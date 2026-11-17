"""Phase 3, the pre-registered one-shot TEST evaluation of the coordinate predictors.

phase3_eval never reads the official test split; this module does, once, and only for the cells,
hyperparameters and hypotheses frozen in configs/phase3_final.json (docs/V2_PIPELINE.md, "Phase 3
pre-registration"). Nothing is selected here: every lr, epoch, k and m comes from the file.

  prereg   write the pre-registration from the finished dev runs (dev files only); commit it
  probes   GPU session, test NOT attached: the file must be committed and unmodified
           (prereg.git_state); its sha256, commit and the start time are recorded
           (prereg_provenance.json). Each declared cell probe is retrained once at its frozen lr
           for exactly its frozen number of epochs (same seed, last head kept: the head the dev
           run restored), its head is saved, and its dev_seen top-5 cells are compared with the
           stored dev ones (reproduction check, repro.json). Exit 1 if any cell is outside the
           tolerance.
  test     CPU session (numpy fp32 kNN, the dev setting): refuses unless the file is the one the
           probes run recorded and repro.json passed. The saved heads are applied on the CPU and
           the probes, the medoid and the hybrids are recomputed on dev_seen with the frozen
           values and checked again (dev_check.json). Only if every cell passes is the test
           split loaded and scored (test_geo_hits.csv, test_row.json per cell), then `primary`.
  primary  primary.csv: one row per pre-registered hypothesis (a linear contrast of per-image
           test GeoScore over cells), paired cluster bootstrap over creators (sequences as a
           sensitivity column), one-sided p, Holm. exploratory_gaps.csv and results.csv hold
           everything else.

The country of a predicted point (phase3_eval's country_top1 sanity check) is not computed on
test: it costs one more pass over the training subset per predictor and is no Phase 3 outcome.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai import geocells as gc
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_eval as p3
from geoguessr_ai import prereg
from geoguessr_ai.v2_data import normalize_creators
from geoguessr_ai.zipq import write_text

DEFAULT_PATH = Path("configs/phase3_final.json")
SIZES = (50_000, 100_000, 200_000, 500_000)
ORDERINGS = ("nat", "bal")
INITS = ("random", "zeroshot")
PREDICTORS = ("medoid", "cell_random", "cell_zeroshot", "hybrid_random", "hybrid_zeroshot")
PRIMARY = "hybrid_zeroshot"
ALPHA, N_BOOT = prereg.PRIMARY_ALPHA, prereg.PRIMARY_N_BOOT
# reproduction check on dev_seen, retrained/recomputed vs the stored dev run
TOLERANCE = {"top1_cell_same_min": 0.99, "top5_set_same_min": 0.95, "probe_geoscore_abs_max": 2.0,
             "hybrid_geoscore_abs_max": 2.0, "medoid_geoscore_abs_max": 0.5}
SHARED = ("seed", "n_boot", "mask_km", "knn_device", "knn_dtype", "hybrid_k", "hybrid_predictor",
          "label_smoothing", "logit_scale", "template", "probe_batch", "probe_epochs",
          "probe_patience", "probe_lrs", "k", "hybrid_m", "data_config_sha256")
CHUNK = 20_000  # test queries per kNN block (memory: CHUNK x hybrid_k neighbours)
HEAD_CHUNK = 20_000  # queries per block of head logits (memory: HEAD_CHUNK x cells)


# ----------------------------------------------------------------- the pre-registration


def _t(w: float, bb: str, o: str, n: int, pred: str) -> dict:
    return {"w": w, "backbone": bb, "ordering": o, "n": n, "predictor": pred}


def _gap(o: str, n: int, pred: str, w: float = 1.0) -> list[dict]:
    """PAIR[0] - PAIR[1] of one cell, times w."""
    return [_t(w, pe.PAIR[0], o, n, pred), _t(-w, pe.PAIR[1], o, n, pred)]


def hypotheses(lo: int, hi: int) -> dict:
    """The primary family. Each statistic is the mean over test images of sum(w * GeoScore of the
    term's cell), so 'gap' = StreetCLIP - L/14-336 on the same images; every claim is '> 0'."""
    a, b = pe.PAIR
    primary = [
        {"id": "H1/gap/bal", "terms": _gap("bal", lo, PRIMARY),
         "claim": f"{a} - {b} test GeoScore, {PRIMARY}, bal N={lo} > 0"},
        {"id": "H2/gap/nat", "terms": _gap("nat", lo, PRIMARY),
         "claim": f"{a} - {b} test GeoScore, {PRIMARY}, nat N={lo} > 0"},
        {"id": "H3/shrink/bal", "terms": _gap("bal", lo, PRIMARY) + _gap("bal", hi, PRIMARY, -1.0),
         "claim": f"gap(N={lo}) - gap(N={hi}), {PRIMARY}, bal > 0 (the gap shrinks)"},
        {"id": "H4/init/nat",
         "terms": _gap("nat", lo, "cell_random") + _gap("nat", lo, "cell_zeroshot", -1.0),
         "claim": f"gap(cell_random) - gap(cell_zeroshot), nat N={lo} > 0 (zero-shot init "
                  "narrows the gap)"},
        {"id": "H5/init/bal",
         "terms": _gap("bal", lo, "cell_random") + _gap("bal", lo, "cell_zeroshot", -1.0),
         "claim": f"gap(cell_random) - gap(cell_zeroshot), bal N={lo} > 0 (zero-shot init "
                  "narrows the gap)"}]
    exploratory = [
        {"id": "E1/hybrid_vs_knn/nat",
         "terms": [_t(0.5 * s, bb, "nat", hi, p) for bb in pe.PAIR
                   for s, p in ((1.0, PRIMARY), (-1.0, "medoid"))],
         "claim": f"GeoScore({PRIMARY}) - GeoScore(medoid), nat N={hi}, average of {a} and {b}; "
                  "not like for like (kNN parts under the 1 km mask, the probe trained "
                  "unmasked): estimate and uncorrected interval only, no verdict"}]
    return {
        "statistic": "mean over test images of sum_terms w * per-image GeoScore of the term's "
                     "cell (5000 * exp(-km / 1492.7)); images scored by every cell of the "
                     "hypothesis (n_images is reported)",
        "criterion": "supported if the Holm-adjusted one-sided p-value < alpha; no other "
                     "outcome of this run changes a verdict",
        "p_value": "one-sided, from n_boot paired cluster-bootstrap resamples of the statistic "
                   "(one resample of clusters applies to every cell of the hypothesis; the same "
                   "resamples for every hypothesis): p = (1 + #{resampled statistic <= 0}) / "
                   "(n_boot + 1)",
        "clusters": "creator_id of test.csv (v2_data.normalize_creators: a missing creator -> "
                    "its sequence)",
        "sensitivity": "the same with sequence clusters (Phase 1's unit), reported next to each "
                       "hypothesis (seq_*); it decides nothing",
        "interval": "uncorrected 95% percentile interval of the same resamples",
        "multiplicity": "Holm-Bonferroni over all primary hypotheses (family-wise alpha). The "
                        "analysis runs once, when every declared cell is scored. If a cell cannot "
                        "be scored after test_read_started.json exists, its hypotheses are not "
                        "evaluable and count with p = 1 (primary --missing-as-p1); a later fix is "
                        "a published amendment and is reported next to that result, not instead",
        "alpha": ALPHA, "n_boot": N_BOOT, "seed": 42, "primary": primary,
        "exploratory": exploratory,
        "secondary": "every other output is exploratory: the contrasts under `exploratory` "
                     "(exploratory_contrasts.csv), 100K/200K, hybrid_random, the gap of "
                     "each other predictor, shrink at nat, median km and accuracy within "
                     "25/200/750/2500 km, dev splits, any per-country or per-creator breakdown"}


def _flags(pred: str, row: dict, cs: dict) -> str:
    """Choices at an edge of their search grid (limitations, kept as they are)."""
    out = []
    if pred.startswith("cell_"):
        grid = sorted({*cs["probe_lrs"], *cs.get("extra_lrs", [])})
        out += ["lr_low"] * (row["lr"] == grid[0]) + ["lr_high"] * (row["lr"] == grid[-1])
        out += ["epoch_cap"] * (row["epoch"] == cs["probe_epochs"])
        out += ["never_stopped_early"] * (not row["stopped_early"])
    elif pred.startswith("hybrid_"):
        out += ["m_high"] * (row["m"] == max(cs["hybrid_m"])) + ["k_high"] * (row["k"] == max(cs["k"]))
    else:
        out += ["k_high"] * (row["k"] == max(cs["k"]))
    return ";".join(out)


def write(dev_run: Path, knn_run: Path, path: Path, dev_results: Path | None = None,
          sizes=SIZES) -> dict:
    """Freeze the declared cells (PAIR x ORDERINGS x sizes x PREDICTORS) with the dev_seen-selected
    values of their stored dev cells: the medoid from knn_run (Step 1), the cell probes and the
    hybrids from dev_run (the merged final dev run). Development files only."""
    dev_run, knn_run = Path(dev_run), Path(knn_run)
    rec = json.loads((dev_run / "cell_count.json").read_text())
    cells, shared, enc = [], {}, {}
    for bb in pe.PAIR:
        for o in ORDERINGS:
            for n in sizes:
                for pred in PREDICTORS:
                    d = pe.cell_dir(knn_run if pred == "medoid" else dev_run, bb, o, n,
                                    p3.METHOD, pred)
                    row = pe.read_row(d)
                    if row is None:
                        raise SystemExit(f"no finished dev cell {d}")
                    cs = json.loads((d / "geo_metrics.json").read_text())["cache_settings"]
                    if cs["cache_version"] != p3.CACHE_VERSION:
                        raise SystemExit(f"{d} predates the current cache version")
                    if pred != "medoid" and cs["geocells_sha256"] != rec["geocells_sha256"]:
                        raise SystemExit(f"{d}: another geocell partition than cell_count.json")
                    for k in SHARED:  # one setting for every declared cell
                        if k in cs and shared.setdefault(k, cs[k]) != cs[k]:
                            raise SystemExit(f"{d}: {k} = {cs[k]}, other cells have {shared[k]}")
                    if enc.setdefault(bb, cs["emb_encoder_sha256"]) != cs["emb_encoder_sha256"]:
                        raise SystemExit(f"{d}: another embedding cache than {bb}'s other cells")
                    cell = {"backbone": bb, "ordering": o, "n": int(n), "predictor": pred,
                            "dev_seen_geoscore": row["dev_seen_geoscore"],
                            "dev_unseen_geoscore": row["dev_unseen_geoscore"],
                            "grid_edge": _flags(pred, row, cs)}
                    if pred == "medoid":
                        cell["k"] = int(row["k"])
                    elif pred.startswith("cell_"):
                        cell |= {"lr": float(row["lr"]), "epoch": int(row["epoch"]),
                                 "extra_lrs": cs.get("extra_lrs", []),
                                 "dev_seen_top_cells_sha256":
                                     prereg.sha256(d / "dev_seen_top_cells.csv")}
                    else:
                        cell |= {"m": int(row["m"]), "k": int(row["k"])}
                    cells.append(cell)
    if shared["hybrid_predictor"] != "medoid" or shared["knn_device"] != "cpu":
        raise SystemExit(f"unexpected dev setting: {shared}")
    reg = {
        "note": "Phase 3 coordinates on the official test split; frozen before any test "
                "distance is read. Every value below is the dev_seen selection of the stored dev "
                "runs, used as it is (no search on test). See docs/V2_PIPELINE.md.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dev_results": str(dev_results).replace("\\", "/") if dev_results else None,
        "dev_results_sha256": prereg.sha256(dev_results) if dev_results else None,
        "primary": {"predictor": PRIMARY, "metric": "geoscore", "pair": list(pe.PAIR),
                    "why": "dev only: by dev_seen GeoScore the best of the five predictors is a "
                           "hybrid in every backbone x ordering x N cell and the zero-shot "
                           "hybrid in most; mean GeoScore is the quantity every selection "
                           "(k, m, lr, cell count) maximised, median km was never selected on"},
        "settings": shared,
        "emb_encoder_sha256": enc,
        "geocells": {"target": rec["target"], "n_cells": rec["n_cells"],
                     "sha256": rec["geocells_sha256"]},
        "search": {"rule": "all grids are final (docs/V2_PIPELINE.md: lr-edge rule, m grid); a "
                           "choice at an edge is kept and listed per cell in grid_edge",
                   "probe_lrs": shared["probe_lrs"], "extra_lrs": p3.EXTRA_LRS,
                   "probe_epochs": shared["probe_epochs"], "probe_patience": shared["probe_patience"],
                   "k": shared["k"], "hybrid_m": shared["hybrid_m"]},
        "reproduction": {
            **TOLERANCE, "split": "dev_seen",
            "probe": "retrained once: frozen lr, exactly `epoch` epochs, same seed, last head. "
                     "top1_cell_same = share of dev_seen queries with the same argmax cell as "
                     "the stored dev_seen_top_cells.csv (sha256 per cell below); top5_set_same "
                     "= share with the same set of 5 cells; geoscore = mean GeoScore of the "
                     "argmax cell's centroid vs dev_seen_geoscore",
            "hybrid_medoid": "recomputed on the CPU (fp32) with the frozen m, k from the "
                             "retrained probe's top cells; mean GeoScore vs dev_seen_geoscore",
            "rule": "checked twice (probes: GPU logits; test: the saved head applied on the CPU). "
                    "Every test session first checks ALL declared cells of BOTH backbones on "
                    "dev_seen; if ANY is outside a tolerance it stops before the test split is "
                    "loaded; no cell is dropped and no hypothesis is evaluated on a subset. A "
                    "tolerance may only be changed by a dated amendment to this file, committed "
                    "before any test distance is read, that publishes every failing number",
            "one_run": "the result is the first run: one probes output (repro.json, its sha256), "
                       "one code commit with no modified tracked file, one pre-registration "
                       "sha256. They are written into dev_check.json, test_read_started.json and "
                       "every test_row.json; `primary` refuses cells of different runs"},
        "hypotheses": hypotheses(min(sizes), max(sizes)), "cells": cells}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2) + "\n", newline="\n")
    return reg


# ----------------------------------------------------------------- shared by both stages


def _cells(reg: dict, bb: str) -> dict:
    return {(c["ordering"], c["n"], c["predictor"]): c for c in reg["cells"] if c["backbone"] == bb}


def _subsets(reg: dict) -> list[tuple[str, int]]:
    return list(dict.fromkeys((c["ordering"], c["n"]) for c in reg["cells"]))


def _geo(reg: dict, dev_run: Path) -> dict:
    """The frozen partition, loaded from the dev run (never rebuilt: see p3.frozen_geo)."""
    want = reg["geocells"]
    tree, sha = gc.load(Path(dev_run) / "geocells" / f"cells_{want['target']}.json")
    if sha != want["sha256"]:
        raise SystemExit(f"geocell partition sha256 {sha[:16]} != pre-registered {want['sha256'][:16]}")
    return p3.geo_of(tree, sha, want["target"])


def _load(reg: dict, data: Path, emb: Path, bb: str, test_emb: Path | None = None):
    """(g.load_data context, its train frame) for one backbone after checking that the inputs
    are the dev runs' (data config and encoder hashes). test_emb None: test is not read."""
    for f, want in ((data / "config.json", reg["settings"]["data_config_sha256"]),
                    (emb / bb / "encoder.json", reg["emb_encoder_sha256"][bb])):
        have = prereg.sha256(f) if f.exists() else None
        if have != want:
            raise SystemExit(f"{f}: sha256 {have} is not the dev runs' {want}")
    if test_emb is not None:
        model = [json.loads((d / "encoder.json").read_text())["model"] for d in (emb / bb, test_emb)]
        if model[0] != model[1]:
            raise SystemExit(f"{test_emb}: embeddings of {model[1]}, the training cache is {model[0]}")
    ctx = g.load_data(data, emb / bb, test_emb is not None, test_emb)
    if test_emb is None and "test" in ctx["evals"]:
        raise AssertionError("test loaded without the test stage")
    if test_emb is not None:  # load_data drops unembedded ids with a warning only
        n = len(pd.read_csv(data / "test.csv", usecols=["id"], dtype=str))
        if len(ctx["dfs"]["test"]) != n:
            raise SystemExit(f"{bb}: {n - len(ctx['dfs']['test'])} of {n} test ids are not embedded")
    tr = ctx["dfs"]["train"]
    return {**ctx, "backbone": bb}, tr


def code_commit(path: Path) -> str:
    """HEAD of the clone holding the pre-registration; SystemExit if a tracked file is modified
    (the test code must be the committed one)."""
    run = lambda *a: subprocess.run(["git", "-C", str(Path(path).resolve().parent), *a],
                                    capture_output=True, text=True, check=False).stdout.strip()
    if run("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files are modified in this clone: run the committed code")
    return run("rev-parse", "HEAD")


def _stored_top(d: Path, ids, sha: str | None = None) -> np.ndarray:
    f = d / "dev_seen_top_cells.csv"
    if sha and prereg.sha256(f) != sha:
        raise SystemExit(f"{f} is not the pre-registered dev file (sha256 differs)")
    t = pd.read_csv(f, dtype={"id": str}).set_index("id")
    return t.loc[list(ids)].filter(regex=r"^c\d+$").to_numpy()


def probe_check(top: np.ndarray, stored: np.ndarray, sel: pd.DataFrame, cent, cell: dict,
                tol: dict) -> dict:
    km = g.haversine(sel["latitude"].to_numpy(), sel["longitude"].to_numpy(), *cent[top[:, 0]].T)
    gs = float(g.geoscore(km).mean())
    r = {"top1_cell_same": float((top[:, 0] == stored[:, 0]).mean()),
         "top5_set_same": float((np.sort(top, 1) == np.sort(stored, 1)).all(1).mean()),
         "geoscore": gs, "stored": cell["dev_seen_geoscore"],
         "diff": gs - cell["dev_seen_geoscore"]}
    r["ok"] = bool(r["top1_cell_same"] >= tol["top1_cell_same_min"]
                   and r["top5_set_same"] >= tol["top5_set_same_min"]
                   and abs(r["diff"]) <= tol["probe_geoscore_abs_max"])
    return r


def apply_head(emb: np.ndarray, head: np.ndarray, scale: float,
               m: int = max(p3.HYBRID_M)) -> np.ndarray:
    """Top-m cells of the saved cosine head, on the CPU, with the probe's own expression
    (geo_eval.probe_logits): exactly tied cells (rows that stayed identical, e.g. empty cells of
    one country after a zero-shot init) then order as in a CPU dev run. Against GPU logits the
    order of such ties can differ, which changes a prediction when the tie is at rank 1 or
    across rank 5; the dev_seen check bounds how often."""
    import torch
    import torch.nn.functional as F

    h = F.normalize(torch.from_numpy(head).float(), dim=-1).T
    with torch.no_grad():
        return np.concatenate([p3.top_cells((scale * torch.from_numpy(emb[s : s + HEAD_CHUNK])
                                             @ h).numpy(), m)
                               for s in range(0, len(emb), HEAD_CHUNK)])


def _report(checks: list[dict], what: str) -> bool:
    t = pd.DataFrame(checks)
    print(t.drop(columns=[c for c in ("stored",) if c in t]).to_string(index=False), flush=True)
    bad = t[~t["ok"]]
    print(f"{what}: {len(t) - len(bad)} of {len(t)} cells inside the tolerance", flush=True)
    return bad.empty


# ----------------------------------------------------------------- stage 1: probes (GPU)


def run_probes(args: argparse.Namespace) -> bool:
    import torch

    path, out, data, emb = Path(args.prereg), Path(args.out), Path(args.data), Path(args.emb_root)
    from geoguessr_ai.train import PROBE_BATCH

    prereg.git_state(path)
    commit = code_commit(path)
    prov = out / "prereg_provenance.json"
    if not prov.exists():  # a resumed run keeps the first session's record
        prereg.provenance(path, prov)
    reg = prereg.check(path, prov)
    st, tol, geo = reg["settings"], reg["reproduction"], _geo(reg, args.dev_run)
    if PROBE_BATCH != st["probe_batch"]:
        raise SystemExit(f"train.PROBE_BATCH {PROBE_BATCH} != pre-registered {st['probe_batch']}")
    cent, n_cells = geo["cent"], len(geo["cent"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"pre-registration {prereg.sha256(path)[:16]}; probes on {device}", flush=True)
    for bb in args.backbones:
        cells = _cells(reg, bb)
        todo = [(o, n, i) for o, n in _subsets(reg) for i in INITS
                if not (pe.cell_dir(out, bb, o, n, p3.METHOD, f"cell_{i}") / "probe.json").exists()]
        if not todo:
            continue
        ctx, tr = _load(reg, data, emb, bb)
        sel, x_sel = ctx["dfs"]["dev_seen"], ctx["embs"]["dev_seen"]
        pool = gc.assign(geo["tree"], tr["latitude"], tr["longitude"])
        q = gc.assign(geo["tree"], sel["latitude"], sel["longitude"])
        text = None
        if any(i == "zeroshot" for *_, i in todo):
            model = json.loads((emb / bb / "encoder.json").read_text())["model"]
            text = g.text_head(model, ctx["classes"], st["template"])
        for o, n, init in todo:
            if args.deadline and time.time() > args.deadline:
                raise p3.StopRequested(f"deadline before {bb} {o} {n} cell_{init}")
            c, t0 = cells[o, n, f"cell_{init}"], time.time()
            mask = (tr[f"{o}_rank"] < n).to_numpy()
            sub, y = tr[mask], pool[mask]
            lat, lon, lab = (sub["latitude"].to_numpy(), sub["longitude"].to_numpy(),
                             sub["label"].to_numpy())
            head = (pe.random_head(n_cells, x_sel.shape[1], st["seed"]) if init == "random" else
                    text[torch.from_numpy(p3.majority_country(y, lab, n_cells, cent, lat, lon))])
            # exactly `epoch` epochs, the last head kept = the head the dev run restored
            a = argparse.Namespace(probe_lr=c["lr"], probe_epochs=c["epoch"], probe_patience=0,
                                   probe_restore_best=False, seed=st["seed"],
                                   label_smoothing=st["label_smoothing"],
                                   logit_scale=st["logit_scale"])
            d = pe.cell_dir(out, bb, o, n, p3.METHOD, f"cell_{init}")
            d.mkdir(parents=True, exist_ok=True)
            logits, _ = g.probe_logits(ctx["embs"]["train"][mask], y, x_sel, q,
                                       {"dev_seen": x_sel}, head, a, device,
                                       save_head=d / "head.npy")
            top = p3.top_cells(logits["dev_seen"], max(p3.HYBRID_M))
            stored = _stored_top(pe.cell_dir(Path(args.dev_run), bb, o, n, p3.METHOD,
                                             f"cell_{init}"), sel["id"],
                                 c["dev_seen_top_cells_sha256"])
            res = {"backbone": bb, "ordering": o, "n": n, "predictor": f"cell_{init}",
                   "lr": c["lr"], "epoch": c["epoch"],
                   **probe_check(top, stored, sel, cent, c, tol), "device": device,
                   "head_sha256": prereg.sha256(d / "head.npy"), "code_commit": commit,
                   "seconds": round(time.time() - t0, 1)}
            pd.DataFrame({"id": sel["id"], **{f"c{j}": top[:, j] for j in range(top.shape[1])}}) \
                .to_csv(d / "dev_seen_top_cells.csv", index=False)
            write_text(d / "probe.json", json.dumps(res, indent=2))  # written last: done
            print(f"{bb} {o} {n} cell_{init}: top-1 same {res['top1_cell_same']:.4f}, top-5 set "
                  f"{res['top5_set_same']:.4f}, GeoScore diff {res['diff']:+.2f} "
                  f"{'OK' if res['ok'] else 'OUTSIDE THE TOLERANCE'} ({res['seconds']:.0f} s)",
                  flush=True)
        del ctx
    want = [(c["backbone"], c["ordering"], c["n"], c["predictor"]) for c in reg["cells"]
            if c["predictor"].startswith("cell_")]
    files = [pe.cell_dir(out, bb, o, n, p3.METHOD, p) / "probe.json" for bb, o, n, p in want]
    checks = [json.loads(f.read_text()) for f in files if f.exists()]
    ok = _report(checks, "probe reproduction (dev_seen)") and len(checks) == len(want)
    if {c["code_commit"] for c in checks} - {commit}:
        raise SystemExit("probes of more than one code commit in this output: start a fresh one")
    write_text(out / "repro.json", json.dumps(
        {"prereg_sha256": prereg.sha256(path), "code_commit": commit, "ok": ok,
         "declared": len(want),
         "done": len(checks), "cells": checks}, indent=2))
    return ok


# ----------------------------------------------------------------- stage 2: dev check, then test


def score_subset(ctx, tr, pool, split: str, o: str, n: int, cells: dict, geo: dict, st: dict,
                 probes: Path, heads: dict) -> tuple[dict, dict, dict]:
    """Coordinates of every declared predictor for one subset and split with the frozen values:
    ({predictor: (lat, lon)}, {init: top cells}, {masked_any_share, fallback shares})."""
    df, x, bb, cent = ctx["dfs"][split], ctx["embs"][split], ctx["backbone"], geo["cent"]
    mask = (tr[f"{o}_rank"] < n).to_numpy()
    sub, db, db_cells = tr[mask], ctx["embs"]["train"][mask], pool[mask]
    lat, lon = sub["latitude"].to_numpy(), sub["longitude"].to_numpy()
    top = {}
    for i in INITS:
        f = pe.cell_dir(probes, bb, o, n, p3.METHOD, f"cell_{i}") / "head.npy"
        if prereg.sha256(f) != heads[bb, o, n, f"cell_{i}"]:
            raise SystemExit(f"{f} is not the head recorded in repro.json")
        top[i] = apply_head(x, np.load(f), st["logit_scale"])
    parts = {p: [] for p in ("medoid", *(f"hybrid_{i}" for i in INITS))}
    fb, masked = {i: 0 for i in INITS}, 0
    for s in range(0, len(df), CHUNK):
        qd = df.iloc[s : s + CHUNK]
        gm = p3.geo_mask(qd, sub, st["mask_km"])
        nb = g.knn(x[s : s + CHUNK], db, st["hybrid_k"], device=None, dtype=st["knn_dtype"],
                   exclude=gm[:2])
        masked += len(np.unique(gm[0]))
        # the first k of the same masked search: Step 1's geo_medoid
        parts["medoid"].append(g.knn_coordinates(*nb, lat, lon, cells[o, n, "medoid"]["k"]))
        for i in INITS:
            c = cells[o, n, f"hybrid_{i}"]
            xy, f = p3.hybrid_coords(nb, db_cells, top[i][s : s + CHUNK], c["m"], "medoid",
                                     c["k"], None, lat, lon, cent)
            parts[f"hybrid_{i}"].append(xy)
            fb[i] += int(f.sum())
    coords = {p: tuple(np.concatenate([xy[j] for xy in v]) for j in (0, 1))
              for p, v in parts.items()}
    coords |= {f"cell_{i}": tuple(cent[top[i][:, 0]].T) for i in INITS}
    info = {"masked_any_share": masked / len(df),
            **{f"hybrid_{i}": fb[i] / len(df) for i in INITS}}
    return coords, top, info


def dev_check(ctx, tr, pool, reg: dict, geo: dict, args) -> list[dict]:
    """dev_seen: the CPU-applied heads vs the stored top cells, and the medoid and hybrids with
    the frozen k / m, k vs their stored GeoScore."""
    cells, tol, bb, sel = _cells(reg, ctx["backbone"]), reg["reproduction"], ctx["backbone"], \
        ctx["dfs"]["dev_seen"]
    out = []
    for o, n in _subsets(reg):
        coords, top, _ = score_subset(ctx, tr, pool, "dev_seen", o, n, cells, geo,
                                      reg["settings"], Path(args.probes), args.heads)
        for pred in PREDICTORS:
            c, row = cells[o, n, pred], {"backbone": bb, "ordering": o, "n": n, "predictor": pred}
            if pred.startswith("cell_"):
                stored = _stored_top(pe.cell_dir(Path(args.dev_run), bb, o, n, p3.METHOD, pred),
                                     sel["id"], c["dev_seen_top_cells_sha256"])
                row |= probe_check(top[pred.removeprefix("cell_")], stored, sel, geo["cent"], c, tol)
            else:
                km = g.haversine(sel["latitude"].to_numpy(), sel["longitude"].to_numpy(),
                                 *coords[pred])
                gs = float(g.geoscore(km).mean())
                lim = tol["medoid_geoscore_abs_max" if pred == "medoid" else
                          "hybrid_geoscore_abs_max"]
                row |= {"geoscore": gs, "stored": c["dev_seen_geoscore"],
                        "diff": gs - c["dev_seen_geoscore"],
                        "ok": bool(abs(gs - c["dev_seen_geoscore"]) <= lim)}
            out.append(row)
        print(f"{bb} {o} {n}: dev_seen checked", flush=True)
    return out


def score_test(ctx, tr, pool, reg: dict, geo: dict, args) -> None:
    out, bb, st, df = Path(args.out), ctx["backbone"], reg["settings"], ctx["dfs"]["test"]
    cells = _cells(reg, bb)
    clusters = normalize_creators(df)["creator_id"].to_numpy()
    for o, n in _subsets(reg):
        dirs = {p: pe.cell_dir(out, bb, o, n, p3.METHOD, p) for p in PREDICTORS}
        if all((d / "test_row.json").exists() for d in dirs.values()):
            continue
        if args.deadline and time.time() > args.deadline:
            raise p3.StopRequested(f"deadline before {bb} {o} {n}")
        t0 = time.time()
        coords, _, info = score_subset(ctx, tr, pool, "test", o, n, cells, geo, st,
                                       Path(args.probes), args.heads)
        for pred, d in dirs.items():
            km = g.haversine(df["latitude"].to_numpy(), df["longitude"].to_numpy(), *coords[pred])
            per = g.per_image_geo(km)
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": df["id"], "pred_lat": coords[pred][0], "pred_lon": coords[pred][1],
                          "km": km, **per}).to_csv(d / "test_geo_hits.csv", index=False)
            lo, hi = g.cluster_bootstrap(per["geoscore"], clusters, st["n_boot"], st["seed"])
            c = cells[o, n, pred]
            row = {"backbone": bb, "ordering": o, "n": n, "predictor": pred, "test_n": len(df),
                   "test_geoscore": float(per["geoscore"].mean()), "test_geoscore_lo": lo,
                   "test_geoscore_hi": hi, "test_median_km": float(np.median(km)),
                   "test_mean_km": float(km.mean()),
                   **{f"test_{k}": float(v.mean()) for k, v in per.items() if k != "geoscore"},
                   "test_masked_any_share": info["masked_any_share"],
                   "test_fallback_share": info.get(pred),
                   "dev_seen_geoscore_prereg": c["dev_seen_geoscore"],
                   "dev_unseen_geoscore_prereg": c["dev_unseen_geoscore"],
                   "grid_edge": c["grid_edge"],
                   **{k: c.get(k) for k in ("lr", "epoch", "m", "k")}, "run": args.run}
            write_text(d / "test_row.json", json.dumps(row, default=float))  # last: done
        print(f"{bb} {o} {n}: test scored in {time.time() - t0:.0f} s", flush=True)


def run_test(args: argparse.Namespace) -> bool:
    path, out, data, emb = Path(args.prereg), Path(args.out), Path(args.data), Path(args.emb_root)
    probes = Path(args.probes)
    if not (probes / "repro.json").exists():
        raise SystemExit(f"no {probes / 'repro.json'}: run the probes stage first")
    reg = prereg.check(path, probes / "prereg_provenance.json")
    rep = json.loads((probes / "repro.json").read_text())
    if not rep["ok"] or rep["prereg_sha256"] != prereg.sha256(path):
        raise SystemExit("the probe reproduction check did not pass for this pre-registration: "
                         "the test split is not read")
    args.run = {"prereg_sha256": rep["prereg_sha256"], "code_commit": code_commit(path),
                "repro_sha256": prereg.sha256(probes / "repro.json")}
    if rep["code_commit"] != args.run["code_commit"]:
        raise SystemExit("the probes ran at another code commit than this clone")
    args.heads = {(c["backbone"], c["ordering"], c["n"], c["predictor"]): c["head_sha256"]
                  for c in rep["cells"]}
    geo, checks = _geo(reg, args.dev_run), []
    for bb in pe.PAIR:  # BOTH backbones, whatever --backbones: before any test image is loaded
        f = out / bb / "dev_check.json"
        if not f.exists() or json.loads(f.read_text())["run"] != args.run:
            ctx, tr = _load(reg, data, emb, bb)
            pool = gc.assign(geo["tree"], tr["latitude"], tr["longitude"])
            write_text(f, json.dumps({"run": args.run, "cells": dev_check(ctx, tr, pool, reg, geo,
                                                                         args)}, indent=2))
            del ctx
        checks += json.loads(f.read_text())["cells"]
    if not _report(checks, "reproduction on dev_seen (CPU)"):
        raise SystemExit("reproduction check failed: the test split is NOT read (see the "
                         "pre-registration's rule)")
    started = out / "test_read_started.json"
    if not started.exists():
        write_text(started, json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        **args.run}))
    elif {k: v for k, v in json.loads(started.read_text()).items() if k != "utc"} != args.run:
        raise SystemExit(f"{started} belongs to another run (probes, code or pre-registration)")
    for bb in args.backbones:
        ctx, tr = _load(reg, data, emb, bb, Path(args.test_emb_root) / bb)
        pool = gc.assign(geo["tree"], tr["latitude"], tr["longitude"])
        score_test(ctx, tr, pool, reg, geo, args)
        del ctx
    return run_primary(args)


# ----------------------------------------------------------------- the analysis


def contrast(out: Path, terms: list[dict]) -> pd.Series:
    """Per test image: sum of w * GeoScore over the terms' cells (images every cell scored).
    FileNotFoundError while a cell is missing."""
    x = None
    for t in terms:
        f = pe.cell_dir(out, t["backbone"], t["ordering"], t["n"], p3.METHOD, t["predictor"])
        h = pd.read_csv(f / "test_geo_hits.csv", dtype={"id": str},
                        usecols=["id", "geoscore"]).set_index("id")["geoscore"] * t["w"]
        x = h if x is None else x + h
    return x.dropna()


def _boot(x: pd.Series, clusters: pd.Series, n_boot: int, seed: int) -> dict:
    means = g.bootstrap_means(x.to_numpy(dtype=float), clusters.loc[x.index].to_numpy(), n_boot, seed)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"ci_lo": float(lo), "ci_hi": float(hi), "p": pe.one_sided_p(means)}


def primary_results(reg: dict, out: Path, data: Path) -> pd.DataFrame:
    """One row per pre-registered hypothesis: estimate, creator-clustered interval, one-sided p,
    Holm-adjusted p and verdict; seq_*: the same with sequence clusters (sensitivity only)."""
    hyp = reg["hypotheses"]
    d = normalize_creators(pd.read_csv(data / "test.csv", dtype=str, keep_default_na=False,
                                       usecols=["id", "sequence", "creator_id"])).set_index("id")
    rows = []
    for h in hyp["primary"]:
        row = {"id": h["id"], "claim": h["claim"], "estimate": np.nan, "ci_lo": np.nan,
               "ci_hi": np.nan, "p": 1.0, "seq_ci_lo": np.nan, "seq_ci_hi": np.nan, "seq_p": 1.0,
               "n_images": 0, "evaluable": False}
        try:
            x = contrast(out, h["terms"])
        except FileNotFoundError:
            rows.append(row)
            continue
        share = d.loc[x.index, "creator_id"].value_counts(normalize=True)
        row |= {"estimate": float(x.mean()), "n_images": len(x), "evaluable": True,
                "n_creators": len(share), "largest_creator_share": float(share.iloc[0]),
                **_boot(x, d["creator_id"], hyp["n_boot"], hyp["seed"]),
                **{f"seq_{k}": v for k, v in _boot(x, d["sequence"], hyp["n_boot"],
                                                   hyp["seed"]).items()}}
        rows.append(row)
    res = pd.DataFrame(rows)
    res["p_holm"], res["seq_p_holm"] = pe.holm(res["p"]), pe.holm(res["seq_p"])
    res["verdict"] = np.where(~res["evaluable"], "not evaluable",
                              np.where(res["p_holm"] < hyp["alpha"], "supported", "not supported"))
    return res.drop(columns="evaluable")


def run_primary(args: argparse.Namespace) -> bool:
    """primary.csv once every declared cell is scored (False before that), then the exploratory
    tables."""
    path, out, data = Path(args.prereg), Path(args.out), Path(args.data)
    reg = json.loads(path.read_text())
    files = [pe.cell_dir(out, c["backbone"], c["ordering"], c["n"], p3.METHOD, c["predictor"])
             / "test_row.json" for c in reg["cells"]]
    missing = [f for f in files if not f.exists()]
    if missing and not getattr(args, "missing_as_p1", False):
        print(f"{len(missing)} of {len(files)} declared cells not scored yet (e.g. {missing[0]}): "
              "no primary analysis; merge the sessions' outputs and run `primary`", flush=True)
        return False
    files = [f for f in files if f.exists()]
    runs = {json.dumps(json.loads(f.read_text())["run"], sort_keys=True) for f in files}
    if len(runs) != 1 or json.loads(next(iter(runs)))["prereg_sha256"] != prereg.sha256(path):
        raise SystemExit(f"the scored cells come from {len(runs)} runs, or from another "
                         "pre-registration: no primary analysis")
    res = primary_results(reg, out, data)
    res.to_csv(out / "primary.csv", index=False)
    print("PRIMARY (pre-registered, Holm over all rows; creator clusters):")
    print(res.drop(columns="claim").to_string(index=False), flush=True)
    pd.DataFrame([json.loads(f.read_text()) for f in files]).to_csv(out / "results.csv", index=False)
    d = normalize_creators(pd.read_csv(data / "test.csv", dtype=str, keep_default_na=False,
                                       usecols=["id", "sequence", "creator_id"])).set_index("id")
    st, rows = reg["settings"], []
    for o, n in _subsets(reg):
        for pred in PREDICTORS:
            try:
                x = contrast(out, _gap(o, n, pred))
            except FileNotFoundError:  # only with --missing-as-p1
                continue
            b = _boot(x, d["creator_id"], st["n_boot"], st["seed"])
            rows.append({"analysis": "exploratory", "ordering": o, "n": n, "predictor": pred,
                         "gap": float(x.mean()), "gap_lo": b["ci_lo"], "gap_hi": b["ci_hi"],
                         "n_images": len(x)})
    pd.DataFrame(rows).to_csv(out / "exploratory_gaps.csv", index=False)
    rows = []
    for h in reg["hypotheses"]["exploratory"]:
        try:
            x = contrast(out, h["terms"])
        except FileNotFoundError:  # only with --missing-as-p1
            continue
        b = _boot(x, d["creator_id"], st["n_boot"], st["seed"])
        rows.append({"analysis": "exploratory", "id": h["id"], "claim": h["claim"],
                     "estimate": float(x.mean()), "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                     "n_images": len(x)})
    pd.DataFrame(rows).to_csv(out / "exploratory_contrasts.csv", index=False)
    print("EXPLORATORY: results.csv, exploratory_gaps.csv, exploratory_contrasts.csv "
          "(uncorrected creator intervals)")
    return True


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("prereg")
    w.add_argument("--dev-run", type=Path, required=True, help="merged final dev phase3_eval dir")
    w.add_argument("--knn-run", type=Path, required=True, help="Step 1 dev phase3_eval dir")
    w.add_argument("--dev-results", type=Path, help="committed dev table to hash")
    for name in ("probes", "test", "primary"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--out", required=True)
        if name != "primary":
            p.add_argument("--emb-root", required=True)
            p.add_argument("--dev-run", required=True,
                           help="dev phase3_eval output holding geocells/ and the cell probes' "
                                "dev_seen_top_cells.csv")
            p.add_argument("--backbones", nargs="+", default=list(pe.PAIR), choices=pe.PAIR)
            p.add_argument("--deadline-min", type=float,
                           help="start no new cell after this many minutes: RESUME NEEDED, exit 3")
        if name == "primary":
            p.add_argument("--missing-as-p1", action="store_true",
                           help="a cell that cannot be scored: its hypotheses count with p = 1")
        if name == "test":
            p.add_argument("--probes", required=True, help="output of the probes stage")
            p.add_argument("--test-emb-root", required=True)
    for p in sub.choices.values():
        p.add_argument("--prereg", type=Path, default=DEFAULT_PATH)
    args = ap.parse_args(argv)
    m = getattr(args, "deadline_min", None)
    args.deadline = time.time() + 60 * m if m else None
    return args


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    try:
        if args.cmd == "prereg":
            reg = write(args.dev_run, args.knn_run, args.prereg, args.dev_results)
            print(f"{args.prereg}: {len(reg['cells'])} cells, "
                  f"{len(reg['hypotheses']['primary'])} primary hypotheses, sha256 "
                  f"{prereg.sha256(args.prereg)}; commit it before the probes run")
        elif args.cmd == "probes":
            sys.exit(0 if run_probes(args) else 1)
        elif args.cmd == "test":
            run_test(args)
        else:
            prereg.git_state(args.prereg)
            run_primary(args)
    except p3.StopRequested as e:
        print(f"RESUME NEEDED: {e}. Rerun with this output seeded; finished cells are skipped.",
              flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
