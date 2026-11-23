"""V4 recipe B: the pre-registered one-shot TEST evaluation (docs/V4_FEASIBILITY.md, section 5d).

Nothing is selected here: the weights, predictors, hypotheses and outcome rules come from
configs/v4_test_prereg.json.

  prereg   write the pre-registration from the finished dev runs (dev files only); commit it
  test     GPU, one job x backbone: refuses (gate) unless the file is committed, unmodified and has
           the sha256 given on the command line, the code is committed, job 6 finished after that
           commit, and V3's comparators reproduce V3's committed table. Then the weights are
           checked and reproduced on 2K dev_unseen images. Only after that is the test split read.
           Job 4 also embeds its bal_50K training images, before test, for the secondary hybrid.
  hybrid   CPU, job 4 x backbone, after `test`: the secondary hybrid with V3's frozen m and k
  analyze  primary.csv, secondary.csv, outcomes.json (Holm within each family)

  python -m geoguessr_ai.v4_test test --prereg-sha <sha256> --job 4 --backbone streetclip \\
      --inputs <bundle> --parts <archive> --runs <v4_runs> --v3-test <phase3_final> \\
      --test-csv <v2>/test.csv --test-zips <dir with test/NN.zip> --out <out>
"""

import argparse
import calendar
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geoguessr_ai import geo_eval as g
from geoguessr_ai import geocells as gc
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_eval as p3
from geoguessr_ai import prereg
from geoguessr_ai import v4_train as vt
from geoguessr_ai.phase3_final import code_commit
from geoguessr_ai.v2_data import normalize_creators
from geoguessr_ai.v4_inputs import dev_subset, file_sha256, read_csv, verify_bundle
from geoguessr_ai.zipq import write_text

DEFAULT_PATH = Path("configs/v4_test_prereg.json")
V3_RESULTS = "results/v3/test_results.csv"
JOBS = (4, 5)
SEED2 = {"k": 6, "name": "B_full_50k_seed2", "seed": 43}
MARGIN, ALPHA, N_BOOT, SEED = 40.0, 0.05, 10_000, 42
TEST_N, TEST_SHARDS, DEV_CHECK_N, DEV_CHECK_MIN = 210_122, 5, 2000, 0.99
CELL, HYBRID = "cell_zeroshot", "hybrid_zeroshot"
CHUNK = 20_000  # test queries per kNN block, as phase3_final
S, L = pe.PAIR


# ----------------------------------------------------------------- the pre-registration


def _gap(src: str, n: int, pred: str, w: float = 1.0) -> list[dict]:
    return [{"w": w, "src": src, "backbone": S, "n": n, "predictor": pred},
            {"w": -w, "src": src, "backbone": L, "n": n, "predictor": pred}]


def _delta(n: int, pred: str = CELL, w: float = 1.0) -> list[dict]:
    """Fine-tuned gap - frozen gap, on the same images."""
    return _gap("b", n, pred, w) + _gap("v3", n, pred, -w)


def hypotheses(lo: int, hi: int) -> dict:
    primary = [
        {"id": "P1/change/500K", "test": "two_sided", "terms": _delta(hi),
         "claim": f"the gap changes at N={hi}: fine-tuned gap - frozen gap != 0"},
        {"id": "P2/keep/500K", "test": "tost", "terms": _delta(hi),
         "claim": f"the gap is kept at N={hi}: |fine-tuned gap - frozen gap| < {MARGIN:g}"},
        {"id": "P3/change/50K", "test": "two_sided", "terms": _delta(lo),
         "claim": f"the gap changes at N={lo}: fine-tuned gap - frozen gap != 0"}]
    secondary = [
        {"id": "S1/keep/50K", "test": "tost", "terms": _delta(lo),
         "claim": f"|fine-tuned gap - frozen gap| < {MARGIN:g} at N={lo}. Moved out of the primary "
                  "family before test because its expected power is about zero; it never yields "
                  "the label 'kept'"},
        {"id": "S2/reversed/500K", "test": "below_zero", "terms": _gap("b", hi, CELL),
         "claim": f"the fine-tuned gap is < 0 at N={hi}"},
        {"id": "S3/reversed/50K", "test": "below_zero", "terms": _gap("b", lo, CELL),
         "claim": f"the fine-tuned gap is < 0 at N={lo}"},
        {"id": "S4/fine_tuned_gap/50K-500K", "test": "two_sided",
         "terms": _gap("b", lo, CELL) + _gap("b", hi, CELL, -1.0),
         "claim": f"fine-tuned gap(N={lo}) - fine-tuned gap(N={hi}) != 0"},
        {"id": "S5/change/50K-500K", "test": "two_sided",
         "terms": _delta(lo) + _delta(hi, w=-1.0),
         "claim": f"the gap change at N={lo} differs from the gap change at N={hi}"},
        {"id": "S6/change/hybrid/50K", "test": "two_sided", "terms": _delta(lo, HYBRID),
         "claim": f"the gap changes at N={lo} under {HYBRID}, with V3's frozen m and k"}]
    return {
        "statistic": "mean over test images of sum_terms w * per-image GeoScore "
                     "(5000 * exp(-km / 1492.7)), on the images every term scored",
        "bootstrap": "paired creator-cluster bootstrap: the same resamples for every term and "
                     "test; sequence clusters are a sensitivity column and decide nothing",
        "p_values": {
            "two_sided": "min(1, 2 * min(p_up, p_down)), p_up = (1 + #{stat <= 0}) / (B + 1), "
                         "p_down = (1 + #{stat >= 0}) / (B + 1)",
            "tost": "max((1 + #{stat <= -margin}) / (B + 1), (1 + #{stat >= margin}) / (B + 1))",
            "below_zero": "(1 + #{stat >= 0}) / (B + 1)"},
        "multiplicity": "Holm within the primary family (3 tests) and, separately, within the "
                        "secondary family (6 tests). Secondary results never change a primary "
                        "verdict.",
        "margin": MARGIN, "alpha": ALPHA, "n_boot": N_BOOT, "seed": SEED,
        "primary": primary, "secondary": secondary,
        "outcomes": {
            "labels": ["narrowed", "widened", "kept", "reversed", "inconclusive",
                       "narrowed, within the margin", "widened, within the margin",
                       "reversed, within the margin"],
            "500K": "P1 significant: narrowed (estimate < 0) or widened (> 0); narrowed becomes "
                    "reversed if S2 is also significant. P2 significant too: add ', within the "
                    "margin'. P2 only: kept. Neither: inconclusive.",
            "50K": "P3 significant: narrowed or widened; narrowed becomes reversed if S3 is also "
                   "significant. Otherwise inconclusive. 'kept' is not available at 50K."}}


LIMITATIONS = [
    "V4 gaps are NOT comparable to V3's H1: H1 is the hybrid_zeroshot gap, the V4 primary "
    "predictor is cell_zeroshot (top-1 cell centroid).",
    "Recipe B was chosen after seeing recipe A's dev results, and its lr search ended at the low "
    "edge of the extended grid (grid_edge).",
    "V3's test values for the frozen comparators exist and are published, so the fine-tuned vs "
    "frozen contrasts are not blind on the frozen side.",
    "The V3 comparators were selected on dev_seen; the V4 lr was selected on dev_unseen.",
    "The contrast is the adapter update under this augmented schedule; nothing separates the "
    "effect of the augmentation.",
    "N and repetition are confounded: both runs see the same number of images, so 50K sees each "
    "image about 10 times and 500K once.",
    "The CIs cover images and creators, not training seeds. One seed per model; job 6 (seed 43, "
    "50K) is descriptive and never goes to test.",
    "The hybrid uses V3's m and k, which were selected for the frozen encoder, not re-selected.",
    "LoRA adapts the last 12 of 24 blocks only.",
]


def _git(path: Path, *a) -> str:
    return subprocess.run(["git", "-C", str(Path(path).resolve().parent), *a], capture_output=True,
                          text=True, check=False).stdout.strip()


def run_id(job: int, name: str, bb: str) -> str:
    return f"k{job}_{name}_{bb}"


def write(runs: Path, config: Path, selection: Path, v3_prereg: Path, path: Path,
          test_n: int = TEST_N, shards: int = TEST_SHARDS) -> dict:
    """Freeze the seed-42 models of jobs 4 and 5 and the analysis. Development files only; V3's
    committed test table is referenced by its git object id and is not read."""
    cfg, sel, v3 = (json.loads(Path(f).read_text()) for f in (config, selection, v3_prereg))
    models, commits = [], set()
    for k in JOBS:
        job = vt.job_of(cfg, k)
        for bb in pe.PAIR:
            rid = run_id(k, job["name"], bb)
            d = Path(runs) / rid
            res = json.loads((d / "result.json").read_text())
            sha = file_sha256(d / "weights_final.pt")
            if not (d / "DONE").read_text().strip() == sha == res["weights_final_sha256"]:
                raise SystemExit(f"{rid}: DONE, result.json and weights_final.pt disagree")
            if not res["head_frozen"] or res["seed"] != job["seed"] or res["lr"] != sel["selected"][bb]:
                raise SystemExit(f"{rid}: not the declared recipe B run")
            commits.add(res["code_commit"])
            models.append({"job": k, "name": job["name"], "backbone": bb, "n": job["n"],
                           "run_id": rid, "run_key": res["key"], "model": res["model"],
                           "lr": res["lr"], "seed": res["seed"], "weights_sha256": sha,
                           "head_sha256": res["head"],
                           "dev_predictions_sha256": file_sha256(d / "final" / "dev_predictions.csv"),
                           "dev_unseen_geoscore": res["dev"]["geoscore"]["mean"],
                           "dev_seen_geoscore": res["dev_seen"]["geoscore"]["mean"]})
    if len(commits) != 1:
        raise SystemExit(f"the declared runs come from several commits: {sorted(commits)}")
    ns = sorted({m["n"] for m in models})
    blob = _git(Path(config), "rev-parse", f"HEAD:{V3_RESULTS}")
    if not blob:
        raise SystemExit(f"{V3_RESULTS} is not committed")
    hyb = {c["backbone"]: {"m": c["m"], "k": c["k"]} for c in v3["cells"]
           if (c["ordering"], c["n"], c["predictor"]) == ("bal", ns[0], HYBRID)}
    reg = {
        "note": "V4 recipe B test pre-registration, frozen before any V4 test image is read and "
                "before job 6 finishes. See docs/V4_FEASIBILITY.md, section 5d.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe": "B", "ordering": "bal", "metric": "geoscore", "pair": list(pe.PAIR),
        "primary_predictor": CELL,
        "config": Path(config).as_posix(), "config_sha256": file_sha256(config),
        "selection": Path(selection).as_posix(), "selection_sha256": file_sha256(selection),
        "train_commit": commits.pop(), "inputs": cfg["inputs"],
        "models": models,
        "seed2_required_before_test": SEED2,
        "comparators": {
            "v3_prereg_sha256": prereg.sha256(v3_prereg), "v3_results": V3_RESULTS,
            "v3_results_git_blob": blob,
            "cells": [{"backbone": bb, "n": n, "predictor": CELL} for bb in pe.PAIR for n in ns]
            + [{"backbone": bb, "n": ns[0], "predictor": HYBRID} for bb in pe.PAIR]},
        "hybrid": {"n": ns[0], "predictor": "medoid", "hybrid_k": v3["settings"]["hybrid_k"],
                   "mask_km": v3["settings"]["mask_km"], "dtype": v3["settings"]["knn_dtype"],
                   "frozen_from_v3": hyb},
        "test": {"images": test_n, "shards": shards},
        "dev_check": {"split": "dev_unseen", "n": DEV_CHECK_N, "subset_seed": cfg["evals"]["subset_seed"],
                      "top1_same_min": DEV_CHECK_MIN,
                      "rule": "before the test split is read, each model's top-1 cell on these "
                              "images must match its stored dev prediction; otherwise stop"},
        "hypotheses": hypotheses(ns[0], ns[-1]),
        "limitations": LIMITATIONS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2) + "\n", newline="\n")
    return reg


# ----------------------------------------------------------------- the gate (reads no V4 test file)


def session_end(quota: Path, k: int) -> int | None:
    """Unix time of the last session that ran job k and exited 0."""
    end, jobs = None, []
    for line in quota.read_text().splitlines():
        w = line.split()
        if w and w[0] == "session_start_utc":
            jobs = w[3:]
        elif w and w[0] == "session_end_utc" and str(k) in jobs and "(exit 0," in line:
            end = calendar.timegm(time.strptime(w[1], "%Y-%m-%dT%H:%M:%SZ"))
    return end


def check_seed2(reg: dict, runs: Path, prereg_time: int) -> None:
    j = reg["seed2_required_before_test"]
    for bb in reg["pair"]:
        d = runs / run_id(j["k"], j["name"], bb)
        if not (d / "DONE").exists():
            raise SystemExit(f"{d}: job {j['k']} has not finished: the test split is not read")
        r = json.loads((d / "result.json").read_text())
        if (r["seed"], r["code_commit"]) != (j["seed"], reg["train_commit"]):
            raise SystemExit(f"{d}: not the declared seed-2 run")
        if not (d / "DONE").read_text().strip() == file_sha256(d / "weights_final.pt") == r["weights_final_sha256"]:
            raise SystemExit(f"{d}: DONE, result.json and weights_final.pt disagree")
    end = session_end(runs / "quota_times.txt", j["k"])
    if end is None:
        raise SystemExit(f"quota_times.txt: no finished session of job {j['k']}")
    if not prereg_time < end:
        raise SystemExit(f"the pre-registration was committed after job {j['k']} finished")


def v3_dir(root: Path, bb: str, n: int, pred: str) -> Path:
    return pe.cell_dir(Path(root), bb, "bal", n, p3.METHOD, pred)


def check_comparators(reg: dict, path: Path, v3_test: Path) -> None:
    """Each V3 hits file reproduces V3's committed test table and names V3's pre-registration."""
    c = reg["comparators"]
    root = Path(_git(path, "rev-parse", "--show-toplevel"))
    if _git(path, "rev-parse", f"HEAD:{c['v3_results']}") != c["v3_results_git_blob"]:
        raise SystemExit(f"{c['v3_results']} is not the pre-registered V3 table")
    table = pd.read_csv(root / c["v3_results"]).query("ordering == 'bal'")
    for t in c["cells"]:
        d = v3_dir(v3_test, t["backbone"], t["n"], t["predictor"])
        if not (d / "test_row.json").exists():
            raise SystemExit(f"{d}: V3 comparator missing")
        if json.loads((d / "test_row.json").read_text())["run"]["prereg_sha256"] != c["v3_prereg_sha256"]:
            raise SystemExit(f"{d}: not from V3's pre-registered run")
        want = table[(table["backbone"] == t["backbone"]) & (table["n"] == t["n"])
                     & (table["predictor"] == t["predictor"])]["test_geoscore"].item()
        have = pd.read_csv(d / "test_geo_hits.csv", usecols=["geoscore"])["geoscore"].mean()
        if abs(have - want) > 1e-6:
            raise SystemExit(f"{d}: hits do not reproduce V3's committed table")


def check_code_frozen(path: Path, prereg_commit: str) -> None:
    """This module is the one committed with the pre-registration."""
    if _git(path, "diff", "--name-only", prereg_commit, "HEAD", "--", ":/src/geoguessr_ai/v4_test.py"):
        raise SystemExit("v4_test.py changed since the pre-registration commit")


def gate(args) -> tuple[dict, dict]:
    """(pre-registration, run identity), or SystemExit before anything of V4's test is read."""
    path = Path(args.prereg)
    prereg_commit, prereg_time = prereg.git_state(path)
    if prereg.sha256(path) != args.prereg_sha:
        raise SystemExit(f"{path}: sha256 {prereg.sha256(path)} is not --prereg-sha")
    commit = code_commit(path)
    check_code_frozen(path, prereg_commit)
    draft = json.loads(path.read_text())
    check_seed2(draft, Path(args.runs), prereg_time)
    check_comparators(draft, path, Path(args.v3_test))
    prov = Path(args.out) / "prereg_provenance.json"
    if not prov.exists():  # only after every check: it dates the first real test session
        prereg.provenance(path, prov)
    return prereg.check(path, prov), {"prereg_sha256": args.prereg_sha, "code_commit": commit}


# ----------------------------------------------------------------- test (GPU)


class TestImages:
    """Test images by row, from the official zips, with V3's eval crop."""

    __test__ = False  # not a pytest class

    def __init__(self, zips: Path, index: dict, ids, size: int):
        missing = [i for i in ids if i not in index]
        if missing:
            raise SystemExit(f"{len(missing)} test ids are not in the zips, e.g. {missing[:3]}")
        self.zips, self.size, self._open = Path(zips), size, {}
        self.items = [index[i] for i in ids]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, row: int):
        from geoguessr_ai.embed import clip_center_crop
        from geoguessr_ai.train import to_tensor_array

        shard, name = self.items[row]
        zf = self._open.get(shard) or self._open.setdefault(
            shard, zipfile.ZipFile(self.zips / "test" / f"{shard:02d}.zip"))
        with Image.open(io.BytesIO(zf.read(name))) as im:
            return to_tensor_array(clip_center_crop(im.convert("RGB"), self.size)), row


def model_of(reg: dict, job: int, bb: str) -> dict:
    m = [x for x in reg["models"] if (x["job"], x["backbone"]) == (job, bb)]
    if len(m) != 1:
        raise SystemExit(f"job {job} x {bb} is not a pre-registered model")
    return m[0]


def load_model(reg, m: dict, cfg: dict, inputs: Path, runs: Path, smoke: bool, device):
    import torch

    f = runs / m["run_id"] / "weights_final.pt"
    if file_sha256(f) != m["weights_sha256"]:
        raise SystemExit(f"{f}: not the pre-registered weights")
    hf = inputs / "heads" / m["backbone"] / f"bal_{m['n']}_cell_zeroshot.npy"
    if file_sha256(hf) != m["head_sha256"]:
        raise SystemExit(f"{hf}: not the pre-registered head")
    model, trainable, size = vt.build_model(cfg, m["model"], np.load(hf), "full", m["seed"], smoke, device)
    w = torch.load(f, map_location="cpu", weights_only=True)
    if set(w) != set(trainable):
        raise SystemExit(f"{f}: its tensors are not this model's LoRA parameters")
    for n, p in trainable.items():
        p.data.copy_(w[n].to(p.device))
    return model.eval(), size


def dev_check(reg, m, model, size, inputs, parts, where, runs, device, amp, workers, batch) -> dict:
    d = reg["dev_check"]
    f = runs / m["run_id"] / "final" / "dev_predictions.csv"
    if file_sha256(f) != m["dev_predictions_sha256"]:
        raise SystemExit(f"{f}: not the pre-registered dev predictions")
    dev = read_csv(inputs / "dev.csv").set_index("id", drop=False)
    ids = dev_subset(dev["id"], d["n"], d["subset_seed"])
    df = dev.loc[ids].reset_index(drop=True)
    top, _ = vt.Eval("dev", df, vt.PartImages(parts, where, df["id"], size, False), batch, workers) \
        .run(model, device, amp)
    stored = pd.read_csv(f, dtype={"id": str}).set_index("id").loc[ids, "c0"].to_numpy()
    same = float((top[:, 0] == stored).mean())
    return {"n": len(ids), "top1_same": same, "min": d["top1_same_min"], "ok": same >= d["top1_same_min"]}


def run_test(args) -> int:
    import torch

    from geoguessr_ai.prep_data import index_shard, make_opener

    reg, run = gate(args)
    m = model_of(reg, args.job, args.backbone)
    out, runs, inputs, parts = Path(args.out) / m["run_id"], Path(args.runs), Path(args.inputs), Path(args.parts)
    if (out / "test_row.json").exists():
        if json.loads((out / "test_row.json").read_text())["run"] != run:
            raise SystemExit(f"{out}: scored by another run")
        print(f"{m['run_id']}: test already scored", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(_git(Path(args.prereg), "rev-parse", "--show-toplevel")) / reg["config"]
    if file_sha256(cfg_path) != reg["config_sha256"]:
        raise SystemExit(f"{cfg_path}: not the pre-registered training config")
    cfg = json.loads(cfg_path.read_text())
    verify_bundle(inputs, reg["inputs"]["bundle_manifest_sha256"])
    tree, gsha = gc.load(inputs / "cells_5000.json")
    if gsha != reg["inputs"]["geocells_sha256"]:
        raise SystemExit("the partition is not the frozen one")
    geo = p3.geo_of(tree, gsha, 5000)
    where = vt.archive_index(parts, reg["inputs"]["archive_manifest_sha256"])
    auto = vt.device_for(args.job, args.backbone) if torch.cuda.is_available() else "cpu"
    device = torch.device(auto if args.device == "auto" else args.device)
    amp, t = device.type == "cuda", cfg["train"]
    batch = args.batch or t["eval_batch"]
    workers = t["workers"] if args.workers is None else args.workers
    runtime = {"device": str(device), "batch": batch, "workers": workers}  # recorded, not part of `run`
    model, size = load_model(reg, m, cfg, inputs, runs, args.smoke, device)

    chk = dev_check(reg, m, model, size, inputs, parts, where, runs, device, amp, workers, batch)
    write_text(out / "dev_check.json", json.dumps({**chk, "runtime": runtime, "run": run}, indent=2))
    if not chk["ok"]:
        raise SystemExit(f"{m['run_id']}: dev reproduction {chk['top1_same']:.4f} < {chk['min']}: "
                         "the test split is NOT read")
    hybrid = m["n"] == reg["hybrid"]["n"]
    if hybrid and not (out / "train_emb.npz").exists():
        tr = read_csv(inputs / "train.csv")
        tr = tr[tr["bal_rank"] < m["n"]].sort_values("bal_rank").reset_index(drop=True)
        _, e = vt.Eval("train", tr, vt.PartImages(parts, where, tr["id"], size, False), batch,
                       workers).run(model, device, amp, keep_emb=True)
        np.savez(out / "train_emb.npz", ids=tr["id"].to_numpy(dtype=str), emb=e)

    started = out / "test_read_started.json"
    if not started.exists():
        write_text(started, json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **run}))
    elif {k: v for k, v in json.loads(started.read_text()).items() if k != "utc"} != run:
        raise SystemExit(f"{started} belongs to another run")
    df = normalize_creators(pd.read_csv(args.test_csv, dtype=str, keep_default_na=False,
                                        usecols=["id", "latitude", "longitude", "sequence", "creator_id"]))
    if len(df) != reg["test"]["images"] or df["id"].duplicated().any():
        raise SystemExit(f"{args.test_csv}: {len(df)} rows, expected {reg['test']['images']} unique ids")
    df[["latitude", "longitude"]] = df[["latitude", "longitude"]].astype(float)
    op, index = make_opener(Path(args.test_zips)), {}
    for s in range(reg["test"]["shards"]):
        index.update(index_shard(op, "test", s))
    ds = TestImages(Path(args.test_zips), index, df["id"].tolist(), size)
    top, e = vt.Eval("test", df, ds, batch, workers).run(model, device, amp, keep_emb=hybrid)
    km = g.haversine(df["latitude"].to_numpy(), df["longitude"].to_numpy(), *geo["cent"][top[:, 0]].T)
    per = g.per_image_geo(km)
    pd.DataFrame({"id": df["id"], **{f"c{j}": top[:, j] for j in range(top.shape[1])},
                  "pred_lat": geo["cent"][top[:, 0], 0], "pred_lon": geo["cent"][top[:, 0], 1],
                  "km": km, **per}).to_csv(out / "test_geo_hits.csv", index=False)
    if hybrid:
        np.savez(out / "test_emb.npz", ids=df["id"].to_numpy(dtype=str), emb=e)
    write_text(out / "test_row.json", json.dumps(
        {"run_id": m["run_id"], "test_n": len(df), "weights_sha256": m["weights_sha256"],
         "runtime": runtime, "run": run}, indent=2))  # written last: done
    print(f"{m['run_id']}: test scored ({len(df)} images)", flush=True)
    return 0


# ----------------------------------------------------------------- hybrid (CPU, secondary)


def run_hybrid(args) -> int:
    reg, run = gate(args)
    m = model_of(reg, args.job, args.backbone)
    h = reg["hybrid"]
    if m["n"] != h["n"]:
        raise SystemExit(f"the hybrid is pre-registered at N={h['n']} only")
    out, inputs = Path(args.out) / m["run_id"], Path(args.inputs)
    row = out / "hybrid_test_row.json"
    if row.exists():
        return 0
    if not (out / "test_row.json").exists() or json.loads((out / "test_row.json").read_text())["run"] != run:
        raise SystemExit(f"{out}: run `test` first, with this pre-registration and code")
    tree, gsha = gc.load(inputs / "cells_5000.json")
    if gsha != reg["inputs"]["geocells_sha256"]:
        raise SystemExit("the partition is not the frozen one")
    cent = p3.geo_of(tree, gsha, 5000)["cent"]
    tr = read_csv(inputs / "train.csv")
    tr = tr[tr["bal_rank"] < m["n"]].sort_values("bal_rank").reset_index(drop=True)
    zt, zq = np.load(out / "train_emb.npz"), np.load(out / "test_emb.npz")
    if list(map(str, zt["ids"])) != tr["id"].tolist():
        raise SystemExit(f"{out}/train_emb.npz: not the bal_{m['n']} rows in order")
    hits = pd.read_csv(out / "test_geo_hits.csv", dtype={"id": str})
    if list(map(str, zq["ids"])) != hits["id"].tolist():
        raise SystemExit(f"{out}/test_emb.npz: not the scored test rows in order")
    df = pd.read_csv(args.test_csv, dtype=str, keep_default_na=False,
                     usecols=["id", "latitude", "longitude"]).set_index("id").loc[hits["id"]].reset_index()
    df[["latitude", "longitude"]] = df[["latitude", "longitude"]].astype(float)
    db, x, top = zt["emb"], zq["emb"], hits.filter(regex=r"^c\d+$").to_numpy()
    lat, lon = tr["latitude"].to_numpy(), tr["longitude"].to_numpy()
    cells = gc.assign(tree, lat, lon)
    f = h["frozen_from_v3"][m["backbone"]]
    las, los, fb = [], [], 0
    chunk = args.knn_chunk
    for s in range(0, len(df), chunk):
        gm = p3.geo_mask(df.iloc[s : s + chunk], tr, h["mask_km"])
        # device=None: numpy on the CPU, V3's frozen kNN numerics (not an environment setting)
        nb = g.knn(x[s : s + chunk], db, min(h["hybrid_k"], len(db)), device=None, dtype=h["dtype"],
                   exclude=gm[:2])
        (la, lo), miss = p3.hybrid_coords(nb, cells, top[s : s + chunk], f["m"], h["predictor"], f["k"],
                                          None, lat, lon, cent)
        las.append(la), los.append(lo)
        fb += int(miss.sum())
    la, lo = np.concatenate(las), np.concatenate(los)
    km = g.haversine(df["latitude"].to_numpy(), df["longitude"].to_numpy(), la, lo)
    pd.DataFrame({"id": df["id"], "pred_lat": la, "pred_lon": lo, "km": km, **g.per_image_geo(km)}) \
        .to_csv(out / "hybrid_test_geo_hits.csv", index=False)
    write_text(row, json.dumps({"run_id": m["run_id"], "fallback_share": fb / len(df), "run": run}, indent=2))
    return 0


# ----------------------------------------------------------------- analysis


def p_value(means: np.ndarray, test: str, margin: float) -> float:
    b = len(means)
    up, down = (1 + (means <= 0).sum()) / (b + 1), (1 + (means >= 0).sum()) / (b + 1)
    if test == "two_sided":
        return float(min(1.0, 2 * min(up, down)))
    if test == "below_zero":
        return float(down)
    if test == "tost":
        return float(max((1 + (means <= -margin).sum()) / (b + 1), (1 + (means >= margin).sum()) / (b + 1)))
    raise ValueError(test)


def outcome(delta: float, changed: bool, kept: bool, reversed_: bool) -> str:
    if changed:
        base = "widened" if delta > 0 else "reversed" if reversed_ else "narrowed"
        return f"{base}, within the margin" if kept else base
    return "kept" if kept else "inconclusive"


def contrast(terms: list[dict], reg: dict, out: Path, v3_test: Path) -> pd.Series:
    """Per test image: sum of w * GeoScore; FileNotFoundError while a term is missing."""
    x = None
    for t in terms:
        if t["src"] == "v3":
            f = v3_dir(v3_test, t["backbone"], t["n"], t["predictor"]) / "test_geo_hits.csv"
        else:
            m = next(a for a in reg["models"] if (a["backbone"], a["n"]) == (t["backbone"], t["n"]))
            f = out / m["run_id"] / ("hybrid_test_geo_hits.csv" if t["predictor"] == HYBRID else "test_geo_hits.csv")
        h = pd.read_csv(f, dtype={"id": str}, usecols=["id", "geoscore"]).set_index("id")["geoscore"] * t["w"]
        x = h if x is None else x.add(h)
    return x.dropna()


def family(rows: list[dict], reg: dict, out: Path, v3_test: Path, d: pd.DataFrame) -> pd.DataFrame:
    hy, res = reg["hypotheses"], []
    for h in rows:
        r = {"id": h["id"], "test": h["test"], "claim": h["claim"], "estimate": np.nan, "ci_lo": np.nan,
             "ci_hi": np.nan, "ci90_lo": np.nan, "ci90_hi": np.nan, "p": 1.0, "seq_p": 1.0,
             "n_images": 0, "evaluable": False}
        try:
            x = contrast(h["terms"], reg, out, v3_test)
        except FileNotFoundError:
            res.append(r)
            continue
        boot = g.bootstrap_means(x.to_numpy(float), d.loc[x.index, "creator_id"].to_numpy(), hy["n_boot"], hy["seed"])
        seq = g.bootstrap_means(x.to_numpy(float), d.loc[x.index, "sequence"].to_numpy(), hy["n_boot"], hy["seed"])
        r |= {"estimate": float(x.mean()), "n_images": len(x), "evaluable": True,
              **dict(zip(("ci_lo", "ci_hi", "ci90_lo", "ci90_hi"), map(float, np.percentile(boot, [2.5, 97.5, 5, 95])))),
              "p": p_value(boot, h["test"], hy["margin"]), "seq_p": p_value(seq, h["test"], hy["margin"])}
        res.append(r)
    t = pd.DataFrame(res)
    t["p_holm"], t["seq_p_holm"] = pe.holm(t["p"]), pe.holm(t["seq_p"])
    t["significant"] = t["evaluable"] & (t["p_holm"] < hy["alpha"])
    return t


def outcomes(p: pd.DataFrame, s: pd.DataFrame) -> dict:
    a, b = p.set_index("id"), s.set_index("id")
    return {
        "500K": outcome(a.at["P1/change/500K", "estimate"], bool(a.at["P1/change/500K", "significant"]),
                        bool(a.at["P2/keep/500K", "significant"]), bool(b.at["S2/reversed/500K", "significant"])),
        "50K": outcome(a.at["P3/change/50K", "estimate"], bool(a.at["P3/change/50K", "significant"]),
                       False, bool(b.at["S3/reversed/50K", "significant"]))}


def run_analyze(args) -> int:
    reg, run = gate(args)
    out = Path(args.out)
    rows = [out / m["run_id"] / "test_row.json" for m in reg["models"]]
    missing = [f for f in rows if not f.exists()]
    if missing and not args.missing_as_p1:
        raise SystemExit(f"{len(missing)} models not scored yet, e.g. {missing[0]}")
    if any(json.loads(f.read_text())["run"] != run for f in rows if f.exists()):
        raise SystemExit("the scored models come from another pre-registration or code commit")
    d = normalize_creators(pd.read_csv(args.test_csv, dtype=str, keep_default_na=False,
                                       usecols=["id", "sequence", "creator_id"])).set_index("id")
    hy = reg["hypotheses"]
    p, s = (family(hy[k], reg, out, Path(args.v3_test), d) for k in ("primary", "secondary"))
    p.to_csv(out / "primary.csv", index=False)
    s.to_csv(out / "secondary.csv", index=False)
    res = {"outcomes": outcomes(p, s), **run}
    write_text(out / "outcomes.json", json.dumps(res, indent=2))
    print(p.drop(columns="claim").to_string(index=False), "\n", res["outcomes"], flush=True)
    return 0


# ----------------------------------------------------------------- CLI


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("prereg")
    w.add_argument("--runs", type=Path, required=True)
    w.add_argument("--config", type=Path, default=Path("configs/v4_predev_v2.json"))
    w.add_argument("--selection", type=Path, default=Path("configs/v4_selection_v2.json"))
    w.add_argument("--v3-prereg", type=Path, default=Path("configs/phase3_final.json"))
    for name in ("test", "hybrid", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--prereg-sha", required=True, help="the sha256 of the committed pre-registration")
        p.add_argument("--runs", required=True, help="the v4_runs output holding jobs 4, 5 and 6")
        p.add_argument("--v3-test", required=True, help="V3's phase3_final test output")
        p.add_argument("--test-csv", required=True)
        p.add_argument("--out", required=True)
        if name != "analyze":
            p.add_argument("--job", type=int, required=True)
            p.add_argument("--backbone", required=True, choices=pe.PAIR)
            p.add_argument("--inputs", required=True)
        if name == "test":
            p.add_argument("--parts", required=True)
            p.add_argument("--test-zips", required=True, help="dir holding test/NN.zip")
            p.add_argument("--device", default="auto", help="auto = the job's GPU, or cpu without one")
            p.add_argument("--batch", type=int, help="eval batch (default: the training config's)")
            p.add_argument("--workers", type=int, help="loader workers (default: the training config's)")
            p.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
        if name == "hybrid":
            p.add_argument("--knn-chunk", type=int, default=CHUNK, help="test queries per kNN block")
        if name == "analyze":
            p.add_argument("--missing-as-p1", action="store_true",
                           help="a model that cannot be scored: its hypotheses count with p = 1")
    for p in sub.choices.values():
        p.add_argument("--prereg", type=Path, default=DEFAULT_PATH)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    a = parse_args(argv)
    if a.cmd == "prereg":
        reg = write(a.runs, a.config, a.selection, a.v3_prereg, a.prereg)
        print(f"{a.prereg}: {len(reg['models'])} models, sha256 {prereg.sha256(a.prereg)}; commit it")
        return
    sys.exit({"test": run_test, "hybrid": run_hybrid, "analyze": run_analyze}[a.cmd](a))


if __name__ == "__main__":
    main()
