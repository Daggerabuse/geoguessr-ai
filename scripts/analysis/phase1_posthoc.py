"""Phase 1, POST-HOC analyses requested by the independent review (none was pre-registered; they
do not replace the pre-registered result in phase1_primary.csv). Test is read only through the
saved per-image top-1 hits (test_hits.csv) and top-1 columns; no test distance column is read.

  1  gap table: test top-1 gap (StreetCLIP - L/14-336) per variant x ordering x N
  2  creator-clustered intervals for the 12 primary hypotheses (the pre-registered bootstrap
     resamples sequences; errors are creator-correlated). The sequence version is recomputed
     with the same resamples as a check that it reproduces phase1_primary.csv.
  3  hyperparameter-matched kNN: one k per ordering x N for both backbones, the k with the
     highest pair-average dev_seen top-1 (the stored dev k sweep; ties -> smaller k). dev_seen
     gaps come from the sweep. A test gap is unchanged where both backbones already used that
     k; otherwise both cells are recomputed on CPU in fp32 at the shared and the frozen k, which
     needs the embeddings: on Kaggle (--data/--emb-root/--test-emb-root, no --run; called by
     kaggle_phase3.sh) it writes phase1_matched_k_cpu.csv.gz (per image: predicted country and
     top-1 hit) and the dev agreement with the saved GPU fp16 predictions of --dev-run; locally,
     --run ... --cpu-preds <that file> adds the test gap and the test agreement.
  4  N-equivalence: the N at which L/14-336 reaches StreetCLIP's 50K test top-1, by piecewise
     log-linear interpolation over the 4 measured N, with a cluster-bootstrap interval
     (sequence and creator clusters). inf = above the measured range (L/14 at 500K is still
     below StreetCLIP at 50K).

  uv run python scripts/analysis/phase1_posthoc.py --run runs/phase1_final \
      --dev-run runs/phase1_eval_v3/phase1_eval --test-csv data/raw/test.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai.phase1_eval import PAIR, holm, one_sided_p
from geoguessr_ai.v2_data import normalize_creators

NS = (50_000, 100_000, 200_000, 500_000)
ORDERINGS = ("bal", "nat")
VARIANTS = ("knn", "probe_random", "probe_zeroshot")


def boot(v: np.ndarray, clusters, n_boot: int, seed: int, chunk: int = 200) -> np.ndarray:
    """geo_eval.bootstrap_means for every column of v (images x m) with the SAME resamples
    (same random stream, so column j equals bootstrap_means(v[:, j], ...)): n_boot x m."""
    codes, _ = pd.factorize(pd.Series(clusters))
    sums = np.stack([np.bincount(codes, weights=v[:, j]) for j in range(v.shape[1])], axis=1)
    counts = np.bincount(codes).astype(float)
    rng, out = np.random.default_rng(seed), []
    for start in range(0, n_boot, chunk):
        w = np.stack([np.bincount(rng.integers(0, len(sums), len(sums)), minlength=len(sums))
                      for _ in range(min(chunk, n_boot - start))])
        out.append((w @ sums) / (w @ counts)[:, None])
    return np.concatenate(out)


def load_hits(ev: Path) -> pd.DataFrame:
    """Test top-1 hits, one column per (backbone, ordering, n, variant), rows = test ids."""
    cols = {}
    for bb in PAIR:
        for o in ORDERINGS:
            for n in NS:
                for v in VARIANTS:
                    h = pd.read_csv(ev / bb / f"{o}_{n}_{v}" / "test_hits.csv",
                                    usecols=["id", "top1"], dtype={"id": str})
                    cols[(bb, o, n, v)] = h.set_index("id")["top1"]
    hits = pd.DataFrame(cols)
    assert not hits.isna().any().any(), "cells scored different test images"
    return hits


def test_clusters(test_csv: Path, ids: pd.Index) -> pd.DataFrame:
    d = pd.read_csv(test_csv, usecols=["id", "sequence", "creator_id"], dtype=str,
                    keep_default_na=False)
    d = normalize_creators(d).set_index("id").loc[ids]
    return d[["sequence", "creator_id"]]


def n_equiv(target: float, accs) -> float:
    """N where the piecewise log-linear curve through (NS, accs) first reaches target;
    NS[0] if already reached at the smallest N, inf if never reached in the range."""
    if target <= accs[0]:
        return float(NS[0])
    x = np.log(NS)
    for i in range(len(NS) - 1):
        a, b = accs[i], accs[i + 1]
        if a < target <= b:
            return float(np.exp(x[i] + (target - a) / (b - a) * (x[i + 1] - x[i])))
    return float("inf")


def quantiles(v: np.ndarray) -> tuple[float, float]:
    """2.5% / 97.5% order statistics (inf-safe: inf sorts last)."""
    s = np.sort(v)
    return float(s[int(0.025 * len(s))]), float(s[int(np.ceil(0.975 * len(s))) - 1])


def gap_table(hits: pd.DataFrame) -> pd.DataFrame:
    rows = [{"variant": v, "ordering": o, "n": n,
             "gap": hits[(PAIR[0], o, n, v)].mean() - hits[(PAIR[1], o, n, v)].mean()}
            for v in VARIANTS for o in ORDERINGS for n in NS]
    return pd.DataFrame(rows).pivot(index=["variant", "ordering"], columns="n", values="gap")


def creator_cis(hits, cl, hyps, n_boot, seed) -> pd.DataFrame:
    def x(h):
        def d(n):
            return hits[(PAIR[0], h["ordering"], n, h["variant"])] \
                - hits[(PAIR[1], h["ordering"], n, h["variant"])]
        return d(h["n"]) if h["quantity"] == "gap" else d(h["n_small"]) - d(h["n_large"])
    v = np.stack([x(h).to_numpy(float) for h in hyps], axis=1)
    out = pd.DataFrame({"id": [h["id"] for h in hyps], "estimate": v.mean(0)})
    for name, col in (("seq", "sequence"), ("creator", "creator_id")):
        means = boot(v, cl[col].to_numpy(), n_boot, seed)
        lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
        p = [one_sided_p(means[:, j]) for j in range(v.shape[1])]
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"], out[f"{name}_p"] = lo, hi, p
        out[f"{name}_p_holm"] = holm(p)
    out["n_creators"], out["n_sequences"] = cl["creator_id"].nunique(), cl["sequence"].nunique()
    return out


def n_equivalence(hits, cl, n_boot, seed) -> pd.DataFrame:
    rows = []
    for v in ("knn", "probe_random", "probe_zeroshot"):
        for o in ORDERINGS:
            keys = [(PAIR[0], o, NS[0], v)] + [(PAIR[1], o, n, v) for n in NS]
            m = hits[keys].to_numpy(float)
            acc = m.mean(0)
            row = {"variant": v, "ordering": o, "streetclip_50k": acc[0],
                   **{f"l14_{n // 1000}k": a for n, a in zip(NS, acc[1:])},
                   "n_equiv": n_equiv(acc[0], acc[1:])}
            for name, col in (("seq", "sequence"), ("creator", "creator_id")):
                b = boot(m, cl[col].to_numpy(), n_boot, seed)
                ne = np.array([n_equiv(r[0], r[1:]) for r in b])
                row[f"{name}_lo"], row[f"{name}_hi"] = quantiles(ne)
                row[f"{name}_share_above_range"] = float(np.isinf(ne).mean())
            rows.append(row)
    t = pd.DataFrame(rows)
    t["ratio_to_50k"] = t["n_equiv"] / NS[0]
    return t


def matched_k(cfg: dict, dev_run: Path, hits: pd.DataFrame | None) -> pd.DataFrame:
    frozen = {(c["backbone"], c["ordering"], c["n"]): c["k_country"]
              for c in cfg["cells"] if c["variant"] == "knn"}
    rows = []
    for o in ORDERINGS:
        for n in NS:
            sw = {bb: {int(k): r["dev_seen_top1"] for k, r in json.loads(
                (dev_run / bb / f"{o}_{n}_knn" / "geo_metrics.json").read_text())
                ["k_sweep_dev_seen"].items()} for bb in PAIR}
            ks = sorted(sw[PAIR[0]])
            k = max(ks, key=lambda k: (np.mean([sw[bb][k] for bb in PAIR]), -k))
            fa, fb = frozen[(PAIR[0], o, n)], frozen[(PAIR[1], o, n)]
            gap = np.nan if hits is None else (hits[(PAIR[0], o, n, "knn")].mean()
                                               - hits[(PAIR[1], o, n, "knn")].mean())
            same = fa == fb == k
            rows.append({"ordering": o, "n": n, "k_streetclip": fa, "k_l14": fb, "k_shared": k,
                         "dev_seen_gap_frozen": sw[PAIR[0]][fa] - sw[PAIR[1]][fb],
                         "dev_seen_gap_matched": sw[PAIR[0]][k] - sw[PAIR[1]][k],
                         "test_gap_frozen": gap, "test_gap_matched": gap if same else np.nan,
                         "recompute": not same})
    return pd.DataFrame(rows)


def cpu_predictions(t: pd.DataFrame, args) -> pd.DataFrame:
    """Kaggle step: the kNN cells of t that need it, per backbone at the shared k and at its
    frozen k, CPU fp32 (geo_eval.knn, device None, dtype fp32), on dev_seen, dev and test.
    Per image: predicted country and top-1 hit only (no coordinate is predicted or read)."""
    out = []
    for bb in PAIR:
        ctx = g.load_data(args.data, args.emb_root / bb, True, args.test_emb_root / bb)
        train, classes = ctx["dfs"]["train"], np.array(ctx["classes"])
        for r in t[t["recompute"]].itertuples():
            ks = sorted({r.k_shared, (r.k_streetclip, r.k_l14)[PAIR.index(bb)]})
            m = (train[f"{r.ordering}_rank"] < r.n).to_numpy()
            db, lab = ctx["embs"]["train"][m], train["label"].to_numpy()[m]
            for s in ctx["evals"]:
                nb = g.knn(ctx["embs"][s], db, max(ks), dtype="fp32")
                for k in ks:
                    pred = g.knn_country_scores(*nb, lab, len(classes), k).argmax(1)
                    out.append(pd.DataFrame({
                        "backbone": bb, "ordering": r.ordering, "n": r.n, "k": k, "split": s,
                        "id": ctx["dfs"][s]["id"].to_numpy(), "pred": classes[pred],
                        "top1": (pred == ctx["dfs"][s]["label"].to_numpy()).astype(int)}))
            print(f"{bb} {r.ordering} {r.n}: k {ks} done", flush=True)
        del ctx
    return pd.concat(out, ignore_index=True)


def merge_cpu(t: pd.DataFrame, cpu: pd.DataFrame, roots: dict) -> pd.DataFrame:
    """Matched gaps from the CPU predictions, and per backbone the share of images whose CPU
    fp32 prediction at the frozen k equals the saved (GPU fp16) one. roots: split -> the
    phase1_eval dir holding that split's saved {split}_predictions.csv."""
    t = t.copy()
    for r in t[t["recompute"]].itertuples():
        c = cpu[(cpu["ordering"] == r.ordering) & (cpu["n"] == r.n)]
        for s in sorted(set(c["split"])):
            top = {bb: c[(c["backbone"] == bb) & (c["k"] == r.k_shared) & (c["split"] == s)]
                   ["top1"].mean() for bb in PAIR}
            t.loc[r.Index, f"{s}_gap_matched_cpu"] = top[PAIR[0]] - top[PAIR[1]]
            if s == "test":
                t.loc[r.Index, "test_gap_matched"] = top[PAIR[0]] - top[PAIR[1]]
            for bb, k in zip(PAIR, (r.k_streetclip, r.k_l14)):
                if s not in roots:
                    continue
                f = roots[s] / bb / f"{r.ordering}_{r.n}_knn" / f"{s}_predictions.csv"
                saved = pd.read_csv(f, dtype={"id": str}, usecols=["id", "pred"],
                                    keep_default_na=False)
                mine = c[(c["backbone"] == bb) & (c["k"] == k) & (c["split"] == s)]
                both = mine.merge(saved, on="id", suffixes=("_cpu", "_saved"))
                t.loc[r.Index, f"{s}_agree_{bb}_k{k}"] = float(
                    (both["pred_cpu"] == both["pred_saved"]).mean())
                t.loc[r.Index, f"{s}_agree_n"] = len(both)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, help="downloaded final eval output (local analyses)")
    ap.add_argument("--dev-run", type=Path, required=True, help="phase1_eval dir of dev_results")
    ap.add_argument("--test-csv", type=Path, help="official OSV-5M test.csv (with --run)")
    ap.add_argument("--config", type=Path, default=Path("configs/phase1_final.json"))
    ap.add_argument("--out", type=Path, default=Path("results/v2"))
    ap.add_argument("--n-boot-neq", type=int, default=2000)
    ap.add_argument("--data", type=Path, help="(Kaggle) v2 split dir, for the matched-k recompute")
    ap.add_argument("--emb-root", type=Path, help="(Kaggle) train embeddings, one dir per backbone")
    ap.add_argument("--test-emb-root", type=Path, help="(Kaggle) test embeddings (emb_test)")
    ap.add_argument("--cpu-preds", type=Path, help="the Kaggle step's phase1_matched_k_cpu.csv.gz")
    args = ap.parse_args()
    kaggle = bool(args.data and args.emb_root and args.test_emb_root)
    if not (kaggle or args.run):
        ap.error("give --run (local analyses) or --data, --emb-root and --test-emb-root (Kaggle)")
    if args.run and not args.test_csv:
        ap.error("--run needs --test-csv")
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200)
    cfg = json.loads(args.config.read_text())
    hyps, seed = cfg["hypotheses"], cfg["settings"]["seed"]
    args.out.mkdir(parents=True, exist_ok=True)
    roots = {"dev_seen": args.dev_run, "dev": args.dev_run}
    hits = None
    if args.run:
        roots["test"] = args.run / "phase1_eval_test"
        hits = load_hits(roots["test"])
        cl = test_clusters(args.test_csv, hits.index)
        gt = gap_table(hits)
        gt.to_csv(args.out / "phase1_posthoc_gap_table.csv")
        print("1  test top-1 gap by N (pt)\n", (100 * gt).round(2))

        ci = creator_cis(hits, cl, hyps["primary"], hyps["n_boot"], seed)
        prim = pd.read_csv(args.out / "phase1_primary.csv").set_index("id").loc[ci["id"]]
        for a, b in (("estimate", "estimate"), ("seq_ci_lo", "ci_lo"), ("seq_ci_hi", "ci_hi"),
                     ("seq_p_holm", "p_holm")):
            assert np.allclose(ci[a].to_numpy(), prim[b].to_numpy()), f"{a} does not reproduce {b}"
        ci.to_csv(args.out / "phase1_posthoc_creator_ci.csv", index=False)
        show = ci[["id", "estimate", "seq_ci_lo", "seq_ci_hi", "creator_ci_lo", "creator_ci_hi",
                   "creator_p_holm"]].copy()
        show.iloc[:, 1:6] = (100 * show.iloc[:, 1:6]).round(2)
        print("2  creator-clustered CIs (pt; sequence CIs reproduce phase1_primary.csv)\n",
              show.to_string(index=False))

    mk = matched_k(cfg, args.dev_run, hits)
    cpu = None
    if kaggle:
        cpu = cpu_predictions(mk, args)
        cpu.to_csv(args.out / "phase1_matched_k_cpu.csv.gz", index=False)
    elif args.cpu_preds:
        cpu = pd.read_csv(args.cpu_preds, dtype={"id": str}, keep_default_na=False)
    if cpu is not None:
        mk = merge_cpu(mk, cpu, roots)
    mk.to_csv(args.out / "phase1_posthoc_matched_k.csv", index=False)
    print("3  matched-k kNN\n", mk.round(4).to_string(index=False))

    if args.run:
        ne = n_equivalence(hits, cl, args.n_boot_neq, seed)
        ne.to_csv(args.out / "phase1_posthoc_n_equivalence.csv", index=False)
        print("4  N-equivalence\n", ne.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
