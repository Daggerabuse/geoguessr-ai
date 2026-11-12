"""Phase 3, development splits only: coordinate prediction from kNN on the frozen embeddings.

Does StreetCLIP's advantage over its matched base (CLIP L/14-336) hold in km, and how does it
change with N? Grid: backbone x ordering (nat, bal) x N, each subset searched once (top-k cosine
neighbours, geo_eval.knn) and three predictors scored from the same neighbours:
  geo_top1    the nearest neighbour's coordinates;
  geo_mean    weighted mean of the k neighbours on the sphere (3D unit vectors), weights the
              similarity or exp((sim - best sim) / tau);
  geo_medoid  the similarity-weighted geographic medoid of the k neighbours (Phase 1's kNN
              coordinates, so its dev scores reproduce Phase 1's).
k (and tau) are chosen by mean GeoScore on dev_seen only; median km is reported, never selected
on. Metrics follow OSV-5M (haversine km, GeoScore = 5000 * exp(-km / 1492.7), accuracy within
1/25/200/750/2500 km); country_top1 is the country of the training image nearest to the predicted
point, a sanity check against Phase 1. Paired StreetCLIP - L/14-336 differences per image
(gaps_<metric>.csv, shrink_<metric>.csv; for km a lower value is better) and of median km
(gaps_median_km.csv) use the cluster bootstrap of Phase 1 (sequences; creators for dev_unseen).

The official test split is never read here: there is no test option, and load_data is always
called without it. A test-level coordinate evaluation needs its own pre-registration first.

  python -m geoguessr_ai.phase3_eval --data <v2 dir> --emb-root <emb dir> --out results/phase3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai.zipq import write_text

CACHE_VERSION = 1
METHOD = "geo"
PREDICTORS = ("top1", "mean", "medoid")
TAUS = (None, 0.1, 0.03, 0.01)  # geo_mean weights: None = the cosine similarity itself
GAP_METRICS = ("geoscore", "acc_25km", "acc_200km", "acc_750km", "acc_2500km", "country_top1", "km")
GROUPS = ("reliable", "head", "middle", "tail", "seen_creator", "unseen_creator")


def predict(pred: str, nb, lat, lon, k: int, tau=None):
    if pred == "top1":
        return g.knn_top1_coords(nb[1], lat, lon)
    if pred == "mean":
        return g.knn_mean_coords(*nb, lat, lon, k, tau)
    return g.knn_coordinates(*nb, lat, lon, k)


def candidates(pred: str, ks) -> list[dict]:
    if pred == "top1":
        return [{"k": 1, "tau": None}]
    return [{"k": k, "tau": t} for k in ks for t in (TAUS if pred == "mean" else (None,))]


def choose(pred: str, nb, sel: pd.DataFrame, lat, lon, ks) -> tuple[dict, list[dict]]:
    """The candidate with the highest mean GeoScore on dev_seen (first one on ties)."""
    sweep = []
    for c in candidates(pred, ks):
        km = g.haversine(sel["latitude"].to_numpy(), sel["longitude"].to_numpy(),
                         *predict(pred, nb, lat, lon, c["k"], c["tau"]))
        sweep.append({**c, "dev_seen_geoscore": float(g.geoscore(km).mean()),
                      "dev_seen_median_km": float(np.median(km))})
    best = max(sweep, key=lambda e: e["dev_seen_geoscore"])
    return {"k": best["k"], "tau": best["tau"]}, sweep


def score_coords(out: Path, name: str, df: pd.DataFrame, coords, country, groups, args,
                 clusters) -> dict:
    """Per-image file {name}_geo_hits.csv and the metrics of every row group."""
    km = g.haversine(df["latitude"].to_numpy(), df["longitude"].to_numpy(), *coords)
    per = {**g.per_image_geo(km),
           "country_top1": (np.asarray(country) == df["label"].to_numpy()).astype(float)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "pred_lat": coords[0], "pred_lon": coords[1], "km": km, **per}) \
        .to_csv(out / f"{name}_geo_hits.csv", index=False)
    clusters = np.asarray(clusters)
    return {grp: g.summarize({k: v[m] for k, v in per.items()}, km[m], clusters[m], args.n_boot,
                             args.seed) for grp, m in groups.items()}


def run_subset(ctx: dict, ordering: str, n: int, args) -> dict[str, dict]:
    """One training subset: one neighbour search per split, then every predictor."""
    dfs, embs, evals = ctx["dfs"], ctx["embs"], ctx["evals"]
    train, sel = dfs["train"], dfs["dev_seen"]
    mask = (train[f"{ordering}_rank"] < n).to_numpy()  # nested subset, see v2_data
    sub, db = train[mask], embs["train"][mask]
    lat, lon, lab = (sub["latitude"].to_numpy(), sub["longitude"].to_numpy(),
                     sub["label"].to_numpy())
    creators = set(sub["creator_id"])
    groups = {s: g.row_groups(dfs[s], ctx["counts"], dfs[s]["creator_id"].isin(creators).to_numpy())
              for s in evals}
    nbrs = {s: g.knn(embs[s], db, max(args.k), device=args.device or None) for s in evals}
    out = {}
    for pred in PREDICTORS:
        d = pe.cell_dir(Path(args.out), ctx["backbone"], ordering, n, METHOD, pred)
        best, sweep = choose(pred, nbrs["dev_seen"], sel, lat, lon, args.k)
        res: dict = {"n_train": int(mask.sum()), **best, "sweep_dev_seen": sweep}
        for s in evals:
            coords = predict(pred, nbrs[s], lat, lon, best["k"], best["tau"])
            res[s] = score_coords(d, s, dfs[s], coords, g.nearest_label(*coords, lat, lon, lab),
                                  groups[s], args, g.clusters_of(s, dfs[s]))
        out[pred] = res
    return out


def cell_row(res: dict, backbone: str, ordering: str, n: int, pred: str, evals) -> dict:
    row = {"backbone": backbone, "ordering": ordering, "n": n, "method": METHOD,
           "predictor": pred, "variant": pe.variant(METHOD, pred), "n_train": res["n_train"],
           "k": res["k"], "tau": res["tau"]}
    for s in evals:
        p, r = pe.SPLIT_NAMES[s], res[s]
        a = r["all"]
        row[f"{p}_n"] = a["n"]
        row[f"{p}_median_km"], row[f"{p}_mean_km"] = a["median_km"], a["mean_km"]
        for m in ("geoscore", "country_top1", *(f"acc_{t}km" for t in g.KM_THRESHOLDS)):
            row[f"{p}_{m}"], row[f"{p}_{m}_lo"], row[f"{p}_{m}_hi"] = (
                a[m]["mean"], a[m]["ci_lo"], a[m]["ci_hi"])
        for grp in GROUPS:
            row[f"{p}_{grp}_n"] = r[grp]["n"]
            row[f"{p}_{grp}_geoscore"] = r[grp]["geoscore"]["mean"] if r[grp]["n"] else np.nan
            row[f"{p}_{grp}_median_km"] = r[grp].get("median_km", np.nan)
    return row


def cell_done(d: Path) -> bool:
    row = pe.read_row(d)
    return row is not None and row["cache_version"] == CACHE_VERSION


def weighted_median(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    cw = np.cumsum(weights)
    return float(sorted_values[np.searchsorted(cw, cw[-1] / 2)])


def median_gap(a: pd.DataFrame, b: pd.DataFrame, clusters: pd.Series, n_boot: int,
               seed: int) -> dict:
    """median(km_a) - median(km_b) over the images both scored, with a paired cluster bootstrap
    (one resample of clusters, as multiplicity weights, applies to both backbones)."""
    m = a.set_index("id")["km"].to_frame("a").join(b.set_index("id")["km"].rename("b"),
                                                  how="inner")
    codes, _ = pd.factorize(m.index.map(clusters))
    if (codes < 0).any():
        raise ValueError("image without a cluster")
    oa, ob = np.argsort(m["a"].to_numpy()), np.argsort(m["b"].to_numpy())
    va, vb = m["a"].to_numpy()[oa], m["b"].to_numpy()[ob]
    n_c, rng = codes.max() + 1, np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n_c, n_c), minlength=n_c)[codes]
        diffs[i] = weighted_median(va, w[oa]) - weighted_median(vb, w[ob])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"median_km_a": float(np.median(va)), "median_km_b": float(np.median(vb)),
            "gap": float(np.median(va) - np.median(vb)), "gap_lo": float(lo), "gap_hi": float(hi),
            "n_images": len(m)}


def median_gaps(out: Path, data: Path, rows: pd.DataFrame, evals, n_boot: int,
                seed: int) -> pd.DataFrame:
    clusters = {}
    for s in evals:
        d = pd.read_csv(data / f"{s}.csv", dtype=str, keep_default_na=False)
        clusters[s] = pd.Series(g.clusters_of(s, d), index=d["id"])
    res = []
    for (ordering, n, var), grp in rows.groupby(["ordering", "n", "variant"]):
        if not set(pe.PAIR) <= set(grp["backbone"]):
            continue
        for s in evals:
            a, b = (pd.read_csv(out / bb / f"{ordering}_{n}_{var}" / f"{s}_geo_hits.csv",
                                dtype={"id": str}) for bb in pe.PAIR)
            res.append({"ordering": ordering, "n": n, "variant": var, "split": pe.SPLIT_NAMES[s],
                        **median_gap(a, b, clusters[s], n_boot, seed)})
    return pd.DataFrame(res)


def compare_phase1(table: pd.DataFrame, phase1: Path) -> str:
    """Largest |geo_medoid - Phase 1 kNN| over dev GeoScore and median km (development rows
    only: a directory holding test results is refused)."""
    diffs = []
    for _, r in table[table["predictor"] == "medoid"].iterrows():
        old = pe.read_row(phase1 / r["backbone"] / f"{r['ordering']}_{r['n']}_knn")
        if old is None:
            continue
        if any(k.startswith("test_") for k in old):
            raise ValueError(f"{phase1} holds test results; compare with the development run only")
        diffs += [abs(r[f"{s}_{m}"] - old[f"{s}_{m}"]) for s in ("dev_seen", "dev_unseen")
                  for m in ("geoscore", "median_km")]
    if not diffs:
        return f"compare-phase1: no matching kNN cells in {phase1}"
    return (f"geo_medoid vs Phase 1 kNN ({len(diffs) // 4} cells): max abs difference "
            f"{max(diffs):.6f} (GeoScore, median km on dev_seen and dev_unseen)")


def run(args: argparse.Namespace) -> pd.DataFrame:
    out, data, emb_root = Path(args.out), Path(args.data), Path(args.emb_root)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("no CUDA device: kNN with numpy on the CPU")
            args.device = ""
    rows = []
    for bb in args.backbones:
        subsets = [(o, n) for o in args.orderings for n in args.sizes]
        todo = [c for c in subsets
                if not all(cell_done(pe.cell_dir(out, bb, *c, METHOD, p)) for p in PREDICTORS)]
        if todo:
            t = time.time()
            ctx = {**g.load_data(data, emb_root / bb, eval_test=False), "backbone": bb}
            if "test" in ctx["evals"]:
                raise AssertionError("phase3_eval never scores the official test split")
            print(f"{bb}: loaded in {time.time() - t:.0f} s; {len(todo)} of {len(subsets)} "
                  f"subsets to run")
            for o, n in todo:
                t = time.time()
                for pred, res in run_subset(ctx, o, n, args).items():
                    d = pe.cell_dir(out, bb, o, n, METHOD, pred)
                    res["seconds"] = round(time.time() - t, 1)
                    write_text(d / "geo_metrics.json", json.dumps(res, indent=2, default=float))
                    write_text(d / "row.json", json.dumps(  # written last: the cell is done
                        {**cell_row(res, bb, o, n, pred, ctx["evals"]), "seconds": res["seconds"],
                         "cache_version": CACHE_VERSION}, default=float))
                print(f"{bb} {o} {n}: {time.time() - t:.0f} s")
            del ctx
        rows += [pe.read_row(pe.cell_dir(out, bb, *c, METHOD, p))
                 for c in subsets for p in PREDICTORS]
    table = pd.DataFrame(rows)
    table.to_csv(out / "results.csv", index=False)
    evals = ("dev_seen", "dev")
    for m in GAP_METRICS:
        gp, sh = pe.gaps(out, data, table, evals, args.n_boot, args.seed, stem="geo_hits", col=m)
        gp.to_csv(out / f"gaps_{m}.csv", index=False)
        sh.to_csv(out / f"shrink_{m}.csv", index=False)
        if m == "geoscore" and not gp.empty:
            pe.plot_gap(gp, out / "gap_geo.png", "GeoScore difference (paired)")
            print(gp.to_string())
            print(sh.to_string())
    mg = median_gaps(out, data, table, evals, args.n_boot, args.seed)
    mg.to_csv(out / "gaps_median_km.csv", index=False)
    pe.plot_scaling(table, out / "scaling_geo.png", [pe.SPLIT_NAMES[s] for s in evals],
                    "geoscore", "GeoScore")
    cols = ["backbone", "ordering", "n", "predictor", "k", "tau", "dev_seen_geoscore",
            "dev_seen_median_km", "dev_unseen_geoscore", "dev_unseen_median_km"]
    print(table[cols].to_string(index=False))
    if args.compare_phase1:
        print(compare_phase1(table, Path(args.compare_phase1)))
    return table


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="v2_data.py output dir")
    ap.add_argument("--emb-root", required=True, help="dir with one embed.py cache per backbone")
    ap.add_argument("--out", default="results/phase3")
    ap.add_argument("--backbones", nargs="+", default=list(pe.BACKBONES))
    ap.add_argument("--orderings", nargs="+", choices=["nat", "bal"], default=["nat", "bal"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[50_000, 100_000, 200_000, 500_000])
    ap.add_argument("--k", type=int, nargs="+", default=list(g.DEFAULT_K))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda", help="'' = CPU (numpy kNN)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-phase1", help="Phase 1 development output (phase1_eval dir): "
                    "check that geo_medoid reproduces its kNN dev coordinates")
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
