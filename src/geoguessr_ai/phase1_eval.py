"""Phase 1 analysis: does geographic pretraining (StreetCLIP) improve sample efficiency over its
matched base (CLIP L/14-336) and B/16, and does the gap close as N grows?

Grid: backbone x ordering (nat, bal) x N x method (kNN, linear probe) x probe_init (random,
zeroshot = the backbone's CLIP text embeddings of the country prompts; random separates the
image representation from the text encoder). Each cell trains on the nested subset
(v2_data.py), chooses its hyperparameters on dev_seen only (k; the probe's learning rate and
stopping epoch, jointly) and is scored on dev_seen and dev_unseen (dev.csv) by
geo_eval.run_cell. The official test is
read only with --eval-test (into a separate results directory).

Cells are cached: <out>/<backbone>/<ordering>_<N>_<variant>/row.json is written last, and a
rerun skips every cell whose row.json carries the method's current CACHE_VERSION (a backbone
whose cells are all done is not even loaded). --seed-from copies such finished cells from an
earlier output first (e.g. the previous Kaggle run attached as input).

Probe search: --probe-lrs x epochs 0..--probe-epochs, a run stopped after --probe-patience
epochs without a dev_seen gain (this script's defaults differ from geo_eval's); results.csv's
grid_edge column flags a choice at the edge of the search grid.

Outputs in <out>:
  results.csv   one row per cell: chosen k / lr / epoch; per split top-1/top-5 with intervals,
                macro top-1 over the split's fixed class set (every country of the split,
                each with its actual accuracy; comparable across N), macro top-1 over the
                classes in the training subset (secondary), top-1 on images whose country is
                absent from the training subset, head/middle/tail and reliable top-1
  gaps.csv      paired StreetCLIP - L/14-336 top-1 difference per ordering x N x variant x split
                (same images; cluster bootstrap: sequences, creators for dev_unseen)
  shrink.csv    gap(smallest N) - gap(largest N) per ordering x variant x split, paired the same
                way (same images, same cluster resamples): does the gap shrink with N?
  scaling.png   top-1 vs N (log x), 95% bands, one panel per variant x ordering
  gap.png       the paired difference vs N with its band, one panel per variant x ordering
(variant: knn, probe_random, probe_zeroshot)
Intervals: geo_eval.cluster_bootstrap (sequence clusters; creator clusters for dev_unseen).

Usage (Kaggle; inputs from the full run, see scripts/kaggle_phase1_eval.sh):
  python -m geoguessr_ai.phase1_eval --data <in>/v2 --emb-root <in>/emb --out /kaggle/working/phase1_eval
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai.zipq import write_text

BACKBONES = ("streetclip", "clip_l14_336", "clip_b16")
PAIR = ("streetclip", "clip_l14_336")  # geographic pretraining vs its matched base
SPLIT_NAMES = {"dev_seen": "dev_seen", "dev": "dev_unseen", "test": "test"}
GROUPS = ("reliable", "head", "middle", "tail")
# bump a method's version when its search or scoring changes: only that method's cells rerun
# (rows written before versioning count as 1)
CACHE_VERSION = {"knn": 1, "probe": 3}
PROBE_LRS, PROBE_EPOCHS, PROBE_PATIENCE = [1e-3, 3e-3, 1e-2, 3e-2], 60, 8


def variant(method: str, init: str | None) -> str:
    return method if method == "knn" else f"{method}_{init}"


def cell_dir(out: Path, backbone: str, ordering: str, n: int, method: str,
             init: str | None = None) -> Path:
    return out / backbone / f"{ordering}_{n}_{variant(method, init)}"


def read_row(d: Path) -> dict | None:
    f = d / "row.json"
    return {"cache_version": 1, **json.loads(f.read_text())} if f.exists() else None


def cell_done(d: Path, method: str) -> bool:
    row = read_row(d)
    return row is not None and row["cache_version"] == CACHE_VERSION[method]


def seed_cells(src: Path, out: Path, backbone: str, cells) -> int:
    """Copy cells finished at the current version from src (row.json last, as when written)."""
    copied = 0
    for c in cells:
        s, d = cell_dir(src, backbone, *c), cell_dir(out, backbone, *c)
        if cell_done(d, c[2]) or not cell_done(s, c[2]):
            continue
        # copyfile: no mode bits (the Kaggle input is read-only)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns("row.json"), dirs_exist_ok=True,
                        copy_function=shutil.copyfile)
        shutil.copyfile(s / "row.json", d / "row.json")
        copied += 1
    return copied


def grid_edge(row, args) -> str:
    """';'-joined flags of a choice at the edge of its search grid ('' = interior)."""
    if row["method"] == "knn":
        ks = sorted(args.k)
        return {ks[0]: "k_min", ks[-1]: "k_max"}.get(row["k"], "") if len(ks) > 1 else ""
    lrs, flags = sorted(args.probe_lrs), []
    if len(lrs) > 1 and row["lr"] <= lrs[0]:
        flags.append("lr_low")
    if len(lrs) > 1 and row["lr"] >= lrs[-1]:
        flags.append("lr_high")
    # best epoch in the final 10% of a run that used them all: more epochs might still help
    if row["epoch"] > 0.9 * args.probe_epochs and row.get("stopped_early") != True:  # noqa: E712 (NaN, numpy bools)
        flags.append("last_epoch")
    return ";".join(flags)


def edge_summary(table: pd.DataFrame, args) -> str:
    p, k = table[table["method"] == "probe"], table[table["method"] == "knn"]
    has = lambda t, f: int(t["grid_edge"].str.contains(f).sum())
    stopped = int((p["stopped_early"] == True).sum())  # noqa: E712 (NaN in old rows)
    return (f"grid edges: probe {len(p)} cells: lr_low {has(p, 'lr_low')}, lr_high "
            f"{has(p, 'lr_high')}, last_epoch {has(p, 'last_epoch')} (stopped early: {stopped}); "
            f"knn {len(k)} cells: k_min {has(k, 'k_min')}, k_max {has(k, 'k_max')}")


def random_head(n_classes: int, dim: int, seed: int):
    import torch

    return torch.randn(n_classes, dim, generator=torch.Generator().manual_seed(seed))


def cell_row(res: dict, backbone: str, ordering: str, n: int, method: str, init, evals) -> dict:
    row = {"backbone": backbone, "ordering": ordering, "n": n, "method": method,
           "probe_init": init or "", "variant": variant(method, init), "n_train": res["n_train"],
           "k": res.get("k_country"), "lr": res.get("best_lr"), "epoch": res.get("best_epoch"),
           "stopped_early": res.get("stopped_early")}
    for s in evals:
        p, r = SPLIT_NAMES[s], res[s]
        row[f"{p}_n"] = r["all"]["n"]
        for m in ("top1", "top5"):
            v = r["all"][f"country_{m}"]
            row[f"{p}_{m}"], row[f"{p}_{m}_lo"], row[f"{p}_{m}_hi"] = v["mean"], v["ci_lo"], v["ci_hi"]
        row[f"{p}_macro_top1"] = r["macro_top1_fixed"]
        row[f"{p}_macro_top1_seen_classes"] = r["class_in_subset"].get("macro_top1")
        absent = r["class_not_in_subset"]  # images whose country has no training image
        row[f"{p}_absent_class_n"] = absent["n"]
        row[f"{p}_absent_class_top1"] = absent["country_top1"]["mean"] if absent["n"] else np.nan
        for grp in GROUPS:
            row[f"{p}_{grp}_top1"] = (r[grp]["country_top1"]["mean"] if r[grp].get("n")
                                      else np.nan)
    return row


def hit_diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """top1_a - top1_b per image id, over the images both scored."""
    return (a.set_index("id")["top1"] - b.set_index("id")["top1"]).dropna()


def mean_ci(d: pd.Series, clusters: pd.Series, n_boot: int, seed: int) -> tuple[float, float]:
    """Cluster-bootstrap interval of d's mean; the same seed and image set give the same
    resamples for every quantity, so differences between them stay paired."""
    return g.cluster_bootstrap(d.to_numpy(dtype=float), d.index.map(clusters).to_numpy(),
                               n_boot, seed)


def paired_gap(a: pd.DataFrame, b: pd.DataFrame, clusters: pd.Series, n_boot: int,
               seed: int) -> dict:
    """mean(top1_a - top1_b) over the images both scored, with a paired cluster bootstrap
    (one resample of clusters applies to both backbones). clusters: id -> cluster."""
    d = hit_diff(a, b)
    lo, hi = mean_ci(d, clusters, n_boot, seed)
    return {"gap": float(d.mean()), "gap_lo": lo, "gap_hi": hi, "n_images": len(d)}


def shrink(d_small: pd.Series, d_large: pd.Series, clusters: pd.Series, n_boot: int,
           seed: int) -> dict:
    """gap(small N) - gap(large N) from per-image paired differences, over the images both
    cells scored; positive = the gap shrinks as N grows."""
    x = (d_small - d_large).dropna()
    lo, hi = mean_ci(x, clusters, n_boot, seed)
    return {"gap_small": float(d_small[x.index].mean()), "gap_large": float(d_large[x.index].mean()),
            "shrink": float(x.mean()), "shrink_lo": lo, "shrink_hi": hi, "n_images": len(x)}


def gaps(out: Path, data: Path, rows: pd.DataFrame, evals, n_boot: int,
         seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gaps, shrink) tables, see the module docstring."""
    clusters = {}
    for s in evals:
        d = pd.read_csv(data / f"{s}.csv", dtype=str, keep_default_na=False)
        clusters[s] = pd.Series(g.clusters_of(s, d), index=d["id"])
    have = set(rows["backbone"])
    out_rows, shrink_rows, diffs = [], [], {}
    if not set(PAIR) <= have:
        return pd.DataFrame(out_rows), pd.DataFrame(shrink_rows)
    for (ordering, n, var), _ in rows.groupby(["ordering", "n", "variant"]):
        for s in evals:
            a, b = (pd.read_csv(out / bb / f"{ordering}_{n}_{var}" / f"{s}_hits.csv",
                                dtype={"id": str}) for bb in PAIR)
            diffs[ordering, var, s, n] = hit_diff(a, b)
            out_rows.append({"ordering": ordering, "n": n, "variant": var,
                             "split": SPLIT_NAMES[s],
                             **paired_gap(a, b, clusters[s], n_boot, seed)})
    for (ordering, var), grp in rows.groupby(["ordering", "variant"]):
        small, large = grp["n"].min(), grp["n"].max()
        if small == large:
            continue
        for s in evals:
            shrink_rows.append({"ordering": ordering, "variant": var, "split": SPLIT_NAMES[s],
                                "n_small": small, "n_large": large,
                                **shrink(diffs[ordering, var, s, small],
                                         diffs[ordering, var, s, large], clusters[s], n_boot, seed)})
    return pd.DataFrame(out_rows), pd.DataFrame(shrink_rows)


def plot_scaling(rows: pd.DataFrame, path: Path, splits) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods, orderings = sorted(rows["variant"].unique()), sorted(rows["ordering"].unique())
    fig, axes = plt.subplots(len(methods), len(orderings), squeeze=False, sharey=True,
                             figsize=(5.5 * len(orderings), 4 * len(methods)))
    for i, m in enumerate(methods):
        for j, o in enumerate(orderings):
            ax = axes[i][j]
            for bb, color in zip(BACKBONES, ("C0", "C1", "C2")):
                d = rows[(rows["backbone"] == bb) & (rows["variant"] == m)
                         & (rows["ordering"] == o)].sort_values("n")
                for s, style in zip(splits, ("-", "--", ":")):
                    if d.empty or f"{s}_top1" not in d:
                        continue
                    ax.plot(d["n"], d[f"{s}_top1"], style, color=color, marker="o",
                            label=f"{bb} {s}")
                    ax.fill_between(d["n"], d[f"{s}_top1_lo"], d[f"{s}_top1_hi"], color=color,
                                    alpha=0.15)
            ax.set_xscale("log")
            ax.set_title(f"{m}, {o}")
            ax.set_xlabel("training images N")
            ax.set_ylabel("country top-1")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gap(gp: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods, orderings = sorted(gp["variant"].unique()), sorted(gp["ordering"].unique())
    fig, axes = plt.subplots(len(methods), len(orderings), squeeze=False, sharey=True,
                             figsize=(5.5 * len(orderings), 4 * len(methods)))
    for i, m in enumerate(methods):
        for j, o in enumerate(orderings):
            ax = axes[i][j]
            for k, (s, d) in enumerate(gp[(gp["variant"] == m) & (gp["ordering"] == o)]
                                       .sort_values("n").groupby("split")):
                ax.plot(d["n"], d["gap"], marker="o", color=f"C{k}", label=s)
                ax.fill_between(d["n"], d["gap_lo"], d["gap_hi"], color=f"C{k}", alpha=0.2)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_title(f"{PAIR[0]} - {PAIR[1]}: {m}, {o}")
            ax.set_xlabel("training images N")
            ax.set_ylabel("top-1 difference (paired)")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(args: argparse.Namespace) -> pd.DataFrame:
    out, data, emb_root = Path(args.out), Path(args.data), Path(args.emb_root)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("no CUDA device: kNN with numpy, probe with torch on the CPU")
            args.device = ""
    if args.eval_test:
        out = out.with_name(out.name + "_test")  # never mixed with development results
    rows = []
    for bb in args.backbones:
        cells = [(o, n, m, i) for o in args.orderings for n in args.sizes for m in args.methods
                 for i in ([None] if m == "knn" else args.probe_inits)]
        if args.seed_from:
            print(f"{bb}: {seed_cells(Path(args.seed_from), out, bb, cells)} cells copied "
                  f"from {args.seed_from}")
        todo = [c for c in cells if not cell_done(cell_dir(out, bb, *c), c[2])]
        if todo:
            t = time.time()
            ctx = g.load_data(data, emb_root / bb, args.eval_test)
            print(f"{bb}: loaded in {time.time() - t:.0f} s; {len(todo)} of {len(cells)} cells to run")
            heads = {}
            if any(c[3] == "zeroshot" for c in todo):
                model = json.loads((emb_root / bb / "encoder.json").read_text())["model"]
                heads["zeroshot"] = g.text_head(model, ctx["classes"], args.template)
            if any(c[3] == "random" for c in todo):
                heads["random"] = random_head(len(ctx["classes"]), ctx["embs"]["train"].shape[1],
                                              args.seed)
            for c in todo:
                d, t = cell_dir(out, bb, *c), time.time()
                res = g.run_cell(ctx, c[0], c[1], c[2], d, args, heads.get(c[3]))
                res["seconds"] = round(time.time() - t, 1)
                write_text(d / "geo_metrics.json", json.dumps(res, indent=2, default=float))
                write_text(d / "row.json", json.dumps(  # written last: the cell is done
                    {**cell_row(res, bb, *c, ctx["evals"]), "seconds": res["seconds"],
                     "cache_version": CACHE_VERSION[c[2]]}, default=float))
                print(f"{bb} {c}: {res['seconds']:.0f} s")
            del ctx
        rows += [read_row(cell_dir(out, bb, *c)) for c in cells]
    table = pd.DataFrame(rows)
    table["grid_edge"] = table.apply(grid_edge, axis=1, args=(args,))
    table.to_csv(out / "results.csv", index=False)
    evals = ("dev_seen", "dev", "test") if args.eval_test else ("dev_seen", "dev")
    gp, sh = gaps(out, data, table, evals, args.n_boot, args.seed)
    gp.to_csv(out / "gaps.csv", index=False)
    sh.to_csv(out / "shrink.csv", index=False)
    plot_scaling(table, out / "scaling.png", [SPLIT_NAMES[s] for s in evals])
    if not gp.empty:
        plot_gap(gp, out / "gap.png")
        print(gp.to_string())
        print(sh.to_string())
    print(edge_summary(table, args))
    return table


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="v2_data.py output dir")
    ap.add_argument("--emb-root", required=True, help="dir with one embed.py cache per backbone")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--orderings", nargs="+", choices=["nat", "bal"], default=["nat", "bal"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[50_000, 100_000, 200_000, 500_000])
    ap.add_argument("--methods", nargs="+", choices=["knn", "probe"], default=["knn", "probe"])
    ap.add_argument("--probe-inits", nargs="+", choices=["random", "zeroshot"],
                    default=["random", "zeroshot"])
    ap.add_argument("--seed-from", help="earlier output dir: copy its finished, current-version "
                    "cells before running")
    g.add_head_args(ap)
    ap.set_defaults(probe_lrs=PROBE_LRS, probe_epochs=PROBE_EPOCHS, probe_patience=PROBE_PATIENCE)
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
