"""Phase 3 step 2, DEVELOPMENT only (no test km / GeoScore is read): geocell probes vs the
Step 1 kNN medoid, and the backbone gaps of the cell probes.

  results/v3/phase3_cell_init_gap.csv      zero-shot - random init, per backbone x ordering x N
                                           x split (paired, same images)
  results/v3/phase3_cell_backbone_gap.csv  StreetCLIP - L/14-336 and StreetCLIP - B/16 per init
                                           x ordering x N x split
  results/v3/phase3_cell_best.csv          per backbone x ordering x N: the best of geo_medoid /
                                           cell_random / cell_zeroshot by dev_seen GeoScore, and
                                           its paired gap vs geo_medoid. kNN is near-duplicate
                                           masked, the cell probe trains on every subset image:
                                           not a like-for-like comparison
  results/v3/phase3_cell_scaling.png, phase3_cell_gap.png

Gaps: mean per-image difference of geoscore (plus acc_25km, acc_750km), paired cluster bootstrap
(phase1_eval.paired_gap: sequences on dev_seen, creators on dev_unseen). --data (the v2 dir with
dev_seen.csv and dev.csv) gives the clusters; without it the intervals are NaN.

  uv run python scripts/analysis/phase3_cell_summary.py --run runs/phase3_session_c/phase3_eval \
      --knn-run runs/phase3_dev/phase3_eval --data <v2 dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai import phase1_eval as pe

SPLITS = {"dev_seen": "dev_seen", "dev": "dev_unseen"}  # file prefix -> report name
COLS = ("geoscore", "acc_25km", "acc_750km")
KEYS = ["backbone", "ordering", "n"]
CELLS, MEDOID = ("geo_cell_random", "geo_cell_zeroshot"), "geo_medoid"
REF = "streetclip"
OTHERS = ("clip_l14_336", "clip_b16")


def load_rows(path: Path, variants) -> pd.DataFrame:
    """Every finished cell's row.json (results.csv lists only the last session's cells)."""
    t = pd.DataFrame([json.loads(f.read_text()) for bb in pe.BACKBONES for v in variants
                      for f in sorted((path / bb).glob(f"*_{v}/row.json"))])
    if t.filter(regex="^test_").size:
        raise ValueError(f"{path} holds test columns; this script is development-only")
    return t


def gap(a_dir: Path, b_dir: Path, split: str, clusters, n_boot: int, seed: int) -> dict:
    a, b = (pd.read_csv(d / f"{split}_geo_hits.csv", dtype={"id": str}) for d in (a_dir, b_dir))
    out = {}
    for c in COLS:
        if clusters is None:
            d = pe.hit_diff(a, b, c)
            r = {"gap": float(d.mean()), "gap_lo": np.nan, "gap_hi": np.nan, "n_images": len(d)}
        else:
            r = pe.paired_gap(a, b, clusters[split], n_boot, seed, c)
        out |= {(f"{k}" if c == "geoscore" else f"{c}_{k}"): v for k, v in r.items()
                if c == "geoscore" or k != "n_images"}
    return out


def plot_gaps(bg: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orderings, splits = sorted(bg["ordering"].unique()), list(SPLITS.values())
    fig, axes = plt.subplots(len(orderings), len(splits), squeeze=False, sharey=True,
                             figsize=(5.5 * len(splits), 4 * len(orderings)))
    for i, o in enumerate(orderings):
        for j, s in enumerate(splits):
            ax = axes[i][j]
            for k, ((other, var), d) in enumerate(bg[(bg["ordering"] == o) & (bg["split"] == s)]
                                                   .sort_values("n").groupby(["other", "variant"])):
                ax.plot(d["n"], d["gap"], marker="o", color=f"C{k}",
                        label=f"{REF} - {other}, {var.removeprefix('geo_')}")
                ax.fill_between(d["n"], d["gap_lo"], d["gap_hi"], color=f"C{k}", alpha=0.15)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_title(f"{o}, {s}")
            ax.set_xlabel("training images N")
            ax.set_ylabel("GeoScore difference (paired)")
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="Step 2 phase3_eval dir (cell probes)")
    ap.add_argument("--knn-run", type=Path, required=True, help="Step 1 phase3_eval dir (kNN)")
    ap.add_argument("--data", type=Path, help="v2 dir with dev_seen.csv and dev.csv (clusters)")
    ap.add_argument("--out", type=Path, default=Path("results/v3"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    clusters = None
    if a.data:
        clusters = {}
        for s in SPLITS:
            d = pd.read_csv(a.data / f"{s}.csv", dtype=str, keep_default_na=False)
            clusters[s] = pd.Series(g.clusters_of(s, d), index=d["id"])
    else:
        print("no --data: point estimates only, intervals are NaN")
    rows = pd.concat([load_rows(a.run, CELLS), load_rows(a.knn_run, [MEDOID])])
    root = {MEDOID: a.knn_run, **dict.fromkeys(CELLS, a.run)}
    cell = lambda bb, o, n, v: root[v] / bb / f"{o}_{n}_{v}"
    a.out.mkdir(parents=True, exist_ok=True)

    ig, bg, best = [], [], []
    for (bb, o, n), grp in rows.groupby(KEYS):
        for s, name in SPLITS.items():
            ig.append({"backbone": bb, "ordering": o, "n": n, "split": name,
                       **gap(cell(bb, o, n, CELLS[1]), cell(bb, o, n, CELLS[0]), s, clusters,
                             a.n_boot, a.seed)})
        r = grp.set_index("variant")
        win = r["dev_seen_geoscore"].idxmax()  # chosen on dev_seen only
        row = {"backbone": bb, "ordering": o, "n": n, "best": win,
               **{f"{v}_{name}_geoscore": r.loc[v, f"{name}_geoscore"]
                  for v in r.index for name in SPLITS.values()}}
        if win != MEDOID:
            for s, name in SPLITS.items():
                gp = gap(cell(bb, o, n, win), cell(bb, o, n, MEDOID), s, clusters, a.n_boot, a.seed)
                row |= {f"{name}_best_minus_medoid": gp["gap"],
                        f"{name}_best_minus_medoid_lo": gp["gap_lo"],
                        f"{name}_best_minus_medoid_hi": gp["gap_hi"]}
        best.append(row)
    for (o, n, v), grp in rows[rows["variant"].isin(CELLS)].groupby(["ordering", "n", "variant"]):
        for other in OTHERS:
            if not {REF, other} <= set(grp["backbone"]):
                continue
            for s, name in SPLITS.items():
                bg.append({"variant": v, "other": other, "ordering": o, "n": n, "split": name,
                           **gap(cell(REF, o, n, v), cell(other, o, n, v), s, clusters,
                                 a.n_boot, a.seed)})
    ig, bg, best = pd.DataFrame(ig), pd.DataFrame(bg), pd.DataFrame(best)
    ig.to_csv(a.out / "phase3_cell_init_gap.csv", index=False)
    bg.to_csv(a.out / "phase3_cell_backbone_gap.csv", index=False)
    best.to_csv(a.out / "phase3_cell_best.csv", index=False)
    pe.plot_scaling(rows, a.out / "phase3_cell_scaling.png", list(SPLITS.values()), "geoscore",
                    "GeoScore")
    plot_gaps(bg, a.out / "phase3_cell_gap.png")
    show = ["gap", "gap_lo", "gap_hi"]
    print("zero-shot - random (GeoScore)\n", ig[KEYS + ["split"] + show].round(1).to_string(index=False))
    print("StreetCLIP - other (GeoScore)\n",
          bg[["variant", "other", "ordering", "n", "split"] + show].round(1).to_string(index=False))
    print("best predictor by dev_seen GeoScore\n", best.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
