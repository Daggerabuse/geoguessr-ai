"""Phase 3 step 2, DEVELOPMENT only (no test km / GeoScore is read): the five coordinate
predictors (Step 1 kNN medoid, cell probe and hybrid, each random / zero-shot init) and their
backbone gaps. --run is the merged final dir (scripts/analysis/phase3_merge.py).

  results/v3/phase3_dev_predictors.csv     GeoScore (row CI) and median km per backbone x ordering
                                           x N x split x predictor
  results/v3/phase3_cell_backbone_gap.csv  StreetCLIP - L/14-336 and StreetCLIP - B/16 per predictor
                                           x ordering x N x split: GeoScore (+ acc_25km, acc_750km)
                                           and median km, paired
  results/v3/phase3_backbone_shrink.csv    gap(50K) - gap(500K) of the GeoScore gap, paired per image
  results/v3/phase3_cell_init_gap.csv      zero-shot - random init, cell and hybrid
  results/v3/phase3_cell_best.csv          per backbone x ordering x N: the best predictor by dev_seen
                                           GeoScore, its paired gap vs the runner-up and vs
                                           geo_medoid. kNN parts are near-duplicate masked, the cell
                                           probe trains on every subset image: not like for like
  results/v3/phase3_cell_flags.csv         cell probes whose lr is at a grid edge or whose epoch hit
                                           the cap, and hybrids whose m or k is at a grid edge, with
                                           the dev_seen GeoScore margin vs each grid neighbour
                                           (pick - neighbour; the other settings fixed). m=1 / k=1
                                           are natural lower bounds, not grid limitations; tau is
                                           not searched by the medoid hybrid
  results/v3/phase3_cell_scaling.png, phase3_cell_gap.png (vs L/14-336), phase3_cell_gap_b16.png

Intervals: paired cluster bootstrap (sequences on dev_seen, creators on dev_unseen), n_boot and
seed as in phase3_eval (checked against the rows' cache_settings); --data: the v2 dir with
dev_seen.csv and dev.csv.

  uv run python scripts/analysis/phase3_cell_summary.py --run runs/phase3_final_dev/phase3_eval \
      --knn-run runs/phase3_dev/phase3_eval --data runs/phase3_hybrid_random/v2_dev
"""

import argparse
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_eval as p3

sys.path.insert(0, str(Path(__file__).parent))
from phase3_cell_lr import cell_rows  # noqa: E402

SPLITS = {"dev_seen": "dev_seen", "dev": "dev_unseen"}  # file prefix -> report name
COLS = ("geoscore", "acc_25km", "acc_750km")
KEYS = ["backbone", "ordering", "n"]
MEDOID = "geo_medoid"
CELLS = ("geo_cell_random", "geo_cell_zeroshot", "geo_hybrid_random", "geo_hybrid_zeroshot")
VARIANTS = (MEDOID, *CELLS)
INIT_PAIRS = {"cell": ("geo_cell_zeroshot", "geo_cell_random"),
              "hybrid": ("geo_hybrid_zeroshot", "geo_hybrid_random")}
REF = "streetclip"
OTHERS = ("clip_l14_336", "clip_b16")


def load_rows(path: Path, variants) -> pd.DataFrame:
    """Every finished cell's row.json (results.csv lists only the last session's cells)."""
    t = pd.DataFrame([json.loads(f.read_text()) for bb in pe.BACKBONES for v in variants
                      for f in sorted((path / bb).glob(f"*_{v}/row.json"))])
    if t.filter(regex="^test_").size:
        raise ValueError(f"{path} holds test columns; this script is development-only")
    return t


@cache
def hits(d: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(d / f"{split}_geo_hits.csv", dtype={"id": str})


def gap(a_dir: Path, b_dir: Path, split: str, clusters, n_boot: int, seed: int) -> dict:
    a, b = hits(a_dir, split), hits(b_dir, split)
    out = {}
    for c in COLS:
        r = pe.paired_gap(a, b, clusters[split], n_boot, seed, c)
        out |= {(f"{k}" if c == "geoscore" else f"{c}_{k}"): v for k, v in r.items()
                if c == "geoscore" or k != "n_images"}
    m = p3.median_gap(a, b, clusters[split], n_boot, seed)
    return out | {"median_km_gap": m["gap"], "median_km_gap_lo": m["gap_lo"],
                  "median_km_gap_hi": m["gap_hi"]}


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


def cell_flags(run: Path) -> pd.DataFrame:
    """Cell probes at an lr-grid edge or the epoch cap: one row per lr neighbour."""
    t, out = cell_rows(run), []
    for _, r in t.iterrows():
        cap = r["epoch"] >= r["probe_epochs"]
        flags = [f"grid_edge_{r['grid_edge']}"] * bool(r["grid_edge"]) + ["epoch_cap"] * cap
        for side in ("lower", "upper"):
            if flags and not np.isnan(r[f"lr_{side}"]):
                out.append({"kind": "cell", **r[KEYS], "predictor": f"geo_cell_{r['init']}",
                            "flags": ";".join(flags), "param": "lr", "grid": r["grid"],
                            "pick": r["lr"], "neighbour": r[f"lr_{side}"],
                            "margin": r["geoscore_sel"] - r[f"geoscore_{side}"],
                            "epoch": r["epoch"], "stopped_early": r["stopped_early"]})
    return pd.DataFrame(out)


def hybrid_flags(run: Path) -> pd.DataFrame:
    """Hybrids whose m or k is at an edge of its grid (tau is not searched by the medoid)."""
    out = []
    for f in sorted(run.glob("*/*_geo_hybrid_*/geo_metrics.json")):
        gm, row = json.loads(f.read_text()), json.loads((f.parent / "row.json").read_text())
        cs, sw = gm["cache_settings"], pd.DataFrame(gm["sweep_dev_seen"])
        assert sw["tau"].isna().all() and cs["hybrid_predictor"] == "medoid"
        pick = {"m": gm["m"], "k": gm["k"]}
        for p, grid in (("m", cs["hybrid_m"]), ("k", cs["k"])):
            i = grid.index(pick[p])
            if 0 < i < len(grid) - 1:
                continue
            nb = grid[1] if i == 0 else grid[-2]
            q = sw[(sw[p] == nb) & (sw[{"m": "k", "k": "m"}[p]] == pick[{"m": "k", "k": "m"}[p]])]
            sel = sw[(sw["m"] == pick["m"]) & (sw["k"] == pick["k"])]
            out.append({"kind": "hybrid", **{k: row[k] for k in KEYS}, "predictor": row["variant"],
                        "flags": ("natural_lower_bound" if i == 0 and pick[p] == 1 else
                                  f"grid_edge_{'low' if i == 0 else 'high'}"),
                        "param": p, "grid": " ".join(map(str, grid)), "pick": pick[p],
                        "neighbour": nb, "margin": float(sel["dev_seen_geoscore"].iloc[0]
                                                         - q["dev_seen_geoscore"].iloc[0])})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="merged phase3_eval dir (cells, hybrids)")
    ap.add_argument("--knn-run", type=Path, required=True, help="Step 1 phase3_eval dir (kNN)")
    ap.add_argument("--data", type=Path, required=True, help="v2 dir with dev_seen.csv and dev.csv")
    ap.add_argument("--out", type=Path, default=Path("results/v3"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    clusters = {}
    for s in SPLITS:
        d = pd.read_csv(a.data / f"{s}.csv", dtype=str, keep_default_na=False)
        clusters[s] = pd.Series(g.clusters_of(s, d), index=d["id"])
    rows = pd.concat([load_rows(a.run, CELLS), load_rows(a.knn_run, [MEDOID])], ignore_index=True)
    root = {MEDOID: a.knn_run, **dict.fromkeys(CELLS, a.run)}
    cell = lambda bb, o, n, v: root[v] / bb / f"{o}_{n}_{v}"
    assert len(rows) == 3 * 2 * 4 * len(VARIANTS), len(rows)
    for _, r in rows.iterrows():  # same bootstrap settings as phase3_eval, and the hits match the row
        d = cell(r["backbone"], r["ordering"], r["n"], r["variant"])
        cs = json.loads((d / "geo_metrics.json").read_text())["cache_settings"]
        assert (cs["n_boot"], cs["seed"]) == (a.n_boot, a.seed), (d, cs["n_boot"], cs["seed"])
        for s, name in SPLITS.items():
            h = hits(d, s)
            assert np.isclose(h["geoscore"].mean(), r[f"{name}_geoscore"]), (d, s)
            assert np.isclose(h["km"].median(), r[f"{name}_median_km"]), (d, s)
    a.out.mkdir(parents=True, exist_ok=True)

    pr = pd.DataFrame([{**r[KEYS], "predictor": r["variant"], "split": name,
                        **{c: r[f"{name}_{c}"] for c in ("geoscore", "geoscore_lo", "geoscore_hi",
                                                         "median_km")}}
                       for _, r in rows.iterrows() for name in SPLITS.values()])

    ig, bg, sh, best = [], [], [], []
    for (bb, o, n), grp in rows.groupby(KEYS):
        for pred, (zs, rnd) in INIT_PAIRS.items():
            for s, name in SPLITS.items():
                ig.append({"predictor": pred, "backbone": bb, "ordering": o, "n": n, "split": name,
                           **gap(cell(bb, o, n, zs), cell(bb, o, n, rnd), s, clusters, a.n_boot, a.seed)})
        r = grp.set_index("variant")["dev_seen_geoscore"].sort_values(ascending=False)
        win, second = r.index[:2]  # chosen on dev_seen only
        row = {"backbone": bb, "ordering": o, "n": n, "best": win, "runner_up": second,
               **{f"{v}_{name}_geoscore": grp.set_index("variant").loc[v, f"{name}_geoscore"]
                  for v in VARIANTS for name in SPLITS.values()}}
        for vs, other in (("runner_up", second), ("medoid", MEDOID)):
            if other == win:
                continue
            for s, name in SPLITS.items():
                gp = gap(cell(bb, o, n, win), cell(bb, o, n, other), s, clusters, a.n_boot, a.seed)
                row |= {f"{name}_best_minus_{vs}{k}": gp[f"gap{k}"] for k in ("", "_lo", "_hi")}
        best.append(row)
    for v in VARIANTS:
        for o in sorted(rows["ordering"].unique()):
            ns = sorted(rows["n"].unique())
            for other in OTHERS:
                for s, name in SPLITS.items():
                    diffs = {}
                    for n in ns:
                        bg.append({"variant": v, "other": other, "ordering": o, "n": n, "split": name,
                                   **gap(cell(REF, o, n, v), cell(other, o, n, v), s, clusters,
                                         a.n_boot, a.seed)})
                        diffs[n] = pe.hit_diff(hits(cell(REF, o, n, v), s), hits(cell(other, o, n, v), s),
                                               "geoscore")
                    sh.append({"variant": v, "other": other, "ordering": o, "split": name,
                               "n_small": ns[0], "n_large": ns[-1],
                               **pe.shrink(diffs[ns[0]], diffs[ns[-1]], clusters[s], a.n_boot, a.seed)})
    ig, bg, sh, best = map(pd.DataFrame, (ig, bg, sh, best))
    flags = pd.concat([cell_flags(a.run), hybrid_flags(a.run)], ignore_index=True)
    pr.to_csv(a.out / "phase3_dev_predictors.csv", index=False)
    ig.to_csv(a.out / "phase3_cell_init_gap.csv", index=False)
    bg.to_csv(a.out / "phase3_cell_backbone_gap.csv", index=False)
    sh.to_csv(a.out / "phase3_backbone_shrink.csv", index=False)
    best.to_csv(a.out / "phase3_cell_best.csv", index=False)
    flags.to_csv(a.out / "phase3_cell_flags.csv", index=False)
    pe.plot_scaling(rows, a.out / "phase3_cell_scaling.png", list(SPLITS.values()), "geoscore",
                    "GeoScore")
    plot_gaps(bg[bg["other"] == OTHERS[0]], a.out / "phase3_cell_gap.png")
    plot_gaps(bg[bg["other"] == OTHERS[1]], a.out / "phase3_cell_gap_b16.png")
    pd.set_option("display.width", 250)
    show = ["gap", "gap_lo", "gap_hi", "median_km_gap", "median_km_gap_lo", "median_km_gap_hi"]
    print("zero-shot - random\n", ig[["predictor", *KEYS, "split", *show]].round(1).to_string(index=False))
    print("StreetCLIP - other\n",
          bg[["variant", "other", "ordering", "n", "split", *show]].round(1).to_string(index=False))
    print("shrink 50K -> 500K (GeoScore)\n", sh.round(1).to_string(index=False))
    print("best predictor by dev_seen GeoScore\n", best.round(1).to_string(index=False))
    print("flags\n", flags.to_string(index=False))
    c = flags[flags["kind"] == "cell"].drop_duplicates(KEYS + ["predictor"])
    print("cell flag counts:", c.groupby(["predictor", "flags"]).size().to_dict())


if __name__ == "__main__":
    main()
