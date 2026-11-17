"""Phase 3 step 2, DEVELOPMENT only: the probe lr-grid edge rule of docs/V2_PIPELINE.md
("Decisions made after Step 2 finished"). Reads only each cell probe's row.json and the
dev_seen lr sweep in geo_metrics.json.

  results/v3/phase3_cell_lr.csv  per cell: selected lr and epoch, dev_seen GeoScore at the
                                 selected lr and at its lower/upper grid neighbours (NaN at an
                                 edge), grid_edge, and the GPU seconds one more 60-epoch lr
                                 would take (worst case, scaled from the cell's own run)

  uv run python scripts/analysis/phase3_cell_lr.py --run runs/phase3_session_c/phase3_eval
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BACKBONES, ORDERINGS, SIZES = ("streetclip", "clip_l14_336", "clip_b16"), ("nat", "bal"), (
    50_000, 100_000, 200_000, 500_000)
INITS, THRESHOLD = ("random", "zeroshot"), 8  # declared before counting, see the docs
EXTEND = {"low": 3e-4, "high": 3e-1}


def cell_rows(run: Path) -> pd.DataFrame:
    rows = []
    for bb in BACKBONES:
        for o in ORDERINGS:
            for n in SIZES:
                for init in INITS:
                    d = run / bb / f"{o}_{n}_geo_cell_{init}"
                    gm = json.loads((d / "geo_metrics.json").read_text())
                    if any(k.startswith("test") for k in gm):
                        raise ValueError(f"{d} holds test results; development only")
                    cs = gm["cache_settings"]
                    sw = {e["lr"]: e for e in gm["sweep_dev_seen"]}
                    grid = sorted(sw)
                    declared = sorted(cs["probe_lrs"] + cs.get("extra_lrs", []))
                    if grid != declared:
                        raise ValueError(f"{d}: sweep {grid} != grid {declared}")
                    i = grid.index(gm["lr"])
                    run_ep = sum(min(e["epoch"] + cs["probe_patience"], cs["probe_epochs"]) + 1
                                 for e in sw.values())
                    rows.append({"backbone": bb, "ordering": o, "n": n, "init": init,
                                 "lr": gm["lr"], "epoch": gm["epoch"],
                                 "stopped_early": gm["stopped_early"],
                                 "probe_epochs": cs["probe_epochs"],
                                 "grid": " ".join(f"{x:g}" for x in grid),
                                 "geoscore_sel": sw[gm["lr"]]["dev_seen_geoscore"],
                                 "lr_lower": grid[i - 1] if i > 0 else float("nan"),
                                 "geoscore_lower": sw[grid[i - 1]]["dev_seen_geoscore"] if i > 0 else float("nan"),
                                 "lr_upper": grid[i + 1] if i + 1 < len(grid) else float("nan"),
                                 "geoscore_upper": (sw[grid[i + 1]]["dev_seen_geoscore"]
                                                    if i + 1 < len(grid) else float("nan")),
                                 "grid_edge": "low" if i == 0 else "high" if i == len(grid) - 1 else "",
                                 "extended": bool(cs.get("extra_lrs")), "seconds": gm["seconds"],
                                 "est_seconds_one_more_lr": gm["seconds"] * (cs["probe_epochs"] + 1) / run_ep})
    return pd.DataFrame(rows)


def verdict(t: pd.DataFrame) -> dict:
    """{(init, edge): (count, extend?)}"""
    return {(i, e): (int(((t["init"] == i) & (t["grid_edge"] == e)).sum()),
                     int(((t["init"] == i) & (t["grid_edge"] == e)).sum()) >= THRESHOLD)
            for i in INITS for e in EXTEND}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/v3"))
    a = ap.parse_args()
    t = cell_rows(a.run)
    assert len(t) == len(BACKBONES) * len(ORDERINGS) * len(SIZES) * len(INITS)
    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / "phase3_cell_lr.csv", index=False)
    print(t.drop(columns=["seconds"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    for (i, e), (c, ext) in verdict(t).items():
        if t.loc[t["init"] == i, "extended"].any():  # the stopping rule: no second extension
            print(f"{i} {e} edge: {c} of 24 -> final grid (already extended): grid_edge limitation")
            continue
        h = t.loc[t["init"] == i, "est_seconds_one_more_lr"].sum() / 3600
        print(f"{i} {e} edge: {c} of 24 -> " + (f"EXTEND with {EXTEND[e]:g} "
              f"(~{h:.1f} GPU h worst case, all 24 cells, + loading)" if ext else "frozen"))


if __name__ == "__main__":
    main()
