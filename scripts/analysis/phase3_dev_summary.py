"""Phase 3 step 1, DEVELOPMENT only (no test km / GeoScore is read): the compact summary table and
the exploratory leakage effect of the near-duplicate mask.

  results/v3/phase3_dev_summary.csv  one row per backbone x ordering x N x predictor (dev_seen and
                                     dev_unseen: GeoScore, median km, acc@25km, country top-1)
  results/v3/phase3_dev_leakage.csv  EXPLORATORY: masked geo_medoid (this run: 1 km mask, CPU
                                     fp32) minus Phase 1's unmasked kNN coordinates (GPU fp16),
                                     per backbone x ordering x N x split. The two runs differ in
                                     neighbour mask AND in device/dtype, so a difference mixes
                                     both; acc@1km is 0 under a 1 km mask by construction.

  uv run python scripts/analysis/phase3_dev_summary.py --run runs/phase3_dev/phase3_eval \
      --phase1-dev runs/phase1_eval_v3/phase1_eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g

SPLITS = {"dev_seen": "dev_seen_geo_predictions.csv", "dev_unseen": "dev_geo_predictions.csv"}
KEYS = ["backbone", "ordering", "n"]
METRICS = ("geoscore", "median_km", "acc_1km", "acc_25km")


def phase1_metrics(cell: Path, split: str) -> dict:
    km = pd.read_csv(cell / SPLITS[split])["km"].to_numpy()
    return {"geoscore": g.geoscore(km).mean(), "median_km": np.median(km),
            "acc_1km": (km <= 1).mean(), "acc_25km": (km <= 25).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--phase1-dev", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/v3"))
    a = ap.parse_args()
    t = pd.read_csv(a.run / "results.csv")
    if t.filter(regex="^test_").size:
        raise ValueError("results.csv holds test columns; this script is development-only")
    a.out.mkdir(parents=True, exist_ok=True)

    cols = ["variant", "k", "mask_km", "knn_device", "knn_dtype"] + [
        f"{s}_{m}" for s in SPLITS for m in (*METRICS[:2], "acc_25km", "country_top1")]
    t[KEYS + cols].sort_values(KEYS + ["variant"]).to_csv(a.out / "phase3_dev_summary.csv", index=False)

    rows = []
    for _, r in t[t["predictor"] == "medoid"].iterrows():
        cell = a.phase1_dev / r["backbone"] / f"{r['ordering']}_{r['n']}_knn"
        k1 = json.loads((cell / "geo_metrics.json").read_text())["k_coords"]
        for s in SPLITS:
            old = phase1_metrics(cell, s)
            row = {"backbone": r["backbone"], "ordering": r["ordering"], "n": r["n"], "split": s,
                   "k_masked": r["k"], "k_phase1": k1}
            for m in METRICS:
                row |= {f"{m}_masked": r[f"{s}_{m}"], f"{m}_phase1": old[m],
                        f"{m}_diff": r[f"{s}_{m}"] - old[m]}
            rows.append(row)
    lk = pd.DataFrame(rows)
    lk.to_csv(a.out / "phase3_dev_leakage.csv", index=False)
    print(lk.groupby("split")[[f"{m}_diff" for m in METRICS]].agg(["mean", "min", "max"]).T.round(4))


if __name__ == "__main__":
    main()
