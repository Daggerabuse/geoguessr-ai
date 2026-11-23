"""Regenerate the recipe B full-run tables of docs/V4_FEASIBILITY.md, section 5c (dev only).

Paired cluster bootstrap on the same images: creators for dev_unseen; for dev_seen, sequences for
the score rows and the N contrast, creators for the B - V3, gap and gap-change rows.

  uv run python scripts/analysis/v4_b_dev_table.py [--runs runs/v4_B_session4/v4_runs]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai.v4_inputs import dev_subset, read_csv

R = Path(__file__).resolve().parents[2]
JOB = {50_000: "k4_B_full_50k", 500_000: "k5_B_full_500k"}
S, L = BB = ("streetclip", "clip_l14_336")
NB, SEED = 10_000, 42


def scores(s: str, df: pd.DataFrame, runs: Path, v3: Path, ref: pd.DataFrame) -> pd.DataFrame:
    out = {}
    label = "dev_unseen" if s == "dev" else s
    for n in JOB:
        for b in BB:
            h = pd.read_csv(v3 / b / f"bal_{n}_geo_cell_zeroshot" / f"{s}_geo_hits.csv", dtype={"id": str}).set_index("id")
            want = ref[(ref.backbone == b) & (ref.ordering == "bal") & (ref.n == n)
                       & (ref.predictor == "geo_cell_zeroshot") & (ref.split == label)].geoscore.item()
            assert abs(h.geoscore.mean() - want) < 1e-6, (b, n, s)
            out["v3", b, n] = h.geoscore
            p = pd.read_csv(runs / f"{JOB[n]}_{b}" / "final" / f"{s}_predictions.csv", dtype={"id": str}).set_index("id")
            out["b", b, n] = 5000 * np.exp(-p.km / 1492.7)
    x = pd.DataFrame(out).reindex(df["id"])
    assert x.notna().all().all()
    return x


def boot(v, cl, seed=SEED) -> np.ndarray:
    return g.bootstrap_means(np.asarray(v, float), cl, NB, seed)


def num(v: float, sign: bool = False) -> str:
    t = f"{v:+.0f}" if sign else f"{v:.0f}"
    return t.replace("-", "−")


def cell(v, cl, sign=False, level=95, bold=False) -> str:
    a = (100 - level) / 2
    lo, hi = np.percentile(boot(v, cl), [a, 100 - a])
    m = num(np.mean(v), sign)
    return f"{f'**{m}**' if bold else m} [{num(lo)}, {num(hi)}]"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=R / "runs/v4_B_session4/v4_runs")
    ap.add_argument("--bundle", type=Path, default=R / "runs/v4_inputs/v4_inputs")
    ap.add_argument("--v3", type=Path, default=R / "runs/phase3_final_dev/phase3_eval")
    a = ap.parse_args()
    ref = pd.read_csv(R / "results/v3/phase3_dev_predictors.csv")
    dfs = {s: read_csv(a.bundle / f"{s}.csv") for s in ("dev", "dev_seen")}
    X = {s: scores(s, dfs[s], a.runs, a.v3, ref) for s in dfs}
    un, se = X["dev"], X["dev_seen"]
    cu, cq = dfs["dev"]["creator_id"].to_numpy(), dfs["dev_seen"]["sequence"].to_numpy()
    cc = dfs["dev_seen"]["creator_id"].to_numpy()
    gap = lambda x, m, n: x[m, S, n] - x[m, L, n]
    delta = lambda x, n: gap(x, "b", n) - gap(x, "v3", n)

    print("| dev GeoScore (top-1 cell centroid) | StreetCLIP dev_unseen | L/14-336 dev_unseen | StreetCLIP dev_seen | L/14-336 dev_seen |")
    print("|---|---|---|---|---|")
    for n, tag, job in ((50_000, "50K", 4), (500_000, "500K", 5)):
        for m, name, bold in (("v3", f"V3 frozen, bal {tag}", False), ("b", f"**B, job {job}: {tag}**", True)):
            print(f"| {name} | " + " | ".join([cell(un[m, b, n], cu, bold=bold) for b in BB]
                                              + [cell(se[m, b, n], cq, bold=bold) for b in BB]) + " |")
        print(f"| B − V3, {tag} | " + " | ".join(
            [cell(un["b", b, n] - un["v3", b, n], cu, True) for b in BB]
            + [cell(se["b", b, n] - se["v3", b, n], cc, True) for b in BB]) + " |")

    print("\n| StreetCLIP − L/14-336 gap | dev_unseen | dev_seen |\n|---|---|---|")
    for n, tag in ((50_000, "50K"), (500_000, "500K")):
        print(f"| V3 frozen, {tag} | {cell(gap(un, 'v3', n), cu, True)} | {cell(gap(se, 'v3', n), cc, True)} |")
        print(f"| B, {tag} | {cell(gap(un, 'b', n), cu, True)} | {cell(gap(se, 'b', n), cc, True)} |")
        print(f"| **Δ = B gap − frozen gap, {tag}** | {cell(delta(un, n), cu, True, bold=True)} | "
              f"{cell(delta(se, n), cc, True, bold=True)} |")
    print(f"| Δ(50K) − Δ(500K) | {cell(delta(un, 50_000) - delta(un, 500_000), cu, True)} | "
          f"{cell(delta(se, 50_000) - delta(se, 500_000), cq, True)} (sequence clusters) |")

    print("\n90% CI of Δ (the ±40 margin, descriptive):")
    for n, tag in ((50_000, "50K"), (500_000, "500K")):
        print(f"- {tag}: dev_unseen {cell(delta(un, n), cu, True, 90)}; dev_seen {cell(delta(se, n), cc, True, 90)}")

    print("\n(B − V3) on dev_seen minus dev_unseen (independent creator bootstraps):")
    for n, tag in ((50_000, "50K"), (500_000, "500K")):
        for b in BB:
            p, q = se["b", b, n] - se["v3", b, n], un["b", b, n] - un["v3", b, n]
            lo, hi = np.percentile(boot(p, cc, 42) - boot(q, cu, 43), [2.5, 97.5])
            print(f"- {b}, {tag}: {num(p.mean() - q.mean(), True)} [{num(lo)}, {num(hi)}]")

    ids = dev_subset(dfs["dev"]["id"], 5000, 42)[:2000]
    x2 = un.loc[ids]
    ev = {b: [json.loads(l) for l in (a.runs / f"{JOB[50_000]}_{b}" / "log.jsonl").read_text().splitlines()] for b in BB}
    cur = {b: [e["geoscore"] for e in ev[b] if e.get("kind") == "diagnostic_epoch"] for b in BB}
    v = {b: [x2["v3", b, 50_000].mean(), *cur[b], x2["b", b, 50_000].mean()] for b in BB}
    print("\n| epoch | V3 frozen | " + " | ".join(map(str, range(1, len(cur[S]) + 1))) + " | final |")
    print("|---|" + "---|" * (len(cur[S]) + 2))
    print("| StreetCLIP | " + " | ".join(num(t) for t in v[S]) + " |")
    print("| L/14-336 | " + " | ".join(num(t) for t in v[L]) + " |")
    print("| gap | " + " | ".join(num(p - q, True) for p, q in zip(v[S], v[L])) + " |")


if __name__ == "__main__":
    main()
