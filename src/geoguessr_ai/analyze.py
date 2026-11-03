"""Confusion matrix analysis: per-country recall/precision, most confused pairs.

Input: results/<run>/<split>_confusion.csv (row = true, column = predicted);
baseline_clip and train write the same format, so this works on both.
Output under results/analysis/:
  precision_recall.csv         all runs, per country (long format)
  <run>_top_confusions.csv     the 10 most confused (true, predicted) pairs
  <run>_pairs.csv              directed pairs of interest
  report.md                    readable summary

Usage:
  python -m geoguessr_ai.analyze                       # all runs under results/*/
  python -m geoguessr_ai.analyze --runs results/clip_vitb16_lpft results/clip_vitb16_lpft/probe

Uncertainty: with n=200 per country the standard error of a proportion is ~3.5 points
and the 95% interval roughly +-7 points (at p=0.5). The CSVs contain Wilson 95%
intervals; do not interpret differences whose intervals overlap.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Default directed pairs of interest: the Nordic four + other commonly confused pairs.
SCANDI = ["SE", "NO", "FI", "DK"]
PAIR_GROUPS = [
    SCANDI,
    ["AR", "CL"],
    ["ES", "PT"],
    ["NG", "GH"],
    ["US", "CA"],
]


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson 95% confidence interval (lower, upper); NaN for n=0."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        den = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def load_confusion(path: Path) -> pd.DataFrame:
    cm = pd.read_csv(path, index_col=0)
    cm.columns = cm.columns.astype(str)
    cm.index = cm.index.astype(str)
    return cm.loc[:, cm.index]  # column order = row order


def per_country(cm: pd.DataFrame, attractor_ratio: float) -> pd.DataFrame:
    """recall, precision, predicted/true ratio and the 'attractor' flag."""
    m = cm.to_numpy()
    tp = np.diag(m).astype(float)
    n_true, n_pred = m.sum(1), m.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall, precision = tp / n_true, tp / n_pred
        ratio = n_pred / n_true
    r_lo, r_hi = wilson(tp, n_true)
    p_lo, p_hi = wilson(tp, n_pred)
    df = pd.DataFrame(
        {
            "country": cm.index,
            "n_true": n_true,
            "n_pred": n_pred,
            "recall": recall,
            "recall_lo": r_lo,
            "recall_hi": r_hi,
            "precision": precision,
            "precision_lo": p_lo,
            "precision_hi": p_hi,
            "pred_ratio": ratio,
        }
    )
    # attractor: over-predicted, and mostly wrong when predicted (precision < recall)
    df["attractor"] = (df["pred_ratio"] >= attractor_ratio) & (df["precision"] < df["recall"])
    # clear: precision upper bound is below the recall lower bound -> intervals do not overlap
    df["attractor_clear"] = df["attractor"] & (df["precision_hi"] < df["recall_lo"])
    return df


def confusions(cm: pd.DataFrame) -> pd.DataFrame:
    """All off-diagonal (true, pred, count, share of true) rows, sorted by descending count."""
    rows = []
    n_true = cm.sum(axis=1)
    for a in cm.index:
        for b in cm.columns:
            if a != b and cm.loc[a, b] > 0:
                c = int(cm.loc[a, b])
                rows.append((a, b, c, c / n_true[a]))
    out = pd.DataFrame(rows, columns=["true", "pred", "count", "share_of_true"])
    return out.sort_values(["count", "true", "pred"], ascending=[False, True, True]).reset_index(drop=True)


def pair_table(cm: pd.DataFrame, groups: list[list[str]]) -> pd.DataFrame:
    rows, seen = [], set()
    n_true = cm.sum(axis=1)
    for g in groups:
        for a in g:
            for b in g:
                if a == b or (a, b) in seen or a not in cm.index or b not in cm.columns:
                    continue
                seen.add((a, b))
                c = int(cm.loc[a, b])
                lo, hi = wilson(np.array([c]), np.array([n_true[a]]))
                rows.append((a, b, c, c / n_true[a], lo[0], hi[0]))
    return pd.DataFrame(rows, columns=["true", "pred", "count", "share_of_true", "lo", "hi"])


def run_name(path: Path, results: Path) -> str:
    try:
        return str(path.relative_to(results)).replace("\\", "/")
    except ValueError:
        return path.name


def find_runs(results: Path, split: str) -> list[Path]:
    found = sorted(
        p.parent for p in results.glob(f"**/{split}_confusion.csv") if "analysis" not in p.parts
    )
    return found


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, fmt[c]) if c in fmt and pd.notna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(name: str, pc: pd.DataFrame, top: pd.DataFrame, pairs: pd.DataFrame,
                 focus: list[str], k: int) -> str:
    acc = float(np.average(pc["recall"], weights=pc["n_true"]))
    out = [f"## {name}", f"top-1 (recall, weighted mean) = {acc:.3f}, countries = {len(pc)}", ""]

    att = pc[pc["attractor"]].sort_values("pred_ratio", ascending=False)
    out.append("### Attractor countries (predicted/true >= threshold and precision < recall)")
    if att.empty:
        out.append("None.")
    else:
        a = att[["country", "recall", "precision", "pred_ratio", "attractor_clear"]]
        out.append(md_table(a, {"recall": ".2f", "precision": ".2f", "pred_ratio": ".2f"}))
        out.append("`attractor_clear=True`: the precision interval is separated from the recall interval.")
    out.append("")

    out.append("### Focus countries")
    f = pc[pc["country"].isin(focus)][
        ["country", "recall", "recall_lo", "recall_hi", "precision", "pred_ratio"]
    ]
    out.append(md_table(f, {c: ".2f" for c in f.columns if c != "country"}))
    out.append("")

    out.append(f"### {k} most confused pairs (true -> predicted)")
    out.append(md_table(top.head(k), {"share_of_true": ".3f"}))
    out.append("")

    out.append("### Pairs of interest (true -> predicted; directed)")
    out.append(md_table(pairs, {"share_of_true": ".3f", "lo": ".3f", "hi": ".3f"}))
    out.append("")
    return "\n".join(out)


def analyze_run(path: Path, results: Path, split: str, out: Path, args) -> pd.DataFrame:
    cm = load_confusion(path / f"{split}_confusion.csv")
    name = run_name(path, results)
    stem = name.replace("/", "__")
    pc = per_country(cm, args.attractor_ratio)
    top = confusions(cm).head(args.top)
    pairs = pair_table(cm, PAIR_GROUPS)
    top.to_csv(out / f"{stem}_top_confusions.csv", index=False)
    pairs.to_csv(out / f"{stem}_pairs.csv", index=False)
    report = build_report(name, pc, top, pairs, args.focus, args.top)
    return pc.assign(run=name), report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--runs", type=Path, nargs="*", help="run directories (default: all)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>/analysis")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--focus", nargs="+", default=["FI", "SE"])
    ap.add_argument("--attractor-ratio", type=float, default=1.3,
                    help="threshold on predicted count / true count")
    args = ap.parse_args()

    out = args.out or args.results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = args.runs or find_runs(args.results, args.split)
    if not runs:
        raise SystemExit(f"no {args.split}_confusion.csv found under {args.results}")

    frames, reports = [], [f"# Confusion analysis ({args.split})", ""]
    for r in runs:
        pc, rep = analyze_run(r, args.results, args.split, out, args)
        frames.append(pc)
        reports.append(rep)
    pd.concat(frames, ignore_index=True).to_csv(out / "precision_recall.csv", index=False)
    (out / "report.md").write_text("\n".join(reports), encoding="utf-8")
    print(f"{len(runs)} runs -> {out}")
    print("\n".join(reports))


if __name__ == "__main__":
    main()
