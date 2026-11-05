"""Confusion matrix analysis: per-country recall/precision, most confused pairs.

Input: results/<run>/<split>_confusion.csv (row = true, column = predicted);
baseline_clip and train write the same format, so this works on both.
Output under results/analysis/:
  precision_recall.csv         all runs, per country (long format)
  <run>_top_confusions.csv     the 10 most confused (true, predicted) pairs
  <run>_pairs.csv              directed pairs of interest
  report.md                    readable summary
  per_country_lpft_vs_streetclip.png   dumbbell plot of per-country top-1 for two runs
                               (default: LP-FT + TTA vs. StreetCLIP zero-shot + TTA)
  paired_comparison.csv        paired test of two runs on the same images (top-1, top-5), one
                               block of rows per pair (default: LP-FT vs. StreetCLIP, with and
                               without TTA): bootstrap 95% interval of the difference, McNemar p

Usage:
  python -m geoguessr_ai.analyze                       # all runs under results/*/
  python -m geoguessr_ai.analyze --runs results/clip_vitb16_lpft results/clip_vitb16_lpft/probe

Uncertainty: with n=200 per country the standard error of a proportion is ~3.5 points
and the 95% interval of one country's accuracy roughly +-7 points (at p=0.5). The difference
between two models has a standard error about sqrt(2) times larger, so its 95% interval is
roughly +-10 points per country (independent errors; paired errors can make it narrower).
The CSVs contain Wilson 95% intervals for single proportions.
"""

import argparse
import math
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV with LF line endings on every OS (pandas defaults to os.linesep, CRLF on Windows)."""
    df.to_csv(path, index=False, lineterminator="\n")


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


# Colors from the validated reference palette (categorical slots 1 and 2, light surface).
INK, INK_2, SURFACE, GRID, LINK = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df", "#b4b3ad"
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"


def plot_dumbbell(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                  split: str, out: Path) -> None:
    """Per-country top-1 of two runs as a dumbbell plot, sorted by the difference (a - b)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df_a.set_index("country")[["n", "top1"]].rename(columns={"top1": "top1_a"}).join(
        df_b.set_index("country")[["top1"]].rename(columns={"top1": "top1_b"}), how="inner"
    )
    d["diff"] = d["top1_a"] - d["top1_b"]
    d = d.sort_values("diff", ascending=False)
    n = int(d["n"].iloc[0])
    half = 1.96 * math.sqrt(0.25 / n) * 100  # 95% half-width in points at p = 0.5

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(d) + 2.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hlines(y, d["top1_a"], d["top1_b"], color=LINK, linewidth=2, zorder=1)
    series = (("top1_a", COLOR_A, label_a), ("top1_b", COLOR_B, label_b))
    for col, color, label in series:
        ax.scatter(d[col], y, s=64, color=color, edgecolor=SURFACE, linewidth=2,
                   zorder=3, label=label)
    # direct labels above the top row, centered over each dot
    top = d.iloc[0]
    for col, _, label in series:
        ax.text(top[col], -0.85, label, ha="center", va="bottom", fontsize=8, color=INK_2)

    ax.set_yticks(y, d.index, fontsize=9, color=INK)
    ax.set_ylim(len(d) - 0.5, -1.2)
    lo = math.floor(min(d["top1_a"].min(), d["top1_b"].min()) * 10) / 10
    ax.set_xlim(max(0.0, lo - 0.02), 1.0)
    ax.set_xlabel(f"{split} top-1", color=INK_2)
    ax.tick_params(axis="x", colors=INK_2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle(f"Per-country {split} top-1: {label_a} vs. {label_b}", x=0.01, ha="left",
                 fontsize=12, color=INK)
    ax.set_title(f"n = {n} per country. 95% interval: about +-{half:.0f} points per dot, "
                 f"+-{half * math.sqrt(2):.0f} for the gap. Sorted by difference.",
                 loc="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def load_predictions(path: Path) -> pd.DataFrame:
    """<split>_predictions.csv: id, true, pred, top5 (space-separated codes)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def correct(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-image correctness (bool) for top-1 and top-5, indexed by image id."""
    top5 = [t in p.split() for t, p in zip(pred["true"], pred["top5"])]
    return pd.DataFrame(
        {"true": pred["true"].to_numpy(), "top1": (pred["pred"] == pred["true"]).to_numpy(),
         "top5": top5},
        index=pred["id"].to_numpy(),
    )


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of b vs. c discordant pairs, p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_comparison(pred_a: pd.DataFrame, pred_b: pd.DataFrame, n_boot: int = 10000,
                      seed: int = 42) -> pd.DataFrame:
    """Paired comparison of two runs scored on the same images (difference = a - b).

    Bootstrap: images are resampled with replacement (n_boot times, fixed seed); the 95%
    interval is the percentile interval of the accuracy difference. Since each image
    contributes -1, 0 or +1 to the difference, resampling images is the same as drawing the
    three counts from a multinomial, which is what is done here. McNemar: exact two-sided
    test on the images where exactly one run is correct.
    """
    ca, cb = correct(pred_a), correct(pred_b)
    if set(ca.index) != set(cb.index) or len(ca) != len(cb):
        raise ValueError("the two runs were not scored on the same images")
    cb = cb.loc[ca.index]
    if not (ca["true"] == cb["true"]).all():
        raise ValueError("the two runs disagree on the true labels")
    rng = np.random.default_rng(seed)
    n, rows = len(ca), []
    for metric in ("top1", "top5"):
        a, b = ca[metric].to_numpy(), cb[metric].to_numpy()
        a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
        probs = np.array([b_only, n - a_only - b_only, a_only]) / n
        counts = rng.multinomial(n, probs, size=n_boot)
        diffs = (counts[:, 2] - counts[:, 0]) / n
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({
            "metric": metric, "n": n, "acc_a": a.mean(), "acc_b": b.mean(),
            "diff": (a_only - b_only) / n, "ci_lo": lo, "ci_hi": hi,
            "a_only": a_only, "b_only": b_only, "mcnemar_p": mcnemar_exact(a_only, b_only),
            "n_boot": n_boot, "seed": seed,
        })
    return pd.DataFrame(rows)


def paired_report(df: pd.DataFrame, name_a: str, name_b: str) -> str:
    note = ("Same images for both runs; diff = a - b. 95% interval: paired bootstrap over images "
            f"({int(df['n_boot'].iloc[0])} resamples, seed {int(df['seed'].iloc[0])}). "
            "a_only / b_only: images only that run gets right. p: exact McNemar test.")
    out = [f"## Paired comparison: {name_a} (a) vs. {name_b} (b)", note, ""]
    cols = ["metric", "n", "acc_a", "acc_b", "diff", "ci_lo", "ci_hi", "a_only", "b_only",
            "mcnemar_p"]
    out.append(md_table(df[cols], {c: ".4f" for c in ("acc_a", "acc_b", "diff", "ci_lo", "ci_hi")}
                        | {"mcnemar_p": ".2g"}))
    out.append("")
    return "\n".join(out)


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


def analyze_run(path: Path, results: Path, split: str, out: Path,
                args) -> tuple[pd.DataFrame, str]:
    cm = load_confusion(path / f"{split}_confusion.csv")
    name = run_name(path, results)
    stem = name.replace("/", "__")
    pc = per_country(cm, args.attractor_ratio)
    top = confusions(cm).head(args.top)
    pairs = pair_table(cm, PAIR_GROUPS)
    write_csv(top, out / f"{stem}_top_confusions.csv")
    write_csv(pairs, out / f"{stem}_pairs.csv")
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
    # both runs with the same 3-crop TTA, so the per-country comparison is like for like
    ap.add_argument("--plot-runs", nargs=2,
                    default=["clip_vitb16_lpft_tta", "streetclip_zeroshot_tta"],
                    metavar=("A", "B"),
                    help="two run directories (under --results) for the dumbbell plot")
    ap.add_argument("--plot-labels", nargs=2,
                    default=["Fine-tuned ViT-B/16 (LP-FT, TTA)", "StreetCLIP (zero-shot, TTA)"])
    ap.add_argument("--plot-name", default="per_country_lpft_vs_streetclip.png")
    ap.add_argument("--paired-runs", nargs="+",
                    default=["clip_vitb16_lpft_tta", "streetclip_zeroshot_tta",
                             "clip_vitb16_lpft", "streetclip_zeroshot"],
                    metavar="RUN",
                    help="pairs of run directories (under --results) for the paired comparison: "
                         "A1 B1 [A2 B2 ...]")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
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
    write_csv(pd.concat(frames, ignore_index=True), out / "precision_recall.csv")
    if len(args.paired_runs) % 2:
        raise SystemExit("--paired-runs needs pairs of runs: A1 B1 [A2 B2 ...]")
    paired_frames = []
    for run_a, run_b in zip(args.paired_runs[::2], args.paired_runs[1::2]):
        preds = [args.results / r / f"{args.split}_predictions.csv" for r in (run_a, run_b)]
        if not all(p.exists() for p in preds):
            print("paired comparison skipped: predictions not found for", run_a, run_b)
            continue
        paired = paired_comparison(*(load_predictions(p) for p in preds), args.n_boot, args.seed)
        paired_frames.append(paired.assign(run_a=run_a, run_b=run_b))
        reports.append(paired_report(paired, run_a, run_b))
    if paired_frames:
        write_csv(pd.concat(paired_frames, ignore_index=True), out / "paired_comparison.csv")
    (out / "report.md").write_text("\n".join(reports), encoding="utf-8", newline="\n")
    csvs = [args.results / r / f"{args.split}_per_country.csv" for r in args.plot_runs]
    if all(c.exists() for c in csvs):
        plot_dumbbell(*(pd.read_csv(c) for c in csvs), *args.plot_labels, args.split,
                      out / args.plot_name)
    else:
        print("dumbbell plot skipped: per-country CSVs not found for", args.plot_runs)
    print(f"{len(runs)} runs -> {out}")
    print("\n".join(reports))


if __name__ == "__main__":
    main()
