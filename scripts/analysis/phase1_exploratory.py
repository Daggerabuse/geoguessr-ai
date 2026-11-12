"""Phase 1 final test evaluation, EXPLORATORY analyses (no hypothesis was declared for any of
this; nothing is tuned or selected on test, the cells and their hyperparameters are the
pre-registered ones):

  2  test vs dev top-1 per cell; why test is lower: country mix (TVD; reweighting each split's
     per-country accuracy to the other's mix), per-country accuracy, head/middle/tail and
     reliable subsets, sequence structure of test (train and dev keep at most 3 images per
     sequence, test keeps all)
  3  subset-seen creators: share of dev_seen / test images whose creator is in the cell's own
     training subset, and top-1 for those vs the rest
  4  does the dev StreetCLIP - L/14-336 gap (and its shrink) replicate on test: sign agreement
  5  nat vs bal on test: fixed-class macro top-1, head/middle/tail

Inputs: the final eval output (results.csv, gaps.csv, shrink.csv, per-cell *_per_country.csv
and test_hits.csv), the official-train country table of the data audit (country_counts.csv:
official train/test images, freq_group, reliable) and the official test CSV (sequence ids).
Writes CSVs and PNGs to --out.

  uv run python scripts/analysis/phase1_exploratory.py --run runs/phase1_final \
      --counts runs/phase0/v2/country_counts.csv --test-csv data/raw/test.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = {"dev_seen": "dev_seen", "dev_unseen": "dev", "test": "test"}  # name -> file prefix
EXAMPLE = ("nat", 500_000, "probe_zeroshot")  # cell shown per country in the report


def cell_dir(ev: Path, r) -> Path:
    return ev / r["backbone"] / f"{r['ordering']}_{r['n']}_{r['variant']}"


def tvd(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return 0.5 * float((p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum())


def top1_table(res: pd.DataFrame) -> pd.DataFrame:
    t = res[["backbone", "variant", "ordering", "n"]].copy()
    for s in SPLITS:
        t[s] = res[f"{s}_top1"]
    t["test_minus_dev_seen"] = t["test"] - t["dev_seen"]
    t["test_minus_dev_unseen"] = t["test"] - t["dev_unseen"]
    return t


def country_mix(ev: Path, res: pd.DataFrame, counts: pd.DataFrame):
    """Image share per country in each split (identical in every cell) and official train."""
    d = cell_dir(ev, res.iloc[0])
    n = {s: pd.read_csv(d / f"{p}_per_country.csv", keep_default_na=False).set_index("country")["n"]
         for s, p in SPLITS.items()}
    cls = counts[counts["in_classes"]].set_index("country")
    n["official_train"] = cls["train_images"]
    share = pd.DataFrame({k: v / v.sum() for k, v in n.items()}).fillna(0)
    share["freq_group"] = cls["freq_group"].reindex(share.index)
    share["reliable"] = cls["reliable"].reindex(share.index).fillna(False)
    pairs = [("test", "dev_seen"), ("test", "dev_unseen"), ("test", "official_train"),
             ("dev_seen", "official_train"), ("dev_unseen", "official_train"),
             ("dev_seen", "dev_unseen")]
    tv = pd.DataFrame([{"a": a, "b": b, "tvd": tvd(share[a], share[b])} for a, b in pairs])
    comp = pd.concat([share.groupby("freq_group")[list(n)].sum(),
                      share[share["reliable"]][list(n)].sum().rename("reliable").to_frame().T])
    return share.sort_values("test", ascending=False), tv, comp


def reweighting(ev: Path, res: pd.DataFrame) -> pd.DataFrame:
    """Per cell: dev_seen accuracy at test's country mix and test accuracy at dev_seen's mix,
    over the countries present in both (coverage: their share of each split)."""
    rows = []
    for r in res.to_dict("records"):
        d = cell_dir(ev, r)
        ds, te = (pd.read_csv(d / f"{p}_per_country.csv", keep_default_na=False).set_index("country")
                  for p in ("dev_seen", "test"))
        common = ds.index.intersection(te.index)
        w_ds, w_te = ds.loc[common, "n"] / ds.loc[common, "n"].sum(), te.loc[common, "n"] / te.loc[common, "n"].sum()
        rows.append({k: r[k] for k in ("backbone", "variant", "ordering", "n")} | {
            "dev_seen": r["dev_seen_top1"], "test": r["test_top1"],
            "dev_seen_at_test_mix": float((w_te * ds.loc[common, "top1"]).sum()),
            "test_at_dev_seen_mix": float((w_ds * te.loc[common, "top1"]).sum()),
            "test_coverage": float(te.loc[common, "n"].sum() / te["n"].sum()),
            "dev_seen_coverage": float(ds.loc[common, "n"].sum() / ds["n"].sum())})
    t = pd.DataFrame(rows)
    t["drop"] = t["test"] - t["dev_seen"]
    t["mix_part"] = t["dev_seen_at_test_mix"] - t["dev_seen"]  # dev_seen accuracy, test's mix
    t["within_country_part"] = t["test"] - t["dev_seen_at_test_mix"]
    return t


def per_country_example(ev: Path, res: pd.DataFrame) -> pd.DataFrame:
    o, n, v = EXAMPLE
    out = []
    for r in res[(res["ordering"] == o) & (res["n"] == n) & (res["variant"] == v)].to_dict("records"):
        d = cell_dir(ev, r)
        ds, te = (pd.read_csv(d / f"{p}_per_country.csv", keep_default_na=False).set_index("country")
                  for p in ("dev_seen", "test"))
        t = te.merge(ds, how="left", left_index=True, right_index=True, suffixes=("_test", "_dev_seen"))
        t["test_share"] = t["n_test"] / t["n_test"].sum()
        # this country's share of test x its accuracy change: sums (over common countries) to
        # the within-country part of the drop
        t["within_contrib"] = t["test_share"] * (t["top1_test"] - t["top1_dev_seen"])
        out.append(t.reset_index().assign(backbone=r["backbone"], ordering=o, n=n, variant=v))
    return pd.concat(out, ignore_index=True)


def sequence_weighted(ev: Path, res: pd.DataFrame, test_csv: Path) -> pd.DataFrame:
    """Test top-1 per image vs with every sequence weighted equally (test keeps all images of a
    sequence; train/dev keep at most 3, so long test sequences weigh more per image)."""
    seq = pd.read_csv(test_csv, usecols=["id", "sequence"], dtype=str).set_index("id")["sequence"]
    rows = []
    for r in res.to_dict("records"):
        h = pd.read_csv(cell_dir(ev, r) / "test_hits.csv", dtype={"id": str})
        h["sequence"] = h["id"].map(seq)
        per_seq = h.groupby("sequence")["top1"].mean()
        long = h.groupby("sequence")["top1"].transform("size") > 3
        rows.append({k: r[k] for k in ("backbone", "variant", "ordering", "n")} | {
            "test_per_image": float(h["top1"].mean()), "test_per_sequence": float(per_seq.mean()),
            "test_in_seq_le3": float(h.loc[~long, "top1"].mean()),
            "test_in_seq_gt3": float(h.loc[long, "top1"].mean()), "share_in_seq_gt3": float(long.mean())})
    return pd.DataFrame(rows)


def groups_table(res: pd.DataFrame) -> pd.DataFrame:
    t = res[["backbone", "variant", "ordering", "n"]].copy()
    for g in ("head", "middle", "tail", "reliable"):
        for s in ("dev_seen", "test"):
            t[f"{s}_{g}"] = res[f"{s}_{g}_top1"]
        t[f"drop_{g}"] = t[f"test_{g}"] - t[f"dev_seen_{g}"]
    return t


def seen_creators(res: pd.DataFrame) -> pd.DataFrame:
    t = res[["backbone", "variant", "ordering", "n"]].copy()
    for s in ("dev_seen", "test"):
        t[f"{s}_seen_frac"] = res[f"{s}_seen_creator_n"] / res[f"{s}_n"]
        t[f"{s}_seen_top1"] = res[f"{s}_seen_creator_top1"]
        t[f"{s}_unseen_top1"] = res[f"{s}_unseen_creator_top1"]
        t[f"{s}_seen_minus_unseen"] = t[f"{s}_seen_top1"] - t[f"{s}_unseen_top1"]
    return t


def replication(t: pd.DataFrame, keys: list[str], stat: str) -> pd.DataFrame:
    w = t.pivot_table(index=keys, columns="split", values=[stat, f"{stat}_lo", f"{stat}_hi"])
    w.columns = [f"{s}_{c}" for c, s in w.columns]
    w = w.reset_index()
    for s in ("dev_seen", "dev_unseen"):
        w[f"sign_agree_{s}_test"] = np.sign(w[f"{s}_{stat}"]) == np.sign(w[f"test_{stat}"])
    for s in ("dev_seen", "dev_unseen", "test"):  # CI above 0 / below 0 / covers 0
        w[f"{s}_ci_side"] = np.select([w[f"{s}_{stat}_lo"] > 0, w[f"{s}_{stat}_hi"] < 0],
                                      ["+", "-"], "0")
    return w


def nat_vs_bal(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["test_top1", "test_macro_top1", "test_head_top1", "test_middle_top1", "test_tail_top1"]
    w = res.pivot_table(index=["backbone", "variant", "n"], columns="ordering", values=cols)
    out = w.copy()
    out.columns = [f"{c}_{o}" for c, o in w.columns]
    for c in cols:
        out[f"{c}_bal_minus_nat"] = w[(c, "bal")] - w[(c, "nat")]
    return out.reset_index()


def figures(out: Path, top: pd.DataFrame, gap: pd.DataFrame, seen: pd.DataFrame,
            share: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"streetclip": "C0", "clip_l14_336": "C1", "clip_b16": "C2"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, s in zip(axes, ("dev_seen", "dev_unseen")):
        for bb, d in top.groupby("backbone"):
            ax.scatter(100 * d[s], 100 * d["test"], s=14, color=colors[bb], label=bb)
        ax.plot([35, 90], [35, 90], "k--", lw=0.8)
        ax.set_xlabel(f"{s} top-1 (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test top-1 (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory: test vs dev top-1, 72 cells (dashed: equal)")
    fig.tight_layout()
    fig.savefig(out / "phase1_test_vs_dev.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, s in zip(axes, ("dev_seen", "dev_unseen")):
        ax.errorbar(100 * gap[f"{s}_gap"], 100 * gap["test_gap"],
                    xerr=100 * np.abs(gap[[f"{s}_gap_lo", f"{s}_gap_hi"]].sub(gap[f"{s}_gap"], axis=0).T.to_numpy()),
                    yerr=100 * np.abs(gap[["test_gap_lo", "test_gap_hi"]].sub(gap["test_gap"], axis=0).T.to_numpy()),
                    fmt="o", ms=4, lw=0.8, alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{s} gap (pt)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test gap (pt)")
    fig.suptitle("Exploratory: StreetCLIP - L/14-336 top-1 gap, dev vs test (24 cells, 95% CIs)")
    fig.tight_layout()
    fig.savefig(out / "phase1_gap_dev_vs_test.png", dpi=120)
    plt.close(fig)

    fr = seen.drop_duplicates(["ordering", "n"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for o, d in fr.groupby("ordering"):
        for s, ls in (("dev_seen", "-"), ("test", "--")):
            axes[0].plot(d["n"], 100 * d[f"{s}_seen_frac"], ls, marker="o", label=f"{o} {s}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training subset N")
    axes[0].set_ylabel("images whose creator is in the subset (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    k = seen[seen["variant"] == "probe_zeroshot"]
    for (bb, o), d in k.groupby(["backbone", "ordering"]):
        axes[1].plot(d["n"], 100 * d["test_seen_minus_unseen"], marker="o",
                     ls="-" if o == "nat" else "--", color=colors[bb], label=f"{bb} {o}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training subset N")
    axes[1].set_ylabel("test top-1: subset-seen - unseen creators (pt)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Exploratory: creators in the training subset (probe_zeroshot in the right panel)")
    fig.tight_layout()
    fig.savefig(out / "phase1_seen_creators.png", dpi=120)
    plt.close(fig)

    top15 = share.head(15)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(top15))
    for i, s in enumerate(("test", "dev_seen", "dev_unseen", "official_train")):
        ax.bar(x + (i - 1.5) * 0.2, 100 * top15[s], 0.2, label=s)
    ax.set_xticks(x, top15.index)
    ax.set_ylabel("share of images (%)")
    ax.legend(fontsize=8)
    ax.set_title("Exploratory: country mix, the 15 largest test countries")
    fig.tight_layout()
    fig.savefig(out / "phase1_country_mix.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="downloaded Kaggle output")
    ap.add_argument("--counts", type=Path, required=True, help="data audit country_counts.csv")
    ap.add_argument("--test-csv", type=Path, required=True, help="official OSV-5M test.csv")
    ap.add_argument("--out", type=Path, default=Path("results/v2"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    ev, out = args.run / "phase1_eval_test", args.out
    out.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(ev / "results.csv")
    counts = pd.read_csv(args.counts, keep_default_na=False)
    counts["in_classes"] = counts["in_classes"].astype(str) == "True"
    counts["reliable"] = counts["reliable"].astype(str) == "True"

    print("ALL OUTPUT BELOW IS EXPLORATORY")
    top = top1_table(res)
    top.to_csv(out / "phase1_test_vs_dev_top1.csv", index=False)
    print("\n[2] test - dev top-1 (pt) over the 72 cells")
    print((100 * top[["test_minus_dev_seen", "test_minus_dev_unseen"]].describe().loc[["mean", "min", "50%", "max"]]).round(1))
    print((100 * top.groupby("backbone")[["dev_seen", "dev_unseen", "test"]].mean()).round(1))

    share, tv, comp = country_mix(ev, res, counts)
    share.to_csv(out / "phase1_country_mix.csv")
    tv.to_csv(out / "phase1_country_tvd.csv", index=False)
    comp.to_csv(out / "phase1_group_composition.csv")
    print("\n[2] country mix: total variation distance")
    print(tv.round(3).to_string(index=False))
    print("share of images by group (head/middle/tail by official train images; reliable)")
    print((100 * comp).round(1))
    print("largest test countries (share %)")
    print((100 * share.head(10)[list(SPLITS) + ["official_train"]]).round(1))

    rw = reweighting(ev, res)
    rw.to_csv(out / "phase1_mix_reweighting.csv", index=False)
    print("\n[2] drop = test - dev_seen, split into country mix (dev_seen accuracy at test's mix) "
          "and within-country (pt; mean over cells)")
    print((100 * rw.groupby("backbone")[["drop", "mix_part", "within_country_part"]].mean()).round(1))
    print(f"coverage: test {rw['test_coverage'].min():.3f}, dev_seen {rw['dev_seen_coverage'].min():.3f} (min)")

    pc = per_country_example(ev, res)
    pc.to_csv(out / "phase1_per_country_example.csv", index=False)
    b = pc[pc["backbone"] == "clip_b16"].sort_values("within_contrib")
    print(f"\n[2] per country, clip_b16 {EXAMPLE}: largest within-country contributions to the drop")
    print(b[["country", "n_test", "n_dev_seen", "test_share", "top1_dev_seen", "top1_test", "within_contrib"]]
          .head(10).round(3).to_string(index=False))

    sw = sequence_weighted(ev, res, args.test_csv)
    sw.to_csv(out / "phase1_test_sequence_weighting.csv", index=False)
    print("\n[2] test: per image vs per sequence, and images in sequences of <=3 vs >3 (pt; mean over cells)")
    print((100 * sw.groupby("backbone")[["test_per_image", "test_per_sequence", "test_in_seq_le3",
                                         "test_in_seq_gt3"]].mean()).round(1))
    print(f"share of test images in sequences > 3: {sw['share_in_seq_gt3'].iloc[0]:.3f}")

    gr = groups_table(res)
    gr.to_csv(out / "phase1_groups_dev_vs_test.csv", index=False)
    print("\n[2] head/middle/tail/reliable top-1: dev_seen -> test (pt; mean over cells)")
    print((100 * gr[[c for c in gr if c.startswith(("dev_seen_", "test_", "drop_"))]].mean()).round(1).to_string())

    seen = seen_creators(res)
    seen.to_csv(out / "phase1_seen_creators.csv", index=False)
    print("\n[3] share of images whose creator is in the cell's training subset (%)")
    print((100 * seen.drop_duplicates(["ordering", "n"]).set_index(["ordering", "n"])[
        ["dev_seen_seen_frac", "test_seen_frac"]]).round(1))
    print("top-1 subset-seen - subset-unseen creators (pt), probe_zeroshot / knn")
    for v in ("probe_zeroshot", "knn"):
        k = seen[seen["variant"] == v].pivot_table(index=["ordering", "n"], columns="backbone",
                                                    values=["dev_seen_seen_minus_unseen", "test_seen_minus_unseen"])
        print(v)
        print((100 * k).round(1))

    gp = pd.read_csv(ev / "gaps.csv")
    gap = replication(gp.assign(split=gp["split"]), ["ordering", "n", "variant"], "gap")
    gap.to_csv(out / "phase1_gap_replication.csv", index=False)
    sh = replication(pd.read_csv(ev / "shrink.csv"), ["ordering", "variant", "n_small", "n_large"], "shrink")
    sh.to_csv(out / "phase1_shrink_replication.csv", index=False)
    print("\n[4] gap StreetCLIP - L/14-336: sign agreement with test")
    for s in ("dev_seen", "dev_unseen"):
        print(f"  {s}: {gap[f'sign_agree_{s}_test'].sum()} of {len(gap)} cells; CI side equal: "
              f"{(gap[f'{s}_ci_side'] == gap['test_ci_side']).sum()} of {len(gap)}")
    print((100 * gap.set_index(["ordering", "variant", "n"])[["dev_seen_gap", "dev_unseen_gap", "test_gap"]]).round(2))
    print("shrink gap(50K) - gap(500K)")
    print((sh.set_index(["ordering", "variant"])[["dev_seen_shrink", "dev_unseen_shrink", "test_shrink"]] * 100).round(2)
          .join(sh.set_index(["ordering", "variant"])[["dev_seen_ci_side", "dev_unseen_ci_side", "test_ci_side"]]))

    nb = nat_vs_bal(res)
    nb.to_csv(out / "phase1_nat_vs_bal_test.csv", index=False)
    print("\n[5] test, bal - nat (pt; mean over backbones per variant x N)")
    d = [c for c in nb if c.endswith("_bal_minus_nat")]
    print((100 * nb.groupby(["variant", "n"])[d].mean()).round(1).rename(columns=lambda c: c[5:-14]))

    figures(out, top, gap, seen, share)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
