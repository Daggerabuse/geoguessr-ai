"""V2 data: dataset audit, global class set, creator-disjoint dev split, nested training subsets.

No images are written here; embed.py streams the listed ids from the shard zips.

Class list: the countries that occur in the official test split AND in the official train
split (sorted ISO-2 codes). Test images of a country that is absent from train cannot be
predicted correctly by any trained head; they stay in test.csv with label -1 and count as
errors in every global metric (see geo_eval.py), so the test split stays the full official one.

Outputs in --out:
  audit.json          split sizes, countries only in train / only in test, train-test overlap
                      of ids, sequences and creators, missing creators
  country_counts.csv  per country: train/test images, sequences, creators, share of test images
                      whose creator is in train, freq_group (head/middle/tail: thirds of the
                      class list by official train images) and reliable (enough test images and
                      sequences for a per-country estimate)
  classes.json, test.csv, dev.csv, train.csv, config.json (skipped with --audit-only):
    test.csv   the full official test split; label = -1 for countries absent from train
    dev.csv    "dev_unseen": --dev-n images (1 per sequence) from a seeded random
               --dev-creator-frac of the creators; none of these creators is in train.csv
               (independent of --max-per-seq). Second report: generalization to new creators.
    dev_seen.csv  --dev-seen-n images (1 per sequence) from seeded random sequences of the
               other creators; those whole sequences leave the pool, and each creator keeps
               at least one other sequence in the official train split. Mirrors the official
               test (sequence-disjoint, 98% of test images have a creator seen in train);
               hyperparameters are chosen on it (geo_eval.py). Unlike the official test it does
               not enforce the 1 km distance to training images.
    train.csv  the training pool with two nested orderings:
                 nat_rank  seeded random order (natural, skewed country distribution)
                 bal_rank  round-robin over countries in nat order (as balanced as the data
                           allows; a country runs out when its images do)
               A subset of size N is `rank < N`, so 50K is inside 100K and so on.
    config.json arguments, counts, rows dropped for leakage, a hash of every id list

Missing creator_id (a few test rows): the sequence stands in for the creator ("seq:<id>"),
which keeps every sequence on one side of the dev split.

Usage (Kaggle):
  python -m geoguessr_ai.v2_data --out /kaggle/working/v2 --audit-only
  python -m geoguessr_ai.v2_data --out /kaggle/working/v2 --sizes 50000 100000 200000 500000
  # diversity arm: same sizes, different frames-per-sequence cap (dev is unchanged)
  python -m geoguessr_ai.v2_data --out /kaggle/working/v2_seq1 --max-per-seq 1
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai.prep_data import cap_per_sequence, get_csv

COLS = ["id", "country", "sequence", "latitude", "longitude", "creator_id", "quadtree_10_1000"]


def read_meta(path: Path) -> pd.DataFrame:
    """The columns V2 needs; fails with the missing column names if the schema differs."""
    head = pd.read_csv(path, nrows=0).columns
    missing = [c for c in COLS if c not in head]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    # keep_default_na=False: "NA" is Namibia's code, pandas would read it as missing
    df = pd.read_csv(path, usecols=COLS, keep_default_na=False,
                     dtype={c: str for c in ("id", "country", "sequence", "creator_id")})
    return normalize_creators(df)


def normalize_creators(df: pd.DataFrame) -> pd.DataFrame:
    """creator_id as written in both CSVs ('123.0') -> '123'; empty -> 'seq:<sequence>'."""
    c = df["creator_id"].str.removesuffix(".0")
    return df.assign(creator_id=c.where(c != "", "seq:" + df["sequence"]))


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------- audit


def audit(tr: pd.DataFrame, te: pd.DataFrame, classes: list[str], min_test_images: int,
          min_test_sequences: int) -> tuple[pd.DataFrame, dict]:
    def counts(df: pd.DataFrame, name: str) -> pd.DataFrame:
        g = df.groupby("country")
        return pd.DataFrame({f"{name}_images": g.size(), f"{name}_sequences": g["sequence"].nunique(),
                             f"{name}_creators": g["creator_id"].nunique()})

    seen = te["creator_id"].isin(set(tr["creator_id"]))
    stats = counts(tr, "train").join(counts(te, "test"), how="outer").fillna(0).astype(int)
    stats["test_seen_creator_frac"] = seen.groupby(te["country"]).mean()
    stats["in_classes"] = stats.index.isin(classes)
    # head/middle/tail: thirds of the class list ordered by official train images
    order = stats.loc[stats["in_classes"], "train_images"].sort_values(ascending=False).index
    third = int(np.ceil(len(order) / 3))
    stats["freq_group"] = ""
    for i, grp in enumerate(("head", "middle", "tail")):
        stats.loc[order[i * third : (i + 1) * third], "freq_group"] = grp
    stats["reliable"] = (stats["in_classes"] & (stats["test_images"] >= min_test_images)
                         & (stats["test_sequences"] >= min_test_sequences))
    train_c, test_c = set(tr["country"]), set(te["country"])
    summary = {
        "train_images": len(tr), "test_images": len(te),
        "train_countries": len(train_c), "test_countries": len(test_c), "classes": len(classes),
        "countries_only_in_train": sorted(train_c - test_c),
        "countries_only_in_test": sorted(test_c - train_c),
        "test_images_of_countries_only_in_test": int((~te["country"].isin(train_c)).sum()),
        "shared_ids": len(set(tr["id"]) & set(te["id"])),
        "shared_sequences": len(set(tr["sequence"]) & set(te["sequence"])),
        "test_seen_creator_frac": float(seen.mean()),
        "sequences_with_several_creators": {
            n: int((d.groupby("sequence")["creator_id"].nunique() > 1).sum())
            for n, d in (("train", tr), ("test", te))},
        "missing_creator_rows": {n: int(d["creator_id"].str.startswith("seq:").sum())
                                 for n, d in (("train", tr), ("test", te))},
        "reliable_countries": int(stats["reliable"].sum()),
        "freq_group_sizes": stats.loc[stats["in_classes"], "freq_group"].value_counts().to_dict(),
    }
    return stats.sort_values("train_images", ascending=False), summary


# ----------------------------------------------------------------- split / subsets


def creator_split(df: pd.DataFrame, frac: float, seed: int) -> pd.Series:
    """Boolean mask: rows whose creator is in a seeded random `frac` of all creators."""
    creators = np.sort(df["creator_id"].unique())
    rng = np.random.default_rng(seed)
    dev = set(creators[rng.random(len(creators)) < frac])
    return df["creator_id"].isin(dev)


def add_ranks(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """nat_rank: seeded random order; bal_rank: round-robin over countries in that order."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["nat_rank"] = np.arange(len(df))
    per_country = df.groupby("country").cumcount()
    order = np.lexsort((df["nat_rank"].to_numpy(), per_country.to_numpy()))
    df["bal_rank"] = 0
    df.loc[order, "bal_rank"] = np.arange(len(df))
    return df


def subset(train: pd.DataFrame, strategy: str, n: int) -> pd.DataFrame:
    """The first n training rows under strategy 'nat' or 'bal' (nested in n)."""
    return train[train[f"{strategy}_rank"] < n]


def pick_dev_seen(rest: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n seeded random sequences of `rest` (1 frame each), dropping a pick whose creator would
    have no other sequence left in `rest` (so every dev_seen creator is seen in training)."""
    chosen = cap_per_sequence(rest, 1, seed).sample(frac=1.0, random_state=seed).head(n)
    left = set(rest.loc[~rest["sequence"].isin(set(chosen["sequence"])), "creator_id"])
    return chosen[chosen["creator_id"].isin(left)]


def subsets_nested(train: pd.DataFrame, sizes) -> bool:
    """Every subset (nat and bal) is contained in the next larger one."""
    for s in ("nat", "bal"):
        ids = [set(subset(train, s, n)["id"]) for n in sorted(sizes)]
        if not all(a <= b for a, b in itertools.pairwise(ids)):
            return False
    return True


def build(tr: pd.DataFrame, te: pd.DataFrame, args: argparse.Namespace):
    """(parts {train, dev, dev_seen, test}, classes, dropped) from normalized official metadata."""
    classes = sorted(set(te["country"]) & set(tr["country"]))
    idx = {c: i for i, c in enumerate(classes)}
    tr = tr[tr["country"].isin(idx)]
    # the official split separates sequences; enforce it (and ids) instead of trusting it
    leak = tr["id"].isin(set(te["id"])) | tr["sequence"].isin(set(te["sequence"]))
    dropped = int(leak.sum())
    tr = tr[~leak]

    is_dev = creator_split(tr, args.dev_creator_frac, args.seed)
    dev = cap_per_sequence(tr[is_dev], 1, args.seed)
    dev = dev.sample(n=min(args.dev_n, len(dev)), random_state=args.seed)
    rest = tr[~is_dev]
    dev_seen = pick_dev_seen(rest, args.dev_seen_n, args.seed)
    rest = rest[~rest["sequence"].isin(set(dev_seen["sequence"]))]
    pool = add_ranks(cap_per_sequence(rest, args.max_per_seq, args.seed), args.seed)
    n_max = max(args.sizes)
    pool = pool[(pool["nat_rank"] < n_max) | (pool["bal_rank"] < n_max)]

    out = {"train": pool, "dev": dev, "dev_seen": dev_seen, "test": te}
    for name, d in out.items():
        out[name] = d.assign(label=d["country"].map(idx).fillna(-1).astype(int)).reset_index(drop=True)
    check_separation(out)
    return out, classes, dropped


def check_separation(parts: dict[str, pd.DataFrame]) -> None:
    """No id or sequence shared by two parts; no dev creator in train or dev_seen."""
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for col in ("id", "sequence"):
                shared = set(parts[a][col]) & set(parts[b][col])
                if shared:
                    raise AssertionError(f"{len(shared)} {col}s in both {a} and {b}")
    for other in ("train", "dev_seen"):
        if other in parts and set(parts["dev"]["creator_id"]) & set(parts[other]["creator_id"]):
            raise AssertionError(f"dev and {other} share creators")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/v2"))
    ap.add_argument("--train-csv", type=Path)
    ap.add_argument("--test-csv", type=Path)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50_000, 100_000, 200_000, 500_000])
    ap.add_argument("--max-per-seq", type=int, default=3)
    ap.add_argument("--dev-creator-frac", type=float, default=0.02)
    ap.add_argument("--dev-n", type=int, default=20_000)
    ap.add_argument("--dev-seen-n", type=int, default=20_000)
    ap.add_argument("--min-test-images", type=int, default=100, help="'reliable' country")
    ap.add_argument("--min-test-sequences", type=int, default=50, help="'reliable' country")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tr = read_meta(get_csv("train.csv", args.train_csv, args.out))
    te = read_meta(get_csv("test.csv", args.test_csv, args.out))
    classes = sorted(set(te["country"]) & set(tr["country"]))
    stats, summary = audit(tr, te, classes, args.min_test_images, args.min_test_sequences)
    stats.to_csv(args.out / "country_counts.csv")
    (args.out / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.audit_only:
        return

    parts, classes, dropped = build(tr, te, args)
    (args.out / "classes.json").write_text(json.dumps(classes))
    for name, d in parts.items():
        d.to_csv(args.out / f"{name}.csv", index=False)
    train = parts["train"]
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_classes": len(classes),
        "n": {name: len(d) for name, d in parts.items()},
        "test_unlabelled": int((parts["test"]["label"] < 0).sum()),
        "train_rows_dropped_for_test_overlap": dropped,
        "subsets": {
            f"{s}_{n}": {"n": len(subset(train, s, n)),
                         "countries": int(subset(train, s, n)["country"].nunique()),
                         "ids_sha256": ids_hash(sorted(subset(train, s, n)["id"]))}
            for s in ("nat", "bal") for n in args.sizes
        },
        "ids_sha256": {name: ids_hash(sorted(d["id"])) for name, d in parts.items()},
        "pool_size": len(train),
        "subsets_nested": subsets_nested(train, args.sizes),
    }
    if not config["subsets_nested"]:
        raise AssertionError("training subsets are not nested")
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("n_classes", "n", "pool_size", "subsets_nested",
                                             "test_unlabelled",
                                             "train_rows_dropped_for_test_overlap", "subsets")}, indent=2))


if __name__ == "__main__":
    main()
