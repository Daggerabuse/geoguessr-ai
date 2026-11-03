"""Data preparation: country selection, sequence cap, train/val/test split, download.

Designed to run on Kaggle/Colab (train.csv is 2.9 GB, images live in zip shards).
Locally it is only tested on a small sample (data/sample); see --local-zips.

Steps:
  1. (--stats-only) print the country distribution and exit.
  2. Filter countries; keep at most --max-per-seq frames per sequence.
  3. train.csv -> train + val: split by sequence (a sequence never lands on both sides).
     test: the official test.csv (--test-per-country per country).
  4. Write manifest.csv + classes.json.
  5. Extract only the needed images from the shard zips (via range requests).

Usage (Kaggle):
  python -m geoguessr_ai.prep_data --stats-only
  python -m geoguessr_ai.prep_data --out /kaggle/working/subset
"""

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO = "osv5m/osv5m"
N_SHARDS = {"train": 98, "test": 5}
COLS = ["id", "country", "sequence", "latitude", "longitude"]

# Default selection (continental/cultural diversity); override with --countries.
DEFAULT_COUNTRIES = [
    "SE", "NO", "FI", "DK", "GB", "IE", "ES", "PT", "DE", "AT", "FR", "IT", "PL", "RU",
    "TR", "US", "CA", "MX", "BR", "AR", "CL", "CO", "JP", "TW", "TH", "ID", "IN", "AU",
    "NZ", "ZA", "MA", "NG", "GH",
]


# ----------------------------------------------------------------- selection / split


def read_meta(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=COLS, dtype={"id": str, "sequence": str})


def cap_per_sequence(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Keep at most k frames per sequence (random, seeded)."""
    df = df.sample(frac=1.0, random_state=seed)
    return df[df.groupby("sequence").cumcount() < k]


def split_train_val(
    df: pd.DataFrame, train_n: int, val_n: int, seed: int
) -> pd.DataFrame:
    """Per-country val/train split by sequence. The returned df has a 'split' column."""
    out = []
    for country, g in df.groupby("country"):
        seqs = g["sequence"].drop_duplicates().sample(frac=1.0, random_state=seed)
        rank = {s: i for i, s in enumerate(seqs)}
        g = g.assign(_r=g["sequence"].map(rank)).sort_values("_r")
        counts = g.groupby("_r", sort=True).size()
        cum = counts.cumsum()
        # val: take sequences in order until val_n is reached; the rest is the train pool
        n_val_seq = int((cum.shift(fill_value=0) < val_n).sum())
        val_seqs = set(counts.index[:n_val_seq])
        is_val = g["_r"].isin(val_seqs)
        val = g[is_val].head(val_n).assign(split="val")
        train = g[~is_val].sample(frac=1.0, random_state=seed).head(train_n)
        out += [val, train.assign(split="train")]
    return pd.concat(out).drop(columns="_r")


def pick_test(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df.sample(frac=1.0, random_state=seed)
    return df[df.groupby("country").cumcount() < n].assign(split="test")


def build_manifest(args: argparse.Namespace, train_csv: Path, test_csv: Path):
    countries = args.countries
    tr = read_meta(train_csv)
    tr = tr[tr["country"].isin(countries)]
    tr = cap_per_sequence(tr, args.max_per_seq, args.seed)
    te = read_meta(test_csv)
    te = te[te["country"].isin(countries)]
    te = cap_per_sequence(te, args.max_per_seq, args.seed)

    trv = split_train_val(tr, args.train_per_country, args.val_per_country, args.seed)
    tst = pick_test(te, args.test_per_country, args.seed)
    trv["source"] = "train"
    tst["source"] = "test"
    m = pd.concat([trv, tst], ignore_index=True)

    classes = sorted(countries)
    idx = {c: i for i, c in enumerate(classes)}
    m["label"] = m["country"].map(idx)
    m["path"] = "images/" + m["source"] + "/" + m["id"] + ".jpg"
    return m, classes


# ----------------------------------------------------------------- download


def make_opener(local_zips: Path | None):
    """(source, shard_no) -> file-like object. Remote: HF range requests."""
    if local_zips is not None:
        return lambda src, i: open(local_zips / src / f"{i:02d}.zip", "rb")
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return lambda src, i: fs.open(
        f"datasets/{REPO}/images/{src}/{i:02d}.zip", "rb"
    )


def index_shard(opener, src: str, i: int) -> dict[str, tuple[int, str]]:
    """id -> (shard, name inside the zip). Only the central directory is read."""
    with opener(src, i) as f, zipfile.ZipFile(f) as zf:
        return {
            Path(n).stem: (i, n) for n in zf.namelist() if not n.endswith("/")
        }


def build_index(opener, src: str, cache: Path, workers: int) -> dict:
    if cache.exists():
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    idx: dict = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(index_shard, opener, src, i) for i in range(N_SHARDS[src])]
        for fu in tqdm(futs, desc=f"index {src}"):
            idx.update(fu.result())
    cache.write_text(json.dumps(idx))
    return idx


def save_image(data: bytes, dst: Path, resize: int) -> None:
    if resize:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = im.size
        s = resize / min(w, h)
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        im.save(dst, quality=92)
    else:
        dst.write_bytes(data)


def extract_shard(opener, src, shard, items, out: Path, resize: int) -> int:
    """items: [(id, zip_name)]; extract in zip order (sequential ranges)."""
    n = 0
    with opener(src, shard) as f, zipfile.ZipFile(f) as zf:
        order = {info.filename: j for j, info in enumerate(zf.infolist())}
        for id_, name in sorted(items, key=lambda t: order[t[1]]):
            dst = out / "images" / src / f"{id_}.jpg"
            if dst.exists() and dst.stat().st_size > 0:
                continue
            save_image(zf.read(name), dst, resize)
            n += 1
    return n


def download(m: pd.DataFrame, out: Path, opener, workers: int, resize: int) -> None:
    for src in ("train", "test"):
        ids = set(m.loc[m["source"] == src, "id"])
        if not ids:
            continue
        (out / "images" / src).mkdir(parents=True, exist_ok=True)
        idx = build_index(opener, src, out / f"index_{src}.json", workers)
        missing = ids - idx.keys()
        if missing:
            print(f"WARNING: skipping {len(missing)} ids not found in {src}")
        by_shard: dict[int, list] = {}
        for id_ in ids - missing:
            shard, name = idx[id_]
            by_shard.setdefault(shard, []).append((id_, name))
        with ThreadPoolExecutor(workers) as ex:
            futs = [
                ex.submit(extract_shard, opener, src, s, it, out, resize)
                for s, it in by_shard.items()
            ]
            for fu in tqdm(futs, desc=f"extract {src}"):
                fu.result()


# ----------------------------------------------------------------- main


def get_csv(name: str, given: Path | None, out: Path) -> Path:
    if given:
        return given
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=out / "raw"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/subset"))
    ap.add_argument("--train-csv", type=Path)
    ap.add_argument("--test-csv", type=Path)
    ap.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    ap.add_argument("--train-per-country", type=int, default=800)
    ap.add_argument("--val-per-country", type=int, default=100)
    ap.add_argument("--test-per-country", type=int, default=100)
    ap.add_argument("--max-per-seq", type=int, default=3)
    ap.add_argument("--resize", type=int, default=0, help="short side in px; 0 = original")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--no-download", action="store_true", help="manifest only")
    ap.add_argument("--local-zips", type=Path, help="DIR/{train,test}/NN.zip (for testing)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train_csv = get_csv("train.csv", args.train_csv, args.out)

    if args.stats_only:
        df = read_meta(train_csv)
        vc = df["country"].value_counts()
        seq = df.groupby("country")["sequence"].nunique().rename("n_sequences")
        stats = pd.concat([vc.rename("n_images"), seq], axis=1)
        stats.to_csv(args.out / "country_counts.csv")
        print(stats.head(40).to_string())
        print(f"\n{len(stats)} countries, {len(df)} images -> {args.out / 'country_counts.csv'}")
        return

    test_csv = get_csv("test.csv", args.test_csv, args.out)
    m, classes = build_manifest(args, train_csv, test_csv)
    m.to_csv(args.out / "manifest.csv", index=False)
    (args.out / "classes.json").write_text(json.dumps(classes))
    print(m.groupby(["split", "country"]).size().unstack(0).fillna(0).astype(int).to_string())

    if args.no_download:
        return
    download(m, args.out, make_opener(args.local_zips), args.workers, args.resize)
    print("done:", args.out)


if __name__ == "__main__":
    main()
