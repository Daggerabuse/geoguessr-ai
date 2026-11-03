"""Download a small OSV-5M sample (range requests, without downloading the whole zip).

Usage: uv run python -m geoguessr_ai.download_sample [--n 3000] [--shard 04]
"""

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download
from tqdm import tqdm

REPO = "osv5m/osv5m"
SEED = 42
DATA = Path("data")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--shard", default="04")
    args = ap.parse_args()

    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = hf_hub_download(REPO, "test.csv", repo_type="dataset", local_dir=raw)
    meta = pd.read_csv(csv_path, dtype={"id": str})
    meta = meta.set_index("id")

    out_img = DATA / "sample" / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    zpath = f"datasets/{REPO}/images/test/{args.shard}.zip"
    with fs.open(zpath, "rb") as f, zipfile.ZipFile(f) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        rng = random.Random(SEED)
        picked = rng.sample(names, min(args.n, len(names)))
        # read in zip order: range requests become sequential, much faster
        order = {n: i for i, n in enumerate(names)}
        picked.sort(key=order.__getitem__)
        for name in tqdm(picked, desc="extract"):
            target = out_img / Path(name).name
            if target.exists() and target.stat().st_size > 0:
                continue
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

    ids = [Path(n).stem for n in picked]
    sample = meta.loc[meta.index.intersection(ids)]
    sample.to_csv(DATA / "sample" / "meta.csv")
    print(f"{len(picked)} images, {len(sample)} metadata rows -> {DATA / 'sample'}")


if __name__ == "__main__":
    main()
