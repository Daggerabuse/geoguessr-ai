"""Phase 0b (CPU only, no GPU quota): how the train shard zips are filled and how fast they load.

1. Are the 98 train zips random samples of train.csv? Per zip: country distribution (total
   variation distance to the overall one), how many zips a creator / a sequence spans, and
   how often consecutive entries share a sequence or a creator. Every statistic is printed
   next to the same statistic after randomly reassigning images to zips (same zip sizes):
   "random-like" means observed ~= baseline.
2. Full-zip download speed (hf_hub_download): one zip alone, then two at once.
3. Byte equality: N seeded ids read by HTTP range requests (embed.py's old path) and from
   the downloaded zip have the same sha256.
4. Decode + preprocessing speed from a local zip (224 and 336 crops), one process and a
   process per CPU.

Usage (Kaggle, CPU session, internet on, HF_TOKEN set as for kaggle_phase0.sh):
  python -m geoguessr_ai.zip_audit --out /kaggle/working/phase0b
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geoguessr_ai.embed import clip_center_crop
from geoguessr_ai.prep_data import REPO, build_index, get_csv, make_opener
from geoguessr_ai.train import to_tensor_array
from geoguessr_ai.v2_data import read_meta

# ----------------------------------------------------------------- 1. randomness


def zip_stats(meta: pd.DataFrame, min_creator_images: int = 20) -> dict:
    """meta: id, country, sequence, creator_id, shard, pos (order inside the zip)."""
    overall = meta["country"].value_counts(normalize=True)
    tvd = [0.5 * (g["country"].value_counts(normalize=True).reindex(overall.index, fill_value=0)
                  - overall).abs().sum() for _, g in meta.groupby("shard")]
    creators = meta.groupby("creator_id")
    big = creators.size()[lambda s: s >= min_creator_images].index
    seq = meta.groupby("sequence")
    multi = seq.size()[lambda s: s >= 2].index
    o = meta.sort_values(["shard", "pos"])
    same_zip = o["shard"].to_numpy()[1:] == o["shard"].to_numpy()[:-1]

    def consecutive(col: str) -> float:
        v = o[col].to_numpy()
        return float((v[1:] == v[:-1])[same_zip].mean())

    return {
        "country_tvd_mean": float(np.mean(tvd)), "country_tvd_max": float(np.max(tvd)),
        f"zips_per_creator_mean(>={min_creator_images} images)":
            float(creators["shard"].nunique()[big].mean()),
        "multi_frame_sequences_in_one_zip": float((seq["shard"].nunique()[multi] == 1).mean()),
        "consecutive_same_sequence": consecutive("sequence"),
        "consecutive_same_creator": consecutive("creator_id"),
    }


def randomness(meta: pd.DataFrame, reps: int, seed: int) -> dict:
    """Observed statistics and their range over `reps` random reassignments to zips."""
    rng = np.random.default_rng(seed)
    base = [zip_stats(meta.assign(shard=rng.permutation(meta["shard"].to_numpy()),
                                  pos=rng.permutation(len(meta)))) for _ in range(reps)]
    return {k: {"observed": v, "random_min": min(b[k] for b in base),
                "random_max": max(b[k] for b in base)} for k, v in zip_stats(meta).items()}


# ----------------------------------------------------------------- 2.-4. speed and bytes


def download(i: int, zips: Path) -> dict:
    from huggingface_hub import hf_hub_download

    t = time.time()
    p = Path(hf_hub_download(REPO, f"images/train/{i:02d}.zip", repo_type="dataset",
                             local_dir=zips))
    s = time.time() - t
    return {"zip": i, "gb": round(p.stat().st_size / 1e9, 3), "seconds": round(s, 1),
            "mb_per_s": round(p.stat().st_size / 1e6 / s, 1)}


def sha256s(opener, shard: int, names: list[str]) -> list[str]:
    with opener("train", shard) as f, zipfile.ZipFile(f) as zf:
        return [hashlib.sha256(zf.read(n)).hexdigest() for n in names]


def prep(blobs: list[bytes], sizes: tuple[int, ...]) -> int:
    """Decode once, crop+normalize for each size (as embed.py does per model)."""
    for b in blobs:
        with Image.open(io.BytesIO(b)) as im:
            rgb = im.convert("RGB")
        for s in sizes:
            to_tensor_array(clip_center_crop(rgb, s))
    return len(blobs)


def prep_speed(blobs: list[bytes], sizes: tuple[int, ...], procs: int) -> float:
    t = time.time()
    if procs == 1:
        prep(blobs, sizes)
    else:
        chunks = [blobs[k::procs] for k in range(procs)]
        with ProcessPoolExecutor(procs) as ex:
            list(ex.map(prep, chunks, [sizes] * procs))
    return round(len(blobs) / (time.time() - t), 1)


# ----------------------------------------------------------------- main


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/phase0b"))
    ap.add_argument("--raw", type=Path, default=Path("/kaggle/temp/raw"))
    ap.add_argument("--zips", type=Path, default=Path("/kaggle/temp/zips"))
    ap.add_argument("--train-csv", type=Path)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-bytes-check", type=int, default=200)
    ap.add_argument("--n-decode", type=int, default=2000)
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res: dict = {"cpu_count": os.cpu_count()}

    def save():
        (args.out / "zip_audit.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))

    # 1. randomness (the index keeps each zip's central-directory order)
    tr = read_meta(get_csv("train.csv", args.train_csv, args.raw.parent))
    index = build_index(make_opener(None), "train", args.raw / "index_train.json", 16)
    pos = pd.DataFrame([(k, s, j) for j, (k, (s, _)) in enumerate(index.items())],
                       columns=["id", "shard", "pos"])
    meta = tr.merge(pos, on="id", how="inner")
    res["ids"] = {"train_csv": len(tr), "in_zips": len(index), "matched": len(meta)}
    res["randomness"] = randomness(meta, args.reps, args.seed)
    save()

    # 2. download speed
    free = shutil.disk_usage(args.zips.parent if args.zips.parent.exists() else "/").free / 1e9
    if free < args.min_free_gb:
        sys.exit(f"only {free:.0f} GB free next to {args.zips}; need {args.min_free_gb}")
    res["disk_free_gb_before"] = round(free, 1)
    res["download_one"] = download(0, args.zips)
    with ThreadPoolExecutor(2) as ex:
        t = time.time()
        res["download_two_parallel"] = list(ex.map(download, [1, 2], [args.zips] * 2))
        res["download_two_parallel_mb_per_s_total"] = round(
            sum(d["gb"] for d in res["download_two_parallel"]) * 1e3 / (time.time() - t), 1)
    save()

    # 3. bytes: range requests vs the downloaded zip
    names = sorted(n for (s, n) in index.values() if s == 0)
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(names, size=args.n_bytes_check, replace=False))
    local = make_opener(args.zips / "images")
    remote = sha256s(make_opener(None), 0, pick)
    res["bytes_equal"] = {"n": len(pick), "equal": int(sum(a == b for a, b in
                                                          zip(remote, sha256s(local, 0, pick))))}
    save()

    # 4. decode + preprocessing speed from the local zip
    with local("train", 0) as f, zipfile.ZipFile(f) as zf:
        blobs = [zf.read(n) for n in names[: args.n_decode]]
    procs = os.cpu_count() or 1
    res["decode_preprocess_images_per_s"] = {
        f"{'+'.join(map(str, sizes))}_{p}proc": prep_speed(blobs, sizes, p)
        for sizes in ((224,), (336,), (224, 336)) for p in (1, procs)}
    if not args.keep_zips:
        shutil.rmtree(args.zips)
    save()
    if res["bytes_equal"]["equal"] != res["bytes_equal"]["n"]:
        sys.exit("FAILED: range-request bytes differ from the downloaded zip")


if __name__ == "__main__":
    main()
