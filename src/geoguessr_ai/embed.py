"""Stream images from the OSV-5M shard zips, embed them with a frozen encoder, keep only embeddings.

No image is written to disk: each needed image is read from its zip (range requests),
preprocessed exactly like the model's CLIPImageProcessor (see clip_center_crop) and embedded.

Cache layout (one directory per encoder, e.g. emb/clip_b16/):
  encoder.json                        model, image size, embedding dim, preprocessing, GPU
                                      type; a run with any of them different is refused (fp16
                                      embeddings of a T4 and a P100 differ slightly)
  <source>_<shard>_<run id>.npz       ids (str), emb (float16, L2-normalized), failed (str)
  throughput.jsonl                    one line per run: end-to-end images/s (all, and after
                                      the first batch), GPU busy fraction, zip read and
                                      decode+preprocess thread-seconds, peak GPU memory
  ../index_<source>.json              id -> (shard, name in zip), shared by all encoders

Guarantees (tested in tests/test_v2.py):
  - written atomically (temp file without the .npz suffix, then rename): readers never see
    a half-written file, a killed run loses only the shards in progress;
  - run ids are unique, so concurrent or same-second runs never overwrite each other;
  - --parts/--part split the shards by shard number: two processes with the same --parts
    never embed the same image;
  - --limit N is global: the target is a seeded sample of N of the wanted ids, the same for
    every --parts/--part and every rerun (a resume finishes exactly that sample); each part
    embeds the target ids in its own shards, so all parts together embed N ids;
  - ids already in the cache for the same source are skipped (resume), and a train id is
    never served for a test id or the other way round;
  - if an id is still found in two files (e.g. overlapping manual runs), the first file in
    name order wins, deterministically, and the duplicate count is reported.

Usage (Kaggle; the second command runs in a second process on the second T4):
  python -m geoguessr_ai.embed --ids /kaggle/working/v2/train.csv --source train \\
      --model openai/clip-vit-base-patch16 --out /kaggle/working/emb/clip_b16 --parts 2 --part 0
  python -m geoguessr_ai.embed ... --parts 2 --part 1 --device cuda:1
  # smoke test / throughput benchmark: a seeded random sample of 2000 training images
  python -m geoguessr_ai.embed --ids /kaggle/working/v2/train.csv --source train ... --limit 2000
  # verify a cache: every target id exactly once; optionally the same embeddings as another cache
  python -m geoguessr_ai.embed --check --ids ... --source train --out DIR --limit 2000 [--reference DIR2]

--model: any CLIPVisionModelWithProjection checkpoint, e.g. openai/clip-vit-base-patch16,
openai/clip-vit-large-patch14-336, geolocal/StreetCLIP, or the V1 export
Daggerabuse/geoguessr-clip-vitb16.
"""

import argparse
import collections
import io
import json
import os
import queue
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geoguessr_ai.baseline_clip import as_embedding, library_versions
from geoguessr_ai.prep_data import build_index, make_opener
from geoguessr_ai.train import tiny_vision_config, to_tensor_array

DONE = object()
PREPROCESSING = "clip_image_processor: shortest edge (int), bicubic, center crop, CLIP mean/std"


# ----------------------------------------------------------------- preprocessing


def clip_center_crop(im: Image.Image, size: int) -> Image.Image:
    """Same pixels as transformers' CLIPImageProcessor (resize + crop part).

    The short side becomes `size` and the long side int(size * long / short), truncated like
    the processor (V1's resize_short rounds instead, a 1-pixel difference in the long side).
    """
    w, h = im.size
    nw, nh = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


# ----------------------------------------------------------------- cache


def cache_files(out: Path, source: str) -> list[Path]:
    return sorted(out.glob(f"{source}_*.npz"))


def cached_ids(out: Path, source: str) -> set[str]:
    return {str(i) for f in cache_files(out, source) for i in np.load(f)["ids"]}


def load_embeddings(out: Path, ids, source: str) -> tuple[np.ndarray, np.ndarray]:
    """(len(ids), D) float32 unit rows aligned with ids (zeros where missing), a found mask."""
    pos = {str(i): j for j, i in enumerate(ids)}
    emb, found, dup = None, np.zeros(len(pos), dtype=bool), 0
    for f in cache_files(out, source):
        z = np.load(f)
        rows = [(pos[i], k) for k, i in enumerate(map(str, z["ids"])) if i in pos]
        dup += sum(found[d] for d, _ in rows)
        rows = [(d, k) for d, k in rows if not found[d]]  # first file in name order wins
        if not rows:
            continue
        if emb is None:
            emb = np.zeros((len(pos), z["emb"].shape[1]), dtype=np.float32)
        dst, src = map(list, zip(*rows))
        emb[dst], found[dst] = z["emb"][src], True
    if emb is None:
        raise FileNotFoundError(f"none of the {len(pos)} {source} ids are embedded in {out}")
    if dup:
        print(f"WARNING: {dup} {source} ids are in more than one file in {out}; kept the first")
    # fp16 storage rounds the unit norm by up to ~1e-3; renormalize so dot product == cosine
    emb[found] /= np.linalg.norm(emb[found], axis=1, keepdims=True)
    return emb, found


def atomic_write(path: Path, write) -> None:
    """write(file) into a temp file next to path (not matching *.npz/*.json), then rename."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_npz(path: Path, ids: list[str], emb: np.ndarray, failed: list[str]) -> None:
    atomic_write(path, lambda f: np.savez(f, ids=np.array(ids, dtype=str),
                                          emb=emb.astype(np.float16), failed=np.array(failed, dtype=str)))


def publish_once(path: Path, make) -> None:
    """Write-once file shared by parallel processes: make(tmp) writes a complete temp file,
    os.link publishes it only if path does not exist yet (atomic; the first writer wins and
    nobody ever sees a partial file). os.replace would let a second writer overwrite it."""
    if path.exists():
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        make(tmp)
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass  # another process published first; the caller reads its version
    finally:
        tmp.unlink(missing_ok=True)


def check_encoder(out: Path, info: dict) -> None:
    """Refuse to mix encoders or preprocessing in one cache directory."""
    path = out / "encoder.json"
    publish_once(path, lambda tmp: tmp.write_text(json.dumps(info, indent=2)))
    have = json.loads(path.read_text())
    if have != info:
        raise ValueError(f"{out} holds embeddings of {have}, not {info}; use another --out")


def shared_index(opener, source: str, path: Path, workers: int) -> dict:
    """prep_data.build_index, published once so parallel processes can share it."""
    publish_once(path, lambda tmp: build_index(opener, source, tmp, workers))
    return build_index(opener, source, path, workers)  # reads the cache file


def target_ids(want, index: dict, limit: int, seed: int) -> list[str]:
    """The ids the cache should end up with: wanted ids that are in the shards, or a seeded
    sample of `limit` of them. Independent of the cache and of --parts/--part."""
    ids = sorted(set(map(str, want)))
    missing = [i for i in ids if i not in index]
    if missing:
        print(f"WARNING: {len(missing)} ids are not in the shards, skipped")
    ids = [i for i in ids if i in index]
    if limit and limit < len(ids):
        ids = sorted(np.random.default_rng(seed).choice(ids, size=limit, replace=False))
    return ids


def plan(want, cached: set[str], index: dict, parts: int, part: int, limit: int,
         seed: int) -> dict[int, list]:
    """shard -> [(id, name in zip)]: target ids in this part's shards, not cached yet."""
    mine = [i for i in target_ids(want, index, limit, seed)
            if i not in cached and index[i][0] % parts == part]
    by_shard: dict[int, list] = {}
    for id_ in mine:
        shard, name = index[id_]
        by_shard.setdefault(shard, []).append((id_, name))
    return by_shard


# ----------------------------------------------------------------- run


def shard_batches(opener, src, shard, items, size, batch_size, q: queue.Queue,
                  stop: threading.Event, timings: list) -> None:
    """Producer: read the shard's images in zip order and queue preprocessed batches.
    Appends (seconds reading the zip incl. download, seconds decoding + preprocessing) to
    timings; time spent waiting on a full queue is in neither."""

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=1)
                return True
            except queue.Full:
                continue
        return False

    ids, xs, failed = [], [], []
    t_read = t_prep = 0.0
    try:
        tic = time.perf_counter()
        with opener(src, shard) as f, zipfile.ZipFile(f) as zf:
            order = {info.filename: j for j, info in enumerate(zf.infolist())}
            t_read += time.perf_counter() - tic
            for id_, name in sorted(items, key=lambda t: order[t[1]]):
                try:
                    tic = time.perf_counter()
                    data = zf.read(name)
                    t_read += time.perf_counter() - tic
                    tic = time.perf_counter()
                    with Image.open(io.BytesIO(data)) as im:
                        crop = clip_center_crop(im.convert("RGB"), size)
                    xs.append(to_tensor_array(crop).astype(np.float16))  # halves queue memory
                    ids.append(id_)
                    t_prep += time.perf_counter() - tic
                except OSError:  # corrupt image: record it, do not stop the shard
                    failed.append(id_)
                if len(xs) == batch_size:
                    if not put((shard, ids, np.stack(xs), [])):
                        return
                    ids, xs = [], []
        put((shard, ids, np.stack(xs) if xs else None, failed))
    except Exception as e:  # noqa: BLE001 - any error (network, zip) must still send DONE below, or the consumer waits forever; ids not yet queued stay missing and a rerun resumes them
        print(f"\nWARNING: shard {src}/{shard:02d} failed: {e!r}")
    timings.append((t_read, t_prep))
    put((shard, DONE, None, None))


def load_encoder(name: str, smoke: bool, device):
    import torch
    from transformers import CLIPVisionModelWithProjection

    vision = (CLIPVisionModelWithProjection(tiny_vision_config()) if smoke
              else CLIPVisionModelWithProjection.from_pretrained(name))
    vision = vision.to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    return vision.eval(), vision.config.image_size, vision.config.projection_dim


def gpu_utilization(device) -> float | None:
    """Instantaneous GPU utilization in % (needs pynvml; None if unavailable)."""
    import torch

    try:
        return float(torch.cuda.utilization(device))
    except Exception:  # noqa: BLE001 - optional metric, never worth failing a run
        return None


def run(args: argparse.Namespace) -> dict:
    import torch

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    vision, size, dim = load_encoder(args.model, args.smoke, device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    check_encoder(args.out, {"model": args.model, "smoke": args.smoke, "image_size": size,
                             "dim": dim, "dtype": "float16", "preprocessing": PREPROCESSING,
                             "gpu": gpu})

    want = read_ids(args.ids)
    cached = cached_ids(args.out, args.source)
    opener = make_opener(args.local_zips)
    t_index = time.time()
    index = shared_index(opener, args.source, args.out.parent / f"index_{args.source}.json",
                         args.workers)
    t_index = time.time() - t_index
    target = target_ids(want, index, args.limit, args.seed)
    in_part = sum(index[i][0] % args.parts == args.part for i in target)
    by_shard = plan(want, cached, index, args.parts, args.part, args.limit, args.seed)
    n_todo = sum(map(len, by_shard.values()))
    print(f"target: {len(target)} unique ids (--limit {args.limit or 'none'}, all parts together); "
          f"part {args.part}/{args.parts}: {in_part} unique ids in its shards, "
          f"{in_part - n_todo} already cached, {n_todo} to embed ({len(by_shard)} shards)")

    q: queue.Queue = queue.Queue(maxsize=args.workers + 2)
    stop = threading.Event()
    acc: dict[int, tuple[list, list, list]] = {s: ([], [], []) for s in by_shard}
    timings: list = []
    t0, t_gpu, n_done, util, first = time.time(), 0.0, 0, [], None
    run_id = f"{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    pool = ThreadPoolExecutor(args.workers)
    try:
        for s, items in by_shard.items():
            pool.submit(shard_batches, opener, args.source, s, items, size, args.batch_size, q,
                        stop, timings)
        remaining = len(by_shard)
        while remaining:
            shard, ids, x, failed = q.get()
            if ids is DONE:
                remaining -= 1
                got_ids, embs, fails = acc.pop(shard)
                if got_ids or fails:
                    emb = np.concatenate(embs) if embs else np.zeros((0, dim), np.float16)
                    save_npz(args.out / f"{args.source}_{shard:02d}_{run_id}.npz", got_ids, emb, fails)
                continue
            acc[shard][2].extend(failed)
            if x is None:
                continue
            t1 = time.time()
            with torch.inference_mode():
                px = torch.from_numpy(x).to(device, dtype=vision.dtype, non_blocking=True)
                e = as_embedding(vision(pixel_values=px), "image", dim).float()
                e = torch.nn.functional.normalize(e, dim=-1).half().cpu().numpy()
            t_gpu += time.time() - t1
            if device.type == "cuda":
                util.append(gpu_utilization(device))
            acc[shard][0].extend(ids)
            acc[shard][1].append(e)
            n_done += len(ids)
            el = time.time() - t0
            if first is None:  # warm-up (CUDA init, first zip reads) left out of the steady rate
                first = (n_done, el)
            print(f"  {n_done}/{n_todo}  {n_done / el:.1f} img/s  gpu busy {t_gpu / el:.0%}", end="\r")
    finally:
        stop.set()  # unblocks producers if the loop above failed (e.g. out of GPU memory)
        pool.shutdown(wait=True, cancel_futures=True)
    elapsed = time.time() - t0
    util = [u for u in util if u is not None]
    t_read, t_prep = (sum(c) for c in zip(*timings)) if timings else (0.0, 0.0)
    steady = first is not None and n_done > first[0] and elapsed > first[1]
    stats = {
        "run_id": run_id, "model": args.model, "source": args.source, "device": str(device),
        "gpu_name": gpu, "image_size": size, "dim": dim, "batch_size": args.batch_size,
        "workers": args.workers, "part": f"{args.part}/{args.parts}", "limit": args.limit,
        "target": len(target), "images": n_done, "index_seconds": round(t_index, 1),
        "seconds": round(elapsed, 1),
        "images_per_s": round(n_done / elapsed, 2) if elapsed else None,
        "images_per_s_after_first_batch":
            round((n_done - first[0]) / (elapsed - first[1]), 2) if steady else None,
        "gpu_busy_frac": round(t_gpu / elapsed, 3) if elapsed else None,
        "gpu_images_per_s": round(n_done / t_gpu, 1) if t_gpu else None,
        # thread-seconds per image, summed over the --workers producer threads
        "zip_read_ms_per_image": round(1000 * t_read / n_done, 2) if n_done else None,
        "decode_preprocess_ms_per_image": round(1000 * t_prep / n_done, 2) if n_done else None,
        "gpu_util_mean_pct": round(float(np.mean(util)), 1) if util else None,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        if device.type == "cuda" else None,
        "cpu_count": os.cpu_count(), "versions": library_versions(), "smoke": args.smoke,
    }
    with open(args.out / "throughput.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")
    print("\n" + json.dumps(stats))
    return stats


def read_ids(path: Path) -> pd.Series:
    return pd.read_csv(path, usecols=["id"], dtype={"id": str}, keep_default_na=False)["id"]


def check(args: argparse.Namespace) -> dict:
    """Is every target id in the cache exactly once, and nothing else? With --reference: are
    the embeddings of the same ids the same (fp16 tolerance) in that other cache?"""
    index = build_index(make_opener(args.local_zips), args.source,
                        args.out.parent / f"index_{args.source}.json", args.workers)
    target = set(target_ids(read_ids(args.ids), index, args.limit, args.seed))
    files = cache_files(args.out, args.source)
    counts = collections.Counter(str(i) for f in files for i in np.load(f)["ids"])
    failed = {str(i) for f in files for i in np.load(f)["failed"]}
    res = {"out": str(args.out), "files": len(files), "expected": len(target),
           "unique": len(counts), "duplicates": sum(c - 1 for c in counts.values()),
           "missing": len(target - counts.keys() - failed), "failed": len(failed),
           "extra": len(counts.keys() - target),
           "temp_files_left": len(list(args.out.glob(".*.tmp")))}
    ok = not (res["duplicates"] or res["missing"] or res["failed"] or res["extra"])
    if args.reference:
        enc = [json.loads((d / "encoder.json").read_text()) for d in (args.out, args.reference)]
        ids = sorted(target)
        a, fa = load_embeddings(args.out, ids, args.source)
        b, fb = load_embeddings(args.reference, ids, args.source)
        both = fa & fb
        diff = np.abs(a[both] - b[both]).max() if both.any() else np.inf
        cos = (a[both] * b[both]).sum(1).min() if both.any() else -1.0
        res |= {"reference": str(args.reference), "same_encoder": enc[0] == enc[1],
                "compared": int(both.sum()), "max_abs_diff": float(diff),
                "min_cosine": float(cos)}
        ok = (ok and res["same_encoder"] and bool(both.all()) and diff <= args.atol
              and cos >= args.min_cos)
    res["ok"] = bool(ok)
    print(json.dumps(res, indent=2))
    return res


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", type=Path, required=True, help="CSV with an 'id' column")
    ap.add_argument("--source", choices=["train", "test"], required=True, help="which shard zips")
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out", type=Path, required=True, help="embedding cache dir (one per encoder)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8, help="shards read in parallel")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parts", type=int, default=1, help="split shards over N processes")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="target a seeded sample of N ids in total, over all --parts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny random model (no weights download)")
    ap.add_argument("--local-zips", type=Path, help="DIR/{train,test}/NN.zip (for testing)")
    ap.add_argument("--check", action="store_true",
                    help="verify the cache in --out instead of embedding (exit 1 on failure)")
    ap.add_argument("--reference", type=Path, help="--check: compare embeddings with this cache")
    ap.add_argument("--atol", type=float, default=1e-3, help="--check: max abs difference")
    ap.add_argument("--min-cos", type=float, default=0.999, help="--check: min cosine")
    args = ap.parse_args(argv)
    if not 0 <= args.part < args.parts:
        ap.error("--part must be in [0, --parts)")
    return args


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.check:
        sys.exit(0 if check(args)["ok"] else 1)
    run(args)


if __name__ == "__main__":
    main()
