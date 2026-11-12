"""Sequential full-zip downloader that feeds long-lived consumers through marker files.

Phase 0b measured one hf_hub_download stream at ~400 MB/s and two parallel ones slower, so
zips are downloaded one at a time, in --shards order, at most --window zips ahead of the
slowest consumer. Layout under --zip-dir (e.g. /kaggle/temp/zips):
  images/<src>/NN.zip                 the downloaded zip (hf_hub_download local_dir layout)
  markers/<src>_NN.ready              zip complete and indexed; consumers may read it
  markers/<src>_NN.<consumer>.done    that consumer has durably saved everything it needs
  markers/<src>_NN.extra.json         ids in the zip that are not in --known-ids (logged)
  markers/ABORT                       someone gave up (the downloader, or the pilot script
                                      when a consumer failed): everybody stops (exit 1)
and under --index-dir (persistent, e.g. /kaggle/working/emb/zip_index):
  <src>_NN.json                       {id: name in zip}, written before the ready marker

A zip is deleted only when every --consumers marker exists. Every marker and index file is
written atomically (temp file + rename), so a killed process never leaves a partial one;
rerunning the downloader resumes (a complete zip without its ready marker is re-indexed, a
corrupt one downloaded again). Before each download --min-free-gb must be free next to
--zip-dir (the Kaggle overlay is shared with other machines); otherwise it waits up to
--wait-timeout, then aborts. Transient download errors are retried with backoff.
The Hugging Face token is read by huggingface_hub from HF_TOKEN; it is never printed here.

Usage (Kaggle):
  python -m geoguessr_ai.zipq --zip-dir /kaggle/temp/zips --index-dir /kaggle/working/emb/zip_index \\
      --shards 0 1 2 --consumers clip_l14_336 streetclip clip_b16 ft_archive \\
      --known-ids /kaggle/temp/raw/train.csv
"""

import argparse
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from pathlib import Path

import pandas as pd

REPO = "osv5m/osv5m"


# ----------------------------------------------------------------- files and markers


def atomic_write(path: Path, write) -> None:
    """write(file) into a temp file next to path (not matching *.npz/*.json), then rename."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda f: f.write(text.encode("utf-8")))


def zip_path(zip_dir: Path, src: str, shard: int) -> Path:
    return zip_dir / "images" / src / f"{shard:02d}.zip"


def marker(zip_dir: Path, src: str, shard: int, what: str) -> Path:
    return zip_dir / "markers" / f"{src}_{shard:02d}.{what}"


def index_path(index_dir: Path, src: str, shard: int) -> Path:
    return index_dir / f"{src}_{shard:02d}.json"


def load_zip_index(index_dir: Path, src: str, shards) -> dict[str, tuple[int, str]]:
    """id -> (shard, name in zip), as embed.py's index, for the given shards."""
    out: dict = {}
    for s in shards:
        out.update({i: (s, n) for i, n in json.loads(index_path(index_dir, src, s).read_text()).items()})
    return out


def consumed(zip_dir: Path, src: str, shard: int, consumers) -> bool:
    return all(marker(zip_dir, src, shard, f"{c}.done").exists() for c in consumers)


def abort(zip_dir: Path, reason: str) -> None:
    write_text(zip_dir / "markers" / "ABORT", reason)


def aborted(zip_dir: Path) -> str | None:
    p = zip_dir / "markers" / "ABORT"
    return p.read_text() if p.exists() else None


def stopped(zip_dir: Path) -> bool:
    """A clean stop was requested (markers/STOP, e.g. at a deadline): finish the zip in hand,
    start no new one. Unlike ABORT this is not a failure; completed zips stay valid."""
    return (zip_dir / "markers" / "STOP").exists()


def wait_for(cond, poll: float, timeout: float, what: str, zip_dir: Path | None = None) -> None:
    """Poll until cond() is true; RuntimeError on timeout or (if zip_dir) on ABORT."""
    t0 = time.time()
    while not cond():
        if zip_dir is not None and (why := aborted(zip_dir)):
            raise RuntimeError(f"downloader aborted while waiting for {what}: {why}")
        if time.time() - t0 > timeout:
            raise RuntimeError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(poll)


def wait_ready(zip_dir: Path, src: str, shard: int, poll: float, timeout: float) -> bool:
    """True once the zip is ready; False if a STOP is (or gets) requested first."""
    ready = marker(zip_dir, src, shard, "ready")
    wait_for(lambda: ready.exists() or stopped(zip_dir), poll, timeout,
             f"{src} zip {shard:02d}", zip_dir)
    return not stopped(zip_dir)


# ----------------------------------------------------------------- downloader


def hf_fetch(src: str, shard: int, zip_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(REPO, f"images/{src}/{shard:02d}.zip", repo_type="dataset", local_dir=zip_dir)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def index_zip(path: Path) -> dict[str, str]:
    """{id: name} from the central directory; BadZipFile if the file is incomplete."""
    with zipfile.ZipFile(path) as zf:
        return {Path(n).stem: n for n in zf.namelist() if not n.endswith("/")}


def download_all(shards, src: str, zip_dir: Path, index_dir: Path, consumers,
                 known_ids: set[str] | None = None, window: int = 2, min_free_gb: float = 20,
                 retries: int = 5, backoff: float = 10, poll: float = 5,
                 wait_timeout: float = 7200, fetch=hf_fetch, free=free_gb, log=print) -> dict:
    """Download, index and mark each shard in order; delete consumed zips; return stats."""
    (zip_dir / "markers").mkdir(parents=True, exist_ok=True)
    for m in ("ABORT", "STOP"):  # a rerun starts clean
        (zip_dir / "markers" / m).unlink(missing_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"downloads": [], "extra_ids": 0, "deleted": [], "stopped": False}

    def cleanup() -> None:
        for s in shards:
            if zip_path(zip_dir, src, s).exists() and consumed(zip_dir, src, s, consumers):
                zip_path(zip_dir, src, s).unlink()
                stats["deleted"].append(s)
                log(f"{src} zip {s:02d}: all consumers done, deleted")

    def on_disk(earlier) -> int:
        cleanup()
        return sum(zip_path(zip_dir, src, s).exists() for s in earlier)

    try:
        for k, shard in enumerate(shards):
            if consumed(zip_dir, src, shard, consumers):
                continue  # resume: finished in an earlier run
            if marker(zip_dir, src, shard, "ready").exists() and zip_path(zip_dir, src, shard).exists():
                continue
            wait_for(lambda k=k: on_disk(shards[:k]) <= window or stopped(zip_dir), poll,
                     wait_timeout, f"consumers to finish zips before {shard:02d}", zip_dir)
            wait_for(lambda: free(zip_dir) >= min_free_gb or stopped(zip_dir), poll,
                     wait_timeout, f"{min_free_gb} GB free disk", zip_dir)
            if stopped(zip_dir):
                log(f"STOP requested: no download from {src} zip {shard:02d} on")
                break
            index = fetch_and_index(src, shard, zip_dir, retries, backoff, fetch, log, stats)
            write_text(index_path(index_dir, src, shard), json.dumps(index))
            if known_ids is not None:
                extra = sorted(set(index) - known_ids)
                stats["extra_ids"] += len(extra)
                write_text(marker(zip_dir, src, shard, "extra.json"), json.dumps(extra))
                if extra:
                    log(f"{src} zip {shard:02d}: {len(extra)} ids not in the metadata CSV, "
                        f"skipped by every consumer: {extra[:5]}")
            write_text(marker(zip_dir, src, shard, "ready"), "")
        wait_for(lambda: on_disk(shards) == 0 or stopped(zip_dir), poll, wait_timeout,
                 "consumers to finish", zip_dir)
        stats["stopped"] = stopped(zip_dir)
    except BaseException as e:
        if not aborted(zip_dir):  # keep the first reason (e.g. a consumer's)
            abort(zip_dir, f"{type(e).__name__}: {e}")
        raise
    return stats


def fetch_and_index(src, shard, zip_dir, retries, backoff, fetch, log, stats) -> dict:
    path = zip_path(zip_dir, src, shard)
    for attempt in range(retries + 1):
        try:
            if not path.exists():
                t = time.time()
                fetch(src, shard, zip_dir)
                s = max(time.time() - t, 1e-6)
                gb = path.stat().st_size / 1e9
                stats["downloads"].append({"shard": shard, "gb": round(gb, 3),
                                           "seconds": round(s, 1), "mb_per_s": round(gb * 1e3 / s, 1)})
                log(f"{src} zip {shard:02d}: {gb:.2f} GB in {s:.1f} s")
            return index_zip(path)
        except Exception as e:  # network error or truncated file: retry, then give up
            path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            wait = backoff * 2**attempt
            log(f"{src} zip {shard:02d}: attempt {attempt + 1} failed ({type(e).__name__}: "
                f"{str(e)[:200]}); retrying in {wait:.0f} s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip-dir", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--source", default="train", choices=["train", "test"])
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--consumers", nargs="+", required=True, help="marker names to wait for")
    ap.add_argument("--known-ids", type=Path, help="metadata CSV; zip ids not in it are logged")
    ap.add_argument("--window", type=int, default=2, help="zips downloaded ahead of consumers")
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--wait-timeout", type=float, default=7200)
    ap.add_argument("--stats-out", type=Path)
    args = ap.parse_args()
    known = None
    if args.known_ids:
        known = set(pd.read_csv(args.known_ids, usecols=["id"], dtype=str,
                                keep_default_na=False)["id"])
    stats = download_all(args.shards, args.source, args.zip_dir, args.index_dir, args.consumers,
                         known, args.window, args.min_free_gb, args.retries,
                         wait_timeout=args.wait_timeout)
    print(json.dumps(stats))
    if args.stats_out:
        write_text(args.stats_out, json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
