"""Phase 1 run modes for scripts/kaggle_phase1.sh: settings, repo guard, progress, resume status.

  config MODE        shell assignments for the mode (eval'ed by the script)
  progress ...       one line whenever a consumer finishes a zip: zips done, img/s, ETA, disk
  status ...         after the pipeline: exit 0 if every consumer finished every zip; otherwise
                     print RESUME NEEDED with what is left and exit 3

Modes:
  pilot  3 train zips, 40 min deadline, uploads only to a *-pilot repo
  full   all 98 train zips, 11 h deadline, uploads only to FULL_REPO
  test   the 5 official test zips (embeddings only, no upload), 4 h deadline; refuses to start
         unless the pre-registration (prereg.py) is committed
"""

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

import pandas as pd

from geoguessr_ai.zipq import aborted, load_zip_index, marker, stopped

FULL_REPO = "Daggerabuse/osv5m-ft-archive"
PILOT_REPO = FULL_REPO + "-pilot"
N_TRAIN_ZIPS, N_TEST_ZIPS = 98, 5
MODES = {
    "pilot": {"shards": [0, 1, 2], "deadline_min": 40, "repo": PILOT_REPO, "out": "pilot"},
    "full": {"shards": list(range(N_TRAIN_ZIPS)), "deadline_min": 660, "repo": FULL_REPO,
             "out": "full"},
    "test": {"shards": list(range(N_TEST_ZIPS)), "deadline_min": 240, "repo": "", "out": "test",
             "source": "test"},
}


def check_repo(mode: str, repo: str) -> None:
    """The pilot never touches the main archive; the full run touches nothing but it; the
    test run uploads nothing."""
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode}; use one of {sorted(MODES)}")
    if mode == "pilot" and not repo.endswith("-pilot"):
        raise SystemExit(f"pilot mode only uploads to a *-pilot repo, not {repo}")
    if mode == "full" and repo != FULL_REPO:
        raise SystemExit(f"full mode only uploads to {FULL_REPO}, not {repo}")
    if mode == "test" and repo:
        raise SystemExit(f"test mode uploads nothing, not to {repo}")


def config(mode: str) -> str:
    check_repo(mode, MODES.get(mode, {}).get("repo", ""))
    m = MODES[mode]
    shell = {"MODE": mode, "SHARDS": " ".join(map(str, m["shards"])),
             "DEADLINE_MIN": os.environ.get("DEADLINE_MIN", m["deadline_min"]),
             "REPO": m["repo"], "OUT_NAME": m["out"], "SOURCE": m.get("source", "train")}
    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in shell.items())


def done_shards(zip_dir: Path, src: str, shards, consumer: str) -> list[int]:
    return [s for s in shards if marker(zip_dir, src, s, f"{consumer}.done").exists()]


def du_gb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:  # deleted while walking
                pass
    return total / 1e9


def progress_line(zip_dir: Path, index_dir: Path, src: str, shards, consumers, wanted: set[str],
                  t0: float, disks, now: float | None = None, counts: dict | None = None) -> str:
    """zips done, img/s and ETA per consumer (images = wanted ids in its finished zips; the
    archive consumer gets zips only), then disk use. `counts` caches wanted ids per zip."""
    now = time.time() if now is None else now
    counts = {} if counts is None else counts
    elapsed = max(now - t0, 1.0)
    parts = []
    for c in consumers:
        done = done_shards(zip_dir, src, shards, c)
        for s in done:
            if s not in counts:
                counts[s] = sum(i in wanted for i in load_zip_index(index_dir, src, [s]))
        eta = (len(shards) - len(done)) * elapsed / len(done) / 3600 if done else float("nan")
        rate = "" if c == "ft_archive" else f" {sum(counts[s] for s in done) / elapsed:.1f} img/s"
        parts.append(f"{c} {len(done)}/{len(shards)} zips{rate} ETA {eta:.1f} h")
    disk = " ".join(f"{name} {du_gb(Path(p)):.1f} GB" for name, p in disks)
    return f"[{time.strftime('%H:%M:%S', time.gmtime(now))} UTC, {elapsed / 60:.0f} min] " + \
        " | ".join(parts) + f" | {disk}"


def status(zip_dir: Path, src: str, shards, consumers) -> tuple[bool, str]:
    left = {c: [s for s in shards if s not in done_shards(zip_dir, src, shards, c)]
            for c in consumers}
    if not any(left.values()):
        return True, f"all {len(shards)} zips done by every consumer"
    lines = ["RESUME NEEDED" + (" (deadline STOP)" if stopped(zip_dir) else "")
             + (f" (ABORT: {aborted(zip_dir)})" if aborted(zip_dir) else "")]
    for c, s in left.items():
        lines.append(f"  {c}: {len(shards) - len(s)}/{len(shards)} zips done; left: {s}")
    lines.append("  Completed zips are saved (embeddings in the cache, archive parts in the repo);"
                 " rerun the same script with the cache restored and the rest is redone.")
    return False, "\n".join(lines)


def read_ids(paths) -> set[str]:
    return set().union(*(set(pd.read_csv(p, usecols=["id"], dtype={"id": str},
                                         keep_default_na=False)["id"]) for p in paths))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("mode")
    for p in (pr := sub.add_parser("progress"), sub.add_parser("status")):
        p.add_argument("--zip-dir", type=Path, required=True)
        p.add_argument("--source", default="train")
        p.add_argument("--shards", type=int, nargs="+", required=True)
        p.add_argument("--consumers", nargs="+", required=True)
    pr.add_argument("--index-dir", type=Path, required=True)
    pr.add_argument("--ids", type=Path, nargs="+", required=True)
    pr.add_argument("--disk", nargs="+", default=["/kaggle/temp", "/kaggle/working"])
    pr.add_argument("--poll", type=float, default=30)
    args = ap.parse_args(argv)
    if args.cmd == "config":
        print(config(args.mode))
    elif args.cmd == "status":
        ok, text = status(args.zip_dir, args.source, args.shards, args.consumers)
        print(text)
        sys.exit(0 if ok else 3)
    else:  # runs until killed by the script
        wanted, t0, counts, last = read_ids(args.ids), time.time(), {}, None
        disks = [(Path(d).name, d) for d in args.disk]
        while True:
            state = [len(done_shards(args.zip_dir, args.source, args.shards, c))
                     for c in args.consumers]
            if state != last:
                print(progress_line(args.zip_dir, args.index_dir, args.source, args.shards,
                                    args.consumers, wanted, t0, disks, counts=counts), flush=True)
                last = state
            time.sleep(args.poll)


if __name__ == "__main__":
    main()
