"""Archive the raw JPEG bytes of the fine-tuning images (Phase 2) during the embedding pass.

Images: nat_rank < --nat-max of train.csv, plus dev.csv and dev_seen.csv. They are taken
byte for byte from the zips that zipq.py downloads (no re-encoding) and end up in ~1 GB
ZIP_STORED parts with a sha256 manifest, uploaded to a PRIVATE Kaggle Dataset.

  consume  a zipq consumer: for each --shards zip, copy the wanted images into a staging
           chunk (chunk_<src>_NN.zip + .csv with id, bytes, sha256 of the source bytes),
           write its done marker (the zip may then be deleted), then pack.
  pack     move whole staging chunks into part_NNN.zip once they add up to --part-gb
           (--final: also the rest). Every entry's sha256 is checked while copying and again
           by re-reading the finished part; part_NNN.csv is the commit record (a part without
           it is incomplete and redone; chunks already in a committed part are dropped).
           --final also writes manifest.csv (all parts).
  upload   kaggle datasets version; if that fails with "not found", create (private);
           other failures are retried. --dataset is the slug, always explicit. Credentials
           only from KAGGLE_USERNAME / KAGGLE_KEY in the environment (no kaggle.json is
           written); the key is redacted from everything this module prints or logs.

Usage (Kaggle):
  python -m geoguessr_ai.ft_archive consume --ids v2/train.csv v2/dev.csv v2/dev_seen.csv \\
      --zip-dir /kaggle/temp/zips --index-dir /kaggle/working/emb/zip_index --shards 0 1 2 \\
      --staging /kaggle/temp/ft_staging --out /kaggle/temp/ft_archive
  python -m geoguessr_ai.ft_archive pack --staging ... --out ... --final
  python -m geoguessr_ai.ft_archive upload --out /kaggle/temp/ft_archive --dataset osv5m-ft-pilot
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

from geoguessr_ai.prep_data import make_opener
from geoguessr_ai.zipq import (
    atomic_write,
    load_zip_index,
    marker,
    wait_ready,
    write_text,
)

MANIFEST_COLS = ["id", "part", "bytes", "sha256", "shard", "chunk"]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wanted_ids(paths, nat_max: int) -> set[str]:
    """ids of the CSVs; rows of a CSV with nat_rank only if nat_rank < nat_max."""
    out: set[str] = set()
    for p in paths:
        df = pd.read_csv(p, dtype={"id": str}, keep_default_na=False)
        if "nat_rank" in df:
            df = df[df["nat_rank"] < nat_max]
        out |= set(df["id"])
    return out


def write_zip(path: Path, entries) -> None:
    """entries: [(name, bytes)] -> ZIP_STORED zip, written atomically."""
    def write(f):
        with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, data in entries:
                zf.writestr(name, data)
    atomic_write(path, write)


# ----------------------------------------------------------------- consume


def stage_shard(opener, src: str, shard: int, items, staging: Path) -> int:
    """Copy [(id, name)] from the zip into chunk_<src>_NN.zip; the .csv (written last) marks
    the chunk complete. Returns the number of images."""
    rows, entries = [], []
    with opener(src, shard) as f, zipfile.ZipFile(f) as zf:
        for id_, name in items:
            data = zf.read(name)
            entries.append((f"{id_}.jpg", data))
            rows.append({"id": id_, "bytes": len(data), "sha256": sha256(data), "shard": shard})
    stem = f"chunk_{src}_{shard:02d}"
    write_zip(staging / f"{stem}.zip", entries)
    csv = pd.DataFrame(rows, columns=["id", "bytes", "sha256", "shard"]).to_csv(index=False)
    write_text(staging / f"{stem}.csv", csv)
    return len(rows)


def consume(args) -> None:
    want = wanted_ids(args.ids, args.nat_max)
    opener = make_opener(args.zip_dir / "images")
    args.staging.mkdir(parents=True, exist_ok=True)
    for shard in args.shards:
        done = marker(args.zip_dir, args.source, shard, f"{args.consumer}.done")
        if not done.exists():
            wait_ready(args.zip_dir, args.source, shard, args.poll, args.wait_timeout)
            index = load_zip_index(args.index_dir, args.source, [shard])
            items = sorted((i, n) for i, (_, n) in index.items() if i in want)
            n = stage_shard(opener, args.source, shard, items, args.staging)
            write_text(done, "")
            print(f"{args.source} zip {shard:02d}: {n} images staged")
        pack(args.staging, args.out, args.part_gb, final=False)
    pack(args.staging, args.out, args.part_gb, final=True)


# ----------------------------------------------------------------- pack


def committed(out: Path) -> pd.DataFrame:
    parts = sorted(out.glob("part_*.csv"))
    return (pd.concat([pd.read_csv(p, dtype={"id": str, "sha256": str}) for p in parts])
            if parts else pd.DataFrame(columns=MANIFEST_COLS))


def pack(staging: Path, out: Path, part_gb: float, final: bool) -> list[Path]:
    """Move complete staging chunks into parts; returns the parts written now."""
    out.mkdir(parents=True, exist_ok=True)
    done = committed(out)
    for z in out.glob("part_*.zip"):  # a part without its csv was cut off: redo it
        if not z.with_suffix(".csv").exists():
            z.unlink()
    chunks = []
    for csv in sorted(staging.glob("chunk_*.csv")):
        if csv.stem in set(done["chunk"]):  # already in a committed part: drop the copy
            csv.with_suffix(".zip").unlink(missing_ok=True)
            csv.unlink()
        else:
            chunks.append(csv)
    written, group, size = [], [], 0
    for csv in chunks:
        group.append(csv)
        size += int(pd.read_csv(csv)["bytes"].sum())
        if size >= part_gb * 1e9:
            written.append(write_part(group, out))
            group, size = [], 0
    if final and group:
        written.append(write_part(group, out))
    if final:
        m = committed(out)
        if m["id"].duplicated().any():
            raise AssertionError("an image is in two parts")
        write_text(out / "manifest.csv", m[MANIFEST_COLS].to_csv(index=False))
    return written


def write_part(chunk_csvs: list[Path], out: Path) -> Path:
    num = len(list(out.glob("part_*.csv")))
    part = out / f"part_{num:03d}.zip"
    rows, entries = [], []
    for csv in chunk_csvs:
        meta = pd.read_csv(csv, dtype={"id": str, "sha256": str})
        with zipfile.ZipFile(csv.with_suffix(".zip")) as zf:
            for r in meta.itertuples():
                data = zf.read(f"{r.id}.jpg")
                if sha256(data) != r.sha256:
                    raise AssertionError(f"{r.id}: staged bytes differ from the source zip")
                entries.append((f"{r.id}.jpg", data))
                rows.append({"id": r.id, "part": part.name, "bytes": r.bytes, "sha256": r.sha256,
                             "shard": r.shard, "chunk": csv.stem})
    write_zip(part, entries)
    m = pd.DataFrame(rows, columns=MANIFEST_COLS)
    verify_part(part, m)
    write_text(part.with_suffix(".csv"), m.to_csv(index=False))  # commit
    for csv in chunk_csvs:
        csv.with_suffix(".zip").unlink()
        csv.unlink()
    print(f"{part.name}: {len(m)} images, {m['bytes'].sum() / 1e9:.2f} GB, sha256 verified")
    return part


def verify_part(part: Path, manifest: pd.DataFrame) -> None:
    with zipfile.ZipFile(part) as zf:
        names = {i.filename: i for i in zf.infolist()}
        if set(names) != {f"{i}.jpg" for i in manifest["id"]}:
            raise AssertionError(f"{part.name}: entries differ from the manifest")
        for r in manifest.itertuples():
            if names[f"{r.id}.jpg"].compress_type != zipfile.ZIP_STORED:
                raise AssertionError(f"{part.name}: {r.id} is compressed")
            if sha256(zf.read(f"{r.id}.jpg")) != r.sha256:
                raise AssertionError(f"{part.name}: {r.id} sha256 mismatch")


# ----------------------------------------------------------------- upload


def redact(text: str, secrets) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def not_found(output: str) -> bool:
    """Does a failed `kaggle datasets version` say the dataset does not exist yet?"""
    o = output.lower()
    return "404" in o or "not found" in o or "does not exist" in o


def upload(out: Path, slug: str, title: str, kaggle_cmd: list[str], retries: int,
           backoff: float, log_path: Path | None = None) -> bool:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        raise SystemExit("KAGGLE_USERNAME and KAGGLE_KEY must be set (Kaggle Secrets)")
    secrets = [key, os.environ.get("HF_TOKEN")]

    def log(msg: str) -> None:
        msg = redact(msg, secrets)
        print(msg)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def kaggle(*a: str) -> subprocess.CompletedProcess:
        r = subprocess.run([*kaggle_cmd, *a], capture_output=True, text=True, check=False)
        log(f"$ kaggle {' '.join(a)} -> exit {r.returncode}\n{r.stdout}{r.stderr}".rstrip())
        return r

    if not (out / "manifest.csv").exists():
        raise SystemExit(f"{out}/manifest.csv missing: run pack --final first")
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "CC-BY-SA-4.0"}]}
    write_text(out / "dataset-metadata.json", json.dumps(meta, indent=2))  # no key in it
    for attempt in range(retries + 1):
        r = kaggle("datasets", "version", "-p", str(out), "-m", "ft archive", "--dir-mode", "skip")
        if r.returncode != 0 and not_found(r.stdout + r.stderr):
            # first upload: new datasets are private unless --public is passed (never here)
            r = kaggle("datasets", "create", "-p", str(out), "--dir-mode", "skip")
        if r.returncode == 0:
            log(f"uploaded {out} to {user}/{slug} (private)")
            return True
        if attempt < retries:
            log(f"upload attempt {attempt + 1} failed; retrying in {backoff * 2**attempt:.0f} s")
            time.sleep(backoff * 2**attempt)
    return False


# ----------------------------------------------------------------- main


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("consume")
    c.add_argument("--ids", type=Path, nargs="+", required=True)
    c.add_argument("--nat-max", type=int, default=200_000)
    c.add_argument("--zip-dir", type=Path, required=True)
    c.add_argument("--index-dir", type=Path, required=True)
    c.add_argument("--source", default="train")
    c.add_argument("--shards", type=int, nargs="+", required=True)
    c.add_argument("--consumer", default="ft_archive")
    c.add_argument("--poll", type=float, default=5)
    c.add_argument("--wait-timeout", type=float, default=7200)
    for p in (c, pk := sub.add_parser("pack")):
        p.add_argument("--staging", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--part-gb", type=float, default=1.0)
    pk.add_argument("--final", action="store_true")
    u = sub.add_parser("upload")
    u.add_argument("--out", type=Path, required=True)
    u.add_argument("--dataset", required=True, help="dataset slug (owner = KAGGLE_USERNAME)")
    u.add_argument("--title", default="OSV-5M fine-tuning subset (raw JPEG bytes)")
    u.add_argument("--retries", type=int, default=3)
    u.add_argument("--backoff", type=float, default=30)
    u.add_argument("--log", type=Path)
    u.add_argument("--kaggle-cmd", nargs="+", default=["kaggle"])
    args = ap.parse_args(argv)
    if args.cmd == "consume":
        consume(args)
    elif args.cmd == "pack":
        pack(args.staging, args.out, args.part_gb, args.final)
    elif not upload(args.out, args.dataset, args.title, args.kaggle_cmd, args.retries,
                    args.backoff, args.log):
        sys.exit("upload failed")


if __name__ == "__main__":
    main()
