"""Archive the raw JPEG bytes of the fine-tuning images (Phase 2) during the embedding pass.

Images: nat_rank < --nat-max of train.csv, plus dev.csv and dev_seen.csv. They are taken
byte for byte from the zips that zipq.py downloads (no re-encoding) and end up in ~1 GB
ZIP_STORED parts with a sha256 manifest, uploaded to a PRIVATE Hugging Face dataset repo.

  consume  a zipq consumer: for each --shards zip, copy the wanted images into a staging
           chunk (chunk_<src>_NN.zip + .csv with id, bytes, sha256 of the source bytes),
           write its done marker (the zip may then be deleted), then pack.
  pack     move whole staging chunks into part_NNN.zip once they add up to --part-gb
           (--final: also the rest). Every entry's sha256 is checked while copying and again
           by re-reading the finished part; part_NNN.csv is the commit record (a part without
           it is incomplete and redone; chunks already in a committed part are dropped).
           --final also writes manifest.csv (all parts). Parts are byte-deterministic;
           part_NNN.sha256 holds the sha256 of the whole part file.
  upload   (consume --repo does this part by part, as soon as each is committed) create the
           repo private and refuse it if it is not; upload each part still on disk unless
           the repo has the same sha256; check the repo's sha256; delete the local zip;
           manifest.csv last. Only network errors, 429 and 5xx are retried; 401/403 and other
           errors fail at once. Token only from HF_TOKEN, redacted from all output.
  smoke    a dummy 2-part archive uploaded to <repo>/smoke/ (*-pilot repos only), then
           re-downloaded and verified (file sha256 and every image's sha256).

Usage (Kaggle):
  python -m geoguessr_ai.ft_archive consume --ids v2/train.csv v2/dev.csv v2/dev_seen.csv \\
      --zip-dir /kaggle/temp/zips --index-dir /kaggle/working/emb/zip_index --shards 0 1 2 \\
      --staging /kaggle/temp/ft_staging --out /kaggle/temp/ft_archive \\
      --repo <user>/osv5m-ft-archive-pilot
  python -m geoguessr_ai.ft_archive pack --staging ... --out ... --final
  python -m geoguessr_ai.ft_archive upload --out /kaggle/temp/ft_archive --repo <user>/...
Usage (local smoke test; HF_TOKEN with write access in the environment):
  uv run python -m geoguessr_ai.ft_archive smoke --repo <user>/osv5m-ft-archive-pilot
"""

import argparse
import hashlib
import io
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from geoguessr_ai.prep_data import make_opener
from geoguessr_ai.zipq import (
    atomic_write,
    load_zip_index,
    marker,
    wait_ready,
    write_text,
)

try:  # the hub client's network errors (httpx2 in huggingface_hub 2.x)
    from httpx2 import TransportError as _TransportError
except ImportError:  # pragma: no cover
    _TransportError = ConnectionError
NETWORK_ERRORS = (ConnectionError, TimeoutError, _TransportError)

MANIFEST_COLS = ["id", "part", "bytes", "sha256", "shard", "chunk"]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def wanted_ids(paths, nat_max: int) -> set[str]:
    """ids of the CSVs; rows of a CSV with nat_rank only if nat_rank < nat_max."""
    out: set[str] = set()
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in ("id", "nat_rank"), dtype={"id": str},
                         keep_default_na=False)  # only these columns: no mixed-type warning
        if "nat_rank" in df:
            df = df[df["nat_rank"] < nat_max]
        out |= set(df["id"])
    return out


def write_zip(path: Path, entries) -> None:
    """entries: [(name, bytes)] -> ZIP_STORED zip, written atomically."""
    def write(f):
        with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, data in entries:  # fixed timestamp: same entries -> same bytes
                zf.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)
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
    hub = None
    if args.repo:
        hub, _ = make_hub(args.repo, "", args.retries, args.backoff, args.log)
        hub.open()  # fails before any work if the token or the repo is wrong
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
        if hub:
            upload_parts(hub, args.out)
    pack(args.staging, args.out, args.part_gb, final=True)
    if hub:
        upload_parts(hub, args.out)
        finish(hub, args.out)


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
    write_text(part.with_suffix(".sha256"), file_sha256(part) + "\n")
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 23):
            h.update(b)
    return h.hexdigest()


def redact(text: str, secrets) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def transient(e: Exception) -> bool:
    """Worth retrying: network errors, 429, 5xx. Anything else (401, 403, 404, ...) fails fast."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code is not None:
        return code == 429 or code >= 500
    return isinstance(e, NETWORK_ERRORS)


class Hub:
    """A PRIVATE Hugging Face dataset repo; files go under `prefix`."""

    def __init__(self, api, repo: str, prefix: str = "", retries: int = 5, backoff: float = 30,
                 log=print):
        self.api, self.repo, self.prefix = api, repo, prefix
        self.retries, self.backoff, self.log = retries, backoff, log

    def call(self, what: str, fn):
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except Exception as e:
                if not transient(e) or attempt == self.retries:
                    raise
                wait = self.backoff * 2**attempt
                self.log(f"{what}: {type(e).__name__} (transient); retry in {wait:.0f} s")
                time.sleep(wait)
        raise AssertionError("unreachable")

    def open(self) -> None:
        """Create the repo private (a no-op if it exists); refuse a repo that is not private."""
        self.call("create_repo", lambda: self.api.create_repo(
            self.repo, repo_type="dataset", private=True, exist_ok=True))
        private = self.call("repo_info", lambda: self.api.repo_info(
            self.repo, repo_type="dataset")).private
        self.log(f"{self.repo}: private={private}")
        if private is not True:
            raise SystemExit(f"{self.repo} is not private: refusing to upload")

    def remote_sha256(self, name: str) -> str | None:
        """sha256 of the file in the repo (its LFS oid), None if it is not there."""
        path = self.prefix + name
        info = self.call(f"paths_info {path}", lambda: self.api.get_paths_info(
            self.repo, [path], repo_type="dataset"))
        if not info:
            return None
        if getattr(info[0], "lfs", None) is None:  # never re-upload or trust what we cannot verify
            raise SystemExit(f"{path} is in the repo but has no LFS sha256")
        return info[0].lfs.sha256

    def put(self, local: Path, name: str) -> None:
        path = self.prefix + name
        self.call(f"upload {path}", lambda: self.api.upload_file(
            path_or_fileobj=str(local), path_in_repo=path, repo_id=self.repo,
            repo_type="dataset", commit_message=f"add {path}"))
        self.log(f"uploaded {path} ({local.stat().st_size / 1e9:.3f} GB)")


def upload_parts(hub: Hub, out: Path) -> None:
    """Upload every committed part still on disk (skipped if the repo already has the same
    sha256), check the repo's sha256, then delete the local zip. Its .csv/.sha256 stay."""
    for csv in sorted(out.glob("part_*.csv")):
        part, sha = csv.with_suffix(".zip"), csv.with_suffix(".sha256")
        if not part.exists():  # uploaded and deleted earlier; finish() checks the repo
            continue
        want = sha.read_text().strip()
        if hub.remote_sha256(part.name) == want:
            hub.log(f"{part.name}: already in {hub.repo} with the same sha256, not uploaded")
        else:
            hub.put(part, part.name)
            hub.put(sha, sha.name)
            if hub.remote_sha256(part.name) != want:
                raise SystemExit(f"{part.name}: sha256 in the repo differs from the local part")
        part.unlink()


def finish(hub: Hub, out: Path) -> None:
    """Every committed part is in the repo with its sha256; then manifest.csv, last."""
    for csv in sorted(out.glob("part_*.csv")):
        name = csv.with_suffix(".zip").name
        if hub.remote_sha256(name) != csv.with_suffix(".sha256").read_text().strip():
            raise SystemExit(f"{name}: missing or different in the repo")
    hub.put(out / "manifest.csv", "manifest.csv")


def make_hub(repo: str, prefix: str, retries: int, backoff: float, log_path: Path | None):
    """(Hub, token). The token comes only from HF_TOKEN and is redacted from every log line."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be set (write access to the dataset repo)")

    def log(msg: str) -> None:
        msg = redact(msg, [token])
        print(msg, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    return Hub(HfApi(token=token), repo, prefix, retries, backoff, log), token


def smoke(repo: str, work: Path, log_path: Path | None) -> None:
    """A dummy 2-part archive through the real upload path, then re-downloaded and checked."""
    if not repo.endswith("-pilot"):
        raise SystemExit("smoke only uploads to a *-pilot repo")
    hub, token = make_hub(repo, "smoke/", retries=3, backoff=5, log_path=log_path)
    staging, out, dl = work / "staging", work / "out", work / "download"
    shutil.rmtree(work, ignore_errors=True)
    staging.mkdir(parents=True)
    rng = np.random.default_rng(0)  # seeded: a rerun builds the same parts and skips them
    for shard in (0, 1):
        rows, entries = [], []
        for k in range(3):
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(buf, "JPEG")
            data, id_ = buf.getvalue(), f"smoke{shard}{k}"
            entries.append((f"{id_}.jpg", data))
            rows.append({"id": id_, "bytes": len(data), "sha256": sha256(data), "shard": shard})
        write_zip(staging / f"chunk_smoke_{shard:02d}.zip", entries)
        write_text(staging / f"chunk_smoke_{shard:02d}.csv", pd.DataFrame(rows).to_csv(index=False))
    pack(staging, out, part_gb=1e-9, final=True)  # one part per chunk
    hub.open()
    upload_parts(hub, out)
    finish(hub, out)
    manifest = pd.read_csv(out / "manifest.csv", dtype={"id": str, "sha256": str})
    for csv in sorted(out.glob("part_*.csv")):
        name = csv.with_suffix(".zip").name
        got = Path(hf_hub_download(repo, f"smoke/{name}", repo_type="dataset", token=token,
                                   local_dir=dl))
        if file_sha256(got) != csv.with_suffix(".sha256").read_text().strip():
            raise SystemExit(f"{name}: downloaded sha256 differs")
        verify_part(got, manifest[manifest["part"] == name])
        hub.log(f"{name}: re-downloaded; file sha256 and every image's sha256 match")
    got = Path(hf_hub_download(repo, "smoke/manifest.csv", repo_type="dataset", token=token,
                               local_dir=dl))
    if got.read_bytes() != (out / "manifest.csv").read_bytes():
        raise SystemExit("manifest.csv differs in the repo")
    private = hub.api.repo_info(repo, repo_type="dataset").private
    hub.log(f"SMOKE OK: {len(manifest)} images in 2 parts; {repo} private={private}")


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
    u, sm = sub.add_parser("upload"), sub.add_parser("smoke")
    for p in (c, u, sm):
        p.add_argument("--repo", required=p is not c,
                       help="private HF dataset repo user/name (consume: upload parts as they are made)")
        p.add_argument("--log", type=Path)
    for p in (c, u):
        p.add_argument("--retries", type=int, default=5)
        p.add_argument("--backoff", type=float, default=30)
    u.add_argument("--out", type=Path, required=True)
    sm.add_argument("--work", type=Path, default=Path("data/ft_smoke"))
    args = ap.parse_args(argv)
    try:
        if args.cmd == "consume":
            consume(args)
        elif args.cmd == "pack":
            pack(args.staging, args.out, args.part_gb, args.final)
        elif args.cmd == "upload":
            hub, _ = make_hub(args.repo, "", args.retries, args.backoff, args.log)
            hub.open()
            upload_parts(hub, args.out)
            finish(hub, args.out)
        else:
            smoke(args.repo, args.work, args.log)
    except Exception as e:  # noqa: BLE001 - no traceback: report it with the token redacted
        sys.exit(redact(f"{args.cmd} failed: {type(e).__name__}: {e}", [os.environ.get("HF_TOKEN")]))

if __name__ == "__main__":
    main()
