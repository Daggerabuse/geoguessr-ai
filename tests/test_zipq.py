"""Local-zip pipeline: zipq downloader + markers, embed.py consumer mode, FT image archive."""

import hashlib
import io
import json
import shutil
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx2
import numpy as np
import pandas as pd
import pytest
from huggingface_hub.errors import HfHubHTTPError

from geoguessr_ai import embed as e
from geoguessr_ai import ft_archive as fa
from geoguessr_ai import zipq as z
from tests.test_prep_data import _make_zips

N_SHARDS = 3
FAST = {"poll": 0.01, "backoff": 0, "wait_timeout": 30}


@pytest.fixture
def hub(tmp_path):
    """A fake hub: 3 train zips of 8 ids each (z0..z23); fetch() copies one into place."""
    ids = [f"z{i}" for i in range(8 * N_SHARDS)]
    _make_zips(tmp_path / "hub", {"train": ids}, {"train": N_SHARDS})
    calls = []

    def fetch(src, shard, zip_dir):
        calls.append(shard)
        dst = z.zip_path(zip_dir, src, shard)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "hub" / src / f"{shard:02d}.zip", dst)

    return tmp_path, ids, fetch, calls


def _downloader(root, fetch, consumers, **kw):
    """Run download_all in a thread; returns (thread, box with 'stats' or 'err', log lines)."""
    box, log = {}, []

    def target():
        try:
            box["stats"] = z.download_all(list(range(N_SHARDS)), "train", root / "zips",
                                          root / "index", consumers, fetch=fetch, log=log.append,
                                          **{**FAST, **kw})
        except Exception as ex:  # noqa: BLE001 - returned to the test
            box["err"] = ex

    t = threading.Thread(target=target)
    t.start()
    return t, box, log


def _consumer(root, name, seen=None, skip=()):
    """A fake consumer: for each zip, record how many zips are on disk, then mark it done."""
    for s in range(N_SHARDS):
        z.wait_ready(root / "zips", "train", s, 0.01, 30)
        if seen is not None:
            seen.append(sum(z.zip_path(root / "zips", "train", k).exists() for k in range(N_SHARDS)))
        time.sleep(0.05)
        if s not in skip:
            z.write_text(z.marker(root / "zips", "train", s, f"{name}.done"), "")


# ----------------------------------------------------------------- downloader


def test_downloader_window_deletion_index_and_extra_ids(hub):
    root, ids, fetch, calls = hub
    seen = []
    t, box, log = _downloader(root, fetch, ["a", "b"], window=1,
                              known_ids=set(ids) - {"z5"})  # z5: in a zip, not in the CSV
    workers = [threading.Thread(target=_consumer, args=(root, "a", seen)),
               threading.Thread(target=_consumer, args=(root, "b"))]
    for w in workers:
        w.start()
    for w in [*workers, t]:
        w.join(60)
    assert "err" not in box and calls == [0, 1, 2]  # sequential, each zip once
    assert max(seen) <= 2  # window 1: the zip in use + at most 1 ahead
    assert not list((root / "zips" / "images").rglob("*.zip"))  # all consumed, all deleted
    idx = z.load_zip_index(root / "index", "train", range(N_SHARDS))
    assert set(idx) == set(ids) and idx["z5"][0] == 5 % N_SHARDS
    assert box["stats"]["extra_ids"] == 1
    extra = json.loads(z.marker(root / "zips", "train", 5 % N_SHARDS, "extra.json").read_text())
    assert extra == ["z5"] and any("not in the metadata CSV" in m and "z5" in m for m in log)
    assert not list((root / "zips").rglob("*.tmp"))


def test_zip_is_kept_until_every_consumer_is_done_then_abort(hub):
    root, _, fetch, _ = hub
    t, box, _ = _downloader(root, fetch, ["a", "b"], window=5, wait_timeout=1)
    _consumer(root, "a")
    _consumer(root, "b", skip={1})  # b never finishes zip 1
    t.join(60)
    zips = root / "zips"
    assert not z.zip_path(zips, "train", 0).exists() and z.zip_path(zips, "train", 1).exists()
    assert "timed out" in str(box["err"]) and z.aborted(zips)
    with pytest.raises(RuntimeError, match="aborted"):  # consumers waiting on a zip stop too
        z.wait_for(lambda: False, 0.01, 30, "x", zips)


def test_downloader_stops_when_someone_else_aborts(hub):
    root, _, fetch, _ = hub
    t, box, _ = _downloader(root, fetch, ["a"], window=0)  # waits for consumer "a" after zip 0
    z.wait_ready(root / "zips", "train", 0, 0.01, 30)
    z.abort(root / "zips", "consumer a crashed")
    t.join(60)
    assert not t.is_alive() and "aborted" in str(box["err"])
    assert z.aborted(root / "zips") == "consumer a crashed"  # first reason kept


def test_transient_errors_are_retried_and_truncated_zips_redownloaded(hub, tmp_path):
    root, _, fetch, _ = hub
    fails = {"n": 0}

    def flaky(src, shard, zip_dir):
        if shard == 1 and fails["n"] < 2:
            fails["n"] += 1
            if fails["n"] == 1:
                raise ConnectionError("reset by peer")
            dst = z.zip_path(zip_dir, src, shard)  # second failure: a truncated file
            dst.write_bytes(b"PK\x03\x04 not a whole zip")
            return
        fetch(src, shard, zip_dir)

    t, box, log = _downloader(root, flaky, ["a"])
    _consumer(root, "a")
    t.join(60)
    assert "err" not in box and fails["n"] == 2
    assert sum("retrying" in m for m in log) == 2
    assert set(z.load_zip_index(root / "index", "train", [1])) == {f"z{i}" for i in range(1, 24, 3)}


def test_low_disk_waits_then_aborts_without_downloading(hub):
    root, _, fetch, calls = hub
    t, box, _ = _downloader(root, fetch, ["a"], free=lambda p: 5.0, min_free_gb=20, wait_timeout=0.3)
    t.join(60)
    assert calls == [] and "20 GB free" in str(box["err"]) and z.aborted(root / "zips")


def test_downloader_resumes_and_skips_consumed_zips(hub):
    root, _, fetch, calls = hub
    zips = root / "zips"
    z.write_text(z.marker(zips, "train", 0, "a.done"), "")  # zip 0 finished in a past session
    fetch("train", 1, zips)  # zip 1 downloaded, killed before indexing
    calls.clear()
    t, box, _ = _downloader(root, fetch, ["a"])
    for s in (1, 2):
        z.wait_ready(zips, "train", s, 0.01, 30)
        z.write_text(z.marker(zips, "train", s, "a.done"), "")
    t.join(60)
    assert "err" not in box and calls == [2]


# ----------------------------------------------------------------- embed.py consumer mode


def _embed_zips(root: Path, out: Path, ids_csv: Path, *extra):
    return e.run(e.parse_args(["--ids", str(ids_csv), "--source", "train", "--out", str(out),
                               "--smoke", "--device", "cpu", "--workers", "3", "--batch-size", "4",
                               "--zip-dir", str(root / "zips"), "--index-dir", str(root / "index"),
                               "--shards", *map(str, range(N_SHARDS)), "--poll", "0.01",
                               "--wait-timeout", "30", *extra]))


def _check_zips(root: Path, out: Path, ids_csv: Path):
    return e.check(e.parse_args(["--check", "--ids", str(ids_csv), "--source", "train",
                                 "--out", str(out), "--zip-dir", str(root / "zips"),
                                 "--index-dir", str(root / "index"),
                                 "--shards", *map(str, range(N_SHARDS))]))


def test_embed_zip_mode_resumes_after_a_crash_mid_zip(hub, monkeypatch):
    pytest.importorskip("transformers")
    root, ids, fetch, _ = hub
    wanted = [i for i in ids if i not in ("z4", "z5")]  # z4/z5: in the zips, not wanted
    pd.DataFrame({"id": wanted[:10]}).to_csv(root / "a.csv", index=False)
    pd.DataFrame({"id": wanted[10:]}).to_csv(root / "b.csv", index=False)
    out = root / "emb" / "m"
    t, box, _ = _downloader(root, fetch, ["m"])

    real = e.embed_jobs

    def crash_on_zip_1(jobs, *a, **k):
        if jobs[0][0] == 1:
            raise RuntimeError("killed mid-zip (simulated)")
        return real(jobs, *a, **k)

    monkeypatch.setattr(e, "embed_jobs", crash_on_zip_1)
    with pytest.raises(RuntimeError, match="simulated"):
        e.run(e.parse_args(["--ids", str(root / "a.csv"), str(root / "b.csv"), "--source", "train",
                            "--out", str(out), "--smoke", "--device", "cpu", "--workers", "3",
                            "--batch-size", "4", "--zip-dir", str(root / "zips"),
                            "--index-dir", str(root / "index"), "--shards", "0", "1", "2",
                            "--poll", "0.01", "--wait-timeout", "30"]))
    zips = root / "zips"
    assert z.marker(zips, "train", 0, "m.done").exists()
    assert not z.marker(zips, "train", 1, "m.done").exists()
    assert z.zip_path(zips, "train", 1).exists()  # kept for the restart
    monkeypatch.setattr(e, "embed_jobs", real)
    stats = e.run(e.parse_args(["--ids", str(root / "a.csv"), str(root / "b.csv"), "--source",
                                "train", "--out", str(out), "--smoke", "--device", "cpu",
                                "--workers", "3", "--batch-size", "4", "--zip-dir", str(zips),
                                "--index-dir", str(root / "index"), "--shards", "0", "1", "2",
                                "--poll", "0.01", "--wait-timeout", "30"]))
    t.join(60)
    assert "err" not in box and stats["mode"] == "zips" and stats["target"] == len(wanted)
    pd.DataFrame({"id": wanted}).to_csv(root / "all.csv", index=False)
    res = _check_zips(root, out, root / "all.csv")
    assert res["ok"] and res["unique"] == len(wanted)  # z4/z5 skipped, nothing twice
    assert not list((zips / "images").rglob("*.zip"))


def test_embed_zip_mode_args_are_validated():
    with pytest.raises(SystemExit):
        e.parse_args(["--ids", "a.csv", "--source", "train", "--out", "o", "--zip-dir", "z"])
    with pytest.raises(SystemExit):
        e.parse_args(["--ids", "a.csv", "--source", "train", "--out", "o", "--zip-dir", "z",
                      "--index-dir", "i", "--shards", "0", "--limit", "5"])
    a = e.parse_args(["--ids", "a.csv", "--source", "train", "--out", "x/clip_b16", "--zip-dir",
                      "z", "--index-dir", "i", "--shards", "0"])
    assert a.consumer == "clip_b16"


# ----------------------------------------------------------------- FT archive


def _hub_opener(root: Path):
    def opener(src, i):
        return open(root / "hub" / src / f"{i:02d}.zip", "rb")  # the caller closes it
    return opener


def _hub_bytes(root: Path) -> dict[str, bytes]:
    out = {}
    for s in range(N_SHARDS):
        with zipfile.ZipFile(root / "hub" / "train" / f"{s:02d}.zip") as zf:
            out.update({Path(n).stem: zf.read(n) for n in zf.namelist()})
    return out


def test_archive_consumer_parts_manifest_and_sha256(hub):
    root, ids, fetch, _ = hub
    pd.DataFrame({"id": ids[:12], "nat_rank": range(12)}).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"id": ["z20", "z21"]}).to_csv(root / "dev.csv", index=False)
    t, box, _ = _downloader(root, fetch, ["ft_archive"])
    out, staging = root / "ft", root / "staging"
    fa.main(["consume", "--ids", str(root / "train.csv"), str(root / "dev.csv"), "--nat-max", "6",
             "--zip-dir", str(root / "zips"), "--index-dir", str(root / "index"),
             "--shards", "0", "1", "2", "--poll", "0.01", "--staging", str(staging),
             "--out", str(out), "--part-gb", "1e-6"])  # 1 KB parts: one part per chunk
    t.join(60)
    assert "err" not in box
    m = pd.read_csv(out / "manifest.csv", dtype={"id": str})
    assert set(m["id"]) == {*ids[:6], "z20", "z21"} and not m["id"].duplicated().any()
    assert m["part"].nunique() > 1 and not list(staging.glob("chunk_*"))
    src = _hub_bytes(root)
    for part in sorted(out.glob("part_*.zip")):
        with zipfile.ZipFile(part) as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED
                assert zf.read(info) == src[Path(info.filename).stem]  # byte for byte
    assert all(hashlib.sha256(src[r.id]).hexdigest() == r.sha256 for r in m.itertuples())


def test_pack_redoes_a_cut_off_part_and_never_duplicates_a_chunk(hub, monkeypatch):
    root, ids, _, _ = hub
    opener = _hub_opener(root)
    staging, out = root / "staging", root / "ft"
    staging.mkdir()
    index = {s: z.index_zip(root / "hub" / "train" / f"{s:02d}.zip") for s in range(N_SHARDS)}
    for s in range(N_SHARDS):
        fa.stage_shard(opener, "train", s, sorted(index[s].items()), staging)
    backup = {p.name: p.read_bytes() for p in staging.iterdir()}

    def crash(*a):
        raise RuntimeError("killed before the commit")

    monkeypatch.setattr(fa, "verify_part", crash)
    with pytest.raises(RuntimeError):
        fa.pack(staging, out, 1.0, final=True)
    assert (out / "part_000.zip").exists() and not (out / "part_000.csv").exists()
    monkeypatch.undo()
    fa.pack(staging, out, 1.0, final=True)  # the uncommitted part is redone
    assert sorted(p.name for p in out.glob("part_*")) == ["part_000.csv", "part_000.sha256",
                                                          "part_000.zip"]
    for name, data in backup.items():  # crash after the commit, before the chunks were removed
        (staging / name).write_bytes(data)
    fa.pack(staging, out, 1.0, final=True)
    m = pd.read_csv(out / "manifest.csv", dtype={"id": str})
    assert len(m) == len(ids) and not list(staging.iterdir()) and len(list(out.glob("part_*.zip"))) == 1


def test_pack_refuses_staged_bytes_that_differ_from_the_source(hub):
    root, _, _, _ = hub
    opener = _hub_opener(root)
    staging = root / "staging"
    staging.mkdir()
    fa.stage_shard(opener, "train", 0, sorted(z.index_zip(root / "hub/train/00.zip").items()), staging)
    csv = staging / "chunk_train_00.csv"
    df = pd.read_csv(csv, dtype=str)
    df.loc[0, "sha256"] = "0" * 64
    df.to_csv(csv, index=False)
    with pytest.raises(AssertionError, match="differ"):
        fa.pack(staging, root / "ft", 1.0, final=True)


class FakeHfApi:
    """huggingface_hub.HfApi stand-in: files in a dict; `fail` = errors upload_file raises first."""

    def __init__(self, private=True, fail=()):
        self.private, self.fail = private, list(fail)
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple] = []

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def repo_info(self, repo_id, *, repo_type):
        return SimpleNamespace(private=self.private)

    def get_paths_info(self, repo_id, paths, *, repo_type):
        return [SimpleNamespace(path=q, lfs=SimpleNamespace(sha256=hashlib.sha256(self.files[q]).hexdigest()))
                for q in paths if q in self.files]

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.calls.append(("upload_file", path_in_repo))
        if self.fail:
            raise self.fail.pop(0)
        self.files[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def uploads(self):
        return [c[1] for c in self.calls if c[0] == "upload_file"]


def _http_error(code: int, msg: str = "") -> HfHubHTTPError:
    req = httpx2.Request("POST", "https://huggingface.co/api/datasets/u/x")
    return HfHubHTTPError(f"{code} Client Error {msg}", response=httpx2.Response(code, request=req))


TOKEN = "hf_SECRETtoken0123456789"


@pytest.fixture
def hf(monkeypatch):
    api = FakeHfApi()
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(fa, "HfApi", lambda token: api)
    return api


def _archive(root: Path) -> Path:
    """3 committed parts (one per shard) and manifest.csv in root/ft."""
    staging, out = root / "staging", root / "ft"
    staging.mkdir()
    for s in range(N_SHARDS):
        items = sorted(z.index_zip(root / "hub" / "train" / f"{s:02d}.zip").items())
        fa.stage_shard(_hub_opener(root), "train", s, items, staging)
    fa.pack(staging, out, 1e-6, final=True)
    return out


def _upload(out: Path, *extra: str):
    fa.main(["upload", "--out", str(out), "--repo", "u/x-pilot", "--backoff", "0", *extra])


def test_hf_upload_is_private_verified_deletes_parts_and_sends_the_manifest_last(hub, hf, capsys):
    out = _archive(hub[0])
    shas = {c.with_suffix(".zip").name: c.with_suffix(".sha256").read_text().strip()
            for c in out.glob("part_*.csv")}
    assert len(shas) == 3
    log = hub[0] / "upload.log"
    _upload(out, "--log", str(log))
    assert hf.calls[0] == ("create_repo", "u/x-pilot", "dataset", True, True)
    assert hf.uploads()[-1] == "manifest.csv"
    for name, sha in shas.items():
        assert hashlib.sha256(hf.files[name]).hexdigest() == sha
        assert hf.files[name.replace(".zip", ".sha256")].decode().strip() == sha
    assert not list(out.glob("part_*.zip"))  # deleted after the upload was checked
    assert len(list(out.glob("part_*.csv"))) == len(list(out.glob("part_*.sha256"))) == 3
    assert "private=True" in log.read_text()
    printed = capsys.readouterr()
    assert TOKEN not in printed.out + printed.err + log.read_text()


def test_hf_upload_refuses_a_repo_that_is_not_private(hub, hf):
    out = _archive(hub[0])
    hf.private = False
    with pytest.raises(SystemExit, match="not private"):
        _upload(out)
    assert hf.uploads() == [] and len(list(out.glob("part_*.zip"))) == 3


def test_hf_upload_resume_skips_parts_already_in_the_repo(hub, hf):
    out = _archive(hub[0])
    hf.files["part_000.zip"] = (out / "part_000.zip").read_bytes()  # uploaded before a crash
    _upload(out)
    assert "part_000.zip" not in hf.uploads() and "part_001.zip" in hf.uploads()
    assert not (out / "part_000.zip").exists()
    hf.calls.clear()
    _upload(out)  # everything is there: only the manifest is sent again
    assert hf.uploads() == ["manifest.csv"]


def test_hf_upload_fails_fast_on_403_and_never_logs_the_token(hub, hf, capsys):
    out = _archive(hub[0])
    hf.fail = [_http_error(403, f"Forbidden (token {TOKEN})")]
    log = hub[0] / "upload.log"
    with pytest.raises(SystemExit) as e:
        _upload(out, "--log", str(log))
    assert hf.uploads() == ["part_000.zip"]  # one attempt, no retry
    assert (out / "part_000.zip").exists() and not hf.files
    msg = str(e.value.code)
    assert "403" in msg and "***" in msg and TOKEN not in msg
    printed = capsys.readouterr()
    assert TOKEN not in printed.out + printed.err + log.read_text()


def test_hf_upload_retries_5xx_429_and_network_errors(hub, hf):
    out = _archive(hub[0])
    hf.fail = [_http_error(503), _http_error(429), ConnectionError("reset")]
    _upload(out)
    assert hf.uploads()[:4] == ["part_000.zip"] * 4 and "manifest.csv" in hf.files
    assert fa.transient(_http_error(500)) and not fa.transient(_http_error(401))
    assert not fa.transient(_http_error(404)) and not fa.transient(ValueError())


def test_hf_upload_needs_hf_token_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        fa.main(["upload", "--out", str(tmp_path), "--repo", "u/x-pilot"])


def test_consume_uploads_each_part_as_it_is_made(hub, hf):
    root, ids, fetch, _ = hub
    pd.DataFrame({"id": ids, "nat_rank": range(len(ids))}).to_csv(root / "train.csv", index=False)
    t, box, _ = _downloader(root, fetch, ["ft_archive"])
    out = root / "ft"
    fa.main(["consume", "--ids", str(root / "train.csv"), "--zip-dir", str(root / "zips"),
             "--index-dir", str(root / "index"), "--shards", "0", "1", "2", "--poll", "0.01",
             "--staging", str(root / "staging"), "--out", str(out), "--part-gb", "1e-6",
             "--repo", "u/x-pilot", "--backoff", "0"])
    t.join(60)
    assert "err" not in box
    assert hf.uploads()[-1] == "manifest.csv" and not list(out.glob("part_*.zip"))
    m = pd.read_csv(out / "manifest.csv", dtype={"id": str})
    assert set(m["id"]) == set(ids)
    src = _hub_bytes(root)
    for part in m["part"].unique():
        with zipfile.ZipFile(io.BytesIO(hf.files[part])) as zf:
            assert all(zf.read(i) == src[Path(i.filename).stem] for i in zf.infolist())


def test_smoke_uploads_two_parts_verifies_the_download_and_reruns_without_reuploading(
        tmp_path, hf, monkeypatch, capsys):
    def download(repo, path, *, repo_type, token, local_dir):
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(hf.files[path])
        return str(dst)

    monkeypatch.setattr(fa, "hf_hub_download", download)
    with pytest.raises(SystemExit, match="pilot"):
        fa.main(["smoke", "--repo", "u/osv5m-ft-archive", "--work", str(tmp_path / "w")])
    fa.main(["smoke", "--repo", "u/x-pilot", "--work", str(tmp_path / "w")])
    assert sorted(f for f in hf.files if f.endswith(".zip")) == [
        "smoke/part_000.zip", "smoke/part_001.zip"]
    assert "SMOKE OK" in capsys.readouterr().out
    hf.calls.clear()
    fa.main(["smoke", "--repo", "u/x-pilot", "--work", str(tmp_path / "w")])  # same bytes
    assert hf.uploads() == ["smoke/manifest.csv"]


def test_wanted_ids_filters_nat_rank_only_where_present(tmp_path):
    pd.DataFrame({"id": ["a", "b", "c"], "nat_rank": [0, 5, 1]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"id": ["NA"]}).to_csv(tmp_path / "d.csv", index=False)
    assert fa.wanted_ids([tmp_path / "t.csv", tmp_path / "d.csv"], 2) == {"a", "c", "NA"}
    assert np.array_equal(sorted(fa.wanted_ids([tmp_path / "t.csv"], 10)), ["a", "b", "c"])
