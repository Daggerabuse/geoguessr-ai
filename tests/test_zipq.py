"""Local-zip pipeline: zipq downloader + markers, embed.py consumer mode, FT image archive."""

import hashlib
import json
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert sorted(p.name for p in out.glob("part_*")) == ["part_000.csv", "part_000.zip"]
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


FAKE_KAGGLE = """
import json, os, sys
state = os.environ["FAKE_STATE"]
calls = json.load(open(state)) if os.path.exists(state) else []
calls.append(sys.argv[1:])
json.dump(calls, open(state, "w"))
print("auth with key", os.environ["KAGGLE_KEY"])  # a CLI that leaks the key
sys.stderr.write("debug: " + os.environ["KAGGLE_KEY"] + "\\n")
exists = os.environ.get("FAKE_EXISTS") == "1" or any(c[1] == "create" for c in calls[:-1])
if sys.argv[2] == "version":
    if not exists:
        print("404 Client Error: Not Found for url: .../datasets/someone/x")
        sys.exit(1)
    versions = sum(c[1] == "version" for c in calls)
    sys.exit(1 if os.environ.get("FAKE_FLAKY") == "1" and versions == 1 else 0)
sys.exit(1 if exists else 0)  # create
"""


def test_upload_is_private_retried_and_never_leaks_the_key(tmp_path, monkeypatch, capsys):
    key = "kgl-SECRET-0123456789"
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", key)
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "calls.json"))
    (tmp_path / "fake_kaggle.py").write_text(FAKE_KAGGLE)
    out = tmp_path / "ft"
    out.mkdir()
    (out / "manifest.csv").write_text("id\n")
    log = tmp_path / "upload.log"
    ok = fa.upload(out, "osv5m-ft-pilot", "t", [sys.executable, str(tmp_path / "fake_kaggle.py")],
                   retries=2, backoff=0, log_path=log)
    assert ok
    calls = json.loads((tmp_path / "calls.json").read_text())
    assert [c[1] for c in calls] == ["version", "create"]  # not found -> create
    assert not any("--public" in c for c in calls)
    printed = capsys.readouterr()
    for text in (printed.out, printed.err, log.read_text(), (out / "dataset-metadata.json").read_text()):
        assert key not in text
    assert "***" in log.read_text()
    assert json.loads((out / "dataset-metadata.json").read_text())["id"] == "someone/osv5m-ft-pilot"
    assert not list(tmp_path.rglob("kaggle.json"))


def test_upload_versions_an_existing_dataset_and_retries_other_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", "k3y-value")
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "calls.json"))
    monkeypatch.setenv("FAKE_EXISTS", "1")
    monkeypatch.setenv("FAKE_FLAKY", "1")  # first version fails with a non-404 error
    (tmp_path / "fake_kaggle.py").write_text(FAKE_KAGGLE)
    out = tmp_path / "ft"
    out.mkdir()
    (out / "manifest.csv").write_text("id\n")
    assert fa.upload(out, "x-pilot", "t", [sys.executable, str(tmp_path / "fake_kaggle.py")],
                     retries=2, backoff=0)
    calls = json.loads((tmp_path / "calls.json").read_text())
    assert [c[1] for c in calls] == ["version", "version"]  # retried, never create, never public
    assert not any("--public" in c for c in calls)


def test_upload_requires_credentials_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    with pytest.raises(SystemExit, match="KAGGLE_KEY"):
        fa.upload(tmp_path, "x", "t", ["kaggle"], 0, 0)


def test_wanted_ids_filters_nat_rank_only_where_present(tmp_path):
    pd.DataFrame({"id": ["a", "b", "c"], "nat_rank": [0, 5, 1]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"id": ["NA"]}).to_csv(tmp_path / "d.csv", index=False)
    assert fa.wanted_ids([tmp_path / "t.csv", tmp_path / "d.csv"], 2) == {"a", "c", "NA"}
    assert np.array_equal(sorted(fa.wanted_ids([tmp_path / "t.csv"], 10)), ["a", "b", "c"])
