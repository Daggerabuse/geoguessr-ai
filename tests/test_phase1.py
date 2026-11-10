import json

import pytest

from geoguessr_ai import phase1 as ph
from geoguessr_ai import zipq as z


def _shell(text: str) -> dict:
    return dict(line.split("=", 1) for line in text.splitlines())


def test_modes_pilot_and_full():
    pilot, full = _shell(ph.config("pilot")), _shell(ph.config("full"))
    assert pilot["SHARDS"] == "'0 1 2'" and pilot["DEADLINE_MIN"] == "40"
    assert pilot["REPO"] == "Daggerabuse/osv5m-ft-archive-pilot" and pilot["OUT_NAME"] == "pilot"
    assert full["SHARDS"] == "'" + " ".join(map(str, range(98))) + "'"
    assert full["DEADLINE_MIN"] == "660" and full["OUT_NAME"] == "full"
    assert full["REPO"] == "Daggerabuse/osv5m-ft-archive"
    with pytest.raises(SystemExit, match="unknown mode"):
        ph.config("test")


def test_deadline_override_from_the_environment(monkeypatch):
    monkeypatch.setenv("DEADLINE_MIN", "5")
    assert _shell(ph.config("full"))["DEADLINE_MIN"] == "5"


@pytest.mark.parametrize(("mode", "repo", "ok"), [
    ("pilot", "Daggerabuse/osv5m-ft-archive-pilot", True),
    ("pilot", "u/anything-pilot", True),
    ("pilot", "Daggerabuse/osv5m-ft-archive", False),
    ("full", "Daggerabuse/osv5m-ft-archive", True),
    ("full", "Daggerabuse/osv5m-ft-archive-pilot", False),
    ("full", "daggerabuse/osv5m-ft-archive", False),
    ("full", "Other/osv5m-ft-archive", False),
    ("test", "Daggerabuse/osv5m-ft-archive", False),
])
def test_repo_guard(mode, repo, ok):
    if ok:
        ph.check_repo(mode, repo)
    else:
        with pytest.raises(SystemExit):
            ph.check_repo(mode, repo)


def _markers(tmp_path, done: dict):
    zips = tmp_path / "zips"
    for consumer, shards in done.items():
        for s in shards:
            z.write_text(z.marker(zips, "train", s, f"{consumer}.done"), "")
    return zips


def test_status_complete_or_resume_needed(tmp_path, capsys):
    zips = _markers(tmp_path, {"a": [0, 1, 2], "ft_archive": [0, 1, 2]})
    assert ph.status(zips, "train", [0, 1, 2], ["a", "ft_archive"])[0]
    zips = _markers(tmp_path, {"b": [0]})
    z.write_text(zips / "markers" / "STOP", "deadline")
    ok, text = ph.status(zips, "train", [0, 1, 2], ["a", "b"])
    assert not ok and text.startswith("RESUME NEEDED (deadline STOP)")
    assert "b: 1/3 zips done; left: [1, 2]" in text and "a: 3/3" in text
    with pytest.raises(SystemExit) as e:
        ph.main(["status", "--zip-dir", str(zips), "--shards", "0", "1", "2", "--consumers", "b"])
    assert e.value.code == 3 and "RESUME NEEDED" in capsys.readouterr().out


def test_progress_line(tmp_path):
    zips = _markers(tmp_path, {"m": [0, 1], "ft_archive": [0]})
    idx = tmp_path / "index"
    for s in range(4):
        z.write_text(z.index_path(idx, "train", s), json.dumps({f"i{s}{k}": f"{k}.jpg" for k in range(10)}))
    (tmp_path / "disk").mkdir()
    (tmp_path / "disk" / "f").write_bytes(b"x" * 2_000_000)
    wanted = {f"i{s}{k}" for s in range(4) for k in range(5)}  # 5 wanted per zip
    line = ph.progress_line(zips, idx, "train", [0, 1, 2, 3], ["m", "ft_archive"], wanted,
                            t0=0.0, disks=[("temp", tmp_path / "disk")], now=3600.0)
    assert "m 2/4 zips 0.0 img/s ETA 1.0 h" in line  # 10 images in an hour; 2 zips/h left 2
    assert "ft_archive 1/4 zips ETA 3.0 h" in line and "temp 0.0 GB" in line
    assert line.startswith("[01:00:00 UTC, 60 min]")
