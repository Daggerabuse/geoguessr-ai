"""find_input.sh: locate the attached run's base dir however deep Kaggle mounts it."""
import shutil
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "find_input.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def run(root, *rel):
    return subprocess.run([BASH, "-c", f". '{HELPER.as_posix()}'; find_input \"$@\"", "_", root.as_posix(), *rel],
                          capture_output=True, text=True, timeout=30, check=False)


def make(root, rel):
    (root / rel / "v2").mkdir(parents=True)
    (root / rel / "v2" / "config.json").write_text("{}")


@pytest.mark.parametrize("rel", ["v2-pilot", "notebooks/daggerabuse/v2-pilot"])
def test_one_match(tmp_path, rel):
    make(tmp_path, rel)
    out = run(tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (tmp_path / rel).as_posix()


def test_two_matches_fail_and_list(tmp_path):
    make(tmp_path, "a")
    make(tmp_path, "notebooks/u/b")
    out = run(tmp_path)
    assert out.returncode != 0 and out.stdout == ""
    assert "found 2" in out.stderr and "a/v2/config.json" in out.stderr and "u/b/v2/config.json" in out.stderr


def test_no_match_fails(tmp_path):
    out = run(tmp_path)
    assert out.returncode != 0 and "found 0" in out.stderr


def test_other_relative_path(tmp_path):
    make(tmp_path, "data")
    (tmp_path / "notebooks/u/old-eval/phase1_eval").mkdir(parents=True)
    (tmp_path / "notebooks/u/old-eval/phase1_eval/results.csv").write_text("")
    out = run(tmp_path, "phase1_eval/results.csv")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (tmp_path / "notebooks/u/old-eval").as_posix()
