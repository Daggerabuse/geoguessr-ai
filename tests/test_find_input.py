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


def _symlink(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted here")


def test_symlinked_nested_mount(tmp_path):
    """Kaggle may mount an input as a symlink: plain find does not descend into it."""
    make(tmp_path / "store", "v2-pilot")
    _symlink(tmp_path / "input/notebooks/daggerabuse/v2-pilot", tmp_path / "store/v2-pilot")
    out = run(tmp_path / "input")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (tmp_path / "input/notebooks/daggerabuse/v2-pilot").as_posix()


def test_symlinked_root(tmp_path):
    make(tmp_path / "real", "notebooks/u/v2-pilot")
    _symlink(tmp_path / "input", tmp_path / "real")
    out = run(tmp_path / "input")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == (tmp_path / "input/notebooks/u/v2-pilot").as_posix()


def test_final_eval_inputs_side_by_side(tmp_path):
    """The final eval's three notebook inputs, nested: each REL picks its own one."""
    nb = tmp_path / "notebooks/daggerabuse"
    make(nb, "v2-pilot")
    (nb / "v2-pilot/emb/streetclip").mkdir(parents=True)
    (nb / "v2-test-embed/emb_test").mkdir(parents=True)
    (nb / "v2-test-embed/emb_test/prereg_provenance.json").write_text("{}")
    (nb / "v2-phase1-eval/phase1_eval").mkdir(parents=True)
    (nb / "v2-phase1-eval/phase1_eval/results.csv").write_text("")
    for rel, want in (((), "v2-pilot"), (("emb_test/prereg_provenance.json",), "v2-test-embed"),
                      (("phase1_eval/results.csv",), "v2-phase1-eval")):
        out = run(tmp_path, *rel)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == (nb / want).as_posix()


def test_no_match_shows_what_is_mounted(tmp_path):
    (tmp_path / "notebooks/u/other").mkdir(parents=True)
    out = run(tmp_path)
    assert out.returncode != 0 and "found 0" in out.stderr and "notebooks/u/other" in out.stderr


SCRIPTS = HELPER.parent


def test_kaggle_scripts_locate_inputs_only_via_find_input():
    for f in SCRIPTS.glob("kaggle_*.sh"):
        for line in f.read_text().splitlines():
            if "/kaggle/input" in line and not line.lstrip().startswith("#"):
                assert "find_input /kaggle/input" in line, f"{f.name}: {line}"


@pytest.mark.parametrize("name", ["kaggle_phase1.sh", "kaggle_phase1_eval.sh", "kaggle_phase1_final.sh"])
def test_scripts_parse(name):
    out = subprocess.run([BASH, "-n", (SCRIPTS / name).as_posix()], capture_output=True, text=True,
                         timeout=30, check=False)
    assert out.returncode == 0, out.stderr
