"""The Kaggle shell scripts end to end, with the GPU work stubbed: python -m geoguessr_ai.<x>
(except `phase1 config` and inline `python -` snippets, which run for real), nvidia-smi and free
are fakes on PATH, and KAGGLE_ROOT points /kaggle at a temp dir."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BASH = shutil.which("bash")
# job control and process groups as on Kaggle; KAGGLE_SCRIPT_TESTS=1 runs them elsewhere too
LINUX = pytest.mark.skipif(BASH is None or (sys.platform != "linux"
                                            and not os.environ.get("KAGGLE_SCRIPT_TESTS")),
                           reason="Linux only (bash job control as on Kaggle)")

STUB = r"""#!/usr/bin/env bash
# fake python: logs each geoguessr_ai call, stubs the GPU work, runs everything else for real
out=; prev=
for a in "$@"; do [ "$prev" = --out ] && out=$a; prev=$a; done
case " $* " in
  *" -m geoguessr_ai."*) echo "$*" >> "$STUB_LOG" ;;
esac
case " $* " in
  *" geoguessr_ai.phase1 config "*) exec "$REAL_PY" "$@" ;;
  *" geoguessr_ai.prereg provenance "*) echo '{}' > "$out"; echo provenance recorded ;;
  *" geoguessr_ai.phase1 progress "*) echo "progress: all zips done"; exec sleep 300 ;;
  *" geoguessr_ai.phase1 status "*) echo "status: all zips done" ;;
  *" geoguessr_ai.embed --check "*)
    n=${STUB_N:-5}; bad=0; [ "$(basename "$out")" = "${STUB_BAD:-}" ] && bad=1
    echo "{\"out\": \"$out\", \"files\": 5, \"expected\": $n, \"unique\": $((n - bad)), \"duplicates\": 0, \"missing\": $bad, \"failed\": 0, \"extra\": 0, \"temp_files_left\": 0, \"ok\": $([ $bad = 0 ] && echo true || echo false)}"
    exit $bad ;;
  *" geoguessr_ai.embed "*|*" geoguessr_ai.zipq "*) sleep 2 ;;
  *" geoguessr_ai.phase1_eval "*) echo "EVAL RAN" ;;
  *" geoguessr_ai.phase3_eval "*)
    stop=; prev=; for a in "$@"; do [ "$prev" = --stop-file ] && stop=$a; prev=$a; done
    if [ -n "${STUB_P3_WAIT:-}" ]; then  # runs until the deadline's STOP marker appears
      for _ in $(seq 1 100); do [ -e "$stop" ] && break; sleep 0.1; done
      [ -e "$stop" ] && { echo "RESUME NEEDED: stub stopped"; exit 3; }
    fi
    echo "P3 RAN" ;;
  *" -m geoguessr_ai."*) echo "unexpected: $*" >&2; exit 9 ;;
  *) exec "$REAL_PY" "$@" ;;
esac
"""


def _env(tmp_path, **extra):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in (("python", STUB), ("nvidia-smi", ("#!/usr/bin/env bash\n"
                        "[ \"${1:-}\" = -L ] && printf 'GPU 0\\nGPU 1\\n' || echo 'Tesla T4'\n")),
                       ("free", "#!/usr/bin/env bash\necho free\n")):
        (bin_ / name).write_text(body, newline="\n")
        (bin_ / name).chmod(0o755)
    for d in ("input", "temp", "working"):
        (tmp_path / "kaggle" / d).mkdir(parents=True)
    return {**os.environ, "PATH": f"{bin_.as_posix()}{os.pathsep}{os.environ['PATH']}",
            "REAL_PY": Path(sys.executable).as_posix(), "STUB_LOG": (tmp_path / "calls.log").as_posix(),
            "KAGGLE_ROOT": (tmp_path / "kaggle").as_posix(), **extra}


def _bash(script, env, *args, timeout=120):
    return subprocess.run([BASH, (SCRIPTS / script).as_posix(), *args], env=env, capture_output=True,
                          text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL)


def _calls(tmp_path):
    f = tmp_path / "calls.log"
    return f.read_text().splitlines() if f.exists() else []


@LINUX
def test_test_mode_post_work_path_reaches_the_checks(tmp_path):
    """After the consumers end, the helpers are reaped (the progress logger dies of SIGTERM:
    143) and the script must go on to the status, the checks and PASSED, exiting 0."""
    inp = tmp_path / "in"
    (inp / "v2").mkdir(parents=True)
    (inp / "v2" / "config.json").write_text("{}")
    (inp / "v2" / "test.csv").write_text("id\n1\n")
    r = _bash("kaggle_phase1.sh", _env(tmp_path, INPUT_DIR=inp.as_posix()), "test")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "=== checks" in out and "TEST RUN PASSED" in out and "summary (test, exit 0)" in out
    checks = [c for c in _calls(tmp_path) if "embed --check" in c]
    assert len(checks) == 3 and all("--source test" in c for c in checks)


def _final_inputs(tmp_path):
    nb = tmp_path / "kaggle" / "input" / "notebooks" / "u"
    for d in ("v2-pilot/v2", *(f"v2-pilot/emb/{m}" for m in ("streetclip", "clip_l14_336", "clip_b16")),
              "v2-test-embed/emb_test/zip_index",
              *(f"v2-test-embed/emb_test/{m}" for m in ("streetclip", "clip_l14_336", "clip_b16"))):
        (nb / d).mkdir(parents=True, exist_ok=True)
    (nb / "v2-pilot/v2/config.json").write_text("{}")
    (nb / "v2-pilot/v2/test.csv").write_text("id\n1\n")
    (nb / "v2-test-embed/emb_test/prereg_provenance.json").write_text("{}")


@LINUX
def test_final_checks_every_backbone_before_evaluating(tmp_path):
    env = _env(tmp_path, TEST_IMAGES="5")
    _final_inputs(tmp_path)
    r = _bash("kaggle_phase1_final.sh", env)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    calls = _calls(tmp_path)
    assert [c.split("--out ")[1].split()[0].rsplit("/", 1)[1] for c in calls[:3]] == [
        "streetclip", "clip_l14_336", "clip_b16"] and "phase1_eval --eval-test" in calls[3]
    assert r.stdout.count("-> PASSED") == 3 and "EVAL RAN" in r.stdout
    res = json.loads((tmp_path / "kaggle/working/phase1_eval_test/check_clip_b16.json").read_text())
    assert res["expected"] == 5 and res["ok"]


@LINUX
@pytest.mark.parametrize("env_extra", [{"TEST_IMAGES": "5", "STUB_BAD": "clip_l14_336"},
                                       {"TEST_IMAGES": "210122"}])  # one image missing; wrong total
def test_final_refuses_before_evaluating_if_a_check_fails(tmp_path, env_extra):
    env = _env(tmp_path, **env_extra)
    _final_inputs(tmp_path)
    r = _bash("kaggle_phase1_final.sh", env)
    assert r.returncode == 1 and "no evaluation" in r.stdout
    assert "-> FAILED" in r.stdout and "EVAL RAN" not in r.stdout
    assert not any("phase1_eval" in c for c in _calls(tmp_path))


def test_reap_never_ends_a_set_e_script():
    """reap under set -e: a helper killed by TERM (wait returns 143), one already gone (kill
    fails) and one killed right after it started (before nap set its trap, formerly)."""
    if BASH is None:
        pytest.skip("needs bash")
    script = (f". '{(SCRIPTS / 'bg_helpers.sh').as_posix()}'; set -euo pipefail; "
              "sleep 30 & a=$!; ( nap 30 ) & b=$!; true & c=$!; sleep 1; reap $a $b $c; "
              "( nap 30 ) & d=$!; reap $d; echo reached")
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60,
                       check=False, stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and r.stdout.strip() == "reached", r.stderr


def _p3_inputs(tmp_path):
    inp = tmp_path / "in"
    for d in ("v2", "emb/streetclip", "emb/clip_l14_336", "emb/clip_b16"):
        (inp / d).mkdir(parents=True)
    (inp / "v2" / "config.json").write_text("{}")
    return inp


def _p3_call(tmp_path):
    (call,) = [c for c in _calls(tmp_path) if "phase3_eval" in c]
    return call


@LINUX
def test_phase3_passes_the_env_settings_through_and_picks_the_device(tmp_path):
    inp = _p3_inputs(tmp_path)
    env = _env(tmp_path, INPUT_DIR=inp.as_posix(), METHODS="cell", BACKBONES="streetclip,clip_l14_336",
               ORDERINGS="nat", SIZES="50000 500000", PROBE_INITS="random")
    r = _bash("kaggle_phase3.sh", env)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    c = _p3_call(tmp_path)
    for part in ("--methods cell", "--device cuda", "--backbones streetclip clip_l14_336",
                 "--orderings nat", "--sizes 50000 500000", "--probe-inits random", "--stop-file"):
        assert part in c, c
    assert "session_end_utc" in r.stdout and "(exit 0" in r.stdout


@LINUX
def test_phase3_hybrid_is_a_cpu_run_and_cell_needs_a_gpu(tmp_path):
    inp = _p3_inputs(tmp_path)
    r = _bash("kaggle_phase3.sh", _env(tmp_path, INPUT_DIR=inp.as_posix(), METHODS="hybrid"))
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert "--methods hybrid --device  --knn-dtype fp32" in _p3_call(tmp_path)  # '' = CPU
    (tmp_path / "bin" / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 1\n", newline="\n")
    (tmp_path / "calls.log").unlink()
    r = _bash("kaggle_phase3.sh", _env_again(tmp_path, inp, METHODS="cell"))
    assert r.returncode == 1 and "needs a GPU" in r.stdout and not _calls(tmp_path)


def _env_again(tmp_path, inp, **extra):  # the same fakes on PATH (the fake nvidia-smi rewritten)
    bin_ = tmp_path / "bin"
    return {**os.environ, "PATH": f"{bin_.as_posix()}{os.pathsep}{os.environ['PATH']}",
            "REAL_PY": Path(sys.executable).as_posix(), "STUB_LOG": (tmp_path / "calls.log").as_posix(),
            "KAGGLE_ROOT": (tmp_path / "kaggle").as_posix(), "INPUT_DIR": inp.as_posix(), **extra}


@LINUX
def test_phase3_deadline_writes_stop_then_exits_3_with_resume_needed(tmp_path):
    inp = _p3_inputs(tmp_path)
    env = _env(tmp_path, INPUT_DIR=inp.as_posix(), METHODS="cell", DEADLINE_MIN="0.02",
               STUB_P3_WAIT="1")
    r = _bash("kaggle_phase3.sh", env)
    out = r.stdout + r.stderr
    assert r.returncode == 3, out[-3000:]
    assert "DEADLINE: STOP written" in out and out.count("RESUME NEEDED") == 2  # python's + script's
    assert "session_end_utc" in out and "(exit 3" in out
    quota = (tmp_path / "kaggle/working/phase3_eval/quota_times.txt").read_text()
    assert "session_start_utc" in quota and "session_end_utc" in quota


@LINUX
def test_phase3_seed_from_copies_a_finished_output_and_refuses_ambiguous_ones(tmp_path):
    inp = _p3_inputs(tmp_path)
    env = _env(tmp_path, INPUT_DIR=inp.as_posix(), SEED_FROM=(tmp_path / "kaggle/input/runA/phase3_eval").as_posix())
    seed = tmp_path / "kaggle/input/runA/phase3_eval"
    (seed / "streetclip/nat_50000_geo_cell_random").mkdir(parents=True)
    (seed / "streetclip/nat_50000_geo_cell_random/row.json").write_text("{}")
    (seed / "cell_count.json").write_text("{}")
    (seed / "STOP").write_text("")
    out = tmp_path / "kaggle/working/phase3_eval"
    r = _bash("kaggle_phase3.sh", env)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert (out / "cell_count.json").exists() and (out / "streetclip/nat_50000_geo_cell_random/row.json").exists()
    assert not (out / "STOP").exists()  # a copied marker would stop the run at once
    shutil.rmtree(out)  # now the refusals: each exits 1 before phase3_eval is called
    for name, path, msg in (("missing", tmp_path / "nope", "not a directory"),
                            ("parent", seed.parent, "not a phase3_eval output dir")):
        (tmp_path / "calls.log").unlink(missing_ok=True)
        r = _bash("kaggle_phase3.sh", _env_again(tmp_path, inp, SEED_FROM=path.as_posix()))
        assert r.returncode == 1 and msg in r.stdout and not _calls(tmp_path), (name, r.stdout)
    assert "found: " + seed.as_posix() in r.stdout  # the nested output is named
    out.mkdir(parents=True, exist_ok=True)
    (out / "x").write_text("")
    r = _bash("kaggle_phase3.sh", _env_again(tmp_path, inp, SEED_FROM=seed.as_posix()))
    assert r.returncode == 1 and "not empty" in r.stdout
