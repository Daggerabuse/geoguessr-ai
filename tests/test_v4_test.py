"""v4_test: the gate, the outcome rules and a full run on the tiny CLIP with a fake test split."""

import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_eval as p3
from geoguessr_ai import prereg
from geoguessr_ai import v4_test as t
from geoguessr_ai import v4_train as vt
from geoguessr_ai.v4_inputs import file_sha256, read_csv
from tests.conftest import write_jpeg
from tests.test_phase1_eval import _git
from tests.test_v4_train import make_world

N_TEST, LO, HI = 8, 12, 24
NAMES = {4: "B_full_50k", 5: "B_full_500k", 6: "B_full_50k_seed2"}


def fake_run(w, cfg, k, n, bb, seed, commit="trainsha"):
    d = w["runs"] / f"k{k}_{NAMES[k]}_{bb}"
    (d / "final").mkdir(parents=True)
    hf = w["inputs"] / "heads" / bb / f"bal_{n}_cell_zeroshot.npy"
    model, tr, size = vt.build_model(cfg, "smoke", np.load(hf), "full", seed, True, torch.device("cpu"))
    torch.save(vt.weights(tr), d / "weights_final.pt")
    dev = read_csv(w["inputs"] / "dev.csv")
    ds = vt.PartImages(w["parts"], vt.archive_index(w["parts"], None), dev["id"], size, False)
    top, _ = vt.Eval("dev", dev, ds, 4, 0).run(model.eval(), torch.device("cpu"), False)
    pd.DataFrame({"id": dev["id"], "c0": top[:, 0]}).to_csv(d / "final" / "dev_predictions.csv", index=False)
    sha = file_sha256(d / "weights_final.pt")
    (d / "DONE").write_text(sha + "\n")
    (d / "result.json").write_text(json.dumps({
        "key": f"key{k}", "model": "smoke", "lr": 1e-4, "seed": seed, "head": file_sha256(hf),
        "head_frozen": True, "code_commit": commit, "weights_final_sha256": sha,
        "dev": {"geoscore": {"mean": 1.0}}, "dev_seen": {"geoscore": {"mean": 2.0}}}))


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    if shutil.which("git") is None:
        pytest.skip("needs git")
    root = tmp_path_factory.mktemp("v4test")
    w = make_world(root, n_train=HI, config="v4_predev_v2.json")
    w["runs"] = root / "runs"
    repo = root / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "results" / "v3").mkdir(parents=True)
    cfg = json.loads(w["config"].read_text())
    for j in cfg["jobs"]:  # job 4 is the small N, as in the real design
        j["n"] = {4: LO, 5: HI, 6: LO}.get(j["k"], j["n"])
    (repo / "configs" / "v4_predev_v2.json").write_text(json.dumps(cfg))
    (repo / "configs" / "v4_selection_v2.json").write_text(json.dumps({"selected": {b: 1e-4 for b in pe.PAIR}}))
    (repo / "configs" / "phase3_final.json").write_text(json.dumps({
        "settings": {"hybrid_k": 5, "mask_km": 1.0, "knn_dtype": "fp32"},
        "cells": [{"backbone": b, "ordering": "bal", "n": LO, "predictor": t.HYBRID, "m": 3, "k": 1}
                  for b in pe.PAIR]}))
    for k, n, seed in ((4, LO, 42), (5, HI, 42), (6, LO, 43)):
        for bb in pe.PAIR:
            fake_run(w, cfg, k, n, bb, seed)

    rng = np.random.default_rng(1)
    ids = [f"q{i:04d}" for i in range(N_TEST)]
    pd.DataFrame({"id": ids, "latitude": rng.uniform(-60, 60, N_TEST), "longitude": rng.uniform(-170, 170, N_TEST),
                  "sequence": [f"s{i}" for i in range(N_TEST)],
                  "creator_id": [f"c{i % 4}" for i in range(N_TEST)]}).to_csv(root / "test.csv", index=False)
    (root / "zips" / "test").mkdir(parents=True)
    for s in range(2):
        with zipfile.ZipFile(root / "zips" / "test" / f"{s:02d}.zip", "w") as zf:
            for j, i in enumerate(ids[s::2]):
                write_jpeg(root / "qjpg" / f"{i}.jpg", 60 + j, 40, seed=j + 100)
                zf.write(root / "qjpg" / f"{i}.jpg", f"x/{i}.jpg")

    rows = []
    for bb in pe.PAIR:
        for n, pred in ((LO, t.CELL), (HI, t.CELL), (LO, t.HYBRID)):
            d = pe.cell_dir(root / "v3", bb, "bal", n, p3.METHOD, pred)
            d.mkdir(parents=True)
            gs = rng.uniform(1000, 4000, N_TEST)
            pd.DataFrame({"id": ids, "geoscore": gs}).to_csv(d / "test_geo_hits.csv", index=False)
            sha = prereg.sha256(repo / "configs" / "phase3_final.json")
            (d / "test_row.json").write_text(json.dumps({"run": {"prereg_sha256": sha}}))
            rows.append({"backbone": bb, "ordering": "bal", "n": n, "predictor": pred,
                         "test_geoscore": pd.read_csv(d / "test_geo_hits.csv")["geoscore"].mean()})
    pd.DataFrame(rows).to_csv(repo / "results" / "v3" / "test_results.csv", index=False)
    _git(root, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    path = repo / "configs" / "v4_test_prereg.json"
    t.write(w["runs"], repo / "configs" / "v4_predev_v2.json", repo / "configs" / "v4_selection_v2.json",
            repo / "configs" / "phase3_final.json", path, test_n=N_TEST, shards=2)
    return root, w, path


def quota(runs, end):
    (runs / "quota_times.txt").write_text(
        f"session_start_utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end - 60))} jobs 6\n"
        f"session_end_utc {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end))} (exit 0, 1 min)\n")


def common(root, w, path, out, sha=None):
    return ["--prereg", str(path), "--prereg-sha", sha or prereg.sha256(path), "--runs", str(w["runs"]),
            "--v3-test", str(root / "v3"), "--test-csv", str(root / "test.csv"), "--out", str(out)]


def test_args(root, w, path, out, job, bb, sha=None):
    return ["test", *common(root, w, path, out, sha), "--job", str(job), "--backbone", bb,
            "--inputs", str(w["inputs"]), "--parts", str(w["parts"]), "--test-zips", str(root / "zips"), "--smoke"]


test_args.__test__ = False


def is_committed(path):
    return bool(subprocess.run(["git", "-C", str(path.parent), "log", "-1", "--format=%H", "--", path.name],
                               capture_output=True, text=True, check=False).stdout.strip())


def committed(root, path):
    if not is_committed(path):
        _git(path.parents[1], "add", "-A")
        _git(path.parents[1], "commit", "-q", "-m", "pre-register")


def test_family_and_declared_hypotheses(base):
    reg = json.loads(base[2].read_text())["hypotheses"]
    assert [h["id"] for h in reg["primary"]] == ["P1/change/500K", "P2/keep/500K", "P3/change/50K"]
    assert "S1/keep/50K" in [h["id"] for h in reg["secondary"]] and reg["margin"] == 40.0
    assert "widened" in reg["outcomes"]["labels"]
    assert any("NOT comparable to V3's H1" in s for s in json.loads(base[2].read_text())["limitations"])


def test_p_values_and_outcome_rules():
    x = np.array([1.0, 2.0, 3.0, -1.0])
    assert t.p_value(x, "two_sided", 40) == pytest.approx(min(1, 2 * min(2 / 5, 4 / 5)))
    assert t.p_value(x, "below_zero", 40) == pytest.approx(4 / 5)
    assert t.p_value(x, "tost", 2.5) == pytest.approx(2 / 5)
    assert t.outcome(-5, True, False, False) == "narrowed"
    assert t.outcome(5, True, False, False) == "widened"
    assert t.outcome(5, True, True, False) == "widened, within the margin"
    assert t.outcome(-5, True, True, True) == "reversed, within the margin"
    assert t.outcome(5, True, False, True) == "widened"  # a reversal needs a narrowing
    assert t.outcome(0, False, True, False) == "kept"
    assert t.outcome(0, False, False, True) == "inconclusive"


def test_gate_refuses_before_any_test_read(base, tmp_path, monkeypatch):
    root, w, path = base
    read = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda f, *a, **k: read.append(str(f)) or real(f, *a, **k))
    args = t.parse_args(test_args(root, w, path, tmp_path / "o", 4, "streetclip"))
    if not is_committed(path):
        with pytest.raises(SystemExit, match="not committed"):
            t.run_test(args)
    committed(root, path)
    with pytest.raises(SystemExit, match="is not --prereg-sha"):
        t.run_test(t.parse_args(test_args(root, w, path, tmp_path / "o", 4, "streetclip", "0" * 64)))
    (w["runs"] / "quota_times.txt").unlink(missing_ok=True)
    (w["runs"] / "quota_times.txt").write_text("")
    with pytest.raises(SystemExit, match="no finished session"):
        t.run_test(args)
    quota(w["runs"], 946684800)  # job 6 finished long before the commit
    with pytest.raises(SystemExit, match="committed after job 6"):
        t.run_test(args)
    done = w["runs"] / "k6_B_full_50k_seed2_streetclip" / "DONE"
    keep = done.read_text()
    done.unlink()
    with pytest.raises(SystemExit, match="has not finished"):
        t.run_test(args)
    done.write_text(keep)
    assert not any(str(root / "test.csv") == f for f in read)
    assert not (tmp_path / "o").exists()  # nothing recorded by a refused run


def test_full_run_then_refuses_changes(base, tmp_path):
    root, w, path = base
    committed(root, path)
    quota(w["runs"], int(time.time()) + 3600)
    out = tmp_path / "o"
    for job in t.JOBS:
        for bb in pe.PAIR:
            assert t.run_test(t.parse_args(test_args(root, w, path, out, job, bb))) == 0
    for bb in pe.PAIR:
        assert json.loads((out / f"k4_B_full_50k_{bb}" / "dev_check.json").read_text())["ok"]
        assert t.run_hybrid(t.parse_args(["hybrid", *common(root, w, path, out), "--job", "4", "--backbone", bb,
                                          "--inputs", str(w["inputs"])])) == 0
    assert t.run_analyze(t.parse_args(["analyze", *common(root, w, path, out)])) == 0
    p, s = pd.read_csv(out / "primary.csv"), pd.read_csv(out / "secondary.csv")
    assert len(p) == 3 and len(s) == 6 and p["evaluable"].all() and s["evaluable"].all()
    res = json.loads((out / "outcomes.json").read_text())
    assert set(res["outcomes"]) == {"50K", "500K"} and res["outcomes"]["50K"] != "kept"

    w4 = w["runs"] / "k4_B_full_50k_streetclip" / "weights_final.pt"
    keep = w4.read_bytes()
    w4.write_bytes(keep + b"0")
    with pytest.raises(SystemExit, match="not the pre-registered weights"):
        t.run_test(t.parse_args(test_args(root, w, path, tmp_path / "o2", 4, "streetclip")))
    w4.write_bytes(keep)
    text = path.read_bytes()
    path.write_bytes(text.replace(b'"margin": 40.0', b'"margin": 60.0'))
    with pytest.raises(SystemExit, match="uncommitted changes"):
        t.run_test(t.parse_args(test_args(root, w, path, tmp_path / "o3", 4, "streetclip")))
    path.write_bytes(text)


def test_a_failed_dev_check_never_reads_test(base, tmp_path, monkeypatch):
    root, w, path = base
    committed(root, path)
    quota(w["runs"], int(time.time()) + 3600)
    monkeypatch.setattr(t, "dev_check", lambda *a, **k: {"n": 4, "top1_same": 0.5, "min": 0.99, "ok": False})
    args = test_args(root, w, path, tmp_path / "o", 5, "streetclip")
    args[args.index("--test-csv") + 1] = str(tmp_path / "absent.csv")  # reading it would raise another error
    with pytest.raises(SystemExit, match="NOT read"):
        t.run_test(t.parse_args(args))
    assert not (tmp_path / "o" / "k5_B_full_500k_streetclip" / "test_read_started.json").exists()


def test_refuses_a_stage_changed_after_the_prereg_commit(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("needs git")
    f = tmp_path / "src" / "geoguessr_ai" / "v4_test.py"
    reg = tmp_path / "configs" / "p.json"
    f.parent.mkdir(parents=True), reg.parent.mkdir()
    f.write_bytes(b"a = 1\n"), reg.write_bytes(b"{}\n")
    _git(tmp_path, "init", "-q", str(tmp_path))
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "prereg")
    commit, _ = prereg.git_state(reg)
    t.check_code_frozen(reg, commit)
    f.write_bytes(b"a = 2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "changed")
    with pytest.raises(SystemExit, match="changed since"):
        t.check_code_frozen(reg, commit)


def test_runtime_settings_are_cli_args_and_leave_the_scores_unchanged(base, tmp_path):
    root, w, path = base
    committed(root, path)
    quota(w["runs"], int(time.time()) + 3600)
    src = Path(t.__file__).read_text(encoding="utf-8").lower()
    assert not any(s in src for s in ("kaggle", "/working", "/temp", "daggerabuse", "hf_token", "huggingface"))
    bb, rid = "streetclip", "k4_B_full_50k_streetclip"
    hybrid = lambda out, *x: t.run_hybrid(t.parse_args(["hybrid", *common(root, w, path, out), "--job", "4",
                                                        "--backbone", bb, "--inputs", str(w["inputs"]), *x]))
    a, b = tmp_path / "default", tmp_path / "override"
    assert t.run_test(t.parse_args(test_args(root, w, path, a, 4, bb))) == 0
    assert t.run_test(t.parse_args([*test_args(root, w, path, b, 4, bb), "--device", "cpu", "--batch", "3",
                                    "--workers", "0"])) == 0
    assert json.loads((b / rid / "test_row.json").read_text())["runtime"] == {"device": "cpu", "batch": 3, "workers": 0}
    assert hybrid(a) == 0 and hybrid(b, "--knn-chunk", "3") == 0
    for f in ("test_geo_hits.csv", "hybrid_test_geo_hits.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(a / rid / f), pd.read_csv(b / rid / f))
