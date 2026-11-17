"""Phase 3 pre-registered test evaluation on synthetic embeddings (no model, no Kaggle)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from geoguessr_ai import geo_eval as g
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_eval as p3
from geoguessr_ai import phase3_final as f
from geoguessr_ai import prereg
from tests.test_phase1_eval import _git, setup  # noqa: F401  (fixture)
from tests.test_phase3_eval import p3_args


@pytest.fixture
def frozen(setup, monkeypatch):  # noqa: F811
    """A finished dev run (kNN, cell probes, hybrids) and its committed pre-registration."""
    if shutil.which("git") is None:
        pytest.skip("needs git")
    root, _ = setup
    monkeypatch.setattr(p3, "SELECT_N", 90)
    monkeypatch.setattr(f, "CHUNK", 4)  # several query blocks on the tiny splits
    p3.run(p3_args(root, "--methods", "knn", "cell", "hybrid", "--cell-targets", "4", "8",
                   "--probe-epochs", "3", "--probe-lrs", "0.01", "0.1", "--mask-km", "20"))
    path = root / "repo" / "configs" / "phase3_final.json"
    reg = f.write(root / "p3", root / "p3", path, root / "p3" / "results.csv", sizes=(30, 90))
    _git(root, "init", "-q", str(root / "repo"))
    _git(root / "repo", "add", "configs/phase3_final.json")
    _git(root / "repo", "commit", "-q", "-m", "pre-register")
    common = ["--data", str(root / "v2"), "--emb-root", str(root / "emb"), "--dev-run",
              str(root / "p3"), "--prereg", str(path)]
    probes = ["probes", *common, "--out", str(root / "probes")]
    test = ["test", *common, "--out", str(root / "final"), "--probes", str(root / "probes"),
            "--test-emb-root", str(root / "emb")]
    return root, reg, path, probes, test


def test_prereg_freezes_the_dev_selection_and_five_hypotheses(frozen):
    root, reg, *_ = frozen
    assert len(reg["cells"]) == 2 * 2 * 2 * 5 and {c["backbone"] for c in reg["cells"]} == set(pe.PAIR)
    for c in reg["cells"]:
        row = pe.read_row(pe.cell_dir(root / "p3", c["backbone"], c["ordering"], c["n"],
                                      p3.METHOD, c["predictor"]))
        assert c["dev_seen_geoscore"] == row["dev_seen_geoscore"]
        assert all(c[k] == row[k] for k in ("lr", "epoch", "m", "k") if k in c)
    hyp = reg["hypotheses"]
    assert [h["id"].split("/")[0] for h in hyp["primary"]] == [f"H{i}" for i in range(1, 6)]
    assert [h["id"] for h in hyp["exploratory"]] == ["E1/hybrid_vs_knn/nat"]  # no verdict
    assert hyp["n_boot"] == 10_000 and hyp["alpha"] == 0.05 and "creator" in hyp["clusters"]
    assert all(abs(sum(t["w"] for t in h["terms"])) < 1e-12 for h in hyp["primary"])  # contrasts
    assert reg["dev_results_sha256"] == prereg.sha256(root / "p3" / "results.csv")
    assert reg["reproduction"]["top1_cell_same_min"] == 0.99


def test_probes_reproduce_then_test_is_scored_once(frozen, monkeypatch):
    root, reg, path, probes, test = frozen
    loads = []
    real = g.load_data
    monkeypatch.setattr(g, "load_data", lambda d, e, t, te=None: loads.append(t) or real(d, e, t, te))
    assert f.run_probes(f.parse_args(probes))
    assert loads and not any(loads)  # the probes stage never loads test
    rep = json.loads((root / "probes" / "repro.json").read_text())
    assert rep["ok"] and rep["declared"] == rep["done"] == 16
    assert all(c["top1_cell_same"] == 1.0 and abs(c["diff"]) < 1e-6 for c in rep["cells"])  # CPU
    n = len(loads)
    assert f.run_probes(f.parse_args(probes)) and len(loads) == n  # cached: nothing retrained

    assert f.run_test(f.parse_args(test))
    assert loads[n:] == [False, False, True, True]  # both dev checks before any test load
    one = [*test, "--backbones", "clip_l14_336"]  # a one-backbone session still checks both
    one[one.index("--out") + 1] = str(root / "one")
    assert not f.run_test(f.parse_args(one))  # scored, but no primary analysis on half the cells
    assert loads[n + 4:] == [False, False, True]
    assert (root / "one" / "streetclip" / "dev_check.json").exists()
    out = root / "final"
    checks = [c for bb in pe.PAIR
              for c in json.loads((out / bb / "dev_check.json").read_text())["cells"]]
    assert len(checks) == 40 and all(c["ok"] and abs(c["diff"]) < 1e-6 for c in checks)
    res = pd.read_csv(out / "primary.csv")
    assert list(res["id"]) == [h["id"] for h in reg["hypotheses"]["primary"]]
    assert set(res["verdict"]) <= {"supported", "not supported"}
    assert (res["p_holm"] >= res["p"]).all() and (res["p"] >= 1 / 10_001).all()
    assert ((res["ci_lo"] <= res["estimate"]) & (res["estimate"] <= res["ci_hi"])).all()
    h1 = reg["hypotheses"]["primary"][0]["terms"]
    a, b = (pd.read_csv(pe.cell_dir(out, t["backbone"], t["ordering"], t["n"], p3.METHOD,
                                    t["predictor"]) / "test_geo_hits.csv")["geoscore"] for t in h1)
    assert res["estimate"][0] == pytest.approx((a - b).mean())
    table = pd.read_csv(out / "results.csv")
    assert len(table) == 40 and table["test_n"].eq(len(pd.read_csv(root / "v2" / "test.csv"))).all()
    assert len(pd.read_csv(out / "exploratory_gaps.csv")) == 2 * 2 * 5
    e1 = pd.read_csv(out / "exploratory_contrasts.csv")
    assert list(e1["id"]) == ["E1/hybrid_vs_knn/nat"] and "p" not in e1 and "verdict" not in e1
    assert table["run"].nunique() == 1 and (res["largest_creator_share"] > 0).all()
    row = pe.cell_dir(out, "streetclip", "nat", 30, p3.METHOD, "medoid") / "test_row.json"
    row.write_text(json.dumps({**json.loads(row.read_text()), "run": {"prereg_sha256": "x"}}))
    with pytest.raises(SystemExit, match="runs"):  # cells of another run are not merged
        f.run_primary(f.parse_args(["primary", "--data", str(root / "v2"), "--out", str(out),
                                    "--prereg", str(path)]))


def test_a_failed_reproduction_never_loads_test(frozen, monkeypatch):
    root, _, _, probes, test = frozen
    assert f.run_probes(f.parse_args(probes))
    head = pe.cell_dir(root / "probes", "streetclip", "nat", 90, p3.METHOD, "cell_random") / "head.npy"
    np.save(head, np.random.default_rng(0).normal(size=np.load(head).shape).astype(np.float32))
    loads = []
    real = g.load_data
    monkeypatch.setattr(g, "load_data", lambda d, e, t, te=None: loads.append(t) or real(d, e, t, te))
    with pytest.raises(SystemExit, match="not the head recorded"):
        f.run_test(f.parse_args(test))
    rep = json.loads((root / "probes" / "repro.json").read_text())  # even a recorded bad head
    cell = next(c for c in rep["cells"] if (c["backbone"], c["ordering"], c["n"], c["predictor"])
                == ("streetclip", "nat", 90, "cell_random"))
    cell["head_sha256"] = prereg.sha256(head)
    (root / "probes" / "repro.json").write_text(json.dumps(rep))
    with pytest.raises(SystemExit, match="NOT read"):
        f.run_test(f.parse_args(test))
    assert not any(loads) and not list((root / "final").rglob("test_*"))


def test_refuses_a_changed_prereg_or_dev_file_or_a_failed_probe_check(frozen):
    root, _, path, probes, test = frozen
    with pytest.raises(SystemExit, match="probes stage first"):  # test is not scored
        f.run_test(f.parse_args(test))
    stored = pe.cell_dir(root / "p3", "streetclip", "nat", 30, p3.METHOD, "cell_random") \
        / "dev_seen_top_cells.csv"
    text = stored.read_text()
    stored.write_text(text + "\n")
    with pytest.raises(SystemExit, match="not the pre-registered dev file"):
        f.run_probes(f.parse_args(probes))
    stored.write_text(text)
    assert f.run_probes(f.parse_args(probes))
    rep = root / "probes" / "repro.json"
    rep.write_text(json.dumps({**json.loads(rep.read_text()), "ok": False}))
    with pytest.raises(SystemExit, match="did not pass"):
        f.run_test(f.parse_args(test))
    path.write_text(path.read_text().replace('"alpha": 0.05', '"alpha": 0.2'))
    with pytest.raises(SystemExit, match="uncommitted changes"):
        f.run_probes(f.parse_args(probes))
