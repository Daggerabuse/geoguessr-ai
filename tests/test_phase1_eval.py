"""Phase 1 analysis runner on synthetic embeddings (no model, no Kaggle)."""

import json

import numpy as np
import pandas as pd
import pytest

from geoguessr_ai import embed as e
from geoguessr_ai import geo_eval as g
from geoguessr_ai import phase1_eval as pe
from tests.test_v2 import _meta, _write_data

NOISE = {"streetclip": 0.3, "clip_l14_336": 1.5, "clip_b16": 2.0}  # streetclip separates best


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import torch

    tr, te = _meta("t", sizes=(40, 30, 20)), _meta("x", sizes=(3, 3, 3))
    data = tmp_path / "v2"
    parts = _write_data(data, tr, te, sizes=[30, 90], dev_n=20, dev_seen_n=20)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 8))
    country = {"AA": 0, "BB": 1, "CC": 2}
    for bb, noise in NOISE.items():
        (tmp_path / "emb" / bb).mkdir(parents=True)
        for src, names in (("train", ("train", "dev", "dev_seen")), ("test", ("test",))):
            d = pd.concat([parts[n] for n in names])
            x = centers[d["country"].map(country)] + noise * rng.normal(size=(len(d), 8))
            e.save_npz(tmp_path / "emb" / bb / f"{src}_00_x.npz", d["id"].tolist(), x, failed=[])
        (tmp_path / "emb" / bb / "encoder.json").write_text(json.dumps({"model": f"m/{bb}"}))
    monkeypatch.setattr(g, "text_head", lambda model, classes, template:
                        torch.from_numpy(np.random.default_rng(1).normal(size=(len(classes), 8))).float())
    args = ["--data", str(data), "--emb-root", str(tmp_path / "emb"), "--out", str(tmp_path / "res"),
            "--sizes", "30", "90", "--k", "1", "5", "--probe-epochs", "3", "--n-boot", "50",
            "--device", ""]
    return tmp_path, args


def test_grid_results_gaps_plots_and_no_test(setup):
    root, args = setup
    table = pe.run(pe.parse_args(args))
    assert len(table) == 3 * 2 * 2 * 3  # backbone x ordering x N x (knn, probe x 2 inits)
    for col in ("dev_seen_top1", "dev_seen_top1_lo", "dev_unseen_top5_hi", "dev_unseen_tail_top1",
                "dev_seen_macro_top1", "dev_seen_macro_top1_seen_classes", "k", "lr", "epoch",
                "probe_init", "n_train"):
        assert col in table
    assert set(table["variant"]) == {"knn", "probe_random", "probe_zeroshot"}
    knn, probe = table[table["method"] == "knn"], table[table["method"] == "probe"]
    assert knn["k"].isin([1, 5]).all() and knn["lr"].isna().all()
    assert probe["lr"].isin([3e-4, 1e-3, 3e-3]).all() and probe["epoch"].between(0, 3).all()
    assert {"dev_seen_absent_class_top1", "dev_unseen_absent_class_n"} <= set(table.columns)
    assert (table["dev_seen_absent_class_n"] >= 0).all()
    assert not any(c.startswith("test") for c in table.columns)
    assert (table["dev_seen_top1_lo"] <= table["dev_seen_top1"]).all()
    res = root / "res"
    assert not list(res.rglob("test_*")) and not (root / "res_test").exists()
    gp = pd.read_csv(res / "gaps.csv")
    assert len(gp) == 2 * 2 * 3 * 2 and set(gp["split"]) == {"dev_seen", "dev_unseen"}
    assert set(gp["variant"]) == {"knn", "probe_random", "probe_zeroshot"}
    assert ((gp["gap_lo"] <= gp["gap"]) & (gp["gap"] <= gp["gap_hi"])).all()
    assert gp["gap"].mean() > 0  # the cleaner backbone wins on average
    assert (res / "scaling.png").stat().st_size > 0 and (res / "gap.png").stat().st_size > 0


def test_finished_cells_are_skipped_and_a_missing_one_is_redone(setup, monkeypatch):
    root, args = setup
    first = pe.run(pe.parse_args(args))
    (pe.cell_dir(root / "res", "clip_b16", "nat", 90, "probe", "random") / "row.json").unlink()
    calls = []
    real = g.run_cell

    def counting(ctx, strategy, n, head, *a):
        calls.append((strategy, n, head))
        return real(ctx, strategy, n, head, *a)

    monkeypatch.setattr(g, "run_cell", counting)
    loads = []
    real_load = g.load_data
    monkeypatch.setattr(g, "load_data", lambda *a: loads.append(a[1].name) or real_load(*a))
    again = pe.run(pe.parse_args(args))
    assert calls == [("nat", 90, "probe")] and loads == ["clip_b16"]  # other backbones not loaded
    pd.testing.assert_frame_equal(first.drop(columns="seconds"), again.drop(columns="seconds"),
                                  check_exact=False, atol=1e-12)


def test_eval_test_writes_to_a_separate_directory(setup):
    root, args = setup
    table = pe.run(pe.parse_args([*args, "--backbones", "streetclip", "--methods", "knn",
                                  "--eval-test"]))
    assert "test_top1" in table and (root / "res_test" / "results.csv").exists()
    assert not (root / "res").exists()


def test_probe_lr_and_epoch_are_chosen_on_dev_seen_only(setup, monkeypatch):
    root, args = setup
    lrs = [3e-4, 1e-3, 3e-3]
    calls = []

    def fake_probe(feats, labels, dev_feats, dev_labels, eval_feats, head_init, a, device):
        calls.append((a.probe_lr, len(dev_feats), list(dev_labels)))
        i = lrs.index(a.probe_lr)
        history = [{"epoch": ep, "val_top1": [0.2, 0.9, 0.5][i] - 0.1 * abs(ep - 2)}
                   for ep in range(4)]  # dev_seen favours 1e-3 at epoch 2
        # outputs: a constant class per lr, so dev_unseen would favour another lr if it were used
        logits = {s: np.tile(np.eye(len(head_init))[(i + 1) % len(head_init)], (len(x), 1))
                  for s, x in eval_feats.items()}
        return logits, history

    monkeypatch.setattr(g, "probe_logits", fake_probe)
    table = pe.run(pe.parse_args([*args, "--backbones", "streetclip", "--methods", "probe",
                                  "--probe-inits", "random", "--orderings", "bal",
                                  "--sizes", "90"]))
    assert table["lr"].tolist() == [1e-3] and table["epoch"].tolist() == [2]
    dev_seen = pd.read_csv(root / "v2" / "dev_seen.csv")
    assert [c[0] for c in calls] == lrs
    assert all(n == len(dev_seen) and lab == dev_seen["label"].tolist() for _, n, lab in calls)
    m = json.loads((pe.cell_dir(root / "res", "streetclip", "bal", 90, "probe", "random")
                    / "geo_metrics.json").read_text())
    assert m["best_lr"] == 1e-3 and set(m["lr_sweep_dev_seen"]) == {"0.0003", "0.001", "0.003"}


def test_fixed_class_macro_keeps_actual_accuracy_of_classes_absent_from_training(tmp_path):
    import argparse

    classes = ["AA", "BB", "CC"]
    df = pd.DataFrame({"id": list("abcd"), "country": ["AA", "AA", "BB", "CC"],
                       "label": [0, 0, 1, 2], "sequence": list("abcd")})
    scores = np.eye(3)[[0, 1, 1, 2]]  # a right, b wrong, c right, d right (CC predicted)
    present = df["label"].isin([0, 1]).to_numpy()  # CC not in training
    groups = {"all": np.ones(4, dtype=bool), "class_in_subset": present,
              "class_not_in_subset": ~present}
    r = g.score_split(tmp_path, "dev", df, scores, classes, None, groups,
                      argparse.Namespace(n_boot=20, seed=0), df["sequence"])
    assert r["macro_top1_fixed"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)  # CC: its real accuracy
    assert r["class_in_subset"]["macro_top1"] == pytest.approx((0.5 + 1.0) / 2)  # secondary
    assert r["class_not_in_subset"]["n"] == 1
    assert r["class_not_in_subset"]["country_top1"]["mean"] == 1.0  # the absent-class column


def test_paired_gap_and_creator_clusters():
    ids = [str(i) for i in range(40)]
    clusters = pd.Series([i // 4 for i in range(40)], index=ids)
    same = pd.DataFrame({"id": ids, "top1": [i % 2 for i in range(40)]})
    r = pe.paired_gap(same, same, clusters, 200, 0)
    assert r["gap"] == r["gap_lo"] == r["gap_hi"] == 0 and r["n_images"] == 40
    better = same.assign(top1=1)
    r = pe.paired_gap(better, same, clusters, 200, 0)
    assert r["gap"] == 0.5 and r["gap_lo"] <= 0.5 <= r["gap_hi"]
    r = pe.paired_gap(better, same.iloc[:20], clusters, 200, 0)  # only images both scored
    assert r["n_images"] == 20
    df = pd.DataFrame({"sequence": ["s1", "s2"], "creator_id": ["c", "c"]})
    assert list(g.clusters_of("dev", df)) == ["c", "c"]
    assert list(g.clusters_of("dev_seen", df)) == ["s1", "s2"]
