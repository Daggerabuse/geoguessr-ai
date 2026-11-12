"""Phase 1 analysis runner on synthetic embeddings (no model, no Kaggle)."""

import json

import numpy as np
import pandas as pd
import pytest

from geoguessr_ai import embed as e
from geoguessr_ai import geo_eval as g
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import prereg
from tests.test_v2 import _meta, _write_data

PREV = ["prev_lr", "prev_dev_seen_top1"]  # probe rows of a resumed cell only
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
                "probe_init", "n_train", "stopped_early"):
        assert col in table
    assert set(table["variant"]) == {"knn", "probe_random", "probe_zeroshot"}
    knn, probe = table[table["method"] == "knn"], table[table["method"] == "probe"]
    assert knn["k"].isin([1, 5]).all() and knn["lr"].isna().all()
    assert probe["lr"].isin(pe.PROBE_LRS).all() and probe["epoch"].between(0, 3).all()
    assert "grid_edge" in table and (table["cache_version"] == table["method"].map(pe.CACHE_VERSION)).all()
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
    sh = pd.read_csv(res / "shrink.csv")
    assert len(sh) == 2 * 3 * 2 and (sh["n_small"] == 30).all() and (sh["n_large"] == 90).all()
    for r in sh.itertuples():  # shrink = gap(30) - gap(90) of gaps.csv, same images
        gap = gp.set_index(["ordering", "variant", "split", "n"])["gap"]
        assert r.shrink == pytest.approx(gap[r.ordering, r.variant, r.split, 30]
                                         - gap[r.ordering, r.variant, r.split, 90])
        assert r.shrink_lo <= r.shrink <= r.shrink_hi


def test_finished_cells_are_skipped_and_a_missing_one_is_redone(setup, monkeypatch):
    root, args = setup
    first = pe.run(pe.parse_args(args))
    (pe.cell_dir(root / "res", "clip_b16", "nat", 90, "probe", "random") / "row.json").unlink()
    calls = []
    real = g.run_cell

    def counting(ctx, strategy, n, head, *a, **kw):
        calls.append((strategy, n, head))
        return real(ctx, strategy, n, head, *a, **kw)

    monkeypatch.setattr(g, "run_cell", counting)
    loads = []
    real_load = g.load_data
    monkeypatch.setattr(g, "load_data", lambda *a: loads.append(a[1].name) or real_load(*a))
    again = pe.run(pe.parse_args(args))
    assert calls == [("nat", 90, "probe")] and loads == ["clip_b16"]  # other backbones not loaded
    pd.testing.assert_frame_equal(first.drop(columns="seconds"), again.drop(columns="seconds"),
                                  check_exact=False, atol=1e-12)


def _git(repo, *a):
    import subprocess

    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                   check=True, capture_output=True)


@pytest.fixture
def final(setup):
    """A finished dev run, its committed pre-registration and the provenance of a test
    embedding run (the synthetic test caches sit in emb/ next to the train caches)."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("needs git")
    root, args = setup
    dev = pe.run(pe.parse_args(args))
    repo = root / "repo"
    path = repo / "configs" / "phase1_final.json"
    prereg.write(root / "res", path, pe.parse_args(args))
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "add", "configs/phase1_final.json")
    _git(repo, "commit", "-q", "-m", "pre-register")
    prereg.provenance(path, root / "emb" / "prereg_provenance.json")
    final_args = [*args, "--eval-test", "--prereg", str(path), "--test-emb-root", str(root / "emb")]
    return root, args, final_args, dev, path


def _tree(d):
    return {str(f.relative_to(d)): f.read_bytes() for f in sorted(d.rglob("*")) if f.is_file()}


def test_prereg_declares_primary_hypotheses(final):
    *_, path = final
    h = json.loads(path.read_text())["hypotheses"]
    ids = {p["id"] for p in h["primary"]}
    assert ids == {f"{x}/{v}/{o}" for x in ("H1", "H2") for o in ("nat", "bal")
                   for v in ("knn", "probe_random", "probe_zeroshot")}
    assert {p["n"] for p in h["primary"] if p["quantity"] == "gap"} == {30}  # smallest N
    assert {(p["n_small"], p["n_large"]) for p in h["primary"] if p["quantity"] == "shrink"} == {(30, 90)}
    assert "exploratory" in h["secondary"] and "Holm" in h["criterion"] + h["multiplicity"]
    assert h["alpha"] == 0.05 and h["n_boot"] >= 10_000
    # a variant without both backbones of the pair gets no hypothesis
    cells = [c for c in json.loads(path.read_text())["cells"]
             if not (c["variant"] == "knn" and c["backbone"] == "clip_l14_336")]
    assert not any("/knn/" in p["id"] for p in prereg.hypotheses(cells, pe.PAIR)["primary"])


def test_one_sided_bootstrap_p_value():
    assert pe.one_sided_p(np.array([0.1, 0.2, 0.3])) == pytest.approx(1 / 4)  # none <= 0: never 0
    assert pe.one_sided_p(np.array([-0.1, 0.0, 0.3, 0.4])) == pytest.approx(3 / 5)  # 0 counts
    assert pe.one_sided_p(np.full(9, -1.0)) == pytest.approx(1.0)
    means = g.bootstrap_means(np.r_[np.ones(30), np.zeros(10)], np.arange(40), 10_000, 0)
    assert len(means) == 10_000 and pe.one_sided_p(means) == pytest.approx(1 / 10_001)


def test_bootstrap_means_in_chunks_is_the_same_stream():
    v, c = np.random.default_rng(0).normal(size=60), np.arange(60) // 3
    np.testing.assert_allclose(g.bootstrap_means(v, c, 450, 1, chunk=7),
                               g.bootstrap_means(v, c, 450, 1, chunk=1000), rtol=1e-12, atol=1e-15)


def test_holm_ordering_and_adjustment():
    # sorted: 0.01*4=0.04, 0.02*3=0.06, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.04 0.06 0.06 0.06
    adj = pe.holm([0.03, 0.01, 0.04, 0.02])
    np.testing.assert_allclose(adj, [0.06, 0.04, 0.06, 0.06])  # returned in input order
    np.testing.assert_allclose(pe.holm([0.5, 0.9]), [1.0, 1.0])  # capped at 1
    np.testing.assert_allclose(pe.holm([0.001, 0.2, 1.0]), [0.003, 0.4, 1.0])
    # Holm step-down: the second hypothesis is tested at alpha/(m-1), not alpha/m
    p = [0.012, 0.024, 0.9]  # Bonferroni (x3) would reject only the first
    assert (pe.holm(p) < 0.05).tolist() == [True, True, False]


def test_primary_results_holm_verdicts(tmp_path):
    ids = [str(i) for i in range(200)]
    (tmp_path / "test.csv").write_text("id,sequence\n" + "".join(f"{i},s{i}\n" for i in ids))
    rng = np.random.default_rng(0)

    def cell(ordering, n, var, top1_a, top1_b):
        for bb, t in zip(pe.PAIR, (top1_a, top1_b)):
            d = tmp_path / bb / f"{ordering}_{n}_{var}"
            d.mkdir(parents=True)
            pd.DataFrame({"id": ids, "top1": t}).to_csv(d / "test_hits.csv", index=False)

    base = rng.integers(0, 2, 200)
    cell("nat", 50, "knn", np.maximum(base, rng.random(200) < 0.3), base)  # clearly better
    cell("nat", 500, "knn", base, base)  # no gap left: shrink = gap(50)
    cell("nat", 50, "probe_random", base, base)  # no gap
    hyp = {"n_boot": 10_000, "alpha": 0.05, "primary": [
        {"id": "H1/knn/nat", "quantity": "gap", "variant": "knn", "ordering": "nat", "n": 50, "claim": "a"},
        {"id": "H2/knn/nat", "quantity": "shrink", "variant": "knn", "ordering": "nat", "n_small": 50,
         "n_large": 500, "claim": "b"},
        {"id": "H1/probe_random/nat", "quantity": "gap", "variant": "probe_random", "ordering": "nat",
         "n": 50, "claim": "c"},
        {"id": "H1/knn/bal", "quantity": "gap", "variant": "knn", "ordering": "bal", "n": 50, "claim": "d"}]}
    prim = pe.primary_results(hyp, tmp_path, tmp_path, 0).set_index("id")
    assert prim["verdict"].to_dict() == {"H1/knn/nat": "supported", "H2/knn/nat": "supported",
                                         "H1/probe_random/nat": "not supported",
                                         "H1/knn/bal": "not evaluable"}
    assert prim.loc["H1/knn/bal", "p"] == 1.0 and np.isnan(prim.loc["H1/knn/bal", "ci_lo"])
    np.testing.assert_allclose(prim["p_holm"], pe.holm(prim["p"]))  # m = 4, not-evaluable counts
    assert prim.loc["H1/knn/nat", "p"] == pytest.approx(1 / 10_001)
    assert prim.loc["H1/knn/nat", "estimate"] == prim.loc["H2/knn/nat", "estimate"]
    assert prim.loc["H1/knn/nat", "ci_lo"] > 0 and prim.loc["H1/probe_random/nat", "ci_lo"] == 0



def test_final_eval_writes_to_a_separate_directory_and_leaves_dev_untouched(final, capsys):
    root, _, final_args, dev, _ = final
    before = _tree(root / "res")
    capsys.readouterr()
    table = pe.run(pe.parse_args(final_args))
    printed = capsys.readouterr().out
    i_prim, i_expl = printed.index("=== PRIMARY"), printed.index("=== EXPLORATORY")
    assert i_prim < i_expl and printed.index("max abs difference") > i_expl  # primary first
    prim = pd.read_csv(root / "res_test" / "primary.csv")
    assert len(prim) == 12 and set(prim["verdict"]) <= {"supported", "not supported"}
    assert {"estimate", "ci_lo", "ci_hi", "p", "p_holm"} <= set(prim.columns)
    assert (prim["p_holm"] >= prim["p"]).all() and (prim["p"] > 0).all()
    assert prim["id"].iloc[0] in printed[i_prim:i_expl]
    assert (table["analysis"] == "exploratory").all()
    gp, sh = (pd.read_csv(root / "res_test" / f) for f in ("gaps.csv", "shrink.csv"))
    assert (gp["analysis"] == "primary").sum() == 6 and (sh["analysis"] == "primary").sum() == 6
    assert set(gp.loc[gp["analysis"] == "primary", "split"]) == {"test"}
    assert _tree(root / "res") == before  # every dev file byte-identical
    res = root / "res_test"
    assert len(table) == len(dev) and "test_top1" in table and (res / "results.csv").exists()
    for col in ("test_macro_top1", "test_tail_top1", "test_reliable_top1", "test_seen_creator_top1",
                "test_unseen_creator_n", "test_top5_hi", "test_geoscore", "test_acc_25km"):
        assert col in table
    assert table.loc[table["method"] == "probe", "test_geoscore"].isna().all()  # country only
    assert table.loc[table["method"] == "knn", "test_median_km"].notna().all()
    assert "test" in set(pd.read_csv(res / "gaps.csv")["split"])
    assert "test" in set(pd.read_csv(res / "shrink.csv")["split"])
    # retrained with the frozen values: the pre-registered dev scores come back
    for s in ("dev_seen", "dev_unseen"):
        np.testing.assert_allclose(table[f"{s}_top1"], table[f"{s}_top1_prereg"], atol=1e-9)


def test_final_eval_uses_the_frozen_hyperparameters_only(final, monkeypatch):
    root, _args, final_args, _, path = final
    reg = json.loads(path.read_text())
    probe = [c for c in reg["cells"] if c["method"] == "probe"]
    probe[0]["lr"], probe[0]["epoch"] = 0.123, 2  # values no search would give: must be used
    path.write_text(json.dumps(reg))
    _git(path.parents[1], "commit", "-q", "-am", "edit")
    prereg.provenance(path, root / "emb" / "prereg_provenance.json")  # a new embedding run
    fits, ks = [], []
    real_fit, real_scores = g.probe_logits, g.knn_country_scores

    def fit(feats, labels, dev_feats, dev_labels, eval_feats, head_init, a, device):
        fits.append((a.probe_lr, a.probe_epochs, a.probe_restore_best, a.probe_patience))
        assert set(eval_feats) == {"dev_seen", "dev", "test"}
        return real_fit(feats, labels, dev_feats, dev_labels, eval_feats, head_init, a, device)

    monkeypatch.setattr(g, "probe_logits", fit)
    monkeypatch.setattr(g, "knn_country_scores", lambda sims, idx, lab, n, k: ks.append(k)
                        or real_scores(sims, idx, lab, n, k))
    table = pe.run(pe.parse_args(final_args))
    assert sorted(fits) == sorted((c["lr"], c["epoch"], False, 0) for c in probe)  # one fit each
    knn = [c for c in reg["cells"] if c["method"] == "knn"]
    assert sorted(ks) == sorted(c["k_country"] for c in knn for _ in range(3))  # 3 splits, no sweep
    row = table[(table["backbone"] == probe[0]["backbone"]) & (table["ordering"] == probe[0]["ordering"])
                & (table["n"] == probe[0]["n"]) & (table["variant"] == probe[0]["variant"])]
    assert row["lr"].tolist() == [0.123] and row["epoch"].tolist() == [2]


def test_final_eval_refuses_without_a_valid_preregistration(final):
    root, args, final_args, _, path = final
    with pytest.raises(SystemExit, match="needs --prereg"):
        pe.run(pe.parse_args([*args, "--eval-test", "--test-emb-root", str(root / "emb")]))
    with pytest.raises(SystemExit, match="cannot be used"):
        pe.run(pe.parse_args([*final_args, "--seed-from", str(root / "res")]))
    prov = root / "emb" / "prereg_provenance.json"
    rec = json.loads(prov.read_text())
    prov.write_text(json.dumps({**rec, "embedding_started": rec["prereg_commit_time"] - 1}))
    with pytest.raises(SystemExit, match="does not predate"):
        pe.run(pe.parse_args(final_args))
    prov.unlink()
    with pytest.raises(SystemExit, match="cannot be dated"):
        pe.run(pe.parse_args(final_args))
    prov.write_text(json.dumps(rec))
    path.write_text(path.read_text().replace('"lr": ', '"lr": 1e-9 + '))  # edited, not committed
    with pytest.raises(SystemExit, match="uncommitted"):
        pe.run(pe.parse_args(final_args))
    _git(path.parents[1], "commit", "-q", "-am", "edit after the embeddings")
    with pytest.raises(SystemExit, match="differs"):
        pe.run(pe.parse_args(final_args))
    with pytest.raises(SystemExit, match="missing"):
        prereg.git_state(root / "repo" / "configs" / "other.json")
    other = root / "repo" / "configs" / "other.json"
    other.write_text("{}")
    with pytest.raises(SystemExit, match="not committed"):
        prereg.git_state(other)
    assert not (root / "res_test").exists()  # nothing scored on test


def test_prereg_refuses_outdated_dev_rows(final):
    root, args, *_ = final
    results = root / "res" / "results.csv"
    t = pd.read_csv(results)
    t.loc[t["method"] == "probe", "cache_version"] = 1
    t.to_csv(results, index=False)
    with pytest.raises(SystemExit, match="cache version"):
        prereg.write(root / "res", root / "p.json", pe.parse_args(args))


def test_probe_lr_and_epoch_are_chosen_on_dev_seen_only(setup, monkeypatch):
    root, args = setup
    lrs = pe.PROBE_LRS
    calls = []

    def fake_probe(feats, labels, dev_feats, dev_labels, eval_feats, head_init, a, device):
        calls.append((a.probe_lr, len(dev_feats), list(dev_labels)))
        i = lrs.index(a.probe_lr)
        history = [{"epoch": ep, "val_top1": [0.2, 0.9, 0.5, 0.4, 0.3][i] - 0.1 * abs(ep - 2)}
                   for ep in range(4)]  # dev_seen favours 3e-3 at epoch 2
        # outputs: a constant class per lr, so dev_unseen would favour another lr if it were used
        logits = {s: np.tile(np.eye(len(head_init))[(i + 1) % len(head_init)], (len(x), 1))
                  for s, x in eval_feats.items()}
        return logits, history

    monkeypatch.setattr(g, "probe_logits", fake_probe)
    table = pe.run(pe.parse_args([*args, "--backbones", "streetclip", "--methods", "probe",
                                  "--probe-inits", "random", "--orderings", "bal",
                                  "--sizes", "90"]))
    assert table["lr"].tolist() == [3e-3] and table["epoch"].tolist() == [2]
    assert table["grid_edge"].tolist() == [""]
    dev_seen = pd.read_csv(root / "v2" / "dev_seen.csv")
    assert [c[0] for c in calls] == lrs
    assert all(n == len(dev_seen) and lab == dev_seen["label"].tolist() for _, n, lab in calls)
    m = json.loads((pe.cell_dir(root / "res", "streetclip", "bal", 90, "probe", "random")
                    / "geo_metrics.json").read_text())
    assert m["best_lr"] == 3e-3 and set(m["lr_sweep_dev_seen"]) == {"0.001", "0.003", "0.01", "0.03", "0.1"}


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


def _count_cells(monkeypatch):
    calls = []
    real = g.run_cell

    def counting(ctx, strategy, n, head, *a, **kw):
        calls.append(head)
        return real(ctx, strategy, n, head, *a, **kw)

    monkeypatch.setattr(g, "run_cell", counting)
    return calls


def _as_old_output(res):
    """Rows as the run before versioning wrote them (no cache_version: counts as 1)."""
    for f in res.rglob("row.json"):
        row = json.loads(f.read_text())
        row.pop("cache_version")
        f.write_text(json.dumps(row))


def test_cache_version_bump_reruns_only_that_method(setup, monkeypatch):
    root, args = setup
    first = pe.run(pe.parse_args(args))
    _as_old_output(root / "res")
    calls = _count_cells(monkeypatch)
    again = pe.run(pe.parse_args(args))
    assert calls.count("probe") == 3 * 2 * 2 * 2 and "knn" not in calls  # old probe rows: v1 < 2
    knn = lambda t: t[t["method"] == "knn"].drop(columns=["cache_version", *PREV], errors="ignore")
    pd.testing.assert_frame_equal(knn(first), knn(again))  # kNN rows untouched (same seconds)
    calls.clear()
    pe.run(pe.parse_args(args))
    assert calls == []  # all current now


def test_seed_from_copies_only_valid_cells(setup, monkeypatch):
    root, args = setup
    first = pe.run(pe.parse_args(args))
    _as_old_output(root / "res")
    (pe.cell_dir(root / "res", "clip_b16", "bal", 90, "knn") / "row.json").unlink()  # unfinished
    calls = _count_cells(monkeypatch)
    out2 = root / "res2"
    again = pe.run(pe.parse_args([*args, "--out", str(out2), "--seed-from", str(root / "res")]))
    assert calls.count("knn") == 1 and calls.count("probe") == 3 * 2 * 2 * 2
    src = pe.cell_dir(root / "res", "streetclip", "nat", 30, "knn")
    dst = pe.cell_dir(out2, "streetclip", "nat", 30, "knn")
    assert sorted(p.name for p in dst.iterdir()) == sorted(p.name for p in src.iterdir())
    assert (dst / "dev_hits.csv").read_bytes() == (src / "dev_hits.csv").read_bytes()
    knn = lambda t: t[t["method"] == "knn"].drop(columns=["seconds", "cache_version", *PREV],
                                                 errors="ignore").reset_index(drop=True)
    pd.testing.assert_frame_equal(knn(first), knn(again), check_exact=False, atol=1e-12)
    assert pe.seed_cells(root / "res", out2, "streetclip", [("nat", 30, "knn", None)]) == 0  # done


def test_shrink_is_paired_over_the_same_images():
    ids = [str(i) for i in range(40)]
    clusters = pd.Series([i // 4 for i in range(40)], index=ids)
    d_small = pd.Series([1.0 if i % 2 else 0.0 for i in range(40)], index=ids)
    r = pe.shrink(d_small, d_small, clusters, 200, 0)
    assert r["shrink"] == r["shrink_lo"] == r["shrink_hi"] == 0  # identical gaps: no width
    r = pe.shrink(d_small, pd.Series(0.0, index=ids[:30]), clusters, 200, 0)
    assert r["n_images"] == 30 and r["gap_large"] == 0
    assert r["shrink"] == pytest.approx(d_small[:30].mean()) and r["shrink_lo"] > 0


def test_grid_edge_flags():
    import argparse

    a = argparse.Namespace(k=[1, 5, 10], probe_lrs=[1e-3, 1e-2], probe_epochs=60)
    assert pe.grid_edge({"method": "knn", "k": 10}, a) == "k_max"
    assert pe.grid_edge({"method": "knn", "k": 5}, a) == ""
    probe = lambda lr, ep, stopped: pe.grid_edge(
        {"method": "probe", "lr": lr, "epoch": ep, "stopped_early": stopped}, a)
    assert probe(1e-2, 60, False) == "lr_high;last_epoch"
    assert probe(1e-3, 12, False) == "lr_low"
    assert probe(3e-3, 55, False) == "last_epoch"  # final 10% of a full-length run
    assert probe(3e-3, 54, False) == ""
    assert probe(3e-3, 58, True) == ""  # stopped early: the epochs ran out of gains, not of budget
    assert probe(3e-3, 58, np.nan) == "last_epoch"  # rows written before early stopping


def test_probe_early_stopping_keeps_the_best_epoch(monkeypatch):
    import argparse

    pytest.importorskip("torch")
    import torch
    from torch import nn

    from geoguessr_ai import train as tr

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Parameter(torch.zeros(2, 4))

        def logits_from_embed(self, e):
            return e @ self.head.T

    def run(curve, patience):
        accs = iter(curve)
        monkeypatch.setattr(tr, "top1", lambda logits, labels: next(accs))
        model, history, x = Head(), [], torch.randn(8, 4)
        args = argparse.Namespace(probe_lr=0.1, probe_epochs=len(curve) - 1, seed=0,
                                  label_smoothing=0.0, probe_patience=patience)
        tr.train_probe(model, x, np.array([0, 1] * 4), x, np.array([0, 1] * 4), args, "cpu", history)
        best = max(history, key=lambda h: h["val_top1"])["epoch"]  # what geo_eval reports
        return history[-1]["epoch"], best

    # epoch: 0    1    2    3    4    5    6    7    8    9    10
    curve = [0.1, 0.5, 0.6, 0.6, 0.5, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6]
    assert run(curve, 3) == (8, 5)  # 0.7 at epoch 5 resets the count; 6, 7, 8 without gain
    assert run(curve, 2) == (4, 2)  # epochs 3 (a tie is not a gain) and 4: stop before the 0.7
    assert run(curve, 0) == (10, 5)  # 0 = off: all epochs


ONE = ["--backbones", "streetclip", "--methods", "probe", "--probe-inits", "random",
       "--orderings", "bal", "--sizes", "90"]
OLD_LRS = [1e-3, 3e-3, 1e-2, 3e-2]


def _fake_probe(monkeypatch, vals):
    """probe_logits with a dev_seen curve peaking at epoch 2 at vals[lr]; predicts class 2 at
    lr 0.1 and class 1 otherwise."""
    calls = []

    def fake(feats, labels, dev_feats, dev_labels, eval_feats, head_init, a, device):
        calls.append(a.probe_lr)
        history = [{"epoch": ep, "val_top1": vals[a.probe_lr] - 0.1 * abs(ep - 2)} for ep in range(4)]
        row = np.eye(len(head_init))[2 if a.probe_lr == 0.1 else 1]
        return {s: np.tile(row, (len(x), 1)) for s, x in eval_feats.items()}, history

    monkeypatch.setattr(g, "probe_logits", fake)
    return calls


def _old_cell(root, args, monkeypatch, vals, legacy=True):
    """One probe cell run on the 4-lr grid, then made a cache version 3 cell (legacy: its
    sweep entries without settings, as version 3 wrote them)."""
    calls = _fake_probe(monkeypatch, vals)
    pe.run(pe.parse_args([*args, *ONE, "--probe-lrs", *map(str, OLD_LRS)]))
    d = pe.cell_dir(root / "res", "streetclip", "bal", 90, "probe", "random")
    m = json.loads((d / "geo_metrics.json").read_text())
    if legacy:
        settings = m["lr_sweep_dev_seen"]["0.003"]["settings"]
        monkeypatch.setattr(pe, "LEGACY_PROBE_SETTINGS", {3: settings})
        for e in m["lr_sweep_dev_seen"].values():
            del e["settings"]
        (d / "geo_metrics.json").write_text(json.dumps(m, indent=2))
    row = json.loads((d / "row.json").read_text())
    (d / "row.json").write_text(json.dumps({**row, "cache_version": 3}))
    calls.clear()
    return d, calls


def test_partial_sweep_resume_runs_only_the_missing_lr(setup, monkeypatch, capsys):
    root, args = setup
    vals = {1e-3: 0.2, 3e-3: 0.9, 1e-2: 0.5, 3e-2: 0.4, 1e-1: 0.3}
    d, calls = _old_cell(root, args, monkeypatch, vals)
    old = json.loads((d / "geo_metrics.json").read_text())["lr_sweep_dev_seen"]
    hits = {f: (d / f).read_bytes() for f in ("dev_seen_hits.csv", "dev_hits.csv")}
    assert pe.partial(d, "probe")
    table = pe.run(pe.parse_args([*args, *ONE]))
    assert calls == [0.1]  # only the missing lr is trained
    m = json.loads((d / "geo_metrics.json").read_text())
    for k, entry in old.items():  # stored entries reused as they were (plus their settings)
        assert json.dumps({f: m["lr_sweep_dev_seen"][k][f] for f in entry}) == json.dumps(entry)
    assert m["reused_lrs"] == OLD_LRS and "settings" in m["lr_sweep_dev_seen"]["0.1"]
    assert {f: (d / f).read_bytes() for f in hits} == hits  # the stored winner's scores stand
    r = table.iloc[0]
    assert r["lr"] == 3e-3 and r["prev_lr"] == 3e-3 and r["cache_version"] == 4
    assert "lr changes: 0 of 1 resumed probe cells" in capsys.readouterr().out
    pe.run(pe.parse_args([*args, *ONE]))
    assert calls == [0.1]  # done now


def test_resume_reselects_over_the_merged_sweep(setup, monkeypatch, capsys):
    root, args = setup
    vals = {1e-3: 0.2, 3e-3: 0.9, 1e-2: 0.5, 3e-2: 0.4, 1e-1: 0.95}
    d, calls = _old_cell(root, args, monkeypatch, vals)
    out2 = root / "res2"  # resumed in a fresh output seeded with the partial cell
    capsys.readouterr()
    table = pe.run(pe.parse_args([*args, *ONE, "--out", str(out2), "--seed-from", str(root / "res")]))
    assert calls == [0.1]
    r = table.iloc[0]
    assert r["lr"] == 0.1 and r["epoch"] == 2 and r["grid_edge"] == "lr_high"
    assert r["dev_seen_top1"] != r["prev_dev_seen_top1"]  # rescored with the new winner
    d2 = pe.cell_dir(out2, "streetclip", "bal", 90, "probe", "random")
    assert (d2 / "dev_hits.csv").read_bytes() != (d / "dev_hits.csv").read_bytes()
    out = capsys.readouterr().out
    assert "lr changes: 1 of 1 resumed probe cells" in out and "lr 0.003 -> 0.1" in out


def test_resume_reruns_lrs_whose_settings_differ(setup, monkeypatch):
    root, args = setup
    vals = {1e-3: 0.2, 3e-3: 0.9, 1e-2: 0.5, 3e-2: 0.4, 1e-1: 0.3}
    d, calls = _old_cell(root, args, monkeypatch, vals, legacy=False)
    m = json.loads((d / "geo_metrics.json").read_text())
    sw = m["lr_sweep_dev_seen"]
    sw["0.01"]["settings"]["seed"] = 7
    sw["0.03"]["settings"]["probe_epochs"] = 30
    del sw["0.001"]["settings"]  # legacy entry whose history disagrees with version 3
    sw["0.001"]["history"] = sw["0.001"]["history"][:-1]
    monkeypatch.setattr(pe, "LEGACY_PROBE_SETTINGS", {3: sw["0.003"]["settings"]})
    m["best_lr"] = 0.01  # stored choice not the merged winner: the reused 3e-3 is refit
    (d / "geo_metrics.json").write_text(json.dumps(m))
    table = pe.run(pe.parse_args([*args, *ONE]))
    assert calls == [1e-3, 1e-2, 3e-2, 0.1, 3e-3]
    assert json.loads((d / "geo_metrics.json").read_text())["reused_lrs"] == [3e-3]
    assert table.iloc[0]["lr"] == 3e-3


def test_history_agrees_with_settings():
    s = {"probe_epochs": 10, "probe_patience": 3}
    h = lambda vals: [{"epoch": i, "val_top1": v} for i, v in enumerate(vals)]
    assert pe.history_agrees({"epoch": 2, "stopped_early": True, "history": h([0, 1, 5, 4, 4, 4])}, s)
    assert not pe.history_agrees({"epoch": 2, "stopped_early": True, "history": h([0, 1, 5, 4, 4])}, s)
    full = h([0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 8])
    assert pe.history_agrees({"epoch": 8, "stopped_early": False, "history": full}, s)
    assert not pe.history_agrees({"epoch": 7, "stopped_early": False, "history": full}, s)
    assert not pe.history_agrees({"epoch": 8, "stopped_early": True, "history": full}, s)
