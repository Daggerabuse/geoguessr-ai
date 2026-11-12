"""Phase 3 coordinate runner on synthetic embeddings (no model, no Kaggle)."""

import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from geoguessr_ai import embed as e
from geoguessr_ai import geo_eval as g
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_eval as p3
from tests.test_phase1_eval import setup  # noqa: F401  (fixture)


def p3_args(tmp_path, *extra):
    return p3.parse_args(["--data", str(tmp_path / "v2"), "--emb-root", str(tmp_path / "emb"),
                          "--out", str(tmp_path / "p3"), "--sizes", "30", "90", "--k", "1", "5",
                          "--n-boot", "50", "--device", "", *extra])


# ----------------------------------------------------------------- predictors


def test_spherical_mean_across_the_antimeridian():
    sims, idx = np.array([[0.9, 0.9]]), np.array([[0, 1]])
    lat, lon = g.knn_mean_coords(sims, idx, np.array([10.0, 10.0]), np.array([179.0, -179.0]), 2)
    assert abs(abs(lon[0]) - 180) < 1e-6 and lat[0] == pytest.approx(10.0, abs=0.01)


def test_mean_of_one_point_is_that_point_and_xyz_round_trip():
    lat, lon = g.knn_mean_coords(np.array([[0.7]]), np.array([[0]]), np.array([-33.9]),
                                 np.array([151.2]), 1)
    assert (lat[0], lon[0]) == pytest.approx((-33.9, 151.2))
    la, lo = g.from_xyz(g.to_xyz(np.array([89.0, -45.0]), np.array([-170.0, 30.0])))
    assert la == pytest.approx([89.0, -45.0]) and lo == pytest.approx([-170.0, 30.0])


def test_antipodal_mean_falls_back_to_the_nearest_neighbour():
    lat, lon = g.knn_mean_coords(np.array([[0.8, 0.8]]), np.array([[1, 0]]), np.array([0.0, 0.0]),
                                 np.array([0.0, 180.0]), 2)
    assert (lat[0], lon[0]) == pytest.approx((0.0, 180.0))


def test_small_tau_gives_the_nearest_neighbour():
    db_lat, db_lon = np.array([40.0, -20.0, 5.0]), np.array([10.0, 60.0, -70.0])
    sims, idx = np.array([[0.90, 0.85, 0.80]]), np.array([[2, 0, 1]])
    lat, lon = g.knn_mean_coords(sims, idx, db_lat, db_lon, 3, tau=1e-4)
    assert (lat[0], lon[0]) == pytest.approx((5.0, -70.0))
    raw = g.knn_mean_coords(sims, idx, db_lat, db_lon, 3)  # cosine weights: a real average
    assert abs(raw[1][0] - (-70.0)) > 1


def test_nearest_label_is_the_closest_point_and_a_db_point_keeps_its_own():
    db_lat, db_lon = np.array([0.0, 0.0, 0.001]), np.array([0.0, 90.0, 0.0])
    lab = np.array([3, 4, 5])
    out = g.nearest_label(np.array([1.0, 0.0, 0.0, 0.001]), np.array([80.0, 89.0, 0.0, 0.0]),
                          db_lat, db_lon, lab)
    assert out.tolist() == [4, 4, 3, 5]


# ----------------------------------------------------------------- runner


def test_grid_outputs_selection_on_dev_seen_and_no_test(setup, monkeypatch):  # noqa: F811
    tmp_path, _ = setup
    read, sources = [], []
    real_read, real_load = pd.read_csv, e.load_embeddings
    monkeypatch.setattr(pd, "read_csv", lambda f, *a, **k: read.append(str(f)) or real_read(f, *a, **k))
    monkeypatch.setattr(e, "load_embeddings",
                        lambda emb, ids, source: sources.append(source) or real_load(emb, ids, source))
    chosen_on = []
    real_choose = p3.choose
    monkeypatch.setattr(p3, "choose", lambda pred, nb, sel, *a: chosen_on.append(len(sel))
                        or real_choose(pred, nb, sel, *a))
    table = p3.run(p3_args(tmp_path))
    out = tmp_path / "p3"
    assert len(table) == 3 * 2 * 2 * 7  # backbones x orderings x sizes x (predictors + baselines)
    assert set(table["variant"]) == {f"geo_{p}" for p in (*p3.PREDICTORS, *p3.BASELINES)}
    assert not any(c.startswith("test_") for c in table.columns)
    dev_seen_n = len(real_read(tmp_path / "v2" / "dev_seen.csv"))
    assert set(chosen_on) == {dev_seen_n}
    assert not any(f.endswith("test.csv") for f in read) and "test" not in sources
    mean = table[table["predictor"] == "mean"]
    assert set(mean["k"]) <= {1, 5}
    assert (table[table["predictor"] == "top1"]["k"] == 1).all()
    for m in p3.GAP_METRICS:
        gp = pd.read_csv(out / f"gaps_{m}.csv")
        assert set(gp["split"]) == {"dev_seen", "dev_unseen"} and len(gp) == 2 * 2 * 5 * 2
        assert (out / f"shrink_{m}.csv").exists()
    mg = pd.read_csv(out / "gaps_median_km.csv")
    assert ((mg["gap_lo"] <= mg["gap"] + 1e-9) & (mg["gap"] <= mg["gap_hi"] + 1e-9)).all()
    assert (out / "scaling_geo.png").exists() and (out / "gap_geo.png").exists()
    hits = pd.read_csv(out / "streetclip" / "nat_30_geo_mean" / "dev_seen_geo_hits.csv")
    assert {"km", "geoscore", "acc_25km", "country_top1", "pred_lat"} <= set(hits.columns)
    sweep = json.loads((out / "streetclip" / "nat_30_geo_mean" / "geo_metrics.json").read_text())
    assert len(sweep["sweep_dev_seen"]) == 2 * len(p3.TAUS)


def test_medoid_reproduces_phase1_knn_coordinates(setup, capsys):  # noqa: F811
    tmp_path, args = setup
    pe.run(pe.parse_args([*args, "--methods", "knn"]))
    table = p3.run(p3_args(tmp_path, "--phase1-dev", str(tmp_path / "res"), "--mask-km", "0"))
    assert "max abs difference 0.000000" in capsys.readouterr().out
    med = table[table["predictor"] == "medoid"]
    for _, r in med.iterrows():
        old = pe.read_row(tmp_path / "res" / r["backbone"] / f"{r['ordering']}_{r['n']}_knn")
        assert r["dev_seen_geoscore"] == pytest.approx(old["dev_seen_geoscore"], abs=1e-12)
        assert r["dev_unseen_median_km"] == pytest.approx(old["dev_unseen_median_km"], abs=1e-9)


def test_compare_refuses_a_test_results_dir(tmp_path):
    d = tmp_path / "streetclip" / "nat_30_knn"
    d.mkdir(parents=True)
    (d / "row.json").write_text(json.dumps({"test_geoscore": 1.0, "dev_seen_geoscore": 1.0}))
    table = pd.DataFrame([{"predictor": "medoid", "backbone": "streetclip", "ordering": "nat",
                           "n": 30}])
    with pytest.raises(ValueError, match="test results"):
        p3.compare_phase1(table, tmp_path)


def test_finished_cells_are_skipped_and_a_version_bump_reruns(setup, monkeypatch):  # noqa: F811
    tmp_path, _ = setup
    p3.run(p3_args(tmp_path))
    calls = []
    real = p3.run_subset
    monkeypatch.setattr(p3, "run_subset", lambda *a: calls.append(a[1:3]) or real(*a))
    p3.run(p3_args(tmp_path))
    assert calls == []
    monkeypatch.setattr(p3, "CACHE_VERSION", p3.CACHE_VERSION + 1)
    p3.run(p3_args(tmp_path, "--backbones", "streetclip"))
    assert sorted(calls) == [("bal", 30), ("bal", 90), ("nat", 30), ("nat", 90)]


# ----------------------------------------------------------------- paired statistics


def test_gaps_defaults_unchanged_and_median_gap_is_paired(tmp_path):
    ids = [f"i{j}" for j in range(40)]
    clusters = pd.Series([f"s{j // 2}" for j in range(40)], index=ids)
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"id": ids, "top1": rng.integers(0, 2, 40), "km": rng.uniform(1, 500, 40)})
    b = a.assign(top1=rng.integers(0, 2, 40), km=a["km"] + 10.0)
    assert pe.paired_gap(a, b, clusters, 100, 1) == pe.paired_gap(a, b, clusters, 100, 1, "top1")
    r = p3.median_gap(a, b, clusters, 200, 1)
    assert r["gap"] == pytest.approx(-10.0)
    assert r["gap_lo"] == pytest.approx(-10.0) and r["gap_hi"] == pytest.approx(-10.0)
    km = pe.paired_gap(a, b, clusters, 100, 1, "km")
    assert km["gap"] == pytest.approx(-10.0) and km["gap_lo"] == pytest.approx(-10.0)


# ----------------------------------------------------------------- masking, numerics, baselines


def test_mask_never_retrieves_an_identical_location_image():
    rng = np.random.default_rng(0)
    db = rng.normal(size=(50, 8)).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    q = db[[0, 1]].copy()  # same image as db 0 / db 1 (embedding and location)
    dbm = pd.DataFrame({"id": [f"d{i}" for i in range(50)], "latitude": rng.uniform(-60, 60, 50),
                        "longitude": rng.uniform(-170, 170, 50)})
    dbm.loc[2, ["latitude", "longitude"]] = dbm.loc[0, ["latitude", "longitude"]] + [0.005, 0]  # ~0.6 km
    qm = dbm.iloc[[0, 1]].assign(id=["q0", "q1"]).reset_index(drop=True)
    for dtype in ("fp32", "fp16"):
        qi, di, near = p3.geo_mask(qm, dbm, 1.0)
        _, idx = g.knn(q, db, 10, dtype=dtype, exclude=(qi, di))
        assert 0 not in idx[0] and 2 not in idx[0] and 1 not in idx[1]
        assert sorted(zip(qi.tolist(), di.tolist())) == [(0, 0), (0, 2), (1, 1)]
        assert near[0] not in (0, 2) and near[1] != 1
    qi, di, _ = p3.geo_mask(qm, dbm, 0.0)
    assert len(qi) == 0 and g.knn(q, db, 1, exclude=(qi, di))[1][:, 0].tolist() == [0, 1]


def test_numpy_fp16_rounds_scores_like_the_gpu_path():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    db = np.array([[0.99995, 0.01], [0.99990, 0.0141]], dtype=np.float32)
    s32 = g.knn(q, db, 2, dtype="fp32")[0]
    s16 = g.knn(q, db, 2, dtype="fp16")[0]
    assert s32[0, 0] != s32[0, 1] and s16[0, 0] == s16[0, 1]  # fp16 cannot tell them apart


def test_cache_key_covers_data_embeddings_and_settings(setup):  # noqa: F811
    tmp_path, _ = setup
    data, emb = tmp_path / "v2", tmp_path / "emb" / "streetclip"
    base = p3.cache_key(p3.cache_settings(p3_args(tmp_path), data, emb))
    for extra in (["--mask-km", "2"], ["--knn-dtype", "fp16"], ["--k", "1", "10"],
                  ["--seed", "1"], ["--phase1-dev", "x"]):
        assert p3.cache_key(p3.cache_settings(p3_args(tmp_path, *extra), data, emb)) != base
    (data / "config.json").write_text("{}")
    with_cfg = p3.cache_key(p3.cache_settings(p3_args(tmp_path), data, emb))
    assert with_cfg != base
    e.save_npz(emb / "train_01_y.npz", ["zz"], np.ones((1, 8)), failed=[])
    assert p3.cache_key(p3.cache_settings(p3_args(tmp_path), data, emb)) != with_cfg


def test_changed_mask_reruns_every_subset(setup, monkeypatch):  # noqa: F811
    tmp_path, _ = setup
    p3.run(p3_args(tmp_path, "--backbones", "streetclip"))
    calls = []
    real = p3.run_subset
    monkeypatch.setattr(p3, "run_subset", lambda *a: calls.append(a[1:3]) or real(*a))
    p3.run(p3_args(tmp_path, "--backbones", "streetclip"))
    assert calls == []
    table = p3.run(p3_args(tmp_path, "--backbones", "streetclip", "--mask-km", "200"))
    assert len(calls) == 4
    assert (table["mask_km"] == 200).all() and (table["knn_dtype"] == "fp32").all()
    assert (table["knn_device"] == "cpu").all() and (table["dev_seen_masked_mean"] > 0).all()


def test_baselines(setup):  # noqa: F811
    tmp_path, _ = setup
    res = tmp_path / "res"
    for bb in pe.BACKBONES:  # a fake Phase 1 probe: always predicts the true country
        for o in ("nat", "bal"):
            for n in (30, 90):
                d = res / bb / f"{o}_{n}_{p3.PROBE_VARIANT}"
                d.mkdir(parents=True)
                for s in ("dev_seen", "dev"):
                    df = pd.read_csv(tmp_path / "v2" / f"{s}.csv", dtype={"id": str},
                                     keep_default_na=False)
                    df[["id", "country"]].rename(columns={"country": "pred"}) \
                        .to_csv(d / f"{s}_predictions.csv", index=False)
    table = p3.run(p3_args(tmp_path, "--phase1-dev", str(res), "--mask-km", "2"))
    out = tmp_path / "p3"

    def hits(bb, pred, s="dev_seen"):
        return pd.read_csv(out / bb / f"nat_90_geo_{pred}" / f"{s}_geo_hits.csv", dtype={"id": str})

    const = hits("streetclip", "constant")
    assert const["pred_lat"].nunique() == 1 and const["pred_lon"].nunique() == 1
    pd.testing.assert_frame_equal(const, hits("clip_b16", "constant"))  # backbone-independent
    oracle = hits("streetclip", "oracle")
    assert (oracle["km"] > 2).all()  # the masked radius is never the answer
    for pred in ("top1", "medoid", "bestk"):  # db points among the unmasked: oracle is a bound
        assert (oracle["km"] <= hits("streetclip", pred)["km"] + 1e-9).all()
    assert (hits("streetclip", "bestk")["km"] <= hits("streetclip", "top1")["km"] + 1e-9).all()
    cent = hits("streetclip", "centroid")
    assert cent["country_top1"].all()  # the fake probe is always right
    m = json.loads((out / "streetclip" / "nat_90_geo_centroid" / "geo_metrics.json").read_text())
    assert m["country_source"]["dev_seen"].startswith(p3.PROBE_VARIANT)
    gp = pd.read_csv(out / "gaps_geoscore.csv")
    assert not set(gp["variant"]) & {"geo_constant", "geo_oracle"}
    assert {"geo_centroid", "geo_bestk"} <= set(gp["variant"])
    assert table[table["predictor"] == "oracle"]["k"].isna().all()


# ----------------------------------------------------------------- Step 2: geocells


def test_majority_country_ties_and_empty_cells():
    cells, labels = np.array([0, 0, 1, 1, 1]), np.array([3, 2, 1, 1, 0])
    cent = np.array([[0.0, 0.0], [0.0, 0.0], [-40.0, 100.0]])
    lat, lon = np.array([0, 0, 0, 0, -41.0]), np.array([0, 0, 0, 0, 101.0])
    assert p3.majority_country(cells, labels, 3, cent, lat, lon).tolist() == [2, 1, 0]


def test_pick_cell_target_uses_only_the_pair_average_on_dev_seen():
    rows = pd.DataFrame([
        {"backbone": "streetclip", "cell_target": 1000, "dev_seen_geoscore": 3000, "dev_unseen_geoscore": 0},
        {"backbone": "clip_l14_336", "cell_target": 1000, "dev_seen_geoscore": 2000, "dev_unseen_geoscore": 0},
        {"backbone": "streetclip", "cell_target": 2000, "dev_seen_geoscore": 2600, "dev_unseen_geoscore": 9e9},
        {"backbone": "clip_l14_336", "cell_target": 2000, "dev_seen_geoscore": 2300, "dev_unseen_geoscore": 9e9},
        {"backbone": "clip_b16", "cell_target": 2000, "dev_seen_geoscore": 9e9, "dev_unseen_geoscore": 9e9}])
    target, avg = p3.pick_cell_target(rows)
    assert target == 1000 and avg == {1000: 2500.0, 2000: 2450.0}  # not streetclip's or b16's best


def test_hybrid_keeps_neighbours_in_the_top_cells_or_falls_back():
    lat, lon = np.array([0.0, 0.0, 0.1, 20.0, 20.1]), np.array([0.0, 0.1, 0.0, 20.0, 20.0])
    db_cells = np.array([0, 0, 0, 1, 1])
    cent = np.array([[0.05, 0.05], [20.05, 20.0], [-30.0, -30.0]])
    sims = np.array([[0.9, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]])
    idx = np.array([[3, 0, 4, 1, 2], [0, 1, 2, 3, 4]])
    top = np.array([[1, 2], [2, 0]])  # query 1: its top cell (2) holds no neighbour
    for pred, tau in (("medoid", None), ("mean", 0.1)):
        (la, lo), fb = p3.hybrid_coords((sims, idx), db_cells, top, 1, pred, 3, tau, lat, lon, cent)
        assert fb.tolist() == [False, True]
        assert (la[1], lo[1]) == (-30.0, -30.0)  # fallback: the top cell's centroid
        assert la[0] == pytest.approx(20.0, abs=0.2) and lo[0] == pytest.approx(20.0, abs=0.2)
    (la, lo), fb = p3.hybrid_coords((sims, idx), db_cells, top, 2, "medoid", 5, None, lat, lon, cent)
    assert not fb.any() and la[1] < 1  # m = 2: cell 0 counts too; cell 1 never does


def test_cell_and_hybrid_grid(setup, monkeypatch):  # noqa: F811
    tmp_path, _ = setup
    dev_seen_n = len(pd.read_csv(tmp_path / "v2" / "dev_seen.csv"))
    probe_calls, hybrid_sel = [], []
    real_probe, real_choose = g.probe_logits, p3.choose_hybrid

    def spy_probe(feats, labels, dev_feats, dev_labels, eval_feats, head_init, a, device):
        probe_calls.append((len(dev_feats), np.asarray(labels).copy(), head_init.clone()))
        return real_probe(feats, labels, dev_feats, dev_labels, eval_feats, head_init, a, device)

    majority = []
    real_major = p3.majority_country
    monkeypatch.setattr(p3, "majority_country",
                        lambda *a: majority.append(real_major(*a)) or majority[-1])
    monkeypatch.setattr(g, "probe_logits", spy_probe)
    monkeypatch.setattr(p3, "choose_hybrid", lambda nb, dbc, top, sel, *a: hybrid_sel.append(len(sel))
                        or real_choose(nb, dbc, top, sel, *a))
    args = ["--methods", "cell", "hybrid", "--cell-targets", "4", "8", "--probe-epochs", "3",
            "--probe-lrs", "0.01", "0.1", "--mask-km", "20", "--k", "1", "5"]
    table = p3.run(p3_args(tmp_path, *args))
    out = tmp_path / "p3"
    choice = json.loads((out / "cell_count.json").read_text())
    assert choice["target"] in (4, 8) and choice["backbones"] == list(pe.PAIR)
    assert set(probe_calls_n := {c[0] for c in probe_calls}) == {dev_seen_n}, probe_calls_n
    assert set(hybrid_sel) == {dev_seen_n}
    # selection: PAIR x 2 targets x the random init only x 2 lrs; main grid: 3 bb x 4 subsets x
    # 2 inits x 2 lrs minus the copied selection cells (PAIR x nat 90 x random x 2 lrs)
    assert len(probe_calls) == 2 * 2 * 2 + 3 * 4 * 2 * 2 - 2 * 2
    assert not list((out / "cell_select").glob("*/*/*zeroshot*"))
    assert choice["probe_init"] == "random"
    assert set(table["variant"]) == {"geo_cell_random", "geo_cell_zeroshot", "geo_hybrid_random",
                                     "geo_hybrid_zeroshot"}
    assert (table["cell_target"] == choice["target"]).all()
    assert table[table["predictor"].str.startswith("cell")]["lr"].isin([0.01, 0.1]).all()
    gp = pd.read_csv(out / "gaps_geoscore.csv")
    assert set(gp["variant"]) == set(table["variant"])
    # zero-shot init: each cell's row = the text embedding of its majority country
    text = g.text_head("m/x", json.loads((tmp_path / "v2" / "classes.json").read_text()), "")
    heads = [c[2] for c in probe_calls]
    assert len(majority) == 3 * 4  # one per zero-shot cell probe (the selection trains none)
    for maj in majority:
        assert sum(h.shape == text[maj].shape and np.allclose(h, text[maj]) for h in heads) >= 2
    for d in (out / "streetclip").glob("*_geo_hybrid_*"):  # masking: a kept neighbour is > 20 km away
        h = pd.read_csv(d / "dev_seen_geo_hits.csv")
        assert (h["fallback"] == 0).any() and (h.loc[h["fallback"] == 0, "km"] > 20).all()
    calls = len(probe_calls)
    p3.run(p3_args(tmp_path, *args))  # everything cached: no probe trained again
    assert len(probe_calls) == calls


# ------------------------------------------- Step 2: saved top cells, CPU-only hybrid, stop/resume

CELL_ARGS = ["--cell-target", "4", "--probe-epochs", "2", "--probe-lrs", "0.1", "--mask-km", "20",
             "--k", "1", "5"]


def test_top_cells_file_holds_five_ids_and_their_probabilities(setup):  # noqa: F811
    tmp_path, _ = setup
    p3.run(p3_args(tmp_path, *CELL_ARGS, "--methods", "cell", "--backbones", "streetclip",
                   "--orderings", "nat", "--sizes", "90", "--probe-inits", "random",
                   "--cell-target", "8"))
    d = tmp_path / "p3" / "streetclip" / "nat_90_geo_cell_random"
    for split in ("dev_seen", "dev"):
        t = pd.read_csv(d / f"{split}_top_cells.csv")
        assert list(t.columns) == ["id", *[f"c{j}" for j in range(5)], *[f"p{j}" for j in range(5)]]
        p = t[[f"p{j}" for j in range(5)]].to_numpy()
        assert (p > 0).all() and (p <= 1 + 1e-9).all() and (np.diff(p, axis=1) <= 1e-12).all()
        assert (p.sum(1) <= 1 + 1e-9).all()


def test_top_cell_probs_are_the_softmax():
    z = np.array([[0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 0.0, 0.0]])
    top = p3.top_cells(z, 2)
    ref = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert np.allclose(p3.top_cell_probs(z, top), np.take_along_axis(ref, top, 1))


def test_hybrid_runs_without_cuda_from_saved_top_cells(setup, monkeypatch):  # noqa: F811
    tmp_path, _ = setup
    base = ["--backbones", "streetclip", "--orderings", "nat", "--sizes", "90", *CELL_ARGS]
    p3.run(p3_args(tmp_path, *base, "--methods", "cell", "hybrid"))
    out = tmp_path / "p3"
    hyb = out / "streetclip" / "nat_90_geo_hybrid_random"
    first = json.loads((hyb / "row.json").read_text())
    for d in out.glob("*/*_geo_hybrid_*"):
        shutil.rmtree(d)

    def no_probe(*a, **k):
        raise AssertionError("the hybrid must not train a probe")

    import torch
    monkeypatch.setattr(g, "probe_logits", no_probe)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    table = p3.run(p3_args(tmp_path, *base, "--methods", "hybrid", "--device", ""))
    again = json.loads((hyb / "row.json").read_text())
    assert set(table["variant"]) == {"geo_cell_random", "geo_cell_zeroshot", "geo_hybrid_random",
                                     "geo_hybrid_zeroshot"}
    assert again["knn_device"] == "cpu" and again["cache_key"] == first["cache_key"]
    assert again["dev_seen_geoscore"] == first["dev_seen_geoscore"]  # same top cells, same kNN


def test_hybrid_key_follows_its_probe_source_and_hybrid_alone_needs_cells(setup, monkeypatch):  # noqa: F811
    tmp_path, _ = setup
    base = ["--backbones", "streetclip", "--orderings", "nat", "--sizes", "90", *CELL_ARGS,
            "--probe-inits", "random"]
    with pytest.raises(SystemExit, match="needs finished cell probes"):
        p3.run(p3_args(tmp_path, *base, "--methods", "hybrid"))
    p3.run(p3_args(tmp_path, *base, "--methods", "cell", "hybrid"))
    calls = []
    real = p3.run_hybrid
    monkeypatch.setattr(p3, "run_hybrid", lambda *a, **k: calls.append(a[2]) or real(*a, **k))
    p3.run(p3_args(tmp_path, *base, "--methods", "hybrid"))
    assert calls == []  # cached: same probe source
    f = tmp_path / "p3" / "streetclip" / "nat_90_geo_cell_random" / "dev_seen_top_cells.csv"
    t = pd.read_csv(f)
    t["c0"] = (t["c0"] + 1) % 4  # a different probe output
    t.to_csv(f, index=False)
    p3.run(p3_args(tmp_path, *base, "--methods", "hybrid"))
    assert calls == [90]  # only that hybrid cell is redone, no cell probe is trained


def test_stop_file_finishes_the_cell_then_asks_to_resume(setup, monkeypatch, capsys):  # noqa: F811
    tmp_path, _ = setup
    stop, ran = tmp_path / "STOP", []
    real = p3.run_cells

    def spy(ctx, o, n, *a, **k):
        if not ran:
            stop.touch()  # the deadline passes while the first cell runs
        ran.append((o, n))
        return real(ctx, o, n, *a, **k)

    monkeypatch.setattr(p3, "run_cells", spy)
    args = ["--backbones", "streetclip", "--probe-inits", "random", "--methods", "cell",
            *CELL_ARGS, "--stop-file", str(stop)]
    with pytest.raises(p3.StopRequested, match="3 cell run"):
        p3.run(p3_args(tmp_path, *args))
    done = sorted(p.parent.name for p in (tmp_path / "p3" / "streetclip").glob("*/row.json"))
    assert len(ran) == 1 and len(done) == 1  # the cell in hand was finished and saved
    stop.unlink()
    p3.run(p3_args(tmp_path, *args))
    assert len(ran) == 4 and len(set(ran)) == 4  # the other three ran; the first did not rerun
    stop.touch()
    fresh = p3_args(tmp_path, *args, "--out", str(tmp_path / "q"))  # before parse_args is patched
    monkeypatch.setattr(p3, "parse_args", lambda: fresh)
    monkeypatch.setattr(sys, "argv", ["phase3_eval"])
    with pytest.raises(SystemExit) as e:
        p3.main()
    assert e.value.code == 3 and "RESUME NEEDED" in capsys.readouterr().out
