"""Phase 3 coordinate runner on synthetic embeddings (no model, no Kaggle)."""

import json

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
