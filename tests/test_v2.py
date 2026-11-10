"""V2 pipeline: data/audit (v2_data), embedding cache (embed), heads and metrics (geo_eval)."""

import argparse
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from geoguessr_ai import embed as e
from geoguessr_ai import geo_eval as g
from geoguessr_ai import v2_data as v
from tests.test_prep_data import _make_zips


def _meta(prefix: str, countries=("AA", "BB", "CC"), sizes=(60, 20, 4), per_seq=3) -> pd.DataFrame:
    """Skewed synthetic metadata: country i has sizes[i] sequences; 2 sequences per creator."""
    rows = []
    for c, n_seq, lat in zip(countries, sizes, (10.0, -30.0, 50.0, 0.0)):
        for s in range(n_seq):
            for k in range(per_seq):
                rows.append({"id": f"{prefix}{c}{s}_{k}", "country": c, "sequence": f"{prefix}{c}{s}",
                             "creator_id": f"{prefix}{c}u{s // 2}", "latitude": lat + 0.01 * s,
                             "longitude": 5.0 + 0.01 * k, "quadtree_10_1000": 0})
    return pd.DataFrame(rows)


def _args(**kw):
    base = {"sizes": [30, 90], "max_per_seq": 3, "dev_creator_frac": 0.2, "dev_n": 20, "seed": 42}
    return argparse.Namespace(**{**base, **kw})


# ----------------------------------------------------------------- v2_data: metadata


def test_read_meta_schema_namibia_and_creators(tmp_path):
    df = _meta("t", countries=("NA", "BB"), sizes=(2, 2))
    df["creator_id"] = df["creator_id"] + ".0"  # as written in the official CSVs
    df.loc[0, "creator_id"] = ""  # missing creator
    df.to_csv(tmp_path / "m.csv", index=False)
    m = v.read_meta(tmp_path / "m.csv")
    assert (m["country"] == "NA").sum() == 6  # Namibia is not a missing value
    assert m.loc[0, "creator_id"] == "seq:" + m.loc[0, "sequence"]
    assert not m["creator_id"].str.endswith(".0").any()
    df.drop(columns="quadtree_10_1000").to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="quadtree_10_1000"):
        v.read_meta(tmp_path / "bad.csv")


def test_audit_class_definition_groups_and_overlap():
    tr = _meta("t", countries=("AA", "BB", "CC", "TT"), sizes=(60, 20, 4, 2))  # TT only in train
    te = pd.concat([_meta("x", sizes=(40, 3, 2)), _meta("x", countries=("ZZ",), sizes=(2,))])
    te.loc[te["country"] == "AA", "creator_id"] = "tAAu0"  # AA test images: creator seen in train
    classes = sorted(set(te["country"]) & set(tr["country"]))
    stats, s = v.audit(tr, te, classes, min_test_images=100, min_test_sequences=30)
    assert classes == ["AA", "BB", "CC"]
    assert s["countries_only_in_train"] == ["TT"] and s["countries_only_in_test"] == ["ZZ"]
    assert s["test_images_of_countries_only_in_test"] == 6
    assert stats.loc["AA", "test_seen_creator_frac"] == 1.0
    assert stats.loc["BB", "test_seen_creator_frac"] == 0.0
    assert stats.loc[classes, "freq_group"].tolist() == ["head", "middle", "tail"]
    assert stats.loc["TT", "freq_group"] == "" and stats.loc["ZZ", "freq_group"] == ""
    assert stats["reliable"].tolist().count(True) == 1 and stats.loc["AA", "reliable"]


# ----------------------------------------------------------------- v2_data: splits


def test_build_classes_dev_and_nested_subsets():
    te = pd.concat([_meta("x", sizes=(5, 5, 5)), _meta("x", countries=("ZZ",), sizes=(2,))])
    parts, classes, dropped = v.build(_meta("t"), te, _args())
    assert classes == ["AA", "BB", "CC"] and dropped == 0
    tr, dev = parts["train"], parts["dev"]
    assert not set(dev["creator_id"]) & set(tr["creator_id"])  # creator-disjoint
    assert dev.groupby("sequence").size().max() == 1
    assert (parts["test"]["label"] == -1).sum() == 6  # ZZ kept, marked unlabelled
    for s in ("nat", "bal"):
        small, big = set(v.subset(tr, s, 30)["id"]), set(v.subset(tr, s, 90)["id"])
        assert len(small) == 30 and small < big  # nested
    assert v.subset(tr, "bal", 12)["country"].value_counts().to_dict() == {"AA": 4, "BB": 4, "CC": 4}


def test_build_drops_train_rows_overlapping_test():
    tr, te = _meta("t"), _meta("x", sizes=(3, 3, 3))
    tr.loc[0, "sequence"] = te.loc[0, "sequence"]  # one leaked sequence, one leaked id
    tr.loc[5, "id"] = te.loc[7, "id"]
    parts, _, dropped = v.build(tr, te, _args(sizes=[10**6]))
    assert dropped == 2
    v.check_separation(parts)


def test_check_separation_rejects_shared_rows():
    parts, _, _ = v.build(_meta("t"), _meta("x"), _args())
    bad = {**parts, "dev": pd.concat([parts["dev"], parts["train"].head(1)])}
    with pytest.raises(AssertionError):
        v.check_separation(bad)


def test_build_is_seeded_ranks_are_permutations_and_dev_ignores_cap():
    a, _, _ = v.build(_meta("t"), _meta("x"), _args())
    b, _, _ = v.build(_meta("t"), _meta("x"), _args())
    assert a["train"]["id"].tolist() == b["train"]["id"].tolist()
    assert a["dev"]["id"].tolist() == b["dev"]["id"].tolist()
    full, _, _ = v.build(_meta("t"), _meta("x"), _args(sizes=[10**6]))
    for col in ("nat_rank", "bal_rank"):
        assert sorted(full["train"][col]) == list(range(len(full["train"])))
    one, _, _ = v.build(_meta("t"), _meta("x"), _args(max_per_seq=1))
    assert one["dev"]["id"].tolist() == a["dev"]["id"].tolist()
    assert one["train"].groupby("sequence").size().max() == 1


# ----------------------------------------------------------------- geo metrics


def test_haversine_and_geoscore():
    assert g.haversine(0, 0, 0, 0) == 0
    assert g.haversine(0, 0, 0, 180) == pytest.approx(np.pi * g.EARTH_RADIUS_KM)
    assert g.haversine(0, 179.5, 0, -179.5) == pytest.approx(111.2, abs=0.2)  # across the date line
    assert g.geoscore(0) == 5000
    assert g.geoscore(1492.7) == pytest.approx(5000 / np.e)


def test_cluster_bootstrap_contains_mean_widens_with_clustering_and_is_seeded():
    rng = np.random.default_rng(0)
    vals = np.repeat(rng.random(50) < 0.5, 10).astype(float)  # 50 clusters of 10 identical values
    clusters = np.repeat(np.arange(50), 10)
    lo, hi = g.cluster_bootstrap(vals, clusters, n_boot=500)
    ilo, ihi = g.cluster_bootstrap(vals, np.arange(500), n_boot=500)  # treated as independent
    assert lo <= vals.mean() <= hi
    assert hi - lo > 1.5 * (ihi - ilo)
    assert (lo, hi) == g.cluster_bootstrap(vals, clusters, n_boot=500)


def test_knn_numpy_matches_brute_force_and_votes():
    rng = np.random.default_rng(0)
    db = rng.normal(size=(300, 8)).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    q = db[:20] + 0.01
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sims, idx = g.knn(q, db, 5)
    assert (idx == np.argsort(-(q @ db.T), axis=1)[:, :5]).all()
    assert (idx[:, 0] == np.arange(20)).all() and (np.diff(sims, axis=1) <= 0).all()
    labels = np.arange(300) % 3
    assert (g.knn_country_scores(sims, idx, labels, 3, k=1).argmax(1) == labels[:20]).all()


def test_knn_coordinates_takes_the_geographic_medoid():
    lat = np.array([10.0, 10.1, 9.9, -40.0])  # three neighbours close together, one outlier
    lon = np.array([20.0, 20.1, 19.9, 100.0])
    sims = np.array([[0.9, 0.8, 0.8, 0.95]])  # the outlier is the single best match
    idx = np.array([[3, 0, 1, 2]])
    assert tuple(x[0] for x in g.knn_coordinates(sims, idx, lat, lon, k=4)) == (10.0, 20.0)
    assert g.knn_coordinates(sims, idx, lat, lon, k=1)[0][0] == -40.0


def test_score_split_counts_unseen_classes_as_errors_and_keeps_full_label_space(tmp_path):
    classes = ["AA", "BB"]
    df = pd.DataFrame({"id": list("abcd"), "country": ["AA", "AA", "BB", "ZZ"], "label": [0, 0, 1, -1],
                       "sequence": list("wxyz"), "latitude": 0.0, "longitude": 0.0,
                       "creator_id": list("pqrs")})
    # row b: BB scores higher than AA; row d (ZZ, no training data) can never be right
    scores = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 3.0], [5.0, 0.0]])
    counts = pd.DataFrame({"reliable": [True, False, False], "freq_group": ["head", "tail", ""]},
                          index=["AA", "BB", "ZZ"])
    groups = g.row_groups(df, counts, np.array([True, False, False, False]))
    r = g.score_split(tmp_path, "dev", df, scores, classes, None, groups,
                      argparse.Namespace(n_boot=50, seed=0))
    assert r["all"]["country_top1"]["mean"] == 0.5  # a, c right; b wrong; d counts as an error
    assert r["all"]["macro_top1"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert r["class_in_train"]["country_top1"]["mean"] == pytest.approx(2 / 3)
    # the AA-only "reliable" subset is still scored with the argmax over both classes
    assert r["reliable"]["country_top1"]["mean"] == 0.5 and r["reliable"]["n"] == 2
    assert r["head"]["n"] == 2 and r["tail"]["n"] == 1 and r["middle"]["n"] == 0
    assert r["seen_creator"]["n"] == 1
    assert r["country_files"]["n_missing"] == 1 and not (tmp_path / "dev_confusion.png").exists()


# ----------------------------------------------------------------- preprocessing


@pytest.mark.parametrize("size", [224, 336])
@pytest.mark.parametrize("wh", [(1086, 512), (682, 512), (511, 512), (512, 853), (640, 640), (300, 200)])
def test_clip_center_crop_matches_clip_image_processor(size, wh):
    pytest.importorskip("transformers")
    from transformers import CLIPImageProcessor

    from geoguessr_ai.train import to_tensor_array

    rng = np.random.default_rng(size + wh[0])
    im = Image.fromarray(rng.integers(0, 256, (wh[1], wh[0], 3), dtype=np.uint8))
    # the values of the openai/clip-vit-* and geolocal/StreetCLIP preprocessor_config.json
    proc = CLIPImageProcessor(size={"shortest_edge": size}, crop_size={"height": size, "width": size},
                              resample=3)
    want = proc(images=im, return_tensors="np")["pixel_values"][0]
    np.testing.assert_allclose(to_tensor_array(e.clip_center_crop(im, size)), want, atol=1e-5)


# ----------------------------------------------------------------- embedding cache


def test_cache_roundtrip_source_separation_and_temp_files(tmp_path):
    e.save_npz(tmp_path / "train_00_r1.npz", ["1", "2"], np.eye(2, 3), failed=["9"])
    e.save_npz(tmp_path / "test_00_r1.npz", ["1"], np.ones((1, 3)), failed=[])
    (tmp_path / ".train_01_r2.npz.dead.tmp").write_bytes(b"half-written")  # killed mid-write
    assert e.cached_ids(tmp_path, "train") == {"1", "2"} and e.cached_ids(tmp_path, "test") == {"1"}
    emb, found = e.load_embeddings(tmp_path, ["2", "x", "1"], "train")
    assert found.tolist() == [True, False, True] and emb[2].tolist() == [1, 0, 0]
    assert e.load_embeddings(tmp_path, ["1"], "test")[0][0].tolist() == [1, 1, 1]  # not the train row


def test_duplicates_resolve_to_first_file_deterministically(tmp_path, capsys):
    e.save_npz(tmp_path / "train_00_a.npz", ["1"], np.zeros((1, 2)), failed=[])
    e.save_npz(tmp_path / "train_00_b.npz", ["1"], np.ones((1, 2)), failed=[])
    emb, _ = e.load_embeddings(tmp_path, ["1"], "train")
    assert emb[0].tolist() == [0, 0] and "more than one file" in capsys.readouterr().out


def test_publish_once_never_overwrites_when_the_race_is_lost(tmp_path, monkeypatch):
    p = tmp_path / "encoder.json"
    p.write_text("first")
    monkeypatch.setattr(Path, "exists", lambda self: False)  # both saw "not there yet"
    e.publish_once(p, lambda t: t.write_text("second"))
    assert p.read_text() == "first" and not list(tmp_path.glob("*.tmp"))


def test_encoder_guard(tmp_path):
    info = {"model": "m", "image_size": 224, "dim": 512}
    e.check_encoder(tmp_path, info)
    e.check_encoder(tmp_path, info)  # same encoder: fine
    with pytest.raises(ValueError, match="use another --out"):
        e.check_encoder(tmp_path, {**info, "model": "other"})
    with pytest.raises(ValueError, match="use another --out"):  # T4 and P100 never mixed
        e.check_encoder(tmp_path, {**info, "gpu": "Tesla P100-PCIE-16GB"})


def test_plan_parts_are_disjoint_complete_and_limit_is_seeded():
    index = {f"i{j}": (j % 5, f"t/i{j}.jpg") for j in range(100)}
    want = [*index, "not_in_shards"]
    parts = [e.plan(want, {"i0"}, index, 2, p, 0, 42) for p in (0, 1)]
    ids = [[i for items in pl.values() for i, _ in items] for pl in parts]
    assert not set(ids[0]) & set(ids[1])
    assert set(ids[0]) | set(ids[1]) == set(index) - {"i0"}
    assert all(s % 2 == p for p, pl in enumerate(parts) for s in pl)
    lim = [e.plan(want, set(), index, 1, 0, 10, 42) for _ in range(2)]
    assert lim[0] == lim[1] and sum(map(len, lim[0].values())) == 10


def _plan_ids(pl):
    return {i for items in pl.values() for i, _ in items}


def test_limit_is_global_over_parts_and_stable_under_resume():
    index = {f"i{j}": (j % 5, f"t/i{j}.jpg") for j in range(100)}
    target = set(e.target_ids(list(index), index, 30, 42))
    assert len(target) == 30
    parts = [_plan_ids(e.plan(list(index), set(), index, 2, p, 30, 42)) for p in (0, 1)]
    assert not parts[0] & parts[1] and parts[0] | parts[1] == target  # 30 in total, not per part
    # after a partial run the rest of the SAME sample is planned, not a fresh one
    done = set(sorted(target)[:12])
    rest = [_plan_ids(e.plan(list(index), done, index, 2, p, 30, 42)) for p in (0, 1)]
    assert rest[0] | rest[1] == target - done


def test_embed_parse_args_checks_part():
    with pytest.raises(SystemExit):
        e.parse_args(["--ids", "a.csv", "--source", "test", "--out", "o", "--parts", "2", "--part", "2"])


# ----------------------------------------------------------------- embedding runs (torch)


@pytest.fixture
def zips(tmp_path, monkeypatch):
    """train: 60 ids in 4 shards; test: 9 ids in 1 shard."""
    monkeypatch.setattr("geoguessr_ai.prep_data.N_SHARDS", {"train": 4, "test": 1})
    ids = {"train": [f"tr{i}" for i in range(60)], "test": [f"te{i}" for i in range(9)]}
    _make_zips(tmp_path / "zips", ids, {"train": 4, "test": 1})
    pd.DataFrame({"id": ids["train"]}).to_csv(tmp_path / "train_ids.csv", index=False)
    return tmp_path, ids


def _embed(root: Path, out: Path, *extra):
    return e.run(e.parse_args(["--ids", str(root / "train_ids.csv"), "--source", "train",
                               "--out", str(out), "--smoke", "--device", "cpu", "--workers", "2",
                               "--batch-size", "8", "--local-zips", str(root / "zips"), *extra]))


def _all_ids(out: Path) -> list[str]:
    return [str(i) for f in e.cache_files(out, "train") for i in np.load(f)["ids"]]


def _in_thread(fn, timeout=120):
    """Run fn in a thread; return (finished, exception)."""
    box = {}

    def target():
        try:
            fn()
        except Exception as ex:  # noqa: BLE001 - returned to the test
            box["err"] = ex

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout)
    return not t.is_alive(), box.get("err")


def test_two_parallel_parts_embed_everything_once(zips):
    pytest.importorskip("transformers")
    root, ids = zips
    out = root / "emb" / "m"
    threads = [threading.Thread(target=_embed, args=(root, out, "--parts", "2", "--part", str(p)))
               for p in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    assert sorted(_all_ids(out)) == sorted(ids["train"])  # complete, no duplicates
    assert json.loads((root / "emb" / "index_train.json").read_text())  # shared index is valid
    assert not list((root / "emb").rglob("*.tmp"))
    assert len((out / "throughput.jsonl").read_text().splitlines()) == 2
    assert json.loads((out / "encoder.json").read_text())["gpu"] == "cpu"


def _check(root: Path, out: Path, *extra):
    return e.check(e.parse_args(["--check", "--ids", str(root / "train_ids.csv"), "--source",
                                 "train", "--out", str(out), "--local-zips", str(root / "zips"),
                                 *extra]))


def test_dual_limit_matches_single_run_and_check_verifies_it(zips):
    pytest.importorskip("transformers")
    root, _ = zips
    single, dual = root / "emb" / "single", root / "emb" / "dual"
    _embed(root, single, "--limit", "25")
    stats = [_embed(root, dual, "--limit", "25", "--parts", "2", "--part", str(p)) for p in (0, 1)]
    assert sum(s["images"] for s in stats) == 25 and all(s["target"] == 25 for s in stats)
    res = _check(root, dual, "--limit", "25", "--reference", str(single))
    assert res["ok"] and res["unique"] == res["compared"] == 25 and res["max_abs_diff"] == 0.0
    # a duplicated file, or a cache checked against the wrong target, fails the check
    f = e.cache_files(dual, "train")[0]
    f.with_name(f.name.replace(".npz", "_copy.npz")).write_bytes(f.read_bytes())
    assert _check(root, dual, "--limit", "25")["duplicates"] > 0
    assert not _check(root, single, "--limit", "26")["ok"]


def test_interrupted_run_resumes_without_duplicates(zips, monkeypatch):
    pytest.importorskip("transformers")
    root, ids = zips
    out = root / "emb" / "m"
    real = e.make_opener(root / "zips")
    out.mkdir(parents=True)
    e.shared_index(real, "train", out.parent / "index_train.json", 2)  # fail during extraction

    def flaky(src, shard):  # shard 2 fails as if the connection dropped
        if shard == 2:
            raise ConnectionError("dropped")
        return real(src, shard)

    monkeypatch.setattr(e, "make_opener", lambda _: flaky)
    first = _embed(root, out)
    assert 0 < first["images"] < len(ids["train"])
    assert _check(root, out)["missing"] > 0
    monkeypatch.setattr(e, "make_opener", lambda _: real)
    second = _embed(root, out)
    assert first["images"] + second["images"] == len(ids["train"])
    assert sorted(_all_ids(out)) == sorted(ids["train"])
    assert _check(root, out)["ok"]
    assert _embed(root, out)["images"] == 0  # nothing left


def test_gpu_side_failure_does_not_hang(zips, monkeypatch):
    pytest.importorskip("transformers")
    root, _ = zips

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory (simulated)")

    monkeypatch.setattr(e, "as_embedding", boom)
    finished, err = _in_thread(lambda: _embed(root, root / "emb" / "m", "--workers", "1",
                                              "--batch-size", "2"), timeout=60)
    assert finished and "simulated" in str(err)
    assert not _all_ids(root / "emb" / "m")


def test_cache_refuses_a_different_encoder(zips):
    pytest.importorskip("transformers")
    root, _ = zips
    out = root / "emb" / "m"
    _embed(root, out, "--limit", "4")
    with pytest.raises(ValueError, match="use another --out"):
        e.run(e.parse_args(["--ids", str(root / "train_ids.csv"), "--source", "train",
                            "--out", str(out), "--model", "some/other-model", "--smoke",
                            "--device", "cpu", "--local-zips", str(root / "zips")]))


# ----------------------------------------------------------------- end to end (torch)


def _write_data(data: Path, tr: pd.DataFrame, te: pd.DataFrame, **kw):
    parts, classes, _ = v.build(tr, te, _args(**kw))
    stats, _ = v.audit(tr, te, classes, 3, 2)
    data.mkdir(parents=True)
    stats.to_csv(data / "country_counts.csv")
    (data / "classes.json").write_text(json.dumps(classes))
    for name, d in parts.items():
        d.to_csv(data / f"{name}.csv", index=False)
    return parts


def test_end_to_end_dev_only_then_test(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.setattr("geoguessr_ai.prep_data.N_SHARDS", {"train": 2, "test": 1})
    tr = _meta("t", sizes=(8, 6, 4))
    te = pd.concat([_meta("x", sizes=(2, 2, 2)), _meta("x", countries=("ZZ",), sizes=(1,))])
    _make_zips(tmp_path / "zips", {"train": tr["id"].tolist(), "test": te["id"].tolist()},
               {"train": 2, "test": 1})
    data = tmp_path / "v2"
    parts = _write_data(data, tr, te, sizes=[20, 40], dev_n=10)

    emb = tmp_path / "emb" / "m"
    for name, src in (("train", "train"), ("dev", "train"), ("test", "test")):
        e.run(e.parse_args(["--ids", str(data / f"{name}.csv"), "--source", src, "--out", str(emb),
                            "--smoke", "--device", "cpu", "--workers", "2", "--batch-size", "8",
                            "--local-zips", str(tmp_path / "zips")]))
    common = ["--data", str(data), "--emb", str(emb), "--tag", "smoke", "--sizes", "20", "40",
              "--heads", "knn", "--k", "1", "5", "--device", "", "--n-boot", "50",
              "--results-dir", str(tmp_path / "res")]
    rows = g.run(g.parse_args(common))
    assert [r["split"] for r in rows] == ["dev", "dev"]
    assert not list((tmp_path / "res").rglob("test_*"))  # official test untouched in development

    rows = g.run(g.parse_args([*common, "--eval-test"]))
    m = json.loads((tmp_path / "res" / "smoke_bal_40_knn" / "geo_metrics.json").read_text())
    assert m["test"]["all"]["n"] == len(parts["test"])  # ZZ rows included...
    assert m["test"]["class_in_train"]["n"] == len(parts["test"]) - 3  # ...and countable apart
    assert m["test"]["seen_creator"]["n"] == 0
    assert 0 <= m["test"]["all"]["geoscore"]["mean"] <= 5000
    assert rows[0]["split"] == "test"


def test_geo_eval_refuses_overlapping_train_and_dev(tmp_path):
    data = tmp_path / "v2"
    parts = _write_data(data, _meta("t"), _meta("x"))
    pd.concat([parts["train"], parts["dev"].head(1).assign(nat_rank=0, bal_rank=0)]) \
        .to_csv(data / "train.csv", index=False)
    emb = tmp_path / "emb"
    emb.mkdir()
    ids = pd.concat([parts["train"]["id"], parts["dev"]["id"]]).tolist()
    e.save_npz(emb / "train_00_x.npz", ids, np.ones((len(ids), 4)), failed=[])
    with pytest.raises(AssertionError, match="share"):
        g.run(g.parse_args(["--data", str(data), "--emb", str(emb), "--tag", "t", "--sizes", "10",
                            "--heads", "knn", "--device", "", "--results-dir", str(tmp_path / "r")]))


def test_probe_trains_on_cached_features():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16))
    y = np.repeat(np.arange(3), 40)
    x = (centers[y] + 0.3 * rng.normal(size=(120, 16))).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    args = argparse.Namespace(probe_lr=0.1, probe_epochs=50, label_smoothing=0.0, seed=0,
                              logit_scale=30.0)
    logits, history = g.probe_logits(x, y, x, y, {"test": x}, torch.randn(3, 16), args, "cpu")
    assert (logits["test"].argmax(1) == y).mean() > 0.9
    assert history[0]["epoch"] == 0


def test_bench_smoke(subset_root, tmp_path):
    pytest.importorskip("transformers")
    from geoguessr_ai import bench

    out = tmp_path / "bench.jsonl"
    res = bench.run(bench.parse_args(["--smoke", "--device", "cpu", "--batch-size", "4", "--steps", "2",
                                      "--warmup", "1", "--infer-batch-size", "6", "--root", str(subset_root),
                                      "--workers", "0",
                                      "--out", str(out)]))
    assert res["train_images_per_s"] > 0 and res["infer_images_per_s"] > 0
    assert res["loader_images_per_s"] > 0 and res["infer_batch_size"] == 6
    assert json.loads(out.read_text().splitlines()[0])["batch_size"] == 4
