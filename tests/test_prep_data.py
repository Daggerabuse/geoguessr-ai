import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from geoguessr_ai import prep_data as p
from tests.conftest import write_jpeg

COUNTRIES = ["AA", "BB"]


def _meta(prefix: str, n_seq: int = 30, per_seq: int = 3) -> pd.DataFrame:
    rows = []
    for c in COUNTRIES:
        for s in range(n_seq):
            for k in range(per_seq):
                rows.append(
                    {"id": f"{prefix}{c}{s}_{k}", "country": c, "sequence": f"{prefix}{c}{s}",
                     "latitude": 1.0, "longitude": 2.0}
                )
    return pd.DataFrame(rows)


def test_cap_per_sequence_limits_frames_and_is_seeded():
    df = _meta("t")
    a = p.cap_per_sequence(df, 2, seed=1)
    assert a.groupby("sequence").size().max() == 2
    b = p.cap_per_sequence(df, 2, seed=1)
    assert a["id"].tolist() == b["id"].tolist()


def test_split_train_val_has_no_sequence_overlap_and_right_counts():
    df = _meta("t")
    out = p.split_train_val(df, train_n=30, val_n=10, seed=42)
    for c, g in out.groupby("country"):
        assert (g["split"] == "val").sum() == 10
        assert (g["split"] == "train").sum() == 30
    val_seqs = set(out.loc[out["split"] == "val", "sequence"])
    train_seqs = set(out.loc[out["split"] == "train", "sequence"])
    assert not (val_seqs & train_seqs)


def test_pick_test_caps_per_country():
    out = p.pick_test(_meta("x"), 7, seed=42)
    assert out.groupby("country").size().tolist() == [7, 7]
    assert (out["split"] == "test").all()


def test_build_manifest_labels_paths_and_sources(tmp_path):
    _meta("t").to_csv(tmp_path / "train.csv", index=False)
    _meta("x").to_csv(tmp_path / "test.csv", index=False)
    args = argparse.Namespace(
        countries=COUNTRIES, max_per_seq=3, seed=42,
        train_per_country=20, val_per_country=5, test_per_country=5)
    m, classes = p.build_manifest(args, tmp_path / "train.csv", tmp_path / "test.csv")
    assert classes == ["AA", "BB"]
    assert set(m["split"]) == {"train", "val", "test"}
    assert (m.loc[m["split"] == "test", "source"] == "test").all()
    assert (m.loc[m["split"] != "test", "source"] == "train").all()
    assert m["path"].str.startswith("images/").all()
    assert (m["label"] == m["country"].map({"AA": 0, "BB": 1})).all()


def _make_zips(root: Path, ids_by_source: dict[str, list[str]], n_shards: dict[str, int]):
    for src, ids in ids_by_source.items():
        (root / src).mkdir(parents=True)
        zs = [zipfile.ZipFile(root / src / f"{i:02d}.zip", "w") for i in range(n_shards[src])]
        for j, id_ in enumerate(ids):
            img = root / "_tmp" / f"{id_}.jpg"
            write_jpeg(img, w=96, h=64, seed=j)
            zs[j % n_shards[src]].write(img, f"{src[0]}/{id_}.jpg")  # nested name inside the zip
        for z in zs:
            z.close()


def test_index_and_extract_from_local_zips(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "N_SHARDS", {"train": 2, "test": 1})
    ids = [f"img{i}" for i in range(6)]
    _make_zips(tmp_path / "zips", {"train": ids, "test": ["t0"]}, {"train": 2, "test": 1})
    opener = p.make_opener(tmp_path / "zips")
    idx = p.build_index(opener, "train", tmp_path / "idx.json", workers=2)
    assert set(idx) == set(ids)
    assert idx["img0"][0] == 0 and idx["img1"][0] == 1
    cached = p.build_index(opener, "train", tmp_path / "idx.json", workers=2)  # from cache
    assert cached == idx

    out = tmp_path / "out"
    (out / "images/train").mkdir(parents=True)
    m = pd.DataFrame({"id": ids, "source": "train"})
    p.download(m, out, opener, workers=2, resize=32)
    for id_ in ids:
        with Image.open(out / "images/train" / f"{id_}.jpg") as im:
            assert min(im.size) == 32  # short side resized, aspect ratio kept


def test_save_image_without_resize_keeps_bytes(tmp_path):
    write_jpeg(tmp_path / "a.jpg")
    data = (tmp_path / "a.jpg").read_bytes()
    p.save_image(data, tmp_path / "b.jpg", 0)
    assert (tmp_path / "b.jpg").read_bytes() == data


def test_main_end_to_end_with_local_zips(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "N_SHARDS", {"train": 2, "test": 1})
    tr, te = _meta("t"), _meta("x")
    tr.to_csv(tmp_path / "train.csv", index=False)
    te.to_csv(tmp_path / "test.csv", index=False)
    _make_zips(tmp_path / "zips", {"train": tr["id"].tolist(), "test": te["id"].tolist()},
               {"train": 2, "test": 1})
    out = tmp_path / "subset"
    monkeypatch.setattr("sys.argv", [
        "prep_data", "--out", str(out), "--train-csv", str(tmp_path / "train.csv"),
        "--test-csv", str(tmp_path / "test.csv"), "--countries", *COUNTRIES,
        "--train-per-country", "20", "--val-per-country", "5", "--test-per-country", "5",
        "--local-zips", str(tmp_path / "zips"), "--workers", "2", "--resize", "32",
    ])
    p.main()
    m = pd.read_csv(out / "manifest.csv")
    assert json.loads((out / "classes.json").read_text()) == COUNTRIES
    assert m.groupby("split").size().to_dict() == {"test": 10, "train": 40, "val": 10}
    assert all((out / path).exists() for path in m["path"])


def test_stats_only_writes_country_counts(tmp_path, monkeypatch):
    _meta("t").to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr("sys.argv", [
        "prep_data", "--out", str(tmp_path / "o"), "--train-csv", str(tmp_path / "train.csv"),
        "--stats-only"])
    p.main()
    stats = pd.read_csv(tmp_path / "o" / "country_counts.csv", index_col=0)
    assert stats["n_images"].to_dict() == {"AA": 90, "BB": 90}
    assert stats["n_sequences"].to_dict() == {"AA": 30, "BB": 30}


def test_default_countries_are_unique_and_include_key_countries():
    c = p.DEFAULT_COUNTRIES
    assert len(c) == len(set(c)) == 33
    for code in ("RU", "CO", "IN", "TH", "TW", "ID", "MA", "NG", "GH"):
        assert code in c


@pytest.mark.parametrize("split", ["train", "test"])
def test_n_shards_matches_dataset(split):
    assert p.N_SHARDS == {"train": 98, "test": 5}
    assert split in p.N_SHARDS
