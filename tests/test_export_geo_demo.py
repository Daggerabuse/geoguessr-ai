"""export_geo_demo end to end on a random tiny CLIP (no download). Skipped without torch/onnx."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from geoguessr_ai import export_geo_demo as d
from geoguessr_ai import export_onnx as eo
from geoguessr_ai import geocells as gc
from geoguessr_ai.prereg import sha256
from tests.conftest import gradient_image


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("geo_demo")
    src = eo.write_tiny_folder(tmp / "src", image_size=32)  # projection_dim 16
    rng = np.random.default_rng(0)
    tree = gc.build(rng.uniform(-60, 70, 400), rng.uniform(-180, 180, 400), max_per_cell=40)
    cells_sha = gc.save(tree, tmp / "cells.json")
    head = rng.normal(size=(gc.n_cells(tree), 16)).astype(np.float32)
    np.save(tmp / "head.npy", head)
    (tmp / "probe.json").write_text(json.dumps({"head_sha256": sha256(tmp / "head.npy")}))
    out = tmp / "out"
    info = d.export(str(src), tmp / "head.npy", tmp / "cells.json", out, cells_sha)
    return tmp, src, out, info, tree, head


def test_files(exported):
    _, _, out, info, tree, head = exported
    cfg = json.loads((out / "inference_config.json").read_text())
    assert cfg["n_cells"] == gc.n_cells(tree) and cfg["embed_dim"] == 16 and cfg["image_size"] == 32
    assert cfg["resize"] == "truncate" and cfg["crop"] == "center"
    assert cfg["head_sha256"] == sha256(out / d.HEAD_FILE)
    cells = json.loads((out / d.CELLS_FILE).read_text())
    assert cells["centroid_lat"] == tree["centroid_lat"] and cells["centroid_lon"] == tree["centroid_lon"]
    h = np.fromfile(out / d.HEAD_FILE, dtype="<f4").reshape(head.shape)
    np.testing.assert_allclose(h, head / np.linalg.norm(head, axis=1, keepdims=True), rtol=1e-6)
    assert set(info["files"]) == {d.FP32_FILE, d.INT8_FILE, d.HEAD_FILE, d.CELLS_FILE}
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "license: cc-by-nc-4.0" in card and "CC BY-SA 4.0" in card
    rows = d.test_rows()  # numbers in the card come from the committed results/v3/test_results.csv
    for pred in ("cell_zeroshot", "hybrid_zeroshot"):
        r = rows[pred]
        assert f"**{r['test_geoscore']:.0f}** [{r['test_geoscore_lo']:.0f}, {r['test_geoscore_hi']:.0f}]" in card
    assert "Must not be used to locate individuals" in card and "not calibrated" in card


def test_onnx_outputs_normalized_embeddings(exported, tmp_path):
    _, src, out, _, _, _ = exported
    import torch

    embedder, _ = d.load_embedder(str(src))
    x = torch.randn(2, 3, 32, 32)
    got = eo.ort_session(out / d.FP32_FILE).run([d.OUTPUT_NAME], {eo.INPUT_NAME: x.numpy()})[0]
    np.testing.assert_allclose(got, embedder(x).numpy(), atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1, atol=1e-5)
    for i, (w, h) in enumerate([(64, 40), (30, 50)]):
        gradient_image(w, h).save(tmp_path / f"{i}.png")
    rep, top1 = d.verify(str(src), out, eo.list_images(tmp_path))
    assert rep["fp32"]["top1_agreement"] == 1.0 and rep["fp32"]["top1_km_max"] == 0.0
    assert set(top1) == {"torch", "fp32", "int8"} and len(top1["fp32"]) == 2


def test_geo_accuracy_paired():
    import pandas as pd

    cells = {"centroid_lat": [0.0, 10.0, 0.0], "centroid_lon": [0.0, 0.0, 90.0]}
    meta = pd.DataFrame({"latitude": [0.0, 0.0, 10.0, 0.0], "longitude": [0.0, 0.0, 0.0, 90.0],
                         "creator_id": [1, 1, 2, 3]})
    top1 = {"torch": np.array([0, 0, 1, 2]), "fp32": np.array([0, 0, 1, 2]), "int8": np.array([0, 1, 1, 2])}
    r = d.geo_accuracy(top1, cells, meta, n_boot=200)
    assert r["fp32"]["geoscore_mean"] == pytest.approx(5000) and r["fp32"]["km_median"] == 0
    loss = 5000 * (1 - np.exp(-1111.95 / 1492.7)) / 4  # one image moved 10 degrees of latitude
    assert r["int8_minus_fp32"]["geoscore_diff_mean"] == pytest.approx(-loss, rel=1e-3)
    assert r["int8_minus_fp32"]["n_images_top1_differs"] == 1
    assert r["fp32_minus_torch"]["geoscore_diff_ci95"] == (0.0, 0.0)
    lo, hi = r["int8_minus_fp32"]["geoscore_diff_ci95"]
    assert lo <= r["int8_minus_fp32"]["geoscore_diff_mean"] <= hi <= 0


def test_refuses_wrong_sources(exported, tmp_path):
    tmp, src, _, _, _, _ = exported
    with pytest.raises(SystemExit, match="pre-registered"):
        d.export(str(src), tmp / "head.npy", tmp / "cells.json", tmp_path, "0" * 64)
    (tmp_path / "head.npy").write_bytes((tmp / "head.npy").read_bytes() + b"x")
    (tmp_path / "probe.json").write_text((tmp / "probe.json").read_text())
    with pytest.raises(SystemExit, match="probe.json"):
        d.normalized_head(tmp_path / "head.npy")
