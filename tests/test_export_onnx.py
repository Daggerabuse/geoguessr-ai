"""export_onnx end to end on a random tiny CLIP (no download). Skipped without torch/onnx."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from geoguessr_ai import export_onnx as e
from tests.conftest import gradient_image


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("onnx")
    src = e.write_tiny_folder(out / "src", n_classes=33, image_size=32)
    info = e.export(src, out)
    return src, out, info


def test_files_and_info(exported):
    _, out, info = exported
    assert (out / e.FP32_FILE).is_file() and (out / e.INT8_FILE).is_file()
    assert info["output_shape"] == ["N", 33] and info["input_shape"] == ["N", 3, 32, 32]
    assert json.loads((out / "export_info.json").read_text()) == info
    assert not list(out.glob("*.data"))  # single-file graphs, no external data


def test_one_graph_with_33_logits_and_dynamic_batch(exported):
    import onnx

    _, out, _ = exported
    m = onnx.load(str(out / e.FP32_FILE))
    assert [i.name for i in m.graph.input] == [e.INPUT_NAME]
    assert [o.name for o in m.graph.output] == [e.OUTPUT_NAME]
    dims = m.graph.output[0].type.tensor_type.shape.dim
    assert dims[0].dim_param and dims[1].dim_value == 33
    sess = e.ort_session(out / e.FP32_FILE)
    for n in (1, 3, 5):
        y = sess.run(None, {e.INPUT_NAME: np.zeros((n, 3, 32, 32), np.float32)})[0]
        assert y.shape == (n, 33)


def test_int8_is_quantized(exported):
    import onnx

    _, out, _ = exported
    m = onnx.load(str(out / e.INT8_FILE))
    weights = {i.name for i in m.graph.initializer}
    assert "MatMulNBits" in {n.op_type for n in m.graph.node}
    # what is left as MatMul multiplies two activations (attention), not a stored weight
    assert not [n.name for n in m.graph.node if n.op_type == "MatMul" and n.input[1] in weights]


def test_dynamic_int8_method(exported, tmp_path):
    import onnx

    _, out, _ = exported
    s = e.quantize_int8(out / e.FP32_FILE, tmp_path / "dyn.onnx", method="dynamic")
    assert s["method"] == "onnxruntime quantize_dynamic"
    ops = {n.op_type for n in onnx.load(str(tmp_path / "dyn.onnx")).graph.node}
    assert "DynamicQuantizeLinear" in ops
    with pytest.raises(ValueError):
        e.quantize_int8(out / e.FP32_FILE, tmp_path / "x.onnx", method="nope")


def test_onnx_matches_torch(exported, tmp_path):
    import torch

    src, out, _ = exported
    model, cfg, _ = e.load_folder(src)
    for i, (w, h) in enumerate([(64, 40), (30, 50)]):
        gradient_image(w, h).save(tmp_path / f"{i}.png")
    rep = e.verify(src, out, e.list_images(tmp_path))
    assert rep["fp32"]["logit_max_abs_diff"] < 1e-3 and rep["fp32"]["top1_agreement"] == 1.0
    x = torch.randn(2, 3, 32, 32)
    want = model(x).numpy()
    got = e.ort_session(out / e.FP32_FILE).run(None, {e.INPUT_NAME: x.numpy()})[0]
    np.testing.assert_allclose(got, want, atol=1e-3)
    # logits are logit_scale * cosine similarity
    assert np.abs(want).max() <= cfg["logit_scale"] + 1e-3


def test_parity_stats_identical_and_different():
    a = np.array([[1.0, 2.0, 3.0, 0.0, 0.5, 0.1], [3.0, 2.0, 1.0, 0.0, 0.5, 0.1]])
    s = e.parity_stats(a, a)
    assert s["logit_max_abs_diff"] == 0 and s["top1_agreement"] == 1.0
    b = a.copy()
    b[0, 0] = 10.0
    s = e.parity_stats(a, b)
    assert s["top1_agreement"] == 0.5 and s["logit_max_abs_diff"] == 9.0


def test_parse_args(tmp_path):
    with pytest.raises(SystemExit):
        e.parse_args(["--model", str(tmp_path)])  # no inference_config.json
    a = e.parse_args(["--tiny"])
    assert a.tiny and str(a.out).replace("\\", "/") == "models/onnx"
