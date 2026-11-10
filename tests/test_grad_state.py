"""V1 inference/export must not change torch's global grad mode, and their outputs must stay the same.

Regression tests for the module-level torch.set_grad_enabled(False) that app/app.py and
export_onnx.load_folder used to run: it silently disabled training code in the same process.
"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from geoguessr_ai import export_onnx as e
from tests.conftest import gradient_image

APP_DIR = Path(__file__).parent.parent / "app"


def reference_logits(folder: Path, x: np.ndarray) -> np.ndarray:
    """logit_scale * cos(image embedding, head), computed directly from the exported files."""
    import json

    from safetensors.torch import load_file
    from transformers import CLIPVisionModelWithProjection

    cfg = json.loads((folder / "inference_config.json").read_text())
    vision = CLIPVisionModelWithProjection.from_pretrained(folder).float().eval()
    head = torch.nn.functional.normalize(load_file(folder / "head.safetensors")["head"].float(), dim=-1)
    with torch.no_grad():
        emb = torch.nn.functional.normalize(vision(pixel_values=torch.from_numpy(x)).image_embeds, dim=-1)
        return (cfg["logit_scale"] * emb @ head.T).numpy()


@pytest.fixture
def tiny(tmp_path):
    return e.write_tiny_folder(tmp_path / "tiny", n_classes=33, image_size=32)


def test_load_folder_keeps_grad_enabled_and_logits(tiny):
    assert torch.is_grad_enabled()
    model, _, _ = e.load_folder(tiny)
    assert torch.is_grad_enabled()
    x = np.random.default_rng(0).normal(size=(2, 3, 32, 32)).astype(np.float32)
    got = model(torch.from_numpy(x))
    assert not got.requires_grad  # callers (verify, webdemo) call .numpy() on it
    np.testing.assert_allclose(got.numpy(), reference_logits(tiny, x), atol=1e-5)


def test_export_and_verify_keep_grad_enabled(tiny, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    out = tmp_path / "onnx"
    e.export(tiny, out)
    assert torch.is_grad_enabled()
    gradient_image(64, 40).save(tmp_path / "a.png")
    rep = e.verify(tiny, out, [tmp_path / "a.png"])
    assert torch.is_grad_enabled()
    assert rep["fp32"]["top1_agreement"] == 1.0


def test_app_predict_keeps_grad_enabled_and_matches_reference(tiny, monkeypatch):
    monkeypatch.syspath_prepend(str(APP_DIR))
    monkeypatch.setitem(sys.modules, "gradio", mock.MagicMock())  # the UI is not under test
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(tiny)):
        spec = importlib.util.spec_from_file_location("app_under_test", APP_DIR / "app.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
    assert torch.is_grad_enabled()
    im = gradient_image(64, 40)
    got = app.predict(im)
    assert torch.is_grad_enabled()

    x = app.preprocess(im, app.cfg, tta=True)
    logits = reference_logits(tiny, x).mean(0)
    want = app.top_k(app.softmax(logits.astype(np.float64)), app.labels, k=5)
    assert list(got) == list(want)
    np.testing.assert_allclose(list(got.values()), list(want.values()), atol=1e-6)
