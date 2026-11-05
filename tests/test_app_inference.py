"""app/inference.py must preprocess exactly like the package (the Space cannot import it)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoguessr_ai import export_hf, train
from tests.conftest import gradient_image

APP_INFERENCE = Path(__file__).parent.parent / "app" / "inference.py"


@pytest.fixture(scope="module")
def inf():
    spec = importlib.util.spec_from_file_location("app_inference", APP_INFERENCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = export_hf.inference_config(
    {"classes": ["DE"], "model": "m", "template": "{}", "img_size": 64}, 100.0
)


@pytest.mark.parametrize("size", [(455, 256), (200, 400), (64, 64), (1000, 333)])
@pytest.mark.parametrize("tta", [False, True])
def test_preprocess_matches_train_eval_views(inf, size, tta):
    im = gradient_image(*size)
    got = inf.preprocess(im, CFG, tta=tta)
    want = np.stack(train.eval_views(im, 64, tta))
    assert got.shape == want.shape and got.dtype == np.float32
    assert np.array_equal(got, want)


def test_preprocess_converts_to_rgb(inf):
    im = Image.fromarray(np.zeros((80, 120), np.uint8))  # grayscale
    assert inf.preprocess(im, CFG).shape == (3, 3, 64, 64)


def test_softmax_and_top_k(inf):
    big = inf.softmax(np.array([1.0, 1000.0]))  # no overflow
    assert np.isfinite(big).all() and big.sum() == pytest.approx(1.0)
    p = inf.softmax(np.array([1.0, 3.0, 2.0, 5.0]))
    assert p.sum() == pytest.approx(1.0)
    top = inf.top_k(p, ["a", "b", "c", "d"], k=3)
    assert list(top) == ["d", "b", "c"]
    assert len(inf.top_k(p, list("abcd"), k=10)) == 4
