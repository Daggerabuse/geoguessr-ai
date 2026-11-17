"""web/preprocess.js must preprocess exactly like the Python evaluation code.

The JS runs in node (nodejs-wheel-binaries, a dev dependency). Both sides get the same
decoded pixels, so this checks resize + crops + normalization, not JPEG decoding (the
browser's decoder is a separate question, see NOTES_ONNX.md).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoguessr_ai.baseline_clip import resize_short
from geoguessr_ai.embed import clip_center_crop
from geoguessr_ai.export_hf import CLIP_MEAN, CLIP_STD
from geoguessr_ai.train import eval_views, to_tensor_array
from tests.conftest import gradient_image

nodejs_wheel = pytest.importorskip("nodejs_wheel")

ROOT = Path(__file__).parent.parent
PREPROCESS_JS = ROOT / "web" / "preprocess.js"
CLI = Path(__file__).parent / "js" / "preprocess_cli.mjs"
TOPCELLS_CLI = Path(__file__).parent / "js" / "topcells_cli.mjs"
SAMPLE = ROOT / "data" / "sample" / "images"


def noise_image(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def checker_image(w: int, h: int) -> Image.Image:
    """Hard edges: largest overshoot of the bicubic kernel, so uint8 clipping is exercised."""
    yy, xx = np.mgrid[0:h, 0:w]
    a = (((xx // 3) + (yy // 5)) % 2 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([a, 255 - a, a // 2], axis=-1))


def cases() -> list[tuple[str, Image.Image, int]]:
    out = [(f"gradient{w}x{h}", gradient_image(w, h), s)
           for (w, h), s in [((455, 256), 224), ((200, 400), 64), ((64, 64), 64),
                             ((1000, 333), 224), ((224, 224), 224), ((225, 224), 224),
                             ((3000, 4000), 224)]]
    out += [(f"noise{w}x{h}", noise_image(w, h, seed), s)
            for seed, ((w, h), s) in enumerate([((317, 211), 224), ((100, 80), 224),
                                                 ((641, 479), 224), ((97, 1301), 64),
                                                 ((400, 1000), 224)])]
    out += [("checker1024x768", checker_image(1024, 768), 224)]
    rgba = noise_image(300, 200, 9).convert("RGBA")
    rgba.putalpha(255)
    out += [("rgba300x200", rgba, 224)]
    if SAMPLE.is_dir():  # real photos if data/ is present (not in CI)
        for p in sorted(SAMPLE.glob("*.jpg"))[:8]:
            with Image.open(p) as im:
                out.append((p.stem, im.convert("RGB"), 224))
    return out


def cfg(size: int) -> dict:
    return {"image_size": size, "mean": CLIP_MEAN, "std": CLIP_STD}


@pytest.fixture(scope="module")
def js_results(tmp_path_factory):
    """Run node once for all cases; return {(name, tta): (float32 array, resized uint8)}."""
    tmp = tmp_path_factory.mktemp("webjs")
    by_size: dict[int, list] = {}
    for name, im, size in cases():
        for tta in (False, True):
            arr = np.asarray(im)
            raw = tmp / f"{name}.raw"
            raw.write_bytes(np.ascontiguousarray(arr).tobytes())
            by_size.setdefault(size, []).append(
                {"name": name, "raw": str(raw), "w": im.width, "h": im.height,
                 "channels": arr.shape[2], "tta": tta, "out": str(tmp / f"{name}_{tta}.f32")})
    results = {}
    for size, jobs in by_size.items():
        job = tmp / f"job{size}.json"
        job.write_text(json.dumps({"cfg": cfg(size), "jobs": jobs}))
        proc = nodejs_wheel.node([str(CLI), str(job), str(PREPROCESS_JS)],
                                 return_completed_process=True, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        for j in jobs:
            data = np.fromfile(j["out"], dtype=np.float32)
            resized = np.fromfile(j["out"] + ".resized", dtype=np.uint8)
            results[(j["name"], j["tta"])] = (data.reshape(-1, 3, size, size), resized, size)
    return results


@pytest.mark.parametrize("tta", [False, True])
@pytest.mark.parametrize("case", cases(), ids=lambda c: c[0])
def test_js_preprocess_matches_python(js_results, case, tta):
    name, im, size = case
    got, resized, _ = js_results[(name, tta)]
    rgb = im.convert("RGB")
    want_resized = np.asarray(resize_short(rgb, size))
    assert resized.size == want_resized.size, "resized image size differs"
    diff_levels = np.abs(resized.reshape(want_resized.shape).astype(int) - want_resized.astype(int))
    assert diff_levels.max() == 0, f"{name}: max pixel diff {diff_levels.max()} levels"
    want = np.stack(eval_views(rgb, size, tta))
    assert got.shape == want.shape
    assert np.array_equal(got, want), f"{name}: max abs diff {np.abs(got - want).max()}"


@pytest.fixture(scope="module")
def js_truncate(tmp_path_factory):
    """The V3 page's preprocessing (cfg.resize "truncate", one center crop): {(name, size): array}."""
    tmp = tmp_path_factory.mktemp("webjs_trunc")
    results = {}
    for size in (64, 224, 336):
        jobs = []
        for name, im, _ in cases():
            arr = np.asarray(im)
            raw = tmp / f"{name}.raw"
            raw.write_bytes(np.ascontiguousarray(arr).tobytes())
            jobs.append({"name": name, "raw": str(raw), "w": im.width, "h": im.height,
                         "channels": arr.shape[2], "tta": False, "out": str(tmp / f"{name}_{size}.f32")})
        job = tmp / f"job{size}.json"
        job.write_text(json.dumps({"cfg": {**cfg(size), "resize": "truncate"}, "jobs": jobs}))
        proc = nodejs_wheel.node([str(CLI), str(job), str(PREPROCESS_JS)],
                                 return_completed_process=True, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        for j in jobs:
            results[(j["name"], size)] = np.fromfile(j["out"], dtype=np.float32).reshape(-1, 3, size, size)
    return results


@pytest.mark.parametrize("size", [64, 224, 336])
@pytest.mark.parametrize("case", cases(), ids=lambda c: c[0])
def test_js_truncate_matches_clip_center_crop(js_truncate, case, size):
    name, im, _ = case
    want = to_tensor_array(clip_center_crop(im.convert("RGB"), size))[None]
    got = js_truncate[(name, size)]
    assert got.shape == want.shape
    assert np.array_equal(got, want), f"{name}: max abs diff {np.abs(got - want).max()}"


def test_js_top_cells_match_numpy(tmp_path):
    """Small integers keep every dot product exact, so ties are real ties (as identical head rows
    after a zero-shot init): both sides must rank them by cell id."""
    rng = np.random.default_rng(0)
    head = rng.integers(-3, 4, size=(300, 8)).astype(np.float32)
    head[[10, 50, 120, 200]] = 3  # tied rows, the largest possible dot product with emb
    emb = rng.integers(1, 3, size=8).astype(np.float32)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"emb": emb.tolist(), "head": head.ravel().tolist(), "scale": 0.5, "k": 5}))
    proc = nodejs_wheel.node([str(TOPCELLS_CLI), str(job), str(PREPROCESS_JS)],
                             return_completed_process=True, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    logits = 0.5 * (head.astype(np.float64) @ emb)
    order = np.argsort(-logits, kind="stable")[:5]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    assert [c for c, _ in got] == order.tolist()
    np.testing.assert_allclose([q for _, q in got], p[order], rtol=1e-12)
    assert order[:4].tolist() == [10, 50, 120, 200]  # the tie is at the top
