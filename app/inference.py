"""Preprocessing and post-processing for the demo; numpy + PIL only (no torch, no gradio).

A copy of the package's evaluation preprocessing (baseline_clip.eval_crops +
train.to_tensor_array), because a Space cannot import the geoguessr_ai package.
tests/test_app_inference.py checks that both give identical arrays.
No flips: the driving side is a country cue.
"""

import numpy as np
from PIL import Image


def resize_short(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    s = size / min(w, h)
    return im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)


def eval_crops(im: Image.Image, size: int, tta: bool) -> list[Image.Image]:
    """Short side `size` + center crop. tta: left/center/right (top/center/bottom) along the long axis."""
    im = resize_short(im, size)
    w, h = im.size
    if w >= h:
        offs = [(0, 0), ((w - size) // 2, 0), (w - size, 0)] if tta else [((w - size) // 2, 0)]
        offs = [(x, (h - size) // 2) for x, _ in offs]
    else:
        offs = [(0, 0), (0, (h - size) // 2), (0, h - size)] if tta else [(0, (h - size) // 2)]
        offs = [((w - size) // 2, y) for _, y in offs]
    return [im.crop((x, y, x + size, y + size)) for x, y in offs]


def to_array(im: Image.Image, mean, std) -> np.ndarray:
    """PIL RGB -> normalized float32 CHW."""
    x = np.asarray(im, dtype=np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((x - mean) / std).transpose(2, 0, 1).copy()


def preprocess(im: Image.Image, cfg: dict, tta: bool = True) -> np.ndarray:
    """(n_crops, 3, S, S) float32 for one image, following inference_config.json."""
    im = im.convert("RGB")
    crops = eval_crops(im, int(cfg["image_size"]), tta)
    return np.stack([to_array(c, cfg["mean"], cfg["std"]) for c in crops])


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def top_k(probs: np.ndarray, labels: list[str], k: int = 5) -> dict[str, float]:
    """The k most likely labels -> probability, most likely first."""
    order = np.argsort(-probs)[:k]
    return {labels[i]: float(probs[i]) for i in order}
