"""Shared fixtures: small synthetic images/manifests, no dependency on data/."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

CLASSES = ["AU", "BR", "CA"]


def write_jpeg(path: Path, w: int = 96, h: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=90)


def gradient_image(w: int = 455, h: int = 256) -> Image.Image:
    """Horizontal gradient: left is dark, right is bright (makes flips detectable)."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return Image.fromarray(np.tile(row, (h, 1))[..., None].repeat(3, axis=2))


@pytest.fixture
def subset_root(tmp_path: Path) -> Path:
    """A prep_data-style subset: manifest.csv, classes.json, images/{train,test}/."""
    root = tmp_path / "subset"
    rows = []
    for ci, c in enumerate(CLASSES):
        for split, n in (("train", 6), ("val", 3), ("test", 3)):
            for i in range(n):
                id_ = f"{c}_{split}_{i}"
                source = "test" if split == "test" else "train"
                path = f"images/{source}/{id_}.jpg"
                write_jpeg(root / path, seed=len(rows))
                rows.append(
                    {
                        "id": id_, "latitude": 0.0, "longitude": 0.0, "country": c,
                        "sequence": f"{c}_{split}_{i // 2}", "split": split,
                        "source": source, "label": ci, "path": path,
                    }
                )
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    (root / "classes.json").write_text(json.dumps(CLASSES))
    return root
