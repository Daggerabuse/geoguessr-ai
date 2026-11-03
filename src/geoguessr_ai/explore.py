"""Exploration: country distribution, image sizes, random samples.

Usage: uv run python -m geoguessr_ai.explore
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

SEED = 42
DATA = Path("data")


def show_counts(title: str, s: pd.Series, k: int = 15) -> None:
    vc = s.value_counts()
    print(f"\n== {title}: {len(vc)} countries, {len(s)} rows ==")
    print("Most:\n", vc.head(k).to_string())
    print("Fewest:\n", vc.tail(k).to_string())
    print("Median per country:", vc.median())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    full = pd.read_csv(DATA / "raw" / "test.csv", dtype={"id": str})
    print("Columns:", list(full.columns))
    show_counts("test.csv (full test set)", full["country"])

    sample = pd.read_csv(DATA / "sample" / "meta.csv", dtype={"id": str})
    show_counts("sample", sample["country"])

    img_dir = DATA / "sample" / "images"
    files = {p.stem: p for p in img_dir.iterdir()}
    sizes = Counter(Image.open(p).size for p in files.values())
    print("\n== Image sizes (w, h) ==")
    for sz, c in sizes.most_common(10):
        print(sz, c)

    pick = sample[sample["id"].isin(files)].sample(12, random_state=SEED)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, (_, r) in zip(axes.flat, pick.iterrows()):
        ax.imshow(Image.open(files[r["id"]]))
        ax.set_title(f"{r['country']} ({r['latitude']:.2f}, {r['longitude']:.2f})")
        ax.axis("off")
    out = DATA / "figures"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=80)
    print("\nSample grid:", out / "samples.png")


if __name__ == "__main__":
    main()
