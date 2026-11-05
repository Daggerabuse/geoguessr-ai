"""CLIP zero-shot baseline: classification with country-name prompt templates.

Input: prep_data.py output (manifest.csv, classes.json, images/).
Output: results/clip_zeroshot/<split>_{metrics.json,per_country.csv,confusion.csv,
        confusion.png,predictions.csv} and run_config.json.

Usage (Kaggle):
  python -m geoguessr_ai.baseline_clip --root /kaggle/working/subset
  python -m geoguessr_ai.baseline_clip --root ... --model geolocal/StreetCLIP \\
      --out results/streetclip_zeroshot
  python -m geoguessr_ai.baseline_clip --root ... --model geolocal/StreetCLIP --tta \\
      --out results/streetclip_zeroshot_tta

--tta averages the logits of the left/center/right crops (the same crops as
train.py --tta, at the model's own image size); without it, the CLIPProcessor's
resize + center crop is used.

Any HF CLIP model loadable with CLIPModel/CLIPProcessor works via --model.
The StreetCLIP model card recommends no template; the default template is used.
"""

import argparse
import importlib.metadata
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_TEMPLATE = "A street view photo taken in {}."

# Countries whose official pycountry name reads unnaturally in the template.
NAME_OVERRIDES = {
    "US": "the United States",
    "GB": "the United Kingdom",
    "RU": "Russia",
    "KR": "South Korea",
    "KP": "North Korea",
    "CD": "the Democratic Republic of the Congo",
    "CG": "the Republic of the Congo",
    "TZ": "Tanzania",
    "TR": "Turkey",
    "IR": "Iran",
    "SY": "Syria",
    "VN": "Vietnam",
    "LA": "Laos",
    "MD": "Moldova",
    "BO": "Bolivia",
    "VE": "Venezuela",
    "CZ": "the Czech Republic",
    "AE": "the United Arab Emirates",
    "NL": "the Netherlands",
    "PH": "the Philippines",
}


# ----------------------------------------------------------------- helpers


def country_names(codes: list[str]) -> dict[str, str]:
    """ISO-2 code -> name used in the template; falls back to the code itself."""
    import pycountry

    out = {}
    for c in codes:
        if c in NAME_OVERRIDES:
            out[c] = NAME_OVERRIDES[c]
            continue
        rec = pycountry.countries.get(alpha_2=c)
        out[c] = (getattr(rec, "common_name", None) or rec.name) if rec else c
    return out


def build_prompts(codes: list[str], template: str) -> list[str]:
    names = country_names(codes)
    return [template.format(names[c]) for c in codes]


def library_versions() -> dict[str, str | None]:
    """Installed versions of torch/transformers (None if not installed)."""
    out = {}
    for name in ("torch", "transformers"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def evaluate(logits: np.ndarray, labels: np.ndarray, k: int = 5):
    """top-1 / top-k hits, predictions, top-k indices."""
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    pred = topk[:, 0]
    hit1 = pred == labels
    hitk = (topk == labels[:, None]).any(axis=1)
    return hit1, hitk, pred, topk


def confusion(labels: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    return cm


def mean_over_views(logits: np.ndarray, n_views: int) -> np.ndarray:
    """(N*V, C) logits, views of one image consecutive -> (N, C) mean over the V views."""
    if logits.shape[0] % n_views:
        raise ValueError(f"{logits.shape[0]} rows are not a multiple of n_views={n_views}")
    return logits.reshape(-1, n_views, logits.shape[-1]).mean(axis=1)


# ----------------------------------------------------------------- crops
# Shared with train.py (eval_views). No flips: the driving side is a country cue.


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


# ----------------------------------------------------------------- model


def as_embedding(out, kind: str, proj_dim: int):
    """Get the projected embedding tensor from a CLIP output.

    Depending on the transformers version, get_*_features / *ModelWithProjection
    return either a tensor or a ModelOutput. A tensor is used as is; for a
    ModelOutput `<kind>_embeds` is used, otherwise `pooler_output` (projected in
    newer versions). last_hidden_state is never used. Raises if the last
    dimension is not proj_dim (meaning the projection was not applied).
    """
    if kind not in ("text", "image"):
        raise ValueError(f"kind must be 'text' or 'image': {kind!r}")
    if hasattr(out, "shape"):
        emb = out
    else:
        emb = getattr(out, f"{kind}_embeds", None)
        if emb is None:
            emb = getattr(out, "pooler_output", None)
        if emb is None:
            raise ValueError(
                f"{type(out).__name__} has neither {kind}_embeds nor pooler_output"
            )
    if emb.shape[-1] != proj_dim:
        raise ValueError(
            f"{kind} embedding dim is {emb.shape[-1]}, expected projection_dim={proj_dim}: "
            "the projection may not have been applied"
        )
    return emb


class HFClip:
    """Wrapper around transformers CLIP; torch/transformers are imported lazily."""

    def __init__(self, name: str, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # pass dtype afterwards: the kwarg name changed across transformers versions
        self.model = CLIPModel.from_pretrained(name).to(self.device, dtype=dtype).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.logit_scale = self.model.logit_scale.exp().item()
        self.proj_dim = self.model.config.projection_dim
        self.image_size = self.model.config.vision_config.image_size

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            t = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            out = as_embedding(self.model.get_text_features(**t), "text", self.proj_dim)
            return out.float().cpu().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        with self.torch.no_grad():
            px = self.processor(images=images, return_tensors="pt")["pixel_values"]
            px = px.to(self.device, dtype=self.model.dtype)
            out = as_embedding(self.model.get_image_features(pixel_values=px), "image", self.proj_dim)
            return out.float().cpu().numpy()


def _open(path: Path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (FileNotFoundError, OSError):
        return None


def image_logits(enc, text_emb, paths, batch_size, workers, tta: bool = False):
    """Image embeddings in batches -> logits; unreadable images are skipped.

    tta: each image becomes 3 crops (left/center/right, see eval_crops) at the
    encoder's image size; the logits of the 3 crops are averaged (as in train.py).

    Returns: (logits, valid_mask)
    """
    ok = np.zeros(len(paths), dtype=bool)
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(paths), batch_size):
            chunk = paths[s : s + batch_size]
            ims = list(ex.map(_open, chunk))
            good = [i for i, im in enumerate(ims) if im is not None]
            ok[[s + i for i in good]] = True
            if good and tta:
                views = [v for i in good for v in eval_crops(ims[i], enc.image_size, True)]
                emb = normalize(enc.encode_images(views))
                rows.append(mean_over_views(enc.logit_scale * emb @ text_emb.T, 3))
            elif good:
                emb = normalize(enc.encode_images([ims[i] for i in good]))
                rows.append(enc.logit_scale * emb @ text_emb.T)
            print(f"  {min(s + batch_size, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    n_cls = text_emb.shape[0]
    return (np.concatenate(rows) if rows else np.zeros((0, n_cls))), ok


# ----------------------------------------------------------------- run


def save_confusion_png(cm: np.ndarray, classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(6, 0.3 * len(classes))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_split_results(
    out: Path,
    split: str,
    ids,
    labels: np.ndarray,
    logits: np.ndarray,
    classes: list[str],
    n_missing: int = 0,
) -> dict:
    """Metrics/files for one split. baseline_clip and train share this format."""
    sub = pd.DataFrame({"id": list(ids)})
    hit1, hitk, pred, topk = evaluate(logits, labels)

    per = (
        pd.DataFrame({"country": [classes[i] for i in labels], "top1": hit1, "top5": hitk})
        .groupby("country")
        .agg(n=("top1", "size"), top1=("top1", "mean"), top5=("top5", "mean"))
        .reset_index()
    )
    cm = confusion(labels, pred, len(classes))
    metrics = {
        "split": split,
        "n": len(sub),
        "n_missing": n_missing,
        "top1": float(hit1.mean()) if len(sub) else None,
        "top5": float(hitk.mean()) if len(sub) else None,
        "macro_top1": float(per["top1"].mean()) if len(per) else None,
        "macro_top5": float(per["top5"].mean()) if len(per) else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    per.to_csv(out / f"{split}_per_country.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"{split}_confusion.csv")
    save_confusion_png(cm, classes, out / f"{split}_confusion.png")
    pd.DataFrame(
        {
            "id": sub["id"],
            "true": [classes[i] for i in labels],
            "pred": [classes[i] for i in pred],
            "top5": [" ".join(classes[j] for j in row) for row in topk],
        }
    ).to_csv(out / f"{split}_predictions.csv", index=False)
    print(f"[{split}] {metrics}")
    return metrics


def run_split(enc, text_emb, df, classes, root, split, out, batch_size, workers,
              tta: bool = False) -> dict:
    sub = df[df["split"] == split].reset_index(drop=True)
    paths = [root / p for p in sub["path"]]
    logits, ok = image_logits(enc, text_emb, paths, batch_size, workers, tta=tta)
    n_missing = int((~ok).sum())
    if n_missing:
        print(f"WARNING [{split}]: {n_missing} images could not be read, skipped")
    sub = sub[ok].reset_index(drop=True)
    return write_split_results(
        out, split, sub["id"], sub["label"].to_numpy(), logits, classes, n_missing
    )


def run(args: argparse.Namespace, enc=None) -> dict:
    np.random.seed(args.seed)
    root = Path(args.root)
    classes = json.loads((root / "classes.json").read_text())
    df = pd.read_csv(root / "manifest.csv", dtype={"id": str})
    if enc is None:
        enc = HFClip(args.model)

    prompts = build_prompts(classes, args.template)
    print("Example prompts:", prompts[:3])
    text_emb = normalize(enc.encode_texts(prompts))

    tta = getattr(args, "tta", False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "template": args.template,
                "seed": args.seed,
                "tta": tta,
                "image_size": getattr(enc, "image_size", None),
                "device": getattr(enc, "device", "n/a"),
                "versions": library_versions(),
                "classes": classes,
                "prompts": prompts,
            },
            indent=2,
        )
    )
    return {
        s: run_split(enc, text_emb, df, classes, root, s, out, args.batch_size, args.workers, tta)
        for s in args.splits
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/subset")
    ap.add_argument("--out", default="results/clip_zeroshot")
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", action="store_true",
                    help="average the logits of 3 crops: left/center/right (as train.py --tta)")
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
