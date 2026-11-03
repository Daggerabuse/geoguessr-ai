"""CLIP zero-shot baseline: ülke adı şablonlarıyla sınıflandırma.

Girdi: prep_data.py çıktısı (manifest.csv, classes.json, images/).
Çıktı: results/clip_zeroshot/<split>_{metrics.json,per_country.csv,confusion.csv,
       confusion.png,predictions.csv} ve run_config.json.

Kullanım (Kaggle):
  python -m geoguessr_ai.baseline_clip --root /kaggle/working/osv5m_subset
  python -m geoguessr_ai.baseline_clip --root ... --model geolocal/StreetCLIP \\
      --out results/streetclip_zeroshot

CLIPModel/CLIPProcessor ile yüklenen her HF CLIP modeli --model ile çalışır.
StreetCLIP model kartı bir şablon önermiyor; varsayılan şablon kullanılır.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_TEMPLATE = "A street view photo taken in {}."

# pycountry'nin resmi adı şablonda doğal okunmayan ülkeler.
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


# ----------------------------------------------------------------- yardımcılar


def country_names(codes: list[str]) -> dict[str, str]:
    """ISO-2 kod -> şablonda kullanılacak ad. Bulunamazsa kodun kendisi."""
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


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def evaluate(logits: np.ndarray, labels: np.ndarray, k: int = 5):
    """top-1 / top-k doğruluk, tahminler, top-k indeksleri."""
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


# ----------------------------------------------------------------- model


class HFClip:
    """transformers CLIP sarmalayıcısı; torch/transformers tembel import edilir."""

    def __init__(self, name: str, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # dtype'ı sonradan ver: transformers sürümleri arasında kwarg adı değişti
        self.model = CLIPModel.from_pretrained(name).to(self.device, dtype=dtype).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.logit_scale = float(self.model.logit_scale.exp())

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            t = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
            return self.model.get_text_features(**t).float().cpu().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        with self.torch.no_grad():
            px = self.processor(images=images, return_tensors="pt")["pixel_values"]
            px = px.to(self.device, dtype=self.model.dtype)
            return self.model.get_image_features(pixel_values=px).float().cpu().numpy()


def _open(path: Path):
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (FileNotFoundError, OSError):
        return None


def image_logits(enc, text_emb, paths, batch_size, workers):
    """Batch'ler halinde görüntü gömmeleri -> logits; okunamayanlar atlanır.

    Döner: (logits, geçerli_maske)
    """
    ok = np.zeros(len(paths), dtype=bool)
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(paths), batch_size):
            chunk = paths[s : s + batch_size]
            ims = list(ex.map(_open, chunk))
            good = [i for i, im in enumerate(ims) if im is not None]
            ok[[s + i for i in good]] = True
            if good:
                emb = normalize(enc.encode_images([ims[i] for i in good]))
                rows.append(enc.logit_scale * emb @ text_emb.T)
            print(f"  {min(s + batch_size, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    n_cls = text_emb.shape[0]
    return (np.concatenate(rows) if rows else np.zeros((0, n_cls))), ok


# ----------------------------------------------------------------- çalıştırma


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
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    fig.colorbar(im, ax=ax, fraction=0.046, label="satır oranı")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_split(enc, text_emb, df, classes, root, split, out, batch_size, workers) -> dict:
    sub = df[df["split"] == split].reset_index(drop=True)
    paths = [root / p for p in sub["path"]]
    logits, ok = image_logits(enc, text_emb, paths, batch_size, workers)
    n_missing = int((~ok).sum())
    if n_missing:
        print(f"UYARI [{split}]: {n_missing} görüntü okunamadı, atlandı")
    sub = sub[ok].reset_index(drop=True)
    labels = sub["label"].to_numpy()
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
        "n": int(len(sub)),
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


def run(args: argparse.Namespace, enc=None) -> dict:
    np.random.seed(args.seed)
    root = Path(args.root)
    classes = json.loads((root / "classes.json").read_text())
    df = pd.read_csv(root / "manifest.csv", dtype={"id": str})
    if enc is None:
        enc = HFClip(args.model)

    prompts = build_prompts(classes, args.template)
    print("Örnek şablonlar:", prompts[:3])
    text_emb = normalize(enc.encode_texts(prompts))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "template": args.template,
                "seed": args.seed,
                "device": getattr(enc, "device", "n/a"),
                "classes": classes,
                "prompts": prompts,
            },
            indent=2,
        )
    )
    return {
        s: run_split(enc, text_emb, df, classes, root, s, out, args.batch_size, args.workers)
        for s in args.splits
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/subset")
    ap.add_argument("--out", default="results/clip_zeroshot")
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
