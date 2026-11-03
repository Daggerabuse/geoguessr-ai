"""CLIP ViT-B/16 fine-tuning: linear probe (LP) followed by full fine-tuning (FT).

Flow:
  1. The classification head is initialised from country-name text embeddings (the
     zero-shot starting point).
  2. LP: the backbone is frozen, image features are extracted once, only the head is trained.
  3. FT: the whole model is trained with a low lr (AMP fp16). NO horizontal flip.
  4. Best checkpoint (val top-1) goes to models/<run>/best.pt; results are written to
     results/<run>/ in the same file format as baseline_clip (LP results in
     results/<run>/probe/).

The test split is evaluated only after training, with the best checkpoint
(for LP too: the head selected by val).

Usage (Kaggle):
  python -m geoguessr_ai.train --root /kaggle/working/subset --smoke   # quick check first
  python -m geoguessr_ai.train --root /kaggle/working/subset [--tta]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from geoguessr_ai.baseline_clip import (
    DEFAULT_TEMPLATE,
    as_embedding,
    build_prompts,
    evaluate,
    library_versions,
    write_split_results,
)

DEFAULT_MODEL = "openai/clip-vit-base-patch16"
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ----------------------------------------------------------------- image transforms
# PIL + numpy only (no torchvision needed). Horizontal/vertical flips are deliberately
# absent: the side of the road traffic drives on is a country cue.


def to_tensor_array(im: Image.Image) -> np.ndarray:
    """PIL RGB -> normalized float32 CHW."""
    x = np.asarray(im, dtype=np.float32) / 255.0
    return ((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1).copy()


def resize_short(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    s = size / min(w, h)
    return im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)


def eval_views(im: Image.Image, size: int, tta: bool) -> list[np.ndarray]:
    """Short side `size` + center crop. tta: left/center/right (top/center/bottom) along the long axis."""
    im = resize_short(im, size)
    w, h = im.size
    if w >= h:
        offs = [(0, 0), ((w - size) // 2, 0), (w - size, 0)] if tta else [((w - size) // 2, 0)]
        offs = [(x, (h - size) // 2) for x, _ in offs]
    else:
        offs = [(0, 0), (0, (h - size) // 2), (0, h - size)] if tta else [(0, (h - size) // 2)]
        offs = [((w - size) // 2, y) for _, y in offs]
    return [to_tensor_array(im.crop((x, y, x + size, y + size))) for x, y in offs]


def train_view(
    im: Image.Image,
    size: int,
    rng: np.random.Generator,
    jitter: bool = True,
    erase_p: float = 0.25,
    blur_p: float = 0.1,
) -> np.ndarray:
    """Random near-square crop (~49-100% of the short-side square) + light color jitter/blur/erasing."""
    w, h = im.size
    c = min(w, h) * rng.uniform(0.7, 1.0)
    r = math.exp(rng.uniform(math.log(0.85), math.log(1.18)))
    cw, ch = min(w, c * math.sqrt(r)), min(h, c / math.sqrt(r))
    x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
    im = im.resize((size, size), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
    if jitter:
        # no hue shift: road-line colors (yellow/white) are a cue
        for enh, mag in ((ImageEnhance.Brightness, 0.2), (ImageEnhance.Contrast, 0.2),
                         (ImageEnhance.Color, 0.15)):
            im = enh(im).enhance(rng.uniform(1 - mag, 1 + mag))
    if rng.random() < blur_p:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    x = to_tensor_array(im)
    if rng.random() < erase_p:
        area = rng.uniform(0.02, 0.08) * size * size
        asp = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        eh, ew = min(size, int(math.sqrt(area / asp))), min(size, int(math.sqrt(area * asp)))
        ey, ex = int(rng.integers(0, size - eh + 1)), int(rng.integers(0, size - ew + 1))
        x[:, ey : ey + eh, ex : ex + ew] = 0.0  # mean color in normalized space
    return x


class GeoDataset:
    """Image dataset from manifest rows (id, path, label); does not depend on torch.

    train=True: random augmentation, (3,H,W). Otherwise: (n_views,3,H,W).
    Images missing on disk are skipped; their count is kept in n_missing.
    """

    def __init__(self, root: Path, df: pd.DataFrame, size: int, train: bool,
                 tta: bool = False, seed: int = 42):
        paths = [Path(root) / p for p in df["path"]]
        ok = np.array([p.exists() for p in paths], dtype=bool)
        self.n_missing = int((~ok).sum())
        if self.n_missing:
            print(f"WARNING: {self.n_missing} images not found, skipped")
        self.df = df[ok].reset_index(drop=True)
        self.paths = [p for p, k in zip(paths, ok) if k]
        self.labels = self.df["label"].to_numpy()
        self.size, self.train, self.tta, self.seed, self.epoch = size, train, tta, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            if self.train:
                rng = np.random.default_rng([self.seed, self.epoch, i])
                x = train_view(im, self.size, rng)
            else:
                x = np.stack(eval_views(im, self.size, self.tta))
        return x, int(self.labels[i])


# ----------------------------------------------------------------- program logic (no torch)


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup + cosine decay (to 0)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class EarlyStopper:
    """Stops when the metric (higher is better) does not improve for `patience` epochs."""

    def __init__(self, patience: int):
        self.patience, self.best, self.bad = patience, -math.inf, 0

    def update(self, metric: float) -> bool:
        """Returns True if this epoch is the best so far."""
        if metric > self.best:
            self.best, self.bad = metric, 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


# ----------------------------------------------------------------- torch part


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backbone(args, n_classes: int, prompts: list[str]):
    """(vision_model, img_size, text_emb[C,D]). --smoke: no download, tiny random model."""
    import torch

    if args.smoke:
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        cfg = CLIPVisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, image_size=32, patch_size=16, projection_dim=16,
        )
        vision = CLIPVisionModelWithProjection(cfg)
        g = torch.Generator().manual_seed(args.seed)
        text = torch.randn(n_classes, cfg.projection_dim, generator=g)
        return vision, cfg.image_size, text

    from transformers import (
        AutoTokenizer,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )

    vision = CLIPVisionModelWithProjection.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    text_model = CLIPTextModelWithProjection.from_pretrained(args.model).eval()
    with torch.no_grad():
        t = tok(prompts, padding=True, return_tensors="pt")
        text = as_embedding(text_model(**t), "text", text_model.config.projection_dim).float()
    del text_model
    return vision, vision.config.image_size, text


def make_model(vision, text_emb, logit_scale: float):
    """Image encoder + (countries x D) head; logits = scale * cos(img, W)."""
    import torch.nn.functional as F
    from torch import nn

    class GeoCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = vision
            self.proj_dim = vision.config.projection_dim
            self.head = nn.Parameter(text_emb.clone().float())
            self.scale = logit_scale

        def embed(self, px):
            out = as_embedding(self.vision(pixel_values=px), "image", self.proj_dim)
            return F.normalize(out.float(), dim=-1)

        def logits_from_embed(self, e):
            return self.scale * e @ F.normalize(self.head.float(), dim=-1).T

        def forward(self, px):
            return self.logits_from_embed(self.embed(px))

    return GeoCLIP()


def make_loader(ds, batch_size, shuffle, workers, seed):
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        drop_last=False, generator=g, pin_memory=torch.cuda.is_available(),
    )


def predict_logits(model, ds, batch_size, workers, device, amp, seed=42) -> np.ndarray:
    """(N, C) logits; with TTA, the mean of the views' logits."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in make_loader(ds, batch_size, False, workers, seed):
            b, n = x.shape[:2]
            x = x.flatten(0, 1).to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                lg = model(x)
            out.append(lg.float().view(b, n, -1).mean(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.head.shape[0]))


def extract_embeddings(model, ds, batch_size, workers, device, amp):
    """(N, D) normalized features from the frozen backbone (single view: center crop)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in make_loader(ds, batch_size, False, workers, 0):
            x = x[:, 0].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                e = model.embed(x)
            out.append(e.float())
    return torch.cat(out)


def top1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(evaluate(logits, labels)[0].mean()) if len(labels) else float("nan")


def train_probe(model, feats, labels, val_feats, val_labels, args, device, history):
    """Train only the head; epoch 0 = zero-shot starting point. The best-val head is restored."""
    import torch
    import torch.nn.functional as F

    y = torch.as_tensor(labels, device=device)

    def val_acc() -> float:
        with torch.no_grad():
            return top1(model.logits_from_embed(val_feats).cpu().numpy(), val_labels)

    best = val_acc()
    best_head = model.head.detach().clone()
    history.append({"phase": "probe", "epoch": 0, "val_top1": best})
    print(f"[probe] epoch 0 (initial) val_top1={best:.4f}")
    opt = torch.optim.AdamW([model.head], lr=args.probe_lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(args.seed)
    for ep in range(1, args.probe_epochs + 1):
        perm = torch.randperm(len(feats), generator=gen).to(device)
        tot = 0.0
        for s in range(0, len(perm), 256):
            idx = perm[s : s + 256]
            loss = F.cross_entropy(
                model.logits_from_embed(feats[idx]), y[idx], label_smoothing=args.label_smoothing
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        acc = val_acc()
        history.append({"phase": "probe", "epoch": ep, "train_loss": tot / len(perm), "val_top1": acc})
        if acc > best:
            best, best_head = acc, model.head.detach().clone()
    model.head.data.copy_(best_head)
    print(f"[probe] best val_top1={best:.4f}")


def finetune(model, train_ds, val_ds, args, device, amp, ckpt_dir: Path, history, meta):
    import torch
    import torch.nn.functional as F

    loader = make_loader(train_ds, args.batch_size, True, args.workers, args.seed)
    decay, no_decay = [], []
    for name, p in model.vision.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or "embedding" in name else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
            {"params": no_decay, "lr": args.lr_backbone, "weight_decay": 0.0},
            {"params": [model.head], "lr": args.lr_head, "weight_decay": 0.0},
        ]
    )
    total = args.epochs * len(loader)
    warm = int(args.warmup_epochs * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warm))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    stopper = EarlyStopper(args.patience)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale:  # fp16 overflow skips the optimizer step: skip the scheduler too
                sched.step()
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        val_logits = predict_logits(model, val_ds, args.batch_size, args.workers, device, amp)
        v = top1(val_logits, val_ds.labels)
        row = {
            "phase": "finetune", "epoch": epoch, "train_loss": tot / max(n, 1),
            "train_top1": correct / max(n, 1), "val_top1": v,
            "lr_backbone": opt.param_groups[0]["lr"], "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"[ft] {row}")
        payload = {"state_dict": model.state_dict(), "val_top1": v, "epoch": epoch, **meta}
        torch.save(payload, ckpt_dir / "last.pt")
        if stopper.update(v):
            torch.save(payload, ckpt_dir / "best.pt")
        if stopper.should_stop:
            print(f"[ft] early stopping (epoch {epoch})")
            break
    return stopper.best


# ----------------------------------------------------------------- run


def run(args: argparse.Namespace) -> dict:
    import torch

    set_seed(args.seed)
    root = Path(args.root)
    classes = json.loads((root / "classes.json").read_text())
    df = pd.read_csv(root / "manifest.csv", dtype={"id": str})
    run_name = args.run_name + ("_smoke" if args.smoke else "")
    if args.smoke:  # small/fast check
        df = df.groupby("split").head(48)
        args.epochs, args.probe_epochs, args.batch_size, args.workers = 2, 2, 8, 0
    out = Path(args.results_dir) / run_name
    ckpt_dir = Path(args.models_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    prompts = build_prompts(classes, args.template)
    vision, size, text_emb = load_backbone(args, len(classes), prompts)
    model = make_model(vision, text_emb, args.logit_scale).to(device)

    def part(name):
        return df[df["split"] == name].reset_index(drop=True)

    ds = {
        "train": GeoDataset(root, part("train"), size, True, seed=args.seed),
        "train_eval": GeoDataset(root, part("train"), size, False),
        "val": GeoDataset(root, part("val"), size, False),
        "test": GeoDataset(root, part("test"), size, False),
    }
    meta = {"classes": classes, "model": args.model, "template": args.template, "img_size": size}
    (out / "run_config.json").write_text(json.dumps(
        {**vars(args), "run_name": run_name, "device": str(device), "img_size": size,
         "versions": library_versions(), "head_init": "text_embeddings", "prompts": prompts, "classes": classes,
         "n_missing": {k: v.n_missing for k, v in ds.items()}},
        indent=2, default=str,
    ))
    history: list[dict] = []

    # --- LP
    ev = lambda d: extract_embeddings(model, d, args.batch_size, args.workers, device, amp)
    f_tr, f_va, f_te = ev(ds["train_eval"]), ev(ds["val"]), ev(ds["test"])
    train_probe(model, f_tr, ds["train_eval"].labels, f_va, ds["val"].labels, args, device, history)
    with torch.no_grad():
        for name, f in (("val", f_va), ("test", f_te)):
            d = ds[name]
            write_split_results(out / "probe", name, d.df["id"], d.labels,
                                model.logits_from_embed(f).cpu().numpy(), classes, d.n_missing)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, ckpt_dir / "probe.pt")

    # --- FT
    best = finetune(model, ds["train"], ds["val"], args, device, amp, ckpt_dir, history, meta)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    print(f"best val_top1={best:.4f}")

    # --- Final evaluation: best checkpoint; the test split is used only here
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device)["state_dict"])
    results = {}
    for name in ("val", "test"):
        d = GeoDataset(root, part(name), size, False, tta=args.tta)
        logits = predict_logits(model, d, args.batch_size, args.workers, device, amp)
        results[name] = write_split_results(
            out, name, d.df["id"], d.labels, logits, classes, d.n_missing
        )
    return results


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="data/subset")
    ap.add_argument("--run-name", default="clip_vitb16_lpft")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--probe-epochs", type=int, default=30)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true", help="average of 3 crops: left/center/right (final eval)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny random model, no download, quick check")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
