"""E0 throughput benchmark: what a training or inference step costs on this GPU.

Measures, for one model / batch size / precision setting:
  - train step (forward + backward + AdamW, fp16 AMP) on synthetic batches: GPU-only images/s
  - inference (fp16, no grad) on synthetic batches: images/s
  - peak GPU memory of each
  - with --root: data-loader images/s of train.py's augmentation pipeline (GeoDataset on
    real images, no GPU), to see whether the CPU or the GPU is the bottleneck.
  - with --lora-rank (V4): LoRA (lora.py) instead of full fine-tuning; the frozen base stays
    fp16 on cuda; --lora-blocks K adapts only the last K blocks; --head-classes 5000 adds the
    geocell head and CE. An out-of-memory train step is recorded (train_oom), not raised.
  - with --archive (V4): a seeded sample of real ft_archive images over all parts; phases
    (--archive-phases): loader alone (train / eval preprocessing), loader + train step,
    loader + inference, checkpoint save/load round trip. The results so far are appended after
    every phase, so a killed run keeps its finished phases; the last line of a run has
    complete=true. Labels are dummies: nothing is scored.
The first --warmup steps are excluded. Results are appended to --out (JSON lines), so
P100 / T4 / 2xT4 runs can be compared side by side.

Usage (Kaggle):
  python -m geoguessr_ai.bench --model openai/clip-vit-base-patch16 --batch-size 64
  python -m geoguessr_ai.bench --model geolocal/StreetCLIP --batch-size 16 --grad-checkpointing
  python -m geoguessr_ai.bench --model openai/clip-vit-base-patch16 --root /kaggle/working/subset
  python -m geoguessr_ai.bench --model geolocal/StreetCLIP --batch-size 16 --lora-rank 16 \\
      --lora-blocks 12 --head-classes 5000 --archive /kaggle/temp/ft_parts --device cuda:1
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geoguessr_ai.baseline_clip import as_embedding, library_versions
from geoguessr_ai.embed import clip_center_crop, load_encoder
from geoguessr_ai.train import GeoDataset, make_loader, to_tensor_array, train_view


def timed(fn, steps: int, warmup: int, sync) -> float:
    """Seconds per step after warmup."""
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(steps):
        fn()
    sync()
    return (time.time() - t0) / steps


class ArchiveImages:
    """A seeded random sample of n images from all ft_archive part_*.zip files (raw OSV-5M JPEG
    bytes), so reads spread over every part. train=True: train.train_view (V1 augmentation, no
    flip); else V3's eval crop (embed.clip_center_crop). Label is always 0: timing only."""

    def __init__(self, parts, n: int, size: int, train: bool, seed: int = 42):
        items = []
        for p in sorted(parts):
            with zipfile.ZipFile(p) as zf:
                items += [(str(p), name) for name in sorted(zf.namelist()) if name.endswith(".jpg")]
        pick = np.random.default_rng(seed).permutation(len(items))[:n]
        self.items = [items[i] for i in pick]
        self.size, self.train, self.seed, self._zips = size, train, seed, {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, name = self.items[i]
        zf = self._zips.get(path) or self._zips.setdefault(path, zipfile.ZipFile(path))  # one handle per worker
        with Image.open(io.BytesIO(zf.read(name))) as im:
            im = im.convert("RGB")
            x = (train_view(im, self.size, np.random.default_rng([self.seed, i])) if self.train
                 else to_tensor_array(clip_center_crop(im, self.size)))
        return x, 0


ARCHIVE_PHASES = ("loader", "train", "eval", "ckpt")


def save(args, res: dict) -> None:
    """Append the results so far: a timeout kill keeps every finished phase (last line wins)."""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")


def archive_timings(args, res, vision, head, params, opt, train_step, size, device, sync) -> None:
    """Real-image timings (--archive-phases): loader alone, loader + train step, loader +
    inference, checkpoint round trip. Updates res and saves it after every phase."""
    import torch

    parts = sorted(Path(args.archive).glob("part_*.zip"))
    if not parts:
        raise SystemExit(f"no part_*.zip in {args.archive}")
    res.update(archive_parts=len(parts), archive_workers=args.workers, archive_phases=list(args.archive_phases))

    def rate(ds, step=None, batch=args.batch_size):
        n, t0 = 0, None
        for i, (xb, _) in enumerate(make_loader(ds, batch, ds.train, args.workers, args.seed)):
            if i == args.warmup:  # worker start-up and the first batches excluded, as in timed()
                sync()
                t0, n = time.time(), 0
            if step is not None:
                step(xb.to(device, non_blocking=True))
            n += len(xb)
        sync()
        if t0 is None:
            raise SystemExit(f"--archive-images too small for --warmup {args.warmup} batches")
        return round(n / (time.time() - t0), 1)

    train_ds = ArchiveImages(parts, args.archive_images, size, True, args.seed)
    eval_ds = ArchiveImages(parts, args.archive_images, size, False, args.seed)
    res["archive_images"] = len(train_ds)
    ph = args.archive_phases
    if "loader" in ph:
        res["loader_train_images_per_s"] = rate(train_ds)
        res["loader_eval_images_per_s"] = rate(eval_ds)
        save(args, res)
    if "train" in ph and not res.get("train_oom"):
        t0 = time.time()
        try:
            res["e2e_train_images_per_s"] = rate(train_ds, train_step)
        except torch.OutOfMemoryError:
            res["e2e_train_oom"] = True
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        res["e2e_train_seconds"] = round(time.time() - t0, 1)
        save(args, res)

    def infer(xb):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            vision(pixel_values=xb)

    if "eval" in ph:
        vision.eval()
        res["e2e_eval_images_per_s"] = rate(eval_ds, infer, args.infer_batch_size or args.batch_size)
        vision.train()
        save(args, res)

    if "ckpt" in ph:  # round trip of trainable params + optimizer state (what a resumed session needs)
        ckpt = Path(args.out).parent / f"bench_ckpt_{os.getpid()}.pt"  # two benches may share the dir
        state = {"params": [p.detach().cpu() for p in params], "opt": opt.state_dict()}
        t0 = time.time()
        torch.save(state, ckpt)
        back = torch.load(ckpt, weights_only=True)
        res["ckpt_seconds"] = round(time.time() - t0, 2)
        res["ckpt_mb"] = round(ckpt.stat().st_size / 1e6, 3)
        assert all(torch.equal(a, b) for a, b in zip(state["params"], back["params"])), "checkpoint round trip differs"
        ckpt.unlink()


def run(args: argparse.Namespace) -> dict:
    import torch

    from geoguessr_ai.lora import add_lora, lora_parameters

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    vision, size, dim = load_encoder(args.model, args.smoke, device)
    res = {"tag": args.tag, "model": args.model, "image_size": size, "batch_size": args.batch_size,
           "grad_checkpointing": args.grad_checkpointing, "device": str(device),
           "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
           "gpu_count": torch.cuda.device_count(), "versions": library_versions()}
    ib = args.infer_batch_size or args.batch_size
    res["infer_batch_size"] = ib
    xi = torch.randn(ib, 3, size, size, device=device)
    x = torch.randn(args.batch_size, 3, size, size, device=device)

    # inference: the fp16 encoder load_encoder returns (as in embed.py)
    def infer():
        with torch.no_grad():
            vision(pixel_values=xi.to(vision.dtype))

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    s = timed(infer, args.steps, args.warmup, sync)
    res["infer_images_per_s"] = round(ib / s, 1)
    res["infer_peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if cuda else None

    res["attn"] = vision.config._attn_implementation
    if args.lora_rank:
        # LoRA: the frozen base keeps its dtype (fp16 on cuda); only A, B and the head train (fp32)
        wrapped = add_lora(vision, args.lora_rank, args.lora_alpha, args.lora_targets, args.lora_blocks)
        vision.train()
        params = lora_parameters(vision)
        res.update(lora_rank=args.lora_rank, lora_alpha=args.lora_alpha or args.lora_rank,
                   lora_targets=list(args.lora_targets), lora_blocks=args.lora_blocks,
                   lora_modules=len(wrapped))
    else:  # full fine-tuning: fp32 master weights + fp16 autocast, as train.py
        vision = vision.float().train()
        params = list(vision.parameters())
    if args.grad_checkpointing:  # non-reentrant: gradients reach LoRA params although the input needs none
        vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head = torch.nn.Linear(dim, args.head_classes, device=device) if args.head_classes else None
    if head is not None:
        params = params + list(head.parameters())
    res["trainable_params"] = sum(p.numel() for p in params)
    res["head_classes"] = args.head_classes
    opt = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    y = (torch.randint(0, args.head_classes, (args.batch_size,), generator=g).to(device) if head is not None
         else torch.randn(args.batch_size, dim, generator=g).to(device))

    def train_step(xb=x):
        with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
            out = as_embedding(vision(pixel_values=xb), "image", dim)
            if head is not None:
                out = head(torch.nn.functional.normalize(out.float(), dim=-1))
        yb = y[: len(xb)]
        loss = (torch.nn.functional.cross_entropy(out.float(), yb) if head is not None
                else torch.nn.functional.mse_loss(out.float(), yb))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        s = timed(train_step, args.steps, args.warmup, sync)
        res["train_step_s"] = round(s, 4)
        res["train_images_per_s"] = round(args.batch_size / s, 1)
        res["train_peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if cuda else None
        # a step that silently trains nothing (e.g. reentrant checkpointing with a frozen input) is fast and wrong
        res["grads_ok"] = all(p.grad is not None for p in params)
        assert res["grads_ok"], "some trainable parameters got no gradient"
    except torch.OutOfMemoryError:  # recorded, so a sweep keeps going and the edge of memory is known
        res["train_oom"] = True
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    if args.archive:
        save(args, res)
        archive_timings(args, res, vision, head, params, opt, train_step, size, device, sync)

    if args.root:
        df = pd.read_csv(Path(args.root) / "manifest.csv", dtype={"id": str})
        ds = GeoDataset(args.root, df[df["split"] == "train"].head(args.loader_images), size, True)
        t0, n = time.time(), 0
        for xb, _ in make_loader(ds, args.batch_size, True, args.workers, args.seed):
            n += len(xb)
        res["loader_images_per_s"] = round(n / (time.time() - t0), 1)
        res["loader_workers"] = args.workers

    res["complete"] = True
    save(args, res)
    print(json.dumps(res, indent=2))
    return res


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--infer-batch-size", type=int, default=0,
                    help="inference batch (default: --batch-size; embed.py uses 256 or 64)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lora-rank", type=int, default=0, help="0 = full fine-tuning (as before)")
    ap.add_argument("--lora-alpha", type=float, default=None, help="default: the rank (scale 1)")
    ap.add_argument("--lora-targets", nargs="+", default=["q_proj", "k_proj", "v_proj", "out_proj"],
                    help="Linear names inside a block, e.g. q_proj k_proj v_proj out_proj fc1 fc2")
    ap.add_argument("--lora-blocks", type=int, default=0, help="LoRA in the last K blocks only (0 = all)")
    ap.add_argument("--head-classes", type=int, default=0,
                    help="linear head on the L2-normalised embedding, CE over this many classes "
                         "(5000 = the V3 cells); 0 = MSE on the embedding, as before")
    ap.add_argument("--archive", help="dir of ft_archive part_*.zip: time loader, loader+train, loader+eval, checkpoint")
    ap.add_argument("--archive-images", type=int, default=2000)
    ap.add_argument("--archive-phases", nargs="+", choices=ARCHIVE_PHASES, default=list(ARCHIVE_PHASES))
    ap.add_argument("--tag", default="", help="config name stored in every result line")
    ap.add_argument("--root", help="prep_data subset: also time the training data loader")
    ap.add_argument("--loader-images", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/v2/bench.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny random model (no download)")
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
