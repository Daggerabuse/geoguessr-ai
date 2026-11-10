"""E0 throughput benchmark: what a training or inference step costs on this GPU.

Measures, for one model / batch size / precision setting:
  - train step (forward + backward + AdamW, fp16 AMP) on synthetic batches: GPU-only images/s
  - inference (fp16, no grad) on synthetic batches: images/s
  - peak GPU memory of each
  - with --root: data-loader images/s of train.py's augmentation pipeline (GeoDataset on
    real images, no GPU), to see whether the CPU or the GPU is the bottleneck.
The first --warmup steps are excluded. Results are appended to --out (JSON lines), so
P100 / T4 / 2xT4 runs can be compared side by side.

Usage (Kaggle):
  python -m geoguessr_ai.bench --model openai/clip-vit-base-patch16 --batch-size 64
  python -m geoguessr_ai.bench --model geolocal/StreetCLIP --batch-size 16 --grad-checkpointing
  python -m geoguessr_ai.bench --model openai/clip-vit-base-patch16 --root /kaggle/working/subset
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from geoguessr_ai.baseline_clip import as_embedding, library_versions
from geoguessr_ai.embed import load_encoder
from geoguessr_ai.train import GeoDataset, make_loader


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


def run(args: argparse.Namespace) -> dict:
    import torch

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    vision, size, dim = load_encoder(args.model, args.smoke, device)
    res = {"model": args.model, "image_size": size, "batch_size": args.batch_size,
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

    # training: fp32 master weights + fp16 autocast, as train.py
    vision = vision.float().train()
    if args.grad_checkpointing:
        vision.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(vision.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    y = torch.randn(args.batch_size, dim, device=device)

    def train_step():
        with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
            out = as_embedding(vision(pixel_values=x), "image", dim)
        loss = torch.nn.functional.mse_loss(out.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    s = timed(train_step, args.steps, args.warmup, sync)
    res["train_step_s"] = round(s, 4)
    res["train_images_per_s"] = round(args.batch_size / s, 1)
    res["train_peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2) if cuda else None

    if args.root:
        df = pd.read_csv(Path(args.root) / "manifest.csv", dtype={"id": str})
        ds = GeoDataset(args.root, df[df["split"] == "train"].head(args.loader_images), size, True)
        t0, n = time.time(), 0
        for xb, _ in make_loader(ds, args.batch_size, True, args.workers, args.seed):
            n += len(xb)
        res["loader_images_per_s"] = round(n / (time.time() - t0), 1)
        res["loader_workers"] = args.workers

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
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
