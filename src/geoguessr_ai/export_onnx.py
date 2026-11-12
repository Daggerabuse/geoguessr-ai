"""Export the export_hf folder (vision encoder + head) as one ONNX graph, fp32 and 8-bit.

Input: a folder written by export_hf (or downloaded from the Hub):
  config.json + model.safetensors  CLIPVisionModelWithProjection (save_pretrained)
  head.safetensors                 {"head": (n_classes, projection_dim)}
  inference_config.json            logit_scale, image_size, mean/std, ...

Output folder (default models/onnx):
  model.onnx       fp32; input pixel_values (N, 3, S, S) float32, output logits (N, n_classes)
  model_int8.onnx  8-bit weights: MatMulNBits (bits=8, blocks of 32 along K, symmetric);
                   activations and arithmetic stay fp32. --int8-method dynamic writes
                   onnxruntime's quantize_dynamic instead (8-bit activations too); the
                   comparison of both is in NOTES_ONNX.md.
  export_info.json opset, quantization settings, file sizes

The graph computes, per crop: logit_scale * normalize(image_embeds) @ normalize(head).T.
N is dynamic so the TTA crops (left/center/right) run as one batch; averaging the crop
logits is left to the caller (web/app.js), the same as app/app.py does it in PyTorch.

Usage:
  python -m geoguessr_ai.export_onnx --model models/hf_export --out models/onnx
  python -m geoguessr_ai.export_onnx --tiny --out models/onnx_tiny   # random tiny CLIP, no download
  python -m geoguessr_ai.export_onnx --model models/hf_export --out models/onnx \\
      --verify data/sample/images --offset 500 --report results/onnx_parity/report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

OPSET = 18
INPUT_NAME = "pixel_values"
OUTPUT_NAME = "logits"
FP32_FILE = "model.onnx"
INT8_FILE = "model_int8.onnx"
INT8_METHODS = ("nbits", "dynamic")
NBITS_BLOCK = 32
SEED = 0


# ----------------------------------------------------------------- model


def build_classifier(vision, head, logit_scale: float):
    """nn.Module: pixel_values (N,3,S,S) -> logits (N, n_classes). The head is normalized once here."""
    import torch

    class GeoClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = vision
            self.register_buffer("head_t", torch.nn.functional.normalize(head.float(), dim=-1).T.contiguous())
            self.logit_scale = float(logit_scale)

        def forward(self, pixel_values):
            emb = self.vision(pixel_values=pixel_values).image_embeds
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return self.logit_scale * (emb @ self.head_t)

    return GeoClassifier().eval()


def load_folder(folder: Path):
    """(classifier, inference_config, classes) from an export_hf folder."""
    from safetensors.torch import load_file
    from transformers import CLIPVisionModelWithProjection

    folder = Path(folder)
    cfg = json.loads((folder / "inference_config.json").read_text())
    classes = json.loads((folder / "classes.json").read_text())
    # eager attention: a plain MatMul/Softmax graph that every ONNX runtime (also the web one) runs
    vision = CLIPVisionModelWithProjection.from_pretrained(folder, attn_implementation="eager")
    vision = vision.float().eval()
    head = load_file(folder / cfg.get("head_file", "head.safetensors"))[cfg.get("head_key", "head")]
    if head.shape != (len(classes), vision.config.projection_dim):
        raise ValueError(f"head shape {tuple(head.shape)} does not match classes/projection_dim")
    # frozen weights: outputs need no autograd (callers call .numpy()), global grad mode untouched
    return build_classifier(vision, head, cfg["logit_scale"]).requires_grad_(False), cfg, classes


def write_tiny_folder(out: Path, n_classes: int = 33, image_size: int = 32) -> Path:
    """A random tiny CLIP vision model in export_hf layout (for tests without downloads)."""
    import torch
    from safetensors.torch import save_file
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    from geoguessr_ai.export_hf import CLIP_MEAN, CLIP_STD

    torch.manual_seed(SEED)
    config = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                              num_attention_heads=2, image_size=image_size, patch_size=8,
                              projection_dim=16, hidden_act="quick_gelu")
    vision = CLIPVisionModelWithProjection(config).eval()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vision.save_pretrained(out)
    save_file({"head": torch.randn(n_classes, config.projection_dim)}, out / "head.safetensors")
    classes = [f"C{i:02d}" for i in range(n_classes)]
    (out / "classes.json").write_text(json.dumps(classes))
    cfg = {"n_classes": n_classes, "image_size": image_size, "mean": CLIP_MEAN, "std": CLIP_STD,
           "tta_crops": ["left", "center", "right"], "logit_scale": 100.0,
           "head_file": "head.safetensors", "head_key": "head",
           "display_names": {c: c for c in classes}}
    (out / "inference_config.json").write_text(json.dumps(cfg, indent=2))
    return out


# ----------------------------------------------------------------- export


def export_fp32(model, image_size: int, path: Path, opset: int = OPSET) -> Path:
    """One self-contained .onnx file (no external data), dynamic batch axis."""
    import onnx
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(3, 3, image_size, image_size)
    batch = torch.export.Dim("batch", min=1, max=64)
    with torch.no_grad():
        prog = torch.onnx.export(model, (dummy,), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                                 dynamic_shapes={"pixel_values": {0: batch}}, opset_version=opset,
                                 dynamo=True, optimize=True)
    prog.save(str(path), external_data=False)
    # The exporter's intermediate shape annotations (value_info) contain an entry that
    # onnx shape inference rejects, which stops quantize_dynamic; value_info is optional
    # metadata, so it is dropped and re-inferred.
    m = onnx.load(str(path))
    del m.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.save(m, str(path))
    onnx.checker.check_model(str(path))
    return path


def quantize_int8(src: Path, dst: Path, method: str = "nbits", block_size: int = NBITS_BLOCK) -> dict:
    """8-bit version of the fp32 graph; returns the settings so they can be written down.

    nbits:   weight-only. Every MatMul with a constant weight becomes MatMulNBits (com.microsoft,
             runs in onnxruntime and onnxruntime-web): int8 weights with one fp32 scale per
             block of `block_size` input rows; the matmul itself is computed in fp32.
    dynamic: quantize_dynamic on MatMul/Gemm: int8 weights (per channel) and activations
             quantized to 8 bit at run time (integer matmul).
    """
    import onnx

    if method == "nbits":
        import logging

        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)
        cfg = DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=True, bits=8)
        q = MatMulNBitsQuantizer(onnx.load(str(src)), algo_config=cfg)
        q.process()
        q.model.save_model_to_file(str(dst), False)
        return {"method": "onnxruntime MatMulNBitsQuantizer (weight-only)", "bits": 8,
                "block_size": block_size, "symmetric": True}
    if method == "dynamic":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        op_types = ["MatMul", "Gemm"]
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                         op_types_to_quantize=op_types)
        return {"method": "onnxruntime quantize_dynamic", "weight_type": "QInt8",
                "per_channel": True, "op_types": op_types}
    raise ValueError(f"unknown int8 method: {method}")


def export(folder: Path, out: Path, opset: int = OPSET, int8_method: str = "nbits") -> dict:
    model, cfg, classes = load_folder(folder)
    out = Path(out)
    t0 = time.time()
    fp32 = export_fp32(model, int(cfg["image_size"]), out / FP32_FILE, opset)
    qset = quantize_int8(fp32, out / INT8_FILE, int8_method)
    info = {"opset": opset, "input": INPUT_NAME, "output": OUTPUT_NAME,
            "input_shape": ["N", 3, int(cfg["image_size"]), int(cfg["image_size"])],
            "output_shape": ["N", len(classes)], "n_classes": len(classes),
            "quantization": qset,
            "files": {p.name: p.stat().st_size for p in (fp32, out / INT8_FILE)},
            "export_seconds": round(time.time() - t0, 1)}
    (out / "export_info.json").write_text(json.dumps(info, indent=2))
    return info


# ----------------------------------------------------------------- parity check


def ort_session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def parity_stats(ref: np.ndarray, got: np.ndarray) -> dict:
    """ref/got: (n_images, n_classes) TTA-averaged logits."""
    d = np.abs(ref - got)
    pr, pg = _softmax(ref.astype(np.float64)), _softmax(got.astype(np.float64))
    top5r = np.argsort(-ref, axis=1)[:, :5]
    top5g = np.argsort(-got, axis=1)[:, :5]
    return {"n": len(ref),
            "logit_max_abs_diff": float(d.max()), "logit_mean_abs_diff": float(d.mean()),
            "logit_p99_abs_diff": float(np.quantile(d.max(1), 0.99)),
            "prob_max_abs_diff": float(np.abs(pr - pg).max()),
            "top1_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
            "top1_in_top5": float(np.mean([r in g for r, g in zip(ref.argmax(1), top5g)])),
            "top5_set_agreement": float(np.mean([set(a) == set(b) for a, b in zip(top5r, top5g)]))}


def verify(folder: Path, onnx_dir: Path, images: list[Path], tta: bool = True) -> dict:
    """PyTorch vs ONNX (fp32, int8) logits on the same preprocessed crops."""
    import torch
    from PIL import Image

    from geoguessr_ai.train import eval_views

    model, cfg, classes = load_folder(folder)
    sess = {name: ort_session(Path(onnx_dir) / f) for name, f in (("fp32", FP32_FILE), ("int8", INT8_FILE))}
    size = int(cfg["image_size"])
    out = {"torch": [], "fp32": [], "int8": []}
    secs = {k: 0.0 for k in out}
    for i, p in enumerate(images):
        with Image.open(p) as im:
            x = np.stack(eval_views(im.convert("RGB"), size, tta))
        t = time.perf_counter()
        with torch.no_grad():
            out["torch"].append(model(torch.from_numpy(x)).numpy().mean(0))
        secs["torch"] += time.perf_counter() - t
        for k, s in sess.items():
            t = time.perf_counter()
            out[k].append(s.run([OUTPUT_NAME], {INPUT_NAME: x})[0].mean(0))
            secs[k] += time.perf_counter() - t
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(images)}", flush=True)
    arr = {k: np.stack(v) for k, v in out.items()}
    return {"n_images": len(images), "tta": tta, "n_classes": len(classes),
            "seconds_per_image": {k: round(v / max(1, len(images)), 4) for k, v in secs.items()},
            "fp32": parity_stats(arr["torch"], arr["fp32"]),
            "int8": parity_stats(arr["torch"], arr["int8"]),
            "int8_vs_fp32": parity_stats(arr["fp32"], arr["int8"])}


def list_images(folder: Path, limit: int | None = None, offset: int = 0) -> list[Path]:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[offset:]
    return files[:limit] if limit else files


# ----------------------------------------------------------------- run


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path, help="export_hf output folder (local or snapshot_download)")
    src.add_argument("--tiny", action="store_true", help="random tiny CLIP (no download), for testing")
    ap.add_argument("--out", type=Path, default=Path("models/onnx"))
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--verify", type=Path, default=None, help="image folder for the parity check")
    ap.add_argument("--int8-method", choices=INT8_METHODS, default="nbits")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N images (sorted by name)")
    ap.add_argument("--limit", type=int, default=None, help="use at most N images")
    ap.add_argument("--report", type=Path, default=None, help="write the parity check as JSON here")
    args = ap.parse_args(argv)
    if args.model is not None and not (args.model / "inference_config.json").is_file():
        ap.error(f"not an export_hf folder (no inference_config.json): {args.model}")
    return args


def run(args: argparse.Namespace) -> dict:
    folder = write_tiny_folder(args.out / "tiny_source") if args.tiny else args.model
    info = export(folder, args.out, args.opset, args.int8_method)
    print(json.dumps(info, indent=2))
    if args.verify:
        images = list_images(args.verify, args.limit, args.offset)
        print(f"parity check on {len(images)} images ...", flush=True)
        rep = verify(folder, args.out, images)
        rep["export"] = info
        print(json.dumps(rep, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rep, indent=2))
        return rep
    return info


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
