"""Export the fine-tuned checkpoint (best.pt) as a Hugging Face model repo.

Output folder (default models/hf_export, must be empty or not exist):
  config.json + model.safetensors  CLIPVisionModelWithProjection (save_pretrained format)
  head.safetensors                 classification head {"head": (n_classes, projection_dim)}
  classes.json                     ISO-2 codes, same order and format as the training classes.json
  inference_config.json            preprocessing (image size, normalization, TTA crops), logit scale
  README.md                        model card; test results are read from results/

Only the model weights are exported (no optimizer state, no training metadata beyond
the base model, template and epoch).

Usage (Kaggle):
  python -m geoguessr_ai.export_hf --checkpoint <path to best.pt> --out /kaggle/working/hf_export
  python -m geoguessr_ai.export_hf --checkpoint ... --out ... --push Daggerabuse/geoguessr-clip-vitb16

--push reads a write token from the HF_WRITE_TOKEN environment variable. The token is
never printed, logged or written to a file.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from geoguessr_ai.baseline_clip import country_names

TOKEN_ENV = "HF_WRITE_TOKEN"
DEFAULT_REPO_ID = "Daggerabuse/geoguessr-clip-vitb16"
GITHUB_URL = "https://github.com/Daggerabuse/geoguessr-ai"
DATASET_URL = "https://huggingface.co/datasets/osv5m/osv5m"
# Same values as train.CLIP_MEAN / CLIP_STD (kept as plain floats so this module does not import train).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

OSV5M_BIBTEX = """@inproceedings{osv5m,
  title     = {{OpenStreetView-5M}: {T}he Many Roads to Global Visual Geolocation},
  author    = {Astruc, Guillaume and Dufour, Nicolas and Siglidis, Ioannis and Aronssohn, Constantin
               and Bouia, Nacim and Fu, Stephanie and Loiseau, Romain and Nguyen, Van Nguyen
               and Raude, Charles and Vincent, Elliot and Xu, Lintao and Zhou, Hongyu and Landrieu, Loic},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}"""


# ----------------------------------------------------------------- pure helpers (no torch)


def display_names(codes: list[str]) -> dict[str, str]:
    """ISO-2 code -> display name ("the United States" -> "United States")."""
    return {c: n.removeprefix("the ") for c, n in country_names(codes).items()}


def inference_config(meta: dict, logit_scale: float) -> dict:
    """Everything needed to reproduce train.py's evaluation preprocessing and logits."""
    classes = list(meta["classes"])
    return {
        "base_model": meta["model"],
        "template": meta["template"],
        "checkpoint_epoch": meta.get("epoch"),
        "n_classes": len(classes),
        "image_size": int(meta["img_size"]),
        "resize": "short side to image_size, bicubic (PIL)",
        "mean": CLIP_MEAN,
        "std": CLIP_STD,
        "eval_crop": "center",
        "tta_crops": ["left", "center", "right"],
        "tta_note": "square crops along the long axis (top/center/bottom for portrait images); "
                    "the logits of the crops are averaged; no flips",
        "logit_scale": float(logit_scale),
        "logits": "logit_scale * normalize(image_embeds) @ normalize(head).T",
        "head_file": "head.safetensors",
        "head_key": "head",
        "display_names": display_names(classes),
    }


def read_metrics(results_dir: Path, run: str, split: str = "test") -> dict:
    """<results_dir>/<run>/<split>_metrics.json; raises if missing (numbers are never made up)."""
    path = Path(results_dir) / run / f"{split}_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"metrics file not found: {path}")
    return json.loads(path.read_text())


def check_run_classes(results_dir: Path, run: str, classes: list[str]) -> None:
    """The results run must use exactly the checkpoint's classes (otherwise its numbers do not apply)."""
    path = Path(results_dir) / run / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"run config not found: {path}")
    run_classes = json.loads(path.read_text()).get("classes")
    if list(run_classes or []) != list(classes):
        raise ValueError(f"classes of {run} differ from the checkpoint classes")


def results_rows(results_dir: Path, runs: list[tuple[str, str]]) -> list[dict]:
    """[(label, run)] -> rows with test top-1/top-5/n read from results/."""
    rows = []
    for label, run in runs:
        m = read_metrics(results_dir, run, "test")
        rows.append({"label": label, "run": run, "top1": m["top1"], "top5": m["top5"], "n": m["n"]})
    return rows


def model_card(meta: dict, rows: list[dict], repo_id: str, github_url: str = GITHUB_URL) -> str:
    classes = list(meta["classes"])
    names = display_names(classes)
    table = "\n".join(
        f"| {r['label']} | {r['top1']:.3f} | {r['top5']:.3f} | {r['n']} |" for r in rows
    )
    countries = ", ".join(f"{names[c]} ({c})" for c in classes)
    return f"""---
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: image-classification
base_model: {meta["model"]}
datasets:
- osv5m/osv5m
tags:
- clip
- geolocation
- country-classification
- street-view
---

# geoguessr-clip-vitb16

Predicts the country of a street-level photo, choosing one of {len(classes)} countries.
This is the vision encoder of [{meta["model"]}](https://huggingface.co/{meta["model"]}),
fine-tuned with a linear probe followed by full fine-tuning (LP-FT), plus a
classification head initialised from the text embeddings of the prompt
`{meta["template"]}`. Training data: a subset of
[OpenStreetView-5M]({DATASET_URL}). Code, training details and the full analysis:
{github_url}

## Countries

{countries}

The model always answers with one of these countries, even for a photo taken elsewhere.

## Test results

Test split of the project's OSV-5M subset (read from `results/*/test_metrics.json` in the
GitHub repo). TTA averages the logits of the left, center and right crops.

| Model | Test top-1 | Test top-5 | n |
|---|---|---|---|
{table}

## Usage

```python
import json
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import CLIPVisionModelWithProjection

path = snapshot_download("{repo_id}")
vision = CLIPVisionModelWithProjection.from_pretrained(path).eval()
head = load_file(f"{{path}}/head.safetensors")["head"]
cfg = json.load(open(f"{{path}}/inference_config.json"))
classes = json.load(open(f"{{path}}/classes.json"))

# pixel_values: (n_crops, 3, S, S), preprocessed as described in inference_config.json
with torch.no_grad():
    emb = torch.nn.functional.normalize(vision(pixel_values=pixel_values).image_embeds, dim=-1)
    w = torch.nn.functional.normalize(head, dim=-1)
    logits = (cfg["logit_scale"] * emb @ w.T).mean(0)  # mean over the TTA crops
```

`inference_config.json` describes the preprocessing: short side resized to
`image_size` (bicubic), square crop(s), CLIP mean/std normalization. Do not flip the
image: the driving side is a cue the model uses.

## Limitations

- Only {len(classes)} countries; they do not represent the world, and the results are not
  comparable to published OSV-5M benchmarks (different country set and metrics).
- A single training run with a single seed; the variance of training is unknown.
- The validation split comes from the same pool as training (split by sequence), so it is
  optimistic; compare models on the test split.
- TTA was selected on the validation split; its test gain is a single measurement.
- There is a clear gap between train and validation accuracy (over-fitting).
- Trained on OSV-5M street-level imagery; other kinds of photos (indoor, aerial, close-ups)
  are out of distribution.

## License

The weights are released under **CC BY-SA 4.0**, because they are derived from
OpenStreetView-5M (CC BY-SA 4.0). The base model is
[{meta["model"]}](https://huggingface.co/{meta["model"]}). The project code is MIT
licensed ({github_url}).

## Data source and citation

Data: [OpenStreetView-5M]({DATASET_URL}), Astruc et al., CVPR 2024.

```bibtex
{OSV5M_BIBTEX}
```
"""


def get_token(env=None) -> str:
    """Write token from HF_WRITE_TOKEN. The error message never contains a token value."""
    env = os.environ if env is None else env
    token = env.get(TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"--push needs a Hugging Face write token in the {TOKEN_ENV} environment variable"
        )
    return token


def check_out_dir(out: Path) -> None:
    """The output folder is uploaded as a whole, so it must not contain stale files."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output folder is not empty: {out}")


def write_metadata(out: Path, meta: dict, rows: list[dict], repo_id: str, logit_scale: float,
                   github_url: str = GITHUB_URL) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(list(meta["classes"])))
    (out / "inference_config.json").write_text(
        json.dumps(inference_config(meta, logit_scale), indent=2)
    )
    (out / "README.md").write_text(model_card(meta, rows, repo_id, github_url), encoding="utf-8")


# ----------------------------------------------------------------- torch part


def export_weights(checkpoint: Path, out: Path) -> dict:
    """Write the vision encoder (save_pretrained) and head.safetensors; return the checkpoint metadata."""
    import torch
    from safetensors.torch import save_file

    from geoguessr_ai.train import build_vision

    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt["state_dict"]
    vision = build_vision(ckpt["model"], smoke=False)
    vision_sd = {k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")}
    vision.load_state_dict(vision_sd, strict=True)
    extra = set(sd) - {f"vision.{k}" for k in vision_sd} - {"head"}
    if extra:
        raise ValueError(f"unexpected keys in the checkpoint state_dict: {sorted(extra)}")
    head = sd["head"].detach().float().contiguous()
    if head.shape != (len(ckpt["classes"]), vision.config.projection_dim):
        raise ValueError(f"head shape {tuple(head.shape)} does not match classes/projection_dim")

    out.mkdir(parents=True, exist_ok=True)
    vision.float().save_pretrained(out)
    if not (out / "model.safetensors").is_file():
        raise RuntimeError("save_pretrained did not write model.safetensors")
    save_file({"head": head}, out / "head.safetensors")
    return {k: ckpt.get(k) for k in ("classes", "model", "template", "img_size", "epoch")}


def push(out: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=get_token())
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model",
                      commit_message="Upload geoguessr-ai export")
    return f"https://huggingface.co/{repo_id}"


# ----------------------------------------------------------------- run


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True, help="best.pt from train.py")
    ap.add_argument("--out", type=Path, default=Path("models/hf_export"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--run", default="clip_vitb16_lpft", help="results run of the checkpoint")
    ap.add_argument("--tta-run", default="clip_vitb16_lpft_tta", help="same checkpoint with --tta")
    ap.add_argument("--baseline-run", default="clip_vitb16_zeroshot",
                    help="zero-shot run on the same backbone (reference row)")
    ap.add_argument("--logit-scale", type=float, default=100.0, help="train.py --logit-scale")
    ap.add_argument("--github-url", default=GITHUB_URL)
    ap.add_argument("--push", metavar="REPO_ID", default=None,
                    help=f"upload to this Hub repo (token from {TOKEN_ENV})")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    args = ap.parse_args(argv)
    if not args.checkpoint.is_file():
        ap.error(f"checkpoint not found: {args.checkpoint}")
    return args


def run(args: argparse.Namespace) -> None:
    if args.push:
        get_token()  # fail before the slow export if the token is missing
    check_out_dir(args.out)
    runs = [("CLIP ViT-B/16, zero-shot (reference)", args.baseline_run),
            ("This model, center crop", args.run),
            ("This model, TTA (3 crops)", args.tta_run)]
    rows = results_rows(args.results_dir, runs)  # before the export: fail early

    meta = export_weights(args.checkpoint, args.out)
    for _, r in runs:
        check_run_classes(args.results_dir, r, meta["classes"])
    write_metadata(args.out, meta, rows, args.push or DEFAULT_REPO_ID, args.logit_scale,
                   args.github_url)
    print(f"exported to {args.out}: {sorted(p.name for p in args.out.iterdir())}")
    if args.push:
        print(f"pushed: {push(args.out, args.push, args.private)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
