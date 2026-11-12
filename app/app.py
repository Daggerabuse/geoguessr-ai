"""Gradio demo: upload a street-level photo, get the top-5 countries (CPU, 3-crop TTA).

The model is loaded from the Hugging Face Hub (MODEL_REPO environment variable,
default below); it was exported with geoguessr_ai.export_hf.
"""

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import CLIPVisionModelWithProjection

from inference import preprocess, softmax, top_k

MODEL_REPO = os.environ.get("MODEL_REPO", "Daggerabuse/geoguessr-clip-vitb16")
GITHUB_URL = "https://github.com/Daggerabuse/geoguessr-ai"

path = Path(snapshot_download(MODEL_REPO))
cfg = json.loads((path / "inference_config.json").read_text())
classes = json.loads((path / "classes.json").read_text())
# cast afterwards: the dtype kwarg name changed across transformers versions
vision = CLIPVisionModelWithProjection.from_pretrained(path).float().eval()
head = torch.nn.functional.normalize(load_file(path / cfg["head_file"])[cfg["head_key"]].float(), dim=-1)
labels = [f"{cfg['display_names'].get(c, c)} ({c})" for c in classes]


def predict(image):
    if image is None:
        return None
    px = torch.from_numpy(preprocess(image, cfg, tta=True))
    with torch.inference_mode():
        emb = torch.nn.functional.normalize(vision(pixel_values=px).image_embeds.float(), dim=-1)
        logits = (cfg["logit_scale"] * emb @ head.T).mean(0)  # mean over the 3 crops
    return top_k(softmax(logits.numpy().astype(np.float64)), labels, k=5)


with gr.Blocks(title="geoguessr-ai: which country is this street in?") as demo:
    gr.Markdown(
        "# geoguessr-ai\n"
        "Upload a street-level photo; the model guesses the country."
    )
    with gr.Row():
        image = gr.Image(type="pil", label="Street-level photo")
        output = gr.Label(num_top_classes=5, label="Top-5 countries")
    button = gr.Button("Guess the country", variant="primary")
    button.click(predict, inputs=image, outputs=output)
    image.upload(predict, inputs=image, outputs=output)
    gr.Markdown(
        f"**Limited to {len(classes)} countries:** the model always answers with one of "
        f"them, even for a photo taken elsewhere. Countries: "
        + ", ".join(labels)
        + ".\n\n"
        "CLIP ViT-B/16 fine-tuned on a subset of "
        "[OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) (CC BY-SA 4.0). "
        "The prediction averages three crops (left, center, right) of the photo; the image "
        "is never flipped, because the driving side is a cue. Probabilities are the model's "
        "softmax outputs, not calibrated confidence.\n\n"
        f"Code, results and limitations: [{GITHUB_URL}]({GITHUB_URL}) · "
        f"Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})"
    )

if __name__ == "__main__":
    demo.launch()
