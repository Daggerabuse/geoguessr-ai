---
title: geoguessr-ai
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
models:
- Daggerabuse/geoguessr-clip-vitb16
datasets:
- osv5m/osv5m
short_description: Guess the country of a street-level photo (33 countries)
---

# geoguessr-ai demo

Upload a street-level photo; a CLIP ViT-B/16 fine-tuned on a subset of
[OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) shows the five most likely
countries. The prediction averages three crops (left, center, right). It runs on CPU.

The model only knows 33 countries and always answers with one of them.

- Model: [Daggerabuse/geoguessr-clip-vitb16](https://huggingface.co/Daggerabuse/geoguessr-clip-vitb16) (weights CC BY-SA 4.0)
- Code: [github.com/Daggerabuse/geoguessr-ai](https://github.com/Daggerabuse/geoguessr-ai) (MIT)
- Data: OpenStreetView-5M, Astruc et al., CVPR 2024 (CC BY-SA 4.0)

Set the `MODEL_REPO` variable in the Space settings to load a different model repo.
