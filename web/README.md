---
title: geoguessr-ai
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Guess the country of a street photo, in your browser
models:
- Daggerabuse/geoguessr-clip-vitb16
---

# geoguessr-ai (browser demo)

Upload a street-level photo; a fine-tuned CLIP ViT-B/16 guesses the country (one of 33).
The model runs in the browser with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/);
the photo is not uploaded anywhere.

- Model: [Daggerabuse/geoguessr-clip-vitb16](https://huggingface.co/Daggerabuse/geoguessr-clip-vitb16)
  (ONNX files under `onnx/`; weights CC BY-SA 4.0, derived from
  [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m), Astruc et al., CVPR 2024)
- Code: [github.com/Daggerabuse/geoguessr-ai](https://github.com/Daggerabuse/geoguessr-ai) (MIT)
