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
- Daggerabuse/geoguessr-streetclip-geocells
---

# geoguessr-ai (browser demo)

Upload a street-level photo; a fine-tuned CLIP ViT-B/16 guesses the country (one of 33).
The model runs in the browser with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/);
the photo is not uploaded anywhere.

- Model: [Daggerabuse/geoguessr-clip-vitb16](https://huggingface.co/Daggerabuse/geoguessr-clip-vitb16)
  (ONNX files under `onnx/`; weights CC BY-SA 4.0, derived from
  [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m), Astruc et al., CVPR 2024)

`coords.html` (V3) guesses coordinates instead: the [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP)
image encoder and a linear head over geographic cells trained on OpenStreetView-5M; the five most
likely cells are shown on an OpenStreetMap map. **Non-commercial use only** (StreetCLIP is CC BY-NC 4.0).

- Files: [Daggerabuse/geoguessr-streetclip-geocells](https://huggingface.co/Daggerabuse/geoguessr-streetclip-geocells)
  (encoder CC BY-NC 4.0; head and cell centroids derived from OpenStreetView-5M, CC BY-SA 4.0;
  licences per file in its model card)
- Code: [github.com/Daggerabuse/geoguessr-ai](https://github.com/Daggerabuse/geoguessr-ai) (MIT)
