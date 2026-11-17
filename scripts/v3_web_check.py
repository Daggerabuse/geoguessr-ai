"""V3 browser check: web/coords.html in headless Edge (browser JPEG decoding + preprocess.js +
onnxruntime-web, one thread) vs Python onnxruntime on PIL-preprocessed crops, same photos.

  uv run --group browser python scripts/v3_web_check.py int8 20 results/onnx_parity/v3_web_int8.json
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from geoguessr_ai import export_geo_demo as d
from geoguessr_ai import export_onnx as eo
from geoguessr_ai import webdemo
from geoguessr_ai.embed import clip_center_crop
from geoguessr_ai.train import to_tensor_array

out = Path("models/streetclip_geocells")
cfg = json.loads((out / "inference_config.json").read_text())
head = np.fromfile(out / d.HEAD_FILE, "<f4").reshape(cfg["n_cells"], -1)
imgs = eo.list_images(Path("data/sample/images"), int(sys.argv[2]), 500)
variant = sys.argv[1]
sess = eo.ort_session(out / cfg["onnx"][variant])
srv = webdemo.start_server(out, out / "onnx", 0)
url = f"http://127.0.0.1:{srv.server_address[1]}/web/coords.html?base=/model/&nocache" + ("&model=fp32" if variant == "fp32" else "")
JS = """async (b64) => { const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
 const {rgba, w, h} = await window.geoguessrCoords.decode(new Blob([bytes], {type: 'image/jpeg'}));
 const r = await window.geoguessrCoords.predictPixels(rgba, w, h); return {top: r.top.map(t => t.cell), ms: r.ms}; }"""
same1 = same5 = 0; ms = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    pg = b.new_page()
    pg.goto(url)
    pg.wait_for_function("window.geoguessrCoords", timeout=60000)
    pg.evaluate("window.geoguessrCoords.ready.then(() => 1)")
    for f in imgs:
        r = pg.evaluate(JS, base64.b64encode(f.read_bytes()).decode())
        with Image.open(f) as im:
            x = to_tensor_array(clip_center_crop(im.convert("RGB"), cfg["image_size"]))[None]
        want = np.argsort(-(sess.run(None, {"pixel_values": x})[0] @ head.T)[0], kind="stable")[:5].tolist()
        same1 += r["top"][0] == want[0]; same5 += set(r["top"]) == set(want); ms.append(r["ms"]["model"])
    print(pg.evaluate("document.getElementById('status').textContent"))
    b.close()
rep = {"variant": variant, "n": len(imgs), "top1_same_as_python_ort": same1 / len(imgs),
                  "top5_set_same": same5 / len(imgs), "model_ms_mean": float(np.mean(ms)), "model_ms_max": float(np.max(ms)),
                  "threads": 1}
print(json.dumps(rep))
Path(sys.argv[3]).write_text(json.dumps(rep, indent=2))
srv.shutdown()
