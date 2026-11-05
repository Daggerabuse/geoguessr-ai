"""Serve the browser demo (web/) locally, and check it in a real browser against PyTorch.

  serve: http://localhost:8000/web/ with the model files served from local folders, so the
         demo can be tried before anything is on the Hub:
           /web/...                -> web/
           /model/onnx/<f>.onnx    -> --onnx-dir (export_onnx output)
           /model/<f>.json         -> --model (export_hf folder)
         Open http://localhost:8000/web/?base=/model/  (add &model=fp32 for the fp32 graph).
         Without ?base= the page loads the model from the Hub.

  check: opens the page in a headless browser (Playwright; --channel msedge or chrome uses an
         installed browser, nothing is downloaded) and, for the first --limit images:
           pixels    PIL-decoded pixels -> preprocess.js -> onnxruntime-web, compared with
                     the PyTorch model on Python-preprocessed crops (logits, top-1)
           jpeg      the browser decodes the JPEG itself (as in the demo): pixel difference
                     to PIL's decoder, and top-1 agreement of the full browser pipeline
           canvas    resizing with canvas drawImage (imageSmoothingQuality "high") instead
                     of preprocess.js: pixel difference to PIL bicubic and top-1 agreement
         The report is printed and optionally written as JSON.

Usage:
  python -m geoguessr_ai.webdemo serve --model models/hf_export --onnx-dir models/onnx
  python -m geoguessr_ai.webdemo check --model models/hf_export --onnx-dir models/onnx \\
      --images data/sample/images --limit 200 --variant int8 --report results/onnx_parity/web_int8.json
"""

import argparse
import base64
import functools
import http.server
import io
import json
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    """Maps /web/ and /model/ to local folders; nothing else is served."""

    def __init__(self, *a, model: Path, onnx_dir: Path, **kw):
        self.model, self.onnx_dir = Path(model), Path(onnx_dir)
        super().__init__(*a, directory=str(WEB_DIR), **kw)

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        parts = [p for p in path.split("/") if p and p not in (".", "..")]
        if parts[:1] == ["web"]:
            return str(WEB_DIR.joinpath(*parts[1:]) if parts[1:] else WEB_DIR)
        if parts[:2] == ["model", "onnx"] and len(parts) == 3:
            return str(self.onnx_dir / parts[2])
        if parts[:1] == ["model"] and len(parts) == 2 and parts[1].endswith(".json"):
            return str(self.model / parts[1])
        return str(WEB_DIR / "__not_found__")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet
        pass


def start_server(model: Path, onnx_dir: Path, port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(Handler, model=model, onnx_dir=onnx_dir)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ----------------------------------------------------------------- browser check

CANVAS_JS = """async ([b64, w, h, rw, rh]) => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { rgba.set(bytes.subarray(i*3, i*3+3), i*4); rgba[i*4+3] = 255; }
  const src = new OffscreenCanvas(w, h);
  src.getContext('2d').putImageData(new ImageData(rgba, w, h), 0, 0);
  const dst = new OffscreenCanvas(rw, rh);
  const ctx = dst.getContext('2d');
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, rw, rh);
  const d = ctx.getImageData(0, 0, rw, rh).data;
  let s = ''; for (let i = 0; i < rw * rh; i++) s += String.fromCharCode(d[i*4], d[i*4+1], d[i*4+2]);
  return btoa(s);
}"""

PREDICT_JS = """async ([b64, w, h]) => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const rgba = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) { rgba.set(bytes.subarray(i*3, i*3+3), i*4); rgba[i*4+3] = 255; }
  const r = await window.geoguessr.predictPixels(rgba, w, h);
  return { mean: r.mean, ms: r.ms };
}"""

JPEG_JS = """async (b64) => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const { rgba, w, h } = await window.geoguessr.decode(new Blob([bytes], { type: 'image/jpeg' }));
  const r = await window.geoguessr.predictPixels(rgba, w, h);
  let s = ''; for (let i = 0; i < w * h; i++) s += String.fromCharCode(rgba[i*4], rgba[i*4+1], rgba[i*4+2]);
  return { mean: r.mean, w, h, rgb: btoa(s) };
}"""


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def _unb64(s: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(shape)


def _pix_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(int) - b.astype(int))
    return {"max": int(d.max()), "mean": float(d.mean()), "frac_nonzero": float((d > 0).mean())}


def _agg(rows: list[dict]) -> dict:
    keys = rows[0].keys() if rows else []
    return {k: {"max": float(np.max([r[k] for r in rows])), "mean": float(np.mean([r[k] for r in rows]))}
            for k in keys}


def check(model: Path, onnx_dir: Path, images: list[Path], variant: str, channel: str,
          port: int, timeout_s: float = 600) -> dict:
    import torch
    from PIL import Image
    from playwright.sync_api import sync_playwright

    from geoguessr_ai.baseline_clip import resize_short
    from geoguessr_ai.export_onnx import load_folder, parity_stats
    from geoguessr_ai.train import eval_views

    clf, cfg, _ = load_folder(model)
    size = int(cfg["image_size"])
    srv = start_server(model, onnx_dir, port)
    url = f"http://127.0.0.1:{port}/web/index.html?base=/model/&nocache" + (
        "&model=fp32" if variant == "fp32" else "")
    ref, web_pix, web_jpeg, canvas_pred = [], [], [], []
    jpeg_px, canvas_px, ms = [], [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=channel, headless=True)
            page = browser.new_page()
            logs = []
            page.on("console", lambda m: logs.append(m.text))
            page.goto(url)
            page.wait_for_function("window.geoguessr !== undefined")
            page.evaluate("window.geoguessr.ready.then(() => true)")
            ua = page.evaluate("navigator.userAgent")
            for i, path in enumerate(images):
                raw = path.read_bytes()
                with Image.open(io.BytesIO(raw)) as im:
                    rgb_im = im.convert("RGB")
                rgb = np.asarray(rgb_im)
                h, w = rgb.shape[:2]
                ref.append(clf(torch.from_numpy(np.stack(eval_views(rgb_im, size, True)))).numpy().mean(0))
                r = page.evaluate(PREDICT_JS, [_b64(rgb), w, h])
                web_pix.append(np.array(r["mean"], np.float32))
                ms.append(r["ms"])
                j = page.evaluate(JPEG_JS, base64.b64encode(raw).decode())
                web_jpeg.append(np.array(j["mean"], np.float32))
                if (j["w"], j["h"]) == (w, h):
                    jpeg_px.append(_pix_stats(_unb64(j["rgb"], (h, w, 3)), rgb))
                resized = resize_short(rgb_im, size)
                rw, rh = resized.size
                c = _unb64(page.evaluate(CANVAS_JS, [_b64(rgb), w, h, rw, rh]), (rh, rw, 3))
                canvas_px.append(_pix_stats(c, np.asarray(resized)))
                # model on canvas-resized pixels: crops/normalization unchanged (already at size)
                cv = np.stack(eval_views(Image.fromarray(c), size, True))
                canvas_pred.append(clf(torch.from_numpy(cv)).numpy().mean(0))
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(images)}", flush=True)
            browser.close()
    finally:
        srv.shutdown()
    ref = np.stack(ref)
    return {"variant": variant, "browser": ua, "n_images": len(images),
            "pixels_vs_torch": parity_stats(ref, np.stack(web_pix)),
            "jpeg_decode_vs_pil": {"n_same_size": len(jpeg_px), **_agg(jpeg_px)},
            "browser_jpeg_pipeline_vs_torch": parity_stats(ref, np.stack(web_jpeg)),
            "canvas_resize_vs_pil": _agg(canvas_px),
            "canvas_resize_pipeline_vs_torch": parity_stats(ref, np.stack(canvas_pred)),
            "ms_per_image": _agg(ms), "console": logs[-20:]}


# ----------------------------------------------------------------- run


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["serve", "check"])
    ap.add_argument("--model", type=Path, default=Path("models/hf_export"))
    ap.add_argument("--onnx-dir", type=Path, default=Path("models/onnx"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--images", type=Path, default=Path("data/sample/images"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N images")
    ap.add_argument("--variant", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--channel", default="msedge", help="Playwright browser channel (msedge, chrome)")
    ap.add_argument("--report", type=Path, default=None)
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.command == "serve":
        srv = start_server(args.model, args.onnx_dir, args.port)
        print(f"demo with local model: http://localhost:{args.port}/web/?base=/model/")
        print(f"demo with the Hub model: http://localhost:{args.port}/web/")
        print("Ctrl+C to stop")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            srv.shutdown()
        return
    from geoguessr_ai.export_onnx import list_images

    images = list_images(args.images)[args.offset:args.offset + args.limit]
    rep = check(args.model, args.onnx_dir, images, args.variant, args.channel, args.port)
    print(json.dumps(rep, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
