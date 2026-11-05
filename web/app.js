// Browser demo: downloads the ONNX model from the Hugging Face Hub once (kept in the Cache
// Storage), preprocesses the photo as in training (preprocess.js) and runs it with
// onnxruntime-web (WebAssembly). The photo is only read locally; nothing is uploaded.
//
// URL parameters (for local testing):
//   ?model=fp32   use onnx/model.onnx instead of onnx/model_int8.onnx
//   ?base=URL     folder with inference_config.json, classes.json and onnx/ (default: the Hub repo)
//   ?nocache      do not use the Cache Storage

import { preprocess, averageLogits, softmax, topK } from "./preprocess.js";

const ORT_VERSION = "1.31.1"; // same as the <script> tag in index.html
const REPO = "Daggerabuse/geoguessr-clip-vitb16";
const HUB_BASE = `https://huggingface.co/${REPO}/resolve/main/`;
const CACHE_NAME = "geoguessr-ai-v1"; // bump when the files on the Hub change
const MAX_PIXELS = 16_777_216; // canvas limit on iOS Safari

const params = new URLSearchParams(location.search);
const variant = params.get("model") === "fp32" ? "fp32" : "int8";
const base = params.get("base") || HUB_BASE;
const useCache = !params.has("nocache") && "caches" in self;
const MODEL_FILE = variant === "fp32" ? "onnx/model.onnx" : "onnx/model_int8.onnx";

const $ = (id) => document.getElementById(id);
const status = (msg, isError = false) => {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
};

function abs(path) {
  return new URL(path, new URL(base, location.href)).href;
}

// fetch with download progress; cached copies are reused across visits
async function fetchBytes(url, onProgress) {
  const cache = useCache ? await caches.open(CACHE_NAME) : null;
  const hit = cache && (await cache.match(url));
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, total);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
    } catch (e) {
      console.warn("could not cache", url, e); // e.g. storage quota; the demo still works
    }
  }
  return bytes;
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return res.json();
}

async function loadModel() {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  // threads need a cross-origin isolated page (COOP/COEP headers); otherwise one thread
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const [cfg, classes] = await Promise.all([
    fetchJSON(abs("inference_config.json")),
    fetchJSON(abs("classes.json")),
  ]);
  const labels = classes.map((c) => `${cfg.display_names?.[c] ?? c} (${c})`);
  $("n-classes").textContent = classes.length;
  $("countries").textContent = labels.join(", ");

  const bar = $("progress");
  bar.hidden = false;
  const bytes = await fetchBytes(abs(MODEL_FILE), (got, total) => {
    const mb = (got / 1e6).toFixed(0);
    if (total) { bar.value = got / total; status(`Downloading the model (${variant}): ${mb} / ${(total / 1e6).toFixed(0)} MB`); }
    else status(`Downloading the model (${variant}): ${mb} MB`);
  });
  bar.hidden = true;
  status("Starting the model…");
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return { cfg, classes, labels, session };
}

// File -> RGBA pixels. EXIF orientation is applied (as photo viewers show it); no colour
// management (colorSpaceConversion "none"), like PIL which ignores embedded ICC profiles.
async function decode(file) {
  const opts = { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" };
  let bmp = await createImageBitmap(file, opts);
  if (bmp.width * bmp.height > MAX_PIXELS) {
    // very large photos: the browser shrinks them first (not PIL-exact, see NOTES_ONNX.md)
    const s = Math.sqrt(MAX_PIXELS / (bmp.width * bmp.height));
    const [w, h] = [Math.floor(bmp.width * s), Math.floor(bmp.height * s)];
    bmp.close();
    bmp = await createImageBitmap(file, { ...opts, resizeWidth: w, resizeHeight: h, resizeQuality: "high" });
  }
  const canvas = document.createElement("canvas");
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return { rgba: ctx.getImageData(0, 0, canvas.width, canvas.height).data, w: canvas.width, h: canvas.height };
}

async function predictPixels(model, rgba, w, h) {
  const { cfg, classes, session } = model;
  const t0 = performance.now();
  const pre = preprocess(rgba, w, h, 4, cfg, true);
  const n = pre.crops.length, s = cfg.image_size;
  const t1 = performance.now();
  const out = await session.run({ pixel_values: new ort.Tensor("float32", pre.data, [n, 3, s, s]) });
  const t2 = performance.now();
  const logits = out.logits.data;
  const mean = averageLogits(logits, n, classes.length);
  return { pre, logits: Array.from(logits), mean: Array.from(mean), probs: softmax(mean),
           ms: { preprocess: t1 - t0, model: t2 - t1 } };
}

function showCrops(pre, size) {
  const box = $("crops");
  box.replaceChildren();
  for (const [x0, y0] of pre.crops) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(size, size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = ((y0 + y) * pre.width + x0 + x) * 3, o = (y * size + x) * 4;
        img.data[o] = pre.resized[i]; img.data[o + 1] = pre.resized[i + 1];
        img.data[o + 2] = pre.resized[i + 2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    box.append(c);
  }
}

function showTop5(probs, labels) {
  const list = $("top5");
  list.replaceChildren();
  for (const [label, p] of topK(probs, labels, 5)) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = label;
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${(100 * p).toFixed(1)}%`;
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.width = `${(100 * p).toFixed(1)}%`;
    li.append(name, pct, bar);
    list.append(li);
  }
}

async function handleFile(model, file) {
  if (!file || !file.type.startsWith("image/")) { status("Please choose an image file.", true); return; }
  status("Reading the photo…");
  if ($("preview").src.startsWith("blob:")) URL.revokeObjectURL($("preview").src);
  $("preview").src = URL.createObjectURL(file);
  try {
    const { rgba, w, h } = await decode(file);
    status("Thinking…");
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0))); // let the status paint
    const r = await predictPixels(model, rgba, w, h);
    showCrops(r.pre, model.cfg.image_size);
    showTop5(r.probs, model.labels);
    $("timing").textContent =
      `Preprocessing ${r.ms.preprocess.toFixed(0)} ms, model ${r.ms.model.toFixed(0)} ms (${variant}, on this device).`;
    $("result").hidden = false;
    status("Done. Try another photo.");
  } catch (e) {
    console.error(e);
    status(`Could not process this photo: ${e.message}`, true);
  }
}

function wireUpload(model) {
  const input = $("file"), drop = $("drop");
  $("pick").onclick = () => input.click();
  drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") input.click(); };
  input.onchange = () => handleFile(model, input.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    handleFile(model, e.dataTransfer.files[0]);
  };
}

function wireVariantLink() {
  $("variant").textContent = variant === "fp32" ? "fp32" : "8-bit";
  const other = new URLSearchParams(location.search);
  if (variant === "fp32") other.delete("model"); else other.set("model", "fp32");
  $("variant-switch").textContent = variant === "fp32" ? "use the smaller 8-bit model" : "use fp32";
  $("variant-switch").href = `?${other}`;
}

wireVariantLink();
const ready = loadModel().then((model) => {
  wireUpload(model);
  status(`Model ready (${variant}). Choose a photo.`);
  return model;
}).catch((e) => {
  console.error(e);
  status(`Could not load the model: ${e.message}`, true);
  throw e;
});

// hook for scripts/web_check.py (browser parity check); not used by the page itself
window.geoguessr = {
  ready,
  predictPixels: async (rgba, w, h) => predictPixels(await ready, rgba, w, h),
  decode,
};
