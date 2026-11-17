// V3 browser demo: StreetCLIP encoder (ONNX) -> normalized embedding -> cell head -> top-5 cell
// centroids on a Leaflet map. The encoder and the head are separate files (different licences,
// see the model card). Loading, caching and decoding follow app.js (the country page).
//
// URL parameters (for local testing):
//   ?model=fp32   use onnx/vision.onnx instead of onnx/vision_int8.onnx
//   ?base=URL     folder with inference_config.json, cells.json, head_f32.bin and onnx/
//   ?nocache      do not use the Cache Storage

import { preprocess, topCells } from "./preprocess.js";

const ORT_VERSION = "1.31.1"; // same as the <script> tag in coords.html
const REPO = "Daggerabuse/geoguessr-streetclip-geocells";
const HUB_BASE = `https://huggingface.co/${REPO}/resolve/main/`;
const CACHE_NAME = "geoguessr-ai-coords-v1"; // bump when the files on the Hub change
const MAX_PIXELS = 16_777_216; // canvas limit on iOS Safari

const params = new URLSearchParams(location.search);
const variant = params.get("model") === "fp32" ? "fp32" : "int8";
const base = params.get("base") || HUB_BASE;
const useCache = !params.has("nocache") && "caches" in self;

const $ = (id) => document.getElementById(id);
const status = (msg, isError = false) => {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
};
const abs = (path) => new URL(path, new URL(base, location.href)).href;

// fetch with download progress; cached copies are reused across visits (as app.js)
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
  const [cfg, cells] = await Promise.all([fetchJSON(abs("inference_config.json")), fetchJSON(abs("cells.json"))]);
  $("n-cells").textContent = cfg.n_cells;
  const headBytes = await fetchBytes(abs(cfg.head_file));
  const head = new Float32Array(headBytes.buffer, 0, headBytes.length / 4);
  if (head.length !== cfg.n_cells * cfg.embed_dim || cells.centroid_lat.length !== cfg.n_cells) {
    throw new Error("head or cells file does not match inference_config.json");
  }
  const bar = $("progress");
  bar.hidden = false;
  const bytes = await fetchBytes(abs(cfg.onnx[variant]), (got, total) => {
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
  return { cfg, cells, head, session };
}

// File -> RGBA pixels (same as app.js decode)
async function decode(file) {
  const opts = { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" };
  let bmp = await createImageBitmap(file, opts);
  if (bmp.width * bmp.height > MAX_PIXELS) {
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
  const { cfg, cells, head, session } = model;
  const t0 = performance.now();
  const pre = preprocess(rgba, w, h, 4, cfg, false); // cfg.resize "truncate", one center crop
  const s = cfg.image_size;
  const t1 = performance.now();
  const out = await session.run({ [cfg.input]: new ort.Tensor("float32", pre.data, [1, 3, s, s]) });
  const t2 = performance.now();
  const emb = out[cfg.output].data;
  const top = topCells(emb, head, cfg.logit_scale, 5).map(([cell, p]) => (
    { cell, p, lat: cells.centroid_lat[cell], lon: cells.centroid_lon[cell] }));
  return { pre, top, emb: Array.from(emb), ms: { preprocess: t1 - t0, model: t2 - t1, head: performance.now() - t2 } };
}

function showCrop(pre, size) {
  const [x0, y0] = pre.crops[0];
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
  $("crops").replaceChildren(c);
}

const fmt = (lat, lon) => `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;

function showTop5(top) {
  const list = $("top5");
  list.replaceChildren();
  for (const t of top) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = fmt(t.lat, t.lon);
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = `${(100 * t.p).toFixed(1)}%`;
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.width = `${(100 * t.p).toFixed(1)}%`;
    li.append(name, pct, bar);
    list.append(li);
  }
}

let map, layer;
function showMap(top) {
  $("map").hidden = false;
  if (!map) {
    map = L.map("map", { worldCopyJump: true });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
  }
  layer?.remove();
  layer = L.layerGroup().addTo(map);
  top.forEach((t, i) => {
    const label = `#${i + 1}: ${fmt(t.lat, t.lon)}, ${(100 * t.p).toFixed(1)}%`;
    const m = i === 0
      ? L.marker([t.lat, t.lon], { title: label, zIndexOffset: 1000 })
      : L.circleMarker([t.lat, t.lon], { radius: 7, weight: 2, color: "#2f6fdb", fillOpacity: 0.4 });
    m.bindTooltip(label).addTo(layer);
  });
  map.invalidateSize();
  map.fitBounds(L.latLngBounds(top.map((t) => [t.lat, t.lon])), { padding: [40, 40], maxZoom: 6 });
}

async function handleFile(model, file) {
  if (!file || !file.type.startsWith("image/")) { status("Please choose an image file.", true); return; }
  status("Reading the photo…");
  if ($("preview").src.startsWith("blob:")) URL.revokeObjectURL($("preview").src);
  $("preview").src = URL.createObjectURL(file);
  try {
    const { rgba, w, h } = await decode(file);
    status("Thinking… (this can take tens of seconds)");
    await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0))); // let the status paint
    const r = await predictPixels(model, rgba, w, h);
    showCrop(r.pre, model.cfg.image_size);
    showTop5(r.top);
    $("result").hidden = false;
    showMap(r.top);
    $("timing").textContent =
      `Preprocessing ${r.ms.preprocess.toFixed(0)} ms, model ${r.ms.model.toFixed(0)} ms, cells ${r.ms.head.toFixed(0)} ms (${variant}, on this device).`;
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
  $("model-size").textContent = variant === "fp32" ? "about 1.2 GB" : "about 350 MB";
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

// hook for a local browser check (as window.geoguessr in app.js); not used by the page itself
window.geoguessrCoords = {
  ready,
  predictPixels: async (rgba, w, h) => predictPixels(await ready, rgba, w, h),
  decode,
};
