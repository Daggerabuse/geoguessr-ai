// Preprocessing for the browser demo, following inference_config.json:
//   short side -> image_size (bicubic, as PIL does it), square crops along the long axis
//   (left/center/right, or top/center/bottom for portrait images), CLIP mean/std, CHW float32.
// No flips: the driving side is a country cue.
//
// resizeBicubic reimplements Pillow's Resample.c (bicubic, a = -0.5, support scaled when
// downscaling, 22-bit fixed-point coefficients, horizontal pass then vertical pass, each
// rounded and clipped to uint8). The canvas' own drawImage scaling is not used because it is
// not the same filter. tests/test_web_preprocess.py compares this file with the Python code.
//
// Pure functions on typed arrays: no DOM, so the same file runs in node for the test.

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;
const HALF = 2 ** (PRECISION_BITS - 1);

function bicubic(x) {
  const a = -0.5;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc for one axis.
function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const w = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    const xmin = Math.max(0, Math.trunc(center - support + 0.5));
    const xmax = Math.min(inSize, Math.trunc(center + support + 0.5)) - xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      w[x] = bicubic((x + xmin - center + 0.5) * ss);
      ww += w[x];
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? w[x] / ww : w[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(ss) {
  const v = Math.floor(ss / ONE); // == arithmetic shift by PRECISION_BITS
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function horizontalPass(src, w, h, channels, outW) {
  const { ksize, bounds, kk } = coeffs(w, outW);
  const out = new Uint8Array(outW * h * 3);
  for (let y = 0; y < h; y++) {
    const row = y * w * channels;
    for (let xx = 0; xx < outW; xx++) {
      const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], k0 = xx * ksize;
      let r = HALF, g = HALF, b = HALF;
      for (let x = 0; x < xmax; x++) {
        const k = kk[k0 + x], p = row + (x + xmin) * channels;
        r += src[p] * k; g += src[p + 1] * k; b += src[p + 2] * k;
      }
      const o = (y * outW + xx) * 3;
      out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b);
    }
  }
  return out;
}

function verticalPass(src, w, h, channels, outH) {
  const { ksize, bounds, kk } = coeffs(h, outH);
  const out = new Uint8Array(w * outH * 3);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], k0 = yy * ksize;
    for (let xx = 0; xx < w; xx++) {
      let r = HALF, g = HALF, b = HALF;
      for (let y = 0; y < ymax; y++) {
        const k = kk[k0 + y], p = ((y + ymin) * w + xx) * channels;
        r += src[p] * k; g += src[p + 1] * k; b += src[p + 2] * k;
      }
      const o = (yy * w + xx) * 3;
      out[o] = clip8(r); out[o + 1] = clip8(g); out[o + 2] = clip8(b);
    }
  }
  return out;
}

// Resize an RGB(A) uint8 image (channels = 3 or 4; alpha is dropped) -> RGB uint8.
// Pass order as in Pillow 13 (ImagingResampleInner): vertical first only when the height
// shrinks by more than twice as many pixels as the width; Pillow <= 12 always went
// horizontal first. Each pass rounds to uint8, so the order changes a few pixel values.
export function resizeBicubic(src, w, h, channels, outW, outH) {
  const horizontalFirst = !(h - outH > 0 && h - outH > (w - outW) * 2);
  let cur = src, c = channels, cw = w, ch = h;
  const steps = horizontalFirst ? ["h", "v"] : ["v", "h"];
  for (const step of steps) {
    if (step === "h" && outW !== cw) {
      cur = horizontalPass(cur, cw, ch, c, outW); cw = outW; c = 3;
    } else if (step === "v" && outH !== ch) {
      cur = verticalPass(cur, cw, ch, c, outH); ch = outH; c = 3;
    }
  }
  return c === 3 && cur !== src ? cur : toRGB(cur, cw, ch, c);
}

export function toRGB(src, w, h, channels) {
  const out = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++, j += channels) {
    out[i * 3] = src[j]; out[i * 3 + 1] = src[j + 1]; out[i * 3 + 2] = src[j + 2];
  }
  return out;
}

// Python's round(): round half to even.
function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// Same as app/inference.py resize_short (size of the resized image).
export function shortSideSize(w, h, size) {
  const s = size / Math.min(w, h);
  return [Math.max(size, roundHalfEven(w * s)), Math.max(size, roundHalfEven(h * s))];
}

// Top-left corners of the crops in the resized image (app/inference.py eval_crops).
export function cropOffsets(w, h, size, tta) {
  if (w >= h) {
    const y = Math.floor((h - size) / 2);
    const xs = tta ? [0, Math.floor((w - size) / 2), w - size] : [Math.floor((w - size) / 2)];
    return xs.map((x) => [x, y]);
  }
  const x = Math.floor((w - size) / 2);
  const ys = tta ? [0, Math.floor((h - size) / 2), h - size] : [Math.floor((h - size) / 2)];
  return ys.map((y) => [x, y]);
}

// RGB uint8 image -> { data: Float32Array (n, 3, size, size), crops, width, height }.
// float32 arithmetic in the same order as numpy: (x / 255 - mean) / std.
export function preprocess(rgb, w, h, channels, cfg, tta = true) {
  const size = cfg.image_size;
  const [rw, rh] = shortSideSize(w, h, size);
  const resized = resizeBicubic(rgb, w, h, channels, rw, rh);
  const crops = cropOffsets(rw, rh, size, tta);
  const plane = size * size;
  const data = new Float32Array(crops.length * 3 * plane);
  const f = Math.fround;
  const mean = cfg.mean.map(f), std = cfg.std.map(f);
  const inv = f(255);
  crops.forEach(([x0, y0], n) => {
    for (let c = 0; c < 3; c++) {
      const base = (n * 3 + c) * plane;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const v = f(resized[((y0 + y) * rw + x0 + x) * 3 + c] / inv);
          data[base + y * size + x] = f(f(v - mean[c]) / std[c]);
        }
      }
    }
  });
  return { data, crops, resized, width: rw, height: rh };
}

// Mean of the per-crop logits (n, k) -> (k), then softmax.
export function averageLogits(logits, n, k) {
  const out = new Float64Array(k);
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[j] += logits[i * k + j] / n;
  return out;
}

export function softmax(x) {
  const m = Math.max(...x);
  const e = Array.from(x, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function topK(probs, labels, k = 5) {
  return Array.from(probs, (p, i) => [labels[i], p])
    .sort((a, b) => b[1] - a[1])
    .slice(0, k);
}
