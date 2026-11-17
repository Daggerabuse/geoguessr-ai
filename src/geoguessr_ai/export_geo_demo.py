"""Files for the V3 browser demo (web/coords.html): StreetCLIP encoder as ONNX + the Phase 3 cell head.

The encoder and the head are separate files on purpose: the encoder is StreetCLIP (CC BY-NC 4.0),
the head and the cell centroids come from OpenStreetView-5M (CC BY-SA 4.0), and the page combines
them at run time. Output folder (default models/streetclip_geocells):

  onnx/vision.onnx       fp32; pixel_values (N, 3, S, S) -> image_embeds (N, D), L2-normalized
  onnx/vision_int8.onnx  8-bit weights (export_onnx.quantize_int8, nbits)
  head_f32.bin           (n_cells, D) little-endian float32, rows L2-normalized
  cells.json             cell centroids (lat, lon) of the frozen Phase 3 partition
  inference_config.json  preprocessing, logit_scale, file names, provenance (sha256 of the sources)
  export_info.json       opset, quantization, file sizes
  README.md              model card (licences per file)

The page computes logit_scale * emb @ head.T, a softmax over the cells, and shows the top-5
centroids. Preprocessing is the one the embeddings of Phase 3 were made with
(embed.clip_center_crop: one center crop, long side truncated, CLIP mean/std).

Usage:
  python -m geoguessr_ai.export_geo_demo \\
      --head runs/phase3_test_probes/phase3_probes/streetclip/nat_500000_geo_cell_zeroshot/head.npy \\
      --cells runs/phase3_final_dev/phase3_eval/geocells/cells_5000.json \\
      --verify data/sample/images --offset 500 --limit 500 --report results/onnx_parity/v3_geocells.json
"""

import argparse
import json
import subprocess
import sys
import time
from gc import collect as gc_collect
from pathlib import Path

import numpy as np

from geoguessr_ai import export_onnx as eo
from geoguessr_ai import geocells as gc
from geoguessr_ai.prereg import sha256

MODEL = "geolocal/StreetCLIP"
REPO_ID = "Daggerabuse/geoguessr-streetclip-geocells"
OUTPUT_NAME = "image_embeds"
FP32_FILE, INT8_FILE = "onnx/vision.onnx", "onnx/vision_int8.onnx"
HEAD_FILE, CELLS_FILE = "head_f32.bin", "cells.json"
PREREG = Path("configs/phase3_final.json")
TEST_RESULTS = Path("results/v3/test_results.csv")  # the committed V3 test rows (model card)
CARD_CELL = ("streetclip", "nat", 500_000)  # backbone, ordering, N of the exported head
LOGIT_SCALE = 100.0  # the probes' logit_scale (configs/phase3_final.json settings)
N_BOOT, BOOT_SEED = 10_000, 42  # as the V3 analysis


def load_embedder(model: str):
    """nn.Module: pixel_values -> normalize(image_embeds), and the image size."""
    import torch
    from transformers import CLIPVisionModelWithProjection

    # eager attention: a plain MatMul/Softmax graph that onnxruntime-web runs (as export_onnx)
    vision = CLIPVisionModelWithProjection.from_pretrained(model, attn_implementation="eager")
    vision = vision.float().eval().requires_grad_(False)

    class Embedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = vision

        def forward(self, pixel_values):
            return torch.nn.functional.normalize(self.vision(pixel_values=pixel_values).image_embeds, dim=-1)

    return Embedder().eval(), int(vision.config.image_size)


def normalized_head(head_npy: Path) -> np.ndarray:
    """The saved probe head, rows L2-normalized (as phase3_final.apply_head), after checking
    that it is the head recorded in the probe.json next to it."""
    head_npy = Path(head_npy)
    want = json.loads((head_npy.parent / "probe.json").read_text())["head_sha256"]
    if sha256(head_npy) != want:
        raise SystemExit(f"{head_npy} is not the head recorded in its probe.json")
    h = np.load(head_npy).astype(np.float32)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def centroids(cells_json: Path, want_sha: str) -> dict:
    tree, sha = gc.load(cells_json)
    if sha != want_sha:
        raise SystemExit(f"{cells_json}: sha256 {sha[:16]} is not the pre-registered partition {want_sha[:16]}")
    return {"partition_sha256": sha, "n_cells": gc.n_cells(tree),
            "centroid_lat": tree["centroid_lat"], "centroid_lon": tree["centroid_lon"]}


def code_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()


def export(model: str, head_npy: Path, cells_json: Path, out: Path, want_sha: str) -> dict:
    from geoguessr_ai.export_hf import CLIP_MEAN, CLIP_STD

    out = Path(out)
    head = normalized_head(head_npy)
    cells = centroids(cells_json, want_sha)
    if head.shape[0] != cells["n_cells"]:
        raise SystemExit(f"head has {head.shape[0]} rows, the partition {cells['n_cells']} cells")
    embedder, size = load_embedder(model)
    t0 = time.time()
    fp32 = eo.export_fp32(embedder, size, out / FP32_FILE, output_name=OUTPUT_NAME)
    qset = eo.quantize_int8(fp32, out / INT8_FILE)
    head.astype("<f4").tofile(out / HEAD_FILE)
    (out / CELLS_FILE).write_text(json.dumps(cells))
    cfg = {"image_size": size, "mean": CLIP_MEAN, "std": CLIP_STD,
           "resize": "truncate", "crop": "center",
           "preprocessing": "short side to image_size (bicubic), long side truncated, one center "
                            "crop, CLIP mean/std (embed.clip_center_crop); no flips",
           "input": eo.INPUT_NAME, "output": OUTPUT_NAME, "embed_dim": int(head.shape[1]),
           "logit_scale": LOGIT_SCALE, "logits": "logit_scale * image_embeds @ head.T",
           "n_cells": int(head.shape[0]), "head_file": HEAD_FILE,
           "head_format": "little-endian float32, (n_cells, embed_dim), rows L2-normalized",
           "cells_file": CELLS_FILE, "onnx": {"fp32": FP32_FILE, "int8": INT8_FILE},
           "source": {"encoder": model, "head": Path(head_npy).as_posix(),
                      "head_npy_sha256": sha256(head_npy),
                      "partition": Path(cells_json).as_posix(),
                      "partition_sha256": cells["partition_sha256"], "code_commit": code_commit()},
           "head_sha256": sha256(out / HEAD_FILE), "cells_sha256": sha256(out / CELLS_FILE)}
    (out / "inference_config.json").write_text(json.dumps(cfg, indent=2))
    files = [fp32, out / INT8_FILE, out / HEAD_FILE, out / CELLS_FILE]
    info = {"opset": eo.OPSET, "input": eo.INPUT_NAME, "output": OUTPUT_NAME,
            "input_shape": ["N", 3, size, size], "output_shape": ["N", int(head.shape[1])],
            "quantization": qset,
            "files": {p.relative_to(out).as_posix(): p.stat().st_size for p in files},
            "export_seconds": round(time.time() - t0, 1)}
    (out / "export_info.json").write_text(json.dumps(info, indent=2))
    write_card(out)
    return info


def verify(model: str, out: Path, images: list[Path]) -> tuple[dict, dict]:
    """PyTorch vs ONNX (fp32, int8): cell logits and top-1 centroid distance on the same crops.
    Returns (report, {run: top-1 cell per image})."""
    import torch
    from PIL import Image

    from geoguessr_ai.embed import clip_center_crop
    from geoguessr_ai.geo_eval import haversine
    from geoguessr_ai.train import to_tensor_array

    out = Path(out)
    cfg = json.loads((out / "inference_config.json").read_text())
    head = np.fromfile(out / HEAD_FILE, dtype="<f4").reshape(cfg["n_cells"], cfg["embed_dim"])
    cells = json.loads((out / CELLS_FILE).read_text())
    lat, lon = np.asarray(cells["centroid_lat"]), np.asarray(cells["centroid_lon"])
    size = cfg["image_size"]

    def load(k):
        if k == "torch":
            m = load_embedder(model)[0]
            return lambda x: m(torch.from_numpy(x)).numpy()
        s = eo.ort_session(out / (FP32_FILE if k == "fp32" else INT8_FILE))
        return lambda x: s.run([OUTPUT_NAME], {eo.INPUT_NAME: x})[0]

    arr, secs = {}, {}
    for k in ("torch", "fp32", "int8"):  # one model in memory at a time
        run, rows, secs[k] = load(k), [], 0.0
        for i, p in enumerate(images):
            with Image.open(p) as im:
                x = to_tensor_array(clip_center_crop(im.convert("RGB"), size))[None]
            t = time.perf_counter()
            with torch.no_grad():
                e = run(x)
            secs[k] += time.perf_counter() - t
            rows.append(cfg["logit_scale"] * (e @ head.T)[0])
            if (i + 1) % 100 == 0:
                print(f"  {k} {i + 1}/{len(images)}", flush=True)
        arr[k] = np.stack(rows)
        del run
        gc_collect()

    def km(a, b):  # distance between the top-1 centroids of two runs
        ia, ib = arr[a].argmax(1), arr[b].argmax(1)
        d = haversine(lat[ia], lon[ia], lat[ib], lon[ib])
        return {"top1_km_mean": float(d.mean()), "top1_km_max": float(d.max())}

    rep = {"n_images": len(images), "crop": "center", "n_cells": cfg["n_cells"],
           "seconds_per_image": {k: round(v / max(1, len(images)), 4) for k, v in secs.items()},
           "fp32": {**eo.parity_stats(arr["torch"], arr["fp32"]), **km("torch", "fp32")},
           "int8": {**eo.parity_stats(arr["torch"], arr["int8"]), **km("torch", "int8")},
           "int8_vs_fp32": eo.parity_stats(arr["fp32"], arr["int8"])}
    return rep, {k: a.argmax(1) for k, a in arr.items()}


def geo_accuracy(top1: dict, cells: dict, meta, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """GeoScore and km of each run's top-1 centroid against the true coordinates (meta: latitude,
    longitude, creator_id, rows aligned with top1), and paired differences between the runs;
    95% intervals from a creator-clustered bootstrap (geo_eval.cluster_bootstrap)."""
    from geoguessr_ai.geo_eval import cluster_bootstrap, geoscore, haversine

    lat, lon = np.asarray(cells["centroid_lat"]), np.asarray(cells["centroid_lon"])
    cl = meta["creator_id"].to_numpy()
    km = {k: haversine(meta["latitude"].to_numpy(), meta["longitude"].to_numpy(), lat[i], lon[i])
          for k, i in top1.items()}
    gs = {k: geoscore(v) for k, v in km.items()}
    res = {k: {"geoscore_mean": float(gs[k].mean()), "geoscore_ci95": cluster_bootstrap(gs[k], cl, n_boot, seed),
               "km_median": float(np.median(km[k]))} for k in top1}
    for a, b in (("int8", "fp32"), ("int8", "torch"), ("fp32", "torch")):
        d = gs[a] - gs[b]
        res[f"{a}_minus_{b}"] = {"geoscore_diff_mean": float(d.mean()),
                                 "geoscore_diff_ci95": cluster_bootstrap(d, cl, n_boot, seed),
                                 "km_median_diff": float(np.median(km[a]) - np.median(km[b])),
                                 "n_images_top1_differs": int((top1[a] != top1[b]).sum())}
    res["setting"] = {"n_images": len(cl), "n_creators": int(meta["creator_id"].nunique()),
                      "bootstrap": f"creator-clustered percentile interval, {n_boot} resamples, seed {seed}",
                      "prediction": "centroid of the top-1 cell"}
    return res


def test_rows(path: Path = TEST_RESULTS) -> dict:
    """{predictor: row} of results/v3/test_results.csv for the exported cell (CARD_CELL)."""
    import pandas as pd

    t = pd.read_csv(path)
    bb, o, n = CARD_CELL
    t = t[(t["backbone"] == bb) & (t["ordering"] == o) & (t["n"] == n)]
    return {r["predictor"]: r for _, r in t.iterrows()}


def write_card(out: Path) -> None:
    """README.md (model card) from the folder's own config and export info and the V3 test rows."""
    out = Path(out)
    cfg = json.loads((out / "inference_config.json").read_text())
    info = json.loads((out / "export_info.json").read_text())
    (out / "README.md").write_text(model_card(cfg, info, test_rows()), encoding="utf-8")


def model_card(cfg: dict, info: dict, rows: dict) -> str:
    def perf(r):
        return (f"mean GeoScore **{r['test_geoscore']:.0f}** [{r['test_geoscore_lo']:.0f}, "
                f"{r['test_geoscore_hi']:.0f}], median error **{r['test_median_km']:.0f} km**")

    cell, best = rows["cell_zeroshot"], rows["hybrid_zeroshot"]
    readme = "https://github.com/Daggerabuse/geoguessr-ai"
    mb = {k: f"{v / 1e6:.0f} MB" if v >= 1e6 else f"{v / 1e3:.0f} kB" for k, v in info["files"].items()}
    return f"""---
license: cc-by-nc-4.0
library_name: onnx
pipeline_tag: image-feature-extraction
base_model: {MODEL}
datasets:
- osv5m/osv5m
tags:
- geolocation
- onnx
---

# geoguessr-streetclip-geocells

Files for the coordinate page of the [geoguessr-ai demo](https://huggingface.co/spaces/Daggerabuse/geoguessr-ai):
the [StreetCLIP]({"https://huggingface.co/" + MODEL}) image encoder as ONNX, and a linear head
over {cfg["n_cells"]} geographic cells trained on [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m)
embeddings (Phase 3 of [geoguessr-ai](https://github.com/Daggerabuse/geoguessr-ai): the cell probe
with zero-shot initialisation, StreetCLIP, natural-order subset of 500,000 training images).
The browser computes the embedding, the cell probabilities and the top-5 cell centroids; the photo
is not uploaded.

**Non-commercial use only** (the encoder is CC BY-NC 4.0).

## Files and licences

| File | Size | What | Licence |
|---|---|---|---|
| `{FP32_FILE}` | {mb[FP32_FILE]} | StreetCLIP vision encoder + projection, fp32, outputs L2-normalized `image_embeds` | CC BY-NC 4.0 (converted from [{MODEL}](https://huggingface.co/{MODEL}); weights unchanged) |
| `{INT8_FILE}` | {mb[INT8_FILE]} | the same with 8-bit weights (MatMulNBits, blocks of 32; activations fp32) | CC BY-NC 4.0 (as above; quantized) |
| `{HEAD_FILE}` | {mb[HEAD_FILE]} | cell head, ({cfg["n_cells"]}, {cfg["embed_dim"]}) little-endian float32, rows L2-normalized | trained on OpenStreetView-5M (CC BY-SA 4.0); initialised from StreetCLIP text embeddings of country names (CC BY-NC 4.0) |
| `{CELLS_FILE}` | {mb[CELLS_FILE]} | cell centroids (lat, lon) of a quadtree built from OpenStreetView-5M training coordinates | CC BY-SA 4.0 (derived from OpenStreetView-5M) |
| `inference_config.json`, `export_info.json` | | preprocessing, file hashes, provenance | MIT |

The files are kept separate because CC BY-NC 4.0 and CC BY-SA 4.0 are not compatible licences for a
single combined work. The head is derived from both sources; it is published only for this
non-commercial demo. This is not legal advice; if you want to reuse a file, check both licences.

## Performance

On the official OpenStreetView-5M test split ({int(cell["test_n"]):,} images), scored once in the
pre-registered V3 evaluation, this head (StreetCLIP, natural-order N = 500,000, cell probe with
zero-shot initialisation; prediction = centroid of the top-1 cell): {perf(cell)}.
GeoScore = 5000 · exp(−km / 1492.7) per image; brackets: 95% interval from a creator-clustered
bootstrap (1,000 resamples). Source: [`results/v3/test_results.csv`]({readme}/blob/master/results/v3/test_results.csv).

The best V3 predictor for the same backbone and subset is the zero-shot hybrid ({perf(best)}), which
also searches the 500,000 training embeddings, so it cannot run in a browser and is not in this repo.

The demo uses the 8-bit encoder by default. Its agreement with the PyTorch model and its GeoScore
next to fp32 on 500 test photos are in the [V3 demo table]({readme}#v3-demo).

## Intended use and limitations

- For the demo and for research only (and non-commercial, see above). Not for any decision about
  where a photo was taken.
- Trained on street-level imagery. Indoor photos, portraits, screenshots or other personal photos give
  meaningless output: the model always returns cells, whatever the photo shows.
- Biased toward regions with many OpenStreetView-5M images: cells are small there and large elsewhere,
  so photos from sparse regions are probably located less precisely (not measured per region here).
- The probabilities are softmax outputs and are not calibrated.
- **Must not be used to locate individuals or private persons**, or to find where someone lives or was.

## Use

Preprocessing (as the embeddings it was trained on): short side to {cfg["image_size"]} px (bicubic),
long side truncated, one center crop, CLIP mean/std, no flips.
`logits = {cfg["logit_scale"]:g} * image_embeds @ head.T`; softmax over the cells. The probabilities
are the model's softmax outputs, not calibrated confidence. A cell is an area; its centroid is a point.

## Sources

- StreetCLIP: Haas, Alberti, Skreta, *Learning Generalized Zero-Shot Learners for Open-Domain Image
  Geolocalization*, arXiv:2302.00275 (2023). CC BY-NC 4.0.
- OpenStreetView-5M: Astruc et al., CVPR 2024. CC BY-SA 4.0.
- Code: [github.com/Daggerabuse/geoguessr-ai](https://github.com/Daggerabuse/geoguessr-ai) (MIT),
  `geoguessr_ai.export_geo_demo`. Results and limitations of the cell probe: `results/v3/` in the repo.
"""


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL, help="Hub id or local folder of a CLIP checkpoint")
    ap.add_argument("--head", type=Path, required=True, help="head.npy of a Phase 3 cell probe")
    ap.add_argument("--cells", type=Path, required=True, help="geocells JSON (frozen partition)")
    ap.add_argument("--partition-sha256", default=None,
                    help=f"expected sha256 of --cells (default: the pre-registered one in {PREREG})")
    ap.add_argument("--out", type=Path, default=Path("models/streetclip_geocells"))
    ap.add_argument("--verify", type=Path, default=None, help="image folder for the parity check")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N images (sorted by name)")
    ap.add_argument("--limit", type=int, default=None, help="use at most N images")
    ap.add_argument("--report", type=Path, default=None, help="write the parity check as JSON here")
    ap.add_argument("--meta", type=Path, default=None,
                    help="CSV with id, latitude, longitude, creator_id of the --verify photos: adds "
                         "GeoScore against the true coordinates (and a per-image CSV next to --report)")
    ap.add_argument("--no-export", action="store_true",
                    help="keep the exported files in --out (only the model card is rewritten)")
    args = ap.parse_args(argv)
    if args.partition_sha256 is None:
        args.partition_sha256 = json.loads(PREREG.read_text())["geocells"]["sha256"]
    return args


def run(args: argparse.Namespace) -> dict:
    if args.no_export:
        info = json.loads((args.out / "export_info.json").read_text())
        write_card(args.out)  # cheap and deterministic; keeps the card in sync with the CSV
    else:
        info = export(args.model, args.head, args.cells, args.out, args.partition_sha256)
        print(json.dumps(info, indent=2))
    if not args.verify:
        return info
    images = eo.list_images(args.verify, args.limit, args.offset)
    print(f"parity check on {len(images)} images ...", flush=True)
    rep, top1 = verify(args.model, args.out, images)
    rep["export"] = info
    if args.meta:
        import pandas as pd

        meta = pd.read_csv(args.meta, dtype={"id": str}).set_index("id").loc[[p.stem for p in images]]
        cells = json.loads((args.out / CELLS_FILE).read_text())
        rep["geo"] = geo_accuracy(top1, cells, meta)
        if args.report:
            meta[["latitude", "longitude", "creator_id"]].assign(
                **{f"top1_cell_{k}": v for k, v in top1.items()}).to_csv(
                args.report.with_name(args.report.stem + "_per_image.csv"))
    print(json.dumps(rep, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2))
    return rep


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
