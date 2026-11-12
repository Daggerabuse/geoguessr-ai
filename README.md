# geoguessr-ai

![CI](https://github.com/Daggerabuse/geoguessr-ai/actions/workflows/ci.yml/badge.svg)

Predicting the country of street-level photos with CLIP and fine-tuning. The task is 33-way
country classification on a subset of [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m),
comparing zero-shot CLIP, a linear probe and full fine-tuning (LP-FT).

Main result: CLIP ViT-B/16 with LP-FT reaches 0.708 top-1 on the test split with a single center crop,
24.0 points above the zero-shot accuracy of the same backbone, and on par in top-1 with StreetCLIP
zero-shot (0.708; ViT-L/14, 336 px; paired test p = 0.95), though StreetCLIP is significantly ahead
in top-5 (0.958 vs. 0.925). Test-time augmentation (TTA, average over 3 crops) was adopted because it
improved the val split (0.787 to 0.808). With the same TTA on both models, LP-FT reaches 0.732 test top-1
and StreetCLIP 0.726. In a paired test on the same 6600 images the 0.6-point difference is not
significant (95% interval -0.8 to +1.8 points, McNemar p = 0.41), so the two are on par in top-1.
In top-5, StreetCLIP is ahead (0.963 vs. 0.935), and this difference is significant
(-2.8 points, 95% interval -3.5 to -2.2, p < 0.001).

- **Live demo (runs in your browser):** https://huggingface.co/spaces/Daggerabuse/geoguessr-ai
- **Model:** https://huggingface.co/Daggerabuse/geoguessr-clip-vitb16

## Why

GeoGuessr is one of my hobbies. Watching pro players name a country in under half a second, often from little more than the colour of a road line or the shape of a bollard, made me wonder what a model would pick up on. I play to improve myself too, but after a data mining course at İzmir Institute of Technology I decided to move towards machine learning. Last year I started a pathing tool for Elden Ring Nightreign that I never finished; I started this project partly because working with images and fine-tuning seemed like the skills that project was missing. I learned a lot here: building a dataset, setting up fair baselines, and reading a confusion matrix instead of trusting a single accuracy number.

## Results

Test split: 6600 images (200 per country, 33 countries); val split: 3300 images. Random guessing
is about 0.03 top-1. All numbers are read from `results/*/{test,val}_metrics.json`.

| Model | Test top-1 | Test top-5 | Val top-1 |
|---|---|---|---|
| CLIP ViT-B/16, zero-shot | 0.468 | 0.830 | 0.485 |
| CLIP ViT-L/14, zero-shot | 0.544 | 0.903 | 0.598 |
| StreetCLIP, zero-shot (ViT-L/14, 336 px) | 0.708 | 0.958 | 0.723 |
| StreetCLIP, zero-shot + TTA (3 crops) | 0.726 | 0.963 | 0.752 |
| CLIP ViT-B/16, linear probe | 0.600 | 0.900 | 0.708 |
| CLIP ViT-B/16, LP-FT (fine-tuned) | 0.708 | 0.925 | 0.787 |
| CLIP ViT-B/16, LP-FT + TTA (3 crops) | 0.732 | 0.935 | 0.808 |

Gain from zero-shot to LP-FT on the same backbone: 24.0 points (test top-1).

TTA (average over the left, center and right crops) was chosen based on the val split: for the same
checkpoint, evaluated with `--eval-only --tta` and no retraining, val top-1 went from 0.787 to 0.808.
On test, top-1 is 0.732 with TTA versus 0.708 without. StreetCLIP was evaluated with the same 3-crop
TTA (logits averaged): test top-1 goes from 0.708 to 0.726.

LP-FT and StreetCLIP were compared image by image, both without TTA and both with the same TTA
(`results/analysis/paired_comparison.csv`, from `results/*/test_predictions.csv`;
difference = LP-FT - StreetCLIP):

| | Difference | 95% interval (paired bootstrap) | Only LP-FT right | Only StreetCLIP right | McNemar p |
|---|---|---|---|---|---|
| Test top-1, no TTA | +0.1 points | -1.3 to +1.4 | 976 | 972 | 0.95 |
| Test top-5, no TTA | -3.3 points | -4.0 to -2.6 | 184 | 402 | < 0.001 |
| Test top-1, TTA | +0.6 points | -0.8 to +1.8 | 967 | 930 | 0.41 |
| Test top-5, TTA | -2.8 points | -3.5 to -2.2 | 169 | 355 | < 0.001 |

In both settings the top-1 difference is not significant, so the two are treated as on par in
top-1; the top-5 difference is significant, so StreetCLIP is better in top-5. On par does not mean
the two get the same images right: each gets about 970 test images right that the other gets wrong.
The bootstrap resamples images
(10000 resamples, seed 42) and treats them as independent; images from the same sequence are
correlated, so the true intervals may be somewhat wider.

## Findings

Details are in `results/analysis/report.md`; per-country accuracy is in `results/*/test_per_country.csv`.
With 200 test images per country, the 95% interval of one country's accuracy is roughly +-7 points.
The difference between two models has a standard error about sqrt(2) times larger, so its 95%
interval is roughly +-10 points; that is why only differences larger than 10 points are discussed below.

- **Where the gain comes from.** On the same ViT-B/16 backbone, the linear probe adds 13.2 points
  over zero-shot and full fine-tuning adds another 10.8 points on top of the probe (test top-1).
- **Test-time augmentation.** Adopted because of the val split, not the test split. For the same checkpoint,
  test top-1 is 0.732 with TTA versus 0.708 without. It changes the confusion picture only slightly:
  SE to FI is 20 (19 without TTA), CA to US 7 (7), NG to GH 32 (32),
  US to CA 32 (34); the attractor countries are ES, TR (DK, ES without TTA).
  The Nordic, Canada/US, unsolved-confusion and attractor counts below are all from runs without TTA
  (LP-FT and StreetCLIP alike); only the per-country comparison uses TTA on both models.
- **Nordic confusion.** Swedish images predicted as Finland (SE to FI, out of 200 per country):
  115 for B/16 zero-shot, 38 after the linear probe, 19 after LP-FT
  (119 for L/14 zero-shot, 23 for StreetCLIP). For StreetCLIP the dominant direction is
  the reverse: FI to SE is 64, versus 11 for LP-FT.
- **Canada vs. United States.** CA to US: 38 for B/16 zero-shot, 14 after the probe,
  7 after LP-FT (150 for L/14 zero-shot, 37 for StreetCLIP).
- **Per country, LP-FT vs. StreetCLIP, both with the same 3-crop TTA (test top-1, differences > 10 points).**
  - Fine-tuned model better (7 countries): GH (87.0% vs 40.0%, 47.0 points), FI (91.5% vs 56.5%, 35.0 points), TW (73.5% vs 44.0%, 29.5 points), PL (70.0% vs 43.0%, 27.0 points), DK (82.0% vs 63.0%, 19.0 points), ZA (84.5% vs 72.0%, 12.5 points), TH (90.5% vs 78.5%, 12.0 points).
  - Fine-tuned model worse (8 countries): IT (48.5% vs 73.5%, 25.0 points), ID (62.5% vs 86.5%, 24.0 points), US (69.0% vs 92.5%, 23.5 points), NG (66.0% vs 86.5%, 20.5 points), CL (54.5% vs 71.5%, 17.0 points), BR (66.0% vs 81.5%, 15.5 points), RU (68.5% vs 82.0%, 13.5 points), JP (83.5% vs 96.5%, 13.0 points).
  - All other countries differ by 10 points or less, which is within the noise level here.
- **Confusions that were not solved, only changed direction.** GH to NG fell from 108 to 6
  while NG to GH is 32 after LP-FT (it was 24 for B/16 zero-shot). US to CA is
  34 after LP-FT (CA to US is only 7); it was 21 for B/16 zero-shot.
- **Attractor countries largely disappear.** Countries that are over-predicted (predicted/true count
  at least 1.3) while precision is below recall: 7 for B/16 zero-shot, 5 for L/14 zero-shot,
  4 for StreetCLIP, 5 after the linear probe, 2 after LP-FT
  (DK, ES). See `results/analysis/precision_recall.csv`.

### Figures

Per-country test top-1 of the fine-tuned ViT-B/16 (LP-FT) and StreetCLIP zero-shot, both with the
same 3-crop TTA, sorted by the difference between them. The 95% interval of a single dot is about
+-7 points, and that of the gap between two dots about +-10 points, so gaps smaller than about
10 points should not be read as real differences.

![Per-country test top-1: LP-FT with TTA vs. StreetCLIP zero-shot with TTA](results/analysis/per_country_lpft_vs_streetclip.png)

Test confusion matrices of the same backbone before and after fine-tuning (rows are the true
country, columns the predicted one, each row normalized):

<p>
  <img src="results/clip_vitb16_zeroshot/test_confusion.png" width="49%" alt="Confusion matrix: ViT-B/16 zero-shot">
  <img src="results/clip_vitb16_lpft/test_confusion.png" width="49%" alt="Confusion matrix: ViT-B/16 LP-FT">
</p>

Left: CLIP ViT-B/16 zero-shot. Right: CLIP ViT-B/16 LP-FT.

## Demo

[Live demo](https://huggingface.co/spaces/Daggerabuse/geoguessr-ai) (a static Hugging Face Space, `web/`).
The fine-tuned model (vision encoder + head, one graph, 33 logits) is exported to ONNX
(`export_onnx`) and runs in the browser with onnxruntime-web, so the photo never leaves the device.
Preprocessing is the same as in evaluation (short side 224 with PIL's bicubic reimplemented in JS,
left/center/right crops, no flips); a test checks that the JS and Python outputs are identical.

The demo uses 8-bit weights (MatMulNBits, blocks of 32; activations stay fp32), 101 MB instead of
346 MB. Activation quantization (`quantize_dynamic`) was tried first and dropped: top-1 agreement
with PyTorch fell to 0.82-0.85 on the 500 images used for choosing the method. Agreement with the
PyTorch model (3-crop TTA), measured on 2500 other images from `data/sample`
(`results/onnx_parity/`):

| | fp32 ONNX | 8-bit ONNX |
|---|---|---|
| Top-1 agreement, onnxruntime (2500 images) | 1.000 | 0.986 |
| Max logit difference, onnxruntime | 1e-4 | 0.27 |
| Top-1 agreement in the browser (Edge, onnxruntime-web) | 1.000 (100 images) | 0.995 (200 images) |

These are agreement numbers, not accuracy. The onnxruntime rows come from `results/onnx_parity/report.json`
(`export_onnx --verify`). The browser row was measured locally in an installed Edge (headless, via
Playwright, `webdemo check`); its results are in `results/onnx_parity/web_fp32.json` and
`results/onnx_parity/web_int8.json`. That check needs the local model files and Edge, so it is not
part of CI. Details: [NOTES_ONNX.md](NOTES_ONNX.md).

## Limitations

- Single run with a single seed; the variance of training itself is unknown.
- The 33 countries do not represent the world, and the numbers are not comparable to published
  OSV-5M benchmark results (different country set and metrics).
- StreetCLIP is a larger model (ViT-L/14) at a higher resolution (336 px) than the fine-tuned ViT-B/16,
  and it is evaluated zero-shot, while the fine-tuned model was trained on this country set.
- TTA was selected on the val split, which is optimistic (see below); the test gain is a single measurement.
- Val is split from the same pool as train (by sequence), so it is optimistic: the val-test gap is
  7.8 points for LP-FT, versus 1.5 for StreetCLIP and 1.6 for B/16 zero-shot. Compare models on test.
- Over-fitting: in the last of 10 epochs, train top-1 is 0.968 versus 0.787 on val. The
  suggestion that more data would help is an untested inference from this gap.
- The torch-dependent code was only run on Kaggle; it is not covered by the local tests.

## Setup
Locally (no torch/transformers; exploration, data prep and code only):
```
uv sync
```
Kaggle/Colab (internet on). torch and transformers usually come preinstalled, and
`pip install -e .` installs only the core dependencies. If they are missing: `pip install -e ".[ml]"`.
```
pip install -e .
```
To get the ML packages locally as well: `uv sync --extra ml` (large download).

## Usage
### Pipeline

```mermaid
flowchart LR
    HF[(OSV-5M on Hugging Face)] --> P[prep_data]
    P --> S[subset: manifest.csv + images]
    S --> B[baseline_clip: zero-shot]
    S --> T[train: linear probe, then fine-tuning]
    B --> R[results/]
    T --> R
    R --> A[analyze]
    A --> AN[results/analysis/]
```

Locally (needs `data/sample`):
```
uv run python -m geoguessr_ai.download_sample
uv run python -m geoguessr_ai.explore
```

On Kaggle:
```
# 1. country distribution, then the subset (manifest + images, resized to short side 256)
python -m geoguessr_ai.prep_data --stats-only
python -m geoguessr_ai.prep_data --out /kaggle/working/subset --resize 256

# 2. zero-shot baselines (results go to results/)
python -m geoguessr_ai.baseline_clip --root /kaggle/working/subset
python -m geoguessr_ai.baseline_clip --root /kaggle/working/subset \
    --model geolocal/StreetCLIP --out results/streetclip_zeroshot
# same backbone as the fine-tuned model (fair comparison)
python -m geoguessr_ai.baseline_clip --root /kaggle/working/subset \
    --model openai/clip-vit-base-patch16 --out results/clip_vitb16_zeroshot
# StreetCLIP with the same 3-crop TTA as train.py --tta (logits averaged)
python -m geoguessr_ai.baseline_clip --root /kaggle/working/subset \
    --model geolocal/StreetCLIP --tta --out results/streetclip_zeroshot_tta

# 3. fine-tuning: quick check with a tiny random model (no download), then the real run
python -m geoguessr_ai.train --root /kaggle/working/subset --smoke
python -m geoguessr_ai.train --root /kaggle/working/subset [--tta]

# evaluate a saved checkpoint only (no probe, no training); results go to results/<run-name>/
python -m geoguessr_ai.train --root /kaggle/working/subset --eval-only --checkpoint <path to best.pt> --tta --run-name clip_vitb16_lpft_tta

# 4. confusion analysis of every run under results/ (writes results/analysis/)
python -m geoguessr_ai.analyze

# 5. export to Hugging Face: vision encoder + head (safetensors), classes.json,
#    inference_config.json and a model card with the test results from results/.
#    Check the files first, then push (needs a write token in HF_WRITE_TOKEN)
python -m geoguessr_ai.export_hf --checkpoint <path to best.pt> --out /kaggle/working/hf_export
python -m geoguessr_ai.export_hf --checkpoint <path to best.pt> --out /kaggle/working/hf_export_push \
    --push Daggerabuse/geoguessr-clip-vitb16
```

The output folder must be empty (it is uploaded as a whole). The repo is created as public;
add `--private` for a private one. On Kaggle, store the token as a secret and set the
environment variable in the notebook (never print it):
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_WRITE_TOKEN"] = UserSecretsClient().get_secret("HF_WRITE_TOKEN")
```

### Demo

```
# ONNX export (fp32 + 8-bit) of the Hub model, with a parity check against PyTorch
uv sync --extra ml --extra onnx
uv run python -m geoguessr_ai.export_onnx --model models/hf_export --out models/onnx \
    --verify data/sample/images --offset 500 --report results/onnx_parity/report.json
# serve web/ locally with the local model: http://localhost:8000/web/?base=/model/
uv run python -m geoguessr_ai.webdemo serve
```

`app/` (a Gradio version of the demo) is kept but not used: Hugging Face Gradio Spaces require a
paid plan, so the live demo is the static page in `web/`.

## Tests

```
uv run pytest
uv run ruff check
```

The tests use small synthetic images and manifests and do not need torch/transformers or any
downloaded data, so they run locally. The torch-dependent code (model loading, linear probe,
fine-tuning loop) is not covered by them; it was only run on Kaggle.

## Fine-tuning (`train.py`)
CLIP ViT-B/16 with a classification head initialised from country-name text embeddings
(the template is stored in `run_config.json`). Two stages: a linear probe on frozen features,
then full fine-tuning (AdamW, fp16 AMP, cosine schedule, label smoothing).
No horizontal flip (driving side is a cue); training uses a random near-square crop plus light
brightness/contrast/saturation jitter, small blur and small random erasing (no hue shift).
Evaluation: short side 224 + center crop; `--tta` averages left/center/right crops.

- Best checkpoint (by val top-1): `models/<run>/best.pt`.
- Results use the same file format as `baseline_clip`, so runs are directly comparable:
  `results/<run>/{val,test}_{metrics.json,per_country.csv,confusion.csv,confusion.png,predictions.csv}`,
  linear-probe results in `results/<run>/probe/`, plus `history.csv` and `run_config.json`.
- The test split is evaluated only after training, with the best checkpoint.
- Val is split from the same pool as train (by sequence), so it can be more optimistic than the
  official test split; compare runs on test.

`data/` and `models/` are not tracked; `results/` is.

## V2: global geolocation (in progress)

V1 (the 33-country model above, its checkpoint, results and demo) is unchanged. V2 studies
frozen-embedding geolocation (kNN retrieval and a linear probe on cached embeddings) over
every country of the official OSV-5M test split. Outputs go to `results/v2/`.

**Class list.** The classes are the countries that occur in the official test split *and*
in the official train split: `classes = test countries ∩ train countries` (`v2_data.py`,
`classes.json`). `audit.json` lists the countries found only in train (never a class, so their
training images are not used) and only in test (no training data). Test images of a test-only
country stay in the evaluation with label -1. A trained head can never predict them, so in
every global metric they **count as errors** (top-1 and top-5), and in the macro average each
such country counts with accuracy 0. They are also reported apart: the `class_in_train` group
contains only images of countries with training data. The number of countries on each side is
written to `audit.json` by the audit step.

**Splits.** `train.csv`: the training pool, at most 3 frames per sequence, with two nested
orderings (`nat_rank`: natural distribution, `bal_rank`: country round-robin); a subset of
size N is `rank < N`, so smaller subsets are contained in larger ones. `dev_seen.csv`:
20K images (1 per sequence) from seeded random sequences of photographers who keep other
sequences in train, like 98% of the official test images; these sequences leave the pool,
and all choices (k, probe epoch) are made on dev_seen. Unlike the official test, it does not
enforce a 1 km distance to training images. `dev.csv` ("dev_unseen", second report): images
(1 per sequence) from a seeded 2% of the photographers, none of whom is in `train.csv`, so
it measures generalization to new photographers and scores lower than test by design.
`test.csv`: the full official test split. It is
scored only with `geo_eval --eval-test` (final configurations), and no smoke test or
benchmark reads it. Train rows sharing an id or a sequence with test are dropped (counted
in `config.json`).

**Reporting.** Every result is also split into row groups: `reliable` countries (at least
100 test images and 50 test sequences), `head`/`middle`/`tail` (thirds of the class list by
official training images) and photographer seen/unseen in the training subset. A group only
selects rows; predictions always range over the full class list. Distance metrics follow
OSV-5M (haversine km, GeoScore = 5000·exp(−km/1492.7), accuracy within 1/25/200/750/2500 km);
intervals are a bootstrap over sequences. Only results on the full official test split with
these metrics are comparable with the OSV-5M paper.

**Embeddings.** `embed.py` streams images from the shard zips (no image is stored) and
preprocesses them exactly like each model's `CLIPImageProcessor` (V1's evaluation crop
rounds the resized long side instead of truncating it, a one-pixel difference). One cache
directory per encoder and GPU type (T4 and P100 caches are never mixed): float16,
L2-normalized, `.npz` per shard and run; resumable, and `--parts 2 --part 1 --device cuda:1`
runs a second process on the second T4. `--limit N` picks a seeded sample of N ids for all
parts together (the same sample on every rerun). `--check` verifies a cache: every target
id exactly once, and with `--reference` the same embeddings as another cache.

`scripts/kaggle_phase0.sh` runs the whole validation session on Kaggle (audit, throughput
of each backbone, 2,000-image smoke test, dual-T4 run with an interrupted and resumed part);
it stops at the first failed check.

**Phase 1 reading (local zips).** Sparse range reads were network-bound in Phase 0, so the
full run downloads each of the 98 train zips once (`zipq.py`: one stream at a time, at most
2 zips ahead, a zip deleted when all consumers marked it done, at least 20 GB free required
before each download). One long-lived `embed.py --zip-dir` process per backbone reads the
same local zip; `ft_archive.py` copies the fine-tuning images (nat_200K, dev, dev_seen) byte
for byte into ~1 GB ZIP_STORED parts with a sha256 manifest. Each part is uploaded to a
private Hugging Face dataset repo as soon as it is committed (skipped if the repo already has
the same sha256), checked against the repo's sha256 and deleted locally; `manifest.csv` goes
last. `scripts/kaggle_phase1.sh` runs this end to end in one of two modes (settings and the
repo guard in `phase1.py`): `kaggle_pilot.sh` = 3 zips, 40 min, archive only to the separate
`-pilot` repo; `kaggle_full.sh` = all 98 zips (pool + dev + dev_seen; no test image), 11 h,
archive only to `Daggerabuse/osv5m-ft-archive`. Both print a progress line whenever a
consumer finishes a zip (zips done, img/s, ETA, disk use), log quota timestamps and disk
peaks, and at the deadline write a STOP marker: every process finishes the zip in hand and
starts no new one, so completed zips stay valid, and the script prints RESUME NEEDED with the
zips left (exit 3) instead of running the checks. A rerun with the embedding cache restored
skips what is already embedded; rebuilt archive parts are byte-identical and are not
uploaded again.
GPU layout: StreetCLIP alone on GPU1; L/14-336 and B/16 together on GPU0. Phase 1 does not
compare its embeddings with the Phase 0 cache (`--reference`): the Phase 0b audit found the
local-zip bytes identical to the range-read bytes (sha256, 200/200), and both read modes feed
the same decode, preprocessing and model code.

Before a Kaggle run, check the token and the upload path locally (a dummy 2-part archive is
uploaded to `smoke/` in the pilot repo, re-downloaded and verified; prints the private flag):
```
$env:HF_TOKEN = "<token with write access>"   # PowerShell; bash: export HF_TOKEN=...
uv run python -m geoguessr_ai.ft_archive smoke --repo Daggerabuse/osv5m-ft-archive-pilot
```

Pilot notebook (2x T4, internet on). Cell 1 (the secret from Kaggle Secrets, never printed):
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```
Cell 2:
```
!git clone <repo url> /kaggle/working/geoguessr-ai && cd /kaggle/working/geoguessr-ai && pip install -q -e .
!mkdir -p /kaggle/working/pilot && cd /kaggle/working/geoguessr-ai && bash scripts/kaggle_pilot.sh 2>&1 | tee /kaggle/working/pilot/pilot.log
```
Full run: the same Cell 1, then
```
!git clone <repo url> /kaggle/working/geoguessr-ai && cd /kaggle/working/geoguessr-ai && pip install -q -e .
!mkdir -p /kaggle/working/full && cd /kaggle/working/geoguessr-ai && bash scripts/kaggle_full.sh 2>&1 | tee /kaggle/working/full/full.log
```
Outputs: `/kaggle/working/full/` (logs, checks, progress, disk peaks, quota times) and the
embeddings in `/kaggle/working/emb/`.

**Phase 1 analysis.** `phase1_eval.py` asks whether geographic pretraining (StreetCLIP)
improves sample efficiency over its matched base (CLIP L/14-336) and B/16, and whether the
gap closes as N grows. Grid: backbone x ordering (nat, bal) x N (50K-500K) x method (kNN,
linear probe) x probe init (random, or zero-shot from the backbone's text encoder, which
separates image-representation quality from text-encoder quality); each cell uses
`geo_eval.run_cell` (k, or the probe's learning rate and stopping epoch jointly, chosen on
dev_seen only; scored on dev_seen and dev_unseen) and is cached, so a rerun skips finished
cells. The primary macro top-1 averages over a fixed class set per split (every country in
it, comparable across N), each with its actual accuracy: a zero-shot head can predict a
country absent from the training subset; top-1 on those images is a separate column. Outputs:
`results.csv` (one row per cell), `gaps.csv` (paired StreetCLIP - L/14-336 differences on the
same images), `scaling.png` and `gap.png`. Intervals are cluster bootstraps over sequences,
and over creators for dev_unseen. The probe search is learning rate in {1e-3, 3e-3, 1e-2,
3e-2, 1e-1} x epochs up to 60, a run stopped after 8 epochs without a dev_seen gain;
`grid_edge` flags a choice at an edge of the search. The 1e-1 step was added after the
previous grid's top edge was chosen often; it was run incrementally (stored runs of the
other learning rates with the same settings reused, choice made over the merged sweep).
Stopping rule: this is the final grid extension. If 1e-1 (or 1e-3) is chosen in some cells,
the grid is frozen as it is and those cells are reported as a limitation; the `grid_edge`
flags stay in `results.csv` and in the pre-registration. The official test is read only with
`--eval-test`, into a separate `<out>_test` directory. On Kaggle, attach the full run's output as input and run:
```
!git clone -b v2-phase0 https://github.com/Daggerabuse/geoguessr-ai.git /kaggle/working/geoguessr-ai && cd /kaggle/working/geoguessr-ai && pip install -q -e .
!cd /kaggle/working/geoguessr-ai && bash scripts/kaggle_phase1_eval.sh
```

**Pre-registration and the one-shot test evaluation.** Before the official test split is
embedded, `python -m geoguessr_ai.prereg write --dev-out <dev phase1_eval dir>` writes
`configs/phase1_final.json` and it is committed. It fixes the cells reported on test, their
hyperparameters (the dev_seen-selected k, or learning rate and epoch, used as they are: no
re-tuning on test, `grid_edge` kept per cell), the head settings, the search grid with its
stopping rule (`search`) and the hypotheses. Primary: (H1) the paired StreetCLIP
- L/14-336 test top-1 gap at N=50K is > 0, and (H2) the gap shrinks from 50K to 500K, each per
variant x ordering: 12 hypotheses. Each gets a one-sided p-value from 10,000 paired
cluster-bootstrap resamples on test (sequence clusters; the same resample applies to both
backbones and both N): p = (1 + number of resampled statistics <= 0) / (10,000 + 1). The 12
p-values are Holm-Bonferroni adjusted (family-wise alpha 0.05; a hypothesis whose cells were
not run counts with p = 1), and a hypothesis is supported when its adjusted p < 0.05. The
uncorrected 95% percentile interval is reported with each. Everything else is exploratory. Then
`bash scripts/kaggle_phase1.sh test` embeds the 5 test zips into `emb_test/`; it refuses to
start unless the pre-registration is committed and unmodified, and records its hash, commit
and the start time. Finally `phase1_eval --eval-test --prereg configs/phase1_final.json
--test-emb-root <emb_test>` retrains each declared cell once with its frozen hyperparameters,
scores test into `<out>_test`, prints the primary results first (`primary.csv`) and labels
every other output exploratory.

What this does and does not guarantee: it is a self-discipline mechanism, not proof. The
checks stop accidental test-driven choices (re-tuning on test, editing the plan after seeing
results, scoring before committing), and the git history shows the order of events. But git
timestamps can be set by whoever makes the commit, history can be rewritten, the provenance
file is a plain JSON written by our own script, and nothing is signed or timestamped by a
third party. A reader has to trust that the steps were followed; anyone who wants to cheat
can. Pushing the commit to GitHub before the test run gives a weaker but external record
(GitHub's push time), nothing more.

```
# audit only (countries, overlap, groups), then the manifests
python -m geoguessr_ai.v2_data --out /kaggle/working/v2 --audit-only
python -m geoguessr_ai.v2_data --out /kaggle/working/v2 --sizes 50000 100000 200000 500000
# throughput: GPU-only benchmark, then a 2000-image end-to-end smoke test on training images
python -m geoguessr_ai.bench --model openai/clip-vit-base-patch16 --batch-size 64
python -m geoguessr_ai.embed --ids /kaggle/working/v2/train.csv --source train \
    --model openai/clip-vit-base-patch16 --out /kaggle/working/emb/clip_b16 --limit 2000
# embeddings of the training pool and dev (test only for final configurations)
python -m geoguessr_ai.embed --ids /kaggle/working/v2/train.csv --source train ...
python -m geoguessr_ai.embed --ids /kaggle/working/v2/dev.csv --source train ...
# kNN and probe per subset size, scored on dev; add --eval-test for final configurations
python -m geoguessr_ai.geo_eval --data /kaggle/working/v2 --emb /kaggle/working/emb/clip_b16 \
    --model openai/clip-vit-base-patch16 --tag clip_b16 --strategy bal \
    --sizes 50000 100000 200000 500000
```

## Future work

- Split val by creator (not only by sequence) to reduce the val-test gap.
- Linear probe on StreetCLIP.
- More images per country.

## License

- Code: MIT, see [LICENSE](LICENSE).
- Data: [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m), **CC BY-SA 4.0**.
  Astruc et al., "OpenStreetView-5M: The Many Roads to Global Visual Geolocation", CVPR 2024.
- Models: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14),
  [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16),
  [geolocal/StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) (**CC BY-NC 4.0**, no commercial use).

## How I built this

I built this project with Claude Code as a pair programmer: it handled much of the implementation and suggested standard techniques from the fine-tuning literature (LP-FT, a head initialised from text embeddings, Wilson intervals). My part was the project itself:

- **Scope and setup:** I chose 33 countries, built the data pipeline on Kaggle's free GPUs, and ran every experiment there.
- **Experiment design:** I set the order (zero-shot baselines, then a linear probe, then full fine-tuning on the same backbone) so that each step's gain could be measured on its own.
- **Ground rules:**
  - no horizontal flip, because driving side is a real cue;
  - the test split is used once, after training;
  - every number in this README comes from `results/`.
- **Review:** I read the code before each commit and kept the AI to fixed rules in a `CLAUDE.md` file: no commits on its own, seeds everywhere, ask before changing behaviour.
- **Interpretation:** I report "on par" with StreetCLIP rather than "beats", and list the limitations above.

Things that went wrong and what they taught me:

- A Kaggle "Quick Save" created an empty notebook version, and my training notebook silently read from it. Now I pin dataset versions.
- A newer `transformers` release changed what CLIP's feature functions return and broke the baseline. That is why library versions are now logged in `run_config.json`.
- My validation split came from the same pool as training, so validation looked about 8 points better than test. I report test numbers only, and a split by photographer is the first item in future work.