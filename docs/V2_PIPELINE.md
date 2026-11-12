# V2 pipeline: splits, embeddings, Kaggle runs and pre-registration

V1 (the 33-country model in the main README, its checkpoint, results and demo) is unchanged. V2 studies
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
19,906 images (1 per sequence; 20K drawn, picks whose photographer would keep no other train sequence are dropped) from seeded random sequences of photographers who keep other
sequences in train, like 98% of the official test images; these sequences leave the pool,
and all choices (k, probe learning rate and epoch) are made on dev_seen. Unlike the official test, it does not
enforce a 1 km distance to training images. `dev.csv` ("dev_unseen", second report): images
(1 per sequence) from a seeded 2% of the photographers, none of whom is in `train.csv`, so it measures generalization to new photographers. (In practice the official test scored lower still; see the exploratory findings in the README.)
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
--test-emb-root <emb_test>` (on Kaggle: `bash scripts/kaggle_phase1_final.sh`, which finds
both attached inputs) retrains each declared cell once with its frozen hyperparameters,
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

**Phase 3 (coordinates, development splits only).** `phase3_eval.py` asks whether StreetCLIP's
advantage over CLIP L/14-336 holds in km. On the same nested subsets, one kNN search per subset
feeds three coordinate predictors: the nearest neighbour, the weighted mean of the k neighbours on
the sphere (3D unit vectors; weights the similarity or exp((sim - best sim)/tau), tau in {0.1,
0.03, 0.01}), and the similarity-weighted geographic medoid (Phase 1's kNN coordinates, which it
reproduces). k and tau are chosen by mean GeoScore on dev_seen only; median km is reported, not
selected on. Country accuracy from a predicted point (the country of the nearest training image in
the subset) is a sanity check against Phase 1. Paired StreetCLIP - L/14-336 differences per image
(GeoScore, accuracy within 25/200/750/2500 km, km, country) and of median km use the Phase 1
cluster bootstrap. The official test split is never read: `phase3_eval` has no test option, and a
test-level coordinate evaluation gets its own pre-registration first. The Phase 1 final evaluation
already computed exploratory kNN (medoid) test distance metrics (`results/v2/test_results.csv`);
Phase 3 code never reads them and they were not looked at. On Kaggle (CPU-only session, the full
run's output attached): `bash scripts/kaggle_phase3.sh`.
