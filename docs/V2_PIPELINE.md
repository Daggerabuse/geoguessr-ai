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

**Phase 1 analysis.** `phase1_eval.py` compares StreetCLIP vs its CLIP L/14-336 base (one model
pair; B/16 is also run) at small training sizes, and asks whether the gap closes as N grows.
Sample efficiency is estimated post hoc via N-equivalence (`phase1_posthoc.py`). Grid: backbone x ordering (nat, bal) x N (50K-500K) x method (kNN,
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

`dev_results` in the pre-registration points to `runs/` (not in the repo). The same file is
committed byte for byte as `results/v2/phase1_dev_results.csv` (dev splits only, no test
column; stored with LF line endings, see `.gitattributes`), so the recorded hash can be checked:
`sha256sum results/v2/phase1_dev_results.csv` must print `dev_results_sha256` of
`configs/phase1_final.json`.

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

Review fixes (dev step). dev_seen shares creators with train without a distance buffer, so
`--mask-km` (default 1) removes, per dev query, every training image within that haversine
distance of its true location before the top-k search; the masked count per split is printed and
stored (`*_masked_mean`, `*_masked_any_share`). Baselines on the same splits: a constant
(spherical mean of the subset), predicted country -> that country's training centroid (Phase 1's
frozen `probe_random` dev predictions from `--phase1-dev`, else the masked kNN vote), the oracle
nearest training image after masking (a lower bound) and the oracle best of the retrieved
neighbours. The constant and the nearest-image oracle do not depend on the backbone and get no
paired gap. kNN numerics are pinned: `--knn-dtype` (default fp32) and the device are recorded in
every `row.json`; the Kaggle script runs numpy fp32 on the CPU, and the Phase 3 selection and any
test run must use the same setting (Phase 1 ran fp16 on the GPU). A finished cell is reused only
if its `cache_key` matches: CACHE_VERSION, the sha256 of the data dir's `config.json`, the
embedding cache (encoder.json hash, file names and sizes), the k and tau grids, mask_km, device,
dtype, seed, n_boot and the Phase 1 probe source (`cache_settings` in `geo_metrics.json`). The
`geo_medoid` vs Phase 1 check runs only with `--mask-km 0`.

Step 1 dev analysis (`scripts/analysis/phase3_dev_summary.py`, dev only, no test km or GeoScore):
`results/v3/phase3_dev_summary.csv` (every predictor per backbone x ordering x N, both dev splits),
`scaling_geo.png` and `gap_geo.png`, and `phase3_dev_leakage.csv`, an exploratory comparison of
the masked `geo_medoid` (1 km mask, CPU fp32) with Phase 1's unmasked kNN coordinates (GPU fp16).
The difference mixes the mask with the device/dtype change and is not a clean leakage estimate.
`geo_mean` selected k=1 in every cell on dev_seen, i.e. it degenerated to `geo_top1`, so the
hybrid's default predictor is `medoid` (`--hybrid-predictor`, chosen after seeing this dev result;
`mean` stays selectable).

Step 2, geocells (`METHODS=cell` in a GPU session with internet on, `METHODS=hybrid` in a CPU
session). `geocells.py` builds one quadtree on (lat, lon) from the coordinates of the full
training pool (no labels, no images): a box splits into 4 while it holds more than
`max_per_cell` points and is larger than 0.01 degrees; `max_per_cell` is set by bisection to hit
the target count; a cell's centroid is the normalised 3D mean of its pool points (or its pool
point nearest that mean when the mean leaves the box). The partition is saved as JSON and its
sha256 enters the cache key. One partition serves every N (the same classes along the scaling
curve), so it uses the pool's geographic density beyond a subset. `--methods cell` trains the
Phase 1 probe (lr grid 1e-3..1e-1, 60 epochs, patience 8) over cell ids, random or zero-shot
initialised (each cell: the text embedding of its majority country in the subset); the epoch is
chosen by cell top-1 and the lr by GeoScore, both on dev_seen. Per cell and split (dev_seen,
dev_unseen) it saves the top-5 cell ids and their probabilities (`<split>_top_cells.csv`).
`--methods hybrid` reads only those files (no probe is trained, no GPU is needed; it refuses if a
cell probe is missing): it keeps the masked global kNN top-1000 inside the probe's top-m cells (m
in {1, 3, 5}, chosen on dev_seen) and applies `--hybrid-predictor`, else the top cell's centroid.
The hybrid's cache key includes its probe source (the cell's key and the hash of its top-cells
files), so a changed probe reruns exactly its hybrid cells. The cell probe's own key leaves out
mask_km, device and dtype (it has no kNN; the device used is still recorded in its `row.json`), so
probes trained with `--device cuda` are found by the CPU session's `--methods hybrid --device ''`;
the kNN and hybrid keys keep all three. The cell probe trains on all of the subset's training
images, including those within the near-duplicate mask radius of dev queries: only the kNN parts
(Step 1 and the hybrid's neighbour search) are masked. The same images are masked for both
backbones, but kNN vs cell is not a like-for-like comparison. (Corrected 2026-11-15, pre-registration
review: an earlier version said the paired StreetCLIP vs L/14-336 gap is unaffected. That is not
guaranteed. The mask removes the same images, but it costs more to the backbone that retrieves
near-duplicates better.)

Decisions made before any Step 2 result (changes to the first Step 2 design):
- **Cell-count selection uses the random init only.** 1K/2K/5K cells for StreetCLIP and L/14-336
  at nat 500K; the count with the best average dev_seen GeoScore of the pair, recorded in
  `cell_count.json` and frozen for both inits, every backbone, ordering and N. (Before: the
  average over both inits.) Rationale: random init only fits the weekly GPU budget (selection
  with both inits ≈ 2.6 h vs ≈ 1.3 h). It stays neutral between the compared backbones (average
  of StreetCLIP and L/14-336); the zero-shot variant uses the frozen count chosen this way. The
  selection refuses to run unless 500000 is among `--sizes` (or `--cell-target` is given), so a
  `SIZES` subset cannot freeze a count chosen at a smaller N. Every later run loads the frozen
  partition file (`geocells/cells_<target>.json`, carried over by `SEED_FROM`) and aborts if
  its sha256 differs from the one in `cell_count.json`. Only without that file is it rebuilt,
  and the rebuild must hash the same. (Fix after the first hybrid session: that run rebuilt
  the partition on a CPU-only host and got another sha256 than the GPU sessions, with the same
  cell count and `max_per_cell`. The tree comes from comparisons and exact halving only, so the
  difference is in the float64 centroids, whose trig and sums follow numpy's CPU-dependent
  SIMD paths. The rebuild also overwrote the seeded file before the check; it no longer does.)
- **The hybrid runs in a CPU-only session.** Cell probes (GPU) and hybrid (CPU) are separate
  runs: the hybrid's kNN is the Step 1 setting (numpy fp32, `--device ''`), so Step 1 and the
  hybrid share one numerical pin. (Before: kNN on cuda in the same GPU session.)
- Not changed: `--hybrid-predictor` defaults to `medoid` (`geo_mean` degenerated to top-1 on
  dev, see above).

Decisions made after Step 2 finished, before the lr counts were computed (2026-11-15):
- **Probe lr-grid edge (cell probes).** Per init (random, zeroshot) and per edge of the grid
  {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}: if the lower edge 1e-3 is the selected lr in at least 8 of
  that init's 24 cells (3 backbones x nat/bal x 4 N), 3e-4 is added for that init; if the upper
  edge 1e-1 is selected in at least 8, 3e-1 is added. The added lr runs on all 24 cells of
  that init, incrementally (stored lr runs reused, choice made over the merged sweep, as in
  Phase 1); selection is unchanged (epoch by cell top-1, lr by GeoScore, both on dev_seen).
  Stopping rule (Phase 1's): this is the final extension. If an added edge lr is then
  selected, or an edge is selected in fewer than 8 cells, the grid stays frozen and those cells
  are flagged `grid_edge` and reported as a limitation.
- Honest note: the threshold was set after partially seeing the live Kaggle logs (a few
  zero-shot cells visibly picked 1e-3; B/16 nat 200K picked 1e-2 (this live reading was mistaken;
  the saved sweep shows 1e-3)). The full counts had not been
  computed when this rule was written. Counts and per-cell neighbour GeoScores:
  `scripts/analysis/phase3_cell_lr.py` -> `results/v3/phase3_cell_lr.csv`.
- Outcome: the zero-shot lower edge met the threshold and no other init x edge did, so
  `phase3_eval.EXTRA_LRS = {"zeroshot": (3e-4,)}`. An init's extra lrs enter its cell key only
  when it has any (`extra_lrs` in `cache_settings`), so the random cells keep their keys. A
  zero-shot cell whose row still has the base-grid key is resumed: its stored lr runs are reused,
  only 3e-4 is trained, and if 3e-4 does not beat the stored choice (a tie keeps it) its scores
  and top-cells files stand. Its hybrid reruns, because the probe key changed. Kaggle notebooks:
  `runs/kaggle_phase3_extend_cells.ipynb` (GPU, zero-shot only) and
  `runs/kaggle_phase3_hybrid.ipynb` (CPU; random now, zero-shot after the extension).
- **Hybrid m grid (2026-11-15, after all 48 hybrids finished).**
  - Honest note: this was decided after the counts had been seen. m = 5, the upper edge of
    {1, 3, 5}, was picked in 35/48 hybrids.
  - **The grid is not extended.** Each cell probe saved only its top-5 cells per query (`top_cells`
    keeps max(HYBRID_M) = 5), and its weights were not saved. So m > 5 cannot be computed from
    stored data. A retrain on the GPU was considered and declined.
  - m is frozen at {1, 3, 5}. The m = 5 picks are flagged `grid_edge_high` in
    `results/v3/phase3_cell_flags.csv` and reported as a limitation, with their margin vs m = 3.
  - The m → all-cells limit is known by construction, though not rerun: no cell filter leaves the
    first k of the same masked CPU fp32 search, which is Step 1's `geo_medoid` over the same k
    grid. In 30 of the 35 m = 5 picks the medoid scores below m = 5 on dev_seen. In 5 it scores
    above, so the dev_seen optimum over m lies above 5 there; all 5 are random init, bal, N ≤ 100K.
  - Cell-probe keys and files are unchanged (nothing reran).

Test-phase requirements for the cell probe and hybrid, decided before any test run (2026-11-15).
The probe weights of the dev runs were not saved, so a test evaluation cannot reuse them:
- **Retrain and save.** The test evaluation retrains each declared cell probe once, with its
  frozen dev_seen-selected lr and epoch, the same seed, and the stored settings (the cell's
  `cache_settings`). There is no lr or epoch search. The weights are saved next to the cell.
- **Reproduction check before any test scoring.** The retrained probe's dev_seen top-5 cells and
  dev_seen GeoScore are compared with the stored dev values (`dev_seen_top_cells.csv`, `row.json`).
  If any cell is outside the tolerance, the run stops before test is read. The tolerance allows
  for GPU nondeterminism. Proposed values, to be fixed in the Phase 3 pre-registration:
  - the top-1 cell is identical for ≥ 99% of dev_seen queries;
  - the top-5 set is identical for ≥ 95%;
  - |dev_seen GeoScore difference| ≤ 2.
  The hybrid is then recomputed from the retrained probe's top cells with its frozen m and k.
- **Cost estimate** (scaled from the dev cells' own timings: one lr trained to the selected
  epoch, plus a dev_seen evaluation per epoch):
  - GPU: ≈ 2.2 h for all 48 cell probes, plus loading, ≈ 3 h.
  - CPU: the hybrids' kNN on test ≈ 2–3.5 h, from the 3.5 h the 48 dev hybrids took over two
    sessions. (Wrong: it ignored that test has 210,122 queries against 39,906 on the two dev
    splits. Corrected in the Phase 3 pre-registration below.)
  - A subset declared in the pre-registration costs proportionally less.

Final dev merge (`scripts/analysis/phase3_merge.py`): the final cell probes (random from session
C, zero-shot from the 3e-4 extension) and the 48 hybrids are copied into
`runs/phase3_final_dev/phase3_eval`. Each cell's key is recomputed from its stored settings, and
each hybrid's key from the merged probe's source, i.e. it was built from the final probe. The
stored settings are compared with the current code, the geocell sha256 is checked everywhere,
and per backbone the data and embedding hashes must match the Step 1 kNN cells (checked for
consistency only: the data and embeddings are not local). Any mismatch is listed and the script
exits 1.

Step 2 dev analysis (`scripts/analysis/phase3_cell_summary.py --run <merged dir>`, dev only, reads
every cell's `row.json` and `*_geo_hits.csv`), over five predictors (geo_medoid, cell and hybrid
x random / zero-shot):
- `results/v3/phase3_dev_predictors.csv`: GeoScore and median km;
- `phase3_cell_backbone_gap.csv`: StreetCLIP - L/14-336 and StreetCLIP - B/16, GeoScore and
  median km;
- `phase3_backbone_shrink.csv`: 50K -> 500K;
- `phase3_cell_init_gap.csv`: zero-shot - random;
- `phase3_cell_best.csv`: the best predictor by dev_seen GeoScore vs the runner-up and vs
  geo_medoid. Masked kNN vs unmasked probe, so not like for like;
- `phase3_cell_flags.csv`: lr grid_edge or epoch-cap cell probes, and hybrid m/k at a grid edge,
  with the margin vs each grid neighbour. m=1 and k=1 are natural lower bounds;
- `phase3_cell_scaling.png`, `phase3_cell_gap.png` and `phase3_cell_gap_b16.png`.

Intervals are paired cluster bootstraps with phase3_eval's n_boot and seed, and need `--data`
(dev_seen.csv and dev.csv). The numbers are in `results/v3/PHASE3_DEV_SUMMARY.md`.

Sessions (`kaggle_phase3.sh`; env: `METHODS`, `BACKBONES`, `ORDERINGS`, `SIZES`, `PROBE_INITS`,
`DEADLINE_MIN`, `SEED_FROM`). At `DEADLINE_MIN` a STOP marker lets the cell in hand finish, then
the run prints RESUME NEEDED and exits 3 (hard kill GRACE_MIN, default 45, later; that cell is
lost). Finished cells are cached; `SEED_FROM=<earlier output dir>` copies it into the empty
output of a fresh session (refused if it is not a phase3_eval output, the output dir itself, or
the output is not empty). A cell probe of an init that is done is never retrained.

Exploratory Phase 1 check in the same session: with the test embedding run (`emb_test`) and the
Phase 1 dev output attached, `kaggle_phase3.sh` ends with `scripts/analysis/phase1_posthoc.py`,
which recomputes the kNN cells whose dev_seen-matched k differs between the two backbones (k shared
by the pair's average dev_seen top-1) on the CPU in fp32. It writes only predicted countries and
top-1 hits (`phase1_matched_k_cpu.csv.gz`) and the agreement with Phase 1's saved GPU fp16 dev
predictions; locally `--run runs/phase1_final ... --cpu-preds <that file>` adds the matched test
gap and the test prediction agreement.

## Phase 3 pre-registration (coordinates on the official test split)

Written 2026-11-15, before any test distance or GeoScore of Phase 3 was read. File:
`configs/phase3_final.json`, written by `python -m geoguessr_ai.phase3_final prereg --dev-run
runs/phase3_final_dev/phase3_eval --knn-run runs/phase3_dev/phase3_eval --dev-results
results/v3/phase3_dev_predictors.csv`. Its sha256 is
`6aef5a15fed9662f8e586be417f48d05e7871561edf096afbb4db26c8228d66c`.
`dev_results_sha256` in it is the hash of the committed `results/v3/phase3_dev_predictors.csv`.
The test code (`src/geoguessr_ai/phase3_final.py`) is committed with it, so the statistic is
fixed by the same commit. The dev evidence is `results/v3/PHASE3_DEV_SUMMARY.md`.

**What is evaluated.** StreetCLIP and CLIP L/14-336 (B/16 is not scored on test), nat and bal,
N = 50K, 100K, 200K, 500K, five predictors: 80 cells. Each cell uses the values chosen on
dev_seen and stored in the file (the medoid's k; the cell probe's lr and epoch; the hybrid's m
and k), the frozen 5000-cell partition (sha256 `9882eb2a…`), seed 42, the 1 km mask and the CPU
fp32 kNN of the dev runs. Nothing is searched on test.

**Primary predictor and metric** (chosen from dev only). Predictor: the zero-shot hybrid
(`hybrid_zeroshot`). By dev_seen GeoScore a hybrid is the best of the five predictors in every
backbone x ordering x N cell, and the zero-shot hybrid in most of them. Metric: mean GeoScore.
It is the quantity every selection maximised (k, m, lr, cell count); median km was reported but
never selected on. "Gap" below is StreetCLIP minus L/14-336 on the same test images.

**Hypotheses (5, each one-sided).**
- H1: at bal 50K, StreetCLIP scores higher than L/14-336 with the primary predictor.
- H2: the same at nat 50K. On dev this gap is small; it is the least certain of the five.
- H3: at bal, the gap is smaller at 500K than at 50K (primary predictor).
- H4: at nat 50K, the gap of the cell probe is larger with a random init than with a zero-shot
  init. In words: the zero-shot init helps L/14-336 more than StreetCLIP. Why is not tested: a
  text-initialised head giving the base model what StreetCLIP already has, and a plain ceiling
  effect, both predict this sign.
- H5: the same at bal 50K.

Exploratory, not in the family (E1): at nat 500K, the primary predictor minus the kNN medoid,
averaged over the two backbones. It was H6 in the first draft and was moved out before the file
was committed: the kNN parts are under the 1 km mask and the probe inside the hybrid is trained
unmasked, so the comparison is not like for like. It is still reported
(`exploratory_contrasts.csv`): estimate and uncorrected creator interval, no p-value, no verdict.

How the five were chosen: by direction and size from roughly a hundred dev contrasts (5
predictors x 2 orderings x 4 N x 2 dev splits, plus shrink and init contrasts). They are not a
random sample of what could have been tested. H1 is close to certain on dev and repeats what
Phase 1 showed for country top-1. H3 is stated for bal only because nat went the other way on dev.
H4 and H5 use the cell probe; the same contrast for the hybrid is near zero on dev_unseen at nat
50K.

Not a hypothesis: whether the gap shrinks at nat. On dev it does not shrink with a zero-shot
init, so no direction is claimed; the nat shrink is reported with its interval as exploratory.

**Test statistic and decision rule.** Each hypothesis is a fixed weighted sum of per-image test
GeoScores over the cells listed under `terms` in the file; the statistic is its mean over the
test images. Resampling: 10,000 paired cluster-bootstrap resamples, seed 42, clusters =
**creators** (Phase 1 resampled sequences; its post-hoc review showed test errors are correlated
within creators and the sequence intervals too narrow). One resample applies to every cell of a
hypothesis. p = (1 + number of resampled statistics <= 0) / 10,001, so p is never below 1/10,001.
The five p-values are Holm-Bonferroni adjusted; a hypothesis is supported when its adjusted p
< 0.05, otherwise not supported. The sequence-clustered interval and p are printed next to each
(`seq_*`) and decide nothing. Every test id must be embedded for both backbones, or the run
stops; the largest creator's share of the test images is printed with each hypothesis.

The analysis runs once, when all 80 cells are scored. If a cell cannot be scored after
`test_read_started.json` exists, its hypotheses count with p = 1 (`primary --missing-as-p1`). A
later fix is a published amendment, reported next to that result and not instead of it.

One run: the result is the first run. Its identity (pre-registration sha256, code commit with no
modified tracked file, sha256 of the probes' `repro.json`, which holds each head's sha256) is
written into `dev_check.json`, `test_read_started.json` and every `test_row.json`. `test`
refuses heads other than the recorded ones, and `primary` refuses cells of different runs.

**Reproduction check (before test is read).** The dev probes' weights were not saved. Each probe
is retrained once: frozen lr, exactly the frozen number of epochs, same seed, last head kept
(this is the head the dev run restored). On dev_seen, against the stored dev files (their sha256
is in the file):
- the top-1 cell is the same for >= 99% of the queries;
- the set of top-5 cells is the same for >= 95%;
- |GeoScore difference| <= 2 for the cell probe and for the hybrid (frozen m, k, recomputed from
  the retrained probe); <= 0.5 for the medoid (CPU fp32, should be near exact).

It runs twice: in the GPU session on the trained logits, and in every CPU session on the saved
heads before the test split is loaded. A CPU session checks all 80 cells of both backbones, also
when it scores only one backbone. If any of the 80 cells is outside a tolerance, the run stops and
test is not read. No cell is dropped and no hypothesis is evaluated on fewer cells. A tolerance
can only be changed by a dated amendment to the file, committed before any test distance is read,
that publishes every failing number. The tolerances were set without data on how much a T4 retrain
varies; a failure is therefore possible for reasons that have nothing to do with the result.

**Reported whatever the outcome.**
- All five hypotheses: estimate, creator interval, raw and Holm p, verdict, and the sequence
  columns (`primary.csv`).
- E1, the hybrid minus the medoid at nat 500K (`exploratory_contrasts.csv`), labelled
  exploratory, with the mask caveat in its `claim` column.
- Every declared cell's test GeoScore, median km and accuracy within 1/25/200/750/2500 km, next
  to its dev_seen and dev_unseen GeoScore and its `grid_edge` flags (`results.csv`).
- The StreetCLIP minus L/14-336 gap of every predictor x ordering x N (`exploratory_gaps.csv`),
  labelled exploratory.
- The reproduction numbers of every cell (`repro.json`, `dev_check.json`), pass or fail.
- The share of test queries with a masked training image and the hybrid's fallback share.

**Known limitations.**
- lr grid edge: 13 of the 32 declared cell probes chose the lowest lr of their grid (`lr_low`).
  The grid is final; the neighbour margins are in `results/v3/phase3_cell_flags.csv`.
- m grid edge: 23 of the 32 declared hybrids chose m = 5, the top of {1, 3, 5} (`m_high`). m > 5
  could not be computed from stored data and was not run.
- Epoch cap: 2 declared probes chose epoch 60, the cap; 14 never stopped early.
- Rule timing: the lr-edge threshold (8/24) was written after some live Kaggle logs had been
  seen, before the counts were computed. The m grid was frozen after its counts were seen.
- The hybrid's medoid predictor was chosen after dev showed the weighted mean collapsing to k = 1.
  The cell count (5000) was chosen on dev_seen, random init only, nat 500K.
- dev_seen optimism: every choice, including the primary predictor, was made on dev_seen, and
  the dev_seen numbers in the file are from the same split. dev_seen shares creators with train
  and has no distance buffer. Test may rank the predictors differently.
- Masked kNN vs unmasked probe: the kNN parts mask training images within 1 km of the query's
  true location and the probe does not. E1 compares the two kinds of predictor, so the hybrid is
  favoured wherever the mask removes something; this is why E1 is exploratory. The mask also need not cost both backbones the
  same (H1 to H3). The mask is applied on test as on dev; the masked share on test is reported
  so the reader can see how much it matters there. An unmasked medoid is not computed.
- The p-value is a percentile-bootstrap p, not a test under a constructed null. With a few very
  large creators it is unstable; the largest creator's share is reported for that reason.
- Each backbone has its own k, m, lr and epoch (not matched between the two). A gap includes
  any difference in how well the grids suit each backbone.
- Phase 1 already scored country top-1 on the same test images, and its results are known:
  StreetCLIP ahead at 50K, test harder than dev. H1 and H2 are not blind to that. Phase 1 also
  wrote exploratory kNN distance metrics on test (`results/v2/test_results.csv`); they were not
  opened for Phase 3.
- The two test sessions (one per backbone) print per-cell test scores before the paired analysis
  exists. Nothing may be changed after `test_read_started.json` is written.
- The retrain is not bit-identical to the dev probes (GPU). The hypotheses are about the
  retrained probes, which the tolerance ties to the dev ones.
- Exactly tied cells (head rows that stay identical, e.g. empty cells of one country after a
  zero-shot init) can be ordered differently by the GPU and the CPU. This changes a prediction
  when the tie is at rank 1 (the cells have different centroids) or across rank 5 (the hybrid's
  filter). The dev_seen check bounds how often; ties are not counted separately.
- As in Phase 1, this is self-discipline, not proof: git timestamps and the provenance file are
  written by us. The provenance shows only that the file was committed before the probes run
  (its `embedding_started` field, named for Phase 1, holds that run's start). The test embeddings
  are older than this pre-registration. Pushing the commit before the probes run gives GitHub's
  push time as the only outside record.

**What we will NOT conclude.**
- From "not supported": that there is no gap, no shrink or no init effect. It only means this
  test did not show it at the corrected level.
- Anything confirmatory about B/16, N = 100K or 200K, the random hybrid, median km, accuracy at a
  distance, countries or creators. These are exploratory, with uncorrected intervals.
- From a supported H4 or H5: anything about the hybrid, about N above 50K, or about why the init
  matters.
- From a supported H1: more than a confirmation, in km, of what Phase 1 showed for countries.
- That StreetCLIP's training causes a gap: the two backbones also differ in how well the shared
  grids fit them.
- That the hybrid is the best coordinate predictor, or better than kNN. No hypothesis tests it;
  E1 is exploratory, one N, one ordering, not like for like, and both methods were tuned on
  dev_seen.
- That the numbers are comparable with the OSV-5M paper beyond the shared metric: the training
  sets here are 50K-500K subsets and the models are frozen.
- Anything about fine-tuned models.

**Running it** (`runs/kaggle_phase3_test.ipynb`, pinned to the commit of the file):
1. GPU session (T4 as in the dev runs), test embeddings not attached: `phase3_final probes`.
   32 probes, ≈ 2 GPU h. Writes the provenance, the heads and `repro.json`; exit 1 if the check
   fails.
2. CPU session per backbone, with the probes output and `emb_test` attached: `phase3_final test`.
   The dev_seen check of both backbones first, then test. Rough estimate 6–7 CPU h per backbone
   plus the check (unmeasured: scaled from the dev sessions to about 5 times the queries); the
   first subset's printed time gives the real rate. A deadline stops cleanly and the next
   session resumes.
3. `phase3_final primary` on the two outputs merged: `primary.csv`.
