# V4 feasibility: LoRA fine-tuning of StreetCLIP vs CLIP L/14-336 (design only)

Status: a feasibility study and pilot design. No V4 training run has happened, no V4 number
exists, and the OSV-5M test split stays unopened for V4 until a V4 pre-registration is
committed. Every rate below is either **measured** (with the file it comes from) or **estimated**
(with how it was derived). Estimates are what the pilot (section 4) replaces. Section 7 lists the
independent review of a first draft and what changed because of it.

**Question.** Frozen encoders (V3) gave StreetCLIP a GeoScore lead over its base CLIP
L/14-336. That lead depends on the predictor and the init, and a text-initialised head narrows it
(H4/H5). Does fine-tuning the vision encoder (LoRA) **close, keep or reverse** the lead, and how
does that depend on N (asked: 50K vs 500K)? The metric stays GeoScore, with the same frozen
5000-geocell partition (sha256 `9882eb2a…`) and the same dev splits.

**Short answer.**
- **50K vs 500K with a real lr search at both N does not fit two Kaggle weeks** (about 60 quota h)
  at any plausible rate: 92–154 h including a reserve for one lost session (section 3).
- With a declared shortcut (the lr chosen at 50K and frozen for 500K), it fits only if the pilot
  measures **≥ 19 img/s** per T4 on the slower backbone. My central estimate is 12.
- At the central rate, the honest options are a **short-horizon lr search** (B-S2, needs
  ≥ 15.6 img/s) or **a smaller large-N arm**: N = 200K, seen once, with a real search at both N.
  The latter fits from 11.3 img/s (57 h at 12).
- The pilot (2.5 h) decides which one by a rule fixed in advance.

**Decisions (2026-11-18, before any V4 run or dev result).**
1. **N:** 50K vs 200K (B-full-200) is accepted as the answer if the pilot-rate rule (section 3)
   selects it. The rule itself is unchanged.
2. **Ordering: bal.** bal needs a new archive. It is built in a CPU session
   (`runs/kaggle_v4_bal_archive.ipynb`, section 1).
3. **Equivalence margin for "keep": 40 GeoScore**, fixed with its reasoning in section 5.
4. **Head lr = 10× the adapter lr**, fixed.
5. **Quota:** a T4×2 session costs 1 quota h per wall-clock hour. This is observed, not assumed
   (section 2).

## 1. Data

**What already exists** (checked read-only on 2026-11-18 with the HF API):

| Source | Content | Size | Per image |
|---|---|---|---|
| `osv5m/osv5m` `images/train/*.zip` | 98 zips, the full train split | 246.4 GB (2.25–2.53 GB per zip) | ≈ 50 KB |
| `osv5m/osv5m` `images/test/*.zip` | 5 zips (V4 must not read them before its pre-registration) | 9.4 GB | |
| `Daggerabuse/osv5m-ft-archive` (private, written by Phase 1's `ft_archive.py`) | nat_rank < 200K + dev_unseen (20,000) + dev_seen (19,906) = **239,906 images**, raw bytes (no re-encode), 13 ZIP_STORED parts, sha256 per part and per image | **12.44 GB** | 51.8 KB |

- The images are short side 512, mostly 682×512 or 910×512 (`data/sample`).
- The existing archive is packed in shard order (`ft_archive.py` packs chunks per source zip). So
  nat_50K, dev_seen and dev_unseen are **spread over all 13 parts**: any session that needs them
  stages the full 12.44 GB.
- nat_200K is complete there. The ordering is bal (decision 2), so V4 does not use this archive
  for training.
- Any **bal** subset needs a new archive. bal_200K + dev + dev_seen is about 12.4 GB; bal_500K
  (which contains bal_50K and bal_200K) is about 26 GB plus dev.

**The bal archive for V4 (CPU session, no GPU quota): `runs/kaggle_v4_bal_archive.ipynb`**
(prepared, not run).
- It runs the Phase 1 path: `zipq` streams the 98 train zips one at a time, and `ft_archive
  consume` writes ~1 GB parts byte for byte, sha256-checked and uploaded part by part. The new
  `--rank-col bal_rank --rank-max BAL_MAX` selects bal_rank < BAL_MAX + dev_unseen + dev_seen.
- **Repo:** the new private repo `Daggerabuse/osv5m-ft-archive-bal`, behind its own guard (mode
  `bal`). bal_rank is refused for any other repo, and the bal repo is refused for nat_rank. The V3
  archive is never written.
- **Manifests:** the notebook reads `v2/train.csv`, `dev.csv` and `dev_seen.csv` from the
  attached Phase 1 output, never `test.csv`, and attaches no image zips.
  - It asserts that `v2/config.json` has V3's pre-registered sha256
    (`configs/phase3_final.json`).
  - It asserts that the ids hashes of bal_50K and bal_BAL_MAX equal the pre-registered ones. So
    V4 trains on exactly V3's subsets.
- **Checks at the end:** manifest = wanted ids exactly (no duplicate, missing or extra id), every
  part uploaded, the remote manifest byte-equal, the repo private.
- **BAL_MAX** = 200,000 for B-full-200. If the pilot's rule picks B-S1 or B-S2, it is 500,000
  (about 28 GB).
- **Run it after the pilot**, so BAL_MAX is known.
- **Packing order:** the parts are packed in shard order, so a 50K session stages the whole
  archive. At 200K that is ≈ 12.4 GB, a few minutes per session. Packing in rank order, with dev
  in its own parts, would only pay off at 500K. It is not done.
- Measured download speeds from Kaggle:
  - **72–200 MB/s** per OSV-5M zip (`runs/pilot/zipq_stats.json`, 2.52 GB in 12.6–35.1 s);
  - one stream at ≈ 400 MB/s in Phase 0b (`zipq.py` docstring).
- 246 GB therefore takes 0.2–1 h to download, plus copying, packing and the HF upload.
  Estimate: a 2–4 h CPU session.
- Pilot cell 2 times one OSV-5M zip download again.
- HF private storage: 12.4 (V3 archive) + 12.4 (bal at 200K; 28 at 500K) GB, inside the 100 GB
  free private limit
  ([HF storage limits](https://huggingface.co/docs/hub/storage-limits)).
- Download volume over V4 is about 100–150 GB from one free private repo. HF rate limiting is
  possible and unmeasured. Every download is sha256-checked and retried (as in `ft_archive`).

**Getting images into a GPU session.**

| | How | Pros | Cons |
|---|---|---|---|
| (a) **HF parts → `/kaggle/temp` at session start** (recommended) | `hf_hub_download` of the needed parts, sha256 checked (pilot cell 2) | proven path (Phase 1, V3); byte-identical to OSV-5M; 28 GB at 72–400 MB/s = 1–7 min per session | repeated every session; relies on `/kaggle/temp` space (below) |
| (b) Kaggle dataset as input | parts uploaded as a private Kaggle dataset, mounted read-only under `/kaggle/input` (does not count against the 20 GB output limit) | no download per session | a 28 GB dataset cannot come from one notebook's ≤ 20 GB output (two datasets needed); `kaggle datasets version` from a notebook failed in the Phase 1 pilot (`runs/pilot.log`, exit 1), which is why the archive went to HF; private dataset quota reported as 200 GB, measured uncompressed ([Kaggle Book, Packt](https://subscription.packtpub.com/book/data/9781801817479/3/ch03lvl1sec07/setting-up-a-dataset); not confirmed in Kaggle's own docs) |
| (c) Pre-resized 336 px re-encoded shards | resize + JPEG re-encode once | ≈ 2× smaller, faster decode | **changes the pixels** vs the V3 frozen embeddings (made from the original bytes) and adds JPEG loss: a confound in every FT vs frozen comparison. Not recommended |
| (d) Streaming from the 98 OSV-5M zips during training | | | sparse range reads were network-bound in Phase 0 (≈ 1.2 s per image, `runs/phase0/emb/*/throughput.jsonl`). Ruled out |

**Disk and RAM.**
- `/kaggle/working`: 20 GB that is kept as output. Secondary sources only; Kaggle's docs pages are
  JS-rendered and could not be read
  ([summary](https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/93babea0b1c303124af8c062c459fbceea4c3208/kaggle_20251121.md)).
  V4 keeps only logs, adapters and heads there. Checkpoints are also pushed to HF during a run,
  because a hard kill can lose `/kaggle/working`.
- `/kaggle/temp` showed a shared overlay with 967 GB free in our sessions (`runs/pilot.log`,
  `df -h`). The enforced limit is not documented, and Phase 1 never kept more than about 13 GB
  there. The pilot prints `df -h` again. A 28 GB staging is untested; if it fails, option (b) is
  the fallback.
- RAM: 31 GB (`free -g`). A 500K session's 28 GB nearly fills the page cache.
- The random-read load is small: 2 runs × ≤ 17 img/s × 52 KB ≈ 2 MB/s. So the cache is not the
  bottleneck.

**Decode budget (CPU): the tightest resource.** A session has 4 cores (`nproc`).
- Measured decode plus eval preprocessing at 336 px, per image:
  - **18.3 ms** uncontended (`runs/phase0/emb/clip_l14_336/throughput.jsonl`);
  - **37.5–49.1 ms** with three encoders sharing the CPU (`runs/pilot/*/throughput.jsonl`:
    StreetCLIP 37.5, L/14 47.2, B/16 49.1).
- Training augmentation (`train.train_view`) adds a box-resize, three enhancers, an occasional
  blur and erasing. **Estimate: 45–60 ms/image under contention.**
- Load: 2 runs × 12–17 img/s × 50 ms ≈ 1.2–1.7 cores for decoding, plus about 1 core per training
  process. A partner in an eval phase (33 img/s × 45 ms ≈ 1.5 cores) adds more.
- **That is about 3.5–4 cores of 4: the loader can become the bottleneck.**
- The pilot measures it: 2 vs 3 workers, a train-while-the-other-evaluates phase, `vmstat`
  throughout.
- Fallbacks, in this order and declared before the pilot:
  1. 3 workers per run;
  2. evals scheduled so both GPUs evaluate at the same time (no mixed phases);
  3. dropping blur and erasing for both backbones, which changes the recipe and must be decided
     before any V4 dev run.

  PIL `draft` decoding is not usable (it scales by 1/2, and 256 < 336). GPU decode (nvjpeg)
  changes the pixels vs PIL, so it would need a frozen re-baseline.

## 2. Compute

**Measured anchors, T4** (`runs/phase0/bench.jsonl`; synthetic batches, torch 2.11, transformers 5.20):

| | L/14-336 | StreetCLIP |
|---|---|---|
| inference fp16, batch 64 | 43.6 img/s, 2.0 GB | 36.2 img/s |
| **full** fine-tuning, batch 16, AMP, grad checkpointing, AdamW | 8.3 img/s, 6.86 GB peak | 7.9 img/s |
| real pipeline inference (decode included, GPU-bound) | 35.3 img/s (`runs/phase0/emb`) | 32.7 img/s (`runs/pilot`, cuda:1) |

- The budget uses **33 img/s per T4 for any evaluation or re-embedding**.
- **StreetCLIP and L/14-336 have the same architecture, yet StreetCLIP measured 10–20% slower
  every time.** Backbone and GPU, run order and temperature are confounded in these numbers. The
  pilot swaps the GPUs for one config and logs SM clock, temperature and power every 10 s.
- The budget uses the **slower backbone's** rate, because a pair of runs lasts as long as its
  slower half.

**LoRA cost model, calibrated on the measured full fine-tuning step.**
- In theory a full fine-tuning step with checkpointing costs about 4 forward passes. The measured
  step costs 43.6 / 8.3 = **5.25**, i.e. a **1.3× overhead**. It comes from fp32 LayerNorm and
  residuals under autocast, memory-bound elementwise backward, and micro-batch 16 vs 64.
- The same overhead is applied to the LoRA counts. LoRA skips the weight gradients of the 304M
  frozen parameters, but still back-propagates the input gradients. The real rate is 0.8× the
  synthetic one.

| LoRA config (r=16, q,k,v,o) | forward units | × 1.3 | synthetic img/s | **real img/s** | name |
|---|---|---|---|---|---|
| all 24 blocks, checkpointing | ≈ 3 | 3.9 | 11 | **≈ 9** | P |
| all 24 blocks, no checkpointing | ≈ 2.2 | 2.9 | 15 | **≈ 12** | C |
| last 12 blocks, no checkpointing | ≈ 1.6 | 2.1 | 21 | **≈ 17** | O |

- In the last-12 config the lower 12 blocks build no graph, since their input needs no
  gradient.
- A rate below P is possible. The decision rule below covers every rate down to 6.9 img/s, and
  below that it stops.

**Memory (estimate, per T4, 15 GB usable).**
- Fixed costs:
  - the frozen base in fp16 is 0.61 GB;
  - LoRA r=16 on q,k,v,o of 24 blocks is 3.1M parameters, and the 5000×768 head is 3.85M;
  - fp32 weights, gradients and two AdamW moments for those 7M parameters take about 0.11 GB.
- Activations, fp16 with SDPA (the transformers default here, checked):
  - per block and image, about 577 tokens × (10·1024 + 3·4096) × 2 B ≈ 26 MB (quick_gelu saves
    3 tensors of 4096 per token);
  - **× 1.3–1.5** for the fp32 LayerNorm outputs and residuals under autocast;
  - this gives ≈ 0.8–0.95 GB per image over 24 blocks.
- Without checkpointing, micro-batch 16 is **borderline** (13–15 GB) and 12 is safe. With
  checkpointing it is about 40 MB per image, so batch 32+ fits. The last 12 blocks need about
  half.
- The pilot measures peak memory. An OOM is recorded, not raised, in the synthetic and the
  real-data step.
- An effective batch of 64 comes from gradient accumulation. The micro-batch and checkpointing
  change speed and memory, not the maths (up to fp16 numerics). The pilot picks them, which is a
  throughput-only choice.
- If C needs micro-batch ≤ 12, its rate drops toward P.

**Fixed a priori (not chosen by the pilot).**
- LoRA r=16, alpha=16, on q,k,v,out_proj of **all 24 blocks**, for both backbones. Same
  architecture, so the trainable parameter count is identical.
- Last-12 is faster, but it is a capacity choice, not a speed setting. It is only listed as an
  alternative.

**Layout: two independent runs per session, not DDP.**
- One run per GPU, the two backbones side by side at the same lr. Example: StreetCLIP lr_i on
  `cuda:0` and L/14-336 lr_i on `cuda:1`.
- There is no gradient sync and no change to batch semantics, and the pattern is already used
  (Phase 1).
- Both halves of a pair share the session, hardware and CPU contention.
- The faster half then idles. The budget charges the pair at the slower half's wall time.

**Quota accounting.**
- Kaggle gives 30 GPU h per week, sessions of at most 12 h, quota shared between P100 and T4×2
  (secondary sources: [summary](https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/93babea0b1c303124af8c062c459fbceea4c3208/kaggle_20251121.md),
  [Notre Dame course page](https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html): "an
  interactive session counts towards your 30 hours, even when no process is running").
- **A T4×2 session costs 1 quota hour per wall-clock hour (observed).** In the week of
  2026-11-18, about 18 h of T4×2 sessions left about 11–12 of the 30 h, i.e. about 18–19 quota h
  were used. At 2× they would have used about 36 h, more than the weekly quota. I found no official
  statement either way.
- The budget therefore counts session wall-clock hours. The pilot's `quota_times.txt` (batch
  only, asserted) confirms it once more.
- Rules: batch sessions only; a deadline stop before 12 h; checkpoints every 30 min, also pushed
  to HF.

**Per-run cost: what one pair job contains** (`scripts/v4_budget.py`; every number is an input to
it):
- **Training:** S images seen per run at rate r.
- **Evals:**
  - the final checkpoint on **dev_unseen** (20,000; the selection split, section 5) and on
    dev_seen (19,906; diagnostic);
  - two intermediate checkpoints on a 5K dev_unseen subset (diagnostic only).

  That is 0.42 h per run at 33 img/s. The first draft had five full dev_seen evals, 1.0 h.
- **Sessions:** a job uses whole sessions of ≤ 11 h. A longer job resumes in the next session and
  pays the setup again: 0.25 h for clone, pip, two model downloads of about 1.7 GB each, data
  staging and preflight.
- **Reserve:** one lost session (min(job, 11 h) + 0.25 h). It replaces the first draft's flat
  10%, which was less than one failed session.
- **Fixed costs:**
  - the pilot, 2.5 h;
  - a zero-init equality check, once per backbone on 2K dev_unseen images (minutes);
  - the augmented-frozen control at 50K (section 3);
  - the 50K hybrid re-embedding: 50K train images per model. The dev embeddings are stored by the
    final eval, so there is no second dev pass;
  - **the later test pass:** 4 models × 210,122 images at 33 img/s ≈ 7.1 wall h (3.5 quota h).
    The test zips (9.4 GB) are staged in that session only.
- **Not needed** (unlike V3): a retrain for the test run. The adapters and heads are small and are
  saved with sha256, so the test run loads exactly the dev-selected weights.
- **Resume state**, all of it in the checkpoint:
  - adapters, head, AdamW, GradScaler;
  - scheduler step;
  - the sampler's permutation seed and position: an index sampler skips without decoding;
  - torch/numpy RNG.

  Estimate: about 85 MB. The pilot times a round trip.

## 3. Design options and an honest budget

Common to all options:
- **Primary predictor:** the 5000-cell probe (top-1 cell centroid). It needs only forward passes,
  on dev and on test.
- **Head init (LP-FT):** each run starts from **its own backbone's V3 `cell_zeroshot` head for the
  same ordering and N** (`runs/phase3_test_probes/phase3_probes/<bb>/<ord>_<N>_geo_cell_zeroshot/head.npy`,
  sha256 in V3's `repro.json`; V3 saved heads for N = 50K, 100K, 200K and 500K).
- **Frozen comparators:** V3's `cell_zeroshot` (the LP-FT starting point) and `cell_random`, for
  the same backbone, ordering and N. They come from the committed V3 dev tables.
- **Schedule:** S images seen per run, the same for both N, so compute is matched and the N
  effect is data, not steps.
  - AdamW, warmup 5% then cosine to zero, effective batch 64;
  - `train_view` augmentation (no flip, no hue), seed 42;
  - the same sampler seed and per-image augmentation RNG (`train_view` is seeded by
    (seed, epoch, index)), so both backbones see identical pixels.
- **lr grid:** {1e-4, 3e-4, 1e-3} for the adapters. The head gets **10× the adapter lr**, as V1's
  head/backbone ratio, declared before any run.
  - Reason: V3's heads were tuned with their own lr, per backbone (grid 1e-3…1e-1, plus 3e-4 for
    the zero-shot init). With a shared lr of ≤ 1e-3 the head would be close to frozen, and its
    starting fit differs by backbone.
  - The initial train loss of each run is logged.
  - One edge extension is reserved (section 5).
- **Augmented-frozen control, in every option.** The frozen encoder is trained head-only with the
  same augmented S-image schedule, at 50K, for both backbones. It separates encoder adaptation
  from "augmentation + more head training". V3's heads were trained on non-augmented center
  crops, and that effect need not cancel in the gap. Cost: one forward-only pair.

**Budget** (`uv run python scripts/v4_budget.py [--seen S] [--rate r…]`). Quota h for P / C / O
(9 / 12 / 17 img/s per T4), **including the one-lost-session reserve**:

| Option | Jobs (pairs = both backbones) | S | P | C | O | break-even for 60 h |
|---|---|---|---|---|---|---|
| A: 50K only | 3 lr + edge + seed 2 + control, hybrid 50K, test (2 models) | 500K | 103.1 | 83.8 | 63.1 | 18.2 img/s |
| **B-full**: real search at both N | (3 lr + edge) × {50K, 500K} + control, hybrid, test (4 models) | 500K | 154.2 | 123.3 | 91.6 | 29.9 |
| **B-S1**: full-length search at 50K, its lr frozen for 500K | 3 lr + edge at 50K, 1 at 500K, control, hybrid, test | 500K | 105.1 | 85.8 | 65.1 | **19.0** |
| **B-S2**: short-horizon search | (3 lr + edge) at 50K for 100K images seen (a complete cosine over 100K), then 50K + 500K full + seed 2 at 50K, control, hybrid, test | 500K | 87.1 | 72.4 | 56.3 | **15.6** |
| C: B-S1 + seed 2 at 50K + 500K hybrid | | 500K | 125.9 | 102.8 | 78.4 | — |
| **B-full-200**: real search at both N, **N ∈ {50K, 200K}** | (3 lr + edge) × {50K, 200K}, ... | 200K | 71.1 | **57.3** | 45.0 | **11.3** |
| B-S1-250: 500K pool, 250K seen (half an epoch) | as B-S1 | 250K | 60.3 | 48.7 | 38.5 | 9.1 |

**Plainly:**
- **50K vs 500K with a real lr search at both N (B-full) needs 92–154 h. It does not fit two
  weeks at any rate.**
- Even with the lr frozen from 50K (B-S1), it needs ≥ 19 img/s, which is above my optimistic
  estimate for the chosen all-blocks config.
- A short-horizon search (B-S2) needs ≥ 15.6 img/s.
- At the central rate the 500K arm as asked does not fit. The options that do fit keep a real
  search but shrink the large N to 200K (seen once), or keep 500K but show it only half an epoch.

What each option can answer, and its risks:
- **A** answers the 50K question with a seed-variance estimate, not the N dependence. It costs
  about as much as B-S1 (the same number of runs), so it is not "cheaper", only less informative.
- **B-S1** answers the N question at matched compute with a full-length search, but only at
  50K.
  - Risk: the 50K optimum (10 epochs, more overfitting) may sit lower than the 500K optimum. The
    rule is the same for both backbones, but the shift may not be.
  - No seed estimate.
- **B-S2** answers the same question with a weaker search. The best lr for a short horizon tends
  to be higher than for the full schedule. V2 and V3 both showed grid choices moving the gaps.
- **B-full-200** gives a real search at both N, with every image seen exactly once at the large N
  and 4 epochs at 50K.
  - V3 has frozen baselines and saved heads at 200K, and nat_200K is already archived.
  - The cost: a 4× N range instead of 10×, and no direct link to V3's 500K result.
- **B-S1-250** keeps the 500K pool, but its run sees only 250K distinct images. "500K" then
  means pool size, not data seen. Not recommended.
- **C** is too big for two weeks at every rate except O.

**Recommendation: a decision rule, fixed now, applied to the pilot's measured rate.** r is the
slower backbone's `e2e_train_images_per_s` in pilot cell 4a (train only, both GPUs at once, real
images, all-blocks config):

| r (img/s per T4) | run |
|---|---|
| ≥ 19.0 | **B-S1** (50K vs 500K, full-length search at 50K) |
| 15.6 – 19.0 | **B-S2** (50K vs 500K, short-horizon search) |
| 11.3 – 15.6 | **B-full-200** (50K vs 200K, real search at both N) |
| 6.9 – 11.3 | **A** (50K only, with seed 2 and the control) |
| < 6.9 | stop; rethink (e.g. the last-12-blocks config, a third week) |

- My expectation (C ≈ 12) lands in **B-full-200**. 50K vs 200K is accepted as the answer
  (decision 1). The alternative, B-S2 at the cost of a third week (72 h at C), is not taken.
- Unused reserve hours (no lost session, no edge extension) buy a seed-2 replicate at 50K,
  declared now. The resulting seed estimate is therefore conditional on no edge binding.

**Weekly packing (B-full-200 at C).** Pairs take about 5 h (two per session). The order is a
strict chain:

1. Week 1, about 30 h: the pilot; the CPU staging session; then the 50K search (3 pairs) and the
   200K search (3 pairs).
2. Week 2, about 27 h: the edge extension (if any); the control; the 50K hybrid; the dev
   analysis; the V4 test pre-registration; the test pass.

The 30 h weekly cap will split a session somewhere. Budget per week, not just in total.

**Ordering: bal (decision 2).** One ordering only; both would double every cost.
- **The reason given for the decision:** "does fine-tuning close the lead" needs a lead. On V3
  test, StreetCLIP leads at bal (H1, +103 GeoScore, hybrid, 50K) and trails at nat (H2, −40).
  **So the choice was made knowing V3's test results.** This is declared, not hidden.
- **Dev agrees:** V3's frozen bal gap had the same sign on dev_seen and dev_unseen at 50K and 500K
  (`results/v3/phase3_cell_backbone_gap.csv`). On dev, the nat zero-shot gap did not shrink
  consistently (V3 pre-registration, "not a hypothesis").
- **Cost:** one CPU archive session (no GPU quota); the nat archive cannot be reused.

**Fair controls (all options).**
- Same LoRA config, schedule, images seen, checkpoints, lr grid, head-lr multiplier, edge rule
  and augmentation RNG for both backbones. Both backbones run side by side in the same sessions.
- **lr selection:** per backbone (as in V3), with matched lr (best pair average) as a sensitivity
  analysis. Where every grid run exists for both backbones, this costs nothing.
- The head init comes from each backbone's own text encoder and V3 tuning, by construction. This
  is the H4/H5 mechanism, and it stays part of "StreetCLIP's advantage".
- The augmented-frozen control (above) separates augmentation from adaptation.
- Preprocessing: evaluation uses V3's `embed.clip_center_crop` on the same original bytes, so a
  model with zero-initialised adapters computes the V3 frozen model, up to fp16/autocast numerics.
  The check runs once per backbone before the first run: with B = 0 and the V3 head, the top-1
  cell matches the V3 frozen model's on ≥ 99% of 2K dev_unseen images.
- Merged export: `LoRALinear.merged` sums in fp32. A merged model must give the same top-1 cells
  as the unmerged one before it is used.

## 4. Pilot (≤ 2.5 quota h, prepared, not run)

Notebook: `runs/kaggle_v4_pilot.ipynb`. GPU T4×2, internet on, secret `HF_TOKEN`, **no
inputs**, **batch only**.
- Code: `src/geoguessr_ai/lora.py` (in-house LoRA, no new dependency).
- New `bench.py` flags: `--lora-rank/--lora-alpha/--lora-targets/--lora-blocks`, `--head-classes`,
  `--archive`, `--archive-phases`, `--tag`.
- bench saves its results after every phase, so a timeout keeps the finished ones. It asserts that
  every trainable parameter received a gradient (`grads_ok`), which catches checkpointing that
  silently trains nothing.
- Tests: `tests/test_lora.py`.

1. **Preflight.**
   - Clone `v4` at the pinned `COMMIT`; assert no tracked file is modified; assert a batch session,
     2× T4, and **no input attached**, so no test data can be present.
   - Print versions, `nproc`, `free`, and `df -h` for /kaggle/working, /kaggle/temp and /tmp.
   - Check that the archive repo is private and its manifest has 239,906 unique ids in 13 parts.
   - Start `nvidia-smi` (clock, temperature, power, utilisation, memory) and `vmstat` logging
     every 10 s.
   - Write `quota_times.txt`.
2. **Staging speed.**
   - Download parts 000–002 (about 3 GB) and check their sha256.
   - Download one original OSV-5M train zip (2.5 GB, for the CPU staging estimate), then delete it.
3. **Bench sweep.**
   - StreetCLIP on `cuda:0` and L/14-336 on `cuda:1` **at the same time**.
   - Rank 16 or 8; all 24 blocks or the last 12; q,k,v,o with or without fc1/fc2; checkpointing on
     or off; micro-batch 16, 24 or 32. All with the 5000-class head and CE, recording img/s, peak
     memory and OOM.
   - Three real-image configs: 2 vs 3 workers, all blocks vs last 12. Each measures the loader
     alone, loader + train, loader + eval, and the checkpoint round trip, on a seeded sample over
     all staged parts.
   - One config repeated **with the GPUs swapped**.
4. **Sustained rates.**
   - (a) **Train only, both GPUs at once**, 8,000 images each, all-blocks config with
     checkpointing. **This gives r for the rule above.**
   - (b) One GPU trains while the other evaluates.
5. **Summary.** `bench_v4_summary.csv` (the last line per tag and device), the session end time,
   and a reminder to stop and note the quota.

What the pilot does not do:
- it reads no labels, coordinates, cells or scores, so nothing is selected;
- the archive parts it times mix nat_200K, dev and dev_seen images; only pixel throughput is
  measured.

What the pilot chooses: micro-batch and checkpointing, and whether the decode fallbacks are
needed. All are throughput-only choices.

Pilot limits:
- 8,000 images is about 11–15 min, not thermal steady state over 10 h. The `nvidia-smi` log shows
  whether clocks drop.
- The page cache is warm: the parts were just downloaded and hashed.
- The first real run's first hour is the true check. Its rate is logged, and the rule is not
  re-applied after it: the option is fixed by the pilot.

## 5. What a V4 pre-registration must fix, and dev-only tuning rules

**Before any V4 training run** (a dated pre-dev note, committed):
- The option (by the rule in section 3, from the pilot's r), ordering, N values, S, and the
  checkpoint schedule.
- The LoRA config (r, alpha, targets, blocks), optimizer, warmup, cosine, effective batch, weight
  decay per parameter group, head-lr multiplier, and augmentation (V1's `train_view`, no flip,
  no hue).
- The head init file and its sha256 per cell; the partition sha256.
- The lr grid, the edge rule, the tie tolerance and the stopping rule (below); the seed(s); what
  unused reserve hours buy.
- **The selection split: all of dev_unseen** (20,000 images, creator-disjoint from train).
  - Why: dev_seen shares creators with train with no distance buffer. Fine-tuning can learn
    creator-specific cues (camera, vehicle, mount), and does so more at 50K × several epochs than
    at a large N seen once. Selecting on dev_seen would bias the 50K lr choice and the frozen vs
    fine-tuned contrast.
  - The official test split is the reporting split, so dev_unseen does not need halving.
  - dev_seen is reported as a declared diagnostic: the dev_seen − dev_unseen difference of the
    fine-tuned vs the frozen model.
  - The V3 frozen comparators were selected on dev_seen. That asymmetry is a declared limitation.
- **Checkpoint rule:** the lr is selected on the **final** checkpoint (fixed schedule, cosine to
  zero). The intermediate checkpoints are diagnostics only.
- Everything stored per run and resumable with cache keys, as in V3: code commit, data config
  hash, archive manifest hash, head sha256, settings.
- **The equivalence margin for "keep": 40 GeoScore (decision 3, fixed 2026-11-18, before any V4
  dev result; never moved afterwards).**
  - **Rule:** Δ = fine-tuned gap − frozen gap on the same test images. "Kept" means the TOST shows
    |Δ| < 40, i.e. both one-sided tests at α = 0.05: the 90% CI of Δ lies inside (−40, +40).
  - **Noise scale:** Δ is a difference of two paired gaps on the same images, like V3's H4/H5.
    Their creator-clustered 95% CI half-widths on test were ±21.8 and ±20.9 (H1–H3: ±20–25;
    `results/v3/test_primary.csv`). So σ ≈ 10.9, and the 90% CI half-width is ≈ 17.9.
  - **Power:** the chance that the TOST shows "kept" (normal approximation, σ = 10.9):

    | margin | true Δ = 0 | Δ = 10 | Δ = 20 | at α = 0.0125 (Holm, 4 tests), Δ = 0 |
    |---|---|---|---|---|
    | 25 | 0.48 | 0.34 | 0.11 | 0.04 |
    | 30 | 0.73 | 0.55 | 0.23 | 0.39 |
    | 35 | 0.88 | 0.74 | 0.39 | 0.67 |
    | **40** | **0.96** | **0.86** | **0.58** | **0.85** |
    | 45 | 0.99 | 0.94 | 0.74 | 0.94 |

    So 25 could hardly show "kept" even when nothing changes: under Holm it does so 4% of the
    time. 40 shows it in 85–96% of cases when nothing changes.
  - **Upper limit:** 40 is below every V3 effect of the same kind: the init effects on the gap,
    H5 = 51 and H4 = 82. A fine-tuning effect as large as a text-initialised head's counts as a
    change.
  - **Caveats, declared:**
    - 40 is above V3's N effect, H3 = +30.7. A change of that size would count as "kept".
    - On dev, the frozen bal cell_zeroshot gap at 200K was ≈ 69 (dev_seen). So "kept" there allows
      a change of more than half the lead.
    - The CI covers images and creators, not training seeds. One seed per cell (decision rules
      below).
  - **Close and keep overlap.** A Δ that is significantly < 0 and also inside ±40 is reported as
    "**narrowed, within the margin**", not as either one alone. "Reversed" means the fine-tuned
    gap itself is significantly < 0.

**Dev-only tuning rules (V3-style, fixed now).**
- lr is chosen per backbone by dev_unseen GeoScore at the final checkpoint.
  - **Tie tolerance:** if the two best lrs are within 5 GeoScore, the lower lr is kept.
  - Ties exactly at 5 also keep the lower.
- **Grid-edge rule:**
  - If the selected lr is at the **same** edge of {1e-4, 3e-4, 1e-3} for at least one backbone,
    the next ×3 step beyond that edge runs for **both** backbones (one grid for both). This is the
    one reserved pair per N where a search runs.
  - If the backbones bind at **opposite** edges, or a lower and an upper edge bind at once,
    **neither is extended**: both are flagged `grid_edge` and reported with the margin to their
    neighbour. One grid stays one grid.
  - **Stopping rule:** this is the only extension. If the extension lr is then selected, or the
    edge still binds, the grid is frozen and flagged.
- This rule is written before any V4 dev result exists. (V3's lr-edge threshold was set after
  seeing live logs.)
- B-S1: the large-N lr is the 50K choice, never re-searched. B-S2: the short-run choice is
  final.
- Seeds: one per cell (seed 42). Seed 2 runs only as declared, and is descriptive.

**The V4 test pre-registration** (after dev, before any test image is read):
- **Hypotheses.** "Close, keep or reverse" is a three-way question: one-sided tests for close and
  reverse, TOST with the margin above for keep. Directions may come from dev; the margin may not.
  Candidates:
  - the change in the paired gap, fine-tuned vs frozen, at each N;
  - the fine-tuned gap at each N;
  - the change in the fine-tuned gap between the two N.
- The statistic: the paired creator-cluster bootstrap, 10,000 resamples, seed 42, Holm over the
  family, as in V3.
- The frozen comparators: V3 test values exist and are published, so the fine-tuned vs frozen
  contrasts are not blind on the frozen side. Declared as a limitation.
- The run identity: adapter and head sha256, prereg sha256, code commit, written into every
  output. Test refuses other weights. No retrain is needed.
- The test split is opened only by a V4 test stage that refuses to start unless the V4
  pre-registration is committed and unmodified (as `phase3_final`). The test zips are staged in
  that session only.

## 6. Open risks

- **The LoRA rates are estimates.** At C, 50K vs 500K does not fit with a full-length search; the
  rule picks the option from the pilot.
- **The CPU decode headroom** is about 1× on 4 cores with two runs. It may cap r below the GPU
  rate.
- **StreetCLIP is 10–20% slower** than its identical-architecture base in every measurement so far;
  cause unknown.
- **`/kaggle/temp` capacity** for 28 GB is undocumented. **HF throttling** of about 100–150 GB of
  private downloads is unmeasured.
- **The frozen comparators** were tuned with their own grids and on dev_seen (V3 limitations). A
  fine-tuned vs frozen difference also includes that.
- **One subset draw**, inherited from V2/V3 (seed 42).
- **StreetCLIP's pretraining** may overlap OSV-5M sources (not bounded), which would favour
  StreetCLIP.
- **Licences:** StreetCLIP is CC BY-NC 4.0, so its LoRA adapters are derivatives and
  non-commercial. OSV-5M is CC BY-SA 4.0.

## 7. Independent review of the first draft (and what changed)

An independent reviewer (a separate Opus agent, read-only) checked the first draft against the repo
and the measured logs. It re-ran the budget script and confirmed that the first draft's tables
matched it. Its findings, by the area each one covered:

| # | Area | Finding | Change |
|---|---|---|---|
| 1 | Decision rule | at C the draft's own rule selected B-S2, not the recommended B-S1 (61.8 > 60 h) | break-even rates stated per option; the rule is a table over r; diagnostic evals cut to 2 × 5K |
| 2 | Rates | P was not pessimistic: the full-FT step shows a 1.3× overhead over FLOP counts | P/C/O recalibrated to 9/12/17 (from 12/16/22); the rule covers rates down to 6.9 |
| 3 | Throughput | backbone and GPU confounded (StreetCLIP 10–20% slower) | swapped-GPU config, `nvidia-smi` log, budget uses the slower backbone |
| 4 | Pilot | the sustained run could produce nothing (JSON only at the end, tight timeout) and was not sustained | per-phase saves, a train-only phase on both GPUs, a mixed train/eval phase, sampling over all parts |
| 5 | Pre-registration | TOST margin "from dev" was exploitable | margin fixed now (25 GeoScore, open for your decision before dev) |
| 6 | Grid edge | the edge reserve was asymmetric (opposite edges); tie rule vacuous; "A-short" undefined | opposite or both edges: no extension; 5 GeoScore tie tolerance; seed-2 conditionality stated |
| 7 | Fairness | augmentation/head-training confound left to Option C | augmented-frozen control at 50K in every option |
| 8 | Fairness | a shared lr made the LP-FT head near-frozen, with backbone-dependent starting fit | head lr = 10× adapter lr, declared; initial loss logged |
| 9 | Selection | dev_seen selection favours memorising creators, more at 50K | select on all of dev_unseen; dev_seen as a diagnostic |
| 10 | Budget | weekly packing and lost sessions ignored; 10% slack < one session | whole-session packing per job, one-lost-session reserve, weekly sketch |
| 11 | Staging | dev and nat_50K spread over all 13 parts | full 12.44 GB per nat session stated; the new archive packs by rank with dev parts separate |
| 12 | Decode | the decode estimate was cherry-picked (37.5 of 37.5–49.1 ms) | 45–60 ms under contention; CPU named the tightest resource; worker sweep and `vmstat` |
| 13 | Quota | the interactive preflight confounded the quota measurement | batch-only, asserted |
| 14 | Budget | missing costs: zero-init check, dev embeddings for the hybrid, hard-kill loss, resume state | once per backbone on 2K; dev embeddings stored at the final eval; checkpoints pushed to HF; resume state listed |
| 15 | Pilot | no check that LoRA gradients flow under checkpointing | `grads_ok` asserted in bench, covered by a test |
| 16 | Memory | the memory estimate was low (quick_gelu, fp32 LayerNorm) | 0.8–0.95 GB per image; batch 16 without checkpointing called borderline; OOM also caught in the real-data step |
| 17 | Export | `merged()` rounded B·A to fp16 before the add | summed in fp32 |
| 18 | Framing | A vs B-S1 framing; B-S2 grid size and short schedule undefined; bal argued from test; dirty-clone check not asserted | all fixed |

Not checked by either of us: Kaggle's official T4×2 quota rule, the `/kaggle/temp` limit, HF rate
limits. The quota rule has since been observed as 1× (section 2). The pilot or the first real
session measures the other two.

**After the review (2026-11-18):** the decisions at the top. The margin changed from the drafted 25
to 40, with the power analysis in section 5, because 25 could hardly show "kept". A `bal_rank`
selector and the CPU archive notebook were added.

## Sources

- HF Hub API `list_repo_tree` for `osv5m/osv5m` (file sizes) and `Daggerabuse/osv5m-ft-archive`
  (`manifest.csv`), read 2026-11-18.
- [Hugging Face storage limits](https://huggingface.co/docs/hub/storage-limits): 100 GB private
  storage, free accounts.
- Kaggle limits (no official page readable; secondary):
  [knowledge-base summary](https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/93babea0b1c303124af8c062c459fbceea4c3208/kaggle_20251121.md),
  [Notre Dame course notes](https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html),
  [The Kaggle Book (Packt), datasets](https://subscription.packtpub.com/book/data/9781801817479/3/ch03lvl1sec07/setting-up-a-dataset).
- Our own Kaggle measurements: `runs/phase0/bench.jsonl`, `runs/phase0/emb/*/throughput.jsonl`,
  `runs/pilot/*/throughput.jsonl`, `runs/pilot/zipq_stats.json`, `runs/pilot.log` (not in the
  repo; `runs/` is git-ignored).
