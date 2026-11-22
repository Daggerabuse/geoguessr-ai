# V4 feasibility: LoRA fine-tuning of StreetCLIP vs CLIP L/14-336 (design only)

Status: a feasibility study and pilot design. No V4 training run has happened, no V4 number
exists, and the OSV-5M test split stays unopened for V4 until a V4 pre-registration is
committed. Every rate below is either **measured** (with the file it comes from) or **estimated**
(with how it was derived). Estimates are what the pilot (section 4) replaces. Section 7 lists the
independent review of a first draft and what changed because of it.

**Update 2026-11-20 (after the first V4 dev results; section 5c).** Recipe A (the head trained with
the adapters at 10× their lr) finished below the frozen V3 heads on both dev splits. It is recorded
as a negative dev result and stops after job 2 (its edge step is dropped). **Recipe B** (the head
frozen at the V3 head, LoRA only) **was chosen after seeing those results** and replaces it
(`configs/v4_predev_v2.json`). The test split is still unopened.

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

**After the pilot (run 2026-11-18, 1 h 27 min; section 4b).**
- **Config:** LoRA on the **last 12 blocks** (decision 6, chosen on speed only, before any dev
  result).
- **Rate:** r = **16.0 img/s**, the slower GPU's real-data train rate.
- **Option:** the rule picks **B-S2**: 50K vs **500K**, with a short-horizon lr search at 50K.
- **Budget:** 49.8 quota h, **59.2 h** with the lost-session reserve. That is just inside 60.
- The all-blocks configs are slower: 8.0 img/s with checkpointing, about 10.9 without. With them
  the rule picks option A (50K only, no N comparison).

**Decisions (2026-11-18, before any V4 run or dev result).**
1. **N:** 50K vs 200K (B-full-200) is accepted as the answer if the pilot-rate rule (section 3)
   selects it. The rule itself is unchanged.
2. **Ordering: bal.** bal needs a new archive. It is built in a CPU session
   (`runs/kaggle_v4_bal_archive.ipynb`, section 1).
3. **Equivalence margin for "keep": 40 GeoScore**, fixed with its reasoning in section 5.
4. **Head lr = 10× the adapter lr**, fixed. (Recipe A only. Recipe B freezes the head; section
   5c, decided 2026-11-20 after seeing recipe A's dev results.)
5. **Quota:** a T4×2 session costs 1 quota h per wall-clock hour. **Confirmed** by the pilot
   (section 2).
6. **LoRA config: r = 16 on q,k,v,out_proj of the last 12 of 24 blocks**, for both backbones.
   This replaces the a-priori "all 24 blocks". The reasoning, and what it can and cannot answer,
   are in section 4b.
7. **GPU assignment:** the backbones alternate between the two GPUs in a fixed, declared pattern
   (section 4b).
8. **BAL_MAX = 500,000** for the bal archive (B-S2 needs bal_500K).

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
- **BAL_MAX = 500,000** (decision 8): the rule picked B-S2. That is bal_500K + dev, about 28 GB.
  It would have been 200,000 for B-full-200.
- The notebook is set to 500,000 and is ready to run now that the pilot is done.
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
- **That is about 3.5–4 cores of 4: the loader can become the bottleneck.** The pilot disproved
  this estimate: the CPU was 43% idle on average, the loader ran ≫ the training rate, and no
  fallback is needed (section 4b).
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

- The budget now uses the pilot's **29.3 img/s per T4 for any evaluation or re-embedding**. That
  is the real-data eval rate of the last-12 config on the slower GPU (section 4b). The first draft
  used 33.
- **StreetCLIP and L/14-336 have the same architecture, yet StreetCLIP measured 10–20% slower
  every time.** The pilot's swapped run shows that this is the **GPU, not the backbone**
  (section 4b).
- The budget uses the **slower GPU's** rate, because a pair of runs lasts as long as its slower
  half.

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
- **Measured** (pilot, real data, slower / faster GPU; section 4b):
  - all 24 blocks with checkpointing: 8.0 / 9.2;
  - all 24 blocks without checkpointing: about 10.9 / 12.6. This is synthetic only, but real data
    matched synthetic within 3% wherever both were measured;
  - last 12 blocks: 16.0 / 18.7.

  The estimates were 9 / 12 / 17, about 10% optimistic.

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
- **Measured peak memory** (pilot):
  - all 24 blocks without checkpointing: 9.7 GB at micro-batch 16, 14.3 GB at 24. The estimate
    above was too high; 16 is safe.
  - with checkpointing: 2.3 GB at 16, 3.9 GB at 32.
  - last 12 blocks: 5.4 GB at 16, 10.0 GB at 32.
  - **Adding fc1/fc2 to the targets (all 24 blocks, no checkpointing, micro-batch 16) ran out of
    memory on both GPUs** (10.9M trainable parameters). It was not tried with checkpointing or a
    smaller batch. MLP targets are not part of any V4 config.

**Fixed a priori, then changed once after the pilot (decision 6; section 4b).**
- The a-priori config was LoRA r=16, alpha=16, on q,k,v,out_proj of **all 24 blocks**. The first
  draft called last-12 "a capacity choice, not a speed setting, listed only as an alternative".
- After the pilot, and before any V4 dev result, it became **the last 12 blocks**, for both
  backbones. The trainable parameter count is identical: 1.57M adapter + 3.85M head = 5.42M.

**Layout: two independent runs per session, not DDP.**
- One run per GPU, the two backbones side by side at the same lr. Which backbone runs on which
  GPU alternates in a fixed pattern (section 4b).
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
- **Confirmed by the pilot.** Its batch session ran 13:40:46–15:08:09 UTC (1 h 27 min,
  `quota_times.txt`). The quota went from about 11 h to 9 h 28 min, i.e. about 1.5 quota h were
  used. At 2× it would have been about 2.9 h.
- The budget counts session wall-clock hours.
- Rules: batch sessions only; a deadline stop before 12 h; checkpoints every 30 min, also pushed
  to HF.

**Per-run cost: what one pair job contains** (`scripts/v4_budget.py`; every number is an input to
it):
- **Training:** S images seen per run at rate r.
- **Evals:**
  - the final checkpoint on **dev_unseen** (20,000; the selection split, section 5) and on
    dev_seen (19,906; diagnostic);
  - two intermediate checkpoints on a 5K dev_unseen subset (diagnostic only).

  That is 0.47 h per run at the measured 29.3 img/s. The first draft had five full dev_seen
  evals, 1.0 h.
- **Sessions:** a job uses whole sessions of ≤ 11 h. A longer job resumes in the next session and
  pays the setup again: 0.25 h for clone, pip, two model downloads of about 1.7 GB each, data
  staging and preflight.
- **Reserve:** one lost session (min(job, 11 h) + 0.25 h). It replaces the first draft's flat
  10%, which was less than one failed session.
- **Fixed costs:**
  - the pilot, 1.5 h (measured; planned 2.5);
  - a zero-init equality check, once per backbone on 2K dev_unseen images (minutes);
  - the augmented-frozen control at 50K (section 3);
  - the 50K hybrid re-embedding: 50K train images per model. The dev embeddings are stored by the
    final eval, so there is no second dev pass;
  - **the later test pass:** 4 models × 210,122 images at 29.3 img/s ≈ 8.0 GPU h, i.e. 4.0 quota h
    as two pairs.
    The test zips (9.4 GB) are staged in that session only.
- **Not needed** (unlike V3): a retrain for the test run. The adapters and heads are small and are
  saved with sha256, so the test run loads exactly the dev-selected weights.
- **Resume state**, all of it in the checkpoint:
  - adapters, head, AdamW, GradScaler;
  - scheduler step;
  - the sampler's permutation seed and position: an index sampler skips without decoding;
  - torch/numpy RNG.

  Measured: 65 MB (last 12 blocks) or 84 MB (all blocks), with a save/load round trip of
  0.2–0.5 s.

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

**Budget before the pilot** (estimated rates, eval 33 img/s, pilot 2.5 h; kept as written). The
measured budget follows the decision rule below. Quota h for P / C / O (9 / 12 / 17 img/s per
T4), **including the one-lost-session reserve**:

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

- Before the pilot, my expectation (C ≈ 12) landed in B-full-200, which decision 1 accepted.
- Unused reserve hours (no lost session, no edge extension) buy a seed-2 replicate at 50K,
  declared now. The resulting seed estimate is therefore conditional on no edge binding.

**The rule applied (2026-11-18, after the pilot, before any V4 dev result).**
- The rule names "the all-blocks config". The config choice was opened again before applying it
  (decision 6, section 4b), because no config fits as declared:
  - all blocks with checkpointing (cell 4a): r = 8.0;
  - all blocks without checkpointing: about 10.9.
  - Checkpointing is a throughput-only choice, so the declared config at its best gives r ≈ 10.9:
    option A.
- **With the chosen last-12 config, r = 16.0** (slower GPU, real data, 3000 images; the sustained
  run with checkpointing matched its short run exactly). **15.6 ≤ 16.0 < 19.0, so the rule picks
  B-S2.**
- **Recomputed with the measured inputs** (eval 29.3 img/s, pilot 1.5 h), the break-evens are:
  B-S2 15.8, B-full-200 11.4, A (S = 200K) 6.9, B-S1 19.2. 16.0 still selects B-S2, but by only
  0.2 img/s. The declared thresholds are applied as written, and they give the same answer.

**Measured budget, B-S2 at r = 16.0** (`uv run python scripts/v4_budget.py`):

| Job (pairs, both backbones) | each | quota h |
|---|---|---|
| short lr search at 50K, 100K images seen: 3 lr + 1 edge | 2.11 h + 0.25 setup | 9.4 |
| full runs, 500K images seen: 50K at the selected lr, 500K at the same lr, seed 2 at 50K | 9.15 h + 0.25 | 28.2 |
| augmented-frozen control (forward-only, 500K seen) | 5.21 + 0.25 | 5.5 |
| 50K hybrid re-embedding | 0.47 + 0.25 | 0.7 |
| test pass, 4 models (2 pairs) | 1.99 + 0.25 | 4.5 |
| pilot (done) | | 1.5 |
| **total** | | **49.8** |
| + one lost 9.4 h session (reserve) | | **59.2** |

- **It fits 60 h only with no slack beyond the reserve.** If the sustained rate falls to 15.8 the
  total is 59.8 h; at 15.0 it is 62.0 h.
- The rule is not re-applied after the first real hour (section 4). If the real rate comes in
  lower, the reserve is spent first. Then the declared seed-2 run at 50K is dropped: it is
  descriptive only.
- The large N is the asked-for **500K**, seen once. Decision 1's 50K vs 200K fallback is not
  needed.

**Weekly packing (B-S2 at 16.0).** This is a strict chain; each full pair fills one session.
1. **Before any GPU run:** the bal_500K archive (CPU, no quota) and the committed V4 pre-dev note.
2. **Week 1, about 28.2 h:**
   - the 4 short search pairs (2 per session, about 9.4 h);
   - the 50K full pair and the 500K full pair (one session each).
3. **Week 2, about 20 h plus the reserve:**
   - seed 2 at 50K, the control and the hybrid;
   - the dev analysis and the V4 test pre-registration;
   - the test pass.

This week has 9 h 28 min left. That is just enough for the 4 short pairs (9.4 h), with no
margin. Starting them next week is safer.

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

## 4. Pilot (≤ 2.5 quota h; run 2026-11-18, 1 h 27 min, all steps exit 0; results in 4b)

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

## 4b. Pilot results (2026-11-18)

Outputs: `runs/v4_pilot/v4_pilot/`, git-ignored:
- `bench_v4.jsonl` and `bench_v4_summary.csv`;
- `gpu_monitor.csv`, `cpu_vmstat.log`;
- `downloads.json`, `quota_times.txt`.

Every config: StreetCLIP and L/14-336 at the same time, one per GPU, LoRA r = 16 unless noted, a
5000-class head with CE, micro-batch 16 unless noted. Rates are img/s per T4.

| config | data | cuda:0 (slower) | cuda:1 | peak GB |
|---|---|---|---|---|
| all 24 blocks + checkpointing | synthetic | 7.9 | 9.2 | 2.3 |
| ″, real images, 2 / 3 workers | e2e train | 8.0 / 8.0 | 9.1 / 9.2 | |
| ″, sustained 8,000 images (cell 4a) | e2e train | **8.0** | 9.2 | |
| ″, micro-batch 32 | synthetic | 8.0 | 9.3 | 3.9 |
| all 24 blocks, no checkpointing | synthetic | **10.9** | 12.6 | 9.7 |
| ″, micro-batch 24 | synthetic | 11.0 | 12.9 | 14.3 |
| ″, rank 8 | synthetic | 10.9 | 12.7 | 9.7 |
| ″, + fc1/fc2 targets | synthetic | **OOM** | **OOM** | |
| **last 12 blocks**, no checkpointing | synthetic | 16.4 | 19.3 | 5.4 |
| ″, real images, 2 workers | e2e train | **16.0** | 18.7 | |
| ″, micro-batch 32 | synthetic | 16.2 | 19.3 | 10.0 |
| eval with LoRA, real images (all / last 12) | e2e eval | 26.4 / **29.3** | 30.9 / 34.6 | |
| mixed: cuda:0 evaluates while cuda:1 trains | e2e | 26.4 eval | 9.2 train | |

- **Real data matches synthetic.** The loader delivers 92–141 img/s of `train_view` images with
  2–3 workers, ≫ 16 needed.
- **CPU:** over the session 39% user, 17% system, 43% idle. It was < 10% idle in 21 of 524
  samples.
  - The CPU is not the bottleneck at these rates, and the decode fallbacks (section 1) are not
    needed.
  - 2 workers per run.
  - Training next to an eval did not slow either side.
- **Staging:**
  - HF archive parts arrived at 101–286 MB/s, and one OSV-5M train zip (2.5 GB) at 240 MB/s. A
    28 GB bal_500K staging therefore takes about 2–5 min.
  - `df` reports no separate `/kaggle/temp` mount. It is a directory on the root overlay (8 TB,
    965 GB free).
- **Checkpoint:** 65 MB (last 12) or 84 MB (all), with a round trip of 0.2–0.5 s.
- **Rank:** 8 vs 16 does not change speed. r = 16 stays.
- **MLP targets:** adding fc1/fc2 ran out of memory on both GPUs (all blocks, no checkpointing,
  micro-batch 16, 10.9M trainable). It was not retried with checkpointing or a smaller batch,
  since V4's targets are q,k,v,o by declaration.
- **Quota:** 1 h 27 min of session cost about 1.5 quota h, so a T4×2 session counts **1×**
  (section 2).

**GPU asymmetry: the GPU, not the backbone.**
- **The swapped run decides it.** StreetCLIP ran at 7.9 on cuda:0 and 9.6 on cuda:1. L/14-336 ran
  at 9.2 on cuda:1 and 8.0 on cuda:0.
- **cuda:0 is 13–17% slower in every config**, whichever backbone runs on it. The earlier "StreetCLIP
  is 10–20% slower" was this confound.
- **`gpu_monitor.csv`** (samples at ≥ 80% utilisation; 443 on cuda:0, 412 on cuda:1):
  - both GPUs ran at ≈ 100% utilisation and **67–68 W**, i.e. at the T4's 70 W power cap;
  - at that same power, **cuda:0's SM clock had a median of 630 MHz vs 855 MHz** on cuda:1
    (−26%; both far below the 1590 MHz boost);
  - cuda:0 ran hotter: median 77 °C vs 76 °C, mean 77 vs 74.
  - It is steady, not a warm-up effect: 10-min means were 593–654 vs 807–893 MHz over the whole
    87 min.
- **Interpretation:** both cards are power-limited, and cuda:0 needs more power per MHz (a hotter
  card leaks more, or chip-to-chip variation), so the cap holds it lower. It is a property of
  this machine's card, not of the code or the backbone. Another session may get another machine,
  and the slower card may be the other index.

**GPU assignment rule (decision 7, declared now).**
- **The GPU affects speed, not results.** Both cards are T4s running the same code and seeds. The
  rule only keeps hardware from lining up with a backbone, and the budget honest.
- **Alternate in a fixed pattern.** Number the pair jobs k = 0, 1, 2, … in the order the pre-dev
  note lists them. In job k, StreetCLIP runs on `cuda:(k mod 2)` and L/14-336 on the other GPU.
  - In B-S2, the 4 short search pairs then put each backbone twice on each GPU.
  - The 50K and 500K full pairs are consecutive, so each backbone has one N on each GPU.
  - The pattern is fixed in advance, never chosen by observed speed.
- **Log it.** Every run records the GPU index, its `nvidia-smi` UUID, and the 10 s clock /
  temperature / power monitor. A hardware effect can then be checked afterwards: GeoScore should
  not depend on the GPU.
- **Budget at the slower card.** A pair lasts as long as its slower run, so the faster GPU idles
  about 15% of each job (about 1.3 h in a 9.15 h full pair). Filling that time (e.g. evaluating
  the partner's checkpoints on it) is not worth the coupling.

**LoRA config: last 12 blocks (decision 6, 2026-11-18, before any V4 dev result).**
- **Why speed may decide it here.** The pilot used dummy labels: no loss value was compared, and
  no accuracy, GeoScore or dev number exists. A choice made now cannot be tuned to the outcome,
  and it is recorded with its date. It does change an a-priori setting (section 2), and that is
  declared.
- **The two candidates, by the rule:**
  - **All 24 blocks, no checkpointing:** r ≈ 10.9 gives **option A**: 50K only, 4 epochs, seed 2,
    the control, about 42 h. It answers close / keep / reverse at 50K with full-depth adaptation,
    but **not the N question**, half of what V4 asks.
  - **Last 12 blocks:** r = 16.0 gives **option B-S2**: 50K vs the asked-for 500K, about 59 h. It
    answers both halves, at the price of adapting only the upper half and a short-horizon lr
    search.
- **Chosen: last 12 blocks.** Only this config answers the N question within two weeks, and it
  keeps N = 500K.
- **What last-12 can answer.** Whether adapting the upper 12 blocks of each encoder (with the
  LP-FT head) closes, keeps or reverses StreetCLIP's lead, and how that changes from 50K to 500K.
  - Both backbones get exactly the same adaptable capacity: the same blocks, rank and parameter
    count (1.57M adapter + 3.85M head).
  - The frozen lower half is V3's. The zero-init identity check still applies.
- **What it cannot answer.**
  - **Full-depth fine-tuning.** The lower 12 blocks stay frozen. Whatever StreetCLIP's
    geolocation pretraining changed there stays in StreetCLIP and cannot be learned by L/14-336:
    low-level, local features such as textures, road markings, script shapes and camera traits.
    If the lead lives there, last-12 is **biased toward "keep"**.
  - So a "keep" is weaker evidence than a "close" or a "reverse". The conclusion is worded as
    "LoRA on the upper 12 blocks", never "fine-tuning" in general.
  - A "close" says that upper-half adaptation was enough. It does not say what full-depth
    adaptation would do, since the same lr need not suit a deeper adapter.
  - Which layers carry the lead is not answered.
- **The all-blocks question** (option A's design at 50K, about 42 h at r ≈ 10.9) stays a possible
  follow-up. It is not part of V4.

## 5. What a V4 pre-registration must fix, and dev-only tuning rules

**Before any V4 training run** (a dated pre-dev note, committed):
- The option (by the rule in section 3, from the pilot's r), ordering, N values, S, and the
  checkpoint schedule.
- The GPU assignment pattern (section 4b) and the pair-job order it is keyed to.
- The LoRA config (r = 16, alpha = 16, q,k,v,out_proj, **last 12 blocks**), optimizer, warmup, cosine, effective batch, weight
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

## 5b. Training pipeline (code ready, inputs fixed, no training run yet; 2026-11-19, before any V4 dev result)

**Files.**
- `configs/v4_predev.json`: the machine half of the pre-dev note, with the input sha256s filled in
  (below). The trainer refuses a null value.
- `src/geoguessr_ai/v4_inputs.py`: the inputs bundle.
- `src/geoguessr_ai/v4_train.py`: `train`, `identity`, `stage` and `hfcheck`.
- `src/geoguessr_ai/v4_select.py`: the section 5 rules.
- `scripts/kaggle_v4.sh`: the pair runner.
- Notebooks: `runs/kaggle_v4_inputs.ipynb` (CPU) and `runs/kaggle_v4_train.ipynb` (T4×2).
- Tests: `tests/test_v4_*.py` and the `kaggle_v4.sh` cases in `tests/test_kaggle_scripts.py`.

**No test data is attached.** A training session attaches only:
- the **inputs bundle**: train rows with bal_rank < 500K, dev, dev_seen, the frozen partition, V3's
  bal zero-shot heads for 50K and 500K, and the 2K V3 dev_unseen embeddings for the identity check.
  Every file is sha-checked against its V3 source, and a `MANIFEST.sha256` whose own sha256 is in
  the config;
- its own previous output, to resume.

**Fixed inputs (2026-11-19, both CPU sessions at commit f538707).**
- Inputs bundle: the output of Kaggle notebook `notebook8c927d8f13`. `MANIFEST.sha256` =
  `1fc4e0e5c1fc71f24948380ddc27f03b96c4b35f815e2e6a79d3f9b1900a27ca`. The downloaded copy passes
  `verify_bundle` locally (10 files). The training notebook also asserts the attached bundle comes
  from `notebook8c927d8f13` (`INPUTS_NAME`).
- Bal archive: `Daggerabuse/osv5m-ft-archive-bal` (private), built by Kaggle notebook
  `notebook5109c74927`. 539,906 images in 25 parts, 27.531 GB; none missing, none extra, no
  duplicates. `manifest.csv` sha256 =
  `0fe28db46b9dc2d14de689c6a55f25387bb66e5505cfaef90fec9e08e10d3752`, computed from the downloaded
  copy; the copy re-fetched from the repo has the same sha256.
- Partition (unchanged): `9882eb2ae7124f3331859bbf742a52c9d30fca517a31d16a83bfd77d6b74662b`.

The bundle is built in a CPU session from the Phase 1 output. That is the only session that has
test.csv mounted, and it never opens it. The training preflight refuses any other input and any
attached path containing "test".

**Recipe values fixed here** (not in sections 3–5 before):
- label smoothing 0.1 and logit scale 100 (V1 and the V3 probe);
- gradient clip 1.0, with the scheduler step skipped on an fp16 overflow (`train.finetune`);
- weight decay 0 on LoRA A/B and on the head;
- AdamW (0.9, 0.999, 1e-8);
- micro-batch 16 × 4 accumulation = effective batch 64;
- warmup = 5% of the steps; steps = S // 64.

**Data order.** One stream over seeded per-epoch permutations. An image's augmentation depends only
on (seed, epoch, bal_rank), so both backbones, and both N where they overlap, see the same pixels.

**Evals.**
- **Per-epoch 2K dev_unseen diagnostic, in the 50K full run (job 4) only.** One eval each time
  an epoch boundary is crossed: 9 evals, about 0.17 h. The short runs, the 500K run, the seed-2
  replicate and the control have none (`evals.diagnostic_jobs = [4]`; decided 2026-11-19 to keep
  the budget inside 60 h). **Diagnostic only: never used for selection, stopping or checkpoint
  choice.** A test checks that stubbing it out leaves the final weights bit-identical.
- 5K dev_unseen at S/3 and 2S/3, in the full and control runs only, as budgeted. The short runs
  have none. Diagnostic only.
- At the final checkpoint: all of dev_unseen (the selection split) and dev_seen.

**Job numbering k** (the GPU pattern is keyed to it; a skipped slot keeps its number):
- 0–2: short runs at 1e-4, 3e-4, 1e-3;
- 3: the edge step, only if `v4_select` asks for it;
- 4: 50K full; 5: 500K full;
- 6: seed-2 replicate at 50K, **seed 43** (confirmed 2026-11-19);
- 7: the augmented-frozen control.

The lr for jobs 3–7 is read from the committed `configs/v4_selection.json`. **The control's head
lr = 10 × that backbone's selected adapter lr** (confirmed 2026-11-19), i.e. the same head lr as
its 50K full run.

**Resume.**
- A checkpoint holds the trainable weights, AdamW, GradScaler, the scheduler, the stream position,
  the RNG states and a run key. The key covers the config, the job, the code commit, and the
  sha256 of the bundle, archive manifest, head and partition.
- Checkpoints are written every 450 steps (≈ 30 min), at each eval point and on STOP, and pushed
  to the private runs repo. A key mismatch is refused, so a code fix restarts an unfinished run.
- Measured on CPU (the tiny model, with STOP from outside at step 60 of 200): the resumed run's
  final weights and its loss/eval log are bit-identical to an uninterrupted run. On GPU,
  nondeterministic kernels make a resume close, not bit-identical.

**HF write access.** The Kaggle secret `HF_TOKEN` needs **write** access, because checkpoints are
pushed to `Daggerabuse/geoguessr-v4-runs`. The notebook preflight and `kaggle_v4.sh` both put and
delete a file there, and assert that the repo is private, before staging or training.

**Budget with the per-epoch diagnostic** (`scripts/v4_budget.py`; the reserve is now the option's
longest job):
- With the diagnostic in job 4 only, B-S2 at r = 16.0 is **50.0 + 9.6 reserve = 59.6 h** (was
  49.8 + 9.4 = 59.2). The reserve grows because job 4 is now the longest job. The break-even is
  **15.9 img/s** (was 15.8); at 15.8 the total is 60.1 h.
- A diagnostic in every training run would have cost 60.0 h at r = 16.0 (break-even 16.1, above
  the measured rate). That is why it is limited to job 4.
- The declared order of cuts is unchanged: the reserve first, then the seed-2 run.
- The other options' rows are unchanged (they carry no diagnostic).

## 5c. Recipe A: a negative dev result; recipe B (2026-11-20, AFTER seeing V4 dev results)

Everything in this section was decided **after** recipe A's first dev results were seen. It is
not part of the pre-dev note of sections 3–5b, and is reported as a post-hoc change.

**Recipe A** (`configs/v4_predev.json`, commit 26d98a4): LoRA r16 on the last 12 blocks plus the
cosine head, initialised from V3's bal `cell_zeroshot` head of the same N and **trained at 10× the
adapter lr**. Short runs at 50K (100K images seen): session 1 (jobs 0, 1; 274 min, exit 0) and
session 2 (job 2; 130 min, exit 0; same commit 26d98a4, seeded from session 1's output):

| dev GeoScore (top-1 cell centroid) | StreetCLIP dev_unseen | L/14-336 dev_unseen | StreetCLIP dev_seen | L/14-336 dev_seen |
|---|---|---|---|---|
| V3 frozen `cell_zeroshot`, bal 50K (`results/v3/phase3_dev_predictors.csv`) | 3372 [3237, 3587] | 3215 [2861, 3460] | 3586 | 3447 |
| A, job 0: lr 1e-4 (head 1e-3) | 3163 [2924, 3383] | 3002 [2607, 3277] | 3427 | 3202 |
| A, job 1: lr 3e-4 (head 3e-3) | 3059 [2801, 3313] | 2952 [2620, 3241] | 3368 | 3225 |
| A, job 2: lr 1e-3 (head 1e-2) | 3065 [2847, 3306] | 2988 [2701, 3226] | 3385 | 3255 |

- Brackets are 95% creator-cluster bootstrap CIs (dev_unseen). Selection would be on dev_unseen;
  dev_seen is diagnostic.
- All three short runs finished below the frozen starting point on **both** splits, so this is not
  only an effect of unseen creators. The lowest lr (1e-4) was best for both backbones; 3e-4 and
  1e-3 are within 36 of each other.
- What A's declared rule says on the full grid (`v4_select` with `configs/v4_predev.json`, run
  locally, not committed): both backbones pick 1e-4, the low edge, so it asks for the edge step,
  job 3 at lr 3.33e-5. That step is not run (below).
- **No bug found** (checked before any further GPU run):
  - **Labels:** the trainer's labels are `geocells.assign` on the same lat/lon as V3's probe,
    over the same `bal_rank < 50000` rows.
    - 3781 cells are used. The median distance from an image to its label's cell centroid is
      13 km (99th percentile 537 km); with lat/lon swapped it would be 5847 km.
    - 94.6% of images share their cell's majority country.
  - **Image/label alignment:** the initial train loss is 3.12 (L/14) / 3.23 (StreetCLIP),
    against ≈ 8.5 for labels unrelated to the images.
  - **fp16:** exactly one overflow skip per run in jobs 0–2, at the GradScaler's first step
    (the logged lr is the schedule's value one step behind).
  - **Eval preprocessing** equals V3's: the zero-init identity check agrees on top-1 for 99.75% /
    99.4% of images, mean cos 0.999998.
- **The head was already fitted to these images.** V3's head reaches a loss of 3.1 on its own
  augmented training images, against 6.5–6.6 on clean dev_unseen images (2K subset). V3 trained
  it for 23 epochs on exactly these 50K images. Recipe A then trains it further on the same
  images, at 10× the adapter lr and batch 64.
- **Post-hoc decomposition** (2K dev_unseen subset, CPU, chosen after seeing the numbers above;
  diagnostic only, not additive, wide CIs):

| GeoScore, 2K subset | V3 emb × V3 head | V3 emb × A head | A emb × V3 head | A emb × A head |
|---|---|---|---|---|
| StreetCLIP, job 0 | 3339 | 3133 | 3287 | 3135 |
| StreetCLIP, job 1 | 3339 | 2736 | 3364 | 3034 |
| L/14-336, job 0 | 3205 | 3046 | 3114 | 3014 |
| L/14-336, job 1 | 3205 | 2724 | 3199 | 2950 |

  - The adapted encoder moved a lot: mean cos(A emb, V3 emb) is 0.69–0.78.
  - Even so, paired with the **V3 head** it scored about as well as V3 (within ±100 on the
    subset).
  - The co-trained head was the worse head on its own encoder's embeddings in all 4 runs, though
    its rows barely changed direction (mean cos ≥ 0.99 to V3's).
  - Reading: in recipe A the head over-fits the training images. This reading comes from a
    post-hoc diagnostic, not from a declared test.
- **Recipe A is closed** (2026-11-20). Job 2 finished with the old config, so A's three-point grid
  is complete. **Its edge step (job 3) is dropped: a decision made after seeing dev.** The edge
  rule asked for it; it is not run. Jobs 3–7 of recipe A never run. Recipe A is reported as a
  negative dev result; none of its models goes to test. The decomposition above covers jobs 0
  and 1 only; it was not extended to job 2.

**Recipe B** (`configs/v4_predev_v2.json`, new run key):
- **Chosen after seeing recipe A's dev results**, because the A encoder × V3 head result above
  suggests the head update, not the encoder update, is what lost GeoScore.
- **Identical to recipe A except `train.freeze_head = true`:** the head is the V3 bal
  `cell_zeroshot` head of the same N and gets no gradient. Only LoRA trains (r16, q,k,v,out_proj,
  last 12 blocks).
  - Its sha256 is in the run key and in `result.json`; `weights_final.pt` holds the adapters only.
  - `head_lr_mult` is removed from the config.
- **Unchanged from A:** inputs, schedule, augmentation and its RNG, the lr grid {1e-4, 3e-4,
  1e-3}, the edge and stopping rules, the 5 GeoScore tie tolerance, the "keep" margin of 40,
  selection on all of dev_unseen at the final checkpoint, the GPU parity rule (decision 7), seeds
  42/43, and the job numbering 0–6.
- **Job names carry a `B_` prefix**, so run ids and paths in the runs repo never collide with
  recipe A's.
  - `v4_select` reads only results whose job is declared in the config it is given.
  - The trainer refuses a finished run directory whose run key differs from the current one.
  - A recipe B session never seeds from a recipe A output.
  - The selection file is `configs/v4_selection_v2.json`:
    `uv run python -m geoguessr_ai.v4_select --runs <B output> --config configs/v4_predev_v2.json --out configs/v4_selection_v2.json`.
- **What B can still answer:** the gap question, with the head held at each backbone's own V3
  head. "Fine-tuned vs frozen" becomes "adapted encoder vs V3 encoder, same head".
- **What B cannot answer:**
  - It is not LP-FT.
  - Its lr choice inherits recipe A's evidence. The grid is the same, but the recipe was picked
    knowing that A lost GeoScore.

**Recipe B short runs** (50K, 100K images seen, head frozen in all 8 runs: `head_frozen` true,
head sha256 the same per backbone in every job, `weights_final.pt` holds LoRA tensors only).
Session 1: jobs 0–2, commit c0aa755 (351 min, exit 0). Session 2: job 3, the edge step that
`v4_select` asked for, commit 723c04b (130 min, exit 0; seeded from session 1's output):

| dev GeoScore (top-1 cell centroid) | StreetCLIP dev_unseen | L/14-336 dev_unseen | StreetCLIP dev_seen | L/14-336 dev_seen |
|---|---|---|---|---|
| V3 frozen `cell_zeroshot`, bal 50K (`results/v3/phase3_dev_predictors.csv`) | 3372 [3237, 3587] | 3215 [2861, 3460] | 3586 | 3447 |
| B, job 3: lr 3.33e-5 (edge) | 3250 [3085, 3472] | 3042 [2624, 3322] | 3509 | 3224 |
| B, job 0: lr 1e-4 | 3225 [3054, 3445] | 3036 [2631, 3310] | 3506 | 3254 |
| B, job 1: lr 3e-4 | 3168 [2967, 3410] | 3022 [2635, 3301] | 3479 | 3277 |
| B, job 2: lr 1e-3 | 3094 [2865, 3357] | 2959 [2589, 3247] | 3461 | 3277 |

- **Selection** (`configs/v4_selection_v2.json`, extended grid): both backbones pick 3.33e-5, the
  new low edge. Per the stopping rule the grid is frozen and both are flagged `grid_edge`:
  - StreetCLIP margin to 1e-4: +25.1;
  - L/14-336 margin to 1e-4: +5.7, only 0.7 above the 5 GeoScore tie tolerance. Had it been
    within 5, 1e-4 would not have been kept either: the lower lr wins a tie.
  - The matched lr (sensitivity only) is also 3.33e-5.
- Every B short run is below the frozen V3 starting point on dev_unseen and dev_seen. dev_unseen
  rises as the lr falls on both backbones.
- B-S2: this short-run choice is final for the full runs (jobs 4–6).

**The control (A's job 7) under B: dropped, not replaced.**
- A's control trained the head on the frozen encoder with the same augmented schedule. Its
  purpose was to separate encoder adaptation from "augmentation plus more head training".
- Under B, no head training remains to separate out. The B = 0 model is exactly the V3 frozen
  model (identity check), so B's fine-tuned vs frozen contrast already isolates the adapter
  update.
- A head-only control would have nothing to train, so the trainer refuses it.
- This saves about 5.5 quota h.
- **Limitation:** B's contrast is "the adapter update **under this augmented schedule**" vs the
  V3 encoder, not encoder adaptation as such. The augmentation (crops of 0.7–1.0 of the short
  side, so objects about 1.15× larger than in the eval center crop, plus jitter, blur and
  erasing) acts only through the adapters, and nothing in B separates its effect. A B result in
  either direction may be partly the augmentation's. Separating it would need a non-augmented
  LoRA run (at least one short pair, 2.4 h); it is not declared and is not in the budget.

**Budget, recipe B** (per-job hours from `uv run python scripts/v4_budget.py --rate 15.4`; its
option totals still add the spent 1.5 h pilot, which the table below leaves out):
- **Rate:** 15.4 img/s, the slower GPU's sustained train rate in session 1 (`log.jsonl`
  `img_per_s`; the pilot measured 16.0).
  - Freezing the head saves nothing measurable here: the head's gradient is negligible next to
    the LoRA backward pass.
  - Session 1 took 274 min for two short pairs and session 2 took 130 min for one, against
    2.43 h per short pair budgeted, so the estimate is slightly conservative.
- Per pair, setup included: short 2.43 h, full 9.74 h, job 4 with its diagnostic 9.91 h, 50K
  hybrid 0.72 h, test pass (2 models) 2.24 h.
- **This week's sessions are not counted**: the pilot, session 1 and session 2 (recipe A) are
  spent. The 60 h are the next two Kaggle weeks, both starting at 30 h.

| item | quota h |
|---|---|
| B short runs, jobs 0–2 | 7.3 |
| B edge job 3 (only if `v4_select` asks for it) | 2.4 |
| B jobs 4, 5, 6 (50K full + diagnostic, 500K full, 50K seed 2) | 29.4 |
| 50K hybrid re-embedding | 0.7 |
| test pass, 4 models | 4.5 |
| **total** | **44.3** (41.9 without the edge) |
| + one lost session of the longest job (reserve) | **54.2** (51.8) |

- **It fits 60 h in total**, with 5.8 h spare beyond the reserve (8.2 h without the edge).
  Break-even for the total: about 13.6 img/s.
- **The weekly 30 h cap binds before the total does** (strict chain; a full pair fills a
  session):
  - week 1 holds B jobs 0–2 (+ job 3), job 4 and job 5: 29.4 h with the edge step, 27.0 h
    without;
  - week 2 holds job 6, the hybrid and the test pass, 14.9 h, plus the full 9.9 h reserve:
    24.8 h.
  - With the edge step, week 1 has only 0.6 h slack; it holds at ≥ 15.0 img/s. Below that, or
    after any lost session in week 1, job 5 moves to week 2, which then holds 24.7 h of jobs and
    only 5.3 h of reserve.
- **Order of cuts (as declared in section 3): the reserve first, then the descriptive seed-2 run
  (job 6, 9.7 h).**
  - If job 5 has moved to week 2, one lost full session there means job 6 is cut.
  - Nothing else is cut without a new decision. The next candidate is moving the test pass into a
    third week, not shrinking N.
- The 500K arm and the test pass are never cut silently.

## 6. Open risks

- **B-S2 has no slack beyond the reserve.** r = 16.0 is only 0.2 img/s above the recomputed
  break-even.
  - The rate comes from a 3,000-image run. The 87-minute clock log was steady, but a 9 h run on
    another machine may differ.
  - Below about 15.8 img/s the plan exceeds 60 h. The order of cuts is declared: the reserve
    first, then the descriptive seed-2 run.
- **Recipe B was chosen after seeing recipe A's dev results** (section 5c). The lr grid and rules
  are unchanged, but the recipe itself is a post-hoc choice. It is reported as such, and the
  V4 test pre-registration has to say so.
- **Last-12 adapts only the upper half** (section 4b): it is biased toward "keep" if
  StreetCLIP's lead sits in the lower blocks.
- **The short-horizon lr search** (100K seen) tends to pick a higher lr than the full schedule
  would.
- **GPU speed differs by card** (13–17% here): budgeted at the slower card and counterbalanced.
  The CPU decode headroom was ample in the pilot.
- **`/kaggle/temp`** has no separate mount: it is a directory on the root overlay, 965 GB free
  in the pilot. 28 GB of staging is still untested. **HF throttling** of about 100–150 GB of
  private downloads is unmeasured; the pilot saw none at 101–286 MB/s.
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
