# Phase 3 dev summary (coordinates, development splits only)

This file is input for the Phase 3 preregistration. **The test split has not been opened**: no
test km/GeoScore, no `runs/phase1_final`, no test embeddings. It holds facts only. It makes no
recommendation about test hypotheses.

- Sources:
  - `runs/phase3_final_dev/phase3_eval`, merged by `scripts/analysis/phase3_merge.py`: random cell
    probes from session C, zero-shot cell probes after the 3e-4 extension, and all 48 hybrids;
  - `runs/phase3_dev/phase3_eval` for the Step 1 kNN `geo_medoid`.
- Tables: `results/v3/phase3_*.csv`, written by `scripts/analysis/phase3_cell_summary.py`.
- Intervals: 95% paired cluster bootstrap with n_boot 1000 and seed 42, as in phase3_eval.
  Clusters are sequences on dev_seen (19,906 images) and creators on dev_unseen (20,000 images).
- "Gap" is the mean per-image difference in GeoScore (0–5000) unless it says median km. Brackets
  are the 95% interval.

Predictors:
- `medoid`: masked kNN geographic medoid (Step 1);
- `cell_r` / `cell_z`: geocell probe (5000 cells), random / zero-shot init;
- `hyb_r` / `hyb_z`: hybrid, i.e. masked kNN top-1000 inside the probe's top-m cells, medoid.

## Merge verification
- 96 cell dirs (3 backbones × nat/bal × 4 N × {cell, hybrid} × {random, zero-shot}) were copied
  byte for byte. `cell_count.json` and `cells_5000.json` are identical in all 4 source sessions.
- Every cell probe's `cache_key` equals the key recomputed from its stored settings. Every
  hybrid's key equals the key recomputed with the **merged** probe's source (cell key plus
  top-cells sha256): 48/48 hybrids were built from the final probe.
  - Negative check: against session C's pre-extension zero-shot probe, the key does not match.
- The stored settings equal the current code and defaults:
  - CACHE_VERSION 2;
  - zero-shot `extra_lrs = [3e-4]`, random none;
  - lr grid, 60 epochs, patience 8;
  - mask 1 km, CPU fp32 kNN for the hybrid;
  - k grid, m grid, hybrid_k 1000, medoid.
- The geocell sha256 is `9882eb2a…` in every cell, every hybrid and the partition file.
- Per backbone, the data config and embedding-cache hashes are identical to the Step 1 kNN cells.
- The 24 zero-shot rows equal the extension session's `results.csv`. Each row's GeoScore and
  median km equal its hits file.
- Limitation: the data dir and embeddings are not local. Their hashes are checked for consistency
  across cells, not recomputed from the files.

## 1. Best predictor
- By dev_seen GeoScore, the best predictor is a hybrid in **24/24** backbone × ordering × N cells.
  - `hyb_z` is best in 20 cells and `hyb_r` in 4 (B/16 nat 200K, L/14 nat 200K and 500K,
    StreetCLIP nat 100K).
- Best vs runner-up:
  - bal, N ≤ 200K: the margin is clear on dev_seen, +21 to +106.
  - nat at N ≥ 100K and every 500K cell: the dev_seen interval includes 0. The two hybrids are
    then within ~10 of each other.
- Best vs `medoid` (`phase3_cell_best.csv`):
  - dev_seen: +41 to +231, every interval above 0.
  - dev_unseen: +22 to +314. The interval includes 0 in 6/24 cells: L/14 bal 50K; StreetCLIP bal
    100K/200K/500K, nat 200K/500K.
  - StreetCLIP: +41 to +97 on dev_seen, smaller than for L/14 (+72 to +108) and B/16 (+150 to
    +231).
- Absolute values, StreetCLIP nat 500K (`phase3_dev_predictors.csv`):

  | predictor | dev_seen GeoScore | dev_seen median km | dev_unseen GeoScore | dev_unseen median km |
  |---|---|---|---|---|
  | hyb_z | 4351 | 43 | 3785 | 296 |
  | medoid | 4256 | 78 | 3763 | 310 |

## 2. StreetCLIP − CLIP L/14-336 by predictor and init (`phase3_cell_backbone_gap.csv`)

GeoScore gap, N = 50K:

| | medoid | cell_r | cell_z | hyb_r | hyb_z |
|---|---|---|---|---|---|
| nat, dev_seen | 47 [36,59] | 56 [43,70] | **9 [−3,22]** | 37 [25,50] | 15 [3,27] |
| nat, dev_unseen | 125 [79,181] | 46 [−6,72] | **−15 [−57,7]** | 53 [17,79] | 50 [13,74] |
| bal, dev_seen | 135 [121,151] | 201 [181,219] | 139 [122,156] | 161 [143,178] | 113 [98,129] |
| bal, dev_unseen | 189 [124,273] | 185 [110,333] | 156 [12,496] | 188 [107,346] | 147 [36,400] |

GeoScore gap, N = 500K:

| | medoid | cell_r | cell_z | hyb_r | hyb_z |
|---|---|---|---|---|---|
| nat, dev_seen | 31 [20,42] | 23 [12,35] | 18 [6,29] | 17 [5,29] | 28 [17,40] |
| nat, dev_unseen | 82 [40,105] | 45 [30,76] | 18 [4,34] | **4 [−22,22]** | **5 [−12,19]** |
| bal, dev_seen | 64 [52,77] | 65 [52,80] | 50 [37,64] | 39 [27,52] | 36 [23,49] |
| bal, dev_unseen | 143 [99,172] | 13 [−26,87] | 56 [18,79] | 45 [19,80] | 47 [23,68] |

- Zero-shot init shrinks the nat 50K cell-probe gap from 56 to 9 on dev_seen; the interval
  includes 0. On dev_unseen it goes from 46 to −15. The reason: zero-shot helps L/14 more than
  StreetCLIP. Zero-shot − random at nat 50K, dev_seen, cell probe: StreetCLIP +48 [36,59],
  L/14 +95 [81,109].
- Interval includes 0 (StreetCLIP − L/14):
  - dev_seen: only cell_z nat 50K.
  - dev_unseen: cell_r nat 50K, cell_z nat 50K, cell_z nat 200K, cell_z bal 100K/200K,
    cell_r bal 500K, hyb_r and hyb_z nat 500K.
- The medoid has the largest dev_unseen gap of the five predictors in all 8 subsets. At bal 50K
  it is only just largest: 189 vs hyb_r 188. For the learned predictors the dev_unseen gap at
  nat 500K is 4–45.
- Median km gap (negative = StreetCLIP closer), nat 50K dev_seen: medoid −12 [−17,−8],
  cell_r −22 [−27,−17], cell_z −6 [−10,−3], hyb_r −16 [−21,−12], hyb_z −8 [−12,−5].
  - bal 50K dev_seen: −30 to −66 km.
  - Most dev_unseen median-km intervals exclude 0 but are wide.

**StreetCLIP − CLIP B/16:**
- Large and positive everywhere: GeoScore 408–888 on dev_seen, 579–1059 on dev_unseen.
- Every interval excludes 0.
- Median km −58 to −607.

## 3. Shrink 50K → 500K of the GeoScore gap (`phase3_backbone_shrink.csv`)

Positive = the gap is smaller at 500K. vs L/14-336:

| | medoid | cell_r | cell_z | hyb_r | hyb_z |
|---|---|---|---|---|---|
| nat, dev_seen | 16 [3,30] | 33 [16,51] | −9 [−24,8] | 19 [3,37] | −14 [−30,2] |
| nat, dev_unseen | 43 [18,98] | 1 [−66,37] | **−33 [−72,−15]** | 49 [24,78] | 45 [17,63] |
| bal, dev_seen | 71 [53,90] | 136 [111,157] | 89 [68,110] | 122 [100,141] | 77 [58,96] |
| bal, dev_unseen | 46 [−5,142] | 172 [74,271] | 100 [−44,427] | 143 [56,293] | 100 [−7,347] |

- nat, zero-shot init: the gap does **not** shrink on dev_seen.
  - cell_z goes from 9 to 18 and hyb_z from 15 to 28; the intervals include 0.
  - cell_z on dev_unseen goes from −15 to 18, so the gap grows: −33 [−72,−15].
- vs B/16, the gap shrinks in every predictor on dev_seen:
  - nat: 75–143;
  - bal: 153–304;
  - all intervals are above 0.
- On dev_unseen, the B/16 intervals include 0 for medoid (bal and nat), cell_r bal, cell_z (bal
  and nat) and hyb_z bal.

## 4. Zero-shot − random init (`phase3_cell_init_gap.csv`)
- Cell probe: +3 to +355 on dev_seen and −5 to +373 on dev_unseen.
  - The interval is above 0 in 20/24 (dev_seen) and 21/24 (dev_unseen).
  - It is largest at bal 50K: B/16 +355, L/14 +345, StreetCLIP +283 (dev_seen).
  - At 500K it is ≈ 0 to +35.
- Hybrid: −8 to +248 on dev_seen and −12 to +284 on dev_unseen; the interval is above 0 in 12/24
  and 18/24. At nat N ≥ 100K, dev_seen, every hybrid interval includes 0.
- No zero-shot − random interval lies entirely below 0 (0/96).
- At N ≤ 100K, StreetCLIP benefits less than L/14 in all 8 cell/hybrid × ordering × N
  comparisons, on dev_seen. It benefits least of the three backbones in 7 of the 8; the
  exception is the cell probe at bal 100K, where B/16 gains +193 and StreetCLIP +225. This is
  why zero-shot init narrows the StreetCLIP − L/14 gap.

## 5. dev_seen vs dev_unseen
- dev_unseen GeoScore is ~290–740 lower than dev_seen. Median km is much larger: e.g.
  StreetCLIP nat 500K hyb_z, 43 km on dev_seen vs 296 km on dev_unseen.
- dev_unseen intervals are wider, because the creator clusters are fewer and larger:
  - bal 50K StreetCLIP − L/14 cell_z: [12,496];
  - B/16 dev_unseen median-km intervals span hundreds of km.
- The StreetCLIP − L/14 gap is larger on dev_unseen for the medoid at every subset: nat 50K,
  125 vs 47. For the cell and hybrid predictors the two splits are closer, and at nat 500K the
  dev_unseen hybrid gaps (4, 5) are smaller than on dev_seen (17, 28).
- Best − medoid is larger on dev_unseen than on dev_seen in 14/16 L/14 and B/16 cells (not L/14
  nat 500K or B/16 bal 500K). For StreetCLIP it is larger in only 4/8.

## Limitations
- **lr grid_edge (cell probes).**
  - Zero-shot: 13/24 picked the added edge 3e-4.
  - Random: 7/24 picked the lower edge 1e-3. Random has no 3e-4 run, so its margin is vs 3e-3.
  - Per the stopping rule the grid is frozen; these cells are a limitation.
  - Margin GS(pick) − GS(neighbour), dev_seen, from `phase3_cell_flags.csv`:
    - zero-shot (3e-4 − 1e-3):
      - nat: L/14 50K 3.4, 100K 2.2, 200K 3.7, 500K 4.9; B/16 50K 8.4, 100K 2.3, 200K 5.4.
      - bal: StreetCLIP 200K 63.9; L/14 200K 71.0, 500K 0.5; B/16 50K 33.1, 100K 7.0,
        200K 17.8.
    - random (1e-3 − 3e-3):
      - StreetCLIP: nat 500K 2.7; bal 50K 95.2, 100K 67.7, 200K 76.5.
      - L/14: nat 50K 22.1; bal 200K 67.5.
      - B/16: bal 50K 60.3.
  - The margins are large (17–95) mostly at bal. They are small (<9) at nat and at L/14 bal 500K.
- **Epoch cap.**
  - 4 cells picked epoch 60, the cap; all 4 are also grid_edge:
    - StreetCLIP nat 500K random;
    - L/14 nat 500K zero-shot;
    - B/16 nat 50K and 100K zero-shot.
  - More broadly, the selected lr's run never early-stopped in 20/48 cells (8 random, 12
    zero-shot). The best epoch was within patience of the cap.
- **Hybrid grid edges.** Grids are m ∈ {1,3,5}, k ∈ {1,5,…,100}; tau is not searched by the
  medoid.
  - m = 5 (the upper edge, a real edge) was picked in 35/48 hybrids. Margin vs m = 3: 1.1–58.0,
    largest at bal 50K.
  - k = 1 (a natural lower bound, not a grid limitation) was picked in 27/48. Margin vs k = 5:
    2.1–180.5.
  - No hybrid picked k = 100. Example: B/16 nat 50K random picked m = 5, k = 10, so only m is at an
    edge.
  - m is **not** extended (docs/V2_PIPELINE.md, 2026-11-15). The probes saved only their top-5
    cells and no weights, so m > 5 cannot be computed without a GPU retrain. That was decided
    after these counts were seen.
  - The m → all-cells limit is Step 1's `geo_medoid`, by construction. It scores below m = 5 in
    30 of the 35 m = 5 picks. It scores above m = 5 in 5: random init, bal, N ≤ 100K (B/16 50K;
    L/14 50K and 100K; StreetCLIP 50K and 100K). There the dev_seen optimum over m lies above 5.
- **Probe weights were not saved.** A test evaluation must retrain each cell probe at its frozen
  lr, epoch and seed. It must pass a reproduction check against the stored dev_seen top-5 and
  GeoScore before test is scored, because GPU training may not be bit-reproducible. Estimated
  cost: ≈ 3 GPU h + 2–3.5 CPU h (docs/V2_PIPELINE.md, test-phase requirements).
- **Masking is not like for like.** The kNN parts (medoid, and the hybrid's neighbour search) mask
  training images within 1 km of each dev query. The cell probe trains on every subset image. The
  hybrid mixes both. Backbone gaps within one predictor are unaffected, because the masking is
  identical for both backbones. Comparisons between predictors are not like for like.
- **Rule timing.**
  - The lr-edge rule (threshold 8/24) was written after partially seeing live Kaggle logs, before
    the counts were computed (see docs/V2_PIPELINE.md).
  - The hybrid's medoid predictor was chosen after seeing on dev that geo_mean degenerated to
    k = 1.
  - The cell count (5000) was chosen on dev_seen with random init only, at nat 500K, using
    StreetCLIP and L/14.
- **Selection and scoring use the same split.** All selections (lr, epoch, k, m, cell count, best
  predictor) were made on dev_seen, and dev_seen numbers are reported on the same split, so they
  are optimistic. dev_unseen is not used for any selection.
- **Verification** of the data and embedding hashes is consistency-only (see above).
- **Dev only.** The intervals are uncorrected 95% intervals over many comparisons. Nothing here
  is a test result.
