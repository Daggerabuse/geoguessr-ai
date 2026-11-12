#!/usr/bin/env bash
# Phase 3 (development splits only) on Kaggle. Step 1 (default, METHODS=knn): kNN coordinate
# prediction in a CPU session (no GPU quota), numpy fp32 (--device '' --knn-dtype fp32). Step 2
# (METHODS=cell,hybrid): geocell probe and hybrid in a GPU session, kNN in fp32 on cuda. The kNN
# setting is recorded per cell; any later test run must use the same one. The official test
# split is never read by phase3_eval (it has no test option).
#   usage: bash scripts/kaggle_phase3.sh [INPUT_DIR]   (or the INPUT_DIR env var; default: the
#          one dir under /kaggle/input holding v2/config.json). Rerunning skips finished cells.
#   PHASE1_DEV  Phase 1 development phase1_eval dir (default: found by phase1_eval/results.csv
#               under /kaggle/input, if attached): probe countries for geo_centroid.
#   MASK_KM     near-duplicate mask radius in km (default 1).
#   METHODS     comma-separated phase3_eval methods (default knn: CPU session). cell,hybrid =
#               Step 2 (geocell probe + hybrid): GPU T4 session with internet on (the zero-shot
#               init loads the CLIP text encoders); probe and kNN on cuda, kNN still fp32.
#   TEST_EMB_DIR  the emb_test dir of the test embedding run (default: found by its
#               prereg_provenance.json, if attached). With PHASE1_DEV, a last exploratory step
#               recomputes Phase 1's hyperparameter-matched kNN cells on the CPU in fp32
#               (scripts/analysis/phase1_posthoc.py): predicted country and top-1 hit per test
#               image only, no coordinate or distance, and the agreement with Phase 1's saved
#               GPU fp16 dev predictions.
set -euo pipefail
export PYTHONUNBUFFERED=1  # live Kaggle logs (python output through tee)
. "$(dirname "${BASH_SOURCE[0]}")/find_input.sh"
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IN=${1:-${INPUT_DIR:-}}
[ -n "$IN" ] || IN=$(find_input /kaggle/input) || { echo "set INPUT_DIR"; exit 1; }
OUT=/kaggle/working/phase3_eval
for d in "$IN/v2" "$IN/emb/streetclip" "$IN/emb/clip_l14_336" "$IN/emb/clip_b16"; do
  [ -e "$d" ] || { echo "FAILED: $d missing (attach the full run's output)"; exit 1; }
done
P1=${PHASE1_DEV:-}
if [ -z "$P1" ] && P=$(find_input /kaggle/input phase1_eval/results.csv 2>/dev/null); then
  P1=$P/phase1_eval
fi
TEST=${TEST_EMB_DIR:-}
if [ -z "$TEST" ] && P=$(find_input /kaggle/input emb_test/prereg_provenance.json 2>/dev/null); then
  TEST=$P/emb_test
fi
echo "inputs: $IN; Phase 1 dev: ${P1:-none}; test embeddings: ${TEST:-none}"
P1ARG=()
[ -z "$P1" ] || P1ARG=(--phase1-dev "$P1")
METHODS=${METHODS:-knn}
DEVICE=""  # knn only: numpy fp32 on the CPU (Step 1)
if [ "$METHODS" != knn ]; then
  DEVICE=cuda
  nvidia-smi --query-gpu=name --format=csv \
    || { echo "FAILED: METHODS=$METHODS needs a GPU session"; exit 1; }
else
  nvidia-smi --query-gpu=name --format=csv || echo "no GPU (not needed)"
fi
nproc; free -g || true
mkdir -p "$OUT"
python -m geoguessr_ai.phase3_eval --data "$IN/v2" --emb-root "$IN/emb" --out "$OUT" \
  --methods ${METHODS//,/ } --device "$DEVICE" --knn-dtype fp32 --mask-km "${MASK_KM:-1}" \
  "${P1ARG[@]}" \
  2>&1 | tee -a "$OUT/phase3_eval.log"

if [ "$METHODS" = knn ] && [ -n "$P1" ] && [ -n "$TEST" ]; then
  echo "=== exploratory: Phase 1 matched-k kNN on the CPU (fp32), top-1 only"
  mkdir -p "$OUT/phase1_matched_k"
  (cd "$REPO_DIR" && python scripts/analysis/phase1_posthoc.py --dev-run "$P1" --data "$IN/v2" \
    --emb-root "$IN/emb" --test-emb-root "$TEST" --out "$OUT/phase1_matched_k") \
    2>&1 | tee -a "$OUT/phase1_matched_k/phase1_matched_k.log"
fi
ls -la "$OUT"
