#!/usr/bin/env bash
# Phase 1 analysis on Kaggle: the full run's output attached as notebook input (internet on, for
# the CLIP text encoders that initialize the probe; GPU optional: without one kNN runs in numpy
# and the probe on the CPU, several times slower). Dev splits only; never --eval-test here.
#   usage: bash scripts/kaggle_phase1_eval.sh [INPUT_DIR]   (or the INPUT_DIR env var; default: the
#          one dir under /kaggle/input holding v2/config.json). Rerunning skips finished cells (cache in OUT).
#   An earlier phase1_eval output attached as input (the one */phase1_eval/results.csv under
#   /kaggle/input, or SEED_FROM=<its phase1_eval dir>) seeds OUT with its still-valid cells;
#   SEED_FROM=none runs without it.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/find_input.sh"
IN=${1:-${INPUT_DIR:-}}
[ -n "$IN" ] || IN=$(find_input /kaggle/input) || { echo "set INPUT_DIR"; exit 1; }
SEED=${SEED_FROM:-}
[ -n "$SEED" ] || SEED=$(find_input /kaggle/input phase1_eval/results.csv)/phase1_eval \
  || { echo "attach the previous phase1_eval output, or set SEED_FROM (none = no seeding)"; exit 1; }
OUT=/kaggle/working/phase1_eval
for d in "$IN/v2" "$IN/emb/streetclip" "$IN/emb/clip_l14_336" "$IN/emb/clip_b16"; do
  [ -e "$d" ] || { echo "FAILED: $d missing (attach the full run's output)"; exit 1; }
done
SEED_ARGS=()
[ "$SEED" = none ] || SEED_ARGS=(--seed-from "$SEED")
nvidia-smi --query-gpu=name --format=csv || echo "no GPU: CPU fallback"
mkdir -p "$OUT"
python -m geoguessr_ai.phase1_eval --data "$IN/v2" --emb-root "$IN/emb" --out "$OUT" "${SEED_ARGS[@]}" \
  2>&1 | tee -a "$OUT/phase1_eval.log"
ls -la "$OUT"
