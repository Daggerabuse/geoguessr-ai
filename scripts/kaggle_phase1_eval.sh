#!/usr/bin/env bash
# Phase 1 analysis on Kaggle: the full run's output attached as notebook input (internet on, for
# the CLIP text encoders that initialize the probe; GPU optional: without one kNN runs in numpy
# and the probe on the CPU, several times slower). Dev splits only; never --eval-test here.
#   usage: bash scripts/kaggle_phase1_eval.sh [INPUT_DIR]   (default: the /kaggle/input/* dir
#          holding v2/config.json). Rerunning skips finished cells (cache in OUT).
set -euo pipefail
IN=${1:-$(dirname "$(dirname "$(ls /kaggle/input/*/v2/config.json | head -n 1)")")}
OUT=/kaggle/working/phase1_eval
for d in "$IN/v2" "$IN/emb/streetclip" "$IN/emb/clip_l14_336" "$IN/emb/clip_b16"; do
  [ -e "$d" ] || { echo "FAILED: $d missing (attach the full run's output)"; exit 1; }
done
nvidia-smi --query-gpu=name --format=csv || echo "no GPU: CPU fallback"
mkdir -p "$OUT"
python -m geoguessr_ai.phase1_eval --data "$IN/v2" --emb-root "$IN/emb" --out "$OUT" \
  2>&1 | tee -a "$OUT/phase1_eval.log"
ls -la "$OUT"
