#!/usr/bin/env bash
# Phase 3 (development splits only): kNN coordinate prediction on Kaggle, the full run's output
# attached as notebook input. No GPU needed (a CPU-only session uses no GPU quota): kNN runs in
# numpy. The official test split is never read (phase3_eval has no test option).
#   usage: bash scripts/kaggle_phase3.sh [INPUT_DIR]   (or the INPUT_DIR env var; default: the
#          one dir under /kaggle/input holding v2/config.json). Rerunning skips finished cells.
#   PHASE1_DEV=<a Phase 1 development phase1_eval dir> also checks that geo_medoid reproduces
#   its kNN dev coordinates (a dir with test results is refused).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/find_input.sh"
IN=${1:-${INPUT_DIR:-}}
[ -n "$IN" ] || IN=$(find_input /kaggle/input) || { echo "set INPUT_DIR"; exit 1; }
OUT=/kaggle/working/phase3_eval
for d in "$IN/v2" "$IN/emb/streetclip" "$IN/emb/clip_l14_336" "$IN/emb/clip_b16"; do
  [ -e "$d" ] || { echo "FAILED: $d missing (attach the full run's output)"; exit 1; }
done
CMP=()
[ -z "${PHASE1_DEV:-}" ] || CMP=(--compare-phase1 "$PHASE1_DEV")
nvidia-smi --query-gpu=name --format=csv || echo "no GPU: CPU (numpy kNN)"
nproc; free -g || true
mkdir -p "$OUT"
python -m geoguessr_ai.phase3_eval --data "$IN/v2" --emb-root "$IN/emb" --out "$OUT" "${CMP[@]}" \
  2>&1 | tee -a "$OUT/phase3_eval.log"
ls -la "$OUT"
