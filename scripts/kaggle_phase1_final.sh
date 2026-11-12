#!/usr/bin/env bash
# The one-shot final evaluation on Kaggle (phase1_eval --eval-test; see geoguessr_ai/prereg.py).
# Inputs attached to the notebook (internet on, for the CLIP text encoders of the zero-shot
# probe init; GPU recommended):
#   the full run's output        v2/ + emb/ (train embeddings)    INPUT_DIR overrides the search
#   kaggle_phase1.sh test output emb_test/ + its provenance       TEST_EMB_DIR overrides (the emb_test dir)
# No earlier phase1_eval output is needed (cells are retrained with the pre-registered
# hyperparameters; nothing is seeded). The pre-registration is this clone's
# configs/phase1_final.json; phase1_eval refuses unless it is committed, unmodified and the one
# recorded by the test embedding run. Output: /kaggle/working/phase1_eval_test.
#   usage: bash scripts/kaggle_phase1_final.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/find_input.sh"
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IN=${INPUT_DIR:-}
[ -n "$IN" ] || IN=$(find_input /kaggle/input) || { echo "attach the full run's output or set INPUT_DIR"; exit 1; }
TEST=${TEST_EMB_DIR:-}
[ -n "$TEST" ] || TEST=$(find_input /kaggle/input emb_test/prereg_provenance.json)/emb_test \
  || { echo "attach the test embedding run's output or set TEST_EMB_DIR"; exit 1; }
for d in "$IN/v2/test.csv" "$IN/emb/streetclip" "$IN/emb/clip_l14_336" "$IN/emb/clip_b16" \
         "$TEST/prereg_provenance.json" "$TEST/streetclip" "$TEST/clip_l14_336" "$TEST/clip_b16"; do
  [ -e "$d" ] || { echo "FAILED: $d missing"; exit 1; }
done
echo "inputs: $IN (v2, emb); $TEST (test embeddings)"
OUT=/kaggle/working/phase1_eval  # phase1_eval writes to ${OUT}_test
mkdir -p "${OUT}_test"
nvidia-smi --query-gpu=name --format=csv || echo "no GPU: CPU fallback"
cd "$REPO_DIR"
python -m geoguessr_ai.phase1_eval --eval-test --data "$IN/v2" --emb-root "$IN/emb" --out "$OUT" \
  --prereg configs/phase1_final.json --test-emb-root "$TEST" 2>&1 | tee -a "${OUT}_test/phase1_final.log"
ls -la "${OUT}_test"
