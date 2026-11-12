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
export PYTHONUNBUFFERED=1  # live Kaggle logs (python output through tee)
. "$(dirname "${BASH_SOURCE[0]}")/find_input.sh"
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
K=${KAGGLE_ROOT:-/kaggle}  # only tests set KAGGLE_ROOT
IN=${INPUT_DIR:-}
[ -n "$IN" ] || IN=$(find_input "$K/input") || { echo "attach the full run's output or set INPUT_DIR"; exit 1; }
TEST=${TEST_EMB_DIR:-}
[ -n "$TEST" ] || TEST=$(find_input "$K/input" emb_test/prereg_provenance.json)/emb_test \
  || { echo "attach the test embedding run's output or set TEST_EMB_DIR"; exit 1; }
for d in "$IN/v2/test.csv" "$IN/emb/streetclip" "$IN/emb/clip_l14_336" "$IN/emb/clip_b16" \
         "$TEST/prereg_provenance.json" "$TEST/zip_index" "$TEST/streetclip" "$TEST/clip_l14_336" \
         "$TEST/clip_b16"; do
  [ -e "$d" ] || { echo "FAILED: $d missing"; exit 1; }
done
echo "inputs: $IN (v2, emb); $TEST (test embeddings)"
OUT=$K/working/phase1_eval  # phase1_eval writes to ${OUT}_test
mkdir -p "${OUT}_test"

# Before any evaluation: every test id embedded exactly once in each backbone's cache. The check
# reads the per-zip indexes the test run kept in emb_test/zip_index (--zip-dir only selects that
# mode; no zip is read). Refuses unless expected = unique = TEST_IMAGES and duplicates = missing
# = failed = extra = temp files = 0.
TEST_IMAGES=${TEST_IMAGES:-210122}
echo "=== checks: test embeddings, per backbone (expected $TEST_IMAGES), before any evaluation"
for m in streetclip clip_l14_336 clip_b16; do
  c=${OUT}_test/check_$m.json
  python -m geoguessr_ai.embed --check --ids "$IN/v2/test.csv" --source test --out "$TEST/$m" \
    --zip-dir "$K/temp/zips" --index-dir "$TEST/zip_index" --shards 0 1 2 3 4 > "$c" \
    || true  # exit 1 on a failed check: judged below, with every count printed
  python - "$c" "$TEST_IMAGES" "$m" <<'EOF' || { echo "FAILED: test embeddings of $m: no evaluation"; exit 1; }
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    sys.exit(f"{sys.argv[3]}: no check result ({e})")
want = int(sys.argv[2])
ok = (r["ok"] and r["expected"] == r["unique"] == want and r["duplicates"] == r["missing"]
      == r["failed"] == r["extra"] == r["temp_files_left"] == 0)
print(f"{sys.argv[3]}: expected {r['expected']} unique {r['unique']} duplicates {r['duplicates']} "
      f"missing {r['missing']} failed {r['failed']} extra {r['extra']} temp files "
      f"{r['temp_files_left']} -> {'PASSED' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
EOF
done

nvidia-smi --query-gpu=name --format=csv || echo "no GPU: CPU fallback"
cd "$REPO_DIR"
python -m geoguessr_ai.phase1_eval --eval-test --data "$IN/v2" --emb-root "$IN/emb" --out "$OUT" \
  --prereg configs/phase1_final.json --test-emb-root "$TEST" 2>&1 | tee -a "${OUT}_test/phase1_final.log"
ls -la "${OUT}_test"
