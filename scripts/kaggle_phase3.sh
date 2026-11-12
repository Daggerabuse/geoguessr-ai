#!/usr/bin/env bash
# Phase 3 (development splits only) on Kaggle. Step 1 (METHODS=knn): kNN coordinate prediction
# in a CPU session, numpy fp32. Step 2: METHODS=cell trains the geocell probe (GPU session, the
# probe on cuda; saves the top-5 cells and probabilities per cell and split); METHODS=hybrid
# reads those and needs NO GPU (CPU session, numpy fp32 kNN as in Step 1; refuses if a cell
# probe is missing). The kNN setting is recorded per cell; any later test run must use the same
# one. The official test split is never read by phase3_eval (it has no test option).
#   usage: bash scripts/kaggle_phase3.sh [INPUT_DIR]   (or the INPUT_DIR env var; default: the
#          one dir under /kaggle/input holding v2/config.json). Rerunning skips finished cells.
#   METHODS     knn (default) | cell | hybrid | cell,hybrid. A GPU is required iff cell is in it
#               (device cuda); otherwise the kNN runs on the CPU (--device '').
#   BACKBONES, ORDERINGS, SIZES, PROBE_INITS   comma- or space-separated; passed through as
#               --backbones/--orderings/--sizes/--probe-inits (default: phase3_eval's).
#   DEADLINE_MIN  wall-clock minutes (may be fractional). At the deadline a STOP marker
#               ($OUT/STOP) makes phase3_eval finish the cell in hand and start no new one: it
#               prints RESUME NEEDED and this script exits 3. phase3_eval is killed only
#               GRACE_MIN (default 45) minutes later (RESUME NEEDED too; that cell is lost).
#   SEED_FROM   a finished phase3_eval output dir (e.g. the earlier session's, attached as an
#               input) copied into the empty output dir before running, to resume in a new
#               session. Refused: not a directory, the output dir itself, a non-empty output
#               dir, a dir that is not itself a phase3_eval output (nested ones are listed).
#   MASK_KM     near-duplicate mask radius in km (default 1).
#   PHASE1_DEV  Phase 1 development phase1_eval dir (default: found by phase1_eval/results.csv
#               under /kaggle/input, if attached): probe countries for geo_centroid.
#   TEST_EMB_DIR  the emb_test dir of the test embedding run (default: found by its
#               prereg_provenance.json, if attached). With PHASE1_DEV, METHODS=knn ends with
#               scripts/analysis/phase1_posthoc.py (predicted country and top-1 hit per test
#               image only, no coordinate or distance).
# On exit (any reason) the end time is printed and appended to $OUT/quota_times.txt.
set -euo pipefail
export PYTHONUNBUFFERED=1  # live Kaggle logs (python output through tee)
. "$(dirname "${BASH_SOURCE[0]}")/bg_helpers.sh"
. "$(dirname "${BASH_SOURCE[0]}")/find_input.sh"
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
K=${KAGGLE_ROOT:-/kaggle}  # only tests set KAGGLE_ROOT
fail() { echo "FAILED: $*"; exit 1; }
IN=${1:-${INPUT_DIR:-}}
[ -n "$IN" ] || IN=$(find_input "$K/input") || { echo "set INPUT_DIR"; exit 1; }
OUT=$K/working/phase3_eval
for d in "$IN/v2" "$IN/emb/streetclip" "$IN/emb/clip_l14_336" "$IN/emb/clip_b16"; do
  [ -e "$d" ] || fail "$d missing (attach the full run's output)"
done
P1=${PHASE1_DEV:-}
if [ -z "$P1" ] && P=$(find_input "$K/input" phase1_eval/results.csv 2>/dev/null); then
  P1=$P/phase1_eval
fi
TEST=${TEST_EMB_DIR:-}
if [ -z "$TEST" ] && P=$(find_input "$K/input" emb_test/prereg_provenance.json 2>/dev/null); then
  TEST=$P/emb_test
fi
echo "inputs: $IN; Phase 1 dev: ${P1:-none}; test embeddings: ${TEST:-none}"
P1ARG=()
[ -z "$P1" ] || P1ARG=(--phase1-dev "$P1")
EXTRA=()
pass() {  # pass FLAG VALUE: "a,b" or "a b" -> FLAG a b
  local -a v
  [ -z "${2:-}" ] || { read -ra v <<<"${2//,/ }"; EXTRA+=("$1" "${v[@]}"); }
}
pass --backbones "${BACKBONES:-}"; pass --orderings "${ORDERINGS:-}"; pass --sizes "${SIZES:-}"
pass --probe-inits "${PROBE_INITS:-}"
METHODS=${METHODS:-knn}
case ",${METHODS// /,}," in
  *,cell,*)
    DEVICE=cuda
    nvidia-smi --query-gpu=name --format=csv \
      || fail "METHODS=$METHODS trains the cell probe: needs a GPU session" ;;
  *)
    DEVICE=""  # numpy fp32 kNN on the CPU, as in Step 1
    nvidia-smi --query-gpu=name --format=csv || echo "no GPU (not needed)" ;;
esac
nproc; free -g || true

mkdir -p "$OUT"
STOP=$OUT/STOP
if [ -n "${SEED_FROM:-}" ]; then
  [ -d "$SEED_FROM" ] || fail "SEED_FROM=$SEED_FROM is not a directory"
  [ "$(cd "$SEED_FROM" && pwd -P)" != "$(cd "$OUT" && pwd -P)" ] || fail "SEED_FROM is the output dir"
  [ -z "$(ls -A "$OUT")" ] || fail "$OUT is not empty: seeding would merge two outputs"
  if [ ! -e "$SEED_FROM/cell_count.json" ] \
     && [ -z "$(find "$SEED_FROM" -mindepth 3 -maxdepth 3 -name row.json -print -quit)" ]; then
    nested=$(find -L "$SEED_FROM" -maxdepth 3 -name cell_count.json 2>/dev/null | sed 's|/cell_count.json$||')
    fail "SEED_FROM=$SEED_FROM is not a phase3_eval output dir (no cell_count.json, no <backbone>/<cell>/row.json)${nested:+; found: $(echo $nested)}"
  fi
  cp -r --no-preserve=mode,ownership "$SEED_FROM"/. "$OUT"/  # inputs are read-only
  chmod -R u+w "$OUT"
  echo "seeded $OUT from $SEED_FROM"
fi
rm -f "$STOP"  # left over from an earlier run
QUOTA=$OUT/quota_times.txt
echo "session_start_utc $(date -u +%FT%TZ)" >> "$QUOTA"
START=$(date +%s)
HARD=""
WATCH=""
summary() {
  local rc=$?
  set +e
  [ -z "$WATCH" ] || reap "$WATCH"
  echo "session_end_utc $(date -u +%FT%TZ) (exit $rc, $(( ($(date +%s) - START) / 60 )) min)" | tee -a "$QUOTA"
  echo "Stop the Kaggle session now: the quota runs until it is stopped."
}
trap summary EXIT
if [ -n "${DEADLINE_MIN:-}" ]; then
  [[ $DEADLINE_MIN =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "DEADLINE_MIN must be a number of minutes"
  GRACE_MIN=${GRACE_MIN:-45}
  SECS=$(awk -v m="$DEADLINE_MIN" 'BEGIN { printf "%d", m * 60 }')
  HARD=$((START + SECS + GRACE_MIN * 60))
  ( nap "$SECS"; touch "$STOP"; echo "DEADLINE: STOP written; the cell in hand finishes, no new one starts" ) & WATCH=$!
fi

RUN=(python -m geoguessr_ai.phase3_eval --data "$IN/v2" --emb-root "$IN/emb" --out "$OUT"
  --methods ${METHODS//,/ } --device "$DEVICE" --knn-dtype fp32 --mask-km "${MASK_KM:-1}"
  --stop-file "$STOP" "${EXTRA[@]}" "${P1ARG[@]}")
if [ -n "$HARD" ]; then
  left=$((HARD - $(date +%s))); [ "$left" -gt 0 ] || left=1
  RUN=(timeout -k 60 "${left}s" "${RUN[@]}")
fi
set +e
"${RUN[@]}" 2>&1 | tee -a "$OUT/phase3_eval.log"
rc=${PIPESTATUS[0]}
set -e
case $rc in
  0) ;;
  3) echo "RESUME NEEDED: rerun with the same settings (SEED_FROM only in a fresh session); finished cells are skipped"
     exit 3 ;;
  124|137) echo "RESUME NEEDED: the hard limit (deadline + ${GRACE_MIN:-45} min) killed phase3_eval mid-cell; finished cells are cached"
     exit 3 ;;
  *) fail "phase3_eval exit $rc" ;;
esac

if [ "$METHODS" = knn ] && [ -n "$P1" ] && [ -n "$TEST" ]; then
  echo "=== exploratory: Phase 1 matched-k kNN on the CPU (fp32), top-1 only"
  mkdir -p "$OUT/phase1_matched_k"
  (cd "$REPO_DIR" && python scripts/analysis/phase1_posthoc.py --dev-run "$P1" --data "$IN/v2" \
    --emb-root "$IN/emb" --test-emb-root "$TEST" --out "$OUT/phase1_matched_k") \
    2>&1 | tee -a "$OUT/phase1_matched_k/phase1_matched_k.log"
fi
ls -la "$OUT"
