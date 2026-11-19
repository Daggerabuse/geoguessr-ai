#!/usr/bin/env bash
# V4 training on Kaggle (GPU T4 x2, batch session): the declared pair jobs of
# configs/v4_predev.json, both backbones side by side, StreetCLIP on cuda:(k mod 2) and
# L/14-336 on the other GPU (decision 7; v4_train --device auto asserts it). Dev splits only:
# the inputs bundle holds no test file, and nothing here reads one.
#   usage: JOBS="0 1" bash scripts/kaggle_v4.sh
#   JOBS        job numbers k, run one after the other in this session (a pair at a time)
#   INPUTS      the V4 inputs bundle (default: the one dir under /kaggle/input holding
#               MANIFEST.sha256)
#   CONFIG      default configs/v4_predev.json (must be committed; v4_train checks)
#   DEADLINE_MIN  wall-clock minutes. At the deadline a STOP marker ($OUT/STOP) makes every run
#               checkpoint (pushed to the runs repo) and exit 3 at its next step: RESUME NEEDED,
#               this script exits 3. The runs are killed GRACE_MIN (default 30) minutes later.
#   SEED_FROM   the previous session's v4_runs output (attached as an input), copied into the
#               empty output dir first, to resume. Refused: not a directory, the output dir, a
#               non-empty output dir, a dir that is not a v4_runs output.
# Order: GPU check, HF write check on the private runs repo (before anything else costs time),
# archive staging (sha256-checked), the zero-init identity check per backbone (once per output),
# then the jobs. nvidia-smi (10 s) and vmstat logs go to $OUT. On exit the end time is appended
# to $OUT/quota_times.txt.
set -euo pipefail
export PYTHONUNBUFFERED=1
. "$(dirname "${BASH_SOURCE[0]}")/bg_helpers.sh"
. "$(dirname "${BASH_SOURCE[0]}")/find_input.sh"
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
K=${KAGGLE_ROOT:-/kaggle}  # only tests set KAGGLE_ROOT
fail() { echo "FAILED: $*"; exit 1; }
[ -n "${JOBS:-}" ] || fail "set JOBS, e.g. JOBS=\"0 1\""
for k in $JOBS; do [[ $k =~ ^[0-9]+$ ]] || fail "JOBS must be job numbers, got '$k'"; done
CONFIG=${CONFIG:-configs/v4_predev.json}
INPUTS=${INPUTS:-}
[ -n "$INPUTS" ] || INPUTS=$(find_input "$K/input" MANIFEST.sha256) || fail "attach the V4 inputs bundle"
OUT=$K/working/v4_runs
PARTS=$K/temp/v4_parts

ngpu=$(nvidia-smi -L | grep -c . || true)
[ "$ngpu" = 2 ] && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "T4" \
  || fail "needs GPU T4 x2 (found $ngpu GPUs)"
nproc; free -g || true

mkdir -p "$OUT"
STOP=$OUT/STOP
if [ -n "${SEED_FROM:-}" ]; then
  [ -d "$SEED_FROM" ] || fail "SEED_FROM=$SEED_FROM is not a directory"
  [ "$(cd "$SEED_FROM" && pwd -P)" != "$(cd "$OUT" && pwd -P)" ] || fail "SEED_FROM is the output dir"
  [ -z "$(ls -A "$OUT")" ] || fail "$OUT is not empty: seeding would merge two outputs"
  [ -e "$SEED_FROM/quota_times.txt" ] && ls "$SEED_FROM"/identity_*.json >/dev/null 2>&1 \
    || fail "SEED_FROM=$SEED_FROM is not a v4_runs output (no quota_times.txt + identity_*.json)"
  cp -r --no-preserve=mode,ownership "$SEED_FROM"/. "$OUT"/  # inputs are read-only
  chmod -R u+w "$OUT"
  echo "seeded $OUT from $SEED_FROM"
fi
rm -f "$STOP"  # left over from an earlier session
QUOTA=$OUT/quota_times.txt
echo "session_start_utc $(date -u +%FT%TZ) jobs $JOBS" >> "$QUOTA"
START=$(date +%s)
HELPERS=()
summary() {
  local rc=$?
  set +e
  [ ${#HELPERS[@]} = 0 ] || reap "${HELPERS[@]}"
  echo "session_end_utc $(date -u +%FT%TZ) (exit $rc, $(( ($(date +%s) - START) / 60 )) min)" | tee -a "$QUOTA"
  echo "Stop the Kaggle session now: the quota runs until it is stopped."
}
trap summary EXIT

python -m geoguessr_ai.v4_train hfcheck --config "$CONFIG" --out "$OUT" \
  || fail "HF write check on the runs repo (HF_TOKEN needs WRITE access; the repo must be private)"
if [ ! -e "$PARTS/STAGED" ]; then
  python -m geoguessr_ai.v4_train stage --config "$CONFIG" --parts "$PARTS" || fail "archive staging"
fi
df -h "$K/temp" "$K/working" || true

nvidia-smi --query-gpu=timestamp,index,uuid,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used \
  --format=csv -l 10 >> "$OUT/gpu_monitor.csv" 2>/dev/null & HELPERS+=($!)
if command -v vmstat >/dev/null; then vmstat -t 10 >> "$OUT/cpu_vmstat.log" & HELPERS+=($!); fi
HARD=""
if [ -n "${DEADLINE_MIN:-}" ]; then
  [[ $DEADLINE_MIN =~ ^[0-9]+(\.[0-9]+)?$ ]] || fail "DEADLINE_MIN must be a number of minutes"
  GRACE_MIN=${GRACE_MIN:-30}
  SECS=$(awk -v m="$DEADLINE_MIN" 'BEGIN { printf "%d", m * 60 }')
  HARD=$((START + SECS + GRACE_MIN * 60))
  ( nap "$SECS"; touch "$STOP"; echo "DEADLINE: STOP written; every run checkpoints and exits" ) & HELPERS+=($!)
fi

COMMON=(--config "$CONFIG" --inputs "$INPUTS" --parts "$PARTS" --out "$OUT" --stop-file "$STOP")
CODES=()
pair() {  # pair ARGS...: both backbones at once; their exit codes go to CODES
  local -a pids=() pre=()
  if [ -n "$HARD" ]; then
    local left=$((HARD - $(date +%s))); [ "$left" -gt 0 ] || left=1
    pre=(timeout -k 60 "${left}s")
  fi
  for bb in streetclip clip_l14_336; do
    ( set +e; "${pre[@]}" python -m geoguessr_ai.v4_train "$@" --backbone "$bb" "${COMMON[@]}" 2>&1 \
        | sed -u "s/^/[$bb] /" | tee -a "$OUT/$bb.log"; exit "${PIPESTATUS[0]}" ) & pids+=($!)
  done
  local p
  CODES=()
  for p in "${pids[@]}"; do
    if wait "$p"; then CODES+=(0); else CODES+=($?); fi
  done
}
verdict() {  # verdict WHAT CODES...: 0 all done, 3 resume needed, else fail
  local what=$1; shift
  local c resume=0
  for c in "$@"; do
    case $c in
      0) ;;
      3|124|137) resume=1 ;;
      *) fail "$what: exit codes $*" ;;
    esac
  done
  if [ $resume = 1 ]; then
    echo "RESUME NEEDED after $what: rerun this notebook with SEED_FROM = this output (finished runs are skipped)"
    exit 3
  fi
}

first=${JOBS%% *}
if ! ls "$OUT"/identity_streetclip.json "$OUT"/identity_clip_l14_336.json >/dev/null 2>&1; then
  s=$((first % 2))  # identity on the GPUs the first job uses
  pids=()
  for spec in "streetclip cuda:$s" "clip_l14_336 cuda:$((1 - s))"; do
    set -- $spec
    python -m geoguessr_ai.v4_train identity --backbone "$1" --device "$2" "${COMMON[@]}" 2>&1 \
      | sed -u "s/^/[$1] /" | tee -a "$OUT/$1.log" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
fi
for bb in streetclip clip_l14_336; do
  python - "$OUT/identity_$bb.json" <<'EOF' || fail "zero-init identity check failed for $bb: no training"
import json, sys
r = json.load(open(sys.argv[1]))
print(sys.argv[1], r)
sys.exit(0 if r["ok"] else 1)
EOF
done

for k in $JOBS; do
  echo "=== job $k: streetclip on cuda:$((k % 2)), clip_l14_336 on cuda:$((1 - k % 2))"
  [ ! -e "$STOP" ] || { echo "RESUME NEEDED: deadline before job $k"; exit 3; }
  pair train --job "$k" --device auto
  verdict "job $k" "${CODES[@]}"
done
ls -la "$OUT"
