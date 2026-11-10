#!/usr/bin/env bash
# Phase 1 on Kaggle (2x T4, internet on): train zips through the whole pipeline.
#   usage: bash scripts/kaggle_phase1.sh pilot|full   (kaggle_pilot.sh / kaggle_full.sh call this)
#   pilot  3 zips, 40 min deadline, archive -> Daggerabuse/osv5m-ft-archive-pilot
#   full   all 98 zips, 11 h deadline, archive -> Daggerabuse/osv5m-ft-archive (exactly; see
#          geoguessr_ai/phase1.py for both modes and the repo guard)
# zipq downloader (sequential, 2 ahead) -> 3 embedding consumers + the FT-archive consumer, which
# uploads each part to the PRIVATE Hugging Face repo as soon as it is committed -> --check of
# every cache -> archive check. Train/dev/dev_seen only: no test image is embedded or scored.
# Deadline: a STOP marker makes every consumer finish the zip in hand and start no new one
# (completed zips stay valid), then RESUME NEEDED lists what is left (exit 3). Processes are
# killed only GRACE_MIN after the deadline. Any failure writes ABORT and stops everyone.
# HF_TOKEN (write access) comes from Kaggle Secrets as an environment variable; never printed.
# On exit, success or not, a trap prints throughput/disk and writes session_end_utc.
# Stop the Kaggle session right after the last cell: the quota runs until it is stopped.
set -euo pipefail
set +x

eval "$(python -m geoguessr_ai.phase1 config "${1:?usage: kaggle_phase1.sh pilot|full}")"
W=/kaggle/working
P=$W/$OUT_NAME
RAW=/kaggle/temp/raw
Z=/kaggle/temp/zips
STAGING=/kaggle/temp/ft_staging
ARCHIVE=/kaggle/temp/ft_archive
V2=$W/v2
EMB=$W/emb
IDX=$EMB/zip_index
GRACE_MIN=$([ "$MODE" = full ] && echo 30 || echo 10)
MAX_WORKING_GB=15
CONSUMERS="clip_l14_336 streetclip clip_b16 ft_archive"
IDS="$V2/train.csv $V2/dev.csv $V2/dev_seen.csv"
ZIPARGS="--zip-dir $Z --index-dir $IDX --shards $SHARDS"

START=$(date +%s)
DEADLINE=$((START + DEADLINE_MIN * 60))
HARD=$((DEADLINE + GRACE_MIN * 60))
step() { echo; echo "=== $* ($(date -u +%T) UTC, $(( ($(date +%s) - START) / 60 )) min)"; }
fail() { echo "FAILED: $*"; mkdir -p "$Z/markers"; echo "$MODE: $*" > "$Z/markers/ABORT"; exit 1; }
# seconds left for `timeout` before the hard limit (deadline + grace); at least 1 s
left() { local s=$((HARD - $(date +%s))); [ "$s" -gt 0 ] || s=1; echo "${s}s"; }

mkdir -p "$P"
echo "session_start_utc $(date -u +%FT%TZ)" > "$P/quota_times.txt"
# disk samples every 10 s: epoch, /kaggle/temp bytes, /kaggle/working bytes. No set -e/pipefail
# in here: du fails on files deleted mid-scan, which must not end the sampling.
( set +e +o pipefail
  while :; do
    t=$(du -sb /kaggle/temp 2>/dev/null | cut -f1); w=$(du -sb "$W" 2>/dev/null | cut -f1)
    echo "$(date +%s) ${t:-0} ${w:-0}"
    sleep 10
  done ) > "$P/disk_bytes.log" &
DU=$!
BG="$DU"
summary() {
  local rc=$?
  set +e
  kill $BG 2>/dev/null
  step "summary ($MODE, exit $rc)"
  for m in clip_l14_336 streetclip clip_b16; do tail -n 1 "$EMB/$m/throughput.jsonl" 2>/dev/null; done
  cat "$P/zipq_stats.json" 2>/dev/null
  du -sh "$W"/* /kaggle/temp/* 2>/dev/null | sort -h
  df -h "$W" /kaggle/temp
  awk '$2 > t {t = $2} $3 > w {w = $3} END {printf "peak /kaggle/temp %.2f GB, peak /kaggle/working %.2f GB (%d samples)\n", t / 1e9, w / 1e9, NR}' \
    "$P/disk_bytes.log" | tee "$P/disk_peaks.txt"
  echo "session_end_utc $(date -u +%FT%TZ)" >> "$P/quota_times.txt"
  cat "$P/quota_times.txt"
  echo "Bring back: $P/ ($OUT_NAME.log, logs, checks, disk_*, progress.log, quota_times.txt, zipq_stats.json), $EMB/*/throughput.jsonl, $V2/config.json"
}
trap summary EXIT

step "$MODE: ${SHARDS// /,} -> $REPO, deadline ${DEADLINE_MIN} min (+${GRACE_MIN} min grace). Note Kaggle's remaining GPU quota NOW, and again after stopping the session"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN not set (Kaggle Secrets)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
[ "$(nvidia-smi -L | wc -l)" -ge 2 ] || fail "this layout needs 2 GPUs"
nproc; free -g; df -h "$W" /kaggle/temp

step "official CSVs -> $RAW"
mkdir -p "$RAW"
python - "$RAW" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
for name in ("train.csv", "test.csv"):
    print(hf_hub_download("osv5m/osv5m", name, repo_type="dataset", local_dir=sys.argv[1]))
EOF

step "manifests with dev_seen (test.csv metadata only for the leakage check)"
timeout "$(left)" python -m geoguessr_ai.v2_data --out "$V2" --train-csv "$RAW/train.csv" \
  --test-csv "$RAW/test.csv" --sizes 50000 100000 200000 500000
python -c "import json, sys; c = json.load(open(sys.argv[1])); print('pool_size', c['pool_size'], 'images', sum(c['n'][k] for k in ('train', 'dev', 'dev_seen')), 'subsets_nested', c['subsets_nested']); sys.exit(0 if c['subsets_nested'] else 1)" "$V2/config.json" || fail "subsets not nested"

step "downloader + 3 embedding consumers + archive consumer"
mkdir -p "$Z/markers"
rm -f "$Z/markers/STOP" "$Z/markers/ABORT"  # left over from an earlier run in this session
# GPU1: StreetCLIP alone (the slowest backbone). GPU0: L/14-336 and B/16 as two concurrent
# processes (all read the same local zip, so no zip is kept for a later pass).
timeout "$(left)" python -m geoguessr_ai.zipq --zip-dir "$Z" --index-dir "$IDX" --shards $SHARDS \
  --consumers $CONSUMERS --known-ids "$RAW/train.csv" \
  --stats-out "$P/zipq_stats.json" > "$P/zipq.log" 2>&1 & PD=$!
EMBED="python -m geoguessr_ai.embed --ids $IDS --source train $ZIPARGS --workers 4"
timeout "$(left)" $EMBED --model openai/clip-vit-large-patch14-336 --out "$EMB/clip_l14_336" \
  --batch-size 64 --device cuda:0 > "$P/clip_l14_336.log" 2>&1 & P1=$!
timeout "$(left)" $EMBED --model geolocal/StreetCLIP --out "$EMB/streetclip" \
  --batch-size 64 --device cuda:1 > "$P/streetclip.log" 2>&1 & P2=$!
timeout "$(left)" $EMBED --model openai/clip-vit-base-patch16 --out "$EMB/clip_b16" \
  --batch-size 256 --device cuda:0 > "$P/clip_b16.log" 2>&1 & P3=$!
timeout "$(left)" python -m geoguessr_ai.ft_archive consume --ids $IDS --nat-max 200000 $ZIPARGS \
  --staging "$STAGING" --out "$ARCHIVE" --repo "$REPO" --mode "$MODE" --log "$P/upload.log" \
  > "$P/ft_archive.log" 2>&1 & P4=$!
# deadline: a clean stop at the next zip boundary (not a failure)
( sleep $((DEADLINE - $(date +%s) > 0 ? DEADLINE - $(date +%s) : 0)); echo "deadline" > "$Z/markers/STOP"
  echo "DEADLINE: STOP written; consumers finish their current zip" ) & WATCH=$!
python -u -m geoguessr_ai.phase1 progress --zip-dir "$Z" --index-dir "$IDX" --shards $SHARDS \
  --consumers $CONSUMERS --ids $IDS --disk /kaggle/temp "$W" > >(tee -a "$P/progress.log") 2>&1 & PROG=$!
BG="$DU $WATCH $PROG"
# wait for whichever ends first; the first failure writes ABORT so the others stop at once
declare -A NAME=([$P1]=clip_l14_336 [$P2]=streetclip [$P3]=clip_b16 [$P4]=ft_archive [$PD]=zipq)
status=0
while [ ${#NAME[@]} -gt 0 ]; do
  if wait -n -p done "${!NAME[@]}"; then rc=0; else rc=$?; fi
  name=${NAME[$done]}; unset "NAME[$done]"
  if [ "$rc" -ne 0 ]; then
    echo "FAILED: $name (exit $rc; 124 = hard time limit; see $P/$name.log)"; status=1
    [ -e "$Z/markers/ABORT" ] || echo "$MODE: $name failed" > "$Z/markers/ABORT"
  fi
done
kill $WATCH $PROG 2>/dev/null || true
tail -n 3 "$P"/*.log
[ $status -eq 0 ] || fail "a pipeline process failed"
python -m geoguessr_ai.phase1 status --zip-dir "$Z" --shards $SHARDS --consumers $CONSUMERS \
  | tee "$P/status.txt" || { echo "Not all zips done: skipping the checks."; exit 3; }

step "checks: every wanted id in the zips embedded exactly once, per backbone"
for m in clip_l14_336 streetclip clip_b16; do
  python -m geoguessr_ai.embed --check --ids $IDS --source train --out "$EMB/$m" $ZIPARGS \
    | tee "$P/check_$m.json" || fail "check of $m"
done

step "archive: manifest = (nat_200K + dev + dev_seen) in these zips, no duplicates; all parts uploaded"
python - "$ARCHIVE" "$IDX" "$V2" $SHARDS <<'EOF' | tee "$P/archive_check.json" || fail "archive check"
import json, sys
from pathlib import Path
import pandas as pd
from geoguessr_ai.ft_archive import wanted_ids
from geoguessr_ai.zipq import load_zip_index
archive, idx, v2, shards = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), map(int, sys.argv[4:])
m = pd.read_csv(archive / "manifest.csv", dtype={"id": str, "sha256": str})
want = wanted_ids([v2 / "train.csv", v2 / "dev.csv", v2 / "dev_seen.csv"], 200_000)
expected = want & load_zip_index(idx, "train", list(shards)).keys()
res = {"images": len(m), "expected": len(expected), "duplicates": int(m["id"].duplicated().sum()),
       "missing": len(expected - set(m["id"])), "extra": len(set(m["id"]) - expected),
       "parts": int(m["part"].nunique()), "gb": round(m["bytes"].sum() / 1e9, 3)}
res["parts_left_locally"] = len(list(archive.glob("part_*.zip")))  # 0: all uploaded and deleted
res["ok"] = res["duplicates"] == res["missing"] == res["extra"] == res["parts_left_locally"] == 0
print(json.dumps(res))
sys.exit(0 if res["ok"] else 1)
EOF

step "disk"
gb=$(du -sb "$W" | awk '{printf "%.2f", $1 / 1e9}')
awk -v g="$gb" -v m="$MAX_WORKING_GB" 'BEGIN { exit !(g <= m) }' || fail "/kaggle/working $gb GB > $MAX_WORKING_GB GB"
echo; echo "${MODE^^} RUN PASSED. Stop the session now and note the remaining GPU quota."
