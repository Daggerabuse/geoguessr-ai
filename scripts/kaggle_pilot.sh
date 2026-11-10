#!/usr/bin/env bash
# Phase 1 pilot on Kaggle (2x T4, internet on): 3 train zips through the full pipeline.
#   zipq downloader (sequential, 2 ahead) -> 3 embedding consumers + the FT-archive consumer
#   -> --check of every cache -> archive manifest check. The archive consumer uploads each part
#   to a PRIVATE Hugging Face dataset repo as soon as it is committed (then deletes it locally).
# Train/dev/dev_seen only: no test image is embedded or scored. Stops at the first failure.
# Budget: everything runs before DEADLINE_MIN after the start (40 min: <= 1.33 GPU-h even if
# Kaggle counts 2x T4 double). Stop the session right after the last cell: the quota runs
# until the session is stopped.
#
# Notebook cells: see README ("Phase 1 pilot"). HF_TOKEN (write access) comes from Kaggle
# Secrets as an environment variable; this script never prints it. On exit, success or not, a
# trap prints the throughput/disk summary and writes session_end_utc.
set -euo pipefail
set +x

W=/kaggle/working
P=$W/pilot
RAW=/kaggle/temp/raw
Z=/kaggle/temp/zips
STAGING=/kaggle/temp/ft_staging
ARCHIVE=/kaggle/temp/ft_archive
V2=$W/v2
EMB=$W/emb
IDX=$EMB/zip_index
SHARDS="0 1 2"
DEADLINE_MIN=${DEADLINE_MIN:-40}
MAX_WORKING_GB=15
REPO=Daggerabuse/osv5m-ft-archive-pilot  # a separate private repo; never the main FT archive
IDS="$V2/train.csv $V2/dev.csv $V2/dev_seen.csv"
ZIPARGS="--zip-dir $Z --index-dir $IDX --shards $SHARDS"

START=$(date +%s)
DEADLINE=$((START + DEADLINE_MIN * 60))
step() { echo; echo "=== $* ($(date -u +%T) UTC, $(( ($(date +%s) - START) / 60 )) min)"; }
fail() { echo "FAILED: $*"; mkdir -p "$Z/markers"; echo "pilot: $*" > "$Z/markers/ABORT"; exit 1; }
# seconds left for `timeout`; past the deadline 1s, so the step is killed and reported (exit 124)
left() { local s=$((DEADLINE - $(date +%s))); [ "$s" -gt 0 ] || s=1; echo "${s}s"; }

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
summary() {
  local rc=$?
  set +e
  kill "$DU" 2>/dev/null
  step "summary (exit $rc)"
  for m in clip_l14_336 streetclip clip_b16; do tail -n 1 "$EMB/$m/throughput.jsonl" 2>/dev/null; done
  cat "$P/zipq_stats.json" 2>/dev/null
  du -sh "$W"/* /kaggle/temp/* 2>/dev/null | sort -h
  df -h "$W" /kaggle/temp
  awk '$2 > t {t = $2} $3 > w {w = $3} END {printf "peak /kaggle/temp %.2f GB, peak /kaggle/working %.2f GB (%d samples)\n", t / 1e9, w / 1e9, NR}' \
    "$P/disk_bytes.log" | tee "$P/disk_peaks.txt"
  echo "session_end_utc $(date -u +%FT%TZ)" >> "$P/quota_times.txt"
  cat "$P/quota_times.txt"
  echo "Bring back: $P/ (pilot.log, logs, checks, disk_*, quota_times.txt, zipq_stats.json), $EMB/*/throughput.jsonl, $V2/config.json"
}
trap summary EXIT
step "environment (note Kaggle's remaining GPU quota NOW, and again after stopping the session)"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN not set (Kaggle Secrets)"
[[ $REPO == *-pilot ]] || fail "the pilot only uploads to a *-pilot repo"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
[ "$(nvidia-smi -L | wc -l)" -ge 2 ] || fail "the pilot layout needs 2 GPUs"
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
python -c "import json, sys; c = json.load(open(sys.argv[1])); print('pool_size', c['pool_size'], 'subsets_nested', c['subsets_nested']); sys.exit(0 if c['subsets_nested'] else 1)" "$V2/config.json" || fail "subsets not nested"

step "downloader + 3 embedding consumers + archive consumer"
# GPU1: StreetCLIP alone (the slowest backbone). GPU0: L/14-336 and B/16 as two concurrent
# processes (all read the same local zip, so no zip is kept for a later pass).
timeout "$(left)" python -m geoguessr_ai.zipq --zip-dir "$Z" --index-dir "$IDX" --shards $SHARDS \
  --consumers clip_l14_336 streetclip clip_b16 ft_archive --known-ids "$RAW/train.csv" \
  --stats-out "$P/zipq_stats.json" > "$P/zipq.log" 2>&1 & PD=$!
EMBED="python -m geoguessr_ai.embed --ids $IDS --source train $ZIPARGS --workers 4"
timeout "$(left)" $EMBED --model openai/clip-vit-large-patch14-336 --out "$EMB/clip_l14_336" \
  --batch-size 64 --device cuda:0 > "$P/clip_l14_336.log" 2>&1 & P1=$!
timeout "$(left)" $EMBED --model geolocal/StreetCLIP --out "$EMB/streetclip" \
  --batch-size 64 --device cuda:1 > "$P/streetclip.log" 2>&1 & P2=$!
timeout "$(left)" $EMBED --model openai/clip-vit-base-patch16 --out "$EMB/clip_b16" \
  --batch-size 256 --device cuda:0 > "$P/clip_b16.log" 2>&1 & P3=$!
timeout "$(left)" python -m geoguessr_ai.ft_archive consume --ids $IDS --nat-max 200000 $ZIPARGS \
  --staging "$STAGING" --out "$ARCHIVE" --repo "$REPO" --log "$P/upload.log" \
  > "$P/ft_archive.log" 2>&1 & P4=$!
# wait for whichever ends first; the first failure writes ABORT so the others stop at once
declare -A NAME=([$P1]=clip_l14_336 [$P2]=streetclip [$P3]=clip_b16 [$P4]=ft_archive [$PD]=zipq)
status=0
while [ ${#NAME[@]} -gt 0 ]; do
  if wait -n -p done "${!NAME[@]}"; then rc=0; else rc=$?; fi
  name=${NAME[$done]}; unset "NAME[$done]"
  if [ "$rc" -ne 0 ]; then
    echo "FAILED: $name (exit $rc; 124 = budget timeout; see $P/$name.log)"; status=1
    mkdir -p "$Z/markers"; [ -e "$Z/markers/ABORT" ] || echo "pilot: $name failed" > "$Z/markers/ABORT"
  fi
done
tail -n 3 "$P"/*.log
[ $status -eq 0 ] || fail "a pipeline process failed"

step "checks: every wanted id in the 3 zips embedded exactly once, per backbone"
for m in clip_l14_336 streetclip clip_b16; do
  python -m geoguessr_ai.embed --check --ids $IDS --source train --out "$EMB/$m" $ZIPARGS \
    | tee "$P/check_$m.json" || fail "check of $m"
done

step "archive: manifest = (nat_200K + dev + dev_seen) in these zips, no duplicates; all parts uploaded"
python - "$ARCHIVE" "$IDX" "$V2" $SHARDS <<'EOF' | tee "$P/archive_check.json"
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
echo; echo "PILOT PASSED. Stop the session now and note the remaining GPU quota."
