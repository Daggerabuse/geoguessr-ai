#!/usr/bin/env bash
# Phase 1 pilot on Kaggle (2x T4, internet on): 3 train zips through the full pipeline.
#   zipq downloader (sequential, 2 ahead) -> 3 embedding consumers + the FT-archive consumer
#   -> --check of every cache -> archive manifest check -> upload to a PRIVATE Kaggle Dataset.
# Train/dev/dev_seen only: no test image is embedded or scored. Stops at the first failure.
# Budget: everything runs before DEADLINE_MIN after the start (40 min: <= 1.33 GPU-h even if
# Kaggle counts 2x T4 double). Stop the session right after the last cell: the quota runs
# until the session is stopped.
#
# Notebook cells: see README ("Phase 1 pilot"). Secrets (HF_TOKEN, KAGGLE_USERNAME,
# KAGGLE_KEY) come from Kaggle Secrets as environment variables; this script never prints them.
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
DATASET=osv5m-ft-pilot  # a separate private dataset; never the main FT archive
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
step "environment (note Kaggle's remaining GPU quota NOW, and again after stopping the session)"
for v in HF_TOKEN KAGGLE_USERNAME KAGGLE_KEY; do [ -n "${!v:-}" ] || fail "$v not set (Kaggle Secrets)"; done
command -v kaggle > /dev/null || fail "kaggle CLI not found"
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
# GPU0: L/14-336 alone. GPU1: StreetCLIP and B/16 as two concurrent processes (B/16 needs ~1/7
# of StreetCLIP's GPU time; both read the same local zip, so no zip is kept for a later pass).
( while sleep 10; do du -sb "$Z" "$STAGING" "$ARCHIVE" 2>/dev/null | awk '{s+=$1} END {print s}'; done ) > "$P/temp_bytes.log" &
DU=$!
timeout "$(left)" python -m geoguessr_ai.zipq --zip-dir "$Z" --index-dir "$IDX" --shards $SHARDS \
  --consumers clip_l14_336 streetclip clip_b16 ft_archive --known-ids "$RAW/train.csv" \
  --stats-out "$P/zipq_stats.json" > "$P/zipq.log" 2>&1 & PD=$!
EMBED="python -m geoguessr_ai.embed --ids $IDS --source train $ZIPARGS --workers 4"
timeout "$(left)" $EMBED --model openai/clip-vit-large-patch14-336 --out "$EMB/clip_l14_336" \
  --batch-size 64 --device cuda:0 > "$P/clip_l14_336.log" 2>&1 & P1=$!
timeout "$(left)" $EMBED --model geolocal/StreetCLIP --out "$EMB/streetclip" \
  --batch-size 64 --device cuda:1 > "$P/streetclip.log" 2>&1 & P2=$!
timeout "$(left)" $EMBED --model openai/clip-vit-base-patch16 --out "$EMB/clip_b16" \
  --batch-size 256 --device cuda:1 > "$P/clip_b16.log" 2>&1 & P3=$!
timeout "$(left)" python -m geoguessr_ai.ft_archive consume --ids $IDS --nat-max 200000 $ZIPARGS \
  --staging "$STAGING" --out "$ARCHIVE" > "$P/ft_archive.log" 2>&1 & P4=$!
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
kill $DU 2>/dev/null || true
tail -n 3 "$P"/*.log
[ $status -eq 0 ] || fail "a pipeline process failed"
echo "peak /kaggle/temp use by the pipeline: $(sort -n "$P/temp_bytes.log" | tail -1 | awk '{printf "%.2f GB", $1/1e9}')"

step "checks: every wanted id in the 3 zips embedded exactly once, per backbone"
for m in clip_l14_336 streetclip clip_b16; do
  python -m geoguessr_ai.embed --check --ids $IDS --source train --out "$EMB/$m" $ZIPARGS \
    | tee "$P/check_$m.json" || fail "check of $m"
done

step "archive: manifest = (nat_200K + dev + dev_seen) in these zips, no duplicates"
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
res["ok"] = res["duplicates"] == res["missing"] == res["extra"] == 0
print(json.dumps(res))
sys.exit(0 if res["ok"] else 1)
EOF

step "upload the pilot archive to the private Kaggle Dataset $DATASET"
[[ $DATASET == *-pilot ]] || fail "the pilot only uploads to a *-pilot dataset"
timeout "$(left)" python -m geoguessr_ai.ft_archive upload --out "$ARCHIVE" --dataset "$DATASET" \
  --title "OSV-5M FT subset pilot (raw JPEG bytes)" --log "$P/upload.log" || fail "upload"

step "throughput and disk"
for m in clip_l14_336 streetclip clip_b16; do tail -n 1 "$EMB/$m/throughput.jsonl"; done
cat "$P/zipq_stats.json"
du -sh "$W"/* /kaggle/temp/* 2>/dev/null | sort -h
df -h "$W" /kaggle/temp
gb=$(du -sb "$W" | awk '{printf "%.2f", $1 / 1e9}')
awk -v g="$gb" -v m="$MAX_WORKING_GB" 'BEGIN { exit !(g <= m) }' || fail "/kaggle/working $gb GB > $MAX_WORKING_GB GB"
echo "session_end_utc $(date -u +%FT%TZ)" >> "$P/quota_times.txt"
cat "$P/quota_times.txt"
echo; echo "PILOT PASSED. Stop the session now and note the remaining GPU quota."
echo "Bring back: $P/ (logs, checks, quota_times.txt, zipq_stats.json), $EMB/*/throughput.jsonl, $V2/config.json"
