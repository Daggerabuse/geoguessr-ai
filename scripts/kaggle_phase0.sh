#!/usr/bin/env bash
# Phase 0 validation on Kaggle (2x T4, internet on). Train/dev only: nothing is embedded or
# evaluated from the test split. Stops at the first failed step or check (set -e).
#
# Notebook, first cell (HF_TOKEN from Kaggle Secrets, never printed):
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
# Second cell:
#   !git clone <repo url> /kaggle/working/geoguessr-ai && cd /kaggle/working/geoguessr-ai && pip install -q -e .
#   !bash /kaggle/working/geoguessr-ai/scripts/kaggle_phase0.sh 2>&1 | tee /kaggle/working/phase0.log
set -euo pipefail

W=/kaggle/working
RAW=/kaggle/temp/raw           # raw CSVs (about 3 GB) stay out of /kaggle/working
V2=$W/v2
EMB=$W/emb
MAX_WORKING_GB=15
B16=openai/clip-vit-base-patch16
L14=openai/clip-vit-large-patch14-336
SC=geolocal/StreetCLIP
EMBED="python -m geoguessr_ai.embed --ids $V2/train.csv --source train"

step() { echo; echo "=== $* ($(date +%T))"; }
fail() { echo "FAILED: $*"; exit 1; }
working_gb() { du -sb "$W" | awk '{printf "%.2f", $1 / 1e9}'; }
disk_check() {
  local gb; gb=$(working_gb)
  echo "/kaggle/working: $gb GB"
  awk -v g="$gb" -v m="$MAX_WORKING_GB" 'BEGIN { exit !(g <= m) }' || fail "/kaggle/working above $MAX_WORKING_GB GB"
}
empty_dir() { [ ! -e "$1" ] || [ -z "$(ls -A "$1")" ] || fail "$1 is not empty"; }

step "environment"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN not set (load it from Kaggle Secrets, see the header)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
[ "$(nvidia-smi -L | wc -l)" -ge 2 ] || fail "dual-T4 test needs 2 GPUs"
nproc; free -g; df -h "$W" /kaggle/temp
python -c "import torch, transformers, huggingface_hub; print(torch.__version__, transformers.__version__, huggingface_hub.__version__)"

step "1. official CSVs -> $RAW"
mkdir -p "$RAW"
python - "$RAW" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
for name in ("train.csv", "test.csv"):
    print(hf_hub_download("osv5m/osv5m", name, repo_type="dataset", local_dir=sys.argv[1]))
EOF

step "1. full metadata audit (fails if train.csv lacks a required column)"
python -m geoguessr_ai.v2_data --out "$V2" --train-csv "$RAW/train.csv" --test-csv "$RAW/test.csv" --audit-only
step "1. manifests (test.csv metadata is read for leakage removal only)"
python -m geoguessr_ai.v2_data --out "$V2" --train-csv "$RAW/train.csv" --test-csv "$RAW/test.csv" \
  --sizes 50000 100000 200000 500000
disk_check

step "2. synthetic GPU throughput (after warm-up steps)"
timeout 15m python -m geoguessr_ai.bench --model $B16 --batch-size 64 --infer-batch-size 256 --out "$W/bench.jsonl"
timeout 15m python -m geoguessr_ai.bench --model $L14 --batch-size 16 --grad-checkpointing --infer-batch-size 64 --out "$W/bench.jsonl"
timeout 15m python -m geoguessr_ai.bench --model $SC --batch-size 16 --grad-checkpointing --infer-batch-size 64 --out "$W/bench.jsonl"

step "2+3. end-to-end 2,000 train images per backbone, single GPU, empty cache dirs"
# the first run also builds the shared shard index ($EMB/index_train.json; index_seconds)
for spec in "clip_b16 $B16 256" "clip_l14_336 $L14 64" "streetclip $SC 64"; do
  set -- $spec
  empty_dir "$EMB/$1"
  timeout 30m $EMBED --model "$2" --out "$EMB/$1" --batch-size "$3" --limit 2000 --device cuda:0
  $EMBED --check --out "$EMB/$1" --limit 2000 || fail "check of $1"
done
disk_check

step "4. dual T4: 2 x part of the same 2,000 ids, part 0 killed mid-run, then resumed"
D=$EMB/dual_b16
empty_dir "$D"
DUAL="$EMBED --model $B16 --out $D --batch-size 256 --limit 2000 --parts 2"
$DUAL --part 0 --device cuda:0 > "$W/dual_part0.log" 2>&1 & P0=$!
$DUAL --part 1 --device cuda:1 > "$W/dual_part1.log" 2>&1 & P1=$!
# part 0 owns the even shards; kill it (SIGKILL, no cleanup) once it has saved one
until compgen -G "$D/train_[0-9][02468]_*.npz" > /dev/null || ! kill -0 $P0 2>/dev/null; do sleep 1; done
kill -0 $P0 2>/dev/null || fail "part 0 finished before it could be interrupted (test not valid)"
kill -9 $P0; wait $P0 || true
wait $P1 || fail "part 1 (see $W/dual_part1.log)"
echo "after the kill (expected: missing > 0, no duplicates):"
if $EMBED --check --out "$D" --limit 2000; then fail "nothing was missing after the kill (test not valid)"; fi
timeout 20m $DUAL --part 0 --device cuda:0 2>&1 | tee "$W/dual_part0_resume.log"
echo "after resume, against the single-GPU cache:"
$EMBED --check --out "$D" --limit 2000 --reference "$EMB/clip_b16" || fail "dual/resume check"

step "GPU hours from the measured end-to-end rates (single GPU, after the first batch)"
python - "$EMB" <<'EOF'
import json, sys
from pathlib import Path
emb = Path(sys.argv[1])
sizes = [50_000, 100_000, 200_000, 500_000]
rows = ["| backbone | img/s | gpu busy | " + " | ".join(f"{n // 1000}K" for n in sizes) + " | dev+test 230K |",
        "|---|---|---|" + "---|" * (len(sizes) + 1)]
for tag in ("clip_b16", "clip_l14_336", "streetclip"):
    r = json.loads((emb / tag / "throughput.jsonl").read_text().splitlines()[0])
    rate = r["images_per_s_after_first_batch"] or r["images_per_s"]
    hours = [n / rate / 3600 for n in [*sizes, 230_122]]
    rows.append(f"| {tag} | {rate:.1f} | {r['gpu_busy_frac']:.0%} | " + " | ".join(f"{h:.2f}" for h in hours) + " |")
print("\n".join(rows))
(emb.parent / "phase0_gpu_hours.md").write_text("\n".join(rows) + "\n")
EOF

step "disk usage"
du -sh "$W"/* /kaggle/temp/* 2>/dev/null | sort -h
df -h "$W" /kaggle/temp
disk_check
echo; echo "ALL PHASE 0 CHECKS PASSED. Bring back: v2/audit.json, v2/country_counts.csv, v2/config.json,"
echo "bench.jsonl, emb/*/throughput.jsonl, emb/*/encoder.json, phase0_gpu_hours.md, phase0.log, dual_*.log"
