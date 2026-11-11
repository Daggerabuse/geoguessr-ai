# Sourced by kaggle_phase1_eval.sh. find_input ROOT [REL]: prints the one dir under ROOT holding
# REL (default v2/config.json: the run's base dir, with v2/ and emb/); Kaggle's mount depth
# varies (/kaggle/input/<slug>/ or .../notebooks/<user>/<slug>/).
find_input() {
  local rel=${2:-v2/config.json} m n
  m=$(find "$1" -maxdepth 6 -path "*/$rel" 2>/dev/null)
  n=$(printf '%s\n' "$m" | grep -c . || true)
  if [ "$n" -ne 1 ]; then
    echo "FAILED: expected exactly 1 $rel under $1, found $n" >&2
    [ -z "$m" ] || printf '  %s\n' $m >&2
    return 1
  fi
  printf '%s\n' "${m%/"$rel"}"
}
