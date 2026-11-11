# Sourced by kaggle_phase1_eval.sh. Prints the attached run's base dir (the one holding v2/ and
# emb/) under $1; Kaggle's mount depth varies (/kaggle/input/<slug>/ or .../notebooks/<user>/<slug>/).
find_input() {
  local m n
  m=$(find "$1" -maxdepth 6 -path '*/v2/config.json' 2>/dev/null)
  n=$(printf '%s\n' "$m" | grep -c . || true)
  if [ "$n" -ne 1 ]; then
    echo "FAILED: expected exactly 1 v2/config.json under $1, found $n (set INPUT_DIR)" >&2
    [ -z "$m" ] || printf '  %s\n' $m >&2
    return 1
  fi
  dirname "$(dirname "$m")"
}
