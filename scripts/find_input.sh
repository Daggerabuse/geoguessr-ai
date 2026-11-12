# Sourced by the Kaggle scripts (kaggle_phase1.sh test, kaggle_phase1_eval.sh,
# kaggle_phase1_final.sh). find_input ROOT [REL]: prints the one dir under ROOT holding REL
# (default v2/config.json: the run's base dir, with v2/ and emb/). Kaggle's mount depth varies
# (/kaggle/input/<slug>/ or .../notebooks/<user>/<slug>/) and mounts may be symlinks: -L follows
# them (find does not descend a symlinked dir, nor a symlinked ROOT, without it).
find_input() {
  local rel=${2:-v2/config.json} m n
  m=$(find -L "$1" -maxdepth 6 -path "*/$rel" 2>/dev/null)
  n=$(printf '%s\n' "$m" | grep -c . || true)
  if [ "$n" -ne 1 ]; then
    echo "FAILED: expected exactly 1 $rel under $1, found $n" >&2
    if [ -n "$m" ]; then
      printf '  %s\n' $m >&2
    else  # show what is mounted, to see why
      find -L "$1" -maxdepth 3 2>/dev/null | head -n 30 | sed 's/^/  /' >&2
    fi
    return 1
  fi
  printf '%s\n' "${m%/"$rel"}"
}
