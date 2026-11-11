# Sourced by kaggle_phase1.sh. Helpers run in subshells; a plain `kill` of a subshell leaves its
# `sleep` child alive (and holding the parent's stdout pipe), so sleep via nap and stop via reap.
nap() { sleep "$1" & local s=$!; trap 'kill $s 2>/dev/null; exit 0' TERM; wait $s; }
reap() { kill "$@" 2>/dev/null; wait "$@" 2>/dev/null; true; }
