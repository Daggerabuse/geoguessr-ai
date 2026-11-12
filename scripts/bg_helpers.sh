# Sourced by kaggle_phase1.sh. Helpers run in subshells; a plain `kill` of a subshell leaves its
# `sleep` child alive (and holding the parent's stdout pipe), so sleep via nap and stop via reap.
# nap: the TERM trap is set before sleep starts, so an early kill cannot orphan the sleep.
nap() { local s=; trap '[ -z "$s" ] || kill $s 2>/dev/null; exit 0' TERM; sleep "$1" & s=$!; wait $s; }
# reap: never fails. The caller runs under set -e, and `wait` returns the killed helper's 143
# (`kill` 1 for one already gone): unguarded, that ended the whole script right there.
reap() { kill "$@" 2>/dev/null || true; wait "$@" 2>/dev/null || true; }
