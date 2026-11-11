"""kaggle_phase1.sh helpers: stopping background helpers must free the tee pipe at once."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HELPERS = Path(__file__).resolve().parents[1] / "scripts" / "bg_helpers.sh"
BASH = shutil.which("bash")

# Same shape as the script: a nap-ing watcher and a sampler loop, 300 s long, output to files,
# (the test reads the pipe like the cell's tee). Work "finishes" after 1 s.
SCRIPT = f"""
. '{HELPERS.as_posix()}'
( nap 300; echo deadline ) > "$1/w.log" 2>&1 & W=$!
( while :; do echo s; nap 300; done ) > "$1/d.log" 2>&1 & D=$!
sleep 1
reap $W $D
echo "alive: $(ps -o pid= -p $W -p $D 2>/dev/null | wc -l)"
"""


@pytest.mark.skipif(
    sys.platform == "win32" or BASH is None,
    reason="git-bash on Windows reaps orphaned sleeps differently, so the test cannot catch the leak; Linux only",
)
def test_reap_leaves_no_child_holding_the_pipe(tmp_path):
    t0 = time.time()
    # capture_output uses pipes, so any child inheriting stdout would block until it exits
    out = subprocess.run([BASH, "-c", SCRIPT, "_", tmp_path.as_posix()],
                         capture_output=True, text=True, timeout=30, check=False)
    assert time.time() - t0 < 10  # a surviving sleep would hold the pipe for 300 s
    assert "alive: 0" in out.stdout
