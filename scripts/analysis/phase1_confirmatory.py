"""Phase 1 final test evaluation, confirmatory part: the pre-registered primary results exactly as
phase1_eval wrote them (primary.csv), plus the run's self-checks from its log (test embedding
checks, dev reproduction). Nothing is recomputed.

  uv run python scripts/analysis/phase1_confirmatory.py --run runs/phase1_final
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="downloaded Kaggle output")
    ap.add_argument("--out", type=Path, default=Path("results/v2"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ev = args.run / "phase1_eval_test"
    prim = pd.read_csv(ev / "primary.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    prim.to_csv(args.out / "phase1_primary.csv", index=False)
    show = prim.assign(**{c: (100 * prim[c]).round(2) for c in ("estimate", "ci_lo", "ci_hi")})
    print("PRIMARY (pre-registered; estimate and CI in percentage points of test top-1)")
    print(show[["id", "estimate", "ci_lo", "ci_hi", "p", "p_holm", "verdict"]].to_string(index=False))
    print(f"supported: {(prim['verdict'] == 'supported').sum()} of {len(prim)}")
    log = (args.run / "final_eval.log").read_text(encoding="utf-8", errors="replace")
    for pat in (r".*-> (PASSED|FAILED)", r"(dev_seen|dev_unseen) top-1 vs pre-registration.*"):
        for m in re.finditer(pat, log):
            print(m.group(0))


if __name__ == "__main__":
    main()
