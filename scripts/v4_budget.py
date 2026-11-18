"""V4 budget in Kaggle quota hours (docs/V4_FEASIBILITY.md, sections 2-3).

1 quota hour = 1 wall-clock hour of a T4x2 session (observed on the account; doc section 2).
Every job is a PAIR: StreetCLIP on cuda:0 and L/14-336 on cuda:1 side by side, so a pair's
wall time is what the quota pays. Rates are inputs: replace them with the pilot's numbers.

  uv run python scripts/v4_budget.py              # the P/C/O table and the break-even rate
  uv run python scripts/v4_budget.py --rate 13.5  # one measured rate (slower backbone, sustained)
  uv run python scripts/v4_budget.py --seen 200000  # shorter runs; with N = 200K this is one epoch
                                                    # (the option labels still say "500K" for the large N)
"""

import argparse
import math

EVAL = 33.0                        # img/s per T4, decode included (measured, V2 pilot)
DEV_SEEN, DEV_UNSEEN, TEST = 19906, 20000, 210122
SUB = 5000                         # dev_unseen subset for the intermediate (diagnostic) evals
S = 500_000                        # images seen per full run: 10 epochs at 50K, 1 epoch at 500K
SHORT = 100_000                    # images seen per short-horizon search run (B-S2)
SESSION, OVH, PILOT, WEEK = 11.0, 0.25, 2.5, 30.0   # usable h per session, setup h, pilot h, quota h/week
BUDGET = 60.0


def evals_h(intermediate=2):
    """Final checkpoint on dev_unseen (selection) + dev_seen (diagnostic), plus intermediates on a subset."""
    return (DEV_UNSEEN + DEV_SEEN + intermediate * SUB) / EVAL / 3600


def full(r):
    return S / r / 3600 + evals_h()


def short(r):
    return SHORT / r / 3600 + (DEV_UNSEEN + DEV_SEEN) / EVAL / 3600


def fwd(n):                        # a forward-only pair job over n images (control, hybrid, test)
    return n / EVAL / 3600


def quota(jobs):
    """Each pair job uses whole sessions of <= SESSION h (a longer job resumes in the next one)."""
    return sum(j + math.ceil(j / SESSION) * OVH for j in jobs) + PILOT


def options(r):
    ctrl = fwd(S) + evals_h()      # frozen encoder, head only, same augmented 500K schedule, 50K
    hyb50, test2 = fwd(50_000), fwd(TEST)          # test2: one pair = 2 models on test
    return {
        "A    50K: 3 lr + edge + seed2 + ctrl, hybrid, test (2 models)":
            [full(r)] * (3 + 1 + 1) + [ctrl, hyb50, test2],
        "B-full  3 lr + edge at both N + ctrl, hybrid, test (4 models)":
            [full(r)] * ((3 + 1) * 2) + [ctrl, hyb50, test2, test2],
        "B-S1  3 lr + edge at 50K, 500K at the 50K lr, ctrl, hybrid, test":
            [full(r)] * (3 + 1 + 1) + [ctrl, hyb50, test2, test2],
        "B-S2  3 lr + edge short (100K seen) at 50K, 2 full, seed2, ctrl, ...":
            [short(r)] * (3 + 1) + [full(r)] * (2 + 1) + [ctrl, hyb50, test2, test2],
        "C    B-S1 + seed2 at 50K + 500K hybrid":
            [full(r)] * (3 + 1 + 1 + 1) + [ctrl, hyb50, fwd(500_000), test2, test2],
    }


def report(r):
    lost = min(full(r), SESSION) + OVH              # reserve: one failed session of the longest job
    print(f"rate {r:5.1f} img/s per T4: full pair {full(r):5.2f} h, short pair {short(r):4.2f} h, "
          f"lost-session reserve {lost:4.1f} h")
    for name, jobs in options(r).items():
        q = quota(jobs)
        print(f"  {name:68s} {q:6.1f} h  +reserve {q + lost:6.1f} h  ({(q + lost) / WEEK:.1f} weeks)")


def break_even(name, budget=BUDGET):
    """Lowest rate at which option `name` plus the lost-session reserve fits the budget."""
    for r10 in range(50, 600):
        r = r10 / 10
        key = next(k for k in options(r) if k.startswith(name))
        if quota(options(r)[key]) + min(full(r), SESSION) + OVH <= budget:
            return r
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rate", type=float, nargs="+", default=[9.0, 12.0, 17.0],
                    help="training img/s per T4 (default: the doc's P, C, O)")
    ap.add_argument("--seen", type=int, default=S, help="images seen per full run")
    a = ap.parse_args()
    S = a.seen
    print(f"images seen per full run: {S:,}")
    for r in a.rate:
        report(r)
    for name in ("A ", "B-full", "B-S1", "B-S2"):
        print(f"break-even rate for {name.strip()} within {BUDGET:.0f} h incl. reserve: {break_even(name)} img/s")
