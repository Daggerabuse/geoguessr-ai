"""Pre-registration of the one-shot Phase 1 test evaluation.

configs/phase1_final.json fixes, before any test image is embedded: the cells reported on test
and their hyperparameters, copied from the dev_seen selection of a finished dev run (no
re-tuning), plus the head settings. Order of events, each enforced:

  write        python -m geoguessr_ai.prereg write --dev-out <dev phase1_eval dir>
               -> configs/phase1_final.json (every cell of results.csv; drop cells by editing
               the file before committing it). Refuses dev rows of an outdated cache version.
  commit       git commit configs/phase1_final.json (and push: Kaggle clones the repo)
  provenance   kaggle_phase1.sh test runs this before reading any test image: the file must be
               committed and unmodified; records its sha256, last commit and the time
               -> emb_test/prereg_provenance.json
  check        phase1_eval --eval-test: the file is still committed and unmodified, its sha256
               equals the provenance's, and its last commit predates the embedding run.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("configs/phase1_final.json")
SETTINGS = ("template", "label_smoothing", "logit_scale", "seed", "n_boot")
PRIMARY_ALPHA, PRIMARY_N_BOOT = 0.05, 10_000  # family-wise alpha; resamples per primary test


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state(path: Path) -> tuple[str, int]:
    """(last commit touching path, its committer time); SystemExit unless the file is committed
    and has no uncommitted change."""
    path = Path(path).resolve()
    if not path.exists():
        raise SystemExit(f"pre-registration missing: {path}")
    run = lambda *a: subprocess.run(["git", "-C", str(path.parent), *a], capture_output=True,
                                    text=True, check=False)
    log = run("log", "-1", "--format=%H %ct", "--", path.name)
    if log.returncode or not log.stdout.strip():
        raise SystemExit(f"pre-registration not committed: {path}")
    dirty = run("status", "--porcelain", "--", path.name)
    if dirty.returncode or dirty.stdout.strip():
        raise SystemExit(f"pre-registration has uncommitted changes: {path}")
    commit, ct = log.stdout.split()
    return commit, int(ct)


def hypotheses(cells: list[dict], pair: tuple[str, str]) -> dict:
    """Primary hypotheses on test, per variant x ordering with both backbones of the pair at
    the smallest and the largest declared N (50K and 500K for the full grid):
      H1  gap(N_min) = top-1(pair[0]) - top-1(pair[1]) > 0   (paired over the same images)
      H2  shrink = gap(N_min) - gap(N_max) > 0
    Everything else is secondary/exploratory."""
    have = {(c["variant"], c["ordering"], c["n"], c["backbone"]) for c in cells}
    ns = sorted({c["n"] for c in cells})
    both = lambda v, o, n: all((v, o, n, b) in have for b in pair)
    primary = []
    for v, o in sorted({(c["variant"], c["ordering"]) for c in cells}):
        lo, hi = ns[0], ns[-1]
        if both(v, o, lo):
            primary.append({"id": f"H1/{v}/{o}", "quantity": "gap", "variant": v, "ordering": o,
                            "n": lo, "claim": f"{pair[0]} - {pair[1]} test top-1 at N={lo} > 0"})
        if lo != hi and both(v, o, lo) and both(v, o, hi):
            primary.append({"id": f"H2/{v}/{o}", "quantity": "shrink", "variant": v,
                            "ordering": o, "n_small": lo, "n_large": hi,
                            "claim": f"gap(N={lo}) - gap(N={hi}) on test > 0 (the gap shrinks)"})
    return {"criterion": "supported if the Holm-adjusted one-sided p-value < alpha; the "
                         "uncorrected 95% percentile interval is reported with each",
            "p_value": "one-sided, from n_boot paired cluster-bootstrap resamples of the test "
                       "statistic (sequence clusters; one resample applies to both backbones and "
                       "both N): p = (1 + #{resampled statistic <= 0}) / (n_boot + 1)",
            "multiplicity": "Holm-Bonferroni over all primary hypotheses (family-wise alpha); a "
                            "hypothesis whose cells were not run is not evaluable and counts "
                            "with p = 1",
            "alpha": PRIMARY_ALPHA, "n_boot": PRIMARY_N_BOOT,
            "primary": primary,
            "secondary": "every other output (other N, dev splits, top-5, macro, head/middle/"
                         "tail, reliable, seen/unseen creators, B/16, geolocation metrics) is "
                         "exploratory"}


def write(dev_out: Path, path: Path, args: argparse.Namespace) -> dict:
    """Freeze every cell of dev_out/results.csv with its dev_seen-selected hyperparameters."""
    from geoguessr_ai import phase1_eval as pe

    results = Path(dev_out) / "results.csv"
    rows = pd.read_csv(results)
    version = rows.get("cache_version", pd.Series(1, index=rows.index)).fillna(1)
    stale = rows[version != rows["method"].map(pe.CACHE_VERSION)]  # rows before versioning: 1
    if len(stale):
        raise SystemExit(f"{len(stale)} rows of {results} predate the current cache version; "
                         "rerun phase1_eval first")
    cells = []
    for r in rows.to_dict("records"):
        cell = {k: r[k] for k in ("backbone", "ordering", "method", "variant")}
        cell |= {"n": int(r["n"]), "probe_init": r["probe_init"] if r["method"] == "probe" else "",
                 "dev_seen_top1": r["dev_seen_top1"], "dev_unseen_top1": r["dev_unseen_top1"],
                 "grid_edge": "" if pd.isna(r.get("grid_edge")) else r["grid_edge"]}
        if r["method"] == "knn":
            m = json.loads((pe.cell_dir(Path(dev_out), r["backbone"], r["ordering"], int(r["n"]),
                                        "knn") / "geo_metrics.json").read_text())
            cell |= {"k_country": int(m["k_country"]), "k_coords": int(m["k_coords"])}
        else:
            cell |= {"lr": float(r["lr"]), "epoch": int(r["epoch"])}
        cells.append(cell)
    reg = {"note": "frozen before the test embeddings; hyperparameters are the dev_seen-selected "
                   "values of dev_results, used as they are (no re-tuning on test)",
           "dev_results": str(results), "dev_results_sha256": sha256(results),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "settings": {k: getattr(args, k) for k in SETTINGS},
           "search": {"k": args.k, "probe_lrs": args.probe_lrs, "probe_epochs": args.probe_epochs,
                      "probe_patience": args.probe_patience, "rule": pe.PROBE_GRID_RULE,
                      "grid_edge": "per cell: a choice at an edge of this search grid"},
           "hypotheses": hypotheses(cells, pe.PAIR), "cells": cells}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2) + "\n")
    return reg


def provenance(path: Path, out: Path) -> dict:
    commit, ct = git_state(path)
    rec = {"prereg_sha256": sha256(path), "prereg_commit": commit, "prereg_commit_time": ct,
           "embedding_started": time.time(),
           "embedding_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if ct >= rec["embedding_started"]:
        raise SystemExit("pre-registration commit is not older than now (clock?)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def check(path: Path, provenance_path: Path) -> dict:
    """The pre-registration, if it is committed, unmodified, and the one recorded by the test
    embedding run, which it predates; SystemExit otherwise."""
    _, ct = git_state(path)
    if not Path(provenance_path).exists():
        raise SystemExit(f"no {provenance_path}: the test embeddings were not made by "
                         "kaggle_phase1.sh test, so the pre-registration cannot be dated")
    rec = json.loads(Path(provenance_path).read_text())
    if rec["prereg_sha256"] != sha256(path):
        raise SystemExit("pre-registration differs from the one recorded when the test "
                         "embeddings were made")
    if not ct < rec["embedding_started"]:
        raise SystemExit("pre-registration commit does not predate the test embeddings")
    return json.loads(Path(path).read_text())


def main(argv=None) -> None:
    from geoguessr_ai import phase1_eval as pe

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--dev-out", type=Path, required=True, help="phase1_eval output of the dev run")
    w.add_argument("--prereg", type=Path, default=DEFAULT_PATH)
    p = sub.add_parser("provenance")
    p.add_argument("--prereg", type=Path, default=DEFAULT_PATH)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "write":
        # head settings: phase1_eval's defaults (the dev run's settings)
        reg = write(args.dev_out, args.prereg, pe.parse_args(["--data", "-", "--emb-root", "-"]))
        print(f"{args.prereg}: {len(reg['cells'])} cells; commit it before the test embedding run")
    else:
        print(json.dumps(provenance(args.prereg, args.out), indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
