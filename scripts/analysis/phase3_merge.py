"""Phase 3, DEVELOPMENT only: merge the final cell probes and hybrids of separate Kaggle sessions
into one phase3_eval dir and verify it. Nothing is patched: every mismatch is listed and the
script exits 1.

  final cell probes: random from --cells-random (session C), zero-shot from --cells-zeroshot
  (the 3e-4 extension); hybrids from --hybrid-random and --hybrid-zeroshot.

Checks: files copied byte for byte; each cell's cache_key == cache_key(its stored cache_settings);
each hybrid's cache_key == cache_key(settings + probe_source of the MERGED cell probe), i.e. it was
built from the final probe; stored settings == the current code's constants and defaults; the
frozen geocell sha256 everywhere; per backbone the same data config / embedding cache as the Step 1
kNN cells (--knn-run). The data dir and the embeddings are not local, so their hashes are checked
for consistency across cells, not recomputed.

  uv run python scripts/analysis/phase3_merge.py --cells-random runs/phase3_session_c/phase3_eval \
      --cells-zeroshot runs/phase3_extend/phase3_eval \
      --hybrid-random runs/phase3_hybrid_random/phase3_eval \
      --hybrid-zeroshot runs/phase3_hybrid_zeroshot/phase3_eval \
      --knn-run runs/phase3_dev/phase3_eval --out runs/phase3_final_dev/phase3_eval
"""

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_eval as p3
from geoguessr_ai import prereg
from geoguessr_ai.train import PROBE_BATCH

ORDERINGS, SIZES = ("nat", "bal"), (50_000, 100_000, 200_000, 500_000)
SHA_PREFIX = "9882eb2a"
SHARED = ("data_config_sha256", "emb_encoder_sha256", "emb_files")  # per backbone


def expected(method: str, init: str) -> dict:
    """The settings fields the current code and defaults give (data/emb hashes excluded)."""
    a = p3.parse_args(["--data", ".", "--emb-root", "."])
    s = {"cache_version": p3.CACHE_VERSION, "method": method, "seed": a.seed, "n_boot": a.n_boot,
         "cell_target": 5000, "probe_lrs": list(a.probe_lrs), "probe_epochs": a.probe_epochs,
         "probe_patience": a.probe_patience, "label_smoothing": a.label_smoothing,
         "logit_scale": a.logit_scale, "template": a.template, "probe_batch": PROBE_BATCH}
    if method == "cell" and p3.EXTRA_LRS.get(init):
        s["extra_lrs"] = list(p3.EXTRA_LRS[init])
    if method == "hybrid":  # the CPU session: --device ''
        s |= {"mask_km": a.mask_km, "knn_device": "cpu", "knn_dtype": a.knn_dtype,
              "hybrid_k": a.hybrid_k, "hybrid_m": list(p3.HYBRID_M),
              "hybrid_predictor": a.hybrid_predictor, "k": list(a.k), "taus": list(p3.TAUS)}
    return s


def main():
    ap = argparse.ArgumentParser()
    for f in ("cells-random", "cells-zeroshot", "hybrid-random", "hybrid-zeroshot", "knn-run", "out"):
        ap.add_argument(f"--{f}", type=Path, required=True)
    a = ap.parse_args()
    src = {"cell_random": a.cells_random, "cell_zeroshot": a.cells_zeroshot,
           "hybrid_random": a.hybrid_random, "hybrid_zeroshot": a.hybrid_zeroshot}
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty")
    bad = []

    rec_files = [s / "cell_count.json" for s in src.values()]
    geo_files = [s / "geocells" / "cells_5000.json" for s in src.values()]
    for fs in (rec_files, geo_files):
        bad += [f"{f} differs from {fs[0]}" for f in fs[1:] if not filecmp.cmp(fs[0], f, shallow=False)]
    rec = json.loads(rec_files[0].read_text())
    want_sha = rec["geocells_sha256"]
    if not want_sha.startswith(SHA_PREFIX) or prereg.sha256(geo_files[0]) != want_sha:
        bad.append(f"geocell sha {want_sha[:16]} / file {prereg.sha256(geo_files[0])[:16]}")
    (a.out / "geocells").mkdir(parents=True)
    shutil.copy2(rec_files[0], a.out / "cell_count.json")
    shutil.copy2(geo_files[0], a.out / "geocells" / "cells_5000.json")

    cells = [(bb, o, n, v) for bb in pe.BACKBONES for o in ORDERINGS for n in SIZES for v in src]
    for bb, o, n, v in cells:
        s, d = (pe.cell_dir(r, bb, o, n, p3.METHOD, v) for r in (src[v], a.out))
        shutil.copytree(s, d)
        cmp = filecmp.dircmp(s, d)
        if cmp.left_only or cmp.right_only or filecmp.cmpfiles(s, d, cmp.common_files, shallow=False)[1:] != ([], []):
            bad.append(f"{d}: copy differs from {s}")

    knn = {}
    for bb in pe.BACKBONES:
        gm = json.loads((pe.cell_dir(a.knn_run, bb, "nat", SIZES[0], p3.METHOD, "medoid")
                         / "geo_metrics.json").read_text())
        knn[bb] = {k: gm["cache_settings"][k] for k in SHARED}
    n_hyb = 0
    for bb, o, n, v in cells:
        d = pe.cell_dir(a.out, bb, o, n, p3.METHOD, v)
        row, gm = pe.read_row(d), json.loads((d / "geo_metrics.json").read_text())
        if any(k.startswith("test") for k in [*row, *gm]):
            raise SystemExit(f"{d} holds test results; development only")
        cs, (method, init) = gm["cache_settings"], v.split("_")
        if method == "cell":
            key = p3.cache_key(cs)
        else:
            src_ = p3.probe_source(a.out, bb, o, n, init)
            key = None if src_ is None else p3.cache_key({**cs, "probe_source": src_})
            n_hyb += key == row["cache_key"]
        if key != row["cache_key"]:
            bad.append(f"{d}: cache_key {row['cache_key']} != recomputed {key}")
        exp = expected(method, init)
        diff = {k: (cs.get(k), w) for k, w in exp.items() if cs.get(k) != w}
        if method == "cell" and "extra_lrs" in cs and "extra_lrs" not in exp:
            diff["extra_lrs"] = (cs["extra_lrs"], None)
        if diff:
            bad.append(f"{d}: settings differ from current code {diff}")
        if cs.get("geocells_sha256") != want_sha or row.get("cell_target") != 5000:
            bad.append(f"{d}: geocells {str(cs.get('geocells_sha256'))[:16]}, target {row.get('cell_target')}")
        if {k: cs.get(k) for k in SHARED} != knn[bb]:
            bad.append(f"{d}: data config / embedding cache differs from the kNN cells")
    print(f"{len(cells)} cell dirs merged into {a.out}; geocells sha256 {want_sha[:16]}; "
          f"{n_hyb} of {len(cells) // 2} hybrids built from the merged cell probe")
    if bad:
        print(f"{len(bad)} MISMATCH(ES):\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
