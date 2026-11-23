"""Read-only check of recipe B's full runs, jobs 4 and 5 (dev only; exit 1 on any failure).

  uv run python scripts/analysis/v4_b_verify.py [--runs runs/v4_B_session4/v4_runs]
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geoguessr_ai.v4_inputs import file_sha256, read_csv

R = Path(__file__).resolve().parents[2]
COMMIT = "9db076b5f9aeb126b920dafae83dfe1483c054a1"
JOBS = ((4, "B_full_50k", 50_000), (5, "B_full_500k", 500_000))
BACKBONES = ("streetclip", "clip_l14_336")
LR = 3.33333e-05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=R / "runs/v4_B_session4/v4_runs")
    ap.add_argument("--bundle", type=Path, default=R / "runs/v4_inputs/v4_inputs")
    a = ap.parse_args()
    manifest = dict(reversed(l.split()) for l in (a.bundle / "MANIFEST.sha256").read_text().splitlines())
    dev = {s: read_csv(a.bundle / f"{s}.csv") for s in ("dev", "dev_seen")}
    fails = []

    def check(ok, msg):
        print(("ok   " if ok else "FAIL ") + msg)
        if not ok:
            fails.append(msg)

    for k, name, n in JOBS:
        for b in BACKBONES:
            d = a.runs / f"k{k}_{name}_{b}"
            r = json.loads((d / "result.json").read_text())
            print(f"--- {d.name}")
            w = file_sha256(d / "weights_final.pt")
            check((d / "DONE").read_text().strip() == w == r["weights_final_sha256"], f"DONE = weights = result ({w[:12]})")
            check(r["code_commit"] == COMMIT, "commit")
            check(r["head_frozen"] is True and r["head_lr"] is None, "head frozen")
            rel = f"heads/{b}/bal_{n}_cell_zeroshot.npy"
            check(r["head"] == manifest[rel] == file_sha256(a.bundle / rel), f"head = bundle ({r['head'][:12]})")
            v3 = {file_sha256(p) for p in R.glob(f"runs/phase3_*/phase3_probes/{b}/bal_{n}_geo_cell_zeroshot/head.npy")}
            check(r["head"] in v3, "head = V3 head.npy")
            check(r["n"] == n and r["seed"] == 42 and abs(r["lr"] - LR) < 1e-12 and r["steps"] == 7812, "n, seed, lr, steps")
            check(r["gpu"]["index"] == (k % 2 if b == "streetclip" else 1 - k % 2), "GPU parity")
            sd = torch.load(d / "weights_final.pt", map_location="cpu", weights_only=True)
            check(sd and all("lora" in x.lower() for x in sd), f"{len(sd)} tensors, LoRA only")
            check(all(torch.isfinite(v.float()).all() for v in sd.values()), "weights finite")
            ev = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()]
            print("     events", dict(collections.Counter(e.get("event") for e in ev)))
            check(all(math.isfinite(e["loss"]) for e in ev if e.get("loss") is not None), "losses finite")
            for s in dev:
                p = pd.read_csv(d / "final" / f"{s}_predictions.csv", dtype={"id": str})
                check(len(p) == len(dev[s]) and set(p["id"]) == set(dev[s]["id"]) and p["id"].is_unique, f"{s}: ids")
                gs = (5000 * np.exp(-p["km"] / 1492.7)).mean()
                check(abs(gs - r[s]["geoscore"]["mean"]) < 1e-6, f"{s}: GeoScore {gs:.2f}")
            hub = (d / "hub.log").read_text()
            check("weights_final.pt" in hub and "result.json" in hub, "uploaded")
    print("--- seeded chain (jobs 0-3 unchanged)")
    for kd in sorted(a.runs.glob("k[0-3]_*")):
        src = list(R.glob(f"runs/v4_B_session[123]/**/{kd.name}/DONE"))
        check(bool(src) and all(p.read_text() == (kd / "DONE").read_text() for p in src), kd.name)
    print("FAILS:", len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
