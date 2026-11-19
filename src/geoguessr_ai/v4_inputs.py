"""The V4 inputs bundle: every small file a V4 GPU session reads, and nothing else.

A V4 training session attaches only this bundle (and its own previous output), so no test file
is ever attached (docs/V4_FEASIBILITY.md, section 5). Built once in a CPU session from the Phase 1
full run's output (v2/ manifests + V3 embeddings) and the V3 probe output (heads):

  train.csv      train rows with bal_rank < 500,000 (the bal_50K and bal_500K ids hashes are
                 checked against v2/config.json, whose sha256 is V3's pre-registered one)
  dev.csv, dev_seen.csv   copied byte for byte (ids hashes checked)
  cells_5000.json         the frozen partition (sha256 = V3's)
  heads/<bb>/bal_<n>_cell_zeroshot.npy   V3's bal zero-shot cell heads (sha256 = repro.json's)
  identity/<bb>_dev2k.npz  V3 embeddings of the 2K dev_unseen diagnostic subset (identity check)
  MANIFEST.sha256          "<sha256>  <path>" per file; its own sha256 goes into the pre-dev config

test.csv and emb_test are never opened; a bundle file whose name contains "test" is refused.

Usage (Kaggle CPU session, runs/kaggle_v4_inputs.ipynb):
  python -m geoguessr_ai.v4_inputs --v2 <p1>/v2 --emb-root <p1>/emb --probes <probes>/phase3_probes \\
      --geocells <dev run>/geocells/cells_5000.json --out /kaggle/working/v4_inputs
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKBONES = ("streetclip", "clip_l14_336")
HEAD_NS = (50_000, 500_000)
BAL_MAX = 500_000
TRAIN_COLS = ["id", "latitude", "longitude", "country", "sequence", "creator_id", "label", "bal_rank"]
CSV_DTYPES = {"id": str, "sequence": str, "creator_id": str}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 23):
            h.update(b)
    return h.hexdigest()


def dev_subset(ids, n: int, seed: int = 42) -> list[str]:
    """The first n of a seeded permutation of the sorted ids: nested in n (2K inside 5K)."""
    ids = sorted(map(str, ids))
    return [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))[:n]]


def read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False: "NA" is Namibia, not a missing value
    return pd.read_csv(path, dtype=CSV_DTYPES, keep_default_na=False)


def write_manifest(out: Path) -> str:
    """MANIFEST.sha256 over every other file in out; returns the manifest's own sha256."""
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    for p in files:
        if "test" in p.relative_to(out).as_posix().lower():
            raise SystemExit(f"refusing a test file in the V4 inputs bundle: {p}")
    lines = [f"{file_sha256(p)}  {p.relative_to(out).as_posix()}" for p in files]
    (out / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", newline="\n")
    return file_sha256(out / "MANIFEST.sha256")


def verify_bundle(root: Path, want_sha: str | None) -> dict[str, str]:
    """Check MANIFEST.sha256 (its own sha256 against the pre-dev config, then every listed file)
    and that the bundle holds nothing unlisted. Returns {relative path: sha256}."""
    root = Path(root)
    man = root / "MANIFEST.sha256"
    if not man.exists():
        raise SystemExit(f"{root}: no MANIFEST.sha256 (not a V4 inputs bundle)")
    have = file_sha256(man)
    if want_sha is not None and have != want_sha:
        raise SystemExit(f"{man}: sha256 {have[:16]} is not the pre-dev config's {want_sha[:16]}")
    listed = {}
    for line in man.read_text().splitlines():
        if line:
            sha, rel = line.split("  ", 1)
            listed[rel] = sha
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*")
               if p.is_file() and p.name != "MANIFEST.sha256"}
    if on_disk != set(listed):
        raise SystemExit(f"{root}: files differ from MANIFEST.sha256 "
                         f"(extra {sorted(on_disk - set(listed))}, missing {sorted(set(listed) - on_disk)})")
    for rel, sha in listed.items():
        if "test" in rel.lower():
            raise SystemExit(f"{root}: a test file is listed in the bundle: {rel}")
        if file_sha256(root / rel) != sha:
            raise SystemExit(f"{root / rel}: sha256 differs from MANIFEST.sha256")
    return listed


def build(args: argparse.Namespace) -> str:
    from geoguessr_ai import geocells as gc
    from geoguessr_ai.embed import load_embeddings
    from geoguessr_ai.v2_data import ids_hash

    reg = json.loads(Path(args.prereg).read_text())
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    v2, emb = Path(args.v2), Path(args.emb_root)
    if "test" in emb.name:
        raise SystemExit(f"--emb-root {emb}: the training embeddings, never emb_test")

    cfg_path = v2 / "config.json"
    if file_sha256(cfg_path) != reg["settings"]["data_config_sha256"]:
        raise SystemExit(f"{cfg_path}: not V3's pre-registered data config")
    cfg = json.loads(cfg_path.read_text())

    train = read_csv(v2 / "train.csv")
    for n in HEAD_NS:  # V4 trains on exactly V3's subsets
        got = ids_hash(sorted(train.loc[train["bal_rank"] < n, "id"]))
        if got != cfg["subsets"][f"bal_{n}"]["ids_sha256"]:
            raise SystemExit(f"bal_{n}: ids hash {got} differs from v2/config.json")
    train = train.loc[train["bal_rank"] < BAL_MAX, TRAIN_COLS].sort_values("bal_rank")
    train.to_csv(out / "train.csv", index=False)

    for name in ("dev", "dev_seen"):
        df = read_csv(v2 / f"{name}.csv")
        if ids_hash(sorted(df["id"])) != cfg["ids_sha256"][name]:
            raise SystemExit(f"{name}.csv: ids hash differs from v2/config.json")
        shutil.copyfile(v2 / f"{name}.csv", out / f"{name}.csv")

    tree, sha = gc.load(Path(args.geocells))
    if sha != reg["geocells"]["sha256"]:
        raise SystemExit(f"{args.geocells}: sha256 {sha[:16]} is not the frozen partition's")
    shutil.copyfile(args.geocells, out / "cells_5000.json")

    probes = Path(args.probes)
    repro = json.loads((probes / "repro.json").read_text())
    want = {(c["backbone"], c["ordering"], c["n"], c["predictor"]): c["head_sha256"]
            for c in repro["cells"]}
    dev_ids = dev_subset(read_csv(v2 / "dev.csv")["id"], args.identity_n, args.seed)
    for bb in BACKBONES:
        enc = emb / bb / "encoder.json"
        if file_sha256(enc) != reg["emb_encoder_sha256"][bb]:
            raise SystemExit(f"{enc}: not V3's encoder record")
        for n in HEAD_NS:
            src = probes / bb / f"bal_{n}_geo_cell_zeroshot" / "head.npy"
            if file_sha256(src) != want[bb, "bal", n, "cell_zeroshot"]:
                raise SystemExit(f"{src}: sha256 differs from repro.json")
            dst = out / "heads" / bb / f"bal_{n}_cell_zeroshot.npy"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        e, found = load_embeddings(emb / bb, dev_ids, "train")
        if not found.all():
            raise SystemExit(f"{bb}: {(~found).sum()} identity ids not embedded")
        (out / "identity").mkdir(exist_ok=True)
        np.savez(out / "identity" / f"{bb}_dev2k.npz", ids=np.array(dev_ids), emb=e)

    sha = write_manifest(out)
    print(f"V4 inputs bundle {out}: MANIFEST.sha256 = {sha}\n"
          "put it into configs/v4_predev.json (inputs.bundle_manifest_sha256) before committing")
    return sha


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v2", required=True, help="the Phase 1 full run's v2/ (train, dev, dev_seen, config)")
    ap.add_argument("--emb-root", required=True, help="its emb/ (training-source caches, dev included)")
    ap.add_argument("--probes", required=True, help="V3's phase3_probes dir (repro.json + heads)")
    ap.add_argument("--geocells", required=True, help="the frozen cells_5000.json")
    ap.add_argument("--prereg", default="configs/phase3_final.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    build(parse_args(argv))


if __name__ == "__main__":
    main()
