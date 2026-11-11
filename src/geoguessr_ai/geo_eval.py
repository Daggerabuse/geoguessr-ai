"""V2/V3 heads on cached embeddings: kNN retrieval (country + coordinates) and a linear probe.

For each training-subset size N (nested, see v2_data.py) and head:
  - hyperparameters (k for kNN, the probe epoch) are chosen on dev_seen only (creators seen
    in train, like 98% of the official test; see v2_data.py);
  - dev_seen and dev (creators unseen in train, second report) are scored; the official test
    split is read and scored only with --eval-test (final configurations), so development
    runs never touch it. The summary has one row per scored split.
Every split is also reported per row group: countries with training data, reliable
countries, head/middle/tail (country_counts.csv), photographer seen/unseen in the training
subset. Groups select rows only; predictions always range over the full class list.
Country metrics use the same files as baseline_clip/train (write_split_results).
Distance metrics follow OSV-5M: haversine km, GeoScore = 5000 * exp(-km / 1492.7),
accuracy within 1/25/200/750/2500 km. Intervals: cluster bootstrap over sequences
(frames of one sequence are not independent); for dev (unseen creators) over creators. Test images whose creator also appears in
the training subset are reported separately ("seen" / "unseen" creator).

Usage (Kaggle):
  python -m geoguessr_ai.geo_eval --data /kaggle/working/v2 --emb /kaggle/working/emb/clip_b16 \\
      --model openai/clip-vit-base-patch16 --tag clip_b16 --strategy bal --sizes 50000 200000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai.baseline_clip import (
    DEFAULT_TEMPLATE,
    as_embedding,
    build_prompts,
    evaluate,
    library_versions,
    write_split_results,
)

EARTH_RADIUS_KM = 6371.0088
KM_THRESHOLDS = (1, 25, 200, 750, 2500)
DEFAULT_K = (1, 5, 10, 25, 50, 100)


# ----------------------------------------------------------------- metrics (numpy only)


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (inputs in degrees, broadcastable)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geoscore(km) -> np.ndarray:
    return 5000 * np.exp(-np.asarray(km) / 1492.7)


def per_image_geo(km: np.ndarray) -> dict[str, np.ndarray]:
    out = {"geoscore": geoscore(km)}
    out.update({f"acc_{t}km": (km <= t).astype(float) for t in KM_THRESHOLDS})
    return out


def cluster_bootstrap(values: np.ndarray, clusters, n_boot: int = 1000,
                      seed: int = 42) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling whole clusters with replacement."""
    codes, _ = pd.factorize(pd.Series(clusters))
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(sums), len(sums)), minlength=len(sums))
                  for _ in range(n_boot)])
    means = (w @ sums) / (w @ counts)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(per_image: dict[str, np.ndarray], km: np.ndarray | None, clusters,
              n_boot: int, seed: int) -> dict:
    """Mean (+ cluster-bootstrap interval) of each per-image metric; median/mean km."""
    out: dict = {"n": len(clusters)}
    if not len(clusters):
        return out
    for name, v in per_image.items():
        lo, hi = cluster_bootstrap(v, clusters, n_boot, seed)
        out[name] = {"mean": float(v.mean()), "ci_lo": lo, "ci_hi": hi}
    if km is not None:
        out["median_km"], out["mean_km"] = float(np.median(km)), float(km.mean())
    return out


# ----------------------------------------------------------------- kNN


def knn(query: np.ndarray, db: np.ndarray, k: int, device: str | None = None):
    """Top-k cosine neighbours (rows are L2-normalized). Returns (sims, idx), best first.

    device: run the matrix products with torch on that device (e.g. "cuda"); None = numpy.
    """
    k = min(k, len(db))
    chunk = 1024 if device else 256  # one chunk of scores: chunk x len(db) values
    sims, idx = [], []
    if device:
        import torch

        dbt = torch.from_numpy(db).to(device, dtype=torch.float16)
    for s in range(0, len(query), chunk):
        if device:
            q = torch.from_numpy(query[s : s + chunk]).to(device, dtype=torch.float16)
            v, i = (q @ dbt.T).topk(k, dim=1)
            sims.append(v.float().cpu().numpy())
            idx.append(i.cpu().numpy())
            continue
        sc = query[s : s + chunk] @ db.T
        part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(sc, part, 1)
        o = np.argsort(-ps, axis=1)
        sims.append(np.take_along_axis(ps, o, 1))
        idx.append(np.take_along_axis(part, o, 1))
    return np.concatenate(sims), np.concatenate(idx)


def knn_country_scores(sims, idx, db_labels, n_classes: int, k: int) -> np.ndarray:
    """Similarity-weighted vote of the k nearest neighbours' countries -> (N, C) scores."""
    scores = np.zeros((len(sims), n_classes))
    rows = np.repeat(np.arange(len(sims)), k)
    np.add.at(scores, (rows, db_labels[idx[:, :k]].ravel()), sims[:, :k].ravel())
    return scores


def knn_coordinates(sims, idx, db_lat, db_lon, k: int, chunk: int = 2048):
    """Geographic medoid of the k neighbours: the neighbour with the smallest similarity-
    weighted sum of distances to the others (robust to a few far-away matches)."""
    lat, lon = db_lat[idx[:, :k]], db_lon[idx[:, :k]]
    best = np.zeros(len(idx), dtype=int)
    for s in range(0, len(idx), chunk):
        la, lo, w = lat[s : s + chunk], lon[s : s + chunk], sims[s : s + chunk, :k]
        d = haversine(la[:, :, None], lo[:, :, None], la[:, None, :], lo[:, None, :])
        best[s : s + chunk] = np.argmin((d * w[:, None, :]).sum(-1), axis=1)
    r = np.arange(len(idx))
    return lat[r, best], lon[r, best]


# ----------------------------------------------------------------- probe (torch)


def text_head(model_name: str, classes: list[str], template: str):
    """(C, D) country-name text embeddings: the zero-shot starting point of the probe."""
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tok = AutoTokenizer.from_pretrained(model_name)
    text = CLIPTextModelWithProjection.from_pretrained(model_name).eval()
    with torch.no_grad():
        t = tok(build_prompts(classes, template), padding=True, return_tensors="pt")
        return as_embedding(text(**t), "text", text.config.projection_dim).float()


def probe_logits(feats, labels, dev_feats, dev_labels, eval_feats: dict, head_init, args, device):
    """Train train.py's probe (cosine head, epoch chosen by top-1 on dev_feats, the
    selection split); return logits per split."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    from geoguessr_ai.train import train_probe

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Parameter(head_init.clone().float())

        def logits_from_embed(self, e):
            return args.logit_scale * e @ F.normalize(self.head, dim=-1).T

    model = Head().to(device)
    t = lambda x: torch.from_numpy(x).to(device)
    history: list[dict] = []
    train_probe(model, t(feats), labels, t(dev_feats), dev_labels, args, device, history)
    with torch.no_grad():
        return {k: model.logits_from_embed(t(v)).cpu().numpy() for k, v in eval_feats.items()}, history


# ----------------------------------------------------------------- run


def row_groups(df: pd.DataFrame, counts: pd.DataFrame, seen_creator) -> dict[str, np.ndarray]:
    """Named boolean row masks. A mask only selects rows: predictions always range over the
    full class list, so a subset result is never a restricted-label-space result."""
    c = df["country"]
    groups = {"all": np.ones(len(df), dtype=bool),
              "class_in_train": (df["label"] >= 0).to_numpy(),
              "reliable": c.map(counts["reliable"]).fillna(False).astype(bool).to_numpy()}
    for g in ("head", "middle", "tail"):
        groups[g] = (c.map(counts["freq_group"]) == g).to_numpy()
    groups["seen_creator"], groups["unseen_creator"] = seen_creator, ~seen_creator
    return groups


def score_split(out: Path, name: str, df: pd.DataFrame, scores: np.ndarray, classes,
                coords, groups: dict[str, np.ndarray], args, clusters) -> dict:
    """Country files (images of countries with training data) + metrics per row group.

    Global country metrics ("all") include images of countries without training data: they
    can never be predicted, so they count as errors (top-1 and top-5). macro_top1 averages
    per-country top-1 over every country in the group (such countries score 0).
    """
    lab = df["label"].to_numpy()
    ok = lab >= 0
    m = write_split_results(out, name, df["id"][ok], lab[ok], scores[ok], classes,
                            n_missing=int((~ok).sum()))
    (out / f"{name}_confusion.png").unlink(missing_ok=True)  # unreadable at ~200 classes; CSV kept
    hit1, hitk = evaluate(scores, lab)[:2]  # label -1 never matches a prediction
    per = {"country_top1": hit1.astype(float), "country_top5": hitk.astype(float)}
    pd.DataFrame({"id": df["id"], "top1": hit1.astype(int), "top5": hitk.astype(int)}) \
        .to_csv(out / f"{name}_hits.csv", index=False)
    km = None
    if coords is not None:
        km = haversine(df["latitude"].to_numpy(), df["longitude"].to_numpy(), *coords)
        per.update(per_image_geo(km))
        pd.DataFrame({"id": df["id"], "pred_lat": coords[0], "pred_lon": coords[1], "km": km}) \
            .to_csv(out / f"{name}_geo_predictions.csv", index=False)
    clusters, country = np.asarray(clusters), df["country"].to_numpy()
    res: dict = {"country_files": m}
    for g, mask in groups.items():
        r = summarize({k: v[mask] for k, v in per.items()}, None if km is None else km[mask],
                      clusters[mask], args.n_boot, args.seed)
        if mask.any():
            r["macro_top1"] = float(pd.Series(hit1[mask]).groupby(country[mask]).mean().mean())
            r["n_countries"] = len(set(country[mask]))
        res[g] = r
    if "class_in_subset" in groups:
        # fixed class set (every country of the split), comparable across N; a country absent
        # from the training subset keeps its actual accuracy (a zero-shot head can predict it)
        res["macro_top1_fixed"] = float(pd.Series(hit1).groupby(country).mean().mean())
    return res


def pick(results: dict, metric: str) -> int:
    return max(results, key=lambda k: results[k][metric])


def clusters_of(name: str, df: pd.DataFrame) -> np.ndarray:
    """Bootstrap clusters: sequences; for dev (creators unseen in train) whole creators, since
    a new photographer's images succeed or fail together."""
    return df["creator_id" if name == "dev" else "sequence"].to_numpy()


def load_data(data: Path, emb: Path, eval_test: bool) -> dict:
    """classes, counts, dfs and embeddings of train + the scored splits (dev_seen first: the
    selection split). The official test is read only with eval_test."""
    from geoguessr_ai.embed import load_embeddings

    classes = json.loads((data / "classes.json").read_text())
    # keep_default_na=False everywhere: "NA" is Namibia, not a missing value
    counts = pd.read_csv(data / "country_counts.csv", index_col=0, keep_default_na=False)
    splits = ("train", "dev_seen", "dev", "test") if eval_test else ("train", "dev_seen", "dev")
    dfs, embs = {}, {}
    for name in splits:
        df = pd.read_csv(data / f"{name}.csv", dtype={"id": str, "sequence": str, "creator_id": str},
                         keep_default_na=False)
        e, found = load_embeddings(emb, df["id"], "test" if name == "test" else "train")
        if not found.all():
            print(f"WARNING: {name}: {(~found).sum()} of {len(df)} ids not embedded, dropped")
        dfs[name], embs[name] = df[found].reset_index(drop=True), e[found]
    evals = splits[1:]
    for name in evals:  # the kNN database / probe data must never contain evaluated images
        for col in ("id", "sequence"):
            if set(dfs["train"][col]) & set(dfs[name][col]):
                raise AssertionError(f"train and {name} share {col}s")
    return {"classes": classes, "counts": counts, "dfs": dfs, "embs": embs, "evals": evals}


def run_cell(ctx: dict, strategy: str, n: int, head: str, out: Path, args, head_init=None) -> dict:
    """One training subset (strategy, n) and head: choose hyperparameters on dev_seen, score
    every split of ctx["evals"]; files under out. Returns the metrics dict."""
    classes, dfs, embs, evals = ctx["classes"], ctx["dfs"], ctx["embs"], ctx["evals"]
    train, sel = dfs["train"], dfs["dev_seen"]
    sub_mask = (train[f"{strategy}_rank"] < n).to_numpy()  # nested subset, see v2_data
    sub, db = train[sub_mask], embs["train"][sub_mask]
    creators, lab = set(sub["creator_id"]), sub["label"].to_numpy()
    groups = {}
    for s in evals:
        groups[s] = row_groups(dfs[s], ctx["counts"], dfs[s]["creator_id"].isin(creators).to_numpy())
        groups[s]["class_in_subset"] = dfs[s]["label"].isin(set(lab)).to_numpy()
        groups[s]["class_not_in_subset"] = ~groups[s]["class_in_subset"]
    lat, lon = sub["latitude"].to_numpy(), sub["longitude"].to_numpy()
    res: dict = {"n_train": int(sub_mask.sum())}
    if head == "knn":
        nbrs = {s: knn(embs[s], db, max(args.k), device=args.device or None) for s in evals}
        sweep = {}
        for k in args.k:
            sc = knn_country_scores(*nbrs["dev_seen"], lab, len(classes), k)
            km = haversine(sel["latitude"].to_numpy(), sel["longitude"].to_numpy(),
                           *knn_coordinates(*nbrs["dev_seen"], lat, lon, k))
            sweep[k] = {"dev_seen_top1": float((sc.argmax(1) == sel["label"].to_numpy()).mean()),
                        "dev_seen_geoscore": float(geoscore(km).mean())}
        k_c, k_g = pick(sweep, "dev_seen_top1"), pick(sweep, "dev_seen_geoscore")
        res.update({"k_sweep_dev_seen": sweep, "k_country": k_c, "k_coords": k_g})
        for s in evals:
            sc = knn_country_scores(*nbrs[s], lab, len(classes), k_c)
            coords = knn_coordinates(*nbrs[s], lat, lon, k_g)
            res[s] = score_split(out, s, dfs[s], sc, classes, coords, groups[s], args,
                                 clusters_of(s, dfs[s]))
    else:  # learning rate and stopping epoch chosen jointly by top-1 on dev_seen only
        best, res["lr_sweep_dev_seen"] = None, {}
        for lr in args.probe_lrs:
            a = argparse.Namespace(**{**vars(args), "probe_lr": lr})
            logits_lr, history = probe_logits(db, lab, embs["dev_seen"], sel["label"].to_numpy(),
                                              {s: embs[s] for s in evals}, head_init, a,
                                              args.device or "cpu")
            top = max(history, key=lambda h: h["val_top1"])  # train_probe restores this epoch
            res["lr_sweep_dev_seen"][lr] = {"dev_seen_top1": top["val_top1"],
                                            "epoch": top["epoch"], "history": history}
            if best is None or top["val_top1"] > best[0]:
                best = (top["val_top1"], lr, top["epoch"], logits_lr)
        _, res["best_lr"], res["best_epoch"], logits = best
        for s in evals:
            res[s] = score_split(out, s, dfs[s], logits[s], classes, None, groups[s], args,
                                 clusters_of(s, dfs[s]))
    return res


def run(args: argparse.Namespace) -> list[dict]:
    ctx = load_data(Path(args.data), Path(args.emb), args.eval_test)
    head = (text_head(args.text_model or args.model, ctx["classes"], args.template)
            if "probe" in args.heads else None)
    rows, root = [], Path(args.results_dir)
    for n in args.sizes:
        for h in args.heads:
            out = root / f"{args.tag}_{args.strategy}_{n}_{h}"
            res = run_cell(ctx, args.strategy, n, h, out, args, head)
            rows += finish(out, res, args, n, h, ctx["evals"])
    summary = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    split = "test" if args.eval_test else "dev"
    summary.to_csv(root / f"{args.tag}_{args.strategy}_{split}_summary.csv", index=False)
    print(summary.to_string())
    return rows


def finish(out: Path, res: dict, args, n: int, head: str, splits) -> list[dict]:
    """Write geo_metrics.json + run_config.json; one summary row per scored split."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "geo_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "run_config.json").write_text(json.dumps(
        {**vars(args), "n_train": n, "head": head, "versions": library_versions()},
        indent=2, default=str))
    rows = []
    for split in splits:
        t = res[split]
        row = {"tag": args.tag, "strategy": args.strategy, "n_train": n, "head": head, "split": split}
        for g in ("all", "reliable", "head", "middle", "tail", "seen_creator", "unseen_creator"):
            if t.get(g, {}).get("n"):
                row[f"{g}_top1"] = t[g]["country_top1"]["mean"]
                row[f"{g}_macro_top1"] = t[g]["macro_top1"]
                row[f"{g}_n"] = t[g]["n"]
        a = t["all"]
        row["all_top1_ci"] = (a["country_top1"]["ci_lo"], a["country_top1"]["ci_hi"])
        row["all_top5"] = a["country_top5"]["mean"]
        if "median_km" in a:
            row.update({"median_km": a["median_km"], "geoscore": a["geoscore"]["mean"]})
        rows.append(row)
    return rows


def add_head_args(ap: argparse.ArgumentParser) -> None:
    """Arguments of run_cell (shared with phase1_eval)."""
    ap.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    ap.add_argument("--eval-test", action="store_true",
                    help="also score the official test split (final configurations only)")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--probe-epochs", type=int, default=30)
    ap.add_argument("--probe-lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3],
                    help="probe learning rates; one is chosen with the epoch on dev_seen")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda",
                    help="'' = CPU (numpy kNN, torch probe on cpu)")
    ap.add_argument("--seed", type=int, default=42)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="v2_data.py output dir")
    ap.add_argument("--emb", required=True, help="embed.py cache dir of one model")
    ap.add_argument("--model", default="openai/clip-vit-base-patch16", help="encoder of --emb")
    ap.add_argument("--text-model", help="CLIP text model for the probe init (default: --model)")
    ap.add_argument("--tag", required=True, help="run name prefix, e.g. clip_b16")
    ap.add_argument("--strategy", choices=["nat", "bal"], default="bal")
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--heads", nargs="+", choices=["knn", "probe"], default=["knn", "probe"])
    ap.add_argument("--results-dir", default="results/v2")
    add_head_args(ap)
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
