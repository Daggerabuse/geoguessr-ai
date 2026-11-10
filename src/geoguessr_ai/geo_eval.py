"""V2/V3 heads on cached embeddings: kNN retrieval (country + coordinates) and a linear probe.

For each training-subset size N (nested, see v2_data.py) and head:
  - hyperparameters (k for kNN, the probe epoch) are chosen on dev only;
  - dev is scored; the official test split is read and scored only with --eval-test
    (final configurations), so development runs never touch it.
Every split is also reported per row group: countries with training data, reliable
countries, head/middle/tail (country_counts.csv), photographer seen/unseen in the training
subset. Groups select rows only; predictions always range over the full class list.
Country metrics use the same files as baseline_clip/train (write_split_results).
Distance metrics follow OSV-5M: haversine km, GeoScore = 5000 * exp(-km / 1492.7),
accuracy within 1/25/200/750/2500 km. Intervals: cluster bootstrap over sequences
(frames of one sequence are not independent). Test images whose creator also appears in
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
    """Train train.py's probe (cosine head, epoch chosen by dev top-1); return logits per split."""
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
                coords, groups: dict[str, np.ndarray], args) -> dict:
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
    km = None
    if coords is not None:
        km = haversine(df["latitude"].to_numpy(), df["longitude"].to_numpy(), *coords)
        per.update(per_image_geo(km))
        pd.DataFrame({"id": df["id"], "pred_lat": coords[0], "pred_lon": coords[1], "km": km}) \
            .to_csv(out / f"{name}_geo_predictions.csv", index=False)
    seq, country = df["sequence"].to_numpy(), df["country"].to_numpy()
    res: dict = {"country_files": m}
    for g, mask in groups.items():
        r = summarize({k: v[mask] for k, v in per.items()}, None if km is None else km[mask],
                      seq[mask], args.n_boot, args.seed)
        if mask.any():
            r["macro_top1"] = float(pd.Series(hit1[mask]).groupby(country[mask]).mean().mean())
            r["n_countries"] = len(set(country[mask]))
        res[g] = r
    return res


def pick(results: dict, metric: str) -> int:
    return max(results, key=lambda k: results[k][metric])


def run(args: argparse.Namespace) -> list[dict]:
    from geoguessr_ai.embed import load_embeddings

    data = Path(args.data)
    classes = json.loads((data / "classes.json").read_text())
    # keep_default_na=False everywhere: "NA" is Namibia, not a missing value
    counts = pd.read_csv(data / "country_counts.csv", index_col=0, keep_default_na=False)
    splits = ("train", "dev", "test") if args.eval_test else ("train", "dev")
    dfs, embs = {}, {}
    for name in splits:
        df = pd.read_csv(data / f"{name}.csv", dtype={"id": str, "sequence": str, "creator_id": str},
                         keep_default_na=False)
        e, found = load_embeddings(Path(args.emb), df["id"], "test" if name == "test" else "train")
        if not found.all():
            print(f"WARNING: {name}: {(~found).sum()} of {len(df)} ids not embedded, dropped")
        dfs[name], embs[name] = df[found].reset_index(drop=True), e[found]
    train, dev, evals = dfs["train"], dfs["dev"], splits[1:]
    for name in evals:  # the kNN database / probe data must never contain evaluated images
        for col in ("id", "sequence"):
            if set(train[col]) & set(dfs[name][col]):
                raise AssertionError(f"train and {name} share {col}s")

    rows, root = [], Path(args.results_dir)
    for n in args.sizes:
        sub_mask = (train[f"{args.strategy}_rank"] < n).to_numpy()  # nested subset, see v2_data
        sub, db = train[sub_mask], embs["train"][sub_mask]
        creators = set(sub["creator_id"])
        groups = {s: row_groups(dfs[s], counts, dfs[s]["creator_id"].isin(creators).to_numpy())
                  for s in evals}
        lab, lat, lon = sub["label"].to_numpy(), sub["latitude"].to_numpy(), sub["longitude"].to_numpy()
        base = f"{args.tag}_{args.strategy}_{n}"

        if "knn" in args.heads:
            nbrs = {s: knn(embs[s], db, max(args.k), device=args.device) for s in evals}
            sweep = {}
            for k in args.k:
                sc = knn_country_scores(*nbrs["dev"], lab, len(classes), k)
                km = haversine(dev["latitude"].to_numpy(), dev["longitude"].to_numpy(),
                               *knn_coordinates(*nbrs["dev"], lat, lon, k))
                sweep[k] = {"dev_top1": float((sc.argmax(1) == dev["label"].to_numpy()).mean()),
                            "dev_geoscore": float(geoscore(km).mean())}
            k_c, k_g = pick(sweep, "dev_top1"), pick(sweep, "dev_geoscore")
            out = root / f"{base}_knn"
            res = {"k_sweep_dev": sweep, "k_country": k_c, "k_coords": k_g}
            for s in evals:
                sc = knn_country_scores(*nbrs[s], lab, len(classes), k_c)
                coords = knn_coordinates(*nbrs[s], lat, lon, k_g)
                res[s] = score_split(out, s, dfs[s], sc, classes, coords, groups[s], args)
            rows.append(finish(out, res, args, n, "knn"))

        if "probe" in args.heads:
            head = text_head(args.text_model or args.model, classes, args.template)
            logits, history = probe_logits(db, lab, embs["dev"], dev["label"].to_numpy(),
                                           {s: embs[s] for s in evals}, head, args, args.device)
            out = root / f"{base}_probe"
            res = {"history": history}
            for s in evals:
                res[s] = score_split(out, s, dfs[s], logits[s], classes, None, groups[s], args)
            rows.append(finish(out, res, args, n, "probe"))
    summary = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    split = "test" if args.eval_test else "dev"
    summary.to_csv(root / f"{args.tag}_{args.strategy}_{split}_summary.csv", index=False)
    print(summary.to_string())
    return rows


def finish(out: Path, res: dict, args, n: int, head: str) -> dict:
    """Write geo_metrics.json + run_config.json; one summary row (test if scored, else dev)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "geo_metrics.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "run_config.json").write_text(json.dumps(
        {**vars(args), "n_train": n, "head": head, "versions": library_versions()},
        indent=2, default=str))
    split = "test" if "test" in res else "dev"
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
    return row


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
    ap.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    ap.add_argument("--eval-test", action="store_true",
                    help="also score the official test split (final configurations only)")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--probe-epochs", type=int, default=30)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda", help="'' = numpy kNN on CPU (probe needs torch)")
    ap.add_argument("--results-dir", default="results/v2")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
