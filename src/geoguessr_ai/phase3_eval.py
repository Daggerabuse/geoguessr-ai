"""Phase 3, development splits only: coordinate prediction from kNN on the frozen embeddings.

Does StreetCLIP's advantage over its matched base (CLIP L/14-336) hold in km, and how does it
change with N? Grid: backbone x ordering (nat, bal) x N, each subset searched once (top-k cosine
neighbours, geo_eval.knn) and three predictors scored from the same neighbours:
  geo_top1    the nearest neighbour's coordinates;
  geo_mean    weighted mean of the k neighbours on the sphere (3D unit vectors), weights the
              similarity or exp((sim - best sim) / tau);
  geo_medoid  the similarity-weighted geographic medoid of the k neighbours (Phase 1's kNN
              coordinates, so its dev scores reproduce Phase 1's).
k (and tau) are chosen by mean GeoScore on dev_seen only; median km is reported, never selected
on.

Near-duplicate masking (--mask-km, default 1 km): before the top-k search, every training image
within that haversine distance of a dev query's true location is removed from that query's
database (dev_seen shares creators with train without a distance buffer, so otherwise the best
predictor is "copy the re-driven road"). The number of masked images is reported per split.
Baselines, scored like the predictors:
  geo_constant  the spherical mean of the training subset's coordinates, for every image;
  geo_centroid  predicted country -> spherical mean of that country's training images; the country
                from the Phase 1 frozen probe (--phase1-dev, PROBE_VARIANT dev predictions; they
                were made without masking) or else the masked kNN country vote (k on dev_seen);
  geo_oracle    the geographically nearest training image after masking (a lower bound on km);
  geo_bestk     the geographically nearest of the max(--k) retrieved neighbours (best-of-k).
geo_constant and geo_oracle do not depend on the backbone, so they get no paired gap.
Step 2, geocells (--methods cell hybrid; geocells.py: a quadtree from the pool coordinates only):
  geo_cell_<init>    linear probe over cell ids (train.py's probe, Phase 1 lr grid, epochs and
                     patience; epoch by cell top-1 on dev_seen, lr by mean GeoScore on dev_seen),
                     prediction = the argmax cell's centroid; init random, or zeroshot: each
                     cell's row is the text embedding (Phase 1 prompt) of its majority country
                     among the subset's training images (an empty cell: the country of the
                     subset image nearest its centroid);
  geo_hybrid_<init>  the probe's top-m cells (m in HYBRID_M) x the masked global kNN top-K
                     (--hybrid-k): the --hybrid-predictor (medoid/mean, k and tau as above) over
                     the kept neighbours, else the top cell's centroid; m, k, tau on dev_seen.
The cell count is chosen once (cell_geo): the cell probe of the PAIR backbones at nat and the
largest N for every --cell-targets count; the count with the best average dev_seen GeoScore of
the pair is recorded in cell_count.json and frozen for every backbone, ordering and N.
kNN numerics are pinned (--device, --knn-dtype; both recorded in every row.json): the selection
here and any later test run must use the same setting. A finished cell is reused only if its
cache_key (cache_settings: data config hash, embedding cache identity, grids, mask, numerics,
seed, n_boot, Phase 1 probe source) is unchanged. Metrics follow OSV-5M (haversine km, GeoScore = 5000 * exp(-km / 1492.7), accuracy within
1/25/200/750/2500 km); country_top1 is the country of the training image nearest to the predicted
point, a sanity check against Phase 1. Paired StreetCLIP - L/14-336 differences per image
(gaps_<metric>.csv, shrink_<metric>.csv; for km a lower value is better) and of median km
(gaps_median_km.csv) use the cluster bootstrap of Phase 1 (sequences; creators for dev_unseen).

The official test split is never read here: there is no test option, and load_data is always
called without it. A test-level coordinate evaluation needs its own pre-registration first.

  python -m geoguessr_ai.phase3_eval --data <v2 dir> --emb-root <emb dir> --out results/phase3
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geoguessr_ai import geo_eval as g
from geoguessr_ai import geocells as gc
from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import prereg
from geoguessr_ai.zipq import write_text

CACHE_VERSION = 2
METHOD = "geo"
PREDICTORS = ("top1", "mean", "medoid")
BASELINES = ("constant", "centroid", "oracle", "bestk")
PROBE_VARIANT = "probe_random"  # Phase 1 probe whose dev predictions feed geo_centroid
CELL_TARGETS = (1000, 2000, 5000)  # geocell counts tried once (nat, largest N, PAIR)
HYBRID_M = (1, 3, 5)  # top-m probe cells the hybrid keeps neighbours in (chosen on dev_seen)
TAUS = (None, 0.1, 0.03, 0.01)  # geo_mean weights: None = the cosine similarity itself
GAP_METRICS = ("geoscore", "acc_25km", "acc_200km", "acc_750km", "acc_2500km", "country_top1", "km")
GROUPS = ("reliable", "head", "middle", "tail", "seen_creator", "unseen_creator")


def predict(pred: str, nb, lat, lon, k: int, tau=None):
    if pred == "top1":
        return g.knn_top1_coords(nb[1], lat, lon)
    if pred == "mean":
        return g.knn_mean_coords(*nb, lat, lon, k, tau)
    return g.knn_coordinates(*nb, lat, lon, k)


def candidates(pred: str, ks) -> list[dict]:
    if pred == "top1":
        return [{"k": 1, "tau": None}]
    return [{"k": k, "tau": t} for k in ks for t in (TAUS if pred == "mean" else (None,))]


def choose(pred: str, nb, sel: pd.DataFrame, lat, lon, ks) -> tuple[dict, list[dict]]:
    """The candidate with the highest mean GeoScore on dev_seen (first one on ties)."""
    sweep = []
    for c in candidates(pred, ks):
        km = g.haversine(sel["latitude"].to_numpy(), sel["longitude"].to_numpy(),
                         *predict(pred, nb, lat, lon, c["k"], c["tau"]))
        sweep.append({**c, "dev_seen_geoscore": float(g.geoscore(km).mean()),
                      "dev_seen_median_km": float(np.median(km))})
    best = max(sweep, key=lambda e: e["dev_seen_geoscore"])
    return {"k": best["k"], "tau": best["tau"]}, sweep


def score_coords(out: Path, name: str, df: pd.DataFrame, coords, country, groups, args,
                 clusters, extra: dict | None = None) -> dict:
    """Per-image file {name}_geo_hits.csv (plus the extra columns) and the metrics of every
    row group."""
    km = g.haversine(df["latitude"].to_numpy(), df["longitude"].to_numpy(), *coords)
    per = {**g.per_image_geo(km),
           "country_top1": (np.asarray(country) == df["label"].to_numpy()).astype(float)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "pred_lat": coords[0], "pred_lon": coords[1], "km": km, **per,
                  **(extra or {})}).to_csv(out / f"{name}_geo_hits.csv", index=False)
    clusters = np.asarray(clusters)
    return {grp: g.summarize({k: v[m] for k, v in per.items()}, km[m], clusters[m], args.n_boot,
                             args.seed) for grp, m in groups.items()}


def run_subset(ctx: dict, ordering: str, n: int, args) -> dict[str, dict]:
    """One training subset: one neighbour search per split, then every predictor."""
    dfs, embs, evals = ctx["dfs"], ctx["embs"], ctx["evals"]
    train, sel = dfs["train"], dfs["dev_seen"]
    mask = (train[f"{ordering}_rank"] < n).to_numpy()  # nested subset, see v2_data
    sub, db = train[mask], embs["train"][mask]
    lat, lon, lab = (sub["latitude"].to_numpy(), sub["longitude"].to_numpy(),
                     sub["label"].to_numpy())
    creators = set(sub["creator_id"])
    groups = {s: g.row_groups(dfs[s], ctx["counts"], dfs[s]["creator_id"].isin(creators).to_numpy())
              for s in evals}
    geo = {s: geo_mask(dfs[s], sub, args.mask_km) for s in evals}
    nbrs = {s: g.knn(embs[s], db, max(args.k), device=args.device or None, dtype=args.knn_dtype,
                     exclude=geo[s][:2]) for s in evals}
    masking = {s: {"n_queries": len(dfs[s]), "masked_total": len(geo[s][0]),
                   "masked_mean": len(geo[s][0]) / max(len(dfs[s]), 1),
                   "masked_any_share": len(np.unique(geo[s][0])) / max(len(dfs[s]), 1)}
               for s in evals}
    out = {}
    for pred in PREDICTORS:
        d = pe.cell_dir(Path(args.out), ctx["backbone"], ordering, n, METHOD, pred)
        best, sweep = choose(pred, nbrs["dev_seen"], sel, lat, lon, args.k)
        res: dict = {"n_train": int(mask.sum()), **best, "sweep_dev_seen": sweep,
                     "masking": masking}
        for s in evals:
            coords = predict(pred, nbrs[s], lat, lon, best["k"], best["tau"])
            res[s] = score_coords(d, s, dfs[s], coords, g.nearest_label(*coords, lat, lon, lab),
                                  groups[s], args, g.clusters_of(s, dfs[s]))
        out[pred] = res
    for pred, (res, preds) in baselines(ctx, ordering, n, sub, nbrs, geo, args).items():
        d = pe.cell_dir(Path(args.out), ctx["backbone"], ordering, n, METHOD, pred)
        res.update({"n_train": int(mask.sum()), "masking": masking})
        for s in evals:
            res[s] = score_coords(d, s, dfs[s], *preds[s], groups[s], args,
                                  g.clusters_of(s, dfs[s]))
        out[pred] = res
    return out


_GEO: dict = {}  # geo_mask results: the backbone loop is outermost, so each subset recurs


def geo_mask(q: pd.DataFrame, db: pd.DataFrame, mask_km: float, chunk: int = 32):
    """(query rows, db rows) of every db image within mask_km (haversine) of each query's true
    location, and per query the geographically nearest db image outside that radius
    (mask_km <= 0: nothing masked). float64 unit vectors: float32 cannot resolve ~1 km."""
    key = (tuple(q["id"]), tuple(db["id"].iloc[[0, -1]]), len(db), mask_km,
           float(q["latitude"].sum() + q["longitude"].sum()),
           float(db["latitude"].sum() + db["longitude"].sum()))
    if key in _GEO:
        return _GEO[key]
    if len(_GEO) >= 32:  # every split x subset of one grid (8 x 2), reused by the next backbone
        _GEO.clear()
    qx = g.to_xyz(q["latitude"].to_numpy(float), q["longitude"].to_numpy(float))
    dx = g.to_xyz(db["latitude"].to_numpy(float), db["longitude"].to_numpy(float))
    thr = np.cos(mask_km / g.EARTH_RADIUS_KM) if mask_km > 0 else np.inf
    qi, di, nearest = [np.zeros(0, int)], [np.zeros(0, int)], np.empty(len(q), dtype=np.int64)
    for s in range(0, len(q), chunk):  # memory: chunk x len(db) float64
        dot = qx[s : s + chunk] @ dx.T
        r, c = np.nonzero(dot >= thr)
        dot[r, c] = -np.inf
        qi.append(r + s)
        di.append(c)
        nearest[s : s + chunk] = dot.argmax(1)
    _GEO[key] = (np.concatenate(qi), np.concatenate(di), nearest)
    return _GEO[key]


def spherical_mean(lat, lon) -> tuple[float, float]:
    la, lo = g.from_xyz(g.to_xyz(lat, lon).sum(0))
    return float(la), float(lo)


def probe_countries(args, backbone: str, ordering: str, n: int, split: str, ids,
                    classes) -> np.ndarray | None:
    """The Phase 1 frozen probe's predicted label per id (-1 where it has none), or None."""
    if not args.phase1_dev:
        return None
    f = (Path(args.phase1_dev) / backbone / f"{ordering}_{n}_{PROBE_VARIANT}"
         / f"{split}_predictions.csv")
    if not f.exists():
        return None
    p = pd.read_csv(f, dtype={"id": str}, usecols=["id", "pred"], keep_default_na=False)
    lab = p.set_index("id")["pred"].map({c: i for i, c in enumerate(classes)})
    return lab.reindex(pd.Index(ids)).fillna(-1).astype(int).to_numpy()


def baselines(ctx, ordering: str, n: int, sub: pd.DataFrame, nbrs, geo, args) -> dict:
    """{predictor: (metrics so far, {split: ((lat, lon), country)})} for BASELINES."""
    dfs, evals, classes = ctx["dfs"], ctx["evals"], ctx["classes"]
    lat, lon, lab = (sub["latitude"].to_numpy(), sub["longitude"].to_numpy(),
                     sub["label"].to_numpy())
    c_lat, c_lon = spherical_mean(lat, lon)
    c_lab = g.nearest_label(np.array([c_lat]), np.array([c_lon]), lat, lon, lab)[0]
    cent = np.full((len(classes), 2), (c_lat, c_lon))  # a country without training images
    for c, grp in sub[sub["label"] >= 0].groupby("label"):
        cent[c] = spherical_mean(grp["latitude"].to_numpy(), grp["longitude"].to_numpy())

    def vote(s, k):
        return g.knn_country_scores(*nbrs[s], lab, len(classes), k).argmax(1)

    sel = dfs["dev_seen"]
    sweep = [{"k": k, "dev_seen_geoscore": float(g.geoscore(g.haversine(
        sel["latitude"].to_numpy(), sel["longitude"].to_numpy(),
        *cent[vote("dev_seen", k)].T)).mean())} for k in args.k]
    k_vote = max(sweep, key=lambda e: e["dev_seen_geoscore"])["k"]
    kmax = max(args.k)
    out = {p: ({"k": None, "tau": None}, {}) for p in BASELINES}
    out["centroid"][0].update({"k": k_vote, "sweep_dev_seen": sweep, "country_source": {}})
    out["bestk"][0]["k"] = kmax
    for s in evals:
        m = len(dfs[s])
        q_lat, q_lon = dfs[s]["latitude"].to_numpy(), dfs[s]["longitude"].to_numpy()
        out["constant"][1][s] = ((np.full(m, c_lat), np.full(m, c_lon)), np.full(m, c_lab))
        country = vote(s, k_vote)
        probe = probe_countries(args, ctx["backbone"], ordering, n, s, dfs[s]["id"], classes)
        src = "knn_vote"
        if probe is not None:
            src = f"{PROBE_VARIANT} ({int((probe < 0).sum())} images without one: knn_vote)"
            country = np.where(probe >= 0, probe, country)
        out["centroid"][0]["country_source"][s] = src
        out["centroid"][1][s] = ((cent[country, 0], cent[country, 1]), country)
        near = geo[s][2]
        out["oracle"][1][s] = ((lat[near], lon[near]), lab[near])
        idx = nbrs[s][1][:, :kmax]
        km = g.haversine(q_lat[:, None], q_lon[:, None], lat[idx], lon[idx])
        best = idx[np.arange(m), km.argmin(1)]
        out["bestk"][1][s] = ((lat[best], lon[best]), lab[best])
    return out


# ----------------------------------------------------------------- Step 2: geocells


def make_geo(data: Path, out: Path, target: int) -> dict:
    """The geocell partition for target cells, from the pool coordinates (train.csv) only."""
    pool = pd.read_csv(data / "train.csv", usecols=["latitude", "longitude"])
    tree = gc.for_target(pool["latitude"].to_numpy(), pool["longitude"].to_numpy(), target)
    sha = gc.save(tree, out / "geocells" / f"cells_{target}.json")
    print(f"geocells: target {target} -> {gc.n_cells(tree)} cells (max_per_cell "
          f"{tree['max_per_cell']}), sha256 {sha[:16]}", flush=True)
    cent = np.stack([tree["centroid_lat"], tree["centroid_lon"]], axis=1)
    return {"tree": tree, "sha": sha, "target": target, "cent": cent}


def majority_country(cells, labels, n_cells: int, cent, lat, lon) -> np.ndarray:
    """Per cell: the most frequent country among the subset's training images in it (ties ->
    the smallest label); a cell without one takes the country of the subset image nearest to
    its centroid. Only the subset's own labels are used."""
    out = np.full(n_cells, -1)
    ok = labels >= 0
    if ok.any():
        tab = pd.crosstab(cells[ok], labels[ok])
        out[tab.index.to_numpy()] = tab.columns.to_numpy()[tab.to_numpy().argmax(1)]
    empty = out < 0
    if empty.any():
        out[empty] = g.nearest_label(cent[empty, 0], cent[empty, 1], lat, lon, labels)
    return out


def top_cells(logits: np.ndarray, m: int) -> np.ndarray:
    """(N, m) cell ids, highest logit first."""
    m = min(m, logits.shape[1])
    part = np.argpartition(-logits, m - 1, axis=1)[:, :m]
    order = np.argsort(-np.take_along_axis(logits, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def split_groups(ctx: dict, sub: pd.DataFrame) -> dict:
    creators = set(sub["creator_id"])
    return {s: g.row_groups(ctx["dfs"][s], ctx["counts"],
                            ctx["dfs"][s]["creator_id"].isin(creators).to_numpy())
            for s in ctx["evals"]}


def run_cells(ctx: dict, ordering: str, n: int, args, geo: dict, text) -> dict[str, dict]:
    """Linear probe over geocell ids per init (train.py's probe: epoch by cell top-1 on
    dev_seen; lr by mean GeoScore on dev_seen of the argmax cell's centroid). Writes each
    split's top cells ({split}_top_cells.csv) for the hybrid."""
    import torch

    dfs, embs, evals, sel = ctx["dfs"], ctx["embs"], ctx["evals"], ctx["dfs"]["dev_seen"]
    mask = (dfs["train"][f"{ordering}_rank"] < n).to_numpy()
    sub, db = dfs["train"][mask], embs["train"][mask]
    lat, lon, lab = (sub["latitude"].to_numpy(), sub["longitude"].to_numpy(),
                     sub["label"].to_numpy())
    cent, n_cells = geo["cent"], len(geo["cent"])
    y = ctx["pool_cells"][mask]
    q = {s: gc.assign(geo["tree"], dfs[s]["latitude"], dfs[s]["longitude"]) for s in evals}
    cent_lab = g.nearest_label(cent[:, 0], cent[:, 1], lat, lon, lab)
    groups, out = split_groups(ctx, sub), {}
    for init in args.probe_inits:
        head = (pe.random_head(n_cells, db.shape[1], args.seed) if init == "random" else
                text[torch.from_numpy(majority_country(y, lab, n_cells, cent, lat, lon))])
        sweep, best, best_top = [], None, None
        for lr in args.probe_lrs:
            a = argparse.Namespace(**{**vars(args), "probe_lr": lr})
            logits, hist = g.probe_logits(db, y, embs["dev_seen"], q["dev_seen"],
                                          {s: embs[s] for s in evals}, head, a,
                                          args.device or "cpu")
            top = {s: top_cells(logits[s], max(HYBRID_M)) for s in evals}
            del logits
            e = max(hist, key=lambda h: h["val_top1"])
            km = g.haversine(sel["latitude"].to_numpy(), sel["longitude"].to_numpy(),
                             *cent[top["dev_seen"][:, 0]].T)
            sweep.append({"lr": lr, "epoch": e["epoch"], "dev_seen_cell_top1": e["val_top1"],
                          "dev_seen_geoscore": float(g.geoscore(km).mean()),
                          "stopped_early": hist[-1]["epoch"] < args.probe_epochs})
            if best is None or sweep[-1]["dev_seen_geoscore"] > best["dev_seen_geoscore"]:
                best, best_top = sweep[-1], top
            print(f"  cell_{init} lr {lr:g}: epoch {e['epoch']}, dev_seen GeoScore "
                  f"{sweep[-1]['dev_seen_geoscore']:.1f}", flush=True)
        pred = f"cell_{init}"
        d = pe.cell_dir(Path(args.out), ctx["backbone"], ordering, n, METHOD, pred)
        res = {"n_train": int(mask.sum()), "k": None, "tau": None, "lr": best["lr"],
               "epoch": best["epoch"], "stopped_early": best["stopped_early"],
               "n_cells": n_cells, "cell_target": geo["target"], "sweep_dev_seen": sweep}
        for s in evals:
            t = best_top[s]
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": dfs[s]["id"], **{f"c{j}": t[:, j] for j in range(t.shape[1])}}) \
                .to_csv(d / f"{s}_top_cells.csv", index=False)
            res[s] = score_coords(d, s, dfs[s], tuple(cent[t[:, 0]].T), cent_lab[t[:, 0]],
                                  groups[s], args, g.clusters_of(s, dfs[s]))
        out[pred] = res
    return out


def hybrid_coords(nb, db_cells, top, m: int, pred: str, k: int, tau, lat, lon, cent):
    """Of the retrieved neighbours (best first), keep those in the query's top-m cells and
    apply the predictor to the first k kept; no kept neighbour -> the top cell's centroid.
    Returns ((lat, lon), fallback mask)."""
    sims, idx = nb
    # a masked image (-inf) only reaches the top-K when fewer than K are left: never kept
    keep = (db_cells[idx][:, :, None] == top[:, None, :m]).any(-1) & np.isfinite(sims)
    order = np.argsort(~keep, axis=1, kind="stable")[:, :k]  # kept first, order preserved
    s2, i2, v2 = (np.take_along_axis(x, order, 1) for x in (sims, idx, keep))
    if pred == "medoid":
        la, lo = g.knn_coordinates(s2, i2, lat, lon, k, valid=v2)
    else:
        la, lo = g.knn_mean_coords(s2, i2, lat, lon, k, tau, valid=v2)
    fb = ~v2[:, 0]
    la, lo = np.array(la, dtype=float), np.array(lo, dtype=float)
    la[fb], lo[fb] = cent[top[fb, 0], 0], cent[top[fb, 0], 1]
    return (la, lo), fb


def choose_hybrid(nb, db_cells, top, sel: pd.DataFrame, lat, lon, cent, args):
    """(m, k, tau) with the highest mean GeoScore on dev_seen (first one on ties)."""
    sweep = []
    for m in HYBRID_M:
        for c in candidates(args.hybrid_predictor, args.k):
            coords, fb = hybrid_coords(nb, db_cells, top, m, args.hybrid_predictor, c["k"],
                                       c["tau"], lat, lon, cent)
            km = g.haversine(sel["latitude"].to_numpy(), sel["longitude"].to_numpy(), *coords)
            sweep.append({"m": m, **c, "dev_seen_geoscore": float(g.geoscore(km).mean()),
                          "dev_seen_fallback_share": float(fb.mean())})
    best = max(sweep, key=lambda e: e["dev_seen_geoscore"])
    return {"m": best["m"], "k": best["k"], "tau": best["tau"]}, sweep


def run_hybrid(ctx: dict, ordering: str, n: int, args, geo: dict) -> dict[str, dict]:
    """Probe top-m cells (from the finished cell_<init> cells) x masked global kNN top-K."""
    dfs, embs, evals = ctx["dfs"], ctx["embs"], ctx["evals"]
    mask = (dfs["train"][f"{ordering}_rank"] < n).to_numpy()
    sub, db = dfs["train"][mask], embs["train"][mask]
    lat, lon, lab = (sub["latitude"].to_numpy(), sub["longitude"].to_numpy(),
                     sub["label"].to_numpy())
    db_cells, cent = ctx["pool_cells"][mask], geo["cent"]
    gm = {s: geo_mask(dfs[s], sub, args.mask_km) for s in evals}
    nb = {s: g.knn(embs[s], db, args.hybrid_k, device=args.device or None, dtype=args.knn_dtype,
                   exclude=gm[s][:2]) for s in evals}
    masking = {s: {"masked_mean": len(gm[s][0]) / max(len(dfs[s]), 1),
                   "masked_any_share": len(np.unique(gm[s][0])) / max(len(dfs[s]), 1)}
               for s in evals}
    groups, out = split_groups(ctx, sub), {}
    for init in args.probe_inits:
        cd = pe.cell_dir(Path(args.out), ctx["backbone"], ordering, n, METHOD, f"cell_{init}")
        top = {}
        for s in evals:
            t = pd.read_csv(cd / f"{s}_top_cells.csv", dtype={"id": str}).set_index("id")
            top[s] = t.loc[dfs[s]["id"]].to_numpy()
        best, sweep = choose_hybrid(nb["dev_seen"], db_cells, top["dev_seen"], dfs["dev_seen"],
                                    lat, lon, cent, args)
        pred = f"hybrid_{init}"
        d = pe.cell_dir(Path(args.out), ctx["backbone"], ordering, n, METHOD, pred)
        res = {"n_train": int(mask.sum()), **best, "sweep_dev_seen": sweep, "masking": masking,
               "n_cells": len(cent), "cell_target": geo["target"], "fallback_share": {}}
        for s in evals:
            coords, fb = hybrid_coords(nb[s], db_cells, top[s], best["m"], args.hybrid_predictor,
                                       best["k"], best["tau"], lat, lon, cent)
            res["fallback_share"][s] = float(fb.mean())
            res[s] = score_coords(d, s, dfs[s], coords, g.nearest_label(*coords, lat, lon, lab),
                                  groups[s], args, g.clusters_of(s, dfs[s]),
                                  extra={"fallback": fb.astype(int)})
        out[pred] = res
    return out


def pick_cell_target(rows: pd.DataFrame) -> tuple[int, dict]:
    """The geocell target with the best average dev_seen GeoScore over the PAIR backbones'
    rows (every init), ties -> the smaller target; also the averages per target."""
    r = rows[rows["backbone"].isin(pe.PAIR)]
    avg = r.groupby("cell_target")["dev_seen_geoscore"].mean().sort_index()
    return int(avg.idxmax()), {int(k): float(v) for k, v in avg.items()}


def cell_row(res: dict, backbone: str, ordering: str, n: int, pred: str, evals) -> dict:
    row = {"backbone": backbone, "ordering": ordering, "n": n, "method": METHOD,
           "predictor": pred, "variant": pe.variant(METHOD, pred), "n_train": res["n_train"],
           "k": res["k"], "tau": res["tau"],
           **{c: res.get(c) for c in ("lr", "epoch", "stopped_early", "m", "n_cells",
                                      "cell_target")}}
    for s in evals:
        p, r = pe.SPLIT_NAMES[s], res[s]
        mk = (res.get("masking") or {}).get(s, {})  # the cell probe has no kNN: nothing masked
        row[f"{p}_masked_mean"] = mk.get("masked_mean", np.nan)
        row[f"{p}_masked_any_share"] = mk.get("masked_any_share", np.nan)
        if "fallback_share" in res:
            row[f"{p}_fallback_share"] = res["fallback_share"][s]
        a = r["all"]
        row[f"{p}_n"] = a["n"]
        row[f"{p}_median_km"], row[f"{p}_mean_km"] = a["median_km"], a["mean_km"]
        for m in ("geoscore", "country_top1", *(f"acc_{t}km" for t in g.KM_THRESHOLDS)):
            row[f"{p}_{m}"], row[f"{p}_{m}_lo"], row[f"{p}_{m}_hi"] = (
                a[m]["mean"], a[m]["ci_lo"], a[m]["ci_hi"])
        for grp in GROUPS:
            row[f"{p}_{grp}_n"] = r[grp]["n"]
            row[f"{p}_{grp}_geoscore"] = r[grp]["geoscore"]["mean"] if r[grp]["n"] else np.nan
            row[f"{p}_{grp}_median_km"] = r[grp].get("median_km", np.nan)
    return row


def cache_settings(args, data: Path, emb: Path, method: str = "knn", cells=None) -> dict:
    """Everything a finished cell of `method` depends on besides the code (CACHE_VERSION).
    cells: (geocell sha256, target) for the cell and hybrid methods."""
    from geoguessr_ai.embed import cache_files
    from geoguessr_ai.train import PROBE_BATCH

    cfg, enc = data / "config.json", emb / "encoder.json"
    s = {"cache_version": CACHE_VERSION, "method": method,
         "data_config_sha256": prereg.sha256(cfg) if cfg.exists() else None,
         "emb_encoder_sha256": prereg.sha256(enc) if enc.exists() else None,
         "emb_files": [[f.name, f.stat().st_size] for f in cache_files(emb, "train")],
         "mask_km": args.mask_km, "knn_device": args.device or "cpu",
         "knn_dtype": args.knn_dtype, "seed": args.seed, "n_boot": args.n_boot}
    if method == "knn":
        return {**s, "k": list(args.k), "taus": list(TAUS), "probe_variant": PROBE_VARIANT,
                "phase1_dev": str(args.phase1_dev) if args.phase1_dev else None}
    s.update({"geocells_sha256": cells[0], "cell_target": cells[1],
              "probe_lrs": list(args.probe_lrs), "probe_epochs": args.probe_epochs,
              "probe_patience": args.probe_patience, "label_smoothing": args.label_smoothing,
              "logit_scale": args.logit_scale, "template": args.template,
              "probe_batch": PROBE_BATCH})
    if method == "hybrid":
        s.update({"hybrid_k": args.hybrid_k, "hybrid_m": list(HYBRID_M),
                  "hybrid_predictor": args.hybrid_predictor, "k": list(args.k),
                  "taus": list(TAUS)})
    return s


def cache_key(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def cell_done(d: Path, key: str) -> bool:
    row = pe.read_row(d)
    return row is not None and row.get("cache_key") == key


def weighted_median(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    cw = np.cumsum(weights)
    return float(sorted_values[np.searchsorted(cw, cw[-1] / 2)])


def median_gap(a: pd.DataFrame, b: pd.DataFrame, clusters: pd.Series, n_boot: int,
               seed: int) -> dict:
    """median(km_a) - median(km_b) over the images both scored, with a paired cluster bootstrap
    (one resample of clusters, as multiplicity weights, applies to both backbones)."""
    m = a.set_index("id")["km"].to_frame("a").join(b.set_index("id")["km"].rename("b"),
                                                  how="inner")
    codes, _ = pd.factorize(m.index.map(clusters))
    if (codes < 0).any():
        raise ValueError("image without a cluster")
    oa, ob = np.argsort(m["a"].to_numpy()), np.argsort(m["b"].to_numpy())
    va, vb = m["a"].to_numpy()[oa], m["b"].to_numpy()[ob]
    n_c, rng = codes.max() + 1, np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n_c, n_c), minlength=n_c)[codes]
        diffs[i] = weighted_median(va, w[oa]) - weighted_median(vb, w[ob])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"median_km_a": float(np.median(va)), "median_km_b": float(np.median(vb)),
            "gap": float(np.median(va) - np.median(vb)), "gap_lo": float(lo), "gap_hi": float(hi),
            "n_images": len(m)}


def median_gaps(out: Path, data: Path, rows: pd.DataFrame, evals, n_boot: int,
                seed: int) -> pd.DataFrame:
    clusters = {}
    for s in evals:
        d = pd.read_csv(data / f"{s}.csv", dtype=str, keep_default_na=False)
        clusters[s] = pd.Series(g.clusters_of(s, d), index=d["id"])
    res = []
    for (ordering, n, var), grp in rows.groupby(["ordering", "n", "variant"]):
        if not set(pe.PAIR) <= set(grp["backbone"]):
            continue
        for s in evals:
            a, b = (pd.read_csv(out / bb / f"{ordering}_{n}_{var}" / f"{s}_geo_hits.csv",
                                dtype={"id": str}) for bb in pe.PAIR)
            res.append({"ordering": ordering, "n": n, "variant": var, "split": pe.SPLIT_NAMES[s],
                        **median_gap(a, b, clusters[s], n_boot, seed)})
    return pd.DataFrame(res)


def compare_phase1(table: pd.DataFrame, phase1: Path) -> str:
    """Largest |geo_medoid - Phase 1 kNN| over dev GeoScore and median km (development rows
    only: a directory holding test results is refused)."""
    if "mask_km" in table and (table["mask_km"] > 0).any():
        return "compare-phase1: skipped (mask_km > 0: the neighbours differ from Phase 1's)"
    diffs = []
    for _, r in table[table["predictor"] == "medoid"].iterrows():
        old = pe.read_row(phase1 / r["backbone"] / f"{r['ordering']}_{r['n']}_knn")
        if old is None:
            continue
        if any(k.startswith("test_") for k in old):
            raise ValueError(f"{phase1} holds test results; compare with the development run only")
        diffs += [abs(r[f"{s}_{m}"] - old[f"{s}_{m}"]) for s in ("dev_seen", "dev_unseen")
                  for m in ("geoscore", "median_km")]
    if not diffs:
        return f"compare-phase1: no matching kNN cells in {phase1}"
    return (f"geo_medoid vs Phase 1 kNN ({len(diffs) // 4} cells): max abs difference "
            f"{max(diffs):.6f} (GeoScore, median km on dev_seen and dev_unseen)")


def method_preds(args) -> dict[str, tuple]:
    """Predictor names per requested method (the hybrid needs its cell probes)."""
    out = {}
    for m in args.methods:
        if m == "knn":
            out[m] = (*PREDICTORS, *BASELINES)
        else:
            if m == "hybrid" and "cell" not in out:
                out["cell"] = tuple(f"cell_{i}" for i in args.probe_inits)
            out[m] = tuple(f"{m}_{i}" for i in args.probe_inits)
    return {m: out[m] for m in ("knn", "cell", "hybrid") if m in out}


def run_grid(args: argparse.Namespace, geo: dict | None) -> pd.DataFrame:
    """Every backbone x subset x method of args (finished cells with the same cache key are
    reused); returns the rows."""
    out, data, emb_root = Path(args.out), Path(args.data), Path(args.emb_root)
    preds, rows = method_preds(args), []
    cells = (geo["sha"], geo["target"]) if geo else None
    for bb in args.backbones:
        settings = {m: cache_settings(args, data, emb_root / bb, m, cells) for m in preds}
        keys = {m: cache_key(s) for m, s in settings.items()}
        subsets = [(o, n) for o in args.orderings for n in args.sizes]
        todo = {}
        for c in subsets:
            ms = [m for m in preds if not all(cell_done(pe.cell_dir(out, bb, *c, METHOD, p),
                                                        keys[m]) for p in preds[m])]
            if "cell" in ms and "hybrid" in preds and "hybrid" not in ms:
                ms.append("hybrid")  # new cell probes: their hybrid is redone too
            if ms:
                todo[c] = ms
        if todo:
            t = time.time()
            ctx = {**g.load_data(data, emb_root / bb, eval_test=False), "backbone": bb}
            if "test" in ctx["evals"]:
                raise AssertionError("phase3_eval never scores the official test split")
            text = None
            if geo:
                tr = ctx["dfs"]["train"]
                ctx["pool_cells"] = gc.assign(geo["tree"], tr["latitude"], tr["longitude"])
                if "zeroshot" in args.probe_inits and any("cell" in ms for ms in todo.values()):
                    model = json.loads((emb_root / bb / "encoder.json").read_text())["model"]
                    text = g.text_head(model, ctx["classes"], args.template)
            print(f"{bb}: loaded in {time.time() - t:.0f} s; {len(todo)} of {len(subsets)} "
                  f"subsets to run", flush=True)
            for i, ((o, n), ms) in enumerate(todo.items(), 1):
                for m in ms:
                    print(f"{bb} {o} {n} {m}: start ({i}/{len(todo)})", flush=True)
                    t = time.time()
                    res_all = (run_subset(ctx, o, n, args) if m == "knn" else
                               run_cells(ctx, o, n, args, geo, text) if m == "cell" else
                               run_hybrid(ctx, o, n, args, geo))
                    for pred, res in res_all.items():
                        d = pe.cell_dir(out, bb, o, n, METHOD, pred)
                        res["seconds"] = round(time.time() - t, 1)
                        write_text(d / "geo_metrics.json", json.dumps(
                            {**res, "cache_settings": settings[m]}, indent=2, default=float))
                        write_text(d / "row.json", json.dumps(  # written last: the cell is done
                            {**cell_row(res, bb, o, n, pred, ctx["evals"]), "method": m,
                             "seconds": res["seconds"], "cache_version": CACHE_VERSION,
                             "cache_key": keys[m], "mask_km": args.mask_km,
                             "knn_device": settings[m]["knn_device"],
                             "knn_dtype": args.knn_dtype}, default=float))
                    masked = next(iter(res_all.values())).get("masking") or {}
                    print(f"{bb} {o} {n} {m}: {time.time() - t:.0f} s"
                          + ("; masked per query: " + ", ".join(
                              f"{pe.SPLIT_NAMES[s]} {v['masked_mean']:.2f} "
                              f"({100 * v['masked_any_share']:.1f}% of queries)"
                              for s, v in masked.items()) if masked else ""), flush=True)
            del ctx
        rows += [pe.read_row(pe.cell_dir(out, bb, *c, METHOD, p))
                 for c in subsets for m in preds for p in preds[m]]
    return pd.DataFrame(rows)


def cell_geo(args: argparse.Namespace) -> dict:
    """The frozen geocell partition: --cell-target, else the recorded choice (cell_count.json),
    else chosen now, once: the cell probe for the PAIR backbones at nat and the largest N for
    every CELL_TARGETS count, then the count with the best average dev_seen GeoScore of the
    pair (pick_cell_target). Its cells are copied into the main grid."""
    out, data = Path(args.out), Path(args.data)
    if args.cell_target:
        return make_geo(data, out, args.cell_target)
    f = out / "cell_count.json"
    if f.exists():
        return make_geo(data, out, json.loads(f.read_text())["target"])
    n, rows, geos = max(args.sizes), [], {}
    for t in args.cell_targets:
        geos[t] = make_geo(data, out, t)
        a = argparse.Namespace(**{**vars(args), "out": str(out / "cell_select" / f"c{t}"),
                                  "backbones": list(pe.PAIR), "orderings": ["nat"],
                                  "sizes": [n], "methods": ["cell"]})
        print(f"cell count selection: target {t} (nat {n}, {', '.join(pe.PAIR)})", flush=True)
        rows.append(run_grid(a, geos[t]))
    target, avg = pick_cell_target(pd.concat(rows))
    write_text(f, json.dumps({"target": target, "n_cells": len(geos[target]["cent"]),
                              "geocells_sha256": geos[target]["sha"],
                              "dev_seen_geoscore_pair_average": avg, "ordering": "nat", "n": n,
                              "backbones": list(pe.PAIR)}, indent=2))
    print(f"cell count selection: {avg} -> target {target}, frozen", flush=True)
    src = out / "cell_select" / f"c{target}"
    for p in src.glob("*/*"):  # same cache key: the main grid reuses these cells
        dst = out / p.relative_to(src)
        if not dst.exists():
            shutil.copytree(p, dst)
    return geos[target]


def run(args: argparse.Namespace) -> pd.DataFrame:
    out, data = Path(args.out), Path(args.data)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("no CUDA device: kNN with numpy on the CPU")
            args.device = ""
    geo = cell_geo(args) if {"cell", "hybrid"} & set(args.methods) else None
    table = run_grid(args, geo)
    table.to_csv(out / "results.csv", index=False)
    evals = ("dev_seen", "dev")
    paired = table[~table["predictor"].isin(("constant", "oracle"))]  # backbone-independent
    for m in GAP_METRICS:
        gp, sh = pe.gaps(out, data, paired, evals, args.n_boot, args.seed, stem="geo_hits", col=m)
        gp.to_csv(out / f"gaps_{m}.csv", index=False)
        sh.to_csv(out / f"shrink_{m}.csv", index=False)
        if m == "geoscore" and not gp.empty:
            pe.plot_gap(gp, out / "gap_geo.png", "GeoScore difference (paired)")
            print(gp.to_string())
            print(sh.to_string())
    mg = median_gaps(out, data, paired, evals, args.n_boot, args.seed)
    mg.to_csv(out / "gaps_median_km.csv", index=False)
    pe.plot_scaling(table, out / "scaling_geo.png", [pe.SPLIT_NAMES[s] for s in evals],
                    "geoscore", "GeoScore")
    cols = ["backbone", "ordering", "n", "predictor", "k", "tau", "dev_seen_geoscore",
            "dev_seen_median_km", "dev_unseen_geoscore", "dev_unseen_median_km"]
    print(table[cols].to_string(index=False))
    if args.phase1_dev and "knn" in args.methods:
        print(compare_phase1(table, Path(args.phase1_dev)))
    return table


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="v2_data.py output dir")
    ap.add_argument("--emb-root", required=True, help="dir with one embed.py cache per backbone")
    ap.add_argument("--out", default="results/phase3")
    ap.add_argument("--backbones", nargs="+", default=list(pe.BACKBONES))
    ap.add_argument("--orderings", nargs="+", choices=["nat", "bal"], default=["nat", "bal"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[50_000, 100_000, 200_000, 500_000])
    ap.add_argument("--methods", nargs="+", choices=["knn", "cell", "hybrid"], default=["knn"])
    ap.add_argument("--k", type=int, nargs="+", default=list(g.DEFAULT_K))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda", help="'' = CPU (numpy kNN, CPU probe)")
    ap.add_argument("--knn-dtype", choices=["fp16", "fp32"], default="fp32",
                    help="kNN score precision (recorded per cell; keep it for any test run)")
    ap.add_argument("--mask-km", type=float, default=1.0,
                    help="drop training images within this distance of each dev query (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase1-dev", help="Phase 1 development output (phase1_eval dir): probe "
                    "countries for geo_centroid and, with --mask-km 0, a check that geo_medoid "
                    "reproduces its kNN dev coordinates")
    # Step 2: geocell probe and hybrid
    ap.add_argument("--probe-inits", nargs="+", choices=["random", "zeroshot"],
                    default=["random", "zeroshot"])
    ap.add_argument("--probe-lrs", type=float, nargs="+", default=list(pe.PROBE_LRS))
    ap.add_argument("--probe-epochs", type=int, default=pe.PROBE_EPOCHS)
    ap.add_argument("--probe-patience", type=int, default=pe.PROBE_PATIENCE)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--template", default=g.DEFAULT_TEMPLATE)
    ap.add_argument("--cell-targets", type=int, nargs="+", default=list(CELL_TARGETS),
                    help="geocell counts tried once to choose the frozen one")
    ap.add_argument("--cell-target", type=int,
                    help="use this geocell count (skips the selection and cell_count.json)")
    ap.add_argument("--hybrid-k", type=int, default=1000, help="global kNN top-K of the hybrid")
    ap.add_argument("--hybrid-predictor", choices=["medoid", "mean"], default="medoid",
                    help="medoid: geo_mean degenerated to top-1 (k=1) on dev")
    return ap.parse_args(argv)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    run(parse_args())


if __name__ == "__main__":
    main()
