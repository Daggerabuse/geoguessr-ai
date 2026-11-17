"""Geocells for Phase 3 Step 2: a quadtree on (lat, lon), built from the full training POOL's
coordinates only (no labels, no images, no dev or test data). One partition serves every
training subset N, so the class set is the same along the scaling curve; it does use the pool's
geographic density beyond a subset (coordinates only), which is documented as such.

A box [lat0, lat1) x [lon0, lon1) splits into 4 equal children while it holds more than
max_per_cell pool points and is larger than min_deg (latitude span). Points on a split line go
to the upper child, the top/right edges of the world box (90, 180) belong to it. Every non-empty
leaf is a cell, numbered in depth-first order (children SW, SE, NW, NE): fully deterministic.
Cell centroid: the normalised 3D mean of its pool points; when that falls outside the box (a
large, sparse box), the cell's pool point nearest to it instead. A point in an empty leaf (only
possible for non-pool coordinates, e.g. a dev image) gets the cell with the nearest centroid.
max_per_cell is found by bisection so the cell count is as close as possible to a target.

Saved as JSON (deterministic bytes); its sha256 identifies the partition in cache keys.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from geoguessr_ai import geo_eval as g

MIN_DEG = 0.01  # ~1 km of latitude: boxes never get smaller than this


def build(lat, lon, max_per_cell: int, min_deg: float = MIN_DEG) -> dict:
    """The quadtree as flat node arrays (bounds, first child, cell id) plus cell centroids."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    bounds, child, cell = [[-90.0, 90.0, -180.0, 180.0]], [-1], [-1]
    members = []  # pool point indices per cell
    stack = [(0, np.arange(len(lat)))]
    while stack:
        node, pts = stack.pop()
        la0, la1, lo0, lo1 = bounds[node]
        if len(pts) > max_per_cell and la1 - la0 > min_deg:
            mla, mlo = (la0 + la1) / 2, (lo0 + lo1) / 2
            child[node] = len(bounds)
            q = 2 * (lat[pts] >= mla) + (lon[pts] >= mlo)
            boxes = [[la0, mla, lo0, mlo], [la0, mla, mlo, lo1], [mla, la1, lo0, mlo],
                     [mla, la1, mlo, lo1]]
            for b in boxes:
                bounds.append(b)
                child.append(-1)
                cell.append(-1)
            for j in (3, 2, 1, 0):  # LIFO: child 0 is processed (and numbered) first
                stack.append((child[node] + j, pts[q == j]))
        elif len(pts):
            cell[node] = len(members)
            members.append((node, pts))
    xyz = g.to_xyz(lat, lon)
    cent = np.empty((len(members), 2))
    for c, (node, pts) in enumerate(members):
        la, lo = g.from_xyz(xyz[pts].sum(0))
        b = bounds[node]
        if not (b[0] <= la <= b[1] and b[2] <= lo <= b[3]):
            j = pts[np.argmax(xyz[pts] @ g.to_xyz(la, lo))]
            la, lo = lat[j], lon[j]
        cent[c] = la, lo
    return {"max_per_cell": int(max_per_cell), "min_deg": min_deg, "n_points": len(lat),
            "bounds": bounds, "child": child, "cell": cell,
            "centroid_lat": cent[:, 0].tolist(), "centroid_lon": cent[:, 1].tolist(),
            "counts": [len(p) for _, p in members]}


def assign(tree: dict, lat, lon) -> np.ndarray:
    """Cell id of each point (descends the tree; an empty leaf -> the nearest centroid)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    bounds, child = np.asarray(tree["bounds"]), np.asarray(tree["child"])
    node = np.zeros(len(lat), dtype=np.int64)
    while True:
        inner = child[node] >= 0
        if not inner.any():
            break
        b = bounds[node[inner]]
        q = 2 * (lat[inner] >= (b[:, 0] + b[:, 1]) / 2) + (lon[inner] >= (b[:, 2] + b[:, 3]) / 2)
        node[inner] = child[node[inner]] + q
    out = np.asarray(tree["cell"])[node]
    empty = out < 0
    if empty.any():
        cx = g.to_xyz(np.asarray(tree["centroid_lat"]), np.asarray(tree["centroid_lon"]))
        out[empty] = np.argmax(g.to_xyz(lat[empty], lon[empty]) @ cx.T, axis=1)
    return out


def n_cells(tree: dict) -> int:
    return len(tree["counts"])


def for_target(lat, lon, target: int, min_deg: float = MIN_DEG) -> dict:
    """The tree whose cell count is closest to target (bisection on max_per_cell; the count
    does not increase with max_per_cell). Ties -> the larger max_per_cell."""
    lo, hi = 1, max(len(lat), 2)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        t = build(lat, lon, mid, min_deg)
        if best is None or (abs(n_cells(t) - target), -mid) < (abs(n_cells(best) - target),
                                                                -best["max_per_cell"]):
            best = t
        if n_cells(t) > target:
            lo = mid + 1
        else:
            hi = mid - 1
    return {**best, "target": int(target)}


def save(tree: dict, path: Path) -> str:
    """Write the tree as JSON and return its sha256."""
    data = json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()
