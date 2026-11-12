"""Geocell quadtree (Phase 3 Step 2)."""

import hashlib

import numpy as np
import pytest

from geoguessr_ai import geocells as gc


def _pool(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 40, n)
    clat, clon = rng.uniform(-50, 65, 40), rng.uniform(-170, 170, 40)
    return (np.clip(clat[c] + rng.normal(0, 1.5, n), -89, 89),
            np.clip(clon[c] + rng.normal(0, 2.5, n), -179.9, 179.9))


def _box(tree, c):
    return tree["bounds"][tree["cell"].index(c)]


def test_every_point_in_exactly_one_cell_and_counts_near_target():
    lat, lon = _pool()
    for target in (20, 60, 150):
        t = gc.for_target(lat, lon, target)
        a = gc.assign(t, lat, lon)
        assert a.min() >= 0 and a.max() < gc.n_cells(t)
        assert (np.bincount(a, minlength=gc.n_cells(t)) == t["counts"]).all()
        assert sum(t["counts"]) == len(lat) and min(t["counts"]) > 0
        assert abs(gc.n_cells(t) - target) <= max(3, 0.1 * target)
        for c in range(gc.n_cells(t)):  # each point lies inside its cell's box
            b, m = _box(t, c), a == c
            assert (lat[m] >= b[0]).all() and (lat[m] <= b[1]).all()
            assert (lon[m] >= b[2]).all() and (lon[m] <= b[3]).all()


def test_split_rule_and_min_box():
    lat, lon = np.full(50, 10.0), np.full(50, 20.0)  # identical points: only min_deg stops it
    t = gc.build(lat, lon, max_per_cell=5, min_deg=1.0)
    assert gc.n_cells(t) == 1 and t["counts"] == [50]
    b = _box(t, 0)
    assert 0.5 < b[1] - b[0] <= 1.0
    t = gc.build(*_pool(), max_per_cell=200)
    assert max(t["counts"]) <= 200


def test_centroids_on_the_sphere_and_inside_their_box():
    lat, lon = _pool()
    t = gc.for_target(lat, lon, 60)
    for c, (la, lo) in enumerate(zip(t["centroid_lat"], t["centroid_lon"])):
        b = _box(t, c)
        assert -90 <= la <= 90 and -180 <= lo <= 180
        assert b[0] <= la <= b[1] and b[2] <= lo <= b[3]
    # a large sparse box: the 3D mean would leave it (pulled to the pole) -> nearest pool point
    # cell [0, 45) x [-180, -90): the mean of (40, -178) and (40, -92) lies at ~49 N
    t = gc.build(np.array([40.0, 40.0, 80.0]), np.array([-178.0, -92.0, -50.0]), max_per_cell=2)
    c = gc.assign(t, [40.0], [-178.0])[0]
    b, la = _box(t, c), t["centroid_lat"][c]
    assert b == [0.0, 45.0, -180.0, -90.0] and la == 40.0


def test_deterministic_and_saved_with_sha(tmp_path):
    lat, lon = _pool()
    s1 = gc.save(gc.for_target(lat, lon, 60), tmp_path / "a.json")
    s2 = gc.save(gc.for_target(lat, lon, 60), tmp_path / "b.json")
    assert s1 == s2 == hashlib.sha256((tmp_path / "a.json").read_bytes()).hexdigest()
    tree, sha = gc.load(tmp_path / "a.json")
    assert sha == s1 and (gc.assign(tree, lat, lon) == gc.assign(gc.for_target(lat, lon, 60),
                                                                   lat, lon)).all()


def test_point_in_an_empty_leaf_gets_the_nearest_centroid():
    lat, lon = np.r_[np.full(20, 10.0), np.full(20, -40.0)], np.r_[np.full(20, 5.0), np.full(20, 100.0)]
    t = gc.build(lat, lon, max_per_cell=25)
    assert gc.n_cells(t) == 2
    near = gc.assign(t, np.array([12.0, -38.0]), np.array([150.0, -170.0]))  # empty leaves
    assert near[1] == gc.assign(t, [-40.0], [100.0])[0]
    assert near.min() >= 0
    assert gc.assign(t, [-90.0, 90.0], [-180.0, 180.0]).min() >= 0  # world edges belong


@pytest.mark.parametrize("target", [1, 5])
def test_tiny_targets(target):
    lat, lon = _pool(300)
    assert gc.n_cells(gc.for_target(lat, lon, target)) >= 1
