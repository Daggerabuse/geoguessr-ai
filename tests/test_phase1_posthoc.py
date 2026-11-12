import importlib.util
from pathlib import Path

import numpy as np

from geoguessr_ai import geo_eval as g

_spec = importlib.util.spec_from_file_location(
    "phase1_posthoc", Path(__file__).parents[1] / "scripts/analysis/phase1_posthoc.py")
ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ph)


def test_n_equiv_log_linear():
    accs = [0.5, 0.6, 0.7, 0.8]
    assert ph.n_equiv(0.5, accs) == 50_000
    assert np.isclose(ph.n_equiv(0.55, accs), np.sqrt(50_000 * 100_000))  # midpoint in log N
    assert np.isclose(ph.n_equiv(0.8, accs), 500_000)
    assert ph.n_equiv(0.81, accs) == float("inf")


def test_boot_matches_bootstrap_means_per_column():
    rng = np.random.default_rng(0)
    v, cl = rng.random((300, 2)), rng.integers(0, 40, 300)
    b = ph.boot(v, cl, 50, 7)
    for j in range(2):
        assert np.allclose(b[:, j], g.bootstrap_means(v[:, j], cl, 50, 7))


def test_quantiles_inf_safe():
    lo, hi = ph.quantiles(np.r_[np.arange(1.0, 98.0), [np.inf] * 3])
    assert lo == 3.0 and hi == np.inf
