import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geoguessr_ai import geo_eval as g
from tests.test_phase1_eval import setup  # noqa: F401  (fixture)

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


def test_cpu_matched_k_predictions_and_agreement(setup, tmp_path):  # noqa: F811
    root, _ = setup
    args = argparse.Namespace(data=root / "v2", emb_root=root / "emb", test_emb_root=root / "emb")
    t = pd.DataFrame([{"ordering": "nat", "n": 30, "k_streetclip": 5, "k_l14": 1, "k_shared": 5,
                       "recompute": True}])
    cpu = ph.cpu_predictions(t, args)
    assert set(cpu.columns) == {"backbone", "ordering", "n", "k", "split", "id", "pred", "top1"}
    assert set(cpu["split"]) == {"dev_seen", "dev", "test"}
    assert set(cpu[cpu["backbone"] == "clip_l14_336"]["k"]) == {1, 5}
    saved = tmp_path / "saved"
    for bb, k in (("streetclip", 5), ("clip_l14_336", 1)):  # saved = the CPU run at frozen k
        d = saved / bb / "nat_30_knn"
        d.mkdir(parents=True)
        for s in ("dev_seen", "dev", "test"):
            c = cpu[(cpu["backbone"] == bb) & (cpu["k"] == k) & (cpu["split"] == s)]
            c[["id", "pred"]].to_csv(d / f"{s}_predictions.csv", index=False)
    m = ph.merge_cpu(t, cpu, {s: saved for s in ("dev_seen", "dev", "test")})
    assert m.loc[0, "test_agree_streetclip_k5"] == 1.0 and m.loc[0, "dev_agree_clip_l14_336_k1"] == 1.0
    top = cpu[(cpu["k"] == 5) & (cpu["split"] == "test")].groupby("backbone")["top1"].mean()
    assert m.loc[0, "test_gap_matched"] == pytest.approx(top["streetclip"] - top["clip_l14_336"])
