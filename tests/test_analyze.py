import math

import numpy as np
import pandas as pd

from geoguessr_ai import analyze as a


def _cm():
    # rows = true, columns = predicted; A is over-predicted (B is often called A)
    return pd.DataFrame(
        [[8, 2, 0], [6, 4, 0], [0, 0, 10]], index=list("ABC"), columns=list("ABC")
    )


def test_wilson_interval():
    lo, hi = a.wilson(np.array([50]), np.array([100]))
    assert math.isclose(lo[0], 0.4038, abs_tol=1e-3)
    assert math.isclose(hi[0], 0.5962, abs_tol=1e-3)
    lo, hi = a.wilson(np.array([0]), np.array([0]))
    assert math.isnan(lo[0]) and math.isnan(hi[0])


def test_per_country_recall_precision_and_attractor_flag():
    pc = a.per_country(_cm(), attractor_ratio=1.3).set_index("country")
    assert math.isclose(pc.loc["A", "recall"], 0.8)
    assert math.isclose(pc.loc["A", "precision"], 8 / 14)
    assert math.isclose(pc.loc["A", "pred_ratio"], 1.4)
    assert pc.loc["A", "attractor"]
    assert not pc.loc["B", "attractor"] and not pc.loc["C", "attractor"]


def test_confusions_sorted_by_count_and_exclude_diagonal():
    c = a.confusions(_cm())
    assert c.iloc[0].tolist()[:3] == ["B", "A", 6]
    assert (c["true"] != c["pred"]).all()
    assert c["count"].is_monotonic_decreasing


def test_pair_table_is_directed():
    t = a.pair_table(_cm(), [["A", "B"]])
    got = {(r.true, r.pred): r.count for r in t.itertuples()}
    assert got == {("A", "B"): 2, ("B", "A"): 6}


def test_main_writes_outputs_and_skips_analysis_dir(tmp_path, monkeypatch):
    results = tmp_path / "results"
    (results / "runA").mkdir(parents=True)
    _cm().to_csv(results / "runA" / "test_confusion.csv")
    monkeypatch.setattr("sys.argv", ["analyze", "--results", str(results)])
    a.main()
    out = results / "analysis"
    assert (out / "report.md").exists() and (out / "precision_recall.csv").exists()
    assert (out / "runA_top_confusions.csv").exists()
    assert (out / "runA_pairs.csv").exists()
    # a second run must not treat the analysis folder as a run
    assert a.find_runs(results, "test") == [results / "runA"]


def _per_country_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"country": list(values), "n": 200, "top1": list(values.values()), "top5": 1.0}
    ).to_csv(path, index=False)


def test_plot_dumbbell_writes_png(tmp_path):
    df_a = pd.DataFrame({"country": ["AA", "BB", "CC"], "n": 200, "top1": [0.9, 0.5, 0.7]})
    df_b = pd.DataFrame({"country": ["AA", "BB", "CC"], "n": 200, "top1": [0.4, 0.6, 0.7]})
    out = tmp_path / "plot.png"
    a.plot_dumbbell(df_a, df_b, "A", "B", "test", out)
    assert out.stat().st_size > 1000


def test_main_writes_dumbbell_only_when_both_runs_exist(tmp_path, monkeypatch):
    results = tmp_path / "results"
    for run in ("runA", "runB"):
        (results / run).mkdir(parents=True)
        _cm().to_csv(results / run / "test_confusion.csv")
    _per_country_csv(results / "runA" / "test_per_country.csv", {"A": 0.8, "B": 0.4, "C": 1.0})
    argv = ["analyze", "--results", str(results), "--plot-runs", "runA", "runB"]
    monkeypatch.setattr("sys.argv", argv)
    a.main()
    assert not (results / "analysis" / "per_country_lpft_vs_streetclip.png").exists()  # runB has no CSV
    _per_country_csv(results / "runB" / "test_per_country.csv", {"A": 0.7, "B": 0.5, "C": 0.9})
    a.main()
    assert (results / "analysis" / "per_country_lpft_vs_streetclip.png").exists()
