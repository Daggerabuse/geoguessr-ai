import math

import numpy as np
import pandas as pd
import pytest

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


def test_default_plot_compares_both_tta_runs(tmp_path, monkeypatch):
    """Without --plot-runs the plot is LP-FT + TTA vs. StreetCLIP + TTA (like for like)."""
    results = tmp_path / "results"
    # a distinct top-1 per run, so the plotted data identifies the run it came from
    top1 = {"clip_vitb16_lpft": 0.1, "clip_vitb16_lpft_tta": 0.2,
            "streetclip_zeroshot": 0.3, "streetclip_zeroshot_tta": 0.4}
    for run, v in top1.items():
        (results / run).mkdir(parents=True)
        _cm().to_csv(results / run / "test_confusion.csv")
        _per_country_csv(results / run / "test_per_country.csv", {"A": v, "B": v})
    calls = []
    monkeypatch.setattr(a, "plot_dumbbell", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["analyze", "--results", str(results)])
    a.main()
    assert len(calls) == 1
    df_a, df_b, label_a, label_b, split, out = calls[0]
    assert set(df_a["top1"]) == {top1["clip_vitb16_lpft_tta"]}
    assert set(df_b["top1"]) == {top1["streetclip_zeroshot_tta"]}
    assert "TTA" in label_a and "TTA" in label_b
    assert split == "test" and out.name == "per_country_lpft_vs_streetclip.png"


def _predictions(path, rows):
    """rows: (id, true, pred, top5 codes); written like baseline_clip / train do."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["id", "true", "pred", "top5"]).to_csv(path, index=False)


def _paired_preds(n_both, n_a_only, n_b_only, n_none):
    """Two prediction tables on the same images with the given top-1 agreement pattern."""
    pattern = ([(1, 1)] * n_both + [(1, 0)] * n_a_only + [(0, 1)] * n_b_only
               + [(0, 0)] * n_none)
    ra, rb = [], []
    for i, (ok_a, ok_b) in enumerate(pattern):
        ra.append((str(i), "A", "A" if ok_a else "B", "A B" if ok_a else "B C"))
        rb.append((str(i), "A", "A" if ok_b else "C", "C A" if ok_b else "C B"))
    cols = ["id", "true", "pred", "top5"]
    return pd.DataFrame(ra, columns=cols), pd.DataFrame(rb, columns=cols)


def test_mcnemar_exact():
    assert a.mcnemar_exact(0, 0) == 1.0
    assert a.mcnemar_exact(3, 3) == 1.0
    assert math.isclose(a.mcnemar_exact(0, 5), 2 / 32)
    assert math.isclose(a.mcnemar_exact(5, 0), 2 / 32)
    # 2 vs. 8: P(X <= 2) for Binomial(10, 0.5) = 56/1024, two-sided
    assert math.isclose(a.mcnemar_exact(2, 8), 2 * 56 / 1024)


def test_paired_comparison_counts_and_difference():
    pa, pb = _paired_preds(n_both=50, n_a_only=30, n_b_only=10, n_none=10)
    df = a.paired_comparison(pa, pb, n_boot=2000, seed=0).set_index("metric")
    t1 = df.loc["top1"]
    assert t1["n"] == 100 and t1["a_only"] == 30 and t1["b_only"] == 10
    assert math.isclose(t1["acc_a"], 0.8) and math.isclose(t1["acc_b"], 0.6)
    assert math.isclose(t1["diff"], 0.2)
    assert t1["ci_lo"] < 0.2 < t1["ci_hi"]
    assert math.isclose(t1["mcnemar_p"], a.mcnemar_exact(30, 10))
    # top-5: a's top-5 has A when right, b's top-5 has A when right -> same pattern here
    assert df.loc["top5", "a_only"] == 30 and df.loc["top5", "b_only"] == 10


def test_paired_comparison_is_seeded_and_matches_image_resampling():
    pa, pb = _paired_preds(n_both=120, n_a_only=40, n_b_only=25, n_none=15)
    r1 = a.paired_comparison(pa, pb, n_boot=4000, seed=7)
    r2 = a.paired_comparison(pa, pb, n_boot=4000, seed=7)
    pd.testing.assert_frame_equal(r1, r2)
    # the multinomial shortcut gives the same interval as resampling images directly
    d = ((pa["pred"] == "A").astype(int) - (pb["pred"] == "A").astype(int)).to_numpy()
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(d), size=(4000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    t1 = r1.set_index("metric").loc["top1"]
    assert abs(t1["ci_lo"] - lo) < 0.015 and abs(t1["ci_hi"] - hi) < 0.015


def test_paired_comparison_rejects_different_images_or_labels():
    pa, pb = _paired_preds(5, 1, 1, 1)
    with pytest.raises(ValueError, match="same images"):
        a.paired_comparison(pa, pb.assign(id=pb["id"] + "x"))
    with pytest.raises(ValueError, match="true labels"):
        a.paired_comparison(pa, pb.assign(true="B"))


def test_main_writes_paired_comparison_for_default_runs(tmp_path, monkeypatch):
    results = tmp_path / "results"
    pa, pb = _paired_preds(n_both=6, n_a_only=3, n_b_only=1, n_none=0)
    for run, pred in (("clip_vitb16_lpft_tta", pa), ("streetclip_zeroshot_tta", pb)):
        (results / run).mkdir(parents=True)
        _cm().to_csv(results / run / "test_confusion.csv")
        _predictions(results / run / "test_predictions.csv", pred.to_numpy().tolist())
    monkeypatch.setattr("sys.argv", ["analyze", "--results", str(results), "--n-boot", "200"])
    a.main()
    out = pd.read_csv(results / "analysis" / "paired_comparison.csv").set_index("metric")
    assert out.loc["top1", "run_a"] == "clip_vitb16_lpft_tta"
    assert out.loc["top1", "run_b"] == "streetclip_zeroshot_tta"
    assert out.loc["top1", "a_only"] == 3 and out.loc["top1", "b_only"] == 1
    assert "Paired comparison" in (results / "analysis" / "report.md").read_text(encoding="utf-8")
    assert len(out) == 2  # the no-TTA pair has no predictions here, so it is skipped


def test_main_writes_one_block_of_rows_per_pair(tmp_path, monkeypatch):
    """Default pairs: LP-FT vs. StreetCLIP with TTA, then without TTA, in one CSV."""
    results = tmp_path / "results"
    tta = _paired_preds(n_both=6, n_a_only=3, n_b_only=1, n_none=0)
    no_tta = _paired_preds(n_both=5, n_a_only=1, n_b_only=4, n_none=0)
    runs = {"clip_vitb16_lpft_tta": tta[0], "streetclip_zeroshot_tta": tta[1],
            "clip_vitb16_lpft": no_tta[0], "streetclip_zeroshot": no_tta[1]}
    for run, pred in runs.items():
        (results / run).mkdir(parents=True)
        _cm().to_csv(results / run / "test_confusion.csv")
        _predictions(results / run / "test_predictions.csv", pred.to_numpy().tolist())
    monkeypatch.setattr("sys.argv", ["analyze", "--results", str(results), "--n-boot", "200"])
    a.main()
    out = pd.read_csv(results / "analysis" / "paired_comparison.csv")
    assert out[["run_a", "run_b", "metric"]].values.tolist() == [
        ["clip_vitb16_lpft_tta", "streetclip_zeroshot_tta", "top1"],
        ["clip_vitb16_lpft_tta", "streetclip_zeroshot_tta", "top5"],
        ["clip_vitb16_lpft", "streetclip_zeroshot", "top1"],
        ["clip_vitb16_lpft", "streetclip_zeroshot", "top5"],
    ]
    top1 = out[out["metric"] == "top1"]
    assert top1["a_only"].tolist() == [3, 1] and top1["b_only"].tolist() == [1, 4]
    report = (results / "analysis" / "report.md").read_text(encoding="utf-8")
    assert report.count("## Paired comparison") == 2


def test_main_rejects_odd_number_of_paired_runs(tmp_path, monkeypatch):
    results = tmp_path / "results"
    (results / "runA").mkdir(parents=True)
    _cm().to_csv(results / "runA" / "test_confusion.csv")
    argv = ["analyze", "--results", str(results), "--paired-runs", "runA", "runB", "runC"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="pairs"):
        a.main()


def test_text_outputs_use_lf_line_endings(tmp_path, monkeypatch):
    """Same bytes on every OS (on Windows pandas/write_text would otherwise write CRLF)."""
    results = tmp_path / "results"
    (results / "runA").mkdir(parents=True)
    _cm().to_csv(results / "runA" / "test_confusion.csv")
    monkeypatch.setattr("sys.argv", ["analyze", "--results", str(results)])
    a.main()
    files = [p for p in (results / "analysis").iterdir() if p.suffix in {".csv", ".md"}]
    assert {p.name for p in files} >= {"report.md", "precision_recall.csv", "runA_pairs.csv",
                                       "runA_top_confusions.csv"}
    for p in files:
        data = p.read_bytes()
        assert b"\n" in data and b"\r" not in data, p.name
