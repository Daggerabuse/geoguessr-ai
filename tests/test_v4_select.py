"""v4_select: the declared lr selection, tie and grid-edge rules (docs/V4_FEASIBILITY.md, section 5)."""

import json

import pytest

from geoguessr_ai import v4_select as vs

GRID = [1e-4, 3e-4, 1e-3]


def sel(a, b, extended=False):
    return vs.select({"streetclip": a, "clip_l14_336": b}, GRID, 3, 5.0, extended)


def test_tie_keeps_the_lower_lr_at_and_below_5():
    assert vs.pick({1e-4: 100, 3e-4: 105, 1e-3: 90}, 5.0) == 1e-4  # exactly 5: the lower
    assert vs.pick({1e-4: 100, 3e-4: 105.01, 1e-3: 90}, 5.0) == 3e-4
    assert vs.pick({1e-4: 90, 3e-4: 100, 1e-3: 103}, 5.0) == 3e-4  # only the two best count


def test_interior_choices_need_no_extension():
    r = sel({1e-4: 1, 3e-4: 9, 1e-3: 2}, {1e-4: 1, 3e-4: 20, 1e-3: 1})
    assert r["edge"] == {"run": False} and r["selected"] == {"streetclip": 3e-4, "clip_l14_336": 3e-4}
    assert r["flags"] == {}


def test_one_backbone_at_an_edge_extends_for_both():
    r = sel({1e-4: 1, 3e-4: 2, 1e-3: 30}, {1e-4: 1, 3e-4: 9, 1e-3: 2})
    assert r["edge"] == {"run": True, "side": "high", "lr": 0.003} and r["selected"] is None
    r = sel({1e-4: 30, 3e-4: 2, 1e-3: 1}, {1e-4: 40, 3e-4: 2, 1e-3: 1})
    assert r["edge"]["lr"] == pytest.approx(3.33333e-05) and r["edge"]["side"] == "low"


def test_opposite_edges_flag_both_and_extend_nothing():
    r = sel({1e-4: 30, 3e-4: 2, 1e-3: 1}, {1e-4: 1, 3e-4: 2, 1e-3: 30})
    assert r["edge"] == {"run": False}
    assert r["selected"] == {"streetclip": 1e-4, "clip_l14_336": 1e-3}
    assert r["flags"] == {"streetclip": "grid_edge", "clip_l14_336": "grid_edge"}
    assert r["margins"] == {"streetclip": 28, "clip_l14_336": 28}


def test_after_the_extension_the_grid_is_frozen_and_flagged():
    a = {1e-4: 1, 3e-4: 2, 1e-3: 30, 0.003: 50}
    b = {1e-4: 1, 3e-4: 9, 1e-3: 2, 0.003: 0}
    r = sel(a, b, extended=True)
    assert r["grid"] == [1e-4, 3e-4, 1e-3, 0.003] and r["edge"]["done"] and not r["edge"]["run"]
    assert r["selected"] == {"streetclip": 0.003, "clip_l14_336": 3e-4}
    assert r["flags"] == {"streetclip": "grid_edge"} and r["margins"]["streetclip"] == 20
    with pytest.raises(SystemExit, match="exactly one"):
        sel({**a, 0.01: 1}, {**b, 0.01: 1}, extended=True)


def test_missing_result_refused_and_matched_lr_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="no final result"):
        sel({1e-4: 1, 3e-4: 2}, {1e-4: 1, 3e-4: 2, 1e-3: 3})
    assert vs.matched_lr({"a": {1e-4: 10, 3e-4: 0}, "b": {1e-4: 0, 3e-4: 11}}) == 3e-4
    jobs = [{"k": k, "name": f"B_short_{lr}", "kind": "short", "lr": lr} for k, lr in enumerate(GRID)]
    cfg = {"backbones": {"streetclip": "", "clip_l14_336": ""}, "jobs": jobs,
           "selection": {"lr_grid": GRID, "step": 3, "tie_tol": 5.0}}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    for bb in cfg["backbones"]:
        for j, gs in zip(jobs, (1, 9, 2)):
            for job, score in ((j, gs), ({**j, "name": f"A_short_{j['lr']}"}, 100 * gs * (j["k"] != 1))):
                d = tmp_path / "runs" / f"{job['name']}_{bb}"  # another recipe's run is ignored
                d.mkdir(parents=True)
                (d / "result.json").write_text(json.dumps({"kind": "short", "backbone": bb, "lr": j["lr"],
                                                          "job": job, "dev": {"geoscore": {"mean": score}}}))
    out = tmp_path / "sel.json"
    vs.main(["--runs", str(tmp_path / "runs"), "--config", str(tmp_path / "c.json"), "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["selected"] == {"streetclip": 3e-4, "clip_l14_336": 3e-4} and r["matched_lr_sensitivity"] == 3e-4
