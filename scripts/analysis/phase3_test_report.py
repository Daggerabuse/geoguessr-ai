"""Phase 3 TEST outputs -> results/v3: verify the run against the pre-registration
(configs/phase3_final.json, docs/V2_PIPELINE.md), copy the frozen tables and provenance, and write
the reproduction summary, the dev-vs-test table and the two test plots. Reads runs/ only; any
failed check stops before anything is written. No new statistic: every number is either copied
from the run or a linear contrast of its per-cell means (as a consistency check).

  results/v3/test_primary.csv, test_results.csv, test_exploratory_{gaps,contrasts}.csv   copies
  results/v3/test_repro.json, test_dev_check_<session>_<backbone>.json,
             test_read_started_<session>.json, test_prereg_provenance.json            provenance
  results/v3/test_repro_summary.csv   every reproduction row (GPU retrain + each CPU check)
  results/v3/test_dev_vs_test.csv     H1-H5 and E1 on dev_seen / dev_unseen next to test
  results/v3/test_exploratory_nat_shrink.csv   the nat shrink with its creator interval (promised
                                      as exploratory; the run's contrast and bootstrap; needs
                                      data/raw/test.csv for the creator ids)
  results/v3/test_scaling_geo.png, test_gap_geo.png

  uv run python scripts/analysis/phase3_test_report.py
"""

import filecmp
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from geoguessr_ai import phase1_eval as pe
from geoguessr_ai import phase3_final as f3
from geoguessr_ai import prereg
from geoguessr_ai.v2_data import normalize_creators

ROOT = Path(__file__).resolve().parents[2]
RUNS, OUT = ROOT / "runs", ROOT / "results/v3"
PREREG = ROOT / "configs/phase3_final.json"
PREREG_COMMIT = "86d9f8b328b0bf93f626d0dea0b17bb04c46f540"
PREREG_SHA = "6aef5a15fed9662f8e586be417f48d05e7871561edf096afbb4db26c8228d66c"
PROBES = RUNS / "phase3_test_probes/phase3_probes"
SESSIONS = {"streetclip": RUNS / "phase3_test_streetclip/phase3_final",
            "clip_l14_336": RUNS / "phase3_test_l14/phase3_final"}  # session -> backbone it scored
MERGED = RUNS / "phase3_test_primary/phase3_final"
TOL = {"medoid": 0.5, "cell": 2.0, "hybrid": 2.0}  # |GeoScore diff|, from the pre-registration


def check(cond, msg: str) -> None:
    if not cond:
        raise SystemExit(f"CHECK FAILED: {msg}")
    print(f"ok  {msg}")


def key(t: dict) -> tuple:
    return t["backbone"], t["ordering"], int(t["n"]), t["predictor"]


def contrast(means: dict, terms: list) -> float:
    return sum(t["w"] * means[key(t)] for t in terms)


def verify(reg: dict) -> dict:
    blob = subprocess.run(["git", "show", f"{PREREG_COMMIT}:configs/phase3_final.json"],
                          cwd=ROOT, capture_output=True, check=True).stdout
    check(hashlib.sha256(blob).hexdigest() == PREREG_SHA, "prereg file at 86d9f8b has sha 6aef5a15")
    print(f"    local configs/phase3_final.json sha256 {prereg.sha256(PREREG)[:8]}")
    run = {"prereg_sha256": PREREG_SHA, "code_commit": PREREG_COMMIT,
           "repro_sha256": prereg.sha256(PROBES / "repro.json")}
    prov = json.loads((PROBES / "prereg_provenance.json").read_text())
    check(prov["prereg_sha256"] == PREREG_SHA and prov["prereg_commit"] == PREREG_COMMIT,
          "probes provenance names the pre-registration and its commit")

    repro = json.loads((PROBES / "repro.json").read_text())
    check(repro["ok"] and repro["declared"] == repro["done"] == len(repro["cells"]) == 32
          and all(c["ok"] and c["code_commit"] == PREREG_COMMIT for c in repro["cells"]),
          "GPU reproduction: 32/32 probes ok, all at the pre-registered commit")

    cells = {key(c) for c in reg["cells"]}
    for s, d in SESSIONS.items():
        checked = []
        for b in SESSIONS:
            dc = json.loads((d / b / "dev_check.json").read_text())
            check(dc["run"] == run, f"{s} session, {b} dev_check.json: same run identity")
            checked += dc["cells"]
        check(len(checked) == 80 and {key(c) for c in checked} == cells
              and all(c["ok"] for c in checked), f"{s} session: dev check 80/80 declared cells ok")
    for d in [*SESSIONS.values(), MERGED]:
        st = json.loads((d / "test_read_started.json").read_text())
        check({k: st[k] for k in run} == run, f"{d.parent.name}: test_read_started run identity")

    rows = []
    for c in reg["cells"]:
        rel = pe.cell_dir(Path("."), c["backbone"], c["ordering"], c["n"], "geo", c["predictor"])
        row = json.loads((MERGED / rel / "test_row.json").read_text())
        assert row["run"] == run, rel
        for f in ("test_row.json", "test_geo_hits.csv"):
            assert filecmp.cmp(MERGED / rel / f, SESSIONS[c["backbone"]] / rel / f, shallow=False), rel
        rows.append(row)
    check(len(rows) == 80, "80 merged test_row.json: one run identity, byte-identical to the "
                           "session that scored them (test_row.json and test_geo_hits.csv)")
    res = pd.read_csv(MERGED / "results.csv")
    check(len(res) == 80 and res["run"].nunique() == 1 and (res["test_n"] == 210122).all(),
          "results.csv: 80 cells, one run, all 210,122 test images")

    for log, must in [(RUNS / "phase3_test_probes/phase3_probes.log", "32 of 32 cells inside"),
                      *[(d.parent / "phase3_final.log", "80 of 80 cells inside")
                        for d in SESSIONS.values()],
                      (RUNS / "phase3_test_primary/phase3_final.log", "PRIMARY (pre-registered")]:
        check(must in log.read_text(encoding="utf-8"), f"{log.parent.name}: '{must}'")
    for nb in sorted(RUNS.glob("phase3_test_*/notebook*.log")):
        t = nb.read_text(encoding="utf-8")
        check(not any(w in t for w in ("Traceback", "FAILED", "RESUME NEEDED"))
              and "__results__.html" in t, f"{nb.parent.name}: no Traceback/FAILED/RESUME NEEDED, "
                                           "notebook ran to the end")
    for b, d in SESSIONS.items():
        t = (d.parent / "phase3_final.log").read_text(encoding="utf-8")
        check(sum(f"{b} {o} {n}: test scored" in t for o in ("nat", "bal")
                  for n in (50000, 100000, 200000, 500000)) == 8, f"{b}: 8 of 8 subsets scored")
    return run


def consistency(reg: dict) -> dict:
    res = pd.read_csv(MERGED / "results.csv")
    means = {key(r): r for r in res.to_dict("records")}
    prim = pd.read_csv(MERGED / "primary.csv").set_index("id")
    expl = pd.read_csv(MERGED / "exploratory_contrasts.csv").set_index("id")
    for h, tab in [*[(h, prim) for h in reg["hypotheses"]["primary"]],
                   *[(h, expl) for h in reg["hypotheses"]["exploratory"]]]:
        est = contrast({k: v["test_geoscore"] for k, v in means.items()}, h["terms"])
        check(abs(est - tab.loc[h["id"], "estimate"]) < 1e-6,
              f"{h['id']}: estimate {est:+.2f} = contrast of the per-cell test means")
    return means


def repro_summary(reg: dict) -> pd.DataFrame:
    rows = [{"stage": "gpu_retrain", **c} for c in json.loads((PROBES / "repro.json")
                                                              .read_text())["cells"]]
    for s, d in SESSIONS.items():
        for b in SESSIONS:
            rows += [{"stage": f"cpu_check_{s}_session", **c}
                     for c in json.loads((d / b / "dev_check.json").read_text())["cells"]]
    t = pd.DataFrame(rows)
    t["abs_diff"] = t["diff"].abs()
    t["tol_geoscore"] = t["predictor"].str.split("_").str[0].map(TOL)
    t["tol_ok"] = (t["abs_diff"] <= t["tol_geoscore"]) & (t["top1_cell_same"].fillna(1) >= 0.99) \
        & (t["top5_set_same"].fillna(1) >= 0.95)
    check(t["tol_ok"].all() and t["ok"].all(), f"{len(t)} reproduction rows inside every tolerance "
                                               "(recomputed here, and as recorded)")
    cols = ["stage", "backbone", "ordering", "n", "predictor", "lr", "epoch", "top1_cell_same",
            "top5_set_same", "geoscore", "stored", "diff", "abs_diff", "tol_geoscore", "tol_ok",
            "head_sha256"]
    t = t[cols]
    print(t.groupby(["stage", "predictor"]).agg(max_abs_diff=("abs_diff", "max"),
                                                min_top1=("top1_cell_same", "min"),
                                                min_top5=("top5_set_same", "min")).to_string())
    return t


def dev_vs_test(reg: dict, means: dict) -> pd.DataFrame:
    gap = pd.read_csv(OUT / "phase3_cell_backbone_gap.csv")
    shrink = pd.read_csv(OUT / "phase3_backbone_shrink.csv")
    prim = pd.read_csv(MERGED / "primary.csv").set_index("id")
    expl = pd.read_csv(MERGED / "exploratory_contrasts.csv").set_index("id")
    rows = []
    for h in [*reg["hypotheses"]["primary"], *reg["hypotheses"]["exploratory"]]:
        t = prim.loc[h["id"]] if h["id"] in prim.index else expl.loc[h["id"]]
        row = {"id": h["id"], "test": t["estimate"], "test_ci_lo": t["ci_lo"],
               "test_ci_hi": t["ci_hi"], "test_verdict": t.get("verdict", "exploratory")}
        a = h["terms"][0]
        for split in ("dev_seen", "dev_unseen"):
            row[split] = contrast({k: v[f"{split}_geoscore_prereg"] for k, v in means.items()},
                                  h["terms"])
            if h["id"].startswith(("H1", "H2")):
                m = gap[(gap["variant"] == f"geo_{a['predictor']}") & (gap["other"] == "clip_l14_336")
                        & (gap["ordering"] == a["ordering"]) & (gap["n"] == a["n"])
                        & (gap["split"] == split)]
                lo, hi, est = m["gap_lo"], m["gap_hi"], m["gap"]
            elif h["id"].startswith("H3"):
                m = shrink[(shrink["variant"] == f"geo_{a['predictor']}")
                           & (shrink["other"] == "clip_l14_336")
                           & (shrink["ordering"] == a["ordering"]) & (shrink["split"] == split)]
                lo, hi, est = m["shrink_lo"], m["shrink_hi"], m["shrink"]
            else:  # no stored dev interval for the init x backbone and hybrid-vs-medoid contrasts
                continue
            assert len(m) == 1 and abs(est.iloc[0] - row[split]) < 1e-6, (h["id"], split)
            row[f"{split}_ci_lo"], row[f"{split}_ci_hi"] = lo.iloc[0], hi.iloc[0]
        rows.append(row)
    cols = ["id", "dev_seen", "dev_seen_ci_lo", "dev_seen_ci_hi", "dev_unseen", "dev_unseen_ci_lo",
            "dev_unseen_ci_hi", "test", "test_ci_lo", "test_ci_hi", "test_verdict"]
    t = pd.DataFrame(rows)[cols]
    print(t.round(1).to_string(index=False))
    return t


def nat_shrink(reg: dict) -> pd.DataFrame:
    """The nat shrink the pre-registration promises as exploratory "with its interval": the
    run's own contrast and creator bootstrap (n_boot and seed of its exploratory intervals)."""
    st, pred = reg["settings"], reg["primary"]["predictor"]
    d = normalize_creators(pd.read_csv(ROOT / "data/raw/test.csv", dtype=str,
                                       keep_default_na=False,
                                       usecols=["id", "sequence", "creator_id"])).set_index("id")
    x = f3.contrast(MERGED, f3._gap("nat", 50000, pred) + f3._gap("nat", 500000, pred, -1.0))
    b = f3._boot(x, d["creator_id"], st["n_boot"], st["seed"])
    t = pd.DataFrame([{"analysis": "exploratory", "id": "nat_shrink",
                       "claim": f"gap(N=50000) - gap(N=500000), {pred}, nat; no direction "
                                "pre-registered, uncorrected creator interval, no verdict",
                       "estimate": float(x.mean()), "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                       "n_images": len(x)}])
    g = pd.read_csv(MERGED / "exploratory_gaps.csv").set_index(["ordering", "n", "predictor"])
    check(abs(t["estimate"][0] - (g.loc[("nat", 50000, pred), "gap"]
                                  - g.loc[("nat", 500000, pred), "gap"])) < 1e-6,
          f"nat shrink {t['estimate'][0]:+.1f} [{b['ci_lo']:.1f}, {b['ci_hi']:.1f}] matches the gaps")
    return t


def plots() -> None:
    res = pd.read_csv(MERGED / "results.csv").rename(columns={"predictor": "variant"})
    pe.plot_scaling(res, OUT / "test_scaling_geo.png", ["test"], "geoscore", "test GeoScore")
    test = pd.read_csv(MERGED / "exploratory_gaps.csv").rename(columns={"predictor": "variant"})
    test["split"] = "test (creator CI)"
    dev = pd.read_csv(OUT / "phase3_cell_backbone_gap.csv")
    dev = dev[dev["other"] == "clip_l14_336"].assign(variant=lambda d: d["variant"].str[4:])
    dev["split"] = dev["split"].map({"dev_seen": "dev_seen (sequence CI)",
                                     "dev_unseen": "dev_unseen (creator CI)"})
    cols = ["variant", "ordering", "n", "split", "gap", "gap_lo", "gap_hi"]
    pe.plot_gap(pd.concat([dev[cols], test[cols]]), OUT / "test_gap_geo.png",
                "GeoScore difference (paired)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    reg = json.loads(PREREG.read_text())
    verify(reg)
    means = consistency(reg)
    rs, dvt, ns = repro_summary(reg), dev_vs_test(reg, means), nat_shrink(reg)
    for src, dst in [(MERGED / "primary.csv", "test_primary.csv"),
                     (MERGED / "results.csv", "test_results.csv"),
                     (MERGED / "exploratory_gaps.csv", "test_exploratory_gaps.csv"),
                     (MERGED / "exploratory_contrasts.csv", "test_exploratory_contrasts.csv"),
                     (PROBES / "repro.json", "test_repro.json"),
                     (PROBES / "prereg_provenance.json", "test_prereg_provenance.json"),
                     *[(d / "test_read_started.json", f"test_read_started_{s}.json")
                       for s, d in SESSIONS.items()],
                     *[(d / b / "dev_check.json", f"test_dev_check_{s}_session_{b}.json")
                       for s, d in SESSIONS.items() for b in SESSIONS]]:
        shutil.copyfile(src, OUT / dst)
    rs.to_csv(OUT / "test_repro_summary.csv", index=False)
    dvt.to_csv(OUT / "test_dev_vs_test.csv", index=False)
    ns.to_csv(OUT / "test_exploratory_nat_shrink.csv", index=False)
    plots()
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
