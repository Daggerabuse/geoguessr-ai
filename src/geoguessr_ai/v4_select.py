"""V4 lr selection from the short-horizon search (docs/V4_FEASIBILITY.md, section 5).

Rules, as declared before any V4 dev result:
- per backbone, the lr with the best dev_unseen GeoScore at the FINAL checkpoint; if the two best
  are within tie_tol (5 GeoScore, a tie at exactly 5 included), the lower of the two is kept;
- grid edge: if the selected lr sits at the SAME edge of the grid for at least one backbone (and
  no backbone at the other edge), the next x3 step beyond it runs for BOTH backbones (one grid);
  opposite edges (or a lower and an upper edge at once): no extension, both flagged grid_edge
  with the margin to their neighbour;
- stopping rule: that extension is the only one. If the extension lr is selected or an edge still
  binds afterwards, the grid is frozen and flagged.
- the matched lr (best average over the backbones) is a sensitivity analysis, never a selection.
B-S2: the short-run choice is final for the full runs.

  uv run python -m geoguessr_ai.v4_select --runs <dir with k*/result.json> --config configs/v4_predev.json \\
      --out configs/v4_selection.json
"""

import argparse
import json
import sys
from pathlib import Path


def pick(scores: dict[float, float], tie_tol: float) -> float:
    """Best lr; the lower of the two best when they are within tie_tol."""
    order = sorted(scores, key=lambda lr: (-scores[lr], lr))
    if len(order) > 1 and scores[order[0]] - scores[order[1]] <= tie_tol:
        return min(order[:2])
    return order[0]


def edge_of(lr: float, grid: list[float]) -> str | None:
    g = sorted(grid)
    return "low" if lr == g[0] else "high" if lr == g[-1] else None


def margin(scores: dict[float, float], lr: float) -> float:
    """GeoScore of the selected edge lr minus its inward neighbour's."""
    g = sorted(scores)
    i = g.index(lr)
    return scores[lr] - scores[g[1] if i == 0 else g[-2]]


def select(scores: dict[str, dict[float, float]], grid: list[float], step: float, tie_tol: float,
           extended: bool) -> dict:
    """scores {backbone: {lr: dev_unseen GeoScore}} over `grid` (plus the extension lr when
    `extended`). Returns the edge decision and, unless an extension must run first, the lr per
    backbone with flags."""
    grid = sorted(grid)
    for bb, s in scores.items():
        missing = set(grid) - set(s)
        if missing:
            raise SystemExit(f"{bb}: no final result for lr {sorted(missing)}")
    chosen = {bb: pick({lr: s[lr] for lr in grid}, tie_tol) for bb, s in scores.items()}
    edges = {bb: edge_of(lr, grid) for bb, lr in chosen.items()}
    bound = {e for e in edges.values() if e}
    out = {"grid": grid, "tie_tol": tie_tol, "extended": extended, "flags": {}, "margins": {}}

    if not extended:
        if len(bound) == 1:
            side = bound.pop()
            lr = grid[0] / step if side == "low" else grid[-1] * step
            return {**out, "edge": {"run": True, "side": side, "lr": float(f"{lr:.6g}")},
                    "selected": None, "first_pass": chosen}
        out["edge"] = {"run": False}
        flagged = set(edges) if len(bound) > 1 else set()  # opposite edges: neither is extended
    else:
        extra = sorted(set().union(*map(set, scores.values())) - set(grid))
        if len(extra) != 1:
            raise SystemExit(f"extended: expected exactly one extension lr, found {extra}")
        full = sorted(grid + extra)
        chosen = {bb: pick({lr: s[lr] for lr in full}, tie_tol) for bb, s in scores.items()}
        edges = {bb: edge_of(lr, full) for bb, lr in chosen.items()}
        out.update(grid=full, edge={"run": False, "done": True, "lr": extra[0]})
        flagged = {bb for bb, e in edges.items() if e}  # stopping rule: frozen, flagged
    for bb in flagged:
        if edges[bb]:
            out["flags"][bb] = "grid_edge"
            out["margins"][bb] = margin({lr: scores[bb][lr] for lr in out["grid"]}, chosen[bb])
    out["selected"] = chosen
    return out


def matched_lr(scores: dict[str, dict[float, float]]) -> float:
    """Sensitivity only: the lr with the best average over the backbones."""
    common = set.intersection(*(set(s) for s in scores.values()))
    return max(sorted(common), key=lambda lr: sum(s[lr] for s in scores.values()) / len(scores))


def read_scores(runs: Path, cfg: dict) -> dict[str, dict[float, float]]:
    """Final dev_unseen GeoScore of every finished short run under `runs`."""
    scores: dict[str, dict[float, float]] = {}
    for f in sorted(Path(runs).glob("*/result.json")):
        r = json.loads(f.read_text())
        if r["kind"] != "short":
            continue
        scores.setdefault(r["backbone"], {})[r["lr"]] = r["dev"]["geoscore"]["mean"]
    if set(scores) != set(cfg["backbones"]):
        raise SystemExit(f"short results for {sorted(scores)}, expected {sorted(cfg['backbones'])}")
    return scores


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/v4_predev.json"))
    ap.add_argument("--out", type=Path, default=Path("configs/v4_selection.json"))
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text())
    sel = cfg["selection"]
    scores = read_scores(a.runs, cfg)
    extended = any(len(s) > len(sel["lr_grid"]) for s in scores.values())
    res = select(scores, sel["lr_grid"], sel["step"], sel["tie_tol"], extended)
    res["scores"] = {bb: {str(lr): v for lr, v in sorted(s.items())} for bb, s in scores.items()}
    res["matched_lr_sensitivity"] = matched_lr(scores)
    a.out.write_text(json.dumps(res, indent=2) + "\n", newline="\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
