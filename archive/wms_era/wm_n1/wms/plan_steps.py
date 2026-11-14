"""Choose per-stage optimizer steps so the full curriculum fits a wall-clock budget.

Reads a smoke training log (whose records are 20-sim-step updates) and converts
its median step time to ms per simulated step (conservative: includes fixed
overheads).  Prints ``a,b,c,...`` optimizer-step counts, one per curriculum
stage.

The stage list is configurable (Run 2 / B2: extend to 300 s).  For a stage of
``s`` seconds and ratio ``r`` the stage costs ``r * s / dt_s`` sim-steps per unit
``N``, so the full run costs ``N * sum(r_i * s_i / dt_s)`` sim-steps.  The
defaults reproduce the original 10/30/60/120 s plan exactly (cost 290 N); the
5-stage run-2 schedule (10/30/60/120/300 s, ratios 1/1/0.75/0.5/0.25) costs
440 N.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

#: 10 / 30 / 60 / 120 s: the frozen Run-1 curriculum (unchanged default).
DEFAULT_STAGES_S = (10.0, 30.0, 60.0, 120.0)
#: Ratios for the default stages; extra stages repeat the last ratio.
DEFAULT_RATIOS = (1.0, 1.0, 0.75, 0.5, 0.25)


def _ratios_for(n_stages: int) -> list[float]:
    if n_stages < 1:
        raise ValueError("need at least one stage")
    if n_stages <= len(DEFAULT_RATIOS):
        return [float(x) for x in DEFAULT_RATIOS[:n_stages]]
    return [float(x) for x in DEFAULT_RATIOS] + [float(DEFAULT_RATIOS[-1])] * (
        n_stages - len(DEFAULT_RATIOS))


def _resolve(spec, default, name: str):
    if spec is None:
        return list(default)
    if isinstance(spec, str):
        spec = [x for x in spec.replace(",", " ").split() if x]
    return [float(x) for x in spec]


def plan(log: Path, budget_s: float, *, stages_s=None, ratios=None, dt_s: float = 0.5,
         n_min: int = 100, n_max: int = 2500, unroll_steps: int | None = None,
         ) -> tuple[list[int], dict]:
    stages = _resolve(stages_s, DEFAULT_STAGES_S, "stages_s")
    ratios = (_ratios_for(len(stages)) if ratios is None
              else _resolve(ratios, _ratios_for(len(stages)), "ratios"))
    if len(ratios) != len(stages):
        raise ValueError(
            f"ratios has {len(ratios)} values but stages has {len(stages)}")
    if any(r <= 0.0 for r in ratios):
        raise ValueError("ratios must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    cost_per_unit = sum(r * s / dt_s for r, s in zip(ratios, stages))
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if unroll_steps is not None:
        # A multi-stage smoke log mixes unroll lengths; keep only the reference
        # stage's records so ``seconds`` is a per-20-sim-step timing, as the
        # conversion below assumes.
        matched = [r for r in rows if int(r.get("unroll_steps", -1)) == int(unroll_steps)]
        if not matched:
            raise ValueError(f"no log rows with unroll_steps={unroll_steps}")
        rows = matched
    secs = [float(r["seconds"]) for r in rows[5:]] or [float(r["seconds"]) for r in rows]
    per_sim = statistics.median(secs) / 20.0
    n = int(budget_s / (cost_per_unit * per_sim))
    n = max(n_min, min(n_max, n))
    steps = [max(1, int(round(r * n))) for r in ratios]
    info = {"median_step_s_20sim": statistics.median(secs), "s_per_sim_step": per_sim,
            "stages_s": stages, "ratios": ratios,
            "cost_per_unit_sim_steps": cost_per_unit,
            "N": n, "stage_steps": steps,
            "estimated_hours": per_sim * cost_per_unit * n / 3600.0}
    return steps, info


def project(log: Path, stage_steps, *, stages_s=None, warmup_drop: int = 1) -> dict:
    """Project wall-clock hours for an **explicit** per-stage step budget.

    Uses the median measured per-stage optimizer-step time from a multi-stage
    smoke log (one record per stage) and multiplies by the requested explicit
    stage step counts.  Unlike :func:`plan` this never shrinks the budget; the
    caller compares ``projected_hours`` against a hard cap and refuses to start
    if it is exceeded.
    """
    stages = _resolve(stages_s, DEFAULT_STAGES_S, "stages_s")
    stage_steps = [int(x) for x in stage_steps]
    if len(stage_steps) != len(stages):
        raise ValueError(
            f"stage_steps has {len(stage_steps)} values but stages has {len(stages)}")
    if any(n < 1 for n in stage_steps):
        raise ValueError("stage_steps must all be positive")
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_stage: dict[float, list[float]] = {}
    for row in rows:
        if "stage_s" not in row or "seconds" not in row:
            continue
        by_stage.setdefault(round(float(row["stage_s"]), 3), []).append(
            float(row["seconds"]))
    total = 0.0
    detail: dict[str, dict] = {}
    for s, n in zip(stages, stage_steps):
        key = round(float(s), 3)
        secs = by_stage.get(key)
        if not secs:
            raise ValueError(f"smoke log has no records for stage {key} s")
        usable = secs[int(warmup_drop):] if len(secs) > int(warmup_drop) else secs
        if any(not math.isfinite(value) or value <= 0 for value in usable):
            raise ValueError("smoke step timing must be finite and positive")
        med = statistics.median(usable)
        if not math.isfinite(med) or med <= 0:
            raise ValueError("smoke step timing must be finite and positive")
        hours = med * n / 3600.0
        detail[str(key)] = {"stage_steps": int(n), "median_step_s": med,
                            "projected_hours": hours}
        total += hours
    return {"mode": "explicit-stage-steps", "stages_s": stages,
            "stage_steps": stage_steps, "per_stage": detail,
            "projected_hours": total, "estimated_hours": total}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--budget-s", type=float, default=5.5 * 3600)
    ap.add_argument("--stages", default=None,
                    help="comma list of stage unroll lengths in seconds, e.g. "
                         "10,30,60,120,300 (default 10,30,60,120)")
    ap.add_argument("--ratios", default=None,
                    help="comma list of per-stage step ratios (default 1,1,0.75,0.5[,0.25...])")
    ap.add_argument("--dt-s", type=float, default=0.5)
    ap.add_argument("--unroll-steps", type=int, default=None,
                    help="keep only log rows with this unroll_steps (multi-stage "
                         "smoke: 20 for the 10 s reference stage)")
    ap.add_argument("--stage-steps", default=None,
                    help="explicit per-stage optimizer steps; when given, report the "
                         "projected hours instead of shrinking a budget")
    ap.add_argument("--max-hours", type=float, default=10.0,
                    help="refuse (exit 3) when the projected hours exceed this cap")
    a = ap.parse_args()
    if a.stage_steps is not None:
        info = project(a.log, [int(x) for x in a.stage_steps.replace(",", " ").split()],
                       stages_s=a.stages)
        Path(a.log).with_name("plan_steps.json").write_text(json.dumps(info, indent=1))
        print(f"projected_hours={info['projected_hours']:.4f}")
        if info["projected_hours"] > float(a.max_hours):
            print(f"PROJECTED {info['projected_hours']:.3f} h > {a.max_hours} h cap",
                  file=sys.stderr)
            raise SystemExit(3)
        raise SystemExit(0)
    steps, info = plan(a.log, a.budget_s, stages_s=a.stages, ratios=a.ratios,
                       dt_s=a.dt_s, unroll_steps=a.unroll_steps)
    Path(a.log).with_name("plan_steps.json").write_text(json.dumps(info, indent=1))
    print(",".join(str(x) for x in steps))
