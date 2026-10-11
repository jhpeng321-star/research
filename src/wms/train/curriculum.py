"""Curriculum stages and per-stage optimizer-step budgets.

Stage lengths are the unroll horizons (default 10/30/60/120 s).  A run either gives
explicit ``stage_steps`` (the registered R3/C1FAST recipe 2500/2500/1875/1250) or
derives ``steps_per_epoch * epochs_per_stage`` per stage.  ``plan_from_log`` sizes a
budget from a smoke timing log; ``project_hours`` only *projects* the wall-clock cost
of an explicit budget (it never shrinks it) so the caller can refuse a too-long run.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .settings import TrainSettings

#: 10 / 30 / 60 / 120 s: the frozen curriculum (unchanged default).
DEFAULT_STAGES_S = (10.0, 30.0, 60.0, 120.0)
#: Step-count ratios for the default stages; extra stages repeat the last ratio.
DEFAULT_RATIOS = (1.0, 1.0, 0.75, 0.5, 0.25)


@dataclass(frozen=True)
class StagePlan:
    """One curriculum stage: horizon, unroll length in physics steps, optimizer steps."""

    stage_s: float
    unroll_steps: int
    steps: int
    truncation_steps: int = 0
    """BPTT chunk length of the stage (0 = the global ``truncation_steps``)."""


def stage_plan(settings: TrainSettings, dt_s: float) -> list[StagePlan]:
    """Explicit per-stage plan of ``settings`` at physics step ``dt_s``.

    A stage may carry its own truncation length (``truncation_steps_per_stage``,
    round 5) so the long-horizon stages back-propagate through more of the rollout;
    stages without an entry keep the global ``truncation_steps``.
    """
    default_steps = settings.steps_per_epoch * settings.epochs_per_stage
    per_stage = settings.truncation_steps_per_stage or ()
    plans = []
    for i, stage_s in enumerate(settings.unroll_curriculum_s):
        steps = settings.stage_steps[i] if settings.stage_steps else default_steps
        trunc = int(per_stage[i]) if i < len(per_stage) else 0
        plans.append(StagePlan(float(stage_s), max(1, round(float(stage_s) / dt_s)),
                               int(steps), trunc or int(settings.truncation_steps)))
    return plans


def ratios_for(n_stages: int) -> list[float]:
    if n_stages < 1:
        raise ValueError("need at least one stage")
    if n_stages <= len(DEFAULT_RATIOS):
        return [float(x) for x in DEFAULT_RATIOS[:n_stages]]
    return [float(x) for x in DEFAULT_RATIOS] + [float(DEFAULT_RATIOS[-1])] * (
        n_stages - len(DEFAULT_RATIOS))


def _floats(spec, default: Sequence[float]) -> list[float]:
    if spec is None:
        return list(default)
    if isinstance(spec, str):
        spec = [x for x in spec.replace(",", " ").split() if x]
    return [float(x) for x in spec]


def _read_rows(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def plan_from_log(log: Path, budget_s: float, *, stages_s=None, ratios=None, dt_s: float = 0.5,
                  n_min: int = 100, n_max: int = 2500,
                  unroll_steps: Optional[int] = None) -> tuple[list[int], dict]:
    """Choose per-stage optimizer steps so the curriculum fits ``budget_s`` seconds.

    Reads a smoke log whose records are 20-sim-step updates; converts the median step
    time to seconds per simulated step (conservative: includes fixed overheads).  A
    stage of ``s`` seconds and ratio ``r`` costs ``r * s / dt_s`` sim-steps per unit
    ``N``, so the run costs ``N * sum(r_i * s_i / dt_s)`` sim-steps.
    """
    stages = _floats(stages_s, DEFAULT_STAGES_S)
    ratios = ratios_for(len(stages)) if ratios is None else _floats(ratios, ratios_for(len(stages)))
    if len(ratios) != len(stages):
        raise ValueError(f"ratios has {len(ratios)} values but stages has {len(stages)}")
    if any(r <= 0.0 for r in ratios):
        raise ValueError("ratios must be positive")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    cost_per_unit = sum(r * s / dt_s for r, s in zip(ratios, stages))
    rows = _read_rows(log)
    if unroll_steps is not None:
        # A multi-stage smoke log mixes unroll lengths; keep only the reference stage.
        rows = [r for r in rows if int(r.get("unroll_steps", -1)) == int(unroll_steps)]
        if not rows:
            raise ValueError(f"no log rows with unroll_steps={unroll_steps}")
    secs = [float(r["seconds"]) for r in rows[5:]] or [float(r["seconds"]) for r in rows]
    per_sim = statistics.median(secs) / 20.0
    n = max(n_min, min(n_max, int(budget_s / (cost_per_unit * per_sim))))
    steps = [max(1, int(round(r * n))) for r in ratios]
    info = {"median_step_s_20sim": statistics.median(secs), "s_per_sim_step": per_sim,
            "stages_s": stages, "ratios": ratios, "cost_per_unit_sim_steps": cost_per_unit,
            "N": n, "stage_steps": steps,
            "estimated_hours": per_sim * cost_per_unit * n / 3600.0}
    return steps, info


def project_hours(log: Path, stage_steps: Sequence[int], *, stages_s=None,
                  warmup_drop: int = 1) -> dict:
    """Project wall-clock hours of an **explicit** per-stage step budget from a smoke log.

    Uses the median measured per-stage optimizer-step time (one record group per stage).
    """
    stages = _floats(stages_s, DEFAULT_STAGES_S)
    stage_steps = [int(x) for x in stage_steps]
    if len(stage_steps) != len(stages):
        raise ValueError(f"stage_steps has {len(stage_steps)} values but stages has {len(stages)}")
    if any(n < 1 for n in stage_steps):
        raise ValueError("stage_steps must all be positive")
    by_stage: dict[float, list[float]] = {}
    for row in _read_rows(log):
        if "stage_s" in row and "seconds" in row:
            by_stage.setdefault(round(float(row["stage_s"]), 3), []).append(float(row["seconds"]))
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
        detail[str(key)] = {"stage_steps": int(n), "median_step_s": med, "projected_hours": hours}
        total += hours
    return {"mode": "explicit-stage-steps", "stages_s": stages, "stage_steps": stage_steps,
            "per_stage": detail, "projected_hours": total, "estimated_hours": total}
