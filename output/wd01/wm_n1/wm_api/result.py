"""``ScoreResult`` — per-plan metric summaries, OOD and paired differences.

Quantile keys are ``q05..q95`` (the frozen 19-level grid 0.05..0.95).
Conventions (all documented in WM_API.md):

* a member's "draw" for a metric is the member's median (q50) of that
  metric; ``mean`` is the unweighted mean of the per-member draws;
* the plan-level quantile curve is the unweighted mean of the member
  quantile curves (the ensemble mixture convention of
  ``model.ensemble.WmN1Ensemble.mixture_quantiles``);
* paired differences vs plan 0 are comonotone at the SAME levels
  (``q_plan_i - q_plan_0``, the ``evalh.metrics.build_delta_pairs``
  convention).  ``cost difference = treatment - reference; negative =
  improvement``; plan 0 is the reference (native by convention) and its
  paired block is all zeros with ``"reference": true``;
* the OOD score is larger = less familiar (``corr.ood`` conventions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .state import K_MAX

#: The frozen 19 quantile levels (model.constants.QUANTILE_LEVELS).
QUANTILE_LEVELS = tuple(round(0.05 * i, 4) for i in range(1, 20))

METRIC_NAMES = ("delay_s_per_veh", "ttc3_exposure_s", "hard_brake_s")


def level_key(level: float) -> str:
    """``0.05 -> 'q05'``, ``0.95 -> 'q95'`` (two-digit percent)."""
    return f"q{int(round(float(level) * 100)):02d}"


def quantile_dict(curve, *, monotone: bool = True) -> dict:
    """``[19]`` quantile curve -> ``{q05..q95: float}``.

    ``monotone=True`` (outcome curves) accumulates the maxima so rounding
    cannot break monotonicity; paired-difference curves pass
    ``monotone=False`` (a comonotone difference is not itself monotone).
    """
    arr = np.asarray(curve, dtype=np.float64)
    if monotone:
        arr = np.maximum.accumulate(arr)
    return {level_key(lv): float(arr[i])
            for i, lv in enumerate(QUANTILE_LEVELS)}


@dataclass
class MetricSummary:
    """One metric for one plan: quantiles + mean + per-member draws."""

    quantiles: dict                 # {q05..q95: float}
    mean: float
    member_draws: list              # [float per scored member]

    def to_json(self) -> dict:
        return dict(quantiles=dict(self.quantiles), mean=float(self.mean),
                    member_draws=[float(x) for x in self.member_draws])


@dataclass
class PlanScore:
    """One candidate plan's score (metrics, OOD, paired differences)."""

    plan_index: int
    descriptor: str
    metrics: dict                   # metric name -> MetricSummary
    ood_score: float
    paired_diff_vs_plan0: Optional[dict]   # metric name -> {quantiles, mean}
    is_reference: bool = False
    wall_gates: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return dict(
            plan_index=int(self.plan_index),
            descriptor=self.descriptor,
            is_reference=bool(self.is_reference),
            metrics={k: v.to_json() for k, v in self.metrics.items()},
            ood_score=float(self.ood_score),
            paired_diff_vs_plan0=(None if self.paired_diff_vs_plan0 is None
                                  else {k: dict(quantiles=dict(v["quantiles"]),
                                                mean=float(v["mean"]))
                                        for k, v in
                                        self.paired_diff_vs_plan0.items()}),
            wall_gates=dict(self.wall_gates))


@dataclass
class ScoreResult:
    """The full scoring response (versioned; JSON-serialisable)."""

    api_version: str
    horizon_s: float
    tau_s: float
    plans: list                     # PlanScore, in candidate order
    members: list                   # scored 1-based member numbers
    seed: int
    n_samples: int
    arrivals_enabled: bool
    checkpoint: dict                # {directory, shas, sha_verified, ...}
    meta: dict = field(default_factory=dict)

    def to_json(self, include_timing: bool = True) -> dict:
        meta = dict(self.meta)
        if not include_timing:
            meta.pop("timing_s", None)
        return dict(
            api_version=str(self.api_version),
            horizon_s=float(self.horizon_s),
            tau_s=float(self.tau_s),
            n_plans=len(self.plans),
            members=[int(m) for m in self.members],
            seed=int(self.seed),
            n_samples=int(self.n_samples),
            arrivals_enabled=bool(self.arrivals_enabled),
            k_slots_max=int(K_MAX),
            plans=[p.to_json() for p in self.plans],
            checkpoint=dict(self.checkpoint),
            meta=meta)

    def to_json_text(self, include_timing: bool = True) -> str:
        import json

        return json.dumps(self.to_json(include_timing=include_timing),
                          indent=2, sort_keys=True)
