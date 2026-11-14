"""``wm_api`` — stable WM scoring API for the planner pilot (api_version 1.0).

Minimal, versioned, documented interface over the composed Idea A bundle
(``adapter.predictor:IdeaABundlePredictor`` modules / ``rollout2`` engine +
P5 correction head):

    scorer = WmScorer(checkpoint_dir)
    result = scorer.score(state, candidate_plans, horizon_s=300,
                          *, members=None, n_samples=None, seed=None)

* ``state``  — the per-vehicle t0 snapshot (:class:`wm_api.state.StateSnapshot`,
  with a converter from the chunk-4 physics-log tick format);
* ``candidate_plans`` — per-CAV plans in the action-descriptor / switch-
  schedule conventions (ACTION_DESCRIPTOR_REQUIREMENT_20261001.md): the
  PLANNED SCHEDULE (30 s segments), never an arm name; per-CAV actions in
  {native, damping, pi, wall v2.2 group membership+params, scripted
  schedules};
* returns :class:`wm_api.result.ScoreResult` — per plan, per metric
  (delay_s_per_veh / ttc3_exposure_s / hard_brake_s): quantiles q05..q95,
  mean and per-member draws; an OOD score; paired differences vs plan 0
  (the reference, native by convention).

Hard refusals: unknown action, missing checkpoint, sha mismatch, K-slot
overflow (reported, never silently dropped CAVs) and ``horizon_s != 300``
unless ``allow_other_horizon=True``.  Deterministic CPU mode (seeded; 1-2
torch threads).  Zero SUMO; tests are synthetic-only.

See ``wm_api/WM_API.md`` for the schema, examples and versioning contract.
"""
from __future__ import annotations

from .errors import (CheckpointError, CheckpointNotFoundError,
                     CheckpointShaMismatch, HorizonError, KOverflowError,
                     PlanSchemaError, StateSchemaError, UnknownActionError,
                     WmApiError)
from .state import ObstacleRecord, StateSnapshot, state_from_physics_tick
from .plans import ResolvedPlan, resolve_plan, wall_gate_records
from .checkpoint import load_checkpoint
from .scorer import WmScorer, load_scorer
from .result import MetricSummary, PlanScore, ScoreResult
from .eval import (has_paired_effect_head, paired_effect_quantiles,
                   paired_effect_quantiles_for_plan,
                   paired_effect_quantiles_from_token)

API_VERSION = "1.0"

__all__ = [
    "API_VERSION",
    "WmScorer", "load_scorer", "score",
    "ScoreResult", "PlanScore", "MetricSummary",
    "StateSnapshot", "ObstacleRecord", "state_from_physics_tick",
    "ResolvedPlan", "resolve_plan", "wall_gate_records",
    "load_checkpoint",
    "has_paired_effect_head", "paired_effect_quantiles",
    "paired_effect_quantiles_for_plan",
    "paired_effect_quantiles_from_token",
    "WmApiError", "UnknownActionError", "CheckpointError",
    "CheckpointNotFoundError", "CheckpointShaMismatch", "HorizonError",
    "KOverflowError", "StateSchemaError", "PlanSchemaError",
]


def score(state, candidate_plans, horizon_s: float = 300.0, *,
          checkpoint, members=None, n_samples: int | None = None,
          seed: int | None = None, **kwargs):
    """Module-level convenience: load ``checkpoint`` and score once.

    Prefer :class:`WmScorer` when scoring repeatedly (the checkpoint is
    loaded once).  Keyword arguments beyond ``checkpoint`` forward to
    :meth:`WmScorer.score`.
    """
    return load_scorer(checkpoint, **kwargs).score(
        state, candidate_plans, horizon_s, members=members,
        n_samples=n_samples, seed=seed)
