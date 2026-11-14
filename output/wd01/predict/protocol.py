"""Predictor protocol (§2 API).

Implementations:
- perfect simulator (SUMO saveState/loadState branching): planner-owned,
  M = 1, K = arrival draws with common random numbers.
- learned (Part A rollout + Part B delay estimator): world-model-owned,
  M = ensemble size, K = hidden-vehicle samples x arrival draws.
- reference_const_speed.ConstSpeedPredictor: conformance reference only.

Contract API (§2):
    predict_many(state_samples, plans, controller_state, *,
                 horizon_s=600.0, trajectory_s=None) -> list[Prediction]
    predict(state_samples, plan, controller_state, **kw)
        == predict_many(state_samples, [plan], controller_state, **kw)[0]

Output order matches `plans`. The native reference (§5) is computed once per
call and shared by every Prediction in the call. trajectory_s=None means the
model's own part_a_horizon_s; implementations may reject
trajectory_s > part_a_horizon_s. All output arrays are numpy (CPU). Every call
is deterministic given inputs, weights (sha256) and arrival seeds.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from .contract import (
    CONTINUATION_PLANNER,
    DEFAULT_COMMIT_S,
    DEFAULT_HORIZON_S,
    ControllerState,
    PlanCandidate,
    Prediction,
    StateSample,
)


@runtime_checkable
class Predictor(Protocol):
    """Structural interface every predict() implementation satisfies."""

    def predict_many(
        self,
        state_samples: Sequence[StateSample],
        plans: Sequence[PlanCandidate],
        controller_state: ControllerState,
        *,
        horizon_s: float = DEFAULT_HORIZON_S,
        trajectory_s: "float | None" = None,
    ) -> "list[Prediction]": ...

    def predict(
        self,
        state_samples: Sequence[StateSample],
        plan: PlanCandidate,
        controller_state: ControllerState,
        **kw: Any,
    ) -> Prediction: ...


class PredictorMixin:
    """Default `predict()` in terms of `predict_many()` (§2 identity).
    Mix into an implementation that defines predict_many."""

    def predict(
        self,
        state_samples: Sequence[StateSample],
        plan: PlanCandidate,
        controller_state: ControllerState,
        **kw: Any,
    ) -> Prediction:
        return self.predict_many(state_samples, [plan], controller_state, **kw)[0]


@runtime_checkable
class PredictorV2(Protocol):
    """§2 (v2) interface: the commit-window API. `commit_s` is Δ (prediction
    semantics only, not a dispatch dwell); after Δ both branches continue
    under `continuation` ("planner" via meta.continuation_policy_id, or
    "native"). trajectory_s=None means commit_s; implementations may reject
    trajectory_s > part_a_horizon_s."""

    def predict_many(
        self,
        state_samples: Sequence[StateSample],
        plans: Sequence[PlanCandidate],
        controller_state: ControllerState,
        *,
        horizon_s: float = DEFAULT_HORIZON_S,
        commit_s: float = DEFAULT_COMMIT_S,
        continuation: str = CONTINUATION_PLANNER,
        trajectory_s: "float | None" = None,
    ) -> "list[Prediction]": ...

    def predict(
        self,
        state_samples: Sequence[StateSample],
        plan: PlanCandidate,
        controller_state: ControllerState,
        **kw: Any,
    ) -> Prediction: ...


class PredictorV2Mixin(PredictorMixin):
    """v2 flavour of the §2 identity mixin (predict == predict_many[0]);
    identical behaviour, typed against the v2 signature."""


__all__ = ["Predictor", "PredictorMixin", "PredictorV2", "PredictorV2Mixin"]
