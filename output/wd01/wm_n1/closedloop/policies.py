"""Closed-loop policy registry (WM-N1-REPL-CL-20261005 Part C).

Every policy maps a decision context (live RSU-observable state + the live
CAV roster) to per-CAV actions for the next decision period. Registered
names: ``pl01_a11`` (legacy), ``fleet_wm`` (legacy), ``pl01_wms`` (primary),
``fleet_wms`` (secondary), ``always_native``,
``always_pi``, ``random``. Another module can add a policy with
:func:`register_policy` (the walls variant registers ``pl01_a11_walls``) and
the runner imports it with ``--policy-module``.

A policy returns :class:`DecisionOut`: ``assignments`` {cav: action} over the
live CAVs (actions in the dispatcher vocabulary native / damping /
pi-external-u), an optional ``walls`` payload (refused by the runner unless
the policy also implements ``apply(session, plan_factory, ...)``), and a
JSON-able ``record`` logged to decisions.jsonl.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

FLEET_CANDIDATES = ("native", "damping", "pi-external-u")
POLICIES: dict = {}


def register_policy(name: str):
    def deco(factory: Callable):
        if name in POLICIES:
            raise ValueError(f"policy {name!r} already registered")
        POLICIES[name] = factory
        return factory
    return deco


def make_policy(name: str, **kwargs):
    """Instantiate a policy; kwargs it does not accept are dropped (the
    runner passes the same predictor/config options to every policy)."""
    import inspect

    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; registered: {sorted(POLICIES)}")
    factory = POLICIES[name]
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        params = {}
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return factory(**kwargs)


@dataclass
class DecisionContext:
    world: str
    seed: int
    t0_s: float
    now_s: float
    decision_index: int
    state: Any                     # closedloop.live_state.LiveState
    live_cavs: tuple               # tracked CAV ids present in the state
    prev_assignments: dict         # {cav: action} currently executed
    model: Any = None              # lazily shared A11 bundle (ModelHandle)


@dataclass
class DecisionOut:
    assignments: dict
    record: dict = field(default_factory=dict)
    walls: tuple = ()


class ModelHandle:
    """Loads the A11 bundle once per process (lazy)."""

    def __init__(self, loader: Optional[Callable] = None):
        self._loader = loader
        self._ck = None

    @property
    def checkpoint(self):
        if self._ck is None:
            from .a11_predictor import load_a11_bundle

            self._ck = (self._loader or load_a11_bundle)()
        return self._ck

    @property
    def bundle(self):
        return self.checkpoint.bundle


def _sample(ctx: DecisionContext):
    from .a11_predictor import LiveSample

    sample = LiveSample(
        hybrid=ctx.state,
        meta={"sample_id": f"{ctx.world}@{ctx.now_s:g}", "time_s": float(ctx.now_s),
              "seed": int(ctx.seed),
              # ideal RSU sensing (reconstructor direction): every tracked
              # vehicle of the physics row is observed
              "merge_observed": 1.0, "coverage": 1.0},
    )
    # PL-04e2 decision-unit ranking input (read only when the planner's
    # max_decision_units is set): metres upstream of the merge start
    # (featurizer s_down is merge-relative, upstream negative); CAVs at/after
    # the merge start or on junction lanes get no entry (held).
    st = ctx.state
    dist = {}
    for cav, slot in st.cav_slots.items():
        s = float(st.veh_kin[slot, 0])
        on_lane = float(st.veh_kin[slot, 1:4].sum()) > 0
        if on_lane and s <= 0.0:
            dist[cav] = -s
    sample.frames_distance_to_merge_m = dist
    return sample


def _fleet(ctx, action):
    return {c: action for c in ctx.live_cavs}


@register_policy("always_native")
class AlwaysNative:
    name = "always_native"
    uses_model = False

    def decide(self, ctx: DecisionContext) -> DecisionOut:
        return DecisionOut(_fleet(ctx, "native"), {"choice": "native"})


@register_policy("always_pi")
class AlwaysPi:
    name = "always_pi"
    uses_model = False

    def decide(self, ctx: DecisionContext) -> DecisionOut:
        return DecisionOut(_fleet(ctx, "pi-external-u"), {"choice": "pi-external-u"})


def decision_seed(world: str, decision_index: int, salt: str = "random") -> int:
    digest = hashlib.sha256(f"{salt}|{world}|{int(decision_index)}".encode()).hexdigest()
    return int(digest[:16], 16)


@register_policy("random")
class RandomFleet:
    """Seeded (world, decision) uniform fleet choice among FLEET_CANDIDATES."""

    name = "random"
    uses_model = False

    def decide(self, ctx: DecisionContext) -> DecisionOut:
        seed = decision_seed(ctx.world, ctx.decision_index)
        choice = FLEET_CANDIDATES[int(np.random.default_rng(seed).integers(len(FLEET_CANDIDATES)))]
        return DecisionOut(_fleet(ctx, choice), {"choice": choice, "rng_seed": seed})


@register_policy("fleet_wm")
class FleetWM:
    """One fleet-wide action: argmin A11 mean dV vs native at tau = 300 s."""

    name = "fleet_wm"
    uses_model = True

    def __init__(self, model: Optional[ModelHandle] = None, predictor_kwargs=None):
        self.model = model
        self.predictor_kwargs = dict(predictor_kwargs or {})
        self._pred = None

    def predictor(self, ctx):
        if self._pred is None:
            from .a11_predictor import A11Predictor

            handle = self.model or ctx.model
            self._pred = A11Predictor(handle.bundle, **self.predictor_kwargs)
        return self._pred

    def decide(self, ctx: DecisionContext) -> DecisionOut:
        from traffic_shepherd.planner.interface import PlanCandidate

        if not ctx.live_cavs:
            return DecisionOut({}, {"choice": "native", "detail": "no live CAVs"})
        pred = self.predictor(ctx)
        plans = [PlanCandidate({c: a for c in ctx.live_cavs}) for a in FLEET_CANDIDATES]
        t = time.perf_counter()
        outs = pred.predict_many([_sample(ctx)], plans)
        dv = {a: float(o.meta["dv_mean_tau"]) for a, o in zip(FLEET_CANDIDATES, outs)}
        q90 = {a: float(o.meta["dv_q90_tau"]) for a, o in zip(FLEET_CANDIDATES, outs)}
        # ties -> candidate order (native first)
        choice = min(FLEET_CANDIDATES, key=lambda a: (dv[a], FLEET_CANDIDATES.index(a)))
        risk_choice = min(FLEET_CANDIDATES, key=lambda a: (q90[a], FLEET_CANDIDATES.index(a)))
        return DecisionOut(_fleet(ctx, choice), {
            "choice": choice, "dv_mean_tau300": dv, "dv_q90_tau300": q90,
            "risk_q90_choice_recorded": risk_choice,
            "head_native_dv_mean": float(outs[0].meta["head_native_dv_mean"]),
            "predict_s": round(time.perf_counter() - t, 3),
            "predictor_calls": list(pred.calls[-1:]),
            "safety": {a: {"min_ttc_s": float(o.min_ttc_s[0, 0]),
                           "collision": bool(o.collision[0, 0]),
                           "hard_brake_s": float(o.hard_brake_s[0, 0])}
                       for a, o in zip(FLEET_CANDIDATES, outs)},
        })


class _WMSMixin:
    """Shared predictor construction for the WM-S policy pair."""

    def predictor(self, ctx):
        if self._pred is None:
            from .wm_s_predictor import WMSPredictor

            handle = self.model or ctx.model
            if handle is None and self.predictor_kwargs.get("predict_states") is None:
                raise RuntimeError(
                    "WM-S policy requires a model handle or predict_states plug-in"
                )
            bundle = None if handle is None else handle.bundle
            checkpoint = None if handle is None else getattr(handle, "checkpoint", None)
            predictor_kwargs = dict(self.predictor_kwargs)
            predictor_kwargs.setdefault(
                "model_sha256", None if checkpoint is None else getattr(checkpoint, "sha256", None)
            )
            self._pred = WMSPredictor(
                bundle,
                **predictor_kwargs,
            )
        return self._pred


@register_policy("fleet_wms")
class FleetWMS(_WMSMixin):
    """One fleet-wide action selected from paired state-rollout delay."""

    name = "fleet_wms"
    uses_model = True

    def __init__(self, model: Optional[ModelHandle] = None, predictor_kwargs=None,
                 predictor=None):
        self.model = model
        self.predictor_kwargs = dict(predictor_kwargs or {})
        self._pred = predictor

    def decide(self, ctx: DecisionContext) -> DecisionOut:
        from traffic_shepherd.planner.interface import PlanCandidate

        if not ctx.live_cavs:
            return DecisionOut({}, {"choice": "native", "detail": "no live CAVs"})
        pred = self.predictor(ctx)
        plans = [PlanCandidate({c: a for c in ctx.live_cavs}) for a in FLEET_CANDIDATES]
        t = time.perf_counter()
        outs = pred.predict_many([_sample(ctx)], plans)
        dv = {a: float(o.delay_vs_native_s[0, 0])
              for a, o in zip(FLEET_CANDIDATES, outs)}
        choice = min(FLEET_CANDIDATES, key=lambda a: (dv[a], FLEET_CANDIDATES.index(a)))
        return DecisionOut(_fleet(ctx, choice), {
            "choice": choice, "delay_vs_native_s": dv,
            "predict_s": round(time.perf_counter() - t, 3),
            "predictor_calls": list(pred.calls[-1:]),
            "safety": {a: {"min_ttc_s": float(o.min_ttc_s[0, 0]),
                           "collision": bool(o.collision[0, 0]),
                           "hard_brake_s": float(o.hard_brake_s[0, 0])}
                       for a, o in zip(FLEET_CANDIDATES, outs)},
        })


@register_policy("pl01_a11")
class PL01A11:
    """PL01 planner decision loop (traffic_shepherd.planner.loop.Planner,
    steps 0-5) with the A11 predictor; walls OFF (max_walls = 0)."""

    name = "pl01_a11"
    uses_model = True
    #: registered policy config: walls excluded; v1.1 predictions are allowed
    #: in the controlled-experiment mode; everything else = planner defaults.
    config_overrides = {"max_walls": 0, "mode": "controlled_experiment"}

    def __init__(self, model: Optional[ModelHandle] = None, predictor_kwargs=None,
                 config_overrides=None, log_candidates: bool = True):
        self.model = model
        self.predictor_kwargs = dict(predictor_kwargs or {})
        if config_overrides is not None:
            self.config_overrides = dict(config_overrides)
        self.log_candidates = bool(log_candidates)
        self._planner = None
        self._pred = None

    def planner(self, ctx):
        if self._planner is None:
            from traffic_shepherd.planner.config import PlannerConfig
            from traffic_shepherd.planner.loop import Planner

            from .a11_predictor import A11Predictor

            handle = self.model or ctx.model
            self._pred = A11Predictor(handle.bundle, **self.predictor_kwargs)
            self._planner = Planner(self._pred, PlannerConfig(**self.config_overrides),
                                    log_candidates=self.log_candidates)
        return self._planner

    @staticmethod
    def controller_state(ctx):
        return {c: {"action": ctx.prev_assignments.get(c, "native"), "wall_id": None,
                    "wall_phase": None, "time_in_phase_s": 0.0, "wall_speed_mps": None,
                    "locked": False} for c in ctx.live_cavs}

    def decide(self, ctx: DecisionContext) -> DecisionOut:
        from traffic_shepherd.planner.interface import PlanCandidate, plan_key

        if not ctx.live_cavs:
            return DecisionOut({}, {"detail": "no live CAVs; nothing to decide"})
        planner = self.planner(ctx)
        # roster sync: the planner's current plan = what is executing now on
        # the CAVs tracked at this decision (entrants run native)
        planner._current_plan = PlanCandidate(
            {c: ctx.prev_assignments.get(c, "native") for c in ctx.live_cavs})
        n_calls = len(self._pred.calls)
        t = time.perf_counter()
        decision = planner.decide([_sample(ctx)], float(ctx.now_s),
                                  controller_state=self.controller_state(ctx))
        wall_s = time.perf_counter() - t
        if decision.plan.walls:
            raise RuntimeError("pl01_a11 emitted a wall plan with walls disabled")
        assignments = {c: decision.plan.per_cav.get(c, "native") for c in ctx.live_cavs}
        chosen = decision.chosen_prediction
        calls = self._pred.calls[n_calls:]
        record = {
            "groups": [list(g) for g in decision.groups],
            "abstain": bool(decision.abstain),
            "switched": bool(decision.switched),
            "non_native": {c: a for c, a in assignments.items() if a != "native"},
            "reasons": [r.__dict__ for r in decision.reasons],
            "chosen_dv_mean_tau300": (None if chosen is None
                                      else float(chosen.meta.get("dv_mean_tau", np.nan))),
            "chosen_dv_q90_tau300": (None if chosen is None
                                     else float(chosen.meta.get("dv_q90_tau", np.nan))),
            "head_native_dv_mean": (None if chosen is None
                                    else chosen.meta.get("head_native_dv_mean")),
            "ttc_vetoes": list(decision.ttc_vetoes),
            "decide_s": round(wall_s, 3),
            "predictor_calls": calls,
            "n_predicted_plans": int(sum(c["n_plans"] for c in calls)),
            "n_rollouts": int(sum(c["n_rollouts"] for c in calls)),
            "rollout_s": round(sum(c["rollout_s"] for c in calls), 3),
            "head_s": round(sum(c["head_s"] for c in calls), 3),
            "plan_key": repr(plan_key(decision.plan))[:200],
        }
        if self.log_candidates:
            record["candidates"] = [dict(r) if isinstance(r, dict) else r
                                    for r in planner.last_candidates]
        return DecisionOut(assignments, record)


@register_policy("pl01_wms")
class PL01WMS(_WMSMixin):
    """PL01 planner using WM-S paired next-state rollouts (walls off)."""

    name = "pl01_wms"
    uses_model = True
    config_overrides = {"max_walls": 0, "mode": "controlled_experiment"}

    def __init__(self, model: Optional[ModelHandle] = None, predictor_kwargs=None,
                 config_overrides=None, log_candidates: bool = True, predictor=None):
        self.model = model
        self.predictor_kwargs = dict(predictor_kwargs or {})
        if config_overrides is not None:
            self.config_overrides = dict(config_overrides)
        self.log_candidates = bool(log_candidates)
        self._planner = None
        self._pred = predictor

    def planner(self, ctx):
        if self._planner is None:
            from traffic_shepherd.planner.config import PlannerConfig
            from traffic_shepherd.planner.loop import Planner

            self._pred = self.predictor(ctx)
            self._planner = Planner(
                self._pred, PlannerConfig(**self.config_overrides),
                log_candidates=self.log_candidates)
        return self._planner

    @staticmethod
    def controller_state(ctx):
        return {c: {"action": ctx.prev_assignments.get(c, "native"), "wall_id": None,
                    "wall_phase": None, "time_in_phase_s": 0.0, "wall_speed_mps": None,
                    "locked": False} for c in ctx.live_cavs}

    def decide(self, ctx: DecisionContext) -> DecisionOut:
        from traffic_shepherd.planner.interface import PlanCandidate, plan_key

        if not ctx.live_cavs:
            return DecisionOut({}, {"detail": "no live CAVs; nothing to decide"})
        planner = self.planner(ctx)
        planner._current_plan = PlanCandidate(
            {c: ctx.prev_assignments.get(c, "native") for c in ctx.live_cavs})
        n_calls = len(self._pred.calls)
        t = time.perf_counter()
        decision = planner.decide(
            [_sample(ctx)], float(ctx.now_s),
            controller_state=self.controller_state(ctx))
        wall_s = time.perf_counter() - t
        if decision.plan.walls:
            raise RuntimeError("pl01_wms emitted a wall plan with walls disabled")
        assignments = {
            c: decision.plan.per_cav.get(c, "native") for c in ctx.live_cavs
        }
        chosen = decision.chosen_prediction
        calls = self._pred.calls[n_calls:]
        record = {
            "groups": [list(g) for g in decision.groups],
            "abstain": bool(decision.abstain),
            "switched": bool(decision.switched),
            "non_native": {c: a for c, a in assignments.items() if a != "native"},
            "reasons": [r.__dict__ for r in decision.reasons],
            "chosen_delay_vs_native_s": (None if chosen is None else
                                          float(chosen.delay_vs_native_s[0, 0])),
            "ttc_vetoes": list(decision.ttc_vetoes),
            "decide_s": round(wall_s, 3),
            "predictor_calls": calls,
            "n_predicted_plans": int(sum(c["n_plans"] for c in calls)),
            "n_rollouts": int(sum(c["n_rollouts"] for c in calls)),
            "rollout_s": None,
            "plan_key": repr(plan_key(decision.plan))[:200],
        }
        if self.log_candidates:
            record["candidates"] = [dict(r) if isinstance(r, dict) else r
                                     for r in planner.last_candidates]
        return DecisionOut(assignments, record)
