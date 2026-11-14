"""The Part A v2g rollout baseline for INC1 (with obstacle support).

Given the frozen ``params_nominal.json`` geometry, the Part A v2g baseline is
exactly ``rollout2`` with the ballistic (``v2g``) integrator, no learned HDV
residual and no arrival head -- the like-for-like comparison mode already
pinned by ``rollout2.tests.test_engine_parity``.  This module is a thin,
INC1-owned wrapper that fixes those switches and forwards the fixed-obstacle
world unchanged, so the obstacle slot is pinned by the same step/validity code
in both the Idea A and the v2g engine.
"""
from __future__ import annotations

from typing import Optional

from rollout2.engine import IdeaABundle, Rollout2Engine
from rollout2.state import Rollout2Config, StrategyActions


def rollout_v2g_baseline(bundle: IdeaABundle, world, *, n_steps: int,
                         actions: Optional[StrategyActions] = None,
                         n_samples: int = 1, seed: Optional[int] = None,
                         params: Optional[dict] = None, t0_s: float = 0.0,
                         lane_changes_enabled: bool = True):
    """Run the Part A v2g rollout baseline from ``world``.

    ``bundle`` supplies the arm controller (``ctrl``); ``hdv``/``arrival`` are
    ignored because the baseline forces zero residual and no arrivals.  The
    world may carry a fixed obstacle (``WorldState.obstacle``); it is held by
    the same step/validity layer as in the Idea A rollout.
    """
    cfg = Rollout2Config(
        params=params, integrator="v2g", residual_enabled=False,
        arrivals_enabled=False, lane_changes_enabled=bool(lane_changes_enabled),
        enforce_lane_legality=False, t0_s=float(t0_s))
    engine = Rollout2Engine(bundle, cfg, seed=seed)
    return engine.run(world, actions, n_steps=int(n_steps),
                      n_samples=int(n_samples))


def idea_a_rollout(bundle: IdeaABundle, world, *, n_steps: int,
                   actions: Optional[StrategyActions] = None,
                   n_samples: int = 1, seed: Optional[int] = None,
                   params: Optional[dict] = None, t0_s: float = 0.0,
                   event_tokens: bool = False):
    """Run the Idea A (D1) rollout with the INC1 event-token default OFF."""
    cfg = Rollout2Config(
        params=params, integrator="d1",
        residual_enabled=bundle.hdv is not None, arrivals_enabled=False,
        t0_s=float(t0_s), event_tokens=bool(event_tokens))
    engine = Rollout2Engine(bundle, cfg, seed=seed)
    return engine.run(world, actions, n_steps=int(n_steps),
                      n_samples=int(n_samples))
