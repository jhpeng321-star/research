"""Evaluation data sources.

A source hands the harness the realised rows for a split.  In production the
rows come from the featurised memmap shards (``INDEX.json`` pairs) joined to
the frozen v0.4a outcome labels (``features/FORMAT.md`` section 7); the CLI
accepts any ``module:attr`` source.  This module ships the protocol and a
deterministic synthetic source used by the CPU tests.

No real data is read here.  HOLDOUT, CAL and chunk-5 are never opened by the
synthetic source; the seals in :mod:`evalh.seals` gate the real splits.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .protocol import HORIZONS_S, OutcomeRow, RolloutRow

# Policy arms (chunk-5 campaign: REGISTRATION_CHUNK5_TEST.md:43-51).  The
# chunk's ``__base`` (damping fleet) arm is the "damping" policy.
DAMPING = "damping"
NATIVE = "native"
PI = "pi"
WALL = "wall"
DEFAULT_POLICIES = (DAMPING, NATIVE, PI, WALL)
DEFAULT_TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)


class EvalSource(Protocol):
    """Minimal data interface consumed by :mod:`evalh.run_eval`."""

    sign_threshold: float

    def outcome_rows(self, split: str) -> list[OutcomeRow]:
        ...

    def rollout_rows(self, split: str) -> list[RolloutRow]:
        ...


class SyntheticSource:
    """Deterministic synthetic worlds for the CPU tests (zero real data).

    The realised delay of a row is ``base(world) + effect(policy) + tau`` plus
    a seeded noise; ``row.features['truth_center']`` carries the noiseless
    centring value so a synthetic predictor can be an oracle-plus-noise.
    """

    def __init__(self, n_worlds: int = 8, n_decision: int = 2,
                 policies: Sequence[str] = DEFAULT_POLICIES,
                 taus: Sequence[float] = DEFAULT_TAUS, seed: int = 0,
                 horizons_s: tuple = HORIZONS_S):
        self.n_worlds = int(n_worlds)
        self.n_decision = int(n_decision)
        self.policies = tuple(policies)
        self.taus = tuple(float(t) for t in taus)
        self.seed = int(seed)
        self.horizons_s = tuple(float(h) for h in horizons_s)
        # Registered-gate effects: damping and PI cut delay, walls cut more.
        self.effect = {DAMPING: 12.0, NATIVE: 0.0, PI: -6.0, WALL: -9.0}
        self.sign_threshold = 5.0

    # ------------------------------------------------------------- helpers --
    def _world_base(self, w: int) -> float:
        return 20.0 + 3.0 * w

    def _center(self, w: int, policy: str, tau: float) -> float:
        return (self._world_base(w) + self.effect.get(policy, 0.0)
                + 0.02 * tau)

    def _rng(self, split: str, salt: int) -> np.random.Generator:
        off = {"train_oof": 11, "cal": 23, "final_test": 37}.get(split, 101)
        return np.random.default_rng(self.seed + 1000 * off + salt)

    # ------------------------------------------------------------- outcome --
    def outcome_rows(self, split: str) -> list[OutcomeRow]:
        rng = self._rng(split, 1)
        rows = []
        for w in range(self.n_worlds):
            world = f"w{w:03d}"
            base = self._world_base(w)
            for d in range(self.n_decision):
                dec = float(120 + 60 * d)
                for tau in self.taus:
                    for pol in self.policies:
                        center = self._center(w, pol, tau)
                        y = float(center + rng.normal(0.0, 1.5))
                        veh_steps = float(200.0 + 10.0 * rng.random())
                        rows.append(OutcomeRow(
                            world=world, t0_s=300.0, decision_t_s=dec,
                            tau_s=tau, policy=pol, y_delay_s=y,
                            collision=float(rng.random() < 0.01),
                            veh_steps=veh_steps,
                            ttc_lt3_s=float(max(0.0, 1.0 + rng.normal(0, 0.3))),
                            hard_brake_s=float(max(0.0, 2.0 + rng.normal(0, 0.4))),
                            n_vehicles=veh_steps / 40.0,
                            n_arrivals=float(max(0.0, rng.poisson(3))),
                            intervention=float(max(0.0, rng.normal(0.5, 0.2))),
                            features=dict(base=base, truth_center=center)))
        # world-bootstrap TRAIN threshold: median |contrast| on train worlds.
        self.sign_threshold = max(1.0, self._median_abs_contrast(rows))
        return rows

    def _median_abs_contrast(self, rows) -> float:
        by = {}
        for r in rows:
            key = (r.world, r.decision_t_s, r.tau_s)
            by.setdefault(key, {})[r.policy] = r.y_delay_s
        d = [abs(v[DAMPING] - v[NATIVE]) for v in by.values()
             if DAMPING in v and NATIVE in v]
        return float(np.median(d)) if d else 0.0

    # ------------------------------------------------------------- rollout --
    def rollout_rows(self, split: str) -> list[RolloutRow]:
        rng = self._rng(split, 2)
        h = len(self.horizons_s)
        v = 6
        rows = []
        for w in range(self.n_worlds):
            world = f"w{w:03d}"
            for pol in self.policies:
                pos = np.empty((h, v), float)
                spd = np.empty((h, v), float)
                dt = np.asarray(self.horizons_s, float)[:, None]
                v0 = 15.0 + 2.0 * rng.random(v)
                pos = 40.0 + v0[None, :] * dt
                spd = np.repeat(v0[None, :], h, axis=0)
                pos = pos + rng.normal(0, 0.2, pos.shape)
                rows.append(RolloutRow(
                    world=world, t0_s=300.0, policy=pol,
                    true_pos=pos, true_speed=spd, horizons_s=self.horizons_s,
                    true_dist=float(np.mean(np.abs(pos[-1] - pos[0]))),
                    true_speed_scale=float(np.mean(v0)),
                    veh_steps=float(200.0), collision=0.0,
                    ttc_lt3_s=float(max(0.0, 1.0 + rng.normal(0, 0.3))),
                    hard_brake_s=float(max(0.0, 2.0 + rng.normal(0, 0.4))),
                    features=dict(policy=pol)))
        return rows
