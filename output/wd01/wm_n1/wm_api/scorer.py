"""``WmScorer`` — the stable scoring interface of wm_api (version 1.0).

    scorer = WmScorer("ckpt/")                 # sha-checked member dir
    result = scorer.score(state, candidate_plans, horizon_s=300,
                          *, members=None, n_samples=None, seed=None)

Interface contract (``WmScorer.score``)::

    score(state, candidate_plans, horizon_s=300, *, members=None,
          n_samples=None, seed=None, allow_other_horizon=False)
          -> ScoreResult

* ``state`` — :class:`wm_api.state.StateSnapshot` or its JSON mapping;
* ``candidate_plans`` — non-empty list of per-CAV plan mappings (or a bare
  action string per plan); plan 0 is the paired reference (native by
  convention);
* ``horizon_s`` — the scoring window; 300 s by registration.  Any other
  value is refused unless ``allow_other_horizon=True`` (the outcome head
  then reports its NEAREST registered tau, recorded in ``result.tau_s``);
* ``members`` — optional 0-based subset into the checkpoint's sorted
  member list (default: all);
* ``n_samples`` — rollout repetitions per member (default 1).  Only the
  seeded arrival sampler varies across samples;
* ``seed`` — default 0.  Deterministic CPU mode: the (member, sample)
  rollout rng is ``numpy.random.default_rng(seed + 1009*member0 +
  7919*sample)``, so a fixed seed reproduces the response exactly.

Refusals: unknown action, missing/corrupt checkpoint, sha mismatch, K-slot
overflow (reported, never silently dropping CAVs) and ``horizon_s != 300``
without ``allow_other_horizon``.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np
import torch

from . import engine as _engine
from .checkpoint import load_checkpoint
from .errors import (HorizonError, PlanSchemaError)
from .plans import resolve_plan
from .result import METRIC_NAMES, MetricSummary, PlanScore, ScoreResult, \
    quantile_dict
from .state import StateSnapshot

DEFAULT_HORIZON_S = 300.0
#: registered outcome-head taus (model.constants.TAUS; frozen).
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
#: documented (member, sample) rng derivation constants.
_SEED_MEMBER, _SEED_SAMPLE = 1009, 7919


def _coerce_state(state) -> StateSnapshot:
    if isinstance(state, StateSnapshot):
        return state
    return StateSnapshot.from_json(state)


def _tau_for(horizon_s: float) -> float:
    taus = np.asarray(TAUS, dtype=float)
    return float(taus[int(np.argmin(np.abs(taus - float(horizon_s))))])


class WmScorer:
    """Sha-checked Idea A checkpoint + the ``score`` interface."""

    API_VERSION = "1.0"
    DEFAULT_HORIZON_S = DEFAULT_HORIZON_S

    def __init__(self, checkpoint, *, threads: int = 2,
                 verify_sha: bool = True) -> None:
        self.threads = int(min(2, max(1, int(threads))))
        torch.set_num_threads(self.threads)
        self.ckpt = load_checkpoint(checkpoint, verify_sha=verify_sha)
        self.checkpoint = dict(
            directory=self.ckpt.directory,
            sha256=dict(self.ckpt.shas),
            sha_verified=bool(self.ckpt.sha_verified),
            members=[int(m.member) for m in self.ckpt.members],
            seeds=[int(m.seed) for m in self.ckpt.members],
            ensemble_k=int(self.ckpt.flags.get("ensemble_k", 0)),
            partial=bool(self.ckpt.flags.get("partial", True)))

    # ------------------------------------------------------------------
    def score(self, state, candidate_plans, horizon_s: float =
              DEFAULT_HORIZON_S, *, members: Optional[Sequence[int]] = None,
              n_samples: Optional[int] = None, seed: Optional[int] = None,
              allow_other_horizon: bool = False) -> ScoreResult:
        """Score candidate plans from one t0 snapshot (see module docstring)."""
        t_start = time.perf_counter()
        snap = _coerce_state(state)
        horizon_s = float(horizon_s)
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            raise HorizonError(
                f"horizon_s must be finite and > 0; got {horizon_s!r}")
        if horizon_s != DEFAULT_HORIZON_S and not allow_other_horizon:
            raise HorizonError(
                f"horizon_s = {horizon_s:g} != {DEFAULT_HORIZON_S:g}; pass "
                "allow_other_horizon=True to score a different window (the "
                "outcome head reports its nearest registered tau)")
        tau_s = _tau_for(horizon_s)
        n_steps = max(1, int(round(horizon_s / 0.5)))
        n_segments = max(1, min(19, int(round(horizon_s / 30.0))))
        if not isinstance(candidate_plans, (list, tuple)) or \
                not candidate_plans:
            raise PlanSchemaError(
                "candidate_plans must be a non-empty list of plans; plan 0 "
                "is the paired reference (native by convention)")
        plans = [resolve_plan(p, snap, n_segments, i)
                 for i, p in enumerate(candidate_plans)]
        member_idx = list(range(self.ckpt.n_members)) if members is None \
            else [int(m) for m in members]
        if not member_idx:
            raise PlanSchemaError("members must select at least one member")
        bad = [m for m in member_idx if not 0 <= m < self.ckpt.n_members]
        if bad:
            raise PlanSchemaError(
                f"members index a 0-based subset of "
                f"{self.ckpt.n_members} loaded member(s); offending {bad}")
        n_samples = 1 if n_samples is None else int(n_samples)
        if n_samples < 1:
            raise PlanSchemaError("n_samples must be >= 1")
        seed = 0 if seed is None else int(seed)
        arrivals_enabled = snap.entrance_counters is not None
        k_slots = _engine.k_slots_for(snap, arrivals_enabled=arrivals_enabled)
        scored_members = [self.ckpt.members[m] for m in member_idx]

        per_member = []
        for m_pos, member in enumerate(scored_members):
            member0 = int(member_idx[m_pos])
            samples = []
            for s_i in range(n_samples):
                rng_seed = seed + _SEED_MEMBER * member0 \
                    + _SEED_SAMPLE * s_i
                by_plan = []
                for plan in plans:
                    by_plan.append(_engine.rollout_member(
                        member, snap, plan, n_steps=n_steps, t0_s=snap.t0_s,
                        seed=rng_seed, arrivals_enabled=arrivals_enabled,
                        counters=snap.entrance_counters, k_slots=k_slots))
                samples.append(by_plan)
            per_member.append((member, samples))

        plan_scores = self._assemble(plans, per_member, tau_s)
        timing = round(time.perf_counter() - t_start, 3)
        return ScoreResult(
            api_version=self.API_VERSION, horizon_s=horizon_s, tau_s=tau_s,
            plans=plan_scores,
            members=[int(m.member) for m, _ in per_member],
            seed=seed, n_samples=n_samples,
            arrivals_enabled=arrivals_enabled,
            checkpoint=dict(self.checkpoint),
            meta=dict(
                k_slots=k_slots, n_steps=n_steps, n_segments=n_segments,
                threads=self.threads,
                timing_s=timing,
                excluded=snap.excluded,
                obstacles=len(snap.obstacles),
                arrivals_inserted=(sum(int(r["inserted"])
                                       for _, samples in per_member
                                       for by_plan in samples
                                       for r in by_plan)
                                   if arrivals_enabled else 0),
                ood=("density_model" if self.ckpt.ood_model is not None
                     else "ensemble_disagreement"),
                validity={f"member{member.member}":
                          samples[0][0]["validity"]
                          for member, samples in per_member},
                n_collisions=sum(int(sum(by_plan[0]["collisions"]
                                         .values()))
                                 for _, samples in per_member
                                 for by_plan in samples),
                ))

    # ------------------------------------------------------------------
    def _assemble(self, plans: list, per_member: list, tau_s: float) -> list:
        tau_i = int(np.argmin(np.abs(np.asarray(TAUS) - tau_s)))
        tgt_index = {name: i for i, name in enumerate(METRIC_NAMES)}
        plan_scores = []
        curves_by_plan = []
        for p_i, plan in enumerate(plans):
            curves = {name: [] for name in METRIC_NAMES}
            summaries = []
            draws = {name: [] for name in METRIC_NAMES}
            for member, samples in per_member:
                member_curves = {name: [] for name in METRIC_NAMES}
                member_summaries = []
                for by_plan in samples:
                    run = by_plan[p_i]
                    q = np.asarray(run["outcome_quantiles"], dtype=float)
                    member_summaries.append(
                        np.asarray(run["summary_raw"], dtype=float))
                    for name in METRIC_NAMES:
                        member_curves[name].append(
                            q[tau_i, tgt_index[name], :])
                for name in METRIC_NAMES:
                    curve = np.mean(member_curves[name], axis=0)
                    curves[name].append(curve)
                    draws[name].append(float(np.median(curve)))
                summaries.append(np.mean(member_summaries, axis=0))
            pooled = {name: np.mean(curves[name], axis=0)
                      for name in METRIC_NAMES}
            curves_by_plan.append(pooled)
            ood = self._ood(summaries)
            plan_scores.append(PlanScore(
                plan_index=p_i, descriptor=plan.describe(),
                metrics={name: MetricSummary(
                    quantiles=quantile_dict(pooled[name]),
                    mean=float(np.mean(draws[name])),
                    member_draws=[float(x) for x in draws[name]])
                    for name in METRIC_NAMES},
                ood_score=float(ood),
                paired_diff_vs_plan0=None,
                is_reference=(p_i == 0),
                wall_gates=dict(plan.gates)))
        ref = curves_by_plan[0]
        ref_means = {p.plan_index: {n: p.metrics[n].mean
                                    for n in METRIC_NAMES}
                     for p in plan_scores}
        for p_i, ps in enumerate(plan_scores):
            if p_i == 0:
                ps.paired_diff_vs_plan0 = {
                    n: dict(quantiles=quantile_dict(np.zeros(19), monotone=False),
                        mean=0.0)
                    for n in METRIC_NAMES}
                continue
            ps.paired_diff_vs_plan0 = {
                n: dict(quantiles=quantile_dict(curves_by_plan[p_i][n] - ref[n],
                                                monotone=False),
                        mean=float(ref_means[p_i][n] - ref_means[0][n]))
                for n in METRIC_NAMES}
        return plan_scores

    def _ood(self, summaries: list) -> float:
        """Per-plan OOD score (larger = less familiar).

        Default: ``corr.ood.ensemble_disagreement`` over the per-member CAV
        motion summaries.  When the checkpoint ships ``ood_model.json``
        (a diagonal-Gaussian density fitted offline on TRAIN summaries),
        the mean member summary is scored with
        ``corr.ood.motion_density_score`` instead.
        """
        from corr.ood import ensemble_disagreement, motion_density_score

        if self.ckpt.ood_model is not None:
            mean_summary = np.mean(np.asarray(summaries, dtype=float),
                                   axis=0, keepdims=True)
            return float(motion_density_score(
                mean_summary, self.ckpt.ood_model)[0])
        per_member = [np.asarray(s, dtype=float).reshape(1, -1)
                      for s in summaries]                   # [M, B=1, F]
        return float(ensemble_disagreement(per_member)[0])


def load_scorer(checkpoint, **kwargs) -> WmScorer:
    """Load once, score many (the repeated-use entry point)."""
    return WmScorer(checkpoint, **kwargs)
