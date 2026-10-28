"""S-COOPW (2026-10-25): switches around the counterfactual cooperation credit (COOPW.md).

Everything here is OFF unless a config turns it on; with the defaults the trainer is bit-identical
to ``c4/loop14c``.

* :class:`UncertaintyConfig` (``[uncertainty]``): sampled futures + a curriculum of the pessimism
  weight, applied to TRAINING imagination only (validation / the independent rescore keep the
  single deterministic rollout).  It is a run-config section (like ``[counterfactual]``) because
  ``DreamerTrainer.load_checkpoint`` restores ``DreamerConfig`` from the checkpoint, so trainer
  fields could not reach a resumed loop policy.
* :func:`multi_cav_flags` / :func:`multi_cav_start` - observable "multi-CAV situation" flags.
* :func:`start_features` / :class:`TiltSampler` - training-start tilt toward cooperation and
  headroom situations (observable features only), with unbiased running target estimates and the
  importance weight of every drawn start (logged; validation restores the target distribution).
* :class:`ExplorationConfig` (``[exploration]``) - group-correlated, persistent exploration noise
  for TRAINING imagination only, exact common-random-number pairing kept.
* :class:`HorizonConfig` (``[horizon]``) - gated imagination-horizon curriculum (training only).
* :func:`term_objectives` - every counterfactual objective term at unit weight (analytic + lane
  REINFORCE path, exactly the decomposition the weights multiply), for the gradient audit.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

import torch
from torch import Tensor

from .contract import MERGE_S0_M, MERGE_S1_M
from .counterfactual import (CounterfactualConfig, Variants, acting_cavs, baseline_index,
                             close_groups, row_coefficients)

RAMP_LANE = 2
TERM_NAMES = ("all", "individual", "group", "coop", "team")


def _interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


# ------------------------------------------------------------------ uncertainty pessimism
@dataclass
class UncertaintyConfig:
    """Pessimism over sampled futures (FUTURES_NOTES.md) for the training imagination.

    ``future_samples`` driver-parameter futures per start (encoder uncertainty; the uncalibrated
    acceleration noise stays off) make ``ensemble_penalty`` = the per-control-step std of the team
    reward over futures (one WM member gives zero otherwise, so the trainer's
    ``uncertainty_weight`` alone is inert).  The rewards become ``r - w(u) * std`` with ``w`` a
    piecewise-linear curriculum over the counterfactual curriculum position ``u``."""
    enabled: bool = False
    future_samples: int = 2
    driver_noise_scale: float = 1.0
    curriculum_updates: list = field(default_factory=lambda: [0])
    weights: list = field(default_factory=lambda: [0.1])

    def __post_init__(self):
        if (isinstance(self.future_samples, bool) or not isinstance(self.future_samples, int)
                or self.future_samples < 2):
            raise ValueError("uncertainty.future_samples must be an integer >= 2")
        if not math.isfinite(self.driver_noise_scale) or self.driver_noise_scale <= 0:
            raise ValueError("uncertainty.driver_noise_scale must be positive")
        xs = [float(x) for x in self.curriculum_updates]
        ys = [float(y) for y in self.weights]
        if not xs or len(xs) != len(ys):
            raise ValueError("uncertainty curriculum lists must be nonempty and of equal length")
        if any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("uncertainty.curriculum_updates must be strictly increasing")
        if any(not math.isfinite(y) or y < 0 for y in ys):
            raise ValueError("uncertainty weights must be finite and nonnegative")
        self.curriculum_updates, self.weights = xs, ys

    def weight(self, position: int) -> float:
        return _interp(self.curriculum_updates, self.weights, position)

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ correlated exploration
@dataclass
class ExplorationConfig:
    """Correlated exploration noise for TRAINING imagination (never an actor input; the
    deterministic / deployed policy is unchanged).  CAVs of one counterfactual group (connected
    component of policy CAVs chained within ``radius_m`` at t0; default the counterfactual
    ``group_radius_m``) share a persistent standard-normal component:

        eps_i(t) = sqrt(rho) * g_G(i)(t) + sqrt(1 - rho) * eps_own_i(t),
        g(t) = a g(t-1) + sqrt(1 - a^2) z(t),  a = exp(-1 / persistence_steps),

    so every eps_i stays N(0, 1) (the policy's marginal noise, entropy and log-density per CAV are
    unchanged) and corr(eps_i, eps_j) = rho inside a group.  The transform is applied per
    (start, decision step, slot) BEFORE the rows of a start are expanded, so the counterfactual
    common-random-number pairing stays exact.  ``rho`` follows a piecewise-linear curriculum on
    the counterfactual curriculum position."""
    enabled: bool = False
    curriculum_updates: list = field(default_factory=lambda: [0])
    rho: list = field(default_factory=lambda: [0.0])
    persistence_steps: float = 4.0
    radius_m: Optional[float] = None
    long_persistence: bool = False
    persistence_curriculum_updates: list = field(default_factory=lambda: [0, 150])
    persistence_seconds: list = field(default_factory=lambda: [30.0, 60.0])

    def __post_init__(self):
        xs = [float(x) for x in self.curriculum_updates]
        ys = [float(y) for y in self.rho]
        if not xs or len(xs) != len(ys) or any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("exploration curriculum must be nonempty, equal length, increasing")
        if any(not 0.0 <= y <= 1.0 for y in ys):
            raise ValueError("exploration rho must lie in [0, 1]")
        if not math.isfinite(self.persistence_steps) or self.persistence_steps <= 0:
            raise ValueError("exploration persistence_steps must be positive")
        if self.radius_m is not None and not (math.isfinite(self.radius_m) and self.radius_m > 0):
            raise ValueError("exploration radius_m must be positive")
        self.curriculum_updates, self.rho = xs, ys
        px, py = self.persistence_curriculum_updates, self.persistence_seconds
        if (not isinstance(self.long_persistence, bool) or not px or len(px) != len(py)
                or any(not math.isfinite(x) for x in px)
                or any(b <= a for a, b in zip(px, px[1:]))
                or any(not math.isfinite(y) or not 30 <= y <= 60 for y in py)):
            raise ValueError("long exploration persistence requires increasing knots and 30..60 seconds")

    def rho_at(self, position: int) -> float:
        return _interp(self.curriculum_updates, self.rho, position)

    def persistence_at(self, position: int) -> float:
        if not self.long_persistence:
            return self.persistence_steps
        from .contract import CONTROL_DT_S
        return _interp(self.persistence_curriculum_updates, self.persistence_seconds, position) / CONTROL_DT_S

    def as_dict(self) -> dict:
        return asdict(self)


def exploration_plan(state: Mapping[str, Tensor], radius_m: float, n_decisions: int,
                     persistence_steps: float, generator) -> tuple:
    """``(group_id [B,K] long (-1 = no group), shared [D,B,G] standard-normal AR(1))``."""
    s = state["s_m"]
    cav = acting_cavs(state)
    b, k = s.shape
    gid = torch.full((b, k), -1, dtype=torch.long)
    n_groups = 0
    for row in range(b):
        for index, members in enumerate(close_groups(s[row], cav[row], radius_m, 2)):
            gid[row, members] = index
            n_groups = max(n_groups, index + 1)
    a = math.exp(-1.0 / float(persistence_steps))
    shared = torch.empty(max(1, n_decisions), b, max(1, n_groups))
    shared[0] = torch.randn(b, max(1, n_groups), generator=generator)
    for t in range(1, shared.shape[0]):
        shared[t] = a * shared[t - 1] + math.sqrt(1.0 - a * a) * torch.randn(
            b, max(1, n_groups), generator=generator)
    return gid.to(s.device), shared.to(s.device)


def correlate(eps: Tensor, gid: Tensor, shared_t: Tensor, rho: float) -> Tensor:
    """Group-correlated version of the [B,K] standard normals ``eps`` (unchanged outside groups)."""
    if rho <= 0.0 or not bool((gid >= 0).any()):
        return eps
    common = shared_t.to(eps).gather(1, gid.clamp_min(0))
    mixed = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * eps
    return torch.where(gid >= 0, mixed, eps)


# ------------------------------------------------------------------ horizon curriculum
GATE_SCHEMA = "coopw_horizon_gate/1"


@dataclass
class HorizonConfig:
    """Imagination-horizon curriculum for TRAINING imagination (``horizons_s`` per stage, stage k
    reachable from curriculum position ``curriculum_updates[k]``), for starts whose observed CAV
    share is above the existing low-share cap (``cav * 10 <= present`` keeps the base horizon).

    Every step-up is GATED (fail closed): stage k > 0 is allowed only when ``gate_file`` (written
    by ``tools/coopw/horizon_gate.py``) is for one of the trainer's WM checkpoints (sha256) and
    reports PASS for ``horizons_s[k]`` and every lower stage.  :meth:`bind` resolves the gate
    once, from the policy CLI's WM receipts; unbound -> stage 0."""
    enabled: bool = False
    horizons_s: list = field(default_factory=lambda: [10.0])
    curriculum_updates: list = field(default_factory=lambda: [0])
    gate_file: Optional[str] = None
    # Entrant-slot cap per stage (the trainer's ``entry_pad_max`` caps the demand-sized pad of
    # ``with_entry_slots`` for the base 10 s horizon); None = the base cap at every stage.
    entry_pad_max: Optional[list] = None

    def pad_cap(self, horizon_s: float) -> int:
        if not self.entry_pad_max:
            return 0
        stage = max(i for i, h in enumerate(self.horizons_s) if h <= horizon_s or i == 0)
        return int(self.entry_pad_max[stage])

    def __post_init__(self):
        if self.entry_pad_max is not None and (
                len(self.entry_pad_max) != len(self.horizons_s)
                or any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in self.entry_pad_max)):
            raise ValueError("horizon.entry_pad_max must list one positive integer per stage")
        hs = [float(h) for h in self.horizons_s]
        xs = [float(x) for x in self.curriculum_updates]
        if not hs or len(hs) != len(xs):
            raise ValueError("horizon.horizons_s and curriculum_updates must be nonempty, equal length")
        if any(b <= a for a, b in zip(hs, hs[1:])) or any(b <= a for a, b in zip(xs, xs[1:])):
            raise ValueError("horizon stages must be strictly increasing")
        if any(h <= 0 or h != int(h) for h in hs):
            raise ValueError("horizon.horizons_s must be positive whole control seconds")
        self.horizons_s, self.curriculum_updates = hs, xs
        self.allowed_stage = 0
        self.gate_record = None

    def bind(self, wm_sha256s, wm_config_sha256s=None) -> dict:
        """Resolve the gate for the loaded WM checkpoints; returns the record logged by the CLI."""
        self.allowed_stage, record = 0, {"gate_file": self.gate_file, "allowed_stage": 0}
        if not self.enabled or len(self.horizons_s) == 1 or not self.gate_file:
            self.gate_record = record
            return record
        import json
        from pathlib import Path
        from .start_states import checked_path

        try:
            doc = json.loads(Path(checked_path(self.gate_file)).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            record["refused"] = f"gate unreadable: {error}"
            self.gate_record = record
            return record
        if doc.get('schema') == 'c4launch_inherited_horizon/1':
            from .c4launch_authority import inherited_gate
            gate = inherited_gate(doc, wm_sha256s)
            self.allowed_stage = max(i for i, h in enumerate(self.horizons_s)
                                     if h <= doc['trusted_horizon_s'])
            record.update(allowed_stage=self.allowed_stage,
                authority='MAIN2026-10-28T15:00', measured_wm3_gate=False,
                inherited_gate_sha256=doc['inherited_gate_sha256'],
                trusted_horizon_s=90, verdicts={k: v['verdict'] for k, v in gate['horizons'].items()})
            self.gate_record = record
            return record
        if doc.get('schema') == 'c4launch_measured_horizon/1':
            from .auditfix_authority import measured_gate
            gate = measured_gate(doc, wm_sha256s)
            if (wm_config_sha256s is not None and
                    set(wm_config_sha256s) != {doc['wm_config_sha256']}):
                raise ValueError('measured authority / loaded WM config mismatch')
            self.allowed_stage = max(i for i, h in enumerate(self.horizons_s)
                                     if h <= doc['trusted_horizon_s'])
            record.update(allowed_stage=self.allowed_stage, wm_sha256=doc['wm_sha256'],
                          authority=doc['schema'], measured_gate=True, trusted_horizon_s=90,
                          measured_gate_sha256=doc['measured_gate_sha256'],
                          cct1_sha256=doc['cct1_sha256'],
                          verdicts={k: v.get('verdict') for k, v in gate['horizons'].items()})
            self.gate_record = record
            return record
        if doc.get("schema") != GATE_SCHEMA or doc.get("wm_sha256") not in set(wm_sha256s):
            record["refused"] = "gate schema / WM checkpoint mismatch"
            self.gate_record = record
            return record
        results = doc.get("horizons", {})
        for stage, horizon in enumerate(self.horizons_s[1:], start=1):
            entry = results.get(str(int(horizon)))
            if not entry or entry.get("verdict") != "PASS":
                break
            self.allowed_stage = stage
        record.update(allowed_stage=self.allowed_stage, wm_sha256=doc.get("wm_sha256"),
                      rule=doc.get("rule"), verdicts={k: v.get("verdict") for k, v in results.items()})
        self.gate_record = record
        return record

    def stage(self, position: int) -> int:
        reached = max(i for i, x in enumerate(self.curriculum_updates) if position >= x or i == 0)
        return min(reached, self.allowed_stage)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["allowed_stage"] = self.allowed_stage
        return out


# ------------------------------------------------------------------ multi-CAV start predicate
def _cells():
    from .rsu_field import CELLS
    return CELLS


def multi_cav_flags(state: Mapping[str, Tensor], cfg: CounterfactualConfig,
                    horizon_s: float) -> dict:
    """Cooperation-situation flags of a start from observable state only (present / is_cav / lane /
    s / v; no driver types, no world labels).  A *group* = >= 2 policy-controllable CAVs chained
    within ``group_radius_m`` (exactly the counterfactual groups of ``build_variants``).

    * ``near_merge``: a group member reaches the merge area [MERGE_S0, MERGE_S1] at its current
      speed within the imagination horizon ``horizon_s`` (so the imagined rollout contains the
      merge; no distance constant);
    * ``congested``: a group member is in an RSU cell that is a queue cell or whose mean speed is
      below ``congestion_speed_mps`` (the existing ``group_rich`` congestion test, at the group);
    * ``ramp_encounter``: ``near_merge`` AND a ramp vehicle also reaches the merge area within the
      horizon (a subset of ``near_merge``, logged separately).
    ``state`` [1,K] (or [K])."""
    from .rsu_field import build_rsu_field

    flags = {"group": False, "near_merge": False, "congested": False, "ramp_encounter": False}
    s = state["s_m"].reshape(-1)
    flat = {k: v.reshape(1, -1) for k, v in state.items() if torch.is_tensor(v) and v.numel() == s.numel()}
    cav = acting_cavs(flat).reshape(-1)
    groups = close_groups(s, cav, cfg.group_radius_m, 2)
    if not groups:
        return flags
    flags["group"] = True
    v = flat["v_mps"].reshape(-1).clamp_min(0.0)
    lane = flat["lane"].reshape(-1).long()
    present = flat["present"].reshape(-1).bool()
    reach = s + v * float(horizon_s)
    in_merge_window = (reach >= MERGE_S0_M) & (s <= MERGE_S1_M)
    ramp_coming = bool((present & (lane == RAMP_LANE) & in_merge_window).any())
    field_ = build_rsu_field(flat)[0]
    congested = (field_[:, 3] > 0) | ((field_[:, 4] > 0) & (field_[:, 1] < cfg.congestion_speed_mps))
    slot_congested = torch.zeros_like(cav)
    for index, (cell_lane, lo, hi) in enumerate(_cells()):
        if bool(congested[index]):
            slot_congested |= present & (lane == cell_lane) & (s >= lo) & (s < hi)
    for members in groups:
        m = torch.tensor(members, dtype=torch.long)
        near = bool(in_merge_window[m].any())
        flags["near_merge"] |= near
        flags["congested"] |= bool(slot_congested[m].any())
        flags["ramp_encounter"] |= near and ramp_coming
    return flags


def multi_cav_start(state: Mapping[str, Tensor], cfg: CounterfactualConfig, horizon_s: float) -> bool:
    """A group in a cooperation situation (:func:`multi_cav_flags`: near merge or congested)."""
    flags = multi_cav_flags(state, cfg, horizon_s)
    return flags["near_merge"] or flags["congested"]


TILT_FEATURES = ("multi_cav", "side_by_side", "high_share", "high_demand", "congested")
# Registered collection grid (LOOP14 collect: demands / penetrations) and the trainer's existing
# nearest-level share buckets (DreamerTrainer._penetration_buckets); used only to name the
# nearest grid level of an OBSERVED quantity, never as a tuned threshold.
SHARE_LEVELS = (0.02, 0.05, 0.1, 0.2, 0.4)
DEMAND_LEVELS_VPH = (1900.0, 2100.0, 2400.0, 2700.0)


def _nearest(value: float, levels) -> float:
    return min(levels, key=lambda level: abs(level - value))


def side_by_side(state: Mapping[str, Tensor], lane_rules=None) -> bool:
    """Two policy-controllable CAVs in adjacent lanes (|lane difference| = 1, the ramp counts as
    adjacent to lane 1) where either lies inside the other's lane-change gap window of the
    policy lane mask (front: ``gap_base_m + gap_time_s * v``; rear: ``rear_gap_base_m +
    rear_gap_time_s * v_follower``, bumper to bumper): a lane change of one needs the other's
    cooperation.  Observable positions / lanes / speeds / lengths only."""
    from .lanes import LaneRules

    rules = lane_rules or LaneRules()
    rear_base = rules.rear_gap_base_m if rules.rear_gap_base_m is not None else rules.gap_base_m
    rear_time = rules.rear_gap_time_s if rules.rear_gap_time_s is not None else rules.gap_time_s
    s = state["s_m"].reshape(-1)
    flat = {k: v.reshape(1, -1) for k, v in state.items() if torch.is_tensor(v) and v.numel() == s.numel()}
    cav = acting_cavs(flat).reshape(-1)
    slots = torch.nonzero(cav).flatten()
    if slots.numel() < 2:
        return False
    s, v = s[slots], flat["v_mps"].reshape(-1)[slots].clamp_min(0.0)
    lane = flat["lane"].reshape(-1)[slots].long()
    length = flat.get("length_m", torch.full_like(flat["s_m"], 5.0)).reshape(-1)[slots]
    adjacent = (lane[:, None] - lane[None, :]).abs() == 1                        # [n,n] i, j
    ds = s[None, :] - s[:, None]                                                 # j ahead of i > 0
    front = ds - length[None, :]                                                 # bumper gap i->j
    rear = -ds - length[:, None]                                                 # bumper gap j->i
    in_front = (ds >= 0) & (front <= rules.gap_base_m + rules.gap_time_s * v[:, None])
    in_rear = (ds < 0) & (rear <= rear_base + rear_time * v[None, :])
    return bool((adjacent & (in_front | in_rear)).any())


def start_features(state: Mapping[str, Tensor], cfg: CounterfactualConfig, horizon_s: float, *,
                   entry_queue: Optional[Tensor] = None, lane_rules=None) -> dict:
    """Observable cooperation / headroom features of one start (``state`` [1,K]).

    * ``multi_cav``: :func:`multi_cav_start` (a CAV group near the merge or in congestion);
    * ``side_by_side``: :func:`side_by_side`;
    * ``high_share``: the observed present-CAV share is nearest to the top level 0.4;
    * ``high_demand``: the RSU-observed mainline entrance arrival rate (``entry_queue`` column 2,
      the V9ENTRY 300 s request-rate count of the start source) is nearest to a grid level >= 2400
      vph (the headroom-study high-demand worlds); absent queue state -> False;
    * ``congested``: any RSU cell is a queue cell or occupied with mean speed below
      ``congestion_speed_mps`` (the ``group_rich`` congestion test, road-wide)."""
    from .rsu_field import build_rsu_field

    s = state["s_m"].reshape(-1)
    flat = {k: v.reshape(1, -1) for k, v in state.items() if torch.is_tensor(v) and v.numel() == s.numel()}
    present = flat["present"].bool()
    share = float((present & flat["is_cav"].bool()).sum()) / max(1, int(present.sum()))
    field_ = build_rsu_field(flat)[0]
    congested = (field_[:, 3] > 0) | ((field_[:, 4] > 0) & (field_[:, 1] < cfg.congestion_speed_mps))
    demand = None
    if entry_queue is not None and entry_queue.numel() >= 3:
        demand = float(entry_queue.reshape(-1)[2]) * 3600.0
    return {"multi_cav": multi_cav_start(flat, cfg, horizon_s),
            "side_by_side": side_by_side(flat, lane_rules),
            "high_share": _nearest(share, SHARE_LEVELS) == SHARE_LEVELS[-1],
            "high_demand": demand is not None and _nearest(demand, DEMAND_LEVELS_VPH) >= 2400.0,
            "congested": bool(congested.any()),
            "observed_share": share, "observed_main_demand_vph": demand}


class TiltSampler:
    """Training-start tilt toward cooperation / headroom situations by rejection sampling.

    A candidate x is drawn from the TARGET distribution (the sampler's plain uniform-window draw)
    and accepted with probability m(x) / m_max, m(x) = prod_f (1 + w_f * feature_f(x)) (each
    present feature multiplies the sampling weight by 1 + w_f; an additive tilt saturates).  Accepted
    starts follow p(x) m(x) / Z with Z = E_p[m]; every candidate (accepted or not) is a target
    draw, so the running means of m, m^2 and each feature over candidates are unbiased target
    estimates.  Logged per update: the features, m, the importance weight Z / m(x) of the start
    (p / q), the running effective-sample-size fraction (E_p m)^2 / E_p[m^2] of the tilt, and
    target base rates.  Batch size 1 (POLICY14); ``max_tries`` bounds the draws (the last
    candidate is kept, a negligible bias when m_max / E_p[m] << max_tries)."""

    def __init__(self, weights: Mapping[str, float]):
        unknown = set(weights) - set(TILT_FEATURES)
        if unknown or not weights:
            raise ValueError(f"start_tilt features must be from {TILT_FEATURES}")
        if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights.values()):
            raise ValueError("start_tilt weights must be finite and nonnegative")
        self.weights = {k: float(v) for k, v in weights.items()}
        self.m_max = math.prod(1.0 + w for w in self.weights.values())
        self.n = 0
        self.sum_m = 0.0
        self.sum_m2 = 0.0
        self.feature_counts = {k: 0 for k in TILT_FEATURES}

    def multiplier(self, features: Mapping[str, bool]) -> float:
        return math.prod(1.0 + w for f, w in self.weights.items() if features.get(f))

    @property
    def z(self) -> float:
        return self.sum_m / self.n if self.n else 1.0

    @property
    def ess_fraction(self) -> float:
        return (self.sum_m / self.n) ** 2 / (self.sum_m2 / self.n) if self.n else 1.0

    def draw(self, sampler, batch_size, device, features_fn, *, max_tries, generator):
        if batch_size != 1:
            raise ValueError("start_tilt supports batch_size = 1 (POLICY14)")
        for attempt in range(max(1, int(max_tries))):
            batch = sampler.sample(1, device=device)
            features = features_fn(batch)
            m = self.multiplier(features)
            self.n += 1
            self.sum_m += m
            self.sum_m2 += m * m
            for name in TILT_FEATURES:
                self.feature_counts[name] += int(bool(features.get(name)))
            if bool(torch.rand((), generator=generator) < m / self.m_max):
                break
        if batch.metadata and isinstance(batch.metadata[0], dict):
            batch.metadata[0]["group_rich"] = bool(features.get("multi_cav"))
        info = {"features": {k: features.get(k) for k in TILT_FEATURES},
                "observed_share": features.get("observed_share"),
                "observed_main_demand_vph": features.get("observed_main_demand_vph"),
                "m": m, "tries": attempt + 1, "importance_weight": self.z / m,
                "ess_fraction": self.ess_fraction, "candidates": self.n,
                "target_base_rates": {k: c / self.n for k, c in self.feature_counts.items()}}
        return batch, info

    def state_dict(self):
        return {"weights": dict(self.weights), "n": self.n, "sum_m": self.sum_m,
                "sum_m2": self.sum_m2, "feature_counts": dict(self.feature_counts)}

    def load_state_dict(self, state):
        if state.get("weights") != self.weights:
            return              # changed tilt: restart the running target estimates
        self.n, self.sum_m, self.sum_m2 = int(state["n"]), float(state["sum_m"]), float(state["sum_m2"])
        self.feature_counts = {k: int(state["feature_counts"].get(k, 0)) for k in TILT_FEATURES}


# ------------------------------------------------------------------ term decomposition / audit
def term_coefficients(variants: Optional[Variants], n_rows: int, n_starts: int, *, device=None,
                      dtype=torch.float32) -> dict:
    """Unit-weight row-coefficient vector of each objective term.  ``row_coefficients`` is linear
    in the weights; ``coop`` = c(ind=1, coop=1) - c(ind=1) keeps every evaluation inside the
    anti-gaming guard."""
    zero = {"individual": 0.0, "group": 0.0, "coop": 0.0, "team": 0.0, "all": 0.0}

    def c(**w):
        return row_coefficients(variants, {**zero, **w}, n_rows, n_starts, device=device, dtype=dtype)
    out = {"all": c(all=1.0), "individual": c(individual=1.0), "group": c(group=1.0),
           "team": c(team=1.0)}
    out["coop"] = c(individual=1.0, coop=1.0) - out["individual"]
    return out


def term_objectives(imagined, scale: float, lane_pg_weight: float, *, uncertainty=True) -> tuple:
    """``({term: maximised objective at unit weight}, {term: start-of-rollout value})`` with the
    advantages and normalisation of ``DreamerTrainer.actor_objective``.  ``uncertainty`` adds the
    unit-weight pessimism term on the all-CAV rows: d R_t / d w = -sum_{k>=t} std_k for POLICY14's
    gamma = lambda = 1 returns (compared with the ``all`` term, its anchor)."""
    n_base = imagined["n_base"]
    returns = imagined["returns"]
    n_rows = returns.shape[-1]
    variants = imagined.get("variants")
    base = baseline_index(variants, n_rows, device=returns.device)
    has_base = base != torch.arange(n_rows, device=returns.device)
    difference = returns - torch.where(has_base, returns[..., base].detach(), torch.zeros_like(returns))
    lanes = imagined.get("lanes")
    joint = advantage = None
    if lanes is not None:
        advantage = difference.detach() / scale
        values = imagined.get("values")
        if values is not None:
            advantage[..., :n_base] = (returns[..., :n_base] - values[:, :-1, :n_base]).detach() / scale
        joint = (lanes["lane_log_prob"] * lanes["lane_active"].to(returns.dtype)).sum(-1)
    coefficients = term_coefficients(variants, n_rows, n_base, device=returns.device,
                                     dtype=returns.dtype)
    terms, values = {}, {}
    for name, c in coefficients.items():
        if not bool(c.abs().sum() > 0):
            continue
        term = (difference.mean((0, 1)) * c).sum() / scale
        if joint is not None:
            term = term + lane_pg_weight * ((joint * advantage).mean((0, 1)) * c).sum()
        terms[name] = term
        values[name] = float((difference[:, 0].detach().mean(0) * c).sum() / scale)
    disagreement = imagined.get("disagreement")
    if uncertainty and disagreement is not None and disagreement.requires_grad:
        # gamma = lambda = 1 (POLICY14): the penalty's return is the tail sum over steps.
        penalty = -disagreement.flip(0).cumsum(0).flip(0)            # [H,N]
        c = coefficients["all"]
        terms["uncertainty"] = (penalty.mean(0) * c).sum() / scale
        values["uncertainty"] = float((penalty[0].detach() * c).sum() / scale)
    return terms, values


def term_gradient_norms(terms: Mapping[str, Tensor], parameters) -> dict:
    """L2 norm over ``parameters`` of each term's gradient (graph retained, no .grad writes)."""
    norms = {}
    for name, term in terms.items():
        if not term.requires_grad:
            norms[name] = 0.0
            continue
        grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
        norms[name] = math.sqrt(sum(float(g.detach().double().square().sum())
                                    for g in grads if g is not None))
    return norms


def coverage(variants: Optional[Variants], state: Mapping[str, Tensor]) -> dict:
    """Counterfactual row coverage of one update (logged with ``coop_audit``)."""
    if variants is None:
        return {"cf_starts": 0}
    cav = acting_cavs(state)
    acting = int(cav[list(variants.base_row)].sum()) if variants.base_row else 0
    covered = sum(len(g["members"]) for g in variants.groups)
    return {"cf_starts": len(variants.base_row), "n_groups": len(variants.groups),
            "group_sizes": [len(g["members"]) for g in variants.groups],
            "team_groups": sum(len(g["members"]) >= 3 for g in variants.groups),
            "rows_individual": len(variants.individual_rows),
            "rows_flip": int((variants.row_kind == 4).sum()),
            "rows_total": variants.n_rows,
            "acting_cavs": acting,
            "cav_in_group_fraction": covered / acting if acting else None}


