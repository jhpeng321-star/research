"""WM-S: a differentiable IDM/MOBIL residual next-state model.

The model owns no controller policy.  HDVs use differentiable physics plus a
gated residual; CAVs are copied from a real-controller teacher trajectory (or
must be supplied by a phase-B controller callback).  ``history`` is deliberately
an explicit interface so the T3 driver-history feature can be merged without
changing the step contract.
"""
from __future__ import annotations

import sys
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hdv.config import HdvConfig, KIN_F  # noqa: E402
from hdv.net import HdvResidualNet  # noqa: E402
from .driver_encoder import (DRIVER_ENCODER_HIDDEN, DriverEncoder,  # noqa: E402
                             PARAMETER_NAMES)
from .lane_head import HistoryLaneHead  # noqa: E402
from .history import history_spec, LONG_HISTORY_F  # noqa: E402

from .stability import FeasibilityProjection, bounded_value  # noqa: E402
from .brakefix import bounded_value_asym  # noqa: E402
from .long_horizon import soft_bound, validate_bounds  # noqa: E402
from .entries import section_ids as _section_ids  # noqa: E402


@dataclass
class PhysicsConfig:
    """Frozen rollout2-compatible physics switches and IDM parameters."""

    dt_s: float = 0.5
    open_road: bool = True
    sumo_fidelity: bool = False
    sumo_idm: bool = True
    speed_factor: float = 1.0
    speed_limit_mps: float = 31.29
    free_road_gap_m: float = 250.0
    max_gap_steps: int = 240
    accel_min_mps2: float = -9.0
    accel_max_mps2: float = 3.0
    lane_gap_m: float = 5.0
    lane_incentive_mps: float = 0.4
    lane_safe_gap_m: float = 6.0
    # s_down-frame geometry (rollout2.baseline.model_frame_params): the accel
    # lane (2) ends at 75.44 m; the mandatory-merge zone is [0, 75.44].
    merge_zone_s0_m: float = 0.0
    merge_zone_s1_m: float = 75.44
    merge_lane: int = 2
    merge_target_lane: int = 1
    mobil_threshold_mps2: float = 0.1
    b_safe_base_mps2: float = 4.0
    b_safe_end_mps2: float = 6.0
    # Calibrated SUMO-like mandatory-merge gap acceptance + cooperative
    # yielding (rollout2.merge_cal / _out/merge_params.json, fit on TRAIN
    # chunk-4).  Flag-gated, default OFF: the legacy MOBIL-merge baseline is
    # byte-identical when False.
    merge_cal: bool = False
    t_lead_base_s: float = 1.0
    t_lead_end_s: float = 0.10
    b_accept_base_mps2: float = 4.0
    b_accept_end_mps2: float = 6.0
    gap_floor_m: float = 0.3
    u_yield: float = 0.35
    yield_lookahead_m: float = 120.0
    b_yield_max_mps2: float = 1.75
    yield_min_speed_mps: float = 2.5
    yield_min_merger_speed_mps: float = 3.5
    end_obstacle_range_m: float = 25.0

    def idm(self) -> tuple[float, float, float, float, int]:
        if self.sumo_fidelity and self.sumo_idm:
            # Probability-weighted hdv_mix values from rollout2.baseline.
            return 2.49, 4.2, 1.135, 2.6, 4
        return 2.6, 4.5, 1.1, 2.5, 4

    @property
    def v0_mps(self) -> float:
        return self.speed_limit_mps * (0.992 if self.sumo_fidelity else 1.0) \
            * float(self.speed_factor)

    @classmethod
    def from_params_file(cls, path: str, **overrides) -> "PhysicsConfig":
        """Build a config from a fitted merge-cal params JSON (unknown keys filtered).

        TRAIN-only calibration artifact (``_out/merge_params.json``); the exported
        params override the built-in defaults.  ``merge_cal`` is not read from the
        file, so the caller must enable it explicitly.
        """
        import json
        known = set(cls.__dataclass_fields__)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"merge-cal params must be a JSON object: {path}")
        data = {k: v for k, v in raw.items()
                if k in known and k != "merge_cal" and v is not None}
        data.update(overrides)
        return cls(**data)


class BatchedPhysics(Protocol):
    """Torch-native batched physics interface for the optional torchphys branch.

    Implementations receive ``[B, K]`` state tensors and return at least
    ``accel_mps2`` and optionally ``lane_logits``.  WM-S does not import the
    torchphys branch; its module only needs to satisfy this structural contract.
    """

    def __call__(self, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]: ...


def _state_kin(state: dict[str, Tensor]) -> Tensor:
    """Build the A11 kinematics-only tensor from a state mapping."""
    # Scalar state fields are authoritative after a rollout step.  The
    # ``veh_kin`` entry is the initial packed view and is retained only for
    # callers that provide no scalar fields.
    if not all(key in state for key in ("s_m", "v_mps", "a_mps2", "lane")) \
            and "veh_kin" in state:
        return state["veh_kin"]
    lane = state["lane"].long()
    one_hot = F.one_hot(lane.clamp_min(0), num_classes=3).to(state["s_m"].dtype)
    one_hot = one_hot * (lane >= 0).unsqueeze(-1).to(one_hot.dtype)
    return torch.cat((state["s_m"].unsqueeze(-1), one_hot,
                      state["v_mps"].unsqueeze(-1),
                      state["a_mps2"].unsqueeze(-1),
                      state.get("length_m", torch.full_like(state["s_m"], 5.0)
                              ).unsqueeze(-1)), dim=-1)


def _present(state: dict[str, Tensor], kin: Tensor) -> Tensor:
    return state.get("present", torch.ones(kin.shape[:2], dtype=torch.bool,
                                             device=kin.device)).bool()


HISTORY_H = 10          # last 5 s at 0.5 s
HISTORY_F = 6           # v/30, a/3, lane one-hot (3), min(gap,250)/100

#: Controller-id vocabulary for the flag-gated black-box variant.  ``none`` is
#: the HDV / no-controller slot; ``other`` collects OPEN-LOOP training policies
#: (trajectory replay, fixed/switching arms) so they are never mislabelled as
#: native.  The black-box model conditions CAV prediction on this id; the
#: structured WM-S ignores the whole embedding.
CONTROLLER_VOCAB = ("none", "native", "pi", "damping", "wall", "other", "direct")
N_CONTROLLERS = len(CONTROLLER_VOCAB)


def controller_index(policy: object) -> int:
    """Map a pack ``policy``/``arm`` string to a :data:`CONTROLLER_VOCAB` index."""
    text = str(policy or "").strip().lower()
    aliases = {"fleet-pi": "pi", "pi-external-u": "pi", "fleet-pi-external-u": "pi",
               "direct-output": "direct"}
    text = aliases.get(text, text)
    if text in CONTROLLER_VOCAB:
        return CONTROLLER_VOCAB.index(text)
    return CONTROLLER_VOCAB.index("other")


def history_step_features(kin: Tensor, gap: Tensor) -> Tensor:
    """Per-step motion features ``[..., K, HISTORY_F]`` from kinematics and gap."""
    return torch.cat((kin[..., 4:5] / 30.0, kin[..., 5:6] / 3.0, kin[..., 1:4],
                      (gap.clamp(0.0, 250.0) / 100.0).unsqueeze(-1)), dim=-1)


class WmSModel(nn.Module):
    """Physics-informed per-vehicle next-state model.

    Inputs use ``state`` with ``veh_kin [B,K,7]`` (or scalar state fields),
    ``present [B,K]`` and optional ``is_cav [B,K]``.  ``history`` is
    ``[B,K,H,F_history]`` with an optional ``history_mask``.  The default
    T3-compatible feature interface is ``(s_down, v, a, lane)``; callers may
    provide any fixed-width feature vector and set ``history_features``.

    ``apply_direct_commands`` (class default ``True`` = historical behaviour)
    controls whether a ``direct`` action-timeline entry drives a CAV
    closed-loop from its issued acceleration / target speed.  The WM-S trainer
    sets it ``False`` for ``--cav-drive teacher`` (default): every CAV then
    follows the recorded trajectory, exactly like the pre-switching training.
    """

    apply_direct_commands: bool = True

    def __init__(self, *, physics: Optional[PhysicsConfig] = None,
                 hdv: Optional[HdvConfig] = None,
                 history_features: int = HISTORY_F, history_len: int = HISTORY_H,
                 history_hidden: int = 64, k_hidden: int = 32,
                 physics_step: Optional[BatchedPhysics] = None,
                 black_box: bool = False,
                 n_controllers: int = N_CONTROLLERS,
                 controller_embed_dim: int = 16,
                 checkpoint_residual: bool = False,
                 residual_cap_mps2: Optional[float] = None,
                 lane_logit_cap: Optional[float] = None,
                 rollout_mode: str = "residual",
                 lane_head: str = "off",
                 residual_cap_neg_mps2: Optional[float] = None,
                 residual_soft_bound_mps2: Optional[float] = None,
                 residual_soft_bound_neg_mps2: Optional[float] = None,
                 history_s: Optional[float] = None,
                 history_stride_s: float = 1.0) -> None:
        super().__init__()
        self.physics = physics or PhysicsConfig()
        if physics_step is not None and not callable(physics_step):
            raise TypeError("physics_step must be a callable batched Torch module")
        self.physics_step = physics_step
        # Flag-gated run-2 bounded correction (B).  ``None`` = off and the
        # rollout is byte-identical to the un-capped model.
        if residual_cap_mps2 is not None and float(residual_cap_mps2) <= 0.0:
            raise ValueError("residual_cap_mps2 must be > 0 when set")
        if lane_logit_cap is not None and float(lane_logit_cap) <= 0.0:
            raise ValueError("lane_logit_cap must be > 0 when set")
        self.residual_cap_mps2 = None if residual_cap_mps2 is None else float(residual_cap_mps2)
        # Braking-fix: optional separate (typically larger) cap on the NEGATIVE
        # (braking) side of the residual.  ``None`` = symmetric/off, byte-identical.
        if residual_cap_neg_mps2 is not None and (
                not math.isfinite(float(residual_cap_neg_mps2))
                or float(residual_cap_neg_mps2) <= 0.0):
            raise ValueError("residual_cap_neg_mps2 must be finite and > 0 when set")
        self.residual_cap_neg_mps2 = (None if residual_cap_neg_mps2 is None
                                      else float(residual_cap_neg_mps2))
        self.lane_logit_cap = None if lane_logit_cap is None else float(lane_logit_cap)
        validate_bounds(residual_soft_bound_mps2, residual_soft_bound_neg_mps2,
                        residual_cap_mps2, residual_cap_neg_mps2)
        self.residual_soft_bound_mps2 = residual_soft_bound_mps2
        self.residual_soft_bound_neg_mps2 = residual_soft_bound_neg_mps2
        if rollout_mode not in ("residual", "driver_params", "hybrid"):
            raise ValueError(
                "rollout_mode must be residual, driver_params or hybrid")
        self.rollout_mode = rollout_mode
        if lane_head not in ("off", "history"):
            raise ValueError("lane_head must be off or history")
        if lane_head == "history" and rollout_mode not in ("driver_params", "hybrid"):
            raise ValueError(
                "history lane head requires driver_params or hybrid rollout")
        self.lane_head_mode = lane_head
        self.history_features = int(history_features)
        self.history_len = int(history_len)
        self.history_s = history_s
        self.history_stride_s = float(history_stride_s)
        self.long_history = history_s is not None
        _, _, self.driver_history_len = history_spec(
            history_s, history_stride_s, self.physics.dt_s)
        if self.long_history and (black_box or self.history_features != HISTORY_F):
            raise ValueError("long history requires structured WM-S with six rolling features")
        if self.history_features < 0:
            raise ValueError("history_features must be >= 0")
        # Fused DriverEncoder embedding width (pooled trunk + interaction proj +
        # last observed feature); also the residual-net conditioning width.
        self.driver_embed_dim = 2 * DRIVER_ENCODER_HIDDEN + self.history_features
        self.hdv = hdv or HdvConfig()
        if (rollout_mode == "hybrid" or self.long_history) and int(self.hdv.conditioning_dim) == 0:
            # Hybrid gives the residual net the inferred driver embedding as an
            # extra input.  Building it with conditioning_dim=0 stays baseline.
            self.hdv = replace(self.hdv, conditioning_dim=self.driver_embed_dim)
        # Flag-gated black-box variant: CAVs are learned movers conditioned on a
        # controller-id embedding; no controller code is ever executed.
        self.black_box = bool(black_box)
        self.n_controllers = int(n_controllers)
        self.controller_embed_dim = int(controller_embed_dim)
        if self.black_box and self.n_controllers < 1:
            raise ValueError("black_box needs at least one controller id")
        if self.black_box and self.controller_embed_dim < 1:
            raise ValueError("controller_embed_dim must be >= 1 in black_box")
        self.checkpoint_residual = bool(checkpoint_residual)
        self.residual_net = HdvResidualNet(self.hdv)
        self.history_proj = nn.Sequential(
            nn.Linear(max(1, self.history_features * self.history_len) + self.history_len,
                      history_hidden), nn.GELU(),
            nn.Linear(history_hidden, history_hidden))
        self.k_embedding = nn.Embedding(self.physics.max_gap_steps + 1, k_hidden)
        temporal_in = history_hidden + k_hidden
        self.controller_embed = None
        self.command_proj = None
        if self.black_box:
            self.controller_embed = nn.Embedding(self.n_controllers,
                                                 self.controller_embed_dim)
            self.command_proj = nn.Linear(2, self.controller_embed_dim, bias=False)
            temporal_in += self.controller_embed_dim
        self.temporal = nn.Sequential(
            nn.Linear(temporal_in, history_hidden), nn.GELU(),
            nn.Linear(history_hidden, 4))  # accel delta + 3 lane logits
        # Exact zero at k=0 is intentional.  Positive increments make the gate
        # monotone; the parameter is initialized small so early steps remain
        # physics-dominated.
        self.gate_increments = nn.Parameter(
            torch.full((self.physics.max_gap_steps,), -5.0))
        self.driver_encoder = DriverEncoder(self.history_features, long_history=self.long_history)
        self.lane_change_head = (HistoryLaneHead(self.driver_embed_dim)
                                 if lane_head == "history" else None)

    def _lane_context(self, kin: Tensor, present: Tensor, lane_base: Tensor,
                      driver_params: Optional[dict[str, Tensor]],
                      lane_age: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """Build current/adjacent-lane leader/follower context and safe masks."""
        s, v = kin[..., 0], kin[..., 4]
        lane = kin[..., 1:4].argmax(-1)
        chunks = [s.unsqueeze(-1) / 250.0, v.unsqueeze(-1) / 30.0,
                  kin[..., 5:6] / 3.0, kin[..., 1:4]]
        target_safe = torch.zeros_like(lane_base, dtype=torch.bool)
        min_gap = (self.physics.lane_safe_gap_m if driver_params is None
                   else driver_params["merge_gap"])
        for target in range(3):
            tl = torch.full_like(lane, target)
            lead_gap, lead_v = self._neighbour(kin, present, tl, ahead=True)
            foll_gap, foll_v = self._neighbour(kin, present, tl, ahead=False)
            # Missing neighbours have the configured free-road gap and zero
            # relative speed; clamp to a finite range before feeding the head.
            chunks.extend(((lead_gap.clamp(0, 250) / 100).unsqueeze(-1),
                           ((lead_v - v) / 30).unsqueeze(-1),
                           (foll_gap.clamp(0, 250) / 100).unsqueeze(-1),
                           ((foll_v - v) / 30).unsqueeze(-1)))
            safe = (lead_gap >= min_gap) & (foll_gap >= min_gap)
            adjacent = (lane - target).abs() == 1
            mandatory = (lane == self.physics.merge_lane) \
                & (s <= self.physics.merge_zone_s1_m)
            allowed = adjacent & safe & (lane_base[..., target] > -7.9)
            # The physics baseline's safety test also includes follower braking;
            # use its masked directional logits as an additional hard gate.
            target_safe[..., target] = allowed
            if target != self.physics.merge_target_lane:
                target_safe[..., target] &= ~mandatory
        end_distance = (self.physics.merge_zone_s1_m - s).clamp(-250, 250) / 100
        age = torch.full_like(s, 999.0) if lane_age is None else lane_age
        age_feature = (age * self.physics.dt_s).clamp(0, 60) / 10
        context = torch.cat((*chunks, end_distance.unsqueeze(-1),
                             age_feature.unsqueeze(-1), lane_base), dim=-1)
        # Three direction classes map stay/left/right, while baseline logits are
        # indexed by absolute lane.  Translate safety by adjacent target lane.
        action_safe = torch.zeros_like(lane_base, dtype=torch.bool)
        action_safe[..., 0] = True
        for action, target in ((1, lane - 1), (2, lane + 1)):
            valid_target = (target >= 0) & (target < 3)
            target_clamped = target.clamp(0, 2)
            action_safe[..., action] = valid_target & target_safe.gather(
                -1, target_clamped.unsqueeze(-1)).squeeze(-1)
        mandatory = ((lane == self.physics.merge_lane)
                     & (s >= self.physics.merge_zone_s1_m - 25.0)
                     & (s <= self.physics.merge_zone_s1_m))
        action_safe[..., 1:] &= ((age >= 6) | mandatory).unsqueeze(-1)
        merge_action = 1 if self.physics.merge_target_lane < self.physics.merge_lane else 2
        can_merge = action_safe[..., merge_action]
        action_safe[..., 0] &= ~(mandatory & can_merge)
        other_action = 3 - merge_action
        action_safe[..., other_action] &= ~mandatory
        return context, action_safe

    def load_a11_state_dict(self, state_dict: dict[str, Tensor], *, strict: bool = False):
        """Initialize the reusable A11 residual core without touching A11."""
        own = self.residual_net.state_dict()
        copied = {}
        for key, value in state_dict.items():
            name = key.removeprefix("hdv.").removeprefix("residual_net.")
            if name in own and own[name].shape == value.shape:
                copied[name] = value
        result = self.residual_net.load_state_dict(copied, strict=strict)
        return {"loaded": len(copied), "available": len(own),
                "missing": list(result.missing_keys)}

    def gate(self, k_steps: Tensor | int) -> Tensor:
        """Return learned monotone gates, with exactly ``g(0) == 0``."""
        k = torch.as_tensor(k_steps, device=self.gate_increments.device,
                            dtype=torch.long)
        k = k.clamp(0, self.physics.max_gap_steps)
        cumulative = F.softplus(self.gate_increments).cumsum(0)
        shape = k.shape
        flat = k.reshape(-1)
        out = torch.zeros_like(flat, dtype=cumulative.dtype)
        positive = flat > 0
        if bool(positive.any()):
            out[positive] = cumulative[flat[positive] - 1]
        return out.reshape(shape)

    def _idm(self, v: Tensor, leader_v: Tensor, gap: Tensor,
             params: Optional[dict[str, Tensor]] = None) -> Tensor:
        a_max, b_comf, time_headway, s0, delta = self.physics.idm()
        v0 = max(self.physics.v0_mps, 1.0e-3)
        if params is not None:
            a_max, b_comf = params["accel"], params["decel"]
            time_headway, s0, v0 = (params["time_headway"], params["min_gap"],
                                     params["v0"].clamp_min(1.0e-3))
        closing = v - leader_v
        desired = s0 + torch.clamp(
            v * time_headway + v * closing / (2.0 * (a_max * b_comf) ** 0.5),
            min=0.0)
        accel = a_max * (1.0 - (v / v0).pow(delta)
                         - (desired / gap.clamp_min(0.3)).pow(2))
        return accel.clamp(self.physics.accel_min_mps2,
                           self.physics.accel_max_mps2)

    def _neighbour(self, kin: Tensor, present: Tensor, query_lane: Tensor, *,
                   ahead: bool) -> tuple[Tensor, Tensor]:
        """Bumper gap and speed of the nearest vehicle ahead/behind in ``query_lane``."""
        s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
        lane = kin[..., 1:4].argmax(-1)
        ds = s.unsqueeze(-2) - s.unsqueeze(-1)          # [B,i,j] = s_j - s_i
        other = present.unsqueeze(-2) & (lane.unsqueeze(-2) == query_lane.unsqueeze(-1))
        valid = other & ((ds > 0) if ahead else (ds < 0))
        if ahead:
            bumper = ds - length.unsqueeze(-2)
        else:
            bumper = -ds - length.unsqueeze(-1)
        gaps = torch.where(valid, bumper, torch.full_like(ds, float("inf")))
        gap, index = gaps.min(dim=-1)
        has = torch.isfinite(gap)
        other_v = torch.gather(v, -1, index)
        free = self.physics.free_road_gap_m if self.physics.open_road else 1.0e3
        return torch.where(has, gap.clamp_min(0.3), torch.full_like(gap, free)),             torch.where(has, other_v, v)

    def _lead(self, kin: Tensor, present: Tensor):
        """Same-lane leader gap/speed, with the accel-lane end as a standing obstacle."""
        lane = kin[..., 1:4].argmax(-1)
        gap, lv = self._neighbour(kin, present, lane, ahead=True)
        end_gap = (self.physics.merge_zone_s1_m - kin[..., 0]).clamp_min(0.3)
        obstacle = (lane == self.physics.merge_lane) & (end_gap < gap)
        if self.physics.merge_cal:
            obstacle = obstacle & (end_gap <= self.physics.end_obstacle_range_m)
        return torch.where(obstacle, end_gap, gap),             torch.where(obstacle, torch.zeros_like(lv), lv)

    def _merge_cal_yield(self, kin: Tensor, present: Tensor, accel: Tensor) -> Tensor:
        """Bounded cooperative mainline yielding (rollout2.merge_cal)."""
        p = self.physics
        lane = kin[..., 1:4].argmax(-1)
        s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
        span = max(p.merge_zone_s1_m - p.merge_zone_s0_m, 1.0)
        urgency = ((s - p.merge_zone_s0_m) / span).clamp(0.0, 1.0)
        urgent = present & (lane == p.merge_lane) & (urgency >= p.u_yield)
        lane1 = present & (lane == p.merge_target_lane)
        ds = s.unsqueeze(-2) - s.unsqueeze(-1)          # [B,i,j] = s_j - s_i
        valid = urgent.unsqueeze(-2) & lane1.unsqueeze(-1) & (ds > 0.0) \
            & (ds <= p.yield_lookahead_m) \
            & (v.unsqueeze(-1) >= p.yield_min_speed_mps) \
            & (v.unsqueeze(-2) >= p.yield_min_merger_speed_mps)
        gap_v = ds - length.unsqueeze(-2)
        a_v = self._idm(v.unsqueeze(-1), v.unsqueeze(-2), gap_v)
        a_v = torch.where(valid, a_v, torch.full_like(a_v, float("inf")))
        a_min = a_v.min(dim=-1).values.clamp_min(-p.b_yield_max_mps2)
        has = valid.any(dim=-1)
        return torch.where(lane1 & has, torch.minimum(accel, a_min), accel)

    def physics_accel(self, state: dict[str, Tensor],
                      driver_params: Optional[dict[str, Tensor]] = None) -> Tensor:
        kin = _state_kin(state)
        present = _present(state, kin)
        gap, leader_v = self._lead(kin, present)
        accel = self._idm(kin[..., 4], leader_v, gap, driver_params)
        if driver_params is not None:
            accel = torch.maximum(accel, -driver_params["decel"])
            accel = torch.minimum(accel, driver_params["accel"])
        if self.physics.merge_cal:
            accel = self._merge_cal_yield(kin, present, accel)
        return torch.where(present, accel, torch.zeros_like(accel))

    def mobil_merge_logits(self, state: dict[str, Tensor],
                           driver_params: Optional[dict[str, Tensor]] = None) -> Tensor:
        """Differentiable-input MOBIL + mandatory-merge baseline lane logits.

        ``stay`` has logit 0; a change to an adjacent lane has
        ``3 tanh((incentive - threshold (1-u))/4) - 2.5 (+ 6 u^2 on the accel lane)``
        and is masked (-8) unless the leader/follower gaps and the follower
        braking in the target lane are safe (b_safe grows with merge urgency u).
        """
        p = self.physics
        kin = _state_kin(state)
        present = _present(state, kin)
        s, v = kin[..., 0], kin[..., 4]
        lane = kin[..., 1:4].argmax(-1)
        gap0, lv0 = self._lead(kin, present)
        a_cur = self._idm(v, lv0, gap0, driver_params)
        current_follower_accel = a_cur
        span = max(p.merge_zone_s1_m - p.merge_zone_s0_m, 1.0)
        urgency = ((s - p.merge_zone_s0_m) / span).clamp(0.0, 1.0)
        threshold = p.mobil_threshold_mps2
        politeness = torch.full_like(v, 0.2)
        min_gap = torch.full_like(v, p.lane_safe_gap_m)
        if driver_params is not None:
            threshold = driver_params["mobil_threshold"]
            politeness = driver_params["politeness"]
            min_gap = driver_params["merge_gap"]
            urgency = (urgency * driver_params["merge_urgency"]).clamp(0.0, 1.0)
        in_zone = (s >= p.merge_zone_s0_m) & (s <= p.merge_zone_s1_m)
        b_safe = p.b_safe_base_mps2 + (p.b_safe_end_mps2 - p.b_safe_base_mps2) * urgency
        logits = torch.full((*s.shape, 3), -8.0, dtype=kin.dtype, device=kin.device)
        for target in range(3):
            tl = torch.full_like(lane, target)
            gap_l, lv_l = self._neighbour(kin, present, tl, ahead=True)
            gap_f, fv_f = self._neighbour(kin, present, tl, ahead=False)
            a_new = self._idm(v, lv_l, gap_l, driver_params)
            ds = s.unsqueeze(-2) - s.unsqueeze(-1)
            follower = (present.unsqueeze(-2) & present.unsqueeze(-1)
                        & (lane.unsqueeze(-2) == tl.unsqueeze(-1)) & (ds < 0))
            follower_dist = torch.where(follower, -ds, torch.full_like(ds, float("inf")))
            follower_gap, follower_index = follower_dist.min(-1)
            follower_has = torch.isfinite(follower_gap)
            follower_old = torch.gather(current_follower_accel, -1, follower_index)
            follower_params = (None if driver_params is None else
                               {name: torch.gather(value, -1, follower_index)
                                for name, value in driver_params.items()})
            a_follow = self._idm(fv_f, v, gap_f, follower_params)
            follower_gain = torch.where(follower_has, a_follow - follower_old,
                                        torch.zeros_like(a_follow))
            adjacent = (lane - target).abs() == 1
            ramp_ok = (lane != p.merge_lane) | (in_zone if target == p.merge_target_lane
                                                else torch.zeros_like(in_zone))
            no_to_ramp = torch.full_like(lane, target) != p.merge_lane
            forced = (lane == p.merge_lane) & (target == p.merge_target_lane)
            safe = (gap_l >= min_gap) & (gap_f >= min_gap) & (a_follow >= -b_safe)
            allowed = adjacent & ramp_ok & no_to_ramp & safe
            u = urgency * forced.to(kin.dtype)
            if driver_params is None:
                score = (a_new - a_cur) - p.mobil_threshold_mps2 * (1.0 - u)
            else:
                score = (a_new - a_cur) + politeness * follower_gain - threshold * (1.0 - u)
            move = 3.0 * torch.tanh(score / 4.0) - 2.5 + 6.0 * u * u
            logits[..., target] = torch.where(allowed, move, logits[..., target])
        if p.merge_cal:
            # Strategic mandatory merge: accept whenever physically safe, with
            # the calibrated urgency-relaxed lead/follower ceilings.
            target = p.merge_target_lane
            tl = torch.full_like(lane, target)
            gap_l, lv_l = self._neighbour(kin, present, tl, ahead=True)
            gap_f, fv_f = self._neighbour(kin, present, tl, ahead=False)
            floor = p.gap_floor_m
            s0i = p.idm()[3]
            t_lead = p.t_lead_base_s + (p.t_lead_end_s - p.t_lead_base_s) * urgency
            b_acc = p.b_accept_base_mps2 + (p.b_accept_end_mps2 - p.b_accept_base_mps2) * urgency
            b_req_foll = (fv_f ** 2 - v ** 2) / (2.0 * gap_f.clamp_min(floor))
            b_req_lead = (v ** 2 - lv_l ** 2) / (2.0 * gap_l.clamp_min(floor))
            cal_safe = (gap_l >= torch.maximum(torch.full_like(gap_l, floor),
                                               s0i + v * t_lead)) \
                & (gap_f >= floor) & (gap_l >= floor) \
                & (b_req_foll <= b_acc) & (b_req_lead <= b_acc)
            mandatory = (lane == p.merge_lane) & in_zone
            logits[..., target] = torch.where(
                mandatory & cal_safe, torch.full_like(s, 3.0),
                torch.where(mandatory, torch.full_like(s, -8.0), logits[..., target]))
        logits = logits.scatter(-1, lane.unsqueeze(-1), torch.zeros_like(s).unsqueeze(-1))
        return logits * present.unsqueeze(-1).to(kin.dtype)

    def _physics_step(self, state: dict[str, Tensor],
                      driver_params: Optional[dict[str, Tensor]] = None) -> tuple[Tensor, Tensor]:
        """Run the injected backend or the frozen in-module Torch baseline."""
        if self.physics_step is None:
            return self.physics_accel(state, driver_params), self.mobil_merge_logits(state, driver_params)
        result = self.physics_step(state)
        if not isinstance(result, Mapping) or "accel_mps2" not in result:
            raise ValueError("physics_step must return mapping with accel_mps2")
        accel = result["accel_mps2"]
        baseline_lane = result.get("lane_logits")
        kin = _state_kin(state)
        expected = kin.shape[:2]
        if tuple(accel.shape) != tuple(expected):
            raise ValueError(f"physics accel shape {tuple(accel.shape)} != {expected}")
        if not torch.isfinite(accel).all():
            raise ValueError("physics_step returned non-finite acceleration")
        if baseline_lane is None:
            baseline_lane = self.mobil_merge_logits(state)
        elif tuple(baseline_lane.shape) != (*expected, 3):
            raise ValueError(
                f"physics lane logits shape {tuple(baseline_lane.shape)} != {tuple(expected) + (3,)}")
        if not torch.isfinite(baseline_lane).all():
            raise ValueError("physics_step returned non-finite lane logits")
        return accel.to(kin.dtype), baseline_lane.to(kin.dtype)

    def _temporal_delta(self, history: Optional[Tensor], history_mask: Optional[Tensor],
                        k_steps: Tensor, shape: tuple[int, int], dtype, device,
                        controller_emb: Optional[Tensor] = None):
        b, k = shape
        h, f = self.history_len, self.history_features
        if history is None:
            flat = torch.zeros(b, k, max(1, f * h) + h, dtype=dtype, device=device)
        else:
            if history.ndim != 4 or history.shape[:2] != (b, k):
                raise ValueError("history must be [B,K,H,F_history]")
            if history.shape[-1] != f or history.shape[2] != h:
                raise ValueError("history window/feature width does not match model")
            m = torch.ones(history.shape[:-1], dtype=dtype, device=device) \
                if history_mask is None else history_mask.to(dtype)
            flat = torch.cat(((history * m.unsqueeze(-1)).reshape(b, k, h * f), m), -1)
        z = self.history_proj(flat)
        kx = self.k_embedding(k_steps.clamp(0, self.physics.max_gap_steps))
        parts = [z, kx]
        if self.black_box or controller_emb is not None:
            if controller_emb is None:
                controller_emb = torch.zeros(b, k, self.controller_embed_dim,
                                             dtype=dtype, device=device)
            parts.append(controller_emb.to(dtype))
        return self.temporal(torch.cat(parts, dim=-1))

    def _controller_embedding(self, controller_id: Optional[Tensor], is_cav: Tensor,
                              shape: tuple[int, int], dtype, device,
                              action_command: Optional[Tensor] = None):
        """Controller-id embedding on CAV slots only (zero elsewhere/structured)."""
        if not self.black_box:
            return None
        b, k = shape
        if controller_id is None:
            cid = torch.zeros(b, k, dtype=torch.long, device=device)
            cav = torch.zeros(b, k, dtype=torch.bool, device=device)
        else:
            cid = torch.as_tensor(controller_id, device=device).long()
            if cid.ndim == 0:
                cid = cid.expand(b, k)
            elif cid.ndim == 1:
                cid = cid.reshape(b, 1).expand(b, k)
            elif tuple(cid.shape) != (b, k):
                raise ValueError("controller_id must be scalar, [B] or [B,K]")
            cid = cid.clamp(0, self.n_controllers - 1)
            cav = is_cav
        emb = self.controller_embed(cid)
        if action_command is not None:
            command = torch.as_tensor(action_command, dtype=dtype, device=device)
            if command.ndim == 2 and tuple(command.shape) == (b, 2):
                command = command.unsqueeze(1).expand(b, k, 2)
            if tuple(command.shape) != (b, k, 2):
                raise ValueError("action_command must have shape [B,K,2]")
            emb = emb + self.command_proj(command)
        return emb * cav.unsqueeze(-1).to(dtype)

    def forward(self, state: dict[str, Tensor], *, k_steps: Tensor | int = 0,
                history: Optional[Tensor] = None,
                history_mask: Optional[Tensor] = None,
                driver_history: Optional[Tensor] = None,
                driver_history_mask: Optional[Tensor] = None,
                 driver_defaults: Optional[Tensor] = None,
                 driver_params: Optional[dict[str, Tensor]] = None,
                 driver_embedding: Optional[Tensor] = None,
                 lane_age: Optional[Tensor] = None,
                 baseline_lane_logits: Optional[Tensor] = None,
                controller_id: Optional[Tensor] = None,
                action_command: Optional[Tensor] = None) -> dict[str, Tensor]:
        kin = _state_kin(state).float()
        present = _present(state, kin)
        if kin.shape[-1] != KIN_F:
            raise ValueError(f"veh_kin must have {KIN_F} features")
        b, slots = kin.shape[:2]
        is_cav = state.get("is_cav", torch.zeros_like(present)).bool()
        # Structured WM-S learns HDVs only and teacher-forces CAVs; the black-box
        # variant treats every present vehicle as a learned mover (no controller
        # code) conditioned on the controller id.
        learned = present if self.black_box else (present & ~is_cav)
        k = torch.as_tensor(k_steps, device=kin.device, dtype=torch.long)
        if k.ndim == 0:
            k = k.expand(b, slots)
        elif tuple(k.shape) != (b, slots):
            raise ValueError("k_steps must be scalar or [B,K]")
        # Driver encoder (driver_params / hybrid): infer per-HDV IDM/MOBIL
        # parameters from the observed pre-t0 history.  Hybrid mode keeps that
        # per-driver physics AND the gated learned residual, and ALSO feeds the
        # fused embedding to the residual net as extra conditioning.  Nothing
        # here runs in residual mode, so that path stays byte-identical.
        encoded = None
        if self.rollout_mode in ("driver_params", "hybrid") or self.long_history:
            observed = driver_history if driver_history is not None else history
            observed_mask = (driver_history_mask if driver_history_mask is not None
                             else history_mask)
            # Hybrid needs the embedding on every step (to condition the
            # residual net), so re-encode whenever it is missing; driver_params
            # only encodes when the parameters themselves are not yet fixed.
            need_encode = ((self.rollout_mode != "residual" and driver_params is None)
                           or ((self.rollout_mode == "hybrid" or self.long_history)
                               and driver_embedding is None))
            if need_encode:
                if observed is None or observed_mask is None:
                    if self.long_history:
                        raise ValueError("--history-s requires masked pre-t0 driver_history")
                    observed = torch.zeros((b, slots, self.history_len, self.history_features),
                                           device=kin.device, dtype=kin.dtype)
                    observed_mask = torch.zeros((b, slots, self.history_len),
                                                device=kin.device, dtype=torch.bool)
                if driver_defaults is None:
                    a_default, b_default, t_default, s0_default, _ = self.physics.idm()
                    defaults = kin.new_tensor((self.physics.v0_mps, t_default, s0_default,
                                               a_default, b_default, 0.2,
                                               self.physics.mobil_threshold_mps2,
                                               self.physics.lane_safe_gap_m, 1.0))
                    driver_defaults = defaults.view(1, 1, -1).expand(b, slots, -1)
                if self.long_history and (observed.shape != (
                        b, slots, self.driver_history_len, LONG_HISTORY_F)
                        or observed_mask.shape != observed.shape[:-1]):
                    raise ValueError("long driver history shape does not match configured duration")
                gap, lead_v = self._lead(kin, present)
                follow_gap, follow_v = self._neighbour(
                    kin, present, kin[..., 1:4].argmax(-1), ahead=False)
                interaction = torch.stack((gap.clamp(0, 250) / 100, (lead_v - kin[..., 4]) / 30,
                                           follow_gap.clamp(0, 250) / 100,
                                           (follow_v - kin[..., 4]) / 30), -1)
                encoded = self.driver_encoder(observed, observed_mask, interaction, driver_defaults)
                driver_embedding = encoded["embedding"]
                if driver_params is None and self.rollout_mode != "residual":
                    values = encoded["values"]
                    driver_params = {name: values[..., i] for i, name in enumerate(PARAMETER_NAMES)}
            physics, lane_base = self._physics_step({**state, "veh_kin": kin}, driver_params)
        else:
            physics, lane_base = self._physics_step({**state, "veh_kin": kin})
        # Hybrid: condition the residual net on the inferred driver embedding.
        # Checkpointing is bypassed in hybrid (its call carries a keyword).
        conditioning = (driver_embedding if self.rollout_mode == "hybrid" or self.long_history
                        else None)
        if self.checkpoint_residual and self.training and conditioning is None:
            base = checkpoint(self.residual_net, kin, present,
                              use_reentrant=False, preserve_rng_state=False)
        else:
            base = self.residual_net(kin, present, conditioning=conditioning)
        controller_emb = self._controller_embedding(controller_id, is_cav,
                                                    (b, slots), kin.dtype, kin.device,
                                                    action_command=action_command)
        temporal = self._temporal_delta(history, history_mask, k, (b, slots),
                                        kin.dtype, kin.device,
                                        controller_emb=controller_emb)
        gate = self.gate(k).to(kin.dtype)
        accel_delta = (torch.zeros_like(physics) if self.rollout_mode == "driver_params"
                       else base["accel_delta_mps2"] + temporal[..., 0])
        # Bounded learned correction (B): tanh cap when configured, identity
        # (byte-identical) when ``None``.
        raw_residual = gate * accel_delta
        if self.residual_soft_bound_mps2 is not None:
            correction = soft_bound(raw_residual, self.residual_soft_bound_mps2,
                                    self.residual_soft_bound_neg_mps2)
        elif self.residual_cap_neg_mps2 is None:
            correction = bounded_value(gate * accel_delta, self.residual_cap_mps2)
        else:
            correction = bounded_value_asym(gate * accel_delta, self.residual_cap_mps2,
                                            self.residual_cap_neg_mps2)
        accel = torch.where(learned, physics + correction, physics)
        if self.rollout_mode != "driver_params":
            accel = accel.clamp(self.physics.accel_min_mps2,
                                self.physics.accel_max_mps2)
        lane_base = lane_base if baseline_lane_logits is None else baseline_lane_logits
        lane_delta = (torch.zeros_like(lane_base) if self.rollout_mode == "driver_params"
                      else base["lane_logit_delta"] + temporal[..., 1:])
        if self.long_history and self.rollout_mode == "driver_params":
            # Even without an explicit history lane head, inferred style must
            # condition the lane correction as well as IDM/MOBIL parameters.
            lane_delta = base["lane_logit_delta"]
        lane_correction = bounded_value(gate.unsqueeze(-1) * lane_delta,
                                        self.lane_logit_cap)
        lane_logits = lane_base + lane_correction
        lane_probabilities = None
        if self.lane_change_head is not None:
            if driver_embedding is None:
                raise ValueError("history lane head requires its pre-t0 encoder embedding")
            context, feasible_actions = self._lane_context(
                kin, present, lane_base, driver_params, lane_age)
            lane_logits = self.lane_change_head(driver_embedding, context,
                                               feasible_actions)
            lane_probabilities = torch.softmax(lane_logits, dim=-1)
        result = {"physics_accel_mps2": physics, "accel_mps2": accel,
                  "raw_residual_mps2": raw_residual,
                  "accel_delta_mps2": accel - physics, "lane_logits": lane_logits,
                  "gate": gate, "residual_log_sigma": base["accel_log_sigma"],
                  "hdv_mask": learned}
        if encoded is not None:
            result.update({"driver_parameters": encoded["values"],
                           "driver_uncertainty": encoded["uncertainty"],
                            "driver_shrinkage": encoded["shrinkage"]})
        if driver_embedding is not None and (self.lane_change_head is not None
                                             or self.rollout_mode == "hybrid" or self.long_history):
            result["driver_embedding"] = driver_embedding
        if lane_probabilities is not None:
            result["lane_probabilities"] = lane_probabilities
        return result

    def hold_teacher(self, state: dict[str, Tensor]) -> Tensor:
        """Native hold-speed CAV motion ``[B,K,4] = s,lane,v,a`` (no controller)."""
        s, v = state["s_m"], state["v_mps"]
        lane = state["lane"].float()
        return torch.stack((s + v * self.physics.dt_s, lane, v,
                            torch.zeros_like(v)), dim=-1)

    def step(self, state: dict[str, Tensor], *, k_steps: Tensor | int = 0,
             history: Optional[Tensor] = None, history_mask: Optional[Tensor] = None,
             driver_history: Optional[Tensor] = None,
             driver_history_mask: Optional[Tensor] = None,
              driver_defaults: Optional[Tensor] = None,
              driver_params: Optional[dict[str, Tensor]] = None,
              driver_embedding: Optional[Tensor] = None,
              lane_age: Optional[Tensor] = None,
             cav_teacher: Optional[Tensor] = None,
             cav_controller: Optional[Callable[[dict[str, Tensor]], Tensor]] = None,
             baseline_lane_logits: Optional[Tensor] = None,
             controller_id: Optional[Tensor] = None,
             action_id: Optional[Tensor] = None,
             action_command: Optional[Tensor] = None,
             projection: Optional[FeasibilityProjection] = None) -> dict[str, Tensor]:
        """Advance one 0.5 s step; CAVs are never learned movers in structured mode.

        With entries enabled the state carries an ``entrant`` mask: t0 CAVs keep
        the recorded teacher, while entrant CAVs (which have no recorded plan)
        run the supplied ``cav_controller`` or native hold-speed motion.
        """
        forward_kwargs = dict(
            k_steps=k_steps, history=history, history_mask=history_mask,
            driver_history=driver_history,
            driver_history_mask=driver_history_mask,
            driver_defaults=driver_defaults, driver_params=driver_params,
            baseline_lane_logits=baseline_lane_logits,
            controller_id=(controller_id if action_id is None else action_id),
            action_command=action_command)
        if self.rollout_mode == "hybrid" or self.lane_change_head is not None or self.long_history:
            forward_kwargs["driver_embedding"] = driver_embedding
        if self.lane_change_head is not None:
            forward_kwargs["lane_age"] = lane_age
        out = self.forward(state, **forward_kwargs)
        s, v = state["s_m"], state["v_mps"]
        present = _present(state, _state_kin(state))
        cav = state.get("is_cav", torch.zeros_like(present)).bool() & present
        next_v = (v + out["accel_mps2"] * self.physics.dt_s).clamp_min(0.0)
        next_s = s + 0.5 * (v + next_v) * self.physics.dt_s
        choice = out["lane_logits"].argmax(-1)
        if self.lane_change_head is None:
            next_lane = choice
        else:
            lane = state["lane"].long()
            next_lane = lane + (choice == 2).long() - (choice == 1).long()
        # Black-box: CAVs are predicted by the network (controller id is only an
        # input feature); controller code/teacher is never consulted.
        direct = (torch.zeros_like(cav)
                  if action_id is None or not self.apply_direct_commands else
                  (action_id == CONTROLLER_VOCAB.index("direct")) & cav)
        if direct.any():
            if action_command is None:
                raise ValueError("direct timeline action requires acceleration and target speed")
            command = torch.as_tensor(action_command, dtype=v.dtype, device=v.device)
            if command.shape[-1] != 2 or command.shape[:2] != v.shape:
                raise ValueError("action_command must be [B,K,2]")
            requested_a, target_v = command[..., 0], command[..., 1]
            lo = torch.maximum(torch.full_like(v, self.physics.accel_min_mps2),
                               -v / float(self.physics.dt_s))
            hi = torch.minimum(torch.full_like(v, self.physics.accel_max_mps2),
                               (target_v - v) / float(self.physics.dt_s))
            feasible = hi >= lo
            feasible_a = torch.minimum(torch.maximum(requested_a, lo), hi)
            direct_v = (v + feasible_a * float(self.physics.dt_s)).clamp_min(0.0)
            direct_s = s + 0.5 * (v + direct_v) * float(self.physics.dt_s)
            applied = direct & feasible
            next_v = torch.where(applied, direct_v, next_v)
            next_s = torch.where(applied, direct_s, next_s)
            direct = applied
            out["accel_mps2"] = torch.where(applied, feasible_a, out["accel_mps2"])
        if cav.any() and not self.black_box:
            # Execute once per physics tick, including direct-only ticks: the
            # dispatcher owns episode-persistent PI/damping lifecycle state.
            callback_teacher = None
            if cav_controller is not None:
                callback_state = {**state, "physics_step_index": k_steps}
                if action_id is not None:
                    callback_state = {**callback_state, "action_id": action_id,
                                      "action_command": action_command}
                callback_teacher = cav_controller(callback_state)
            entrant = state.get("entrant")
            if entrant is None:
                cav_base, cav_new = cav, torch.zeros_like(cav)
            else:
                entrant = entrant.bool() & present
                cav_base, cav_new = cav & ~entrant, cav & entrant
            cav_base = cav_base & ~direct
            cav_new = cav_new & ~direct
            if cav_base.any():
                if cav_teacher is None and cav_controller is None:
                    raise ValueError("CAVs require a real-controller teacher or callback")
                teacher = cav_teacher if cav_teacher is not None else callback_teacher
                if teacher.shape[-1] != 4 or teacher.shape[:2] != s.shape:
                    raise ValueError("CAV teacher must be [B,K,4] = s,lane,v,a")
                next_s = torch.where(cav_base, teacher[..., 0], next_s)
                next_lane = torch.where(cav_base, teacher[..., 1].long(), next_lane)
                next_v = torch.where(cav_base, teacher[..., 2], next_v)
                out["accel_mps2"] = torch.where(cav_base, teacher[..., 3],
                                                out["accel_mps2"])
            if cav_new.any():
                ctrl = (callback_teacher if cav_controller is not None
                        else self.hold_teacher(state))
                next_s = torch.where(cav_new, ctrl[..., 0], next_s)
                next_lane = torch.where(cav_new, ctrl[..., 1].long(), next_lane)
                next_v = torch.where(cav_new, ctrl[..., 2], next_v)
                out["accel_mps2"] = torch.where(cav_new, ctrl[..., 3],
                                                out["accel_mps2"])
        next_present = present.clone()
        exit_s = state.get("exit_s_m")
        if exit_s is not None:
            next_present = next_present & (next_s <= exit_s)
        # (A) flag-gated differentiable physics feasibility projection.  Absent
        # (None) leaves the step byte-identical.
        if projection is not None:
            length = state.get("length_m", torch.full_like(s, 5.0))
            proj = projection(next_s, next_v, next_lane, next_present, length,
                              is_cav=state.get("is_cav"))
            next_s, next_v, next_lane = proj["s_m"], proj["v_mps"], proj["lane"]
            out["projection_active"] = proj["active"]
        out.update({"next_s_m": next_s, "next_v_mps": next_v,
                    "next_lane": next_lane, "next_present": next_present})
        return out

    def rollout(self, state: dict[str, Tensor], *, n_steps: int, k0: int = 0,
                cav_teacher: Optional[Tensor] = None,
                present_seq: Optional[Tensor] = None,
                junction_seq: Optional[Tensor] = None,
                is_cav_seq: Optional[Tensor] = None,
                 history: Optional[Tensor] = None,
                 history_mask: Optional[Tensor] = None,
                 driver_history: Optional[Tensor] = None,
                 driver_history_mask: Optional[Tensor] = None,
                 driver_defaults: Optional[Tensor] = None,
                 driver_params: Optional[dict[str, Tensor]] = None,
                 driver_embedding: Optional[Tensor] = None,
                 cav_controller: Optional[Callable[[dict[str, Tensor]], Tensor]] = None,
                 truncation_steps: Optional[int] = None,
                 controller_id: Optional[Tensor] = None,
                 action_timeline: Optional[dict[str, Tensor]] = None,
                 entries: Optional[object] = None,
                state_noise_std_m: float = 0.0,
                state_noise_std_mps: float = 0.0,
                projection: Optional[FeasibilityProjection] = None) -> dict[str, Tensor]:
        """Multi-step rollout; ``k`` (steps since the last real observation) is ``k0 + i``.

        ``cav_teacher[:, i+1]`` is the CAV state after step ``i``; ``present_seq
        [:, i+1]`` (optional) is the recorded presence after step ``i`` (exits are
        exogenous).  ``junction_seq[:, i+1]`` (optional) marks a step where the
        slot is on a junction internal lane: the slot is kept present across it
        (state carried / last known) instead of being dropped.  The motion
        history is a rolling buffer fed with the model's
        own predicted states.  The returned ``state``/``history`` allow a chunked
        (truncated-BPTT) continuation.

        ``entries`` (flag-gated, default ``None``) is a
        :class:`wms.entries.TorchEntryProcess`: it inserts observed-counter
        arrivals into free padding slots before each step.  The returned
        ``entrant`` mask marks slots that ever held a post-t0 newcomer, so a
        caller excludes them from the per-car loss while traffic metrics still
        count them.  When entries are active the per-step ``is_cav_seq`` override
        is skipped (the entry process owns ``is_cav`` of new slots).

        ``state_noise_std_m`` / ``state_noise_std_mps`` (Run 2 / B4, default
        ``0.0``) add zero-mean Gaussian noise to the *predicted* HDV position and
        speed each step, acting as a stability regulariser for long rollouts.  A
        value of ``0.0`` performs no random draw, so the default is byte-identical
        to the unregularised rollout.

        Two extra tensors expose the entries interface for a traffic-level loss
        (the ``wms/trafloss`` branch):

        * ``entrant_seq`` ``[B, n_steps, K]`` bool - per-step entrant membership
          (a slot that has been spawned by ``entries`` is marked from the step it
          is inserted onward; all-False when ``entries`` is ``None``);
        * ``sections`` ``[B, n_steps, K]`` int64 - model-frame section id per
          slot per step (0 upstream / 1 merge / 2 downstream, -1 where absent),
          from :func:`wms.entries.section_ids`.
        """
        current = {key: value for key, value in state.items()}
        if controller_id is None:
            controller_id = current.get("controller_id")
        current.pop("controller_id", None)
        kin0 = _state_kin(current)
        pres0 = _present(current, kin0)
        # Entrant membership is carried across truncated-BPTT chunks so a
        # continuation keeps the slots it already spawned.
        entrant = current.get("entrant")
        entrant = (entrant.clone().bool() if torch.is_tensor(entrant)
                   else torch.zeros(pres0.shape, dtype=torch.bool, device=pres0.device))
        noise_on = float(state_noise_std_m) > 0.0 or float(state_noise_std_mps) > 0.0
        if history is None:
            gap0, _ = self._lead(kin0, pres0)
            feat = history_step_features(kin0, gap0)
            h, f = self.history_len, self.history_features
            history = torch.zeros(*feat.shape[:2], h, f, dtype=feat.dtype, device=feat.device)
            history[:, :, -1] = feat
            history_mask = torch.zeros(*feat.shape[:2], h, dtype=torch.bool, device=feat.device)
            history_mask[:, :, -1] = pres0
        positions, speeds, lanes, masks, accels, lane_logits = [], [], [], [], [], []
        lane_probabilities = []
        raw_residuals, change_probabilities = [], []
        projection_totals: dict[str, float] = {}
        entrant_hist: list = []
        fixed_driver_params = driver_params
        fixed_driver_embedding = driver_embedding
        lane_age = current.get("lane_age") if self.lane_change_head is not None else None
        if self.lane_change_head is not None and lane_age is None:
            lane_age = torch.full_like(pres0, 999, dtype=kin0.dtype)
            if driver_history is not None and driver_history_mask is not None \
                    and (not self.long_history or driver_history.shape[-2] > 1):
                hist_lane = driver_history[..., 2:5].argmax(-1)
                valid_pair = (driver_history_mask[..., 1:].bool()
                              & driver_history_mask[..., :-1].bool())
                changed_hist = (hist_lane[..., 1:] != hist_lane[..., :-1]) & valid_pair
                index = torch.arange(changed_hist.shape[-1], device=kin0.device)
                latest = torch.where(changed_hist, index, -1).amax(-1)
                since = changed_hist.shape[-1] - latest - 1
                if self.long_history:
                    since = since * round(self.history_stride_s / self.physics.dt_s)
                lane_age = torch.where(latest >= 0, since.to(kin0.dtype), lane_age)
        encoded_parameters = None
        # Hybrid diagnostics: per-rollout absolute physics vs learned-residual
        # acceleration on learned (HDV) slots, used for the share log and the
        # optional residual-magnitude penalty.  Not allocated in other modes.
        physics_magnitude = None
        residual_magnitude = None
        residual_count = 0
        timeline_dt = float(action_timeline.get("control_dt_s", 1.0)) \
            if action_timeline is not None else 1.0
        if timeline_dt <= 0:
            raise ValueError("action timeline control_dt_s must be positive")
        for step_i in range(int(n_steps)):
            if entries is not None:
                current = {key: (value.clone() if torch.is_tensor(value) else value)
                           for key, value in current.items()}
                entrant = entrant | entries.insert(current)
                current["entrant"] = entrant
            elif is_cav_seq is not None:
                current = {**current, "is_cav": is_cav_seq[:, step_i].bool()}
            entrant_hist.append(entrant)
            teacher = None if cav_teacher is None else cav_teacher[:, step_i + 1]
            action_id = None
            action_command = None
            if action_timeline is not None:
                ctrl = torch.as_tensor(action_timeline["controller"], device=pres0.device,
                                       dtype=torch.long)
                if ctrl.ndim != 3 or ctrl.shape[0] != pres0.shape[0] \
                        or ctrl.shape[2] != pres0.shape[1]:
                    raise ValueError("action timeline controller must be [B,control_steps,K]")
                control_index = min(int(((k0 + step_i) * float(self.physics.dt_s)) // timeline_dt),
                                    ctrl.shape[1] - 1)
                action_id = ctrl[:, control_index]
                # Timeline ids are deliberately the compact action-contract
                # indices (native/damping/PI/direct); map them to the legacy
                # black-box embedding vocabulary without changing old labels.
                action_id = action_id.new_tensor((1, 3, 2, 6))[action_id.clamp(0, 3)]
                action_id = action_id.expand_as(pres0)
                action_command = torch.stack((
                    torch.as_tensor(action_timeline["acceleration_mps2"], device=pres0.device,
                                    dtype=current["v_mps"].dtype)[:, control_index],
                    torch.as_tensor(action_timeline["target_speed_mps"], device=pres0.device,
                                    dtype=current["v_mps"].dtype)[:, control_index]), dim=-1)
            out = self.step(current, k_steps=k0 + step_i, history=history,
                            history_mask=history_mask, cav_teacher=teacher,
                            driver_history=driver_history if step_i == 0 else None,
                            driver_history_mask=driver_history_mask if step_i == 0 else None,
                            driver_defaults=driver_defaults if step_i == 0 else None,
                             driver_params=fixed_driver_params,
                             driver_embedding=fixed_driver_embedding,
                             lane_age=lane_age,
                            cav_controller=cav_controller,
                            controller_id=controller_id, action_id=action_id,
                            action_command=action_command, projection=projection)
            active = out.get("projection_active")
            if active is not None:
                for key, value in active.items():
                    projection_totals[key] = projection_totals.get(key, 0.0) + float(value)
            if self.rollout_mode in ("driver_params", "hybrid") \
                    and fixed_driver_params is None:
                # Reconstruct the differentiable dict from the encoded values once;
                # the same driver identity/parameters persist through the rollout.
                fixed_driver_params = {name: out["driver_parameters"][..., i]
                                       for i, name in enumerate(PARAMETER_NAMES)}
                encoded_parameters = out["driver_parameters"]
            if fixed_driver_embedding is None and "driver_embedding" in out:
                # Keep the inferred driver identity fixed across the rollout for
                # both the history lane head and hybrid residual conditioning.
                fixed_driver_embedding = out["driver_embedding"]
            if self.rollout_mode == "hybrid":
                mag = out["hdv_mask"].to(out["accel_mps2"].dtype)
                phys_step = (out["physics_accel_mps2"].abs() * mag).sum()
                res_step = (out["accel_delta_mps2"].abs() * mag).sum()
                physics_magnitude = (phys_step if physics_magnitude is None
                                     else physics_magnitude + phys_step)
                residual_magnitude = (res_step if residual_magnitude is None
                                      else residual_magnitude + res_step)
                residual_count += int(mag.sum())
            nxt_present = out["next_present"]
            if present_seq is not None:
                nxt_present = nxt_present & present_seq[:, step_i + 1]
            if junction_seq is not None:
                # A junction internal-lane transit is not a departure: carry the
                # slot through it even if the recorded presence dropped that step.
                nxt_present = nxt_present | junction_seq[:, step_i + 1].bool()
            if noise_on:
                # B4: perturb only predicted (non-CAV) movers.  No draw happens
                # when both stds are 0, so the default stays byte-identical.
                noise_present = current.get("present")
                if noise_present is None:
                    noise_present = _present(current, _state_kin(current))
                noise_cav = current.get("is_cav")
                learned = noise_present.bool() if noise_cav is None else (
                    noise_present.bool() & ~noise_cav.bool())
                learned = learned.to(out["next_s_m"].dtype)
                if float(state_noise_std_m) > 0.0:
                    out["next_s_m"] = out["next_s_m"] + torch.randn_like(
                        out["next_s_m"]) * float(state_noise_std_m) * learned
                if float(state_noise_std_mps) > 0.0:
                    out["next_v_mps"] = (out["next_v_mps"] + torch.randn_like(
                        out["next_v_mps"]) * float(state_noise_std_mps)
                        * learned).clamp_min(0.0)
            positions.append(out["next_s_m"])
            speeds.append(out["next_v_mps"])
            lanes.append(out["next_lane"])
            masks.append(nxt_present)
            accels.append(out["accel_mps2"])
            lane_logits.append(out["lane_logits"])
            # Monitoring uses raw g*delta_a BEFORE either residual/plant bound.
            # Pure learned subclasses lack this physics-residual quantity.
            raw_residuals.append(out.get("raw_residual_mps2", torch.zeros_like(out["accel_mps2"])))
            probs = out.get("lane_probabilities")
            if probs is None:
                probs = torch.softmax(out["lane_logits"], dim=-1)
            stay = (probs[..., 0] if self.lane_change_head is not None else
                    probs.gather(-1, current["lane"].clamp(0, 2).unsqueeze(-1)).squeeze(-1))
            change_probabilities.append(1.0 - stay)
            if self.lane_change_head is not None:
                lane_probabilities.append(out["lane_probabilities"])
            if self.lane_change_head is not None:
                changed = out["next_lane"] != current["lane"]
                lane_age = torch.where(changed, torch.zeros_like(lane_age), lane_age + 1)
            current = {**current, "s_m": out["next_s_m"], "v_mps": out["next_v_mps"],
                       "a_mps2": out["accel_mps2"], "lane": out["next_lane"],
                       "present": nxt_present}
            if self.lane_change_head is not None:
                current["lane_age"] = lane_age
            current.pop("veh_kin", None)
            kin = _state_kin(current)
            gap, _ = self._lead(kin, nxt_present)
            feat = history_step_features(kin, gap)
            history = torch.cat((history[:, :, 1:], feat.unsqueeze(2)), dim=2)
            history_mask = torch.cat((history_mask[:, :, 1:],
                                      nxt_present.unsqueeze(-1)), dim=2)
            if truncation_steps and (step_i + 1) % int(truncation_steps) == 0:
                current = {key: (value.detach() if torch.is_tensor(value) else value)
                           for key, value in current.items()}
                history, history_mask = history.detach(), history_mask
        entrant_seq = (torch.stack(entrant_hist, 1) if entrant_hist
                       else torch.zeros(pres0.shape[0], 0, pres0.shape[1],
                                        dtype=torch.bool, device=pres0.device))
        sections = (torch.stack([_section_ids(pos, msk)
                                 for pos, msk in zip(positions, masks)], 1)
                    if positions else torch.zeros(pres0.shape[0], 0, pres0.shape[1],
                                                  dtype=torch.long, device=pres0.device))
        result = {"positions": torch.stack(positions, 1),
                  "raw_residual_mps2": torch.stack(raw_residuals, 1),
                  "lane_change_probabilities": torch.stack(change_probabilities, 1),
                  "speeds": torch.stack(speeds, 1), "lanes": torch.stack(lanes, 1),
                  "masks": torch.stack(masks, 1), "accel": torch.stack(accels, 1),
                  "lane_logits": torch.stack(lane_logits, 1),
                  "entrant": entrant, "entrant_seq": entrant_seq, "sections": sections,
                  "state": current, "history": history, "history_mask": history_mask}
        if projection is not None:
            result["projection_stats"] = projection_totals
        if encoded_parameters is not None:
            result["driver_parameters"] = encoded_parameters
        if fixed_driver_embedding is not None:
            result["driver_embedding"] = fixed_driver_embedding
        if self.rollout_mode == "hybrid":
            result["physics_accel_magnitude"] = (
                physics_magnitude if physics_magnitude is not None
                else torch.zeros((), dtype=kin0.dtype, device=kin0.device))
            result["residual_magnitude"] = (
                residual_magnitude if residual_magnitude is not None
                else torch.zeros((), dtype=kin0.dtype, device=kin0.device))
            result["residual_count"] = int(residual_count)
        if lane_probabilities:
            result["lane_probabilities"] = torch.stack(lane_probabilities, 1)
        return result
