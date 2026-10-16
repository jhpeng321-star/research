"""Batched, device-agnostic IDM/MOBIL physics for world-model rollouts.

The state uses the frozen model-frame longitudinal coordinate ``s_down``
(``features.geometry.s_down_of`` = physical s - lane offset), which increases
downstream.  Tensor shapes are ``[B, K]`` unless noted otherwise.  The module
does not own CAV control: CAV acceleration and lane updates are supplied by the
caller, while human vehicles use IDM/MOBIL.

This is the batched torch mirror of the numpy ``rollout2`` baseline
(``rollout2.baseline.V2GCore`` / ``rollout2.step.step_once``) for the
zero-residual, no-arrival physics path.  In particular:

* the per-lane segment speed limits (``v0``) come from the model-frame geometry,
  so ``s_down`` positions must be used with the ``sumo_fidelity`` parameter
  view (``baseline.model_frame_params``);
* the accel-lane end obstacle and the mandatory-merge zone are clamped/derived
  from the frozen ``model_frame_params`` contract in the SAME shared ``s_down``
  axis as ``s`` (using the physical 378.66/454.1 here was the frame bug that
  silently disabled the merge and the end obstacle);
* the default exit geometry is the lane-specific model-frame lane end
  (``MODEL_FRAME_LANE_LENGTHS``); the legacy scalar ``downstream_exit_m`` is
  only a fallback for a lane count that does not match the supplied geometry.
* with ``bumper_gap=True`` IDM uses the bumper gap of the numpy contract
  (``_leaders_vector``); the default measures centre distance, matching the
  synthetic module tests.  Real-parity callers supply lengths and enable it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

GAP_MAX = 250.0

#: Frozen model-frame geometry derived from
#: ``rollout2.baseline.model_frame_params(None, sumo_idm=True)``:
#: physical lane length - lane offset.  Mainline ends are 3199.0 - 2645.67 and
#: 3197.55 - 2644.22; the accel lane end is 454.1 - 378.66.  The merge zone is
#: the physical 378.66..454.1 shifted by -off[2]; lane offsets are zeroed.
MODEL_FRAME_LANE_LENGTHS = (553.33, 553.33, 75.44)
MODEL_FRAME_LANE_OFFSETS = (0.0, 0.0, 0.0)
MODEL_FRAME_MERGE_START_M = 0.0
MODEL_FRAME_MERGE_END_M = 75.44


@dataclass
class PhysicsState:
    """Fixed-capacity batched vehicle state (model ``s_down`` frame)."""

    s: Tensor
    v: Tensor
    lane: Tensor
    active: Tensor
    cav: Tensor

    def validate(self) -> None:
        if self.s.ndim != 2 or self.v.shape != self.s.shape:
            raise ValueError("s and v must both have shape [B, K]")
        if self.lane.shape != self.s.shape:
            raise ValueError("lane must have shape [B, K]")
        if self.active.shape != self.s.shape or self.cav.shape != self.s.shape:
            raise ValueError("active and cav must have shape [B, K]")
        if self.lane.dtype not in (torch.int32, torch.int64):
            raise ValueError("lane must be an integer tensor")
        if self.active.dtype != torch.bool or self.cav.dtype != torch.bool:
            raise ValueError("active and cav must be bool tensors")


@dataclass
class IDMParameters:
    """Per-slot IDM/MOBIL parameters, broadcastable to ``[B, K]``."""

    v0: Tensor
    accel: Tensor
    decel: Tensor
    b_comf: Tensor
    tau: Tensor
    min_gap: Tensor
    length: Tensor
    politeness: Tensor
    safe_decel: Tensor
    delta: Tensor

    @classmethod
    def constant(
        cls,
        batch_size: int,
        vehicle_count: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        v0: float = 31.29,
        accel: float = 2.6,
        decel: float = 4.5,
        b_comf: float = 4.5,
        tau: float = 1.1,
        min_gap: float = 2.5,
        length: float = 5.0,
        politeness: float = 0.2,
        safe_decel: float = 4.0,
        delta: int = 4,
    ) -> "IDMParameters":
        shape = (batch_size, vehicle_count)

        def full(value: float) -> Tensor:
            return torch.full(shape, value, device=device, dtype=dtype)

        return cls(
            v0=full(v0),
            accel=full(accel),
            decel=full(decel),
            b_comf=full(b_comf),
            tau=full(tau),
            min_gap=full(min_gap),
            length=full(length),
            politeness=full(politeness),
            safe_decel=full(safe_decel),
            delta=full(delta),
        )

    def broadcast(self, shape: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> "IDMParameters":
        def expand(value: Tensor) -> Tensor:
            return torch.as_tensor(value, device=device, dtype=dtype).expand(shape)

        return IDMParameters(
            v0=expand(self.v0),
            accel=expand(self.accel),
            decel=expand(self.decel),
            b_comf=expand(self.b_comf),
            tau=expand(self.tau),
            min_gap=expand(self.min_gap),
            length=expand(self.length),
            politeness=expand(self.politeness),
            safe_decel=expand(self.safe_decel),
            delta=expand(self.delta),
        )


def parameters_from_vtypes(
    vtypes: Tensor,
    *,
    v0_i: Optional[Tensor] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> IDMParameters:
    """Build per-vehicle parameters from normal/aggressive/cautious vTypes.

    ``vtypes`` uses ids 0=normal, 1=aggressive, and 2=cautious.  A supplied
    ``v0_i`` replaces only the desired speed, which is useful for inferred
    driver-specific free speeds without changing the calibrated vType means.
    """

    vtypes = torch.as_tensor(vtypes, device=device, dtype=torch.long)
    if vtypes.ndim != 2:
        raise ValueError("vtypes must have shape [B, K]")
    table = torch.tensor(
        [
            [31.29, 2.52, 4.2, 4.2, 1.165, 2.6, 5.0, 0.5, 4.0, 4],
            [34.0, 3.0, 5.0, 4.5, 0.95, 1.5, 5.0, 0.1, 4.5, 4],
            [28.0, 2.0, 4.0, 3.0, 1.5, 3.5, 5.0, 0.3, 3.5, 4],
        ],
        device=vtypes.device,
        dtype=dtype,
    )
    if torch.any((vtypes < 0) | (vtypes >= table.shape[0])):
        raise ValueError("vtypes contains an unknown vType id")
    values = table[vtypes]
    if v0_i is not None:
        values = values.clone()
        values[..., 0] = torch.as_tensor(v0_i, device=vtypes.device, dtype=dtype).expand(vtypes.shape)
    return IDMParameters(
        v0=values[..., 0],
        accel=values[..., 1],
        decel=values[..., 2],
        b_comf=values[..., 3],
        tau=values[..., 4],
        min_gap=values[..., 5],
        length=values[..., 6],
        politeness=values[..., 7],
        safe_decel=values[..., 8],
        delta=values[..., 9],
    )


@dataclass(frozen=True)
class BatchedPhysicsConfig:
    """Numerical and model-frame geometry contract.

    ``seg_starts``/``seg_limits`` are per-lane 1-D tensors of the model-frame
    (``s_down``) segment boundaries and speed limits; ``lane_lengths`` are the
    model-frame lane ends.  When supplied, ``v0`` is the containing segment
    limit for every active vehicle.  ``merge_*`` holds the mandatory-merge
    geometry and the (configurable) longitudinal gap-acceptance thresholds;
    calibrated values are supplied by the caller.

    The defaults are the frozen ``model_frame_params`` contract: per-lane
    model-frame lane ends (mainline 553.33 m, accel 75.44 m), zero lane offsets
    and the merge zone ``0.0..75.44`` (physical 378.66..454.1 shifted by the
    accel-lane offset).  This keeps ``s_down`` inputs in one frame.
    """

    dt: float = 0.5
    lane_count: int = 3
    speed_cap_factor: float = 1.2
    free_gap_m: float = GAP_MAX
    #: Fallback scalar exit, used only when the supplied lane geometry does not
    #: match ``lane_count``.  ``inf`` = open road; matching geometry uses the
    #: model-frame per-lane ends below.
    downstream_exit_m: float = float("inf")
    entry_position_m: float = 0.0
    entry_speed_mps: float = 13.9
    entry_lane: int = 1
    # model-frame geometry (model frame = physical s - lane offset)
    seg_starts: Optional[Tuple[Tensor, ...]] = None
    seg_limits: Optional[Tuple[Tensor, ...]] = None
    lane_lengths: Optional[Tensor] = field(
        default_factory=lambda: torch.tensor(
            MODEL_FRAME_LANE_LENGTHS, dtype=torch.float64))
    lane_offsets: Optional[Tensor] = field(
        default_factory=lambda: torch.tensor(
            MODEL_FRAME_LANE_OFFSETS, dtype=torch.float64))
    physical_exits: bool = False
    # dynamics clip (None -> use IDMParameters accel/decel)
    a_clip_min: Optional[float] = None
    a_clip_max: Optional[float] = None
    #: Centre distance by default (synthetic tests); ``True`` uses the numpy
    #: bumper-gap contract and requires real vehicle lengths.
    bumper_gap: bool = False
    # lane-change stage (discretionary); off for the longitudinal parity path
    lane_changes_enabled: bool = True
    # mandatory merge (lane merge_lane -> merge_target_lane); model-frame zone
    merge_lane: int = 2
    merge_target_lane: int = 1
    merge_start_m: float = MODEL_FRAME_MERGE_START_M
    merge_end_m: float = MODEL_FRAME_MERGE_END_M
    merge_leader_gap_m: float = 0.3    # frozen merge.gap_floor_m
    merge_follower_gap_m: float = 2.5  # frozen idm.s0_m (sumo vType mean)
    merge_b_safe_mps2: float = 4.0
    accel_lane_end_clamp: bool = True
    mobil_threshold_mps2: float = 0.1
    keep_right_bias_mps2: float = 0.3

    def clip_bounds(self, accel: Tensor, decel: Tensor) -> Tuple[Tensor, Tensor]:
        lo = accel.new_full((), self.a_clip_min) if self.a_clip_min is not None else -decel
        hi = accel.new_full((), self.a_clip_max) if self.a_clip_max is not None else accel
        return lo, hi


@dataclass
class StepOutput:
    """Result of one differentiable physics step."""

    state: PhysicsState
    acceleration: Tensor
    lane_logits: Tensor
    lane_change: Tensor
    exited: Tensor
    entered: Tensor
    merge_blocked: Tensor
    leader_index: Tensor


def _neighbors(
    s: Tensor,
    v: Tensor,
    lane: Tensor,
    active: Tensor,
    length: Tensor,
    query_lane: Tensor,
    *,
    ahead: bool,
    free_gap_m: float,
    bumper: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return nearest same-lane neighbour state for every query slot.

    With ``bumper=True`` ``gap`` is the bumper gap (centre distance minus the
    leader's length), matching ``partA_v2g.engine._leaders_vector``.  With the
    default ``bumper=False`` it is the centre distance, matching the synthetic
    module tests.  For a fixed query, minimizing the centre distance is the
    same ordering as minimizing the bumper gap.
    """

    same_lane = lane[:, None, :] == query_lane[:, :, None]
    active_pair = active[:, None, :] & active[:, :, None]
    if ahead:
        distance = s[:, None, :] - s[:, :, None]
        direction = distance > 0.0
    else:
        distance = s[:, :, None] - s[:, None, :]
        direction = distance > 0.0
    valid = same_lane & active_pair & direction
    inf = torch.full_like(distance, float("inf"))
    nearest = torch.where(valid, distance, inf)
    gap, index = torch.min(nearest, dim=-1)
    has_neighbor = torch.isfinite(gap)
    index = torch.where(has_neighbor, index, torch.zeros_like(index))
    neighbor_length = torch.gather(length, 1, index)
    neighbor_v = torch.gather(v, 1, index)
    if bumper:
        gap = torch.where(has_neighbor, gap - neighbor_length, torch.full_like(gap, free_gap_m))
    else:
        gap = torch.where(has_neighbor, gap, torch.full_like(gap, free_gap_m))
    neighbor_v = torch.where(has_neighbor, neighbor_v, v)
    return gap, neighbor_v, index, has_neighbor


def _idm_acceleration(v: Tensor, leader_v: Tensor, gap: Tensor, v0: Tensor, p: IDMParameters) -> Tensor:
    """Textbook IDM, math-identical to ``partA_v2g.engine.idm_accel``."""

    gap = gap.clamp_min(1e-6)
    v0 = v0.clamp_min(1e-6)
    closing = v - leader_v
    s_star = p.min_gap + (v * p.tau + v * closing / (2.0 * torch.sqrt(p.accel * p.b_comf).clamp_min(1e-6))).clamp_min(0.0)
    return p.accel * (1.0 - (v / v0).pow(p.delta) - (s_star / gap).pow(2))


def _seg_limit(s: Tensor, lane: Tensor, active: Tensor, config: BatchedPhysicsConfig, fallback: Tensor) -> Tensor:
    """Per-vehicle desired speed from the model-frame segment table."""

    if config.seg_starts is None or config.seg_limits is None:
        return fallback
    out = fallback.clone()
    for li in range(config.lane_count):
        m = active & (lane == li)
        if not bool(m.any()):
            continue
        starts = config.seg_starts[li].to(device=s.device, dtype=s.dtype).contiguous()
        lims = config.seg_limits[li].to(device=s.device, dtype=s.dtype)
        values = s[m].contiguous()
        idx = torch.searchsorted(starts, values, right=True) - 1
        idx = idx.clamp(0, lims.shape[0] - 1)
        out = out.clone()
        out[m] = lims[idx]
    return out


def _project_gaps(
    s: Tensor, v: Tensor, lane: Tensor, active: Tensor, p: IDMParameters, lane_count: int, dt: float
) -> Tuple[Tensor, Tensor]:
    """Project positions back to a non-overlapping lane ordering.

    Mirror of ``rollout2.validity.project_same_lane_gaps``: a follower that
    would overlap its leader is placed at the contact limit and its speed is
    reduced so the implied displacement stays feasible.
    """

    result_s = s
    result_v = v
    for lane_id in range(lane_count):
        in_lane = active & (lane == lane_id)
        keyed = torch.where(in_lane, s, torch.full_like(s, -float("inf")))
        _, order = torch.sort(keyed, dim=1, descending=True)
        valid = torch.gather(in_lane, 1, order)
        ordered_s = torch.gather(s, 1, order)
        ordered_v = torch.gather(v, 1, order)
        ordered_length = torch.gather(p.length, 1, order)
        # Functional (no in-place) so projection stays inside the autograd graph.
        safe_s = [ordered_s[:, 0]]
        safe_v = [ordered_v[:, 0]]
        for position in range(1, s.shape[1]):
            min_gap = 0.5 * (ordered_length[:, position - 1] + ordered_length[:, position])
            limit = safe_s[position - 1] - min_gap
            old_s = ordered_s[:, position]
            old_v = ordered_v[:, position]
            overlap = valid[:, position] & (old_s > limit)
            new_s = torch.where(overlap, limit, old_s)
            vmax = ((limit - old_s) / dt + old_v).clamp_min(0.0)
            new_v = torch.where(overlap, torch.minimum(old_v, vmax), old_v)
            safe_s.append(torch.where(valid[:, position], new_s, old_s))
            safe_v.append(torch.where(valid[:, position], new_v, old_v))
        stacked_s = torch.stack(safe_s, dim=1)
        stacked_v = torch.stack(safe_v, dim=1)
        projected_s = torch.zeros_like(s).scatter(1, order, stacked_s)
        projected_v = torch.zeros_like(v).scatter(1, order, stacked_v)
        result_s = torch.where(in_lane, projected_s, result_s)
        result_v = torch.where(in_lane, projected_v, result_v)
    return result_s, result_v


def _sample_entries(rate: Tensor, free: Tensor, seed: int) -> Tensor:
    """Draw a deterministic per-slot entrance mask over the currently free slots.

    ``rate`` is the per-world entry intensity for this step (veh/s, broadcast to
    ``[B]``).  Each free slot admits one entry with probability
    ``clamp(rate, 0, 1)``; a rate at or above 1 therefore always fills every
    free slot.  The draw uses one seeded generator, so repeated batches are
    deterministic and ``rate == 0`` never enters.  This mirrors the seeded
    common-random-numbers arrival process of ``closedloop.rollout_entries`` at
    the step granularity while staying CPU-only and dependency-free.
    """

    generator = torch.Generator(device=free.device)
    generator.manual_seed(int(seed))
    prob = rate.clamp(0.0, 1.0).unsqueeze(-1).expand_as(free)
    uniform = torch.rand(free.shape, device=free.device, dtype=prob.dtype, generator=generator)
    return free & (uniform < prob)


def step(
    state: PhysicsState,
    params: IDMParameters,
    *,
    config: BatchedPhysicsConfig = BatchedPhysicsConfig(),
    learned_lane_logits: Optional[Tensor] = None,
    cav_acceleration: Optional[Tensor] = None,
    cav_lane: Optional[Tensor] = None,
    observed_entry_rate: Optional[Tensor] = None,
    entry_seed: int = 0,
    step_index: int = 0,
) -> StepOutput:
    """Advance all worlds by one 0.5 s step (D1 trapezoidal integration)."""

    # The reference is the float64 numpy ``rollout2`` core (``_world_f64``), so
    # the physics is evaluated in float64 regardless of the input dtype.  This
    # keeps a long-horizon rollout within the parity tolerance; casts are
    # differentiable, so gradients still reach the caller's state.
    if state.s.dtype != torch.float64:
        state = PhysicsState(state.s.to(torch.float64), state.v.to(torch.float64),
                             state.lane, state.active, state.cav)
    state.validate()
    batch_size, vehicle_count = state.s.shape
    device, dtype = state.s.device, state.s.dtype
    p = params.broadcast((batch_size, vehicle_count), device, dtype)
    if torch.any(state.v < 0):
        raise ValueError("state speed must be non-negative")
    if torch.any(state.cav) and cav_acceleration is None:
        raise ValueError("CAV slots require externally supplied acceleration")

    v0 = _seg_limit(state.s, state.lane, state.active, config, p.v0)
    current_gap, current_leader_v, leader_index, _ = _neighbors(
        state.s, state.v, state.lane, state.active, p.length, state.lane,
        ahead=True, free_gap_m=config.free_gap_m, bumper=config.bumper_gap,
    )
    baseline_accel = _idm_acceleration(state.v, current_leader_v, current_gap, v0, p)
    # accel-lane end obstacle (engine.py:639-643); model-frame s1z.  The
    # engine passes ``dv = v`` (a stopped obstacle), i.e. ``leader_v = 0``.
    accel_end = state.active & (state.lane == config.merge_lane)
    gap_end = (config.merge_end_m - state.s).clamp_min(0.3)
    obstacle_accel = _idm_acceleration(state.v, torch.zeros_like(state.v), gap_end, v0, p)
    baseline_accel = torch.where(accel_end, torch.minimum(baseline_accel, obstacle_accel), baseline_accel)

    lo, hi = config.clip_bounds(p.accel, p.decel)
    baseline_accel = baseline_accel.clamp(lo, hi)

    # ---- lane-change stage -------------------------------------------------
    auto_lane = state.lane
    lane_logits = torch.zeros((batch_size, vehicle_count, 2), device=device, dtype=dtype)
    merge_blocked = torch.zeros((batch_size, vehicle_count), dtype=torch.bool, device=device)
    if config.lane_changes_enabled and batch_size > 0:
        target_lanes = torch.stack((state.lane - 1, state.lane + 1), dim=-1)
        target_valid = (target_lanes >= 0) & (target_lanes < config.lane_count)

        def expected_accel(query_lane: Tensor) -> Tensor:
            l_gap, l_v, _, _ = _neighbors(
                state.s, state.v, state.lane, state.active, p.length, query_lane,
                ahead=True, free_gap_m=config.free_gap_m,
            )
            return _idm_acceleration(state.v, l_v, l_gap, v0, p)

        left = expected_accel(target_lanes[..., 0])
        right = expected_accel(target_lanes[..., 1])
        incentive = torch.stack((left, right), dim=-1) - baseline_accel.unsqueeze(-1)
        lane_logits = incentive - config.mobil_threshold_mps2
        if learned_lane_logits is not None:
            if learned_lane_logits.shape != lane_logits.shape:
                raise ValueError("learned_lane_logits must have shape [B, K, 2]")
            lane_logits = lane_logits + learned_lane_logits
        eligible = target_valid & (lane_logits > 0.0)
        masked = torch.where(eligible, lane_logits, torch.full_like(lane_logits, -float("inf")))
        best_side = torch.argmax(masked, dim=-1)
        chosen = torch.gather(target_lanes, -1, best_side.unsqueeze(-1)).squeeze(-1)
        chosen = torch.where(torch.any(eligible, dim=-1), chosen, auto_lane)
        auto_lane = chosen

    # mandatory merge (lane merge_lane -> merge_target_lane) with configurable
    # gap acceptance; measured against the CURRENT lane-1 occupancy.
    mandatory = (
        state.active & ~state.cav & (state.lane == config.merge_lane)
        & (state.s >= config.merge_start_m) & (state.s <= config.merge_end_m)
    )
    if bool(mandatory.any()):
        lead_gap, _, _, _ = _neighbors(
            state.s, state.v, state.lane, state.active, p.length,
            torch.full_like(state.lane, config.merge_target_lane),
            ahead=True, free_gap_m=config.free_gap_m,
        )
        follow_gap, _, _, _ = _neighbors(
            state.s, state.v, state.lane, state.active, p.length,
            torch.full_like(state.lane, config.merge_target_lane),
            ahead=False, free_gap_m=config.free_gap_m,
        )
        gap_ok = (lead_gap >= config.merge_leader_gap_m) & (follow_gap >= config.merge_follower_gap_m)
        accepted = mandatory & gap_ok
        auto_lane = torch.where(accepted, torch.full_like(auto_lane, config.merge_target_lane), auto_lane)
        merge_blocked = mandatory & ~gap_ok
    if config.lane_changes_enabled:
        auto_lane = auto_lane.clamp(0, config.lane_count - 1)
        auto_lane = torch.where(state.active, auto_lane, state.lane)
    else:
        auto_lane = state.lane
    chosen_lane = auto_lane
    if cav_lane is not None:
        cav_lane = torch.as_tensor(cav_lane, device=device, dtype=state.lane.dtype).expand_as(state.lane)
        chosen_lane = torch.where(state.cav, cav_lane, chosen_lane)
    chosen_lane = chosen_lane.clamp(0, config.lane_count - 1)
    chosen_lane = torch.where(state.active, chosen_lane, state.lane)

    lane_change = (chosen_lane != state.lane).to(dtype)

    acceleration = baseline_accel
    if cav_acceleration is not None:
        cav_acceleration = torch.as_tensor(cav_acceleration, device=device, dtype=dtype).expand_as(acceleration)
        acceleration = torch.where(state.cav, cav_acceleration, acceleration)
    acceleration = acceleration.clamp(lo, hi)

    next_v = (state.v + acceleration * config.dt).clamp_min(0.0)
    next_s = state.s + 0.5 * (state.v + next_v) * config.dt
    # accel-lane end guard (engine.py:662-669): hold at the model-frame s1z,
    # but only while the vehicle actually remains on the merge lane.
    if config.accel_lane_end_clamp:
        past = (state.active & (state.lane == config.merge_lane)
                & (chosen_lane == config.merge_lane) & (next_s > config.merge_end_m))
        next_s = torch.where(past, torch.full_like(next_s, config.merge_end_m), next_s)
        next_v = torch.where(past, torch.zeros_like(next_v), next_v)

    # projection first (validity.py), then the road-exit mask, exactly as the
    # numpy step order.
    next_s, next_v = _project_gaps(next_s, next_v, chosen_lane, state.active, p, config.lane_count, config.dt)

    # The lane-specific model-frame geometry applies only when it matches the
    # configured lane count; otherwise fall back to the scalar exit (open road
    # when ``downstream_exit_m`` is infinite), e.g. the single-lane synthetic
    # parity worlds.
    lengths = config.lane_lengths
    if lengths is not None and lengths.numel() == config.lane_count:
        lengths = lengths.to(device=device, dtype=dtype)
        if config.physical_exits and config.lane_offsets is not None:
            offs = config.lane_offsets.to(device=device, dtype=dtype)
            limits = lengths - offs
        else:
            limits = lengths
        lim = torch.gather(limits.expand(batch_size, -1), 1, chosen_lane.clamp(0, config.lane_count - 1))
    else:
        lim = torch.full_like(next_s, config.downstream_exit_m)
    exited = state.active & (next_s > lim)
    next_active = state.active & ~exited

    entered = torch.zeros_like(next_active)
    if observed_entry_rate is not None:
        rate = torch.as_tensor(observed_entry_rate, device=device, dtype=dtype).expand(batch_size)
        free = ~next_active
        entered = _sample_entries(rate, free, entry_seed + step_index)
        next_active = next_active | entered
        next_s = torch.where(entered, torch.full_like(next_s, config.entry_position_m), next_s)
        next_v = torch.where(entered, torch.full_like(next_v, config.entry_speed_mps), next_v)
        chosen_lane = torch.where(entered, torch.full_like(chosen_lane, config.entry_lane), chosen_lane)
        next_s, next_v = _project_gaps(next_s, next_v, chosen_lane, next_active, p, config.lane_count, config.dt)

    next_lane = torch.where(next_active, chosen_lane, torch.full_like(chosen_lane, -1))
    next_state = PhysicsState(next_s, next_v, next_lane, next_active, state.cav & next_active)
    return StepOutput(next_state, acceleration, lane_logits, lane_change, exited, entered, merge_blocked, leader_index)


def rollout(
    state: PhysicsState,
    params: IDMParameters,
    steps: int,
    *,
    config: BatchedPhysicsConfig = BatchedPhysicsConfig(),
    learned_lane_logits: Optional[Tensor] = None,
    cav_acceleration: Optional[Sequence[Tensor]] = None,
    cav_lane: Optional[Sequence[Tensor]] = None,
    observed_entry_rate: Optional[Tensor] = None,
    entry_seed: int = 0,
) -> Tuple[Tuple[PhysicsState, ...], Tuple[StepOutput, ...]]:
    """Run a fixed number of steps while retaining every state and diagnostic."""

    states = [state]
    outputs = []
    current = state
    for tick in range(steps):
        output = step(
            current,
            params,
            config=config,
            learned_lane_logits=learned_lane_logits,
            cav_acceleration=None if cav_acceleration is None else cav_acceleration[tick],
            cav_lane=None if cav_lane is None else cav_lane[tick],
            observed_entry_rate=observed_entry_rate,
            entry_seed=entry_seed,
            step_index=tick,
        )
        outputs.append(output)
        current = output.state
        states.append(current)
    return tuple(states), tuple(outputs)
