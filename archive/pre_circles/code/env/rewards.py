"""All reward functions in one place — edit here to change training incentives.

Two reward families:

1. Shepherd / Flat-MARL (human-centric), `compute_shepherd_reward`:

       R = w_e * R_eff + w_s * R_safe + w_t * R_stab + w_f * R_fair + w_c * C_ctrl

   Term            | Helper          | Meaning
   ----------------|-----------------|---------------------------------------------
   R_eff           | `_efficiency`   | F_flow = clip(sys_norm,0,2) - 1  (no ego speed)
   R_safe          | `_safety`       | ego (brake/leader-TTC/side-TTC/collision) - 2 x violating-neighbor share, bounded [-10, 0]
   R_stab          | `_stability`    | ego accel + ego jerk (no neighbor term)
   R_fair          | `_fairness`     | clip((r_ref - mean HDV delay)/r_ref, -1, +1) vs All-HDV baseline
   C_ctrl          | `_control_cost` | 0 if mode == 0; else need-gated in [-1, -c_min]
                     (default: health = clip(upstream s_norm, 0, 1); calm traffic
                     pays full -1, jammed traffic pays closer to -c_min)
   C_mask          | (inline)        | -1 if hard health mask remapped this step's
                     selected mode (weight ``mask_penalty``); 0 otherwise. Breaks
                     coast vs masked-flow_stab reward aliasing.
   C_inelig        | (inline)        | -1 if a non-coast mode was selected but
                     eligibility remapped it to coast (wrong lane / no trigger;
                     weight ``ineligible_penalty``). Breaks vacate-off-drop ↔
                     coast aliasing.
   B_mode_use      | (inline)        | +1 when any non-coast mode executes
                     (eligible, not remapped). Weight ``mode_use_bonus``
                     (default 0.1 ≪ |F_flow|≤1). Paid every such step so a
                     multi-step window can still be preferred *when* it helps
                     flow; too small to make the mode automatic if it hurts
                     F_flow.
   B_fair_win      | `_fair_win`     | clip((gap_ref - |gap|)_+ / scale, 0, 1):
                     pays strict stream-balance wins below the paired no_op
                     ref (mirror of the excess-above-ref fairness penalty);
                     weight ``fair_win``. 0 when gap unknown; ties earn 0.
   C_easy_tax      | `_easy_intervention_tax` | -1 when a non-coast mode is
                     selected while |gap| <= gap_ref (nothing to win, so
                     intervention is pure risk). Soft signal, not a mask;
                     weight ``easy_intervention_tax``. 0 for coast / unknown
                     gap / hard states.

   R_eff is the beat-HDV flow objective only (ego speed removed — it fought
   metering/yield). sys_norm = 0.5 * merge-discharge q_norm + 0.5 *
   upstream-speed s_norm (EMA, `MergeOutflowTracker`), each normalized by
   All-HDV refs (per-density defaults, or per-seed via ``paired_hdv_refs``).
   Matching the baseline scores F_flow = 0; jamming scores -1; beating the
   baseline scores up to +1. Control cost weight is 0 for the flow-only
   retrain (probe: even 0.1 preferred coast over recover). Fairness uses
   per-step getTimeLoss rates over ALL local HDVs via `TimeLossTracker`.
   Need-gated C_ctrl plumbing remains for later re-enable.

2. Selfish (personal), `compute_selfish_reward`: speed + comfort + safety only,
   weighted by ``selfish_reward_weights`` in ``config/defaults.py``.

To plug in a custom reward without editing the environment, pass ``reward_fn=``
to :class:`env.core.TrafficShepherdMultiAgentEnv`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import traci

from config.constants import MODE_NAMES, Mode
from env.features import wave_risk_signal
from env.snapshot import get_active_snapshot
from env.traci_query import (
    leader_ttc,
    merge_arrival_pet,
    percentile_or_zero,
    side_ttc,
    vehicles_within_radius,
)


def _accel_or_none(veh_id: str) -> Optional[float]:
    """Vehicle acceleration; None mirrors the direct getter raising."""
    snap = get_active_snapshot()
    if snap is not None:
        return snap.accel(veh_id)
    try:
        return float(traci.vehicle.getAcceleration(veh_id))
    except Exception:
        return None


class JerkTracker:
    """Track per-vehicle acceleration to compute jerk once per env step."""

    def __init__(self, dt: float = 1.0):
        self.dt = max(float(dt), 1e-6)
        self._prev_a: Dict[str, float] = {}
        self._jerk: Dict[str, float] = {}

    def reset(self) -> None:
        self._prev_a.clear()
        self._jerk.clear()

    def update(self, vehicle_ids: Optional[Iterable[str]] = None) -> None:
        """Compute jerk for this timestep; call once before reward evaluation."""
        snap = get_active_snapshot()
        if vehicle_ids is None:
            if snap is not None:
                vehicle_ids = snap.ordered_ids()
            else:
                try:
                    vehicle_ids = list(traci.vehicle.getIDList())
                except Exception:
                    vehicle_ids = []
        active = set()
        self._jerk = {}
        for vid in vehicle_ids:
            active.add(vid)
            if snap is not None:
                a = snap.accel(vid)
                if a is None:
                    continue
                a = float(a)
            else:
                try:
                    a = float(traci.vehicle.getAcceleration(vid))
                except Exception:
                    continue
            prev = self._prev_a.get(vid)
            if prev is None:
                self._jerk[vid] = 0.0
            else:
                self._jerk[vid] = (a - prev) / self.dt
            self._prev_a[vid] = a
        stale = [vid for vid in self._prev_a if vid not in active]
        for vid in stale:
            del self._prev_a[vid]

    def get(self, veh_id: str) -> float:
        return float(self._jerk.get(veh_id, 0.0))


class MergeOutflowTracker:
    """EMAs of bottleneck health per env step: merge discharge + upstream speed.

    Two complementary signals:
      - discharge: vehicles completing the watched (merge) edge per step.
        Collapses only when the merge deadlocks — the gridlock detector.
      - upstream speed: mean speed on ``speed_edge_id`` (the mainline feeding
        the merge). Drops as soon as a queue grows upstream — the
        queue-growth detector, which pure discharge is blind to (discharge
        stays at capacity while the queue builds).

    Both are smoothed with an EMA; call :meth:`update` once per step before
    reward evaluation.
    """

    def __init__(self, edge_id: str, alpha: float = 0.1, speed_edge_id: Optional[str] = None):
        self.edge_id = edge_id
        self.speed_edge_id = speed_edge_id
        self.alpha = float(alpha)
        self._prev_road: Dict[str, str] = {}
        self._ema = 0.0
        self._v_ema = 0.0
        self._started = False

    def reset(self) -> None:
        self._prev_road.clear()
        self._ema = 0.0
        self._v_ema = 0.0
        self._started = False

    def update(self, vehicle_ids: Optional[Iterable[str]] = None) -> None:
        snap = get_active_snapshot()
        if vehicle_ids is None:
            if snap is not None:
                vehicle_ids = snap.ordered_ids()
            else:
                try:
                    vehicle_ids = list(traci.vehicle.getIDList())
                except Exception:
                    vehicle_ids = []
        active = set()
        count = 0
        speed_sum = 0.0
        speed_n = 0
        for vid in vehicle_ids:
            active.add(vid)
            if snap is not None:
                road = snap.road(vid)
            else:
                try:
                    road = traci.vehicle.getRoadID(vid)
                except Exception:
                    road = None
            prev = self._prev_road.get(vid)
            if prev == self.edge_id and road is not None and road != self.edge_id:
                count += 1
            if road is not None:
                self._prev_road[vid] = road
            if self.speed_edge_id is not None and road == self.speed_edge_id:
                if snap is not None:
                    v = snap.speed(vid)
                else:
                    try:
                        v = traci.vehicle.getSpeed(vid)
                    except Exception:
                        v = None
                if v is not None:
                    speed_sum += float(v)
                    speed_n += 1
        stale = [vid for vid in self._prev_road if vid not in active]
        for vid in stale:
            # Arrived (left the network) from the watched edge also counts.
            if self._prev_road.pop(vid) == self.edge_id:
                count += 1
        v_mean = (speed_sum / speed_n) if speed_n else None
        if not self._started:
            self._ema = float(count)
            self._v_ema = float(v_mean) if v_mean is not None else 0.0
            self._started = True
        else:
            self._ema = self.alpha * count + (1.0 - self.alpha) * self._ema
            if v_mean is not None:
                self._v_ema = self.alpha * v_mean + (1.0 - self.alpha) * self._v_ema

    def rate(self) -> float:
        """Smoothed discharge in veh/step."""
        return float(self._ema)

    def q_norm(self, q_ref: float) -> float:
        """Discharge relative to the All-HDV reference, clipped to [0, 2.0]."""
        return float(min(max(self._ema / max(float(q_ref), 1e-6), 0.0), 2.0))

    def s_norm(self, v_ref: float) -> float:
        """Upstream mean speed relative to the All-HDV reference, [0, 2.0]."""
        if self.speed_edge_id is None:
            return 1.0
        return float(min(max(self._v_ema / max(float(v_ref), 1e-6), 0.0), 2.0))

    def sys_norm(self, q_ref: float, v_ref: float) -> float:
        """Combined bottleneck-health signal: 0.5*q_norm + 0.5*s_norm."""
        return 0.5 * self.q_norm(q_ref) + 0.5 * self.s_norm(v_ref)


class TimeLossTracker:
    """Track per-vehicle cumulative getTimeLoss to compute per-step delay rate."""

    def __init__(self):
        self._prev_time_loss: Dict[str, float] = {}
        self._rates: Dict[str, float] = {}

    def reset(self) -> None:
        self._prev_time_loss.clear()
        self._rates.clear()

    def update(self, vehicle_ids: Optional[Iterable[str]] = None) -> None:
        """Compute per-step time-loss rate; call once before reward evaluation."""
        snap = get_active_snapshot()
        if vehicle_ids is None:
            if snap is not None:
                vehicle_ids = snap.ordered_ids()
            else:
                try:
                    vehicle_ids = list(traci.vehicle.getIDList())
                except Exception:
                    vehicle_ids = []
        active = set()
        self._rates = {}
        for vid in vehicle_ids:
            active.add(vid)
            if snap is not None:
                tl = snap.time_loss(vid)
                if tl is None:
                    continue
                tl = float(tl)
            else:
                try:
                    tl = float(traci.vehicle.getTimeLoss(vid))
                except Exception:
                    continue
            prev = self._prev_time_loss.get(vid)
            if prev is None:
                self._rates[vid] = 0.0
            else:
                self._rates[vid] = max(0.0, tl - prev)
            self._prev_time_loss[vid] = tl
        stale = [vid for vid in self._prev_time_loss if vid not in active]
        for vid in stale:
            del self._prev_time_loss[vid]

    def get(self, veh_id: str) -> float:
        return float(self._rates.get(veh_id, 0.0))


# Local proximity alias bound to this module's default look-ahead, so reward
# term helpers keep their original call sites (e.g. `_local_vehicles(...)`).
def _local_vehicles(cav_id: str, observation_radius: float) -> list:
    """Same-edge vehicles within longitudinal radius of the CAV (excl. self)."""
    return vehicles_within_radius(cav_id, observation_radius, same_edge_only=True)


# ---------------------------------------------------------------------------
# Shepherd / Flat-MARL reward terms
# ---------------------------------------------------------------------------
def _efficiency(
    cav_id: str,
    max_speed: float,
    flow_tracker: Optional[MergeOutflowTracker] = None,
    flow_q_ref: float = 0.9,
    flow_v_ref: float = 27.0,
    w_flow: float = 1.0,
) -> float:
    """Beat-HDV flow objective only (no ego speed).

    ``F_flow = clip(sys_norm, 0, 2.0) - 1.0`` ∈ [-1, +1], where
    ``sys_norm = 0.5 * merge-discharge q_norm + 0.5 * upstream-speed s_norm``.
    Matching All-HDV scores 0; jam → -1; beating the baseline → up to +1.
    ``w_flow`` scales F_flow (default 1.0; legacy ego term removed). Without a
    tracker, returns 0 (neutral). ``cav_id`` / ``max_speed`` kept for call-site
    compatibility.
    """
    del cav_id, max_speed  # unused after dropping ego speed from R_eff
    if flow_tracker is None:
        return 0.0
    f_flow = min(max(flow_tracker.sys_norm(flow_q_ref, flow_v_ref), 0.0), 2.0) - 1.0
    return float(w_flow) * f_flow


def _safety(
    cav_id: str,
    observation_radius: float,
    hard_brake: float = -4.5,
    ttc_threshold: float = 1.5,
    collision_penalty: float = 6.5,
) -> float:
    """Ego penalties plus a bounded, density-independent neighborhood term.

    Ego part: hard brake -1, low leader-TTC -1, low side-TTC (converging
    lane-drop race at the merge) -1, low merge-arrival-PET (arriving at the
    lane-drop together with a converging-lane vehicle) -1, collision
    ``-collision_penalty`` (default 6.5; was 5.0).
    Neighborhood part: -2.0 x (share of neighbors hard-braking or low-TTC),
    bounded [-2, 0] so the same weight means the same thing at any density.
    """
    penalty = 0.0
    ego_accel = _accel_or_none(cav_id)
    if ego_accel is None:
        return 0.0

    if ego_accel < hard_brake:
        penalty -= 1.0
    if leader_ttc(cav_id) < ttc_threshold:
        penalty -= 1.0
    if side_ttc(cav_id) < ttc_threshold:
        penalty -= 1.0
    if merge_arrival_pet(cav_id) < ttc_threshold:
        penalty -= 1.0

    neighbors = _local_vehicles(cav_id, observation_radius)
    if neighbors:
        violating = 0
        for veh_id in neighbors:
            veh_accel = _accel_or_none(veh_id)
            hard_braking = veh_accel is not None and veh_accel < hard_brake
            if hard_braking or leader_ttc(veh_id) < ttc_threshold:
                violating += 1
        penalty -= 2.0 * (violating / len(neighbors))

    try:
        colliding = set(traci.simulation.getCollidingVehiclesIDList())
        if cav_id in colliding:
            penalty -= float(collision_penalty)
    except Exception:
        pass

    return penalty


def _stability(
    cav_id: str,
    observation_radius: float,
    jerk_tracker: Optional[JerkTracker] = None,
    a_ref: float = 3.0,
    jerk_ref: float = 3.0,
) -> float:
    """Penalize ego accel and ego jerk (comfort).

    j = da/dt from JerkTracker. First observation for a vehicle has j=0.
    The old neighbor-jerk RMS term was dropped: it scaled with density, not
    with anything the CAV did, and was pure noise under no-op (-0.1..-0.2/step).
    """
    a = _accel_or_none(cav_id)
    if a is None:
        return 0.0
    a = float(a)

    a_ref = max(a_ref, 1e-6)
    jerk_ref = max(jerk_ref, 1e-6)
    j_ego = jerk_tracker.get(cav_id) if jerk_tracker is not None else 0.0

    # Comfort: smooth speed changes (accel) + smooth accel changes (jerk).
    r = -0.4 * (a / a_ref) ** 2 - 0.6 * (j_ego / jerk_ref) ** 2
    return float(r)


def _percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile (0.0 for empty input — reward semantics)."""
    return percentile_or_zero(values, p)


def _stream_fairness(
    stream_delay_gap: Optional[float],
    gap_scale: float = 0.15,
    gap_ref: float = 0.0,
) -> float:
    """Penalize main↔ramp delay-rate imbalance *above* a paired no_op baseline.

    gap = ema_main_delay_rate - ema_ramp_delay_rate.
    excess = max(|gap| - gap_ref, 0) — no penalty when already as balanced as
    (or better than) the paired no_op seed.
    Returns -clip(excess/scale, 0, 1) ∈ [-1, 0]. Neutral 0 if gap unknown.
    """
    if stream_delay_gap is None:
        return 0.0
    scale = max(float(gap_scale), 1e-6)
    excess = max(abs(float(stream_delay_gap)) - max(float(gap_ref), 0.0), 0.0)
    return -min(excess / scale, 1.0)


def _fairness(
    cav_id: str,
    observation_radius: float,
    max_speed: float,
    cav_ids: Optional[Set[str]] = None,
    time_loss_tracker: Optional[TimeLossTracker] = None,
    r_ref: float = 0.5,
    stream_delay_gap: Optional[float] = None,
    stream_fairness_gap_scale: float = 0.15,
    stream_delay_gap_ref: float = 0.0,
) -> float:
    """Fairness for GOAL_FAIR_SAFE: prefer stream balance; legacy HDV term as tie-break.

    Primary: ``_stream_fairness(gap, scale, gap_ref)``. If gap is unavailable,
    fall back to signed local HDV delay vs ``r_ref`` (paired no_op).
    """
    stream = _stream_fairness(stream_delay_gap, stream_fairness_gap_scale, stream_delay_gap_ref)
    if stream_delay_gap is not None:
        return stream

    cav_ids = cav_ids or set()
    if time_loss_tracker is None:
        return 0.0

    rates = []
    for veh_id in _local_vehicles(cav_id, observation_radius):
        if veh_id in cav_ids:
            continue
        rates.append(time_loss_tracker.get(veh_id))

    if not rates:
        return 0.0

    mean_r = sum(rates) / len(rates)
    r_ref = max(r_ref, 1e-6)
    return min(max((r_ref - mean_r) / r_ref, -1.0), 1.0)


def _delay_budget(
    corridor_delay_rate: Optional[float],
    delay_rate_ref: Optional[float],
    scale: float = 0.20,
) -> float:
    """One-sided paired delay budget: punish only worse-than-no_op delay rate.

    ``corridor_delay_rate`` = mean per-step timeLoss rate on pre∪merge∪ramp.
    ``delay_rate_ref`` = paired no_op mean_time_loss / mean_tt.
    Returns -clip((rate - ref)_+ / scale, 0, 1) ∈ [-1, 0].
    """
    if corridor_delay_rate is None or delay_rate_ref is None:
        return 0.0
    scale = max(float(scale), 1e-6)
    excess = max(float(corridor_delay_rate) - float(delay_rate_ref), 0.0)
    return -min(excess / scale, 1.0)


def _delay_win(
    corridor_delay_rate: Optional[float],
    delay_rate_ref: Optional[float],
    scale: float = 0.20,
) -> float:
    """Modest credit when corridor delay beats paired no_op (do-no-harm complement).

    Returns clip((ref - rate)_+ / scale, 0, 1) ∈ [0, 1]. Zero when worse/equal
    or inputs missing. Keep ``delay_win`` weight ≪ ``delay_budget`` / fairness.
    """
    if corridor_delay_rate is None or delay_rate_ref is None:
        return 0.0
    scale = max(float(scale), 1e-6)
    gain = max(float(delay_rate_ref) - float(corridor_delay_rate), 0.0)
    return min(gain / scale, 1.0)


def _fair_win(
    stream_delay_gap: Optional[float],
    gap_scale: float = 0.40,
    gap_ref: float = 0.0,
) -> float:
    """Pay strict stream-balance improvement *below* the paired no_op ref.

    gain = max(gap_ref - |gap|, 0); returns clip(gain/scale, 0, 1) in [0, 1].
    Mirror image of ``_stream_fairness``: that term punishes excess above the
    ref, this one pays wins below it — ties earn nothing. 0 when gap unknown.
    """
    if stream_delay_gap is None:
        return 0.0
    scale = max(float(gap_scale), 1e-6)
    gain = max(float(gap_ref) - abs(float(stream_delay_gap)), 0.0)
    return min(gain / scale, 1.0)


def _easy_intervention_tax(
    mode: int,
    stream_delay_gap: Optional[float],
    gap_ref: float = 0.0,
) -> float:
    """-1 when a non-coast mode is selected while |gap| <= gap_ref.

    "Easy" = already at least as balanced as the paired no_op seed, so there
    is nothing to win and intervention is pure risk. Soft reward signal, not
    a mask: the policy stays free to act; abstention is simply what pays.
    Charged on the SELECTED mode (same convention as ``_control_cost``).
    0 when gap unknown, coast, or |gap| above ref.
    """
    if stream_delay_gap is None or int(mode) == 0:
        return 0.0
    if abs(float(stream_delay_gap)) <= max(float(gap_ref), 0.0):
        return -1.0
    return 0.0


def _corridor_ttc_penalty(corridor_ttc_count: Optional[int]) -> float:
    """-1 per corridor TTC event this step (raw, uncapped).

    The TTC-redistribution mechanism (wake probe 2026-08-04) is systemic:
    compression waves surface as sub-threshold TTC between HDV pairs anywhere
    in the corridor, not traceable to a specific CAV. A pairwise follower
    term cannot see it; a corridor-level count can. The RSU already tracks
    all vehicles, so this term aligns the reward with the eval-side safety
    leg directly. ``None`` (no snapshot) -> 0.
    """
    if corridor_ttc_count is None:
        return 0.0
    return -float(corridor_ttc_count)


def _tail_penalty(
    cav_id: str,
    observation_radius: float,
    cav_ids: Optional[Set[str]] = None,
    time_loss_tracker: Optional[TimeLossTracker] = None,
    tail_r_ref: float = 0.34,
) -> float:
    """One-sided p90 local-delay penalty (tail protection).

    -clip((p90 - r_ref)/r_ref, 0, 1) over local HDV per-step delay rates:
    punishes ONLY when the local worst-decile HDV delay exceeds the All-HDV
    baseline p90 — never rewards, so it cannot pay for compressing everyone
    into a uniform queue (the pathology that killed the old variance term).
    Returns 0.0 with fewer than 3 local HDVs (statistic unstable).
    """
    cav_ids = cav_ids or set()
    if time_loss_tracker is None:
        return 0.0

    rates = []
    for veh_id in _local_vehicles(cav_id, observation_radius):
        if veh_id in cav_ids:
            continue
        rates.append(time_loss_tracker.get(veh_id))

    if len(rates) < 3:
        return 0.0

    p90 = _percentile(rates, 90.0)
    tail_r_ref = max(tail_r_ref, 1e-6)
    return -min(max((p90 - tail_r_ref) / tail_r_ref, 0.0), 1.0)


def _control_health(
    flow_tracker: Optional[MergeOutflowTracker],
    flow_q_ref: float = 0.9,
    flow_v_ref: float = 27.0,
    source: str = "upstream",
) -> Optional[float]:
    """Traffic-health in [0, 1] for need-gated control cost (1 = calm / free-flow).

    ``upstream`` uses clip(s_norm, 0, 1) — the easy-mainline signal.
    ``sys`` uses clip(sys_norm, 0, 1). Returns None without a tracker.
    """
    if flow_tracker is None:
        return None
    if source == "sys":
        raw = flow_tracker.sys_norm(flow_q_ref, flow_v_ref)
    else:
        raw = flow_tracker.s_norm(flow_v_ref)
    return float(min(max(raw, 0.0), 1.0))


def _control_cost(
    mode: int,
    *,
    health: Optional[float] = None,
    need_gated: bool = False,
    c_min: float = 0.25,
) -> float:
    """Discourage unnecessary intervention (mode 0 / coast = no cost).

    Flat (need_gated=False or health unknown): -1 for any non-zero mode.
    Need-gated: cost = -(c_min + (1 - c_min) * health) with health in [0, 1],
    so calm traffic (health=1) still pays -1 and jammed traffic (health=0)
    pays only -c_min.
    """
    if int(mode) == 0:
        return 0.0
    if not need_gated or health is None:
        return -1.0
    h = min(max(float(health), 0.0), 1.0)
    c_lo = min(max(float(c_min), 0.0), 1.0)
    return -(c_lo + (1.0 - c_lo) * h)


def compute_shepherd_reward(
    cav_id: str,
    mode: int,
    max_speed: float,
    reward_weights: Dict[str, float],
    observation_radius: float,
    cav_ids: Optional[Set[str]] = None,
    jerk_tracker: Optional[JerkTracker] = None,
    time_loss_tracker: Optional[TimeLossTracker] = None,
    fairness_r_ref: float = 0.5,
    stream_delay_gap: Optional[float] = None,
    stream_fairness_gap_scale: float = 0.15,
    stream_delay_gap_ref: float = 0.0,
    corridor_delay_rate: Optional[float] = None,
    delay_rate_ref: Optional[float] = None,
    delay_budget_scale: float = 0.20,
    delay_win_scale: float = 0.20,
    flow_tracker: Optional[MergeOutflowTracker] = None,
    flow_q_ref: float = 0.9,
    flow_v_ref: float = 27.0,
    tail_r_ref: float = 0.34,
    control_cost_need_gated: bool = False,
    control_cost_c_min: float = 0.25,
    control_cost_health: str = "upstream",
    mask_fired: bool = False,
    ineligible: bool = False,
    mode_use_bonus_modes: Optional[list] = None,
    mode_use_bonus_overrides: Optional[Dict[str, float]] = None,
    safety_collision_penalty: float = 6.5,
    corridor_ttc_count: Optional[int] = None,
    action_mag: Optional[float] = None,
    action_mag_full: float = 1.5,
    action_mag_cost_floor: float = 0.3,
    metastable_health: Optional[float] = None,
) -> float:
    """Return a scalar human-centric reward for one CAV agent.

    ``mode_use_bonus_modes``: mode names that receive ``mode_use_bonus`` when
    eligible. ``None`` → any non-coast (legacy). Empty list → no bonus modes.

    ``metastable_health``: RSU merge-zone health h in [0,1] (see
    ``env.core.TrafficShepherdMultiAgentEnv.metastable_health_detail``). When
    the ``health`` weight is set, R_health = clip((h - 0.2)/0.8, 0, 1) credits
    the CAV for keeping the corridor on the metastable upper branch — the
    preventive objective Experiment C's mask-only design lacked.
    """
    w_e = reward_weights.get("efficiency", 1.0)
    w_s = reward_weights.get("safety", 1.0)
    w_t = reward_weights.get("stability", 0.25)
    w_f = reward_weights.get("fairness", 0.2)
    w_budget = reward_weights.get("delay_budget", 0.0)
    w_delay_win = reward_weights.get("delay_win", 0.0)
    w_fair_win = reward_weights.get("fair_win", 0.0)
    w_easy_tax = reward_weights.get("easy_intervention_tax", 0.0)
    w_c = reward_weights.get("control_cost", 0.0)
    w_mask = reward_weights.get("mask_penalty", 0.0)
    w_inelig = reward_weights.get("ineligible_penalty", 0.0)
    w_mode_use = reward_weights.get("mode_use_bonus", 0.0)
    w_cttc = reward_weights.get("corridor_ttc_penalty", 0.0)
    w_health = reward_weights.get("health", 0.0)
    # Legacy key: kept so old configs don't crash. R_eff is F_flow only now;
    # scale lives in ``efficiency`` (default 1.0). ``flow`` defaults to 1.0.
    w_flow = reward_weights.get("flow", 1.0)
    w_tail = reward_weights.get("tail", 0.25)

    r_eff = _efficiency(
        cav_id,
        max_speed,
        flow_tracker=flow_tracker,
        flow_q_ref=flow_q_ref,
        flow_v_ref=flow_v_ref,
        w_flow=w_flow,
    )
    r_safe = _safety(cav_id, observation_radius, collision_penalty=safety_collision_penalty)
    r_stab = _stability(cav_id, observation_radius, jerk_tracker=jerk_tracker)
    r_fair = _fairness(
        cav_id,
        observation_radius,
        max_speed,
        cav_ids,
        time_loss_tracker=time_loss_tracker,
        r_ref=fairness_r_ref,
        stream_delay_gap=stream_delay_gap,
        stream_fairness_gap_scale=stream_fairness_gap_scale,
        stream_delay_gap_ref=stream_delay_gap_ref,
    )
    r_budget = _delay_budget(
        corridor_delay_rate,
        delay_rate_ref,
        scale=delay_budget_scale,
    )
    r_win = _delay_win(
        corridor_delay_rate,
        delay_rate_ref,
        scale=delay_win_scale,
    )
    r_fair_win = _fair_win(
        stream_delay_gap,
        stream_fairness_gap_scale,
        stream_delay_gap_ref,
    )
    c_easy = _easy_intervention_tax(mode, stream_delay_gap, stream_delay_gap_ref)
    # wave_damp is the do-no-harm mode (active wave damping): exempt from
    # the easy tax, which would otherwise discourage it exactly on the
    # smooth corridors where it is most useful.
    if int(mode) == int(Mode.WAVE_DAMP):
        c_easy = 0.0
    c_cttc = _corridor_ttc_penalty(corridor_ttc_count)
    r_tail = _tail_penalty(
        cav_id,
        observation_radius,
        cav_ids,
        time_loss_tracker=time_loss_tracker,
        tail_r_ref=tail_r_ref,
    )
    # Metastable-health credit (retry, 2026-08-12): R_health = clip((h-0.2)/0.8)
    # — full credit only on the upper branch (h>=1), zero at/below h=0.2 where
    # the merge is already collapsing. The mask unlocks actions when h falls;
    # this term is what makes keeping h high (early, gentle action) worth it.
    r_health = 0.0
    if w_health and metastable_health is not None:
        r_health = min(max((float(metastable_health) - 0.2) / 0.8, 0.0), 1.0)
    health = (
        _control_health(
            flow_tracker,
            flow_q_ref,
            flow_v_ref,
            source=control_cost_health,
        )
        if control_cost_need_gated
        else None
    )
    c_ctrl = _control_cost(
        mode,
        health=health,
        need_gated=control_cost_need_gated,
        c_min=control_cost_c_min,
    )
    # wave_damp is FREE (2026-08-05): the do-no-harm mode pays no control
    # cost, same as coast. Without this the policy rationally never picks it
    # (v1 6-mode run: 0.0 usage at iter_25 — pure cost, no benefit term).
    # Its benefit arrives via the safety neighbor-share term when damping
    # actually calms neighbors; exploration starts once it is free.
    if int(mode) == int(Mode.WAVE_DAMP):
        c_ctrl = 0.0
    # Continuous-action shaping (2026-08-05): with a Gaussian policy the
    # binary intervention cost is an undescendable cliff (every step pays -1
    # because |a| > 1e-3 almost surely). Scale the intervention costs by
    # |a|/a_full so gentler actions are cheaper — the smooth gradient toward
    # gentle metering is exactly what the continuous action space is FOR.
    # Discrete path is untouched (action_mag stays None).
    # 2026-08-05 v2 (hybrid): floor the scale at ``action_mag_cost_floor``
    # — mag-only scaling halved the coast pressure at init (dial starts at
    # ~0.5 -> interventions pay half), and hybrid v1 intervened everywhere.
    if action_mag is not None and int(mode) != 0:
        mag_scale = min(max(float(action_mag), 0.0) / max(float(action_mag_full), 1e-9), 1.0)
        floor = min(max(float(action_mag_cost_floor), 0.0), 1.0)
        mag_scale = floor + (1.0 - floor) * mag_scale
        c_ctrl *= mag_scale
        c_easy *= mag_scale
    c_mask = -1.0 if mask_fired else 0.0
    c_inelig = -1.0 if ineligible else 0.0
    # Small per-step shaping when a selected mode actually runs. By default
    # only recover/vacate (see ``mode_use_bonus_modes``); fair_balance/protect
    # are excluded so metering is not paid for its own sake.
    # ``mode_use_bonus_overrides`` gives an ABSOLUTE per-mode bonus (replacing
    # the weight scaling): wave_damp gets 0.3 to break the zero-usage/
    # zero-signal cycle its first runs showed (2026-08-05).
    mode_i = int(mode)
    name = MODE_NAMES[mode_i] if 0 <= mode_i < len(MODE_NAMES) else ""
    _override = (mode_use_bonus_overrides or {}).get(name)
    mode_use_term = 0.0
    if not ineligible and name:
        if _override is not None:
            mode_use_term = float(_override)
        elif mode_use_bonus_modes is None:
            mode_use_term = w_mode_use if mode_i != int(Mode.COAST) else 0.0
        elif name in set(mode_use_bonus_modes):
            mode_use_term = w_mode_use

    return float(
        w_e * r_eff
        + w_s * r_safe
        + w_t * r_stab
        + w_f * r_fair
        + w_budget * r_budget
        + w_delay_win * r_win
        + w_fair_win * r_fair_win
        + w_easy_tax * c_easy
        + w_cttc * c_cttc
        + w_c * c_ctrl
        + w_mask * c_mask
        + w_inelig * c_inelig
        + mode_use_term
        + w_tail * r_tail
        + w_health * r_health
    )


def compute_shepherd_rewards(
    action_dict: Dict[str, int],
    max_speed: float,
    reward_weights: Dict[str, float],
    observation_radius: float,
    cav_ids: Optional[Set[str]] = None,
    jerk_tracker: Optional[JerkTracker] = None,
    time_loss_tracker: Optional[TimeLossTracker] = None,
    fairness_r_ref: float = 0.5,
    stream_delay_gap: Optional[float] = None,
    stream_fairness_gap_scale: float = 0.15,
    stream_delay_gap_ref: float = 0.0,
    corridor_delay_rate: Optional[float] = None,
    corridor_edges: Optional[tuple] = None,
    delay_rate_ref: Optional[float] = None,
    delay_budget_scale: float = 0.20,
    delay_win_scale: float = 0.20,
    flow_tracker: Optional[MergeOutflowTracker] = None,
    flow_q_ref: float = 0.9,
    flow_v_ref: float = 27.0,
    tail_r_ref: float = 0.34,
    control_cost_need_gated: bool = False,
    control_cost_c_min: float = 0.25,
    control_cost_health: str = "upstream",
    mask_fired: Optional[Dict[str, bool]] = None,
    ineligible: Optional[Dict[str, bool]] = None,
    mode_use_bonus_modes: Optional[list] = None,
    mode_use_bonus_overrides: Optional[Dict[str, float]] = None,
    safety_collision_penalty: float = 6.5,
    corridor_ttc_count: Optional[int] = None,
    action_mags: Optional[Dict[str, float]] = None,
    action_mag_full: float = 1.5,
    action_mag_cost_floor: float = 0.3,
    metastable_health: Optional[float] = None,
) -> Dict[str, float]:
    """Return a reward dict for all CAV agents that acted."""
    cav_ids = set(cav_ids) if cav_ids is not None else set(action_dict.keys())
    if jerk_tracker is not None:
        jerk_tracker.update()
    if time_loss_tracker is not None:
        time_loss_tracker.update()
    if flow_tracker is not None:
        flow_tracker.update()
    # Corridor delay rate for the paired budget term (after tracker update).
    if corridor_delay_rate is None and corridor_edges and time_loss_tracker is not None:
        snap = get_active_snapshot()
        edges = set(corridor_edges)
        rates: List[float] = []
        if snap is not None:
            for vid in snap.ordered_ids():
                road = snap.road(vid)
                if road is None or road not in edges:
                    continue
                rates.append(float(time_loss_tracker.get(vid)))
        if rates:
            corridor_delay_rate = float(sum(rates) / len(rates))
    snap = get_active_snapshot()
    mask_fired = mask_fired or {}
    ineligible = ineligible or {}
    action_mags = action_mags or {}
    rewards = {}
    for cav_id, mode in action_dict.items():
        if snap is not None:
            if not snap.has(cav_id):
                continue
        elif cav_id not in traci.vehicle.getIDList():
            continue
        rewards[cav_id] = compute_shepherd_reward(
            cav_id,
            mode,
            max_speed,
            reward_weights,
            observation_radius,
            cav_ids=cav_ids,
            jerk_tracker=jerk_tracker,
            time_loss_tracker=time_loss_tracker,
            fairness_r_ref=fairness_r_ref,
            stream_delay_gap=stream_delay_gap,
            stream_fairness_gap_scale=stream_fairness_gap_scale,
            stream_delay_gap_ref=stream_delay_gap_ref,
            corridor_delay_rate=corridor_delay_rate,
            delay_rate_ref=delay_rate_ref,
            delay_budget_scale=delay_budget_scale,
            delay_win_scale=delay_win_scale,
            flow_tracker=flow_tracker,
            flow_q_ref=flow_q_ref,
            flow_v_ref=flow_v_ref,
            tail_r_ref=tail_r_ref,
            control_cost_need_gated=control_cost_need_gated,
            control_cost_c_min=control_cost_c_min,
            control_cost_health=control_cost_health,
            mask_fired=bool(mask_fired.get(cav_id, False)),
            ineligible=bool(ineligible.get(cav_id, False)),
            mode_use_bonus_modes=mode_use_bonus_modes,
            mode_use_bonus_overrides=mode_use_bonus_overrides,
            safety_collision_penalty=safety_collision_penalty,
            corridor_ttc_count=corridor_ttc_count,
            action_mag=action_mags.get(cav_id),
            action_mag_full=action_mag_full,
            action_mag_cost_floor=action_mag_cost_floor,
            metastable_health=metastable_health,
        )
    return rewards


def compute_idm_damping_rewards(
    residual_actions: Dict[str, float],
    all_action_ids: Iterable[str],
    max_speed: float,
    reward_weights: Dict[str, float],
    observation_radius: float,
    *,
    cav_ids: Optional[Set[str]] = None,
    jerk_tracker: Optional[JerkTracker] = None,
    time_loss_tracker: Optional[TimeLossTracker] = None,
    delay_rate_ref: Optional[float] = None,
    delay_budget_scale: float = 0.20,
    corridor_edges: Optional[tuple] = None,
    flow_tracker: Optional[MergeOutflowTracker] = None,
    flow_q_ref: float = 0.9,
    flow_v_ref: float = 27.0,
    corridor_ttc_count: Optional[int] = None,
    corridor_severe_braking_count: Optional[int] = None,
    corridor_severe_braking_count_ref_per_s: float = 1.0,
    corridor_acceleration_energy_sum_m2ps4: Optional[float] = None,
    corridor_acceleration_energy_sum_ref_m2ps4: float = 1.0,
    residual_scale_mps2: float = 0.5,
    shared_reward_ids: Optional[Iterable[str]] = None,
    center_corridor_terms: bool = False,
    diagnostics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Phase-4 selected-absorber damping reward.

    Only residuals that the deterministic actuator executed receive learning
    signal. Safety and comfort are local. Corridor delay, discharge, and TTC
    are shared once across selected absorbers so their magnitude does not grow
    with the number of simultaneously controlled lanes. Nonselected policy
    samples receive exactly zero.
    """
    selected = dict(residual_actions)
    action_ids = tuple(str(agent_id) for agent_id in all_action_ids)
    rewards = {agent_id: 0.0 for agent_id in action_ids}
    recipients = (
        tuple(
            agent_id
            for agent_id in dict.fromkeys(str(value) for value in shared_reward_ids)
            if agent_id in rewards
        )
        if shared_reward_ids is not None
        else tuple(selected)
    )
    local_weights = {
        "efficiency": 0.0,
        "safety": (
            0.0 if center_corridor_terms else float(reward_weights.get("safety", 2.0))
        ),
        "stability": (
            0.0
            if center_corridor_terms
            else float(reward_weights.get("stability", 0.5))
        ),
        "fairness": 0.0,
        "delay_budget": 0.0,
        "delay_win": 0.0,
        "fair_win": 0.0,
        "easy_intervention_tax": 0.0,
        "control_cost": 0.0,
        "mask_penalty": 0.0,
        "ineligible_penalty": 0.0,
        "mode_use_bonus": 0.0,
        "corridor_ttc_penalty": 0.0,
        "health": 0.0,
        "flow": 0.0,
        "tail": 0.0,
    }
    local = compute_shepherd_rewards(
        {agent_id: int(Mode.COAST) for agent_id in recipients},
        max_speed,
        local_weights,
        observation_radius,
        cav_ids=cav_ids,
        jerk_tracker=jerk_tracker,
        time_loss_tracker=time_loss_tracker,
        flow_tracker=flow_tracker,
        flow_q_ref=flow_q_ref,
        flow_v_ref=flow_v_ref,
    )
    corridor_delay_rate = None
    if corridor_edges and time_loss_tracker is not None:
        snap = get_active_snapshot()
        edges = set(corridor_edges)
        rates = []
        if snap is not None:
            for vehicle_id in snap.ordered_ids():
                if snap.road(vehicle_id) in edges:
                    rates.append(float(time_loss_tracker.get(vehicle_id)))
        if rates:
            corridor_delay_rate = float(sum(rates) / len(rates))
    efficiency_term = 0.0
    delay_term = 0.0
    ttc_term = 0.0
    severe_braking_term = 0.0
    corridor_wave_energy_term = 0.0
    if recipients:
        exemplar = recipients[0]
        efficiency_term = float(reward_weights.get("efficiency", 0.35)) * _efficiency(
            exemplar, max_speed, flow_tracker, flow_q_ref, flow_v_ref, 1.0
        )
        delay_term = float(reward_weights.get("delay_budget", 0.5)) * _delay_budget(
            corridor_delay_rate, delay_rate_ref, scale=delay_budget_scale
        )
        ttc_term = float(reward_weights.get("corridor_ttc_penalty", 1.0)) * _corridor_ttc_penalty(
            corridor_ttc_count
        )
        severe_count = max(float(corridor_severe_braking_count or 0), 0.0)
        severe_ref = max(float(corridor_severe_braking_count_ref_per_s), 1e-6)
        severe_weight = float(reward_weights.get("corridor_severe_braking", 0.0))
        severe_braking_term = (
            severe_weight * (1.0 - severe_count / severe_ref)
            if center_corridor_terms
            else -severe_weight * severe_count / severe_ref
        )
        acceleration_energy = max(float(corridor_acceleration_energy_sum_m2ps4 or 0.0), 0.0)
        acceleration_ref = max(float(corridor_acceleration_energy_sum_ref_m2ps4), 1e-6)
        wave_energy_weight = float(reward_weights.get("corridor_wave_energy", 0.0))
        corridor_wave_energy_term = (
            wave_energy_weight * (1.0 - acceleration_energy / acceleration_ref)
            if center_corridor_terms
            else -wave_energy_weight * acceleration_energy / acceleration_ref
        )
    # Legacy selected-only residual control shares one global term across
    # simultaneous actuators.  The decentralized activation actor instead
    # gives each feasible decision agent the same cooperative state reward so
    # abstention has a real counterfactual return rather than a hard-coded 0.
    divisor = max(len(selected), 1) if shared_reward_ids is None else 1
    wave_weight = float(reward_weights.get("wave", 1.0))
    effort_weight = float(reward_weights.get("residual_effort", 0.15))
    scale = max(abs(float(residual_scale_mps2)), 1e-6)
    component_rows: Dict[str, Dict[str, float]] = {}
    for agent_id in recipients:
        residual = float(selected.get(agent_id, 0.0))
        risk, _forming = wave_risk_signal(agent_id)
        effort = (
            min(abs(residual) / scale, 1.0) ** 2 if agent_id in selected else 0.0
        )
        row = {
            "local": float(local.get(agent_id, 0.0)),
            "efficiency": float(efficiency_term / divisor),
            "delay_budget": float(delay_term / divisor),
            "corridor_ttc": float(ttc_term / divisor),
            "corridor_severe_braking": float(severe_braking_term / divisor),
            "corridor_wave_energy": float(corridor_wave_energy_term / divisor),
            "wave": float(-wave_weight * float(risk)),
            "residual_effort": float(-effort_weight * effort),
            "wave_risk": float(risk),
            "executed_residual_mps2": float(residual),
        }
        row["total"] = float(
            sum(
                row[key]
                for key in (
                    "local",
                    "efficiency",
                    "delay_budget",
                    "corridor_ttc",
                    "corridor_severe_braking",
                    "corridor_wave_energy",
                    "wave",
                    "residual_effort",
                )
            )
        )
        component_rows[agent_id] = row
        rewards[agent_id] = row["total"]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(component_rows)
    return rewards


# ---------------------------------------------------------------------------
# Selfish (personal) reward
# ---------------------------------------------------------------------------
def compute_selfish_reward(
    cav_id: str,
    max_speed: float,
    weights: Optional[Dict[str, float]] = None,
    ttc_threshold: float = 2.5,
    accel_ref: float = 3.0,
    hard_brake: float = -4.5,
) -> float:
    """Return selfish scalar reward for one CAV.

    Speed is down-weighted vs comfort/safety so free-flow cruising is not
    already near-optimal; TTC uses a soft continuous penalty.
    """
    weights = weights or {"speed": 0.4, "comfort": 1.0, "safety": 2.0}
    snap = get_active_snapshot()
    if snap is not None:
        if not snap.has(cav_id):
            return 0.0
        speed = snap.speed(cav_id)
        accel = snap.accel(cav_id)
        if speed is None or accel is None:
            return 0.0
    else:
        if cav_id not in traci.vehicle.getIDList():
            return 0.0

        try:
            speed = traci.vehicle.getSpeed(cav_id)
            accel = traci.vehicle.getAcceleration(cav_id)
        except Exception:
            return 0.0

    # Prefer mid-to-high speed but leave headroom so safety/comfort matter.
    speed_frac = speed / max(max_speed, 1e-6)
    r_v = speed_frac * (2.0 - speed_frac)  # peaks below max_speed, softer ceiling
    r_c = -((accel / max(accel_ref, 1e-6)) ** 2)

    ttc = leader_ttc(cav_id)
    r_s = 0.0
    if ttc < ttc_threshold:
        # Soft: 0 at threshold, -1 as TTC → 0.
        r_s -= (1.0 - ttc / ttc_threshold) ** 2
    if accel < hard_brake:
        r_s -= 1.0

    return float(
        weights.get("speed", 0.4) * r_v
        + weights.get("comfort", 1.0) * r_c
        + weights.get("safety", 2.0) * r_s
    )


def compute_selfish_rewards(
    cav_ids,
    max_speed: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    snap = get_active_snapshot()
    if snap is not None:
        return {
            cav_id: compute_selfish_reward(cav_id, max_speed, weights)
            for cav_id in cav_ids
            if snap.has(cav_id)
        }
    return {
        cav_id: compute_selfish_reward(cav_id, max_speed, weights)
        for cav_id in cav_ids
        if cav_id in traci.vehicle.getIDList()
    }
