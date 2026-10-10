"""All action execution in one place — edit here to change how RL outputs move cars.

Shepherd discrete modes (docs/GOAL_FAIR_SAFE.md): fair + safe + TT budget.

   Mode | Name            | Eligibility                         | Physics
   -----|-----------------|-------------------------------------|---------------------------
   0    | coast           | always                              | setSpeed -1, restore tau
   1    | fair_balance    | tip + |G| > G_ref+tau (no_op-relative) | mild slow of advantaged side
   2    | recover_speed   | pre-merge, merge/down clear, ego slow | absolute speed push
   3    | vacate_drop     | on merge L0                         | changeLane to L1 + soft
   4    | protect         | low leader/side TTC or merge PET    | brief soft brake + tau

   Comfort rate limits apply to every speed change. Ineligible remaps → coast
   and ``last_ineligible`` for ``ineligible_penalty``.
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import traci

from config.constants import (
    ACCELERATION_MAX,
    ACCELERATION_MIN,
    MERGE_DROPPED_LANE_INDEX,
    MERGE_EDGE,
    MERGE_SURVIVING_LANE_INDEX,
    MODE_NAMES,
    Mode,
    NUM_MODES,
    PRE_MERGE_EDGE,
    RAMP_EDGE,
    RL_SPEED_MODE,
)
from env.traci_query import (
    DEFAULT_LOOK_AHEAD,
    leader_info,
    leader_ttc,
    merge_arrival_pet,
    side_ttc,
)
from env.snapshot import get_active_snapshot
from env.features import distance_to_merge, wave_risk_signal

__all__ = [
    "MODE_NAMES",
    "NUM_MODES",
    "RL_SPEED_MODE",
    "ACCELERATION_MIN",
    "ACCELERATION_MAX",
    "ShepherdActionEngine",
    "make_action_space",
    "apply_continuous_acceleration",
    "apply_continuous_accelerations",
]


def make_action_space(num_modes: int = NUM_MODES) -> gym.spaces.Discrete:
    """Return the per-agent discrete action space."""
    return gym.spaces.Discrete(num_modes)


def _rate_limit(
    v_ego: float,
    v_target: float,
    dt: float,
    a_max_dec: float = 2.0,
    a_max_acc: float = 1.0,
    deadband: float = 0.5,
) -> float:
    """Rate-limit speed change: brake promptly, accelerate slowly, with deadband."""
    diff = v_target - v_ego
    if abs(diff) < deadband:
        return v_ego
    if diff < 0:
        max_change = a_max_dec * dt
        return max(v_target, v_ego - max_change)
    max_change = a_max_acc * dt
    return min(v_target, v_ego + max_change)


def _safe_speed_from_leader(
    cav_id: str,
    current_speed: float,
    desired_speed: float,
    tau: float,
) -> float:
    """Reduce desired speed if the leader is too close for the given tau."""
    leader = leader_info(cav_id, DEFAULT_LOOK_AHEAD)
    if leader is None:
        return desired_speed
    _, distance, leader_speed = leader
    min_gap = max(2.0, current_speed * tau)
    if distance > min_gap and desired_speed <= leader_speed:
        return desired_speed
    return min(desired_speed, max(0.0, leader_speed - 0.5))


def _set_speed_safe(cav_id: str, target_speed: float, speed_mode: int = RL_SPEED_MODE) -> None:
    """Set speed with speedMode for RL control."""
    try:
        traci.vehicle.setSpeedMode(cav_id, speed_mode)
        traci.vehicle.setSpeed(cav_id, target_speed)
    except Exception:
        pass


def _edge_mean_speed(edge_id: str, min_n: int = 3) -> Optional[float]:
    try:
        ids = traci.edge.getLastStepVehicleIDs(edge_id)
        if len(ids) < min_n:
            return None
        return float(mean(traci.vehicle.getSpeed(v) for v in ids))
    except Exception:
        return None


def _lane_role(cav_id: str) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """Return (edge, lane_index, remaining_m_on_lane) or (None, None, None)."""
    try:
        lane = traci.vehicle.getLaneID(cav_id)
        edge = lane.rsplit("_", 1)[0]
        li = int(lane.rsplit("_", 1)[1])
        pos = float(traci.vehicle.getLanePosition(cav_id))
        length = float(traci.lane.getLength(lane))
        return edge, li, max(0.0, length - pos)
    except Exception:
        return None, None, None


class ShepherdActionEngine:
    """Eligibility-gated Shepherd modes → TraCI speed / lane commands."""

    # Pre-rename mode labels (2026-08-20): checkpoints and stored configs from
    # before the rename carry "recover_speed"/"protect"; they execute as
    # "recover"/"guard" everywhere (the modes themselves are unchanged — the
    # rename is mechanics-accurate naming for the paper: recover accelerates
    # to fill rarefactions, guard slows to absorb conflicts).
    _MODE_ALIASES = {"recover_speed": "recover", "protect": "guard"}

    def __init__(
        self,
        base_tau: float,
        tau_deltas: Dict[str, float],
        speed_factors: Dict[str, Optional[float]],
        max_speed: float,
        dt: float = 1.0,
        flow_stab_config: Optional[Dict] = None,
        mode_config: Optional[Dict] = None,
        mode_names: Optional[List[str]] = None,
    ):
        self.base_tau = base_tau
        self.tau_deltas = {self._MODE_ALIASES.get(k, k): v for k, v in tau_deltas.items()}
        self.speed_factors = {self._MODE_ALIASES.get(k, k): v for k, v in speed_factors.items()}
        self.max_speed = max_speed
        self.dt = dt
        self.flow_stab_config = flow_stab_config or {}  # unused; kept for ctor compat
        mode_config = mode_config or {}

        # Active mode set (pruned-champion support, 2026-08-20): the
        # action-index -> mode-name mapping. Historically this was the global
        # MODE_NAMES registry, which only works when the run's mode set is a
        # PREFIX of the registry — pruning fair_balance/vacate_drop out of
        # the middle (the pruned baseline: coast/damp/guard) shifts
        # indices and silently mis-dispatches. The env now passes its
        # configured shepherd_modes here; default stays the global registry.
        self.mode_names: List[str] = [
            self._MODE_ALIASES.get(n, n)
            for n in (list(mode_names) if mode_names else list(MODE_NAMES))
        ]
        self.mode_count = (
            len(self.mode_names) if mode_names else int(mode_config.get("mode_count", 5))
        )

        self.mode_max_decel = float(mode_config.get("max_decel", 1.0))
        self.mode_max_accel = float(mode_config.get("max_accel", 1.0))
        self.mode_deadband = float(mode_config.get("deadband", 0.5))
        self.guard_speed = float(mode_config.get("guard_speed", 8.0))
        # T1 (2026-08-15): install a bigger car-following tau GRADUALLY
        # (``tau_ramp_step`` s per step). An instant setTau makes SUMO's CF
        # demand the full larger gap at once, so the CAV brakes hard to open
        # it — the intervention itself would nucleate the shockwave the
        # big-headway hold is supposed to prevent.
        self.tau_ramp_step = float(mode_config.get("tau_ramp_step", 0.3))
        self._tau_applied: Dict[str, float] = {}

        # Eval-time smooth execution (literature probe, 2026-08-06, Stern-style
        # wave damping): when ``smooth_exec`` is on, every mode's speed change
        # is executed as a slow smooth ramp (``smooth_a_max_*``) from the CAV's
        # previously commanded speed instead of the mode's 1.0 m/s2 step limit.
        # When ``look_behind_exec`` is on, the executed reference is the EMA of
        # the BEHIND-band mean speed (upstream info; LCC "looking-behind"
        # topology, Wang 2020 / 2023) instead of the mode's own target — the
        # CAV becomes an upstream wave-absorber. Both are eval-only overrides
        # (mode_config), default OFF so training behavior is unchanged.
        self.smooth_exec = bool(mode_config.get("smooth_exec", False))
        self.smooth_a_max_dec = float(mode_config.get("smooth_a_max_dec", 0.4))
        self.smooth_a_max_acc = float(mode_config.get("smooth_a_max_acc", 0.4))
        self.smooth_deadband = float(mode_config.get("smooth_deadband", 0.3))
        self.smooth_ema_alpha = float(mode_config.get("smooth_ema_alpha", 0.3))
        self.look_behind_exec = bool(mode_config.get("look_behind_exec", False))
        self.behind_near = float(mode_config.get("behind_near", 150.0))
        self.behind_far = float(mode_config.get("behind_far", 200.0))
        self.behind_ema_alpha = float(mode_config.get("behind_ema_alpha", 0.3))
        # Reactive dissolve (mechanism probe, 2026-08-08): when on, every
        # speed command is overridden to a Stern-style gentle wave-match —
        # slow to the leader's speed when a wave is ahead, else recover to
        # free flow, all rate-limited. Eval-time, default OFF.
        self.dissolve_exec = bool(mode_config.get("dissolve_exec", False))
        self.dissolve_slow_v = float(mode_config.get("dissolve_slow_v", 12.0))
        self.dissolve_recover_v = float(mode_config.get("dissolve_recover_v", 20.0))
        self.dissolve_look = float(mode_config.get("dissolve_look", 500.0))  # really downstream
        self.dissolve_lead_min = float(
            mode_config.get("dissolve_lead_min", 150.0)
        )  # abort if closer
        # String-stable SUPPORT (paper variant, 2026-08-08): when a CAV is
        # upstream of the dissolving region and abstains, extend its time gap
        # by this delta to damp the absorber's wake. 0 = disabled.
        self.dissolve_support_tau = float(mode_config.get("dissolve_support_tau", 0.0))
        # Merge-stream smoothing (2026-08-08): when merge-zone congestion is
        # within the (buffered) RSU range, match the merge zone's current speed
        # LONG BEFORE reaching it, gently, absorbing the merge-induced wave.
        # Eval-time, default OFF.
        self.merge_match_exec = bool(mode_config.get("merge_match_exec", False))
        self.merge_match_min_speed = float(mode_config.get("merge_match_min_speed", 15.0))
        self.merge_match_comm = float(mode_config.get("merge_match_comm", 1500.0))
        self.merge_match_floor = float(mode_config.get("merge_match_floor", 4.0))
        # Proper-JAD probe (2026-08-09): ONE absorbing CAV holds an absorbing
        # speed while upstream CAVs extend LARGE time gaps (support driving) —
        # the paper's design. Eval-time, default OFF.
        self.jad_exec = bool(mode_config.get("jad_exec", False))
        self.jad_absorb_speed = float(mode_config.get("jad_absorb_speed", 13.0))
        self.jad_congest_thresh = float(mode_config.get("jad_congest_thresh", 15.0))
        self.jad_support_tau = float(mode_config.get("jad_support_tau", 4.0))
        # Wave-triggered JAD (2026-08-09): the RSU detects a backward-
        # propagating shockwave at each CAV from its neighbors (ahead slow AND
        # ego slowed vs free-flow behind) instead of a merge-speed threshold.
        self.jad_slow_thresh = float(mode_config.get("jad_slow_thresh", 12.0))
        self.jad_ego_delta = float(mode_config.get("jad_ego_delta", 1.5))
        self.jad_look = float(mode_config.get("jad_look", 200.0))
        self.jad_nbr_n = int(mode_config.get("jad_nbr_n", 2))
        # Gap-selection / strategic-slowdown probe (2026-08-09): the mainline
        # CAV nearest the merge opens a gap when a ramp vehicle approaches —
        # Yang/Choi style. Eval-time, default OFF.
        self.gap_select_exec = bool(mode_config.get("gap_select_exec", False))
        self.gap_select_range = float(mode_config.get("gap_select_range", 150.0))
        self.gap_select_factor = float(mode_config.get("gap_select_factor", 0.85))
        self.gap_select_floor = float(mode_config.get("gap_select_floor", 12.0))
        # Per-step selection state (reset in apply_all).
        self._jad_absorber: Optional[str] = None
        self._gap_opener: Optional[str] = None
        # Last commanded speed per CAV (execution-profile state).
        self._exec_prev: dict[str, float] = {}

        disabled = mode_config.get("disabled_modes") or ()
        self.disabled_modes = frozenset(self._MODE_ALIASES.get(str(m), str(m)) for m in disabled)
        # mode_count already set from mode_names when the env passes the
        # configured set; mode_config is the fallback for direct construction.
        if not mode_names:
            self.mode_count = int(mode_config.get("mode_count", 5))
        self.health_mask_enabled = bool(mode_config.get("health_mask", False))
        self.health_mask_threshold = float(mode_config.get("health_mask_threshold", 0.8))
        masked = mode_config.get("health_mask_modes") or ()
        self.health_mask_modes = frozenset(str(m) for m in masked)
        self.health_fn = None
        # Fairness gate: allow listed modes only when F < -threshold
        # (F = (r_ref - mean_hdv_delay_rate) / r_ref; negative => worse than ref).
        self.fairness_mask_enabled = bool(mode_config.get("fairness_mask", False))
        self.fairness_mask_threshold = float(mode_config.get("fairness_mask_threshold", 0.1))
        fmodes = mode_config.get("fairness_mask_modes") or ()
        self.fairness_mask_modes = frozenset(str(m) for m in fmodes)
        self.fairness_fn = None
        # mask-v1 (2026-08-04): veto on discretionary modes when the corridor
        # is already REALLY fair in absolute terms:
        #   |gap| <= easy_intervention_mask_gap_abs
        # (user steer 2026-08-04: absolute fairness, NOT "fair vs the paired
        # no_op ref". gap_abs <= 0 falls back to the old ref-based condition
        # |gap| <= gap_ref, same as rewards._easy_intervention_tax.)
        # Coast is always legal; protect is exempt (safety). fair_balance
        # self-gates on the ref condition already but is kept in the list
        # for defense in depth.
        # ``easy_intervention_mask_discount`` interpolates hard vs soft:
        #   0.0  -> hard ban (mask bit 0, apply() remaps to coast)
        #   (0,1) -> soft mask (mask bit = discount; logits penalty log(d);
        #            apply() does NOT remap, the mode stays physically allowed)
        #   1.0  -> no effect
        self.easy_intervention_mask_enabled = bool(mode_config.get("easy_intervention_mask", False))
        emodes = mode_config.get("easy_intervention_mask_modes") or ()
        self.easy_intervention_mask_modes = frozenset(str(m) for m in emodes)
        self.easy_intervention_mask_discount = min(
            max(float(mode_config.get("easy_intervention_mask_discount", 0.0)), 0.0),
            1.0,
        )
        # Absolute "really fair" threshold on |gap| (delay-rate units; dense
        # coast |gap| ~ 0.10, so 0.05 = clearly balanced). <= 0 -> ref-based.
        self.easy_intervention_mask_gap_abs = float(
            mode_config.get("easy_intervention_mask_gap_abs", 0.0)
        )
        # Graduated soft mask (user steer 2026-08-04): instead of a step at
        # gap_abs, fade the discount linearly from ``discount`` at
        # |gap| <= gap_abs up to 1.0 (no effect) at |gap| >= gap_outer —
        # the fairer the corridor, the stronger the logit penalty.
        self.easy_intervention_mask_graduated = bool(
            mode_config.get("easy_intervention_mask_graduated", False)
        )
        self.easy_intervention_mask_gap_outer = float(
            mode_config.get("easy_intervention_mask_gap_outer", 0.10)
        )
        # Hybrid mask (2026-08-05): hard BAN only in the ultra-fair zone
        # (|gap| <= hard_below), soft/graduated above it. Targets the
        # structural-harm seeds (near-fair no_op: nothing to win, any
        # intervention is pure risk) without paying the soft mask's leak on
        # the genuinely-close seeds. 0.0 = off (pure soft behavior).
        self.easy_intervention_mask_hard_below = float(
            mode_config.get("easy_intervention_mask_hard_below", 0.0)
        )

        # Metastable-health mask (Experiment C, 2026-08-11): RSU aggregates
        # merge-zone CAV data each step into a health scalar h in [0,1] — 1 =
        # deep in the metastable upper branch (free coast), 0 = collapse
        # imminent (meter). Graduated soft mask: the healthier the merge, the
        # stronger the logit penalty on the discretionary metering/recovery
        # modes; as h drops (approaching collapse) the discount rises to 1.0
        # (unrestricted) so the policy can meter freely to restore metastable.
        # Coast always legal; protect exempt (safety). Same graduated semantics
        # as easy_intervention_mask (mask bit = discount; logits penalty
        # log(d); apply() does NOT remap).
        self.met_health_mask_enabled = bool(mode_config.get("metastable_health_mask", False))
        mh_modes = mode_config.get("metastable_health_mask_modes") or ()
        self.met_health_mask_modes = frozenset(str(m) for m in mh_modes)
        self.met_health_mask_discount = min(
            max(float(mode_config.get("metastable_health_mask_discount", 0.05)), 0.0),
            1.0,
        )
        # h window: discount at h <= h_lo (worst health), 1.0 (no effect) at
        # h >= h_hi (healthiest). Linear fade between.
        self.met_health_mask_h_lo = float(mode_config.get("metastable_health_mask_h_lo", 0.2))
        self.met_health_mask_h_hi = float(mode_config.get("metastable_health_mask_h_hi", 0.7))
        # Feasibility mask (2026-08-13): the SAME soft-mask mechanics, gated
        # on the runway margin margin = L_pre - L_min(rho) (fitted from the
        # density x length controllability grid). Inverse semantics of the
        # health mask: INFEASIBLE state (margin <= lo, i.e. this highway is
        # too short for control at this density) -> strong logit penalty on
        # the discretionary modes (coast — control is futile); feasible state
        # (margin >= hi) -> ungated.
        self.feasibility_mask_enabled = bool(mode_config.get("feasibility_mask", False))
        self.feasibility_mask_discount = min(
            max(float(mode_config.get("feasibility_mask_discount", 0.05)), 0.0),
            1.0,
        )
        self.feasibility_margin_lo = float(mode_config.get("feasibility_margin_lo", 0.0))
        self.feasibility_margin_hi = float(mode_config.get("feasibility_margin_hi", 300.0))
        self.feas_fn = None  # set by the env each step (stored margin)
        # RSU health provider (set by the env each step). Returns h in [0,1]
        # or None when unavailable (→ no masking).
        self.met_health_fn = None

        # Gated-mode knobs (GOAL_FAIR_SAFE)
        self.fair_gap_tau = float(mode_config.get("fair_balance_gap_tau", 0.05))
        self.fair_tip = float(mode_config.get("fair_balance_tip", 120.0))
        self.fair_factor = float(mode_config.get("fair_balance_factor", 0.90))
        self.fair_floor = float(mode_config.get("fair_balance_floor", 8.0))
        self.fair_ema_alpha = float(mode_config.get("fair_balance_ema_alpha", 0.2))
        self.ema_main_delay: Optional[float] = None
        self.ema_ramp_delay: Optional[float] = None
        # Paired no_op stream-gap baseline; fair_balance only if |G| > ref + tau.
        self.stream_delay_gap_ref = float(mode_config.get("stream_delay_gap_ref", 0.0))
        # Adaptive gentleness (2026-08-05): modulate the fair_balance slowdown
        # factor by the trailing corridor-TTC EMA. The holdout probe showed the
        # policy ADDS TTC events on smooth corridors (d_ttc vs noop_ttc pearson
        # -0.65) and REMOVES them on chaotic ones. ema <= ttc_lo (smooth) ->
        # ``adaptive_gentle_factor``; ema >= ttc_hi (chaotic) -> base
        # ``fair_factor``; linear between. NOTE: the global scouting eval
        # REFUTED the "gentler = fewer waves" hypothesis (0.93 was TTC-worse
        # than 0.90) — the indicated direction is FIRMER on smooth corridors,
        # so the factor is NOT clamped against fair_factor anymore.
        self.adaptive_gentle = bool(mode_config.get("adaptive_gentle", False))
        self.adaptive_gentle_factor = min(
            max(float(mode_config.get("adaptive_gentle_factor", 0.85)), 0.3),
            1.0,
        )
        self.adaptive_gentle_ttc_lo = float(mode_config.get("adaptive_gentle_ttc_lo", 0.5))
        self.adaptive_gentle_ttc_hi = float(mode_config.get("adaptive_gentle_ttc_hi", 2.0))
        self.adaptive_gentle_ema_alpha = float(mode_config.get("adaptive_gentle_ema_alpha", 0.1))
        # Injected by the env when adaptive_gentle or recover_ttc_gate is on
        # (snapshot-based corridor TTC pair count, same source as the
        # corridor_ttc_penalty reward term).
        self.corridor_ttc_fn = None
        self._corridor_ttc_ema = 0.0
        # Flow stabilization / wave damping (ported from research_v2, 2026-08-05):
        # the CAV matches the EMA of ahead-band lane speeds with smooth
        # rate-limited commands, absorbing compression waves instead of
        # amplifying them. The v2-era mode set had this as a dedicated mode;
        # the dense lineage lost it in the fairness redesign.
        self.flow_stab_near = float(mode_config.get("flow_stab_near", 150.0))
        self.flow_stab_far = float(mode_config.get("flow_stab_far", 200.0))
        self.flow_stab_ema_alpha = float(mode_config.get("flow_stab_ema_alpha", 0.3))
        self.flow_stab_deadband = float(mode_config.get("flow_stab_deadband", 0.5))
        self.flow_stab_a_max_dec = float(mode_config.get("flow_stab_a_max_dec", 1.0))
        self.flow_stab_a_max_acc = float(mode_config.get("flow_stab_a_max_acc", 1.0))
        self.flow_stab_preview = float(mode_config.get("flow_stab_preview_gain", 0.5))
        # Recover TTC gate (2026-08-05): recover is load-bearing on wins
        # (chaotic corridors) but a TTC injector on smooth ones — the policy
        # ADDS corridor TTC exactly where the no_op was smooth (pearson
        # -0.65). Ban recover when the trailing corridor-TTC EMA is below
        # ``recover_ttc_lo`` (smooth), keep it when already chaotic.
        self.recover_ttc_gate = bool(mode_config.get("recover_ttc_gate", False))
        self.recover_ttc_lo = float(mode_config.get("recover_ttc_lo", 0.5))
        # Benefit gate (2026-08-06, seed-variance finding): the policy helps
        # only where the baseline is chaotic (corr(no-op TTC, d_ttc) = -0.55);
        # on calm corridors its own metering injects waves. When ``ttc_gate``
        # is on, fairness/recover/vacate meter only while the corridor-TTC EMA
        # is at/above ``ttc_gate_lo``; protect (safety) and coast are exempt.
        self.ttc_gate_enabled = bool(mode_config.get("ttc_gate", False))
        self.ttc_gate_lo = float(mode_config.get("ttc_gate_lo", 1.0))
        self.ttc_gate_modes = frozenset(
            str(m)
            for m in mode_config.get("ttc_gate_modes") or ("fair_balance", "recover", "vacate_drop")
        )
        # Meter-cap / V2V coordination (2026-08-06): cap concurrent metering
        # CAVs (``meter_cap``, None = unlimited). ``_metering_load`` is reset
        # once per env step (start of apply_all).
        mc = mode_config.get("meter_cap")
        self.meter_cap = int(mc) if mc not in (None, "", 0) else None
        self.meter_cap_modes = frozenset(
            str(m)
            for m in mode_config.get("meter_cap_modes")
            or ("fair_balance", "recover", "vacate_drop")
        )
        self._metering_load = 0
        # Follower-aware execution (user steer 2026-08-05): the mode DECISION
        # stays discrete, but a braking mode's magnitude adapts to what's
        # behind the CAV — full strength only when the road behind is open,
        # gentler (factor toward 1.0) when a close follower would inherit
        # the brake as a compression wave. Attacks the wave-injection
        # mechanism directly instead of the action-selection layer.
        self.follower_aware_exec = bool(mode_config.get("follower_aware_exec", False))
        self.follower_gentle_cap = float(mode_config.get("follower_gentle_cap", 0.97))
        self.follower_gap_near = float(mode_config.get("follower_gap_near", 15.0))
        self.follower_gap_far = float(mode_config.get("follower_gap_far", 60.0))
        # Easy-corridor coast-force: if |G| and delay are already ≤ paired
        # no_op (+slack) for ``coast_force_hold_steps`` consecutive updates,
        # remap listed modes to coast (default: fair_balance + protect only —
        # recover/vacate stay available for hard-seed discharge).
        self.coast_force_enabled = bool(mode_config.get("coast_force", False))
        self.coast_force_delay_slack = float(mode_config.get("coast_force_delay_slack", 0.0))
        cf_modes = mode_config.get("coast_force_modes")
        if cf_modes is None:
            # None → legacy "all non-coast"; empty tuple → disabled list.
            self.coast_force_modes: Optional[frozenset] = None
        else:
            self.coast_force_modes = frozenset(str(m) for m in cf_modes)
        self.coast_force_hold_steps = max(int(mode_config.get("coast_force_hold_steps", 1)), 1)
        # Don't arm until this many env steps (lets hard seeds intervene early).
        self.coast_force_min_step = max(int(mode_config.get("coast_force_min_step", 0)), 0)
        self._coast_force_healthy_streak = 0
        self._step_count = 0
        dref = mode_config.get("delay_rate_ref")
        self.delay_rate_ref: Optional[float] = float(dref) if dref is not None else None
        self.recover_merge_clear = float(mode_config.get("recover_merge_clear", 10.0))
        self.recover_down_clear = float(mode_config.get("recover_down_clear", 18.0))
        self.recover_ego_max = float(mode_config.get("recover_ego_max", 12.0))
        self.recover_push = float(mode_config.get("recover_push", 1.5))
        self.recover_v_cap = float(mode_config.get("recover_v_cap", 18.0))
        # Recover sign gate (2026-08-04): recover is a corrective tool for a
        # MAIN-disadvantaged corridor. Firing it when the main stream is not
        # behind (gap <= tau) is unfair-making by construction — the ablation
        # showed recover is load-bearing on wins but harmful on near-fair /
        # ramp-disadvantaged seeds. Only fires when gap > recover_gap_tau.
        self.recover_gap_sign_gate = bool(mode_config.get("recover_gap_sign_gate", True))
        self.recover_gap_tau = float(
            mode_config.get("recover_gap_tau", mode_config.get("fair_balance_gap_tau", 0.05))
        )
        self.vacate_duration = float(mode_config.get("vacate_lane_duration", 3.0))
        self.vacate_soft = float(mode_config.get("vacate_soft_factor", 0.90))
        self.protect_ttc = float(mode_config.get("protect_ttc", 1.5))
        self.protect_factor = float(mode_config.get("protect_factor", 0.85))
        self.protect_tau_delta = float(mode_config.get("protect_tau_delta", 0.5))

        self.pre_edge = str(mode_config.get("pre_edge", PRE_MERGE_EDGE))
        self.merge_edge = str(mode_config.get("merge_edge", MERGE_EDGE))
        self.ramp_edge = str(mode_config.get("ramp_edge", RAMP_EDGE))
        self.down_edge = str(mode_config.get("down_edge", "1487606178"))
        self.drop_lane = int(mode_config.get("drop_lane", MERGE_DROPPED_LANE_INDEX))
        self.survive_lane = int(mode_config.get("survive_lane", MERGE_SURVIVING_LANE_INDEX))

        self.ema_state: Dict[str, float] = {}
        self._tau_cache: Dict[str, float] = {}
        self.last_mask_fired: Dict[str, bool] = {}
        self.last_ineligible: Dict[str, bool] = {}

    def reset(self) -> None:
        self.ema_state.clear()
        self._tau_cache.clear()
        self._tau_applied.clear()
        self._exec_prev.clear()
        self.last_mask_fired.clear()
        self.last_ineligible.clear()
        self.ema_main_delay = None
        self.ema_ramp_delay = None
        self._coast_force_healthy_streak = 0
        self._step_count = 0
        self._corridor_ttc_ema = 0.0
        self.hard_brake_steps = 0

    def _ramped_tau(self, cav_id: str, target_tau: float) -> float:
        """Move the installed tau toward ``target_tau`` by at most
        ``tau_ramp_step`` per step (T1: no instant gap jump -> no
        self-induced shockwave)."""
        cur = self._tau_applied.get(cav_id)
        if cur is None:
            try:
                cur = float(traci.vehicle.getTau(cav_id))
            except Exception:
                cur = self._original_tau(cav_id)
        step = max(0.0, float(self.tau_ramp_step))
        if target_tau > cur:
            nxt = min(target_tau, cur + step)
        elif target_tau < cur:
            nxt = max(target_tau, cur - step)
        else:
            nxt = target_tau
        self._tau_applied[cav_id] = nxt
        return nxt

    def _gap_limited_tau(self, cav_id: str, current_speed: float, target_tau: float) -> float:
        """Cap the installed tau by the gap the CAV has ACTUALLY achieved
        (T1b, 2026-08-15, user design): the CAV brakes very lightly and
        KEEPS braking until the measured gap supports the big tau. If the
        tau demanded more gap than exists, SUMO's own car-following would
        brake hard to satisfy it — the exact hand-brake we must not have.
        tau_eff <= gap / v means the CF safe-speed never drops below the
        commanded speed, so no hard brake from either channel."""
        try:
            info = leader_info(cav_id, DEFAULT_LOOK_AHEAD)
        except Exception:
            info = None
        if info is None or current_speed <= 0.5:
            return target_tau  # no leader / stopped: ramp toward target
        _, distance, _ = info
        gap_tau = max(0.5, distance / max(current_speed, 1.0))
        return min(target_tau, gap_tau)

    def _install_tau(self, cav_id: str, target_tau: float, current_speed: float = 0.0) -> float:
        """setTau through the gap limit + ramp; track hard brakes."""
        tau_eff = self._gap_limited_tau(cav_id, current_speed, target_tau)
        tau_eff = self._ramped_tau(cav_id, tau_eff)
        try:
            traci.vehicle.setTau(cav_id, tau_eff)
        except Exception:
            pass
        return tau_eff

    def update_stream_delay_rates(
        self, main_rate: Optional[float], ramp_rate: Optional[float]
    ) -> None:
        """EMA of per-step corridor delay rates for fair_balance gating / reward."""
        a = min(max(self.fair_ema_alpha, 0.0), 1.0)
        if main_rate is not None:
            if self.ema_main_delay is None:
                self.ema_main_delay = float(main_rate)
            else:
                self.ema_main_delay = (1.0 - a) * self.ema_main_delay + a * float(main_rate)
        if ramp_rate is not None:
            if self.ema_ramp_delay is None:
                self.ema_ramp_delay = float(ramp_rate)
            else:
                self.ema_ramp_delay = (1.0 - a) * self.ema_ramp_delay + a * float(ramp_rate)
        # One streak / step tick per env step (this method is called once per step).
        self._step_count += 1
        # Corridor-TTC EMA ticks here too (once-per-step guarantee); shared by
        # adaptive gentleness and the recover TTC gate.
        if (
            self.adaptive_gentle or self.recover_ttc_gate or self.ttc_gate_enabled
        ) and self.corridor_ttc_fn is not None:
            try:
                c = self.corridor_ttc_fn()
            except Exception:
                c = None
            if c is not None:
                a = min(max(self.adaptive_gentle_ema_alpha, 0.0), 1.0)
                self._corridor_ttc_ema = (1.0 - a) * self._corridor_ttc_ema + a * float(c)
        self.update_coast_force_streak()

    def stream_delay_gap(self) -> Optional[float]:
        """Signed gap ema_main - ema_ramp (None if either side missing)."""
        if self.ema_main_delay is None or self.ema_ramp_delay is None:
            return None
        return float(self.ema_main_delay - self.ema_ramp_delay)

    def set_stream_delay_gap_ref(self, gap_ref: float) -> None:
        """Update paired no_op |G| baseline used by fair_balance eligibility."""
        self.stream_delay_gap_ref = max(float(gap_ref), 0.0)

    def easy_intervention_blocked(self) -> bool:
        """True when the corridor is already fair, so intervening is pure risk.

        Absolute semantics (user steer 2026-08-04): blocked iff
        |gap| <= easy_intervention_mask_gap_abs ("really fair"). When
        gap_abs <= 0, falls back to the no_op-relative rule |gap| <= gap_ref.
        """
        gap = self.stream_delay_gap()
        if gap is None:
            return False
        if self.easy_intervention_mask_gap_abs > 0.0:
            return abs(float(gap)) <= self.easy_intervention_mask_gap_abs
        return abs(float(gap)) <= max(float(self.stream_delay_gap_ref), 0.0)

    def easy_intervention_discount(self) -> Optional[float]:
        """Effective mask value for easy-gated modes, or None if ungated.

        Hybrid override: when ``easy_intervention_mask_hard_below`` > 0 and
        |gap| is at or under it, returns 0.0 (hard ban: mask bit 0 + apply()
        remaps to coast) regardless of step/graduated mode.
        Step mode (graduated off): ``discount`` when blocked, else None.
        Graduated mode: linear fade — ``discount`` at |gap| <= gap_abs,
        rising to 1.0 (returned as None = ungated) at |gap| >= gap_outer.
        The fairer the corridor, the stronger the logit penalty.
        """
        if not (self.easy_intervention_mask_enabled and self.easy_intervention_mask_modes):
            return None
        if self.easy_intervention_mask_hard_below > 0.0:
            gap = self.stream_delay_gap()
            if gap is not None and abs(float(gap)) <= self.easy_intervention_mask_hard_below:
                return 0.0
        d = self.easy_intervention_mask_discount
        if not self.easy_intervention_mask_graduated:
            return d if self.easy_intervention_blocked() else None
        gap = self.stream_delay_gap()
        if gap is None:
            return None
        g = abs(float(gap))
        inner = max(self.easy_intervention_mask_gap_abs, 0.0)
        outer = max(self.easy_intervention_mask_gap_outer, inner)
        if g >= outer:
            return None
        if g <= inner or outer <= inner:
            return d
        # inner < g < outer: linear interpolation d -> 1.0
        return d + (1.0 - d) * (g - inner) / (outer - inner)

    def met_health_discount(self) -> Optional[float]:
        """Effective discount for metastable-health-gated modes, or None.

        Experiment C (2026-08-11): the RSU computes merge-zone health h in
        [0,1] each step (1 = metastable upper branch, free coast; 0 = collapse
        imminent). Graduated soft mask following the user's dynamic design:
        HEALTHY merge (h high) -> strong logit penalty on metering/recovery
        (the policy is pushed to coast / protect); as h falls toward collapse
        the discount fades to 1.0 (None = ungated) so the CAV can meter and
        recover to restore metastable. ``discount`` at h >= h_hi, None at
        h <= h_lo, linear fade between.
        """
        if not (self.met_health_mask_enabled and self.met_health_mask_modes):
            return None
        if self.met_health_fn is None:
            return None
        try:
            h = float(self.met_health_fn())
        except Exception:
            return None
        if h != h or h < 0.0 or h > 1.0:
            return None
        d = self.met_health_mask_discount
        hi = max(self.met_health_mask_h_hi, self.met_health_mask_h_lo)
        lo = min(self.met_health_mask_h_lo, hi)
        if h >= hi or hi <= lo:
            return d
        if h <= lo:
            return None  # sickest: ungated (meter freely to restore)
        # lo < h < hi: linear fade 1.0 (open) -> d (healthy penalty).
        return 1.0 - (1.0 - d) * (h - lo) / (hi - lo)

    def feasibility_discount(self) -> Optional[float]:
        """Effective discount for feasibility-gated modes, or None.

        Feasibility mask (2026-08-13): the RSU computes the runway margin
        margin = L_pre - L_min(rho) each step, where L_min(rho) is the
        minimum upstream runway needed for sparse-CAV control to hold the
        merge metastable at density rho (fitted from the measured density x
        length controllability grid). INFEASIBLE (margin <= lo) -> strong
        logit penalty on the discretionary modes (coast — acting is futile);
        feasible (margin >= hi) -> None (ungated); linear fade between.
        """
        if not (self.feasibility_mask_enabled and self.met_health_mask_modes):
            return None
        if self.feas_fn is None:
            return None
        try:
            margin = float(self.feas_fn())
        except Exception:
            return None
        if margin != margin:  # NaN
            return None
        d = self.feasibility_mask_discount
        hi = max(self.feasibility_margin_hi, self.feasibility_margin_lo)
        lo = min(self.feasibility_margin_lo, hi)
        if margin <= lo or hi <= lo:
            return d  # infeasible: coast pressure
        if margin >= hi:
            return None  # feasible: ungated
        # lo < margin < hi: linear fade d (infeasible penalty) -> 1.0 (open).
        return d + (1.0 - d) * (margin - lo) / (hi - lo)

    def set_delay_rate_ref(self, delay_rate_ref: Optional[float]) -> None:
        """Update paired no_op corridor delay-rate baseline for coast-force."""
        if delay_rate_ref is None:
            self.delay_rate_ref = None
        else:
            self.delay_rate_ref = max(float(delay_rate_ref), 0.0)

    def fair_balance_gap_threshold(self) -> float:
        """Minimum |G| to allow fair_balance: G_ref + tau (closes reward dead band)."""
        return float(self.stream_delay_gap_ref) + float(self.fair_gap_tau)

    def corridor_is_healthy(self) -> bool:
        """True when live stream gap and delay are already at/better than no_op.

        Requires both EMAs so we never coast-force on missing signal (hard
        episodes often look "calm" for the first few steps).
        """
        gap = self.stream_delay_gap()
        if gap is None or abs(gap) > self.fair_balance_gap_threshold():
            return False
        if self.delay_rate_ref is None:
            return False
        if self.ema_main_delay is None or self.ema_ramp_delay is None:
            return False
        live = 0.5 * (float(self.ema_main_delay) + float(self.ema_ramp_delay))
        return live <= float(self.delay_rate_ref) + float(self.coast_force_delay_slack)

    def update_coast_force_streak(self) -> bool:
        """Advance healthy streak; return True when coast-force is armed."""
        if not self.coast_force_enabled:
            self._coast_force_healthy_streak = 0
            return False
        if self.corridor_is_healthy():
            self._coast_force_healthy_streak += 1
        else:
            self._coast_force_healthy_streak = 0
        return self._coast_force_healthy_streak >= self.coast_force_hold_steps

    def coast_force_armed(self) -> bool:
        """Whether the hold streak is currently long enough (no side effects)."""
        if not self.coast_force_enabled:
            return False
        if self._step_count < self.coast_force_min_step:
            return False
        return (
            self._coast_force_healthy_streak >= self.coast_force_hold_steps
            and self.corridor_is_healthy()
        )

    def _coast_force_blocks(
        self,
        mode_name: str,
        cav_id: Optional[str] = None,
        current_speed: Optional[float] = None,
    ) -> bool:
        """True if coast-force should remap this mode to coast.

        ``protect`` is exempt when local TTC/PET eligibility fires — mode-mix
        showed mid-episode protect is what CF was wiping on lost hard seeds.
        """
        if mode_name == "coast" or not self.coast_force_armed():
            return False
        listed = True if self.coast_force_modes is None else mode_name in self.coast_force_modes
        if not listed:
            return False
        if (
            mode_name == "guard"
            and cav_id is not None
            and current_speed is not None
            and self._protect_eligible(cav_id, float(current_speed))
        ):
            return False
        return True

    def _original_tau(self, cav_id: str) -> float:
        tau = self._tau_cache.get(cav_id)
        if tau is None:
            try:
                tau = float(traci.vehicle.getTau(cav_id))
            except Exception:
                tau = self.base_tau
            self._tau_cache[cav_id] = tau
        return tau

    def _apply_coast(self, cav_id: str) -> None:
        """Return full control to SUMO and restore the CAV's original tau."""
        try:
            traci.vehicle.setTau(cav_id, self._original_tau(cav_id))
            traci.vehicle.setSpeed(cav_id, -1)
        except Exception:
            pass
        # Sync the tau-ramp tracker with the restored (smaller) tau so the
        # next hold ramps up from the true baseline, not the stale big value.
        try:
            self._tau_applied[cav_id] = self._original_tau(cav_id)
        except Exception:
            pass
        self.ema_state.pop(cav_id, None)
        self._exec_prev.pop(cav_id, None)

    # Alias for older call sites / tests
    _apply_non_intervention = _apply_coast

    def _exec_cmd(self, cav_id: str, current_speed: float, v_target: float) -> float:
        """Execution-profile reference for the literature probes (eval-time).

        Returns the rate-limited, EMA-smoothed command speed. When
        ``look_behind_exec`` is on the reference is the upstream (behind) band
        mean speed instead of the mode's own target. Pure helper (no TraCI)
        so the ramp logic is unit-testable.
        """
        ref = v_target
        if self.look_behind_exec:
            vb = self._behind_band_mean_speed(cav_id)
            if vb is not None:
                ref = vb
        prev = self._exec_prev.get(cav_id, current_speed)
        a = min(max(self.smooth_ema_alpha, 0.0), 1.0)
        ref_s = a * ref + (1.0 - a) * prev
        self._exec_prev[cav_id] = ref_s
        return _rate_limit(
            current_speed,
            ref_s,
            self.dt,
            a_max_dec=self.smooth_a_max_dec,
            a_max_acc=self.smooth_a_max_acc,
            deadband=self.smooth_deadband,
        )

    def _dissolve_target(self, cav_id: str, current_speed: float):
        """Soft, lead-windowed dissolve. Returns a gentle target speed when the
        CAV is positioned to dissolve safely, or None to ABSTAIN (coast and let
        the wave pass to the next CAV upstream).

        Looks FAR downstream (``dissolve_look``) for a slow leader = the wave
        front. If the front is within the usable window [lead_min, look] the CAV
        decelerates gradually (target interpolated free -> wave-speed as it
        approaches). If the front is CLOSER than lead_min (too late to react
        gently) or absent, it abstains — acting then would brake hard and
        amplify, so it waits for a better-positioned (further-upstream) CAV.
        """
        snap = get_active_snapshot()
        # nearest slow leader ahead (wave front) and its distance
        front_dist = front_speed = None
        if snap is not None and snap.has(cav_id):
            lane = snap.lane_id(cav_id)
            pos = snap.lane_pos(cav_id)
            if lane is not None and pos is not None:
                best = None
                for vid in snap.ordered_ids():
                    if vid == cav_id or snap.lane_id(vid) != lane:
                        continue
                    p = snap.lane_pos(vid)
                    if p is None or p <= pos:
                        continue
                    d = p - pos
                    if d > self.dissolve_look:
                        continue
                    sp = snap.speed(vid)
                    if sp is not None and sp < self.dissolve_slow_v:
                        if best is None or d < best[0]:
                            best = (d, float(sp))
                if best:
                    front_dist, front_speed = best[0], best[1]
        free = float(self.dissolve_recover_v)
        if front_dist is None:
            return None  # no wave ahead -> abstain/coast
        if front_dist < self.dissolve_lead_min:
            return None  # too close to react gently -> let it pass
        # usable window: gentle target scaled by proximity within [lead_min, look]
        span = max(self.dissolve_look - self.dissolve_lead_min, 1e-6)
        frac = 1.0 - min(1.0, max(0.0, (front_dist - self.dissolve_lead_min) / span))
        frac = float(frac)  # 0 = at min lead (near wave), 1 = far
        wave_spd = max(0.0, front_speed - 0.5)
        return free - (free - wave_spd) * frac

    def _slow_ahead_within(self, cav_id: str) -> bool:
        """True if any same-lane vehicle ahead (within ``dissolve_look``) is
        slow — i.e. a wave can still reach this CAV even though it is not
        inside the dissolving lead window."""
        snap = get_active_snapshot()
        if snap is None or not snap.has(cav_id):
            return False
        lane = snap.lane_id(cav_id)
        pos = snap.lane_pos(cav_id)
        if lane is None or pos is None:
            return False
        for vid in snap.ordered_ids():
            if vid == cav_id or snap.lane_id(vid) != lane:
                continue
            p = snap.lane_pos(vid)
            if p is None or p <= pos:
                continue
            d = p - pos
            if d > self.dissolve_look:
                continue
            sp = snap.speed(vid)
            if sp is not None and sp < self.dissolve_slow_v:
                return True
        return False

    def _merge_match_target(self, cav_id: str, current_speed: float) -> Optional[float]:
        """Merge-stream smoothing (2026-08-08).

        Returns a target speed matching the CURRENT merge-zone speed when a
        merge bottleneck is present, so the CAV begins to gently absorb the
        merge-induced wave LONG BEFORE it reaches the join — provided it is
        within the buffered RSU range (``merge_match_comm``) and far enough
        upstream to react. Returns None to fall through to normal control.
        """
        if not self.merge_match_exec:
            return None
        ms = _edge_mean_speed(self.merge_edge, min_n=3)
        if ms is None:
            return None
        if ms >= self.merge_match_min_speed:
            return None  # merge is flowing free -> nothing to match
        dist = distance_to_merge(cav_id, self.merge_edge, self.pre_edge)
        if dist is None or dist > self.merge_match_comm or dist <= 0.0:
            return None  # out of RSU range, or already at/past the join
        target = max(ms, self.merge_match_floor)  # sane slow floor
        if target >= current_speed - self.smooth_deadband:
            return None  # already at/below the match speed -> nothing to do
        return target

    def _band_speed_around(self, cav_id: str) -> Optional[tuple]:
        """Mean speed of the nearest ``jad_nbr_n`` vehicles ahead and behind
        (same edge, within ``jad_look``) from the RSU snapshot view."""
        snap = get_active_snapshot()
        if snap is None or not snap.has(cav_id):
            return None
        lane = snap.lane_id(cav_id)
        pos = snap.lane_pos(cav_id)
        if lane is None or pos is None:
            return None
        ahead = []
        behind = []
        for vid in snap.ordered_ids():
            if vid == cav_id or snap.lane_id(vid) != lane:
                continue
            p = snap.lane_pos(vid)
            if p is None:
                continue
            d = p - pos
            sp = snap.speed(vid)
            if sp is None:
                continue
            if 0.0 < d <= self.jad_look:
                ahead.append((d, float(sp)))
            elif -self.jad_look <= d < 0.0:
                behind.append((-d, float(sp)))
        ahead.sort()
        behind.sort()
        ahead = [sp for _, sp in ahead[: self.jad_nbr_n]]
        behind = [sp for _, sp in behind[: self.jad_nbr_n]]
        if not ahead or not behind:
            return None
        return (sum(ahead) / len(ahead), sum(behind) / len(behind))

    def _wave_at_cav(self, cav_id: str, current_speed: float) -> bool:
        """True if a backward-propagating shockwave has reached this CAV:
        traffic ahead is slow AND the CAV is slowed relative to the free-flow
        behind it (the wave front has passed into the upstream string)."""
        band = self._band_speed_around(cav_id)
        if band is None:
            return False
        ahead_mean, behind_mean = band
        return (
            ahead_mean < self.jad_slow_thresh and current_speed < behind_mean - self.jad_ego_delta
        )

    def _select_absorber_and_opener(self, action_dict) -> None:
        """Per-step selection for the JAD and gap-select probes.

        JAD absorber = the most-UPSTREAM CAV where a backward-propagating wave
        has arrived (``_wave_at_cav``) — the wave's front, with free flow
        behind. Falls back to the nearest-to-merge CAV when the merge is
        congested but no wave front has been flagged yet. This catches both
        merge-origin waves and waves created by a CAV's own action.
        """
        self._jad_absorber = None
        self._gap_opener = None
        if not (self.jad_exec or self.gap_select_exec) or not action_dict:
            return
        snap = get_active_snapshot()
        if self.jad_exec:
            front = None
            front_dist = None
            for cid in action_dict:
                d = distance_to_merge(cid, self.merge_edge, self.pre_edge)
                if d is None or d <= 0.0:
                    continue
                sp = None
                if snap is not None and snap.has(cid):
                    sp = snap.speed(cid)
                else:
                    try:
                        sp = traci.vehicle.getSpeed(cid)
                    except Exception:
                        sp = None
                if sp is None or not self._wave_at_cav(cid, float(sp)):
                    continue
                if front_dist is None or d > front_dist:  # most upstream front
                    front, front_dist = cid, d
            if front is not None:
                self._jad_absorber = front
            else:
                # fallback: merge congested but no flagged front yet
                ms = _edge_mean_speed(self.merge_edge, min_n=3)
                if ms is not None and ms < self.jad_congest_thresh:
                    best = None
                    bd = None
                    for cid in action_dict:
                        d = distance_to_merge(cid, self.merge_edge, self.pre_edge)
                        if d is None or d <= 0.0:
                            continue
                        if bd is None or d < bd:
                            best, bd = cid, d
                    self._jad_absorber = best
        if self.gap_select_exec:
            op = None
            op_pos = None
            for cid in action_dict:
                if snap is not None:
                    if snap.road(cid) != self.pre_edge:
                        continue
                    p = snap.lane_pos(cid)
                else:
                    try:
                        if traci.vehicle.getRoadID(cid) != self.pre_edge:
                            continue
                        p = traci.vehicle.getLanePosition(cid)
                    except Exception:
                        continue
                if p is not None and (op_pos is None or p > op_pos):
                    op, op_pos = cid, p
            self._gap_opener = op

    def _ramp_vehicle_near_merge(self) -> bool:
        """True if any ramp vehicle is within ``gap_select_range`` of the join."""
        try:
            ramp_len = traci.edge.getLength(self.ramp_edge)
        except Exception:
            ramp_len = 400.0
        snap = get_active_snapshot()
        if snap is not None:
            for vid in snap.ordered_ids():
                if snap.road(vid) != self.ramp_edge:
                    continue
                p = snap.lane_pos(vid)
                if p is not None and p > ramp_len - self.gap_select_range:
                    return True
            return False
        try:
            for vid in traci.vehicle.getIDList():
                if traci.vehicle.getRoadID(vid) == self.ramp_edge:
                    if traci.vehicle.getLanePosition(vid) > ramp_len - self.gap_select_range:
                        return True
        except Exception:
            pass
        return False

    def _cmd_speed(self, cav_id: str, current_speed: float, v_target: float, tau: float) -> None:
        # Merge-stream smoothing probe: match the merge zone speed long before
        # the join, gently. Only engages when a merge bottleneck is in range;
        # otherwise falls through to the normal mode execution.
        if self.merge_match_exec and not self.dissolve_exec:
            mt = self._merge_match_target(cav_id, current_speed)
            if mt is not None:
                v_cmd = _rate_limit(
                    current_speed,
                    mt,
                    self.dt,
                    a_max_dec=self.smooth_a_max_dec,
                    a_max_acc=self.smooth_a_max_acc,
                    deadband=self.smooth_deadband,
                )
                v_cmd = _safe_speed_from_leader(cav_id, current_speed, v_cmd, tau)
                try:
                    traci.vehicle.setTau(cav_id, tau)
                except Exception:
                    pass
                _set_speed_safe(cav_id, max(0.0, min(v_cmd, self.max_speed)))
                return
        # Proper-JAD probe (2026-08-09): ONE absorbing CAV holds an absorbing
        # speed while upstream CAVs extend LARGE time gaps (support driving).
        if self.jad_exec and not (self.dissolve_exec or self.merge_match_exec):
            ms = _edge_mean_speed(self.merge_edge, min_n=3)
            if ms is not None and ms < self.jad_congest_thresh:
                if cav_id == self._jad_absorber:
                    v_cmd = _rate_limit(
                        current_speed,
                        self.jad_absorb_speed,
                        self.dt,
                        a_max_dec=self.smooth_a_max_dec,
                        a_max_acc=self.smooth_a_max_acc,
                        deadband=self.smooth_deadband,
                    )
                    v_cmd = _safe_speed_from_leader(cav_id, current_speed, v_cmd, tau)
                    try:
                        traci.vehicle.setTau(cav_id, tau)
                    except Exception:
                        pass
                    _set_speed_safe(cav_id, max(0.0, min(v_cmd, self.max_speed)))
                    return
                if self._jad_absorber is not None:
                    dab = distance_to_merge(cav_id, self.merge_edge, self.pre_edge)
                    dabs = distance_to_merge(self._jad_absorber, self.merge_edge, self.pre_edge)
                    if dab is not None and dabs is not None and dab > dabs:
                        # upstream SUPPORT: large time gap, follow normally.
                        try:
                            traci.vehicle.setTau(
                                cav_id, self._original_tau(cav_id) + self.jad_support_tau
                            )
                            traci.vehicle.setSpeed(cav_id, -1)
                        except Exception:
                            pass
                        self.ema_state.pop(cav_id, None)
                        self._exec_prev.pop(cav_id, None)
                        return
            # not congested (or downstream): fall through to normal control.
        # Gap-selection / strategic-slowdown probe (2026-08-09): the mainline
        # CAV nearest the merge opens a gap when a ramp vehicle approaches.
        if self.gap_select_exec and not (
            self.dissolve_exec or self.merge_match_exec or self.jad_exec
        ):
            if self._gap_opener is not None and cav_id == self._gap_opener:
                if self._ramp_vehicle_near_merge():
                    g_target = max(self.gap_select_floor, current_speed * self.gap_select_factor)
                    v_cmd = _rate_limit(
                        current_speed,
                        g_target,
                        self.dt,
                        a_max_dec=self.smooth_a_max_dec,
                        a_max_acc=self.smooth_a_max_acc,
                        deadband=self.smooth_deadband,
                    )
                    v_cmd = _safe_speed_from_leader(cav_id, current_speed, v_cmd, tau)
                    try:
                        traci.vehicle.setTau(cav_id, tau)
                    except Exception:
                        pass
                    _set_speed_safe(cav_id, max(0.0, min(v_cmd, self.max_speed)))
                    return
        if self.dissolve_exec:
            # Mechanism probe (2026-08-08): Stern-style reactive wave absorption.
            ref = self._dissolve_target(cav_id, current_speed)
            if ref is None:
                # ABSTAIN: no wave in the usable lead window (absent, too far
                # upstream, or too close to react gently). Coast and let it pass
                # so a better-positioned upstream CAV dissolves it. If a slow
                # leader is still within look (this CAV is upstream of the
                # dissolving region), act as a string-stable SUPPORT vehicle:
                # extend our time gap to damp the absorber's wake, then coast.
                if self.dissolve_support_tau > 0 and self._slow_ahead_within(cav_id):
                    try:
                        traci.vehicle.setTau(
                            cav_id,
                            self._original_tau(cav_id) + self.dissolve_support_tau,
                        )
                        traci.vehicle.setSpeed(cav_id, -1)
                    except Exception:
                        pass
                    self.ema_state.pop(cav_id, None)
                    self._exec_prev.pop(cav_id, None)
                else:
                    self._apply_coast(cav_id)
                return
            v_cmd = _rate_limit(
                current_speed,
                ref,
                self.dt,
                a_max_dec=self.smooth_a_max_dec,
                a_max_acc=self.smooth_a_max_acc,
                deadband=self.smooth_deadband,
            )
        else:
            v_cmd = _rate_limit(
                current_speed,
                v_target,
                self.dt,
                a_max_dec=self.mode_max_decel,
                a_max_acc=self.mode_max_accel,
                deadband=self.mode_deadband,
            )
        if self.smooth_exec or self.look_behind_exec:
            # Literature probe (2026-08-06): execute the mode as a slow smooth
            # ramp, optionally referenced to the upstream (behind) band speed
            # instead of the mode's own target — the CAV absorbs upstream
            # compression instead of injecting its own abrupt speed change.
            v_cmd = self._exec_cmd(cav_id, current_speed, v_target)
        else:
            self._exec_prev.pop(cav_id, None)
        # Install the tau through the gap limit + ramp FIRST so the
        # leader-gap clamp below uses the same gradually-growing gap
        # requirement (T1b 2026-08-15: the installed tau never exceeds what
        # the achieved gap supports — a tau that demands more gap than
        # exists would make SUMO's CF brake hard to satisfy it, the exact
        # hand-brake the design must not produce).
        if self.tau_ramp_step > 0.0:
            tau_eff = self._install_tau(cav_id, tau, current_speed)
        else:
            tau_eff = tau
            try:
                traci.vehicle.setTau(cav_id, tau)
            except Exception:
                pass
        v_cmd = _safe_speed_from_leader(cav_id, current_speed, v_cmd, tau_eff)
        # T1b (2026-08-15): the leader-gap clamp is ALSO rate-limited — with
        # a big tau the required gap is ~60 m and the clamp could otherwise
        # cut the commanded speed by many m/s in ONE step. Clamp to the same
        # gentle decel envelope; SUMO's own safe-speed logic (RL_SPEED_MODE)
        # still prevents a real collision.
        #
        # Shockwave-formation audit: count steps where the ACTUAL executed
        # decel exceeded the gentle envelope (measured on the commanded
        # speed after rate-limiting — the physical brake the followers
        # experience). With the gap-limited tau this should stay ~0.
        v_cmd = _rate_limit(
            current_speed,
            v_cmd,
            self.dt,
            a_max_dec=self.mode_max_decel,
            a_max_acc=self.mode_max_accel,
            deadband=self.mode_deadband,
        )
        executed_decel = (current_speed - v_cmd) / max(self.dt, 1e-6)
        if executed_decel > self.mode_max_decel + 0.05:
            self.hard_brake_steps += 1
        _set_speed_safe(cav_id, max(0.0, min(v_cmd, self.max_speed)))

    def _meter_cap_blocks(self, mode_name: str) -> bool:
        """Pure decision for the meter-cap / V2V coordination gate: True when
        this CAV should coast because the concurrent metering load is at cap.
        Protect (safety) and coast are exempt."""
        if self.meter_cap is None:
            return False
        if mode_name == "coast" or mode_name == "guard":
            return False
        if mode_name not in self.meter_cap_modes:
            return False
        return self._metering_load >= self.meter_cap

    def _ttc_gate_blocks(self, mode_name: str) -> bool:
        """Pure decision for the benefit gate (2026-08-06): True when the mode
        should be remapped to coast because the corridor is too calm. Protect
        (safety) and coast are exempt."""
        if not self.ttc_gate_enabled:
            return False
        if mode_name == "coast" or mode_name == "guard":
            return False
        if mode_name not in self.ttc_gate_modes:
            return False
        return self._corridor_ttc_ema < self.ttc_gate_lo

    def apply(
        self,
        cav_id: str,
        mode: int,
        cav_ids: Optional[set] = None,
        mag: Optional[float] = None,
    ) -> tuple[bool, bool]:
        """Execute one mode.

        ``mag`` (hybrid action space, 2026-08-05): optional execution
        magnitude in [0, 1]; None = legacy fixed strengths. m=1 is full
        champion strength, m=0 is no-op. See the mode laws for mappings.

        Returns ``(mask_fired, ineligible)``:
          - mask_fired: hard health mask remapped to coast
          - ineligible: non-coast mode selected but gate remapped to coast
        """
        snap = get_active_snapshot()
        if snap is not None:
            if not snap.has(cav_id):
                return False, False
            current_speed = snap.speed(cav_id)
            if current_speed is None:
                return False, False
            current_speed = float(current_speed)
        else:
            if cav_id not in traci.vehicle.getIDList():
                return False, False
            current_speed = float(traci.vehicle.getSpeed(cav_id))

        mi = int(mode)
        mode_name = self.mode_names[mi] if 0 <= mi < len(self.mode_names) else "coast"

        if mode_name in self.disabled_modes:
            self._apply_coast(cav_id)
            return False, True

        # Do-no-harm: if corridor already matches paired no_op on G and delay
        # for long enough, block listed intervention modes (protect exempted
        # when local TTC/PET eligibility is true).
        if self._coast_force_blocks(mode_name, cav_id, current_speed):
            self._apply_coast(cav_id)
            return True, False

        if (
            self.health_mask_enabled
            and mode_name in self.health_mask_modes
            and self.health_fn is not None
        ):
            try:
                health = float(self.health_fn())
            except Exception:
                health = None
            if health is not None and health >= self.health_mask_threshold:
                self._apply_coast(cav_id)
                return True, False

        if (
            self.fairness_mask_enabled
            and mode_name in self.fairness_mask_modes
            and self.fairness_fn is not None
        ):
            try:
                fair = float(self.fairness_fn())
            except Exception:
                fair = None
            # Block unless fairness is clearly worse than the paired ref.
            if fair is not None and fair >= -self.fairness_mask_threshold:
                self._apply_coast(cav_id)
                return True, False

        easy_disc = self.easy_intervention_discount()
        if (
            easy_disc is not None
            and easy_disc <= 0.0
            and mode_name in self.easy_intervention_mask_modes
        ):
            self._apply_coast(cav_id)
            return True, False

        # Metastable-health soft mask (Experiment C, 2026-08-11): the RSU
        # health discount is SOFT — the mode stays physically allowed, the
        # logits carry the penalty. Only a hard 0.0 discount remaps to coast
        # (that case = mask bit 0, and the model cannot legally select it);
        # anything > 0 passes through untouched here so the executed action
        # matches the sampled one (no gradient mismatch).
        mh_disc = self.met_health_discount()
        feas_disc = self.feasibility_discount()
        if (
            (mh_disc is not None and mh_disc <= 0.0) or (feas_disc is not None and feas_disc <= 0.0)
        ) and mode_name in self.met_health_mask_modes:
            self._apply_coast(cav_id)
            return True, False

        # Benefit gate (2026-08-06): meter only where the corridor is already
        # disturbed (corridor-TTC EMA >= ttc_gate_lo). On calm corridors the
        # CAV's own metering injects the waves (corr(no-op TTC, d_ttc)=-0.55).
        # Protect is exempt (safety); coast is always legal.
        if self._ttc_gate_blocks(mode_name):
            self._apply_coast(cav_id)
            return True, False

        # Meter-cap / V2V coordination gate (2026-08-06): cap the number of
        # CAVs metering concurrently. Each metering CAV injects a wave source;
        # at high penetration (p30 over-injects) capping the load keeps the
        # baseline-smoothing benefit of more CAVs while limiting the injection.
        # Protect (safety) is exempt.
        if self._meter_cap_blocks(mode_name):
            self._apply_coast(cav_id)
            return True, False
        if mode_name in self.meter_cap_modes:
            self._metering_load += 1

        if mode_name == "coast":
            self._apply_coast(cav_id)
            return False, False

        # Hybrid magnitude: a near-zero magnitude on an execution mode is a
        # no-op — release control to SUMO (coast) rather than speed-locking.
        if mag is not None and float(mag) < 0.05:
            self._apply_coast(cav_id)
            return False, False

        if mode_name == "fair_balance":
            ineligible = self._fair_balance(cav_id, current_speed, mag=mag)
        elif mode_name == "recover":
            ineligible = self._recover_speed(cav_id, current_speed, mag=mag)
        elif mode_name == "vacate_drop":
            ineligible = self._vacate_drop(cav_id, current_speed, mag=mag)
        elif mode_name == "guard":
            ineligible = self._protect(cav_id, current_speed, mag=mag)
        elif mode_name == "wave_damp":
            ineligible = self._wave_damp(cav_id, current_speed)
        elif mode_name == "merge_match":
            ineligible = self._merge_match_mode(cav_id, current_speed)
        elif mode_name == "jad_absorb":
            ineligible = self._jad_absorb_mode(cav_id, current_speed)
        elif mode_name == "jad_support":
            ineligible = self._jad_support_mode(cav_id, current_speed)
        else:
            self._apply_coast(cav_id)
            ineligible = True
        return False, bool(ineligible)

    def apply_all(
        self, action_dict: Dict[str, object], cav_ids: Optional[set] = None
    ) -> Dict[str, bool]:
        fired: Dict[str, bool] = {}
        inelig: Dict[str, bool] = {}
        self._metering_load = 0  # per-step cap for V2V coordination
        # Per-step selections for the JAD / gap-select mechanism probes.
        self._select_absorber_and_opener(action_dict)
        for cav_id, action in action_dict.items():
            # Hybrid values arrive as (mode, mag) pairs; plain ints = legacy.
            if isinstance(action, (int, np.integer)):
                mode, mag = int(action), None
            else:
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                mode, mag = int(arr[0]), float(arr[1])
            mask, bad = self.apply(cav_id, mode, cav_ids=cav_ids, mag=mag)
            fired[cav_id] = bool(mask)
            inelig[cav_id] = bool(bad)
        self.last_mask_fired = fired
        self.last_ineligible = inelig
        if cav_ids is not None:
            stale = [vid for vid in self._tau_cache if vid not in cav_ids]
            for vid in stale:
                del self._tau_cache[vid]
        return fired

    # -- eligibility (no side effects) ---------------------------------------
    def _current_speed(self, cav_id: str) -> Optional[float]:
        snap = get_active_snapshot()
        if snap is not None:
            if not snap.has(cav_id):
                return None
            sp = snap.speed(cav_id)
            return float(sp) if sp is not None else None
        try:
            if cav_id not in traci.vehicle.getIDList():
                return None
            return float(traci.vehicle.getSpeed(cav_id))
        except Exception:
            return None

    def _fair_balance_eligible(self, cav_id: str, current_speed: float) -> bool:
        """Near tip on the advantaged stream when |G| exceeds no_op ref + tau.

        Threshold is ``stream_delay_gap_ref + fair_balance_gap_tau`` so metering
        is illegal in the dead band where |G| > absolute tau but R_fair is still
        zero (no excess over the paired no_op baseline).
        """
        gap = self.stream_delay_gap()
        if gap is None or abs(gap) <= self.fair_balance_gap_threshold():
            return False
        edge, _, rem = _lane_role(cav_id)
        if rem is None or rem > self.fair_tip:
            return False
        # gap > 0 → main more delayed → slow ramp; gap < 0 → slow main
        if gap > 0:
            if edge != self.ramp_edge:
                return False
        else:
            if edge not in (self.pre_edge, self.merge_edge):
                return False
        if self.guard_speed > 0.0 and current_speed < self.guard_speed:
            return False
        return True

    def _recover_eligible(self, cav_id: str, current_speed: float) -> bool:
        edge, _, _ = _lane_role(cav_id)
        if edge != self.pre_edge or current_speed >= self.recover_ego_max:
            return False
        if self.recover_gap_sign_gate:
            # Recover only when the MAIN stream is actually behind: firing it
            # on a fair or ramp-disadvantaged corridor widens the gap.
            gap = self.stream_delay_gap()
            if gap is not None and float(gap) <= self.recover_gap_tau:
                return False
        if self.recover_ttc_gate:
            # Recover accelerations inject TTC events into smooth flow; keep
            # recover only where the corridor is already chaotic (EMA >= lo).
            if self._corridor_ttc_ema < self.recover_ttc_lo:
                return False
        ms = _edge_mean_speed(self.merge_edge, min_n=3)
        ds = _edge_mean_speed(self.down_edge, min_n=3)
        if ms is None or ds is None:
            return False
        if ms < self.recover_merge_clear or ds < self.recover_down_clear:
            return False
        return True

    def _vacate_eligible(self, cav_id: str, current_speed: float) -> bool:
        del current_speed
        edge, li, _ = _lane_role(cav_id)
        return edge == self.merge_edge and li == self.drop_lane

    def _protect_eligible(self, cav_id: str, current_speed: float) -> bool:
        del current_speed
        try:
            if leader_ttc(cav_id) < self.protect_ttc:
                return True
            if side_ttc(cav_id) < self.protect_ttc:
                return True
            if merge_arrival_pet(cav_id) < self.protect_ttc:
                return True
        except Exception:
            return False
        return False

    def eligibility_mask(self, cav_id: str) -> np.ndarray:
        """Per-mode legality for action masking: 1=allowed, 0=blocked.

        Sized by the run's mode set (``mode_count``, default 5): wave_damp
        is registered globally but only emits a bit when the run opted in.
        """
        n_modes = int(getattr(self, "mode_count", 5))
        mask = np.zeros(n_modes, dtype=np.float32)
        mask[int(Mode.COAST)] = 1.0
        speed = self._current_speed(cav_id)
        if speed is None:
            return mask

        if self.coast_force_armed():
            # Only coast is legal for blocked modes; leave others to eligibility.
            if self.coast_force_modes is None:
                return mask

        checks = {
            "fair_balance": self._fair_balance_eligible,
            "recover": self._recover_eligible,
            "vacate_drop": self._vacate_eligible,
            "guard": self._protect_eligible,
            # wave_damp: always eligible (the do-no-harm mode, like coast).
            "wave_damp": lambda cav_id, speed: True,
            # merge_match: always eligible (coast-like no-op when no merge
            # congestion). Opt-in 7th mode.
            "merge_match": lambda cav_id, speed: True,
            # JAD modes: absorb only at the wave front (free-flow behind);
            # support when a wave is forming. Opt-in 9th/10th modes.
            "jad_absorb": lambda cav_id, speed: self._wave_front_eligible(cav_id, speed),
            "jad_support": lambda cav_id, speed: wave_risk_signal(cav_id)[1] > 0,
        }
        health_block = False
        if self.health_mask_enabled and self.health_fn is not None and self.health_mask_modes:
            try:
                health = float(self.health_fn())
                health_block = health >= self.health_mask_threshold
            except Exception:
                health_block = False

        fair_block = False
        if self.fairness_mask_enabled and self.fairness_fn is not None and self.fairness_mask_modes:
            try:
                fair = float(self.fairness_fn())
                # Same rule as apply(): block unless clearly worse than ref.
                fair_block = fair >= -self.fairness_mask_threshold
            except Exception:
                fair_block = False

        easy_disc = self.easy_intervention_discount()
        mh_disc = self.met_health_discount()
        feas_disc = self.feasibility_discount()

        for name, fn in checks.items():
            if name not in self.mode_names:
                continue  # not in this run's mode set (e.g. pruned champion)
            idx = self.mode_names.index(name)
            if idx >= len(mask):
                continue  # registered but not in this run's mode set
            if name in self.disabled_modes:
                continue
            if health_block and name in self.health_mask_modes:
                continue
            if fair_block and name in self.fairness_mask_modes:
                continue
            if self._coast_force_blocks(name, cav_id, speed):
                continue
            if not fn(cav_id, speed):
                continue
            if easy_disc is not None and name in self.easy_intervention_mask_modes:
                # Hard (discount 0) blocks outright; soft/graduated emits the
                # discount so the model only discounts the logits of this mode.
                if easy_disc <= 0.0:
                    continue
                mask[idx] = easy_disc
            elif mh_disc is not None and name in self.met_health_mask_modes:
                # Metastable-health mask (Experiment C): same graduated soft
                # semantics — healthy merge → discount close to 0.05 (strong
                # logit penalty, coast pressure); sick merge → 1.0 (open).
                # A hard 0.0 would block outright (model cannot select).
                if mh_disc <= 0.0:
                    continue
                mask[idx] = mh_disc
            elif feas_disc is not None and name in self.met_health_mask_modes:
                # Feasibility mask (2026-08-13): runway margin below the
                # minimum-requirement band → coast pressure on the same
                # discretionary modes (control is futile on this highway at
                # this density); feasible → 1.0 (open).
                if feas_disc <= 0.0:
                    continue
                mask[idx] = feas_disc
            else:
                mask[idx] = 1.0
        return mask

    # -- gated mode laws -----------------------------------------------------
    def _follower_gap(self, cav_id: str) -> Optional[float]:
        """Bumper gap (m) to the closest vehicle directly behind on the same
        lane, or None when no follower / no snapshot."""
        snap = get_active_snapshot()
        if snap is None or not snap.has(cav_id):
            return None
        lane = snap.lane_id(cav_id)
        pos = snap.lane_pos(cav_id)
        if lane is None or pos is None:
            return None
        best: Optional[float] = None
        for vid in snap.ordered_ids():
            if vid == cav_id:
                continue
            if snap.lane_id(vid) != lane:
                continue
            p = snap.lane_pos(vid)
            if p is None or p >= pos:
                continue
            gap = pos - p - 5.0  # vehicle length
            if best is None or gap < best:
                best = gap
        return best

    def _follower_aware_factor(self, cav_id: str, base_factor: float) -> float:
        """Scale a braking factor by follower proximity: full strength when
        the road behind is open (gap >= far), gentle cap when a close
        follower (gap <= near) would inherit the brake as a wave."""
        if not self.follower_aware_exec:
            return base_factor
        gap = self._follower_gap(cav_id)
        if gap is None:
            return base_factor
        near, far = self.follower_gap_near, self.follower_gap_far
        cap = self.follower_gentle_cap
        if gap <= near:
            return cap
        if gap >= far:
            return base_factor
        frac = (gap - near) / max(far - near, 1e-9)
        return cap + frac * (base_factor - cap)

    def _current_fair_factor(self) -> float:
        """fair_balance slowdown factor, TTC-adaptive when enabled.

        Smooth corridor (TTC EMA <= lo) -> gentle factor (milder slowdown,
        fewer injected compression waves); chaotic (>= hi) -> full-strength
        fair_factor; linear interpolation between.
        """
        if not self.adaptive_gentle:
            return self.fair_factor
        lo, hi = self.adaptive_gentle_ttc_lo, self.adaptive_gentle_ttc_hi
        ema = self._corridor_ttc_ema
        if ema <= lo:
            return self.adaptive_gentle_factor
        if ema >= hi:
            return self.fair_factor
        frac = (ema - lo) / max(hi - lo, 1e-9)
        return self.adaptive_gentle_factor + frac * (self.fair_factor - self.adaptive_gentle_factor)

    def _fair_balance(self, cav_id: str, current_speed: float, mag: Optional[float] = None) -> bool:
        if not self._fair_balance_eligible(cav_id, current_speed):
            self._apply_coast(cav_id)
            return True
        if mag is not None:
            # Hybrid dial, PER-MODE range (2026-08-11): metering is the
            # density-control lever — it must be able to REALLY hold back the
            # flow at the merge (m=1 -> 0.80 strong metering), while m=0.5 ~
            # champion's 0.90 and m=0 -> 1.0 no-op. The modest tau bump
            # (+0.25) lets the follower absorb the brake without a shockwave.
            factor = 1.0 - float(mag) * 0.20
        else:
            factor = self._follower_aware_factor(cav_id, self._current_fair_factor())
        v_target = max(self.fair_floor, current_speed * factor)
        tau = max(0.1, self._original_tau(cav_id) + self.tau_deltas.get("fair_balance", 0.25))
        self._cmd_speed(cav_id, current_speed, v_target, tau)
        return False

    def _recover_speed(
        self, cav_id: str, current_speed: float, mag: Optional[float] = None
    ) -> bool:
        if not self._recover_eligible(cav_id, current_speed):
            self._apply_coast(cav_id)
            return True
        # Hybrid dial: push in [0, 2.0] m/s^2 (legacy fixed 1.5 ~ m=0.75).
        push = float(mag) * 2.0 if mag is not None else self.recover_push
        v_target = min(current_speed + push * self.dt, self.recover_v_cap)
        tau = max(0.1, self._original_tau(cav_id) + self.tau_deltas.get("recover", -0.2))
        self._cmd_speed(cav_id, current_speed, v_target, tau)
        return False

    def _vacate_drop(self, cav_id: str, current_speed: float, mag: Optional[float] = None) -> bool:
        if not self._vacate_eligible(cav_id, current_speed):
            self._apply_coast(cav_id)
            return True
        try:
            traci.vehicle.setLaneChangeMode(cav_id, 1621)
            traci.vehicle.changeLane(cav_id, self.survive_lane, self.vacate_duration)
        except Exception:
            pass
        # Hybrid dial: soft factor in [0.90, 1.0].
        soft = 1.0 - float(mag) * 0.10 if mag is not None else self.vacate_soft
        v_target = current_speed * soft
        tau = max(0.1, self._original_tau(cav_id) + self.tau_deltas.get("vacate_drop", 0.0))
        self._cmd_speed(cav_id, current_speed, v_target, tau)
        return False

    def _protect(self, cav_id: str, current_speed: float, mag: Optional[float] = None) -> bool:
        if not self._protect_eligible(cav_id, current_speed):
            self._apply_coast(cav_id)
            return True
        if mag is not None:
            # Hybrid dial, PER-MODE range (2026-08-11): protect is a SAFETY
            # action whose job is to CREATE HEADWAY, not to slam the brakes.
            # The dial scales the tau bump (bigger tau = bigger safe gap,
            # which SUMO's car-following translates into a gentle, wave-free
            # slowdown): m=1 -> +protect_tau_delta (full headway extension),
            # m=0.5 -> half. The speed factor stays mild (0.97..1.0) — the
            # gap, not the brake, is what protects.
            tau_delta = float(self.protect_tau_delta) * float(mag)
            factor = 1.0 - 0.03 * float(mag)
        else:
            tau_delta = self.protect_tau_delta
            factor = self.protect_factor
        v_target = current_speed * factor
        tau = max(0.1, self._original_tau(cav_id) + tau_delta)
        self._cmd_speed(cav_id, current_speed, v_target, tau)
        return False

    def _ahead_band_mean_speed(self, cav_id: str) -> Optional[float]:
        """Mean speed of vehicles on the same lane in [near, far] ahead
        (driving direction = larger lane position). Snapshot-only."""
        snap = get_active_snapshot()
        if snap is None or not snap.has(cav_id):
            return None
        lane = snap.lane_id(cav_id)
        pos = snap.lane_pos(cav_id)
        if lane is None or pos is None:
            return None
        speeds = []
        for vid in snap.ordered_ids():
            if vid == cav_id or snap.lane_id(vid) != lane:
                continue
            p = snap.lane_pos(vid)
            if p is None:
                continue
            d = p - pos
            if self.flow_stab_near <= d <= self.flow_stab_far:
                s = snap.speed(vid)
                if s is not None:
                    speeds.append(float(s))
        return float(sum(speeds) / len(speeds)) if speeds else None

    def _behind_band_mean_speed(self, cav_id: str) -> Optional[float]:
        """Mean speed of vehicles on the same lane in [near, far] BEHIND
        (upstream; driving direction = larger lane position). Snapshot-only.
        Mirrors ``_ahead_band_mean_speed`` for the LCC "looking-behind"
        topology: the CAV references the traffic that will inherit its
        braking instead of the traffic ahead of it."""
        snap = get_active_snapshot()
        if snap is None or not snap.has(cav_id):
            return None
        lane = snap.lane_id(cav_id)
        pos = snap.lane_pos(cav_id)
        if lane is None or pos is None:
            return None
        speeds = []
        for vid in snap.ordered_ids():
            if vid == cav_id or snap.lane_id(vid) != lane:
                continue
            p = snap.lane_pos(vid)
            if p is None:
                continue
            d = pos - p  # positive => behind
            if self.behind_near <= d <= self.behind_far:
                s = snap.speed(vid)
                if s is not None:
                    speeds.append(float(s))
        return float(sum(speeds) / len(speeds)) if speeds else None

    def _wave_damp(self, cav_id: str, current_speed: float) -> bool:
        """Flow stabilization (research_v2 port): match the EMA of ahead-band
        speeds with smooth rate-limited commands — the CAV absorbs
        compression waves instead of amplifying them. Anticipates slower
        traffic by preview_gain when the band is slower than ego. Always
        eligible (the do-no-harm mode); returns False (never ineligible)."""
        v_down = self._ahead_band_mean_speed(cav_id)
        if v_down is None:
            v_down = current_speed  # no info, hold
        elif v_down < current_speed - self.flow_stab_deadband:
            v_down = max(0.0, v_down - self.flow_stab_preview)
        prev = self.ema_state.get(cav_id, v_down)
        a = min(max(self.flow_stab_ema_alpha, 0.0), 1.0)
        v_smooth = a * v_down + (1.0 - a) * prev
        self.ema_state[cav_id] = v_smooth
        v_cmd = _rate_limit(
            current_speed,
            v_smooth,
            self.dt,
            a_max_dec=self.flow_stab_a_max_dec,
            a_max_acc=self.flow_stab_a_max_acc,
            deadband=self.flow_stab_deadband,
        )
        v_cmd = max(0.0, min(v_cmd, self.max_speed))
        tau = max(0.1, self._original_tau(cav_id) + self.tau_deltas.get("wave_damp", 0.0))
        try:
            traci.vehicle.setTau(cav_id, tau)
        except Exception:
            pass
        _set_speed_safe(cav_id, v_cmd)
        return False

    def _merge_match_mode(self, cav_id: str, current_speed: float) -> bool:
        """Merge-stream smoothing mode (opt-in 7th, 2026-08-08).

        When a merge bottleneck is present and the CAV is within the buffered
        RSU range, smoothly rate-limit down to the merge-zone speed LONG BEFORE
        arrival (absorbing the merge-origin wave). When no merge congestion or
        out of range, it coasts (do-no-harm) — like wave_damp, always eligible.
        """
        mt = self._merge_match_target(cav_id, current_speed)
        if mt is None:
            self._apply_coast(cav_id)
            return False
        v_cmd = _rate_limit(
            current_speed,
            mt,
            self.dt,
            a_max_dec=self.smooth_a_max_dec,
            a_max_acc=self.smooth_a_max_acc,
            deadband=self.smooth_deadband,
        )
        v_cmd = _safe_speed_from_leader(cav_id, current_speed, v_cmd, self._original_tau(cav_id))
        try:
            traci.vehicle.setTau(cav_id, self._original_tau(cav_id))
        except Exception:
            pass
        _set_speed_safe(cav_id, max(0.0, min(v_cmd, self.max_speed)))
        return False

    def _wave_front_eligible(self, cav_id: str, current_speed: float) -> bool:
        """JAD absorb eligibility (trained mode): a wave is forming at this CAV
        (compound RSU signal) AND the traffic behind is still free-flow — i.e.
        this CAV is at the wave's upstream FRONT (the single-absorber position).
        """
        risk, forming = wave_risk_signal(cav_id)
        if not forming or risk <= 0.0:
            return False
        band = self._band_speed_around(cav_id)
        if band is None:
            return False
        return band[1] > 15.0  # behind is free-flow -> front boundary

    def _jad_absorb_mode(self, cav_id: str, current_speed: float) -> bool:
        """JAD absorbing mode (opt-in 9th, 2026-08-09): the CAV at the wave
        front holds the absorbing speed (rate-limited) so no new vehicles feed
        the wave. Coasts (do-no-harm) when not at a real wave front."""
        if not self._wave_front_eligible(cav_id, current_speed):
            self._apply_coast(cav_id)
            return False
        v_cmd = _rate_limit(
            current_speed,
            self.jad_absorb_speed,
            self.dt,
            a_max_dec=self.smooth_a_max_dec,
            a_max_acc=self.smooth_a_max_acc,
            deadband=self.smooth_deadband,
        )
        v_cmd = _safe_speed_from_leader(cav_id, current_speed, v_cmd, self._original_tau(cav_id))
        try:
            traci.vehicle.setTau(cav_id, self._original_tau(cav_id))
        except Exception:
            pass
        _set_speed_safe(cav_id, max(0.0, min(v_cmd, self.max_speed)))
        return False

    def _jad_support_mode(self, cav_id: str, current_speed: float) -> bool:
        """JAD support mode (opt-in 10th, 2026-08-09): when a wave is forming
        ahead, extend a LARGE time gap (string-stable support driving) and
        follow normally, damping the secondary upstream propagation."""
        risk, forming = wave_risk_signal(cav_id)
        if not forming or risk <= 0.0:
            self._apply_coast(cav_id)
            return False
        try:
            traci.vehicle.setTau(cav_id, self._original_tau(cav_id) + self.jad_support_tau)
            traci.vehicle.setSpeed(cav_id, -1)
        except Exception:
            pass
        return False

    # Legacy helper retained for older unit tests that call it directly.
    def _gentle_mode(self, cav_id: str, current_speed: float, mode_name: str) -> None:
        factor = self.speed_factors.get(mode_name, 1.0)
        if factor is None:
            return
        v_target = max(0.0, min(current_speed * float(factor), self.max_speed))
        tau = max(0.1, self.base_tau + self.tau_deltas.get(mode_name, 0.0))
        self._cmd_speed(cav_id, current_speed, v_target, tau)


# ---------------------------------------------------------------------------
# Continuous longitudinal acceleration (Selfish / Flat-MARL)
# ---------------------------------------------------------------------------
def _as_scalar_acceleration(
    action: object,
    acceleration_min: float = ACCELERATION_MIN,
    acceleration_max: float = ACCELERATION_MAX,
) -> float:
    value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    return float(np.clip(value, acceleration_min, acceleration_max))


def apply_continuous_acceleration(
    cav_id: str,
    action: object,
    dt: float = 1.0,
    max_speed: float = 35.0,
    speed_mode: int = RL_SPEED_MODE,
    acceleration_min: float = ACCELERATION_MIN,
    acceleration_max: float = ACCELERATION_MAX,
) -> None:
    """Apply continuous acceleration: ``v_next = clip(v + a * dt)``."""
    snap = get_active_snapshot()
    if snap is not None:
        if not snap.has(cav_id):
            return
        current_speed = snap.speed(cav_id)
        if current_speed is None:
            return
        current_speed = float(current_speed)
    else:
        if cav_id not in traci.vehicle.getIDList():
            return
        try:
            current_speed = traci.vehicle.getSpeed(cav_id)
        except Exception:
            return

    try:
        acceleration = _as_scalar_acceleration(action, acceleration_min, acceleration_max)
        traci.vehicle.setSpeedMode(cav_id, int(speed_mode))
    except (TypeError, ValueError, IndexError):
        return

    next_speed = max(0.0, min(float(max_speed), current_speed + acceleration * dt))
    try:
        traci.vehicle.setSpeed(cav_id, next_speed)
    except Exception:
        try:
            traci.vehicle.slowDown(cav_id, next_speed, duration=dt)
        except Exception:
            pass


def apply_continuous_accelerations(
    action_dict: Dict[str, object],
    dt: float = 1.0,
    max_speed: float = 35.0,
    speed_mode: int = RL_SPEED_MODE,
    acceleration_min: float = ACCELERATION_MIN,
    acceleration_max: float = ACCELERATION_MAX,
) -> None:
    for cav_id, action in action_dict.items():
        apply_continuous_acceleration(
            cav_id,
            action,
            dt=dt,
            max_speed=max_speed,
            speed_mode=speed_mode,
            acceleration_min=acceleration_min,
            acceleration_max=acceleration_max,
        )
