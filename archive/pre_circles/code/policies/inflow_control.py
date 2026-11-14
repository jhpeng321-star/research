"""R2 inflow-control mechanisms (2026-08-15, user-directed).

The Phase-1..5 investigation established the core physics: flow past a point
is 1/(time headway), so slowing isolated CAVs does NOT reduce merge inflow —
the HDV follower keeps its own tau and the stream re-converges. Controlling
the inflow needs a mechanism that either (a) throttles EVERY lane (an
all-lane CAV wall — no lane to route around), (b) affects HDVs directly
(road-level VSL, which SUMO HDVs obey), or (c) gates the flow with a
duty cycle (periodic release).

All three are implemented here as DIRECT-CONTROL policies (they apply TraCI
commands themselves and return {} so the env skips the discrete modes):

  WallHoldPolicy     — CAVs within a longitudinal band [d_lo, d_hi] of the
                       merge hold ``wall_speed`` in EVERY lane (the band spans
                       all lanes by construction — distance_to_merge is
                       lane-independent). The wall is a moving bottleneck:
                       flow through it = wall_speed / headway, and HDVs cannot
                       bypass because every lane is blocked at that position.
                       Optional ``wall_tau`` raises the CAV headway inside the
                       wall (the big-tau idea applied to the wall itself).
  VSLPolicy          — sets the pre-edge edge speed limit to ``vsl_limit``
                       (m/s) via traci.edge.setMaxSpeed — affects ALL vehicles
                       including HDVs (SUMO HDVs obey limits). Sustained for
                       the whole episode.
  GatePolicy         — duty-cycle release: for G seconds the wall holds a
                       crawl speed (near-zero outflow), then for R seconds it
                       releases (coast). Average inflow = duty * capacity.

The inflow counter (``count_merge_inflow``) is separate: every step it counts
vehicles crossing from the pre-edge onto the merge edge (the physical join),
so the probe row reports the ACTUAL merge inflow in veh/h per variant — the
measurement that was missing from every prior phase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

import traci

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, RAMP_EDGE
from env.features import distance_to_merge
from env.snapshot import get_active_snapshot
from policies import DirectControlPolicy
from policies.traci_control import configure_cav, release_to_sumo


def _rate_limited(
    v_ego: float,
    v_target: float,
    dt: float,
    a_max_dec: float = 0.3,
    a_max_acc: float = 1.0,
    deadband: float = 0.5,
) -> float:
    """Same semantics as env.actions._rate_limit: brake gently (0.3 m/s² by
    default — the user's light-brake envelope), accelerate slowly."""
    diff = v_target - v_ego
    if abs(diff) < deadband:
        return v_ego
    if diff < 0:
        return max(v_target, v_ego - a_max_dec * dt)
    return min(v_target, v_ego + a_max_acc * dt)


class _InflowCounterMixin:
    """Count vehicles entering the merge edge each step (the physical join)."""

    def __init__(self) -> None:
        self.inflow_history: List[float] = []  # veh/h per step
        self._seen_on_merge: Set[str] = set()

    def _reset_inflow(self) -> None:
        self.inflow_history = []
        self._seen_on_merge = set()

    def _count_inflow(self, dt: float = 1.0) -> None:
        snap = get_active_snapshot()
        if snap is None:
            return
        now = set()
        for v in snap.ordered_ids():
            if snap.road(v) == MERGE_EDGE:
                now.add(v)
        # New arrivals = on merge now and never seen before.
        new = now - self._seen_on_merge
        self._seen_on_merge |= now
        if dt > 0:
            self.inflow_history.append(len(new) * 3600.0 / dt)


class WallHoldPolicy(DirectControlPolicy, _InflowCounterMixin):
    """All-lane moving wall: every CAV in the band [d_lo, d_hi] upstream of
    the merge holds ``wall_speed`` (m/s), across all lanes simultaneously.

    Design notes:
      - The band is lane-independent (distance_to_merge), so at any instant
        the wall spans every lane at the same longitudinal position — an HDV
        cannot route around it without leaving the road.
      - Speed change is rate-limited to ``a_max_dec`` (default 0.3 m/s² — the
        user's light-brake envelope), so the wall forms gently.
      - ``wall_tau`` (optional) raises the CAV's car-following tau inside the
        wall: the CAV holds a bigger headway, stretching the queue upstream.
      - ``arm_time``/``release_time`` (optional) turn it into the duty-cycle
        gate: hold the wall for arm_time steps, coast for release_time.
    """

    direct_control = True

    def __init__(
        self,
        d_lo: float = 200.0,
        d_hi: float = 1500.0,
        wall_speed: float | None = 10.0,
        wall_tau: float = 0.0,
        a_max_dec: float = 0.3,
        tau_ramp_step: float = 0.0,
        arm_time: float = 0.0,
        release_time: float = 0.0,
        trigger_h: float | None = None,
        release_h: float | None = None,
        tau_fraction: float = 1.0,
        assembly_hi: float | None = None,
        assembly_window: float = 500.0,
        assembly_max_adj: float = 4.0,
        assembly_mode: str = "sync",
        trigger_v: float | None = None,
        trigger_k: float | None = None,
        ramp_pace: bool = False,
        pace_margin: float = 3.0,
        max_speed: float = 30.0,
    ):
        super().__init__()
        _InflowCounterMixin.__init__(self)
        self.d_lo = float(d_lo)
        self.d_hi = float(d_hi)
        # wall_speed None = tau-only mode (hold the big headway, no speed cmd)
        self.wall_speed = None if wall_speed is None else float(wall_speed)
        self.wall_tau = float(wall_tau)
        # tau_fraction (2026-08-18, user's high-penetration question): at HIGH
        # penetration the band is already CAV-covered, so only a SUBSET of CAVs
        # needs to hold the big tau (keeping the injection-smoothing benefit)
        # while the rest coast at normal tau — cutting the aggregate stretch
        # delay. The wall SPEED still applies to every CAV in the band (the
        # throttle); the big TAU applies to a deterministic subset. 1.0 = all.
        self.tau_fraction = max(0.0, min(1.0, float(tau_fraction)))
        # Rendezvous-light assembly (2026-08-19, user's on-road formation
        # question): CAVs in the pre-band zone (d_hi, assembly_hi] adjust
        # speed within +/- assembly_max_adj so that CAVs within
        # assembly_window metres of each other ARRIVE at the wall band a few
        # seconds apart; the band's uniform wall_speed then does the final
        # compression into a coherent wall. This is the on-road alternative
        # to block placement for scattered low-penetration populations.
        # None = assembly disabled (default).
        self.assembly_hi = float(assembly_hi) if assembly_hi is not None else None
        self.assembly_window = float(assembly_window)
        self.assembly_max_adj = float(assembly_max_adj)
        # "sync" (wait + speed up) cost ~10x delay and doubled TTC at p=0.1
        # scatter (Phase-13 interim, 2026-08-19): slowing to WAIT for a
        # trailing neighbour makes HDVs pile up and cut in. "catchup" only
        # SPEEDS UP toward the nearest CAV ahead — never waits — so it
        # cannot add delay; weaker grouping but delay-safe.
        if assembly_mode not in ("sync", "catchup"):
            raise ValueError(f"assembly_mode must be sync|catchup, got {assembly_mode}")
        self.assembly_mode = assembly_mode
        # Upstream predictive trigger (Phase-16 lever 2): arm when the mean
        # speed of traffic UPSTREAM of the band (pre-edge, further than
        # d_hi + 200 m from the merge) drops below trigger_v — the incoming
        # platoon is detected 20-60 s before it reaches the merge, so a
        # sparse low-penetration wall is already engaged when the wave
        # arrives (the merge-health trigger fires too late for sparse walls).
        # Release when the upstream speed recovers past trigger_v + 2.
        self.trigger_v = float(trigger_v) if trigger_v is not None else None
        # Density-based upstream trigger (Phase-17 lever 2): the L2 speed
        # trigger failed because at dense demand there is no upstream speed
        # break — but a wave is a high-DENSITY packet even at uniform speed.
        # Arm when the upstream density (pre-edge, beyond d_hi + 200 m, in
        # veh/km/lane) exceeds trigger_k; release below trigger_k - 4.
        self.trigger_k = float(trigger_k) if trigger_k is not None else None
        # Ramp pace (Phase-16 lever 4): CAVs on the ramp edge are paced to
        # the merge-zone mean speed + pace_margin — the injection-dv lever
        # (Stage-7 merge_pace) in direct-control form. The delay is paid by
        # the ramp stream (which expects to wait), not the mainline.
        # Self-gating: when the merge flows freely the pace target is above
        # traffic speed, so it does nothing.
        self.ramp_pace = bool(ramp_pace)
        self.pace_margin = float(pace_margin)
        self._pre_len: float | None = None
        self._pre_lanes: int | None = None
        self._detail_cache = None
        self._detail_step = -1
        self.a_max_dec = float(a_max_dec)
        # Backward-compatible by default: zero keeps the historical instant
        # tau change. Counterfactual escalation uses a positive increment so
        # the desired headway cannot jump from damping tau to the full wall
        # tau in one simulation step.
        self.tau_ramp_step = max(0.0, float(tau_ramp_step))
        self.arm_time = float(arm_time)
        self.release_time = float(release_time)
        # Health-triggered engagement (economics fix, 2026-08-17): the wall
        # only runs while the merge is sick (h < trigger_h); once it recovers
        # past release_h it coasts. This pays the delay ONLY during the
        # threatened window instead of the whole episode — the difference
        # between 110 s/veh and ~15-30 s/veh.
        self.trigger_h = float(trigger_h) if trigger_h is not None else None
        self.release_h = (
            float(release_h)
            if release_h is not None
            else (self.trigger_h + 0.2 if trigger_h is not None else None)
        )
        self._wall_on = False
        self._env = None
        self.max_speed = float(max_speed)
        self._step = 0
        self._phase = "arm" if arm_time > 0 else "always"
        self._phase_count = 0
        self._tau_cache: Dict[str, float] = {}
        self._tau_offset_cache: Dict[str, float] = {}
        self.last_speed_commands_mps: Dict[str, float] = {}
        self.simulation_step = 1.0

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._phase = "arm" if self.arm_time > 0 else "always"
        self._phase_count = 0
        self._wall_on = False
        self._tau_cache = {}
        self._tau_offset_cache = {}
        self.last_speed_commands_mps = {}
        self._reset_inflow()
        self.simulation_step = float(getattr(env, "simulation_step", 1.0))

    def reset_transition(self) -> None:
        """Start the next externally commanded wall pulse from zero tau offset."""
        self._tau_offset_cache = {}

    def restore_tau(self, cav_ids) -> list[dict[str, object]]:
        """Restore cached pre-wall headway without releasing speed control.

        The counterfactual overlay keeps the base damping controller active
        throughout a wall pulse.  Consequently, ending wall membership must
        restore only the wall-specific ``tau`` change; calling
        :func:`release_to_sumo` here would erase the damping speed command and
        violate the causal treatment contract.
        """

        try:
            live_ids = {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        except Exception:
            live_ids = set()
        diagnostics: list[dict[str, object]] = []
        for cav_id in sorted({str(vehicle_id) for vehicle_id in cav_ids}):
            offset = float(self._tau_offset_cache.pop(cav_id, 0.0))
            original_tau = self._tau_cache.get(cav_id)
            restored = False
            error = None
            if cav_id in live_ids and original_tau is not None:
                try:
                    traci.vehicle.setTau(cav_id, float(original_tau))
                    restored = True
                except Exception as exc:
                    error = type(exc).__name__
            diagnostics.append(
                {
                    "vehicle_id": cav_id,
                    "original_tau_s": original_tau,
                    "removed_wall_tau_offset_s": offset,
                    "restored": restored,
                    "error": error,
                }
            )
        return diagnostics

    def set_tau_blend(self, cav_ids, fraction: float) -> list[dict[str, object]]:
        """Apply an exact monotone fraction of the wall-only tau offset.

        This narrow actuator is used only by the versioned moving-wall taper.
        It never releases longitudinal control and never changes the cached
        pre-wall tau, so a zero fraction is exactly the damping/native tau.
        """

        blend = min(1.0, max(0.0, float(fraction)))
        try:
            live_ids = {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        except Exception:
            live_ids = set()
        diagnostics: list[dict[str, object]] = []
        for cav_id in sorted({str(vehicle_id) for vehicle_id in cav_ids}):
            original_tau = self._tau_cache.get(cav_id)
            target_offset = max(0.0, float(self._tau_for(cav_id))) * blend
            applied = False
            error = None
            if cav_id in live_ids and original_tau is not None:
                try:
                    traci.vehicle.setTau(cav_id, float(original_tau) + target_offset)
                    self._tau_offset_cache[cav_id] = target_offset
                    applied = True
                except Exception as exc:
                    error = type(exc).__name__
            diagnostics.append(
                {
                    "vehicle_id": cav_id,
                    "blend_fraction": blend,
                    "original_tau_s": original_tau,
                    "wall_tau_offset_s": target_offset,
                    "applied": applied,
                    "error": error,
                }
            )
        return diagnostics

    def _detail(self):
        """metastable_health_detail cached per step — the accessor's EMA
        advances once per CALL, and the recording wrapper already calls it
        once per step, so this policy must add at most one extra call."""
        if self._detail_step != self._step:
            self._detail_cache = None
            try:
                if self._env is not None:
                    self._detail_cache = self._env.metastable_health_detail()
            except Exception:
                self._detail_cache = None
            self._detail_step = self._step
        return self._detail_cache

    def _health(self) -> float | None:
        """Merge-zone metastable health h in [0,1] (None if unavailable)."""
        d = self._detail()
        if d:
            try:
                return float(d[0])
            except Exception:
                pass
        return None

    def _merge_speed(self) -> float | None:
        """Merge-zone mean speed in m/s (None if unavailable)."""
        d = self._detail()
        if d and len(d) > 2:
            try:
                return float(d[2])
            except Exception:
                pass
        return None

    def _upstream_speed(self) -> float | None:
        """Mean speed of pre-edge traffic UPSTREAM of the band (further than
        d_hi + 200 m from the merge) — the incoming-platoon detector. None
        if no such vehicles or the snapshot is unavailable."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        if self._pre_len is None:
            try:
                self._pre_len = float(traci.lane.getLength(PRE_MERGE_EDGE + "_0"))
            except Exception:
                return None
        vs = []
        try:
            for vid in snap.ordered_ids():
                if snap.road(vid) != PRE_MERGE_EDGE:
                    continue
                lp = snap.lane_pos(vid)
                if lp is None or (self._pre_len - lp) <= self.d_hi + 200.0:
                    continue
                sp = snap.speed(vid)
                if sp is not None:
                    vs.append(sp)
        except Exception:
            return None
        return (sum(vs) / len(vs)) if vs else None

    def _upstream_density(self) -> float | None:
        """Density of pre-edge traffic UPSTREAM of the band (veh/km/lane).
        None if the snapshot or geometry is unavailable."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        if self._pre_len is None or self._pre_lanes is None:
            try:
                lane_ids = [
                    lane_id
                    for lane_id in traci.lane.getIDList()
                    if lane_id.startswith(PRE_MERGE_EDGE)
                ]
                self._pre_lanes = max(1, len(lane_ids))
                self._pre_len = float(traci.lane.getLength(PRE_MERGE_EDGE + "_0"))
            except Exception:
                return None
        n = 0
        try:
            for vid in snap.ordered_ids():
                if snap.road(vid) != PRE_MERGE_EDGE:
                    continue
                lp = snap.lane_pos(vid)
                if lp is not None and (self._pre_len - lp) > self.d_hi + 200.0:
                    n += 1
        except Exception:
            return None
        span_km = (self._pre_len - (self.d_hi + 200.0)) / 1000.0
        if span_km <= 0:
            return None
        return n / (span_km * self._pre_lanes)

    def _wall_active(self) -> bool:
        # Duty-cycle gate mode (arm/release times) takes precedence.
        if self.arm_time > 0:
            self._phase_count += 1
            if self._phase == "arm" and self._phase_count >= self.arm_time:
                self._phase = "release"
                self._phase_count = 0
            elif self._phase == "release" and self._phase_count >= self.release_time:
                self._phase = "arm"
                self._phase_count = 0
            return self._phase == "arm"
        # Density-based upstream trigger: arm when the upstream density
        # exceeds trigger_k (a wave packet), release on clearing.
        if self.trigger_k is not None:
            k = self._upstream_density()
            if k is None:
                return self._wall_on
            if not self._wall_on and k > self.trigger_k:
                self._wall_on = True
            elif self._wall_on and k < self.trigger_k - 4.0:
                self._wall_on = False
            return self._wall_on
        # Upstream predictive trigger: arm when the upstream flow breaks
        # (platoon detected before it reaches the merge), release on recovery.
        if self.trigger_v is not None:
            uv = self._upstream_speed()
            if uv is None:
                return self._wall_on
            if not self._wall_on and uv < self.trigger_v:
                self._wall_on = True
            elif self._wall_on and uv > self.trigger_v + 2.0:
                self._wall_on = False
            return self._wall_on
        # Health-triggered mode: arm when sick, release when healthy
        # (hysteresis: release only after recovery past release_h).
        if self.trigger_h is not None:
            h = self._health()
            if h is None:
                return self._wall_on
            if not self._wall_on and h < self.trigger_h:
                self._wall_on = True
            elif self._wall_on and h > self.release_h:
                self._wall_on = False
            return self._wall_on
        return True

    def _original_tau(self, cav_id: str) -> float:
        if cav_id not in self._tau_cache:
            try:
                self._tau_cache[cav_id] = float(traci.vehicle.getTau(cav_id))
            except Exception:
                self._tau_cache[cav_id] = 1.0
        return self._tau_cache[cav_id]

    def _tau_for(self, cav_id: str) -> float:
        """Big tau to apply to this CAV (tau-fraction subset, 2026-08-18).

        At high penetration the band is already covered, so only a
        deterministic subset of CAVs holds the big tau (keeping the
        injection-smoothing benefit) while the rest coast at normal tau —
        reducing the aggregate stretch delay. Deterministic by ID hash so a
        vehicle keeps the same assignment across steps."""
        if self.wall_tau <= 0 or self.tau_fraction >= 1.0:
            return self.wall_tau
        return self.wall_tau if (abs(hash(cav_id)) % 100) < (self.tau_fraction * 100) else 0.0

    def _ramped_tau_for(self, cav_id: str) -> float:
        """Return the applied wall-tau offset for this step.

        Vehicles that leave the wall are restored by the ordinary damping
        action path. If they later re-enter, the offset starts gently again.
        """
        target = max(0.0, float(self._tau_for(cav_id)))
        if self.tau_ramp_step <= 0.0:
            applied = target
        else:
            previous = max(0.0, float(self._tau_offset_cache.get(cav_id, 0.0)))
            applied = min(target, previous + self.tau_ramp_step)
        self._tau_offset_cache[cav_id] = applied
        return applied

    def _assembly_speed(self, cav_id: str, zone: Dict[str, tuple]) -> float | None:
        """Speed that syncs this CAV's wall-band arrival with its nearest
        zone neighbour (None = no neighbour in the window, just coast).

        Each CAV estimates its time to the band entrance t = (d - d_hi)/v
        and matches the nearest neighbour's estimate: arrive later -> speed
        up, arrive sooner -> wait a little. Adjustments are bounded to
        +/- assembly_max_adj and rate-limited like the wall, so the
        gathering itself cannot inject a shockwave (SUMO's car-following
        safety stays on via speedMode 31)."""
        d, v = zone[cav_id]
        best = None
        for other_id, (d2, v2) in zone.items():
            if other_id == cav_id:
                continue
            if self.assembly_mode == "catchup":
                # only neighbours AHEAD (smaller distance-to-merge)
                if d2 >= d or (d - d2) > self.assembly_window:
                    continue
            elif abs(d2 - d) > self.assembly_window:
                continue
            dist = abs(d2 - d)
            if best is None or dist < best[0]:
                best = (dist, (d2 - self.d_hi) / max(v2, 0.5))
        if best is None:
            return None
        t_i = (d - self.d_hi) / max(v, 0.5)
        t_star = max(best[1], 0.5)
        if self.assembly_mode == "catchup" and t_star >= t_i:
            return None  # the CAV ahead isn't arriving sooner: nothing to catch
        v_des = (d - self.d_hi) / t_star
        lo = v if self.assembly_mode == "catchup" else v - self.assembly_max_adj
        v_des = min(v + self.assembly_max_adj, max(lo, v_des))
        return _rate_limited(v, v_des, self.simulation_step, a_max_dec=self.a_max_dec)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        self._count_inflow(self.simulation_step)
        active = self._wall_active()
        # Pass 1: assembly-zone membership (id -> (distance_to_merge, speed))
        # for the neighbour sync. Only gathered when the wall is engaged.
        zone: Dict[str, tuple] = {}
        if active and self.assembly_hi is not None:
            for cav_id in obs_dict.keys():
                if cav_id not in traci.vehicle.getIDList():
                    continue
                try:
                    d = distance_to_merge(cav_id)
                    if self.d_hi < d <= self.assembly_hi:
                        zone[cav_id] = (d, float(traci.vehicle.getSpeed(cav_id)))
                except Exception:
                    pass
        for cav_id in obs_dict.keys():
            if cav_id not in traci.vehicle.getIDList():
                continue
            configure_cav(cav_id)
            if self.ramp_pace:
                # Ramp-side pacing: hold ramp CAVs at merge-zone speed +
                # margin so the injection dv shrinks. Self-gating — a fast
                # merge sets a pace target above traffic speed (no-op).
                try:
                    if traci.vehicle.getRoadID(cav_id) == RAMP_EDGE:
                        v_m = self._merge_speed()
                        if v_m is not None:
                            v = float(traci.vehicle.getSpeed(cav_id))
                            v_cmd = _rate_limited(
                                v,
                                v_m + self.pace_margin,
                                self.simulation_step,
                                a_max_dec=self.a_max_dec,
                            )
                            traci.vehicle.setSpeed(cav_id, max(0.0, min(v_cmd, self.max_speed)))
                        continue
                except Exception:
                    pass
            if not active:
                release_to_sumo(cav_id, tau=self._original_tau(cav_id))
                continue
            try:
                d = distance_to_merge(cav_id)
            except Exception:
                d = float("inf")
            if not (self.d_lo <= d <= self.d_hi):
                # Assembly zone: sync arrival with nearby CAVs so a scattered
                # population reaches the band as a coherent group.
                if cav_id in zone:
                    v_cmd = self._assembly_speed(cav_id, zone)
                    if v_cmd is not None:
                        try:
                            traci.vehicle.setSpeed(cav_id, max(0.0, min(v_cmd, self.max_speed)))
                            continue
                        except Exception:
                            pass
                release_to_sumo(cav_id, tau=self._original_tau(cav_id))
                continue
            try:
                # TAU-ONLY mode (wall_speed None): hold the big headway but
                # never command a speed — the CAV follows its leader at the
                # enlarged tau, so the headway (not the speed) does the
                # throttling with ~zero added transit delay. Tests whether
                # the collapse benefit comes from tau alone (spacing
                # smoothing) rather than the speed throttle.
                if self.wall_speed is None:
                    tf = self._ramped_tau_for(cav_id)
                    if tf > 0:
                        traci.vehicle.setTau(cav_id, self._original_tau(cav_id) + tf)
                    continue
                traci.vehicle.setTau(cav_id, self._original_tau(cav_id))
                v = float(traci.vehicle.getSpeed(cav_id))
                v_cmd = _rate_limited(
                    v,
                    self.wall_speed,
                    self.simulation_step,
                    a_max_dec=self.a_max_dec,
                )
                traci.vehicle.setSpeed(cav_id, max(0.0, min(v_cmd, self.max_speed)))
                tf = self._ramped_tau_for(cav_id)
                if tf > 0:
                    traci.vehicle.setTau(cav_id, self._original_tau(cav_id) + tf)
            except Exception:
                pass
        return {}

    def act_with_targets(
        self,
        obs_dict: Dict[str, Any],
        speed_targets_mps: Dict[str, float],
    ) -> Dict[str, object]:
        """Apply externally computed per-CAV wall targets.

        The counterfactual overlay owns membership and treatment state.  This
        narrow actuator path preserves WallHoldPolicy's CAV configuration,
        rate limits, tau ramp, and inflow accounting while allowing a compact
        formation controller to use bounded member-specific speed offsets.
        """
        self._step += 1
        self._count_inflow(self.simulation_step)
        self.last_speed_commands_mps = {}
        live_ids = {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        for cav_id in obs_dict:
            if cav_id not in live_ids or cav_id not in speed_targets_mps:
                continue
            configure_cav(cav_id)
            try:
                traci.vehicle.setTau(cav_id, self._original_tau(cav_id))
                speed = float(traci.vehicle.getSpeed(cav_id))
                target = float(speed_targets_mps[cav_id])
                command = _rate_limited(
                    speed,
                    target,
                    self.simulation_step,
                    a_max_dec=self.a_max_dec,
                )
                applied_command = max(0.0, min(command, self.max_speed))
                traci.vehicle.setSpeed(
                    cav_id,
                    applied_command,
                )
                self.last_speed_commands_mps[cav_id] = applied_command
                tau_offset = self._ramped_tau_for(cav_id)
                if tau_offset > 0.0:
                    traci.vehicle.setTau(
                        cav_id,
                        self._original_tau(cav_id) + tau_offset,
                    )
            except Exception:
                continue
        return {}


class VSLPolicy(DirectControlPolicy, _InflowCounterMixin):
    """Sustained road-level VSL: set the pre-edge speed limit once at reset
    (affects ALL vehicles incl. HDVs — SUMO HDVs obey the edge limit), then
    coast. ``vsl_limit`` in m/s (e.g. 10, 12, 14)."""

    direct_control = True

    def __init__(self, vsl_limit: float = 10.0):
        super().__init__()
        _InflowCounterMixin.__init__(self)
        self.vsl_limit = float(vsl_limit)
        self.simulation_step = 1.0

    def reset(self, env, seed: int = None) -> None:
        self._reset_inflow()
        self.simulation_step = float(getattr(env, "simulation_step", 1.0))
        self._applied = False

    def _apply_limit(self) -> None:
        if self._applied:
            return
        try:
            traci.edge.setMaxSpeed(PRE_MERGE_EDGE, self.vsl_limit)
            self._applied = True
        except Exception as exc:
            print(f"  VSL setMaxSpeed failed (retry next step): {exc}", flush=True)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        # Reset runs before the TraCI connection exists (evaluate_policy
        # calls policy.reset(env) before SUMO starts), so apply the limit
        # lazily on the first live step.
        self._apply_limit()
        self._count_inflow(self.simulation_step)
        return {}


class SmoothCruisePolicy(DirectControlPolicy):
    """Wave-filter cruise (2026-08-24, after Stern et al. field experiment +
    I-24/MegaVanderTest): the CAV drives to the low-pass filtered (EMA) velocity
    of its own traffic stream and never mirrors wave-frequency fluctuations —
    damping string-instability feedback at each CAV. Analytic, no learning.
    Optional health gate: with trigger_h, the filter only engages while h is
    below the threshold (the metastable-gated form)."""

    def __init__(
        self,
        tau_ema: float = 30.0,
        trigger_h: float | None = None,
        release_h: float = 0.45,
        smooth_a_max: float = 1.0,
    ):
        super().__init__()
        self.tau_ema = float(tau_ema)
        self.alpha = 1.0 / max(self.tau_ema, 1.0)
        self.trigger_h = trigger_h
        self.release_h = float(release_h)
        self.smooth_a_max = float(smooth_a_max)
        self._env = None
        self._step = 0
        self._v_ema: dict = {}
        self._on = False

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._v_ema = {}
        self._on = False

    def _health(self):
        if self._env is None:
            return None
        try:
            d = self._env.metastable_health_detail()
            return None if d is None else float(d[0])
        except Exception:
            return None

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        if self.trigger_h is not None:
            h = self._health()
            if h is None:
                pass
            elif not self._on and h < self.trigger_h:
                self._on = True
            elif self._on and h > self.release_h:
                self._on = False
            active = self._on
        else:
            active = True
        if not active:
            self._v_ema = {}
            return {}
        snap = get_active_snapshot()
        if snap is None:
            return {}
        from policies.inflow_control import traci as _t

        for cav_id in obs_dict.keys():
            try:
                v = float(_t.vehicle.getSpeed(cav_id))
            except Exception:
                continue
            # Filter the LEADER's speed (the traffic signal), not the CAV's
            # own: own-speed EMA is a positive feedback (slowing lowers the
            # reference -> runaway crawl). The leader's smoothed velocity is
            # the I-24/Stern reference the CAV converges to.
            v_lead = v
            try:
                lead = _t.vehicle.getLeader(cav_id, 80.0)
                if lead and lead[0]:
                    v_lead = float(_t.vehicle.getSpeed(lead[0]))
            except Exception:
                pass
            prev = self._v_ema.get(cav_id, v_lead)
            v_ema = prev + self.alpha * (v_lead - prev)
            self._v_ema[cav_id] = v_ema
            # Free-flow floor: the smoothed reference may lag a transient
            # deceleration, but never sinks below cruise (the runaway case);
            # approach it gently, safety checks stay on.
            v_ref = max(v_ema, 0.75 * 31.0)
            try:
                _t.vehicle.setSpeed(cav_id, max(0.0, min(v_ref, v + self.smooth_a_max)))
            except Exception:
                pass
        return {}


class SpatialCruisePolicy(DirectControlPolicy):
    """Literature-replica smooth cruise (2026-08-24, after Martinez 2023 +
    I-24/MegaVanderTest): the CAV drives to the mean speed of the vehicles in
    the upstream RSU comm horizon (spatial averaging = the anti-wave filter;
    Martinez: multi-leader averaging enables low MPR). Healthy-gated: active
    only while corridor health is ABOVE trigger_h (steady-state free flow —
    the regime the research controllers operate in), latched off below
    release_h (the wall's hand-off threshold). Never fights the onset
    transient: that is the measured failure mode of time-lagged filters."""

    def __init__(
        self,
        horizon_m: float = 1500.0,
        trigger_h: float = 0.4,
        release_h: float = 0.25,
        smooth_a_max: float = 1.0,
    ):
        super().__init__()
        self.horizon_m = float(horizon_m)
        self.trigger_h = float(trigger_h)
        self.release_h = float(release_h)
        self.smooth_a_max = float(smooth_a_max)
        self._env = None
        self._step = 0
        self._on = False

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._on = False

    def _health(self):
        if self._env is None:
            return None
        try:
            d = self._env.metastable_health_detail()
            return None if d is None else float(d[0])
        except Exception:
            return None

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        h = self._health()
        if self.trigger_h is None:
            self._on = True
        elif h is not None:
            if not self._on and h > self.trigger_h:
                self._on = True
            elif self._on and h < self.release_h:
                self._on = False
        if not self._on:
            return {}
        snap = get_active_snapshot()
        if snap is None:
            return {}
        from policies.inflow_control import traci as _t

        # Spatial reference: mean speed of vehicles upstream within the
        # comm horizon (window per CAV, snapshot-based — no per-step TraCI
        # leader chains).
        rows = []
        try:
            for v in snap.ordered_ids():
                sp = snap.speed(v)
                if sp is None:
                    continue
                d = distance_to_merge(v)
                if d is not None:
                    rows.append((d, float(sp)))
        except Exception:
            return {}
        for cav_id in obs_dict.keys():
            try:
                v = float(_t.vehicle.getSpeed(cav_id))
                d0 = distance_to_merge(cav_id)
            except Exception:
                continue
            if d0 is None:
                continue
            win = [sp for d, sp in rows if d0 < d <= d0 + self.horizon_m]
            if not win:
                continue
            v_ref = sum(win) / len(win)
            try:
                _t.vehicle.setSpeed(cav_id, max(0.0, min(v_ref, v + self.smooth_a_max)))
            except Exception:
                pass
        return {}


class GatePolicy(WallHoldPolicy):
    """Duty-cycle gate: arm (crawl wall) for ``arm_time`` steps, release for
    ``release_time`` steps. Crawl speed ``gate_speed`` (default 2 m/s) blocks
    outflow during the red phase; release lets the stored queue drain. The
    average merge inflow = duty-cycle * capacity — a genuine throttle."""

    def __init__(
        self, gate_speed: float = 2.0, arm_time: float = 20.0, release_time: float = 10.0, **kw
    ):
        kw.setdefault("wall_speed", gate_speed)
        kw.setdefault("arm_time", arm_time)
        kw.setdefault("release_time", release_time)
        kw.setdefault("d_lo", 200.0)
        kw.setdefault("d_hi", 2500.0)
        super().__init__(**kw)
        self.gate_speed = float(gate_speed)


class AlineaPolicy(DirectControlPolicy):
    """ALINEA ramp metering (Papageorgiou 1991), SUMO realization.

    Classic integral occupancy feedback at the merge throat:
        r(k+1) = r(k) + K_R * (o_target - o(k))      [veh/h]
    with r clamped to [r_min, r_max]. The metering rate is executed as a
    release budget on the ramp edge: each step budgets r/3600 vehicles past
    the gate; the front-of-queue ramp vehicle is released when the budget
    allows, the rest crawl at gate_speed. Occupancy proxy: vehicles in the
    merge zone times the effective vehicle length over lane-meters.

    The canonical baseline for the wall: compares the corridor's health-gated
    wall against occupancy-feedback ramp metering in the same harness."""

    def __init__(
        self,
        o_target: float = 0.15,
        k_r: float = 70.0,
        r_min: float = 150.0,
        r_max: float = 1000.0,
        gate_speed: float = 3.0,
        ctrl_interval: float = 10.0,
        veh_len: float = 5.0,
        zone_m: float = 400.0,
        gate_m_from_merge: float = 250.0,
    ):
        super().__init__()
        self.o_target = float(o_target)
        self.k_r = float(k_r)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.gate_speed = float(gate_speed)
        self.ctrl_interval = float(ctrl_interval)
        self.veh_len = float(veh_len)
        self.zone_m = float(zone_m)
        self.gate_m = float(gate_m_from_merge)
        self._env = None
        self._step = 0
        self._rate = 400.0  # nominal ramp demand = initial meter rate
        self._budget = 0.0
        self._gated: set = set()
        self._geo = None

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._rate = 400.0
        self._budget = 0.0
        self._gated = set()
        self._geo = None

    def _occupancy(self) -> float | None:
        """Merge-zone occupancy proxy = n*veh_len / (lanes*zone_m)."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        if self._geo is None:
            try:
                from config.constants import MERGE_EDGE, PRE_MERGE_EDGE

                lanes = len(
                    [
                        lane_id
                        for lane_id in traci.lane.getIDList()
                        if lane_id.startswith(MERGE_EDGE)
                    ]
                )
                self._geo = (lanes, PRE_MERGE_EDGE, MERGE_EDGE)
            except Exception:
                return None
        lanes, pre, merge = self._geo
        n = 0
        pre_len = None
        try:
            pre_len = float(traci.lane.getLength(pre + "_0"))
        except Exception:
            pass
        for v in snap.ordered_ids():
            road = snap.road(v)
            if road == merge:
                n += 1
            elif road == pre and pre_len is not None:
                lp = snap.lane_pos(v)
                if lp is not None and (pre_len - lp) <= self.zone_m:
                    n += 1
        if lanes <= 0:
            return None
        return n * self.veh_len / (lanes * self.zone_m)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        dt = float(getattr(self._env, "simulation_step", 1.0))
        # Integral occupancy feedback at the control interval.
        if self._step % max(1, int(self.ctrl_interval / dt)) == 0:
            o = self._occupancy()
            if o is not None:
                self._rate = min(
                    self.r_max, max(self.r_min, self._rate + self.k_r * (self.o_target - o))
                )
        self._budget += self._rate * dt / 3600.0
        snap = get_active_snapshot()
        if snap is None:
            return {}
        from config.constants import RAMP_EDGE
        from env.features import distance_to_merge

        try:
            ramp = sorted(
                (
                    (distance_to_merge(v) or 1e9, v)
                    for v in snap.ordered_ids()
                    if snap.road(v) == RAMP_EDGE
                )
            )
        except Exception:
            ramp = []
        if not ramp:
            return {}
        front_d, front = ramp[0]
        if front_d <= self.gate_m and self._budget >= 1.0:
            # release one vehicle past the gate
            try:
                traci.vehicle.setSpeed(front, -1.0)
                self._budget -= 1.0
                self._gated.discard(front)
            except Exception:
                pass
        # gate (crawl) everything else still on the ramp
        new_gated = set()
        for d, v in ramp:
            if v != front or not (front_d <= self.gate_m and self._budget < 1.0):
                if d <= self.gate_m:
                    try:
                        traci.vehicle.setSpeed(v, self.gate_speed)
                        new_gated.add(v)
                    except Exception:
                        pass
        for v in self._gated - new_gated:
            try:
                traci.vehicle.setSpeed(v, -1.0)
            except Exception:
                pass
        self._gated = new_gated
        return {}


class RampGatePolicy(WallHoldPolicy):
    """Merge-inflow gate (2026-08-24, the user's "block the inflow of the
    merge" mechanism): when corridor health drops below ``trigger_h``, EVERY
    vehicle on the ramp edge — HDVs included — crawls at ``gate_speed``,
    starving the merge of the ramp stream that tips the pinch into breakdown.

    The leak asymmetry is the point: the wall throttles only the 20% CAV
    share of the mainline, so its flow reduction is diluted; the gate
    controls 100% of the ramp flow. ALINEA analog (health-triggered ramp
    metering). Releases on h > release_h. With ``wall_params``, the mainline
    band wall engages simultaneously (ramp_gate_wall composite)."""

    def __init__(
        self,
        trigger_h: float = 0.25,
        release_h: float = 0.45,
        gate_speed: float = 3.0,
        wall_params: dict | None = None,
        **kw,
    ):
        kw.setdefault("d_lo", 200.0)
        kw.setdefault("d_hi", 1500.0)
        kw.setdefault("wall_speed", 10.0)
        kw.setdefault("wall_tau", 0.0)
        super().__init__(trigger_h=trigger_h, release_h=release_h, **kw)
        self.gate_speed = float(gate_speed)
        self._wall = WallHoldPolicy(**wall_params) if wall_params else None
        self._gated: set = set()

    def reset(self, env, seed: int = None) -> None:
        super().reset(env, seed=seed)
        if self._wall is not None:
            self._wall.reset(env, seed=seed)
        self._gated = set()

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        self._count_inflow(self.simulation_step)
        closed = self._wall_active()
        snap = get_active_snapshot()
        on_ramp: set = set()
        if snap is not None:
            from config.constants import RAMP_EDGE

            on_ramp = {v for v in snap.ordered_ids() if snap.road(v) == RAMP_EDGE}
        if closed:
            for v in on_ramp:
                try:
                    traci.vehicle.setSpeed(v, self.gate_speed)
                    self._gated.add(v)
                except Exception:
                    pass
        else:
            for v in self._gated:
                try:
                    traci.vehicle.setSpeed(v, -1.0)  # restore default
                except Exception:
                    pass
            self._gated.clear()
        if self._wall is not None:
            self._wall.act(obs_dict)
        return {}


def build_inflow_control(kind: str, params: dict):
    """Variant-builder for the R2 mechanisms (mirrors _build_policy)."""
    if kind == "wall":
        return WallHoldPolicy(
            d_lo=params.get("d_lo", 200.0),
            d_hi=params.get("d_hi", 1500.0),
            wall_speed=params.get("wall_speed", 10.0),
            wall_tau=params.get("wall_tau", 0.0),
            a_max_dec=params.get("a_max_dec", 0.3),
            trigger_h=params.get("trigger_h"),
            release_h=params.get("release_h"),
            tau_fraction=params.get("tau_fraction", 1.0),
            assembly_hi=params.get("assembly_hi"),
            assembly_window=params.get("assembly_window", 500.0),
            assembly_max_adj=params.get("assembly_max_adj", 4.0),
            assembly_mode=params.get("assembly_mode", "sync"),
            trigger_v=params.get("trigger_v"),
            trigger_k=params.get("trigger_k"),
            ramp_pace=params.get("ramp_pace", False),
            pace_margin=params.get("pace_margin", 3.0),
            max_speed=params.get("max_speed", 30.0),
        )
    if kind == "gate":
        return GatePolicy(
            gate_speed=params.get("gate_speed", 2.0),
            arm_time=params.get("arm_time", 20.0),
            release_time=params.get("release_time", 10.0),
            trigger_h=params.get("trigger_h"),
            release_h=params.get("release_h", 0.45),
            d_lo=params.get("d_lo", 200.0),
            d_hi=params.get("d_hi", 2500.0),
        )
    if kind == "alinea":
        return AlineaPolicy(
            o_target=params.get("o_target", 0.15),
            k_r=params.get("k_r", 70.0),
            r_min=params.get("r_min", 150.0),
            r_max=params.get("r_max", 1000.0),
            gate_speed=params.get("gate_speed", 3.0),
            ctrl_interval=params.get("ctrl_interval", 10.0),
            veh_len=params.get("veh_len", 5.0),
            zone_m=params.get("zone_m", 400.0),
            gate_m_from_merge=params.get("gate_m_from_merge", 250.0),
        )
    if kind == "ramp_gate":
        return RampGatePolicy(
            trigger_h=params.get("trigger_h", 0.25),
            release_h=params.get("release_h", 0.45),
            gate_speed=params.get("gate_speed", 3.0),
            wall_params=params.get("wall_params"),
        )
    if kind == "vsl":
        return VSLPolicy(vsl_limit=params.get("vsl_limit", 10.0))
    if kind == "spatial_cruise":
        return SpatialCruisePolicy(
            horizon_m=params.get("horizon_m", 1500.0),
            trigger_h=params.get("trigger_h", 0.4),
            release_h=params.get("release_h", 0.25),
            smooth_a_max=params.get("smooth_a_max", 1.0),
        )
    if kind == "smooth_cruise":
        return SmoothCruisePolicy(
            tau_ema=params.get("tau_ema", 30.0),
            trigger_h=params.get("trigger_h"),
            release_h=params.get("release_h", 0.45),
            smooth_a_max=params.get("smooth_a_max", 1.0),
        )
    raise ValueError(f"Unknown inflow-control kind: {kind}")
