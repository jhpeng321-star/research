"""Metastable-control feasibility probe policies (design spec 2026-08-12).

  - `metastable_rule_decision`   — pure h-table decision (unit-testable)
  - `MetastableRulePolicy`       — Stage-2 h-table controller (+ ablations)
  - `ScriptedInterventionPolicy` — Stage-1 loop-gain scripted interventions
  - `HealthRecordingWrapper`     — records per-step (h, rho, v_mean) for ANY
                                   policy (incl. no_op) via the env accessor

Stage-2 rule table (per CAV, role-aware):

  h >= h_coast              -> coast (upper branch, free flow)
  h in [h_pace_lo, h_coast) -> mainline paces (protect, gentle dial), ramp flows
  h <  h_pace_lo            -> ramp meters (fair_balance, dial by rho excess)
                               only if rho >= rho_crit; mainline paces

Constraints: metering releases only after h >= h_release for
h_release_steps consecutive steps (KWG upper/lower branch hysteresis); leader
TTC < ttc_floor downgrades meter -> protect (never meter into a near-miss);
dial rate capped at dial_rate_cap/step with min_dwell mode dwell.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config.constants import MODE_NAMES, RAMP_EDGE
from env.snapshot import get_active_snapshot
from env.traci_query import leader_ttc
from policies import Policy

COAST = MODE_NAMES.index("coast")
FAIR = MODE_NAMES.index("fair_balance")
PROTECT = MODE_NAMES.index("guard")

DEFAULT_RULE_CFG = {
    "h_coast": 0.6,  # h >= this -> coast
    "h_pace_lo": 0.35,  # h in [h_pace_lo, h_coast) -> pace mainline
    "h_release": 0.55,  # metering releases only after sustained h >= this
    "h_release_steps": 5,
    "rho_crit": 5.5,  # metering density gate (veh/km/lane, h calibration)
    "rho_dial_hi": 6.5,  # rho where the metering dial saturates at its max
    "pace_dial": 0.2,  # protect dial used for pacing (gentle)
    "meter_dial_lo": 0.3,  # dial at rho == rho_crit
    "meter_dial_hi": 0.7,  # dial at rho == rho_dial_hi
    "meter_dial_fixed": None,  # ablation: force a fixed metering dial
    "ttc_floor": 1.8,  # leader TTC below this -> meter downgraded
    "ttc_floor_on": True,
    "smoothness_on": True,  # ablation: cap dial rate + enforce mode dwell
    "dial_rate_cap": 0.05,
    "min_dwell": 3,
}


def _new_state() -> dict:
    return {
        "metering": False,
        "release_count": 0,
        "prev_mode": COAST,
        "prev_dial": 0.0,
        "dwell": 0,
    }


def metastable_rule_decision(
    state: dict,
    h: Optional[float],
    rho: Optional[float],
    role: str,
    ttc: Optional[float],
    cfg: dict,
) -> tuple:
    """h-table decision: returns (mode_index, dial, state').

    ``role`` is "ramp" (on the ramp approach edge) or "mainline". ``state``
    carries the hysteresis / dwell / dial-rate memory and is mutated in place.
    """
    if h is None:
        return COAST, 0.0, state

    # --- TTC floor: never meter into a near-miss (per-CAV downgrade) ---
    floor = cfg.get("ttc_floor")
    if cfg.get("ttc_floor_on", True) and floor is not None and ttc is not None and ttc < floor:
        return PROTECT, cfg.get("pace_dial", 0.2), state

    # --- hysteresis: release only after sustained recovery ---
    if state["metering"]:
        if h >= cfg.get("h_release", 0.55):
            state["release_count"] += 1
            if state["release_count"] >= cfg.get("h_release_steps", 5):
                state["metering"] = False
                state["release_count"] = 0
        else:
            state["release_count"] = 0

    h_coast = cfg.get("h_coast", 0.6)
    h_pace_lo = cfg.get("h_pace_lo", 0.35)
    sick = h < h_pace_lo
    meter_ok = sick and role == "ramp" and rho is not None and rho >= cfg.get("rho_crit", 5.5)
    fixed = cfg.get("meter_dial_fixed")
    if fixed is not None:
        meter_target = float(fixed)
    else:
        span = max(cfg.get("rho_dial_hi", 6.5) - cfg.get("rho_crit", 5.5), 1e-6)
        frac = (
            min(max((rho - cfg.get("rho_crit", 5.5)) / span, 0.0), 1.0) if rho is not None else 0.0
        )
        meter_target = cfg.get("meter_dial_lo", 0.3) + frac * (
            cfg.get("meter_dial_hi", 0.7) - cfg.get("meter_dial_lo", 0.3)
        )

    if state["metering"]:
        # release hold: keep the regime until the sustained-recovery counter
        if role == "mainline":
            mode, target = PROTECT, cfg.get("pace_dial", 0.2)
        elif rho is not None and rho >= cfg.get("rho_crit", 5.5):
            mode, target = FAIR, meter_target
        else:
            mode, target = COAST, 0.0
    elif meter_ok:
        mode, target = FAIR, meter_target
        state["metering"] = True
        state["release_count"] = 0
    elif h >= h_coast:
        mode, target = COAST, 0.0
    elif role == "mainline":
        mode, target = PROTECT, cfg.get("pace_dial", 0.2)
    else:  # ramp in the transition band: let it flow
        mode, target = COAST, 0.0

    # --- smoothness: dial rate cap + min mode dwell ---
    if cfg.get("smoothness_on", True):
        cap = cfg.get("dial_rate_cap", 0.05)
        prev = state.get("prev_dial", 0.0)
        dial = min(max(target, prev - cap), prev + cap)
        if mode != state.get("prev_mode"):
            # a mode switch is allowed only after the current mode has been
            # held for min_dwell steps (count grows every step regardless)
            if state.get("dwell", 0) < cfg.get("min_dwell", 3):
                mode = state["prev_mode"]
                dial = prev
            else:
                state["dwell"] = 0
        state["dwell"] = state.get("dwell", 0) + 1
    else:
        dial = target
    state["prev_mode"] = mode
    state["prev_dial"] = dial
    return mode, dial, state


class MetastableRulePolicy(Policy):
    """Stage-2 h-table controller over the (mode, dial) action space.

    Reads h/rho/v_mean from the env's public health accessor once per step
    and applies `metastable_rule_decision` per CAV (ramp approach edge
    meters, mainline paces). Ablations:
      "full"       — hysteresis + TTC floor + smoothness budget
      "no_floor"   — TTC floor disabled
      "aggressive" — no TTC floor, no smoothness, fixed full metering dial
    """

    ABLATIONS = {
        "full": {},
        "no_floor": {"ttc_floor_on": False},
        "aggressive": {
            "ttc_floor_on": False,
            "smoothness_on": False,
            "meter_dial_fixed": 1.0,
        },
    }

    def __init__(self, ablation: str = "full"):
        cfg = dict(DEFAULT_RULE_CFG)
        cfg.update(self.ABLATIONS.get(ablation, {}))
        self.cfg = cfg
        self._env = None
        self._state = _new_state()

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._state = _new_state()

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, tuple]:
        detail = self._env.metastable_health_detail() if self._env is not None else None
        h = detail[0] if detail else None
        rho = detail[1] if detail else None
        snap = get_active_snapshot()
        actions: Dict[str, tuple] = {}
        for cav_id in obs_dict.keys():
            role = "ramp" if snap is not None and snap.road(cav_id) == RAMP_EDGE else "mainline"
            ttc = None
            if self.cfg.get("ttc_floor_on"):
                try:
                    ttc = leader_ttc(cav_id)
                except Exception:  # noqa: BLE001 - transient TraCI gap, act blind
                    ttc = None
            mode, dial, self._state = metastable_rule_decision(
                self._state, h, rho, role, ttc, self.cfg
            )
            actions[cav_id] = (mode, dial)
        return actions


class ScriptedInterventionPolicy(Policy):
    """Stage-1 loop-gain policy: scripted h-triggered interventions, all CAVs.

    With ``trigger_h=None`` the mode is forced every step (continuous variant,
    e.g. pace). Otherwise the mode is forced for ``on_steps`` whenever
    h < ``trigger_h`` (and the off-window elapsed), then coast for
    ``off_steps``, then re-arms.
    """

    def __init__(
        self,
        mode: int,
        dial: float,
        on_steps: int = 10,
        off_steps: int = 5,
        trigger_h: Optional[float] = 0.4,
    ):
        self.mode = int(mode)
        self.dial = float(dial)
        self.on_steps = int(on_steps)
        self.off_steps = int(off_steps)
        self.trigger_h = trigger_h
        self._env = None
        self._on = False
        self._timer = 0
        self._off_timer = 0

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._on = False
        self._timer = 0
        self._off_timer = 0

    def _health(self) -> Optional[float]:
        if self._env is None:
            return None
        d = self._env.metastable_health_detail()
        return None if d is None else d[0]

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, tuple]:
        h = self._health()
        if self.trigger_h is not None:
            if self._on:
                self._timer -= 1
                if self._timer <= 0:
                    self._on = False
                    self._off_timer = self.off_steps
            elif h is not None and h < self.trigger_h and self._off_timer <= 0:
                self._on = True
                self._timer = self.on_steps
            elif self._off_timer > 0:
                self._off_timer -= 1
        take = self.trigger_h is None or self._on
        action = (self.mode, self.dial) if take else (COAST, 0.0)
        return {cav_id: action for cav_id in obs_dict.keys()}


def _corridor_dv(v_mean: Optional[float]) -> Optional[float]:
    """Injection velocity delta |v_mean_merge - v_ramp_approach| (m/s).

    Perturbation-magnitude proxy: the merge-edge mean speed vs the mean speed
    of ramp-approach vehicles near the junction (the vehicles about to
    inject). The KWG breakdown trigger is the speed DIFFERENCE of injections
    into the stream, so the absolute value is the quantity the metering is
    supposed to reduce. Snapshot-only.
    """
    if v_mean is None:
        return None
    snap = get_active_snapshot()
    if snap is None:
        return None
    from config.constants import RAMP_LANE_LENGTH

    speeds = [
        snap.speed(v)
        for v in snap.ordered_ids()
        if snap.road(v) == RAMP_EDGE
        and snap.lane_pos(v) is not None
        and snap.lane_pos(v) >= RAMP_LANE_LENGTH - 120.0
    ]
    speeds = [s for s in speeds if s is not None]
    if not speeds:
        return None
    return float(abs(v_mean - sum(speeds) / len(speeds)))


class HealthRecordingWrapper(Policy):
    """Wrap any policy; record per-step (h, rho, v_mean) for probe rows.

    Uses the env's public `metastable_health_detail()` accessor once per step.
    The probe never enables the Experiment-C mask, so the EMA advances exactly
    once per step and the recorded series is comparable across variants.
    Also records the injection velocity delta (see `_corridor_dv`) so every
    variant — including no_op — carries the process evidence.
    """

    def __init__(self, inner: Policy):
        self.inner = inner
        self.direct_control = bool(getattr(inner, "direct_control", False))
        self.history: list = []
        self.dv_history: list = []
        self.spill_history: list = []  # (step, min lane pos of queued tail)
        self.hard_brake_steps: int = 0  # engine steps with over-gentle decel
        self.inflow_history: list = []  # veh/h into the merge per step (R2)
        self.outflow_history: list = []  # veh/h OFF the merge edge per step
        self._seen_on_merge: set = set()
        self._prev_on_merge: set = set()
        self._env = None
        self._step = 0

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self.history = []
        self.dv_history = []
        self.spill_history = []
        self.hard_brake_steps = 0
        self.inflow_history = []
        self.outflow_history = []
        self._seen_on_merge = set()
        self._prev_on_merge = set()
        self._step = 0
        self.inner.reset(env, seed=seed)

    def _queue_tail_pos(self) -> float | None:
        """Min lane position of the upstream-most SLOW vehicle on the
        pre-edge (v < 15 m/s ~ half of the ~31 m/s free flow) — how far the
        stored queue's slow region extends toward the corridor entry. None
        when no pre-edge vehicle is slowed (no queue on the corridor yet).

        Threshold note (measured 2026-08-14): the 2900-long "collapse" is a
        MOVING bottleneck — pre-edge vehicles slow to ~18 m/s but do not
        stop, so a v < 2 m/s (stopped) threshold never fires. The probe's
        sustained-collapse metric (h < 0.15) captures the breakdown; this
        recorder captures the queue's upstream extent."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        from config.constants import PRE_MERGE_EDGE

        tail = None
        for v in snap.ordered_ids():
            if snap.road(v) != PRE_MERGE_EDGE:
                continue
            sp = snap.speed(v)
            lp = snap.lane_pos(v)
            if sp is not None and lp is not None and sp < 15.0:
                tail = lp if tail is None else min(tail, lp)
        return tail

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1
        detail = self._env.metastable_health_detail() if self._env is not None else None
        n_veh = None
        snap = get_active_snapshot()
        if snap is not None:
            n_veh = len(snap.ordered_ids())  # on-road vehicle count (corridor)
        # Merge-inflow counter (R2): vehicles newly on the merge edge this
        # step. Raw per-step counts (not rate-scaled) — the driver converts
        # to an episode-average veh/h = total unique arrivals / episode time,
        # which is immune to the initial departure burst (flows with
        # departSpeed=max over-insert at t=0).
        if snap is not None:
            from config.constants import MERGE_EDGE

            now = {v for v in snap.ordered_ids() if snap.road(v) == MERGE_EDGE}
            new = now - self._seen_on_merge
            self._seen_on_merge |= now
            # Append EVERY step (incl. zeros) — the driver splits the series
            # into collapse/healthy windows by index, so alignment matters.
            self.inflow_history.append(len(new))
            # Merge-THROAT discharge (2026-08-23, Hall & Agyemang-Duah point):
            # vehicles that were on the merge edge last step and are gone now
            # (they discharged downstream). Measures the drop where it counts.
            left = self._prev_on_merge - now if self._prev_on_merge else set()
            self.outflow_history.append(len(left))
            self._prev_on_merge = now
        if detail is not None:
            self.history.append((float(detail[0]), float(detail[1]), float(detail[2]), n_veh))
            dv = _corridor_dv(float(detail[2]))
            if dv is not None:
                self.dv_history.append(dv)
        elif snap is not None:
            # keep history step-aligned with the flow counters (h None early)
            self.history.append((None, None, None, n_veh))
        else:
            self.history.append((None, None, None, n_veh))
        tail = self._queue_tail_pos()
        if tail is not None:
            self.spill_history.append((self._step, tail))
        # Shockwave-formation audit (T1): the engine counts steps where the
        # raw commanded decel exceeded the gentle envelope; mirror the latest
        # count so the probe row reports whether the intervention itself
        # braked hard enough to nucleate a wave.
        eng = getattr(self._env, "_shepherd_engine", None) if self._env is not None else None
        if eng is not None:
            self.hard_brake_steps = int(getattr(eng, "hard_brake_steps", 0))
        return self.inner.act(obs_dict)

    def close(self) -> None:
        self.inner.close()


# ---------------------------------------------------------------------------
# Stage 3 (2026-08-13): PREVENTIVE injection control — the user's mechanism.
# The stage-1/2 controllers gated on h < 0.35 / rho >= 5.5 (at-collapse,
# reactive rescue — the expC pattern). Stage 3 meters the ramp BEFORE the
# merge sickens, acting on the two factors that determine the KWG critical
# velocity v_c(rho): (1) the road density (hold while rho_ema >= rho_set,
# still on the upper branch) and (2) the injection velocity delta (hold the
# ramp CAV until its merge speed >= v_c(rho) - margin, so the injection
# cannot nucleate breakdown). A fairness gate releases the ramp whenever its
# delay already exceeds the mainline's (ramp metering must not exploit the
# ramp: fairness = TTC > G > delay).
# ---------------------------------------------------------------------------

DEFAULT_PREVENTIVE_CFG = {
    "rho_set": 4.0,  # preventive metering setpoint (upper branch; rho_free 2.0, rho_crit 5.5)
    "rho_crit": 5.5,  # collapse density (h calibration)
    "rho_free": 2.0,
    "v_free": 25.0,  # upper-branch velocity at rho_free
    "v_break": 12.0,  # upper-branch velocity at rho_crit
    "dv_margin": 2.0,  # release the ramp CAV once v_ramp >= v_c(rho) - margin
    "fair_thresh": 0.1,  # ramp delay-rate excess (s/s) that blocks metering
    "levers": ("rho", "dv"),  # which triggers hold the ramp ("rho", "dv", both)
    "fairness_gate": True,
    "meter_dial_lo": 0.3,  # dial at rho == rho_set
    "meter_dial_hi": 0.7,  # dial at rho == rho_crit
    "dv_dial": 0.3,  # gentle hold for the velocity gate alone
    # NOTE (2026-08-13): "mainline pacing" via protect is NOT expressible in
    # this engine — _protect_eligible gates protect to active TTC/PET threats
    # (emergency headway), so a gentle pacing dial silently remaps to coast.
    # The mainline role therefore coasts by default; wiring wave_damp in would
    # be the real gap-provider if this probe shows the ramp lever matters.
    "mainline_pace": False,
    "pace_dial": 0.2,  # (unused unless mainline_pace)
    "h_pace_hi": 0.6,  # (unused unless mainline_pace)
    "ttc_floor": 1.8,
    "ttc_floor_on": True,
    "release_steps": 5,  # hysteresis: release after N clear steps
    "smoothness_on": True,
    "dial_rate_cap": 0.05,
    "min_dwell": 3,
}


def preventive_vc(rho: Optional[float], cfg: dict) -> Optional[float]:
    """KWG upper-branch critical velocity v_c(rho) (linear ramp calibration).

    v_c = v_free at rho_free, falling linearly to v_break at rho_crit — the
    velocity below which an injection into the stream nucleates breakdown
    (Krauss-Wagner-Gawron 1997 §V). None when rho is unknown.
    """
    if rho is None:
        return None
    v_free = float(cfg.get("v_free", 25.0))
    v_break = float(cfg.get("v_break", 12.0))
    rho_free = float(cfg.get("rho_free", 2.0))
    rho_crit = float(cfg.get("rho_crit", 5.5))
    frac = min(max((rho - rho_free) / max(rho_crit - rho_free, 1e-6), 0.0), 1.0)
    return v_free - (v_free - v_break) * frac


def preventive_decision(
    state: dict,
    h: Optional[float],
    rho: Optional[float],
    role: str,
    ttc: Optional[float],
    v_ramp: Optional[float],
    gap: Optional[float],
    cfg: dict,
    corr_ttc: Optional[int] = None,
) -> tuple:
    """Preventive injection decision: returns (mode_index, dial, state').

    ``role`` is "ramp" (on the ramp approach edge) or "mainline". ``gap`` is
    the engine's signed stream delay gap ema_main - ema_ramp (negative = the
    ramp is already worse). ``corr_ttc`` is the corridor-level sub-threshold
    TTC count this step (the corridor-TTC suppression, stage 4). ``state``
    carries the hold hysteresis / dwell / dial-rate memory (mutated in
    place).
    """
    if h is None:
        return COAST, 0.0, state

    # --- TTC floor: never hold into a near-miss (per-CAV downgrade) ---
    floor = cfg.get("ttc_floor")
    if cfg.get("ttc_floor_on", True) and floor is not None and ttc is not None and ttc < floor:
        return PROTECT, cfg.get("pace_dial", 0.2), state

    levers = set(cfg.get("levers", ("rho", "dv")))
    rho_set = float(cfg.get("rho_set", 4.0))
    rho_trigger = "rho" in levers and rho is not None and rho >= rho_set
    v_c = preventive_vc(rho, cfg)
    # The velocity gate acts only on the metastable band (rho >= rho_free):
    # below rho_c the free branch is unconditionally stable and an injection
    # below v_c cannot nucleate breakdown — holding there would just queue
    # the ramp for nothing.
    dv_trigger = (
        "dv" in levers
        and role == "ramp"
        and rho is not None
        and rho >= float(cfg.get("rho_free", 2.0))
        and v_ramp is not None
        and v_c is not None
        and v_ramp < v_c - float(cfg.get("dv_margin", 2.0))
    )
    # Fairness gate: the ramp is the metered stream — never hold it while it
    # is already worse off than the mainline (gap < -fair_thresh). Stops the
    # hold immediately; the release itself runs through the hysteresis
    # countdown below so the controller does not chatter between the gate
    # and the density trigger.
    fair_thresh = float(cfg.get("fair_thresh", 0.1))
    fairness_blocked = (
        cfg.get("fairness_gate", True) and role == "ramp" and gap is not None and gap < -fair_thresh
    )
    # Corridor-TTC suppression (stage 4): metering creates compression —
    # never hold while the corridor already has sub-threshold TTC pairs.
    cttc_blocked = bool(
        cfg.get("corridor_ttc_block", False) and corr_ttc is not None and corr_ttc > 0
    )
    hold = (
        role == "ramp" and (rho_trigger or dv_trigger) and not fairness_blocked and not cttc_blocked
    )

    # --- hysteresis: once holding, release only after the triggers stay
    # clear for release_steps consecutive steps ---
    if hold:
        state["metering"] = True
        state["release_count"] = 0
    elif state["metering"]:
        state["release_count"] = state.get("release_count", 0) + 1
        if state["release_count"] >= cfg.get("release_steps", 5):
            state["metering"] = False
            state["release_count"] = 0

    if role == "mainline":
        # Protect is threat-gated in this engine (emergency headway only), so
        # gentle pacing is NOT expressible — mainline coasts unless the
        # mainline_pace flag is explicitly set (see cfg note).
        if cfg.get("mainline_pace", False) and h < float(cfg.get("h_pace_hi", 0.6)):
            mode, target = PROTECT, cfg.get("pace_dial", 0.2)
        else:
            mode, target = COAST, 0.0
    elif state["metering"]:
        mode = FAIR
        fixed = cfg.get("meter_dial_fixed")
        if fixed is not None:
            # Stage 6 (VSL / always-on): a fixed speed-cut dial, independent
            # of the rho excess.
            target = float(fixed)
        elif rho_trigger and rho is not None:
            span = max(float(cfg.get("rho_crit", 5.5)) - rho_set, 1e-6)
            frac = min(max((rho - rho_set) / span, 0.0), 1.0)
            target = cfg.get("meter_dial_lo", 0.3) + frac * (
                cfg.get("meter_dial_hi", 0.7) - cfg.get("meter_dial_lo", 0.3)
            )
        else:  # velocity-gate-only hold: gentle, fixed
            target = cfg.get("dv_dial", 0.3)
    else:
        mode, target = COAST, 0.0

    # --- smoothness: dial rate cap + min mode dwell ---
    if cfg.get("smoothness_on", True):
        cap = cfg.get("dial_rate_cap", 0.05)
        prev = state.get("prev_dial", 0.0)
        dial = min(max(target, prev - cap), prev + cap)
        if mode != state.get("prev_mode"):
            if state.get("dwell", 0) < cfg.get("min_dwell", 3):
                mode = state["prev_mode"]
                dial = prev
            else:
                state["dwell"] = 0
        state["dwell"] = state.get("dwell", 0) + 1
    else:
        dial = target
    state["prev_mode"] = mode
    state["prev_dial"] = dial
    return mode, dial, state


class PreventiveInjectionPolicy(Policy):
    """Stage-3 PREVENTIVE injection controller (the user's mechanism).

    Holds the ramp CAV while the merge-zone density is at/above the
    preventive setpoint (rho >= rho_set, still on the upper branch — dial
    proportional to the rho excess) and while its own merge speed is below
    the KWG critical velocity v_c(rho) - margin. Releases through a
    fairness gate (never meter a ramp that already suffers) and a hold
    hysteresis. Mainline CAVs pace (protect) below h_pace_hi to provide the
    merge gaps and damp compression waves.

    Ablations (levers): ("rho",) density-setpoint only; ("dv",) velocity
    gate only; ("rho", "dv") both. fairness_gate on for all variants.
    """

    def __init__(
        self,
        levers=("rho", "dv"),
        fairness_gate: bool = True,
        meter_side: str = "ramp",
        trigger_src: str = "merge",
        cfg_overrides: Optional[dict] = None,
    ):
        cfg = dict(DEFAULT_PREVENTIVE_CFG)
        cfg["levers"] = tuple(levers)
        cfg["fairness_gate"] = bool(fairness_gate)
        if cfg_overrides:
            cfg.update(cfg_overrides)
        self.cfg = cfg
        # Stage 4: "mainline" meters the MAINLINE stream instead of the ramp
        # (the stage-1 all-CAV burst — the lever that actually moved h at
        # 2700 — in a designed, floor-guarded form); ramp CAVs coast.
        self.meter_side = meter_side
        # Stage 6: trigger source — "merge" (merge-zone rho, as before) or
        # "pre_v" (platoon-breaking VSL: fire when the UPSTREAM pre-edge mean
        # speed drops below v_trig — the platoon is detected where it forms,
        # ~20-60 s before it reaches the merge). With "pre_v" the signal is
        # mapped to rho=99.0 (fire) / 0.0 (idle) and the dial is expected to
        # be meter_dial_fixed.
        self.trigger_src = trigger_src
        self._env = None
        self._state = _new_state()
        # Ramp-approach roads: the ramp edge + its connector(s) into the
        # merge. Vehicles leave RAMP_EDGE and spend the last ~seconds before
        # injection on the connector — the exact moment the velocity gate
        # must act — so the role check must include it. Built lazily at the
        # first act() of each episode (TraCI is connected by then).
        self._ramp_roads: Optional[frozenset] = None

    def reset(self, env, seed: int = None) -> None:
        self._env = env
        self._state = _new_state()
        self._ramp_roads = None

    def _pre_v_signal(self) -> Optional[float]:
        """Mean speed of pre-edge (upstream mainline) vehicles, m/s."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        from config.constants import PRE_MERGE_EDGE

        speeds = [snap.speed(v) for v in snap.ordered_ids() if snap.road(v) == PRE_MERGE_EDGE]
        speeds = [s for s in speeds if s is not None]
        if not speeds:
            return None
        return float(sum(speeds) / len(speeds))

    def _ramp_approach_speed(self) -> Optional[float]:
        """Mean speed of ramp-approach vehicles near the junction (m/s)."""
        snap = get_active_snapshot()
        if snap is None:
            return None
        from config.constants import RAMP_LANE_LENGTH

        speeds = [
            snap.speed(v)
            for v in snap.ordered_ids()
            if snap.road(v) == RAMP_EDGE
            and snap.lane_pos(v) is not None
            and snap.lane_pos(v) >= RAMP_LANE_LENGTH - 120.0
        ]
        speeds = [s for s in speeds if s is not None]
        if not speeds:
            return None
        return float(sum(speeds) / len(speeds))

    def _ramp_role(self, road: Optional[str]) -> bool:
        """True when ``road`` is on the ramp approach (edge or connector)."""
        if road is None:
            return False
        if road == RAMP_EDGE:
            return True
        if self._ramp_roads is None:
            roads = {RAMP_EDGE}
            try:
                import traci as _traci

                for lane in _traci.lane.getIDList():
                    if not lane.startswith(RAMP_EDGE):
                        continue
                    for link in _traci.lane.getLinks(lane) or []:
                        nid = link[0] if link else None
                        if not nid:
                            continue
                        edge = nid.rsplit("_", 1)[0]
                        if edge.startswith(":"):  # ramp connector into the merge
                            roads.add(edge)
            except Exception:  # noqa: BLE001 - act with the plain ramp edge
                pass
            self._ramp_roads = frozenset(roads)
        return road in self._ramp_roads

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, tuple]:
        detail = self._env.metastable_health_detail() if self._env is not None else None
        h = detail[0] if detail else None
        rho = detail[1] if detail else None
        if self.trigger_src == "pre_v":
            # Platoon-breaking VSL trigger: upstream pre-edge speed below
            # v_trig -> fire (mapped to rho 99.0), else idle (0.0).
            v_pre = self._pre_v_signal()
            rho = 99.0 if v_pre is not None and v_pre < float(self.cfg.get("v_trig", 20.0)) else 0.0
        elif self.trigger_src == "match_v":
            # Ramp merge-speed matching (2026-08-14): at critical mainline
            # demand the injection delta is INVERTED — ramp vehicles arrive
            # faster than the congested merge. The ramp CAV paces the whole
            # ramp approach down to the merge-zone mean speed (single-lane
            # queue effect: followers must brake too), shrinking |dv|.
            v_app = self._ramp_approach_speed()
            v_merge = detail[2] if detail else None
            rho = (
                99.0
                if (
                    v_app is not None
                    and v_merge is not None
                    and v_app > v_merge + float(self.cfg.get("match_v_margin", 3.0))
                )
                else 0.0
            )
        gap = None
        corr_ttc = None
        if self._env is not None:
            eng = getattr(self._env, "_shepherd_engine", None)
            if eng is not None:
                try:
                    gap = eng.stream_delay_gap()
                except Exception:  # noqa: BLE001 - transient, act without the gate
                    gap = None
            if self.cfg.get("corridor_ttc_block", False):
                try:
                    corr_ttc = self._env._corridor_ttc_count()
                except Exception:  # noqa: BLE001 - transient, act without the gate
                    corr_ttc = None
        snap = get_active_snapshot()
        actions: Dict[str, tuple] = {}
        for cav_id in obs_dict.keys():
            road = snap.road(cav_id) if snap is not None else None
            on_ramp = self._ramp_role(road)
            if self.meter_side == "mainline":
                # Stage 4 maindamp: the MAINLINE is the metered role; the
                # ramp coasts. The decision function's "ramp" role is the
                # one that holds, so invert the mapping.
                role = "mainline" if on_ramp else "ramp"
            elif self.meter_side == "both":
                # Stage 5: every CAV is a metered role (mainline + ramp
                # combined gentle metering).
                role = "ramp"
            else:
                role = "ramp" if on_ramp else "mainline"
            ttc = None
            if self.cfg.get("ttc_floor_on"):
                try:
                    ttc = leader_ttc(cav_id)
                except Exception:  # noqa: BLE001 - transient TraCI gap, act blind
                    ttc = None
            v_ramp = snap.speed(cav_id) if snap is not None else None
            mode, dial, self._state = preventive_decision(
                self._state,
                h,
                rho,
                role,
                ttc,
                v_ramp,
                gap,
                self.cfg,
                corr_ttc=corr_ttc,
            )
            actions[cav_id] = (mode, dial)
        return actions


class HoldZoneWrapper(Policy):
    """Restrict an inner controller to CAVs within a distance-from-merge
    window [d_lo, d_hi] (meters, positive = upstream). CAVs outside the
    window coast (mode 0). Tests whether intervening further upstream makes
    the merge healthier (U1 probe)."""

    def __init__(self, inner: Policy, d_lo: float, d_hi: float):
        self.inner = inner
        self.d_lo = float(d_lo)
        self.d_hi = float(d_hi)
        self.direct_control = bool(getattr(inner, "direct_control", False))

    def reset(self, env, seed: int = None) -> None:
        self.inner.reset(env, seed=seed)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        from env.features import distance_to_merge

        actions = self.inner.act(obs_dict)
        if not actions:
            return actions
        out = {}
        for cav_id, act in actions.items():
            try:
                d = distance_to_merge(cav_id)
            except Exception:  # noqa: BLE001 - transient TraCI gap, coast
                d = float("inf")
            out[cav_id] = act if self.d_lo <= d <= self.d_hi else (0, 0.0)
        return out

    def close(self) -> None:
        self.inner.close()


class ReleaseAfterWrapper(Policy):
    """Run the inner controller for the first ``T`` steps, then release to
    no-op (mode 0) for the rest of the episode. Distinguishes a true
    metastable hold (upper branch persists after control stops) from a
    postponed collapse (H1 probe)."""

    def __init__(self, inner: Policy, T: int):
        self.inner = inner
        self.T = int(T)
        self._step = 0
        self.direct_control = bool(getattr(inner, "direct_control", False))

    def reset(self, env, seed: int = None) -> None:
        self._step = 0
        self.inner.reset(env, seed=seed)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self._step < self.T:
            self._step += 1
            return self.inner.act(obs_dict)
        self._step += 1
        return {cav_id: (0, 0.0) for cav_id in obs_dict.keys()}

    def close(self) -> None:
        self.inner.close()
