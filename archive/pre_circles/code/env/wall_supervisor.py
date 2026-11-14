"""Single-agent RSU wall-supervisor environment (2026-08-21).

The supervisor is the RSU: once per controller step it picks the corridor's
wall state — off / soft / strong — and every in-band CAV executes it (the
wall is a GLOBAL decision: the state it depends on is corridor-level and
identical for all CAVs, so per-CAV learning would only add noise and invite
defections). All other traffic follows SUMO's default driving (v1: the
pruned champion is frozen out of the loop — at dense demand it is a coin
flip anyway, measured 2026-08-21).

The wall rungs are the two measured operating points, trigger removed (the
supervisor replaces the hand-set threshold):
  soft   = w10_t30, band 200-400 m   (the p=0.2 champion config)
  strong = w12_t30, band 200-800 m   (the low-penetration champion config)

Execution reuses policies.inflow_control.WallHoldPolicy verbatim, so
supervisor episodes behave bit-identically to the probe cells whose n=200
numbers are the comparison baseline.

Reward (corridor-level, per step): total timeLoss across vehicles this step
(collapse prices itself through accumulated loss) minus a TTC charge. The
wall's own delay is captured by the same timeLoss term — no separate
engagement cost (that would double-count).

The adapter wraps TrafficShepherdMultiAgentEnv with an empty action dict
(env skips the mode machinery when the dict is empty) and applies the wall
via direct TraCI exactly like the probe harness.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import traci
import gymnasium as gym

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE
from config.defaults import DEFAULT_CONFIG
from policies.inflow_control import WallHoldPolicy
from policies.traci_control import release_to_sumo

# Rung definitions (measured operating points, trigger removed). v3 is binary
# (off / wall): the soft rung is the p=0.2 champion config; the credit split
# between rungs slowed early learning for a second-order gain (v1/v2 lesson).
RUNGS = {
    1: {"d_lo": 200.0, "d_hi": 400.0, "wall_speed": 10.0, "wall_tau": 3.0},
    2: {"d_lo": 200.0, "d_hi": 800.0, "wall_speed": 12.0, "wall_tau": 3.0},
}
OBS_DIM = 16  # 12 corridor dims + 4 health-trajectory lags (v4)
H_LAGS = (5, 10, 20, 30)


def sustained_collapse(
    h_hist: list, start: int = 100, run_len: int = 60, thresh: float = 0.15
) -> bool:
    """The probe's exact collapse criterion: h < thresh for >= run_len
    consecutive steps after step ``start`` (calibrated 2026-08-14)."""
    run = 0
    for i, h in enumerate(h_hist):
        if i < start:
            continue
        run = run + 1 if h < thresh else 0
        if run >= run_len:
            return True
    return False


def corridor_state(
    ma_env,
    prev_h: Optional[float],
    rung: int,
    engaged_steps: int,
    step_i: int,
    inflow_tracker: dict,
    h_hist: Optional[list] = None,
) -> np.ndarray:
    """Corridor-level observation for the RSU supervisor (all snapshot/EMA
    based — the same signals the rule trigger and the probes use).

    v4: appends the h trajectory lags (h_{t-5..30}). The skip decision needs
    to distinguish a recovering dip from a collapsing slide — that lives in
    the trajectory shape, not the instant value (the v3 reward forensics
    showed instant-h policies drift to passivity)."""
    from env.snapshot import get_active_snapshot

    detail = ma_env.metastable_health_detail()
    if detail is None:
        h, rho_e, v_mean = 1.0, 0.0, 25.0
    else:
        h, rho_e, v_mean = float(detail[0]), float(detail[1]), float(detail[2])
    dh = 0.0 if prev_h is None else h - prev_h

    snap = get_active_snapshot()
    queue_n = 0
    tail_rel = 1.0
    inband = 0
    upstream = 0
    merge_now = set()
    if snap is not None:
        try:
            pre_len = float(traci.lane.getLength(PRE_MERGE_EDGE + "_0"))
        except Exception:
            pre_len = 1.0
        for vid in snap.ordered_ids():
            try:
                rd = snap.road(vid)
                if rd == PRE_MERGE_EDGE:
                    upstream += 1
                    sp = snap.speed(vid)
                    lp = snap.lane_pos(vid)
                    if sp is not None and sp < 15.0:
                        queue_n += 1
                        if lp is not None and pre_len > 0:
                            tail_rel = min(tail_rel, lp / pre_len)
                    d = pre_len - (lp or 0.0)
                    if 200.0 <= d <= 800.0 and vid in ma_env.active_cavs:
                        inband += 1
                elif rd == MERGE_EDGE:
                    merge_now.add(vid)
            except Exception:
                continue
    # Merge inflow this step (unique arrivals), veh/h — same counter logic as
    # the probe policies' inflow mixin.
    seen = inflow_tracker.setdefault("seen", set())
    new = merge_now - seen
    seen |= merge_now
    dt = float(getattr(ma_env, "simulation_step", 1.0))
    inflow = (len(new) * 3600.0 / dt) if dt > 0 else 0.0

    max_steps = float(getattr(ma_env, "max_episode_steps", 0) or 0)
    step_frac = (step_i / max_steps) if max_steps > 0 else 0.0
    # v4: trajectory lags (fallback to current h when history is short).
    lags = []
    for L in H_LAGS:
        if h_hist and len(h_hist) >= L:
            lags.append(float(h_hist[-L]))
        elif h_hist:
            lags.append(float(h_hist[0]))
        else:
            lags.append(h)
    return np.array(
        [
            h,  # merge metastable health (the signal)
            dh,  # short trend
            rho_e / 20.0,  # merge-zone density EMA (veh/km/lane)
            v_mean / 30.0,  # merge-zone mean speed
            queue_n / 100.0,  # slow (<15 m/s) vehicles on the pre-edge
            1.0 - tail_rel,  # queue spillback toward the entry
            inband / 5.0,  # convoy presence in the wall band
            upstream / 40.0,  # upstream CAV count (penetration context)
            rung / 2.0,  # current wall state (0, 0.5, 1)
            engaged_steps / max(1.0, float(step_i)),  # engagement duty so far
            step_frac,  # episode progress
            inflow / 3000.0,  # merge inflow this step (veh/h)
            *lags,  # h_{t-5}, h_{t-10}, h_{t-20}, h_{t-30}
        ],
        dtype=np.float32,
    )


class WallSupervisorEnv(gym.Env):
    """Single-agent gym env: the RSU picks the corridor wall state each step."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        config = config or {}
        from env.core import TrafficShepherdMultiAgentEnv

        # v4.1: num_actions 3 re-adds the escalation rung (RUNGS[2], the
        # strong long-band wall) for the "escalate in the worst windows"
        # fine-tune; default 2 (off/soft) — v4's binary credit-clean choice.
        self.num_actions = int(config.get("num_actions", 2))
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # v4.2 (2026-08-23): the soft rung's config is overridable per run —
        # the protection-first variant uses the long band (800 m) measured at
        # the low-penetration champion. Defaults stay the measured p=0.2 config.
        rung_soft = dict(RUNGS[1])
        for k in ("d_lo", "d_hi", "wall_speed", "wall_tau"):
            if config.get(f"rung_soft_{k}") is not None:
                rung_soft[k] = float(config[f"rung_soft_{k}"])
        rungs = dict(RUNGS)
        rungs[1] = rung_soft

        env_cfg = dict(DEFAULT_CONFIG)
        env_cfg.update(config.get("env_overrides") or {})
        # The supervisor is the only controller: the per-CAV mode machinery
        # stays unused (empty action dict every step).
        env_cfg["control_mode"] = "shepherd"
        self._ma = TrafficShepherdMultiAgentEnv(env_cfg)
        self.delay_norm = float(config.get("delay_norm", 100.0))
        self.ttc_penalty = float(config.get("ttc_penalty", 2.0))
        # v4 reward (forensic-driven, 2026-08-22): v1's delay-only term was
        # misaligned (harmful); v2's health-LEVEL bonus was lagging and
        # non-attributional (weak); v3's terminal bonus alone couldn't hold
        # the warm start against steady per-step delay pressure (iter-5
        # trigger-level → iter-50 passive). v4 rebalances the TIMESCALES:
        # the contrastive Δh term pays engagement credit within 30-60 steps
        # (the delay term's own timescale), and the terminal ±800 dominates
        # the episode-level accounting.
        self.health_bonus = float(config.get("health_bonus", 0.0))
        self.health_delta = float(config.get("health_delta", 50.0))
        self.collapse_bonus = float(config.get("collapse_bonus", 800.0))
        # gamma for the potential-form shaping (passed from the trainer so
        # F = w(γh' − h) matches the training discount exactly)
        self.gamma = float(config.get("gamma", 0.997))
        self._h_hist: list = []
        self._walls = {k: WallHoldPolicy(**v) for k, v in rungs.items()}
        self._inflow_tracker: dict = {}
        self._prev_h: Optional[float] = None
        self._rung = 0
        self._engaged = 0
        self._step_i = 0
        self._last_obs: Dict[str, Any] = {}

    # -- gym API ------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, _info = self._ma.reset(seed=seed)
        self._last_obs = obs
        self._inflow_tracker = {}
        self._prev_h = None
        self._rung = 0
        self._engaged = 0
        self._step_i = 0
        self._h_hist = []
        for w in self._walls.values():
            w.reset(self._ma, seed)
        return corridor_state(self._ma, None, 0, 0, 0, self._inflow_tracker, self._h_hist), {}

    def step(self, action: int):
        action = int(action)
        cav_ids = list(self._last_obs.keys())
        if action in self._walls:
            # Wall ON: the probe-identical direct-control path (in-band CAVs
            # get wall speed/tau; out-of-band CAVs are released by the wall
            # policy itself).
            self._walls[action].act({c: None for c in cav_ids})
        else:
            # Wall OFF: release every active CAV to SUMO with its original
            # tau (same release the probe walls use when disengaged). Guard
            # membership — departed vehicles answer TraCI with server-side
            # errors.
            live = traci.vehicle.getIDList()
            for cav_id in cav_ids:
                if cav_id not in live:
                    continue
                try:
                    release_to_sumo(cav_id)
                except Exception:
                    pass
        self._rung = action
        if action in self._walls:
            self._engaged += 1

        obs, _rewards, terminated, truncated, _infos = self._ma.step({})
        self._last_obs = obs
        self._step_i += 1

        sup_obs = corridor_state(
            self._ma,
            self._prev_h,
            self._rung,
            self._engaged,
            self._step_i,
            self._inflow_tracker,
            self._h_hist,
        )
        # previous step's h for the Δh shaping (before _prev_h is updated
        # in the reward block below)
        h_prev = float(sup_obs[0]) if self._prev_h is None else self._prev_h

        # Corridor reward: total timeLoss this step (collapse is priced
        # through accumulated loss) minus the per-event TTC charge.
        from env.snapshot import get_active_snapshot

        snap = get_active_snapshot()
        loss_sum = 0.0
        if snap is not None:
            for vid in snap.ordered_ids():
                try:
                    tl = snap.time_loss(vid)
                    if tl is not None:
                        loss_sum += float(tl)
                except Exception:
                    continue
        try:
            ttc = self._ma._corridor_ttc_count() or 0
        except Exception:
            ttc = 0
        reward = -(loss_sum / self.delay_norm) - self.ttc_penalty * float(ttc)
        if self.health_bonus:
            reward += self.health_bonus * float(sup_obs[0])  # h level (v2)
        if self.health_delta:
            # v4: potential-based shaping (Ng, Harada & Russell 1999):
            # F = w(γ·h' − h) — the ONLY state-only shaping form provably
            # policy-invariant. v2's raw +w·h level bonus was NOT invariant
            # (changed the optimum; measured weak). The Δh-dominant term
            # pays engagement credit within 30-60 steps — the delay term's
            # own timescale.
            reward += self.health_delta * (self.gamma * float(sup_obs[0]) - h_prev)
        self._prev_h = float(sup_obs[0])

        done = bool(terminated.get("__all__", False))
        trunc = bool(truncated.get("__all__", False))
        # v3: terminal collapse bonus — the true objective, priced directly at
        # episode end over the recorded h history (probe's sustained-collapse
        # criterion). Symmetric: survival earns +bonus, collapse −bonus, so
        # the engagement decision carries a clear contrast.
        self._h_hist.append(float(sup_obs[0]))
        if done or trunc:
            collapsed = sustained_collapse(self._h_hist)
            reward += -self.collapse_bonus if collapsed else self.collapse_bonus
        return sup_obs, float(reward), done, trunc, {}

    def close(self):
        try:
            self._ma.close()
        except Exception:
            pass
