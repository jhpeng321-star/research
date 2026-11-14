"""Closed-loop h-rule and single-engagement wall arms (Stage 1b-prep-3).

HRULE reproduces the legacy health-gated wall hysteresis on the sustained
wall actuator, using the exact semantics of the legacy mechanism:

- Signal: ``env.metastable_health_detail()[0]`` — the EMA-based merge-zone
  health h, read once per controller step (pre-step state: the rule that
  fires on the health observed at the start of a step commands the wall for
  the next simulated second). This reproduces the legacy probe pattern
  (WallHoldPolicy._detail cache + the corridor post-step read).
- Arm: h < 0.25, instant (no persistence, no minimum-on time).
- Release: h > 0.45, instant (no minimum-off time, no taper on release).
- Re-engagement: allowed immediately after release (latch resets).
- Engagement gate: the wall is only permitted from step 100 onward per the
  prompt; the legacy WallHoldPolicy trigger has no such gate (ambiguity
  recorded in the contract: this task adds the 100 s gate explicitly).

SE(t_d) is the single-engagement variant: engage at the fixed rule-blind
step t_d regardless of h; release by the same legacy rule (h > 0.45,
instant) or at the demand end (t = 900 s); no re-engagement.

In both arms the actuator is ``WallHoldPolicy`` with its trigger removed
(pure band-passage actuator: in-band CAVs hold 10 m/s / tau + 3.0 s under
the 0.3 m/s^2 clip; out-of-band CAVs are released to native IDM with their
original tau). The wrapper owns the rule state machine and the health reads,
so the number of health-EMA advances per step is identical to the no-op and
sustained-wall arms (one pre-step read by the policy, one post-step read by
the corridor observation), which is what makes prefix identity exact.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import traci

from config.constants import PRE_MERGE_EDGE
from env.features import distance_to_merge
from policies import DirectControlPolicy
from policies.inflow_control import WallHoldPolicy
from policies.traci_control import release_to_sumo

DEFAULT_WALL_PARAMS: Dict[str, float] = {
    "d_lo": 200.0,
    "d_hi": 400.0,
    "wall_speed": 10.0,
    "wall_tau": 3.0,
    "a_max_dec": 0.3,
}


def _build_actuator(wall_params: Mapping[str, Any]) -> WallHoldPolicy:
    params = dict(DEFAULT_WALL_PARAMS)
    params.update(dict(wall_params))
    params["trigger_h"] = None  # pure actuator; the wrapper owns the rule
    return WallHoldPolicy(**params)


class _RuleWallBase(DirectControlPolicy):
    """Shared band-passage actuator plumbing for the rule-driven arms."""

    direct_control = True

    def __init__(self) -> None:
        super().__init__()
        self._wall: Optional[WallHoldPolicy] = None
        self._env: Any = None
        self._step = 0
        self.last_audit: Dict[str, Any] = {}

    # -- live state helpers -------------------------------------------
    @staticmethod
    def _live_ids() -> set[str]:
        try:
            return {str(v) for v in traci.vehicle.getIDList()}
        except Exception:
            return set()

    @staticmethod
    def _road_id(vehicle_id: str) -> Optional[str]:
        try:
            return str(traci.vehicle.getRoadID(vehicle_id))
        except Exception:
            return None

    def _mainline_obs(self, obs_dict: Mapping[str, Any], live_ids: set[str]) -> Dict[str, Any]:
        return {
            str(cav_id): obs_dict[cav_id]
            for cav_id in obs_dict
            if str(cav_id) in live_ids and self._road_id(str(cav_id)) == PRE_MERGE_EDGE
        }

    def _band_members(self, obs_dict: Mapping[str, Any], live_ids: set[str]) -> list[str]:
        members = []
        for cav_id in sorted(str(v) for v in obs_dict.keys()):
            if cav_id not in live_ids:
                continue
            if self._road_id(cav_id) != PRE_MERGE_EDGE:
                continue
            try:
                distance = float(distance_to_merge(cav_id))
            except Exception:
                continue
            if DEFAULT_WALL_PARAMS["d_lo"] <= distance <= DEFAULT_WALL_PARAMS["d_hi"]:
                members.append(cav_id)
        return members

    def _release_touched(self, obs_dict: Mapping[str, Any], live_ids: set[str]) -> None:
        """Restore wall-touched CAVs (tau + speed override) and clear the set.

        Only CAVs actually commanded by the wall are touched, because even a
        same-value ``setTau`` on a vehicle whose type comes from the hdv mix
        distribution makes SUMO clone the type (``normal_car@vehID``), which
        changes the captured type id and would break prefix identity. Before
        the first engagement the touched set is empty and no TraCI write
        happens at all.
        """
        for cav_id in sorted(self._wall_touched):
            if cav_id not in live_ids:
                continue
            try:
                tau = self._wall._original_tau(cav_id)
                release_to_sumo(cav_id, tau=tau)
            except Exception:
                continue
        self._wall_touched = set()

    def _mark_touched(self, members: list[str]) -> None:
        self._wall_touched.update(members)

    def _wall_audit(self, *, wall_active: bool, members: list[str], state: str) -> Dict[str, Any]:
        return {
            "relative_step": self._step,
            "wall_active": bool(wall_active),
            "treatment_state": state,
            "wall_override_ids": tuple(sorted(members)),
            "member_count": len(members),
        }

    def close(self) -> None:
        pass


class HysteresisWallPolicy(_RuleWallBase):
    """HRULE: legacy h-hysteresis on the sustained wall actuator."""

    def __init__(
        self,
        *,
        arm_start_step: int = 100,
        trigger_h: float = 0.25,
        release_h: float = 0.45,
        wall_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._arm_start_step = int(arm_start_step)
        self._trigger_h = float(trigger_h)
        self._release_h = float(release_h)
        self._wall_params = dict(wall_params or {})
        self._wall = _build_actuator(self._wall_params)
        self._on = False
        self._wall_touched: set[str] = set()
        self.engagements = 0
        self.first_engage_step: Optional[int] = None

    def reset(self, env: Any, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._on = False
        self._wall_touched = set()
        self.engagements = 0
        self.first_engage_step = None
        self.last_audit = {}
        self._wall.reset(env, seed=seed)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        live_ids = self._live_ids()
        members = self._band_members(obs_dict, live_ids)
        # One pre-step health read (EMA parity with the other arms).
        health: Optional[float] = None
        if self._env is not None:
            try:
                detail = self._env.metastable_health_detail()
                if detail:
                    health = float(detail[0])
            except Exception:
                health = None
        if self._on and health is not None and health > self._release_h:
            self._on = False
        # The rule observes h at the end of each step; the earliest arm
        # condition that can fire is the one observed at t = 100 (the pre-step
        # read of act(101)), which commands the wall from t = 101 onward.
        # Prefix identity with Arm C therefore holds exactly through t = 100.
        if (
            not self._on
            and self._step > self._arm_start_step
            and health is not None
            and health < self._trigger_h
        ):
            self._on = True
            self.engagements += 1
            self.first_engage_step = self.first_engage_step or self._step
        if self._on:
            self._wall.act(self._mainline_obs(obs_dict, live_ids))
            self._mark_touched(members)
            state = "HOLDING"
        else:
            self._release_touched(obs_dict, live_ids)
            state = "DAMPING"
        audit = self._wall_audit(wall_active=self._on, members=members, state=state)
        audit["engagements"] = self.engagements
        audit["first_engage_step"] = self.first_engage_step
        self.last_audit = audit
        return {}


class SingleEngagementWallPolicy(_RuleWallBase):
    """SE(t_d): engage at fixed rule-blind step; release on h > 0.45 or t=900."""

    def __init__(
        self,
        *,
        decision_step: int,
        release_h: float = 0.45,
        demand_end_s: float = 900.0,
        wall_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._decision_step = int(decision_step)
        self._release_h = float(release_h)
        self._demand_end_s = float(demand_end_s)
        self._wall_params = dict(wall_params or {})
        self._wall = _build_actuator(self._wall_params)
        self._state = "DAMPING"
        self._wall_touched: set[str] = set()

    def reset(self, env: Any, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._state = "DAMPING"
        self._wall_touched = set()
        self.last_audit = {}
        self._wall.reset(env, seed=seed)

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        live_ids = self._live_ids()
        members = self._band_members(obs_dict, live_ids)
        health: Optional[float] = None
        if self._env is not None:
            try:
                detail = self._env.metastable_health_detail()
                if detail:
                    health = float(detail[0])
            except Exception:
                health = None
        if self._state == "DAMPING" and self._step == self._decision_step + 1:
            self._state = "HOLDING"
            reason = None  # engagement step applies the wall; release checked next act
        elif self._state == "HOLDING":
            if health is not None and health > self._release_h:
                self._state = "RELEASED"
                reason = "health_release"
            elif self._step > self._demand_end_s:
                self._state = "RELEASED"
                reason = "demand_end"
            else:
                reason = None
        else:
            reason = None
        if self._state == "HOLDING" and reason is None:
            self._wall.act(self._mainline_obs(obs_dict, live_ids))
            self._mark_touched(members)
            state = "HOLDING"
        else:
            # Even during the release step, restore only wall-touched CAVs
            # (tau clone + override). Before engagement nothing is written.
            self._release_touched(obs_dict, live_ids)
            state = self._state
        audit = self._wall_audit(wall_active=self._state == "HOLDING", members=members, state=state)
        audit["release_reason"] = reason
        self.last_audit = audit
        return {}
