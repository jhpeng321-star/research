"""Stage 1c deployable release rules (R1, R2) on the damping-plus-wall arm.

Extends ``DampingPlusWallPolicy`` with an online release rule evaluated at
1 Hz on the pre-step state:

- W_R1: after a 120 s minimum hold, release when for 60 consecutive
  readings ALL hold: (a) upstream arrival rate into the 400-800 m approach
  region per 30 s (scaled to veh/h) <= 0.9 x merge-edge discharge over the
  same 30 s; (b) mean speed over 200-800 m >= 0.8 x free-flow speed
  (approach-edge speed limit x median speedFactor). Hard max 900 s.
- W_R2: after the minimum hold, release when the 200-800 m approach density
  has stayed below a treatment-blind free-flow threshold for 60 consecutive
  seconds; the threshold is computed from the same seed's D-prefix density
  over steps 1-100 (90th percentile) and passed in. Hard max 900 s.

Both release into the existing 5 s linear taper and tau restore, exactly as
the damping+wall composite already does.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import traci

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH, RAMP_EDGE
from policies.damping_plus_wall import DampingPlusWallPolicy

APPROACH_LO = 200.0
APPROACH_HI = 800.0
ENTRY_REGION_LO = 400.0
ENTRY_REGION_HI = 800.0
APPROACH_LANES = 3


def _live_ids() -> set[str]:
    try:
        return {str(v) for v in traci.vehicle.getIDList()}
    except Exception:
        return set()


def approach_metrics(prev_ids: Optional[set[str]], prev_on_merge: Optional[set[str]],
                     env: Any) -> Dict[str, float]:
    """Compute the release-rule inputs from the live state.

    Entries are counted by membership change vs the previous step's sets,
    so previous sets must be passed back in (``_prev_entry_ids`` /
    ``_prev_on_merge`` on the policy are used).
    """
    live = _live_ids()
    speeds = []
    entry_ids: set[str] = set()
    on_merge: set[str] = set()
    for vid in sorted(live):
        try:
            road = str(traci.vehicle.getRoadID(vid))
            if road != PRE_MERGE_EDGE:
                continue
            lane_pos = float(traci.vehicle.getLanePosition(vid))
            speed = float(traci.vehicle.getSpeed(vid))
        except Exception:
            continue
        distance = max(0.0, PRE_MERGE_LANE_LENGTH - lane_pos)
        if APPROACH_LO <= distance <= APPROACH_HI:
            speeds.append(speed)
        if ENTRY_REGION_LO < distance <= ENTRY_REGION_HI:
            entry_ids.add(vid)
    for vid in sorted(live):
        try:
            if str(traci.vehicle.getRoadID(vid)) == MERGE_EDGE:
                on_merge.add(vid)
        except Exception:
            continue
    new_entries = len(entry_ids - (prev_ids or set()))
    discharge = len((prev_on_merge or set()) - on_merge)
    density = len(speeds) / ((APPROACH_HI - APPROACH_LO) / 1000.0 * APPROACH_LANES)
    return {
        "arrival_rate_veh_h": new_entries * 120.0,
        "merge_discharge_veh_h": discharge * 120.0,
        "approach_mean_speed_mps": (sum(speeds) / len(speeds)) if speeds else 0.0,
        "approach_density_veh_km_lane": density,
        "_entry_ids": entry_ids,
        "_on_merge": on_merge,
    }


class ReleaseRuleDampingWallPolicy(DampingPlusWallPolicy):
    """Damping + wall with an online release rule (R1 or R2)."""

    def __init__(
        self,
        *,
        decision_step: int = 250,
        rule: str = "R1",
        r2_threshold_veh_km_lane: Optional[float] = None,
        min_hold_s: float = 120.0,
        hard_max_hold_s: float = 900.0,
        release_window_s: float = 60.0,
        free_flow_mps: Optional[float] = None,
    ) -> None:
        super().__init__(decision_step=decision_step, wall_cap_s=hard_max_hold_s)
        if rule not in ("R1", "R2"):
            raise ValueError(f"unknown release rule {rule!r}")
        self._rule = rule
        self._r2_threshold = r2_threshold_veh_km_lane
        self._min_hold_s = float(min_hold_s)
        self._hard_max_s = float(hard_max_hold_s)
        self._release_window_s = float(release_window_s)
        self._free_flow_mps = free_flow_mps
        self._window_count = 0
        self._prev_entry_ids: Optional[set[str]] = None
        self._prev_on_merge: Optional[set[str]] = None
        self.rule_inputs: Dict[int, Dict[str, float]] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        super().reset(env, seed=seed)
        self._window_count = 0
        self._prev_entry_ids = None
        self._prev_on_merge = None
        self.rule_inputs = {}
        if self._free_flow_mps is None:
            try:
                self._free_flow_mps = float(traci.lane.getMaxSpeed(PRE_MERGE_EDGE + "_0"))
            except Exception:
                self._free_flow_mps = 30.0

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        self._damping.act(obs_dict)
        live_ids = self._live_ids()
        members = self._band_members(obs_dict, live_ids)
        now = approach_metrics(self._prev_entry_ids, self._prev_on_merge, self._env)
        self._prev_entry_ids = now["_entry_ids"]
        self._prev_on_merge = now["_on_merge"]
        self.rule_inputs[self._step] = {
            "arrival_rate_veh_h": now["arrival_rate_veh_h"],
            "merge_discharge_veh_h": now["merge_discharge_veh_h"],
            "approach_mean_speed_mps": now["approach_mean_speed_mps"],
            "approach_density_veh_km_lane": now["approach_density_veh_km_lane"],
        }

        if self._state == "DAMPING":
            if self._step != self._decision_step + 1:
                self.last_audit = {"relative_step": self._step, "wall_active": False,
                                   "treatment_state": "DAMPING", "wall_override_ids": (),
                                   "member_count": 0}
                return {}
            self._state = "HOLDING"
        if self._state == "HOLDING":
            elapsed = self._step - (self._decision_step + 1)
            if self._step > self._hard_max_s:
                self._begin_taper(members)
            elif elapsed >= self._min_hold_s and self._release_condition(now):
                self._window_count += 1
                if self._window_count >= self._release_window_s:
                    self._begin_taper(members)
            else:
                self._window_count = 0
            if self._state == "HOLDING":
                if members:
                    self._wall.act({m: obs_dict[m] for m in members})
                    self._touched.update(members)
                self._restore_tau([m for m in self._touched if m not in members])
                self.last_audit = {"relative_step": self._step, "wall_active": True,
                                   "treatment_state": "HOLDING",
                                   "wall_override_ids": tuple(sorted(members)),
                                   "member_count": len(members),
                                   "release_window_count": self._window_count}
                return {}
        if self._state == "TAPERING":
            self._apply_taper(obs_dict)
            self.last_audit = {"relative_step": self._step, "wall_active": True,
                               "treatment_state": "TAPERING",
                               "wall_override_ids": tuple(sorted(self._taper_members)),
                               "member_count": len(self._taper_members)}
            return {}
        self.last_audit = {"relative_step": self._step, "wall_active": False,
                           "treatment_state": "RELEASED", "wall_override_ids": (),
                           "member_count": 0}
        return {}

    def _release_condition(self, now: Dict[str, float]) -> bool:
        if self._rule == "R1":
            arrival = now["arrival_rate_veh_h"]
            discharge = now["merge_discharge_veh_h"]
            speed_ok = now["approach_mean_speed_mps"] >= 0.8 * float(self._free_flow_mps)
            return arrival <= 0.9 * discharge and speed_ok
        if self._r2_threshold is None:
            return False
        return now["approach_density_veh_km_lane"] < self._r2_threshold
