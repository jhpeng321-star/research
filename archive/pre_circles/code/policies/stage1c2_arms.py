"""Stage 1c-2 treatment arms: fixed holds, onset-triggered holds, R1b release.

All arms extend :class:`DampingPlusWallPolicy` (frozen damping every step,
in-band 200-400 m wall override at 10 m/s / tau+3.0 / 0.3 clip, touched-set
tau restore, 5 s linear taper). Hold durations are RELATIVE to the first
command step: the base cap is set to t_d + hold_s, so exactly ``hold_s``
seconds elapse between the first wall command (t_d + 1) and the taper start
(cap + 1). Under this convention the 1c references are W_HOLD hold_s = 650
and W_ORACLE hold_s = 950.

- :class:`FixedHoldDampingWallPolicy` - engagement at t_d + 1 (t_d = 250).
- :class:`OnsetTriggeredDampingWallPolicy` - engagement at a precomputed
  treatment-blind onset step (detected on the D branch before the run);
  bit-identical to D through that step.
- :class:`R1bReleaseDampingWallPolicy` - the corrected state release rule.
  Two fixes vs the 1c R1: the speed gate is evaluated over 400-800 m ONLY
  (1c gated on 200-800 m, which includes the walled 10 m/s band and was
  therefore unreachable - proven by 120/120 digest-identical branches), and
  the hard max is 900 s of hold relative to engagement (1c capped at
  absolute t = 900, i.e. only 650 s of hold).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import traci

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from policies.damping_plus_wall import DampingPlusWallPolicy

SPEED_REGION_LO_M = 400.0
SPEED_REGION_HI_M = 800.0
ENTRY_REGION_LO_M = 400.0
ENTRY_REGION_HI_M = 800.0
R1_ARRIVAL_RATIO = 0.9
R1_SPEED_FRACTION = 0.8


def _live_ids() -> set:
    try:
        return {str(v) for v in traci.vehicle.getIDList()}
    except Exception:
        return set()


def entry_and_speed_metrics(
    prev_entries: Optional[set], prev_on_merge: Optional[set]
) -> Dict[str, float]:
    """Arrival rate, discharge rate, and 400-800 m mean speed (veh/h, m/s).

    Entries are counted by membership change vs the previous step's sets;
    the caller stores them back on the policy.
    """
    live = _live_ids()
    speeds = []
    entries: set = set()
    on_merge: set = set()
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
        if SPEED_REGION_LO_M <= distance <= SPEED_REGION_HI_M:
            speeds.append(speed)
        if ENTRY_REGION_LO_M < distance <= ENTRY_REGION_HI_M:
            entries.add(vid)
    for vid in sorted(live):
        try:
            if str(traci.vehicle.getRoadID(vid)) == MERGE_EDGE:
                on_merge.add(vid)
        except Exception:
            continue
    new_entries = len(entries - (prev_entries or set()))
    discharge = len((prev_on_merge or set()) - on_merge)
    return {
        "arrival_rate_veh_h": float(new_entries * 120.0),
        "merge_discharge_veh_h": float(discharge * 120.0),
        "mean_speed_400_800_mps": (sum(speeds) / len(speeds)) if speeds else 0.0,
        "_entries": entries,
        "_on_merge": on_merge,
    }


class FixedHoldDampingWallPolicy(DampingPlusWallPolicy):
    """Damping + wall engaged at t_d + 1, held exactly ``hold_s``, 5 s taper."""

    def __init__(
        self,
        *,
        decision_step: int = 250,
        hold_s: float = 300.0,
        taper_duration_s: float = 5.0,
    ) -> None:
        super().__init__(
            decision_step=int(decision_step),
            wall_cap_s=float(decision_step) + float(hold_s),
            taper_duration_s=taper_duration_s,
        )
        self.hold_s = float(hold_s)


class OnsetTriggeredDampingWallPolicy(DampingPlusWallPolicy):
    """Damping + wall engaged at a precomputed onset step, fixed hold.

    ``onset_step`` is the D-branch step whose trailing 10 s of health are
    all below 0.15 (treatment-blind: computed on the D branch before the
    run). The base class issues the first command at decision_step + 1, so
    the branch is bit-identical to D through ``onset_step``.
    """

    def __init__(
        self,
        *,
        onset_step: int,
        hold_s: float = 300.0,
        taper_duration_s: float = 5.0,
    ) -> None:
        super().__init__(
            decision_step=int(onset_step),
            wall_cap_s=float(onset_step) + float(hold_s),
            taper_duration_s=taper_duration_s,
        )
        self.onset_step = int(onset_step)
        self.hold_s = float(hold_s)


class R1bReleaseDampingWallPolicy(DampingPlusWallPolicy):
    """Damping + wall with the corrected state release rule (R1b).

    Release when, for ``release_window_s`` consecutive seconds, ALL hold:
    (a) upstream arrival rate into 400-800 m (per 30 s, veh/h) <= 0.9 x the
    merge-edge discharge rate over the same 30 s; (b) mean speed over
    400-800 m ONLY >= 0.8 x free-flow. Minimum hold 120 s; hard max 900 s
    of hold relative to engagement; 5 s taper into the existing tau restore.
    """

    def __init__(
        self,
        *,
        decision_step: int = 250,
        min_hold_s: float = 120.0,
        hard_max_hold_s: float = 900.0,
        release_window_s: float = 60.0,
        free_flow_mps: Optional[float] = None,
    ) -> None:
        super().__init__(decision_step=int(decision_step), wall_cap_s=1.0e9)
        self._min_hold_s = float(min_hold_s)
        self._hard_max_hold_s = float(hard_max_hold_s)
        self._release_window_s = float(release_window_s)
        self._free_flow_mps = free_flow_mps
        self._window_count = 0
        self._prev_entries: Optional[set] = None
        self._prev_on_merge: Optional[set] = None
        self.rule_inputs: Dict[int, Dict[str, float]] = {}
        self.release_reason: Optional[str] = None

    def reset(self, env: Any, seed: int = None) -> None:
        super().reset(env, seed=seed)
        self._window_count = 0
        self._prev_entries = None
        self._prev_on_merge = None
        self.rule_inputs = {}
        self.release_reason = None
        if self._free_flow_mps is None:
            try:
                self._free_flow_mps = float(traci.lane.getMaxSpeed(PRE_MERGE_EDGE + "_0"))
            except Exception:
                self._free_flow_mps = 30.0

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        self._damping.act(obs_dict)
        live_ids = _live_ids()
        members = self._band_members(obs_dict, live_ids)
        now = entry_and_speed_metrics(self._prev_entries, self._prev_on_merge)
        self._prev_entries = now["_entries"]
        self._prev_on_merge = now["_on_merge"]
        self.rule_inputs[self._step] = {
            "arrival_rate_veh_h": now["arrival_rate_veh_h"],
            "merge_discharge_veh_h": now["merge_discharge_veh_h"],
            "mean_speed_400_800_mps": now["mean_speed_400_800_mps"],
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
            if elapsed >= self._hard_max_hold_s:
                self._begin_taper(members)
                self.release_reason = "hard_max_hold"
            elif elapsed >= self._min_hold_s and self._release_condition(now):
                self._window_count += 1
                if self._window_count >= self._release_window_s:
                    self._begin_taper(members)
                    self.release_reason = "rule"
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
        arrival = now["arrival_rate_veh_h"]
        discharge = now["merge_discharge_veh_h"]
        speed_ok = now["mean_speed_400_800_mps"] >= R1_SPEED_FRACTION * float(self._free_flow_mps)
        return arrival <= R1_ARRIVAL_RATIO * discharge and speed_ok
