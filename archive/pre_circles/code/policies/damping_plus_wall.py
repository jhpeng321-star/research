"""Damping-plus-wall full-hold composite (Stage 1b-prep-5 amendment C).

Treatment = the frozen analytical damping controller (applied every step to
every active CAV) PLUS the band-passage wall override on in-band mainline
CAVs (200-400 m, 10 m/s, tau +3.0, 0.3 m/s^2 clip, instant tau) engaged
rule-blind at t_d + 1 and held until t = 900 s, then the existing 5 s linear
taper and tau restore. This mirrors the counterfactual escalation contract:
the wall can only override a subset of the damping commands (in-band CAVs);
everything else keeps the damping action. No health-based release.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import traci

from config.constants import PRE_MERGE_EDGE
from env.features import distance_to_merge
from policies import DirectControlPolicy
from policies.idm_analytical import build_frozen_idm_analytical_policy
from policies.inflow_control import WallHoldPolicy

DEFAULT_WALL_PARAMS: Dict[str, float] = {
    "d_lo": 200.0,
    "d_hi": 400.0,
    "wall_speed": 10.0,
    "wall_tau": 3.0,
    "a_max_dec": 0.3,
}


class DampingPlusWallPolicy(DirectControlPolicy):
    direct_control = True

    def __init__(
        self,
        *,
        decision_step: int = 250,
        wall_params: Optional[Mapping[str, Any]] = None,
        wall_cap_s: float = 900.0,
        taper_duration_s: float = 5.0,
    ) -> None:
        super().__init__()
        self._decision_step = int(decision_step)
        self._wall_params = dict(DEFAULT_WALL_PARAMS)
        self._wall_params.update(dict(wall_params or {}))
        self._wall_params["trigger_h"] = None
        self._wall = WallHoldPolicy(**self._wall_params)
        self._damping = build_frozen_idm_analytical_policy()
        self._env: Any = None
        self._step = 0
        self._state = "DAMPING"
        self._taper_duration_s = max(1.0, float(taper_duration_s))
        self._wall_cap_s = float(wall_cap_s)
        self._taper_members: list[str] = []
        self._release_reason: Optional[str] = None
        self._release_step: Optional[int] = None
        self._touched: set[str] = set()
        self._taper_elapsed_steps = 0
        self._taper_origin_targets_mps: dict[str, float] = {}
        self.last_audit: Dict[str, Any] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._state = "DAMPING"
        self._touched = set()
        self._taper_elapsed_steps = 0
        self._taper_origin_targets_mps = {}
        self.last_audit = {}
        self._damping.reset(env, seed=seed)
        self._wall.reset(env, seed=seed)

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
            if self._wall_params["d_lo"] <= distance <= self._wall_params["d_hi"]:
                members.append(cav_id)
        return members

    def _damping_target_mps(self, vehicle_id: str) -> Optional[float]:
        diagnostics = getattr(self._damping, "last_diagnostics", {}) or {}
        for row in diagnostics.get("commands", ()):
            if isinstance(row, Mapping) and str(row.get("vehicle_id")) == vehicle_id:
                value = row.get("target_speed_mps")
                if value is not None:
                    try:
                        return max(0.0, float(value))
                    except (TypeError, ValueError):
                        return None
        return None

    def _taper_step_count(self) -> int:
        return max(1, round(self._taper_duration_s / 1.0))

    def _begin_taper(self, members: list[str]) -> None:
        self._state = "TAPERING"
        self._taper_members = list(members)
        self._taper_elapsed_steps = 0
        self._taper_origin_targets_mps = {}
        live_ids = self._live_ids()
        for member_id in self._taper_members:
            if member_id in live_ids:
                try:
                    self._taper_origin_targets_mps[member_id] = max(
                        0.0, float(traci.vehicle.getSpeed(member_id))
                    )
                except Exception:
                    continue
        self._release_reason = "demand_end"
        self._release_step = self._step

    def _apply_taper(self, obs_dict: Mapping[str, Any]) -> None:
        self._taper_elapsed_steps += 1
        total = self._taper_step_count()
        blend = max(0.0, 1.0 - self._taper_elapsed_steps / total)
        live_ids = self._live_ids()
        targets: dict[str, float] = {}
        member_obs: dict[str, Any] = {}
        for member_id in self._taper_members:
            if member_id not in live_ids or member_id not in obs_dict:
                continue
            base = self._damping_target_mps(member_id)
            if base is None:
                try:
                    base = max(0.0, float(traci.vehicle.getSpeed(member_id)))
                except Exception:
                    continue
            origin = self._taper_origin_targets_mps.get(member_id, base)
            targets[member_id] = max(0.0, base + blend * (origin - base))
            member_obs[member_id] = obs_dict[member_id]
        if member_obs:
            self._wall.act_with_targets(member_obs, targets)
            self._wall.set_tau_blend(self._taper_members, blend)
        if blend <= 0.0:
            self._complete_taper()

    def _complete_taper(self) -> None:
        self._restore_tau(self._taper_members)
        self._state = "RELEASED"

    def _restore_tau(self, ids: list[str]) -> None:
        live_ids = self._live_ids()
        for cav_id in ids:
            if cav_id not in live_ids:
                continue
            try:
                tau = self._wall._original_tau(cav_id)
                traci.vehicle.setTau(cav_id, tau)
                traci.vehicle.setSpeed(cav_id, -1.0)
            except Exception:
                continue

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        # Damping first, every step: the control arm action that stays active
        # for every non-overridden CAV (and for the whole episode in D).
        self._damping.act(obs_dict)
        live_ids = self._live_ids()
        members = self._band_members(obs_dict, live_ids)

        if self._state == "DAMPING":
            if self._step != self._decision_step + 1:
                self.last_audit = {
                    "relative_step": self._step,
                    "wall_active": False,
                    "treatment_state": "DAMPING",
                    "wall_override_ids": (),
                    "member_count": 0,
                }
                return {}
            self._state = "HOLDING"
        if self._state == "HOLDING":
            if self._step > self._wall_cap_s:
                self._begin_taper(members)
            else:
                if members:
                    self._wall.act({m: obs_dict[m] for m in members})
                    self._touched.update(members)
                # Release touched CAVs that left the band (damping resumes next step).
                self._restore_tau([m for m in self._touched if m not in members])
                self.last_audit = {
                    "relative_step": self._step,
                    "wall_active": True,
                    "treatment_state": "HOLDING",
                    "wall_override_ids": tuple(sorted(members)),
                    "member_count": len(members),
                    "touched_count": len(self._touched),
                }
                return {}
        if self._state == "TAPERING":
            self._apply_taper(obs_dict)
            self.last_audit = {
                "relative_step": self._step,
                "wall_active": True,
                "treatment_state": "TAPERING",
                "wall_override_ids": tuple(sorted(self._taper_members)),
                "member_count": len(self._taper_members),
            }
            return {}
        self.last_audit = {
            "relative_step": self._step,
            "wall_active": False,
            "treatment_state": "RELEASED",
            "wall_override_ids": (),
            "member_count": 0,
        }
        return {}

    def close(self) -> None:
        try:
            self._damping.close()
        except Exception:
            pass
