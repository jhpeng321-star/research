"""Sustained health-gated moving wall (Stage 1a effect-existence pilot).

Arm W definition (see ``experiments/idm/sustained_wall_v1_contract.py`` for
the frozen numeric values):

- The wall regime is commanded by an externally supplied decision landmark
  ``t_d`` (computed on the identical control prefix). The first wall command
  is issued at relative step ``t_d + 1`` — the twin-replay convention where
  the decision is the last observation step and commands start the next step.
- Actuator semantics are the legacy band-passage moving wall
  (``policies.inflow_control.WallHoldPolicy``): every CAV whose
  distance-to-merge lies in [200, 400] m on a mainline lane is a member while
  it is in the band; members hold wall_speed = 10 m/s with wall_tau = 3.0 s
  (set via TraCI ``setTau``), speed reduction rate-limited to 0.3 m/s²; a CAV
  leaving the band is released to native IDM immediately (tau restored).
  Membership is by passage: no latched cohort, no Gate F, no compactness
  requirement.
- Release: first of (i) h > 0.45 for 20 consecutive seconds, (ii) t_d + 180 s
  of wall, (iii) t = 900 s. On release the existing 5 s linear taper is
  applied to the current members, then tau is restored.

Implementation choice (recorded): the CounterfactualEscalationPolicy wall
path was examined and rejected for this arm — its latched member-slot cohort,
formation feasibility gates, activation-safety vetos, abort machinery and
step-boundary-only taper do not implement the specified membership-by-passage
regime, and it cannot express the health-persistence release. WallHoldPolicy
is therefore wrapped directly by the new policy class below. Prefix identity,
digests and outcomes still reuse ``counterfactual/twin_replay`` helpers
(``capture_step``, ``compare_traces``, ``ReplayTrace``) and
``counterfactual/outcomes`` in the new runner.

Health-EMA discipline: every arm's policy calls
``env.metastable_health_detail()`` exactly once per ``act`` (pre-step state).
The runner's corridor observation adds exactly one more call per step
(post-step state), so the EMA advance pattern is arm-invariant and prefix
health series stay comparable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import traci

from config.constants import PRE_MERGE_EDGE
from env.features import distance_to_merge
from policies import DirectControlPolicy
from policies.inflow_control import WallHoldPolicy
from policies.traci_control import release_to_sumo


def band_distance_in_m(candidate: Tuple[str, str, float]) -> bool:
    """Band-passage predicate on a (road_id, lane_id, distance_to_merge_m)
    row, restricted to mainline approach lanes."""
    road_id, _lane_id, distance_m = candidate
    return road_id == PRE_MERGE_EDGE and 200.0 <= float(distance_m) <= 400.0


def select_nudge_candidate(
    candidates: Sequence[Tuple[str, float]], landmark_m: float
) -> Optional[str]:
    """The CAV nearest the landmark distance-to-merge point (ties by ID order).

    ``candidates`` are (vehicle_id, distance_to_merge_m) rows already
    restricted to mainline lanes.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda row: (abs(float(row[1]) - float(landmark_m)), str(row[0])))[
        0
    ]


class SustainedWallPolicy(DirectControlPolicy):
    """Native IDM plus the sustained health-gated moving wall (Arm W)."""

    direct_control = True

    def __init__(
        self,
        *,
        decision_step: int,
        wall_params: Optional[Mapping[str, Any]] = None,
        recovery_h: float = 0.45,
        recovery_persist_s: float = 20.0,
        max_wall_duration_s: float = 180.0,
        wall_cap_s: float = 900.0,
        taper_duration_s: float = 5.0,
    ) -> None:
        super().__init__()
        if int(decision_step) < 0:
            raise ValueError("decision_step must be non-negative")
        params = dict(
            d_lo=200.0,
            d_hi=400.0,
            wall_speed=10.0,
            wall_tau=3.0,
            a_max_dec=0.3,
            tau_ramp_step=0.0,
        )
        params.update(dict(wall_params or {}))
        self._decision_step = int(decision_step)
        self._recovery_h = float(recovery_h)
        self._recovery_persist_s = max(1.0, float(recovery_persist_s))
        self._max_wall_duration_s = max(1.0, float(max_wall_duration_s))
        self._wall_cap_s = float(wall_cap_s)
        self._taper_duration_s = max(1.0, float(taper_duration_s))
        self._wall = WallHoldPolicy(**params)
        self._env: Any = None
        self._step = 0
        self._state = "DAMPING"
        self._healthy_run = 0.0
        self._release_reason: Optional[str] = None
        self._release_step: Optional[int] = None
        self._taper_members: list[str] = []
        self._taper_origin_targets_mps: dict[str, float] = {}
        self._taper_elapsed_steps = 0
        self.dose_telemetry: list[dict[str, Any]] = []
        self.last_audit: Dict[str, Any] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._state = "DAMPING"
        self._healthy_run = 0.0
        self._release_reason = None
        self._release_step = None
        self._taper_members = []
        self._taper_origin_targets_mps = {}
        self._taper_elapsed_steps = 0
        self.dose_telemetry = []
        self.last_audit = {}
        self._wall.reset(env, seed=seed)

    # -- live state helpers -------------------------------------------
    @staticmethod
    def _live_ids() -> set[str]:
        try:
            return {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        except Exception:
            return set()

    @staticmethod
    def _road_id(vehicle_id: str) -> Optional[str]:
        try:
            return str(traci.vehicle.getRoadID(vehicle_id))
        except Exception:
            return None

    @staticmethod
    def _lane_id(vehicle_id: str) -> Optional[str]:
        try:
            return str(traci.vehicle.getLaneID(vehicle_id))
        except Exception:
            return None

    def _mainline_band_members(
        self, obs_dict: Mapping[str, Any], live_ids: set[str]
    ) -> tuple[list[str], dict[str, int], dict[str, int]]:
        """(member ids, members per lane, covered lanes) for mainline CAVs."""
        members: list[str] = []
        per_lane: dict[str, int] = {}
        covered: dict[str, int] = {}
        for cav_id in sorted(str(value) for value in obs_dict.keys()):
            if cav_id not in live_ids:
                continue
            if self._road_id(cav_id) != PRE_MERGE_EDGE:
                continue
            try:
                distance_m = float(distance_to_merge(cav_id))
            except Exception:
                continue
            lane_id = self._lane_id(cav_id)
            if 200.0 <= distance_m <= 400.0:
                members.append(cav_id)
                if lane_id is not None:
                    per_lane[lane_id] = per_lane.get(lane_id, 0) + 1
            if lane_id is not None:
                covered[lane_id] = covered.get(lane_id, 0) + 1
        return sorted(members), per_lane, covered

    def _mainline_obs(
        self, obs_dict: Mapping[str, Any], live_ids: set[str]
    ) -> dict[str, Any]:
        return {
            str(cav_id): obs_dict[cav_id]
            for cav_id in obs_dict
            if str(cav_id) in live_ids and self._road_id(str(cav_id)) == PRE_MERGE_EDGE
        }

    # -- treatment state machine --------------------------------------
    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        # Health-EMA discipline: exactly one pre-step advance, in every state.
        health: Optional[float] = None
        if self._env is not None:
            try:
                detail = self._env.metastable_health_detail()
                if detail:
                    health = float(detail[0])
            except Exception:
                health = None

        live_ids = self._live_ids()
        members, per_lane, covered_lanes = self._mainline_band_members(obs_dict, live_ids)

        if self._state == "DAMPING":
            if self._step == self._decision_step + 1:
                self._state = "HOLDING"
                self._healthy_run = 0.0
                self._release_reason = None
                self._release_step = None
                self._taper_members = []
                self._taper_origin_targets_mps = {}
                self._taper_elapsed_steps = 0
            else:
                self.last_audit = self._audit(
                    state="DAMPING", members=(), per_lane={}, coverage_fraction=0.0
                )
                return {}

        if self._state == "HOLDING":
            if health is not None and health > self._recovery_h:
                self._healthy_run += 1.0
            else:
                self._healthy_run = 0.0
            applied_seconds = self._step - (self._decision_step + 1)
            if self._healthy_run >= self._recovery_persist_s:
                self._begin_taper("health_recovery", members)
            elif applied_seconds >= self._max_wall_duration_s:
                self._begin_taper("max_duration", members)
            elif self._step > self._wall_cap_s:
                self._begin_taper("time_cap", members)
            else:
                self._apply_wall(members, obs_dict, live_ids)
                coverage_fraction = self._coverage_fraction(per_lane, covered_lanes, live_ids)
                self.dose_telemetry.append(
                    {
                        "step": self._step,
                        "member_count": len(members),
                        "members_per_lane": dict(per_lane),
                        "lane_coverage_fraction": coverage_fraction,
                    }
                )
                self.last_audit = self._audit(
                    state="HOLDING",
                    members=members,
                    per_lane=per_lane,
                    coverage_fraction=coverage_fraction,
                    applied_seconds=applied_seconds,
                    healthy_run=self._healthy_run,
                )
                return {}

        if self._state == "TAPERING":
            self._apply_taper(obs_dict)
            self.last_audit = self._audit(
                state="TAPERING",
                members=list(self._taper_members),
                per_lane={},
                coverage_fraction=0.0,
                applied_seconds=self._step - (self._decision_step + 1),
                healthy_run=self._healthy_run,
            )
            return {}

        # RELEASED: no further commands.
        self.last_audit = self._audit(
            state="RELEASED",
            members=(),
            per_lane={},
            coverage_fraction=0.0,
            applied_seconds=self._step - (self._decision_step + 1),
            healthy_run=self._healthy_run,
        )
        return {}

    def _coverage_fraction(
        self, per_lane: dict[str, int], covered_lanes: dict[str, int], live_ids: set[str]
    ) -> float:
        """Fraction of occupied mainline lanes (any vehicle) that have a member."""
        occupied = len(covered_lanes)
        if occupied <= 0:
            return 0.0
        return len(per_lane) / occupied

    def _apply_wall(
        self, members: list[str], obs_dict: Dict[str, Any], live_ids: set[str]
    ) -> None:
        # Legacy band-passage actuator: every mainline CAV is passed through so
        # in-band CAVs hold wall speed/tau and out-of-band CAVs are released to
        # native IDM with their original tau. This runs even with zero current
        # members so CAVs that left the band are released immediately.
        self._wall.act(self._mainline_obs(obs_dict, live_ids))

    def _begin_taper(self, reason: str, members: list[str]) -> None:
        self._state = "TAPERING"
        self._taper_reason = reason
        self._taper_elapsed_steps = 0
        self._taper_members = list(members)
        self._taper_origin_targets_mps = {}
        live_ids = self._live_ids()
        for member_id in self._taper_members:
            if member_id not in live_ids:
                continue
            try:
                self._taper_origin_targets_mps[member_id] = max(
                    0.0, float(traci.vehicle.getSpeed(member_id))
                )
            except Exception:
                continue
        self._release_reason = reason
        self._release_step = self._step
        if not self._taper_members:
            self._state = "RELEASED"

    def _taper_step_count(self) -> int:
        return max(1, round(self._taper_duration_s / 1.0))

    def _native_speed_mps(self, member_id: str) -> Optional[float]:
        try:
            return max(0.0, float(traci.vehicle.getSpeedWithoutTraCI(member_id)))
        except Exception:
            try:
                return max(0.0, float(traci.vehicle.getSpeed(member_id)))
            except Exception:
                return None

    def _apply_taper(self, obs_dict: Dict[str, Any]) -> None:
        self._taper_elapsed_steps += 1
        total = self._taper_step_count()
        blend = max(0.0, 1.0 - self._taper_elapsed_steps / total)
        live_ids = self._live_ids()
        member_obs: dict[str, Any] = {}
        targets: dict[str, float] = {}
        for member_id in self._taper_members:
            if member_id not in live_ids or member_id not in obs_dict:
                continue
            base = self._native_speed_mps(member_id)
            if base is None:
                continue
            origin = self._taper_origin_targets_mps.get(member_id, base)
            targets[member_id] = max(0.0, base + blend * (origin - base))
            member_obs[member_id] = obs_dict[member_id]
        if member_obs:
            self._wall.act_with_targets(member_obs, targets)
            # Existing linear taper of the wall-only tau offset.
            self._wall.set_tau_blend(tuple(self._taper_members), blend)
        if blend <= 0.0:
            self._complete_taper()

    def _complete_taper(self) -> None:
        self._release_reason = self._release_reason or "max_duration"
        self._release_step = self._step
        tau_restorations = self._wall.restore_tau(self._taper_members)
        live_ids = self._live_ids()
        releases: list[dict[str, Any]] = []
        for member_id in self._taper_members:
            if member_id not in live_ids:
                continue
            try:
                traci.vehicle.setSpeed(member_id, -1.0)
                releases.append({"vehicle_id": member_id, "released": True})
            except Exception as exc:
                releases.append(
                    {"vehicle_id": member_id, "released": False, "error": type(exc).__name__}
                )
        self.last_taper_release = {
            "tau_restorations": tau_restorations,
            "speed_releases": releases,
        }
        self._state = "RELEASED"

    def _audit(
        self,
        *,
        state: str,
        members: Iterable[str],
        per_lane: dict[str, int],
        coverage_fraction: float,
        applied_seconds: Optional[float] = None,
        healthy_run: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "relative_step": self._step,
            "wall_active": state in ("HOLDING", "TAPERING"),
            "treatment_state": state,
            "wall_override_ids": tuple(sorted(str(x) for x in members)),
            "member_count": len(tuple(members)),
            "members_per_lane": dict(per_lane),
            "lane_coverage_fraction": float(coverage_fraction),
            "wall_elapsed_s": float(applied_seconds) if applied_seconds is not None else None,
            "healthy_run_s": float(healthy_run) if healthy_run is not None else None,
            "release_reason": self._release_reason,
            "release_step": self._release_step,
            "taper_elapsed_steps": self._taper_elapsed_steps,
        }

    def close(self) -> None:
        pass


class PerturbationPolicy(DirectControlPolicy):
    """Arm P: a single one-step 0.5 m/s slowDown at the landmark CAV.

    At relative step ``t_d + 1`` (same convention as the wall arm) the CAV
    nearest the 300 m distance-to-merge point on any mainline lane has its
    speed command reduced by 0.5 m/s for exactly one controller step; at
    ``t_d + 2`` the override is cleared (setSpeed -1) and the CAV returns to
    native IDM. Nothing else is commanded.
    """

    direct_control = True

    def __init__(
        self,
        *,
        decision_step: int,
        slow_down_mps: float = 0.5,
        landmark_distance_m: float = 300.0,
    ) -> None:
        super().__init__()
        self._decision_step = int(decision_step)
        self._slow_down_mps = float(slow_down_mps)
        self._landmark_distance_m = float(landmark_distance_m)
        self._env: Any = None
        self._step = 0
        self._nudge_vehicle_id: Optional[str] = None
        self.last_audit: Dict[str, Any] = {}

    def reset(self, env: Any, seed: int = None) -> None:
        self._env = env
        self._step = 0
        self._nudge_vehicle_id = None
        self.last_audit = {}

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
        # Health-EMA discipline (identical call pattern to all other arms).
        if self._env is not None:
            try:
                self._env.metastable_health_detail()
            except Exception:
                pass
        nudge_applied = False
        if self._step == self._decision_step + 1:
            candidates: list[tuple[str, float]] = []
            for cav_id in obs_dict:
                try:
                    if str(traci.vehicle.getRoadID(cav_id)) != PRE_MERGE_EDGE:
                        continue
                    distance_m = float(distance_to_merge(cav_id))
                except Exception:
                    continue
                candidates.append((str(cav_id), distance_m))
            self._nudge_vehicle_id = select_nudge_candidate(
                candidates, self._landmark_distance_m
            )
            if self._nudge_vehicle_id is not None:
                try:
                    speed = max(
                        0.0, float(traci.vehicle.getSpeed(self._nudge_vehicle_id))
                    )
                    traci.vehicle.setSpeed(
                        self._nudge_vehicle_id,
                        max(0.0, speed - self._slow_down_mps),
                    )
                    nudge_applied = True
                except Exception as exc:
                    self.last_audit = {
                        "relative_step": self._step,
                        "nudge_vehicle_id": self._nudge_vehicle_id,
                        "nudge_applied": False,
                        "nudge_error": type(exc).__name__,
                    }
                    return {}
        elif self._step == self._decision_step + 2 and self._nudge_vehicle_id is not None:
            try:
                traci.vehicle.setSpeed(self._nudge_vehicle_id, -1.0)
            except Exception:
                pass
        self.last_audit = {
            "relative_step": self._step,
            "nudge_vehicle_id": self._nudge_vehicle_id,
            "nudge_applied": nudge_applied,
        }
        return {}

    def close(self) -> None:
        pass
