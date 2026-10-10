"""Minimum-hold single-engagement wall (Stage 1b-prep-4 Part B).

SE250_H{120,300}: engage at the fixed rule-blind step t_d (first command at
t_d + 1); hold the wall for at least H_min seconds; after H_min, release
when h > 0.45 has persisted for 30 consecutive seconds (pre-step health
readings, one read per controller step, same pattern as every other arm), or
at the demand end (t = 900 s); then the existing 5 s linear taper and tau
restore; no re-engagement.

The actuator and taper machinery are inherited from
``policies.sustained_wall.SustainedWallPolicy`` (WallHoldPolicy band-passage
actuator, 0.3 m/s^2 clip, act_with_targets + set_tau_blend taper); only the
release state machine differs, so only ``act`` is overridden.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from policies.sustained_wall import SustainedWallPolicy


class MinimumHoldWallPolicy(SustainedWallPolicy):
    """Single engagement with a minimum-hold and a 30 s health-persistence release."""

    def __init__(
        self,
        *,
        decision_step: int,
        min_hold_s: float = 120.0,
        recovery_h: float = 0.45,
        recovery_persist_s: float = 30.0,
        wall_cap_s: float = 900.0,
        taper_duration_s: float = 5.0,
        wall_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            decision_step=int(decision_step),
            wall_params=wall_params,
            recovery_h=recovery_h,
            recovery_persist_s=recovery_persist_s,
            max_wall_duration_s=1.0e9,
            wall_cap_s=wall_cap_s,
            taper_duration_s=taper_duration_s,
        )
        self._min_hold_s = float(min_hold_s)
        self._min_hold_reached = False
        self._release_reason = None
        self._release_step = None

    def reset(self, env: Any, seed: int = None) -> None:
        super().reset(env, seed=seed)
        self._min_hold_reached = False
        self._release_reason = None
        self._release_step = None

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        self._step += 1
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
                self._min_hold_reached = False
                self._release_reason = None
                self._release_step = None
            else:
                self.last_audit = self._audit(
                    state="DAMPING", members=(), per_lane={}, coverage_fraction=0.0
                )
                return {}

        if self._state == "HOLDING":
            applied_seconds = self._step - (self._decision_step + 1)
            if applied_seconds >= self._min_hold_s and not self._min_hold_reached:
                self._min_hold_reached = True
                self._healthy_run = 0.0  # persistence counted only inside the window
            if self._min_hold_reached:
                if health is not None and health > self._recovery_h:
                    self._healthy_run += 1.0
                else:
                    self._healthy_run = 0.0
                if self._healthy_run >= self._recovery_persist_s:
                    self._begin_taper("health_release", members)
            if self._step > self._wall_cap_s and self._state == "HOLDING":
                self._begin_taper("demand_end", members)
            if self._state == "HOLDING":
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

        self.last_audit = self._audit(
            state="RELEASED",
            members=(),
            per_lane={},
            coverage_fraction=0.0,
            applied_seconds=self._step - (self._decision_step + 1),
            healthy_run=self._healthy_run,
        )
        return {}
