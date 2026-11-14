"""Version 2.1 recovery prototype, not an accepted controller.

Native-selectable local feedback; no nominal-IDM substitution.

Active mode commands an explicitly bounded acceleration target. Native mode
releases speed ownership; the active target is never called a native residual.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from env.native_damping import DampingContract, LocalHistory
from env.native_damping_sumo import SumoDampingActuator


@dataclass(frozen=True)
class SelectableContract:
    control_dt_s: float = 1.0
    filter_seconds: float = 5.0
    cooldown_seconds: float = 5.0
    residual_min: float = -0.5
    residual_max: float = 0.3
    comfortable_decel: float = 1.0
    comfortable_accel: float = 0.5
    jerk_limit: float = 0.5
    observation_dim: int = 89

    def manifest(self):
        payload = dict(
            schema="native_selectable_recovery_damping/2.1",
            **asdict(self),
            sensor=DampingContract().manifest(),
            extra_fields=[
                "filtered_ego_speed_over_30",
                "filter_ready",
                "owned_mode",
                "native_cooldown_fraction",
            ],
            action=["native_or_active", "active_expert_correction_mps2"],
            active="gap_relative_speed_ema_feedback_plus_learned_correction",
            recovery_gains={
                "gap_per_s2": 0.25,
                "relative_speed_per_s": 0.8,
                "ema_per_s": 0.05,
                "headway_s": 1.5,
                "standstill_gap_m": 2.5,
            },
            native="setSpeed(-1)_on_release_no_tau_change",
        )
        return {
            **payload,
            "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        }


class FilteredHistory(LocalHistory):
    def __init__(self):
        super().__init__(DampingContract())
        self.filtered = {}
        self.first_time = {}

    def add(self, frame, *, now_s):
        previous = self.frames.get(frame.vehicle_id, [])
        duplicate = bool(previous and previous[-1].timestamp_s == frame.timestamp_s)
        super().add(frame, now_s=now_s)
        if duplicate:
            return
        cav = frame.vehicle_id
        self.first_time.setdefault(cav, now_s)
        alpha = 1 - math.exp(-1 / SelectableContract().filter_seconds)
        self.filtered[cav] = (1 - alpha) * self.filtered.get(
            cav, frame.speed_mps
        ) + alpha * frame.speed_mps

    def drop(self, cav):
        super().drop(cav)
        self.filtered.pop(cav, None)
        self.first_time.pop(cav, None)

    def vector(self, cav, *, now_s, actuator):
        base = super().vector(cav, now_s=now_s)
        return np.r_[
            base,
            np.asarray(
                [
                    self.filtered[cav] / 30,
                    float(now_s - self.first_time[cav] >= 5),
                    float(cav in actuator.owned),
                    max(0, actuator.blocked_until.get(cav, 0) - now_s) / 5,
                ],
                dtype=np.float32,
            ),
        ].astype(np.float32)


def expert_from_observation(observation):
    """Local gap/relative-speed feedback with a weak ego-EMA damping term.

    The leader-speed and gap terms remain active during recovery. Unlike v2.0,
    increasing gap cannot saturate an ego-referenced speed offset at +2 m/s.
    These fixed gains have an idealized linear damping derivation; bounded
    commands, sampling, native safety and switching require actual SUMO tests.
    """
    x = np.asarray(observation)
    f = x[..., 68:85]
    speed, front_speed = f[..., 0] * 30, (f[..., 0] - f[..., 8]) * 30
    gap, filtered = f[..., 7] * 100, x[..., 85] * 30
    desired_gap = 2.5 + 1.5 * speed
    acceleration = np.clip(
        0.8 * (front_speed - speed) + 0.25 * (gap - desired_gap) + 0.05 * (filtered - speed),
        -1,
        0.5,
    )
    # Missing leader is unknown. The unchanged hybrid prior chooses native;
    # zero expert output is only a storage target if a trained mode overrides it.
    acceleration = np.where(f[..., 9] > 0.5, acceleration, 0.0)
    excited = (np.abs(front_speed - speed) > 0.5) | (np.abs(filtered - speed) > 0.5)
    preferred = (f[..., 9] > 0.5) & (x[..., 86] > 0.5) & excited
    return acceleration, preferred


class SelectableActuator(SumoDampingActuator):
    def __init__(self, api):
        super().__init__(api, DampingContract())
        self.blocked_until = {}

    def apply(self, frames, actions, *, now_s, live_cav_ids, observations):
        if set(actions) != set(frames) or set(frames) - live_cav_ids:
            raise ValueError("action/frame population mismatch")
        decisions = {}
        cfg = SelectableContract()
        for cav, frame in frames.items():
            mode, delta = map(float, actions[cav])
            if (
                mode not in (0, 1)
                or not math.isfinite(delta)
                or not cfg.residual_min - 1e-6 <= delta <= cfg.residual_max + 1e-6
            ):
                raise ValueError("invalid native/active action")
            if frame.vehicle_id != cav or now_s < frame.timestamp_s:
                raise ValueError("invalid frame identity/time")
            reason, target = "selected_native", None
            expert = float(expert_from_observation(observations[cav])[0])
            if mode:
                reason = "active_target"
                if now_s < self.blocked_until.get(cav, 0):
                    reason = "native_cooldown"
                elif (
                    now_s - frame.timestamp_s > 1
                    or min(frame.front_coverage_m, frame.rear_coverage_m) <= 0
                ):
                    reason = "unavailable_observation"
                else:
                    own = self.sensor.own_parameters(cav)
                    v = frame.speed_mps
                    lower = max(-1, frame.acceleration_mps2 - 0.5, -v)
                    upper = min(0.5, frame.acceleration_mps2 + 0.5, frame.allowed_speed_mps - v)
                    if frame.front is not None:
                        upper = min(
                            upper,
                            frame.front.relative_speed_mps
                            + max(0, frame.front.gap_m - own.min_gap_m) / 3,
                        )
                    if frame.rear is not None:
                        lower = max(lower, frame.rear.relative_speed_mps - frame.rear.gap_m / 4)
                    if lower > upper:
                        reason = "infeasible_command_interval"
                    else:
                        target = float(np.clip(expert + delta, lower, upper))
                        if self.api.vehicle.getSpeedMode(cav) != 31:
                            raise RuntimeError("native safety speedMode=31 required")
            decisions[cav] = dict(
                vehicle_id=cav,
                requested_mode=int(mode),
                requested_delta=delta,
                expert_acceleration=expert,
                target_acceleration=target,
                target_speed=None if target is None else max(0, frame.speed_mps + target),
                projected_delta=0 if target is None else target - expert,
                reasons=(reason,),
            )
        self.owned.intersection_update(live_cav_ids)
        self.blocked_until = {k: v for k, v in self.blocked_until.items() if k in live_cav_ids}
        self.previous_command_delta = {
            k: v for k, v in self.previous_command_delta.items() if k in live_cav_ids
        }
        rows = []
        for cav in sorted(self.owned - set(frames)):
            self.api.vehicle.setSpeed(cav, -1)
            self.owned.remove(cav)
            self.blocked_until[cav] = now_s + cfg.cooldown_seconds
            self.previous_command_delta[cav] = 0
            rows.append(dict(vehicle_id=cav, operation="release_missing_observation"))
        for cav, row in decisions.items():
            row["operation"] = "native_no_command"
            if row["target_speed"] is not None:
                self.api.vehicle.slowDown(cav, row["target_speed"], cfg.control_dt_s)
                self.owned.add(cav)
                row["operation"] = "slow_down"
            elif cav in self.owned:
                self.api.vehicle.setSpeed(cav, -1)
                self.owned.remove(cav)
                self.blocked_until[cav] = now_s + cfg.cooldown_seconds
                row["operation"] = "release_to_native"
            self.previous_command_delta[cav] = row["projected_delta"]
            self.pending[cav] = {
                **row,
                "speed_before": frames[cav].speed_mps,
                "timestamp_before": now_s,
            }
            rows.append(row)
        return rows
