"""The sole writer of CAV longitudinal commands, including temporary wall ownership."""
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping

from .damping import CirclesController, DampingDecision, direct_target, frame_problem, record_dict


@dataclass(frozen=True)
class CommandRecord:
    vehicle_id: str
    timestamp_s: float
    owner: str
    network_output_mps2: float | None
    wall_request_mps2: float | None
    issued_acceleration_mps2: float | None
    issued_target_speed_mps: float | None
    fallback_reason: str | None
    released_to_native: bool
    actor_input_si: tuple[float, float, float] | None = None

    def as_dict(self) -> dict:
        return record_dict(self)


@dataclass(frozen=True)
class ResponseRecord:
    vehicle_id: str
    timestamp_s: float
    command_timestamp_s: float
    owner: str
    measured_speed_mps: float
    measured_acceleration_mps2: float

    def as_dict(self) -> dict:
        return record_dict(self)


class CommandOwnership:
    """One call per control tick. Removing a wall member resumes the same damping.

    Wall requests are absolute acceleration, keyed by a validated live CAV mask.
    No planner is permitted to write independently to TraCI.
    """
    def __init__(self, api, cav_ids, controller: CirclesController, *, cav_speed_mode: int = 31):
        if type(cav_speed_mode) is not int or cav_speed_mode not in (31, 27):
            raise ValueError("only legacy31 or explicit emergency-braking candidate27 is supported")
        self.api, self.controller = api, controller
        self.cav_speed_mode = cav_speed_mode
        self._mode_configured = set()
        self.last_mode_changes = []
        self.cav_ids = frozenset(cav_ids)
        self._owned: dict[str, str] = {}
        self._last: dict[str, CommandRecord] = {}
        self._time: float | None = None
        # Upper bound of wall/direct requests; None = the adopted config bound (1.5).  Set only by
        # CavSumoEnv for C4 DIRECT-output ticks (ACCELCAP_NOTES.md).
        self.wall_accel_max_mps2: float | None = None

    def step(self, frames: Mapping, *, now_s: float, live_ids,
             wall_commands: Mapping[str, float] | None = None, native_ids=(),
             record_callback: Callable[[CommandRecord], None] | None = None) -> list[CommandRecord]:
        """Notify optional stateful adapters after each completed vehicle decision.

        A callback receives an already persisted command record, after any writer
        call succeeds. Native rows can represent a no-write fallback. Exceptions
        propagate; callbacks must not treat a partial batch as rolled back.
        """
        if record_callback is not None and not callable(record_callback):
            raise TypeError("record_callback must be callable or None")
        wall = dict(wall_commands or {})
        native = frozenset(native_ids)
        live = frozenset(live_ids)
        live_cavs = live & self.cav_ids
        if not isfinite(now_s) or (self._time is not None and now_s < self._time + self.controller.config.control_dt_s - 1e-6):
            raise ValueError("command clock must advance by at least one control interval")
        if abs(now_s / self.controller.config.control_dt_s - round(now_s / self.controller.config.control_dt_s)) > 1e-6:
            raise ValueError("command clock must lie on the configured control grid")
        if not set(wall) <= live_cavs:
            raise ValueError("wall mask contains an absent vehicle or HDV")
        if not native <= live_cavs or native & set(wall):
            raise ValueError("native mask must contain only live CAVs and be disjoint from wall ownership")
        if not set(frames) <= live_cavs or any(f is not None and f.vehicle_id != v for v, f in frames.items()):
            raise ValueError("sensor mask contains a foreign or mismatched vehicle")
        try:
            wall = {v: float(a) for v, a in wall.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("wall requests must be numeric absolute acceleration") from exc
        if any(not isfinite(a) for a in wall.values()):
            raise ValueError("wall requests must be finite absolute acceleration")
        # Validate the full mask before any command, rather than partially acting.
        modes = {v: self.api.vehicle.getSpeedMode(v) for v in live_cavs}
        if self.cav_speed_mode == 31:
            if any(mode != 31 for mode in modes.values()):
                raise ValueError("native SUMO speedMode31 is required")
        elif any(mode not in (31, 27) or v in self._mode_configured and mode != 27
                 for v, mode in modes.items()):
            raise ValueError("CAV command mode changed outside its declared interface")
        # Validate all masks and modes before mutation. Only this owner configures
        # CAVs, once on first observation; native selection does not reset the mode.
        self.last_mode_changes = []
        if self.cav_speed_mode == 27:
            for v in sorted(live_cavs):
                if modes[v] == 31:
                    self.api.vehicle.setSpeedMode(v, 27)
                    if self.api.vehicle.getSpeedMode(v) != 27:
                        raise RuntimeError("CAV emergency-braking command mode was not applied")
                    self.last_mode_changes.append(dict(vehicle_id=v, timestamp_s=now_s,
                                                       prior_mode=31, speed_mode=27))
            self._mode_configured = (self._mode_configured & live_cavs) | live_cavs
        self._owned = {v: owner for v, owner in self._owned.items() if v in live_cavs}
        self._last = {v: row for v, row in self._last.items() if v in live_cavs}
        records = []
        for v in sorted(live_cavs):
            frame = frames.get(v)
            if v in native:
                decision = DampingDecision(fallback_reason="selected_native")
                owner = "native"
            elif v in wall:
                reason = frame_problem(frame, now_s, self.controller.config)
                a, target = (None, None) if reason else direct_target(
                    frame, wall[v], self.controller.config, self.wall_accel_max_mps2)
                decision = DampingDecision(command_acceleration_mps2=a, target_speed_mps=target,
                                           fallback_reason=reason or ("infeasible_speed_bounds" if a is None else None))
                owner = "wall"
            else:
                decision = self.controller.decide(frame, now_s=now_s)
                owner = "damping"
            released = False
            if decision.fallback_reason is not None:
                owner = "native"
                if v in self._owned:
                    self.api.vehicle.setSpeed(v, -1)
                    del self._owned[v]
                    released = True
            else:
                self.api.vehicle.slowDown(v, decision.target_speed_mps, self.controller.config.control_dt_s)
                self._owned[v] = owner
            row = CommandRecord(v, now_s, owner, decision.network_output_mps2, wall.get(v),
                decision.command_acceleration_mps2, decision.target_speed_mps, decision.fallback_reason,
                released, decision.actor_input_si)
            self._last[v] = row
            records.append(row)
            if record_callback is not None:
                record_callback(row)
        self._time = now_s
        return records

    def record_response(self, vehicle_id: str, *, timestamp_s: float,
                        speed_mps: float, acceleration_mps2: float) -> ResponseRecord:
        prior = self._last.get(vehicle_id)
        if prior is None or not prior.timestamp_s < timestamp_s <= prior.timestamp_s + self.controller.config.control_dt_s + 1e-6:
            raise ValueError("response must follow a known command within its interval")
        if not all(isfinite(x) for x in (timestamp_s, speed_mps, acceleration_mps2)) or speed_mps < 0:
            raise ValueError("invalid measured response")
        return ResponseRecord(vehicle_id, timestamp_s, prior.timestamp_s, prior.owner, speed_mps, acceleration_mps2)

    def release_all(self, *, live_ids) -> None:
        """Explicit lifecycle handback; never command an HDV or departed vehicle."""
        for v in sorted(set(self._owned) & set(live_ids) & self.cav_ids):
            self.api.vehicle.setSpeed(v, -1)
        self._owned.clear()
        self._last.clear()
