"""SUMO sensor emulator and command adapter for sensor-local damping v1.

The lane queries below emulate onboard range-limited perception. Only the
selected visible neighbor measurements cross the LocalFrame boundary. No HDV
car-following parameters, intention, global health, or density are read.
"""

from dataclasses import asdict
from typing import Mapping

from config.constants import IDM_SAFE_SPEED_MODE
from env.native_damping import (
    DampingContract,
    LocalFrame,
    NeighborMeasurement,
    OwnIDMParameters,
    project_residual,
)


class SumoLocalSensor:
    def __init__(
        self,
        api,
        contract: DampingContract = DampingContract(),
        lane_end_distance_to_merge: Mapping[str, float] | None = None,
    ):
        self.api = api
        self.contract = contract
        self.lane_end_distance_to_merge = dict(lane_end_distance_to_merge or {})

    def capture(
        self, cav_id: str, *, now_s: float, previous_command_delta: float = 0
    ) -> LocalFrame:
        vehicle, lane = self.api.vehicle, self.api.lane
        lane_id = vehicle.getLaneID(cav_id)
        position = float(vehicle.getLanePosition(cav_id))
        length = float(vehicle.getLength(cav_id))
        lane_length = float(lane.getLength(lane_id))
        speed = float(vehicle.getSpeed(cav_id))
        limit = self.contract.sensor_range_m
        front_coverage = min(limit, max(0.0, lane_length - position))
        rear_coverage = min(limit, max(0.0, position - length))
        front_candidates, rear_candidates = [], []
        for other in lane.getLastStepVehicleIDs(lane_id):
            if other == cav_id:
                continue
            other_position = float(vehicle.getLanePosition(other))
            if other_position >= position:
                gap = other_position - float(vehicle.getLength(other)) - position
                if gap <= front_coverage:
                    front_candidates.append((gap, str(other)))
            else:
                gap = position - length - other_position
                if gap <= rear_coverage:
                    rear_candidates.append((gap, str(other)))

        def measurement(candidates):
            if not candidates:
                return None
            gap, other = min(candidates)
            # Overlaps are errors, not silently recoded as an empty road.
            return NeighborMeasurement(gap, float(vehicle.getSpeed(other)) - speed)

        distance = self.lane_end_distance_to_merge.get(lane_id)
        if distance is not None:
            distance += lane_length - position
        return LocalFrame(
            cav_id,
            now_s,
            speed,
            float(vehicle.getAcceleration(cav_id)),
            float(vehicle.getAllowedSpeed(cav_id)),
            int(vehicle.getLaneIndex(cav_id)),
            distance,
            measurement(front_candidates),
            measurement(rear_candidates),
            front_coverage,
            rear_coverage,
            previous_command_delta,
        )

    def own_parameters(self, cav_id: str) -> OwnIDMParameters:
        vehicle = self.api.vehicle
        return OwnIDMParameters(
            float(vehicle.getAccel(cav_id)),
            float(vehicle.getDecel(cav_id)),
            float(vehicle.getTau(cav_id)),
            float(vehicle.getMinGap(cav_id)),
            float(vehicle.getEmergencyDecel(cav_id)),
        )


class SumoDampingActuator:
    """Exclusive damping command owner; never silently composes with a wall.

    `live_cav_ids` is controller lifecycle bookkeeping, not an actor feature.
    A later overlay must explicitly transfer ownership; it cannot stack this
    adapter with a second independent speed-command writer.
    """

    def __init__(self, api, contract: DampingContract = DampingContract()):
        self.api = api
        self.contract = contract
        self.sensor = SumoLocalSensor(api, contract)
        self.owned: set[str] = set()
        self.previous_command_delta: dict[str, float] = {}
        self.pending: dict[str, dict] = {}

    def apply(
        self,
        frames: Mapping[str, LocalFrame],
        actions: Mapping[str, float],
        *,
        now_s: float,
        live_cav_ids: set[str],
    ) -> list[dict]:
        if set(actions) - set(frames) or set(frames) - live_cav_ids:
            raise ValueError("action/frame outside the live CAV observation population")
        if any(key != frame.vehicle_id for key, frame in frames.items()):
            raise ValueError("frame identity mismatch")
        # Compute/validate the whole batch before the first command mutation.
        decisions = {
            cav: project_residual(
                frame,
                self.sensor.own_parameters(cav),
                float(actions.get(cav, 0.0)),
                now_s=now_s,
                contract=self.contract,
            )
            for cav, frame in sorted(frames.items())
        }
        for cav, decision in decisions.items():
            if decision.target_speed is not None:
                if self.api.vehicle.getSpeedMode(cav) != IDM_SAFE_SPEED_MODE:
                    raise RuntimeError("native safety speedMode=31 is required; no bypass allowed")
        self.owned.intersection_update(live_cav_ids)
        self.previous_command_delta = {
            cav: value for cav, value in self.previous_command_delta.items() if cav in live_cav_ids
        }
        audit = []
        for cav in sorted(self.owned - set(frames)):
            self.api.vehicle.setSpeed(cav, -1.0)
            self.owned.remove(cav)
            self.previous_command_delta[cav] = 0.0
            audit.append({"vehicle_id": cav, "operation": "release_missing_observation"})
        for cav, decision in decisions.items():
            operation = "native_no_command"
            if decision.target_speed is not None:
                # A duration-matched speed ramp avoids treating a 1 s target
                # as an instantaneous speed change at the first 0.5 s tick.
                self.api.vehicle.slowDown(cav, decision.target_speed, self.contract.control_dt_s)
                self.owned.add(cav)
                operation = "slow_down"
            elif cav in self.owned:
                self.api.vehicle.setSpeed(cav, -1.0)
                self.owned.remove(cav)
                operation = "release_to_native"
            self.previous_command_delta[cav] = decision.projected_delta
            row = {"vehicle_id": cav, "operation": operation, **asdict(decision)}
            audit.append(row)
            self.pending[cav] = {
                **row,
                "speed_before": frames[cav].speed_mps,
                "timestamp_before": now_s,
            }
        return audit

    def observe_execution(self, *, now_s: float, live_cav_ids: set[str]) -> list[dict]:
        """Measure command tracking; this is NOT a paired causal residual."""
        rows = []
        for cav, command in sorted(self.pending.items()):
            if cav not in live_cav_ids:
                rows.append({**command, "execution_observed": False, "reason": "vehicle_departed"})
                continue
            elapsed = now_s - command["timestamp_before"]
            if elapsed <= 0:
                raise ValueError("execution must be observed after the command")
            speed = float(self.api.vehicle.getSpeed(cav))
            observed = (speed - command["speed_before"]) / elapsed
            projected = command["target_acceleration"]
            rows.append(
                {
                    **command,
                    "execution_observed": True,
                    "observed_average_acceleration": observed,
                    "tracking_error": None if projected is None else observed - projected,
                }
            )
        self.pending.clear()
        return rows
