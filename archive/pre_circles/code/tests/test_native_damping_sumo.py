from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from env.native_damping import LocalHistory
from env.native_damping_sumo import SumoDampingActuator, SumoLocalSensor


class FakeVehicle:
    def __init__(self):
        self.positions = {"cav": 200, "front": 250, "occluded": 280, "far": 450, "rear": 150}
        self.speeds = {key: 30 for key in self.positions}
        self.calls = []
        self.mode = 31

    def getLaneID(self, _):
        return "lane0"

    def getLanePosition(self, cav):
        return self.positions[cav]

    def getLength(self, _):
        return 5.0

    def getSpeed(self, cav):
        return self.speeds[cav]

    def getAcceleration(self, _):
        return 0.0

    def getAllowedSpeed(self, _):
        return 30.0

    def getLaneIndex(self, _):
        return 0

    def getSpeedMode(self, _):
        return self.mode

    def _own(self, cav, value):
        assert cav == "cav", "private HDV parameter access"
        return value

    def getAccel(self, cav):
        return self._own(cav, 2.6)

    def getDecel(self, cav):
        return self._own(cav, 4.5)

    def getTau(self, cav):
        return self._own(cav, 1.5)

    def getMinGap(self, cav):
        return self._own(cav, 2.5)

    def getEmergencyDecel(self, cav):
        return self._own(cav, 9)

    def slowDown(self, *args):
        self.calls.append(("slowDown", *args))

    def setSpeed(self, *args):
        self.calls.append(("setSpeed", *args))


def api():
    vehicle = FakeVehicle()
    return SimpleNamespace(
        vehicle=vehicle,
        lane=SimpleNamespace(
            getLength=lambda _: 1000, getLastStepVehicleIDs=lambda _: list(vehicle.positions)
        ),
    )


def test_sensor_excludes_occluded_far_and_private_state():
    connection = api()
    sensor = SumoLocalSensor(connection)
    first = sensor.capture("cav", now_s=0)
    assert first.front.gap_m == 45  # physical bumper gap, not SUMO minGap-subtracted gap
    assert first.rear.gap_m == 45
    connection.vehicle.speeds["occluded"] = 0
    connection.vehicle.speeds["far"] = 100
    second = sensor.capture("cav", now_s=0)
    assert first == second
    h1, h2 = LocalHistory(), LocalHistory()
    h1.add(first, now_s=0)
    h2.add(second, now_s=0)
    np.testing.assert_array_equal(h1.vector("cav", now_s=0), h2.vector("cav", now_s=0))
    assert sensor.own_parameters("cav").tau_s == 1.5


def test_lane_boundary_is_partial_coverage_not_100m_of_empty_road():
    connection = api()
    connection.vehicle.positions = {"cav": 999}
    observed = SumoLocalSensor(connection).capture("cav", now_s=0)
    assert observed.front is None and observed.front_coverage_m == 1


def test_zero_never_mutates_native_vehicle_and_release_occurs_only_once():
    connection = api()
    actuator = SumoDampingActuator(connection)
    observed = replace(SumoLocalSensor(connection).capture("cav", now_s=0), front=None, rear=None)
    for _ in range(2):
        actuator.apply({"cav": observed}, {"cav": 0}, now_s=0, live_cav_ids={"cav"})
    assert connection.vehicle.calls == []
    actuator.apply({"cav": observed}, {"cav": -0.2}, now_s=0, live_cav_ids={"cav"})
    assert connection.vehicle.calls == [("slowDown", "cav", 29.8, 1.0)]
    actuator.apply({"cav": observed}, {"cav": 0}, now_s=0, live_cav_ids={"cav"})
    actuator.apply({"cav": observed}, {"cav": 0}, now_s=0, live_cav_ids={"cav"})
    assert connection.vehicle.calls[-1] == ("setSpeed", "cav", -1.0)
    assert len(connection.vehicle.calls) == 2


def test_invalid_batch_is_rejected_before_any_command_and_safety_never_disabled():
    connection = api()
    actuator = SumoDampingActuator(connection)
    observed = replace(SumoLocalSensor(connection).capture("cav", now_s=0), front=None, rear=None)
    with pytest.raises(ValueError):
        actuator.apply({"cav": observed}, {"cav": float("nan")}, now_s=0, live_cav_ids={"cav"})
    assert connection.vehicle.calls == []
    connection.vehicle.mode = 7
    with pytest.raises(RuntimeError, match="no bypass"):
        actuator.apply({"cav": observed}, {"cav": -0.2}, now_s=0, live_cav_ids={"cav"})
    assert connection.vehicle.calls == []


def test_actual_acceleration_is_measured_and_not_named_causal_residual():
    connection = api()
    actuator = SumoDampingActuator(connection)
    observed = replace(SumoLocalSensor(connection).capture("cav", now_s=0), front=None, rear=None)
    actuator.apply({"cav": observed}, {"cav": -0.2}, now_s=0, live_cav_ids={"cav"})
    connection.vehicle.speeds["cav"] = 29.9
    row = actuator.observe_execution(now_s=1, live_cav_ids={"cav"})[0]
    assert row["observed_average_acceleration"] == pytest.approx(-0.1)
    assert row["tracking_error"] == pytest.approx(0.1)
    assert "executed_residual" not in row


def test_missing_observation_releases_owned_command_and_departed_cav_is_retired():
    connection = api()
    actuator = SumoDampingActuator(connection)
    observed = replace(SumoLocalSensor(connection).capture("cav", now_s=0), front=None, rear=None)
    actuator.apply({"cav": observed}, {"cav": -0.2}, now_s=0, live_cav_ids={"cav"})
    actuator.apply({}, {}, now_s=1, live_cav_ids={"cav"})
    assert connection.vehicle.calls[-1] == ("setSpeed", "cav", -1.0)
    actuator.apply({}, {}, now_s=2, live_cav_ids=set())
    assert not actuator.owned and not actuator.previous_command_delta
