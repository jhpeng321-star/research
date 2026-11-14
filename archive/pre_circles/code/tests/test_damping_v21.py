from dataclasses import replace
import inspect

import numpy as np

from env import damping_v2, damping_v21
from env.native_damping_sumo import SumoLocalSensor
from tests.test_native_damping_sumo import api


def test_actuator_physical_bounds_and_release_code_are_unchanged():
    assert inspect.getsource(damping_v21.SelectableActuator.apply) == inspect.getsource(
        damping_v2.SelectableActuator.apply
    )
    a, b = damping_v2.SelectableContract(), damping_v21.SelectableContract()
    assert vars(a) == vars(b)
    assert a.manifest()["sha256"] != b.manifest()["sha256"]


def test_large_gap_recovery_can_use_existing_acceleration_bound_without_extra_inputs():
    x = np.zeros(89)
    x[68] = 7.75 / 30
    x[75] = 0.91
    x[76] = (7.75 - 14.5) / 30
    x[77:79] = 1
    x[84] = 1
    x[85] = 6.8 / 30
    x[86] = 1
    assert damping_v2.expert_from_observation(x)[0] < 0.22
    assert damping_v21.expert_from_observation(x)[0] == 0.5
    x[77] = 0
    acceleration, mode = damping_v21.expert_from_observation(x)
    assert acceleration == 0 and not mode


def test_native_release_and_zero_active_remain_distinct_in_recovery_version():
    connection = api()
    actuator = damping_v21.SelectableActuator(connection)
    frame = replace(SumoLocalSensor(connection).capture("cav", now_s=0), front=None, rear=None)
    x = np.zeros(89)
    for now, mode in ((0, 0), (1, 1), (2, 0), (3, 1), (7, 1), (8, 0)):
        current = replace(frame, timestamp_s=now)
        actuator.apply(
            {"cav": current},
            {"cav": (mode, 0)},
            now_s=now,
            live_cav_ids={"cav"},
            observations={"cav": x},
        )
    assert [row[0] for row in connection.vehicle.calls] == [
        "slowDown",
        "setSpeed",
        "slowDown",
        "setSpeed",
    ]
    assert not actuator.owned


def test_ideal_continuous_feedback_has_stable_poles_and_nonamplifying_magnitude():
    # Direct polynomial check of the documented derivation, not a SUMO claim.
    denominator = np.array([5.0, 7.125, 2.425, 0.25])
    numerator = np.array([4.0, 2.05, 0.25])
    assert (np.roots(denominator).real < 0).all()
    for w in np.logspace(-4, 3, 200):
        d = np.polyval(denominator, 1j * w)
        n = np.polyval(numerator, 1j * w)
        assert abs(n / d) <= 1 + 1e-10
        expected = 0.115625 * w * w + 10.515625 * w**4 + 25 * w**6
        assert np.isclose(abs(d) ** 2 - abs(n) ** 2, expected, rtol=1e-7, atol=1e-9)
