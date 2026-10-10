"""Tests for the HDVTypeAssigner class."""

import pytest

from env.vehicles import HDVTypeAssigner


def test_default_proportions():
    assigner = HDVTypeAssigner([0.1, 0.7, 0.2])
    assert assigner.proportions == [0.1, 0.7, 0.2]


def test_proportions_must_sum_to_one():
    with pytest.raises(ValueError, match="proportions must sum to 1.0"):
        HDVTypeAssigner([0.1, 0.7, 0.1])


def test_proportions_length():
    with pytest.raises(ValueError, match="proportions must have 3 entries"):
        HDVTypeAssigner([0.5, 0.5])


def test_assign_all_departed_returns_types():
    class DummyTraciVehicle:
        calls = []

        @staticmethod
        def setType(veh_id, vtype):
            DummyTraciVehicle.calls.append((veh_id, vtype))

    class DummyTraci:
        vehicle = DummyTraciVehicle()

    import env.vehicles as hta_module

    original_traci = getattr(hta_module, "traci", None)
    hta_module.traci = DummyTraci()
    try:
        assigner = HDVTypeAssigner([0.0, 1.0, 0.0], seed=42)
        result = assigner.assign_all_departed(["v1", "v2", "v3"])
        assert result == {"v1": "normal_car", "v2": "normal_car", "v3": "normal_car"}
        assert len(DummyTraciVehicle.calls) == 3
    finally:
        if original_traci is None:
            del hta_module.traci
        else:
            hta_module.traci = original_traci


def test_skip_already_assigned():
    assigner = HDVTypeAssigner([0.0, 1.0, 0.0], seed=1)
    assigner._assigned["v1"] = "normal_car"
    # Monkey-patch traci to avoid requiring a live SUMO connection.
    import env.vehicles as hta_module

    original_traci = getattr(hta_module, "traci", None)

    class DummyTraciVehicle:
        calls = []

        @staticmethod
        def setType(veh_id, vtype):
            DummyTraciVehicle.calls.append((veh_id, vtype))

    class DummyTraci:
        vehicle = DummyTraciVehicle()

    hta_module.traci = DummyTraci()
    try:
        result = assigner.assign_all_departed(["v1", "v2"], skip={"v1"})
        assert result == {"v2": "normal_car"}
        assert len(DummyTraciVehicle.calls) == 1
    finally:
        if original_traci is None:
            del hta_module.traci
        else:
            hta_module.traci = original_traci


def test_count_by_type():
    assigner = HDVTypeAssigner([0.0, 1.0, 0.0], seed=1)
    assigner._assigned = {
        "v1": "aggressive_car",
        "v2": "normal_car",
        "v3": "normal_car",
        "v4": "cautious_car",
    }
    counts = assigner.count_by_type()
    assert counts == {"aggressive_car": 1, "normal_car": 2, "cautious_car": 1}


def test_reset_clears_state():
    assigner = HDVTypeAssigner([0.0, 1.0, 0.0], seed=1)
    assigner._assigned["v1"] = "normal_car"
    assigner.reset(seed=2)
    assert assigner._assigned == {}


if __name__ == "__main__":
    test_default_proportions()
    test_proportions_must_sum_to_one()
    test_proportions_length()
    test_assign_all_departed_returns_types()
    test_skip_already_assigned()
    test_count_by_type()
    test_reset_clears_state()
    print("test_hdv_type_assigner passed")
