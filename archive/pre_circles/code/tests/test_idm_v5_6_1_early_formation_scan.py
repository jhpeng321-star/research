from types import SimpleNamespace

from experiments.idm.run_idm_v5_6_1_early_formation_scan import (
    ASSEMBLY_SPAN_M, FINAL_SPAN_M, HOLD_START_S, _project, assess_snapshot,
)


def test_projection_uses_low_authority_and_separate_final_span():
    members = ({"id": "a", "distance_m": 800.0, "speed_mps": 25.0}, {"id": "b", "distance_m": 850.0, "speed_mps": 25.0})
    result = _project(members, 5)
    assert result["maximum_assembly_span_m"] <= ASSEMBLY_SPAN_M
    assert result["initial_span_m"] == FINAL_SPAN_M


def test_scan_rejects_when_final_member_is_not_continuous():
    start = SimpleNamespace(active_cavs=("a", "b"), vehicles=(("a", "1191554636", "1191554636_0", 300.0, 20.0, 0.0, 0, 0, "normal", 1.4), ("b", "1191554636", "1191554636_1", 330.0, 20.0, 0.0, 0, 0, "normal", 1.4)))
    hold = SimpleNamespace(active_cavs=(), vehicles=())
    result = assess_snapshot(start, hold, assembly_start_s=HOLD_START_S - 5)
    assert result["feasible"] is False
