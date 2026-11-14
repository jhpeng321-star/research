import numpy as np

from counterfactual.idm_two_stage_gate import (
    PreparedGateB,
    select_thresholds,
)


def test_threshold_selection_enforces_safety_and_benefit_precision() -> None:
    prepared = PreparedGateB(
        rows=tuple({"case_id": str(i)} for i in range(20)),
        matrix=np.zeros((20, 2)),
        groups=np.asarray([f"seed{i}" for i in range(20)]),
        safety=np.asarray([1] * 19 + [0]),
        joint_benefit=np.asarray([1] * 12 + [0] * 8),
    )
    safety_probability = np.asarray([0.95] * 19 + [0.1])
    benefit_probability = np.asarray([0.9] * 12 + [0.1] * 8)
    selected, sweep = select_thresholds(
        prepared, safety_probability, benefit_probability
    )
    assert selected is not None
    assert selected["selected_safety_rate"] >= 0.90
    assert selected["joint_benefit_precision"] >= 0.50
    assert selected["command_seed_count"] >= 8
    assert any(row["passes_development_policy_gate"] for row in sweep)


def test_threshold_selection_refuses_unsafe_policy() -> None:
    prepared = PreparedGateB(
        rows=tuple({"case_id": str(i)} for i in range(20)),
        matrix=np.zeros((20, 2)),
        groups=np.asarray([f"seed{i}" for i in range(20)]),
        safety=np.asarray([1] * 5 + [0] * 15),
        joint_benefit=np.asarray([1] * 10 + [0] * 10),
    )
    selected, _ = select_thresholds(
        prepared,
        np.asarray([0.9] * 20),
        np.asarray([0.9] * 20),
    )
    assert selected is None
