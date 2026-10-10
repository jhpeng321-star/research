from counterfactual.twin_replay import ReplayTrace, StepSnapshot
from config.constants import MERGE_EDGE, PRE_MERGE_EDGE
from experiments.counterfactual.generate_counterfactual_escalation_headroom import (
    _first_stage_inflow,
)


def _snapshot(step, roads):
    vehicles = tuple(
        (vehicle_id, road, f"{road}_0", 0.0, 10.0, 0.0, 0.0, 0, "passenger", 1.0)
        for vehicle_id, road in sorted(roads.items())
    )
    return StepSnapshot(
        relative_step=step,
        sim_time=float(step),
        action=0,
        supervisor_observation=(0.0,),
        vehicles=vehicles,
        active_cavs=(),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=str(step),
        vehicle_count=len(vehicles),
        mean_speed=10.0,
        active_time_loss_sum=0.0,
    )


def _trace(snapshots):
    return ReplayTrace(
        seed=1,
        decision_step=0,
        horizon_steps=len(snapshots) - 1,
        prefix_action=0,
        post_action=0,
        treatment_steps=2,
        reset_info={},
        snapshots=tuple(snapshots),
        terminated_early=False,
    )


def test_first_stage_inflow_counts_during_and_equal_post_windows():
    control = _trace(
        [
            _snapshot(0, {"a": PRE_MERGE_EDGE, "b": PRE_MERGE_EDGE}),
            _snapshot(1, {"a": MERGE_EDGE, "b": PRE_MERGE_EDGE}),
            _snapshot(2, {"a": MERGE_EDGE, "b": PRE_MERGE_EDGE}),
            _snapshot(3, {"a": MERGE_EDGE, "b": MERGE_EDGE}),
            _snapshot(4, {"a": MERGE_EDGE, "b": MERGE_EDGE}),
        ]
    )
    treatment = _trace(
        [
            _snapshot(0, {"a": PRE_MERGE_EDGE, "b": PRE_MERGE_EDGE}),
            _snapshot(1, {"a": PRE_MERGE_EDGE, "b": PRE_MERGE_EDGE}),
            _snapshot(2, {"a": PRE_MERGE_EDGE, "b": PRE_MERGE_EDGE}),
            _snapshot(3, {"a": MERGE_EDGE, "b": PRE_MERGE_EDGE}),
            _snapshot(4, {"a": MERGE_EDGE, "b": PRE_MERGE_EDGE}),
        ]
    )

    result = _first_stage_inflow(control, treatment, decision_step=0, treatment_steps=2)

    assert result["during_treatment"]["merge_entry_reduction"] == 1
    assert result["equal_post_window"]["merge_entry_reduction"] == 0
    assert result["combined_merge_entry_reduction"] == 1
