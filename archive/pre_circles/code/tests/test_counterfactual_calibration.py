import pytest

from config.constants import PRE_MERGE_EDGE
from counterfactual.calibration import calibrate_outcome_definition
from counterfactual.pilot import generate_paired_case
from counterfactual.twin_replay import ReplayTrace, StepSnapshot


def _snapshot(step, *, congested, time_loss, digest, action=0):
    speed = 10.0 if congested else 25.0
    rows = tuple(
        (
            f"veh-{index}",
            PRE_MERGE_EDGE,
            f"{PRE_MERGE_EDGE}_0",
            10.0 + 10.0 * index,
            speed,
            0.0,
            time_loss,
            0,
            "car",
        )
        for index in range(2)
    )
    return StepSnapshot(
        relative_step=step,
        sim_time=float(step),
        action=action,
        supervisor_observation=(
            0.4,
            -0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.6,
            0.7,
            0.8,
        ),
        vehicles=rows,
        active_cavs=("veh-0",),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=digest,
        vehicle_count=2,
        mean_speed=speed,
        active_time_loss_sum=2 * time_loss,
    )


def _trace(seed, *, congested, post_action=0):
    snapshots = tuple(
        _snapshot(
            step,
            congested=congested,
            time_loss=step * (5.0 if congested else 0.5),
            digest=f"{seed}-{step}",
            action=post_action if step > 0 else 0,
        )
        for step in range(4)
    )
    return ReplayTrace(
        seed=seed,
        decision_step=0,
        horizon_steps=3,
        prefix_action=0,
        post_action=post_action,
        treatment_steps=None,
        reset_info={},
        snapshots=snapshots,
        terminated_early=False,
    )


def test_calibration_uses_no_control_windows_and_finds_two_class_definition():
    traces = [
        _trace(1, congested=True),
        _trace(2, congested=True),
        _trace(3, congested=False),
        _trace(4, congested=False),
    ]

    report = calibrate_outcome_definition(
        traces,
        decision_steps=[0],
        outcome_steps=3,
        queue_speed_thresholds=[15.0],
        congestion_queue_counts=[2, 3],
        congestion_speed_thresholds=[15.0],
        sustain_steps=[2],
        minimum_seed_count_for_freeze=4,
    )

    assert report["calibration_scope"] == "no_control_only"
    assert report["recommended_definition"]["congestion_queue_count"] == 2
    assert report["recommended_diagnostics"]["positive_window_count"] == 2
    assert report["recommended_diagnostics"]["negative_window_count"] == 2
    assert "health" in report["explicitly_excluded_from_selection"]


def test_calibration_prefers_candidate_meeting_freeze_counts(monkeypatch):
    traces = [_trace(seed, congested=seed <= 5) for seed in range(1, 11)]
    report = calibrate_outcome_definition(
        traces,
        decision_steps=[0],
        outcome_steps=3,
        queue_speed_thresholds=[15.0],
        congestion_queue_counts=[2],
        congestion_speed_thresholds=[15.0],
        sustain_steps=[2],
        minimum_seed_count_for_freeze=10,
    )

    assert report["freeze_ready"] is True
    assert report["recommendation_policy"] == ("highest_score_meeting_freeze_requirements")


def test_calibration_rejects_treated_trace():
    with pytest.raises(ValueError, match="only off-control"):
        calibrate_outcome_definition(
            [_trace(1, congested=True, post_action=1)],
            decision_steps=[0],
            outcome_steps=3,
        )


def test_fixed_decision_case_skips_warning_search(monkeypatch):
    calls = []

    def fake_run_replay(**kwargs):
        calls.append(kwargs)
        action = kwargs["post_action"]
        decision = kwargs["decision_step"]
        horizon = kwargs["horizon_steps"]
        snapshots = []
        for step in range(decision + horizon + 1):
            applied = 0 if step <= decision else action
            digest = f"prefix-{step}" if step <= decision else f"action-{action}-{step}"
            snapshots.append(
                _snapshot(
                    step,
                    congested=False,
                    time_loss=float(step),
                    digest=digest,
                    action=applied,
                )
            )
        return ReplayTrace(
            seed=kwargs["seed"],
            decision_step=decision,
            horizon_steps=horizon,
            prefix_action=0,
            post_action=action,
            treatment_steps=kwargs.get("treatment_steps"),
            reset_info={},
            snapshots=tuple(snapshots),
            terminated_early=False,
        )

    monkeypatch.setattr("counterfactual.pilot.run_replay", fake_run_replay)
    row = generate_paired_case(
        project_root=".",
        sumo_binary="sumo",
        sumo_config="scenario.sumocfg",
        seed=101,
        search_steps=10,
        decision_step_override=2,
        outcome_steps=2,
        treatment_steps=1,
    )

    assert len(calls) == 3
    assert row["case_id"] == "seed-101-step-2"
    assert row["eligibility"]["selection_mode"] == "predeclared_fixed_step"
    assert row["search_trace_digest"] is None
