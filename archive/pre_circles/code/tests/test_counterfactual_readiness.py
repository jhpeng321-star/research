from counterfactual.readiness import evaluate_readiness


def _calibration():
    candidate = {
        "definition": {
            "queue_speed_threshold_mps": 10.0,
            "congestion_queue_count": 10,
            "congestion_speed_threshold_mps": 10.0,
            "sustain_steps": 30,
        },
        "valid_two_class_split": True,
        "positive_window_count": 8,
        "negative_window_count": 32,
    }
    return {
        "seed_count": 10,
        "freeze_requirements": {
            "minimum_seed_count": 10,
            "minimum_positive_windows": 5,
            "minimum_negative_windows": 5,
        },
        "top_candidates": [candidate],
        "empirical_no_control_state": {
            "window_hard_brake_rate_vehicle_steps_per_s": {"p95": 0.2},
            "window_low_ttc_exposure_fraction": {"p95": 0.05},
        },
    }


def _outcome(*, queue, delay, discharge, hard_brakes, low_ttc):
    return {
        "sustained_congestion": False,
        "queue_area_vehicle_s": queue,
        "incremental_delay_s": delay,
        "merge_discharge_count": discharge,
        "hard_brake_vehicle_steps": hard_brakes,
        "low_ttc_exposure_steps": low_ttc,
        "collision_unique_vehicle_count": 0,
        "starting_teleport_unique_vehicle_count": 0,
        "observed_outcome_steps": 100,
        "step_length_s": 1.0,
    }


def _row():
    definition = {
        "queue_speed_threshold_mps": 10.0,
        "congestion_queue_count": 10,
        "congestion_speed_threshold_mps": 10.0,
        "sustain_steps": 30,
    }
    return {
        "eligible": True,
        "case_id": "seed-1-step-10",
        "seed": 1,
        "decision_step": 10,
        "outcome_definition": definition,
        "arms": {
            "0": {
                "outcomes": _outcome(queue=100, delay=100, discharge=10, hard_brakes=10, low_ttc=3)
            },
            "1": {
                "outcomes": _outcome(queue=80, delay=80, discharge=12, hard_brakes=15, low_ttc=2)
            },
            "2": {
                "outcomes": _outcome(queue=60, delay=90, discharge=11, hard_brakes=30, low_ttc=2)
            },
        },
    }


def test_readiness_applies_no_control_safety_budget():
    report = evaluate_readiness(
        _calibration(),
        [_row()],
        minimum_paired_cases=1,
        minimum_off_labels=0,
        minimum_active_labels=1,
    )

    assert report["cases"][0]["selected_action_name"] == "soft"
    assert report["cases"][0]["actions"]["1"]["safety_admissible"] is True
    assert report["cases"][0]["actions"]["2"]["safety_admissible"] is False
    assert report["ready_to_train_deployable_gate"] is True


def test_readiness_accepts_off_soft_only_dataset():
    row = _row()
    del row["arms"]["2"]
    report = evaluate_readiness(
        _calibration(),
        [row],
        minimum_paired_cases=1,
        minimum_off_labels=0,
        minimum_active_labels=1,
    )

    assert report["cases"][0]["available_actions"] == [0, 1]
    assert report["cases"][0]["selected_action_name"] == "soft"
    assert report["selected_action_counts"]["strong"] == 0
