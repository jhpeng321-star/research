from counterfactual.oracle import (
    analyze_oracle,
    choose_least_intensive_beneficial_action,
    choose_oracle_action,
    choose_traffic_oracle_action,
)


def _arm(*, congested, queue, delay, hard_brakes=0, low_ttc=0, duty=0):
    return {
        "outcomes": {
            "sustained_congestion": congested,
            "queue_area_vehicle_s": queue,
            "incremental_delay_s": delay,
            "merge_discharge_count": 10,
            "hard_brake_vehicle_steps": hard_brakes,
            "collision_unique_vehicle_count": 0,
            "starting_teleport_unique_vehicle_count": 0,
            "low_ttc_exposure_steps": low_ttc,
            "duty_steps": duty,
            "duty_fraction": duty / 10,
        }
    }


def test_oracle_rejects_unsafe_strong_arm_and_selects_soft_rescue():
    row = {
        "eligible": True,
        "seed": 101,
        "decision_step": 50,
        "arms": {
            "0": _arm(congested=True, queue=100, delay=50),
            "1": _arm(congested=False, queue=50, delay=40, duty=5),
            "2": _arm(congested=False, queue=20, delay=20, hard_brakes=1, duty=5),
        },
    }

    assert choose_oracle_action(row) == 1
    assert choose_traffic_oracle_action(row) == 2
    assert choose_least_intensive_beneficial_action(row) == 1
    report = analyze_oracle([row])
    assert report["oracle_action_counts"] == {"off": 0, "soft": 1, "strong": 0}
    assert report["case_classification_counts"] == {"rescue": 1}
    assert report["traffic_oracle_action_counts"] == {
        "off": 0,
        "soft": 0,
        "strong": 1,
    }
    assert report["oracle_headroom_vs_off"]["sustained_congestion_cases_avoided"] == 1


def test_oracle_prefers_off_when_treatments_are_worse():
    row = {
        "eligible": True,
        "seed": 102,
        "decision_step": 50,
        "arms": {
            "0": _arm(congested=False, queue=10, delay=5),
            "1": _arm(congested=False, queue=20, delay=7, duty=5),
            "2": _arm(congested=True, queue=30, delay=9, duty=5),
        },
    }

    assert choose_oracle_action(row) == 0
    assert choose_least_intensive_beneficial_action(row) == 0
    assert analyze_oracle([row])["case_classification_counts"] == {"harmful_treatments_only": 1}
