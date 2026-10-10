from counterfactual.win_sensitivity import analyze_split, is_traffic_win


def _outcome(
    *,
    congested=False,
    queue=100.0,
    delay=100.0,
    discharge=10,
    merge_speed=10.0,
    pre_merge_speed=15.0,
):
    return {
        "sustained_congestion": congested,
        "queue_area_vehicle_s": queue,
        "incremental_delay_s": delay,
        "merge_discharge_count": discharge,
        "mean_merge_speed_mps": merge_speed,
        "mean_pre_merge_speed_mps": pre_merge_speed,
        "hard_brake_vehicle_steps": 0,
        "low_ttc_exposure_steps": 0,
        "collision_unique_vehicle_count": 0,
        "starting_teleport_unique_vehicle_count": 0,
    }


def _row(off, soft):
    return {
        "eligible": True,
        "case_id": "seed-1-step-10",
        "seed": 1,
        "decision_step": 10,
        "arms": {"0": {"outcomes": off}, "1": {"outcomes": soft}},
    }


def _readiness(safe=True):
    return {
        "cases": [
            {
                "case_id": "seed-1-step-10",
                "actions": {"1": {"safety_admissible": safe}},
            }
        ]
    }


def test_delay_margin_accepts_only_bounded_delay_tradeoff():
    row = _row(
        _outcome(),
        _outcome(queue=90.0, delay=103.0),
    )

    assert is_traffic_win(row, "current_pareto") is False
    assert is_traffic_win(row, "delay_margin_2pct") is False
    assert is_traffic_win(row, "delay_margin_5pct") is True
    assert is_traffic_win(row, "delay_excluded") is True


def test_excluding_costs_does_not_make_no_effect_a_win():
    row = _row(
        _outcome(),
        _outcome(queue=120.0, delay=120.0),
    )

    assert is_traffic_win(row, "stability_throughput") is False
    assert is_traffic_win(row, "bottleneck_service_exact") is False


def test_stability_throughput_rejects_induced_pre_merge_congestion():
    row = _row(
        _outcome(congested=False),
        _outcome(congested=True, discharge=11),
    )

    assert is_traffic_win(row, "stability_throughput") is False


def test_bottleneck_service_requires_speed_and_discharge_noninferiority():
    speed_only = _row(_outcome(), _outcome(merge_speed=10.5))
    speed_with_two_vehicle_loss = _row(_outcome(), _outcome(merge_speed=10.5, discharge=8))
    speed_with_one_vehicle_loss = _row(_outcome(), _outcome(merge_speed=10.5, discharge=9))

    assert is_traffic_win(speed_only, "bottleneck_service_exact") is True
    assert is_traffic_win(speed_with_two_vehicle_loss, "bottleneck_service_exact") is False
    assert is_traffic_win(speed_with_two_vehicle_loss, "bottleneck_service_1veh") is False
    assert is_traffic_win(speed_with_one_vehicle_loss, "bottleneck_service_1veh") is True


def test_safety_remains_a_hard_gate():
    row = _row(_outcome(), _outcome(merge_speed=10.5))
    report = analyze_split([row], _readiness(safe=False))
    result = report["rules"]["bottleneck_service_exact"]

    assert result["traffic_win_count"] == 1
    assert result["selected_safe_soft_count"] == 0
    assert result["safety_rejected_traffic_win_count"] == 1
