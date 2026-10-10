import json

import pytest

from tools.analyze_idm_compact_wall_feasibility import (
    AvailabilityRule,
    FROZEN_DECISION_STEPS,
    FROZEN_PENETRATIONS,
    FROZEN_SEEDS,
    _row_admission,
    analyze,
)


def _formation(feasible=True):
    return {
        "feasible": feasible,
        "reason": "feasible" if feasible else "uncovered_occupied_lane",
        "controlled_cav_ids": ["c0", "c1"] if feasible else [],
        "cav_lane_ids": ["lane0", "lane1"],
        "occupied_lane_ids": ["lane0", "lane1"],
        "required_lane_ids": ["lane0", "lane1"],
        "longitudinal_span_m": 12.0 if feasible else 0.0,
        "predicted_max_span_m": 18.0 if feasible else None,
        "predicted_terminal_span_m": 10.0 if feasible else None,
    }


def _result(step, admitted):
    return {
        "decision_step": step,
        "request_step": step + 1,
        "sim_time_s": float(step),
        "active_cav_count": 20,
        "damping_command_count": 2,
        "formation": _formation(admitted),
        "lane_coverage": {
            "selected_lane_ids": ["lane0", "lane1"] if admitted else [],
            "full_occupied_lane_coverage": admitted,
            "coverage_fraction": 1.0 if admitted else 0.0,
        },
        "projected_dose": {
            "requested_duration_s": 20.0,
            "projected_duration_s": 20.0 if admitted else 0.0,
            "projected_duration_fraction": 1.0 if admitted else 0.0,
            "is_predecision_projection": True,
            "is_realized_treatment_dose": False,
        },
        "activation_safety": (
            {
                "safe": True,
                "reason": "safe",
                "checked_cav_ids": ["c0", "c1"],
                "violations": [],
            }
            if admitted
            else None
        ),
        "admitted": admitted,
    }


def _write_scan(root, admission_fn):
    from experiments.idm.wall_v5_2_contract import profile_manifest
    from policies.idm_analytical import frozen_idm_analytical_manifest

    profile = profile_manifest()
    manifest = {
        "schema": "idm_compact_wall_feasibility_manifest/1",
        "scientific_role": "treatment_blind_compact_wall_feasibility",
        "demand_mainline_vph": 2700,
        "demand_ramp_vph": 400,
        "cav_penetrations": list(FROZEN_PENETRATIONS),
        "seeds": list(FROZEN_SEEDS),
        "decision_steps": list(FROZEN_DECISION_STEPS),
        "wall_was_applied": False,
        "treatment_outcomes_read": False,
        "overlay_overrides": profile["overlay_overrides"],
        "wall_profile": profile,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "prospective_seed_artifact_audit": {
            "passed": True,
            "assigned_seeds": list(FROZEN_SEEDS),
            "conflicts": [],
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for penetration in FROZEN_PENETRATIONS:
        token = f"p{round(penetration * 100):02d}"
        directory = root / token / "seeds"
        directory.mkdir(parents=True)
        for seed in FROZEN_SEEDS:
            rows = [
                _result(step, admission_fn(penetration, seed, step))
                for step in FROZEN_DECISION_STEPS
            ]
            payload = {
                "schema": "idm_compact_wall_feasibility_seed/1",
                "scientific_role": "treatment_blind_compact_wall_feasibility",
                "seed": seed,
                "demand_mainline_vph": 2700,
                "demand_ramp_vph": 400,
                "cav_penetration": penetration,
                "decision_steps": list(FROZEN_DECISION_STEPS),
                "results": rows,
                "wall_was_applied": False,
                "treatment_outcomes_read": False,
            }
            (directory / f"seed{seed}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_selects_lowest_penetration_passing_predeclared_rule(tmp_path):
    def admission(penetration, seed, step):
        seed_limit = 7 if penetration == 0.10 else 8
        return seed < FROZEN_SEEDS[seed_limit] and step in (620, 660)

    _write_scan(tmp_path, admission)
    result = analyze(tmp_path)

    assert result["selected_lowest_feasible_penetration"] == 0.15
    p10, p15 = result["penetration_summaries"][:2]
    assert p10["admitted_seed_count"] == 7
    assert p10["admitted_seed_window_count"] == 14
    assert p10["passes_availability_rule"] is False
    assert p15["admitted_seed_count"] == 8
    assert p15["admitted_seed_window_count"] == 16
    assert p15["admitted_decision_steps"] == [620, 660]
    assert p15["passes_availability_rule"] is True


def test_admission_requires_compact_full_occupied_lane_coverage():
    row = _result(620, True)
    row["lane_coverage"]["selected_lane_ids"] = ["lane0"]
    row["lane_coverage"]["coverage_fraction"] = 0.5

    computed = _row_admission(row, AvailabilityRule())

    assert computed["compact_geometry"] is True
    assert computed["full_occupied_lane_coverage"] is False
    assert computed["admitted"] is False


def test_admission_requires_safety_and_projected_dose():
    unsafe = _result(620, True)
    unsafe["activation_safety"]["safe"] = False
    assert _row_admission(unsafe, AvailabilityRule())["admitted"] is False

    short = _result(620, True)
    short["projected_dose"]["projected_duration_s"] = 15.0
    short["projected_dose"]["projected_duration_fraction"] = 0.75
    assert _row_admission(short, AvailabilityRule())["admitted"] is False


def test_analyzer_rejects_any_treatment_outcome_field(tmp_path):
    _write_scan(tmp_path, lambda _p, _seed, _step: False)
    path = tmp_path / "p10" / "seeds" / "seed4301.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outcomes"] = {"delay": 1.0}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="treatment-outcome field"):
        analyze(tmp_path)


def test_analyzer_fails_closed_on_incomplete_declared_grid(tmp_path):
    _write_scan(tmp_path, lambda _p, _seed, _step: False)
    (tmp_path / "p30" / "seeds" / "seed4312.json").unlink()

    with pytest.raises(ValueError, match="scan integrity failed"):
        analyze(tmp_path)


def test_scan_runner_freezes_high_loading_grid_and_v52_geometry():
    from experiments.idm.scan_idm_compact_wall_feasibility import (
        CAV_PENETRATIONS,
        DECISION_STEPS,
        DEMAND_MAINLINE_VPH,
        DEMAND_RAMP_VPH,
        SEEDS,
    )
    from experiments.idm.wall_v5_2_contract import overlay_overrides

    assert DEMAND_MAINLINE_VPH == 2700
    assert DEMAND_RAMP_VPH == 400
    assert CAV_PENETRATIONS == FROZEN_PENETRATIONS
    assert DECISION_STEPS == FROZEN_DECISION_STEPS
    assert SEEDS == FROZEN_SEEDS
    overlay = overlay_overrides()
    assert overlay["geometry_mode"] == "compact_moving"
    assert overlay["compact_selection_mode"] == "trajectory_coherent"
    assert overlay["require_all_occupied_lanes"] is True
    assert overlay["max_longitudinal_span_m"] == 50.0
    assert overlay["planned_treatment_duration_s"] == 20.0
    assert overlay["activation_min_front_gap_m"] == 20.0
    assert overlay["activation_min_rear_gap_m"] == 20.0
