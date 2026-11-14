from experiments.idm.idm_v5_5_t880_contract import (
    ACTIVATION_TIME_S,
    RELEASE_TIME_S,
    TREATMENT_DURATION_S,
    contract_manifest,
)
from experiments.idm.scan_idm_v5_5_t880_gate_f import _validate_seed_payload
from experiments.idm.wall_v5_4_contract import profile_manifest
from idm.wall_t880 import gate_f_assessment
from policies.idm_analytical import frozen_idm_analytical_manifest


def _formation(**overrides):
    value = {
        "feasible": True,
        "reason": "feasible",
        "occupied_lane_ids": ["lane0", "lane1"],
        "required_lane_ids": ["lane0", "lane1"],
        "longitudinal_span_m": 20.0,
        "predicted_max_span_m": 30.0,
        "predicted_terminal_span_m": 25.0,
    }
    value.update(overrides)
    return value


def test_timing_contract_distinguishes_activation_and_release() -> None:
    manifest = contract_manifest()
    assert ACTIVATION_TIME_S == 880
    assert RELEASE_TIME_S == 900
    assert TREATMENT_DURATION_S == 20
    assert manifest["primary_outcome"]["role"] == (
        "prevention_or_shortening_of_sustained_premerge_congestion"
    )
    assert manifest["secondary_diagnostic"]["not_an_outcome_gate"] is True


def test_gate_f_admits_only_complete_safe_full_dose_geometry() -> None:
    result = gate_f_assessment(
        formation=_formation(),
        selected_lane_ids=["lane0", "lane1"],
        activation_safety={"safe": True},
        terminal_runway_margin_m=10.0,
        requested_duration_s=20.0,
    )
    assert result["admitted"] is True
    assert result["projected_duration_fraction"] == 1.0


def test_gate_f_rejects_each_required_layer() -> None:
    cases = [
        dict(formation=_formation(feasible=False)),
        dict(selected_lane_ids=["lane0"]),
        dict(formation=_formation(predicted_max_span_m=51.0)),
        dict(formation=_formation(longitudinal_span_m=None)),
        dict(terminal_runway_margin_m=-0.1),
        dict(terminal_runway_margin_m=float("inf")),
        dict(activation_safety={"safe": False}),
    ]
    common = {
        "formation": _formation(),
        "selected_lane_ids": ["lane0", "lane1"],
        "activation_safety": {"safe": True},
        "terminal_runway_margin_m": 10.0,
        "requested_duration_s": 20.0,
    }
    for changes in cases:
        assert gate_f_assessment(**{**common, **changes})["admitted"] is False


def _valid_seed_payload() -> dict:
    formation = {
        **_formation(),
        "controlled_cav_ids": ["cav0", "cav1"],
    }
    safety = {"safe": True, "checked_cav_ids": ["cav0", "cav1"]}
    gate_f = gate_f_assessment(
        formation=formation,
        selected_lane_ids=["lane0", "lane1"],
        activation_safety=safety,
        terminal_runway_margin_m=10.0,
        requested_duration_s=20.0,
    )
    from experiments.idm.idm_v5_5_t880_contract import contract_manifest
    from experiments.idm.scan_idm_v5_5_t880_gate_f import _scenario_provenance

    return {
        "schema": "idm_v5_5_t880_gate_f_seed/1",
        "scientific_role": "timing_aligned_late_congestion_wall_headroom_development",
        "seed": 4401,
        "decision_sim_time_s": 880.0,
        "contract_fingerprint_sha256": contract_manifest()[
            "contract_fingerprint_sha256"
        ],
        "scenario_provenance": _scenario_provenance(),
        "base_control_arm": frozen_idm_analytical_manifest(),
        "wall_profile": profile_manifest(),
        "formation": formation,
        "selected_members": [
            {"vehicle_id": "cav0", "lane_id": "lane0"},
            {"vehicle_id": "cav1", "lane_id": "lane1"},
        ],
        "selected_lane_ids": ["lane0", "lane1"],
        "activation_safety": safety,
        "terminal_runway_margin_m": 10.0,
        "gate_f": gate_f,
        "wall_was_applied": False,
        "treatment_outcomes_read": False,
    }


def test_seed_provenance_validation_fails_closed() -> None:
    payload = _valid_seed_payload()
    assert all(_validate_seed_payload(payload, 4401).values())
    payload["cav_penetration"] = 0.10
    # Unrecognized loose fields cannot override the fingerprinted scenario object.
    assert all(_validate_seed_payload(payload, 4401).values())
    payload["scenario_provenance"]["cav_penetration"] = 0.10
    try:
        _validate_seed_payload(payload, 4401)
    except RuntimeError as exc:
        assert "scenario" in str(exc)
    else:
        raise AssertionError("mismatched p30 provenance was accepted")


def test_seed_provenance_recomputes_gate_f() -> None:
    payload = _valid_seed_payload()
    payload["gate_f"]["admitted"] = False
    try:
        _validate_seed_payload(payload, 4401)
    except RuntimeError as exc:
        assert "raw_gate_f_recomputed" in str(exc)
    else:
        raise AssertionError("tampered Gate-F result was accepted")
