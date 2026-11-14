import json

import pytest

from experiments.idm.run_idm_wall_pilot import (
    _base_policy,
    overlay_overrides,
    supervisor_overrides,
)
from policies.idm_analytical import (
    FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID,
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


def test_wall_pilot_freezes_audited_v5_1_treatment():
    overlay = overlay_overrides()
    assert overlay["geometry_mode"] == "compact_moving"
    assert overlay["compact_selection_mode"] == "trajectory_coherent"
    assert overlay["compact_control_mode"] == "adaptive_servo"
    assert overlay["require_all_occupied_lanes"] is True
    assert overlay["max_longitudinal_span_m"] == 50.0
    assert overlay["planned_treatment_duration_s"] == 20.0
    assert overlay["compact_max_speed_reduction_fraction"] == 0.20
    assert overlay["compact_servo_a_max_dec_mps2"] == 1.0
    assert overlay["wall_params"] == {
        "d_lo": 200.0,
        "d_hi": 1000.0,
        "wall_speed": 10.0,
        "wall_tau": 3.0,
        "a_max_dec": 0.3,
        "tau_ramp_step": 0.05,
    }


def test_wall_pilot_uses_frozen_idm_environment_contract():
    env = supervisor_overrides()
    assert env["simulation_step"] == 1.0
    assert env["assign_hdv_types"] is False
    assert env["cav_force_normal_type"] is True
    assert env["cav_warmup_steps"] == 300
    assert env["centralized_critic"] is False
    assert env["action_masking"] is False
    assert env["control_mode"] == "shepherd"


def test_wall_pilot_base_factory_is_the_frozen_analytical_control_arm():
    from_runner = _base_policy()
    from_contract = build_frozen_idm_analytical_policy()

    assert type(from_runner) is type(from_contract)
    assert from_runner.config == from_contract.config
    assert from_runner.config.reference_mode == "disturbance_p30"


def test_v5_2_wall_contract_fails_closed_without_exact_analytical_factory():
    from experiments.idm.wall_v5_2_contract import require_accepted_damping_factory

    assert (
        require_accepted_damping_factory(build_frozen_idm_analytical_policy)
        is build_frozen_idm_analytical_policy
    )
    with pytest.raises(RuntimeError, match="explicitly injected"):
        require_accepted_damping_factory(None)
    with pytest.raises(RuntimeError, match="only build_frozen"):
        require_accepted_damping_factory(lambda: _base_policy())


def test_wall_control_and_treatment_branches_share_control_arm_fingerprint(monkeypatch):
    import experiments.idm.run_idm_wall_pilot as module

    calls = []
    monkeypatch.setattr(
        module,
        "run_replay",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )
    case = {
        "seed": 4101,
        "decision_step": 720,
        "outcome_steps": 260,
        "treatment_steps": 20,
        "sumo_binary": "sumo",
    }

    module._run_branch(case, treatment=False)
    module._run_branch(case, treatment=True)

    expected = frozen_idm_analytical_manifest()
    assert len(calls) == 2
    for call_args in calls:
        assert call_args["base_policy_factory"] is _base_policy
        assert call_args["controller_metadata"]["base_controller"] == (
            FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID
        )
        assert call_args["controller_metadata"]["base_control_arm"] == expected
        assert call_args["controller_metadata"]["base_control_arm"][
            "legacy_krauss_fallback"
        ] is False


def test_timing_scan_matches_pilot_environment():
    from experiments.idm.scan_idm_wall_timing import DEFAULT_DECISIONS, _env_config
    from experiments.idm.wall_v5_2_contract import DECISION_STEPS

    assert DEFAULT_DECISIONS == DECISION_STEPS
    cfg = _env_config(4101, max(DEFAULT_DECISIONS) + 1, "sumo")
    for key, value in supervisor_overrides().items():
        assert cfg[key] == value
    assert cfg["sumo_seed"] == 4101
    assert cfg["max_episode_steps"] == max(DEFAULT_DECISIONS) + 6


def test_multiwindow_pilot_freezes_diagnostic_action_set():
    from experiments.idm.run_idm_wall_multiwindow_pilot import (
        DECISION_STEPS,
        OUTCOME_STEPS,
        _infeasible_dose,
    )
    from experiments.idm.wall_v5_2_contract import PROFILE_ID, profile_manifest

    assert DECISION_STEPS == (620, 660, 680, 700)
    assert OUTCOME_STEPS == 260
    assert profile_manifest()["profile_id"] == PROFILE_ID
    assert profile_manifest()["base_controller"] == frozen_idm_analytical_manifest()
    dose = _infeasible_dose({"admitted": False})
    assert dose["activation_feasible"] is False
    assert dose["realized_duration_fraction"] == 0.0
    assert dose["predecision_scan"] == {"admitted": False}


def test_multiwindow_replays_share_identical_total_horizon(monkeypatch):
    import experiments.idm.run_idm_wall_multiwindow_pilot as module

    calls = []
    monkeypatch.setattr(
        module,
        "run_replay",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )
    module._replay(
        seed=4101,
        decision=620,
        total_steps=960,
        sumo_binary="sumo",
        treatment=False,
        cav_penetration=0.30,
        base_policy_factory=build_frozen_idm_analytical_policy,
    )
    module._replay(
        seed=4101,
        decision=700,
        total_steps=960,
        sumo_binary="sumo",
        treatment=True,
        cav_penetration=0.30,
        base_policy_factory=build_frozen_idm_analytical_policy,
    )

    assert calls[0]["decision_step"] + calls[0]["horizon_steps"] == 960
    assert calls[1]["decision_step"] + calls[1]["horizon_steps"] == 960
    assert calls[0]["horizon_steps"] == 340
    assert calls[1]["horizon_steps"] == 260
    assert calls[0]["cav_penetration"] == 0.30
    assert calls[1]["cav_penetration"] == 0.30
    assert calls[0]["base_policy_factory"] is build_frozen_idm_analytical_policy
    assert calls[1]["base_policy_factory"] is build_frozen_idm_analytical_policy
    assert calls[0]["controller_metadata"]["base_control_arm"] == (
        frozen_idm_analytical_manifest()
    )
    assert calls[1]["controller_metadata"]["base_control_arm"] == (
        frozen_idm_analytical_manifest()
    )


def test_v5_2_profile_changes_only_declared_activation_gap_floors():
    from experiments.idm.wall_v5_2_contract import (
        PROFILE_ID,
        overlay_overrides as v5_2_overrides,
        profile_manifest,
        treatment_repair_criteria,
    )

    previous = overlay_overrides()
    current = v5_2_overrides()
    assert current.pop("activation_min_front_gap_m") == 20.0
    assert current.pop("activation_min_rear_gap_m") == 20.0
    assert current == previous
    assert PROFILE_ID == "adaptive_compact_wall_v5_2_prospective"
    assert profile_manifest()["collapse_prevention_role"] == (
        "prespecified_secondary_not_gate"
    )
    assert treatment_repair_criteria().require_collapse_prevention_diversity is False


def test_v5_2_multiwindow_replay_fails_closed_without_factory(monkeypatch):
    import experiments.idm.run_idm_wall_multiwindow_pilot as module

    monkeypatch.setattr(module, "run_replay", lambda **kwargs: kwargs)
    with pytest.raises(RuntimeError, match="explicitly injected"):
        module._replay(
            seed=4301,
            decision=620,
            total_steps=960,
            sumo_binary="sumo",
            treatment=False,
            cav_penetration=0.30,
        )


def test_v5_2_repeat_cases_are_first_two_admitted_seeds(tmp_path):
    from experiments.idm.wall_v5_2_contract import select_repeat_cases

    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    rows = {
        4301: [{"decision_step": 620, "admitted": False}],
        4302: [
            {"decision_step": 660, "admitted": True},
            {"decision_step": 620, "admitted": True},
        ],
        4303: [{"decision_step": 680, "admitted": False}],
        4304: [{"decision_step": 700, "admitted": True}],
    }
    for seed, results in rows.items():
        (seed_dir / f"seed{seed}.json").write_text(
            json.dumps({"results": results}), encoding="utf-8"
        )

    assert select_repeat_cases(tmp_path, rows) == {4302: 620, 4304: 700}


def test_v5_2_seed_artifact_audit_allows_only_declared_roots(tmp_path):
    from experiments.idm.wall_v5_2_contract import assert_no_prior_seed_artifacts

    allowed = tmp_path / "allowed"
    conflict = tmp_path / "other"
    allowed.mkdir()
    conflict.mkdir()
    (allowed / "seed4301.json").write_text('{"seed": 4301}', encoding="utf-8")
    (conflict / "unrelated.json").write_text('{"seed": 9999}', encoding="utf-8")
    audit = assert_no_prior_seed_artifacts(
        tmp_path,
        (4301, 4302),
        allowed_roots=(allowed,),
    )
    assert audit["passed"] is True

    (conflict / "manifest.json").write_text(
        '{"seeds": [4302]}', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="already exist"):
        assert_no_prior_seed_artifacts(
            tmp_path,
            (4301, 4302),
            allowed_roots=(allowed,),
        )


def test_multiwindow_decision_features_merge_control_prefix_and_preflight(
    monkeypatch,
):
    import experiments.idm.run_idm_wall_multiwindow_pilot as module

    monkeypatch.setattr(
        module,
        "features_from_trace",
        lambda control, *, decision_step, definition, dose_context=None: {
            "health": 0.4,
            "source_step": decision_step,
        },
    )
    values = module._decision_features(
        object(),
        decision=660,
        definition=object(),
        scan={
            "active_cav_count": 22,
            "damping_command_count": 2,
            "formation": {
                "controlled_cav_ids": ["c0", "c1"],
                "required_lane_ids": ["lane0", "lane1"],
                "longitudinal_span_m": 5.0,
                "predicted_max_span_m": 8.0,
                "predicted_terminal_span_m": 3.0,
            },
            "activation_safety": {
                "min_front_gap_m": 30.0,
                "min_rear_gap_m": 25.0,
                "min_front_ttc_s": None,
                "min_rear_ttc_s": 20.0,
                "min_acceleration_mps2": -0.2,
            },
        },
    )

    assert values["health"] == 0.4
    assert values["source_step"] == 660
    assert values["preflight_member_count"] == 2.0
    assert values["preflight_min_rear_gap_m"] == 25.0


def test_multiwindow_summary_uses_selected_penetration_not_global_default(tmp_path):
    import experiments.idm.run_idm_wall_multiwindow_pilot as module

    (tmp_path / "manifest.json").write_text(
        json.dumps({"cav_penetration": 0.30}), encoding="utf-8"
    )
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "seed4301.json").write_text(
        json.dumps({"cases": [], "repeat_checks": None}), encoding="utf-8"
    )

    summary = module.build_summary(tmp_path, [4301])

    assert summary["cav_penetration"] == 0.30
    assert summary["contract"]["cav_penetration"] == 0.30
    assert summary["contract"]["cav_penetration_source"] == (
        "treatment_blind_compact_wall_feasibility_selection"
    )
