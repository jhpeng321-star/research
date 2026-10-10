from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from policies.idm_analytical import (
    FROZEN_IDM_ANALYTICAL_CONTROL_ARM_FINGERPRINT_SHA256,
    FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID,
    FROZEN_IDM_ANALYTICAL_CONTROL_ARM_SCHEMA,
    IDMAnalyticalDampingPolicy,
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_config,
    frozen_idm_analytical_manifest,
)
from policies.counterfactual_escalation import (
    DAMPING_ONLY,
    DAMPING_PLUS_WALL,
    CounterfactualEscalationPolicy,
)


def test_policy_applies_engine_to_active_cavs_and_exposes_diagnostics():
    policy = IDMAnalyticalDampingPolicy()
    engine = MagicMock()
    diagnostics = {"commands": [{"vehicle_id": "cav.1", "target_speed_mps": 18.0}]}
    engine.apply_all.return_value = diagnostics
    policy.engine = engine
    env = SimpleNamespace(_active_cavs=("cav.1", "cav.2"))

    policy.reset(env, seed=4101)
    actions = policy.act({"cav.1": None, "cav.2": None})

    assert policy.direct_control is True
    assert actions == {}
    engine.reset.assert_called_once_with()
    engine.apply_all.assert_called_once_with({}, cav_ids=("cav.1", "cav.2"))
    assert policy.last_diagnostics is diagnostics


def test_policy_falls_back_to_observation_ids_without_bound_environment():
    policy = IDMAnalyticalDampingPolicy()
    engine = MagicMock()
    engine.apply_all.return_value = {"commands": []}
    policy.engine = engine

    actions = policy.act({"cav.2": None, "cav.1": None})

    assert actions == {}
    engine.apply_all.assert_called_once_with({}, cav_ids=dict.fromkeys(("cav.2", "cav.1")).keys())


def test_frozen_factory_pins_exact_analytical_control_arm_without_krauss_fallback():
    policy = build_frozen_idm_analytical_policy()
    manifest = frozen_idm_analytical_manifest()

    assert isinstance(policy, IDMAnalyticalDampingPolicy)
    assert policy.config == frozen_idm_analytical_config()
    assert policy.config.reference_mode == "disturbance_p30"
    assert policy.config.decentralized_damping_enabled is False
    assert manifest["schema"] == FROZEN_IDM_ANALYTICAL_CONTROL_ARM_SCHEMA
    assert manifest["controller_id"] == FROZEN_IDM_ANALYTICAL_CONTROL_ARM_ID
    assert manifest["configuration"]["reference_mode"] == "disturbance_p30"
    assert manifest["residual_action_input"] == "exact_empty_mapping_zero_residual"
    assert manifest["learned_checkpoint"] is None
    assert manifest["fallback_controller"] is None
    assert manifest["legacy_krauss_fallback"] is False
    assert (
        manifest["contract_fingerprint_sha256"]
        == FROZEN_IDM_ANALYTICAL_CONTROL_ARM_FINGERPRINT_SHA256
    )


def test_frozen_manifest_and_factory_are_fresh_and_deterministic():
    first_manifest = frozen_idm_analytical_manifest()
    first_manifest["configuration"]["reference_mode"] = "tampered"
    second_manifest = frozen_idm_analytical_manifest()

    first_policy = build_frozen_idm_analytical_policy()
    second_policy = build_frozen_idm_analytical_policy()

    assert second_manifest["configuration"]["reference_mode"] == "disturbance_p30"
    assert first_policy is not second_policy
    assert first_policy.engine is not second_policy.engine
    assert first_policy.config == second_policy.config


def test_frozen_damping_vs_damping_wrappers_apply_identical_zero_residual_inputs():
    policies = [build_frozen_idm_analytical_policy() for _ in range(2)]
    env = SimpleNamespace(_active_cavs=("cav.1", "cav.2"))
    observations = {"cav.1": None, "cav.2": None}

    for policy in policies:
        engine = MagicMock()
        engine.apply_all.return_value = {
            "commands": [
                {"vehicle_id": "cav.1", "projected_acceleration_mps2": -0.2},
                {"vehicle_id": "cav.2", "projected_acceleration_mps2": -0.1},
            ]
        }
        policy.engine = engine
        policy.reset(env, seed=4101)

    actions = [policy.act(observations) for policy in policies]

    assert actions == [{}, {}]
    assert policies[0].last_diagnostics == policies[1].last_diagnostics
    for policy in policies:
        policy.engine.reset.assert_called_once_with()
        policy.engine.apply_all.assert_called_once_with(
            {}, cav_ids=("cav.1", "cav.2")
        )


def test_frozen_base_actions_continue_through_infeasible_wall_request_and_release():
    base = build_frozen_idm_analytical_policy()
    engine = MagicMock()
    diagnostics = {
        "commands": [
            {"vehicle_id": "cav.1", "projected_acceleration_mps2": -0.2},
            {"vehicle_id": "cav.2", "projected_acceleration_mps2": -0.1},
        ]
    }
    engine.apply_all.return_value = diagnostics
    base.engine = engine
    wall = MagicMock()
    policy = CounterfactualEscalationPolicy(
        base_policy=base,
        wall_factory=MagicMock(return_value=wall),
    )
    env = SimpleNamespace(simulation_step=1.0, _active_cavs=("cav.1", "cav.2"))
    observations = {"cav.1": None, "cav.2": None}
    policy.reset(env, seed=4101)

    with patch("traci.vehicle.getIDList", return_value=[]):
        policy.set_command(DAMPING_PLUS_WALL)
        requested = policy.act(observations)
        requested_audit = dict(policy.last_audit)
        policy.set_command(DAMPING_ONLY)
        released = policy.act(observations)
        released_audit = dict(policy.last_audit)

    assert requested == released == {}
    assert engine.apply_all.call_count == 2
    assert engine.apply_all.call_args_list[0] == engine.apply_all.call_args_list[1]
    assert requested_audit["base_diagnostics"] == released_audit["base_diagnostics"]
    assert requested_audit["base_direct_control_ids"] == ("cav.1", "cav.2")
    assert released_audit["base_direct_control_ids"] == ("cav.1", "cav.2")
    assert requested_audit["wall_override_ids"] == ()
    assert released_audit["wall_override_ids"] == ()
    wall.act.assert_not_called()
