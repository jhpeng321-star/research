import pytest

from tools.run_idm_decentralized_authority_verification import (
    _arms,
    _blend_contract_for_action_set,
    _feedback_contract_for_action_set,
    _effect,
    _has_command_opportunity,
    _landmark_at_state,
    _matched_decision_source_purpose,
    _matched_seed_scope_is_valid,
    _paired_projection_survival,
    _p50_shadow_command,
    _nonclassic_observation_hash,
    _observation_fingerprint,
    _physical_state_fingerprint,
    _projected_release_contract_for_action_set,
    _raw_decision_context_fingerprint,
    _service_horizon_steps,
    _terminal_speed_for_action_set,
    _treatment_fingerprint,
    _treatment_identity,
    _runner_training_authorized,
    _validate_feedback_option_steps,
)
from env.features import (
    RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
    RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1,
)
from env.idm_residual import IDMResidualConfig
from idm.action_local import LandmarkContract
from idm.decentralized_damping import (
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
    ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
    ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
    ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P50_ACTION_SET,
)
from idm.rear_aware_feedback import REAR_AWARE_FEEDBACK_V1, REAR_AWARE_FEEDBACK_V2
from types import SimpleNamespace


def test_service_horizon_never_censors_a_longer_safety_horizon():
    assert _service_horizon_steps(40) == 60
    assert _service_horizon_steps(60) == 60
    assert _service_horizon_steps(69) == 69
    assert _service_horizon_steps(70) == 70


def test_p30_phase_spatial_replay_is_a_narrow_approved_source_purpose():
    assert (
        _matched_decision_source_purpose(
            action_set=ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="idm_phase_response/1",
            diagnostic_spatial_history=True,
        )
        == "p30_modified_followerstopper_match"
    )


def test_projected_blend_match_is_narrow_and_development_only():
    assert (
        _matched_decision_source_purpose(
            action_set=ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="idm_phase_response/1",
            diagnostic_spatial_history=True,
        )
        == "p30_projected_blend_match"
    )
    assert _matched_seed_scope_is_valid(
        source_seeds=range(4121, 4133),
        requested_seeds=(4123, 4127, 4129),
        purpose="p30_projected_blend_match",
    )
    identity = _treatment_identity(
        ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET
    )
    assert identity["terminal_reference_percentile"] == "p30"
    assert identity["blend_contract"] == _blend_contract_for_action_set(
        ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET
    ).to_dict()
    arms = _arms(ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET, (1,))
    assert len(arms) == 6
    assert [row["mode_index"] for row in arms[2:]] == [1, 2, 3, 4]
    assert (
        _matched_decision_source_purpose(
            action_set="activation_speed_ceiling_v1",
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="idm_phase_response/1",
            diagnostic_spatial_history=True,
        )
        == "p30_phase_spatial_replay"
    )
    assert (
        _matched_decision_source_purpose(
            action_set="activation_speed_ceiling_v1",
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="none",
            diagnostic_spatial_history=True,
        )
        is None
    )
    assert (
        _matched_decision_source_purpose(
            action_set="activation_speed_ceiling_v1",
            discover_only=True,
            rsu_observation_contract=RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1,
            response_trace_contract="none",
            diagnostic_spatial_history=False,
        )
        == "rsu_observation_replay"
    )
    assert (
        _matched_decision_source_purpose(
            action_set=ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="idm_phase_response/1",
            diagnostic_spatial_history=True,
        )
        == "p30_projected_release_match"
    )
    assert (
        _matched_decision_source_purpose(
            action_set=ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="idm_phase_response/1",
            diagnostic_spatial_history=True,
        )
        == "p30_projected_release_match"
    )
    assert (
        _matched_decision_source_purpose(
            action_set=ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="idm_phase_response/1",
            diagnostic_spatial_history=True,
        )
        == "p30_projected_release_match"
    )


def test_phase_spatial_replay_allows_only_requested_subset_of_frozen_seeds():
    assert _matched_seed_scope_is_valid(
        source_seeds=range(4121, 4133),
        requested_seeds=(4123, 4127, 4129, 4130, 4131, 4132),
        purpose="p30_phase_spatial_replay",
    )
    assert not _matched_seed_scope_is_valid(
        source_seeds=(4121, 4122),
        requested_seeds=(4121, 9999),
        purpose="p30_phase_spatial_replay",
    )
    assert not _matched_seed_scope_is_valid(
        source_seeds=range(4121, 4133),
        requested_seeds=(4123,),
        purpose="p50_treatment_match",
    )
    assert _matched_seed_scope_is_valid(
        source_seeds=(4123, 4127, 4129),
        requested_seeds=(4127, 4129),
        purpose="p30_projected_release_match",
    )
    assert _matched_seed_scope_is_valid(
        source_seeds=(4123, 4127, 4129),
        requested_seeds=(4129,),
        purpose="p30_modified_followerstopper_match",
    )


def _branch(**overrides):
    row = {
        "arm": "candidate1_nominal",
        "candidate_slot": 1,
        "mode_index": 2,
        "mode": "nominal",
        "completed_horizon_steps": 60,
        "target_command_step_count": 20,
        "unexpected_command_count": 0,
        "collision_observations": 0,
        "teleport_observations": 0,
        "low_ttc_exposure_steps": 0,
        "mean_corridor_delay_rate": 0.2,
        "arrived_vehicles": 10,
        "merge_discharge_count": 20,
        "severe_braking_count_sum": 4.0,
        "acceleration_energy_sum": 40.0,
        "spatial_totals": {
            "influence_region": {"acceleration_energy_m2ps4": 8.0}
        },
        "projection_survival_values": [],
    }
    row.update(overrides)
    return row


def test_frozen_screen_has_two_controls_and_eight_treatments():
    arms = _arms()
    assert len(arms) == 10
    assert len({row["arm"] for row in arms}) == 10
    assert sum(row["is_control"] for row in arms) == 2
    assert {
        (row["candidate_slot"], row["mode_index"])
        for row in arms
        if not row["is_control"]
    } == {(slot, mode) for slot in (1, 2) for mode in (1, 2, 3, 4)}
    cap_arms = _arms("acceleration_caps_v1")
    assert len(cap_arms) == 8
    assert {
        (row["candidate_slot"], row["mode_index"])
        for row in cap_arms
        if not row["is_control"]
    } == {(slot, mode) for slot in (1, 2) for mode in (1, 2, 3)}


def test_effect_never_hides_a_service_constraint_failure():
    control = _branch(
        arm="abstain",
        candidate_slot=None,
        mode_index=0,
        mode="abstain",
        severe_braking_count_sum=10.0,
        acceleration_energy_sum=100.0,
        mean_corridor_delay_rate=0.0,
        spatial_totals={
            "influence_region": {"acceleration_energy_m2ps4": 10.0}
        },
    )
    action = _branch(mean_corridor_delay_rate=2.0)
    effect = _effect(
        control,
        action,
        {"severe": 1.0, "wave": 10.0},
        deadband=0.02,
    )
    assert effect["safety_composite_reference_units"] > 0.0
    assert effect["constraints"]["delay_within_budget"] is False
    assert effect["constraints_pass"] is False
    assert effect["classification"] == "harmful"


def test_projection_survival_uses_paired_nominal_execution():
    nominal = _branch(
        target_vehicle_id="cav",
        command_rows=[
            {
                "relative_step": 1,
                "vehicle_id": "cav",
                "desired_acceleration_mps2": 2.0,
                "projected_acceleration_mps2": 0.5,
            }
        ],
    )
    action = _branch(
        target_vehicle_id="cav",
        command_rows=[
            {
                "relative_step": 1,
                "vehicle_id": "cav",
                "desired_acceleration_mps2": 1.5,
                "projected_acceleration_mps2": 0.5,
            }
        ],
    )
    assert _paired_projection_survival(nominal, action) == 0.0


def test_risk_onset_uses_frozen_wave_then_severe_priority():
    state = {
        "pattern_context": {
            "idm_wave_energy_level": 1.2,
            "idm_wave_energy_trend": 0.2,
            "idm_severe_level": 1.4,
            "idm_severe_trend": 0.3,
        }
    }
    assert _landmark_at_state(state, "risk_onset", LandmarkContract()) == "wave_onset"
    assert _landmark_at_state(state, "severe_onset", LandmarkContract()) == "severe_onset"


def test_command_opportunity_requires_a_physically_active_candidate():
    state = {
        "candidates": [
            {"preview_nominal_acceleration_mps2": -0.05},
            {"preview_nominal_acceleration_mps2": -0.30},
        ]
    }
    assert _has_command_opportunity(state, None)
    assert _has_command_opportunity(state, -0.2)
    assert not _has_command_opportunity(state, -0.4)


def test_p50_shadow_is_diagnostic_and_does_not_mutate_controller_state():
    references = {}
    command = {
        "action_family": "speed_ceiling",
        "lane_id": "lane_0",
        "minimum_gap_m": 2.5,
        "projection_input": {
            "current_speed_mps": 25.0,
            "desired_acceleration_mps2": -1.0,
            "prior_acceleration_mps2": 0.0,
            "allowed_speed_mps": 35.0,
            "leader_gap_m": None,
            "leader_speed_mps": None,
            "follow_speed_mps": None,
            "follower_gap_m": None,
            "follower_speed_mps": None,
        },
    }
    result = _p50_shadow_command(
        command,
        {"speed_p50_mps": 23.0},
        IDMResidualConfig(),
        references,
    )
    assert result["diagnostic_only"] is True
    assert result["filtered_reference_speed_mps"] == 23.0
    assert result["nominal_acceleration_mps2"] == -0.3
    assert result["desired_acceleration_mps2"] == -2.0
    assert result["projected_acceleration_mps2"] == -0.5
    assert references == {"lane_0": 23.0}
    assert command["projection_input"]["desired_acceleration_mps2"] == -1.0

    fallback = {**command, "reference": {"shadow_p50_raw_reference_mps": 24.0}}
    fallback_result = _p50_shadow_command(
        fallback,
        {},
        IDMResidualConfig(),
        {},
    )
    assert fallback_result["raw_reference_speed_mps"] == 24.0


def test_p50_runner_target_and_fingerprint_identity_are_artifact_separated():
    disturbance = SimpleNamespace(
        speed_p30_mps=10.0, speed_p50_mps=17.0, speed_p70_mps=24.0
    )
    assert _terminal_speed_for_action_set(
        ACTIVATION_SPEED_CEILING_ACTION_SET, disturbance
    ) == 10.0
    assert _terminal_speed_for_action_set(
        ACTIVATION_SPEED_CEILING_P50_ACTION_SET, disturbance
    ) == 17.0

    p30 = _treatment_identity(ACTIVATION_SPEED_CEILING_ACTION_SET)
    p50 = _treatment_identity(ACTIVATION_SPEED_CEILING_P50_ACTION_SET)
    assert p30 == {
        "action_set": "activation_speed_ceiling_v1",
        "terminal_reference_percentile": "p30",
    }
    assert p50 == {
        "action_set": "activation_speed_ceiling_p50_v1",
        "terminal_reference_percentile": "p50",
    }
    assert p30 != p50

    feedback = _treatment_identity(
        ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET
    )
    assert feedback["terminal_reference_percentile"] == "p30"
    assert feedback["feedback_contract"]["option_steps"] == 20
    assert feedback["feedback_contract"]["efficacy_conditioned_hold"] is False
    assert feedback != p30


def test_treatment_safety_abort_is_a_hard_effect_constraint():
    effect = _effect(
        _branch(arm="abstain"),
        _branch(
            treatment_safety_abort_rows=[
                {"relative_step": 3, "reason": "projection_interval_conflict"}
            ]
        ),
        {"severe": 1.0, "wave": 10.0},
        0.02,
    )
    assert effect["treatment_safety_abort_count"] == 1
    assert effect["constraints"]["zero_treatment_safety_aborts"] is False
    assert effect["constraints_pass"] is False


def test_feedback_treatment_enforces_duration_and_never_authorizes_training():
    _validate_feedback_option_steps(
        ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET, 20
    )
    with pytest.raises(ValueError, match="freezes option_steps=20"):
        _validate_feedback_option_steps(
            ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET, 25
        )
    assert (
        _runner_training_authorized(
            ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
            {"all_gates": True},
        )
        is False
    )
    assert _runner_training_authorized(
        ACTIVATION_SPEED_CEILING_ACTION_SET,
        {"all_gates": True},
    )

    _validate_feedback_option_steps(
        ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET, 20
    )
    with pytest.raises(ValueError, match="freezes option_steps=20"):
        _validate_feedback_option_steps(
            ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET, 25
        )
    assert not _runner_training_authorized(
        ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
        {"all_gates": True},
    )


def test_rear_aware_feedback_is_a_distinct_matched_development_treatment():
    assert (
        _matched_decision_source_purpose(
            action_set=ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
            discover_only=False,
            rsu_observation_contract=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            response_trace_contract="idm_phase_response/1",
            diagnostic_spatial_history=True,
        )
        == "rear_aware_feedback_match"
    )
    assert _matched_seed_scope_is_valid(
        source_seeds=range(4121, 4133),
        requested_seeds=(4123, 4127),
        purpose="rear_aware_feedback_match",
    )
    identity = _treatment_identity(ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET)
    assert identity["terminal_reference_percentile"] == "p50"
    assert identity["feedback_contract"]["probe_steps"] == 5
    assert identity["release_contract"]["schema"] == "idm_p30_projected_release/2"
    assert _feedback_contract_for_action_set(
        ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET
    ).maximum_active_steps == 20
    assert _projected_release_contract_for_action_set(
        ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET
    ).nominal_budget_steps == 10
    assert len(_arms(ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET, (1,))) == 3
    _validate_feedback_option_steps(ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET, 20)
    with pytest.raises(ValueError, match="freezes option_steps=20"):
        _validate_feedback_option_steps(ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET, 25)
    assert not _runner_training_authorized(
        ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
        {"all_gates": True},
    )
    v2 = _treatment_identity(ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET)
    assert v2["feedback_contract"]["schema"] == "idm_rear_aware_feedback_pulse/2"
    assert v2["feedback_contract"]["risk_conditioned_rear_response"] is True
    assert v2 != identity
    assert (
        _feedback_contract_for_action_set(ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET)
        is REAR_AWARE_FEEDBACK_V2
    )
    assert (
        _feedback_contract_for_action_set(ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET)
        is REAR_AWARE_FEEDBACK_V1
    )
    assert _feedback_contract_for_action_set(
        ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET
    ).soft_response_checks_after_probe is True
    _validate_feedback_option_steps(ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET, 20)
    assert not _runner_training_authorized(
        ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        {"all_gates": True},
    )


def test_rsu_observation_change_preserves_physical_and_treatment_fingerprints():
    candidate = {
        "slot": 1,
        "vehicle_id": "cav",
        "lane_id": "edge_0",
        "geometric_rank": 1,
        "distance_to_merge_m": 1200.0,
        "disturbance_distance_to_merge_m": 600.0,
        "runway_m": 600.0,
        "predicted_dose_s": 20.0,
        "preview_reference_speed_mps": 18.0,
        "preview_nominal_acceleration_mps2": -0.2,
        "terminal_speed_previews": {},
    }
    padded = [0.0] * 134
    padded[9:17] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    live = list(padded)
    live[9:17] = [0.1, 0.6, 0.2, 0.4, 0.05, 0.3, 4.0, 0.7]
    base = {
        "action_set": "activation_speed_ceiling_v1",
        "terminal_reference_percentile": "p30",
        "simulation_time_s": 100.0,
        "lane_id": "edge_0",
        "candidates": [candidate],
        "pattern_context": {"idm_severe_level": 1.0},
    }
    old = {**base, "candidate_observations": {"cav": padded}}
    new = {
        **base,
        "candidate_observations": {"cav": live},
        "candidate_observation_metadata": {
            "cav": {"observation_schema_version": "diagnostic"}
        },
    }
    assert _physical_state_fingerprint(old) == _physical_state_fingerprint(new)
    assert _treatment_fingerprint(old) == _treatment_fingerprint(new)
    assert _observation_fingerprint(old) != _observation_fingerprint(new)
    assert _nonclassic_observation_hash(padded) == _nonclassic_observation_hash(live)


def test_raw_decision_context_ignores_treatment_specific_rear_preview():
    base = {
        "simulation_time_s": 100.0,
        "lane_id": "edge_0",
        "pattern_context": {"idm_severe_level": 1.0},
        "candidates": [
            {
                "slot": 1,
                "vehicle_id": "cav",
                "lane_id": "edge_0",
                "geometric_rank": 1,
                "distance_to_merge_m": 1200.0,
                "disturbance_distance_to_merge_m": 600.0,
                "runway_m": 600.0,
                "predicted_dose_s": 20.0,
                "prospective_rear_secure_gap_m": 10.0,
                "prospective_rear_headroom_m": 5.0,
            }
        ],
    }
    changed = {
        **base,
        "candidates": [
            {
                **base["candidates"][0],
                "prospective_rear_secure_gap_m": 99.0,
                "prospective_rear_headroom_m": -50.0,
            }
        ],
    }
    assert _raw_decision_context_fingerprint(base) == (
        _raw_decision_context_fingerprint(changed)
    )
