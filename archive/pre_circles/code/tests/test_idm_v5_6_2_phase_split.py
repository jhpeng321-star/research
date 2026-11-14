from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from experiments.idm.wall_v5_6_2_phase_split_contract import overlay_overrides, profile_manifest
from policies.counterfactual_escalation import CounterfactualEscalationPolicy, TreatmentState


class _Damping:
    def reset(self, _env, seed=None):
        self.seed = seed

    def act(self, _obs):
        return {}

    def close(self):
        pass


def _phase_split_policy(wall):
    policy = CounterfactualEscalationPolicy(
        base_policy=_Damping(),
        wall_factory=MagicMock(return_value=wall),
        **overlay_overrides(),
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    return policy


def test_phase_split_contract_has_one_high_step_and_low_sustained_caps():
    values = overlay_overrides()
    assert values["compact_initial_assembly_servo_a_max_dec_mps2"] == 1.0
    assert values["compact_high_authority_assembly_steps"] == 1
    assert values["compact_settling_jerk_limit_mps3"] == 0.5
    assert values["compact_settling_max_steps"] == 2
    assert values["compact_servo_a_max_dec_mps2"] == 0.3
    assert len(profile_manifest()["profile_sha256"]) == 64


def test_projection_and_runtime_share_the_one_step_high_authority_cap():
    wall = MagicMock(); wall.a_max_dec = 0.3
    observed = []
    wall.act_with_targets.side_effect = lambda _obs, _targets: observed.append(wall.a_max_dec)
    policy = _phase_split_policy(wall)
    policy._treatment_state = TreatmentState.ASSEMBLING
    policy._assembly_target_speed_mps = 18.0
    with (
        patch.object(policy, "_mean_speed_in_distance_band", return_value=15.0),
        patch.object(policy, "_adaptive_member_targets", return_value={"cav.0": 18.0}),
    ):
        policy._project_compact_formation(
            ("cav.0", "cav.1"),
            {"cav.0": 300.0, "cav.1": 301.0},
            {"cav.0": 20.0, "cav.1": 19.0},
        )
        policy._apply_wall_actuator({"cav.0": None})
    assert observed == [1.0]
    assert policy._assembly_elapsed_steps == 1
    assert policy._last_wall_a_max_dec_mps2 == 1.0


def test_holding_and_taper_never_restore_high_authority():
    wall = MagicMock(); wall.a_max_dec = 0.3
    observed = []
    wall.act_with_targets.side_effect = lambda _obs, _targets: observed.append(wall.a_max_dec)
    policy = _phase_split_policy(wall)
    policy._treatment_state = TreatmentState.HOLDING
    policy._adaptive_common_target_mps = 18.0
    with patch.object(policy, "_adaptive_member_targets", return_value={"cav.0": 18.0}):
        policy._apply_wall_actuator({"cav.0": None})
    policy._treatment_state = TreatmentState.TAPERING
    policy._taper_origin_targets_mps = {"cav.0": 18.0}
    with (
        patch.object(policy, "_base_target_speed_mps", return_value=20.0),
        patch.object(policy, "_safety_project_target", return_value=(19.0, {})),
    ):
        policy._apply_taper_actuator({"cav.0": None})
    assert observed == [0.3, 0.3]
    assert wall.a_max_dec == 0.3


def test_unready_one_step_assembly_fails_closed_to_damping():
    policy = _phase_split_policy(MagicMock())
    policy._wall_active = True
    policy._treatment_state = TreatmentState.ASSEMBLING
    policy._assembly_elapsed_steps = 1
    with patch.object(policy, "_member_speed_spread", return_value=3.0):
        reason = policy._advance_compact_assembly(("cav.0", "cav.1"))
    assert reason == "assembly_abort:high_authority_step_not_compact"
    assert policy.wall_active is False
    assert policy.treatment_state == TreatmentState.RELEASING


def test_settling_uses_realized_assembly_acceleration_and_jerk_bound():
    wall = MagicMock(); wall.a_max_dec = 0.3; wall._tau_for = None; wall.set_tau_blend = None
    applied = []
    wall.act_with_targets.side_effect = lambda _obs, targets: applied.append((wall.a_max_dec, dict(targets)))
    policy = _phase_split_policy(wall)
    policy._treatment_state = TreatmentState.ASSEMBLING
    policy._assembly_elapsed_steps = 1
    values = {("getAcceleration", "cav.0"): -1.0, ("getSpeed", "cav.0"): 20.0}
    with (
        patch.object(policy, "_member_speed_spread", return_value=0.0),
        patch.object(policy, "_vehicle_float", side_effect=lambda name, member: values[(name, member)]),
    ):
        assert policy._advance_compact_assembly(("cav.0",)) == "wall_assembly_complete_settling_started"
        policy._apply_settling_actuator({"cav.0": None})
    assert policy.treatment_state == TreatmentState.SETTLING
    assert applied == [(0.5, {"cav.0": 19.5})]
    assert policy._settling_target_accelerations_mps2 == {"cav.0": -0.5}
    assert policy._settling_max_jerk_mps3 == 0.5
    assert policy._settling_elapsed_steps == 1


def test_settling_relaxes_monotonically_to_low_authority_without_tau_tightening():
    wall = MagicMock(); wall.a_max_dec = 0.3
    wall._tau_offset_cache = {"cav.0": 0.1}; wall._tau_for.return_value = 1.0
    applied = []
    wall.act_with_targets.side_effect = lambda _obs, targets: applied.append((wall.a_max_dec, dict(targets)))
    policy = _phase_split_policy(wall)
    policy._treatment_state = TreatmentState.SETTLING
    policy._settling_prior_accelerations_mps2 = {"cav.0": -0.5}
    with patch.object(policy, "_vehicle_float", side_effect=lambda name, _member: 20.0 if name == "getSpeed" else -0.5):
        policy._apply_settling_actuator({"cav.0": None})
    assert applied == [(0.3, {"cav.0": 19.7})]
    assert policy._settling_target_accelerations_mps2 == {"cav.0": -0.3}
    assert policy._settling_tau_frozen is True
    wall.set_tau_blend.assert_called_with(("cav.0",), 0.1)


def test_settling_hard_fails_after_its_frozen_two_step_budget():
    policy = _phase_split_policy(MagicMock())
    policy._wall_active = True
    policy._treatment_state = TreatmentState.SETTLING
    policy._settling_elapsed_steps = 2
    with patch.object(policy, "_vehicle_float", return_value=-0.8):
        reason = policy._advance_settling(("cav.0",))
    assert reason == "settling_abort:max_duration_not_low_authority"
    assert policy.wall_active is False
    assert policy.treatment_state == TreatmentState.RELEASING


def test_settling_completion_requires_actual_low_authority_response():
    policy = _phase_split_policy(MagicMock())
    policy._treatment_state = TreatmentState.SETTLING
    with patch.object(policy, "_vehicle_float", return_value=-0.3):
        reason = policy._advance_settling(("cav.0",))
    assert reason == "settling_complete_low_authority_envelope"
    assert policy.treatment_state == TreatmentState.HOLDING
