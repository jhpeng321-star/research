from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from experiments.idm.wall_v5_4_contract import profile_manifest as v54_manifest
from experiments.idm.wall_v5_5_1_contract import overlay_overrides, profile_manifest
from policies.counterfactual_escalation import CounterfactualEscalationPolicy, TreatmentState


class _Base:
    last_diagnostics = {}

    def reset(self, env, seed=None):
        pass

    def act(self, obs):
        return {str(key): 0 for key in obs}

    def close(self):
        pass


def _policy() -> CounterfactualEscalationPolicy:
    overrides = overlay_overrides()
    policy = CounterfactualEscalationPolicy(
        base_policy=_Base(),
        wall_factory=MagicMock(return_value=MagicMock()),
        **overrides,
    )
    policy.reset(SimpleNamespace(simulation_step=1.0), seed=101)
    policy._treatment_state = TreatmentState.HOLDING
    return policy


def _sample(policy, *, follower="hdv.1", gap=19.0, follower_speed=25.0, accel=0.0):
    def neighbor(kind, vehicle):
        if kind == "getFollower" and vehicle == "cav.1":
            return follower, gap
        return None, None

    speeds = {"cav.1": 20.0, follower: follower_speed}
    with (
        patch.object(policy, "_neighbor", side_effect=neighbor),
        patch.object(policy, "_vehicle_float", side_effect=lambda key, vehicle: (
            speeds.get(vehicle) if key == "getSpeed" else accel if vehicle == follower else 0.0
        )),
    ):
        return policy._adaptive_release_status(("cav.1",), {"cav.1", follower})


def test_v551_profile_is_distinct_and_enables_only_the_versioned_repair():
    assert profile_manifest()["profile_sha256"] != v54_manifest()["profile_sha256"]
    assert overlay_overrides()["adaptive_release_enabled"] is True
    assert overlay_overrides()["adaptive_release_soft_rear_ttc_s"] == 6.0
    assert overlay_overrides()["adaptive_release_persistence_samples"] == 1


def test_gap_loss_without_the_risk_gate_does_not_trigger_release():
    policy = _policy()
    _sample(policy, gap=40.0, follower_speed=20.0)
    status = _sample(policy, gap=30.0, follower_speed=20.0)
    assert status["should_taper"] is False
    assert status["soft_reasons"] == ()


def test_risk_gated_same_follower_compression_latches_release_deterministically():
    policy = _policy()
    _sample(policy, gap=30.0, follower_speed=21.0)
    status = _sample(policy, gap=19.0, follower_speed=24.0)
    assert status["should_taper"] is True
    assert status["soft_samples"] == 1
    assert status["soft_reasons"] == ("risk_gated_compression:cav.1",)


def test_identity_change_requests_safe_release_without_rebasing_baseline():
    policy = _policy()
    _sample(policy, follower="hdv.1", gap=30.0, follower_speed=20.0)
    status = _sample(policy, follower="hdv.2", gap=30.0, follower_speed=20.0)
    assert status["should_taper"] is True
    assert status["identity_or_missing_reasons"] == (
        "follower_identity_changed:cav.1:hdv.1->hdv.2",
    )
    assert policy._adaptive_release_follower_ids["cav.1"] == "hdv.1"


def test_hard_abort_is_not_an_adaptive_release():
    policy = _policy()
    policy._last_controlled_ids = ("cav.1",)
    with (
        patch.object(policy, "_event_ids", return_value=()),
        patch("policies.counterfactual_escalation.traci.vehicle.getAcceleration", return_value=-5.0),
    ):
        assert policy._safety_abort_reason({"cav.1"}) == "controlled_cav_hard_brake"


def test_adaptive_release_latch_uses_existing_monotone_taper_state():
    policy = _policy()
    policy._last_member_speed_targets_mps = {"cav.1": 15.0}
    with patch.object(policy, "_vehicle_float", return_value=15.0):
        policy._begin_taper("adaptive_soft_compression:risk_gated", ("cav.1",))
    assert policy.treatment_state == TreatmentState.TAPERING
    assert policy._taper_reason == "adaptive_soft_compression:risk_gated"
    assert policy._taper_origin_targets_mps == {"cav.1": 15.0}
