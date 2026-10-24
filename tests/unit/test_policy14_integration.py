"""Readiness invariants: population accounting, abstention, and command parity."""
import copy

import pytest
import torch

from wms.control.contract import RewardWeights, SPEED_LIMIT_MPS
from wms.control.objective14 import authoritative_reward, deployment_accel, score_components
from wms.control.policy import TeamCritic
from wms.control.policy14_probes import probe_state
from wms.control.counterfactual import CounterfactualConfig, build_variants, row_coefficients


def test_primary_j_counts_uninserted_requests_and_does_not_normalize_by_admissions():
    state = probe_state()
    state["entry_queue"] = torch.tensor([8.])
    nxt = {**state, "v_mps": torch.full_like(state["v_mps"], SPEED_LIMIT_MPS)}
    parts = authoritative_reward(state, nxt, torch.zeros_like(state["v_mps"]), RewardWeights())
    assert parts["J"].item() == 4.
    assert parts["reward"].item() == -4.
    # Admitting one queued stopped vehicle transfers its half-second cost to road.
    extra = {key: (torch.cat((value, value[:, :1]), -1) if value.ndim == 2 else value)
             for key, value in state.items()}
    extra["entry_queue"] = torch.tensor([7.])
    added_next = {**extra, "v_mps": torch.cat((nxt["v_mps"], torch.zeros(1, 1)), -1)}
    new = authoritative_reward(extra, added_next, torch.zeros_like(extra["v_mps"]), RewardWeights())
    assert new["J"].item() == parts["J"].item()
    comps = {name: new[name].reshape(1, 1, 1) for name in ("delay", "source_wait")}
    assert score_components(comps, [12.])["seconds_per_requested_vehicle"].item() == pytest.approx(4 / 12)


def test_hdv_follower_braking_is_reported_independently_of_cav_ttc_constraint():
    state = probe_state()
    accel = torch.tensor([[0., 0., -5., 0.]])
    parts = authoritative_reward(state, state, accel, RewardWeights(ttc_pairs="cav_follower"))
    assert parts["hdv_hard_brake_seconds"].item() == .5
    assert parts["hdv_hard_brake"].item() == 4.


@pytest.mark.parametrize("probability,supported,accept", [
    (.8, True, True), (.79999, True, False), (.99, False, False),
    (None, True, False), (float("nan"), True, False), (1.1, True, False),
])
def test_no_uncalibrated_deployment(probability, supported, accept):
    proposal, fallback = object(), object()
    result = deployment_accel(proposal, fallback, calibrated_probability=probability,
                              calibration_supported=supported)
    assert result is (proposal if accept else fallback)


def test_backlog_only_critic_can_learn_value_instead_of_forcing_zero():
    torch.manual_seed(9)
    critic = TeamCritic(entry_backlog=True)
    state = probe_state()
    state["present"].zero_()
    state["entry_backlog"] = torch.tensor([[3., 2.]], requires_grad=True)
    value = critic(state)
    grad = torch.autograd.grad(value.sum(), state["entry_backlog"])[0]
    assert value.abs().item() > 0 and grad.abs().sum().item() > 0


def test_zero_synergy_with_uncovered_members_is_valid_and_team_count_independent():
    state = probe_state()
    state["is_cav"][:] = True
    cfg = CounterfactualConfig(enabled=True, max_starts=1, n_individual=0, n_group=1,
                              n_flip=1, cover_group_members=False,
                              curriculum_updates=[0], w_individual=[0], w_coop=[0],
                              w_group=[0], w_team=[0], w_all=[1])
    variants = build_variants(state, cfg, torch.Generator().manual_seed(3))
    coefficients = row_coefficients(variants, cfg.weights(0), variants.n_rows, 1)
    assert coefficients[0] == 1 and coefficients[1:].sum() == 0


def test_bridge_masks_forced_merges_and_applies_native_cap_symmetrically(monkeypatch):
    from wms.control.bridge import make_cav_controller
    from wms.control.lanes import LaneRules
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.train_cli import GroupSignal
    import wms.model.cav as cav
    state = probe_state()
    state["s_m"][0, 0] = 74.5
    state["v_mps"][0, 0] = 1.
    state["s_m"][0, 2] = 74.5  # overlapping target-lane follower
    seen = []
    def cap(state, action, mask):
        seen.append(mask.clone())
        return action
    monkeypatch.setattr(cav, "apply_exec_cap", cap)
    policy = CavPolicy(lane_head=True, accel_max=1.5)
    controller = make_cav_controller(policy, GroupSignal(RsuGroupEncoder()),
                  lambda state, action, **kwargs: action, lane_rules=LaneRules(),
                  act_mask=torch.zeros_like(state["is_cav"]),
                  native=lambda state: torch.zeros_like(state["v_mps"]),
                  symmetric_exec_cap=True, mask_forced_merge=True)
    motion = controller(copy.deepcopy(state))
    assert motion[0, 0, 1].item() == 2
    assert torch.equal(seen[0], state["present"] & state["is_cav"])


def test_launch_defaults_enable_available_guards_and_supplied_train_accel_box():
    from wms.control.dreamer import DreamerConfig
    from wms.control.lanes import LaneRules
    from wms.control.train_cli import _read_config
    config = _read_config("configs/c4_policy14.toml")
    trainer = DreamerConfig(**config["trainer"])
    rules = LaneRules.from_mapping({k: v for k, v in config["lanes"].items() if k != "enabled"})
    assert rules.consequence_mask and rules.secure_gap_hook == ""
    assert trainer.low_penetration_horizon_cap and trainer.wm_safety_report_only
    assert config["policy"]["accel_max_mps2"] == config["counterfactual"]["native_accel_max_mps2"] == 1.2


def test_consequence_mask_cannot_be_bypassed_by_forced_ramp_merge():
    from wms.control.policy14 import controller_factory
    from wms.control.lanes import LaneRules
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.train_cli import GroupSignal
    from wms.model.driver_encoder import PARAMETER_NAMES
    state = probe_state()
    state["s_m"][0, 2] = -10.
    state["v_mps"][0, 2] = 15.
    defaults = dict(v0=30., time_headway=1.2, min_gap=2., accel=1.2, decel=.8)
    state["inferred_driver_parameters"] = torch.tensor(
        [defaults.get(name, 1.) for name in PARAMETER_NAMES]).expand(1, 4, -1).clone()
    state["inferred_idm_delta"] = 4
    rules = LaneRules(consequence_mask=True, gap_base_m=0., gap_time_s=0., closing_time_s=0.,
                      rear_gap_base_m=0., rear_gap_time_s=0., rear_closing_time_s=0.)
    controller = controller_factory(CavPolicy(lane_head=True), GroupSignal(RsuGroupEncoder()),
                                    lambda state, action, **kwargs: action, lane_rules=rules)
    assert controller(state)[0, 0, 1].item() == 2


def test_courtesy_repair_keeps_previous_acceleration_history_derivative():
    from wms.model.follower_courtesy import follower_courtesy_pairs
    from wms.physics.config import PhysicsConfig
    from wms.physics.state import state_kin
    state = probe_state()
    kin = state_kin(state)
    history = torch.zeros(1, 4, 3, 6, requires_grad=True)
    mask = torch.ones(1, 4, 3, dtype=torch.bool)
    off = follower_courtesy_pairs(kin, state["present"], PhysicsConfig(), version=2,
                                  history=history, history_mask=mask, state_grad=False)
    on = follower_courtesy_pairs(kin, state["present"], PhysicsConfig(), version=2,
                                 history=history, history_mask=mask, state_grad=True)
    assert torch.equal(off["features"], on["features"].detach())
    grad = torch.autograd.grad(on["features"][..., 11].sum(), history)[0]
    assert grad[0, 0, -2, 1].item() == 1.
