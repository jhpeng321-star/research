"""Learned native-residual pathways and restart invariants, no simulator/data."""
from dataclasses import replace

import pytest
import torch

from wms.control.counterfactual import CounterfactualConfig
from wms.control.dreamer import DreamerConfig
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.policy14 import Policy14Trainer, make_controller_factory
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.start_states import StartBatch
from wms.control.train_cli import GroupSignal
from wms.model import ModelSpec, WmSModel
from wms.physics.config import PhysicsConfig

from .test_longmem import assert_tree_close


class Starts:
    def sample(self, batch_size, device="cpu"):
        def row(values, dtype=torch.float32):
            return torch.tensor([values], dtype=dtype, device=device).expand(batch_size, -1).clone()
        return StartBatch(dict(s_m=row([-60., -80., -100., -120., -140., -160.]),
            v_mps=row([5., 6., 7., 4., 5., 6.]), a_mps2=row([0.] * 6), length_m=row([5.] * 6),
            lane=row([1, 0, 1, 1, 0, 0], torch.long), present=row([True] * 6, torch.bool),
            is_cav=row([True, True, True, False, False, False], torch.bool)))


def trainer(group=False, trunk=False, memory=False):
    torch.manual_seed(101)
    model = WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              merge_decision="first_passage", lane_decision="sample"),
                     physics=PhysicsConfig(secure_gap_veto=True))
    result = Policy14Trainer([model], CavPolicy(lane_head=False, residual_on_native=True, accel_max=1.2),
        GroupSignal(RsuGroupEncoder()), TeamCritic(), Starts(), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel,
        controller_factory=make_controller_factory(native_ramp_merge="wm_law", native_lane_change=True),
        config=DreamerConfig(batch_size=1, horizon=3, actor_lr=5e-5,
            gamma=1., lambda_=1., entropy_weight=0., uncertainty_weight=0.,
            checkpoint_every=0, validation_every=0, wm_safety_report_only=True,
            native_residual_train_group=group, native_residual_train_trunk=trunk,
            imagination_microbatch=int(memory), imagination_checkpoint_segment=2 if memory else 0,
            imagination_offload_futures=memory, imagination_offload_saved_tensors=memory),
        counterfactual=CounterfactualConfig(enabled=True, n_individual=0, n_group=1, n_flip=3,
            native_cav="idm", group_min_size=3, group_max_size=3, max_starts=1))
    result.long_credit_requested = True
    return result


@pytest.mark.parametrize("group,trunk", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_pathways_and_zero_init(group, trunk):
    t = trainer(group, trunk)
    state = Starts().sample(1).state
    obs = build_observations(state)
    action = t.policy(obs, t.group_encoder(obs, state["s_m"], None))[0]
    assert torch.equal(action, torch.zeros_like(action))
    first = t.update()
    assert first["actor_module_grad_norms"]["group_encoder"] == 0
    assert first["actor_module_grad_norms"]["own_encoder"] == 0
    second = t.update()
    for name, enabled in (("group_encoder", group), ("own_encoder", trunk),
                          ("report_encoder", trunk), ("attention", trunk)):
        assert (second["actor_module_grad_norms"][name] > 0) == enabled
        assert second["actor_module_lr"][name] == ([5e-5] if enabled else [])
    assert second["physical_residual_mps2"]["mean_abs"] > 0
    assert all(p.grad is None for p in t.policy.head.parameters())


def test_legacy_head_adam_migration_and_full_restart(tmp_path):
    legacy = trainer()
    legacy.update()
    path = tmp_path / "legacy.pt"
    legacy.save_checkpoint(path)
    full = trainer(True, True, True)
    full.load_checkpoint(path)
    assert full.config.native_residual_train_group and full.config.native_residual_train_trunk
    for old_p, new_p in zip(legacy.policy.residual_head.parameters(), full.policy.residual_head.parameters()):
        assert_tree_close(legacy.actor_optimizer.state[old_p], full.actor_optimizer.state[new_p])
    assert len(full.actor_optimizer.state) == 2
    row = full.update()
    assert row["actor_module_grad_norms"]["group_encoder"] > 0
    assert row["physical_residual_mps2"]["mean_abs"] > 0
    path = tmp_path / "full.pt"
    full.save_checkpoint(path)
    again = trainer(True, True, True)
    again.load_checkpoint(path)
    assert_tree_close(full.actor_optimizer.state_dict(), again.actor_optimizer.state_dict())
    a = full.update()
    b = again.update()
    assert_tree_close(a["actor_module_grad_norms"], b["actor_module_grad_norms"])
    assert_tree_close(full.policy.state_dict(), again.policy.state_dict())
    assert_tree_close(full.group_encoder.state_dict(), again.group_encoder.state_dict())


def test_flags_are_strict_booleans():
    with pytest.raises(ValueError, match="boolean"):
        replace(DreamerConfig(), native_residual_train_trunk=1)
