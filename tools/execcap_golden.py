"""EXECCAP cross-commit golden check: with the switch OFF, imagination is bit-identical to a2d3716.

Run once per source tree and compare the printed digests (EXECCAP_NOTES.md):
    PYTHONPATH=<tree>/src;<tree> python tools/execcap_golden.py
(<tree> = an extracted ``git archive a2d3716 src tests`` and this worktree).  Tiny random WM-S,
CPU, single thread; covers the bridge, the coop-imagine callable, a direct action timeline and a
full DreamerTrainer update (counterfactual rows, entries, lanes).
"""
from __future__ import annotations

import copy
import hashlib
import json

import torch


def digest(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().float().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    torch.set_num_threads(1)
    from tests.unit.test_c4_srcwait_imagination import Starts
    from wms.control.bridge import make_cav_controller
    from wms.control.contract import RewardWeights
    from wms.control.coop_imagine import CallableCavController
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig, DreamerTrainer
    from wms.control.lanes import LaneRules
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
    from wms.control.reward import team_reward
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal
    from wms.model.controllers import DIRECT_ID
    from wms.model.residual_net import HdvConfig
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.manual_seed(11)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32)).eval()
    k = 6
    x = {"s_m": torch.tensor([[-600.0, -560.0, -500.0, -470.0, 30.0, 60.0]]),
         "v_mps": torch.tensor([[15.0, 10.0, 20.0, 25.0, 12.0, 14.0]]), "a_mps2": torch.zeros(1, k),
         "lane": torch.tensor([[1, 1, 0, 0, 2, 1]]), "present": torch.ones(1, k, dtype=torch.bool),
         "is_cav": torch.tensor([[True, False, True, False, True, True]]),
         "length_m": torch.full((1, k), 5.0)}
    out = {}

    def policy(obs, signal):
        return torch.full(obs.cav_mask.shape, 2.6), torch.zeros(obs.cav_mask.shape)
    policy.accel_max = 2.6

    def group(obs, s, field=None):
        return torch.zeros(s.shape[0], 4)

    roll = model.rollout(dict(x), n_steps=10,
                         cav_controller=make_cav_controller(policy, group, safe_accel, accel_max=2.6))
    out["bridge"] = digest(roll["positions"], roll["speeds"], roll["accel"])

    def fn(full):
        return torch.full((1, full["s_m"].shape[1]), 2.0)
    ctrl = CallableCavController([fn], torch.arange(k), k, x["is_cav"].clone(), accel_max=2.6)
    roll = model.rollout(dict(x), n_steps=10, cav_controller=ctrl)
    out["coop_callable"] = digest(roll["positions"], roll["speeds"], roll["accel"])

    step = model.step(dict(x), k_steps=0, action_id=torch.where(x["is_cav"], DIRECT_ID, 0),
                      action_command=torch.tensor([[[2.6, 40.0]] * k]),
                      cav_teacher=torch.zeros(1, k, 4))
    out["direct_timeline"] = digest(step["next_s_m"], step["next_v_mps"], step["accel_mps2"])

    torch.manual_seed(3)
    trainer = DreamerTrainer(
        [copy.deepcopy(model)], CavPolicy(hidden_dim=16), GroupSignal(RsuGroupEncoder(hidden_dim=16)),
        TeamCritic(hidden_dim=16), Starts(True), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(batch_size=2, horizon=4, checkpoint_every=0, validation_every=0,
                             actor_lr=1e-3, critic_lr=1e-3),
        reward_weights=RewardWeights(), lane_rules=LaneRules(),
        counterfactual=CounterfactualConfig(enabled=True, n_individual=2, n_group=2, n_flip=2,
                                            group_max_size=4))
    trainer.update_count = 1500
    torch.manual_seed(7)
    metrics = trainer.update()
    metrics.pop("wall_time_s", None)
    out["trainer_metrics"] = hashlib.sha256(json.dumps(metrics, sort_keys=True, default=str)
                                            .encode()).hexdigest()[:16]
    out["trainer_actor_params"] = digest(*[p for p in trainer.actor_parameters])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
