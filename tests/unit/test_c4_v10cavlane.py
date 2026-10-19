"""CAVLANE: CAV lane execution parity + policy lane-action switch (CPU only).

Guarantees:
1. ``trainer.policy_lane_actions`` defaults to True (the previous behaviour) and is validated.
2. False removes the policy lane path from the trainer even when lane rules and a lane head are
   present; True keeps it.
3. The shipped v10entzone policy chain resolves to ``[policy] lane_control = "native"``: it has
   no policy lane head, so CAVs cannot issue discretionary lane changes in imagination (only the
   bridge's forced ramp merge).  The switch gives the loop an explicit off independent of that.
4. An executed lane change is safe in BOTH the lane being left and the target lane, matching
   ``PolicyExecutor`` in SUMO deployment: the target lane must not unlock an acceleration that
   the old-lane leader forbids (the source of the test-D over-credit).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import RewardWeights
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.lanes import LaneRules
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, PolicyOutput, RsuGroupEncoder, TeamCritic
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.start_states import StartBatch
from wms.control.train_cli import GroupSignal, _read_config, policy_settings
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel

ROOT = Path(__file__).resolve().parents[2]


class Starts:
    def __init__(self, batch):
        self.batch = batch

    def sample(self, batch_size, device="cpu"):
        return StartBatch({k: v.clone() for k, v in self.batch.state.items()}, {})


def _batch():
    k = 4
    state = {"s_m": torch.tensor([[-600.0, -400.0, -200.0, -100.0]]),
             "v_mps": torch.tensor([[20.0, 22.0, 24.0, 26.0]]),
             "a_mps2": torch.zeros(1, k),
             "lane": torch.tensor([[1, 1, 0, 0]]),
             "present": torch.ones(1, k, dtype=torch.bool),
             "is_cav": torch.tensor([[True, True, False, False]]),
             "length_m": torch.full((1, k), 5.0)}
    return StartBatch({key: value for key, value in state.items()}, {})


def _trainer(policy_lane_actions, lane_head=True):
    torch.manual_seed(5)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32)).eval()
    cfg = DreamerConfig(batch_size=1, horizon=2, checkpoint_every=0, validation_every=0,
                        checkpoint_imagination=False,
                        policy_lane_actions=policy_lane_actions)
    return DreamerTrainer(
        [model], CavPolicy(hidden_dim=16, lane_head=lane_head),
        GroupSignal(RsuGroupEncoder(hidden_dim=16)), TeamCritic(hidden_dim=16), Starts(_batch()),
        observation_builder=build_observations, reward_fn=team_reward, safety=safe_accel,
        controller_factory=make_cav_controller, config=cfg, reward_weights=RewardWeights(),
        lane_rules=LaneRules())


def test_policy_lane_actions_defaults_true_and_validates():
    assert DreamerConfig().policy_lane_actions is True
    with pytest.raises(ValueError):
        DreamerConfig(policy_lane_actions=1)


def test_policy_lane_actions_switch_gates_the_lane_path():
    on = _trainer(True)
    off = _trainer(False)
    assert on.use_lanes is True
    assert off.use_lanes is False
    # No lane head: already inactive, the switch changes nothing.
    assert _trainer(True, lane_head=False).use_lanes is False


class _LaneLeft(torch.nn.Module):
    """Scripted actor: full acceleration plus LANE_LEFT for slot 0 (test D command)."""

    has_lane_head = True
    supports_noise = False
    accel_max = 1.2

    def forward(self, obs, signal, deterministic=False, lane_mask=None, noise=None, **_):
        accel = torch.full_like(obs.cav_mask, 2.0, dtype=torch.float32)
        choice = torch.zeros_like(obs.cav_mask, dtype=torch.long)
        choice[:, 0] = 1
        if lane_mask is None:
            return accel, torch.zeros_like(accel)
        zero = torch.zeros_like(accel)
        return PolicyOutput(accel, choice, zero, zero, zero, zero, torch.ones_like(choice))


def _permissive_rules():
    return LaneRules(gap_base_m=0.0, gap_time_s=0.0, closing_time_s=0.0,
                     rear_gap_base_m=0.0, rear_gap_time_s=0.0, rear_closing_time_s=0.0,
                     edge_guard_m=0.0, dwell_s=0.0)


def test_lane_change_is_safe_in_both_lanes():
    """A change must not unlock acceleration the lane being left forbids (deployment parity)."""
    state = {"s_m": torch.tensor([[-300.0, -288.0, -100.0]]),
             "v_mps": torch.tensor([[20.0, 0.0, 30.0]]),
             "a_mps2": torch.zeros(1, 3),
             "lane": torch.tensor([[1, 1, 0]]),
             "present": torch.ones(1, 3, dtype=torch.bool),
             "is_cav": torch.tensor([[True, False, False]]),
             "length_m": torch.full((1, 3), 5.0)}
    controller = make_cav_controller(
        _LaneLeft(), lambda obs, s, field: torch.zeros(obs.cav_mask.shape + (8,)),
        safe_accel, lane_rules=_permissive_rules())
    out = controller(state)[0]
    assert int(out[0, 1]) == 0                      # lane change executed
    assert float(out[0, 3]) <= -1.0                 # old-lane leader (7 m, stopped) binds
    assert float(out[0, 2]) < 20.0                  # no free-lane acceleration credit
    assert float(out[1, 3]) == 0.0                  # HDV slots are untouched


def test_v10entzone_policy_cannot_issue_lane_actions_in_imagination():
    raw = _read_config(ROOT / "configs" / "c4_policy_v10entzone_v5m.toml")
    settings = policy_settings(raw)
    assert settings.get("lane_control") == "native"
    # The actor built by the launcher therefore has no lane head; with lane rules supplied the
    # existing trainer condition would still be false.  The switch default preserves that.
    policy = CavPolicy(lane_head=False)
    policy.lane_control = settings["lane_control"]
    assert policy.has_lane_head is False
    assert raw["trainer"].get("policy_lane_actions") is None      # default True in DreamerConfig
    assert DreamerConfig(**raw["trainer"]).policy_lane_actions is True
