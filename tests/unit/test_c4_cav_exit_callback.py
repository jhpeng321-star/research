"""CAV callback contract at the road end: one invocation per physics tick, even with no CAV.

V10EXIT lets CAVs leave at the road end, so an imagination/eval rollout can reach steps whose
state has an all-false CAV mask.  The controller owns episode-persistent lifecycle state, so
the WM must still call it exactly once per physics tick (see WmSModel.step).
"""
from __future__ import annotations

import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import RewardWeights
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.lanes import LaneRules
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.start_states import StartBatch
from wms.control.train_cli import GroupSignal
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig


def _model() -> WmSModel:
    return WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                    hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32),
                    physics=PhysicsConfig(exit_at_road_end=True)).eval()


def _state(exit_s: float) -> dict[str, torch.Tensor]:
    k = 4
    return {
        "s_m": torch.tensor([[100.0, 90.0, 80.0, 0.0]]),
        "v_mps": torch.tensor([[20.0, 18.0, 16.0, 0.0]]),
        "a_mps2": torch.zeros(1, k),
        "lane": torch.tensor([[1, 1, 0, -1]]),
        "present": torch.tensor([[True, True, True, False]]),
        "is_cav": torch.tensor([[True, True, False, False]]),
        "length_m": torch.full((1, k), 5.0),
        "exit_s_m": torch.full((1, k), float(exit_s)),
    }


class _Starts:
    def __init__(self, state: dict[str, torch.Tensor]):
        self.state = state

    def sample(self, batch_size, device="cpu"):
        return StartBatch({key: value.clone() for key, value in self.state.items()}, {})


def test_rollout_calls_the_cav_controller_once_per_tick_after_every_cav_exits():
    model = _model()
    calls = []

    def controller(state):
        calls.append(int(state["physics_step_index"]))
        s, v = state["s_m"], state["v_mps"]
        return torch.stack((s + 0.5 * v, state["lane"].to(s.dtype), v,
                            torch.zeros_like(v)), -1)

    state = _state(105.0)
    with torch.no_grad():
        out = model.rollout(state, n_steps=6, cav_controller=controller)

    assert calls == list(range(6))
    cav_present = out["masks"][0, -1] & state["is_cav"][0]
    assert not bool(cav_present.any()), "all CAVs should have left at the exit bound"


def test_imagine_does_not_require_a_present_cav_at_every_step():
    torch.manual_seed(5)
    state = _state(105.0)
    trainer = DreamerTrainer(
        [_model()], CavPolicy(hidden_dim=16), GroupSignal(RsuGroupEncoder(hidden_dim=16)),
        TeamCritic(hidden_dim=16), _Starts(state), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=DreamerConfig(batch_size=1, horizon=3, checkpoint_every=0,
                             validation_every=0, checkpoint_imagination=False),
        reward_weights=RewardWeights(), lane_rules=LaneRules())

    imagined = trainer.imagine(_Starts(state).sample(1), deterministic=True)
    assert imagined["rewards"].shape == (1, 3, 1)
    assert torch.isfinite(imagined["rewards"]).all()
