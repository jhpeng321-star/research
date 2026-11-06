"""V10ENTZONE: entrance-zone native driving in WM imagination (CPU only).

Guarantees:
1. ``entrance_zone_mask`` selects CAVs inside the entrance zones (same insertion points as the
   SUMO autocheck) and nothing else.
2. The bridge releases a zoned CAV to the native/IDM twin command while the other CAVs still
   follow the policy.
3. Inside Dreamer imagination the zone CAVs are excluded from the actor loss (``action_mask``)
   and counted in ``entrance_zone_fraction``; with the flag off nothing changes.
4. The shipped policy config resolves and carries exactly the entrance-zone switch.

No GPU, SUMO or checkpoint is touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import RewardWeights
from wms.control.counterfactual import CounterfactualConfig, build_variants, native_fn
from wms.control.dreamer import DreamerConfig, DreamerTrainer
from wms.control.entrance import (MAINLINE_ENTRY_S, RAMP_ENTRY_S, DEFAULT_ENTRANCE_ZONE_M,
                                  entrance_zone_cavs, entrance_zone_mask, make_entrance_zone_fn)
from wms.control.obs import build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.reward import team_reward
from wms.control.safety import safe_accel
from wms.control.start_states import StartBatch
from wms.control.train_cli import GroupSignal
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------------ 1. zone mask
def test_entrance_zone_mask_boundaries_and_defaults():
    lane = torch.tensor([0, 0, 0, 1, 1, 2, 2, -1])
    s = torch.tensor([MAINLINE_ENTRY_S[0], MAINLINE_ENTRY_S[0] + DEFAULT_ENTRANCE_ZONE_M,
                      MAINLINE_ENTRY_S[0] + DEFAULT_ENTRANCE_ZONE_M + 0.01,
                      MAINLINE_ENTRY_S[1] + 10.0, MAINLINE_ENTRY_S[1] - 0.01,
                      RAMP_ENTRY_S + 24.0, RAMP_ENTRY_S + 26.0, RAMP_ENTRY_S + 5.0])
    expected = torch.tensor([True, True, False, True, False, True, False, False])
    assert torch.equal(entrance_zone_mask(lane, s), expected)


def test_entrance_zone_mask_configurable_lengths():
    lane = torch.tensor([0, 2])
    s = torch.tensor([MAINLINE_ENTRY_S[0] + 40.0, RAMP_ENTRY_S + 40.0])
    assert not entrance_zone_mask(lane, s).any()                    # 25 m default
    assert torch.equal(entrance_zone_mask(lane, s, main_m=50.0, ramp_m=50.0),
                       torch.tensor([True, True]))


def test_entrance_zone_cavs_requires_present_cav_and_valid_lane():
    state = {"s_m": torch.tensor([[MAINLINE_ENTRY_S[1] + 5.0, MAINLINE_ENTRY_S[1] + 5.0,
                                   MAINLINE_ENTRY_S[1] + 5.0]]),
             "lane": torch.tensor([[1, 1, -1]]),
             "present": torch.tensor([[True, True, True]]),
             "is_cav": torch.tensor([[True, False, True]])}
    assert torch.equal(entrance_zone_cavs(state), torch.tensor([[True, False, False]]))


# ------------------------------------------------------------------ 2. config
def test_config_defaults_off_and_validation():
    cfg = DreamerConfig()
    assert cfg.entrance_zone_native is False
    assert cfg.entrance_zone_main_m == DEFAULT_ENTRANCE_ZONE_M == 25.0
    assert cfg.entrance_zone_ramp_m == 25.0
    with pytest.raises(ValueError):
        DreamerConfig(entrance_zone_native=1)
    with pytest.raises(ValueError):
        DreamerConfig(entrance_zone_native=True, entrance_zone_main_m=0.0)
    with pytest.raises(ValueError):
        DreamerConfig(entrance_zone_native=True, entrance_zone_ramp_m=-1.0)


def test_policy_config_loads_and_resolves():
    from wms.control.train_cli import _read_config
    root = Path(__file__).resolve().parents[2]
    raw = _read_config(root / "configs" / "c4_policy_v10entzone_v5m.toml")
    trainer = raw["trainer"]
    assert trainer["entrance_zone_native"] is True
    assert trainer["entrance_zone_main_m"] == 25.0 and trainer["entrance_zone_ramp_m"] == 25.0
    # identical to v9noyield except the entrance switch: the WM rule-off is retained.
    assert trainer["wm_physics_overrides"] == {"mainline_merge_yield": False}
    assert trainer["entry_queue_state"] is True
    assert raw["data"]["wm_config"] == "configs/c4_wm_v5m_refresh_q.toml"
    cfg = DreamerConfig(**trainer)
    assert cfg.entrance_zone_native is True and cfg.entrance_zone_main_m == 25.0


def test_v10_config_differs_from_v9noyield_by_entrance_keys_only():
    from wms.control.train_cli import _read_config
    root = Path(__file__).resolve().parents[2]
    v9 = _read_config(root / "configs" / "c4_policy_v9noyield_v5m.toml")
    v10 = _read_config(root / "configs" / "c4_policy_v10entzone_v5m.toml")

    def flatten(prefix, value):
        out = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                out.update(flatten(name, item))
            else:
                out[name] = item
        return out

    flat9, flat10 = flatten("", v9), flatten("", v10)
    diffs = {key: (flat9.get(key), flat10.get(key))
             for key in set(flat9) | set(flat10) if flat9.get(key) != flat10.get(key)}
    assert set(diffs) == {"run.output_dir", "trainer.entrance_zone_native",
                          "trainer.entrance_zone_main_m", "trainer.entrance_zone_ramp_m"}


# ------------------------------------------------------------------ 3. bridge native twin
class _ConstPolicy(torch.nn.Module):
    """Stub actor: a fixed acceleration on every CAV, with the bridge-facing bound."""

    def __init__(self):
        super().__init__()
        self.accel_max = 2.6

    def forward(self, obs, signal, *, deterministic=False):
        accel = torch.full_like(obs.cav_mask.to(torch.float32), 1.0)
        return accel, torch.zeros_like(accel)


def _cross_state():
    return {"s_m": torch.tensor([[-2644.22 + 5.0, -500.0, -2645.67 + 3.0]]),
            "v_mps": torch.tensor([[5.0, 20.0, 8.0]]),
            "a_mps2": torch.zeros(1, 3), "lane": torch.tensor([[1, 1, 0]]),
            "present": torch.ones(1, 3, dtype=torch.bool),
            "is_cav": torch.tensor([[True, True, False]]),
            "length_m": torch.full((1, 3), 5.0)}


def test_bridge_zone_releases_zoned_cav_to_native_only():
    state = _cross_state()
    policy = _ConstPolicy()
    encoder = RsuGroupEncoder(hidden_dim=8)
    native = native_fn("idm", a_max=1.2)
    zone = make_entrance_zone_fn()
    assert bool(zone(state)[0, 0]) and not bool(zone(state)[0, 1])   # CAV 0 in, CAV 1 out

    plain = make_cav_controller(policy, encoder, safe_accel, accel_max=2.6, native_accel_max=1.2)
    all_native = make_cav_controller(policy, encoder, safe_accel,
                                     act_mask=torch.zeros(1, 3, dtype=torch.bool),
                                     native=native, accel_max=2.6, native_accel_max=1.2)
    zoned = make_cav_controller(policy, encoder, safe_accel, zone=zone, native=native,
                                accel_max=2.6, native_accel_max=1.2)

    plain_cmd = plain(state)[..., 3]
    native_cmd = all_native(state)[..., 3]
    zone_cmd = zoned(state)[..., 3]
    torch.testing.assert_close(zone_cmd[:, 0], native_cmd[:, 0])    # zoned CAV follows native twin
    torch.testing.assert_close(zone_cmd[:, 1], plain_cmd[:, 1])     # outside CAV still policy
    assert not torch.equal(plain_cmd[:, 0], native_cmd[:, 0])       # the two behaviours differ


def test_bridge_zone_requires_native_behaviour():
    with pytest.raises(ValueError):
        make_cav_controller(_ConstPolicy(), RsuGroupEncoder(hidden_dim=8), safe_accel,
                            zone=make_entrance_zone_fn())


# ------------------------------------------------------------------ 4. imagination exclusion
def _batch():
    s = [-2644.22 + 5.0, -500.0, -2645.67 + 3.0, RAMP_ENTRY_S + 35.0, RAMP_ENTRY_S + 10.0, -600.0]
    lane = [1, 1, 0, 2, 2, 1]
    v = [5.0, 20.0, 8.0, 18.0, 4.0, 22.0]
    cav = [True, True, False, True, True, False]
    k = len(s)
    state = {"s_m": torch.tensor([s]), "v_mps": torch.tensor([v]), "a_mps2": torch.zeros(1, k),
             "lane": torch.tensor([lane]), "present": torch.ones(1, k, dtype=torch.bool),
             "is_cav": torch.tensor([cav]), "length_m": torch.full((1, k), 5.0)}
    return StartBatch({key: value.expand(2, -1).clone() for key, value in state.items()}, {})


def _world_model():
    torch.manual_seed(11)
    return WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                    hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32)).eval()


def _trainer(entrance_zone_native):
    torch.manual_seed(3)
    cfg = DreamerConfig(batch_size=2, horizon=3, checkpoint_every=0, validation_every=0,
                        actor_lr=1e-3, critic_lr=1e-3, checkpoint_imagination=False,
                        entrance_zone_native=entrance_zone_native)
    return DreamerTrainer(
        [_world_model()], CavPolicy(hidden_dim=16, lane_head=False),
        GroupSignal(RsuGroupEncoder(hidden_dim=16)), TeamCritic(hidden_dim=16), Starts(_batch()),
        observation_builder=build_observations, reward_fn=team_reward, safety=safe_accel,
        controller_factory=make_cav_controller, config=cfg, reward_weights=RewardWeights(),
        counterfactual=CounterfactualConfig(enabled=True, n_individual=1, n_group=1, n_flip=1,
                                            group_max_size=3, max_starts=2))


class Starts:
    def __init__(self, batch):
        self.batch = batch

    def sample(self, batch_size, device="cpu"):
        return StartBatch({key: value.clone() for key, value in self.batch.state.items()}, {})


def test_imagination_excludes_entrance_zone_cavs_from_actor_loss():
    batch_state = Starts(_batch()).sample(2).state
    expected_zone = entrance_zone_cavs(batch_state)             # [B, K] at t0
    assert int(expected_zone.sum()) == 2 * 2                    # CAVs 0 and 4 in the zones, per row

    off = _trainer(False)
    on = _trainer(True)
    torch.manual_seed(7)
    variants_off = build_variants(batch_state, off.counterfactual)
    imagined_off = off.imagine(Starts(_batch()).sample(2), variants=variants_off)
    torch.manual_seed(7)
    variants_on = build_variants(batch_state, on.counterfactual)
    imagined_on = on.imagine(Starts(_batch()).sample(2), variants=variants_on)

    # flag off: bit-identical default; zone CAVs still act on the all-CAV rows.
    assert imagined_off["entrance_zone"] is None
    assert imagined_off["entrance_zone_fraction"] is None
    for slot in range(expected_zone.shape[1]):
        if expected_zone[0, slot]:
            assert bool(imagined_off["action_mask"][0, 0, 0, slot])

    # flag on: the zone CAVs are excluded from action_mask and counted in the fraction.
    assert imagined_on["entrance_zone"] is not None
    mask_on = imagined_on["action_mask"][0, 0, 0]               # member 0, step 0, all-CAV row 0
    for slot in range(expected_zone.shape[1]):
        if expected_zone[0, slot]:
            assert not bool(mask_on[slot]), slot
            assert bool(imagined_on["entrance_zone"][0, 0, 0, slot])
        elif batch_state["is_cav"][0, slot]:
            assert bool(mask_on[slot]), slot
    assert 0.0 < imagined_on["entrance_zone_fraction"] < 1.0
    assert imagined_on["entrance_zone"][0, 0, 0].sum() == 2
