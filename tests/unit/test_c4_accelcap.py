"""ACCELCAP (ACCELCAP_NOTES.md): the policy's acceleration cap is a stored actor property.

Legacy checkpoints (no stored bounds) decode bit-identically with the 1.5 cap; new actors carry
their own cap (2.6 in configs/c4_policy_final.toml) end to end: actor -> safety -> bridge /
deployment executor -> SUMO direct command bound; MAPPO keeps 1.5 unless configured.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.contract import (ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, LEGACY_ACCEL_MAX_MPS2,
                                  NATIVE_CAV_ACCEL_MAX_MPS2, POLICY_ACCEL_MAX_MPS2)
from wms.control.counterfactual import CounterfactualConfig, idm_native_accel, native_fn
from wms.control.lanes import LaneRules
from wms.control.obs import build_observations
from wms.control.policy import (CavPolicy, PolicyExecutor, RsuGroupEncoder, accel_bounds_payload,
                                draw_policy_noise, load_policy_components, load_policy_fn)
from wms.control.safety import safe_accel, with_accel_max

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state(s=(0.0, 30.0, -20.0, 50.0), v=(10.0, 12.0, 8.0, 15.0), lane=(1, 1, 1, 0),
          cav=(True, False, True, True)):
    k = len(s)
    return {"s_m": torch.tensor([s]), "v_mps": torch.tensor([v]), "a_mps2": torch.zeros(1, k),
            "lane": torch.tensor([lane], dtype=torch.long), "present": torch.ones(1, k, dtype=torch.bool),
            "is_cav": torch.tensor([cav]), "length_m": torch.full((1, k), 5.0)}


def free_road():
    """One CAV alone in the right mainline lane, far upstream of the merge, at 10 m/s."""
    return state(s=(-600.0,), v=(10.0,), lane=(1,), cav=(True,))


def legacy_state_dict(policy):
    return {k: v.clone() for k, v in policy.state_dict().items() if k != "accel_bounds"}


def full_throttle(policy):
    with torch.no_grad():
        policy.head[-1].weight.zero_()
        policy.head[-1].bias.copy_(torch.tensor([8.0, -5.0]))   # tanh(8) ~ 1: the cap
    return policy


# ------------------------------------------------------------------ legacy decoding
def test_legacy_state_dict_decodes_bit_identically_with_the_15_cap():
    torch.manual_seed(3)
    source = CavPolicy()
    legacy = legacy_state_dict(source)
    loaded = CavPolicy()                       # fresh default is 2.6 ...
    assert loaded.accel_max == POLICY_ACCEL_MAX_MPS2
    loaded.load_state_dict(legacy)             # ... but a state dict without bounds is legacy
    assert (loaded.accel_min, loaded.accel_max) == (ACCEL_MIN_MPS2, LEGACY_ACCEL_MAX_MPS2) == (-3.0, 1.5)
    explicit = CavPolicy(accel_max=1.5)
    explicit.load_state_dict(legacy)
    x = state()
    obs = build_observations(x)
    torch.manual_seed(5)
    group = RsuGroupEncoder()(obs, x["s_m"]).detach()
    noise = draw_policy_noise((1, 4), generator=torch.Generator().manual_seed(1))
    mask = torch.ones(1, 4, 3, dtype=torch.bool)
    heads = {}
    handle = loaded.head.register_forward_hook(lambda m, i, o: heads.__setitem__("head", o))
    with torch.no_grad():
        for deterministic in (True, False):
            out = loaded(obs, group, deterministic=deterministic, lane_mask=mask, noise=noise)
            ref = explicit(obs, group, deterministic=deterministic, lane_mask=mask, noise=noise)
            for a, b in zip(out, ref):
                assert torch.equal(a, b)
            mean, log_std = heads["head"].unbind(-1)
            latent = mean if deterministic else mean + log_std.clamp(-5.0, 1.0).exp() * noise.accel
            # The pre-ACCELCAP formula with the module constants of the time (-3, 1.5).
            legacy_accel = (ACCEL_MAX_MPS2 + ACCEL_MIN_MPS2) / 2.0 + (
                (ACCEL_MAX_MPS2 - ACCEL_MIN_MPS2) / 2.0) * torch.tanh(latent)
            legacy_accel = torch.where(obs.cav_mask, legacy_accel, torch.zeros_like(legacy_accel))
            assert torch.equal(out.accel, legacy_accel)
    handle.remove()


def test_legacy_checkpoint_file_loads_with_15_through_deploy_loader(tmp_path):
    torch.manual_seed(4)
    policy, encoder = CavPolicy(), RsuGroupEncoder()
    path = tmp_path / "legacy.pt"
    torch.save({"format": "c4_dreamer/3", "policy": legacy_state_dict(policy),
                "group_encoder": encoder.state_dict(), "lane_rules": None}, path)
    loaded, enc, _, _ = load_policy_components(path)
    assert loaded.accel_max == 1.5
    exe = load_policy_fn(path)
    assert exe.accel_max == 1.5
    reference = CavPolicy(accel_max=1.5)
    reference.load_state_dict(legacy_state_dict(policy))
    ref_exe = PolicyExecutor(reference.eval(), enc, lane_rules=LaneRules())
    x = state()
    a, lanes = exe(x, with_lanes=True)
    b, lanes_b = ref_exe(x, with_lanes=True)
    assert ((a == b) | ((a != a) & (b != b))).all() and (lanes == lanes_b).all()


def test_new_checkpoint_roundtrip_keeps_its_cap(tmp_path):
    policy, encoder = CavPolicy(accel_max=2.6), RsuGroupEncoder()
    path = tmp_path / "new.pt"
    torch.save({"format": "c4_dreamer/3", "policy": policy.state_dict(),
                "group_encoder": encoder.state_dict(), "lane_rules": None,
                **accel_bounds_payload(policy)}, path)
    loaded, _, _, payload = load_policy_components(path)
    assert loaded.accel_max == 2.6 and payload["accel_max_mps2"] == 2.6
    assert load_policy_fn(path).accel_max == 2.6
    # Payload fields that disagree with the stored buffer are refused.
    torch.save({"format": "c4_dreamer/3", "policy": policy.state_dict(),
                "group_encoder": encoder.state_dict(), "accel_max_mps2": 1.5}, path)
    with pytest.raises(ValueError, match="disagree"):
        load_policy_components(path)
    # A float32 cast of the module must not perturb the stored 2.6.
    other = CavPolicy(accel_max=1.5)
    other.load_state_dict(CavPolicy(accel_max=2.6).float().state_dict())
    assert other.accel_max == 2.6


# ------------------------------------------------------------------ end to end > 1.5
def test_new_policy_commands_above_15_through_bridge_safety_and_executor_on_free_road():
    x = free_road()
    new = full_throttle(CavPolicy(accel_max=2.6)).eval()
    old = full_throttle(CavPolicy(accel_max=1.5)).eval()
    encoder = RsuGroupEncoder().eval()
    with torch.no_grad():
        # Training/imagination bridge (policy -> safety -> direct_command_motion).
        applied_new = make_cav_controller(new, encoder, safe_accel)({**x, "physics_step_index": 0})[0, 0, 3]
        applied_old = make_cav_controller(old, encoder, safe_accel)({**x, "physics_step_index": 0})[0, 0, 3]
        # Lane-head bridge path too.
        lane_new = make_cav_controller(new, encoder, safe_accel, lane_rules=LaneRules())(
            {**x, "physics_step_index": 0})[0, 0, 3]
    assert float(applied_new) > 2.5 and float(lane_new) > 2.5
    assert float(applied_old) <= 1.5 + 1e-6
    # Deployment executor (SUMO path): the command handed to CavSumoEnv.step.
    cmd_new = PolicyExecutor(new, encoder, lane_rules=LaneRules())(x)
    cmd_old = PolicyExecutor(old, encoder, lane_rules=LaneRules())(x)
    assert cmd_new[0, 0] > 2.5 and cmd_old[0, 0] <= 1.5 + 1e-6
    # The safety layer's default stays the legacy ceiling; the policy bound is passed explicitly.
    proposal = torch.tensor([[2.6]])
    assert float(safe_accel(x, proposal)) == 1.5
    assert float(safe_accel(x, proposal, accel_max=2.6)) == pytest.approx(2.6)
    # Custom safety layers without the keyword keep working.
    assert torch.equal(with_accel_max(lambda s, a: a)(x, proposal, accel_max=9.0), proposal)


def test_sumo_direct_target_honours_the_direct_ceiling_only_when_given():
    from traffic_shepherd.damping import direct_target
    config = SimpleNamespace(control_dt_s=1.0, acceleration_min_mps2=-3.0, acceleration_max_mps2=1.5)
    frame = SimpleNamespace(speed_mps=10.0, allowed_speed_mps=31.29)
    assert direct_target(frame, 2.6, config) == (1.5, 11.5)                 # damping/wall/PI: adopted 1.5
    assert direct_target(frame, 2.6, config, 3.0) == pytest.approx((2.6, 12.6))   # C4 direct mode
    assert direct_target(frame, 1.2, config, 3.0) == direct_target(frame, 1.2, config)


def test_counterfactual_native_rows_use_sumo_cav_accel_not_the_policy_cap():
    x = free_road()
    assert CounterfactualConfig().native_accel_max_mps2 == NATIVE_CAV_ACCEL_MAX_MPS2 == 2.6
    native = idm_native_accel(x)
    assert float(native) > 1.5 and float(native) <= 2.6
    assert float(native_fn("idm", a_max=1.5)(x)) <= 1.5
    # Bridge: a legacy (1.5) policy row whose CAV is native gets the native ceiling, not 1.5.
    old = full_throttle(CavPolicy(accel_max=1.5)).eval()
    controller = make_cav_controller(old, RsuGroupEncoder().eval(), safe_accel,
                                     act_mask=torch.zeros(1, 1, dtype=torch.bool),
                                     native=native_fn("idm"))
    with torch.no_grad():
        applied = controller({**x, "physics_step_index": 0})[0, 0, 3]
    assert float(applied) == pytest.approx(float(native)) and float(applied) > 1.5


# ------------------------------------------------------------------ config plumbing
def test_config_plumbing_policy_section_and_final_config():
    from wms.control.train_cli import _read_config, policy_settings
    assert policy_settings({}) == {"accel_max": POLICY_ACCEL_MAX_MPS2}
    assert policy_settings({"policy": {"accel_max_mps2": 1.5}}) == {"accel_max": 1.5}
    with pytest.raises(ValueError):
        policy_settings({"policy": {"accel_max_mps2": 5.0}})
    with pytest.raises(ValueError):
        policy_settings({"policy": {"accel_max": 2.6}})
    raw = _read_config(REPO / "configs" / "c4_policy_final.toml")
    assert raw["policy"]["accel_max_mps2"] == 2.6
    assert policy_settings(raw) == {"accel_max": 2.6}
    assert CounterfactualConfig(**raw["counterfactual"]).native_accel_max_mps2 == 2.6


# ------------------------------------------------------------------ MAPPO
def test_mappo_legacy_default_legacy_load_and_configurable_cap(tmp_path):
    from wms.control.mappo.agent import (ActorStep, build_networks, evaluate_actions,
                                         export_payload, latent_log_prob)
    from wms.control.mappo.config_helpers import build_trainer
    from wms.control.mappo.fakeenv import FakeCavEnv

    policy, encoder, _ = build_networks()
    assert policy.accel_max == 1.5                             # unchanged MAPPO default
    # A running MAPPO run's checkpoint (no stored bound) loads with 1.5 into any network.
    torch.manual_seed(2)
    old_policy, old_encoder, _ = build_networks()
    target, _, _ = build_networks(accel_max=2.6)
    target.load_state_dict(legacy_state_dict(old_policy))
    assert target.accel_max == 1.5
    # Exports carry the bound; load_policy_fn decodes each export with its own bound.
    new_policy, _, _ = build_networks(accel_max=2.6)
    for pol, cap in ((old_policy, 1.5), (new_policy, 2.6)):
        path = tmp_path / f"export_{cap}.pt"
        torch.save(export_payload(pol, old_encoder, LaneRules()), path)
        assert load_policy_fn(path).accel_max == cap
    # Log-probabilities use the actor's own tanh scale (PPO ratio consistency at 2.6).
    env = FakeCavEnv()
    x = env.reset(seed=4110, demand_vph=2600, penetration=0.4, warmup_s=0.0, horizon_s=4.0)
    step = ActorStep(new_policy, old_encoder, LaneRules())
    out = step(x, deterministic=False, generator=torch.Generator().manual_seed(0))
    ev = evaluate_actions(new_policy, old_encoder, out["state"], out["lane_mask"], out["latent"],
                          out["lane_choice"])
    index, cav = ev["index"], ev["cav"]
    torch.testing.assert_close(ev["logp_accel"][cav], out["logp_accel"].gather(1, index)[cav])
    assert latent_log_prob(torch.zeros(1), torch.zeros(1), torch.zeros(1)).item() == pytest.approx(
        -0.5 * math.log(2 * math.pi) - math.log(2.25))
    # Config: legacy 1.5 when absent, explicit [policy] accel_max_mps2 otherwise.
    assert build_trainer({}, workers=1, out_dir=tmp_path / "a").policy.accel_max == 1.5
    assert build_trainer({"policy": {"accel_max_mps2": 2.6}}, workers=1,
                         out_dir=tmp_path / "b").policy.accel_max == 2.6
