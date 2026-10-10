"""New WM-S model vs the legacy implementation: state_dict keys and outputs."""
from __future__ import annotations

import sys

import pytest
import torch

from wms.model import HdvConfig, ModelSpec, WmSModel
from wms.physics import PhysicsConfig

from . import inputs
from .legacy import C1_CHECKPOINT, legacy, load_legacy

SMALL = dict(d_model=32, n_heads=4, n_layers=2, ffn=64)
LONG = dict(history_s=4.0, history_stride_s=1.0)

CASES = {
    "residual": dict(),
    "residual_merge_cal": dict(physics=dict(sumo_fidelity=True, merge_cal=True)),
    "soft_bound": dict(residual_soft_bound_mps2=2.0, residual_soft_bound_neg_mps2=4.0),
    "asym_cap": dict(residual_cap_mps2=1.5, residual_cap_neg_mps2=3.0, lane_logit_cap=2.0),
    "driver_params": dict(rollout_mode="driver_params"),
    "hybrid": dict(rollout_mode="hybrid"),
    "hybrid_lane_head": dict(rollout_mode="hybrid", lane_head="history",
                             physics=dict(sumo_fidelity=True, merge_cal=True)),
    "long_history": dict(rollout_mode="hybrid", lane_head="history", **LONG),
    "black_box": dict(black_box=True),
}


def build(case: str, seed: int = 7):
    kw = dict(CASES[case])
    phys = kw.pop("physics", {})
    leg = legacy("model")
    lm = sys.modules["hdv.config"]
    torch.manual_seed(seed)
    old = leg.WmSModel(physics=leg.PhysicsConfig(**phys), hdv=lm.HdvConfig(**SMALL), **kw)
    torch.manual_seed(seed)
    new = WmSModel(ModelSpec(**kw), physics=PhysicsConfig(**phys), hdv=HdvConfig(**SMALL))
    old.eval()
    new.eval()
    return old, new


def history_kwargs(model, seed=1):
    if model.long_history:
        return dict(driver_history=inputs.make_long_history(seed, model.driver_history_len)[0],
                    driver_history_mask=inputs.make_long_history(seed, model.driver_history_len)[1])
    hist, mask = inputs.make_history(seed)
    return dict(history=hist, history_mask=mask)


def assert_same(a, b, path="", atol=1e-6):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a) ^ set(b)}"
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}", atol)
    elif torch.is_tensor(a):
        assert a.shape == b.shape, path
        if a.dtype.is_floating_point:
            assert torch.allclose(a, b, atol=atol, rtol=0, equal_nan=True), \
                f"{path}: max diff {(a - b).abs().max()}"
        else:
            assert torch.equal(a, b), path
    else:
        assert a == b, path


@pytest.fixture(scope="module", autouse=True)
def _legacy():
    load_legacy()


@pytest.mark.parametrize("case", sorted(CASES))
def test_fresh_init_matches(case):
    old, new = build(case)
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn)
    for key in so:
        assert torch.equal(so[key], sn[key]), key


@pytest.mark.parametrize("case", sorted(CASES))
def test_forward_step_rollout(case):
    old, new = build(case)
    state = inputs.make_state()
    hk = history_kwargs(new)
    with torch.no_grad():
        for k in (0, 5):
            assert_same(old(dict(state), k_steps=k, **hk), new(dict(state), k_steps=k, **hk))
        teacher = inputs.make_teacher(state, inputs.N_STEPS)[:, 1]
        so = old.step(dict(state), k_steps=3, cav_teacher=teacher, **hk)
        sn = new.step(dict(state), k_steps=3, cav_teacher=teacher, **hk)
        assert_same(so, sn)
        teacher = inputs.make_teacher(state, inputs.N_STEPS)
        ro = old.rollout(dict(state), n_steps=inputs.N_STEPS, cav_teacher=teacher, **hk)
        rn = new.rollout(dict(state), n_steps=inputs.N_STEPS, cav_teacher=teacher, **hk)
        assert_same(ro, rn, atol=1e-5)


def test_direct_timeline_and_noise_and_truncation():
    old, new = build("black_box")
    state = inputs.make_state()
    hist, mask = inputs.make_history()
    n = 8
    timeline = {"controller": torch.randint(0, 4, (2, 4, inputs.K)),
                "acceleration_mps2": torch.randn(2, 4, inputs.K),
                "target_speed_mps": torch.rand(2, 4, inputs.K) * 25, "control_dt_s": 1.0}
    kw = dict(n_steps=n, history=hist, history_mask=mask, action_timeline=timeline,
              cav_teacher=inputs.make_teacher(state, n), truncation_steps=3)
    with torch.no_grad():
        assert_same(old.rollout(dict(state), **kw), new.rollout(dict(state), **kw), atol=1e-5)
    old_s, new_s = build("residual")
    for m in (old_s, new_s):
        m.apply_direct_commands = True
    kw = dict(n_steps=6, history=hist, history_mask=mask, action_timeline=timeline,
              cav_teacher=inputs.make_teacher(state, 6), state_noise_std_m=0.1,
              state_noise_std_mps=0.05)
    with torch.no_grad():
        torch.manual_seed(3)
        ro = old_s.rollout(dict(state), **kw)
        torch.manual_seed(3)
        rn = new_s.rollout(dict(state), **kw)
    assert_same(ro, rn, atol=1e-5)


def test_projection_and_entries():
    entries_old, entries_new = legacy("entries"), sys.modules["wms.model.entries"]
    stab = legacy("stability")
    from wms.physics import FeasibilityProjection
    old, new = build("residual")
    state = inputs.make_state(n_present=30)
    counter = torch.rand(2, 30).numpy() * 3
    hist, mask = inputs.make_history()
    n = 10
    outs = []
    for mod, model, proj in ((entries_old, old, stab.FeasibilityProjection()),
                             (entries_new, new, FeasibilityProjection())):
        pool = (~state["present"]).numpy()
        proc = mod.TorchEntryProcess(counter, seed=4, batch=2, pool=pool)
        outs.append(model.rollout({k: v.clone() for k, v in state.items()}, n_steps=n,
                                  history=hist, history_mask=mask, entries=proc,
                                  cav_teacher=inputs.make_teacher(state, n),
                                  projection=proj))
    assert_same(outs[0], outs[1], atol=1e-5)


@pytest.mark.skipif(not C1_CHECKPOINT.is_file(), reason="C1 checkpoint not available")
def test_c1_checkpoint_loads_and_matches():
    payload = torch.load(C1_CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    leg = legacy("model")
    kw = dict(history_features=int(cfg.get("history_features", 6)),
              rollout_mode=cfg.get("rollout_mode", "residual"),
              lane_head=cfg.get("lane_head", "off"), history_s=cfg.get("history_s"),
              history_stride_s=cfg.get("history_stride_s", 1.0))
    old = leg.WmSModel(physics=leg.PhysicsConfig(), **kw)
    new = WmSModel(ModelSpec.from_mapping(kw), physics=PhysicsConfig())
    ro = old.load_state_dict(payload["model"], strict=False)
    rn = new.load_state_dict(payload["model"], strict=False)
    assert sorted(ro.missing_keys) == sorted(rn.missing_keys)
    assert not rn.unexpected_keys
    assert list(old.state_dict()) == list(new.state_dict())
    old.eval()
    new.eval()
    state = inputs.make_state()
    hk = history_kwargs(new)
    teacher = inputs.make_teacher(state, 20)
    with torch.no_grad():
        assert_same(old.rollout(dict(state), n_steps=20, cav_teacher=teacher, **hk),
                    new.rollout(dict(state), n_steps=20, cav_teacher=teacher, **hk), atol=1e-5)
