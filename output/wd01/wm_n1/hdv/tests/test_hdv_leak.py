"""P2 gate G3 (label leak): the HDV net never sees actions/strategy/CAV flag.

The adversarial test builds a full featurizer feature dict (the v1 16-column
``veh_step`` plus Idea A channels), extracts the kinematics-only view and then
perturbs every label-like channel.  The HDV output must be bit-identical.
"""
from __future__ import annotations

import inspect

import torch

from features.idea_a import KIN_SOURCE_INDICES, kinematics_view
from hdv import KIN_F, HdvConfig, HdvResidualNet, input_schema
from synth_hdv import random_kin


def _full_veh_step(batch=2, slots=6, seed=0):
    """A 16-column v1 ``veh_step`` [B,K,16] with labels populated."""
    g = torch.Generator().manual_seed(int(seed))
    veh = torch.zeros(batch, slots, 16)
    veh[..., 0] = torch.rand(batch, slots, generator=g) * 300.0
    lane = torch.randint(0, 3, (batch, slots), generator=g)
    for li in range(3):
        veh[..., 1 + li] = (lane == li).to(veh.dtype)
    veh[..., 4] = torch.rand(batch, slots, generator=g) * 30.0
    veh[..., 5] = (torch.rand(batch, slots, generator=g) - 0.5) * 4.0
    veh[..., 6] = 5.0
    veh[..., 7] = (torch.rand(batch, slots, generator=g) > 0.5).to(veh.dtype)
    veh[..., 8] = (torch.rand(batch, slots, generator=g) > 0.5).to(veh.dtype)
    veh[..., 9] = (torch.rand(batch, slots, generator=g) > 0.5).to(veh.dtype)
    veh[..., 10] = (torch.rand(batch, slots, generator=g) > 0.5).to(veh.dtype)
    veh[..., 11] = (torch.rand(batch, slots, generator=g) > 0.5).to(veh.dtype)
    veh[..., 12] = (torch.rand(batch, slots, generator=g) > 0.5).to(veh.dtype)
    return veh


def test_input_schema_headline_declares_label_free_surface():
    s = input_schema()
    assert s["kin_f"] == KIN_F == len(KIN_SOURCE_INDICES)
    assert "is_cav" not in s["kin_fields"]
    assert not any(f.startswith("action") for f in s["kin_fields"])
    assert s["optional_inputs"] == ()
    assert s["in_features"] == KIN_F
    for ch in ("action_code", "strategy", "is_cav", "cav_flag", "hdv_base",
               "cav_plan_ch", "outcome"):
        assert ch in s["forbidden_inputs"]
    # ablation moves exactly cav_flag out of the forbidden set.
    s_abl = input_schema(HdvConfig(use_cav_flag=True))
    assert s_abl["optional_inputs"] == ("cav_flag",)
    assert "cav_flag" not in s_abl["forbidden_inputs"]
    assert s_abl["in_features"] == KIN_F + 1


def test_forward_signature_accepts_no_label_arguments():
    params = list(inspect.signature(HdvResidualNet.forward).parameters)
    assert params[:3] == ["self", "kin", "kin_mask"]
    assert set(params[3:]) <= {"cav_flag", "conditioning"}


def test_perturbing_labels_flags_strategy_leaves_output_identical():
    net = HdvResidualNet(HdvConfig(use_cav_flag=False))
    net.eval()
    veh = _full_veh_step()
    kin = kinematics_view(veh)
    mask = torch.ones(veh.shape[0], veh.shape[1], dtype=torch.bool)
    out0 = net(kin, mask)

    perturbed = veh.clone()
    perturbed[..., 7] = 1.0 - perturbed[..., 7]              # CAV flag
    perturbed[..., 8:13] = 1.0 - perturbed[..., 8:13]        # action one-hots
    kin2 = kinematics_view(perturbed)
    assert torch.equal(kin, kin2)
    out1 = net(kin2, mask)
    for key in ("accel_delta_mps2", "lane_logit_delta", "accel_log_sigma"):
        assert torch.equal(out0[key], out1[key])

    # A "full feature dict" with strategy/action/flag channels changes nothing
    # because only veh_kin is consumed.
    full_a = {"veh_kin": kin, "is_cav": torch.zeros_like(mask),
              "action_code": torch.zeros(veh.shape[0], veh.shape[1],
                                         dtype=torch.long),
              "strategy": "native", "hdv_base": torch.randn_like(kin[..., :1])}
    full_b = {"veh_kin": kin, "is_cav": torch.ones_like(mask),
              "action_code": torch.ones(veh.shape[0], veh.shape[1],
                                        dtype=torch.long),
              "strategy": "wall", "hdv_base": torch.randn_like(kin[..., :1])}
    for key in ("accel_delta_mps2", "lane_logit_delta"):
        assert torch.equal(net(full_a["veh_kin"], mask)[key],
                           net(full_b["veh_kin"], mask)[key])


def test_full_v1v2_dict_every_nonkinematic_channel_perturbed():
    """Acceptance G3 (stronger): independently perturb EVERY non-kinematic
    channel of a full v1+v2 feature dict; the HDV output must be identical.

    v1 ``veh_step`` columns 7-15 (is_cav, five action one-hots, wall_member,
    junction, present) plus the v2 channels (``hdv_base``, ``cav_plan_ch``,
    ``event_tok``, ``event_mask``) and a strategy name are all pertinently
    absent from the net's input surface.
    """
    veh = _full_veh_step(batch=2, slots=6, seed=5)
    kin = kinematics_view(veh)
    mask = torch.ones(veh.shape[0], veh.shape[1], dtype=torch.bool)
    net = HdvResidualNet(HdvConfig(use_cav_flag=False))
    net.eval()
    base = net(kin, mask)

    for col in range(7, 16):
        p = veh.clone()
        p[..., col] = torch.rand_like(p[..., col]) * 1000.0 - 500.0
        assert torch.equal(kinematics_view(p), kin), col
        o = net(kinematics_view(p), mask)
        for key in ("accel_delta_mps2", "lane_logit_delta", "accel_log_sigma"):
            assert torch.equal(base[key], o[key]), (col, key)

    full = {
        "veh_kin": kin, "veh_step": veh,
        "hdv_base": torch.randn(2, 6, 6),
        "cav_plan_ch": torch.randn(2, 6, 8),
        "event_tok": torch.randn(2, 8, 8),
        "event_mask": torch.ones(2, 8, dtype=torch.uint8),
        "is_cav": torch.ones(2, 6, dtype=torch.bool),
        "action_code": torch.ones(2, 6, dtype=torch.long),
        "strategy": "wall-v2.2",
    }
    o = net(full["veh_kin"], mask)
    for key in ("accel_delta_mps2", "lane_logit_delta", "accel_log_sigma"):
        assert torch.equal(base[key], o[key]), key


def test_headline_ignores_cav_flag_argument():
    net = HdvResidualNet(HdvConfig(use_cav_flag=False))
    net.eval()
    kin, mask = random_kin(batch=2, slots=6)
    a = net(kin, mask, cav_flag=torch.zeros(2, 6, dtype=torch.bool))
    b = net(kin, mask, cav_flag=torch.ones(2, 6, dtype=torch.bool))
    assert torch.equal(a["accel_delta_mps2"], b["accel_delta_mps2"])


def test_ablation_adds_exactly_one_cav_flag_channel():
    cfg = HdvConfig(use_cav_flag=True)
    net = HdvResidualNet(cfg)
    net.eval()
    assert net.embed[0].in_features == KIN_F + 1
    kin, mask = random_kin(batch=2, slots=6)
    out0 = net(kin, mask, cav_flag=torch.zeros(2, 6, dtype=torch.bool))
    out1 = net(kin, mask, cav_flag=torch.ones(2, 6, dtype=torch.bool))
    assert not torch.equal(out0["accel_delta_mps2"], out1["accel_delta_mps2"])
