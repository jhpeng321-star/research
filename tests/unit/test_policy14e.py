"""POLICY14E: policy / closed-loop paths use the same driver settings as the 13E WM step."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wms.control.lanes import LaneRules, lane_feasibility, secure_gap_gs
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.merge_pivot import FirstPassageNet, geometry
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train import cav_drive
from wms.train.inputs import state_from_kin


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def physics(**kw):
    return PhysicsConfig(lane_learnable=True, lane_rule="forced_merge_only",
                         lane_gap_rule="speed", stop_time_integration=True, **kw)


def model(yield_on=False, emergency=False, gs=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1319)
        # merge_pivot (first_passage) hands CAV callbacks the WM's own native accel; the
        # fallback path (merge_pivot None) is the one that calls ``physics_accel``.
        extra = (dict(merge_decision="first_passage", lane_model="hazard_v5",
                      lane_decision="sample") if gs else {})
        spec = ModelSpec(rollout_mode="driver_params", mainline_merge_yield_learnable=yield_on,
                         follower_courtesy_intent="none", **extra)
        return WmSModel(spec, physics=physics(emergency_braking=emergency,
                                              secure_gap_veto=gs)).cpu().eval()


def scene(positions=(70., 15., 5.), speeds=(0., 17., 12.), lanes=(2, 1, 1)):
    kin = torch.zeros(1, len(positions), 7)
    kin[..., 0] = kin.new_tensor(positions)
    kin[..., 1:4] = F.one_hot(torch.tensor([lanes]), 3).to(kin.dtype)
    kin[..., 4] = kin.new_tensor(speeds)
    kin[..., 6] = 5.
    present = torch.ones(kin.shape[:2], dtype=torch.bool)
    cav = torch.zeros_like(present)
    cav[0, 0] = True
    return {**state_from_kin(kin, present), "is_cav": cav}


def settings_tensor(state, **over):
    values = dict(v0=31.29, time_headway=1.1, min_gap=2.5, accel=2.6, decel=4.5,
                  politeness=.2, mobil_threshold=.1, merge_gap=6., merge_urgency=1.)
    values.update(over)
    ref = state["s_m"]
    return torch.stack([torch.full_like(ref, values[n]) for n in PARAMETER_NAMES], -1)


def timeline():
    ids = np.zeros((1, 3, 3), dtype=np.int64)  # every CAV native
    return {"controller": ids, "acceleration_mps2": np.zeros(ids.shape, np.float32),
            "target_speed_mps": np.full(ids.shape, 8., np.float32)}


@pytest.mark.parametrize("flags", [dict(yield_on=True), dict(emergency=True),
                                   dict(yield_on=True, emergency=True)])
def test_cav_drive_native_uses_wm_driver_settings_when_13e_flags_are_on(flags):
    m = model(**flags)
    st = settings_tensor(scene())
    state = {**scene(), "inferred_driver_parameters": st}
    params = {n: st[..., i] for i, n in enumerate(PARAMETER_NAMES)}
    expected = m.physics_accel(state, params)
    out = cav_drive.TimelineControllers(m, timeline())(state)
    assert torch.isfinite(out).all()
    assert torch.equal(out[..., 3], expected)


def test_cav_drive_learned_yield_without_settings_fails_with_a_clear_message():
    m = model(yield_on=True)
    with pytest.raises(ValueError, match="inferred_driver_parameters"):
        cav_drive.TimelineControllers(m, timeline())(scene())


def test_cav_drive_flags_off_is_the_legacy_default_driver_call():
    m = model()
    state = scene()
    legacy = m.physics_accel(state)
    with_settings = {**state, "inferred_driver_parameters": settings_tensor(state, decel=2.)}
    out = cav_drive.TimelineControllers(m, timeline())(with_settings)
    assert torch.equal(out[..., 3], legacy)  # settings ignored: bit-identical legacy behaviour


def test_step_callback_to_cav_drive_with_learned_yield_does_not_raise():
    m = model(yield_on=True, emergency=True)  # merge_pivot None: callback computes native accel
    state = scene()
    st = settings_tensor(state)
    params = {n: st[..., i] for i, n in enumerate(PARAMETER_NAMES)}
    ctl = cav_drive.TimelineControllers(m, timeline())
    out = m.step(state, driver_params=params, cav_controller=ctl,
                 baseline_lane_logits=state["veh_kin"].new_full((1, 3, 3), -1e6).scatter(
                     -1, state["lane"].unsqueeze(-1), 0.))
    assert torch.isfinite(out["next_s_m"]).all() and ctl.step_index == 1


def test_policy_gs_hook_and_wm_merge_veto_make_the_same_decision():
    """Policy lane mask (gs hook) == WM FirstPassageNet.secure_gap_margin >= 0 (GAPACC, no s0)."""
    gen = torch.Generator().manual_seed(7)
    net = FirstPassageNet()
    agree = 0
    for _ in range(200):
        pos = torch.tensor([[40. + 20 * torch.rand(1, generator=gen).item(),
                             10. + 30 * torch.rand(1, generator=gen).item(),
                             60. + 40 * torch.rand(1, generator=gen).item()]])
        spd = 20 * torch.rand(1, 3, generator=gen)
        kin = torch.zeros(1, 3, 7)
        kin[..., 0] = pos
        kin[0, 0, 1:4] = torch.tensor([0., 0., 1.])
        kin[0, 1:, 1:4] = torch.tensor([0., 1., 0.])
        kin[..., 4], kin[..., 6] = spd, 5.
        present = torch.ones(1, 3, dtype=torch.bool)
        settings = {"time_headway": .5 + 1.5 * torch.rand(1, 3, generator=gen),
                    "decel": 1. + 4 * torch.rand(1, 3, generator=gen),
                    "min_gap": .5 + 3 * torch.rand(1, 3, generator=gen)}
        geo = geometry(kin, present, PhysicsConfig())
        wm_ok = bool(net.secure_gap_margin(geo, settings)[0, 0] >= 0.)
        state = {**state_from_kin(kin, present), "is_cav": torch.tensor([[True, False, False]]),
                 "secure_gap_T_s": settings["time_headway"], "secure_gap_b_mps2": settings["decel"]}
        lane = state["lane"].long()
        hook_ok = bool(secure_gap_gs(state, lane, lane - 1)[0, 0])
        agree += hook_ok == wm_ok
        assert hook_ok == wm_ok
    assert agree == 200


def test_consequence_mask_is_a_veto_not_an_additive_term_on_top_of_the_emergency_envelope():
    """The mask judges the UNENVELOPED IDM response of the new follower (the WM's pre-envelope
    baseline); the WM envelope only bounds dynamics. Mask output is identical with the 13E
    envelope flag on or off (no double counting)."""
    results = []
    for emergency in (False, True):
        m = model(emergency=emergency)
        state = scene(positions=(30., 22., 90.), speeds=(14., 14., 14.), lanes=(1, 0, 0))
        state["is_cav"] = torch.tensor([[True, False, False]])
        state["inferred_driver_parameters"] = settings_tensor(state)
        state["inferred_idm_delta"] = m.physics.idm()[4]
        results.append(lane_feasibility(state, rules=LaneRules(consequence_mask=True)))
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize("flags", [dict(emergency=True), dict(yield_on=True, emergency=True)])
def test_13e_action_adjoint_matches_finite_differences(flags):
    from wms.control.policy14_probes import gradient_probe_13e
    result = gradient_probe_13e(model(**flags), n_steps=3)
    assert all(row["passed"] for row in result["differences"])
    assert abs(result["autograd"]) > 1e-7 and result["flags"]["emergency_braking"]
