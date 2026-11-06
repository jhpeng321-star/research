"""Native-only lane delegation uses the HDV path, without changing CAV longitudinal ownership."""
from unittest.mock import patch

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.policy14 import controller_factory, make_controller_factory
from wms.model import ModelSpec, WmSModel
from wms.model.cav import WM_LANE_KEY, WM_LANE_TARGET_KEY
from wms.physics.config import PhysicsConfig
from .test_loop14c import ScriptedLanes, group, identity_safety, wm_scene, fixed_forward


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    torch.manual_seed(17)
    return WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              merge_decision="first_passage", lane_decision="sample"),
                    physics=PhysicsConfig(secure_gap_veto=True)).eval()


def controller(mask, **kwargs):
    return make_cav_controller(ScriptedLanes(accel=.3), group, identity_safety,
                               act_mask=mask, native=lambda st: torch.zeros_like(st["s_m"]),
                               native_lane_change=True, native_ramp_merge="wm_law", **kwargs)


def test_default_off_factory_and_validation():
    assert make_controller_factory(native_lane_change=False) is controller_factory
    with pytest.raises(ValueError, match="boolean"):
        make_controller_factory(native_lane_change=1)
    with pytest.raises(ValueError, match="WM-computed"):
        controller(torch.zeros((1, 3), dtype=torch.bool))(wm_scene(2., True))


def test_dynamic_native_mask_tracks_zone_presence_and_policy_ownership():
    state = wm_scene(2., True)
    state["is_cav"][:] = True
    state["present"][0, 2] = False
    c = controller(torch.tensor([[True, True, False]]), zone=lambda st: st["s_m"] < 20)
    assert c.native_lane_mask(state).tolist() == [[False, True, False]]
    state["s_m"][0, 1] = 25
    assert not c.native_lane_mask(state).any()


def test_native_lane_moves_policy_lane_stays_and_native_acceleration_is_retained():
    state = wm_scene(2., True)
    state["lane"][:] = 1
    state["is_cav"][:] = True
    state[WM_LANE_TARGET_KEY] = torch.zeros_like(state["lane"])
    c = controller(torch.tensor([[False, True, False]]))
    out = c(state)
    assert out[..., 1].long().tolist() == [[0, 1, 0]]
    assert out[..., 3].tolist() == [[0., .3, 0.]]


def test_native_head_and_safety_run_before_the_native_lane_substep():
    state = wm_scene(2., True)
    state["lane"][:] = 1
    state["is_cav"][:] = True
    state[WM_LANE_TARGET_KEY] = torch.zeros_like(state["lane"])
    seen = []

    def head(st):
        seen.append(st["lane"].clone())
        return st["lane"].to(st["s_m"].dtype)

    def safety(st, accel, accel_max=None):
        seen.append(st["lane"].clone())
        return accel

    c = make_cav_controller(ScriptedLanes(accel=.3), group, safety,
                            act_mask=torch.tensor([[False, True, False]]), native=head,
                            native_lane_change=True, native_ramp_merge="wm_law")
    result = c(state)
    assert all(x.tolist() == [[1, 1, 1]] for x in seen)
    assert result[..., 1].long().tolist() == [[0, 1, 0]]
    assert result[..., 3].tolist() == [[1., .3, 1.]]


def test_native_forward_lane_logits_and_carry_match_hdv_but_not_longitudinal_mask():
    m = model()
    state = {k: v.float() if v.is_floating_point() else v for k, v in wm_scene(2., True).items()}
    native = {**state, WM_LANE_KEY: state["is_cav"]}
    human = {**state, "is_cav": torch.zeros_like(state["is_cav"])}
    with torch.no_grad():
        a, b = m.forward(native), m.forward(human)
    for key in ("lane_logits", "lane_v5_acc", "lane_sample", "lane_feasible", "driver_parameters"):
        assert torch.equal(a[key], b[key]), key
    assert not a["hdv_mask"][0, 0]
    assert a["lane_decider_mask"][0, 0]


def test_step_does_not_overwrite_native_discretionary_lane_with_teacher_stay():
    m = model()
    state = wm_scene(2., True)
    state["lane"] = torch.tensor([[1, 1, 1]])
    state["s_m"] = state["s_m"] + 300

    def forward(st, **kw):
        out = fixed_forward(st, **kw)
        out["lane_logits"][0, 0] = out["lane_logits"].new_tensor([20., -20., -20.])
        out["lane_sample"][:] = False
        out["lane_decider_mask"] = st["present"]
        return out

    with patch.object(m, "forward", side_effect=forward):
        out = m.step(state, cav_controller=controller(torch.zeros_like(state["present"])),
                     lane_uniform=torch.full_like(state["s_m"], .5))
    assert out["next_lane"][0, 0] == 0
    assert out["accel_mps2"][0, 0] == 0


def test_native_rule_requires_a_learned_structured_first_passage_model():
    m = WmSModel(ModelSpec())
    state = wm_scene(2., True)
    with pytest.raises(ValueError, match="first-passage"):
        m.step(state, cav_controller=controller(torch.zeros_like(state["present"])))


def test_preforward_native_ownership_receives_the_same_tick_as_execution():
    m = model()
    state = wm_scene(2., True)
    c = controller(torch.zeros_like(state["present"]))
    prior, ticks = c.native_lane_mask, []

    def mask(st):
        ticks.append(int(st["physics_step_index"]))
        return prior(st)
    c.native_lane_mask = mask
    with patch.object(m, "forward", side_effect=fixed_forward):
        m.step(state, k_steps=8, cav_controller=c,
               lane_uniform=torch.full_like(state["s_m"], .5))
    assert ticks == [8]


def test_native_mainline_insertion_competes_with_ramp_exactly_like_hdv():
    m = model()
    state = wm_scene(2., True)
    state["lane"] = torch.tensor([[0, 2, 1]])
    state["s_m"][0, 1] = state["s_m"][0, 0]
    outputs = []

    def forward(st, **kw):
        out = fixed_forward(st, **kw)
        out["lane_logits"][0, 0] = out["lane_logits"].new_tensor([-20., 20., -20.])
        out["lane_sample"][:] = False
        out["lane_decider_mask"] = st["present"]
        return out

    for human in (True, False):
        st = {**state, "is_cav": torch.zeros_like(state["is_cav"])} if human else state
        c = None if human else controller(torch.zeros_like(state["present"]))
        with patch.object(m, "forward", side_effect=forward):
            outputs.append(m.step(st, cav_controller=c, lane_uniform=torch.full_like(st["s_m"], .5)))
    for key in ("next_lane", "merge_pivot_eligible", "merge_pivot_committed", "merge_pivot_c"):
        assert torch.equal(outputs[0][key], outputs[1][key]), key


def test_gate_checkpoint_metadata_preserves_native_mask_and_merge_delegation(tmp_path):
    from wms.control.confident_gate import ArmImagination
    from wms.control.policy import CavPolicy, RsuGroupEncoder, load_policy_components
    p, enc = CavPolicy(lane_head=True), RsuGroupEncoder()
    kwargs = {"native_lane_change": True, "native_ramp_merge": "wm_law",
              "symmetric_exec_cap": True, "mask_forced_merge": True}
    path = tmp_path / "policy.pt"
    torch.save({"format": "c4_dreamer/2", "policy": p.state_dict(),
                "group_encoder": enc.state_dict(), "wm_controller_kwargs": kwargs}, path)
    policy, encoder, rules, _ = load_policy_components(path)
    m = model()
    state = {k: v.float() if v.is_floating_point() else v for k, v in wm_scene(2., True).items()}
    imagine = ArmImagination([m], policy, encoder, lane_rules=rules, horizon_s=1.)
    with patch.object(m, "forward", wraps=m.forward) as seen, \
            patch.object(m, "_apply_merge_pivot", wraps=m._apply_merge_pivot) as merging:
        result = imagine.arms_J(state, [None, torch.zeros(3, dtype=torch.bool)])
    assert result.shape == (1, 2, 1) and torch.isfinite(result).all()
    mask = seen.call_args_list[0].args[0][WM_LANE_KEY]
    assert not mask[0].any() and mask[1].tolist() == [True, False, False]
    assert merging.call_args_list[0].kwargs["delegated"][1, 0]
    hold = ArmImagination([m], policy, encoder, reference="hold")
    assert not hold.controller_kwargs["native_lane_change"]
