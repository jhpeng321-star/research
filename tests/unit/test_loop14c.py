"""LOOP14C CAV-side ramp-merge / native-twin switches: synthetic CPU inputs only.

(1) ``policy_ramp_merge``: a policy ramp CAV is never auto-merged; only its own masked lane-head
    request merges it.  (2) ``native_ramp_merge='wm_law'``: a non-policy ramp CAV is handed to the
    WM's HDV merge law (first passage + 13E secure-gap veto) and decides exactly like an HDV ramp car
    in the same scene.  Both off = the legacy forced 6 m rule.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from wms.control.bridge import make_cav_controller
from wms.control.lanes import LANE_LEFT, LANE_STAY, LaneRules
from wms.control.native_cav import head_native_accel
from wms.control.policy import PolicyOutput
from wms.control.policy14 import controller_factory, make_controller_factory
from wms.model import ModelSpec, WmSModel
from wms.model.cav import WM_MERGE_KEY
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.wms import _MERGE_PIVOT_STATE_KEYS
from wms.physics.config import PhysicsConfig


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ------------------------------------------------------------------ helpers
class ScriptedLanes:
    """Policy stub: zero acceleration, lane choice from ``schedule[decision index]``."""

    accel_max = 1.2

    def __init__(self, schedule=(), accel=0.0):
        self.schedule, self.accel, self.calls = list(schedule), accel, 0

    def __call__(self, obs, signal, lane_mask=None, **_):
        cav = obs.cav_mask
        accel = torch.where(cav, torch.full(cav.shape, float(self.accel), dtype=torch.float64),
                            torch.zeros(cav.shape, dtype=torch.float64))
        choice = self.schedule[self.calls] if self.calls < len(self.schedule) else LANE_STAY
        self.calls += 1
        zeros = torch.zeros_like(accel)
        if lane_mask is None:
            return accel, zeros
        lane = torch.where(cav, torch.full(cav.shape, int(choice)), torch.zeros(cav.shape, dtype=torch.long))
        return PolicyOutput(accel, lane, zeros, zeros, zeros, zeros, torch.ones_like(lane))


def group(obs, s, field):
    return torch.zeros((*s.shape, 8), dtype=s.dtype)


def identity_safety(state, accel, accel_max=None):
    return accel


def settings(shape, dtype=torch.float64):
    values = {"v0": 30.0, "time_headway": 1.2, "min_gap": 2.0, "accel": 1.5, "decel": 3.0}
    return torch.stack([torch.full(shape, values.get(name, 1.0), dtype=dtype)
                        for name in PARAMETER_NAMES], -1)


def ramp_scene(follower_s=0.0, follower_v=5.0):
    """slot 0 = ramp CAV in the merge zone, 1 = lane-1 follower (HDV), 2 = lane-1 leader (HDV)."""
    s = torch.tensor([[30.0, follower_s, 90.0]], dtype=torch.float64)
    return {"s_m": s, "v_mps": s.new_tensor([[5.0, follower_v, 8.0]]), "a_mps2": torch.zeros_like(s),
            "length_m": torch.full_like(s, 5.0), "lane": torch.tensor([[2, 1, 1]]),
            "present": torch.ones_like(s, dtype=torch.bool),
            "is_cav": torch.tensor([[True, False, False]]),
            "inferred_driver_parameters": settings(s.shape), "inferred_idm_delta": 4.0}


def drive(controller, state, ticks):
    """Feed the CAV's own callback motion back; HDVs keep their speed.  Returns per-tick lanes."""
    lanes = []
    state = dict(state)
    for tick in range(ticks):
        out = controller({**state, "physics_step_index": tick})
        cav = state["is_cav"]
        s = torch.where(cav, out[..., 0], state["s_m"] + 0.5 * state["v_mps"])
        state.update(s_m=s, lane=torch.where(cav, out[..., 1].long(), state["lane"]),
                     v_mps=torch.where(cav, out[..., 2], state["v_mps"]))
        lanes.append(int(state["lane"][0, 0]))
    return lanes


# ------------------------------------------------------------------ flags off
def test_flags_off_factory_is_the_legacy_object_and_defaults_match_explicit_off():
    assert make_controller_factory() is controller_factory
    rules = LaneRules(consequence_mask=True)
    runs = []
    for extra in ({}, {"policy_ramp_merge": False, "native_ramp_merge": "gap6"}):
        c = make_cav_controller(ScriptedLanes(), group, identity_safety, lane_rules=rules,
                                mask_forced_merge=True, **extra)
        runs.append(drive(c, ramp_scene(), 6))
    assert runs[0] == runs[1] == [1] * 6        # legacy: forced merge at the first tick (gaps >= 6 m)


def test_switch_validation():
    with pytest.raises(ValueError):
        make_cav_controller(ScriptedLanes(), group, identity_safety, native_ramp_merge="x")
    with pytest.raises(ValueError):
        make_cav_controller(ScriptedLanes(), group, identity_safety, policy_ramp_merge=True)
    with pytest.raises(ValueError):
        make_controller_factory(policy_ramp_merge=1)
    with pytest.raises(ValueError):
        make_controller_factory(native_ramp_merge="six")


# ------------------------------------------------------------------ (1) policy ramp CAVs
def test_policy_ramp_cav_stays_until_it_requests_a_feasible_merge():
    rules = LaneRules(consequence_mask=True)
    stay = make_cav_controller(ScriptedLanes([LANE_STAY] * 4), group, identity_safety,
                               lane_rules=rules, mask_forced_merge=True, policy_ramp_merge=True)
    assert drive(stay, ramp_scene(), 8) == [2] * 8
    # Requests ``left`` at the third decision (tick 4): merged exactly then, never before.
    late = make_cav_controller(ScriptedLanes([LANE_STAY, LANE_STAY, LANE_LEFT]), group,
                               identity_safety, lane_rules=rules, mask_forced_merge=True,
                               policy_ramp_merge=True)
    assert drive(late, ramp_scene(), 8) == [2, 2, 2, 2, 1, 1, 1, 1]


def test_policy_ramp_merge_request_is_vetoed_by_the_mask():
    # Close fast follower (rear bumper gap 3 m at 15 m/s): the feasibility/consequence mask refuses.
    rules = LaneRules(consequence_mask=True)
    scene = ramp_scene(follower_s=22.0, follower_v=15.0)
    asks = make_cav_controller(ScriptedLanes([LANE_LEFT]), group, identity_safety,
                               lane_rules=rules, mask_forced_merge=True, policy_ramp_merge=True)
    assert drive(asks, scene, 1) == [2]


def test_policy_ramp_merge_secure_gap_tier_is_optional_second_tier():
    # Feasible under the default mask; the 13E gs hook (follower T=1.2 at 14 m/s) refuses it.
    scene = ramp_scene(follower_s=-2.0, follower_v=14.0)
    scene["secure_gap_T_s"] = torch.full_like(scene["s_m"], 1.2)
    scene["secure_gap_b_mps2"] = torch.full_like(scene["s_m"], 3.0)
    lanes = {}
    for hook in ("", "wms.control.lanes:secure_gap_gs"):
        rules = LaneRules(gap_base_m=0.0, gap_time_s=0.0, closing_time_s=0.0, rear_gap_base_m=0.0,
                          rear_gap_time_s=0.0, rear_closing_time_s=0.0, secure_gap_hook=hook)
        c = make_cav_controller(ScriptedLanes([LANE_LEFT]), group, identity_safety,
                                lane_rules=rules, policy_ramp_merge=True)
        lanes[hook] = drive(c, scene, 1)
    assert lanes[""] == [1] and lanes["wms.control.lanes:secure_gap_gs"] == [2]


def test_non_acting_cavs_keep_the_legacy_rule_under_policy_ramp_merge_alone():
    native = make_cav_controller(ScriptedLanes(), group, identity_safety,
                                 lane_rules=LaneRules(consequence_mask=True), policy_ramp_merge=True,
                                 act_mask=torch.tensor([[False, False, False]]),
                                 native=lambda st: torch.zeros_like(st["s_m"]))
    assert drive(native, ramp_scene(), 2) == [1, 1]


# ------------------------------------------------------------------ (2) native ramp CAVs (WM law)
def pivot_model(hybrid):
    torch.manual_seed(17)
    m = WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                           merge_decision="first_passage", lane_decision="sample",
                           merge_passage_law="hybrid" if hybrid else "strict"),
                 physics=PhysicsConfig(secure_gap_veto=True)).double().eval()
    with torch.no_grad():
        m.merge_pivot.tau_raw.copy_(torch.log(torch.expm1(torch.tensor(2.))))
    return m


def drivers_tensor(s):
    values = {"time_headway": [1., 1.2, .7], "decel": [2., 3., 5.], "min_gap": [2., 1.5, 3.]}
    return torch.stack([torch.tensor([values[n]], dtype=s.dtype) if n in values else torch.ones_like(s)
                        for n in PARAMETER_NAMES], -1)


def fixed_forward(state, **kwargs):
    lane, s = state["lane"], state["s_m"]
    logits = s.new_full((*lane.shape, 3), -20.)
    logits.scatter_(-1, lane.unsqueeze(-1), 20.)
    return dict(accel_mps2=torch.zeros_like(s), lane_logits=logits,
                lane_feasible=torch.ones_like(logits, dtype=torch.bool),
                hdv_mask=state["present"] & ~state["is_cav"], driver_parameters=drivers_tensor(s),
                lane_age=torch.zeros_like(lane), lane_sample=torch.zeros_like(state["present"]))


def wm_scene(follower_v, cav):
    s = torch.tensor([[30., 5., 90.]], dtype=torch.float64)
    return dict(s_m=s, v_mps=s.new_tensor([[5., follower_v, 8.]]), a_mps2=torch.zeros_like(s),
                length_m=torch.full_like(s, 5.), lane=torch.tensor([[2, 1, 1]]),
                present=torch.ones_like(s, dtype=torch.bool),
                is_cav=torch.tensor([[cav, False, False]]))


def native_controller(mode, accel=None):
    accel_fn = (lambda st: torch.zeros_like(st["s_m"])) if accel is None else accel
    return make_cav_controller(ScriptedLanes(), group, identity_safety,
                               act_mask=torch.tensor([[False, False, False]]),
                               native=accel_fn, native_ramp_merge=mode)


def wm_rollout(m, cav, mode, follower_v, steps):
    torch.manual_seed(5)      # the hybrid law seeds its keyed latent draws from the global RNG
    state = wm_scene(follower_v, cav)
    controller = native_controller(mode) if cav else None
    lanes, outs = [], []
    for step in range(steps):
        with patch.object(m, "forward", side_effect=fixed_forward):
            out = m.step(state, k_steps=step, cav_controller=controller,
                         lane_uniform=torch.full_like(state["s_m"], .6))
        outs.append(out)
        lanes.append(int(out["next_lane"][0, 0]))
        nxt = {key: state[key] for key in ("length_m", "present", "is_cav")}
        nxt.update(s_m=out["next_s_m"], v_mps=out["next_v_mps"], a_mps2=out["accel_mps2"],
                   lane=out["next_lane"].long())
        for key in _MERGE_PIVOT_STATE_KEYS:
            if "merge_pivot_" + key in out:
                nxt["merge_pivot_" + key] = out["merge_pivot_" + key]
        state = nxt
    return lanes, outs


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("follower_v", [2.0, 15.0])
def test_native_ramp_cav_merges_exactly_like_an_hdv_ramp_car(hybrid, follower_v):
    m = pivot_model(hybrid)
    hdv, hdv_out = wm_rollout(m, False, None, follower_v, 10)
    cav, cav_out = wm_rollout(m, True, "wm_law", follower_v, 10)
    legacy, _ = wm_rollout(m, True, "gap6", follower_v, 10)
    assert cav == hdv
    for a, b in zip(hdv_out, cav_out):
        for key in ("merge_pivot_c", "merge_pivot_eligible", "merge_pivot_committed", "next_s_m"):
            assert torch.equal(a[key][0, 0], b[key][0, 0]), key
    assert legacy[0] == 1                       # legacy: >= 6 m gaps -> lane 1 at the first tick
    if follower_v == 15.0:
        # 13E veto (fast follower): the HDV law keeps the ramp car waiting; so does the CAV now.
        assert cav[0] == 2 and hdv[0] == 2


def test_wm_law_delegation_reaches_the_wm_and_needs_a_first_passage_wm():
    m = pivot_model(False)
    _, outs = wm_rollout(m, True, "wm_law", 15.0, 1)
    assert outs[0][WM_MERGE_KEY].tolist() == [[True, False, False]]
    torch.manual_seed(3)
    plain = WmSModel(ModelSpec(rollout_mode="driver_params"), physics=PhysicsConfig()).double().eval()
    with patch.object(plain, "forward", side_effect=fixed_forward), pytest.raises(ValueError):
        plain.step(wm_scene(15.0, True), cav_controller=native_controller("wm_law"))


def test_fd_gradient_through_the_delegated_cav_merge_boundary():
    m = pivot_model(False)

    def boundary(a0):
        def accel(st):
            return torch.full_like(st["s_m"], 0.0) + a0
        controller = native_controller("wm_law", accel)
        with patch.object(m, "forward", side_effect=fixed_forward):
            out = m.step(wm_scene(2.0, True), cav_controller=controller,
                         lane_uniform=torch.full((1, 3), .6, dtype=torch.float64))
        return out["merge_pivot_c"][0, 0]

    a0 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    value = boundary(a0)
    assert torch.isfinite(value)
    (auto,) = torch.autograd.grad(value, a0)
    eps = 1e-6
    with torch.no_grad():
        fd = (boundary(a0 + eps) - boundary(a0 - eps)) / (2 * eps)
    assert float(auto) != 0.0
    assert torch.allclose(auto, fd, rtol=1e-5, atol=1e-8)


def test_fd_gradient_of_the_learned_native_head():
    s = torch.tensor([[0.0, 25.0]], dtype=torch.float32, requires_grad=True)
    def accel(pos):
        st = {"s_m": pos, "v_mps": torch.tensor([[12.0, 8.0]]), "lane": torch.tensor([[1, 1]]),
              "present": torch.ones(1, 2, dtype=torch.bool), "length_m": torch.full((1, 2), 5.0)}
        return head_native_accel(st)[0, 0]
    (auto,) = torch.autograd.grad(accel(s), s)
    eps = 1e-2
    shift = torch.tensor([[0.0, eps]])
    with torch.no_grad():
        fd = (accel(s + shift) - accel(s - shift)) / (2 * eps)
    assert torch.allclose(auto[0, 1], fd, rtol=2e-2, atol=1e-4)
