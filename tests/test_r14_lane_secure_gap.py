"""R14 D1 contracts: HDV mainline lane-change secure-gap mask (synthetic tensors, CPU only).

The mask is discrete (no new differentiable term); the FD test certifies that the
unmasked lane logits and the acceleration keep exactly the flag-off gradients.
"""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from wms.cli.assemble import build_physics
from wms.config.train import PhysicsOptions, TrainConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import MASKED_LOGIT, anticipated_kin, lane_change_secure_ok, mobil_merge_logits
from wms.train.inputs import state_from_kin

DT = 0.5


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def physics(on=False, **kw):
    return PhysicsConfig(lane_learnable=True, lane_rule="learned_discretionary", lane_gap_rule="speed",
                         mainline_lane_mode="courtesy", lane_change_secure_gap=on, **kw)


def scene(pos, speeds, lanes, dtype=torch.float64, accel=None):
    kin = torch.zeros(1, len(pos), 7, dtype=dtype)
    kin[..., 0] = kin.new_tensor(pos)
    kin[..., 1:4] = F.one_hot(torch.tensor([lanes]), 3).to(dtype)
    kin[..., 4] = kin.new_tensor(speeds)
    if accel is not None:
        kin[..., 5] = kin.new_tensor(accel)
    kin[..., 6] = 5.
    return kin


def drivers(kin, **over):
    vals = dict(v0=31.29, time_headway=1.1, min_gap=2.5, accel=2.6, decel=4.5, politeness=0.2,
                mobil_threshold=0.1, merge_gap=6., merge_urgency=1.)
    out = {k: torch.full_like(kin[..., 0], v) for k, v in vals.items()}
    for k, v in over.items():
        out[k] = kin.new_tensor([v])
    return out


def need(v_back, v_front, t_back, b_back, b_front):
    """Independent oracle (plain floats) of the GAPACC gs law."""
    return t_back * v_back + max(0.0, v_back ** 2 / (2 * b_back) - v_front ** 2 / (2 * b_front))


def present(kin):
    return torch.ones(kin.shape[:2], dtype=torch.bool)


# ------------------------------------------------------------------ configuration
def test_config_default_off_omitted_roundtrip_and_requires_lane_learnable():
    assert not PhysicsConfig().lane_change_secure_gap
    assert "lane_change_secure_gap" not in TrainConfig().to_legacy_flat()
    cfg = TrainConfig.from_dict(dict(physics=dict(lane_learnable=True, lane_change_secure_gap=True,
                                                  inference={"lane_change_secure_gap": False})))
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()) == cfg
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    p = build_physics(cfg)
    assert p.lane_change_secure_gap and p.lane_learnable
    assert not replace(p, **cfg.physics.inference).lane_change_secure_gap
    with pytest.raises(ValueError, match="lane_learnable"):
        PhysicsConfig(lane_change_secure_gap=True)
    with pytest.raises(ValueError, match="lane_learnable"):
        TrainConfig.from_dict({"physics": {"lane_change_secure_gap": True}})


@pytest.mark.parametrize("value", [1, "true", None])
def test_boolean_validation(value):
    with pytest.raises(ValueError, match="lane_change_secure_gap"):
        PhysicsConfig(lane_learnable=True, lane_change_secure_gap=value)
    with pytest.raises(ValueError, match="lane_change_secure_gap"):
        PhysicsOptions(lane_learnable=True, lane_change_secure_gap=value)


# ------------------------------------------------------------------ feasibility law
@pytest.mark.parametrize("delta,ok", [(-0.01, False), (0.0, True), (0.01, True)])
def test_follower_side_threshold_exact_learned_settings(delta, ok):
    # Equal-speed-free check on the anticipated configuration: ego lane 1 slow, lane-0 follower fast.
    ve, vf, tf, bf, be = 5., 20., 1.3, 2.5, 4.0
    kin0 = scene([0., -100.], [ve, vf], [1, 0])
    kin_a = anticipated_kin(kin0, DT)
    gap = float(kin_a[0, 0, 0] - 5. - kin_a[0, 1, 0])
    req = need(vf, ve, tf, bf, be)
    kin = kin0.clone()
    kin[0, 1, 0] += gap - req - delta          # place the follower exactly at gap = need + delta
    d = drivers(kin, time_headway=[0.4, tf], decel=[be, bf])
    got = lane_change_secure_ok(anticipated_kin(kin, DT), present(kin), torch.zeros(1, 2, dtype=torch.long),
                                physics(True), d)
    assert bool(got[0, 0]) is ok
    logits_on = mobil_merge_logits(kin, present(kin), physics(True), d)
    logits_off = mobil_merge_logits(kin, present(kin), physics(False), d)
    assert logits_off[0, 0, 0] > MASKED_LOGIT
    assert (logits_on[0, 0, 0] == logits_off[0, 0, 0]) if ok else (logits_on[0, 0, 0] == MASKED_LOGIT)


def test_leader_side_vetoes_with_ego_settings_and_neighbour_permutation():
    # Ego (lane 1, fast) behind a slow lane-0 leader; ego's own T/b decide.
    kin = scene([0., 30., 200.], [20., 2., 20.], [1, 0, 1])
    kin_a = anticipated_kin(kin, DT)
    gap_l = float(kin_a[0, 1, 0] - 5. - kin_a[0, 0, 0])
    for t_e, ok in ((0.1, gap_l >= need(20., 2., 0.1, 9., 3.)), (1.5, gap_l >= need(20., 2., 1.5, 4., 3.))):
        b_e = 9. if t_e == 0.1 else 4.
        d = drivers(kin, time_headway=[t_e, 1., 1.], decel=[b_e, 3., 3.])
        got = lane_change_secure_ok(kin_a, present(kin), torch.zeros(1, 3, dtype=torch.long), physics(True), d)
        assert bool(got[0, 0]) is bool(ok)
        # same decision when the slots are permuted (settings follow the vehicles)
        perm = [2, 0, 1]
        dp = {k: v[:, perm] for k, v in d.items()}
        gotp = lane_change_secure_ok(kin_a[:, perm], present(kin), torch.zeros(1, 3, dtype=torch.long),
                                     physics(True), dp)
        assert bool(gotp[0, 1]) is bool(ok)
    assert 30. - 5. - 0. < need(20., 2., 1.5, 4., 3.)   # the second case is a real veto


def test_missing_neighbours_never_block_and_stopped_follower_inside_zero_need():
    kin = scene([0.], [10.], [1])
    ok = lane_change_secure_ok(anticipated_kin(kin, DT), present(kin), torch.zeros(1, 1, dtype=torch.long),
                               physics(True), drivers(kin))
    assert bool(ok[0, 0])
    # stopped follower 1 m behind a stopped ego: need = 0 -> feasible (no s0 term)
    kin = scene([0., -6.], [0., 0.], [1, 0])
    ok = lane_change_secure_ok(anticipated_kin(kin, DT), present(kin), torch.zeros(1, 2, dtype=torch.long),
                               physics(True), drivers(kin))
    assert bool(ok[0, 0])


@pytest.mark.parametrize("name,value", [("decel", float("nan")), ("decel", 0.0), ("time_headway", float("inf")),
                                        ("time_headway", -1.0)])
def test_invalid_follower_setting_fails_closed(name, value):
    kin = scene([0., -400.], [5., 20.], [1, 0])
    d = drivers(kin)
    assert bool(lane_change_secure_ok(anticipated_kin(kin, DT), present(kin), torch.zeros(1, 2, dtype=torch.long),
                                      physics(True), d)[0, 0])
    d[name] = kin.new_tensor([[d[name][0, 0].item(), value]])
    assert not bool(lane_change_secure_ok(anticipated_kin(kin, DT), present(kin),
                                          torch.zeros(1, 2, dtype=torch.long), physics(True), d)[0, 0])


def test_cav_and_ramp_rows_exempt_absent_ignored_and_off_is_identical():
    # 0: lane-1 HDV (gated), 1: lane-1 CAV (exempt), 2: ramp car (merge keeps pivot veto),
    # 3: fast lane-0 follower, 4: absent fast lane-0 car right behind HDV 0.
    kin = scene([0., 40., 30., -60., -6.], [5., 5., 5., 25., 25.], [1, 1, 2, 0, 0])
    pr = torch.tensor([[True, True, True, True, False]])
    cav = torch.tensor([[False, True, False, False, False]])
    d = drivers(kin)
    off = mobil_merge_logits(kin, pr, physics(False), d)
    on = mobil_merge_logits(kin, pr, physics(True), d, is_cav=cav)
    on_all = mobil_merge_logits(kin, pr, physics(True), d)
    assert off[0, 0, 0] > MASKED_LOGIT and on[0, 0, 0] == MASKED_LOGIT     # HDV vetoed
    assert torch.equal(on[0, 1], off[0, 1]) and on_all[0, 1, 0] == MASKED_LOGIT  # CAV exempt only with is_cav
    assert torch.equal(on[0, 2], off[0, 2])                               # ramp merge row untouched
    # lane-0 -> lane-1 is gated too: the fast HDV 3 cannot cut in behind slow HDV 0 (leader side)
    assert off[0, 3, 1] > MASKED_LOGIT and on[0, 3, 1] == MASKED_LOGIT
    assert torch.equal(on[0, 4], off[0, 4])                               # absent row
    mask = torch.zeros_like(off, dtype=torch.bool)
    mask[0, 0, 0] = mask[0, 3, 1] = True
    assert torch.equal(on[~mask], off[~mask])
    # explicit off == default construction, bit for bit
    p_default = PhysicsConfig(lane_learnable=True, lane_rule="learned_discretionary", lane_gap_rule="speed",
                              mainline_lane_mode="courtesy")
    assert torch.equal(mobil_merge_logits(kin, pr, p_default, d).view(torch.uint8),
                       mobil_merge_logits(kin, pr, physics(False), d, is_cav=cav).view(torch.uint8))


def test_default_settings_without_driver_params():
    kin = scene([0., -100.], [5., 20.], [1, 0])
    _, b, t, _, _ = physics(True).idm()
    kin_a = anticipated_kin(kin, DT)
    gap = float(kin_a[0, 0, 0] - 5. - kin_a[0, 1, 0])
    ok = lane_change_secure_ok(kin_a, present(kin), torch.zeros(1, 2, dtype=torch.long), physics(True), None)
    assert bool(ok[0, 0]) is (gap >= need(20., 5., t, b, b))


# ------------------------------------------------------------------ model level
def model(on, double=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1419)
        m = WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5", lane_decision="sample"),
                     physics=physics(on, stop_time_integration=True)).eval()
        return m.double() if double else m


def state(kin, cav=None):
    pr = present(kin)
    return {**state_from_kin(kin, pr), "is_cav": torch.zeros_like(pr) if cav is None else cav}


def test_model_lane_logits_and_step_follow_the_mask():
    kin = scene([-120., -200., 60., -125.], [3., 25., 25., 2.], [1, 0, 0, 1], dtype=torch.float32)
    st = state(kin)
    d = drivers(kin)
    off, on = model(False), model(True)
    on.load_state_dict(off.state_dict())
    with torch.no_grad():
        lo, ln = off.mobil_merge_logits(st, d), on.mobil_merge_logits(st, d)
        assert lo[0, 0, 0] > MASKED_LOGIT and ln[0, 0, 0] == MASKED_LOGIT
        # CAV flag in the state exempts the row
        lc = on.mobil_merge_logits(state(kin, torch.tensor([[True, False, False, False]])), d)
        assert torch.equal(lc[0, 0], lo[0, 0])
        # forced draw: uniform ~0 takes any open change; the vetoed row must stay in lane 1
        so = off.step(st, driver_params=d, cav_teacher=off.hold_teacher(st),
                      lane_uniform=torch.full_like(kin[..., 0], 1e-12))
        sn = on.step(st, driver_params=d, cav_teacher=on.hold_teacher(st),
                     lane_uniform=torch.full_like(kin[..., 0], 1e-12))
    assert int(so["next_lane"][0, 0]) == 0 and int(sn["next_lane"][0, 0]) == 1
    assert torch.equal(so["accel_mps2"], sn["accel_mps2"])


def test_fd_gradients_unchanged_on_feasible_rows_and_accel():
    # A feasible change (follower far): the new mask is constant locally, so lane logits and
    # acceleration keep the flag-off values and gradients; FD agrees with autograd.
    base = scene([-120., -400., 80.], [10., 20., 25.], [1, 0, 0])
    d = drivers(base)
    outs = {}
    for on in (False, True):
        m = model(on, double=True)
        kin = base.clone().requires_grad_(True)
        lg = m.mobil_merge_logits(state(kin), d)
        acc = m.physics_accel(state(kin), d)
        assert lg[0, 0, 0] > MASKED_LOGIT
        y = lg[0, 0, 0] + acc.sum()
        (g,) = torch.autograd.grad(y, kin)
        outs[on] = (lg.detach(), acc.detach(), g)
        h = 1e-6
        for idx in ((0, 0, 0), (0, 1, 0), (0, 0, 4), (0, 1, 4)):
            kp, km = base.clone(), base.clone()
            kp[idx] += h
            km[idx] -= h
            with torch.no_grad():
                fp = m.mobil_merge_logits(state(kp), d)[0, 0, 0] + m.physics_accel(state(kp), d).sum()
                fm = m.mobil_merge_logits(state(km), d)[0, 0, 0] + m.physics_accel(state(km), d).sum()
            assert torch.isclose((fp - fm) / (2 * h), g[idx], rtol=1e-5, atol=1e-7)
    for a, b in zip(outs[False], outs[True]):
        assert torch.equal(a, b)
