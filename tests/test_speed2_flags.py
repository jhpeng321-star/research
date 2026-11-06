"""S-SPEED2 opt-in speed switches: default off (legacy path), and flag-on equivalence."""
import pytest
import torch
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from wms.core import speed_flags
from wms.model import ModelSpec, WmSModel
from wms.model import merge_pivot as pivot
from wms.model.residual_net import COURTESY_F, HdvConfig, HdvResidualNet
from wms.physics.config import PhysicsConfig

_DIAG = ("logS", "event_mass", "cdf")


def _helpers(path):
    spec = importlib.util.spec_from_file_location("_speed2_helpers", Path(__file__).parent / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _flags_off(monkeypatch):
    for env in speed_flags.FLAGS.values():
        monkeypatch.delenv(env, raising=False)


def test_all_switches_default_off():
    assert speed_flags.active() == {name: (1 if name == "grad_audit_every" else False)
                                    for name in speed_flags.FLAGS}


def test_audit_period_parsing(monkeypatch):
    for raw, period in (("", 1), ("0", 1), ("x", 1), ("-3", 1), ("²", 1), ("1", 1), ("4", 4)):
        monkeypatch.setenv("WMS_SPEED_GRAD_AUDIT_EVERY", raw)
        assert speed_flags.every("grad_audit_every") == period


def test_only_exact_one_enables(monkeypatch):
    for value in ("0", "true", "yes", " 1", ""):
        monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", value)
        assert not speed_flags.enabled("compact_slots")
    monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", "1")
    assert speed_flags.enabled("compact_slots")


# --------------------------------------------------------------- compact_slots
def _net(seed=0):
    torch.manual_seed(seed)
    net = HdvResidualNet(HdvConfig(d_model=32, n_heads=4, n_layers=2, ffn=64,
                                   courtesy_inputs=True, conditioning_dim=3))
    for p in net.parameters():           # zero-initialised adapters must matter in the test
        with torch.no_grad():
            p.add_(0.05 * torch.randn_like(p))
    return net


def _inputs(b=3, k=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    present = torch.rand(b, k, generator=g) < 0.3
    present[2] = False                    # an empty row
    present[1, :3] = True
    kin = torch.zeros(b, k, 7)
    lane = torch.randint(0, 3, (b, k), generator=g)
    kin[..., 0] = torch.rand(b, k, generator=g) * 400 - 200
    kin[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    kin[..., 4] = torch.rand(b, k, generator=g) * 25
    kin[..., 5] = torch.randn(b, k, generator=g)
    kin[..., 6] = 5.0
    kin = kin * present.unsqueeze(-1)
    courtesy = torch.rand(b, k, COURTESY_F, generator=g) * present.unsqueeze(-1)
    cond = torch.randn(b, k, 3, generator=g)
    return kin, present, courtesy, cond


def _run(net, kin, present, courtesy, cond):
    kin = kin.clone().requires_grad_(True)
    cond = cond.clone().requires_grad_(True)
    out = net(kin, present, courtesy=courtesy, conditioning=cond)
    w = torch.linspace(-1, 1, out["lane_logit_delta"].numel()).reshape(out["lane_logit_delta"].shape)
    loss = ((out["accel_delta_mps2"] * present).sum() + (out["accel_log_sigma"] * present).sum()
            + (out["lane_logit_delta"] * w).sum())
    grads = torch.autograd.grad(loss, [kin, cond, *net.parameters()], allow_unused=True)
    return out, grads


def test_flag_off_forward_is_the_dense_path():
    net, (kin, present, courtesy, cond) = _net(), _inputs()
    a = net(kin, present, courtesy=courtesy, conditioning=cond)
    b = net._forward_dense(kin, present, courtesy=courtesy, conditioning=cond)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_compact_slots_matches_dense_values_and_gradients(monkeypatch):
    net, inputs = _net(), _inputs()
    dense, g_dense = _run(net, *inputs)
    monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", "1")
    comp, g_comp = _run(net, *inputs)
    present = inputs[1]
    for key in ("accel_delta_mps2", "accel_log_sigma", "lane_logit_delta", "attn"):
        assert comp[key].shape == dense[key].shape, key
        torch.testing.assert_close(comp[key], dense[key], atol=1e-5, rtol=1e-5)
    # Absent slots: exactly the dense values (zeros / the absent-row log sigma).
    absent = ~present
    assert torch.equal(comp["accel_delta_mps2"][absent], dense["accel_delta_mps2"][absent])
    assert torch.equal(comp["lane_logit_delta"][absent], dense["lane_logit_delta"][absent])
    torch.testing.assert_close(comp["accel_log_sigma"][absent], dense["accel_log_sigma"][absent],
                               atol=1e-6, rtol=0)
    for gd, gc in zip(g_dense, g_comp):
        if gd is None:
            assert gc is None or not gc.abs().any()
        else:
            torch.testing.assert_close(gc, gd, atol=1e-5, rtol=1e-4)


def test_compact_slots_full_rows_fall_back_to_dense(monkeypatch):
    net = _net()
    kin, _, courtesy, cond = _inputs()
    present = torch.ones(kin.shape[:2], dtype=torch.bool)
    ref = net(kin, present, courtesy=courtesy, conditioning=cond)
    monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", "1")
    out = net(kin, present, courtesy=courtesy, conditioning=cond)
    for key in ref:
        assert torch.equal(ref[key], out[key]), key


def test_compact_slots_one_padding_column_keeps_dense_bits(monkeypatch):
    net = _net()
    kin, _, courtesy, cond = _inputs()
    present = torch.ones(kin.shape[:2], dtype=torch.bool)
    present[:, 5] = False
    ref = net(kin, present, courtesy=courtesy, conditioning=cond)
    monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", "1")
    out = net(kin, present, courtesy=courtesy, conditioning=cond)
    for key in ref:
        assert torch.equal(ref[key], out[key]), key


@pytest.mark.parametrize("layers,zero,cav", [(0, False, True), (1, True, False), (2, False, True)])
def test_compact_optional_architectures_and_unmasked_sigma_loss(monkeypatch, layers, zero, cav):
    net = HdvResidualNet(HdvConfig(d_model=16, n_heads=2, n_layers=layers, ffn=32,
                                   use_cav_flag=cav, zero_residual=zero))
    kin, mask, _, _ = _inputs()
    cav_flag = mask.clone()
    results = []
    for flag in ("0", "1"):
        monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", flag)
        out = net(kin, mask, cav_flag=cav_flag)
        loss = sum(out[key].square().sum() for key in
                   ("accel_delta_mps2", "lane_logit_delta", "accel_log_sigma"))
        grads = (torch.autograd.grad(loss, tuple(net.parameters()), allow_unused=True)
                 if loss.requires_grad else ())
        results.append((out, loss, grads))
    a, b = results
    torch.testing.assert_close(a[1], b[1], atol=1e-5, rtol=1e-5)
    for key in a[0]:
        if a[0][key] is None:
            assert b[0][key] is None
        else:
            torch.testing.assert_close(a[0][key], b[0][key], atol=1e-5, rtol=1e-5)
    for ga, gb in zip(a[2], b[2]):
        if ga is None:
            assert gb is None
        else:
            torch.testing.assert_close(ga, gb, atol=1e-4, rtol=1e-4)


def test_compaction_preserves_training_dropout_and_rng(monkeypatch):
    net = HdvResidualNet(HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32, dropout=.3))
    kin, mask, _, _ = _inputs()
    torch.manual_seed(31)
    expected = net(kin, mask)
    rng = torch.get_rng_state()
    monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", "1")
    torch.manual_seed(31)
    actual = net(kin, mask)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(expected[key], actual[key]) for key in expected)


def test_compaction_keeps_optional_input_validation(monkeypatch):
    net = _net()
    kin, mask, courtesy, cond = _inputs()
    monkeypatch.setenv("WMS_SPEED_COMPACT_SLOTS", "1")
    with pytest.raises(ValueError, match="courtesy"):
        net(kin, mask, courtesy=courtesy[:, :-1])
    with pytest.raises(ValueError, match="conditioning"):
        net(kin, mask, conditioning=cond[:, :-1])
    # These channels are ignored by an architecture without either adapter.
    net = HdvResidualNet(HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32))
    ref = net._forward_dense(kin, mask, courtesy=torch.tensor(1), conditioning=torch.tensor(1))
    got = net(kin, mask, courtesy=torch.tensor(1), conditioning=torch.tensor(1))
    for key in ref:
        torch.testing.assert_close(ref[key], got[key], atol=1e-5, rtol=1e-5)


# -------------------------------------------------------------- pivot_diag_off
def _hybrid_model():
    torch.manual_seed(19)
    return WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                              lane_decision="sample", merge_decision="first_passage",
                              merge_passage_law="hybrid", merge_w_init=0.45),
                    physics=PhysicsConfig()).eval()


def _state():
    s = torch.tensor([[30.0, 37.0, 23.0, 49.0, 10.0, 0.0]])
    return {"s_m": s, "v_mps": torch.full_like(s, 5.0), "a_mps2": torch.zeros_like(s),
            "length_m": torch.full_like(s, 5.0), "lane": torch.tensor([[2, 2, 2, 1, 1, -1]]),
            "present": torch.tensor([[True] * 5 + [False]]),
            "is_cav": torch.zeros_like(s, dtype=torch.bool)}


def test_update_state_without_diagnostics_keeps_every_other_carry():
    gen = torch.Generator().manual_seed(3)
    rows, steps = 6, 9
    present = torch.ones(1, rows, dtype=torch.bool)
    ramp = torch.tensor([[True, True, True, True, False, True]])
    tau, w = torch.tensor(0.7), torch.tensor(0.45)
    carries = {True: None, False: None}
    for t in range(steps):
        c = torch.randn(1, rows, generator=gen) * 2.0 - 1.0
        c[0, 1] = -torch.inf if t % 3 == 0 else c[0, 1]
        uni = torch.rand(1, rows, generator=gen).clamp(1e-6, 1 - 1e-6)
        outs = {}
        for diag in (True, False):
            torch.manual_seed(100 + t)
            outs[diag] = pivot.update_state(c, tau, ramp, present, carry=carries[diag],
                                            identity=torch.arange(rows)[None], uniform=uni,
                                            w=w, quadrature_order=8, diagnostics=diag)
        on, off = outs[True], outs[False]
        assert set(on) == set(off)
        for key in on:
            if key in _DIAG:
                assert torch.isnan(off[key]).all(), key
            else:
                assert torch.equal(on[key], off[key]), (t, key)
        carries = {diag: {k: v for k, v in outs[diag].items() if k != "cross"}
                   for diag in (True, False)}


@pytest.mark.parametrize("grad", [False, True])
def test_pivot_diag_off_rollout_bit_identical_and_rng_neutral(monkeypatch, grad):
    results = {}
    for flag in ("", "1"):
        monkeypatch.setenv("WMS_SPEED_PIVOT_DIAG_OFF", flag)
        m = _hybrid_model()
        assert m.merge_pivot is not None and m.merge_pivot.hybrid
        if grad:
            m.train()
        torch.manual_seed(5)
        with torch.set_grad_enabled(grad):
            roll = m.rollout(_state(), n_steps=24)
            grads = None
            if grad:
                loss = roll["positions"].sum() + roll["speeds"].square().sum()
                params = [p for p in m.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss, params, allow_unused=True)
        results[flag] = (roll, grads, torch.random.get_rng_state())
    (on, g_on, rng_on), (off, g_off, rng_off) = results[""], results["1"]
    for key in ("positions", "speeds", "lanes", "masks", "accel", "lane_logits"):
        assert torch.equal(on[key], off[key]), key
    assert torch.equal(rng_on, rng_off)
    assert torch.isfinite(on["merge_pivot_cdf"]).any()
    assert torch.isnan(off["merge_pivot_cdf"]).all()
    if grad:
        for a, b in zip(g_on, g_off):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.equal(a, b)


# ----------------------------------------------------------- fast_long_history
def test_fast_long_history_is_bit_identical(monkeypatch):
    from wms.data.history import long_history_features
    g = torch.Generator().manual_seed(4)
    t, k = 37, 23
    kin = torch.zeros(t, k, 7)
    lane = torch.randint(0, 3, (t, k), generator=g)
    kin[..., 0] = (torch.rand(t, k, generator=g) * 300).round()   # many exact position ties
    kin[..., 1:4] = torch.nn.functional.one_hot(lane, 3).float()
    kin[..., 4] = torch.rand(t, k, generator=g) * 30
    kin[..., 5] = torch.randn(t, k, generator=g)
    kin[..., 6] = 4.0 + torch.rand(t, k, generator=g) * 4         # ties pick different lengths
    mask = torch.rand(t, k, generator=g) < 0.7
    target = torch.rand(t - 1, k, generator=g) < 0.9
    present = torch.rand(k, generator=g) < 0.8
    for history_s, stride in ((10.0, 1.0), (3.0, 0.5), (60.0, 1.0)):
        ref = long_history_features(kin, mask, target, present, history_s=history_s,
                                    history_stride_s=stride)
        monkeypatch.setenv("WMS_SPEED_FAST_LONG_HISTORY", "1")
        new = long_history_features(kin, mask, target, present, history_s=history_s,
                                    history_stride_s=stride)
        monkeypatch.delenv("WMS_SPEED_FAST_LONG_HISTORY")
        assert torch.equal(ref[0], new[0]) and torch.equal(ref[1], new[1])
    one = long_history_features(kin[:1], mask[:1], target[:0], present, history_s=5.0)
    monkeypatch.setenv("WMS_SPEED_FAST_LONG_HISTORY", "1")
    assert all(torch.equal(a, b) for a, b in zip(
        one, long_history_features(kin[:1], mask[:1], target[:0], present, history_s=5.0)))


# -------------------------------------------------------------- fast_cav_drive
class _StubModel:
    class physics:
        dt_s = 0.5
        stop_time_integration = False
        speed_limit_mps = 30.0

    mainline_merge_yield_learnable = False

    @staticmethod
    def physics_accel(state, params=None):
        return 0.3 * torch.sin(state["s_m"]) - 0.1 * state["v_mps"]


def test_fast_cav_drive_is_bit_identical(monkeypatch):
    from wms.train.cav_drive import TimelineControllers
    g = torch.Generator().manual_seed(8)
    b, k, steps = 2, 9, 8
    ids = torch.randint(0, 4, (b, steps, k), generator=g)
    ids = torch.where(ids == 3, ids, torch.zeros_like(ids))       # native / direct only
    timeline = {"controller": ids, "acceleration_mps2": torch.randn(b, steps, k, generator=g),
                "target_speed_mps": 10 + 5 * torch.rand(b, steps, k, generator=g),
                "control_dt_s": 1.0}
    outs = {}
    for flag in ("", "1"):
        monkeypatch.setenv("WMS_SPEED_FAST_CAV_DRIVE", flag)
        ctrl = TimelineControllers(_StubModel(), timeline)
        gen = torch.Generator().manual_seed(9)
        rows = []
        for _ in range(2 * steps):
            state = {"s_m": torch.rand(b, k, generator=gen) * 100,
                     "v_mps": torch.rand(b, k, generator=gen) * 20,
                     "lane": torch.randint(-1, 3, (b, k), generator=gen),
                     "present": torch.rand(b, k, generator=gen) < 0.8,
                     "is_cav": torch.rand(b, k, generator=gen) < 0.6,
                     "a_mps2": torch.zeros(b, k)}
            rows.append(ctrl(state))
        outs[flag] = (torch.stack(rows), dict(ctrl.active))
    assert torch.equal(outs[""][0], outs["1"][0])
    assert outs[""][1] == outs["1"][1]


def test_fast_cav_drive_preserves_stateful_controller_dispatch(monkeypatch):
    """Exercise damping/PI handbacks without importing their frozen external assets."""
    from wms.train.cav_drive import TimelineControllers
    g = torch.Generator().manual_seed(81)
    b, k, steps = 2, 7, 6
    ids = torch.randint(0, 4, (b, steps, k), generator=g)
    timeline = {"controller": torch.zeros_like(ids),
                "acceleration_mps2": torch.randn(b, steps, k, generator=g),
                "target_speed_mps": 7 + 4 * torch.rand(b, steps, k, generator=g)}
    outcomes = []
    for flag in ("0", "1"):
        monkeypatch.setenv("WMS_SPEED_FAST_CAV_DRIVE", flag)
        ctrl = TimelineControllers(_StubModel(), timeline)
        ctrl.ids = ids.numpy()  # inject stateful dispatch with controlled external proposals
        events = []
        ctrl.pi = SimpleNamespace(invalidate=lambda **kw: events.append(
            ("invalidate", sorted(kw["vehicle_ids"]), kw["native_ids"])))

        def command(action, vid, row, slot, control, state):
            events.append((action, vid, control, float(state["v_mps"][row, slot])))
            if action == "pi-external-u":
                ctrl.pi_active.add(vid)
            ctrl.damping[vid] = control
            return .6 if action == "damping" else -.8

        ctrl._new_command = command
        gen = torch.Generator().manual_seed(82)
        rows = []
        for _ in range(2 * steps):
            state = {"s_m": torch.rand(b, k, generator=gen) * 100,
                     "v_mps": torch.rand(b, k, generator=gen) * 20,
                     "lane": torch.randint(-1, 3, (b, k), generator=gen),
                     "present": torch.rand(b, k, generator=gen) < .8,
                     "is_cav": torch.rand(b, k, generator=gen) < .8,
                     "a_mps2": torch.zeros(b, k)}
            rows.append(ctrl(state))
        outcomes.append((torch.stack(rows), events, dict(ctrl.active), dict(ctrl.damping),
                         set(ctrl.pi_active)))
    assert torch.equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1:] == outcomes[1][1:]


# ------------------------------------------------------------- fast_merge_lane
def test_fast_merge_lane_is_bit_identical(monkeypatch):
    from wms.control.bridge import _merge_lane
    g = torch.Generator().manual_seed(12)
    b, k = 5, 40
    for trial in range(6):
        s = torch.rand(b, k, generator=g) * 120 - 30
        s[:, :6] = torch.linspace(0, 70, 6)                    # dense ramp candidates
        state = {"s_m": s, "lane": torch.randint(0, 3, (b, k), generator=g),
                 "present": torch.rand(b, k, generator=g) < 0.8,
                 "length_m": 4 + 2 * torch.rand(b, k, generator=g)}
        state["lane"][:, :6] = 2
        if trial == 0:
            state["lane"][:] = 0                                # no candidate at all
        cav = torch.rand(b, k, generator=g) < 0.7
        monkeypatch.delenv("WMS_SPEED_FAST_MERGE_LANE", raising=False)
        ref = _merge_lane(state, cav)
        monkeypatch.setenv("WMS_SPEED_FAST_MERGE_LANE", "1")
        assert torch.equal(ref, _merge_lane(state, cav))


# --------------------------------------------------------- training likelihood
def test_skipping_rollout_diagnostics_preserves_hybrid_training_nll_and_gradients(monkeypatch):
    _fixture = _helpers("test_merge_pivot_training.py")._fixture
    from wms.train.losses.base import LossContext
    from wms.train.losses.merge_pivot import MergePivotConfig, MergePivotTerm

    results = []
    original_quad = pivot.hybrid_log_masses
    for flag in ("0", "1"):
        monkeypatch.setenv("WMS_SPEED_PIVOT_DIAG_OFF", flag)
        model, inp, targets = _fixture("hybrid")
        term = MergePivotTerm(MergePivotConfig(tf_weight=1., weight=.1, passage_law="hybrid"))
        term.prepare(targets)
        # A censored, eligible sequence has a finite, nonzero likelihood and gradients.
        prep = targets.prepared["merge_pivot"]
        prep["event"].zero_()
        for side in ("f", "l"):
            prep["geometry"][f"gap_{side}"][:, :, 0] = torch.tensor([10., 15., 20., 25.])
        torch.manual_seed(33)
        pred = model.rollout(inp.initial, n_steps=4)
        value = term(LossContext(pred=pred, current=inp.initial, targets=targets, c0=0, c1=4)).value
        grads = torch.autograd.grad(value, tuple(model.merge_pivot.parameters()), allow_unused=True)
        assert torch.isfinite(value) and value > 0
        assert any(g is not None and g.abs().any() for g in grads)
        results.append((value, grads, torch.get_rng_state()))
    assert pivot.hybrid_log_masses is original_quad  # likelihood integrator was never bypassed
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][2], results[1][2])
    for ga, gb in zip(results[0][1], results[1][1]):
        assert (ga is None and gb is None) or torch.equal(ga, gb)


def _repeat_input(inp):
    return SimpleNamespace(**{key: (val.repeat(2, *([1] * (val.ndim - 1)))
                                    if torch.is_tensor(val) and val.shape[0] == 1 else deepcopy(val))
                              for key, val in vars(inp).items()})


@pytest.mark.parametrize("long_history", [False, True])
@pytest.mark.parametrize("commands", [False, True])
def test_batched_ragged_prefix_replay_matches_certification_and_geometry(
        monkeypatch, long_history, commands):
    import numpy as np
    _fixture = _helpers("test_merge_pivot_pre_window.py")._fixture
    from wms.train.losses.merge_pivot import MergePivotConfig, build_pre_window_history

    model, inp, _ = _fixture()
    if long_history:
        model = WmSModel(replace(model.spec, history_s=5.), physics=model.physics).eval()
    inp = _repeat_input(inp)
    prefix = inp.merge_pivot_prefix[0]
    if commands:
        prefix["is_cav"][:, 1] = True
        ids = np.zeros((2, 3), np.int64)
        ids[:, 1] = 3
        prefix["action_timeline"] = dict(controller=ids, acceleration_mps2=np.full((2, 3), -2.),
                                         target_speed_mps=np.full((2, 3), 2.), control_dt_s=1.)
    shorter = {key: val[-(2 if key == "continue" else 3):, :2].copy()
               for key, val in prefix.items() if key != "action_timeline"}
    if commands:
        shorter["action_timeline"] = {key: (val if key == "control_dt_s" else val[:, :2].copy())
                                      for key, val in prefix["action_timeline"].items()}
    inp.merge_pivot_prefix = [prefix, shorter]
    ref = build_pre_window_history(model, inp, MergePivotConfig())
    calls = []
    forward = model.forward

    def observe(state, *args, **kwargs):
        calls.append(state["s_m"].shape[0])
        return forward(state, *args, **kwargs)

    monkeypatch.setattr(model, "forward", observe)
    monkeypatch.setenv("WMS_SPEED_BATCH_REPLAY", "1")
    got = build_pre_window_history(model, inp, MergePivotConfig())
    assert 2 in calls and len(calls) == 3
    for a, b in zip(ref, got):
        assert a is not None and b is not None
        assert a["n_steps"] == b["n_steps"]
        for key in a:
            if key == "geometry":
                for name in a[key]:
                    torch.testing.assert_close(a[key][name], b[key][name], atol=1e-5, rtol=1e-5)
            elif torch.is_tensor(a[key]):
                assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize("second_scene", [False, True])
def test_batched_window_scene_replay_and_hooks(monkeypatch, second_scene):
    _fixture = _helpers("test_merge_pivot_training.py")._fixture
    from wms.train.losses.merge_pivot import build_recorded_geometry

    model, inp, targets = _fixture("hybrid")
    inp = _repeat_input(inp)
    scene = {key: getattr(inp, key)[0].clone() for key in ("states", "present", "is_cav")}
    scene["continuity"] = torch.ones(4, 3, dtype=torch.bool)
    scene["continuity"][1, 0] = False
    scene["action_timeline"] = dict(controller=torch.zeros(2, 3, dtype=torch.long),
                                     acceleration_mps2=torch.zeros(2, 3),
                                     target_speed_mps=torch.zeros(2, 3), control_dt_s=1.)
    inp.merge_pivot_scene = [scene, deepcopy(scene) if second_scene else None]
    ref = build_recorded_geometry(model, inp, targets)
    monkeypatch.setenv("WMS_SPEED_BATCH_REPLAY", "1")
    got = build_recorded_geometry(model, inp, targets)
    for key in ref:
        torch.testing.assert_close(ref[key], got[key], atol=1e-5, rtol=1e-5)
    calls = []
    inp.replay_observer = lambda i, state, *args: calls.append(state["s_m"].shape[0])
    build_recorded_geometry(model, inp, targets)
    assert calls == [1] * 8  # observers keep their original per-scene semantics


def test_batch_replay_with_training_dropout_uses_legacy_calls(monkeypatch):
    _fixture = _helpers("test_merge_pivot_training.py")._fixture
    from wms.train.losses.merge_pivot import _batch_replay_enabled
    model, _, _ = _fixture()
    model.add_module("test_dropout", torch.nn.Dropout(.2))
    model.train()
    monkeypatch.setenv("WMS_SPEED_BATCH_REPLAY", "1")
    assert not _batch_replay_enabled(model)
    model.eval()
    assert _batch_replay_enabled(model)


def test_batched_replay_preserves_hybrid_likelihood_and_parameter_gradients(monkeypatch):
    from wms.train.losses.merge_pivot import build_recorded_geometry, score_chunk
    model, inp, targets = _helpers("test_merge_pivot_training.py")._fixture("hybrid")
    inp = _repeat_input(inp)
    scene = {key: getattr(inp, key)[0].clone() for key in ("states", "present", "is_cav")}
    scene["continuity"] = torch.ones(4, 3, dtype=torch.bool)
    scene["action_timeline"] = dict(controller=torch.zeros(2, 3, dtype=torch.long),
                                     acceleration_mps2=torch.zeros(2, 3),
                                     target_speed_mps=torch.zeros(2, 3))
    inp.merge_pivot_scene = [scene, deepcopy(scene)]
    results = []
    for flag in ("0", "1"):
        monkeypatch.setenv("WMS_SPEED_BATCH_REPLAY", flag)
        geo = build_recorded_geometry(model, inp, targets)
        c = model.merge_pivot(geo)
        params = model.merge_pivot.params()
        risk = geo["ramp"]
        nll, _, _ = score_chunk(c, params["tau"], risk, torch.zeros_like(risk),
                                 known_entry=torch.ones_like(risk), reset=torch.zeros_like(risk),
                                 eligible=geo["eligible"], w=params["w"], quadrature_order=8)
        grads = torch.autograd.grad(nll, tuple(model.merge_pivot.parameters()))
        assert torch.isfinite(nll) and nll > 0
        results.append((nll, grads))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-6, rtol=1e-5)
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------- policy update
@pytest.mark.parametrize("checkpoint", [False, True])
def test_thinned_actor_audit_preserves_updates_optimizers_and_rng(monkeypatch, checkpoint):
    D = _helpers("unit/test_c4_dreamer.py")
    results = []
    for period in ("1", "3"):
        monkeypatch.setenv("WMS_SPEED_GRAD_AUDIT_EVERY", period)
        D._seed(43)
        trainer = D._trainer(horizon=2, ensemble=2, actor_gradient_audit=True,
                             uncertainty_weight=.5, checkpoint_imagination=checkpoint,
                             reward_fn=D.hdv_reward, entropy_weight=.1)
        metrics = [trainer.update() for _ in range(4)]
        results.append((trainer, metrics, torch.get_rng_state()))
    a, b = results
    assert ["actor_grad_norms" in row for row in b[1]] == [True, False, False, True]
    for left, right in zip(a[1], b[1]):
        for key in ("actor_grad_norms", "wall_time_s"):
            left.pop(key, None)
            right.pop(key, None)
        assert left == right
    for name in ("policy", "group_encoder", "critic", "target_critic"):
        D._assert_equal(getattr(a[0], name).state_dict(), getattr(b[0], name).state_dict())
    for name in ("actor_optimizer", "critic_optimizer"):
        D._assert_equal(getattr(a[0], name).state_dict(), getattr(b[0], name).state_dict())
    assert torch.equal(a[2], b[2])
