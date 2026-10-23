"""R10B learned pre-merge follower courtesy: contract checks on synthetic states (CPU, no eval data)."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wms.cli.assemble import build_model, build_physics
from wms.config.load import load_raw, load_train_config
from wms.config.train import TrainConfig
from wms.model import checkpoint as ckpt_module
from wms.model.follower_courtesy import COURTESY_PAIR_F, follower_courtesy_pairs
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.inputs import build_targets, prepare_inputs
from wms.train.losses.base import UnrollContext
from wms.train.losses.compose import build_loss
from wms.train.losses.cutin_reaction import (CutinReactionConfig, CutinReactionLoss,
                                             one_step_replay)
from wms.train.losses.follower_courtesy import (REPLAY_KEYS, SHARED_KEY, FollowerCourtesyConfig,
                                                FollowerCourtesyLoss, courtesy_rows)
from wms.train.settings import TrainSettings
from wms.train.trainer import build_optimizer
from wms.train.unroll import UnrollRunner

ROOT = Path(__file__).resolve().parents[1]
ROUND10 = ROOT / "configs/c4_wm_v10round10.toml"
ROUND10B = ROOT / "configs/c4_wm_v10round10b.toml"
HEAD_PREFIX = "follower_courtesy."


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sequence(n=9, cav=True):
    """Slots: 0 lane-1 follower (s 0), 1 lane-1 far ahead (s 100), 2 ramp car merging at t=2 (s 40),
    3 ramp car (s 20), 4 lane-1 CAV behind everything (s -30)."""
    states = torch.zeros(1, n, 5, 7)
    states[..., 0] = torch.tensor([0., 100., 40., 20., -30.])
    states[..., 4] = torch.tensor([8., 12., 2., 1., 9.])
    states[..., 5] = -0.5
    states[..., 6] = 5.
    lanes = torch.tensor([1, 1, 2, 2, 1]).expand(1, n, 5).clone()
    lanes[:, 2:, 2] = 1
    states[..., 1:4] = F.one_hot(lanes, 3).float()
    is_cav = torch.zeros(1, n, 5, dtype=torch.bool)
    is_cav[..., 4] = cav
    return {"states": states, "present": torch.ones(1, n, 5, dtype=torch.bool), "is_cav": is_cav}


def targets(batch, n=None):
    """Trainer targets; ``n`` < window rows reproduces real batches (window longer than the unroll)."""
    class Model:
        long_history = False
        rollout_mode = "residual"
        physics = None
    cfg = TrainSettings()
    n = batch["states"].shape[1] - 1 if n is None else n
    inp = prepare_inputs(batch, Model(), cfg, torch.device("cpu"), n)
    return build_targets(inp, batch, cfg, .5, torch.device("cpu"))


def context(model, t):
    return UnrollContext(model, {"veh_kin": t.states[:, 0], "present": t.loss_valid[:, 0],
                                 "is_cav": t.is_cav[:, 0]}, t.loss_valid[:, 0], t)


def small_model(courtesy=False, randomize=False, **kwargs):
    torch.manual_seed(31)
    spec = ModelSpec(history_hidden=16, k_hidden=8, follower_courtesy=courtesy, **kwargs)
    model = WmSModel(spec, hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32)).eval()
    if randomize and model.follower_courtesy is not None:
        torch.manual_seed(5)
        with torch.no_grad():
            for p in model.follower_courtesy.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
    return model


# ------------------------------------------------------------------ pairing / features
def test_pairing_is_geometric_nearest_owner_and_zero_without_partner():
    batch = sequence()
    kin = batch["states"][:, 0]
    pairs = follower_courtesy_pairs(kin, batch["present"][:, 0], PhysicsConfig())
    has, partner, x = pairs["has_partner"][0], pairs["partner"][0], pairs["features"][0]
    # slot 0 is the target follower of BOTH ramp cars (2 at 40 m, 3 at 20 m): nearest owner = 3
    assert has.tolist() == [True, False, False, False, False]
    assert int(partner[0]) == 3
    assert x.shape == (5, COURTESY_PAIR_F)
    assert torch.equal(x[1:], torch.zeros_like(x[1:]))          # no partner -> exactly zero inputs
    gap = 20. - 5. - 0.
    assert x[0, 0].item() == pytest.approx(float(torch.asinh(torch.tensor(gap / 5.))))
    # a lane-1 car between the follower and the ramp car takes over as the target follower
    kin2 = kin.clone()
    kin2[0, 1, 0] = 10.
    p2 = follower_courtesy_pairs(kin2, batch["present"][:, 0], PhysicsConfig())
    assert p2["has_partner"][0].tolist() == [False, True, False, False, False]
    assert int(p2["partner"][0, 1]) == 3


def test_inputs_are_current_state_only_and_never_future_rows():
    model = small_model(courtesy=True, randomize=True)
    t = targets(sequence())
    _, first = one_step_replay(model, t, collect=REPLAY_KEYS)
    modified = targets(sequence())
    modified.states[:, 5:, :, 4:6] += 2.
    modified.states[:, 5:, 0, 0] += 3.
    _, second = one_step_replay(model, modified, collect=REPLAY_KEYS)
    for key in REPLAY_KEYS:
        assert torch.equal(first[key][:, :5], second[key][:, :5]), key
    # the pairing function has no time axis: it only receives the state at t
    import inspect
    params = inspect.signature(follower_courtesy_pairs).parameters
    assert list(params)[:3] == ["kin", "present", "physics"]
    assert not any("future" in name or "next" in name for name in params)


# ------------------------------------------------------------------ bit identity / exact zero
def test_flag_off_is_bit_identical_and_creates_no_tensor():
    default = small_model()
    explicit = small_model(courtesy=False)
    assert default.follower_courtesy is None and explicit.follower_courtesy is None
    a, b = default.state_dict(), explicit.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert not any(k.startswith(HEAD_PREFIX) for k in a)
    batch = sequence()
    for t in range(3):
        oa, ob = stepped(default, batch, t), stepped(explicit, batch, t)
        assert torch.equal(oa["accel_mps2"], ob["accel_mps2"])
        assert torch.equal(oa["next_s_m"], ob["next_s_m"])
        assert not any(key.startswith("courtesy_") for key in oa)


def stepped(model, batch, t):
    state = state_at_step(batch, t)
    return model.step(state, cav_teacher=model.hold_teacher(state))


def state_at_step(batch, t):
    from wms.train.inputs import state_from_kin
    return {**state_from_kin(batch["states"][:, t], batch["present"][:, t]),
            "is_cav": batch["is_cav"][:, t]}


def test_enabled_head_is_an_exact_noop_at_init_and_keeps_every_other_tensor():
    off, on = small_model(), small_model(courtesy=True)
    a, b = off.state_dict(), on.state_dict()
    assert set(b) - set(a) == {k for k in b if k.startswith(HEAD_PREFIX)}
    assert all(torch.equal(a[k], b[k]) for k in a)               # same RNG stream for the rest
    batch = sequence()
    for t in range(3):
        oa, ob = stepped(off, batch, t), stepped(on, batch, t)
        assert torch.equal(oa["accel_mps2"], ob["accel_mps2"])
        assert torch.equal(oa["next_s_m"], ob["next_s_m"]) and torch.equal(oa["next_v_mps"], ob["next_v_mps"])
        assert torch.equal(ob["courtesy_accel_mps2"], torch.zeros_like(ob["courtesy_accel_mps2"]))


def test_offset_is_exactly_zero_without_partner_and_for_cavs():
    off, on = small_model(), small_model(courtesy=True, randomize=True)
    on.load_state_dict({**on.state_dict(), **off.state_dict()})
    batch = sequence()
    state = state_at_step(batch, 0)
    oa, ob = off(state), on(state)
    offset = ob["courtesy_accel_mps2"][0]
    assert offset[0].abs().item() > 0                           # the target follower moves
    assert torch.equal(offset[1:], torch.zeros_like(offset[1:]))
    assert torch.equal(oa["accel_mps2"][0, 1:], ob["accel_mps2"][0, 1:])
    # a CAV in the target-follower position never receives the offset
    batch["states"][:, :, 0, 0] = -40.                          # CAV slot 4 (s -30) is now the follower
    s2 = state_at_step(batch, 0)
    o2 = on(s2)
    assert not bool(o2["courtesy_applies"][0, 4]) and o2["courtesy_accel_mps2"][0, 4].item() == 0.0
    assert follower_courtesy_pairs(s2["veh_kin"], s2["present"], PhysicsConfig())["has_partner"][0, 4]


def test_no_residual_clamp_on_the_head_output():
    on = small_model(courtesy=True)
    with torch.no_grad():
        on.follower_courtesy.net[-1].bias.fill_(-1.7)
    out = on(state_at_step(sequence(), 0))
    assert out["courtesy_accel_mps2"][0, 0].item() == pytest.approx(-1.7)
    pre = out["accel_pre_courtesy_mps2"][0, 0]
    expected = (pre - 1.7).clamp(on.physics.accel_min_mps2, on.physics.accel_max_mps2)
    assert out["accel_mps2"][0, 0].item() == pytest.approx(expected.item())


# ------------------------------------------------------------------ loss and gradient scope
def test_courtesy_loss_rows_value_and_gradient_only_on_the_head():
    model = small_model(courtesy=True, randomize=True)
    t = targets(sequence())
    term = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=2.0))
    value = term(context(model, t))
    assert term.last_stats["n_rows"] > 0 and term.last_stats["shared_replay"] == 0
    assert torch.isfinite(value.value) and value.value.item() > 0
    assert value.value.item() == pytest.approx(2.0 * value.log.item())
    value.value.backward()
    reached = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert reached and all(n.startswith(HEAD_PREFIX) for n in reached)
    # rows: lane-1 HDV transitions whose follower has a partner (slot 0 only; slot 4 is a CAV)
    _, out = one_step_replay(model, t, collect=REPLAY_KEYS)
    rows = courtesy_rows(t, out["courtesy_applies"])
    assert rows[..., 0].all() and not rows[..., 1:].any()


def test_zero_init_head_gets_a_gradient_and_weight_zero_is_off():
    model = small_model(courtesy=True)
    t = targets(sequence())
    value = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))(context(model, t)).value
    value.backward()
    assert model.follower_courtesy.net[-1].weight.grad.abs().sum() > 0
    off = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=0.0))(context(model, t))
    assert off.value.item() == 0 and off.valid_count == 0
    with torch.no_grad():
        nograd = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))(context(model, t))
    assert nograd.value.item() == 0


def test_shared_cutin_replay_gives_the_identical_courtesy_loss():
    model = small_model(courtesy=True, randomize=True)
    t = targets(sequence())
    own = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))(context(model, t)).value
    t2 = targets(sequence())
    cut = CutinReactionLoss(CutinReactionConfig(weight=1.0), share_keys=REPLAY_KEYS, share_to=SHARED_KEY)
    cut(context(model, t2))
    assert SHARED_KEY in t2.prepared
    term = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))
    shared = term(context(model, t2)).value
    assert term.last_stats["shared_replay"] == 1
    assert torch.equal(own.detach(), shared.detach())


@pytest.mark.parametrize("rollout_grad", [False, True])
def test_cutin_and_rollout_losses_reach_the_head_only_when_rollout_grad(rollout_grad):
    model = small_model(courtesy=True, randomize=True, follower_courtesy_rollout_grad=rollout_grad)
    t = targets(sequence())
    CutinReactionLoss(CutinReactionConfig(weight=1.0))(context(model, t)).value.backward()
    head = [p.grad for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    reached = any(g is not None and g.abs().sum() > 0 for g in head)
    assert reached == rollout_grad
    # whole runner (position/speed losses + backward): same rule for the head weights
    model.zero_grad(set_to_none=True)
    cfg = TrainSettings(truncation_steps=2)
    UnrollRunner(model, cfg, build_loss(cfg), projection=None,
                 device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
    head = [p.grad for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    assert any(g is not None and g.abs().sum() > 0 for g in head) == rollout_grad
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if not n.startswith(HEAD_PREFIX))


def test_runner_composes_both_terms_shares_the_replay_and_logs_telemetry():
    model = small_model(courtesy=True, randomize=True)
    cfg = replace(TrainSettings(truncation_steps=2), cutin_reaction_weight=1.0,
                  follower_courtesy=True, follower_courtesy_weight=0.5)
    loss = build_loss(cfg)
    assert loss.names[-2:] == ("cutin_reaction", "follower_courtesy")
    record = UnrollRunner(model, cfg, loss, projection=None,
                          device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=True)
    assert record["follower_courtesy"] > 0 and record["follower_courtesy_n_rows"] > 0
    assert record["follower_courtesy_shared_replay"] == 1
    assert all(k in record for k in ("follower_courtesy_head_mean", "follower_courtesy_base_residual_mean"))
    off = TrainSettings(truncation_steps=2)
    other = UnrollRunner(model, off, build_loss(off), projection=None,
                         device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8, backward=False)
    assert not any(key.startswith("follower_courtesy") for key in other)


def test_loss_without_head_and_weight_without_model_flag_are_refused():
    with pytest.raises(ValueError):
        TrainSettings(follower_courtesy_weight=1.0)
    model = small_model()
    with pytest.raises(RuntimeError):
        FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))(context(model, targets(sequence())))


# ------------------------------------------------------------------ window longer than the unroll
@pytest.mark.parametrize("shared", [False, True])
def test_window_longer_than_unroll_real_shaped_batch(shared):
    """Real batches carry 300 s windows with 120 s unrolls: every mask must be cut to n_steps (+1)."""
    model = small_model(courtesy=True, randomize=True)
    t = targets(sequence(n=9), n=5)
    assert t.n_steps == 5 and t.is_cav.shape[1] == 9 and t.states.shape[1] >= 6
    if shared:
        CutinReactionLoss(CutinReactionConfig(weight=1.0), share_keys=REPLAY_KEYS,
                          share_to=SHARED_KEY)(context(model, t))
        assert t.prepared[SHARED_KEY]["courtesy_applies"].shape[1] == 5
    term = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))
    value = term(context(model, t))
    assert term.last_stats["shared_replay"] == int(shared)
    assert term.last_stats["n_rows"] > 0 and torch.isfinite(value.value)
    full = targets(sequence(n=9))
    _, out_full = one_step_replay(model, full, collect=REPLAY_KEYS)
    _, out_cut = one_step_replay(model, targets(sequence(n=9), n=5), collect=REPLAY_KEYS)
    for key in REPLAY_KEYS:                                     # causal: the cut replay is a prefix
        assert torch.equal(out_cut[key], out_full[key][:, :5]), key


def test_runner_with_window_longer_than_unroll():
    model = small_model(courtesy=True, randomize=True)
    cfg = replace(TrainSettings(truncation_steps=2), cutin_reaction_weight=1.0,
                  follower_courtesy=True, follower_courtesy_weight=0.5)
    record = UnrollRunner(model, cfg, build_loss(cfg), projection=None, device=torch.device("cpu")).run(
        sequence(n=12, cav=False), unroll_steps=6, backward=True)
    assert record["follower_courtesy_n_rows"] > 0 and record["follower_courtesy"] > 0


# ------------------------------------------------------------------ config pins
def leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1], node


def test_round10b_differs_from_round10_only_by_the_courtesy():
    a, b = dict(leaves(load_raw(ROUND10))), dict(leaves(load_raw(ROUND10B)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert changed == {
        "name", "runtime.out_dir", "optim.lr_scale.follower_courtesy",
        "model.follower_courtesy", "model.follower_courtesy_hidden",
        "model.follower_courtesy_embedding", "model.follower_courtesy_rollout_grad",
        "loss.follower_courtesy.weight", "loss.follower_courtesy.huber_delta_mps2"}, sorted(changed)
    assert b["model.follower_courtesy"] is True and b["model.follower_courtesy_embedding"] is False
    assert b["model.follower_courtesy_rollout_grad"] is False
    # round 9's binding decisions and fix B are inherited unchanged
    for key, value in (("model.merge_pivot_state_st", True), ("model.checkpoint_forward", True),
                       ("model.merge_passage_law", "hybrid"), ("loss.merge_core.weight", 0.0),
                       ("loss.merge_core.tf_weight", 0.6), ("physics.ramp_lane_end_stop", True),
                       ("physics.ramp_risk_retain", True)):
        assert a[key] == value and b[key] == value, key
    assert a["model.init_from"] == b["model.init_from"]
    assert b["model.init_from"].endswith("runs/c4_wm_v10_round9/wm_s_stage300.pt")


def test_round10b_loss_composition_and_trainable_set():
    cfg = load_train_config(ROUND10B)
    names10 = set(build_loss(TrainSettings.from_config(load_train_config(ROUND10))).names)
    names10b = set(build_loss(TrainSettings.from_config(cfg)).names)
    assert names10b - names10 == {"follower_courtesy"} and names10 <= names10b
    assert "follower_courtesy" in cfg.required_log_terms()
    model = build_model(cfg, build_physics(cfg))
    assert model.follower_courtesy is not None and not model.follower_courtesy_rollout_grad
    _, report = build_optimizer(model, lr=1e-4, weight_decay=0,
                                freeze_prefixes=tuple(cfg.optim.freeze_prefixes),
                                lr_scale=dict(cfg.optim.lr_scale))
    assert report["unmatched_freeze_prefixes"] == [] and report["unmatched_lr_scale_prefixes"] == []
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    head = {n for n, _ in model.named_parameters() if n.startswith(HEAD_PREFIX)}
    assert head and head <= trainable
    assert all(n.startswith(("temporal", "gate_g_max_logit", "gate_tau_raw", HEAD_PREFIX))
               for n in trainable), sorted(trainable)
    assert not any(n.startswith(("merge_pivot", "merge_core", "driver_", "lane_", "residual_net",
                                 "history_proj", "k_embedding")) for n in trainable)


def test_round10b_weight_and_lr_scale_come_from_the_documented_rule_file():
    rule = json.loads((ROOT / "docs/r10b_weight_rule.json").read_text(encoding="utf-8"))
    cfg = load_train_config(ROUND10B)
    assert rule["status"] == "MEASURED"
    assert cfg.loss.follower_courtesy.weight == pytest.approx(rule["chosen_weight"], rel=1e-9)
    assert cfg.optim.lr_scale["follower_courtesy"] == pytest.approx(rule["lr_scale_chosen"], rel=1e-9)


def test_round10b_gate_script_is_launchable():
    gate = (ROOT / "scripts/round10b_gate.sh").read_bytes()
    assert b"\r" not in gate and b"configs/c4_wm_v10round10b.toml" in gate
    assert b"runs/c4_wm_v10_round10b" in gate and b"repo_r10b" in gate
    assert b"\\n  --set" not in gate and b"\\n --set" not in gate
    assert b"selection.val_max_windows=4" in gate and b"selection.val_max_windows=16" in gate
    assert b"curriculum.max_hours=1.5" in gate and b"wms.train.round10b_gate" in gate
    assert b"R10B_INIT" in gate and b"set -u -o pipefail -o noclobber" in gate


def test_legacy_flat_round_trip_and_default_omission():
    flat = load_train_config(ROUND10B).to_legacy_flat()
    assert flat["follower_courtesy"] is True and flat["follower_courtesy_weight"] > 0
    again = TrainConfig.from_legacy_flat(flat)
    assert again.model.follower_courtesy is True
    assert again.loss.follower_courtesy.weight == flat["follower_courtesy_weight"]
    assert not any(k.startswith("follower_courtesy") for k in load_train_config(ROUND10).to_legacy_flat())
    assert ModelSpec.from_mapping(flat).follower_courtesy is True
    assert HEAD_PREFIX in ckpt_module._OPTIONAL_PREFIXES
