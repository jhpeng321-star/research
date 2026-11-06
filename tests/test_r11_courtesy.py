"""R11 structural courtesy fix: contract checks on synthetic states (CPU, no eval data)."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from wms.cli.assemble import build_model, build_physics
from wms.config.load import load_raw, load_train_config
from wms.config.train import TrainConfig
from wms.model.checkpoint import save_checkpoint
from wms.model.follower_courtesy import (COURTESY_PAIR_F, COURTESY_V2_F, ZONE_STATE_KEY,
                                         FollowerCourtesyHead, expand_head_inputs,
                                         follower_courtesy_pairs, initial_zone_time,
                                         update_zone_time)
from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.losses.compose import build_loss
from wms.train.losses.cutin_reaction import (CutinReactionConfig, CutinReactionLoss, insertion_plan,
                                             one_step_replay, recorded_partnered)
from wms.train.losses.follower_courtesy import (MATCH_F, REPLAY_KEYS, SHARED_KEY,
                                                FollowerCourtesyConfig, FollowerCourtesyLoss,
                                                matched_control_residual, matching_features)
from wms.train.settings import TrainSettings
from wms.train.trainer import WmSTrainer, build_optimizer
from wms.train.unroll import UnrollRunner

from test_r10b_courtesy import context, sequence, state_at_step, stepped, targets

ROOT = Path(__file__).resolve().parents[1]
ROUND10B = ROOT / "configs/c4_wm_v10round10b.toml"
ROUND11 = ROOT / "configs/c4_wm_v10round11.toml"
HEAD_PREFIX = "follower_courtesy."


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_v(version=2, randomize=False, **kwargs):
    torch.manual_seed(31)
    spec = ModelSpec(history_hidden=16, k_hidden=8, follower_courtesy=version > 0,
                     follower_courtesy_version=max(version, 1), **kwargs)
    model = WmSModel(spec, hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32)).eval()
    if randomize and model.follower_courtesy is not None:
        torch.manual_seed(5)
        with torch.no_grad():
            for p in model.follower_courtesy.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
    return model


def crowded(n=9):
    """sequence() plus unpartnered lane-1 controls (slots 5-7) so the contrast term has controls."""
    base = sequence(n=n, cav=False)
    states = torch.zeros(1, n, 8, 7)
    states[:, :, :5] = base["states"]
    states[:, :, 5:, 0] = torch.tensor([60., 75., -60.])
    states[:, :, 5:, 4] = torch.tensor([6., 7., 10.])
    states[:, :, 5:, 5] = 0.3
    states[:, :, 5:, 6] = 5.
    states[:, :, 5:, 2] = 1.                                     # lane 1 one-hot
    is_cav = torch.zeros(1, n, 8, dtype=torch.bool)
    return {"states": states, "present": torch.ones(1, n, 8, dtype=torch.bool), "is_cav": is_cav}


# ------------------------------------------------------------------ v2 inputs
def test_v2_features_append_ramp_accel_and_zone_time_and_stay_zero_without_partner():
    batch = sequence()
    kin = batch["states"][:, 0].clone()
    kin[0, 3, 5] = -1.2                                           # partner (slot 3) accel at t
    present = batch["present"][:, 0]
    hist = torch.zeros(1, 5, 4, 6)
    mask = torch.zeros(1, 5, 4, dtype=torch.bool)
    hist[0, 3, -2, 1] = 0.5                                       # accel/3 at t-1 -> 1.5 m/s^2
    mask[0, 3, -2] = True
    zone = torch.tensor([[0., 0., 3., 7.5, 0.]])
    v1 = follower_courtesy_pairs(kin, present, PhysicsConfig())
    v2 = follower_courtesy_pairs(kin, present, PhysicsConfig(), version=2, zone_time_s=zone,
                                 history=hist, history_mask=mask)
    x1, x2 = v1["features"][0], v2["features"][0]
    assert x2.shape == (5, COURTESY_V2_F) and COURTESY_V2_F == COURTESY_PAIR_F + 3
    assert torch.equal(x2[:, :COURTESY_PAIR_F], x1)              # round-10b inputs unchanged (appended)
    assert x2[0, COURTESY_PAIR_F].item() == pytest.approx(-1.2 / 3)
    assert x2[0, COURTESY_PAIR_F + 1].item() == pytest.approx(1.5 / 3)
    assert x2[0, COURTESY_PAIR_F + 2].item() == pytest.approx(float(torch.asinh(torch.tensor(0.75))))
    assert torch.equal(x2[1:], torch.zeros_like(x2[1:]))
    with pytest.raises(ValueError):
        follower_courtesy_pairs(kin, present, PhysicsConfig(), version=3)


def test_zone_time_counter_is_causal_geometric_and_resets():
    phys = PhysicsConfig()
    kin = sequence()["states"][:, 0].clone()                      # ramp cars at s 40 / 20 m (in zone)
    present = torch.ones(1, 5, dtype=torch.bool)
    z = update_zone_time(None, kin, present, phys)
    assert z[0].tolist() == [0., 0., 0.5, 0.5, 0.]
    z = update_zone_time(z, kin, present, phys)
    assert z[0].tolist() == [0., 0., 1.0, 1.0, 0.]
    kin2 = kin.clone()
    kin2[0, 2, 1:4] = torch.tensor([0., 1., 0.])                  # slot 2 merged -> reset
    kin2[0, 3, 0] = -10.                                          # slot 3 upstream of the zone -> reset
    assert update_zone_time(z, kin2, present, phys)[0].tolist() == [0.] * 5
    assert update_zone_time(z, kin, torch.zeros_like(present), phys)[0].tolist() == [0.] * 5
    # pre-t0 history: 3 observed zone samples (stride 2 steps = 1 s each) then t0 -> 3.5 s
    hist = torch.zeros(1, 5, 4, 9)
    mask = torch.ones(1, 5, 4, dtype=torch.bool)
    hist[..., 3] = 1.                                             # lane 1 one-hot (index 2:5 -> lane 1)
    hist[0, 3, :, 2:5] = torch.tensor([0., 0., 1.])               # slot 3 in lane 2, at t0's position
    init = initial_zone_time(present, kin, phys, hist, mask, stride_steps=2)
    assert init[0, 3].item() == pytest.approx(3.5) and init[0, 0].item() == 0.0


def test_v2_inputs_are_current_state_only_no_leakage():
    model = model_v(randomize=True)
    t = targets(sequence())
    _, first = one_step_replay(model, t, collect=REPLAY_KEYS)
    assert first["courtesy_inputs"].shape[-1] == COURTESY_V2_F
    modified = targets(sequence())
    modified.states[:, 5:, :, 4:6] += 2.
    modified.states[:, 5:, :, 0] += 3.
    _, second = one_step_replay(model, modified, collect=REPLAY_KEYS)
    for key in REPLAY_KEYS:
        assert torch.equal(first[key][:, :5], second[key][:, :5]), key


def test_replay_and_rollout_carry_the_same_zone_counter():
    """One code path: the rollout's carried zone time on held states equals the replay's on recorded rows."""
    model = model_v(randomize=True)
    batch = sequence(n=5, cav=False)
    state = state_at_step(batch, 0)
    roll = model.rollout(dict(state), n_steps=3)
    zone = roll["state"][ZONE_STATE_KEY]
    from wms.physics.state import state_kin
    kin = state_kin(roll["state"])
    inside = ((kin[..., 1:4].argmax(-1) == 2) & roll["state"]["present"].bool()
              & (kin[..., 0] >= 0) & (kin[..., 0] <= 75.44))
    assert torch.isfinite(zone).all() and torch.all(zone[~inside] == 0)
    assert torch.all(zone[inside] > 0)
    t = targets(sequence())
    _, out = one_step_replay(model, t, collect=REPLAY_KEYS)
    zone_input = out["courtesy_inputs"][0, :, 0, COURTESY_PAIR_F + 2]   # follower 0's partner (slot 3)
    expect = torch.asinh(torch.arange(1, t.n_steps + 1) * 0.5 / 10.0)
    assert torch.allclose(zone_input, expect)


# ------------------------------------------------------------------ bit identity / exact zero
def test_version1_model_is_unchanged_and_has_no_zone_state():
    a, b = model_v(version=1), model_v(version=1)
    assert a.follower_courtesy.n_in == COURTESY_PAIR_F and not a._tracks_courtesy_zone
    sa = a.state_dict()
    assert all(torch.equal(sa[k], v) for k, v in b.state_dict().items())
    batch = sequence(cav=False)
    roll = a.rollout(dict(state_at_step(batch, 0)), n_steps=2)
    assert ZONE_STATE_KEY not in roll["state"]


def test_flag_off_defaults_are_bit_identical_to_round10b_losses():
    model = model_v(version=1, randomize=True)
    t1, t2 = targets(crowded()), targets(crowded())
    plain = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))(context(model, t1)).value
    explicit = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0, contrast=False))(context(model, t2)).value
    assert torch.equal(plain.detach(), explicit.detach())
    p1 = insertion_plan(targets(crowded()), CutinReactionConfig(weight=1.0))
    p2 = insertion_plan(targets(crowded()), CutinReactionConfig(weight=1.0, exclude_partnered_controls=False))
    assert torch.equal(p1["control"], p2["control"]) and p1["n_partnered_controls_excluded"] == 0
    s = TrainSettings()
    assert (s.follower_courtesy_version, s.follower_courtesy_contrast, s.follower_courtesy_init_zero,
            s.cutin_reaction_exclude_partnered_controls) == (1, False, False, False)


def test_v2_head_is_exact_noop_at_init_and_zero_without_partner_and_on_cavs():
    off, on = model_v(version=0), model_v(version=2)
    a, b = off.state_dict(), on.state_dict()
    assert set(b) - set(a) == {k for k in b if k.startswith(HEAD_PREFIX)}
    batch = sequence()
    for t in range(3):
        oa, ob = stepped(off, batch, t), stepped(on, batch, t)
        assert torch.equal(oa["accel_mps2"], ob["accel_mps2"])
    rnd = model_v(randomize=True)
    rnd.load_state_dict({**rnd.state_dict(), **off.state_dict()})
    out = rnd(state_at_step(batch, 0))
    offset = out["courtesy_accel_mps2"][0]
    assert offset[0].abs().item() > 0 and torch.equal(offset[1:], torch.zeros_like(offset[1:]))
    batch["states"][:, :, 0, 0] = -40.                            # CAV slot 4 becomes the target follower
    o2 = rnd(state_at_step(batch, 0))
    assert not bool(o2["courtesy_applies"][0, 4]) and o2["courtesy_accel_mps2"][0, 4].item() == 0.0


# ------------------------------------------------------------------ contrast term
def test_matched_control_residual_recovers_the_adjacency_effect_on_synthetic_rows():
    torch.manual_seed(0)
    phi_c = torch.rand(4000, 3) * 4
    generic = lambda p: 0.5 * p[:, 0] - 0.3 * p[:, 1]           # state-generic base error  # noqa: E731
    resid_c = generic(phi_c) + 0.05 * torch.randn(4000)
    phi_p = torch.rand(300, 3) * 3 + 0.5
    effect = -0.4
    resid_p = generic(phi_p) + effect
    shift, neff = matched_control_residual(phi_p, phi_c, resid_c)
    target = resid_p - shift
    assert target.mean().item() == pytest.approx(effect, abs=0.08)  # the generic part is removed
    assert (neff > 5).all()
    # no controls: zero shift (falls back to the plain recorded fit)
    s0, n0 = matched_control_residual(phi_p, phi_c[:0], resid_c[:0])
    assert torch.equal(s0, torch.zeros(300)) and torch.equal(n0, torch.zeros(300))
    # no support at all: shrinks to the control mean (one pseudo-row)
    far, _ = matched_control_residual(phi_p + 1000., phi_c, resid_c)
    assert torch.allclose(far, resid_c.mean().expand(300), atol=1e-4)


def test_matching_features_are_the_r9_yield_covariates_and_never_a_model_input():
    t = targets(crowded())
    from wms.train.losses.cutin_reaction import _recorded
    kin, _, valid, _, _, _ = _recorded(t)
    phi = matching_features(kin, valid, FollowerCourtesyConfig(weight=1.0))
    assert phi.shape == (1, t.n_steps, 8, MATCH_F)
    # slot 0 (s 0, v 8) has lane-1 leader slot 5 (s 60, len 5): headway 55/8
    assert phi[0, 0, 0, 0].item() == pytest.approx(0.8)
    assert phi[0, 0, 0, 1].item() == pytest.approx(float(torch.asinh(torch.tensor(55. / 8.))))
    assert phi[0, 0, 0, 2].item() == pytest.approx((8. - 6.) / 5.)
    import inspect
    from wms.model import follower_courtesy as model_side
    assert "matching" not in inspect.getsource(model_side)


def test_contrast_loss_targets_recorded_minus_matched_control_residual_and_trains_only_the_head():
    model = model_v(randomize=True)
    t = targets(crowded())
    term = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0, contrast=True))
    value = term(context(model, t))
    stats = term.last_stats
    assert stats["n_rows"] > 0 and stats["contrast_n_controls"] > 0
    assert all(k in stats for k in ("contrast_shift_mean", "contrast_neff_median", "contrast_control_resid_mean"))
    value.value.backward()
    reached = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert reached and all(n.startswith(HEAD_PREFIX) for n in reached)
    # explicit recomputation of the target
    model.zero_grad(set_to_none=True)
    _, out = one_step_replay(model, targets(crowded()), collect=REPLAY_KEYS)
    from wms.train.losses.follower_courtesy import courtesy_rows, lane1_rows
    from wms.train.losses.cutin_reaction import _recorded
    t2 = targets(crowded())
    rows = courtesy_rows(t2, out["courtesy_applies"])
    ctrl = lane1_rows(t2) & ~out["courtesy_applies"]
    kin, _, valid, _, _, _ = _recorded(t2)
    phi = matching_features(kin, valid, FollowerCourtesyConfig(weight=1.0))
    rec = t2.states[:, 1:t2.n_steps + 1, :, 5]
    pre = out["accel_pre_courtesy_mps2"]
    resid = rec - model.follower_courtesy_total(pre, torch.zeros_like(pre))
    shift, _ = matched_control_residual(phi[rows], phi[ctrl], resid[ctrl])
    with torch.no_grad():
        pred = model.follower_courtesy_total(pre[rows], model.follower_courtesy(out["courtesy_inputs"][rows]))
    expected = F.huber_loss(pred, rec[rows] - shift, delta=1.0)
    assert value.log.item() == pytest.approx(expected.item(), rel=1e-5)
    # contrast differs from the plain fit when the controls carry a residual
    plain = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0))(context(model, targets(crowded())))
    assert plain.log.item() != pytest.approx(value.log.item())


@pytest.mark.parametrize("shared", [False, True])
def test_contrast_with_window_longer_than_unroll(shared):
    model = model_v(randomize=True)
    t = targets(crowded(n=9), n=5)
    assert t.n_steps == 5 and t.is_cav.shape[1] == 9
    if shared:
        CutinReactionLoss(CutinReactionConfig(weight=1.0, exclude_partnered_controls=True),
                          share_keys=REPLAY_KEYS, share_to=SHARED_KEY)(context(model, t))
    term = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.0, contrast=True))
    value = term(context(model, t))
    assert term.last_stats["shared_replay"] == int(shared)
    assert term.last_stats["n_rows"] > 0 and torch.isfinite(value.value)


# ------------------------------------------------------------------ fix-B control exclusion
def test_fixb_controls_exclude_partnered_rows_only():
    batch = crowded()
    batch["states"][:, :, 2, 1:4] = torch.tensor([0., 0., 1.])   # no insertion: slot 0 stays a control row
    t = targets(batch)
    base = insertion_plan(t, CutinReactionConfig(weight=1.0))
    excl = insertion_plan(t, CutinReactionConfig(weight=1.0, exclude_partnered_controls=True))
    partnered = recorded_partnered(t, CutinReactionConfig(weight=1.0))
    assert partnered[..., 0].any()                               # slot 0 is the target follower
    assert torch.equal(excl["control"], base["control"] & ~partnered)
    assert excl["n_partnered_controls_excluded"] == int((base["control"] & partnered).sum()) > 0
    assert torch.equal(excl["event_weights"], base["event_weights"])  # event rows unchanged
    # the recorded pairing is the courtesy head's pairing (same function, recorded state)
    from wms.train.losses.cutin_reaction import _recorded
    kin, _, valid, _, _, _ = _recorded(t)
    for step in range(t.n_steps):
        ref = follower_courtesy_pairs(kin[:, step], valid[:, step], PhysicsConfig())["has_partner"]
        assert torch.equal(partnered[:, step], ref)


def test_runner_composes_r11_terms_and_logs_telemetry():
    model = model_v(randomize=True)
    cfg = replace(TrainSettings(truncation_steps=2), cutin_reaction_weight=1.0,
                  cutin_reaction_exclude_partnered_controls=True, follower_courtesy=True,
                  follower_courtesy_version=2, follower_courtesy_weight=0.5,
                  follower_courtesy_contrast=True)
    record = UnrollRunner(model, cfg, build_loss(cfg), projection=None, device=torch.device("cpu")).run(
        crowded(n=12), unroll_steps=6, backward=True)
    assert record["follower_courtesy_n_rows"] > 0 and record["follower_courtesy_contrast_n_controls"] > 0
    assert record["cutin_reaction_n_partnered_controls_excluded"] >= 0
    assert record["follower_courtesy_shared_replay"] == 1
    with pytest.raises(ValueError):
        TrainSettings(follower_courtesy_contrast=True)
    with pytest.raises(ValueError):
        TrainSettings(follower_courtesy_version=2)


@pytest.mark.parametrize("rollout_grad", [False, True])
def test_v2_gradient_scope_matches_round10b(rollout_grad):
    model = model_v(randomize=True, follower_courtesy_rollout_grad=rollout_grad)
    t = targets(sequence())
    CutinReactionLoss(CutinReactionConfig(weight=1.0, exclude_partnered_controls=True))(
        context(model, t)).value.backward()
    head = [p.grad for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    assert any(g is not None and g.abs().sum() > 0 for g in head) == rollout_grad


# ------------------------------------------------------------------ init: expand / zero switch
def test_round10b_head_zero_expands_to_the_identical_function():
    torch.manual_seed(3)
    old = FollowerCourtesyHead(version=1)
    with torch.no_grad():
        for p in old.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    new = FollowerCourtesyHead(version=2)
    src = {HEAD_PREFIX + k: v for k, v in old.state_dict().items()}
    own = {HEAD_PREFIX + k: v for k, v in new.state_dict().items()}
    grown = expand_head_inputs(src, own)
    new.load_state_dict({k[len(HEAD_PREFIX):]: v for k, v in grown.items()})
    x = torch.randn(50, COURTESY_V2_F)
    assert torch.allclose(new(x), old(x[:, :COURTESY_PAIR_F]), atol=1e-6)
    assert expand_head_inputs(own, own) is own                    # same shape: untouched


@pytest.mark.parametrize("zero", [False, True])
def test_trainer_init_loads_expanded_round10b_head_or_zero_head(tmp_path, zero):
    base = {"rollout_mode": "hybrid", "follower_courtesy": True}
    cfg10b = TrainConfig.from_dict({"model": base})
    cfg11 = TrainConfig.from_dict({"model": {**base, "follower_courtesy_version": 2,
                                             "follower_courtesy_init_zero": zero}})
    physics = PhysicsConfig()
    torch.manual_seed(0)
    m10b = WmSModel(ModelSpec.from_mapping(cfg10b.to_legacy_flat()), physics=physics)
    with torch.no_grad():
        for p in m10b.follower_courtesy.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    init = tmp_path / "r10b.pt"
    save_checkpoint(init, m10b, cfg10b)
    torch.manual_seed(1)
    m11 = WmSModel(ModelSpec.from_mapping(cfg11.to_legacy_flat()), physics=physics)
    fake = SimpleNamespace(model=m11, settings=TrainSettings.from_config(cfg11), out_dir=tmp_path / "out")
    report = WmSTrainer._init_from_checkpoint(fake, str(init))
    head_keys = sorted(k for k in m11.state_dict() if k.startswith(HEAD_PREFIX))
    assert not report["shape_mismatch"] and not report["unused_from_checkpoint"]
    if zero:
        assert report["follower_courtesy_init"] == "zero" and report["new_in_model"] == head_keys
        assert m11.follower_courtesy.net[-1].weight.abs().sum() == 0
    else:
        assert report["follower_courtesy_init"] == "checkpoint" and report["new_in_model"] == []
        assert report["expanded"] == [HEAD_PREFIX + "net.0.weight"]
        x = torch.randn(20, COURTESY_V2_F)
        assert torch.allclose(m11.follower_courtesy(x), m10b.follower_courtesy(x[:, :COURTESY_PAIR_F]), atol=1e-6)
    assert json.loads((tmp_path / "out" / "INIT_FROM.json").read_text())["follower_courtesy_init"] == report[
        "follower_courtesy_init"]


# ------------------------------------------------------------------ config pins
def leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1], node


def test_round11_differs_from_round10b_only_by_the_structural_fix():
    a, b = dict(leaves(load_raw(ROUND10B))), dict(leaves(load_raw(ROUND11)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    allowed = {"name", "runtime.out_dir", "model.init_from", "model.follower_courtesy_version",
               "model.follower_courtesy_init_zero", "loss.follower_courtesy.contrast",
               "loss.cutin_reaction.exclude_partnered_controls", "loss.follower_courtesy.weight",
               "optim.lr_scale.follower_courtesy"}
    assert changed <= allowed and {"model.follower_courtesy_version", "loss.follower_courtesy.contrast",
                                   "loss.cutin_reaction.exclude_partnered_controls",
                                   "model.init_from"} <= changed, sorted(changed)
    assert b["model.follower_courtesy_version"] == 2 and b["model.follower_courtesy_init_zero"] is False
    assert b["model.init_from"].endswith("runs/c4_wm_v10_round10b/wm_s_stage300.pt")
    for key, value in (("model.merge_pivot_state_st", True), ("model.checkpoint_forward", True),
                       ("model.merge_passage_law", "hybrid"), ("loss.merge_core.weight", 0.0),
                       ("loss.merge_core.tf_weight", 0.6), ("physics.ramp_lane_end_stop", True),
                       ("physics.ramp_risk_retain", True), ("model.follower_courtesy_rollout_grad", False),
                       ("loss.cutin_reaction.weight", 2.174)):
        assert b[key] == value, key


def test_round11_trainable_set_and_loss_composition():
    cfg = load_train_config(ROUND11)
    settings = TrainSettings.from_config(cfg)
    assert settings.follower_courtesy_contrast and settings.cutin_reaction_exclude_partnered_controls
    names = build_loss(settings).names
    assert names[-2:] == ("cutin_reaction", "follower_courtesy")
    model = build_model(cfg, build_physics(cfg))
    assert model.follower_courtesy.version == 2 and model._tracks_courtesy_zone
    _, report = build_optimizer(model, lr=1e-4, weight_decay=0, freeze_prefixes=tuple(cfg.optim.freeze_prefixes),
                                lr_scale=dict(cfg.optim.lr_scale))
    assert report["unmatched_freeze_prefixes"] == [] and report["unmatched_lr_scale_prefixes"] == []
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.startswith(("temporal", "gate_g_max_logit", "gate_tau_raw", HEAD_PREFIX)) for n in trainable)
    assert {n for n in trainable if n.startswith(HEAD_PREFIX)}


def test_round11_weight_and_lr_scale_come_from_the_documented_rule_file():
    rule = json.loads((ROOT / "docs/r11_weight_rule.json").read_text(encoding="utf-8"))
    cfg = load_train_config(ROUND11)
    assert rule["status"] == "MEASURED"
    assert cfg.loss.follower_courtesy.weight == pytest.approx(rule["chosen_weight"], rel=1e-9)
    assert cfg.optim.lr_scale["follower_courtesy"] == pytest.approx(rule["lr_scale_chosen"], rel=1e-9)


def test_round11_gate_script_is_launchable():
    gate = (ROOT / "scripts/round11_gate.sh").read_bytes()
    assert b"\r" not in gate and b"configs/c4_wm_v10round11.toml" in gate
    assert b"runs/c4_wm_v10_round11" in gate and b"repo_r11" in gate
    assert b"selection.val_max_windows=4" in gate and b"selection.val_max_windows=16" in gate
    assert b"curriculum.max_hours=1.5" in gate and b"wms.train.round11_gate" in gate
    assert b"R11_INIT" in gate and b"R11_HEAD_INIT" in gate and b"set -u -o pipefail -o noclobber" in gate
    assert b"runs/c4_wm_v10_round10b/wm_s_stage300.pt" in gate


def test_legacy_flat_round_trip():
    flat = load_train_config(ROUND11).to_legacy_flat()
    assert flat["follower_courtesy_version"] == 2 and flat["follower_courtesy_contrast"] is True
    assert flat["cutin_reaction_exclude_partnered_controls"] is True
    again = TrainConfig.from_legacy_flat(flat)
    assert again.model.follower_courtesy_version == 2 and again.loss.follower_courtesy.contrast is True
    assert not any(k in load_train_config(ROUND10B).to_legacy_flat() for k in (
        "follower_courtesy_version", "follower_courtesy_contrast", "cutin_reaction_exclude_partnered_controls"))
    assert ModelSpec.from_mapping(flat).follower_courtesy_version == 2
