"""R13A courtesy pair-contract repair (STRUCTURE_ADVICE section 2 option A, minimal part).

Synthetic CPU states only (no eval data, no SUMO).  Covers:
* the two round-12 defects (verified with the flags at their round-12 defaults);
* ``model.follower_courtesy_state_grad``: non-zero, finite pair-feature state derivative that agrees
  with central finite differences (fixed seeds), discrete pairing unchanged;
* ``model.follower_courtesy_intent = 'pivot' | 'none'``: causal step-start pivot clearance input;
* ``loss.follower_courtesy.absolute_weight``: added absolute term, default off;
* flags off = identical forward outputs and loss values (round-12 behaviour).
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wms.config.load import load_raw, load_train_config
from wms.model import merge_pivot as pivot
from wms.model.follower_courtesy import INTENT_SOURCES, follower_courtesy_pairs, pivot_clearance_intent
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.train.inputs import state_from_kin
from wms.train.losses.compose import build_loss
from wms.train.losses.follower_courtesy import FollowerCourtesyConfig, FollowerCourtesyLoss
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

from test_r10b_courtesy import context, sequence, state_at_step, targets
from test_r11_courtesy import crowded, leaves, model_v

ROOT = Path(__file__).resolve().parents[1]
ROUND12 = ROOT / "configs/c4_wm_v10round12.toml"
ROUND13A = ROOT / "configs/c4_wm_v10round13a.toml"
HEAD_PREFIX = "follower_courtesy."
INTENT = 6                                                   # COURTESY_PAIR_FIELDS index of "intent"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pivot_model(intent="lane_v5", state_grad=False, randomize=True):
    torch.manual_seed(19)
    model = WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                               lane_decision="sample", merge_decision="first_passage",
                               merge_passage_law="hybrid", follower_courtesy=True,
                               follower_courtesy_version=2, follower_courtesy_intent=intent,
                               follower_courtesy_state_grad=state_grad),
                     physics=PhysicsConfig()).eval()
    if randomize:
        torch.manual_seed(5)
        with torch.no_grad():
            for p in model.follower_courtesy.parameters():
                p.copy_(torch.randn_like(p) * 0.3)
    return model


def pivot_kin(requires_grad=False):
    """0 ramp car (s 30, in the merge zone), 1 lane-1 leader (s 49), 2 lane-1 target follower (s 10),
    3 lane-1 CAV far behind (s -60)."""
    kin = torch.zeros(1, 4, 7)
    kin[..., 0] = torch.tensor([30., 49., 10., -60.])
    kin[..., 4] = torch.tensor([4., 6., 7., 9.])
    kin[..., 5] = torch.tensor([-0.4, 0.2, 0.1, 0.0])
    kin[..., 6] = 5.
    kin[..., 1:4] = F.one_hot(torch.tensor([[2, 1, 1, 1]]), 3).float()
    return kin.requires_grad_(requires_grad)


def as_state(kin, like):
    """Scalar-field state built FROM ``kin`` (so a gradient w.r.t. ``kin`` is a state gradient)."""
    return {**state_from_kin(kin, like["present"]), "is_cav": like["is_cav"]}


def pivot_state(kin):
    return as_state(kin, {"present": torch.ones(1, 4, dtype=torch.bool),
                          "is_cav": torch.tensor([[False, False, False, True]])})


def central_difference(fn, kin, slot, channel, h):
    plus, minus = kin.detach().clone(), kin.detach().clone()
    plus[0, slot, channel] += h
    minus[0, slot, channel] -= h
    with torch.no_grad():
        return (fn(plus) - fn(minus)) / (2.0 * h)


# ------------------------------------------------------------------ (a) round-12 defects verified
@pytest.mark.parametrize("rollout_grad", [False, True])
def test_round12_defect1_courtesy_offset_has_no_state_derivative(rollout_grad):
    model = model_v(randomize=True, follower_courtesy_rollout_grad=rollout_grad)
    assert model.follower_courtesy_state_grad is False              # round-12 default
    state = state_at_step(sequence(cav=False), 0)
    kin = state["veh_kin"].clone().requires_grad_(True)
    out = model(as_state(kin, state))
    offset = out["courtesy_accel_mps2"]
    assert bool(out["courtesy_applies"][0, 0]) and offset[0, 0].abs().item() > 0
    if not rollout_grad:
        assert not offset.requires_grad                             # no graph at all
    else:
        grad = torch.autograd.grad(offset[0, 0], kin, allow_unused=True)[0]
        assert grad is None or float(grad.abs().sum()) == 0.0       # graph to the weights only


def test_round12_defect2_intent_is_lane_v5_while_the_pivot_owns_the_ramp_decision():
    # zero head: the courtesy offset cannot move the follower, so any change in the pivot outputs
    # could only come from the lane-v5 merge head itself
    model = pivot_model(randomize=False)                            # intent = lane_v5 (round 12)
    state = pivot_state(pivot_kin())
    forced = {}
    original = model._lane_v5_forward

    def forced_v5(*args, **kwargs):
        logits, *rest = original(*args, **kwargs)
        logits = logits.clone()
        logits[..., 1] = forced["merge_logit"]                     # merge-head score of every slot
        forced["p"] = float(torch.softmax(logits.float(), -1)[0, 0, 1])
        return (logits, *rest)

    model._lane_v5_forward = forced_v5
    results = {}
    for value in (-8.0, 8.0):
        forced["merge_logit"] = value
        out = model.step(state, cav_teacher=model.hold_teacher(state),
                         lane_uniform=torch.full((1, 4), 0.5))
        results[value] = (out, forced["p"])
    (lo, p_lo), (hi, p_hi) = results[-8.0], results[8.0]
    # the courtesy intent input of the follower (slot 2) IS the lane-v5 P(merge) of its partner ...
    assert lo["courtesy_inputs"][0, 2, INTENT].item() == pytest.approx(p_lo, abs=1e-6)
    assert hi["courtesy_inputs"][0, 2, INTENT].item() == pytest.approx(p_hi, abs=1e-6)
    assert p_hi > 0.9 > 0.1 > p_lo
    # ... but the pivot's boundary and the ramp car's lane decision do not depend on it
    assert torch.equal(lo["merge_pivot_c"], hi["merge_pivot_c"])
    assert torch.equal(lo["next_lane"][0, 0], hi["next_lane"][0, 0])


# ------------------------------------------------------------------ (b1) state gradient repair
def test_state_grad_flag_keeps_values_and_pairing_and_adds_a_graph():
    batch = sequence(cav=False)
    kin = batch["states"][:, 0].clone().requires_grad_(True)
    off = follower_courtesy_pairs(kin, batch["present"][:, 0], PhysicsConfig(), version=2)
    on = follower_courtesy_pairs(kin, batch["present"][:, 0], PhysicsConfig(), version=2,
                                 state_grad=True)
    assert not off["features"].requires_grad and on["features"].requires_grad
    assert torch.equal(off["features"], on["features"].detach())
    assert torch.equal(off["partner"], on["partner"]) and torch.equal(off["has_partner"], on["has_partner"])
    assert not on["partner"].requires_grad and on["partner"].dtype == torch.long


def offset_fn(model, base_state, slot):
    def fn(kin):
        return model(as_state(kin, base_state))["courtesy_accel_mps2"][0, slot]
    return fn


def test_state_grad_offset_derivative_is_nonzero_finite_and_matches_finite_differences():
    model = model_v(randomize=True, follower_courtesy_state_grad=True)
    torch.manual_seed(7)
    with torch.no_grad():                                       # larger fixed draws: visible slopes
        for p in model.follower_courtesy.parameters():
            p.mul_(5.0)
    state = state_at_step(sequence(cav=False), 0)
    kin = state["veh_kin"].clone().requires_grad_(True)
    out = model(as_state(kin, state))
    follower, partner = 0, int(follower_courtesy_pairs(kin, state["present"], PhysicsConfig())
                               ["partner"][0, 0])
    assert partner == 3
    grad = torch.autograd.grad(out["courtesy_accel_mps2"][0, follower], kin, retain_graph=True)[0]
    assert torch.isfinite(grad).all()
    # partner (ramp car) position / speed / accel and the follower's own position / speed
    checked = [(partner, 0), (partner, 4), (partner, 5), (follower, 0), (follower, 4)]
    assert all(grad[0, s, c].abs().item() > 1e-4 for s, c in checked)
    fn = offset_fn(model, state, follower)
    for s, c in checked:
        fd = central_difference(fn, kin, s, c, h=1e-2)
        assert grad[0, s, c].item() == pytest.approx(fd.item(), rel=3e-2, abs=2e-4), (s, c)
    # the total follower acceleration now responds to the ramp car (IDM itself does not see it)
    total = torch.autograd.grad(model(as_state(kin, state))["accel_mps2"][0, follower], kin)[0]
    assert total[0, partner, 0].abs().item() > 1e-4
    # slots without a partner keep an exactly zero offset with an exactly zero derivative
    unpartnered = torch.autograd.grad(out["courtesy_accel_mps2"][0, 1], kin, allow_unused=True)[0]
    assert unpartnered is None or float(unpartnered.abs().sum()) == 0.0


def test_state_grad_reaches_cav_partner_state_but_never_offsets_a_cav():
    model = model_v(randomize=True, follower_courtesy_state_grad=True)
    batch = sequence(cav=False)
    batch["is_cav"][..., 3] = True                                # the ramp partner is a CAV
    state = state_at_step(batch, 0)
    kin = state["veh_kin"].clone().requires_grad_(True)
    out = model(as_state(kin, state))
    grad = torch.autograd.grad(out["courtesy_accel_mps2"][0, 0], kin)[0]
    assert grad[0, 3, 0].abs().item() > 0 and grad[0, 3, 4].abs().item() > 0   # d(HDV)/d(CAV state)
    assert not bool(out["courtesy_applies"][0, 3])


def test_flags_off_forward_outputs_identical_to_state_grad_on():
    off, on = model_v(randomize=True), model_v(randomize=True, follower_courtesy_state_grad=True)
    on.load_state_dict(off.state_dict())
    batch = sequence(cav=False)
    for t in range(3):
        state = state_at_step(batch, t)
        a, b = off(state), on(state)
        for key in a:
            if torch.is_tensor(a[key]):
                assert torch.equal(a[key], b[key].detach()), key


def test_rollout_adjoint_now_includes_the_courtesy_path_values_unchanged():
    """Round-12 training impact: identical forward / loss value, different motion-loss gradient."""
    grads, values = {}, {}
    for flag in (False, True):
        model = model_v(randomize=True, follower_courtesy_state_grad=flag)
        with torch.no_grad():
            for p in model.follower_courtesy.parameters():
                p.mul_(5.0)
        cfg = TrainSettings(truncation_steps=8)
        res = UnrollRunner(model, cfg, build_loss(cfg), projection=None,
                           device=torch.device("cpu")).run(sequence(cav=False), unroll_steps=8,
                                                          backward=True)
        values[flag] = float(res["loss"]) if isinstance(res, dict) and "loss" in res else None
        grads[flag] = torch.cat([p.grad.flatten() for n, p in model.named_parameters()
                                 if p.grad is not None and not n.startswith(HEAD_PREFIX)])
        assert torch.isfinite(grads[flag]).all()
        head = [p.grad for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
        assert all(g is None or float(g.abs().sum()) == 0.0 for g in head)   # rollout_grad off
    assert values[False] is not None and values[False] == values[True]
    assert grads[False].shape == grads[True].shape and not torch.equal(grads[False], grads[True])


# ------------------------------------------------------------------ (b2) causal pivot intent
def test_pivot_intent_is_the_step_start_clearance_cdf_and_ignores_lane_v5():
    model = pivot_model(intent="pivot")
    kin = pivot_kin()
    state = pivot_state(kin)
    out = model(state)
    intent = out["courtesy_inputs"][0, 2, INTENT]
    geo = pivot.geometry(kin, state["present"], model.physics, decision_mask=~state["is_cav"])
    c = model.merge_pivot(geo)[0, 0]
    tau = model.merge_pivot.params()["tau"]
    assert torch.isfinite(c) and intent.item() == pytest.approx(torch.sigmoid(c / tau).item(), rel=1e-6)
    # computed BEFORE this step's acceleration: changing the head output leaves the inputs unchanged
    with torch.no_grad():
        model.follower_courtesy.net[-1].bias.add_(2.0)
    again = model(state)
    assert torch.equal(again["courtesy_inputs"], out["courtesy_inputs"])
    assert not torch.equal(again["accel_mps2"], out["accel_mps2"])
    # lane-v5 merge logits no longer enter
    original = model._lane_v5_forward

    def forced(*args, **kwargs):
        logits, *rest = original(*args, **kwargs)
        logits = logits.clone()
        logits[..., 1] = 8.0
        return (logits, *rest)

    model._lane_v5_forward = forced
    assert torch.equal(model(state)["courtesy_inputs"], again["courtesy_inputs"])


def test_pivot_intent_exact_limits_veto_and_outside_zone():
    model = pivot_model(intent="pivot")
    physics, present = model.physics, torch.ones(1, 4, dtype=torch.bool)
    cav = torch.tensor([[False, False, False, True]])
    kin = pivot_kin()
    outside = kin.clone()
    outside[0, 0, 0] = physics.merge_zone_s1_m + 20.0               # ramp car beyond the zone
    assert pivot_clearance_intent(model.merge_pivot, outside, present, physics, cav)[0, 0] == 0.0
    overlap = kin.clone()
    overlap[0, 2, 0] = 28.0                                         # follower body overlaps the ramp car
    assert pivot_clearance_intent(model.merge_pivot, overlap, present, physics, cav)[0, 0] == 0.0
    alone = kin.clone()
    alone[0, 1:, 1:4] = F.one_hot(torch.tensor([0]), 3).float()      # nobody on the target lane
    assert pivot_clearance_intent(model.merge_pivot, alone, present, physics, cav)[0, 0] == 1.0
    # a CAV ramp car is scored on its own neighbours (commanded decision, same physical veto)
    cav_ramp = torch.tensor([[True, False, False, True]])
    p_cav = pivot_clearance_intent(model.merge_pivot, kin, present, physics, cav_ramp)[0, 0]
    p_hdv = pivot_clearance_intent(model.merge_pivot, kin, present, physics, cav)[0, 0]
    assert p_cav.item() == pytest.approx(p_hdv.item())


def test_pivot_intent_state_gradient_matches_finite_differences_and_no_pivot_param_grad():
    model = pivot_model(intent="pivot", state_grad=True)
    state = pivot_state(pivot_kin())
    kin = state["veh_kin"].clone().requires_grad_(True)

    def intent_of(k):
        return model(as_state(k, state))["courtesy_accel_mps2"][0, 2]

    value = intent_of(kin)
    grad = torch.autograd.grad(value, kin)[0]
    assert torch.isfinite(grad).all()
    for s, c in ((0, 0), (0, 4), (1, 0), (2, 0), (2, 4)):
        fd = central_difference(intent_of, kin, s, c, h=1e-2)
        assert grad[0, s, c].item() == pytest.approx(fd.item(), rel=3e-2, abs=2e-4), (s, c)
    assert grad[0, 1, 0].abs().item() > 0                          # the ramp car's LEADER enters via c
    model.zero_grad(set_to_none=True)
    intent_of(kin).backward()
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith("merge_pivot."))


def test_intent_none_zeroes_the_slot_and_validation():
    model = pivot_model(intent="none")
    out = model(pivot_state(pivot_kin()))
    assert out["courtesy_inputs"][0, 2, INTENT].item() == 0.0
    assert set(INTENT_SOURCES) == {"lane_v5", "pivot", "none"}
    with pytest.raises(ValueError):
        ModelSpec(follower_courtesy=True, follower_courtesy_intent="pivot")   # no first_passage
    with pytest.raises(ValueError):
        ModelSpec(follower_courtesy=True, follower_courtesy_intent="merge_head")


# ------------------------------------------------------------------ (b3) absolute loss term
def test_absolute_term_default_off_is_identical_and_on_adds_recorded_target_on_head_only():
    model = model_v(randomize=True)
    base = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.5, contrast=True))
    v0 = base(context(model, targets(crowded())))
    explicit = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.5, contrast=True, absolute_weight=0.0))
    v1 = explicit(context(model, targets(crowded())))
    assert torch.equal(v0.value.detach(), v1.value.detach()) and "absolute_loss_raw" not in explicit.last_stats
    term = FollowerCourtesyLoss(FollowerCourtesyConfig(weight=1.5, contrast=True, absolute_weight=0.7))
    model.zero_grad(set_to_none=True)
    v2 = term(context(model, targets(crowded())))
    absolute = term.last_stats["absolute_loss_raw"]
    assert absolute > 0 and absolute != pytest.approx(v2.log.item())
    assert v2.value.item() == pytest.approx(1.5 * v2.log.item() + 0.7 * absolute, rel=1e-6)
    v2.value.backward()
    reached = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert reached and all(n.startswith(HEAD_PREFIX) for n in reached)
    with pytest.raises(ValueError):
        FollowerCourtesyConfig(weight=1.0, absolute_weight=-1.0)
    with pytest.raises(ValueError):
        TrainSettings(follower_courtesy=True, follower_courtesy_absolute_weight=1.0)   # weight 0


# ------------------------------------------------------------------ configs
def test_round12_config_keeps_round12_behaviour():
    cfg = load_train_config(ROUND12)
    assert cfg.model.follower_courtesy_state_grad is False
    assert cfg.model.follower_courtesy_intent == "lane_v5"
    assert cfg.loss.follower_courtesy.absolute_weight == 0.0
    flat = cfg.to_legacy_flat()
    assert not {"follower_courtesy_state_grad", "follower_courtesy_intent",
                "follower_courtesy_absolute_weight"} & set(flat)            # omitted at default
    settings = TrainSettings.from_config(cfg)
    assert settings.follower_courtesy_absolute_weight == 0.0
    spec = ModelSpec.from_mapping(flat)
    assert spec.follower_courtesy_state_grad is False and spec.follower_courtesy_intent == "lane_v5"


def test_round13a_config_differs_from_round12_only_by_the_repair_flags():
    # inline copy of round 12 (round 12 already sits at the 16-file extends cap)
    a, b = dict(leaves(load_raw(ROUND12))), dict(leaves(load_raw(ROUND13A)))
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    allowed = {"name", "runtime.out_dir", "model.init_from", "model.follower_courtesy_state_grad",
               "model.follower_courtesy_intent", "loss.follower_courtesy.absolute_weight"}
    assert changed <= allowed, sorted(changed - allowed)
    assert {"model.follower_courtesy_state_grad", "model.follower_courtesy_intent",
            "loss.follower_courtesy.absolute_weight"} <= changed
    cfg = load_train_config(ROUND13A)
    assert cfg.model.follower_courtesy_state_grad is True
    assert cfg.model.follower_courtesy_intent == "pivot"
    assert cfg.model.follower_courtesy_init_zero is True           # fresh head: intent meaning changed
    assert cfg.loss.follower_courtesy.absolute_weight == 4.525          # docs/r13a_absolute_weight_rule.json
    assert cfg.loss.follower_courtesy.contrast is True
    assert cfg.model.follower_courtesy_rollout_grad is False
    assert cfg.model.merge_pivot_state_st is True and cfg.model.merge_decision == "first_passage"
    settings = TrainSettings.from_config(cfg)
    assert settings.follower_courtesy_absolute_weight == 4.525
    spec = ModelSpec.from_mapping(cfg.to_legacy_flat())
    assert spec.follower_courtesy_state_grad and spec.follower_courtesy_intent == "pivot"
