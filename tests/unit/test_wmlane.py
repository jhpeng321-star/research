"""WMLANE switches: (A) the dedicated lane-change hazard model + teacher-forced event
loss and (C) the in-rollout driver-setting update (EMA) with the reveal training mode.

Covers: switch-off bit-identity against numbers captured on the pre-WMLANE code
(c4/int 51cb52d), config plumbing (flat keys omitted while default), shapes, the hard
physics feasibility mask under an extreme hazard, the event-loss targets / weights /
sign, the driver update reducing a wrong v0, open-loop no-op, the reveal mask and an
end-to-end unroll with every switch on.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from wms.config.load import load_train_config
from wms.config.train import ModelConfig, TrainConfig
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.factory import build_model
from wms.model.lane_hazard import HAZARD_F, hazard_features, relative_logits
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import MASKED_LOGIT, mobil_merge_logits
from wms.train.losses.compose import build_loss
from wms.train.losses.lane_event import (category_weights, lane_event_loss,
                                         lane_event_targets, recorded_lane_age)
from wms.train.inputs import state_from_kin
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

NEW_FLAT_KEYS = ("lane_model", "lane_hazard_hidden", "lane_hazard_cap", "driver_update",
                 "driver_update_rate", "driver_reveal_prob", "driver_reveal_max_s",
                 "lane_event_weight", "lane_event_gamma", "lane_event_max_weight")

def _batch():
    g = torch.Generator().manual_seed(7)
    B, T, K, H = 2, 13, 6, 61
    states = torch.zeros(B, T, K, 7)
    states[..., 0] = torch.rand(B, T, K, generator=g) * 60 + torch.arange(K) * 30 - 40
    lane = torch.randint(0, 3, (B, 1, K), generator=g).expand(B, T, K)
    states[..., 1:4] = F.one_hot(lane, 3).float()
    states[..., 4] = torch.rand(B, T, K, generator=g) * 15 + 5
    states[..., 5] = 0.1
    states[..., 6] = 5.0
    present = torch.ones(B, T, K, dtype=torch.bool)
    present[:, :, 5] = False
    is_cav = torch.zeros(B, T, K, dtype=torch.bool)
    is_cav[:, :, 0] = True
    teacher = torch.zeros(B, T, K, 4)
    teacher[..., 0] = states[..., 0]
    teacher[..., 1] = lane.float()
    teacher[..., 2] = states[..., 4]
    hist = torch.rand(B, K, 10, 6, generator=g)
    hist_mask = torch.ones(B, K, 10, dtype=torch.bool)
    dh = torch.rand(B, K, H, 9, generator=g)
    dh[..., 2:5] = F.one_hot(lane[:, 0], 3).float().unsqueeze(2)
    dhm = torch.ones(B, K, H, dtype=torch.bool)
    return {"states": states, "present": present, "is_cav": is_cav, "cav_teacher": teacher,
            "hist": hist, "hist_mask": hist_mask, "driver_hist": dh, "driver_hist_mask": dhm}


def _c4(overrides=()):
    cfg = load_train_config("configs/c4_wm.toml", ["runtime.device=cpu", *overrides])
    flat = cfg.to_legacy_flat()
    return cfg, flat, TrainSettings.from_flat(flat)


def _model(flat, seed=0):
    torch.manual_seed(seed)
    physics = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary")
    return build_model(ModelSpec.from_mapping(flat), physics), physics


def _run(overrides=(), *, perturb=True, steps=12, trunc=5):
    _, flat, settings = _c4(overrides)
    model, physics = _model(flat)
    if perturb:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
    model.train()
    composer = build_loss(settings, physics=physics)
    runner = UnrollRunner(model, settings, composer, projection=None, device=torch.device("cpu"))
    rec = runner.run(_batch(), unroll_steps=steps, backward=True, truncation_steps=trunc)
    return model, rec, flat


# --------------------------------------------------------------- bit identity
#: Captured with this file's ``_run()`` on the pre-WMLANE tree (c4/int 51cb52d).
GOLDEN_C4_SWITCH_OFF = {"loss": 195.94624710083008, "position": 186.95134735107422, "speed": 7.406633615493774, "lane": 1.536797046661377, "flip_rate": 0.051463119336403906, "rate": 0.02211921475827694, "grad_abs_sum": 516.2632219315183, "n_params": 816681, "keys": 105}


def test_switch_off_is_bit_identical_to_pre_wmlane_code():
    model, rec, flat = _run()
    grad = sum(float(p.grad.double().abs().sum()) for p in model.parameters()
               if p.grad is not None)
    got = {"loss": rec["loss"], "position": rec["position"], "speed": rec["speed"],
           "lane": rec["lane"], "flip_rate": rec["flip_rate"],
           "rate": rec["predicted_lane_change_rate"], "grad_abs_sum": grad,
           "n_params": sum(p.numel() for p in model.parameters()),
           "keys": len(model.state_dict())}
    # Forward values are exact; the summed |grad| is compared to 1e-9 relative because
    # multi-threaded CPU reductions in backward vary with the intra-op thread count (the
    # HEAD-vs-branch comparison at equal threads was bit-identical, see WMLANE_NOTES.md).
    grad_golden = GOLDEN_C4_SWITCH_OFF["grad_abs_sum"]
    assert got.pop("grad_abs_sum") == pytest.approx(grad_golden, rel=1e-9)
    assert got == {k: v for k, v in GOLDEN_C4_SWITCH_OFF.items() if k != "grad_abs_sum"}
    assert model.lane_hazard is None and model.driver_updater is None
    assert not any(k.startswith(("lane_hazard.", "driver_updater."))
                   for k in model.state_dict())


def test_flat_forms_omit_every_new_key_while_default():
    for cfg in (TrainConfig(), load_train_config("configs/c4_wm.toml"),
                load_train_config("r4")):
        flat = cfg.to_legacy_flat()
        assert not set(NEW_FLAT_KEYS) & set(flat)
    on = load_train_config("configs/c4_wm_v2_lane.toml")
    flat = on.to_legacy_flat()
    assert flat["lane_model"] == "hazard" and flat["driver_update"] == "ema"
    assert flat["lane_event_weight"] > 0 and flat["driver_reveal_prob"] > 0
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    spec = ModelSpec.from_mapping(flat)
    assert spec.lane_model == "hazard" and spec.driver_update == "ema"
    assert TrainSettings.from_flat(flat).lane_event_weight == flat["lane_event_weight"]


def test_switch_validation():
    with pytest.raises(ValueError):
        ModelSpec(lane_model="hazard")             # residual rollout has no embedding
    with pytest.raises(ValueError):
        ModelSpec(rollout_mode="hybrid", driver_update="gru")
    with pytest.raises(Exception):
        ModelConfig(lane_model="hazard")
    with pytest.raises(Exception):
        ModelConfig(rollout_mode="hybrid", lane_model="hazard", lane_head="history")
    ModelSpec(rollout_mode="hybrid", lane_model="hazard", driver_update="ema")


# ----------------------------------------------------------------- A: hazard
def _scene():
    """Mainline car with a slow leader (overtake incentive), ramp car near the lane end,
    lane-0 car, a car whose left target is blocked (unsafe gap)."""
    rows = [(100.0, 1, 25.0), (112.0, 1, 5.0), (70.0, 2, 8.0), (150.0, 0, 25.0),
            (300.0, 1, 20.0), (302.0, 0, 20.0)]
    kin = torch.zeros(1, len(rows), 7)
    for k, (s, lane, v) in enumerate(rows):
        kin[0, k, 0], kin[0, k, 1 + lane], kin[0, k, 4], kin[0, k, 6] = s, 1.0, v, 5.0
    return kin


def _hazard_model(**extra):
    spec = ModelSpec(rollout_mode="hybrid", lane_model="hazard", **extra)
    torch.manual_seed(0)
    return build_model(spec, PhysicsConfig(lane_rule="learned_discretionary"))


def test_hazard_shapes_and_zero_init_reproduces_physics_prior():
    model = _hazard_model()
    kin = _scene()
    present = torch.ones(1, kin.shape[1], dtype=torch.bool)
    state = {"veh_kin": kin, "present": present}
    out = model(state)
    assert out["lane_logits"].shape == (1, kin.shape[1], 3)
    assert "driver_embedding" in out
    base = mobil_merge_logits(kin, present, model.physics, {
        n: out["driver_parameters"][..., i] for i, n in enumerate(PARAMETER_NAMES)})
    assert torch.equal(out["lane_logits"], base)
    feats, priors = hazard_features(kin, present, base, model.physics, None, None)
    assert feats.shape == (1, kin.shape[1], HAZARD_F) and priors.shape == (1, kin.shape[1], 2)
    rel = relative_logits(out["lane_logits"], kin[..., 1:4].argmax(-1))
    assert torch.all(rel[0, 3, 1] == MASKED_LOGIT)     # no lane left of lane 0


@pytest.mark.parametrize("bias", [60.0, -60.0])
def test_hazard_never_selects_an_infeasible_change(bias):
    model = _hazard_model()
    with torch.no_grad():
        model.lane_hazard.net[-1].bias.fill_(bias)
    kin = _scene()
    present = torch.ones(1, kin.shape[1], dtype=torch.bool)
    state = {"veh_kin": kin, "present": present, "s_m": kin[..., 0], "v_mps": kin[..., 4],
             "a_mps2": kin[..., 5], "lane": kin[..., 1:4].argmax(-1), "length_m": kin[..., 6]}
    out = model.step(state)
    params = {n: out["driver_parameters"][..., i] for i, n in enumerate(PARAMETER_NAMES)}
    base = mobil_merge_logits(kin, present, model.physics, params)
    masked = base <= MASKED_LOGIT
    assert torch.all(out["lane_logits"][masked] == base[masked])
    chosen = base.gather(-1, out["next_lane"].unsqueeze(-1)).squeeze(-1)
    assert torch.all(chosen > MASKED_LOGIT)             # never an infeasible lane
    lane = state["lane"]
    assert torch.all((out["next_lane"] - lane).abs() <= 1)
    if bias < 0:
        # The mandatory end-of-lane merge (ramp car 5 m before the end) is never vetoed.
        assert int(out["next_lane"][0, 2]) == int(base[0, 2].argmax())


def test_hazard_lane_event_logits_match_forward_on_the_same_state():
    model = _hazard_model()
    with torch.no_grad():
        model.lane_hazard.net[-1].bias.fill_(1.5)
    kin = _scene()
    present = torch.ones(1, kin.shape[1], dtype=torch.bool)
    out = model({"veh_kin": kin, "present": present})
    params = {n: out["driver_parameters"][..., i] for i, n in enumerate(PARAMETER_NAMES)}
    one = model.lane_event_logits({"veh_kin": kin, "present": present}, driver_params=params,
                                  driver_embedding=out["driver_embedding"])
    assert torch.allclose(one, out["lane_logits"])


# ------------------------------------------------------------- A: event loss
def test_event_targets_and_categories():
    lanes = torch.tensor([[[1, 0, 2, 1], [0, 1, 1, 1], [0, 1, 1, 1]]])   # [1,T+1=3,K=4]
    valid = torch.ones_like(lanes, dtype=torch.bool)
    action, category, mask = lane_event_targets(lanes, valid, 2)
    assert action[0, 0].tolist() == [1, 2, 1, 0]       # left, right, left, stay
    assert category[0, 0].tolist() == [1, 2, 3, 0]     # overtake, return, merge, stay
    assert bool(mask.all())
    age = recorded_lane_age(lanes, valid, torch.full((1, 4), 999.0), 2)
    assert age[0, 1].tolist() == [0.0, 0.0, 0.0, 1000.0]


def test_category_weights_balance_and_cap():
    category = torch.tensor([0] * 100 + [1] * 10 + [2] * 2)
    w = category_weights(category, max_weight=20.0)
    assert w[0] == 1.0
    assert torch.isclose(w[1], torch.tensor(100 / (2 * 10)))     # 5
    assert w[2] == 20.0                                          # capped (25 -> 20)
    assert w[3] == 1.0 and w[4] == 1.0                           # absent categories


def test_event_loss_sign_weights_and_infeasible_rows():
    logits = torch.zeros(4, 3, requires_grad=True)
    action = torch.tensor([0, 0, 0, 1])
    category = torch.tensor([0, 0, 0, 1])
    loss, stats = lane_event_loss(logits, action, category, gamma=0.0, max_weight=20.0)
    loss.backward()
    assert logits.grad[3, 1] < 0          # raising the recorded event's logit lowers the loss
    assert logits.grad[0, 0] < 0
    # Event weight 3 (n_stay / n_event): equals a hand-weighted CE.
    ce = F.cross_entropy(torch.zeros(4, 3), action, reduction="none")
    w = torch.tensor([1.0, 1.0, 1.0, 3.0])
    assert torch.isclose(loss, (w * ce).sum() / w.sum())
    # Focal gamma down-weights confident rows.
    confident = torch.tensor([[5.0, 0.0, 0.0]] * 3 + [[0.0, 0.0, 0.0]])
    plain, _ = lane_event_loss(confident, action, category, gamma=0.0, max_weight=20.0)
    focal, _ = lane_event_loss(confident, action, category, gamma=2.0, max_weight=20.0)
    assert focal < plain
    # A recorded change into a physics-masked target is dropped, never infinite.
    masked = torch.tensor([[0.0, MASKED_LOGIT, 0.0], [0.0, 0.0, 0.0]])
    value, stats = lane_event_loss(masked, torch.tensor([1, 0]), torch.tensor([1, 0]),
                                   gamma=2.0, max_weight=20.0)
    assert torch.isfinite(value) and stats["infeasible_events"] == 1


# ------------------------------------------------------------- C: driver update
def _free_car_states(v0_true=33.0, steps=40):
    """A lone free-road car accelerating by the IDM towards ``v0_true`` (a_max 2.49)."""
    v, s = 24.0, 500.0
    rows = []
    for _ in range(steps + 1):
        a = 2.49 * (1.0 - (v / v0_true) ** 4)
        rows.append((s, v, a))
        v_next = v + a * 0.5
        s += 0.5 * (v + v_next) * 0.5
        v = v_next
    kin = torch.zeros(1, steps + 1, 1, 7)
    for t, (s_t, v_t, a_t) in enumerate(rows):
        kin[0, t, 0, 0], kin[0, t, 0, 2], kin[0, t, 0, 4] = s_t, 1.0, v_t
        kin[0, t, 0, 5], kin[0, t, 0, 6] = a_t, 5.0
    return kin


def _ema_model():
    spec = ModelSpec(rollout_mode="hybrid", driver_update="ema", driver_update_rate=0.3)
    torch.manual_seed(0)
    return build_model(spec, PhysicsConfig(sumo_fidelity=True,
                                           lane_rule="learned_discretionary"))


def _wrong_params(v0=25.0):
    names = dict(zip(PARAMETER_NAMES, (v0, 1.135, 2.6, 2.49, 4.2, 0.2, 0.1, 6.0, 1.0)))
    return {n: torch.full((1, 1), float(val)) for n, val in names.items()}


def test_driver_update_reduces_v0_error_on_a_synthetic_car():
    model = _ema_model()
    kin = _free_car_states()
    params = _wrong_params()
    errors = [abs(float(params["v0"]) - 33.0)]
    for t in range(1, kin.shape[1]):
        state = {"veh_kin": kin[:, t], "present": torch.ones(1, 1, dtype=torch.bool)}
        params, _ = model.update_drivers(params, None, state, torch.ones(1, 1, dtype=torch.bool))
        errors.append(abs(float(params["v0"].detach()) - 33.0))
    assert errors[-1] < 0.25 * errors[0]
    assert all(b <= a + 1e-6 for a, b in zip(errors, errors[1:]))   # monotone here
    # Unobserved slots never change.
    same, _ = model.update_drivers(_wrong_params(), None, {"veh_kin": kin[:, 5]},
                                   torch.zeros(1, 1, dtype=torch.bool))
    assert float(same["v0"].detach()) == 25.0


def test_rollout_observation_updates_drivers_and_open_loop_is_a_noop():
    model = _ema_model()
    model.eval()
    kin = _free_car_states(steps=20)
    state = state_from_kin(kin[:, 0], torch.ones(1, 1, dtype=torch.bool))
    state["is_cav"] = torch.zeros(1, 1, dtype=torch.bool)
    params = _wrong_params()
    emb = torch.zeros(1, 1, model.driver_embed_dim)
    with torch.no_grad():
        free = model.rollout(state, n_steps=20, driver_params=params, driver_embedding=emb)
        observe = torch.ones(1, 21, 1, dtype=torch.bool)
        seen = model.rollout(state, n_steps=20, driver_params=params, driver_embedding=emb,
                             observe_states=kin, observe_mask=observe)
        off_spec = ModelSpec(rollout_mode="hybrid")
        torch.manual_seed(0)
        off = build_model(off_spec, model.physics)
        off.load_state_dict(model.state_dict(), strict=False)
        off.eval()
        plain = off.rollout(state, n_steps=20, driver_params=params, driver_embedding=emb)
    assert "driver_parameters_updated" not in free
    assert torch.equal(free["positions"], plain["positions"])       # open loop: no-op
    v0_new = float(seen["driver_parameters_updated"][0, 0, PARAMETER_NAMES.index("v0")])
    assert abs(v0_new - 33.0) < 0.5 * abs(25.0 - 33.0)
    # The logged prediction of the first step is the model's own (same as open loop).
    assert torch.equal(seen["positions"][:, 0], free["positions"][:, 0])


# ------------------------------------------------------------ reveal + unroll
def test_reveal_mask_is_a_hdv_prefix():
    _, flat, settings = _c4(["model.driver_reveal_prob=1.0", "model.driver_reveal_max_s=2.0",
                             "model.driver_update=\"ema\""])
    model, physics = _model(flat)
    runner = UnrollRunner(model, settings, build_loss(settings, physics=physics),
                          projection=None, device=torch.device("cpu"))
    from wms.train.inputs import prepare_inputs
    inp = prepare_inputs(_batch(), model, settings, torch.device("cpu"), 12)
    mask = runner._reveal_mask(inp)
    assert mask.shape == inp.states.shape[:3]
    assert not mask[:, 0].any() and not mask[:, 5:].any()            # <= 4 steps (2 s)
    assert not mask[:, :, 0].any() and not mask[:, :, 5].any()       # CAV / absent slots
    assert mask[:, 1, 1:5].all()                                     # every HDV revealed


def test_unroll_with_every_switch_on_trains_hazard_and_update():
    model, rec, flat = _run([
        "model.lane_model=\"hazard\"", "model.driver_update=\"ema\"",
        "model.driver_reveal_prob=0.5", "loss.lane_event.weight=1.0"], perturb=False)
    assert "lane_event" in rec and rec["lane_event"] > 0
    assert torch.isfinite(torch.tensor(rec["loss"]))
    hazard_grad = sum(float(p.grad.abs().sum()) for p in model.lane_hazard.parameters()
                      if p.grad is not None)
    assert hazard_grad > 0
    assert model.driver_updater.rate_logits.grad is not None
