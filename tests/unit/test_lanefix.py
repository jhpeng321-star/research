"""LANEFIX switches (2026-10-14): lane-change fixes of the WM-S hazard lane model.

* ``model.lane_decision = "sample"`` (+ ``model.lane_logit_offsets``): seeded draws from the
  masked hazard categorical instead of the argmax;
* ``loss.lane_hazard_decouple``: rollout lane CE / flip-rate never reach the hazard;
* ``model.lane_hazard_features = "v2"``: ranking features (lookahead speed vs v0, leader
  trend, density ahead, time blocked behind a slower leader);
* ``model.merge_hazard_zone = "full"``: the hazard owns ramp merges, physics forces the last 5 m;
* ``model.gap_stopped_relax``: relaxed gaps for stopped ramp mergers (speed rule).

Covers: switch-off bit-identity (mask, one hazard training step, rollout; golden numbers
captured on the pre-LANEFIX tree c4/int a2d3716), sampled decisions (seed determinism,
feasibility, common random numbers), decouple gradients, v2 features (finite, causal),
merge ownership, stopped-merger gaps, config plumbing and the v4-lane config.
"""
from __future__ import annotations

import math
import types
from dataclasses import asdict

import pytest
import torch

from .test_gaprule import _kin, fuzz_scene
from .test_wmlane import _batch, _c4, _model
from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.factory import build_model
from wms.model.lane_hazard import (FORCED_MERGE_LOGIT, HAZARD_F, HAZARD_F_V2,
                                   MERGE_HAZARD_PRIOR_LOGIT, hazard_features_v2_extra,
                                   hazard_lane_logits, initial_lane_blocked, sample_lane_choice)
from wms.model.spec import ModelSpec
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import MASKED_LOGIT, mobil_merge_logits
from wms.train.inputs import state_from_kin
from wms.train.losses.compose import build_loss
from wms.train.losses.lane_event import recorded_event_logits, recorded_lane_blocked
from wms.train.settings import TrainSettings
from wms.train.trainer import WmSTrainer
from wms.train.unroll import UnrollRunner

#: The hazard (v3-like) training recipe the golden numbers were captured with.
HAZ = ['model.lane_model="hazard"', 'model.driver_update="ema"', "loss.lane_event.weight=1.0",
       "loss.lane_event.gamma=0.0", "loss.lane_event.max_weight=1.0", 'model.lane_gap_rule="speed"']
NEW_FLAT_KEYS = ("lane_decision", "lane_logit_offsets", "lane_hazard_features",
                 "merge_hazard_zone", "gap_stopped_relax", "lane_hazard_decouple")
V4_LANE_DIFF = {"lane_decision": "sample", "lane_hazard_features": "v2",
                "merge_hazard_zone": "full", "gap_stopped_relax": True,
                "lane_hazard_decouple": True, "lane_event_weight": 3.0,
                "out_dir": "runs/active/c4_wm_v4_lane"}


def _phys(**kw):
    return PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                         lane_gap_rule="speed", **kw)


def _hazard_run(overrides=(), *, perturb=True, detach=False, threads=1):
    torch.set_num_threads(threads)
    _, flat, settings = _c4([*HAZ, *overrides])
    model, _ = _model(flat)
    physics = _phys(gap_stopped_relax=bool(settings.gap_stopped_relax))
    model.physics = physics
    if perturb:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
    model.lane_hazard_detach = detach
    model.train()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=physics),
                          projection=None, device=torch.device("cpu"))
    rec = runner.run(_batch(), unroll_steps=12, backward=True, truncation_steps=5)
    return model, rec


def _grad(params):
    return sum(float(p.grad.double().abs().sum()) for p in params if p.grad is not None)


def _initial(batch, t=0):
    state = state_from_kin(batch["states"][:, t], batch["present"][:, t])
    state["is_cav"] = batch["is_cav"][:, t]
    return state


# --------------------------------------------------------------- bit identity
#: Captured with this file's helpers on the pre-LANEFIX tree (c4/int a2d3716, one thread):
#: one hazard training step (HAZ recipe), an eval rollout with a biased hazard, the physics
#: mask on a slowed batch state and on ``fuzz_scene()`` (speed rule).
GOLDEN = {"loss": 196.15622609853745, "position": 186.6270294189453, "speed": 7.427229285240173,
          "lane": 1.5391623675823212, "flip_rate": 0.0548754176707007,
          "lane_event": 0.5079182982444763, "grad_abs_sum": 491.92612335205,
          "hazard_grad_abs_sum": 1.084284509345537,
          "rollout": {"pos": 13776.432180404663, "lanes": 133, "logit": -1643.30654835701,
                      "speed": 1880.6822290420532},
          "mask": {"fixed": [-121.52728319168091, 13], "speed": [-121.52728319168091, 13]},
          "fuzz_speed": {"False": -1211.4535709619522, "True": -1213.0717169940472}}


def test_switch_off_hazard_training_step_is_bit_identical():
    model, rec = _hazard_run()
    got = {k: rec[k] for k in ("loss", "position", "speed", "lane", "flip_rate", "lane_event")}
    assert got == {k: GOLDEN[k] for k in got}
    assert _grad(model.parameters()) == pytest.approx(GOLDEN["grad_abs_sum"], rel=1e-9)
    assert _grad(model.lane_hazard.parameters()) == pytest.approx(
        GOLDEN["hazard_grad_abs_sum"], rel=1e-9)
    assert model.lane_hazard.n_features == HAZARD_F
    assert model.lane_decision == "argmax" and not model._samples_lanes


def test_switch_off_rollout_is_bit_identical():
    model, _ = _hazard_run()
    model.eval()
    b = _batch()
    with torch.no_grad():
        model.lane_hazard.net[-1].bias.fill_(4.0)
        roll = model.rollout(_initial(b), n_steps=12, cav_teacher=b["cav_teacher"],
                             driver_history=b["driver_hist"],
                             driver_history_mask=b["driver_hist_mask"])
    got = {"pos": float(roll["positions"].double().sum()), "lanes": int(roll["lanes"].sum()),
           "logit": float(roll["lane_logits"].double().sum()),
           "speed": float(roll["speeds"].double().sum())}
    assert got == GOLDEN["rollout"]
    assert "lane_blocked" not in roll["state"]


def test_switch_off_mask_is_bit_identical():
    b = _batch()
    kin = b["states"][:, 3].clone()
    kin[..., 4] = kin[..., 4] * 0.2
    present = b["present"][:, 3]
    for rule in ("fixed", "speed"):
        p = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary", lane_gap_rule=rule)
        lg = mobil_merge_logits(kin, present, p)
        assert [float(lg.double().sum()), int((lg <= -8.0).sum())] == GOLDEN["mask"][rule]
    kin, present, params = fuzz_scene()
    for drivers in (None, params):
        off = mobil_merge_logits(kin, present, _phys(), drivers)
        assert float(off.double().sum()) == GOLDEN["fuzz_speed"][str(drivers is not None)]


# ------------------------------------------------------------ 1: sampled decisions
def _sampling_model(seed=0, bias=(3.0, 3.0), **spec_kw):
    spec = ModelSpec(**{"rollout_mode": "hybrid", "lane_model": "hazard",
                        "lane_decision": "sample", "history_s": 60.0, "history_stride_s": 1.0,
                        **spec_kw})
    torch.manual_seed(seed)
    model = build_model(spec, _phys())
    with torch.no_grad():
        model.lane_hazard.net[-1].bias.copy_(torch.tensor(bias))
    model.eval()
    return model


def _roll(model, **kw):
    b = _batch()
    with torch.no_grad():
        return model.rollout(_initial(b), n_steps=12, cav_teacher=b["cav_teacher"],
                             driver_history=b["driver_hist"],
                             driver_history_mask=b["driver_hist_mask"], **kw)


def test_sampled_decisions_are_deterministic_per_seed_and_change_lanes():
    model = _sampling_model()
    a, again, other = _roll(model, lane_seed=3), _roll(model, lane_seed=3), _roll(model, lane_seed=4)
    assert torch.equal(a["lanes"], again["lanes"]) and torch.equal(a["positions"], again["positions"])
    assert not torch.equal(a["lanes"], other["lanes"])
    argmax = _sampling_model()
    argmax.lane_decision = "argmax"
    det = _roll(argmax)
    assert not torch.equal(a["lanes"], det["lanes"])      # the draws matter
    # The global RNG is untouched by a seeded rollout.
    torch.manual_seed(11)
    _roll(model, lane_seed=3)
    after = torch.rand(1)
    torch.manual_seed(11)
    assert torch.equal(after, torch.rand(1))


def test_sampled_decisions_respect_the_feasibility_mask():
    model = _sampling_model(bias=(10.0, 10.0), history_s=None)   # wants every open change
    kin, present, _ = fuzz_scene(seed=5, b=6, k=24)
    state = state_from_kin(kin, present)
    state["is_cav"] = torch.zeros_like(present)
    with torch.no_grad():
        out = model.forward(state)
        feasible = out["lane_feasible"]
        assert torch.equal(feasible, (out["lane_logits"] > MASKED_LOGIT)
                           | torch.nn.functional.one_hot(kin[..., 1:4].argmax(-1), 3).bool())
        g = torch.Generator().manual_seed(0)
        changes = 0
        for _ in range(200):
            u = torch.rand(present.shape, generator=g)
            choice = sample_lane_choice(out["lane_logits"], feasible, out["hdv_mask"], u)
            assert feasible.gather(-1, choice.unsqueeze(-1)).all()
            changes += int((choice != kin[..., 1:4].argmax(-1))[present].sum())
        assert changes > 0
    # Float-rounding edge: a draw at the top of the CDF never lands on a masked lane.
    logits = torch.tensor([[[0.3, 0.0, MASKED_LOGIT]]])
    feas = torch.tensor([[[True, True, False]]])
    pick = sample_lane_choice(logits, feas, torch.ones(1, 1, dtype=torch.bool),
                              torch.full((1, 1), 1.0 - 1e-9))   # rounds to 1.0 in float32
    assert int(pick) == 0                                            # argmax fallback, feasible


def test_common_random_numbers_across_batch_rows():
    model = _sampling_model()
    b = _batch()
    two = {k: v[:1].expand(2, *v.shape[1:]).clone() for k, v in b.items()}
    def roll(**kw):
        with torch.no_grad():
            return model.rollout(_initial(two), n_steps=12, cav_teacher=two["cav_teacher"],
                                 driver_history=two["driver_hist"],
                                 driver_history_mask=two["driver_hist_mask"], **kw)
    crn = roll(lane_seed=5, lane_crn=True)
    assert torch.equal(crn["lanes"][0], crn["lanes"][1])
    indep = roll(lane_seed=5, lane_crn=False)
    assert not torch.equal(indep["lanes"][0], indep["lanes"][1])
    # Caller-owned draws [1, T, K] broadcast over rows; [B, T, K] per row.
    u = torch.rand(1, 12, two["states"].shape[2], generator=torch.Generator().manual_seed(2))
    shared = roll(lane_uniform=u)
    assert torch.equal(shared["lanes"][0], shared["lanes"][1])
    assert torch.equal(roll(lane_uniform=u.expand(2, -1, -1))["lanes"], shared["lanes"])


def test_lane_logit_offsets_shift_only_their_category():
    kin = _kin([(-30.0, 1, 20.0), (300.0, 0, 20.0), (40.0, 2, 10.0)])
    present = torch.ones(1, 3, dtype=torch.bool)
    physics = _phys()
    base = mobil_merge_logits(kin, present, physics)
    spec = ModelSpec(rollout_mode="hybrid", lane_model="hazard")
    model = build_model(spec, physics)
    kw = dict(driver_params=None, driver_embedding=None, lane_age=None)
    plain, _ = hazard_lane_logits(model.lane_hazard, kin, present, present, base, physics, **kw)
    shifted, _ = hazard_lane_logits(model.lane_hazard, kin, present, present, base, physics, **kw,
                                    offsets=(1.0, 2.0, 3.0))
    diff = (shifted - plain).detach()
    assert bool((plain[0, 0, 0] > MASKED_LOGIT) & (plain[0, 1, 1] > MASKED_LOGIT)
                & (plain[0, 2, 1] > MASKED_LOGIT))                    # all three open
    assert float(diff[0, 0, 0]) == pytest.approx(1.0)                 # overtake 1 -> 0
    assert float(diff[0, 1, 1]) == pytest.approx(2.0)                 # return 0 -> 1
    assert float(diff[0, 2, 1]) == pytest.approx(3.0)                 # merge 2 -> 1
    assert float(diff.abs().sum()) == pytest.approx(6.0)              # nothing else moves
    with pytest.raises(ValueError):
        ModelSpec(rollout_mode="hybrid", lane_logit_offsets=(1.0, 0.0, 0.0))   # residual lanes


# ------------------------------------------------------------------- 2: decouple
def test_decouple_zero_hazard_grad_from_lane_and_flip_rate():
    no_event = ["loss.lane_event.weight=0.0"]
    model, rec = _hazard_run(no_event, detach=True)
    assert rec["lane"] > 0 and rec["flip_rate"] > 0
    assert _grad(model.lane_hazard.parameters()) == 0.0
    assert _grad(p for n, p in model.named_parameters() if not n.startswith("lane_hazard.")) > 0
    coupled, _ = _hazard_run(no_event, detach=False)                    # the path it cuts
    assert _grad(coupled.lane_hazard.parameters()) > 0
    with_event, rec = _hazard_run(detach=True)
    assert rec["lane_event"] > 0 and _grad(with_event.lane_hazard.parameters()) > 0
    # The forward values are unchanged by the detach.
    assert rec["lane"] == _hazard_run(detach=False)[1]["lane"]


def test_trainer_applies_the_decouple_switch():
    for on in (False, True):
        _, flat, settings = _c4([*HAZ, f"loss.lane_hazard_decouple={str(on).lower()}"])
        model, _ = _model(flat)
        WmSTrainer._apply_model_overrides(types.SimpleNamespace(settings=settings, model=model))
        assert model.lane_hazard_detach is on
    with pytest.raises(Exception):
        _c4(["loss.lane_hazard_decouple=true"])                       # residual lanes


# ------------------------------------------------------------------ 3: features
def test_v2_features_are_finite_and_widen_the_hazard():
    kin, present, params = fuzz_scene(seed=9)
    extra = hazard_features_v2_extra(kin, present, _phys(), params,
                                     torch.full(present.shape, 300.0))
    assert extra.shape == (*present.shape, HAZARD_F_V2 - HAZARD_F)
    assert torch.isfinite(extra).all()
    assert float(extra[..., -2].max()) == pytest.approx(6.0)            # blocked time capped (60 s)
    model = _sampling_model(lane_hazard_features="v2")
    assert model.lane_hazard.n_features == HAZARD_F_V2
    roll = _roll(model, lane_seed=0)
    assert torch.isfinite(roll["positions"]).all()
    assert roll["state"]["lane_blocked"].shape == roll["state"]["s_m"].shape


def test_v2_lookahead_and_blocked_semantics():
    physics = _phys()
    # Car 0 (lane 1, v0 30) behind a slow leader 20 m ahead; lane 0 is free.
    kin = _kin([(0.0, 1, 10.0), (25.0, 1, 8.0), (500.0, 0, 30.0)])
    kin[0, 1, 5] = -1.5
    present = torch.ones(1, 3, dtype=torch.bool)
    params = {"v0": torch.full((1, 3), 30.0)}
    blocked = torch.tensor([[4.0, 0.0, 0.0]])
    x = hazard_features_v2_extra(kin, present, physics, params, blocked)[0, 0]
    left, right, cur = x[0:4], x[4:8], x[8:12]
    assert float(left[0]) == 0.0 and float(left[2]) == 0.0              # free target lane
    assert float(cur[0]) == pytest.approx((8.0 - 30.0) / 10.0)          # slow lane ahead
    assert float(cur[1]) == pytest.approx(-0.5)                         # leader braking
    assert float(cur[2]) == pytest.approx(7.5 / 100.0) and float(cur[3]) == pytest.approx(7.5 / 200)
    assert float(x[12]) == pytest.approx(4 * 0.5 / 10) and float(x[13]) == 1.0
    assert float(right[0]) == 0.0 and float(right[3]) == 0.0           # lane 2: empty
    # Pre-t0 blocked run from the long history (last sample = t0 is excluded).
    hist = torch.zeros(1, 3, 5, 9)
    hist[..., 0] = 10 / 30
    hist[..., 5] = 0.2                                                   # 20 m gap
    hist[..., 7] = -2 / 30                                               # leader 8 m/s
    hist[..., 8] = 1.0
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    mask[0, 1, 1] = False
    run = initial_lane_blocked(present, torch.float32, hist, mask, params["v0"], 2)
    assert run.tolist() == [[8.0, 4.0, 8.0]]


def test_v2_teacher_forced_features_have_no_future_leakage():
    model = _sampling_model(lane_hazard_features="v2")
    b = _batch()
    states = b["states"].clone()
    present, is_cav = b["present"], b["is_cav"]
    defaults = dict(zip(PARAMETER_NAMES, (30.0, 1.135, 2.6, 2.49, 4.2, 0.2, 0.1, 6.0, 1.0)))
    params = {n: torch.full(present.shape[::2], v) for n, v in defaults.items()}
    emb = torch.zeros(*present.shape[::2], model.driver_embed_dim)
    def logits(st):
        blocked = recorded_lane_blocked(model.physics, st, present, params,
                                        torch.zeros(present.shape[::2]))
        age = torch.full(present.shape, 999.0)
        with torch.no_grad():
            return recorded_event_logits(model, st, present, is_cav, params, emb, age,
                                         blocked), blocked
    ref, blocked_ref = logits(states)
    future = states.clone()
    future[:, 7:, :, 0] += 13.0
    future[:, 7:, :, 4] *= 0.3
    out, blocked_out = logits(future)
    assert torch.equal(ref[:, :7], out[:, :7]) and torch.equal(blocked_ref[:, :7], blocked_out[:, :7])
    assert not torch.equal(ref[:, 7:], out[:, 7:])


# ---------------------------------------------------------------- 4: merges
def _merge_model(zone, bias_left):
    spec = ModelSpec(rollout_mode="hybrid", lane_model="hazard", merge_hazard_zone=zone)
    model = build_model(spec, _phys())
    with torch.no_grad():
        model.lane_hazard.net[-1].bias.copy_(torch.tensor([bias_left, 0.0]))
    model.eval()
    return model


def _merge_logit(model, s, v=10.0):
    kin = _kin([(s, 2, v)])
    state = state_from_kin(kin, torch.ones(1, 1, dtype=torch.bool))
    state["is_cav"] = torch.zeros(1, 1, dtype=torch.bool)
    with torch.no_grad():
        out = model.step(state, cav_teacher=None)
    return float(out["lane_logits"][0, 0, 1]), int(out["next_lane"][0, 0])


def test_full_merge_zone_early_merge_and_forced_end():
    s1 = _phys().merge_zone_s1_m
    logit, lane = _merge_logit(_merge_model("full", 5.0), 20.0)       # hazard: merge now
    assert logit == pytest.approx(MERGE_HAZARD_PRIOR_LOGIT + 10.0 * math.tanh(0.5)) and lane == 1
    logit, lane = _merge_logit(_merge_model("full", -9.0), 20.0)      # hazard: wait
    assert lane == 2 and logit < -8.0 + 1e-6 + 4.0
    logit, lane = _merge_logit(_merge_model("full", -9.0), s1 - 3.0)  # last 5 m: forced
    assert logit >= FORCED_MERGE_LOGIT and lane == 1
    logit, lane = _merge_logit(_merge_model("full", -9.0), s1 - 10.0)  # hazard owns it
    assert lane == 2
    late = _merge_model("forced_late", -9.0)                          # legacy 25 m window
    physics_logit = float(mobil_merge_logits(_kin([(s1 - 10.0, 2, 10.0)]),
                                             torch.ones(1, 1, dtype=torch.bool), _phys())[0, 0, 1])
    assert _merge_logit(late, s1 - 10.0)[0] == pytest.approx(physics_logit)


def test_full_merge_zone_never_unmasks_an_unsafe_merge():
    s1 = _phys().merge_zone_s1_m
    kin = _kin([(s1 - 2.0, 2, 0.0), (s1 - 2.0, 1, 0.0)])               # side by side: overlap
    present = torch.ones(1, 2, dtype=torch.bool)
    model = _merge_model("full", 10.0)
    base = mobil_merge_logits(kin, present, model.physics)
    logits, _ = hazard_lane_logits(model.lane_hazard, kin, present, present, base, model.physics,
                                   None, None, None, merge_zone="full")
    assert float(logits.detach()[0, 0, 1]) <= MASKED_LOGIT


def test_stopped_relax_allows_a_jam_merge_but_not_overlap():
    present = torch.ones(1, 3, dtype=torch.bool)
    # Ego stopped on the acceleration lane, 4 m bumper gaps to a stopped leader and to a
    # follower creeping at 3 m/s in lane 1.
    kin = _kin([(60.0, 2, 0.0), (69.0, 1, 0.0), (51.0, 1, 3.0)])
    strict = mobil_merge_logits(kin, present, _phys())
    relaxed = mobil_merge_logits(kin, present, _phys(gap_stopped_relax=True))
    assert float(strict[0, 0, 1]) <= MASKED_LOGIT
    assert float(relaxed[0, 0, 1]) > MASKED_LOGIT
    # A target-lane car alongside (overlap) still blocks; so does a fast follower.
    for blocker in ((58.0, 1, 0.0), (54.5, 1, 12.0)):
        kin2 = _kin([(60.0, 2, 0.0), (69.0, 1, 0.0), blocker])
        assert float(mobil_merge_logits(kin2, present, _phys(gap_stopped_relax=True))[0, 0, 1]) \
            <= MASKED_LOGIT
    # A moving ego is judged by the unrelaxed rule.
    kin3 = _kin([(60.0, 2, 3.0), (69.0, 1, 0.0), (51.0, 1, 3.0)])
    assert torch.equal(mobil_merge_logits(kin3, present, _phys(gap_stopped_relax=True)),
                       mobil_merge_logits(kin3, present, _phys()))
    # Relaxation only adds admissions, and only for merges.
    kin, present, params = fuzz_scene(seed=4)
    for drivers in (None, params):
        a = mobil_merge_logits(kin, present, _phys(), drivers)
        r = mobil_merge_logits(kin, present, _phys(gap_stopped_relax=True), drivers)
        assert not ((a > MASKED_LOGIT) & (r <= MASKED_LOGIT)).any()
        changed = (a != r).any(-1)
        lanes = kin[..., 1:4].argmax(-1)
        assert ((lanes == 2) | ~changed).all()
    with pytest.raises(ValueError):
        PhysicsConfig(gap_stopped_relax=True)                          # needs the speed rule


# ------------------------------------------------------------------- plumbing
def test_flat_forms_omit_new_keys_while_default_and_round_trip():
    for cfg in (TrainConfig(), load_train_config("configs/c4_wm.toml"),
                load_train_config("configs/c4_wm_v3.toml")):
        assert not set(NEW_FLAT_KEYS) & set(cfg.to_legacy_flat())
    flat = load_train_config("configs/c4_wm_v4_lane.toml").to_legacy_flat()
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    s = TrainSettings.from_flat(flat)
    assert s.lane_decision == "sample" and s.lane_hazard_decouple and s.gap_stopped_relax
    spec = ModelSpec.from_mapping(flat)
    assert (spec.lane_decision, spec.lane_hazard_features, spec.merge_hazard_zone) == (
        "sample", "v2", "full")


def test_v4_lane_config_differs_from_v3_only_in_the_lanefix_keys():
    v3 = load_train_config("configs/c4_wm_v3.toml").to_legacy_flat()
    v4 = load_train_config("configs/c4_wm_v4_lane.toml").to_legacy_flat()
    diff = {k for k in set(v3) | set(v4) if v3.get(k) != v4.get(k)}
    assert diff == set(V4_LANE_DIFF)
    assert {k: v4[k] for k in diff} == V4_LANE_DIFF
    from wms.cli.assemble import build_physics
    physics = build_physics(load_train_config("configs/c4_wm_v4_lane.toml"), lambda n: n)
    assert physics.gap_stopped_relax and physics.lane_gap_rule == "speed"


def test_switch_validation():
    bad = [['model.lane_decision="sample"'], ['model.lane_hazard_features="v2"'],
           ['model.merge_hazard_zone="full"'], ["model.gap_stopped_relax=true"],
           ['model.lane_decision="maybe"', *HAZ]]
    for overrides in bad:
        with pytest.raises(Exception):
            _c4(overrides)
    with pytest.raises(ValueError):
        TrainSettings(lane_decision="sample")
    with pytest.raises(ValueError):
        ModelSpec(rollout_mode="hybrid", lane_model="hazard", merge_hazard_zone="early")


def test_checkpoint_round_trip_keeps_switches_and_physics(tmp_path):
    cfg = load_train_config("configs/c4_wm_v4_lane.toml", ["runtime.device=cpu"])
    from wms.cli.assemble import build_model as assemble_model, build_physics
    physics = build_physics(cfg, lambda n: n)
    model = assemble_model(cfg, physics)
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, cfg)
    import json
    (tmp_path / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")
    loaded = load_checkpoint(path).model
    assert loaded.lane_decision == "sample" and loaded.merge_hazard_zone == "full"
    assert loaded.lane_hazard.n_features == HAZARD_F_V2
    assert loaded.physics.gap_stopped_relax and not loaded.lane_hazard_detach
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_imagination_lane_draws_are_seeded_and_only_for_sampling_models():
    from wms.control.futures import lane_uniform_draws
    argmax = build_model(ModelSpec(rollout_mode="hybrid", lane_model="hazard"), _phys())
    assert lane_uniform_draws([argmax], 3, 4, 5, seed=1) is None
    sampler = _sampling_model()
    a = lane_uniform_draws([argmax, sampler], 3, 4, 5, seed=1)
    assert a.shape == (3, 4, 5) and torch.equal(a, lane_uniform_draws([sampler], 3, 4, 5, seed=1))
    assert not torch.equal(a, lane_uniform_draws([sampler], 3, 4, 5, seed="val0"))
    assert float(a.min()) >= 0.0 and float(a.max()) < 1.0


def test_calibration_tool_rows_and_summary():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "tools" / "lanehazard_calib.py"
    spec = importlib.util.spec_from_file_location("lanehazard_calib", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    import numpy as np
    # Two cars, 3 steps: car 0 overtakes 1 -> 0 at t=1, car 1 (lane 2) merges at t=2.
    lanes = np.array([[1, 2], [1, 2], [0, 2], [0, 1]])
    valid = np.ones_like(lanes, dtype=bool)
    probs = np.full((3, 2, 3), 0.1)
    masked = np.zeros((3, 2, 3), dtype=bool)
    masked[:, 1, 2] = True
    rows = tool.transition_rows(lanes, valid, probs, masked)
    p, y, cat, m = rows.T
    assert len(rows) == 2 + 2 + 1 + 3    # car 0: 2 dirs x 2 steps + 1 on lane 0; car 1: 3 merges
    assert y.sum() == 2 and set(cat[y == 1]) == {1.0, 3.0}
    out = tool.summarize(rows)
    assert out["categories"]["overtake 1->0"]["events"] == 1
    assert out["categories"]["merge 2->1"]["p_over_obs"] == pytest.approx(0.3)
    assert out["pooled_feasible"]["logloss_constant"] <= out["pooled_feasible"]["logloss"] + 1e-12 \
        or out["pooled_feasible"]["rate"] == pytest.approx(0.1)
    assert tool.auc(np.array([0.1, 0.9, 0.5]), np.array([0, 1, 0])) == 1.0


def test_candidate_batches_share_lane_draws_across_arms():
    from wms.control.futures import arm_major_lane_uniform
    argmax = build_model(ModelSpec(rollout_mode="hybrid", lane_model="hazard"), _phys())
    gen = torch.Generator().manual_seed(0)
    assert arm_major_lane_uniform([argmax], 3, 2, 4, 5, gen) == {}
    assert torch.equal(gen.get_state(), torch.Generator().manual_seed(0).get_state())  # no draw
    u = arm_major_lane_uniform([_sampling_model()], 3, 2, 4, 5, gen)["lane_uniform"]
    assert u.shape == (6, 4, 5)
    assert torch.equal(u[0:2], u[2:4]) and torch.equal(u[0:2], u[4:6])   # arm-major CRN
    assert not torch.equal(u[0], u[1])
