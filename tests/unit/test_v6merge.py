"""V6MERGE: lane-end waiting (``physics.merge_end_wait``) and cooperative mainline yielding
(``physics.mainline_merge_yield``); both default OFF and the OFF path is unchanged."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, replace

import pytest
import torch
import torch.nn.functional as F

from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.model.lane_v5 import instant_features
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import (MASKED_LOGIT, merge_end_blocked, merge_end_gap_safe,
                               mobil_merge_logits)
from wms.physics.state import state_kin
from wms.train.settings import TrainSettings

END = 75.44
#: The v5/v6 trained physics (fetched c4_wm_v5/physics.json): SUMO IDM, speed gap rule,
#: stopped-merger relaxation, learned discretionary lane changes.
V5 = dict(sumo_fidelity=True, sumo_idm=True, lane_gap_rule="speed", gap_stopped_relax=True,
          lane_rule="learned_discretionary")


def _physics(wait=False, yield_=False, **kw):
    return PhysicsConfig(**{**V5, **kw}, merge_end_wait=wait, mainline_merge_yield=yield_)


def _state(s, v, lane, a=None):
    s = torch.tensor([s], dtype=torch.float32)
    return dict(s_m=s, v_mps=torch.tensor([v], dtype=torch.float32),
                a_mps2=torch.zeros_like(s) if a is None else torch.tensor([a], dtype=torch.float32),
                lane=torch.tensor([lane]), length_m=torch.full_like(s, 5.0),
                present=torch.ones_like(s, dtype=torch.bool))


def _kin(state):
    return state_kin(state), state["present"]


def _merge_logit(physics, state, k=0):
    kin, present = _kin(state)
    return float(mobil_merge_logits(kin, present, physics)[0, k, 1])


# --------------------------------------------------------------------------- gap acceptance
def test_stopped_relaxation_admits_tight_gap_only_when_off():
    # Stopped ramp car at the end; lane-1 follower 2 m behind its rear at 3 m/s.
    state = _state([73.0, 66.0], [0.0, 3.0], [2, 1])
    assert _merge_logit(_physics(), state) > MASKED_LOGIT           # root cause: 2 m accepted
    assert _merge_logit(_physics(wait=True), state) == MASKED_LOGIT
    kin, present = _kin(state)
    assert merge_end_blocked(kin, present, _physics(wait=True)).tolist() == [[True, False]]


@pytest.mark.parametrize("follower_s,leader_s,safe", [
    (66.0, None, False),     # follower too close
    (20.0, None, True),      # both secure (follower 48 m back at 3 m/s; no leader)
    (20.0, 79.0, False),     # leader rear 1 m ahead of the merger front: lead gap insecure
    (20.0, 84.0, True),      # leader rear 6 m ahead (>= s0 = 2.6 m for a standing merger)
])
def test_both_gaps_must_be_secure(follower_s, leader_s, safe):
    s, v, lane = [73.0, follower_s], [0.0, 3.0], [2, 1]
    if leader_s is not None:
        s, v, lane = s + [leader_s], v + [0.0], lane + [1]
    state = _state(s, v, lane)
    kin, present = _kin(state)
    physics = _physics(wait=True)
    assert bool(merge_end_gap_safe(kin, present, physics)[0, 0]) is safe
    assert (_merge_logit(physics, state) > MASKED_LOGIT) is safe


def test_rule_only_touches_slow_ramp_cars_near_the_end():
    # Far from the end (s = 30 m) or fast: the trained rule applies unchanged.
    for s0, v0 in ((30.0, 0.0), (73.0, 6.0)):
        state = _state([s0, s0 - 7.0], [v0, 3.0], [2, 1])
        assert _merge_logit(_physics(), state) == _merge_logit(_physics(wait=True), state)


def test_lane_v5_fast_mask_matches_mobil_veto():
    torch.manual_seed(0)
    physics = _physics(wait=True)
    s = torch.rand(64, 12) * 120.0 - 20.0
    lane = torch.randint(0, 3, (64, 12))
    lane = torch.where(s > 70.0, torch.full_like(lane, 2), lane)
    v = torch.where(torch.rand(64, 12) < 0.5, torch.zeros(64, 12), torch.rand(64, 12) * 12.0)
    state = dict(s_m=s, v_mps=v, a_mps2=torch.randn(64, 12), lane=lane,
                 length_m=torch.full_like(s, 5.0), present=torch.rand(64, 12) < 0.9)
    kin, present = _kin(state)
    feasible = instant_features(kin, present, physics, None)["target_feasible"]
    logits = mobil_merge_logits(kin, present, physics)
    tq = torch.tensor([1, 0, 1])[lane]
    from_mobil = (logits.gather(-1, tq.unsqueeze(-1)).squeeze(-1) > MASKED_LOGIT) & present
    assert torch.equal(feasible, from_mobil)
    assert (merge_end_blocked(kin, present, physics) & present).any()


# --------------------------------------------------------------------------- closed loop
def _physics_model(physics):
    """In-module physics only: the learned residual and lane deltas are exactly zero."""
    torch.manual_seed(0)
    return WmSModel(ModelSpec(rollout_mode="driver_params"), physics=physics).eval()


def _driver_params(state):
    values = dict(v0=30.0, time_headway=1.135, min_gap=2.6, accel=2.49, decel=4.2,
                  politeness=0.2, mobil_threshold=0.1, merge_gap=6.0, merge_urgency=1.0)
    return {k: torch.full_like(state["s_m"], x) for k, x in values.items()}


def _run(physics, state, steps):
    model = _physics_model(physics)
    trace = []
    with torch.no_grad():
        for _ in range(steps):
            out = model.step(state, driver_params=_driver_params(state))
            trace.append((state["s_m"][0].clone(), state["v_mps"][0].clone(),
                          state["lane"][0].clone(), out["next_lane"][0].clone()))
            accel = (out["next_v_mps"] - state["v_mps"]) / physics.dt_s
            state = dict(state, s_m=out["next_s_m"], v_mps=out["next_v_mps"],
                         lane=out["next_lane"], a_mps2=accel)
    return trace


def _queue_then_gap():
    # A dense lane-1 queue crawling past a stopped merger, then a large gap.
    s = [73.0] + [76.0 - 8.0 * i for i in range(1, 7)] + [-60.0]
    v = [0.0] + [3.0] * 6 + [3.0]
    return _state(s, v, [2] + [1] * 7)


def test_blocked_merger_waits_for_both_gaps_then_merges():
    off = _run(_physics(), _queue_then_gap(), 80)
    on = _run(_physics(wait=True), _queue_then_gap(), 80)
    first = lambda tr: next(i for i, (_, _, ln, nl) in enumerate(tr) if nl[0] == 1)  # noqa: E731
    t_off, t_on = first(off), first(on)
    assert t_off <= 4                       # OFF squeezes into a 1 m / 4 m gap within 2 s
    assert t_on > t_off + 10                # ON waits ...
    s, v, lane, _ = on[t_on]
    state = _state(s.tolist(), v.tolist(), lane.tolist())
    kin, present = _kin(state)
    assert bool(merge_end_gap_safe(kin, present, _physics(wait=True))[0, 0])   # ... for both gaps
    for s_i, _, lane_i, _ in on[:t_on]:
        assert lane_i[0] == 2 and float(s_i[0]) <= END


def test_never_passes_lane_end():
    # A fast ramp car with a lane-1 neighbour alongside (merge infeasible) and a residual push.
    def backend(state):
        return {"accel_mps2": torch.full_like(state["s_m"], 3.0),
                "lane_logits": F.one_hot(state["lane"], 3).float() * 8.0}
    for wait, passes in ((False, True), (True, False)):
        model = WmSModel(ModelSpec(), physics=_physics(wait=wait), physics_step=backend).eval()
        state = _state([74.0, 74.0], [6.0, 6.0], [2, 1])
        with torch.no_grad():
            out = model.step(state)
        assert (float(out["next_s_m"][0, 0]) > END + 1e-3) is passes
        if not passes:
            assert float(out["next_s_m"][0, 0]) == pytest.approx(END)
            assert float(out["next_v_mps"][0, 0]) == 0.0
            assert int(out["next_lane"][0, 0]) == 2
    # Closed loop with real physics: the car stays at or behind the end for 60 s.
    trace = _run(_physics(wait=True), _queue_then_gap(), 120)
    assert all(float(s[0]) <= END + 1e-4 for s, _, lane, _ in trace if lane[0] == 2)


# --------------------------------------------------------------------------- mainline yield
def test_follower_behind_blocked_merger_decelerates():
    # Lane-1 follower 28 m behind the blocked merger's rear, free lane ahead otherwise.
    state = _state([73.0, 40.0], [0.0, 10.0], [2, 1])
    kin, present = _kin(state)
    base = baseline_accel(kin, present, _physics(wait=True))
    yielded = baseline_accel(kin, present, _physics(wait=True, yield_=True))
    assert base[0, 1] > 0.0
    assert yielded[0, 1] < 0.0
    assert yielded[0, 1] >= -PhysicsConfig().mainline_merge_yield_decel_mps2 - 1e-6
    assert torch.equal(base[0, 0], yielded[0, 0])         # the merger itself is unchanged


@pytest.mark.parametrize("case", ["merger_behind", "overlapping", "too_far", "secure_gap",
                                  "not_nearest"])
def test_no_phantom_braking(case):
    s, v, lane = {
        "merger_behind": ([73.0, 80.0], [0.0, 10.0], [2, 1]),     # follower already ahead
        "overlapping": ([73.0, 71.0], [0.0, 3.0], [2, 1]),        # alongside: may pass
        "too_far": ([73.0, -20.0], [0.0, 10.0], [2, 1]),          # 93 m back
        "secure_gap": ([73.0, 20.0], [0.0, 2.0], [2, 1]),         # merger not blocked
        "not_nearest": ([73.0, 60.0, 30.0], [0.0, 3.0, 3.0], [2, 1, 1]),
    }[case]
    state = _state(s, v, lane)
    kin, present = _kin(state)
    base = baseline_accel(kin, present, _physics(wait=True))
    yielded = baseline_accel(kin, present, _physics(wait=True, yield_=True))
    if case == "not_nearest":
        assert yielded[0, 1] <= base[0, 1] and torch.equal(base[0, 2], yielded[0, 2])
    else:
        assert torch.equal(base, yielded)


def test_yield_uses_per_driver_idm_and_has_gradients():
    state = _state([73.0, 40.0], [0.0, 10.0], [2, 1])
    params = _driver_params(state)
    params["time_headway"] = params["time_headway"].clone().requires_grad_(True)
    kin, present = _kin(state)
    accel = baseline_accel(kin, present, _physics(wait=True, yield_=True), params)
    assert accel[0, 1] < 0.0
    accel[0, 1].backward()
    assert params["time_headway"].grad[0, 1] != 0.0


# --------------------------------------------------------------------------- OFF path / config
def _fixed_batch():
    torch.manual_seed(1)
    b, k = 4, 24
    s = torch.rand(b, k) * 400.0 - 250.0
    lane = torch.randint(0, 3, (b, k))
    s[:, :4] = torch.tensor([73.0, 70.0, 60.0, 50.0])
    lane[:, :4] = 2
    v = torch.rand(b, k) * 25.0
    v[:, :3] = 0.0
    return dict(s_m=s, v_mps=v, a_mps2=torch.randn(b, k), lane=lane,
                length_m=torch.full_like(s, 5.0), present=torch.ones(b, k, dtype=torch.bool))


GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "v6merge_off_golden.pt"


def _off_outputs(physics):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)          # the captured fixture is single-threaded (bitwise)
    try:
        return _off_outputs_1t(physics)
    finally:
        torch.set_num_threads(threads)


def _off_outputs_1t(physics):
    state = _fixed_batch()
    kin, present = _kin(state)
    out = {"logits": mobil_merge_logits(kin, present, physics),
           "accel": baseline_accel(kin, present, physics),
           "feasible": instant_features(kin, present, physics, None)["target_feasible"]}
    cfg = load_train_config("configs/c4_wm_v5.toml")
    torch.manual_seed(3)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics).eval()
    b, k = present.shape
    hist = torch.zeros(b, k, model.driver_history_len, 9)
    hist[:, :, -1, 0] = state["v_mps"] / 30.0
    hist[:, :, -1, 2:5] = F.one_hot(state["lane"], 3).float()
    mask = torch.zeros(hist.shape[:-1], dtype=torch.bool)
    mask[:, :, -1] = True
    with torch.no_grad():
        step = model.step(dict(state), k_steps=5, lane_uniform=torch.full((b, k), 0.5),
                          driver_history=hist, driver_history_mask=mask)
    for key in ("next_s_m", "next_v_mps", "next_lane", "accel_mps2", "lane_logits"):
        out["step_" + key] = step[key]
    return out


def test_off_path_bit_identical():
    """OFF outputs equal the ones captured on unmodified HEAD 916559d (fixture, 1 thread), bit for bit:
    physics logits/accel, the lane-v5 fast mask and a full v5-config model step."""
    assert not PhysicsConfig().merge_end_wait and not PhysicsConfig().mainline_merge_yield
    physics = PhysicsConfig(**V5)
    assert physics == _physics()
    ref = torch.load(GOLDEN)
    out = _off_outputs(physics)
    for key, value in ref.items():
        assert torch.equal(value, out[key]), key
    # The V6MERGE constants are unused while the switches are OFF.
    other = replace(physics, merge_end_wait_distance_m=3.0, mainline_merge_yield_distance_m=10.0,
                    merge_end_wait_speed_mps=9.0, mainline_merge_yield_decel_mps2=0.5)
    out = _off_outputs(other)
    assert all(torch.equal(value, out[key]) for key, value in ref.items())
    # ... and ON really differs on the same batch (the fixture exercises the rules).
    on = _off_outputs(_physics(wait=True, yield_=True))
    assert not torch.equal(on["logits"], ref["logits"])
    assert not torch.equal(on["accel"], ref["accel"])


def test_config_plumbing_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("WMS_DATA_V6_LONGPAIR", str(tmp_path))
    old = load_train_config("configs/c4_wm_v6_cont.toml")
    assert not {"merge_end_wait", "mainline_merge_yield"} & old.to_legacy_flat().keys()
    assert not build_physics(old).merge_end_wait
    cfg = load_train_config("configs/c4_wm_v6_merge_ft.toml")
    physics = build_physics(cfg, resolver=lambda name: name)
    assert physics.merge_end_wait and physics.mainline_merge_yield
    assert physics.stop_time_integration and physics.emergency_decel_mps2 == 9.0
    assert cfg.curriculum.unroll_s == (60.0, 120.0)
    assert cfg.curriculum.stage_steps == (200, 120)
    assert cfg.curriculum.truncation_steps_per_stage == (120, 120)
    assert cfg.optim.lr_per_stage == (1.0e-4, 5.0e-5) and cfg.curriculum.max_hours == 3.5
    assert cfg.runtime.out_dir == "runs/active/c4_wm_v6_merge_ft"
    flat = cfg.to_legacy_flat()
    assert TrainConfig.from_legacy_flat(flat).to_legacy_flat() == flat
    settings = TrainSettings.from_flat(flat)
    assert settings.merge_end_wait and settings.mainline_merge_yield
    small = TrainConfig.from_dict({"physics": {"merge_end_wait": True,
                                               "mainline_merge_yield": True}})
    physics = build_physics(small)
    model = WmSModel(ModelSpec.from_mapping(small.to_legacy_flat()), physics=physics)
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, small)
    (tmp_path / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")
    loaded = load_checkpoint(path).model
    assert loaded.physics == physics and loaded.physics.merge_end_wait
    # Old physics.json without the keys loads with both switches OFF.
    raw = {k: v for k, v in asdict(physics).items() if "merge_end" not in k
           and "mainline_merge" not in k}
    (tmp_path / "physics.json").write_text(json.dumps(raw), encoding="utf-8")
    assert not load_checkpoint(path).model.physics.merge_end_wait
    # The switches are never read from a merge-cal params file.
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"merge_end_wait": True, "mainline_merge_yield": True}))
    assert not PhysicsConfig.from_params_file(params).merge_end_wait


@pytest.mark.parametrize("name", ["merge_end_wait_distance_m", "merge_end_wait_speed_mps",
                                  "mainline_merge_yield_distance_m",
                                  "mainline_merge_yield_decel_mps2"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_constant_validation(name, value):
    with pytest.raises(ValueError):
        PhysicsConfig(**{name: value})


def test_on_step_backward_finite():
    cfg = TrainConfig.from_dict({"model": {"lane_model": "hazard_v5", "rollout_mode": "hybrid"},
                                 "physics": {"merge_end_wait": True,
                                             "mainline_merge_yield": True}})
    physics = replace(build_physics(cfg), **V5)
    torch.manual_seed(0)
    model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=physics)
    state = _fixed_batch()
    out = model.step(state, k_steps=5)
    out["next_s_m"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_yield_never_deadlocks_the_merge():
    """A yielding follower whose own s0 (1 m) is below the road s0 (2.6 m) stops leaving the
    merger's standstill secure gap, so the blocked merger gets in (and sooner than without
    yielding) in a crawling lane-1 queue."""
    def scenario():
        s = [73.0] + [90.0 - 7.0 * i for i in range(1, 30)]
        return _state(s, [0.0] + [1.0] * 29, [2] + [1] * 29)

    def first_merge(physics):
        model = _physics_model(physics)
        state = scenario()
        with torch.no_grad():
            for i in range(200):
                params = _driver_params(state)
                params["min_gap"] = torch.full_like(state["s_m"], 1.0)
                out = model.step(state, driver_params=params)
                if int(out["next_lane"][0, 0]) == 1:
                    return i
                accel = (out["next_v_mps"] - state["v_mps"]) / physics.dt_s
                state = dict(state, s_m=out["next_s_m"], v_mps=out["next_v_mps"],
                             lane=out["next_lane"], a_mps2=accel)
        return None

    wait_only = first_merge(_physics(wait=True))
    with_yield = first_merge(_physics(wait=True, yield_=True))
    assert with_yield is not None and (wait_only is None or with_yield < wait_only)


def test_fused_gap_check_equals_reference():
    from wms.physics.mobil import _secure_gaps_now, anticipated_kin
    physics = _physics(wait=True)
    for seed in range(50):
        g = torch.Generator().manual_seed(seed)
        b, k = 3, 30
        s = torch.rand(b, k, generator=g) * 200 - 60
        v = torch.where(torch.rand(b, k, generator=g) < 0.4, torch.zeros(b, k),
                        torch.rand(b, k, generator=g) * 15)
        state = dict(s_m=s, v_mps=v, a_mps2=torch.randn(b, k, generator=g) * 2,
                     lane=torch.randint(0, 3, (b, k), generator=g),
                     length_m=4 + torch.rand(b, k, generator=g) * 4,
                     present=torch.rand(b, k, generator=g) < 0.9)
        kin, present = _kin(state)
        target = torch.ones(b, k, dtype=torch.long)
        ref = (_secure_gaps_now(kin, present, target, physics)
               & _secure_gaps_now(anticipated_kin(kin, physics.gap_anticipation_s), present,
                                  target, physics))
        assert torch.equal(ref, merge_end_gap_safe(kin, present, physics))
