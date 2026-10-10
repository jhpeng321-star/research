"""Contract tests for the closed-loop A11 Predictor adapter (fakes + synthetic).

The fake-rollout tests need no model; the real-checkpoint parity tests skip
when the pinned A11 member is not on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_WM = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[5]
for p in (str(_WM), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from closedloop import a11_predictor as AP  # noqa: E402
from closedloop.a11_predictor import (A11Predictor, A11PredictorError,  # noqa: E402
                                      BatchRollout, LiveSample)
from closedloop.live_state import (DepartureHistory, LiveState,  # noqa: E402
                                   base_departures_upto, live_state_from_row)

K = 8


def synth_state(n_cav=3, n_hdv=3) -> LiveState:
    kin = np.zeros((K, 7), np.float32)
    mask = np.zeros(K, bool)
    is_cav = np.zeros(K, bool)
    vid = [None] * K
    for i in range(n_cav + n_hdv):
        kin[i, 0] = -500.0 + 40.0 * i            # s
        kin[i, 2] = 1.0                          # right lane
        kin[i, 4] = 20.0                         # v
        kin[i, 6] = 5.0
        mask[i] = True
        is_cav[i] = i < n_cav
        vid[i] = (f"cav{i}" if i < n_cav else f"hdv{i}")
    return LiveState(time_s=400.0, veh_kin=kin, veh_mask=mask, is_cav=is_cav,
                     slot_id=np.where(mask, np.arange(K), -1).astype(np.int64),
                     counter=np.zeros((2, 30), np.float32), slot_vid=vid)


def fake_rollout_factory(log):
    """Deterministic fake: each CAV code shifts that CAV's speed."""

    def fake(bundle, state, codes, *, n_steps, t0_s=0.0, **kw):
        log.append(np.array(codes))
        b_n = codes.shape[0]
        T = n_steps
        pos = np.zeros((T + 1, b_n, K))
        spd = np.zeros((T + 1, b_n, K))
        for b in range(b_n):
            v = state.veh_kin[:, 4].astype(float) + np.where(codes[b] > 0, -1.0 * codes[b], 0.0)
            spd[:, b] = v
            pos[:, b] = state.veh_kin[:, 0] + np.outer(np.arange(T + 1) * 0.5, v)
        msk = np.repeat(np.repeat(state.rollout_mask[None, None], T + 1, 0), b_n, 1)
        lns = np.ones((T + 1, b_n, K), np.int64)
        ids = np.repeat(np.repeat(state.slot_id[None, None], T + 1, 0), b_n, 1)
        acc = np.zeros((T, b_n, K))
        return BatchRollout(pos=pos, spd=spd, acc=acc, lns=lns, msk=msk, ids=ids,
                            length=np.full((b_n, K), 5.0), a0=np.zeros((b_n, K)))

    return fake


def fake_head(bundle, state, plan_long):
    """dV = 100 * (20 - bin-0 mean speed); 5 taus x 19 quantiles."""
    base = 100.0 * (20.0 - plan_long[:, 0].numpy().astype(float))
    q = np.repeat(base[:, None, None], 5, 1).repeat(19, 2)
    q = q + np.linspace(-1, 1, 19)[None, None, :]
    return q


def make_pred(log, **kw):
    return A11Predictor(object(), rollout_fn=fake_rollout_factory(log), head_fn=fake_head,
                        fast=False, **kw)


def plans_for(state):
    from traffic_shepherd.planner.interface import PlanCandidate

    cavs = sorted(state.cav_slots)
    native = PlanCandidate({c: "native" for c in cavs})
    one = PlanCandidate({**{c: "native" for c in cavs}, cavs[0]: "damping"})
    fleet = PlanCandidate({c: "pi-external-u" for c in cavs})
    return native, one, fleet


def test_native_is_zero_and_dedupe():
    log = []
    st = synth_state()
    pred = make_pred(log)
    native, one, fleet = plans_for(st)
    out = pred.predict_many([LiveSample(st)], [native, one, fleet, one])
    assert len(out) == 4
    assert out[0].delay_vs_native_s.shape == (1, 1)
    assert out[0].delay_vs_native_s[0, 0] == 0.0
    # one damping CAV slows by 1 m/s -> bin-0 mean speed over CAVs drops 1/3
    assert out[1].delay_vs_native_s[0, 0] == pytest.approx(100.0 / 3.0, rel=1e-2)
    assert out[2].delay_vs_native_s[0, 0] == pytest.approx(200.0, rel=1e-3)
    assert out[3].delay_vs_native_s[0, 0] == out[1].delay_vs_native_s[0, 0]
    assert len(log) == 1 and log[0].shape[0] == 3     # native + 2 distinct
    assert out[1].meta["head_native_dv_mean"] == pytest.approx(0.0, abs=1e-9)
    assert out[1].meta["dv_q90_tau"] > out[1].meta["dv_mean_tau"]


def test_native_reference_head_subtracts():
    log = []
    st = synth_state()
    st.veh_kin[: 3, 4] = 18.0          # CAVs slower -> head(native) = +200
    pred = make_pred(log, native_reference="head")
    native, one, _ = plans_for(st)
    out = pred.predict_many([LiveSample(st)], [native, one])
    assert out[1].meta["head_native_dv_mean"] == pytest.approx(200.0, rel=1e-3)
    assert out[1].delay_vs_native_s[0, 0] == pytest.approx(100.0 / 3.0, rel=1e-2)


def test_codes_and_fail_closed():
    from traffic_shepherd.planner.interface import PlanCandidate, WallSpec

    st = synth_state()
    pred = make_pred([])
    codes, _ = pred.plan_codes(st, PlanCandidate({"cav0": "damping", "cav1": "pi-external-u",
                                                  "cav2": "native"}))
    assert codes[:3].tolist() == [1, 2, 0] and (codes[3:] == -1).all()
    with pytest.raises(A11PredictorError):
        pred.plan_codes(st, PlanCandidate({"ghost": "native"}))
    wall = WallSpec("w1", ("cav0", "cav1"), 16.0, "assemble", hold_s=60.0)
    with pytest.raises(A11PredictorError):
        pred.plan_codes(st, PlanCandidate({"cav2": "native"}, walls=(wall,)))
    with pytest.raises(A11PredictorError):
        pred.predict_many([LiveSample(st), LiveSample(st)], [PlanCandidate({})])


def test_prediction_passes_planner_contract_and_planner_decides():
    from traffic_shepherd.planner.config import PlannerConfig
    from traffic_shepherd.planner.loop import Planner
    from traffic_shepherd.planner.safety import safety_check

    st = synth_state()
    pred = make_pred([])
    native, one, fleet = plans_for(st)
    cfg = PlannerConfig(max_walls=0, mode="controlled_experiment")
    out = pred.predict_many([LiveSample(st)], [native, one])
    for o in out:
        verdict = safety_check(o, cfg)
        assert verdict.ok
        assert set(o.trajectories_tracked) == set(v for v in st.slot_vid if v)
        arr = o.trajectories_tracked["cav0"]
        assert arr.shape == (1, 1, 121, 4)
    planner = Planner(pred, cfg, current_plan=native)
    sample = LiveSample(st, meta={"merge_observed": 1.0, "time_s": 400.0})
    decision = planner.decide([sample], 400.0)
    # every alternative is slower (positive dV) -> stays native
    assert all(a == "native" for a in decision.plan.per_cav.values())
    assert decision.groups and not decision.abstain


def test_planner_switches_when_saving_exceeds_bar():
    from traffic_shepherd.planner.config import PlannerConfig
    from traffic_shepherd.planner.loop import Planner

    def saving_head(bundle, state, plan_long):
        return -fake_head(bundle, state, plan_long)   # slower CAVs = saving

    st = synth_state()
    pred = A11Predictor(object(), rollout_fn=fake_rollout_factory([]), head_fn=saving_head,
                        fast=False)
    native, _, _ = plans_for(st)
    cfg = PlannerConfig(max_walls=0, mode="controlled_experiment")
    planner = Planner(pred, cfg, current_plan=native)
    decision = planner.decide([LiveSample(st, meta={"merge_observed": 1.0})], 400.0)
    # pi (code 2) slows a CAV by 2 m/s -> 66.7 s saving > 60 s bar; the
    # group check combines picks
    assert any(a != "native" for a in decision.plan.per_cav.values())


def test_safety_stats_ttc_and_collision():
    T, B = 4, 1
    pos = np.zeros((T + 1, B, K))
    spd = np.zeros((T + 1, B, K))
    msk = np.zeros((T + 1, B, K), bool)
    msk[:, 0, :2] = True
    pos[:, 0, 0] = 100.0 + 10.0 * np.arange(T + 1) * 0.5   # leader 10 m/s
    pos[:, 0, 1] = 80.0 + 20.0 * np.arange(T + 1) * 0.5    # follower 20 m/s
    spd[:, 0, 0] = 10.0
    spd[:, 0, 1] = 20.0
    lns = np.ones((T + 1, B, K), np.int64)
    acc = np.zeros((T, B, K))
    acc[0, 0, 1] = -4.0
    r = BatchRollout(pos=pos, spd=spd, acc=acc, lns=lns, msk=msk,
                     ids=np.repeat(np.arange(K)[None, None], T + 1, 0),
                     length=np.full((B, K), 5.0), a0=np.zeros((B, K)))
    s = AP.safety_stats(r, 0, T, ["L", "F"] + [None] * (K - 2))
    # gap at step 1: (105 - 90) - 5 = 10 m, closing 10 m/s -> 1.0 s ... min later
    assert s.has_closing_pair and s.min_ttc_s < 1.0
    assert s.min_ttc_pair == ("F", "L")
    assert s.hard_brake_s == 0.5
    assert s.ttc_lt1p5_s > 0 and s.ttc_lt3_s >= s.ttc_lt1p5_s
    assert s.collision                     # gap goes negative by step 2-3


def test_long_plan_matches_training_construction():
    """long_plan_from_rollout == training_plan_long's recipe on arrays."""
    from features.idea_a import CAV_ACCEL_SCALE, CAV_TGT_SPEED_SCALE
    from model.paired_effect import long_plan_summary

    rng = np.random.default_rng(0)
    T, B = 240, 2
    spd = rng.uniform(0, 30, (T + 1, B, K))
    acc = rng.uniform(-3, 2, (T, B, K))
    a0 = rng.uniform(-1, 1, (B, K))
    msk = np.ones((T + 1, B, K), bool)
    msk[100:, :, 2] = False                    # slot 2 leaves at tick 100
    msk[150:, :, 3] = False
    msk[160:, :, 3] = True                     # slot 3 reused later: stays dead
    ids = np.repeat(np.arange(K)[None, None], T + 1, 0).repeat(1, 1)
    ids = np.repeat(np.arange(K)[None, None, :], T + 1, 0).repeat(B, 1)
    ids[160:, :, 3] = 99
    r = BatchRollout(pos=np.zeros_like(spd), spd=spd, acc=acc, lns=np.ones_like(ids),
                     msk=msk, ids=ids, length=np.full((B, K), 5.0), a0=a0)
    is_cav = np.array([1, 0, 1, 1, 0, 0, 1, 0], bool)
    got = AP.long_plan_from_rollout(r, is_cav, 120.0)
    # training recipe: cav_plan fields stored float16 at the pack scales,
    # alive = present0 * cumprod(target_mask[t0 : t0+n-1])
    n = 240
    plan0 = (spd[:n] / CAV_TGT_SPEED_SCALE).astype(np.float16)
    a_tick = np.concatenate([a0[None], acc[: n - 1]], 0)
    plan1 = (a_tick / CAV_ACCEL_SCALE).astype(np.float16)
    target = msk[1:n] & msk[: n - 1] & (ids[1:n] == ids[: n - 1])
    alive = np.cumprod(np.concatenate([msk[:1], target[: n - 1]], 0).astype(np.int64), 0) > 0
    sp = torch.as_tensor(plan0.astype(np.float32)) * float(CAV_TGT_SPEED_SCALE)
    ac = torch.as_tensor(plan1.astype(np.float32)) * float(CAV_ACCEL_SCALE)
    cav = torch.as_tensor((is_cav[None, None] & alive).astype(np.float32))
    want = long_plan_summary(sp, ac, cav, horizon_s=120.0)
    assert torch.allclose(got, want, atol=1e-5)
    assert got.shape == (B, 60)


def test_base_departures_upto_matches_parse_departures(tmp_path):
    from features.featurizer import parse_departures

    run = tmp_path / "run"
    run.mkdir()
    rows = []
    for k in range(6):
        t = 0.5 * k
        rows.append({"time_s": t, "departed_ids": [f"mainline.{k}", f"ramp.{k}"],
                     "vehicles": {f"mainline.{k}": {"road_id": None},
                                  f"ramp.{k}": {"road_id": None}}})
    (run / "physics_steps.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    full = parse_departures(tmp_path)
    want = [(t, e) for t, e in zip(*full.arrays()) if t <= 1.5 + 1e-9]
    got = base_departures_upto(tmp_path, 1.5)
    assert [(float(t), int(e)) for t, e in want] == got
    with pytest.raises(Exception):
        base_departures_upto(tmp_path / "nope", 1.5)


def test_live_state_from_real_row_shapes():
    base = Path("E:/research-dense/runs/campaign_stage1_chunk4/w8676m2700p20t360__base")
    man = base / "saved_state" / "STATE_MANIFEST.json"
    if not man.is_file():
        pytest.skip("TRAIN base run not on disk")
    row = json.loads(man.read_text(encoding="utf-8"))["physics_row_at_t0"]
    dep = DepartureHistory(None, 360.0)
    st = live_state_from_row(row, dep)
    assert st.veh_kin.shape == (192, 7) and st.veh_mask.sum() == len(row["vehicles"])
    assert set(st.vehicle_slots) == set(row["vehicles"])
    cavs = set(row["cav_ids"]) & set(row["vehicles"])
    assert set(st.cav_slots) == cavs


def _real_bundle():
    if not AP.A11_MEMBER_PATH.is_file():
        pytest.skip("A11 member not on disk")
    return AP.load_a11_bundle().bundle


def test_batched_head_equals_wm_api_single():
    from wm_api.eval import paired_effect_quantiles_for_plan

    bundle = _real_bundle()
    st = synth_state(3, 3)
    st.counter[0, :3] = 2.0
    rng = np.random.default_rng(1)
    pl = torch.as_tensor(rng.uniform(0, 25, (3, 60)).astype(np.float32))
    got = AP.head_dv_quantiles(bundle, st, pl)
    for i in range(3):
        want = paired_effect_quantiles_for_plan(
            bundle, t0_kin=st.veh_kin, t0_mask=st.veh_mask, t0_counter=st.counter,
            summary_raw=None, plan_long=pl[i])
        assert np.allclose(got[i], want, atol=1e-9)


def test_fast_paths_exact_on_real_bundle():
    from closedloop.fast_hdv import (fast_bundle, fast_project_same_lane_gaps)
    from rollout2.validity import project_same_lane_gaps

    bundle = _real_bundle()
    st = synth_state(3, 4)
    st.veh_kin[1, 0] = st.veh_kin[0, 0] + 3.0    # overlapping pair
    codes = np.full((2, K), -1)
    codes[0, :3] = 0
    codes[1, :3] = 1
    r1 = AP.batched_rollout(bundle, st, codes, n_steps=6, t0_s=400.0)
    r2 = AP.batched_rollout(fast_bundle(bundle), st, codes, n_steps=6, t0_s=400.0)
    assert np.allclose(r1.pos, r2.pos, atol=1e-9) and np.allclose(r1.spd, r2.spd, atol=1e-9)
    rng = np.random.default_rng(3)
    s = torch.as_tensor(rng.uniform(0, 60, (3, 12)))
    lane = torch.as_tensor(rng.integers(0, 3, (3, 12)))
    v = torch.as_tensor(rng.uniform(0, 30, (3, 12)))
    ln = torch.full((3, 12), 5.0, dtype=torch.float64)
    pres = torch.as_tensor(rng.uniform(size=(3, 12)) > 0.2)
    a = project_same_lane_gaps(s, lane, v, ln, pres)
    b = fast_project_same_lane_gaps(s, lane, v, ln, pres)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[2].gap_projection == b[2].gap_projection


def test_policy_registry_and_random_seeded():
    from closedloop import policies as P
    from closedloop.policies import DecisionContext

    assert {"pl01_a11", "fleet_wm", "always_native", "always_pi", "random"} <= set(P.POLICIES)
    st = synth_state()
    ctx = DecisionContext(world="wX", seed=1, t0_s=0.0, now_s=0.0, decision_index=2,
                          state=st, live_cavs=("cav0", "cav1"), prev_assignments={})
    a = P.make_policy("random").decide(ctx)
    b = P.make_policy("random").decide(ctx)
    assert a.assignments == b.assignments and len(set(a.assignments.values())) == 1
    assert P.make_policy("always_pi").decide(ctx).assignments == {
        "cav0": "pi-external-u", "cav1": "pi-external-u"}
    with pytest.raises(ValueError):
        P.register_policy("random")(object)


def test_rollout_cache_reused_across_calls():
    log = []
    st = synth_state()
    pred = make_pred(log)
    native, one, fleet = plans_for(st)
    pred.predict_many([LiveSample(st)], [native])
    out = pred.predict_many([LiveSample(st)], [one, fleet])
    assert len(log) == 2 and log[1].shape[0] == 2       # native not re-rolled
    assert pred.calls[-1]["n_cached"] == 1
    assert out[0].delay_vs_native_s[0, 0] > 0


def test_identical_skip_matches_full_rollouts_real_bundle():
    from closedloop.fast_hdv import fast_bundle, install_fast_gap_projection
    from closedloop.skip import identical_slots

    bundle = _real_bundle()
    install_fast_gap_projection()
    fb = fast_bundle(bundle)
    st = synth_state(3, 4)
    st.veh_kin[3:, 4] = 8.0                       # slow HDVs ahead -> leaders
    cav = st.is_cav & st.veh_mask
    ref = np.where(cav, 0, -1)
    n = 16
    r0 = AP.batched_rollout(fb, st, ref[None], n_steps=n, t0_s=400.0, trace=True)
    for a in (1, 2):
        ident = identical_slots(fb, r0, 0, ref, a, cav, 400.0, n)
        slots = np.flatnonzero(cav)
        codes = np.repeat(ref[None], len(slots), 0)
        for j, s in enumerate(slots):
            codes[j, s] = a
        rf = AP.batched_rollout(fb, st, codes, n_steps=n, t0_s=400.0)
        for j, s in enumerate(slots):
            same = np.abs(rf.pos[:, j] - r0.pos[:, 0]).max() <= 1e-9
            if ident[s]:
                assert same, (a, s)


def test_junction_rows_excluded_from_rollout_but_tracked():
    log = []
    st = synth_state()
    st.veh_kin[1, 1:4] = 0.0                       # cav1 on a junction lane
    pred = make_pred(log)
    native, one, fleet = plans_for(st)
    out = pred.predict_many([LiveSample(st)], [native, fleet])
    assert log[0][:, 1].tolist() == [-1, -1]       # cav1 not driven in the rollout
    assert np.isnan(out[1].trajectories_tracked["cav1"][..., 1]).all()
    assert out[1].meta["n_junction_excluded"] == 1
