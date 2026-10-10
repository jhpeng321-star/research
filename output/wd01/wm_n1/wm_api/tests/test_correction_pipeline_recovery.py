"""DRAFT, NOT RUN: API identity/state alignment, pending reviewed guard.

Collect separately from evalh's same-basename draft (or use importlib mode).
All trajectories and model weights are synthetic in memory.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from corr import CorrConfig, StateGlobalToken
from train.g4_real import RealCorrBundle
from wm_api.engine import _state_channels, build_world, outcome_head, rollout_member
from wm_api.state import ObstacleRecord, StateSnapshot


def _state():
    return StateSnapshot(
        ids=["junction", "self-cav", "neighbor"], lane_phys=np.array([-1, 1, 0]),
        s_down_m=np.array([999., 0., 100.]), v_mps=np.array([99., 10., 20.]),
        a_mps2=np.array([9., -2., 1.]), length_m=np.array([9., 4.5, 7.]),
        is_cav=np.array([True, True, False]), entrance_counters=np.zeros((2, 30)),
        obstacles=(ObstacleRecord("fixed", 1, 200., 6.),))


def _bundle(on=True):
    torch.manual_seed(1)
    return SimpleNamespace(corr=RealCorrBundle(CorrConfig(d_model=16)).eval(),
                           state_enc=StateGlobalToken(16).eval() if on else None,
                           cfg=SimpleNamespace(state_global_token=on))


def _trajectory(state):
    data = _state_channels(state, 4)
    kin = data["t0_kin"]
    pos = np.stack([kin[..., 0], kin[..., 0] + np.array([[5., 10., 0., 0.]]),
                    kin[..., 0] + np.array([[11., 20., 0., 0.]])])
    spd = np.repeat(kin[..., 4][None], 3, axis=0)
    acc = np.zeros((2, 1, 4))
    masks = np.repeat(data["t0_mask"][None], 3, axis=0)
    ids = np.repeat(data["t0_slot_id"][None], 3, axis=0)
    return pos, spd, acc, masks, ids


def test_snapshot_filters_junctions_and_matches_build_world_roster():
    state = _state()
    plan = SimpleNamespace(wall=None)
    world, _, _ = build_world(state, plan, 4)
    data = _state_channels(state, 4)
    from rollout2.engine import _t0_kin

    np.testing.assert_array_equal(data["t0_kin"], _t0_kin(world).numpy())
    np.testing.assert_array_equal(data["t0_mask"], world.present.numpy())
    np.testing.assert_array_equal(data["t0_is_cav"], world.is_cav.numpy())
    pos, spd, acc, masks, ids = _trajectory(state)
    q_state, raw_state = outcome_head(_bundle(), pos, spd, acc, masks,
                                     state=state, slot_ids=ids)
    q_explicit, raw_explicit = outcome_head(_bundle(), pos, spd, acc, masks,
                                           slot_ids=ids, **data)
    np.testing.assert_array_equal(q_state, q_explicit)
    np.testing.assert_array_equal(raw_state, raw_explicit)
    assert raw_state[0] == 0.25 and raw_state[9] == -2. and raw_state[12] == 11.


def test_api_hdv_motion_invariant_and_cav_motion_sensitive():
    state, bundle = _state(), _bundle()
    pos, spd, acc, masks, ids = _trajectory(state)
    expected_q, expected = outcome_head(bundle, pos, spd, acc, masks, state=state, slot_ids=ids)
    pos[1:, 0, 1] += 1000.
    spd[1:, 0, 1] += 100.
    acc[:, 0, 1] += 20.
    q, raw = outcome_head(bundle, pos, spd, acc, masks, state=state, slot_ids=ids)
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(q, expected_q)
    pos[-1, 0, 0] += 50.
    q, raw = outcome_head(bundle, pos, spd, acc, masks, state=state, slot_ids=ids)
    assert not np.array_equal(raw, expected)
    assert not np.array_equal(q, expected_q)


@pytest.mark.parametrize("field", ["t0_kin", "t0_mask", "t0_is_cav", "t0_slot_id", "t0_counter"])
def test_explicit_partial_channels_fail_closed_even_with_snapshot(field):
    state = _state()
    data = _state_channels(state, 4)
    data.pop(field)
    pos, spd, acc, masks, ids = _trajectory(state)
    with pytest.raises(ValueError):
        outcome_head(_bundle(), pos, spd, acc, masks, state=state, slot_ids=ids, **data)


def test_missing_ids_counter_or_required_encoder_fail_closed():
    state = _state()
    pos, spd, acc, masks, ids = _trajectory(state)
    with pytest.raises(ValueError, match="slot_ids"):
        outcome_head(_bundle(), pos, spd, acc, masks, state=state)
    state.entrance_counters = None
    with pytest.raises(ValueError, match="counter"):
        outcome_head(_bundle(), pos, spd, acc, masks, state=state, slot_ids=ids)
    bundle = _bundle()
    bundle.state_enc = None
    with pytest.raises(ValueError, match="no state encoder"):
        outcome_head(bundle, pos, spd, acc, masks, state=state, slot_ids=ids)


def test_a6_off_allows_absent_counter_but_not_missing_identity():
    state = _state()
    state.entrance_counters = None
    pos, spd, acc, masks, ids = _trajectory(state)
    q, _ = outcome_head(_bundle(False), pos, spd, acc, masks, state=state, slot_ids=ids)
    assert np.isfinite(q).all()
    with pytest.raises(ValueError, match="t0_is_cav"):
        outcome_head(_bundle(False), pos, spd, acc, masks, slot_ids=ids)


def test_rollout_member_carries_identity_after_each_step(monkeypatch):
    # Source-only DRAFT: stubs avoid physics, model rollout, and file reads.
    import rollout2.baseline
    import rollout2.step
    import wm_api.engine as engine

    monkeypatch.setattr(rollout2.baseline, "Params", lambda unused: SimpleNamespace(dt=0.5))
    state = _state()
    plan = SimpleNamespace(wall=None, n_segments=1,
                           segment_codes=[{"self-cav": 0}])
    seen = {}
    calls = []
    def step(world, actions, *, bundle, config, t_s, rng):
        calls.append(config.state.next_slot_id.clone())
        if len(calls) == 1:
            world.present[0, 0] = False
        else:
            world.present[0, 0] = True
            world.slot_id[0, 0] = config.state.next_slot_id[0]
            world.is_cav[0, 0] = False
            world.s_m[0, 0] = -500.
        return world, SimpleNamespace(counts=SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(rollout2.step, "step_once", step)
    monkeypatch.setattr(engine, "_add_counts", lambda a, b: a)
    original = engine.outcome_head
    def capture(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "outcome_head", capture)
    result = rollout_member(SimpleNamespace(bundle=_bundle()), state, plan,
                            n_steps=2, t0_s=0., seed=1, arrivals_enabled=False,
                            counters=None, k_slots=4)
    assert seen["t0_is_cav"].tolist() == [[True, False, False, False]]
    assert seen["slot_ids"][:, 0, 0].tolist() == [0, 0, 4]
    assert seen["t0_slot_id"].tolist() == [[0, 1, 2, 3]]
    assert calls[0].tolist() == [4]
    assert result["summary_raw"][12] == 0.  # departed before first next state
