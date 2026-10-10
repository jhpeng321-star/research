"""A6 state-only global token on the wm_api outcome path (synthetic).

The scorer's outcome head must build the correction head's global token from
the t0 state with the checkpoint's ``StateGlobalToken`` weights, and must
refuse to score a state-token checkpoint with a zero/absent state.  With the
flag OFF it reproduces the legacy zero-token path.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from train.idea_a import IdeaATrainConfig, build_idea_a_bundle
from wm_api.engine import _state_channels, outcome_head
from wm_api.state import StateSnapshot

from _tiny import TINY_CONFIG, tiny_state


def _scratch(state=None, t=3, k=8):
    # DRAFT migration: original t0 + explicit identity replaces the old
    # incoherent trajectory/StateSnapshot pairing and all-vehicle selection.
    state = tiny_state(counters=True) if state is None else state
    data = _state_channels(state, k)
    pos = np.repeat(data["t0_kin"][..., 0][None], t + 1, axis=0)
    spd = np.repeat(data["t0_kin"][..., 4][None], t + 1, axis=0)
    acc = np.zeros((t, 1, k), dtype=np.float64)
    msk = np.repeat(data["t0_mask"][None], t + 1, axis=0)
    ids = np.repeat(data["t0_slot_id"][None], t + 1, axis=0)
    return pos, spd, acc, msk, ids


def _bundle(state_global_token=True):
    torch.manual_seed(11)
    cfg = IdeaATrainConfig(**TINY_CONFIG,
                           state_global_token=state_global_token)
    bundle = build_idea_a_bundle(cfg)
    # the wm_api loader casts the learned modules to float64 (its math dtype);
    # mirror that so the outcome path runs exactly as in production
    for name in ("hdv", "corr", "state_enc"):
        module = getattr(bundle, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(torch.float64)
            module.eval()
    return bundle


def _empty_state() -> StateSnapshot:
    return StateSnapshot(ids=[], lane_phys=np.zeros(0, dtype=np.int64),
                         s_down_m=np.zeros(0), v_mps=np.zeros(0),
                         a_mps2=np.zeros(0), length_m=np.zeros(0),
                         is_cav=np.zeros(0, dtype=bool), t0_s=0.0)


def test_outcome_head_uses_state_token_not_zeros():
    bundle = _bundle(True)
    assert bundle.state_enc is not None
    state = tiny_state(counters=True)
    pos, spd, acc, msk, ids = _scratch(state)
    q0, _ = outcome_head(bundle, pos, spd, acc, msk, state=state, slot_ids=ids)
    # a second, different t0 state must move the outcome head output, proving
    # the state token (not a constant zero vector) reaches it
    alt = tiny_state(counters=True)
    # Counter-only state change holds the CAV summary fixed, isolating A6.
    alt.entrance_counters += 5.0
    q1, _ = outcome_head(bundle, pos, spd, acc, msk, state=alt, slot_ids=ids)
    assert not np.allclose(q0, q1)


def test_outcome_head_refuses_missing_state_for_state_token_checkpoint():
    bundle = _bundle(True)
    pos, spd, acc, msk, ids = _scratch()
    with pytest.raises(ValueError):
        outcome_head(bundle, pos, spd, acc, msk, state=None, slot_ids=ids)


def test_outcome_head_refuses_empty_state_for_state_token_checkpoint():
    bundle = _bundle(True)
    pos, spd, acc, msk, ids = _scratch(_empty_state())
    with pytest.raises(ValueError):
        outcome_head(bundle, pos, spd, acc, msk, state=_empty_state(), slot_ids=ids)


def test_outcome_head_flag_off_needs_cav_identity_but_not_counter():
    bundle = _bundle(False)
    assert bundle.state_enc is None
    state = tiny_state(counters=False)
    pos, spd, acc, msk, ids = _scratch(state)
    q, _ = outcome_head(bundle, pos, spd, acc, msk, state=state, slot_ids=ids)
    assert np.all(np.isfinite(q))
    with pytest.raises(ValueError, match="t0_is_cav"):
        outcome_head(bundle, pos, spd, acc, msk, state=None, slot_ids=ids)
