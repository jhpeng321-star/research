"""Generated-only witnesses for the frozen training/evaluation time contract.

These checks describe source behavior.  They do not inspect a checkpoint or
measure any model's performance on a 120-second horizon.
"""
from __future__ import annotations

import pytest

from evalh.protocol import HORIZONS_S
from model import constants as C
from train.dataset import map_outcome_targets
from train.idea_a import IdeaATrainConfig


def _outcome(t: float, tau: float, delay: float, available: bool) -> dict:
    return {
        "t": t,
        "tau": tau,
        "total_delay_ref_s": delay,
        "ttc_lt3_veh_s": delay / 10.0,
        "hard_brake_s": delay / 100.0,
        "collision_any": float(delay > 5.0),
        "tau_available": available,
    }


def test_train_target_mapping_last_duplicate_tau_wins_across_decision_times():
    # Both generated rows are at tau=120, but their decision timestamps and
    # values differ.  Training's by_tau dictionary retains the final row;
    # there is no timestamp dimension or same-decision check in this mapper.
    early = _outcome(t=100.0, tau=120.0, delay=2.0, available=True)
    late = _outcome(t=110.0, tau=120.0, delay=9.0, available=False)
    idx = C.TAUS.index(120.0)

    targets, mask, collision = map_outcome_targets([early, late])
    assert targets[idx].tolist() == pytest.approx([9.0, 0.9, 0.09])
    assert not bool(mask[idx])
    assert collision[idx] == 1.0

    reversed_targets, reversed_mask, reversed_collision = \
        map_outcome_targets([late, early])
    assert reversed_targets[idx].tolist() == pytest.approx([2.0, 0.2, 0.02])
    assert bool(reversed_mask[idx])
    assert reversed_collision[idx] == 0.0


def test_frozen_training_summary_window_is_shorter_than_eval_report_horizon():
    # IdeaACorpus constructs summary_raw from its actual (at most 20-step)
    # teacher window.  Eval's HORIZONS_S includes a 120-second report point,
    # and its variant predictor defaults to max(HORIZONS_S) / DT_S steps.
    # A 120-second rollout is thus an extrapolation of a 10-second trained
    # summary duration, not evidence that the M1 correction head learned a
    # 120-second CAV-motion summary contract.
    train_window_s = IdeaATrainConfig().rollout_horizon_steps * C.DT_S
    eval_window_s = max(HORIZONS_S)
    assert train_window_s == 10.0
    assert eval_window_s == 120.0
    assert eval_window_s / train_window_s == 12.0
