"""Synthetic tests for the label-time fix (a) t0 assertion and (b) exclusion.

No real pack, no CAL/HOLDOUT data, no simulation.  The pack is built by the
byte-layout stub ``packsynth.write_pack``; the trainer telemetry test fits
tiny explicit synthetic windows only.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from model.outcome_norm import OutcomeNorm
from packsynth import write_pack
from train.dataset import (OUTCOME_SUPERVISION_EXCLUDED_ARMS, MemmapCorpus,
                           is_outcome_supervision_excluded, label_t0_rows,
                           outcome_supervision_exclusion)
from train.idea_a import IdeaATrainConfig, IdeaATrainer, fit_outcome_norm


def _train_runs():
    """All-TRAIN sibling arms (no INDEX -> the legacy mixed-table read path)."""
    return [
        dict(chunk="chunk4", world="w1", arm="native", policy="native",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w1", arm="base", policy="damping",
             split="TRAIN", t0_s=540.0),
        dict(chunk="chunk4", world="w2", arm="native_base", policy="native",
             split="TRAIN", t0_s=600.0),
        dict(chunk="chunk4", world="w2", arm="pi", policy="pi",
             split="TRAIN", t0_s=600.0),
    ]


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "pack"
    write_pack(root, _train_runs())
    return MemmapCorpus(root)


# ------------------------------------------------------- (a) t0 assertion ----
def test_label_t0_rows_counts_rows_selected_at_item_t0(corpus):
    # every TRAIN item carries one t0 row per tau (5 taus) in the fixture.
    item = corpus[0]
    assert item["label_t0_rows"] == 5
    assert item["label_t0_violations"] == 0
    assert item["outcome_supervision_excluded"] is False


def test_label_t0_rows_legacy_none_reports_zero():
    rows = [{"t": 0.0, "tau": 30.0, "total_delay_ref_s": 1.0,
             "ttc_lt3_veh_s": 0.0, "hard_brake_s": 0.0, "collision_any": 0.0,
             "tau_available": True}]
    assert label_t0_rows(rows, None) == 0
    assert label_t0_rows(rows, 0.0) == 1


def test_label_t0_rows_ignores_other_decision_times():
    rows = [
        {"t": 600.0, "tau": 30.0, "total_delay_ref_s": 9.0,
         "ttc_lt3_veh_s": 0.0, "hard_brake_s": 0.0, "collision_any": 0.0,
         "tau_available": True},
        {"t": 0.0, "tau": 30.0, "total_delay_ref_s": 1.0,
         "ttc_lt3_veh_s": 0.0, "hard_brake_s": 0.0, "collision_any": 0.0,
         "tau_available": True},
    ]
    # only the item's own t0 row is selected
    assert label_t0_rows(rows, 0.0) == 1
    # no row at this decision time -> nothing asserted
    assert label_t0_rows(rows, 5.0) == 0


def test_label_t0_rows_duplicate_t0_row_fails_closed():
    dup = [dict(t=0.0, tau=30.0, total_delay_ref_s=1.0, ttc_lt3_veh_s=0.0,
                hard_brake_s=0.0, collision_any=0.0, tau_available=True),
           dict(t=0.0, tau=30.0, total_delay_ref_s=2.0, ttc_lt3_veh_s=0.0,
                hard_brake_s=0.0, collision_any=0.0, tau_available=True)]
    with pytest.raises(ValueError, match="duplicate"):
        label_t0_rows(dup, 0.0)


# ---------------------------------------------------- (b) supervision drop ---
def test_excluded_arms_tuple_is_the_registered_pair():
    assert OUTCOME_SUPERVISION_EXCLUDED_ARMS == ("base", "native_base")
    assert is_outcome_supervision_excluded({"arm": "base"})
    assert is_outcome_supervision_excluded("native_base")
    assert not is_outcome_supervision_excluded({"arm": "native"})


def test_excluded_items_deliver_zero_outcome_supervision(corpus):
    excluded_positions = [i for i in range(len(corpus))
                          if is_outcome_supervision_excluded(corpus[i])]
    assert excluded_positions == [1, 2]          # w1 base + w2 native_base
    for pos in excluded_positions:
        item = corpus[pos]
        assert item["outcome_supervision_excluded"] is True
        assert not bool(item["outcome_mask"].any())
        assert not bool(item["outcome_quantiles"].abs().any())
        assert not bool(item["collision"].abs().any())
        y, m, c = corpus.outcome_target(pos)
        assert not m.any() and not np.any(y) and not np.any(c)
    # a non-excluded item keeps its supervision
    keep = corpus[0]
    assert keep["outcome_supervision_excluded"] is False
    assert bool(keep["outcome_mask"].any())


def test_exclusion_receipt_counts_and_lists_excluded_items(corpus):
    receipt = outcome_supervision_exclusion(corpus)
    assert receipt["arms"] == list(OUTCOME_SUPERVISION_EXCLUDED_ARMS)
    assert receipt["count"] == 2
    assert receipt["items"] == ["w1__base", "w2__native_base"]


def test_outcome_norm_excludes_the_excluded_arms(corpus):
    full = fit_outcome_norm(corpus)
    kept = [i for i in range(len(corpus))
            if not is_outcome_supervision_excluded(corpus[i])]
    targets = [corpus.outcome_target(i)[0] for i in kept]
    masks = [corpus.outcome_target(i)[1] for i in kept]
    assert full.to_dict() == OutcomeNorm.fit(targets, masks).to_dict()


def test_outcome_supervision_exclusion_empty_for_synthetic_windows():
    receipt = outcome_supervision_exclusion(object())
    assert receipt["count"] == 0 and receipt["items"] == []


# ------------------------------------------------ trainer telemetry/receipt --
def _small_windows(n=1):
    import test_idea_a_fakes as fakes

    return fakes.make_windows(n)


def test_trainer_logs_label_t0_fields_on_the_step_record(tmp_path):
    cfg = IdeaATrainConfig(epochs=1, steps_per_epoch=2, batch_size=2,
                           out_dir=str(tmp_path / "out"), num_workers=0,
                           state_global_token=False)
    tr = IdeaATrainer(cfg, corpus=None, bundle=_bundle(), device="cpu")
    tr.fit(windows=_small_windows(2))
    rows = [json.loads(line) for line in
            (tmp_path / "out" / "train_log.jsonl").read_text(
                encoding="utf-8").strip().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["label_t0_asserted"] == 0        # synthetic meta has none
        assert row["label_t0_violations"] == 0


def test_trainer_label_t0_telemetry_sums_batch_meta():
    batch = {"meta": [
        {"label_t0_rows": 5, "label_t0_violations": 0},
        {"label_t0_rows": 3, "label_t0_violations": 0},
        {"arm": "native"},
    ]}
    out = IdeaATrainer.label_t0_telemetry(batch)
    assert out == {"label_t0_asserted": 8, "label_t0_violations": 0}
    bad = {"meta": [{"label_t0_rows": 5, "label_t0_violations": 2}]}
    assert IdeaATrainer.label_t0_telemetry(bad)["label_t0_violations"] == 2


def test_train_receipt_records_outcome_supervision_exclusion(tmp_path):
    cfg = IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                           out_dir=str(tmp_path / "out"), num_workers=0,
                           state_global_token=False)
    tr = IdeaATrainer(cfg, corpus=None, bundle=_bundle(), device="cpu")
    receipt = tr.fit(windows=_small_windows(1))
    assert receipt["outcome_supervision_excluded_arms"] == \
        list(OUTCOME_SUPERVISION_EXCLUDED_ARMS)
    assert receipt["outcome_supervision_excluded_count"] == 0
    assert receipt["outcome_supervision_excluded_items"] == []


def _bundle():
    import test_idea_a_fakes as fakes

    return fakes.seeded_bundle(0)
