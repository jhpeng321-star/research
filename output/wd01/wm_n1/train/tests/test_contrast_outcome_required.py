"""T6 pair-dense contrast: opt-in outcome-required eligible pool.

The M1FIX1 training census (overnight T6) found that the pair-aware sampler
spent 61.7% of its forced sibling pairs on HDV-only pairs whose treatment arm
carries no t0 outcome row, so the registered outcome-contrast sub-term was
absent on those steps (console ``out=0/0``).  The opt-in
``contrast_require_outcome_support`` flag restricts the eligible pool to pairs
that share at least one realised outcome tau, so every forced pair carries the
outcome contrast.  These are CPU-only synthetic contracts: no pack, no
simulation, no training.
"""
from __future__ import annotations

import train.idea_a as ia
from model import constants as C
from train import contrast_support as CS
from test_contrastfix_recovery import (
    _IdSource,
    _PairSource,
    _mask,
    _pair,
)
from test_idea_a_fakes import seeded_bundle


def _mixed_source():
    """Group a: outcome-shared (HDV disjoint).  Group b: HDV-only."""
    taus = C.N_TAU
    specs = []
    specs += _pair(_mask([0, 1, 2], taus), _mask([2, 3], taus),
                   [0, 1], [2, 3], key="a@t0=0")
    specs += _pair(_mask([0], taus), _mask([1], taus),
                   [0, 1], [0, 1], key="b@t0=0")
    return _PairSource(specs)


def test_build_eligible_pair_pool_require_outcome_keeps_only_outcome_pairs():
    src = _mixed_source()
    default = CS.build_eligible_pair_pool(src, src._items)
    required = CS.build_eligible_pair_pool(
        src, src._items, require_outcome=True)
    # both groups are eligible under the historical outcome-OR-HDV rule
    assert default == [(0, 1), (2, 3)]
    # only the outcome-shared group survives the opt-in requirement
    assert required == [(0, 1)]


def test_contrast_pool_honours_the_config_flag():
    src = _mixed_source()
    off = ia.IdeaATrainer(
        ia.IdeaATrainConfig(out_dir="unused", num_workers=0),
        corpus=None, bundle=seeded_bundle(0))
    on = ia.IdeaATrainer(
        ia.IdeaATrainConfig(out_dir="unused", num_workers=0,
                            contrast_require_outcome_support=True),
        corpus=None, bundle=seeded_bundle(0))
    assert off._contrast_pool(src, src._items) == [(0, 1), (2, 3)]
    assert on._contrast_pool(src, src._items) == [(0, 1)]


def test_fit_refuses_when_required_outcome_pair_is_absent(tmp_path):
    """An HDV-only source cannot satisfy the outcome-required pool."""
    taus = C.N_TAU
    specs = _pair(_mask([0], taus), _mask([1], taus), [0, 1], [0, 1])
    src = _IdSource(specs, identity="ok")
    cfg = ia.IdeaATrainConfig(epochs=1, steps_per_epoch=1, batch_size=2,
                              out_dir=str(tmp_path / "out"), num_workers=0,
                              contrast_require_outcome_support=True)
    tr = ia.IdeaATrainer(cfg, src, member=1, seed=101,
                         bundle=seeded_bundle(101))
    try:
        tr.fit()
    except ia.IdeaAConfigError as exc:
        assert "no eligible sibling pair" in str(exc)
        assert "shared outcome-tau" in str(exc)
    else:                                             # pragma: no cover
        raise AssertionError(
            "fit ran with contrast_require_outcome_support=True and no "
            "outcome-sharing pair")


def test_flag_default_off_preserves_historical_pool():
    cfg = ia.IdeaATrainConfig(out_dir="unused", num_workers=0)
    assert cfg.contrast_require_outcome_support is False
