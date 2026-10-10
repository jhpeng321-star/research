"""Pair-dense batching (T2) synthetic contracts; no pack, parquet or SUMO.

The registered CONTRAST loss trained on ~1 eligible sibling pair per batch of
8 (see ``train_log`` ``contrast[pairs=..]``).  ``pairs_per_batch=K`` makes every
batch drawn from the eligible pool hold exactly K complete pairs.  These tests
pin the contract on hand-built item metadata only:

* every pool-drawn batch holds exactly K eligible pairs (K=1..6);
* the epoch is a pure function of ``(seed, epoch, pairs)`` and the in-process
  path is byte-identical to :class:`PairAwareBatchSampler`;
* a pair whose support is NOT in the eligible pool is never forced into a
  batch (the "excluded item" contract), and coverage batches never form an
  unsupported pair;
* ``label_t0_rows`` telemetry survives the reorder and is still summed;
* ``--pairs-per-batch`` / ``--lambda-contrast`` parse, and a config whose
  ``2K`` exceeds ``batch_size`` fails closed.
"""
from __future__ import annotations

import pytest

from train.dataset import (PairAwareBatchSampler, batch_has_pair,
                           contrast_batch_indices, contrast_pairs,
                           epoch_batches)
from train.idea_a import (IdeaAConfigError, IdeaATrainConfig, IdeaATrainer,
                          build_parser)

TAUS = 5


def _it(world, arm, policy, *, t0=100.0, rows=3):
    return dict(world=world, arm=arm, policy=policy, chunk="chunk4",
                split="TRAIN", t0_s=t0, pair_key=f"{world}@t0={t0:g}",
                label_t0_rows=rows, label_t0_violations=0)


def _items(*, excluded_treatment=False):
    """Hand-built metadata with a known eligible pair count.

    Positions 0..5 form group ``a`` (one native reference, five treatments);
    positions 6/7 form group ``b`` (reference + treatment); positions 8/9 are
    a reference-free group (never pairable); position 10 is an extra group-a
    treatment excluded from the eligible pool when requested.
    """
    items = [
        _it("a", "native", "native"),        # 0: group-a reference
        _it("a", "pi", "pi"),                # 1
        _it("a", "base", "damping"),         # 2
        _it("a", "switching1", "switching"),  # 3
        _it("a", "fixed1", "fixed"),         # 4
        _it("a", "pi2", "pi"),               # 5
        _it("b", "native", "native"),        # 6: group-b reference
        _it("b", "pi", "pi"),                # 7
        _it("c", "native", "switching"),     # 8: policy switching -> not a ref
        _it("c", "switching2", "switching"),  # 9
    ]
    if excluded_treatment:
        items.append(_it("a", "base2", "damping"))   # 10: omitted from pool
    return items


def _pairs_excluding(items, omitted_treatment=None):
    pairs = contrast_pairs(items)
    if omitted_treatment is None:
        return pairs
    return [p for p in pairs if p[0] != omitted_treatment]


def _n_pairs(positions, items):
    local = contrast_batch_indices([items[p] for p in positions])
    return int(local["a"].numel())


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_pool_batches_hold_exactly_k_pairs(k):
    items = _items()
    pairs = _pairs_excluding(items)
    assert len(pairs) == 6                       # group a x5 + group b x1
    batches = epoch_batches(items, batch_size=max(2 * k, 8), seed=11,
                            epoch=0, pairs=pairs, pairs_per_batch=k)
    pool_batches = [b for b in batches
                    if batch_has_pair(b, items, pairs)]
    assert len(pool_batches) == 6 // k
    for positions in pool_batches:
        assert _n_pairs(positions, items) == k
    # the pool is fully consumed: the union of pool batches is exactly the
    # set of paired positions (a shared reference may recur across batches)
    covered = {p for b in pool_batches for p in b}
    assert covered == {p for pair in pairs for p in pair}


def test_pair_dense_is_deterministic_and_sampler_parity():
    items = _items()
    pairs = _pairs_excluding(items)
    first = epoch_batches(items, 8, 11, 0, pairs=pairs, pairs_per_batch=3)
    assert first == epoch_batches(items, 8, 11, 0, pairs=pairs,
                                  pairs_per_batch=3)
    assert epoch_batches(items, 8, 11, 1, pairs=pairs, pairs_per_batch=3) \
        == epoch_batches(items, 8, 11, 1, pairs=pairs, pairs_per_batch=3)
    # the DataLoader sampler yields the identical seeded batches
    got = list(PairAwareBatchSampler(items, 8, 11, epochs=1, pairs=pairs,
                                     pairs_per_batch=3))
    assert got == first
    assert len(PairAwareBatchSampler(items, 8, 11, epochs=2, pairs=pairs,
                                     pairs_per_batch=3)) == 2 * len(first)


def test_unsupported_pair_is_never_forced_into_a_batch():
    items = _items(excluded_treatment=True)
    pairs = _pairs_excluding(items, omitted_treatment=10)
    assert (10, 0) not in pairs                  # (10,0) is the excluded pair
    batches = epoch_batches(items, 8, 11, 0, pairs=pairs, pairs_per_batch=3)
    for positions in batches:
        assert not all(p in positions for p in (0, 10))
        # every complete pair inside a batch is an ELIGIBLE pool pair
        if batch_has_pair(positions, items, pairs):
            assert _n_pairs(positions, items) == 3
        else:
            # coverage batch: at most one item per pair_key -> no pair
            assert _n_pairs(positions, items) == 0
    # every item (excluded treatment included) is still covered; references
    # may recur across pair batches, so the union (not the multiset) is the
    # coverage contract
    seen = {p for b in batches for p in b}
    assert seen == set(range(len(items)))


def test_label_t0_telemetry_survives_the_reorder():
    items = _items()
    pairs = _pairs_excluding(items)
    batches = epoch_batches(items, 8, 11, 0, pairs=pairs, pairs_per_batch=3)
    asserted = 0
    for positions in batches:
        out = IdeaATrainer.label_t0_telemetry(
            {"meta": [items[p] for p in positions]})
        asserted += out["label_t0_asserted"]
        assert out["label_t0_violations"] == 0
    assert asserted > 0                          # fields are still counted


def test_default_pairs_per_batch_none_is_unchanged():
    items = _items()
    pairs = _pairs_excluding(items)
    assert epoch_batches(items, 8, 11, 0, pairs) == epoch_batches(
        items, 8, 11, 0, pairs, pairs_per_batch=None)
    # a config without the flag keeps the historical default
    assert IdeaATrainConfig().pairs_per_batch is None


def test_config_validation_and_cli_flags():
    cfg = IdeaATrainConfig(pairs_per_batch=4, batch_size=8)
    assert cfg.pairs_per_batch == 4
    with pytest.raises(IdeaAConfigError):
        IdeaATrainConfig(pairs_per_batch=5, batch_size=8)   # 2K > batch_size
    with pytest.raises(IdeaAConfigError):
        IdeaATrainConfig(pairs_per_batch=0)
    parser = build_parser()
    args = parser.parse_args(["--member", "1", "--corpus", "P", "--out", "O",
                              "--pairs-per-batch", "4",
                              "--lambda-contrast", "2.5"])
    assert args.pairs_per_batch == 4 and args.lambda_contrast == 2.5
