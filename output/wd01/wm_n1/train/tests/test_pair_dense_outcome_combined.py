"""Combined T2+T6 contract: pair-dense K batches always carry outcome support.

Integration of the pair-dense sampler (T2, ``pairs_per_batch=K``) with the
outcome-required contrast pool (T6, ``contrast_require_outcome_support``): with
``K=4`` and a pool restricted to pairs sharing a realised outcome tau, EVERY
batch drawn from that pool must hold at least one outcome-supported pair (in
fact exactly K), and a coverage batch - an item the pool filtered out - must
never complete an unsupported sibling pair.  CPU-only synthetic metadata: no
pack, no simulation, no training.
"""
from __future__ import annotations

import torch

from model import constants as C
from test_contrastfix_recovery import _mask, _PairSource, _spec
from train import contrast_support as CS
from train.dataset import (batch_has_pair, contrast_batch_indices,
                           epoch_batches)


def _zero(k: int = 6) -> torch.Tensor:
    return torch.zeros(k, dtype=torch.bool)


def _outcome_group(key: str, n_treatments: int) -> list:
    """One native reference + ``n_treatments`` arms sharing an outcome tau.

    The treatments' HDV slots are DISJOINT from the reference's, so the pair's
    only real support is the shared outcome tau: the pool that keeps it is
    exactly the outcome-required pool.
    """
    taus = C.N_TAU
    specs = [_spec("w", "native", "native", key,
                   _mask([0, 1, 2], taus), _mask([2, 3], 6), _zero())]
    for i in range(n_treatments):
        specs.append(_spec("w", f"pi{i}", "pi", key,
                           _mask([0, 1, 2], taus), _mask([0, 1], 6), _zero()))
    return specs


def _hdv_only_group(key: str) -> list:
    """A native reference + treatment sharing an HDV slot but NOT an outcome tau.

    Eligible under the historical outcome-OR-HDV rule, excluded by the
    outcome-required pool.
    """
    taus = C.N_TAU
    return [
        _spec("w", "native", "native", key,
              _mask([1], taus), _mask([0], 6), _zero()),
        _spec("w", "damp", "damping", key,
              _mask([0], taus), _mask([0], 6), _zero()),
    ]


def _pairs_in_batch(positions, items):
    """The (treatment_global, reference_global) pairs completed in a batch."""
    local = contrast_batch_indices([items[p] for p in positions])
    return [(positions[int(x)], positions[int(y)])
            for x, y in zip(local["a"], local["b"])]


def test_k4_outcome_required_every_batch_has_an_outcome_pair():
    # Phase 1: only outcome-supported groups, so every item is in the pool and
    # there is no coverage batch - EVERY batch must carry exactly K outcome
    # pairs.
    src = _PairSource(_outcome_group("a@t0=0", 8))
    items = src._items
    pool = CS.build_eligible_pair_pool(src, items, require_outcome=True)
    assert len(pool) == 8 and pool                       # such pairs exist
    batches = epoch_batches(items, batch_size=8, seed=7, epoch=0,
                            pairs=pool, pairs_per_batch=4)
    assert len(batches) == 2                             # 8 pairs / K=4
    for positions in batches:
        pairs = _pairs_in_batch(positions, items)
        assert len(pairs) == 4                           # >=1, exactly K
        assert all(p in set(pool) for p in pairs)

    # Phase 2: add an HDV-only sibling group.  The outcome-required pool drops
    # it, so its items become coverage - and no batch may then complete that
    # unsupported pair (the T2 "coverage holds one item per pair_key" contract).
    src2 = _PairSource(_outcome_group("a@t0=0", 8)
                       + _hdv_only_group("b@t0=0"))
    items2 = src2._items
    default = CS.build_eligible_pair_pool(src2, items2)
    required = CS.build_eligible_pair_pool(src2, items2, require_outcome=True)
    assert (10, 9) in default and (10, 9) not in required
    assert required == pool
    pool_set = set(required)
    batches2 = epoch_batches(items2, batch_size=8, seed=7, epoch=0,
                             pairs=required, pairs_per_batch=4)
    pair_batches = 0
    for positions in batches2:
        for pair in _pairs_in_batch(positions, items2):
            assert pair in pool_set, positions        # never an unsupported pair
        if _pairs_in_batch(positions, items2):
            pair_batches += 1
        # the semantic (pool-free) check agrees: a coverage batch completes no
        # sibling pair at all, so no batch is outcome-unsupported
        assert (not batch_has_pair(positions, items2, None)
                or batch_has_pair(positions, items2, required))
    assert pair_batches == 2
    # every item (the filtered HDV-only pair included) is still visited
    assert {p for b in batches2 for p in b} == set(range(len(items2)))


def test_merge_coverage_keeps_k_pairs_in_every_batch_and_visits_all():
    """A8 batch fix: coverage items fold INTO the pair batches.

    With ``merge_coverage`` every batch still carries exactly K pool pairs (a
    pair-less trailing batch cannot supply response_contrast), no batch ever
    completes the filtered unsupported pair, and every item is visited
    exactly once per epoch.
    """
    src = _PairSource(_outcome_group("a@t0=0", 8) + _hdv_only_group("b@t0=0"))
    items = src._items
    required = CS.build_eligible_pair_pool(src, items, require_outcome=True)
    pool_set = set(required)
    batches = epoch_batches(items, batch_size=8, seed=7, epoch=0,
                            pairs=required, pairs_per_batch=4,
                            merge_coverage=True)
    assert len(batches) == 2
    for positions in batches:
        pairs = _pairs_in_batch(positions, items)
        assert len(pairs) == 4
        assert all(p in pool_set for p in pairs)
        assert (not batch_has_pair(positions, items, None)
                or batch_has_pair(positions, items, required))
    flat = [p for b in batches for p in b]
    assert set(flat) == set(range(len(items)))
    # the shared native reference recurs across a group's pairs, but no
    # coverage item is duplicated
    cov = {9, 10}
    assert sum(1 for p in flat if p in cov) == 2
