"""Eligible-pool ``ensure_pair`` regression (DRAFT — tests pending guard).

Coordinator source review (2026-10-02): ``train.dataset.ensure_pair`` used to
return a batch unchanged whenever it completed ANY semantic sibling pair,
even when the supplied ``pairs`` argument was a FILTERED eligible pool and
the batch held only an unsupported (out-of-pool) pair — the eligible pool
was bypassed.  These synthetic-only tests (no pack, no parquet, no SUMO, no
real data) pin the fixed contract:

* a batch whose only complete pair is OUTSIDE the eligible pool is
  completed with a pair FROM the pool (append-only missing members; the
  cursor advances exactly once);
* a batch already completing an IN-POOL pair is returned unchanged and the
  cursor is untouched;
* every primary item still appears exactly once per epoch in
  :func:`train.dataset.epoch_batches`, and the in-process path stays
  byte-identical to :class:`train.dataset.PairAwareBatchSampler` for a
  filtered pool (parity);
* partner choice stays deterministic and cycles the eligible pool;
* an explicitly EMPTY pool never invents a pair, and ``pairs=None`` keeps
  the upstream contract (the caller already computed the default pool).

DRAFT / CODE-ONLY: per the coordinator brief these tests have NOT been run
(awaiting the reviewed guard + exact test list).
"""
from __future__ import annotations

import torch

from train.dataset import (PairAwareBatchSampler, batch_has_pair,
                           contrast_pairs, ensure_pair, epoch_batches)


def _items():
    """Hand-built item metadata: two complete sibling groups.

    Group ``a`` has a true native reference plus two treatments; group ``b``
    is an independent world with its own native reference.  The full pool is
    ``[(1, 0), (2, 0), (4, 3)]`` (sorted pair_key, then position).
    """
    def it(world, arm, policy):
        return dict(world=world, arm=arm, policy=policy, chunk="chunk4",
                    split="TRAIN", t0_s=100.0,
                    pair_key=f"{world}@t0=100")

    return [
        it("a", "native", "native"),          # 0: reference of group a
        it("a", "switching1", "switching"),   # 1: treatment
        it("a", "base", "damping"),           # 2: treatment
        it("b", "native", "native"),          # 3: reference of group b
        it("b", "pi", "pi"),                  # 4: treatment
    ]


def test_unsupported_semantic_pair_no_longer_bypasses_eligible_pool():
    items = _items()
    eligible = [(4, 3)]                       # only group b is supported
    # positions 0+1 complete a semantic pair of group a, OUTSIDE the pool
    assert batch_has_pair([0, 1], items)          # semantic check: complete
    assert not batch_has_pair([0, 1], items, eligible)
    out, cursor = ensure_pair([0, 1], items, eligible, 0)
    assert (out, cursor) == ([0, 1, 4, 3], 1)     # pool pair appended
    assert batch_has_pair(out, items, eligible)


def test_out_of_pool_pair_gets_only_the_missing_member_appended():
    items = _items()
    eligible = [(1, 0)]                       # group-a pool: switching only
    # positions 2+0 complete a semantic pair, but (2, 0) is NOT the eligible
    # pair: only the missing treatment of the chosen pool pair is appended
    out, cursor = ensure_pair([2, 0], items, eligible, 3)
    assert (out, cursor) == ([2, 0, 1], 4)
    assert batch_has_pair(out, items, eligible)


def test_in_pool_complete_pair_returned_unchanged():
    items = _items()
    eligible = [(4, 3)]
    assert batch_has_pair([3, 4], items, eligible)
    assert ensure_pair([3, 4], items, eligible, 7) == ([3, 4], 7)


def test_empty_pool_never_invents_and_none_defers_to_caller():
    items = _items()
    # an explicitly EMPTY pool invents no eligible pair
    assert ensure_pair([1], items, [], 5) == ([1], 5)
    # pairs=None keeps the upstream contract: callers already computed the
    # default pool; ensure_pair itself invents nothing
    assert ensure_pair([1], items, None, 5) == ([1], 5)
    # and epoch_batches(pairs=None) resolves the FULL default pool upstream
    assert epoch_batches(items, 2, 11, 0, pairs=None) == epoch_batches(
        items, 2, 11, 0, pairs=contrast_pairs(items))


def test_filtered_pool_partner_choice_is_deterministic_and_cycling():
    items = _items()
    eligible = [(1, 0), (2, 0)]               # group-a-only pool
    first, cursor = ensure_pair([4], items, eligible, 0)
    assert (first, cursor) == ([4, 1, 0], 1)  # pairs[0] chosen at cursor 0
    second, cursor = ensure_pair([4], items, eligible, cursor)
    assert (second, cursor) == ([4, 2, 0], 2)  # cursor cycled to pairs[1]
    assert ensure_pair([4], items, eligible, 0) == ([4, 1, 0], 1)


def test_epoch_batches_filtered_pool_keeps_primary_coverage_and_parity():
    items = _items()
    eligible = [(4, 3)]
    bs, seed, epoch, n = 2, 11, 0, len(items)
    batches = epoch_batches(items, bs, seed, epoch, eligible)
    # the primary order is exactly the seeded permutation, untouched
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int((seed * 1000003 + epoch) % (2 ** 63 - 1)))
    order = torch.randperm(n, generator=generator).tolist()
    primaries = [order[s:s + bs] for s in range(0, n, bs)]
    assert [b[:len(p)] for b, p in zip(batches, primaries)] == primaries
    # every batch completes an ELIGIBLE pair; partners are append-only
    for batch in batches:
        assert batch_has_pair(batch, items, eligible)
        assert len(batch) <= bs + 2
    # every primary item still appears exactly once per epoch
    seen = [p for batch, pri in zip(batches, primaries)
            for p in batch[:len(pri)]]
    assert sorted(seen) == list(range(n))
    # DataLoader parity: the sampler yields byte-identical filtered batches
    got = list(PairAwareBatchSampler(items, bs, seed, epochs=1,
                                     pairs=eligible))
    assert got == batches
