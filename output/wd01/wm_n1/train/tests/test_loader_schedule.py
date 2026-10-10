"""Loader schedule tests: the persistent DataLoader iterator must cycle
pair-aware seeded corpus passes, and ``fit`` must refuse an unexpected early
exhaustion instead of writing an under-scheduled success receipt.

Pure synthetic (no real workers, no real data, no SUMO, no GPU).  The loader
branch is exercised exactly the way ``fit`` selects it for
``num_workers > 0`` while ``num_workers = 0`` keeps the SAME code path with
zero spawned ``DataLoader`` worker processes:

* ``_iter_batches_loader`` constructs exactly ONE ``torch.utils.data.DataLoader``
  and re-iterates it once per corpus pass; ``set_epoch`` advances the seeded
  pass once per COMPLETE pass (never per step), so pass ``e`` is
  position-identical to the in-process :func:`train.dataset.epoch_batches`
  order and to the in-process generator (seeded order parity);
* primary coverage is retained (one pass visits every item exactly once; a
  pair partner may repeat; nothing is dropped);
* a full ``fit`` through the REAL loader iterator completes the FULL
  requested schedule over MORE than one pass with the correct actual step
  count (fake bundle/source, real ``fit``/iterator);
* an empty / zero-yield source raises fail-closed (never a busy infinite
  loop), and a fit hitting an unexpected early exhaustion refuses WITHOUT a
  success receipt or checkpoint while preserving the partial train log.

Corpus-pass vs training-epoch seeding: a pass counter ``e`` seeds the pure
``epoch_batches(items, batch_size, seed, e)`` order, exactly like the
in-process path.  Both paths start at pass 0 when ``fit`` (re)builds the
iterator - identical behavior, no path divergence; the requested-schedule
resume semantics (which epoch a resumed fit continues from) are unchanged
here and remain flagged for owner review.
"""
from __future__ import annotations

import math

import pytest
import torch

import train.idea_a as ia
from train.dataset import contrast_pairs, epoch_batches
from test_idea_a_fakes import make_windows, seeded_bundle


# ------------------------------------------------------ synthetic corpus ----
def _split_window(window, row):
    """One single-arm canonical window ([1, ...] tensors) from a 2-arm row."""
    out = {}
    for key, val in window.items():
        if key == "meta":
            out["meta"] = [dict(window["meta"][row])]
        elif key in ia.BatchTime_KEYS:
            out[key] = val[:, row:row + 1]
        elif key in ia.Batch2D_KEYS:
            out[key] = val[row:row + 1]
        else:                                          # pragma: no cover
            raise AssertionError(f"unexpected fixture window key {key}")
    return out


class _PairedWindowCorpus:
    """A tiny TRAIN corpus stand-in: one single-arm canonical window/item.

    Items are the split arm rows of the ``make_windows`` fixture (complete
    sibling pairs); ``_items``/``_order`` drive the pair-aware sampler exactly
    like ``MemmapCorpus`` metadata does, and ``verified_identities`` satisfies
    the strict fit-preparation gate without any sidecar.
    """

    def __init__(self, n_pairs: int = 4, k: int = 8, seed: int = 0):
        windows = make_windows(n_pairs, k=k, seed=seed)
        self.split = "TRAIN"
        self.pack_dir = None               # corpus-style for _iter_batches
        self.windows = []
        self._items = []
        for window in windows:
            self._items.append(dict(window["meta"][0]))    # treatment arm
            self._items.append(dict(window["meta"][1]))    # native reference
            self.windows.append(_split_window(window, 0))
            self.windows.append(_split_window(window, 1))
        self._order = list(range(len(self._items)))

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, i: int) -> dict:
        return self.windows[self._order[i]]

    def verified_identities(self) -> dict:
        return {"status": "verified", "mode": "synthetic-fixture",
                "n_items": len(self), "receipt_sha256": "cd" * 32}


class _EmptyTrainCorpus:
    """A TRAIN-split corpus stand-in that yields no item at all."""

    def __init__(self):
        self.split = "TRAIN"
        self.pack_dir = None
        self._items = []
        self._order = []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, i: int) -> dict:   # pragma: no cover - never reached
        raise IndexError(i)

    def verified_identities(self) -> dict:
        return {"status": "verified", "mode": "synthetic-fixture",
                "n_items": 0, "receipt_sha256": "cd" * 32}


def _trainer(src, tmp_path, **over) -> ia.IdeaATrainer:
    values = dict(epochs=2, steps_per_epoch=5, batch_size=2,
                  out_dir=str(tmp_path / "out"), num_workers=0,
                  state_global_token=False)
    values.update(over)
    cfg = ia.IdeaATrainConfig(**values)
    return ia.IdeaATrainer(cfg, src, member=1, seed=101,
                           bundle=seeded_bundle(101))


def _loader_trainer(src) -> ia.IdeaATrainer:
    cfg = ia.IdeaATrainConfig(out_dir="unused", num_workers=0,
                              state_global_token=False, batch_size=2)
    return ia.IdeaATrainer(cfg, src, member=1, seed=101,
                           bundle=seeded_bundle(0))


# ------------------------------------------------------------- spy fixtures --
@pytest.fixture
def loader_builds(monkeypatch) -> list:
    """Count DataLoader CONSTRUCTIONS performed by the loader path."""
    builds = []
    real = torch.utils.data.DataLoader

    class _CountingDataLoader(real):
        def __init__(self, *args, **kwargs):
            builds.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", _CountingDataLoader)
    return builds


@pytest.fixture
def set_epoch_calls(monkeypatch) -> list:
    """Record every ``set_epoch`` call on the loader's sampler."""
    calls = []
    real = ia._LoaderPassSampler

    class _RecordingSampler(real):
        def set_epoch(self, epoch):
            calls.append(int(epoch))
            super().set_epoch(epoch)

    monkeypatch.setattr(ia, "_LoaderPassSampler", _RecordingSampler)
    return calls


# ------------------------------------------------------- seeded order parity --
def test_loader_passes_match_in_process_and_seeded_epoch_batches():
    """>1 pass: every pass equals the in-process order and epoch_batches."""
    src = _PairedWindowCorpus(n_pairs=4, seed=0)     # 8 items, bs=2
    tr = _loader_trainer(src)
    meta = ia._source_item_meta(src)
    pairs = contrast_pairs(meta)
    batches_per_pass = len(epoch_batches(meta, 2, 101, 0, pairs))
    assert batches_per_pass == 4                     # ceil(8 / 2)
    loader = tr._iter_batches_loader(src)
    in_proc = tr._iter_batches(src)
    for epoch in range(3):                           # THREE complete passes
        positions = epoch_batches(meta, 2, 101, epoch, pairs)
        for want in positions:
            got_loader = next(loader)
            got_in_proc = next(in_proc)
            assert got_loader["meta"] == [meta[p] for p in want]
            assert got_in_proc["meta"] == got_loader["meta"]
            for key, val in got_loader.items():
                if key == "meta":
                    continue
                assert torch.equal(val, got_in_proc[key])
    # primary coverage: one pass visits every item exactly once (a pair
    # partner may repeat inside a batch, but nothing is ever dropped)
    one_pass = [p for batch in epoch_batches(meta, 2, 101, 0, pairs)
                for p in batch]
    assert set(one_pass) == set(range(len(meta)))


def test_loader_sampler_set_epoch_selects_the_seeded_pass():
    """``set_epoch`` advances the finite sampler to the requested pass."""
    src = _PairedWindowCorpus(n_pairs=4, seed=1)
    meta = ia._source_item_meta(src)
    pairs = contrast_pairs(meta)
    sampler = ia._LoaderPassSampler(meta, 2, 101, epoch=0, epochs=1,
                                    pairs=pairs)
    assert len(sampler) == 4                         # finite: no worker hang
    assert list(sampler) == epoch_batches(meta, 2, 101, 0, pairs)
    sampler.set_epoch(2)
    assert list(sampler) == epoch_batches(meta, 2, 101, 2, pairs)
    orders = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        orders.append([list(batch) for batch in sampler])
    assert orders[0] != orders[1] and orders[1] != orders[2]


def test_loader_builds_one_dataloader_and_advances_once_per_complete_pass(
        loader_builds, set_epoch_calls):
    """ONE persistent DataLoader construction; set_epoch only at pass ends."""
    src = _PairedWindowCorpus(n_pairs=4, seed=2)
    tr = _loader_trainer(src)
    batches = tr._iter_batches_loader(src)
    for _ in range(4):                               # complete pass 0
        next(batches)
    assert set_epoch_calls == [0]
    assert loader_builds == [1]
    for _ in range(4):                               # complete pass 1
        next(batches)
    assert set_epoch_calls == [0, 1]                 # once per COMPLETE pass
    assert loader_builds == [1]                      # still the SAME loader
    for _ in range(4):                               # pass 2 starts (+1 batch)
        next(batches)
    assert set_epoch_calls == [0, 1, 2]
    assert loader_builds == [1]                      # never rebuilt per step


# --------------------------------------------- full fit through the loader --
def test_full_fit_through_loader_iterator_completes_the_schedule(
        tmp_path, monkeypatch, loader_builds, set_epoch_calls):
    """Real ``fit`` on the real loader iterator: >1 pass, exact step count."""
    src = _PairedWindowCorpus(n_pairs=4, seed=3)
    tr = _trainer(src, tmp_path, epochs=2, steps_per_epoch=5, batch_size=2)
    # select the loader branch exactly as fit does for num_workers > 0 while
    # num_workers stays 0: the REAL path with zero spawned worker processes
    monkeypatch.setattr(ia.IdeaATrainer, "_iter_batches",
                        ia.IdeaATrainer._iter_batches_loader)
    receipt = tr.fit()
    assert receipt["steps"] == 10                    # full requested schedule
    assert receipt["epochs"] == 2
    assert len(receipt["losses"]) == 10
    assert all(math.isfinite(v) for v in receipt["losses"])
    assert [entry["epoch"] for entry in receipt["history"]] == [1, 2]
    assert all(math.isfinite(entry["mean_loss"])
               for entry in receipt["history"])
    assert set_epoch_calls == [0, 1, 2]              # 10 steps over 3 passes
    assert loader_builds == [1]                      # one persistent DataLoader
    out = tmp_path / "out"
    assert (out / "member_1.pt").is_file()
    assert (out / "TRAIN_RECEIPT.json").is_file()


# ------------------------------------------------ zero-yield / exhaustion ----
def test_zero_yield_loader_pass_raises_fail_closed():
    """An empty source raises on the first pass instead of spinning forever."""
    src = _EmptyTrainCorpus()
    tr = _loader_trainer(src)
    batches = tr._iter_batches_loader(src)
    with pytest.raises(ia.IdeaAConfigError) as ei:
        next(batches)
    assert "0 batches" in str(ei.value)
    assert "pass 0" in str(ei.value)


def test_fit_refuses_unexpected_early_exhaustion(tmp_path):
    """A batch stream ending before the schedule refuses; nothing is cleaned."""
    src = _EmptyTrainCorpus()
    tr = _trainer(src, tmp_path, epochs=2, steps_per_epoch=3,
                  allow_incomplete_losses=True)    # TEST-ONLY flag documented
    with pytest.raises(ia.IdeaAConfigError) as ei:
        tr.fit()
    assert "ended early" in str(ei.value)
    out = tmp_path / "out"
    assert not (out / "TRAIN_RECEIPT.json").exists()
    assert not (out / "member_1.pt").exists()
    assert not (out / "OUTCOME_NORM.json").exists()
    assert (out / "train_log.jsonl").is_file()       # partial log preserved


def test_fit_refuses_zero_yield_loader_source(tmp_path, monkeypatch):
    """The loader path's zero-yield raise propagates through fit (no swallow)."""
    src = _EmptyTrainCorpus()
    tr = _trainer(src, tmp_path, epochs=1, steps_per_epoch=2,
                  allow_incomplete_losses=True)
    monkeypatch.setattr(ia.IdeaATrainer, "_iter_batches",
                        ia.IdeaATrainer._iter_batches_loader)
    with pytest.raises(ia.IdeaAConfigError) as ei:
        tr.fit()
    assert "0 batches" in str(ei.value)
    out = tmp_path / "out"
    assert not (out / "TRAIN_RECEIPT.json").exists()
    assert not (out / "member_1.pt").exists()
    assert not (out / "OUTCOME_NORM.json").exists()
    assert (out / "train_log.jsonl").is_file()       # partial log preserved
