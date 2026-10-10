"""Equivalence of the clean data layer (``wms.data``) with the legacy loader on REAL data.

The legacy loader (``output/wd01/wm_n1/wms/train.py``) is imported directly under its
repository path; the new one comes from ``src/wms``.  Both are read-only consumers of the
same packs, sidecars and window lists.  Tests skip when a path is absent; golden comparisons
additionally skip when the golden artefacts are not present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch


def _data(name: str) -> Path:
    """Registered location of ``name`` (git-ignored ``data_cache/``, configs/data_registry.toml)."""
    from wms.data.registry import resolve

    return resolve(name)


PACKS = tuple(_data(n).as_posix() for n in ("pack_v3", "pack_dspeed", "pack_damping"))
SIDE = _data("sidecars_packv3_v2")
IDENTITIES = (str(SIDE / "identities"), _data("identities_dspeed").as_posix(),
              _data("identities_damping").as_posix())
OVERLAYS = (_data("overlay_dspeed").as_posix(), _data("overlay_damping").as_posix())
OUTCOMES = str(SIDE / "outcomes")
LISTS = _data("lpwo_v3")
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "data"

EXPECTED_TRAIN = {"pack_v3": 294, "pack_dspeed": 444, "pack_damping": 173}
EXPECTED_VAL = {"pack_v3": 47, "pack_dspeed": 76, "pack_damping": 28}

SEED_BATCH_PERM, SEED_ITEM, N_BATCHES, BATCH_SIZE = 2026, 7000, 3, 2
HISTORY_S = 60.0
BATCH_TENSORS = ("states", "present", "slot_index", "loss_valid", "vehicle_ids",
                 "cav_teacher", "is_cav", "hist", "hist_mask", "driver_hist",
                 "driver_hist_mask", "controller_id")

REQUIRED = [*PACKS, *IDENTITIES, *OVERLAYS, OUTCOMES, str(LISTS)]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in REQUIRED),
    reason="real packs / sidecars / lpwo_v3 lists not present in data_cache/ "
           "(registry names pack_v3, pack_dspeed, pack_damping, sidecars_packv3_v2, "
           "identities_*, overlay_*, lpwo_v3)")


@pytest.fixture(scope="module")
def stacks():
    """``(legacy TrajectoryDataset class, collate, new class, new collate)``."""
    os.environ["WM_N1_TRAIN_OUTCOMES"] = OUTCOMES
    os.environ["WM_N1_TRAIN_IDENTITIES"] = IDENTITIES[0]
    # The new package must claim the top-level name ``wms`` BEFORE the legacy tree puts
    # output/wd01/wm_n1 (which holds a different ``wms``) on sys.path.
    from wms.data import collate as new_collate
    from wms.data import dataset as new
    from output.wd01.wm_n1.wms import train as legacy

    return legacy, new_collate, new


def _kwargs(window_list: str, *, horizon_steps: int, shift: float, history_s):
    return dict(
        identity_dirs=list(IDENTITIES), overlays=list(OVERLAYS), horizon_steps=horizon_steps,
        start_shift_prob=shift, window_list=str(LISTS / window_list),
        exclude_window_list=[str(LISTS / "excluded_runs_PI_wall_v3.json")],
        history_s=history_s, history_stride_s=1.0)


@pytest.fixture(scope="module")
def built(stacks):
    """Cache of (legacy dataset, new dataset) per (list, horizon, shift, history)."""
    legacy, _, new = stacks
    cache: dict = {}

    def get(window_list="windows_train_noPI_noWall.json", horizon_steps=240, shift=0.0,
            history_s=None):
        key = (window_list, horizon_steps, shift, history_s)
        if key not in cache:
            kwargs = _kwargs(window_list, horizon_steps=horizon_steps, shift=shift,
                             history_s=history_s)
            cache[key] = (legacy.TrajectoryDataset.from_real_packs(list(PACKS), **kwargs),
                          new.TrajectoryDataset.from_real_packs(list(PACKS), **kwargs))
        return cache[key]

    return get


def test_window_counts_per_pack(built):
    old_train, new_train = built()
    old_val, new_val = built("windows_val_noPI_noWall.json")
    assert new_train.receipt["kept_by_pack"] == EXPECTED_TRAIN
    assert new_val.receipt["kept_by_pack"] == EXPECTED_VAL
    assert old_train.receipt["kept_by_pack"] == new_train.receipt["kept_by_pack"]
    assert old_val.receipt["kept_by_pack"] == new_val.receipt["kept_by_pack"]


@pytest.mark.parametrize("window_list", ["windows_train_noPI_noWall.json",
                                         "windows_val_noPI_noWall.json"])
def test_selection_and_receipt_identical(built, window_list):
    old, new = built(window_list)
    assert len(old) == len(new)
    assert old.positions == new.positions
    assert old.receipt == new.receipt


def test_window_counts_helper_matches(stacks):
    from wms.data.corpus import PackCorpus
    from wms.data.windows import window_counts

    corpus = PackCorpus.open(PACKS[0], extra_pack_dirs=PACKS[1:],
                             extra_overlay_dirs=OVERLAYS, extra_identity_dirs=IDENTITIES[1:])
    counts = window_counts(corpus, train_list=LISTS / "windows_train_noPI_noWall.json",
                           val_list=LISTS / "windows_val_noPI_noWall.json",
                           exclude_window_list=[LISTS / "excluded_runs_PI_wall_v3.json"])
    assert counts["pack_v3"] == (294, 47)
    assert counts["pack_dspeed"] == (444, 76)
    assert counts["pack_damping"] == (173, 28)


def _decode(ds, collate, unroll_steps_perm_len):
    """3 fixed batches of 2 windows with per-item seeded start-shift draws."""
    generator = torch.Generator().manual_seed(SEED_BATCH_PERM)
    perm = [int(i) for i in torch.randperm(len(ds), generator=generator)[
        :N_BATCHES * BATCH_SIZE]]
    batches, items = [], []
    for j in range(N_BATCHES):
        records = []
        for n in range(BATCH_SIZE):
            torch.manual_seed(SEED_ITEM + j * BATCH_SIZE + n)
            records.append(ds[perm[j * BATCH_SIZE + n]])
            items.append(int(records[-1].get("start_shift_ticks") or 0))
        batches.append(collate(records))
    return perm, items, batches


def _assert_batches_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], torch.Tensor):
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
            assert torch.equal(a[key], b[key]), key
        elif key == "action_timeline":
            for sub in a[key]:
                assert np.array_equal(np.asarray(a[key][sub]), np.asarray(b[key][sub])), sub
        else:
            assert json.dumps(a[key], default=str) == json.dumps(b[key], default=str), key


@pytest.mark.parametrize("unroll_s", [10, 60])
@pytest.mark.parametrize("history_s", [None, HISTORY_S])
def test_decoded_batches_byte_identical(stacks, built, unroll_s, history_s):
    legacy, new_collate, _ = stacks
    old, new = built(horizon_steps=int(unroll_s / 0.5), shift=0.75, history_s=history_s)
    old_perm, old_shift, old_batches = _decode(old, legacy.collate, N_BATCHES)
    new_perm, new_shift, new_batches = _decode(new, new_collate.collate, N_BATCHES)
    assert old_perm == new_perm and old_shift == new_shift
    assert any(old_shift) or unroll_s == 10   # the start-shift branch is exercised
    for old_batch, new_batch in zip(old_batches, new_batches):
        _assert_batches_equal(old_batch, new_batch)
    assert ("driver_hist" in new_batches[0]) == (history_s is not None)


def test_every_window_loads_identically_without_shift(built):
    """All 911 train windows (shift off) decode to identical records (sampled stride 37)."""
    old, new = built(horizon_steps=120)
    for i in range(0, len(old), 37):
        a, b = old[i], new[i]
        assert a.keys() == b.keys()
        for key in a:
            if key == "action_timeline":
                for sub in a[key]:
                    assert np.array_equal(a[key][sub], b[key][sub]), (i, sub)
            elif isinstance(a[key], np.ndarray):
                assert a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), (i, key)
            else:
                assert json.dumps(a[key], default=str) == json.dumps(b[key], default=str)


# ------------------------------------------------------------------ golden ----
golden = pytest.mark.skipif(not (GOLDEN / "selection.json").exists(),
                            reason="golden artefacts not present")


@golden
def test_golden_selection(built):
    doc = json.loads((GOLDEN / "selection.json").read_text(encoding="utf-8"))
    assert doc["expected_pack_v3"] == {"train": 294, "val": 47}
    from wms.data.corpus import PackCorpus
    from wms.data.windows import item_key, select_windows

    corpus = PackCorpus.open(PACKS[0], extra_pack_dirs=PACKS[1:], extra_overlay_dirs=OVERLAYS,
                             extra_identity_dirs=IDENTITIES[1:])
    names = ["pack_v3", "pack_dspeed", "pack_damping"]
    for split, name in (("train", "windows_train_noPI_noWall.json"),
                        ("val", "windows_val_noPI_noWall.json")):
        selection = select_windows(corpus, window_list=LISTS / name, names=names,
                                   exclude_window_list=[LISTS / "excluded_runs_PI_wall_v3.json"])
        rows = [[names[corpus.pack_index(i)], *item_key(corpus.meta(i))]
                for i in selection.positions]
        assert json.loads(json.dumps(rows)) == doc["kept_keys"][split]


@golden
@pytest.mark.parametrize("unroll_s", [10, 60])
@pytest.mark.parametrize("hist", [True, False])
def test_golden_batches(stacks, built, unroll_s, hist):
    tag = f"batches_u{unroll_s}_hist{'on' if hist else 'off'}"
    if not (GOLDEN / f"{tag}.npz").exists():
        pytest.skip(f"{tag} not captured")
    _, new_collate, _ = stacks
    _, new = built(horizon_steps=int(unroll_s / 0.5), shift=0.75,
                   history_s=HISTORY_S if hist else None)
    perm, _, batches = _decode(new, new_collate.collate, N_BATCHES)
    meta = json.loads((GOLDEN / f"{tag}.json").read_text(encoding="utf-8"))
    assert perm == meta["indices"]
    with np.load(GOLDEN / f"{tag}.npz") as arrays:
        for j, batch in enumerate(batches):
            for name in BATCH_TENSORS:
                if name in batch:
                    assert np.array_equal(arrays[f"b{j}_{name}"], batch[name].numpy()), (j, name)
