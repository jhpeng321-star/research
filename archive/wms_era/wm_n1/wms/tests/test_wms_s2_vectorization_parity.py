"""WM-S2 CPU fast-path parity: vectorised driver-stat / dense-truth vs reference.

WM-S2-only paths were checked for the same anti-patterns the run-3 speed fix
removed from the shared rollout (per-vehicle Python loops and per-step host/device
syncs).  Two were still present and are vectorised here; these tests pin them to
identical numerics:

* ``WmSTrainer._driver_param_stats`` previously issued a leading
  ``bool(valid.any())`` sync plus 3 device syncs per parameter (28 per optimizer
  step in ``driver_params`` mode); the vectorised form does one boolean gather and
  a single host transfer.
* ``dense_truth.build_run`` previously scattered the per-(tick, vehicle) slot map
  with a Python loop; ``_scatter_records`` does one fancy-index scatter.

The driver encoder, the ``driver_params`` rollout and ``val_split`` were checked
and contain no per-vehicle Python loop and no device sync inside the rollout.

Synthetic CPU-only, no real packs, no SUMO, no GPU.
"""
from __future__ import annotations

import numpy as np
import torch

from output.wd01.wm_n1.wms import dense_truth as DT
from output.wd01.wm_n1.wms.driver_encoder import PARAMETER_NAMES
from output.wd01.wm_n1.wms.train import WmSTrainer


def _reference_stats(values, valid):
    """The original per-parameter loop (kept verbatim for parity)."""
    if values is None or valid is None:
        return None
    valid = valid.to(values.device).bool()
    if not bool(valid.any()):
        return None
    stats = {}
    for i, name in enumerate(PARAMETER_NAMES):
        column = values[..., i][valid]
        if column.numel() == 0:
            continue
        stats[name] = {"min": float(column.min()), "median": float(column.median()),
                       "max": float(column.max())}
    return stats


def _assert_stats_equal(got, ref):
    assert (got is None) == (ref is None)
    if ref is None:
        return
    assert set(got) == set(ref)
    for name in ref:
        for key in ("min", "median", "max"):
            assert got[name][key] == ref[name][key], (name, key,
                                                      got[name][key], ref[name][key])


def test_driver_param_stats_matches_reference_loop_dtypes_and_shapes():
    g = torch.Generator().manual_seed(11)
    for dtype in (torch.float32, torch.float64):
        for seed in range(40):
            g.manual_seed(seed)
            b = int(torch.randint(1, 5, (1,), generator=g).item())
            k = int(torch.randint(1, 12, (1,), generator=g).item())
            p = len(PARAMETER_NAMES)
            values = torch.rand(b, k, p, generator=g, dtype=dtype) * 60.0
            # both the [B] (trainer) and [B,K] valid-mask conventions
            mask_b = torch.rand(b, generator=g) > 0.3
            mask_bk = torch.rand(b, k, generator=g) > 0.3
            for valid in (mask_b, mask_bk):
                got = WmSTrainer._driver_param_stats(values, valid)
                ref = _reference_stats(values, valid)
                _assert_stats_equal(got, ref)


def test_driver_param_stats_empty_and_none():
    g = torch.Generator().manual_seed(3)
    values = torch.rand(3, 4, len(PARAMETER_NAMES), generator=g)
    assert WmSTrainer._driver_param_stats(None, None) is None
    assert WmSTrainer._driver_param_stats(values, None) is None
    assert WmSTrainer._driver_param_stats(
        values, torch.zeros(3, dtype=torch.bool)) is None
    # no valid element in a [B,K] mask -> None (matches the reference)
    _assert_stats_equal(
        WmSTrainer._driver_param_stats(values, torch.zeros(3, 4, dtype=torch.bool)),
        _reference_stats(values, torch.zeros(3, 4, dtype=torch.bool)))


def test_dense_truth_scatter_matches_reference_loop():
    rng = np.random.RandomState(0)
    n_ticks, k_max = 40, 16
    # unique (tick, slot) pairs, as produced by assign_slots
    pairs = {(int(rng.randint(n_ticks)), int(rng.randint(k_max)))
             for _ in range(300)}
    records = [(k, slot, int(rng.randint(1000))) for (k, slot) in pairs]
    target = np.full((n_ticks, k_max), -1, dtype=np.int32)
    ref = target.copy()
    for k, slot, vi in records:
        ref[k, slot] = vi
    DT._scatter_records(target, records)
    np.testing.assert_array_equal(target, ref)


def test_dense_truth_scatter_empty_is_noop():
    target = np.full((3, 4), -1, dtype=np.int32)
    before = target.copy()
    DT._scatter_records(target, [])
    np.testing.assert_array_equal(target, before)
