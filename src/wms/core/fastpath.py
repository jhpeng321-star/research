"""S-WMSPEED fast-path switches (all default OFF = bit-identical legacy behaviour).

``WMS_WMSPEED=1`` enables every EXACT speedup (same loss value and optimizer gradient up to
floating-point reassociation; no change of objective, data, RNG streams or selection).
Objective-changing options have their own flags and are never implied by ``WMS_WMSPEED``.

Individual exact switches (each also honoured on its own, ``=0`` disables under the master):
  WMS_WMSPEED_SKIP_PAIR_GRADNORM  skip the logged-only ``paired_grad_norm`` extra VJP.
  WMS_WMSPEED_DEDUPE_MOBIL        build the baseline MOBIL lane logits once per forward when the
                                  native-lane CAV mask recomputes them (first result discarded).
  WMS_WMSPEED_SKIP_FROZEN_PIVOT_TF  training passes skip the merge-pivot recorded replays when
                                  that term is gradient-free (frozen head, on-objective weight 0):
                                  identical gradient; logged ``merge_pivot`` = 0 and merge-pivot
                                  diagnostics absent (the constant TF NLL is not computed).
  WMS_WMSPEED_MOBIL_LAZY_SAFE     with ``lane_learnable`` physics, MOBIL skips the behavioural
                                  gap tests that the physical envelope always overwrites.
"""
from __future__ import annotations

import os


def _flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is not None:
        return value == '1'
    return os.environ.get('WMS_WMSPEED') == '1'


def skip_pair_grad_norm() -> bool:
    return _flag('WMS_WMSPEED_SKIP_PAIR_GRADNORM')


def dedupe_mobil() -> bool:
    return _flag('WMS_WMSPEED_DEDUPE_MOBIL')


def skip_frozen_pivot_tf() -> bool:
    return _flag('WMS_WMSPEED_SKIP_FROZEN_PIVOT_TF')


def mobil_lazy_safe() -> bool:
    return _flag('WMS_WMSPEED_MOBIL_LAZY_SAFE')
