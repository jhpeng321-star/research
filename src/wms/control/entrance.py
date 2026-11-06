"""V10ENTZONE: entrance-zone native driving inside WM imagination and SUMO autochecks.

A CAV inside an entrance zone (the first ``entrance_zone_*_m`` metres downstream of its lane's
insertion point) must drive native (its policy action is replaced by the IDM/native twin command
and the step is excluded from the actor loss), so the learned policy cannot win by metering the
entrances (REGISTRATION.md 12:05 10-19; blocking_diag/out/blocking_table.md: every slow policy-CAV
blocker sits within 0-25 m of the insertion point, 0-10 s after its own insertion).

The zone geometry is the WM/model frame of ``wms.control.contract`` (merge zone 0..75.44 m,
upstream negative) and is identical to the SUMO-side reference
``runs/active/C4-V7FT-20261017/entrance_mask/entrance_mask_loader.py`` (same insertion points).

This module is the single source of truth for the mask so imagination and the SUMO autocheck
cannot drift; the SUMO loader re-implements the same test because it must stay import-free on the
evaluation box.
"""
from __future__ import annotations

from typing import Callable, Mapping

from torch import Tensor

# Insertion points in the WM/model frame (traffic_shepherd/c4/geometry.py HERE_OFFSETS =
# (2645.67, 2644.22, 378.66)); lane 0/1 = mainline pre-merge edge, lane 2 = ramp edge.
MAINLINE_ENTRY_S = {0: -2645.67, 1: -2644.22}
RAMP_ENTRY_S = -378.66
#: Default downstream zone length (metres) used when ``trainer.entrance_zone_native`` is on and
#: the per-entrance lengths are not overridden.
DEFAULT_ENTRANCE_ZONE_M = 25.0
_EPS = 1e-6


def entrance_zone_mask(lane: Tensor, s_m: Tensor, *, main_m: float = DEFAULT_ENTRANCE_ZONE_M,
                       ramp_m: float = DEFAULT_ENTRANCE_ZONE_M) -> Tensor:
    """Boolean mask (same shape as ``lane``): slot is on an entrance lane inside its zone.

    ``main_m`` applies to lanes 0/1, ``ramp_m`` to lane 2; junction-transit rows (lane -1) are
    never masked.  ``s_m`` is the WM/model-frame position.
    """
    lane = lane.long()
    s = s_m.to(lane.device)
    mask = (lane == 0) & (s >= MAINLINE_ENTRY_S[0] - _EPS) & (s <= MAINLINE_ENTRY_S[0] + main_m + _EPS)
    mask = mask | ((lane == 1) & (s >= MAINLINE_ENTRY_S[1] - _EPS)
                   & (s <= MAINLINE_ENTRY_S[1] + main_m + _EPS))
    mask = mask | ((lane == 2) & (s >= RAMP_ENTRY_S - _EPS)
                   & (s <= RAMP_ENTRY_S + ramp_m + _EPS))
    return mask


def entrance_zone_cavs(state: Mapping[str, Tensor], *, main_m: float = DEFAULT_ENTRANCE_ZONE_M,
                       ramp_m: float = DEFAULT_ENTRANCE_ZONE_M) -> Tensor:
    """Present CAV slots whose ``lane``/``s_m`` lie inside an entrance zone."""
    present = state["present"].bool()
    cav = present & state["is_cav"].bool() & (state["lane"] >= 0)
    if not bool(cav.any()):
        return cav
    return cav & entrance_zone_mask(state["lane"], state["s_m"], main_m=main_m, ramp_m=ramp_m)


def make_entrance_zone_fn(main_m: float = DEFAULT_ENTRANCE_ZONE_M,
                          ramp_m: float = DEFAULT_ENTRANCE_ZONE_M
                          ) -> Callable[[Mapping[str, Tensor]], Tensor]:
    """``state -> [B,K] bool`` predicate handed to the CAV controller factory as ``zone``."""

    def zone(state: Mapping[str, Tensor]) -> Tensor:
        return entrance_zone_cavs(state, main_m=main_m, ramp_m=ramp_m)

    return zone
