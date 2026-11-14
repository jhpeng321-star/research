"""Split-leak guards for the WM-N1 combination layer (design section 5).

Two independent contracts:

  * the out-of-fold tree fit sees TRAIN rows only -- a world-grouped fold
    may hold a world out of the fit, but it must never hold out a split;
  * the blend weights and the conformal recalibration are CAL-only -- the
    truth labels that choose the pool weight and its shifts come from CAL,
    never from TRAIN and never from an evaluation split.

HOLDOUT and the freshly generated chunk-5 TEST are SEALED: the whole
combination layer must never open their files.  These guards are the last
line of defence if a caller ever mixes a sealed split into a fit table.
"""
from __future__ import annotations

import numpy as np

from evalh import cal_split

TRAIN = "TRAIN"
CAL = "CALIBRATION"
TEST = "TEST"
HOLDOUT = "HOLDOUT"
SEALED = (TEST, HOLDOUT)


class SplitLeakError(RuntimeError):
    """A sealed or otherwise non-permitted split reached a fit."""


def _splits(splits) -> list[str]:
    return [str(s) for s in np.asarray(splits, dtype=object).ravel().tolist()]


def guard_train_only(splits, context: str = "") -> bool:
    """Raise unless every row is TRAIN (the OOF-tree fit contract)."""
    bad = sorted({s for s in _splits(splits) if s != TRAIN})
    if bad:
        raise SplitLeakError(
            f"{context}: fit received non-TRAIN split(s) {bad}; the "
            f"out-of-fold tree fit may see TRAIN rows only")
    return True


def guard_cal_only(splits, context: str = "", *, worlds=None, half: str = None,
                  cal_split_path=None) -> bool:
    """Raise unless every row is CALIBRATION (and, when asked, the right half).

    TEST/HOLDOUT rows are called out explicitly: they are sealed and must
    never reach a combination fit at all.  When ``half`` is given (``cal_cal``
    for blend weights/calibration/variant choice, ``cal_tune`` for the
    improvement-loop selection), every ``world`` must belong to that half of
    ``CAL_SPLIT_A4.json`` (amendment A4.2) or :class:`CalSplitError` is raised.
    """
    vals = _splits(splits)
    sealed = sorted({s for s in vals if s in SEALED})
    if sealed:
        raise SplitLeakError(
            f"{context}: fit received SEALED split(s) {sealed}; TEST/HOLDOUT "
            f"rows must never reach a combination fit")
    bad = sorted({s for s in vals if s != CAL})
    if bad:
        raise SplitLeakError(
            f"{context}: blend/calibration fit received non-CALIBRATION "
            f"split(s) {bad}; pool weights and conformal shifts are "
            f"CAL-only")
    if half is not None:
        if worlds is None:
            raise SplitLeakError(
                f"{context}: half={half!r} needs the row 'worlds' to enforce "
                "the A4.2 CAL split")
        cal_split.require_worlds(worlds, half, path=cal_split_path,
                                 context=context or "combine")
    return True


def selection_worlds(supplied=None, *, cal_split_path=None) -> list:
    """CAL-tune worlds the improvement-loop selection may use."""
    return cal_split.resolve_worlds(
        cal_split.CAL_TUNE, supplied, path=cal_split_path,
        context="combine selection")


def calibration_worlds(supplied=None, *, cal_split_path=None) -> list:
    """CAL-cal worlds the blend/calibration/variant choice may use."""
    return cal_split.resolve_worlds(
        cal_split.CAL_CAL, supplied, path=cal_split_path,
        context="combine calibration/variant")
