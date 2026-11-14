"""J-PULSE hook (registration s.13): the future TRAIN data and the
registered short-action diagnostic -- a HOOK, not a run.

Sequencing (s.13.3): first fit (J4 + aux, THIS registration) -> PULSE
returns -> DISCLOSED retrain including PULSE (same model family, tuning
grid, selection rule and splits; a new disclosed version, never a
silent refit, never used for selection on the chunk-5 TEST) -> report.
If PULSE never runs, the first-fit registration stands on its own.

The registered PULSE out-of-fold diagnostic (s.13.2) runs for each
v0.4b model RETRAINED with PULSE: world-grouped 5-fold CV over the 44
PULSE worlds (the evaluated fold's worlds excluded from its fit), sign
accuracy and dY CRPS against the zero-contrast and TRAIN-cell-mean
baselines at tau in {120, 300, 570}, stratified by pulse length
(60/120/240 s) and controller (damping/PI).  DIAGNOSTIC EVIDENCE ONLY:
it never changes the s.6 verdict of the first fit and never feeds
selection.

This module is implemented against the generic v0.4b model interface so
it is unit-testable on synthetic data today; the PULSE-specific
anchored parts (the campaign launcher, gated on the owner's
AUTHORIZATION.json) stay out of scope here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from . import reg
from .acceptance_v04b import (cell_mean_of, crps_cell, crps_model,
                              crps_zero, train_references_v04b)

PULSE_TAUS = (120.0, 300.0, 570.0)
PULSE_LENGTHS_S = (60.0, 120.0, 240.0)
PULSE_CONTROLLERS = ("damping", "pi")
PULSE_ARMS = tuple(f"__pulse_{c}{int(s)}"
                   for c in PULSE_CONTROLLERS for s in PULSE_LENGTHS_S)
N_FOLDS = 5                        # world-grouped 5-fold over 44 worlds
SEED = 20261001


def pulse_retrain_config(first_fit_meta: dict,
                         pulse_rows: pd.DataFrame | None = None) -> dict:
    """The disclosed-retrain plan for the report/ledger (same grid,
    selection rule and splits; PULSE pairs enter as TRAINING rows with
    the s.12 planned-schedule descriptor so 'damp/PI 60/120/240 s then
    native' are distinct inputs).  Never executed implicitly: the caller
    (post-PULSE, owner-visible) runs it as a NEW disclosed version."""
    return dict(
        schema="wd01_partB_v04b_pulse_retrain/1",
        base_registration_lf_sha256=reg.reg_sha(),
        disclosed_new_version=True,
        silent_refit=False,
        adds_training_rows="PULSE (__pulse_{damping,pi}{60,120,240} vs the "
                           "same __native sibling, restored from the same "
                           "__base t0 saved state)",
        action_descriptor="planned schedule (s.12): 19 x 30 s segments + "
                          "wall hold/speed/cap; pulse length is a "
                          "first-segments difference",
        model_family="DirectDiffGBM (unchanged)",
        tuning_grid=[list(c) for c in reg.GRID],
        selection_rule="mean pinball over the 7 levels, delay head, "
                       "wall-native rows, world-grouped CV (unchanged)",
        splits="J4 TRAIN/CAL by name (unchanged); PULSE worlds are "
               "TRAIN-only rows",
        chunk5_usage="never a selection input",
        first_fit_meta=first_fit_meta,
        pulse_rows_present=(None if pulse_rows is None
                            else int(len(pulse_rows))),
    )


def _pulse_arm_label(arm_source: str) -> tuple[str, float]:
    """'__pulse_damping120' -> ('damping', 120.0) (loud on drift)."""
    s = str(arm_source)
    if not s.startswith("__pulse_"):
        raise ValueError(f"not a PULSE arm: {arm_source!r}")
    rest = s[len("__pulse_"):]
    for c in PULSE_CONTROLLERS:
        if rest.startswith(c):
            return c, float(rest[len(c):])
    raise ValueError(f"PULSE arm {arm_source!r} carries no known "
                     f"controller ({PULSE_CONTROLLERS})")


def pulse_oof_diagnostic(pairs: pd.DataFrame, pred_fn_factory,
                         ref: dict | None = None,
                         taus=PULSE_TAUS, n_folds=N_FOLDS,
                         seed=SEED) -> list[dict]:
    """The registered OOF diagnostic over PULSE pairs.

    pairs: one row per (world, pulse arm, tau) with columns world, key,
    contrast, arm_source, tau, demand, penetration, dy_delay_{tau};
    arm_source encodes '__pulse_{damping,pi}{60,120,240}'.
    pred_fn_factory(train_rows) -> callable(rows)->{q: [n]}: the caller
    re-fits per fold EXCLUDING the evaluated fold's worlds (world-
    grouped 5-fold over the PULSE worlds).
    Returns flat diagnostic records stratified by pulse length x
    controller x tau (reported only; NOT a selection input).
    """
    worlds = np.array(sorted(pairs["world"].astype(str).unique()))
    if len(worlds) < n_folds:
        raise ValueError(f"PULSE diagnostic needs >= {n_folds} worlds, "
                         f"got {len(worlds)}")
    from sklearn.model_selection import KFold
    folds = KFold(n_splits=n_folds, shuffle=True,
                  random_state=seed).split(worlds)
    out = []
    for k, (tr_w, te_w) in enumerate(folds):
        tr_set = set(worlds[tr_w].tolist())
        te_set = set(worlds[te_w].tolist())
        tr_rows = pairs[pairs["world"].astype(str).isin(tr_set)]
        te_rows = pairs[pairs["world"].astype(str).isin(te_set)]
        fold_ref = ref if ref is not None else \
            train_references_v04b(tr_rows)
        pred = pred_fn_factory(tr_rows)
        for tau in taus:
            col = f"dy_{reg.GATING_HEAD}_{int(tau)}"
            g = te_rows[np.isclose(te_rows["tau"], tau)].dropna(
                subset=[col])
            for arm, gg in g.groupby("arm_source"):
                ctrl, length = _pulse_arm_label(arm)
                gg = gg.reset_index(drop=True)
                dY = gg[col].to_numpy(float)
                preds = pred(gg)
                rr = fold_ref.get((str(gg["contrast"].iloc[0]),
                                   float(tau)))
                thr = rr["median_abs"] if rr else float("nan")
                elig = np.abs(dY) > thr if rr else \
                    np.zeros(len(dY), bool)
                acc = float((np.sign(preds[0.5]) == np.sign(dY))[elig]
                            .mean()) if elig.any() else float("nan")
                out.append(dict(
                    fold=k, pulse_arm=str(arm), controller=ctrl,
                    length_s=float(length), tau=float(tau),
                    n=int(len(gg)), n_eligible=int(elig.sum()),
                    sign_acc=acc,
                    crps_model=float(crps_model(preds, dY).mean()),
                    crps_zero=float(crps_zero(dY).mean()),
                    crps_cell_mean=float(crps_cell(
                        dY, cell_mean_of(gg, fold_ref, tau)).mean()),
                    diagnostic_only=True))
    return out
