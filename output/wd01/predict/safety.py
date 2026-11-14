"""Consumer safety rules — the contract's reference helpers for the planner.

`safety_veto(pred) -> bool[M, K]` (v1 §9): True marks a cell that FAILS
safety. Evaluated BEFORE `min_ttc_s` is read. Per §9 a cell is vetoed when

- `cell_valid` is False (plan) or `native_cell_valid` is False (native)
  [contract 1.1 outputs; absent on 1.0 inputs, where the condition is
  vacuously False because v1 has no invalid-cell concept — the NaN,
  collision and abort vetoes below still apply];
- any delay float is NaN: `plan_delay_s`, `native_delay_s`,
  `delay_vs_native_s`, `delay_part_a_s`, `delay_part_b_s`
  (NaN never signals a legitimate prediction, §8.7);
- `collision` or `native_collision` is True;
- any wall in the call has `safety_abort == True` for that (m, k) cell
  (planner policy: an aborted wall fails the plan).

`safety_veto_v2(pred, tol_future_collision=0.0, p_max=0.01,
rel_safety_tol_s=0.0, rel_safety_quantile=0.90, native_collision_vetoes=False,
window_must_not_increase=True) -> bool[M, K]` (v2 §5, `wd01_predict/2` outputs
only) — the §5 rules 1-5 over
ALL M×K cells, aligned EXACTLY to the planner's (PL-01, 2026-09-27) reading:

1. Fail closed: veto the cell when `cell_valid`, `native_cell_valid`,
   `future_cell_valid` or `native_future_cell_valid` is False; any delay
   float is NaN; the PLAN window collides; or a wall `safety_abort` holds
   in the cell. Owner decision 2026-09-27 (§7): the NATIVE window collision
   does NOT veto (it is logged) — the planner can re-enable the veto with
   `native_collision_vetoes=True`.
2. Absolute window gate: the plan cell must have `collision` False and
   `min_ttc_s >= 1.5` wherever `has_closing_pair` (a plan NaN TTC with a
   pair vetoes — fail closed). The NATIVE min TTC is NOT floor-checked:
   `native_min_ttc_s < 1.5` does NOT veto; a native NaN TTC with
   `native_has_closing_pair` DOES veto.
3. Future collision (owner): veto the whole call when `p_future_collision
   > p_max` in ANY cell; veto the paired cell when `p_future_collision >
   native_p_future_collision + tol_future_collision`; a NaN probability
   (either side) vetoes.
4. Relative safety, per metric m in {`ttc_lt3_exposure_s`,
   `hard_brake_s`}, per cell: the SUMMED delta
   `(plan_m - native_m) + (plan_future_m - native_future_m)`; the call is
   ok iff `mean(delta) <= rel_safety_tol_s` AND `mean(delta <=
   rel_safety_tol_s) >= rel_safety_quantile` (defaults 0.0 / 0.90). Any
   non-finite delta anywhere fails the call. A rule-4 breach vetoes the
   whole call. Owner decision 2026-09-27 (§7, default True): the
   WINDOW-only delta `(plan_m - native_m)` must ADDITIONALLY pass the same
   mean + quantile gates (non-finite window delta -> veto all); pass
   `window_must_not_increase=False` for the pre-decision summed-only run.
5. G-1: if `meta.future_min_ttc_validated`, veto plan cells with
   `future_has_closing_pair` and `future_min_ttc_q05_s < 1.5` (NaN with a
   pair vetoes). Independently — validated or not — a native
   `future_min_ttc_q05_s` NaN with a native future closing pair vetoes.

`live_mode_allowed(meta) -> bool` — the §5 refusal rule: False unless
`future_min_ttc_validated` is True AND `future_min_ttc_validation_id` is set.

A v2 output missing any REQUIRED v2 field is rejected (ValueError), never
accepted through a default: fail closed. These are consumer rules, not
validators and not models.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .contract import CONTRACT_VERSION_V2, Prediction
from .validate import _V2_BOOL_ARRAYS, _V2_FLOAT_ARRAYS

_DELAY_FIELDS: Tuple[str, ...] = (
    "plan_delay_s",
    "native_delay_s",
    "delay_vs_native_s",
    "delay_part_a_s",
    "delay_part_b_s",
)

_TTC_FLOOR_S = 1.5


def safety_veto(pred: Prediction) -> np.ndarray:
    """Evaluate the §9 FAIL CLOSED rule for every (m, k) cell.

    Returns a bool [M, K] array; True = the cell fails safety and the planner
    must treat it as unsafe (fail closed), before consulting `min_ttc_s`."""
    if not isinstance(pred, Prediction):
        raise TypeError(f"safety_veto expects a Prediction, got {type(pred).__name__}")
    veto = np.zeros(pred.collision.shape, dtype=bool)
    # §9: invalid cells fail closed — plan side and native side
    if pred.cell_valid is not None:
        veto |= ~pred.cell_valid.astype(bool)
    if pred.native_cell_valid is not None:
        veto |= ~pred.native_cell_valid.astype(bool)
    # NaN in any delay field fails closed (§8.7/§9), in both versions
    for name in _DELAY_FIELDS:
        veto |= np.isnan(getattr(pred, name))
    veto |= pred.collision.astype(bool)
    veto |= pred.native_collision.astype(bool)
    # planner policy (§9): a wall safety_abort fails the plan cell
    for wall_outcome in pred.walls.values():
        veto |= wall_outcome.safety_abort.astype(bool)
    return veto


def _require_v2_fields(pred: Prediction) -> None:
    """§3/§4 v2 REQUIRED outputs: a consumer rejects a missing one, never
    defaults it (fail closed)."""
    missing = [
        f_name
        for f_name in _V2_BOOL_ARRAYS + _V2_FLOAT_ARRAYS
        if getattr(pred, f_name) is None
    ]
    if missing:
        raise ValueError(
            "v2 output is missing REQUIRED future-safety fields "
            f"{missing}; a consumer rejects them, never defaults them "
            "(§3 v2: fail closed)"
        )


def _rel_gate_ok(delta: np.ndarray, tol: float, quantile: float) -> bool:
    """PL-01 rule-4 gate: mean(delta) <= tol AND mean(delta <= tol) >=
    quantile. The caller guarantees finiteness."""
    return (
        float(delta.mean()) <= float(tol)
        and float((delta <= float(tol)).mean()) >= float(quantile)
    )


def safety_veto_v2(
    pred: Prediction,
    tol_future_collision: float = 0.0,
    p_max: float = 0.01,
    rel_safety_tol_s: float = 0.0,
    rel_safety_quantile: float = 0.90,
    native_collision_vetoes: bool = False,
    window_must_not_increase: bool = True,
) -> np.ndarray:
    """Evaluate the v2 §5 rules 1-5 for every (m, k) cell, mirroring the
    planner's PL-01 reading EXACTLY.

    Returns a bool [M, K] array; True = the cell fails safety. Rules 3 and 4
    are call-level owner/planner vetoes: when they fire, every cell is marked.
    `tol_future_collision` is the paired-veto tolerance (default 0); `p_max`
    the owner's absolute future-collision bound (default 0.01);
    `rel_safety_tol_s` / `rel_safety_quantile` the rule-4 mean/quantile gates
    (defaults 0.0 / 0.90); `native_collision_vetoes` re-enables the NATIVE
    window-collision veto (owner decision 2026-09-27: a native collision is
    logged, not a veto — default False); `window_must_not_increase`
    additionally gates the WINDOW-only rule-4 delta with the same mean +
    quantile gates (owner decision 2026-09-27: default True; False restores
    the summed-only behaviour). Requires a `wd01_predict/2` output with
    all REQUIRED v2 fields."""
    if not isinstance(pred, Prediction):
        raise TypeError(f"safety_veto_v2 expects a Prediction, got {type(pred).__name__}")
    if pred.meta.contract_version != CONTRACT_VERSION_V2:
        raise ValueError(
            "safety_veto_v2 consumes 'wd01_predict/2' outputs, got "
            f"{pred.meta.contract_version!r} (use safety_veto for v1/v1.1)"
        )
    _require_v2_fields(pred)
    veto = np.zeros(pred.collision.shape, dtype=bool)

    # --- rule 1: fail closed (validity, NaN delay, window collisions, aborts) ---
    veto |= ~pred.cell_valid.astype(bool)
    veto |= ~pred.native_cell_valid.astype(bool)
    veto |= ~pred.future_cell_valid.astype(bool)
    veto |= ~pred.native_future_cell_valid.astype(bool)
    for name in _DELAY_FIELDS:
        veto |= np.isnan(getattr(pred, name))
    veto |= pred.collision.astype(bool)
    # NATIVE window collision: NOT a veto (owner decision 2026-09-27 — it is
    # logged); the planner can re-enable the veto with native_collision_vetoes=True
    if native_collision_vetoes:
        veto |= pred.native_collision.astype(bool)
    # planner policy (§9): a wall safety_abort fails the plan cell
    for wall_outcome in pred.walls.values():
        veto |= wall_outcome.safety_abort.astype(bool)

    # --- rule 2: absolute window gate ---
    # plan side: collision False and min_ttc_s >= 1.5 wherever a closing pair
    # exists (a NaN TTC fails >= 1.5 -> veto, fail closed)
    veto |= pred.has_closing_pair.astype(bool) & ~(pred.min_ttc_s >= _TTC_FLOOR_S)
    # native side: NO 1.5 s floor (PL-01) — only a NaN native TTC with a
    # closing pair vetoes
    veto |= pred.native_has_closing_pair.astype(bool) & np.isnan(pred.native_min_ttc_s)

    # --- rule 3: owner's future-collision vetoes (absolute + paired) ---
    p_plan = pred.p_future_collision.astype(np.float64)
    p_native = pred.native_p_future_collision.astype(np.float64)
    # a NaN probability never signals a legitimate prediction (§8.7): veto
    veto |= np.isnan(p_plan)
    veto |= np.isnan(p_native)
    if bool((p_plan > p_max).any()):
        veto |= True  # absolute: one bad cell vetoes the whole call
    veto |= p_plan > (p_native + float(tol_future_collision))  # paired cell

    # --- rule 4: relative safety per metric, SUMMED window + future delta ---
    for plan_m, native_m, plan_future_m, native_future_m in (
        (
            "ttc_lt3_exposure_s",
            "native_ttc_lt3_exposure_s",
            "future_ttc_lt3_exposure_s",
            "native_future_ttc_lt3_exposure_s",
        ),
        (
            "hard_brake_s",
            "native_hard_brake_s",
            "future_hard_brake_s",
            "native_future_hard_brake_s",
        ),
    ):
        window_delta = getattr(pred, plan_m) - getattr(pred, native_m)
        delta = window_delta + (
            getattr(pred, plan_future_m) - getattr(pred, native_future_m)
        )
        # any non-finite delta anywhere fails (veto all)
        if not (np.isfinite(window_delta).all() and np.isfinite(delta).all()):
            veto |= True
            continue
        ok_m = _rel_gate_ok(delta, rel_safety_tol_s, rel_safety_quantile)
        # owner decision 2026-09-27: the WINDOW part alone must also pass
        # the same mean + quantile gates (default True)
        if window_must_not_increase and not _rel_gate_ok(
            window_delta, rel_safety_tol_s, rel_safety_quantile
        ):
            ok_m = False
        if not ok_m:
            veto |= True

    # --- rule 5 / G-1: future min-TTC floor (validated) + native NaN veto ---
    if pred.meta.future_min_ttc_validated:
        veto |= pred.future_has_closing_pair.astype(bool) & ~(
            pred.future_min_ttc_q05_s >= _TTC_FLOOR_S
        )
    # a native future q05 NaN with a native future pair vetoes, whether or
    # not the head is validated
    veto |= pred.native_future_has_closing_pair.astype(bool) & np.isnan(
        pred.native_future_min_ttc_q05_s
    )
    return veto


def live_mode_allowed(meta) -> bool:
    """§5 G-1 refusal rule: live / closed-loop mode outside controlled
    experiments requires a VALIDATED `future_min_ttc_q05_s` head, with the
    `future_min_ttc_validation_id` pointing at the validating record."""
    return bool(
        getattr(meta, "future_min_ttc_validated", False) is True
        and isinstance(getattr(meta, "future_min_ttc_validation_id", None), str)
        and meta.future_min_ttc_validation_id
    )


__all__ = ["safety_veto", "safety_veto_v2", "live_mode_allowed"]
