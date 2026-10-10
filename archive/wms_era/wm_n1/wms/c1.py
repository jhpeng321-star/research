"""C1 leave-one-controller-out evaluation protocol and metrics.

Controller-agnostic pure-array functions: no SUMO, no controller code, no
sealed data.  The protocol trains on TRAIN windows whose CAVs never used the
held-out controller (PI) and scores rollouts against matched SUMO truth.

Reported quantities (see ``docs/WORLD_MODEL_RESEARCH_GOAL`` / ``_ref/
WM_S2_ROADMAP.md`` C1):
* car-level ``|ds|`` at 10/30/120/300 s on CAV-affected HDVs;
* traffic-level per-section speed / queue / delay error;
* paired PI-native difference vs SUMO truth (sign agreement, rank correlation);
* seen-vs-unseen error ratio for structured WM-S vs the black-box variant.
"""
from __future__ import annotations

import numpy as np

C1_HORIZONS_S = (10.0, 30.0, 120.0, 300.0)
DEFAULT_DT_S = 0.5
#: The C1 held-out controller and the controllers present in LPO training.
HELD_OUT_CONTROLLER = "pi"
TRAIN_CONTROLLERS = ("native", "damping", "wall")


def horizon_step(horizon_s: float, dt_s: float = DEFAULT_DT_S) -> int:
    return int(round(float(horizon_s) / float(dt_s)))


def _valid_mask(valid, shape, dtype=bool):
    if valid is None:
        return np.ones(shape, dtype=dtype)
    return np.asarray(valid, dtype=dtype)


def cav_affected_hdvs(true_s, counterfactual_s, *, threshold_m: float = 1.0) -> np.ndarray:
    """HDVs whose trajectory differs from the no-CAV counterfactual.

    ``true_s`` / ``counterfactual_s`` are ``[T, N]`` positions of the same HDVs
    in the real CAV run and the matched no-CAV replay.  A vehicle is
    CAV-affected when its position differs by more than ``threshold_m`` at any
    step.
    """
    true_s = np.asarray(true_s, dtype=float)
    cf = np.asarray(counterfactual_s, dtype=float)
    if true_s.shape != cf.shape or true_s.ndim != 2:
        raise ValueError("true_s and counterfactual_s must share shape [T,N]")
    return np.any(np.abs(true_s - cf) > float(threshold_m), axis=0)


def car_level_abs_ds(pred_s, true_s, affected, *, dt_s: float = DEFAULT_DT_S,
                     horizons_s=C1_HORIZONS_S, valid=None) -> dict:
    """Median ``|ds|`` (m) over CAV-affected HDVs at each horizon.

    ``pred_s``/``true_s``: ``[T, N]`` positions; ``affected``: ``[N]`` bool;
    ``valid``: optional ``[T, N]`` bool.  Returns ``{horizon_s: {...}}``.
    """
    pred_s = np.asarray(pred_s, dtype=float)
    true_s = np.asarray(true_s, dtype=float)
    if pred_s.shape != true_s.shape or pred_s.ndim != 2:
        raise ValueError("pred_s and true_s must share shape [T,N]")
    affected = np.asarray(affected, dtype=bool).reshape(-1)
    if affected.shape[0] != pred_s.shape[1]:
        raise ValueError("affected must have one entry per HDV column")
    valid = _valid_mask(valid, pred_s.shape)
    out = {}
    for horizon in horizons_s:
        step = min(horizon_step(horizon, dt_s), pred_s.shape[0] - 1)
        err = np.abs(pred_s[step] - true_s[step])
        sel = affected & valid[step]
        values = err[sel]
        values = values[np.isfinite(values)]
        out[float(horizon)] = {
            "abs_ds_median_m": float(np.median(values)) if values.size else None,
            "abs_ds_mean_m": float(np.mean(values)) if values.size else None,
            "n": int(values.size)}
    return out


def traffic_level_per_section(pred_v, true_v, section_id, *, valid=None,
                              dt_s: float = DEFAULT_DT_S,
                              free_flow_mps: float = 30.0,
                              queue_speed_mps: float = 0.1) -> dict:
    """Per-section speed MAE, queue-count error and delay error.

    ``pred_v``/``true_v``: ``[T, N]`` speeds; ``section_id``: ``[N]`` int.
    Delay of a vehicle is ``sum_t (free_flow - v)+ / free_flow * dt``.
    """
    pred_v = np.asarray(pred_v, dtype=float)
    true_v = np.asarray(true_v, dtype=float)
    if pred_v.shape != true_v.shape or pred_v.ndim != 2:
        raise ValueError("pred_v and true_v must share shape [T,N]")
    section_id = np.asarray(section_id).reshape(-1)
    if section_id.shape[0] != pred_v.shape[1]:
        raise ValueError("section_id must have one entry per vehicle column")
    valid = _valid_mask(valid, pred_v.shape)
    out = {}
    for section in sorted(set(section_id.tolist())):
        cols = section_id == section
        sel = valid[:, cols]
        pe, te = pred_v[:, cols], true_v[:, cols]
        speed_err = np.abs(pe - te)[sel]
        queue_pred = ((pe < queue_speed_mps) & sel).sum(axis=0)
        queue_true = ((te < queue_speed_mps) & sel).sum(axis=0)
        delay_pred = (np.clip(free_flow_mps - pe, 0.0, None) / free_flow_mps
                      * dt_s * sel).sum(axis=0)
        delay_true = (np.clip(free_flow_mps - te, 0.0, None) / free_flow_mps
                      * dt_s * sel).sum(axis=0)
        out[int(section)] = {
            "speed_mae_mps": float(np.mean(speed_err)) if speed_err.size else None,
            "queue_count_error": float(np.mean(np.abs(queue_pred - queue_true)))
            if queue_pred.size else None,
            "delay_error_s": float(np.mean(np.abs(delay_pred - delay_true)))
            if delay_pred.size else None,
            "n_veh": int(cols.sum())}
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties share the mean rank); no scipy dependency."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_v = values[order]
    i = 0
    while i < len(sorted_v):
        j = i + 1
        while j < len(sorted_v) and sorted_v[j] == sorted_v[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = np.mean(ranks[order[i:j]])
        i = j
    return ranks


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def paired_pi_native(pred_pi, pred_native, true_pi, true_native, *,
                     valid=None) -> dict:
    """Predicted vs true paired (PI - native) difference over shared units.

    Each argument is ``[..., N]``; the difference is taken over the last axis and
    reduced over all leading axes so one value per unit remains.  Returns the
    mean signed error, sign agreement and Spearman rank correlation.
    """
    pred_pi = np.asarray(pred_pi, dtype=float)
    pred_native = np.asarray(pred_native, dtype=float)
    true_pi = np.asarray(true_pi, dtype=float)
    true_native = np.asarray(true_native, dtype=float)
    if not (pred_pi.shape == pred_native.shape == true_pi.shape == true_native.shape):
        raise ValueError("all paired arrays must share shape")
    pred_diff = (pred_pi - pred_native).reshape(-1, pred_pi.shape[-1]).mean(axis=0)
    true_diff = (true_pi - true_native).reshape(-1, true_pi.shape[-1]).mean(axis=0)
    if valid is not None:
        mask = np.asarray(valid, bool).reshape(-1, pred_pi.shape[-1]).all(axis=0)
        pred_diff, true_diff = pred_diff[mask], true_diff[mask]
    finite = np.isfinite(pred_diff) & np.isfinite(true_diff)
    pred_diff, true_diff = pred_diff[finite], true_diff[finite]
    sign_agree = (float(np.mean(np.sign(pred_diff) == np.sign(true_diff)))
                  if pred_diff.size else None)
    return {"mean_signed_error": (float(np.mean(pred_diff - true_diff))
                                  if pred_diff.size else None),
            "sign_agreement": sign_agree,
            "spearman": spearman(pred_diff, true_diff) if pred_diff.size else
            float("nan"),
            "n": int(pred_diff.size)}


def _aggregate(values, kind: str = "median") -> float:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                        dtype=float)
    if values.size == 0:
        return float("nan")
    if kind == "median":
        return float(np.median(values))
    if kind == "mean":
        return float(np.mean(values))
    raise ValueError("kind must be 'median' or 'mean'")


def seen_unseen_error_ratio(seen_errors, unseen_errors, *, kind: str = "median") -> dict:
    """``unseen / seen`` error ratio (the C1 headline comparison)."""
    seen = _aggregate(seen_errors, kind)
    unseen = _aggregate(unseen_errors, kind)
    ratio = float(unseen / seen) if np.isfinite(seen) and seen > 0 else float("nan")
    return {"seen": seen, "unseen": unseen, "ratio": ratio}


def seen_unseen_by_model(models: dict, *, kind: str = "median") -> dict:
    """``{model_name: seen_unseen_error_ratio(...)}`` for structured vs black-box."""
    return {name: seen_unseen_error_ratio(payload.get("seen", []),
                                          payload.get("unseen", []), kind=kind)
            for name, payload in models.items()}


def protocol_descriptor() -> dict:
    """Machine-readable C1 protocol summary (no data access)."""
    return {
        "schema": "wm_s_c1_protocol/1",
        "held_out_controller": HELD_OUT_CONTROLLER,
        "train_controllers": list(TRAIN_CONTROLLERS),
        "horizons_s": list(C1_HORIZONS_S),
        "dt_s": DEFAULT_DT_S,
        "car_level": "median |ds| on CAV-affected HDVs at each horizon",
        "traffic_level": "per-section speed MAE / queue-count / delay error",
        "paired": "PI-native paired difference: mean signed error, sign agreement, "
                  "Spearman rank correlation",
        "headline": "seen-vs-unseen error ratio: structured WM-S vs black-box",
        "controller_enters": {
            "structured": "known controller code (CAVs teacher-forced at train, "
                          "controller run at eval)",
            "black_box": "controller-id embedding feature only; CAVs are learned "
                         "movers; no controller code runs"},
    }
