"""INC1 section-5 metrics and the run-once gate.

Primary (delay): top-1 hit rate, mean per-world Spearman rho and mean regret
(s/veh), each with a world-cluster bootstrap 95 % CI (2000 resamples, seed
20261001), plus the paired WM - v2g difference.  Secondary repeats the same
three on TTC < 3 s exposure and hard-brake time.  OOD-vs-error is the
Spearman over (world, arm) between the A3.5 OOD score and the absolute delay
error, reported descriptively.

The script is gated to run ONCE: it refuses when ``INC1_EVAL_DONE.json``
exists and requires a verified ``INC1_MODEL_LOCK.json`` (registration
section 6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from evalh.metrics import world_bootstrap_ci

from .model_lock import (DONE_NAME, LOCK_NAME, ModelLockError, require_lock,
                         sha256_file)

METRICS_NAME = "INC1_EVAL_METRICS.json"
N_BOOT = 2000
SEED = 20261001
ALPHA = 0.05


class EvalAlreadyDone(RuntimeError):
    """The one-shot INC1 evaluation has already run."""


@dataclass
class Channel:
    """One outcome channel: true and predicted ``[W, A]`` arrays."""

    true: np.ndarray
    idea_a: np.ndarray
    v2g: np.ndarray

    def __post_init__(self) -> None:
        self.true = np.asarray(self.true, dtype=float)
        self.idea_a = np.asarray(self.idea_a, dtype=float)
        self.v2g = np.asarray(self.v2g, dtype=float)


@dataclass
class MetricsInputs:
    worlds: list
    arms: list
    delay: Channel
    ttc_lt3_s: Channel
    hard_brake_s: Channel
    ood: Optional[np.ndarray] = None
    #: ``[W, A]`` bool mask of the per-world feasible set (A1.3).  ``None``
    #: means every arm is feasible (the pre-addendum panel).
    feasible: Optional[np.ndarray] = None
    #: feasibility bookkeeping: ``{"per_world": {...}, "reasons": {...},
    #: "arm_feasible_counts": {...}}`` (A1.4).
    feasibility: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------- primitives
def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rho (average ranks for ties); NaN when a ranking is constant."""
    ra = _rankdata(np.asarray(a, dtype=float))
    rb = _rankdata(np.asarray(b, dtype=float))
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def feasible_mask(feasible, shape) -> np.ndarray:
    """Normalize a caller feasible mask to an ``[W, A]`` bool array.

    ``None`` means every arm is feasible (the pre-addendum panel).
    """
    if feasible is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(feasible, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError(
            f"feasible mask shape {mask.shape} != {tuple(shape)}")
    return mask


def feasible_set_sizes(feasible) -> np.ndarray:
    mask = np.asarray(feasible, dtype=bool)
    return mask.sum(axis=1)


def _feasible_idx(mask_row) -> np.ndarray:
    return np.where(np.asarray(mask_row, dtype=bool))[0]


def top1_hit_per_world(pred, true, feasible=None) -> np.ndarray:
    """1.0 per world when the predicted best FEASIBLE arm equals the true best.

    ``feasible`` is the ``[W, A]`` mask; ``None`` = all arms feasible.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = feasible_mask(feasible, pred.shape)
    out = np.full(pred.shape[0], np.nan, dtype=float)
    for w in range(pred.shape[0]):
        idx = _feasible_idx(mask[w])
        if idx.size == 0:
            continue
        out[w] = 1.0 if np.argmin(pred[w][idx]) == np.argmin(true[w][idx]) \
            else 0.0
    return out


def spearman_per_world(pred, true, feasible=None) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = feasible_mask(feasible, pred.shape)
    out = np.full(pred.shape[0], np.nan, dtype=float)
    for w in range(pred.shape[0]):
        idx = _feasible_idx(mask[w])
        if idx.size < 2:
            continue
        out[w] = spearman(pred[w][idx], true[w][idx])
    return out


def regret_per_world(pred, true, feasible=None) -> np.ndarray:
    """True value of the predicted best FEASIBLE arm minus the feasible best.

    This is the A1.4 feasible-set regret (s/veh); ``None`` mask = full set.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = feasible_mask(feasible, pred.shape)
    out = np.full(pred.shape[0], np.nan, dtype=float)
    for w in range(pred.shape[0]):
        idx = _feasible_idx(mask[w])
        if idx.size == 0:
            continue
        p = pred[w][idx]
        t = true[w][idx]
        out[w] = t[np.argmin(p)] - t.min()
    return out


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size and np.isfinite(x).any():
        return float(np.nanmean(x))
    return float("nan")


def _boot(values, worlds, *, n_boot: int, seed: int, alpha: float) -> dict:
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if keep.sum() == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0}
    lo, hi, point = world_bootstrap_ci(values[keep],
                                       np.asarray(worlds)[keep],
                                       n_boot=n_boot, alpha=alpha, seed=seed)
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n": int(keep.sum())}


def _model_metrics(pred, true, worlds, *, n_boot, seed, alpha,
                   feasible=None) -> dict:
    hit = top1_hit_per_world(pred, true, feasible)
    rho = spearman_per_world(pred, true, feasible)
    reg = regret_per_world(pred, true, feasible)
    return {
        "top1_hit": {"point": _nanmean(hit), "n": int(np.isfinite(hit).sum()),
                     "ci": _boot(hit, worlds, n_boot=n_boot, seed=seed,
                                 alpha=alpha)},
        "spearman": {"point": _nanmean(rho),
                     "ci": _boot(rho, worlds, n_boot=n_boot, seed=seed,
                                 alpha=alpha)},
        # A1.4 PRIMARY metric: feasible-set regret (s/veh)
        "feasible_set_regret_s_per_veh": {
            "point": _nanmean(reg),
            "ci": _boot(reg, worlds, n_boot=n_boot, seed=seed, alpha=alpha)},
        # kept name for report stability (same number, feasible-set based)
        "regret_s_per_veh": {"point": _nanmean(reg),
                             "ci": _boot(reg, worlds, n_boot=n_boot,
                                         seed=seed, alpha=alpha)},
        "_per_world": {"top1_hit": hit, "spearman": rho, "regret": reg},
    }


def _paired(a_per_world, b_per_world, worlds, *, n_boot, seed, alpha) -> dict:
    diff = np.asarray(a_per_world, dtype=float) - \
        np.asarray(b_per_world, dtype=float)
    return {"point": _nanmean(diff),
            "ci": _boot(diff, worlds, n_boot=n_boot, seed=seed, alpha=alpha)}


def _channel_report(ch: Channel, worlds, *, n_boot, seed, alpha,
                    feasible=None) -> dict:
    out = {"idea_a": {}, "v2g": {}}
    per = {}
    for model in ("idea_a", "v2g"):
        pred = getattr(ch, model)
        out[model] = _model_metrics(pred, ch.true, worlds, n_boot=n_boot,
                                    seed=seed, alpha=alpha, feasible=feasible)
        per[model] = out[model].pop("_per_world")
    out["paired_wm_minus_v2g"] = {
        "top1_hit": _paired(per["idea_a"]["top1_hit"],
                            per["v2g"]["top1_hit"], worlds, n_boot=n_boot,
                            seed=seed, alpha=alpha),
        "spearman": _paired(per["idea_a"]["spearman"],
                            per["v2g"]["spearman"], worlds, n_boot=n_boot,
                            seed=seed, alpha=alpha),
        "feasible_set_regret_s_per_veh": _paired(
            per["idea_a"]["regret"], per["v2g"]["regret"], worlds,
            n_boot=n_boot, seed=seed, alpha=alpha),
    }
    return out


# ------------------------------------------------- A2.4 descriptive breakdown
#: ADDENDUM_2 A2.4: the multi-CAV responses vs the single-CAV contrasts.
MULTI_CAV_ARMS = ("R3", "R4", "R5", "R6")
SINGLE_CAV_ARMS = ("R1", "R2")


def subgroup_indices(arms, subgroup) -> list:
    """Column indices of ``arms`` that name an arm in ``subgroup``."""
    wanted = {str(a) for a in subgroup}
    return [i for i, a in enumerate(arms) if str(a) in wanted]


def _subgroup_mask(mask, idx, shape) -> np.ndarray:
    """A feasible mask restricted to the ``idx`` columns (others infeasible)."""
    out = np.zeros(shape, dtype=bool)
    if idx:
        out[:, idx] = np.asarray(mask, dtype=bool)[:, idx]
    return out


def true_best_indices(true, mask) -> np.ndarray:
    """Per world, the column index of the feasible true minimum (or -1)."""
    true = np.asarray(true, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    out = np.full(true.shape[0], -1, dtype=int)
    for w in range(true.shape[0]):
        idx = _feasible_idx(mask[w])
        if idx.size:
            out[w] = int(idx[np.argmin(true[w][idx])])
    return out


def a24_breakdown(inputs: MetricsInputs, mask: np.ndarray) -> dict:
    """A2.4 descriptive breakdowns (no bootstrap; report-only).

    * feasible-set regret restricted to multi-CAV {R3,R4,R5,R6} vs single-CAV
      {R1,R2};
    * the WM's top-1 hit on worlds whose SUMO-best is a multi-CAV response;
    * the OOD score per arm.
    """
    arms = [str(a) for a in inputs.arms]
    shape = mask.shape
    multi_idx = subgroup_indices(arms, MULTI_CAV_ARMS)
    single_idx = subgroup_indices(arms, SINGLE_CAV_ARMS)
    multi_mask = _subgroup_mask(mask, multi_idx, shape)
    single_mask = _subgroup_mask(mask, single_idx, shape)
    out: dict = {"multi_cav_arms": [arms[i] for i in multi_idx],
                 "single_cav_arms": [arms[i] for i in single_idx],
                 "delay_feasible_set_regret_s_per_veh": {}}
    for name, sub_mask in (("multi_cav", multi_mask),
                           ("single_cav", single_mask)):
        row = {}
        for model in ("idea_a", "v2g"):
            pred = getattr(inputs.delay, model)
            reg = regret_per_world(pred, inputs.delay.true, sub_mask)
            row[model] = {"point": _nanmean(reg),
                          "n": int(np.isfinite(reg).sum())}
        out["delay_feasible_set_regret_s_per_veh"][name] = row
    # worlds whose SUMO-best is a multi-CAV response
    best = true_best_indices(inputs.delay.true, mask)
    multi_best = np.isin(best, multi_idx) if multi_idx else np.zeros(
        len(best), dtype=bool)
    out["worlds_with_multi_cav_sumo_best"] = int(multi_best.sum())
    out["top1_on_multi_cav_best_worlds"] = {}
    for model in ("idea_a", "v2g"):
        pred = getattr(inputs.delay, model)
        hit = top1_hit_per_world(pred, inputs.delay.true, mask)
        hit = hit[multi_best]
        out["top1_on_multi_cav_best_worlds"][model] = {
            "point": _nanmean(hit), "n": int(np.isfinite(hit).sum())}
    # OOD per arm (descriptive)
    if inputs.ood is not None:
        ood = np.asarray(inputs.ood, dtype=float)
        if ood.ndim == 1 and ood.size == mask.size:
            ood = ood.reshape(mask.shape)
        if ood.shape == mask.shape:
            out["ood_per_arm"] = {
                arms[i]: float(np.nanmean(ood[:, i])) if np.isfinite(
                    ood[:, i]).any() else float("nan")
                for i in range(len(arms))}
        else:
            out["ood_per_arm"] = None
    else:
        out["ood_per_arm"] = None
    return out


def ood_vs_error(ood, pred, true) -> dict:
    """Spearman over (world, arm) between OOD score and |delay error|."""
    ood = np.asarray(ood, dtype=float).reshape(-1)
    err = np.abs(np.asarray(pred, dtype=float)
                 - np.asarray(true, dtype=float)).reshape(-1)
    keep = np.isfinite(ood) & np.isfinite(err)
    if keep.sum() < 2:
        return {"spearman": float("nan"), "n": int(keep.sum())}
    return {"spearman": spearman(ood[keep], err[keep]),
            "n": int(keep.sum())}


def compute_report(inputs: MetricsInputs, *, n_boot: int = N_BOOT,
                   seed: int = SEED, alpha: float = ALPHA) -> dict:
    """The full section-5 report (pure; no gating, no file access).

    A1.4: the top-1 / Spearman / regret metrics are computed on the per-world
    FEASIBLE set (``inputs.feasible``); the feasible-set regret is the primary
    metric.  Per-world feasible-set sizes and feasibility counts/reasons are
    reported alongside.
    """
    worlds = [str(w) for w in inputs.worlds]
    mask = feasible_mask(inputs.feasible,
                         (len(worlds), len(inputs.arms)))
    counts = inputs.arms and mask.sum(axis=0)
    arm_counts = ({str(a): int(counts[i])
                   for i, a in enumerate(inputs.arms)} if inputs.arms else {})
    report = {
        "schema": "wm_n1_inc1_metrics/1",
        "worlds": worlds, "arms": [str(a) for a in inputs.arms],
        "n_worlds": len(worlds), "n_arms": len(inputs.arms),
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed),
                      "alpha": float(alpha), "cluster": "world"},
        "meta": dict(inputs.meta),
        "feasible_set": {
            "primary_metric": "feasible_set_regret_s_per_veh",
            "per_world_size": {worlds[w]: int(mask[w].sum())
                               for w in range(len(worlds))},
            "mean_size": (float(mask.sum(axis=1).mean()) if worlds
                          else float("nan")),
            "arm_feasible_counts": arm_counts,
            "reasons": dict(inputs.feasibility.get("reasons", {})),
            "per_world": dict(inputs.feasibility.get("per_world", {})),
        },
    }
    for name, ch in (("delay", inputs.delay),
                     ("ttc_lt3_s", inputs.ttc_lt3_s),
                     ("hard_brake_s", inputs.hard_brake_s)):
        report[name] = _channel_report(ch, worlds, n_boot=n_boot, seed=seed,
                                       alpha=alpha, feasible=mask)
    if inputs.ood is not None:
        report["ood_vs_error"] = {
            "idea_a": ood_vs_error(inputs.ood, inputs.delay.idea_a,
                                   inputs.delay.true),
            "v2g": ood_vs_error(inputs.ood, inputs.delay.v2g,
                                inputs.delay.true),
        }
    else:
        report["ood_vs_error"] = None
    report["a2_4_breakdown"] = a24_breakdown(inputs, mask)
    return report


# ---------------------------------------------------------------- the gate
def metrics_path(out_dir) -> Path:
    return Path(out_dir) / METRICS_NAME


def done_path(out_dir) -> Path:
    return Path(out_dir) / DONE_NAME


def assert_not_done(out_dir) -> None:
    if done_path(out_dir).exists():
        raise EvalAlreadyDone(
            f"{DONE_NAME} exists in {out_dir}: the INC1 evaluation is sealed "
            "and runs exactly once (registration section 6)")


def write_done(out_dir, payload: dict) -> Path:
    path = done_path(out_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def run_inc1_metrics(out_dir, inputs: MetricsInputs, *, members,
                     pack_manifest, v2g_params, code_commit=None,
                     n_boot: int = N_BOOT, seed: int = SEED,
                     alpha: float = ALPHA, write: bool = True) -> dict:
    """The gated one-shot metrics run.

    Order: refuse if already done, then verify the model lock against the
    files actually loaded, then (and only then) compute and write.
    """
    out_dir = Path(out_dir)
    assert_not_done(out_dir)
    if not (out_dir / LOCK_NAME).is_file():
        raise ModelLockError(
            f"missing {LOCK_NAME} in {out_dir}: refusing to run INC1 metrics "
            "before the model lock (registration section 6)")
    require_lock(out_dir, members=members, pack_manifest=pack_manifest,
                 v2g_params=v2g_params, code_commit=code_commit)
    report = compute_report(inputs, n_boot=n_boot, seed=seed, alpha=alpha)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path(out_dir).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        write_done(out_dir, {
            "schema": "wm_n1_inc1_eval_done/1",
            "metrics": METRICS_NAME,
            "members": {str(k): sha256_file(v) for k, v in dict(members).items()},
            "pack_sha": sha256_file(pack_manifest),
            "v2g_params_sha": sha256_file(v2g_params),
            "code_commit": code_commit,
        })
    return report


def main(argv=None) -> int:                                # pragma: no cover
    raise SystemExit(
        "inc1eval.metrics is driven by the sealed INC1 runner; there is no "
        "standalone CLI that could bypass the one-shot gate")
