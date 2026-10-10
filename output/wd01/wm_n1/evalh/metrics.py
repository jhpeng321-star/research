"""The WM-N1 registration metrics and the frozen decision rule.

Every function cites the registration line it implements
(``REGISTRATION_WM_N1.md``, frozen 2026-09-30).  The registration's section 4
is the authority; the design section 6 describes the same quantities.

Implemented here (citation -> function):

* paired dV CRPS with comonotone quantile pairing --
  REGISTRATION_WM_N1.md:82-84 (design:201-204) -> :func:`crps_of_quantiles`,
  :func:`build_delta_pairs`, :func:`delta_metrics`;
* sign accuracy on ``|dY| > TRAIN-median`` pairs with a world-bootstrap
  CI -- REGISTRATION_WM_N1.md:83-84,92 -> :func:`sign_accuracy`,
  :func:`world_bootstrap_ci`;
* level CRPS and MAE -- REGISTRATION_WM_N1.md:85-86,92 ->
  :func:`level_metrics`;
* 10-90 coverage and PIT -- REGISTRATION_WM_N1.md:92-96 (design:199-200)
  -> :func:`coverage`, :func:`pit_values`;
* short-horizon position/speed RMSE -- REGISTRATION_WM_N1.md:87
  (design:195-198) -> :func:`rmse_percent`, :func:`rollout_metrics`;
* collision / TTC error -- REGISTRATION_WM_N1.md:88,90 ->
  :func:`per_1000`, :func:`rollout_metrics`;
* the decision rule -- REGISTRATION_WM_N1.md:79-88 -> :func:`decide`.

Sign convention: a LOWER CRPS/RMSE/collision count is better, so a contrast
"improves" when the variant value is below the hybrid's.  ``cost difference =
treatment - reference; negative = improvement``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .protocol import (GATING_CONTRASTS, GATING_TAUS, QUANTILE_LEVELS,
                       OutcomeRow, RolloutBatch, RolloutRow)

# REGISTRATION_WM_N1.md:79-88, verbatim (Unicode operators preserved).
DECISION_RULE_QUOTE = (
    "**Primary (planner relevance):** \"The planner variant beats the hybrid\" "
    "iff,\n"
    "on TEST, ALL of:\n"
    "\n"
    "1. \u0394V CRPS is lower than T's for \u2265 2 of the gating contrasts "
    "{damping\u2212native,\n"
    "   PI\u2212native, wall\u2212native (if available)} at \u2265 2 of the "
    "\u03c4 {60, 120, 300}. The\n"
    "   world-bootstrap 90 % CI of the difference must exclude 0.\n"
    "2. Outcome CRPS is not worse than T's by more than 5 % at any \u03c4 "
    "(delay; the\n"
    "   safety heads are reported).\n"
    "3. H10/H30 s_RMSE is not worse than Part A v2g by more than 10 %.\n"
    "4. Collisions \u2264 T + 0.1 per 1000 veh-steps. Interventions are "
    "reported."
)

# contrast -> (treatment policy, reference policy); REGISTRATION_WM_N1.md:82.
CONTRASTS = {
    "damping-native": ("damping", "native"),
    "pi-native": ("pi", "native"),
    "wall-native": ("wall", "native"),
}
OUTCOME_CRPS_TOLERANCE = 0.05   # item 2: not worse by more than 5 %
S_RMSE_TOLERANCE = 0.10         # item 3: not worse by more than 10 %
COLLISION_MARGIN_PER_1000 = 0.1  # item 4

# --------------------------------------------------------------- primitives --

def _levels(levels) -> np.ndarray:
    return np.asarray(QUANTILE_LEVELS if levels is None else levels, float)


def level_index(level: float, levels=None) -> int:
    """Index of ``level`` in the quantile grid (nearest)."""
    lv = _levels(levels)
    return int(np.argmin(np.abs(lv - float(level))))


def median_index(levels=None) -> int:
    return level_index(0.5, levels)


def pinball_loss(q, y, levels=None) -> np.ndarray:
    """Quantile (pinball) loss, ``levels`` broadcast over the last axis."""
    lv = _levels(levels)
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    diff = y[..., None] - q
    return np.where(diff >= 0.0, lv * diff, (lv - 1.0) * diff)


def crps_of_quantiles(q, y, levels=None) -> np.ndarray:
    """CRPS from a quantile forecast: ``2 * mean_alpha pinball``.

    Exact up to the quadrature of the 19-level grid; reduces to ``|x - m|``
    for a point mass at the median.  Used for the level outcome CRPS
    (REGISTRATION_WM_N1.md:85) and, on comonotone quantile differences, for
    the paired dV CRPS (REGISTRATION_WM_N1.md:82).
    """
    return 2.0 * np.mean(pinball_loss(q, y, levels), axis=-1)


def sign_accuracy(pred_median, dy, threshold: float) -> tuple:
    """Sign accuracy on ``|dY| > threshold`` pairs (REGISTRATION_WM_N1.md:92).

    Returns ``(accuracy, n_used)``; ``accuracy`` is NaN when no pair qualifies.
    """
    pred_median = np.asarray(pred_median, float)
    dy = np.asarray(dy, float)
    keep = np.abs(dy) > float(threshold)
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0
    ok = np.sign(pred_median[keep]) == np.sign(dy[keep])
    return float(np.mean(ok)), n


def sign_accuracy_ci(pred_median, dy, worlds, threshold: float, *,
                     n_boot: int = 2000, alpha: float = 0.10,
                     seed: int = 0) -> tuple:
    """Sign accuracy with a world-bootstrap CI (design:203, reg:92).

    Returns ``(accuracy, n_used, lo, hi)`` at ``1 - alpha`` coverage on the
    ``|dY| > threshold`` pairs; all-NaN when no pair qualifies.
    """
    pred_median = np.asarray(pred_median, float)
    dy = np.asarray(dy, float)
    worlds = np.asarray([str(w) for w in worlds])
    keep = np.abs(dy) > float(threshold)
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0, float("nan"), float("nan")
    ok = (np.sign(pred_median[keep]) == np.sign(dy[keep])).astype(float)
    lo, hi, _ = world_bootstrap_ci(ok, worlds[keep], n_boot=n_boot,
                                   alpha=alpha, seed=seed, statistic=np.mean)
    return float(np.mean(ok)), n, lo, hi


def world_bootstrap_ci(values, worlds, *, n_boot: int = 2000, alpha: float = 0.10,
                       seed: int = 0, statistic=np.mean) -> tuple:
    """World-level bootstrap CI of ``statistic`` (REGISTRATION_WM_N1.md:83).

    Whole worlds are resampled with replacement, so within-world dependence is
    preserved.  Returns ``(lo, hi, point)`` at ``1 - alpha`` coverage.
    """
    values = np.asarray(values, float)
    worlds = np.asarray([str(w) for w in worlds])
    uniq = np.unique(worlds)
    if uniq.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx_by = {w: np.where(worlds == w)[0] for w in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([idx_by[w] for w in pick])
        stats[b] = statistic(values[idx])
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi, float(statistic(values))


def coverage(q_low, q_high, y) -> float:
    """Empirical 10-90 % coverage (REGISTRATION_WM_N1.md:92)."""
    y = np.asarray(y, float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((np.asarray(q_low, float) <= y)
                         & (y <= np.asarray(q_high, float))))


def pit_values(q, y, levels=None) -> np.ndarray:
    """PIT = F(y) from discrete quantiles, linear between levels."""
    lv = _levels(levels)
    q = np.sort(np.asarray(q, float), axis=-1)
    y = np.asarray(y, float)
    out = np.empty(y.shape, float)
    flat_q = q.reshape(-1, q.shape[-1])
    flat_y = y.reshape(-1)
    xs = np.concatenate([[0.0], lv, [1.0]])
    for i in range(flat_q.shape[0]):
        ys = np.concatenate([[flat_q[i, 0] - 1.0], flat_q[i],
                             [flat_q[i, -1] + 1.0]])
        out.reshape(-1)[i] = np.interp(flat_y[i], ys, xs)
    return out


def mae(pred, y) -> float:
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    if y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - y)))


def rmse_percent(pred, true, denominator: float) -> float:
    """RMSE as a % of the true distance/speed (REGISTRATION_WM_N1.md:87)."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    if pred.size == 0 or denominator <= 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(100.0 * np.sqrt(np.nanmean((pred - true) ** 2))
                     / denominator)


def per_1000(count_sum: float, veh_steps_sum: float) -> float:
    """Count per 1000 vehicle-steps (REGISTRATION_WM_N1.md:88)."""
    if veh_steps_sum <= 0:
        return float("nan")
    return float(1000.0 * count_sum / veh_steps_sum)


# ------------------------------------------------------------- pair tables --

def build_delta_pairs(rows: Sequence[OutcomeRow], variant_q: np.ndarray,
                      hybrid_q: np.ndarray, *, levels=None,
                      contrasts=None, taus=GATING_TAUS) -> list:
    """One dict per contrast pair on an identical window.

    Comonotone pairing: ``dV`` quantiles are ``q_variant[a] - q_variant[b]``
    at the SAME levels (design:114-115), and likewise for the hybrid.
    """
    lv = _levels(levels)
    mi = median_index(lv)
    contrasts = CONTRASTS if contrasts is None else contrasts
    taus = {round(float(t), 6) for t in taus}
    idx = {}
    for i, r in enumerate(rows):
        idx[(r.world, round(float(r.t0_s), 6), round(float(r.decision_t_s), 6),
             round(float(r.tau_s), 6), r.policy)] = i
    keys = sorted({(r.world, round(float(r.t0_s), 6),
                    round(float(r.decision_t_s), 6),
                    round(float(r.tau_s), 6)) for r in rows})
    pairs = []
    for (world, t0, dec, tau) in keys:
        if tau not in taus:
            continue
        for name, (pol_a, pol_b) in contrasts.items():
            ia = idx.get((world, t0, dec, tau, pol_a))
            ib = idx.get((world, t0, dec, tau, pol_b))
            if ia is None or ib is None:
                continue
            dy = float(rows[ia].y_delay_s - rows[ib].y_delay_s)
            dv_variant = np.asarray(variant_q[ia] - variant_q[ib], float)
            dv_hybrid = np.asarray(hybrid_q[ia] - hybrid_q[ib], float)
            pairs.append(dict(
                world=world, t0_s=t0, decision_t_s=dec, tau_s=tau,
                contrast=name, policy_a=pol_a, policy_b=pol_b, dy=dy,
                variant_crps=float(crps_of_quantiles(dv_variant, dy, lv)),
                hybrid_crps=float(crps_of_quantiles(dv_hybrid, dy, lv)),
                variant_sign=float(np.sign(dv_variant[mi])),
                hybrid_sign=float(np.sign(dv_hybrid[mi])),
                variant_dv=float(np.mean(dv_variant)),
                hybrid_dv=float(np.mean(dv_hybrid))))
    return pairs


def delta_metrics(pairs, *, threshold: float, levels=None, n_boot: int = 2000,
                  alpha: float = 0.10, seed: int = 0) -> dict:
    """Aggregate the contrast pairs into per-(contrast, tau) results.

    Returns ``{contrast: {tau: {...}}}`` with the variant/hybrid mean dV CRPS,
    the world-bootstrap CI of their per-pair difference, and the sign
    accuracies on ``|dY| > threshold`` pairs.
    """
    out: dict = {}
    for p in pairs:
        out.setdefault(p["contrast"], {}).setdefault(float(p["tau_s"]), {
            "variant_crps": [], "hybrid_crps": [], "diff": [], "worlds": [],
            "dy": [], "variant_sign": [], "hybrid_sign": []})
        cell = out[p["contrast"]][float(p["tau_s"])]
        cell["variant_crps"].append(p["variant_crps"])
        cell["hybrid_crps"].append(p["hybrid_crps"])
        cell["diff"].append(p["variant_crps"] - p["hybrid_crps"])
        cell["worlds"].append(p["world"])
        cell["dy"].append(p["dy"])
        cell["variant_sign"].append(p["variant_sign"])
        cell["hybrid_sign"].append(p["hybrid_sign"])
    result: dict = {}
    for contrast, by_tau in out.items():
        result[contrast] = {}
        for tau, cell in by_tau.items():
            lo, hi, point = world_bootstrap_ci(
                cell["diff"], cell["worlds"], n_boot=n_boot, alpha=alpha,
                seed=seed)
            va, n_used, v_lo, v_hi = sign_accuracy_ci(
                cell["variant_sign"], cell["dy"], cell["worlds"], threshold,
                n_boot=n_boot, alpha=alpha, seed=seed)
            ha, _, h_lo, h_hi = sign_accuracy_ci(
                cell["hybrid_sign"], cell["dy"], cell["worlds"], threshold,
                n_boot=n_boot, alpha=alpha, seed=seed)
            result[contrast][tau] = dict(
                n_pairs=len(cell["diff"]),
                n_worlds=len(set(cell["worlds"])),
                variant_crps=float(np.mean(cell["variant_crps"])),
                hybrid_crps=float(np.mean(cell["hybrid_crps"])),
                diff_mean=point, ci_lo=lo, ci_hi=hi,
                ci_excludes_zero=bool(lo > 0.0 or hi < 0.0),
                improves=bool(hi < 0.0),
                variant_sign_accuracy=va, hybrid_sign_accuracy=ha,
                variant_sign_ci=[v_lo, v_hi], hybrid_sign_ci=[h_lo, h_hi],
                sign_n_used=n_used,
                sign_threshold=float(threshold))
    return result


# ------------------------------------------------------------ level metrics --

def level_metrics(rows: Sequence[OutcomeRow], q: np.ndarray, *, name: str,
                  levels=None) -> dict:
    """CRPS, MAE, 10-90 coverage and PIT per tau (REGISTRATION_WM_N1.md:85-96)."""
    lv = _levels(levels)
    mi = median_index(lv)
    lo_i = level_index(0.10, lv)
    hi_i = level_index(0.90, lv)
    out: dict = {}
    taus = sorted({round(float(r.tau_s), 6) for r in rows})
    for tau in taus:
        sel = [i for i, r in enumerate(rows)
               if round(float(r.tau_s), 6) == tau]
        if not sel:
            continue
        qq = np.asarray(q, float)[sel]
        y = np.asarray([rows[i].y_delay_s for i in sel], float)
        out[tau] = dict(
            name=name, n=len(sel),
            crps=float(np.mean(crps_of_quantiles(qq, y, lv))),
            mae=mae(qq[:, mi], y),
            coverage_10_90=coverage(qq[:, lo_i], qq[:, hi_i], y),
            pit_mean=float(np.mean(pit_values(qq, y, lv))),
            mean_delay=float(np.mean(y)))
    return out


def level_crps_pair(rows, variant_q, hybrid_q, *, levels=None) -> dict:
    """Per-tau (variant, hybrid) level CRPS for decision-rule item 2."""
    vm = level_metrics(rows, variant_q, name="variant", levels=levels)
    hm = level_metrics(rows, hybrid_q, name="hybrid", levels=levels)
    out = {}
    for tau in sorted(set(vm) & set(hm)):
        out[tau] = dict(variant_crps=vm[tau]["crps"],
                        hybrid_crps=hm[tau]["crps"],
                        ratio=vm[tau]["crps"] / hm[tau]["crps"]
                        if hm[tau]["crps"] else float("nan"))
    return out


# ---------------------------------------------------------- rollout metrics --

def rollout_metrics(pred: RolloutBatch, true_rows: Sequence[RolloutRow],
                    *, name: str, horizons: Sequence[float] | None = None) -> dict:
    """Short-horizon s/v RMSE and validity counters (REGISTRATION_WM_N1.md:87-88).

    ``s_RMSE``/``v_RMSE`` are per horizon, as a % of the true distance / speed
    over all rows; collisions and TTC<3 exposure are per 1000 veh-steps.
    """
    hp = tuple(horizons if horizons is not None else pred.horizons_s)
    pos = np.asarray(pred.pos, float)
    speed = np.asarray(pred.speed, float)
    per_h = {}
    dist_denom = float(np.mean([r.true_dist for r in true_rows])) \
        if true_rows else 0.0
    speed_denom = float(np.mean([r.true_speed_scale for r in true_rows])) \
        if true_rows else 0.0
    n = len(true_rows)
    for hi, h in enumerate(hp):
        p = pos[:, hi, :]
        s = speed[:, hi, :]
        tp = np.stack([r.true_pos[min(hi, r.true_pos.shape[0] - 1)]
                       for r in true_rows]) if n else np.zeros((0, 0))
        ts = np.stack([r.true_speed[min(hi, r.true_speed.shape[0] - 1)]
                       for r in true_rows]) if n else np.zeros((0, 0))
        if pred.mask is not None:
            m = np.asarray(pred.mask, float)[:, hi, :]
            p = np.where(m > 0, p, np.nan)
        per_h[float(h)] = dict(
            s_rmse_pct=rmse_percent(p, tp, dist_denom),
            v_rmse_pct=rmse_percent(s, ts, speed_denom),
            s_rmse_m=float(np.sqrt(np.nanmean((p - tp) ** 2)))
            if p.size else float("nan"),
            v_rmse_mps=float(np.sqrt(np.nanmean((s - ts) ** 2)))
            if s.size else float("nan"))
    veh_steps = float(np.sum([r.veh_steps for r in true_rows])) if n else 0.0
    true_coll = float(np.sum([r.collision for r in true_rows])) if n else 0.0
    true_ttc = float(np.sum([r.ttc_lt3_s for r in true_rows])) if n else 0.0
    pred_coll = float(np.sum(pred.collision)) if pred.collision is not None else 0.0
    pred_ttc = float(np.sum(pred.ttc_lt3_s)) if pred.ttc_lt3_s is not None else 0.0
    return dict(
        name=name, n_windows=n,
        s_rmse_pct=per_h,
        collision_per_1000=per_1000(pred_coll, veh_steps),
        true_collision_per_1000=per_1000(true_coll, veh_steps),
        collision_error=pred_coll - true_coll,
        ttc_lt3_s=pred_ttc, true_ttc_lt3_s=true_ttc,
        ttc_error=pred_ttc - true_ttc,
        intervention_per_1000=per_1000(
            float(np.sum(pred.intervention))
            if pred.intervention is not None else 0.0, veh_steps),
        veh_steps=veh_steps)


# ---------------------------------------------------------- decision rule --

@dataclass
class ContrastResult:
    """One (contrast, tau) cell of the decision-rule evidence."""

    contrast: str
    tau: float
    variant_crps: float
    hybrid_crps: float
    ci_lo: float
    ci_hi: float


@dataclass
class DecisionEvidence:
    """All quantities the decision rule consumes (REGISTRATION_WM_N1.md:79-88)."""

    contrasts: list = field(default_factory=list)          # ContrastResult
    level_crps: dict = field(default_factory=dict)         # tau -> {variant_crps, hybrid_crps}
    s_rmse: dict = field(default_factory=dict)             # horizon -> (variant, partA)
    collisions_per_1000: tuple = (float("nan"), float("nan"))
    gating_contrasts: tuple = GATING_CONTRASTS
    gating_taus: tuple = GATING_TAUS


def _as_contrast_results(delta: dict) -> list:
    out = []
    for contrast, by_tau in delta.items():
        for tau, cell in by_tau.items():
            out.append(ContrastResult(
                contrast=contrast, tau=float(tau),
                variant_crps=cell["variant_crps"],
                hybrid_crps=cell["hybrid_crps"],
                ci_lo=cell["ci_lo"], ci_hi=cell["ci_hi"]))
    return out


def decide(ev: DecisionEvidence) -> dict:
    """The frozen decision rule, item by item (REGISTRATION_WM_N1.md:79-88).

    Returns ``dict(verdict, checks, quote)`` where ``verdict`` is
    ``"WM-N1_BEATS_HYBRID"`` iff every check passes, else
    ``"HYBRID_STAYS"``.
    """
    taus = {round(float(t), 6) for t in ev.gating_taus}
    contrasts = list(ev.gating_contrasts)
    checks: dict = {}

    # Item 1 (line 82-84): lower dV CRPS with the CI excluding 0, for >=2
    # contrasts each at >=2 taus.
    qualifying: dict = {c: [] for c in contrasts}
    for cell in ev.contrasts:
        if cell.contrast not in qualifying:
            continue
        if round(cell.tau, 6) not in taus:
            continue
        improved = (cell.variant_crps < cell.hybrid_crps
                    and (cell.ci_hi < 0.0 or cell.ci_lo > 0.0))
        if improved:
            qualifying[cell.contrast].append(float(cell.tau))
    n_goods = sum(1 for c in contrasts if len(set(qualifying[c])) >= 2)
    checks["1_dv_crps_ge2_contrasts_ge2_taus"] = dict(
        passed=bool(n_goods >= 2),
        n_qualifying_contrasts=int(n_goods),
        per_contrast={c: sorted(set(qualifying[c])) for c in contrasts},
        citation="REGISTRATION_WM_N1.md:82-84")

    # Item 2 (line 85-86): outcome CRPS not worse than T by > 5 % at any tau.
    worst = None
    item2 = True
    for tau, cell in ev.level_crps.items():
        ratio = (cell["variant_crps"] / cell["hybrid_crps"]
                 if cell["hybrid_crps"] else float("inf"))
        if worst is None or ratio > worst[1]:
            worst = (float(tau), ratio)
        if ratio > 1.0 + OUTCOME_CRPS_TOLERANCE:
            item2 = False
    checks["2_outcome_crps_within_5pct"] = dict(
        passed=bool(item2),
        worst_tau_ratio=worst,
        tolerance=OUTCOME_CRPS_TOLERANCE,
        citation="REGISTRATION_WM_N1.md:85-86")

    # Item 3 (line 87): H10/H30 s_RMSE not worse than Part A v2g by > 10 %.
    s_ok = True
    s_detail = {}
    for h in (10.0, 30.0):
        cell = ev.s_rmse.get(h)
        if cell is None:
            continue
        variant, parta = float(cell[0]), float(cell[1])
        ratio = variant / parta if parta else float("inf")
        s_detail[h] = ratio
        if ratio > 1.0 + S_RMSE_TOLERANCE:
            s_ok = False
    checks["3_h10_h30_s_rmse_within_10pct"] = dict(
        passed=bool(s_ok), ratios=s_detail,
        tolerance=S_RMSE_TOLERANCE,
        citation="REGISTRATION_WM_N1.md:87")

    # Item 4 (line 88): collisions <= T + 0.1 / 1000 veh-steps.
    v_coll, t_coll = ev.collisions_per_1000
    item4 = bool(np.isfinite(v_coll) and np.isfinite(t_coll)
                 and v_coll <= t_coll + COLLISION_MARGIN_PER_1000)
    checks["4_collisions_within_margin"] = dict(
        passed=item4, variant=v_coll, hybrid=t_coll,
        margin=COLLISION_MARGIN_PER_1000,
        citation="REGISTRATION_WM_N1.md:88")

    passed = all(c["passed"] for c in checks.values())
    return dict(
        verdict="WM-N1_BEATS_HYBRID" if passed else "HYBRID_STAYS",
        passed=passed, checks=checks, quote=DECISION_RULE_QUOTE,
        citation="REGISTRATION_WM_N1.md:79-88")


def evidence_from_metrics(delta: dict, level_crps: dict, rollout_variant: dict,
                          rollout_hybrid: dict, *,
                          gating_contrasts=GATING_CONTRASTS,
                          gating_taus=GATING_TAUS) -> DecisionEvidence:
    """Assemble :class:`DecisionEvidence` from aggregated metric dicts."""
    s_variant = rollout_variant["s_rmse_pct"]
    s_hybrid = rollout_hybrid["s_rmse_pct"]
    s_rmse = {}
    for h in (10.0, 30.0):
        if h in s_variant and h in s_hybrid:
            s_rmse[h] = (s_variant[h]["s_rmse_pct"],
                         s_hybrid[h]["s_rmse_pct"])
    return DecisionEvidence(
        contrasts=_as_contrast_results(delta),
        level_crps={float(t): c for t, c in level_crps.items()},
        s_rmse=s_rmse,
        collisions_per_1000=(rollout_variant["collision_per_1000"],
                             rollout_hybrid["collision_per_1000"]),
        gating_contrasts=tuple(gating_contrasts),
        gating_taus=tuple(gating_taus))
