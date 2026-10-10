"""The CAL-tune scoring metric loop (E4): paired dV of A vs Part A v2g.

:func:`score` consumes ``evalh.protocol.OutcomeRow`` rows and one
``[N, 19]`` delay-quantile array per model (``{"A": ..., "v2g": ...}``) and
returns, per ``(contrast, tau)`` cell, the paired-dV evidence of the
world-cluster registration (shared context, 2026-10-03):

* **Paired dV** with native as the reference policy, per
  ``(world, t0, decision_t, tau, contrast)``.  The dV quantiles are the
  COMONOTONE same-level differences ``q[treatment] - q[reference]`` --
  exactly what ``evalh.metrics.build_delta_pairs`` computes -- and the
  contrasts are the registration's policy-name conventions
  (``damping-native``, ``pi-native``, ``wall-native`` when the wall arm is
  present); row policies are canonicalised through
  ``evalh.idea_a.canon_action`` (so e.g. ``pi-external-u`` pairs as ``pi``).
* **Reused evalh.metrics primitives** (cited per the brief):
  ``build_delta_pairs`` (authoritative pairing; this loop recomputes the
  same pairs once more to obtain the raw dV quantiles for coverage and
  cross-checks every pair against it), ``crps_of_quantiles`` (paired-dV
  CRPS), ``sign_accuracy_ci`` (sign accuracy + world-bootstrap CI),
  ``world_bootstrap_ci`` (every CI here: whole worlds resampled with
  replacement), ``coverage`` (10-90 interval coverage).
  ``evalh.metrics.delta_metrics`` does NOT fit: it lacks coverage, the
  per-metric CIs and the A-minus-v2g naming this scorer must report, so the
  same primitives are applied directly to ``build_delta_pairs`` pairs.
* Per cell: ``crps_A`` / ``crps_v2g`` (mean per-pair paired-dV CRPS), the
  A minus v2g CRPS difference (mean + 90% CI; negative = A better), sign
  accuracy of each model's dV median on ``|dY| > sign_threshold`` pairs,
  10-90 coverage of each model's paired-dV interval, ``n_pairs`` /
  ``n_worlds``, and world-cluster bootstrap 90% CIs (``n_boot`` draws,
  resampling worlds, fixed ``seed`` = 20261001) for EVERY metric and the
  paired difference.
* ``masks`` (bool array aligned with ``rows``; ``True`` = keep) exclude
  cells BEFORE pairing; every exclusion is counted in the returned
  ``counts`` block (mask, tau filter, unknown policy after canonicalisation).
  Quantile arrays whose width does not match the rows are never stacked:
  a mismatching ``[N, 19]`` shape raises immediately.

**TTC<1.5 (A7) is deliberately NOT reported here.**  The A7 trajectory
metric lives on the unmerged ``feat/ttc15`` branch and does not exist in
this tree, and the WM-N1-A7-TTC15-20261003 SOURCE_REVIEW keeps that branch
on HOLD with P1 integration findings (2-D/3-D mask indexing on the evalh
rollout path, divergent TTC<3 / TTC<1.5 formulas, slot-count
mis-estimation).  ``evalh.protocol.OutcomeRow`` labels also carry only
``ttc_lt3_s``, never a TTC<1.5 label.  Per the E4 brief the extra outcome is
therefore skipped until the A7 function exists cleanly in this tree AND the
labels carry it.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from evalh.metrics import (CONTRASTS, build_delta_pairs, coverage,
                           crps_of_quantiles, level_index,
                           sign_accuracy_ci, world_bootstrap_ci)
from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS, OutcomeRow
from evalh.scorer import arm_contrasts as AC

#: All contrasts the metric loop reports: the registration's policy contrasts
#: plus the E2 per-arm contrasts (fixed/switching mix, wall pilot, trajectory)
#: from :mod:`evalh.scorer.arm_contrasts`.  Pooled group contrasts are filled
#: after pairing (see :func:`_pool_e2_groups`).
CONTRASTS_ALL = {**CONTRASTS, **AC.member_contrast_specs()}

SCHEMA = "wm_n1_cal_tune_scorer_metrics/1"

#: Paired-dV construction behind the ``dv_method`` flag.
#:
#: ``comonotone`` (historical default) reports ``q_a[level] - q_b[level]`` at
#: the SAME level.  That is the perfectly rank-correlated difference of the two
#: marginal predictive distributions and is only a *lower bound* on the
#: dispersion of the true difference; it produces far-too-narrow intervals
#: whenever the two arms' outcomes are not perfectly rank-correlated.
#:
#: ``independent`` samples each arm's marginal predictive distribution (via
#: inverse-CDF interpolation of the 19 level/quantile pairs) at a fixed
#: quasi-random probability grid and takes the quantiles of the sampled
#: difference.  It is the honest construction when only the two marginals are
#: available and the two simulation arms are treated as independent (which is
#: what the model predicts -- one marginal per arm, no joint).
DV_METHOD_COMONOTONE = "comonotone"
DV_METHOD_INDEPENDENT = "independent"
#: ``paired_head`` consumes paired-dV quantiles produced DIRECTLY by the
#: model's optional climatology-anchored paired-effect head (owner-approved
#: 2026-10-04), on the same 19-level grid, instead of deriving them from the
#: two per-arm marginals.  It is registered only for the direct-DV comparator
#: path (``dv_models`` given); the historical two-marginal path refuses it.
DV_METHOD_PAIRED_HEAD = "paired_head"
DV_METHODS = (DV_METHOD_COMONOTONE, DV_METHOD_INDEPENDENT,
              DV_METHOD_PAIRED_HEAD)
DEFAULT_DV_SAMPLES = 4096

#: Sign pairs qualify when ``|dY| > SIGN_THRESHOLD`` (registration uses the
#: TRAIN median on its own gate; the CAL-tune scorer reports the raw
#: direction, i.e. threshold 0.0).
SIGN_THRESHOLD = 0.0

_ALPHA = 0.10           # 90% world-bootstrap CIs


def physical_world(name) -> str:
    """Bootstrap cluster key: the world name without its trailing decision
    time (``w8481m2700p02t480`` -> ``w8481m2700p02``), so items that share a
    physical world (different t0) resample together (conservative CIs)."""
    import re
    return re.sub(r"t\d+$", "", str(name))

#: Canonical policy names that may enter a registered contrast; anything
#: else (after ``canon_action``) is excluded and counted, never substituted.
#: The E2 arm tokens are accepted too so the fixed/switching mix, wall pilot
#: and trajectory arms are scored, not silently dropped as unknown.
_KNOWN_POLICIES = frozenset(("native", "damping", "pi", "wall")) | frozenset(
    AC.ALL_ARMS)


def _canonical_rows(rows: Sequence[OutcomeRow]):
    """Canonicalise row policies.

    Returns ``(rows, kept_positions, n_unknown)`` where ``kept_positions``
    index the input list (unknown policies are dropped and counted).
    """
    from evalh.idea_a import canon_action

    out, positions = [], []
    n_unknown = 0
    for i, r in enumerate(rows):
        pol = canon_action(getattr(r, "policy", ""))
        if pol not in _KNOWN_POLICIES:
            n_unknown += 1
            continue
        if pol != r.policy:
            r = OutcomeRow(
                world=r.world, t0_s=r.t0_s, decision_t_s=r.decision_t_s,
                tau_s=r.tau_s, policy=pol, y_delay_s=r.y_delay_s,
                collision=r.collision, veh_steps=r.veh_steps,
                ttc_lt3_s=r.ttc_lt3_s, hard_brake_s=r.hard_brake_s,
                n_vehicles=r.n_vehicles, n_arrivals=r.n_arrivals,
                intervention=r.intervention, features=r.features)
        out.append(r)
        positions.append(i)
    return out, positions, n_unknown


def _pair_dv(rows, q_a, q_v2g, *, taus, levels) -> list:
    """One dict per contrast pair with the raw dV quantiles per model.

    Mirrors ``evalh.metrics.build_delta_pairs`` exactly (same key rounding,
    same comonotone ``q[a] - q[b]`` dV quantiles,     same skip-when-absent
    rule) but keeps the per-pair dV vectors so coverage can reuse them.
    """
    idx = {}
    for i, r in enumerate(rows):
        idx[(r.world, round(float(r.t0_s), 6),
             round(float(r.decision_t_s), 6), round(float(r.tau_s), 6),
             r.policy)] = i
    keys = sorted({(r.world, round(float(r.t0_s), 6),
                    round(float(r.decision_t_s), 6),
                    round(float(r.tau_s), 6)) for r in rows})
    pairs = []
    for (world, t0, dec, tau) in keys:
        if tau not in taus:
            continue
        for name, (pol_a, pol_b) in CONTRASTS_ALL.items():
            ia = idx.get((world, t0, dec, tau, pol_a))
            ib = idx.get((world, t0, dec, tau, pol_b))
            if ia is None or ib is None:
                continue
            dy = float(rows[ia].y_delay_s - rows[ib].y_delay_s)
            pairs.append(dict(
                world=world, t0_s=t0, decision_t_s=dec, tau_s=tau,
                contrast=name, dy=dy,
                # Raw per-arm marginal quantiles: the comonotone dV below is
                # the historical default; ``independent`` recomputes dV from
                # these marginals (see :func:`_apply_dv_method`).
                qa_a=np.asarray(q_a[ia], dtype=float),
                qa_b=np.asarray(q_a[ib], dtype=float),
                qa_v=np.asarray(q_v2g[ia], dtype=float),
                qa_vb=np.asarray(q_v2g[ib], dtype=float),
                dv_a=np.asarray(q_a[ia] - q_a[ib], dtype=float),
                dv_v2g=np.asarray(q_v2g[ia] - q_v2g[ib], dtype=float)))
    return pairs


def _pair_seed(seed, key) -> int:
    """A stable per-pair RNG seed so results do not depend on pair order."""
    if key is None:
        return int(seed)
    digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:8], "little")) % (2 ** 63)


def independent_difference_quantiles(q_a, q_b, *, levels=QUANTILE_LEVELS,
                                     n_samples: int = DEFAULT_DV_SAMPLES,
                                     seed: int = 0, key=None) -> np.ndarray:
    """Quantiles of ``Q_a - Q_b`` when only the two marginals are known.

    Each marginal is specified by its ``(level, quantile)`` pairs (the 19
    registered levels).  We sample each marginal by inverse-CDF (linear)
    interpolation on a deterministic stratified grid and return the empirical
    quantiles of the sampled difference.  Unlike the comonotone subtraction
    ``q_a - q_b`` this does not assume perfect rank correlation between the two
    arms; it is exact for independent arms and conservative (wider) when the
    arms are positively correlated.  A fixed ``seed`` (optionally mixed with a
    per-pair ``key``) makes the result reproducible.
    """
    lv = np.asarray(levels, dtype=float)
    if lv.ndim != 1 or lv.size < 2 or not np.all(np.diff(lv) > 0):
        raise ValueError("levels must be a strictly increasing 1-D grid")
    if int(n_samples) < 2:
        raise ValueError("n_samples must be at least 2")
    qa = np.sort(np.asarray(q_a, dtype=float))
    qb = np.sort(np.asarray(q_b, dtype=float))
    if qa.shape != lv.shape or qb.shape != lv.shape:
        raise ValueError(
            f"each marginal must carry {lv.size} quantiles; got "
            f"{qa.shape} and {qb.shape}")
    if not (np.isfinite(qa).all() and np.isfinite(qb).all()):
        raise ValueError("marginal quantiles must be finite")
    rng = np.random.default_rng(_pair_seed(seed, key))
    n = int(n_samples)
    # Independent draws for the two arms.  A shared (even jittered) grid would
    # reimpose comonotonicity elementwise and collapse the difference.
    u_a = rng.random(n)
    u_b = rng.random(n)
    sample = np.interp(u_a, lv, qa) - np.interp(u_b, lv, qb)
    return np.quantile(sample, lv)


def _apply_dv_method(pairs, *, method: str, n_samples: int, seed: int) -> None:
    """Rewrite every pair's ``dv_a`` / ``dv_v2g`` for the selected method.

    ``comonotone`` keeps the historical ``q_a - q_b`` vectors; anything else
    (only ``independent`` is registered) replaces them with the sampled
    independent-difference quantiles built from the stored marginals.
    """
    if method == DV_METHOD_COMONOTONE:
        return
    if method == DV_METHOD_PAIRED_HEAD:
        raise ValueError(
            "dv_method 'paired_head' supplies direct paired-dV quantiles and "
            "is not built from the per-arm marginals; pass a paired_head "
            "predictor on the direct-DV comparator path")
    if method not in DV_METHODS:
        raise ValueError(
            f"unknown dv_method {method!r}; registered methods are "
            f"{list(DV_METHODS)}")
    for p in pairs:
        key = (p["world"], p["t0_s"], p["decision_t_s"], p["tau_s"],
               p["contrast"])
        p["dv_a"] = independent_difference_quantiles(
            p["qa_a"], p["qa_b"], n_samples=n_samples, seed=seed,
            key=(key, "A"))
        p["dv_v2g"] = independent_difference_quantiles(
            p["qa_v"], p["qa_vb"], n_samples=n_samples, seed=seed,
            key=(key, "v2g"))
    return


def _crosscheck(pairs, rows, q_a, q_v2g, *, taus, levels) -> None:
    """Fail closed unless every legacy pair matches ``build_delta_pairs``.

    ``build_delta_pairs`` only knows the registration's policy contrasts, so
    the comparison is restricted to those pairs; the additive E2 arm contrasts
    are validated by their own synthetic tests.
    """
    legacy = [p for p in pairs if p["contrast"] in CONTRASTS]
    official = build_delta_pairs(rows, q_a, q_v2g, levels=levels, taus=taus)
    if len(official) != len(legacy):
        raise RuntimeError(
            f"pairing drift: {len(legacy)} local pairs vs "
            f"{len(official)} evalh.metrics pairs")
    for mine, theirs in zip(legacy, official):
        crps_a = float(crps_of_quantiles(mine["dv_a"], mine["dy"], levels))
        crps_v = float(crps_of_quantiles(mine["dv_v2g"], mine["dy"], levels))
        if (mine["world"] != theirs["world"]
                or mine["contrast"] != theirs["contrast"]
                or float(mine["tau_s"]) != float(theirs["tau_s"])
                or abs(mine["dy"] - theirs["dy"]) > 1e-12
                or abs(crps_a - theirs["variant_crps"]) > 1e-12
                or abs(crps_v - theirs["hybrid_crps"]) > 1e-12):
            raise RuntimeError(
                "pairing drift vs evalh.metrics.build_delta_pairs at "
                f"{theirs['world']} tau={theirs['tau_s']} "
                f"{theirs['contrast']}")


def _pool_e2_groups(grouped) -> None:
    """Add each E2 pooled group contrast as the union of its member pairs.

    Member pairs (e.g. ``fixed1-native``, ``fixed2-native``) are pooled into
    ``fixed_mix-native`` by concatenating their per-tau pair lists, so the
    pooled cell is the empirical distribution over all member observations.
    Groups with no present members are not created (no empty cells).
    """
    for group, arms in AC.ARM_GROUPS:
        pool: dict = {}
        for arm in arms:
            for tau, items in grouped.get(AC.member_contrast(arm), {}).items():
                pool.setdefault(tau, []).extend(items)
        if pool:
            grouped[AC.group_contrast(group)] = pool


def _cell(pairs_cell, *, levels, n_boot, seed) -> dict:
    """Aggregate one (contrast, tau) cell from its pair dicts."""
    worlds = [physical_world(p["world"]) for p in pairs_cell]
    dy = np.asarray([p["dy"] for p in pairs_cell], dtype=float)
    dv_a = np.stack([p["dv_a"] for p in pairs_cell])
    dv_v = np.stack([p["dv_v2g"] for p in pairs_cell])
    lo_i = level_index(0.10, levels)
    hi_i = level_index(0.90, levels)
    crps_a = np.asarray(crps_of_quantiles(dv_a, dy, levels), dtype=float)
    crps_v = np.asarray(crps_of_quantiles(dv_v, dy, levels), dtype=float)
    diff = crps_a - crps_v
    sign_a = np.sign(dv_a[:, level_index(0.5, levels)])
    sign_v = np.sign(dv_v[:, level_index(0.5, levels)])
    cov_a = (dv_a[:, lo_i] <= dy) & (dy <= dv_a[:, hi_i])
    cov_v = (dv_v[:, lo_i] <= dy) & (dy <= dv_v[:, hi_i])

    def _ci(values) -> list:
        lo, hi, _ = world_bootstrap_ci(values, worlds, n_boot=n_boot,
                                       alpha=_ALPHA, seed=seed)
        return [float(lo), float(hi)]

    sa, n_used, sa_lo, sa_hi = sign_accuracy_ci(
        sign_a, dy, worlds, SIGN_THRESHOLD, n_boot=n_boot, alpha=_ALPHA,
        seed=seed)
    sv, _, sv_lo, sv_hi = sign_accuracy_ci(
        sign_v, dy, worlds, SIGN_THRESHOLD, n_boot=n_boot, alpha=_ALPHA,
        seed=seed)
    return dict(
        n_pairs=int(dy.size), n_worlds=int(len(set(worlds))),
        crps_A=float(np.mean(crps_a)), crps_v2g=float(np.mean(crps_v)),
        crps_A_ci=_ci(crps_a), crps_v2g_ci=_ci(crps_v),
        diff_mean=float(np.mean(diff)), diff_ci=_ci(diff),
        sign_accuracy_A=(float("nan") if n_used == 0 else float(sa)),
        sign_accuracy_v2g=(float("nan") if n_used == 0 else float(sv)),
        sign_accuracy_A_ci=[float(sa_lo), float(sa_hi)],
        sign_accuracy_v2g_ci=[float(sv_lo), float(sv_hi)],
        sign_n_used=int(n_used), sign_threshold=float(SIGN_THRESHOLD),
        coverage_10_90_A=float(coverage(dv_a[:, lo_i], dv_a[:, hi_i], dy)),
        coverage_10_90_v2g=float(coverage(dv_v[:, lo_i], dv_v[:, hi_i], dy)),
        coverage_10_90_A_ci=_ci(cov_a.astype(float)),
        coverage_10_90_v2g_ci=_ci(cov_v.astype(float)))


def _score_two_models(rows: Sequence[OutcomeRow], q_by_model: dict, *,
                      taus=(30, 60, 120, 300, 570), masks=None,
                      n_boot: int = 2000, seed: int = 20261001,
                      dv_method: str = DV_METHOD_COMONOTONE,
                      dv_samples: int = DEFAULT_DV_SAMPLES,
                      dv_seed: int = 20261001) -> dict:
    """Score paired dV of model A against Part A v2g (see module docstring).

    ``rows`` are ``evalh.protocol.OutcomeRow``; ``q_by_model`` maps
    ``"A"``/``"v2g"`` to ``[len(rows), 19]`` delay quantiles.  ``masks`` is
    an optional keep-mask aligned with ``rows``; excluded cells are counted,
    never silently dropped, and unequal widths are never stacked.
    """
    rows = list(rows)
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("score needs at least one OutcomeRow")
    levels = np.asarray(QUANTILE_LEVELS, dtype=float)
    if set(q_by_model) != {"A", "v2g"}:
        raise ValueError(
            f"q_by_model keys must be exactly {{'A', 'v2g'}}; got "
            f"{sorted(q_by_model)!r}")
    q = {}
    for name, arr in q_by_model.items():
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n_rows, N_QUANTILES):
            raise ValueError(
                f"q_by_model[{name!r}] has shape {arr.shape}; never stack "
                f"unequal widths, expected ({n_rows}, {N_QUANTILES})")
        if not np.isfinite(arr).all():
            raise ValueError(f"q_by_model[{name!r}] contains nonfinite "
                             "quantiles")
        q[name] = arr
    if masks is None:
        keep = np.ones(n_rows, dtype=bool)
    else:
        keep = np.asarray(masks, dtype=bool)
        if keep.shape != (n_rows,):
            raise ValueError(
                f"masks has shape {keep.shape}; expected ({n_rows},) bools "
                "aligned with rows")
    n_excluded_by_mask = int((~keep).sum())
    rows_k = [r for r, k in zip(rows, keep) if k]
    tau_set = {round(float(t), 6) for t in taus}
    if not tau_set:
        raise ValueError("taus must name at least one reporting tau")
    rows_tau = [r for r in rows_k if round(float(r.tau_s), 6) in tau_set]
    n_excluded_by_tau = len(rows_k) - len(rows_tau)
    rows_c, kept_idx, n_unknown = _canonical_rows(rows_tau)
    if not rows_c:
        raise ValueError(
            "no canonical policy rows remain after mask/tau filters; "
            f"excluded_by_mask={n_excluded_by_mask}, "
            f"excluded_by_tau={n_excluded_by_tau}, "
            f"unknown_policy={n_unknown}")
    q_a = q["A"][kept_idx]
    q_v2g = q["v2g"][kept_idx]
    pairs = _pair_dv(rows_c, q_a, q_v2g, taus=tau_set, levels=levels)
    if not pairs:
        raise ValueError(
            "no native-referenced contrast pairs on the requested taus; "
            "nothing to score")
    _crosscheck(pairs, rows_c, q_a, q_v2g, taus=tau_set, levels=levels)
    _apply_dv_method(pairs, method=dv_method, n_samples=dv_samples,
                     seed=dv_seed)

    cells: dict = {}
    grouped: dict = {}
    for p in pairs:
        grouped.setdefault(p["contrast"], {}).setdefault(
            float(p["tau_s"]), []).append(p)
    _pool_e2_groups(grouped)
    for contrast in sorted(grouped):
        cells[contrast] = {}
        for tau in sorted(grouped[contrast]):
            cells[contrast][f"{tau:g}"] = _cell(
                grouped[contrast][tau], levels=levels, n_boot=n_boot,
                seed=seed)
    by_policy: dict = {}
    for r in rows_c:
        by_policy[r.policy] = by_policy.get(r.policy, 0) + 1
    return dict(
        schema=SCHEMA,
        seed=int(seed), n_boot=int(n_boot), alpha=_ALPHA,
        taus=[float(t) for t in sorted(tau_set)],
        models=["A", "v2g"],
        reference_policy="native",
        sign_threshold=float(SIGN_THRESHOLD),
        dv_method=dv_method, dv_samples=int(dv_samples),
        counts=dict(
            n_rows=n_rows,
            n_rows_used=len(rows_c),
            n_excluded_by_mask=n_excluded_by_mask,
            n_excluded_by_tau=n_excluded_by_tau,
            n_excluded_unknown_policy=n_unknown,
            n_pairs=len(pairs),
            n_rows_by_policy=by_policy),
        contrasts_present=sorted(cells),
        cells=cells)


# ------------------------------------------------- direct-dV comparator path --
SCHEMA_BASELINES = "wm_n1_cal_tune_scorer_metrics_baselines/1"


def _ci(values, worlds, *, n_boot, seed) -> list:
    lo, hi, _ = world_bootstrap_ci(values, worlds, n_boot=n_boot,
                                   alpha=_ALPHA, seed=seed)
    return [float(lo), float(hi)]


def _cell_multi(pairs_cell, model_names, *, levels, n_boot, seed,
                sign_defined) -> dict:
    """One (contrast, tau) cell for model A plus direct-dV comparator(s)."""
    worlds = [physical_world(p["world"]) for p in pairs_cell]
    dy = np.asarray([p["dy"] for p in pairs_cell], dtype=float)
    dvs = {name: np.stack([p["dvs"][name] for p in pairs_cell])
           for name in model_names}
    lo_i = level_index(0.10, levels)
    hi_i = level_index(0.90, levels)
    crps = {name: np.asarray(crps_of_quantiles(dvs[name], dy, levels),
                             dtype=float)
            for name in model_names}
    cov_bool = {name: (dvs[name][:, lo_i] <= dy) & (dy <= dvs[name][:, hi_i])
                for name in model_names}
    out = dict(
        n_pairs=int(dy.size), n_worlds=int(len(set(worlds))),
        sign_threshold=float(SIGN_THRESHOLD))
    n_used = 0
    sa = float("nan")
    for name in model_names:
        out[f"crps_{name}"] = float(np.mean(crps[name]))
        out[f"crps_{name}_ci"] = _ci(crps[name], worlds, n_boot=n_boot,
                                     seed=seed)
        out[f"coverage_10_90_{name}"] = float(coverage(
            dvs[name][:, lo_i], dvs[name][:, hi_i], dy))
        out[f"coverage_10_90_{name}_ci"] = _ci(cov_bool[name].astype(float),
                                               worlds, n_boot=n_boot, seed=seed)
        if sign_defined.get(name, True):
            acc, used, lo, hi = sign_accuracy_ci(
                np.sign(dvs[name][:, level_index(0.5, levels)]), dy, worlds,
                SIGN_THRESHOLD, n_boot=n_boot, alpha=_ALPHA, seed=seed)
            out[f"sign_accuracy_{name}"] = (
                float("nan") if used == 0 else float(acc))
            out[f"sign_accuracy_{name}_ci"] = [float(lo), float(hi)]
            out[f"sign_accuracy_{name}_defined"] = True
            if name == "A":
                n_used, sa = int(used), float(acc)
        else:
            out[f"sign_accuracy_{name}"] = float("nan")
            out[f"sign_accuracy_{name}_ci"] = [float("nan"), float("nan")]
            out[f"sign_accuracy_{name}_defined"] = False
    out["sign_n_used"] = int(n_used)
    out["sign_accuracy_A"] = float("nan") if n_used == 0 else float(sa)
    for name in model_names:
        if name == "A":
            continue
        diff = crps["A"] - crps[name]
        out[f"diff_{name}_mean"] = float(np.mean(diff))
        out[f"diff_{name}_ci"] = _ci(diff, worlds, n_boot=n_boot, seed=seed)
    return out


def _score_with_dv_baselines(rows, q_by_model, dv_models, *, taus, masks,
                             n_boot, seed,
                             dv_method: str = DV_METHOD_COMONOTONE,
                             dv_samples: int = DEFAULT_DV_SAMPLES,
                             dv_seed: int = 20261001,
                             paired_head=None) -> dict:
    """Score model A (per-row quantiles) against direct-dV baselines."""
    rows = list(rows)
    n_rows = len(rows)
    if n_rows == 0:
        raise ValueError("score needs at least one OutcomeRow")
    levels = np.asarray(QUANTILE_LEVELS, dtype=float)
    if not isinstance(q_by_model, dict) or set(q_by_model) != {"A"}:
        raise ValueError(
            "q_by_model must be exactly {'A': ...} when dv_models is given; "
            f"got {sorted(q_by_model) if isinstance(q_by_model, dict) else q_by_model!r}")
    arr = np.asarray(q_by_model["A"], dtype=float)
    if arr.shape != (n_rows, N_QUANTILES):
        raise ValueError(
            "q_by_model['A'] has shape "
            f"{arr.shape}; never stack unequal widths, expected "
            f"({n_rows}, {N_QUANTILES})")
    if not np.isfinite(arr).all():
        raise ValueError("q_by_model['A'] contains nonfinite quantiles")

    if masks is None:
        keep = np.ones(n_rows, dtype=bool)
    else:
        keep = np.asarray(masks, dtype=bool)
        if keep.shape != (n_rows,):
            raise ValueError(
                f"masks has shape {keep.shape}; expected ({n_rows},) bools")
    n_excluded_by_mask = int((~keep).sum())
    rows_k = [r for r, k in zip(rows, keep) if k]
    tau_set = {round(float(t), 6) for t in taus}
    if not tau_set:
        raise ValueError("taus must name at least one reporting tau")
    rows_tau = [r for r in rows_k if round(float(r.tau_s), 6) in tau_set]
    n_excluded_by_tau = len(rows_k) - len(rows_tau)
    rows_c, kept_idx, n_unknown = _canonical_rows(rows_tau)
    if not rows_c:
        raise ValueError(
            "no canonical policy rows remain after mask/tau filters; "
            f"excluded_by_mask={n_excluded_by_mask}, "
            f"excluded_by_tau={n_excluded_by_tau}, unknown_policy={n_unknown}")
    q_a = arr[kept_idx]
    pairs = _pair_dv(rows_c, q_a, q_a, taus=tau_set, levels=levels)
    if not pairs:
        raise ValueError(
            "no native-referenced contrast pairs on the requested taus; "
            "nothing to score")
    _crosscheck(pairs, rows_c, q_a, q_a, taus=tau_set, levels=levels)
    pair_infos = [dict(world=p["world"], t0_s=p["t0_s"],
                       decision_t_s=p["decision_t_s"], tau_s=p["tau_s"],
                       contrast=p["contrast"]) for p in pairs]
    if dv_method == DV_METHOD_PAIRED_HEAD:
        if paired_head is None:
            raise ValueError(
                "dv_method 'paired_head' requires a paired_head predictor "
                "implementing dv_quantiles(pair_infos)")
        head_dv = np.asarray(paired_head.dv_quantiles(pair_infos), dtype=float)
        if head_dv.shape != (len(pairs), N_QUANTILES):
            raise ValueError(
                f"paired_head returned {head_dv.shape}; expected "
                f"({len(pairs)}, {N_QUANTILES})")
        if not np.isfinite(head_dv).all():
            raise ValueError("paired_head returned nonfinite quantiles")
        for p, row in zip(pairs, head_dv):
            p["dv_a"] = np.asarray(row, dtype=float)
    else:
        _apply_dv_method(pairs, method=dv_method, n_samples=dv_samples,
                         seed=dv_seed)
    for p in pairs:
        p["dvs"] = {"A": np.asarray(p["dv_a"], dtype=float)}
    model_names = ["A"]
    sign_defined: dict = {"A": True}
    seen = {"A"}
    for key, model in dv_models.items():
        name = str(getattr(model, "name", key))
        if name in seen:
            raise ValueError(f"duplicate dv model name {name!r}")
        dv = np.asarray(model.dv_quantiles(pair_infos), dtype=float)
        if dv.shape != (len(pairs), N_QUANTILES):
            raise ValueError(
                f"dv model {name!r} returned {dv.shape}; expected "
                f"({len(pairs)}, {N_QUANTILES})")
        if not np.isfinite(dv).all():
            raise ValueError(f"dv model {name!r} returned nonfinite quantiles")
        for p, row in zip(pairs, dv):
            p.setdefault("dvs", {})[name] = row
        model_names.append(name)
        sign_defined[name] = bool(getattr(model, "sign_defined", True))
        seen.add(name)

    cells: dict = {}
    grouped: dict = {}
    for p in pairs:
        grouped.setdefault(p["contrast"], {}).setdefault(
            float(p["tau_s"]), []).append(p)
    _pool_e2_groups(grouped)
    for contrast in sorted(grouped):
        cells[contrast] = {}
        for tau in sorted(grouped[contrast]):
            cells[contrast][f"{tau:g}"] = _cell_multi(
                grouped[contrast][tau], model_names, levels=levels,
                n_boot=n_boot, seed=seed, sign_defined=sign_defined)
    by_policy: dict = {}
    for r in rows_c:
        by_policy[r.policy] = by_policy.get(r.policy, 0) + 1
    return dict(
        schema=SCHEMA_BASELINES,
        seed=int(seed), n_boot=int(n_boot), alpha=_ALPHA,
        taus=[float(t) for t in sorted(tau_set)],
        models=model_names,
        reference_policy="native",
        sign_threshold=float(SIGN_THRESHOLD),
        dv_method=dv_method, dv_samples=int(dv_samples),
        comparators=[name for name in model_names if name != "A"],
        counts=dict(
            n_rows=n_rows,
            n_rows_used=len(rows_c),
            n_excluded_by_mask=n_excluded_by_mask,
            n_excluded_by_tau=n_excluded_by_tau,
            n_excluded_unknown_policy=n_unknown,
            n_pairs=len(pairs),
            n_rows_by_policy=by_policy),
        contrasts_present=sorted(cells),
        cells=cells)


def score(rows: Sequence[OutcomeRow], q_by_model: dict | None = None, *,
          dv_models: dict | None = None, taus=(30, 60, 120, 300, 570),
          masks=None, n_boot: int = 2000, seed: int = 20261001,
          dv_method: str = DV_METHOD_COMONOTONE,
          dv_samples: int = DEFAULT_DV_SAMPLES,
          dv_seed: int = 20261001, paired_head=None) -> dict:
    """Score paired dV.

    Backward-compatible two-per-row-model path (``dv_models`` omitted):
    ``q_by_model`` maps ``"A"``/``"v2g"`` to ``[N, 19]`` delay quantiles, as
    before.  Comparator path (``dv_models`` given): ``q_by_model`` maps only
    ``"A"`` and ``dv_models`` maps names to objects exposing
    ``dv_quantiles(pairs)`` (see :mod:`evalh.scorer.baselines`), which predict
    paired-dV quantiles DIRECTLY.

    ``dv_method`` selects how per-arm marginal predictors (A and v2g) build
    their paired-dV distribution: ``"comonotone"`` (default, historical
    same-level subtraction), ``"independent"`` (sample each marginal and take
    the difference quantiles), or ``"paired_head"`` (registered only for the
    direct-DV comparator path: model A's paired-dV quantiles come DIRECTLY from
    the optional climatology-anchored paired-effect head, passed as
    ``paired_head``).  Direct-dV comparator models are unaffected.
    """
    if dv_models is None:
        if dv_method == DV_METHOD_PAIRED_HEAD:
            raise ValueError(
                "dv_method 'paired_head' is registered only for the direct-DV "
                "comparator path (dv_models given)")
        return _score_two_models(rows, q_by_model, taus=taus, masks=masks,
                                 n_boot=n_boot, seed=seed,
                                 dv_method=dv_method, dv_samples=dv_samples,
                                 dv_seed=dv_seed)
    return _score_with_dv_baselines(rows, q_by_model, dv_models, taus=taus,
                                    masks=masks, n_boot=n_boot, seed=seed,
                                    dv_method=dv_method,
                                    dv_samples=dv_samples, dv_seed=dv_seed,
                                    paired_head=paired_head)
