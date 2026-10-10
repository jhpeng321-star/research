"""v0.4b acceptance metrics (registration s.6), fixed BEFORE running.

Pure numpy/pandas over the wide pairs frame so every rule is unit-
testable on synthetic truth tables.  The distribution primitives are
imported READ-ONLY from partB_v0.acceptance_v04a -- level_grid /
quantile_samples / crps_samples (the K = 40 midpoint grid, piecewise-
linear quantile function with flat tails) give the model and both
baselines the IDENTICAL CRPS code path, and the world bootstrap is the
same whole-world resampling.

  A1 calibration:  at every tau, CRPS(model dY) < CRPS(zero-contrast);
        AND the overall (all tau pooled) 10-90 % coverage of the
        conformal dY band lies in [0.75, 0.85].
  A3 contrast skill, contrast wall-native, tau in {60, 120, 300}:
        (i)  sign accuracy of sign(median predicted dY) vs sign(dY)
             over pairs with |dY| above the TRAIN median |dY| of that
             (contrast, tau) (near-ties don't count); the world-
             bootstrap 90 % CI lower bound must be > 0.5
             (N_BOOT 2000, seed 20261001);
        (ii) the model dY CRPS must be below BOTH the zero-contrast
             baseline AND the TRAIN-cell-mean baseline (TRAIN mean dY
             per demand x penetration cell as a point mass; the cell
             key is a baseline label, never a model input);
        A3 passes if (i) AND (ii) hold at >= 2 of the 3 gating taus.
  VERDICT: ACCEPT iff A1 AND A3; PARTIAL iff exactly one; FAIL
        otherwise.

Always reported (s.6): per-arm / per-speed-factor / per-timing
breakdowns, aux contrasts' own A3-style numbers (reported only), the
wall-only ablation verdict, TRAIN-OOB versions of A1/A3 (labelled), and
the s.12.1 item 3 STRATIFIED reporting by action duration/onset
(<= 60 s, 60-240 s, whole window) x tau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401  (sys.path shims; read-only imports)
from . import reg
from partB_v0.acceptance_v04a import (crps_samples, level_grid,  # noqa: E402
                                      quantile_samples,
                                      world_bootstrap_ci)

K_LEVELS = 40                     # identical grid as v0.4a
GATING_CONTRAST = reg.HEADLINE_CONTRAST


# --------------------------------------------------------------- baselines --

def train_references_v04b(train_pairs: pd.DataFrame) -> dict:
    """TRAIN-side references per (contrast, tau): median |dY| (the sign
    threshold), mean dY per demand x penetration cell (point mass) and
    the pooled mean fallback -- the v0.4a semantics over the wide frame
    (delay head)."""
    ref: dict = {}
    for tau in reg.TAUS:
        col = f"dy_{reg.GATING_HEAD}_{int(tau)}"
        if col not in train_pairs.columns:
            continue                     # no rows at this tau
        d = train_pairs[np.isclose(train_pairs["tau"], tau)].dropna(
            subset=[col])
        for c, g in d.groupby("contrast"):
            cells = {(int(dd), round(float(pp), 6)): float(vv)
                     for (dd, pp), vv in
                     g.groupby(["demand", "penetration"])[col].mean()
                     .items()}
            ref[(str(c), float(tau))] = dict(
                median_abs=float(np.median(np.abs(g[col].to_numpy()))),
                cell_mean=cells, pooled_mean=float(g[col].mean()),
                n=int(len(g)))
    return ref


def cell_mean_of(pairs: pd.DataFrame, ref: dict, tau: float) -> np.ndarray:
    """Per-pair TRAIN-cell-mean point mass (pooled fallback)."""
    out = np.zeros(len(pairs))
    for i, (dem, pen) in enumerate(zip(pairs["demand"],
                                       pairs["penetration"])):
        rr = ref.get((str(pairs["contrast"].iloc[0]), float(tau)))
        if rr is None:
            continue
        out[i] = rr["cell_mean"].get((int(dem), round(float(pen), 6)),
                                     rr["pooled_mean"])
    return out


def crps_zero(dY: np.ndarray) -> np.ndarray:
    """CRPS of the zero-contrast baseline (point mass at 0) = |dY|."""
    return np.abs(np.asarray(dY, float))


def crps_cell(dY: np.ndarray, cellm: np.ndarray) -> np.ndarray:
    """CRPS of the TRAIN-cell-mean baseline (point mass)."""
    return np.abs(np.asarray(dY, float) - np.asarray(cellm, float))


def crps_model(quantile_preds: dict, dY: np.ndarray,
               k: int = K_LEVELS) -> np.ndarray:
    """Per-row CRPS of the 7-quantile predictive on the SAME K=40
    midpoint grid as v0.4a (piecewise-linear, flat tails)."""
    u = level_grid(k)
    s = quantile_samples(sorted(quantile_preds), quantile_preds, u)
    return crps_samples(s, np.asarray(dY, float))


# ---------------------------------------------------------------- A1 / A3 --

def a1_records(pairs: pd.DataFrame, pred_fn, head=reg.GATING_HEAD
               ) -> dict:
    """A1 over the pairs frame (rows of ALL taus).  pred_fn(pairs_tau)
    -> {q: [n]} calibrated quantile predictions for that tau's rows."""
    per_tau, cov_num = [], []
    for tau in reg.TAUS:
        col = f"dy_{head}_{int(tau)}"
        if col not in pairs.columns:
            continue                     # no rows at this tau
        g = pairs[np.isclose(pairs["tau"], tau)].dropna(subset=[col])
        if not len(g):
            continue
        dY = g[f"dy_{head}_{int(tau)}"].to_numpy(float)
        preds = pred_fn(g)
        cr = crps_model(preds, dY)
        per_tau.append(dict(tau=float(tau), n=int(len(g)),
                            crps_model=float(cr.mean()),
                            crps_zero=float(crps_zero(dY).mean())))
        q10, q90 = preds[0.1], preds[0.9]
        cov_num.append(((dY >= q10) & (dY <= q90)).astype(float))
    cov_overall = float(np.concatenate(cov_num).mean()) if cov_num \
        else float("nan")
    beats = {f"{r['tau']:.0f}": bool(r["crps_model"] < r["crps_zero"])
             for r in per_tau}
    a1 = dict(per_tau=per_tau, coverage_overall=cov_overall,
              band=list(reg.COVERAGE_BAND),
              coverage_in_band=bool(reg.COVERAGE_BAND[0] <= cov_overall
                                    <= reg.COVERAGE_BAND[1]),
              crps_beats_zero=beats,
              all_taus_beat_zero=bool(beats and all(beats.values())))
    a1["passes"] = bool(a1["all_taus_beat_zero"] and a1["coverage_in_band"])
    return a1


def a3_records(pairs: pd.DataFrame, tau: float, pred_fn, ref: dict,
               head=reg.GATING_HEAD, n_boot=reg.N_BOOT,
               seed=reg.BOOT_SEED) -> list[dict]:
    """A3 records for one tau over the pairs frame (CAL pairs at
    scoring time; TRAIN-OOB pairs for the labelled TRAIN version)."""
    g = pairs[np.isclose(pairs["tau"], tau)].dropna(
        subset=[f"dy_{head}_{int(tau)}"])
    out = []
    for c, gg in g.groupby("contrast", sort=True):
        gg = gg.reset_index(drop=True)
        dY = gg[f"dy_{head}_{int(tau)}"].to_numpy(float)
        preds = pred_fn(gg)
        med = preds[0.5]
        rr = ref.get((str(c), float(tau)))
        thr = rr["median_abs"] if rr else float("nan")
        elig = np.abs(dY) > thr if rr else np.zeros(len(dY), bool)
        correct = (np.sign(med) == np.sign(dY)).astype(float)
        boot = world_bootstrap_ci(correct[elig], gg["key"].to_numpy()[elig],
                                  n_boot=n_boot, seed=seed)
        crm = crps_model(preds, dY)
        crz = crps_zero(dY)
        crc = crps_cell(dY, cell_mean_of(gg, ref, tau))
        rec = dict(
            contrast=str(c), tau=float(tau), head=head,
            n_pairs=int(len(gg)), n_worlds=int(gg["key"].nunique()),
            train_median_abs_dY=thr, n_eligible=int(elig.sum()),
            sign_acc=boot["acc"], sign_ci_lo=boot["lo"],
            sign_ci_hi=boot["hi"], sign_n_worlds=boot["n_worlds"],
            crps_model=float(crm.mean()), crps_zero=float(crz.mean()),
            crps_cell_mean=float(crc.mean()),
            dY_mean=float(dY.mean()))
        rec["pass_i"] = bool(np.isfinite(rec["sign_ci_lo"])
                             and rec["sign_ci_lo"] > 0.5)
        rec["pass_ii"] = bool(rec["crps_model"] < rec["crps_zero"]
                              and rec["crps_model"] < rec["crps_cell_mean"])
        rec["pass_tau"] = bool(rec["pass_i"] and rec["pass_ii"])
        out.append(rec)
    return out


def a3_verdict(records: list[dict], contrast=GATING_CONTRAST) -> dict:
    """A3 for one contrast: (i) AND (ii) at >= 2 of the 3 gating taus."""
    rs = [r for r in records if r["contrast"] == contrast
          and r["tau"] in reg.TAUS_A3]
    n_pass = sum(1 for r in rs if r["pass_tau"])
    return dict(contrast=contrast, n_taus_evaluated=len(rs),
                n_taus_pass=int(n_pass), passes=bool(n_pass >= 2),
                per_tau={f"{r['tau']:.0f}": dict(pass_i=r["pass_i"],
                                                pass_ii=r["pass_ii"])
                         for r in rs})


def verdict(a1: dict, a3: dict) -> dict:
    """ACCEPT iff A1 AND A3; PARTIAL iff exactly one; FAIL otherwise."""
    n = int(bool(a1["passes"])) + int(bool(a3["passes"]))
    v = "ACCEPT" if n == 2 else ("PARTIAL" if n == 1 else "FAIL")
    return dict(verdict=v, a1=bool(a1["passes"]), a3=bool(a3["passes"]))


# -------------------------------------------------------------- strata ----

def stratified_records(pairs: pd.DataFrame, tau: float, pred_fn,
                       ref: dict, head=reg.GATING_HEAD) -> list[dict]:
    """s.12.1 item 3: sign accuracy + CRPS vs the baselines REPORTED
    STRATIFIED by action duration/onset stratum x tau (plus per-arm and
    per speed-factor / timing breakdowns of the wall contrast)."""
    g = pairs[np.isclose(pairs["tau"], tau)].dropna(
        subset=[f"dy_{head}_{int(tau)}"])
    out = []
    for stratum, gg in g.groupby("dur_stratum", sort=True):
        out.append(_stratum_rec(f"dur:{stratum}", gg, tau, pred_fn, ref,
                                head))
    for onb, gg in g.groupby("onset_bin", sort=True):
        out.append(_stratum_rec(f"onset:{onb}", gg, tau, pred_fn, ref,
                                head))
    for arm, gg in g.groupby("arm_source", sort=True):
        out.append(_stratum_rec(f"arm:{arm}", gg, tau, pred_fn, ref, head))
    for fam, gg in g[g["contrast"] == GATING_CONTRAST].groupby(
            "family", sort=True):
        out.append(_stratum_rec(f"wall_family:{fam}", gg, tau, pred_fn,
                                ref, head))
    return out


def _stratum_rec(label, gg, tau, pred_fn, ref, head) -> dict:
    gg = gg.reset_index(drop=True)
    dY = gg[f"dy_{head}_{int(tau)}"].to_numpy(float)
    preds = pred_fn(gg)
    rr = ref.get((str(gg["contrast"].iloc[0]), float(tau)))
    thr = rr["median_abs"] if rr else float("nan")
    elig = np.abs(dY) > thr if rr else np.zeros(len(dY), bool)
    correct = (np.sign(preds[0.5]) == np.sign(dY)).astype(float)
    boot = world_bootstrap_ci(correct[elig], gg["key"].to_numpy()[elig])
    return dict(
        stratum=str(label), tau=float(tau), n=int(len(gg)),
        n_eligible=int(elig.sum()), sign_acc=boot["acc"],
        sign_ci_lo=boot["lo"], sign_ci_hi=boot["hi"],
        crps_model=float(crps_model(preds, dY).mean()),
        crps_zero=float(crps_zero(dY).mean()),
        crps_cell_mean=float(crps_cell(
            dY, cell_mean_of(gg, ref, tau)).mean()),
        dY_mean=float(dY.mean()))
