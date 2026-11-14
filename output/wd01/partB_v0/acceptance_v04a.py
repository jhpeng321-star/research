"""WD-01-PB-V04A acceptance metrics (registration section 6), fixed
BEFORE running.  Pure numpy / pandas so every rule is unit-testable on
synthetic data (tests/test_acceptance_v04a.py).

  A1 calibration:  at every tau, CRPS(model) < CRPS(climatology) AND the
        overall (all tau pooled) 10-90 % coverage lies in [0.75, 0.85].
  A2 vs ridge:     per tau the HARD comparison (strictly better
        MAE(median) AND CRPS than ridge; a tie is a TIE).  GATING at
        tau = 30, 60, 120; tau = 300, 570 reported with their verdict.
  A3 contrast skill, for each gating contrast c in {damping-native,
        pi-native} at tau in {60, 120, 300}:
          (i)  sign accuracy of sign(E[dV]) vs sign(dY) over CAL pairs
               with |dY| above the TRAIN median |dY| of that (c, tau);
               the world-bootstrap 90 % CI LOWER bound must be > 0.5;
          (ii) CRPS of dV below BOTH the zero-contrast baseline (point
               mass at 0) and the TRAIN-cell-mean baseline (TRAIN mean dY
               per demand x penetration cell, point mass; the cell key is
               a label for the baseline only, never a model input);
        c passes A3 if (i) AND (ii) hold at >= 2 of the 3 tau.
  VERDICT: ACCEPT iff A1 AND A2-gating AND A3 for BOTH gating contrasts;
        PARTIAL iff A1 AND A2-gating AND A3 for exactly ONE contrast;
        FAIL otherwise.

dV comes from PAIRED predictions of the same model (no delta head):
  point:  dV = V_med(s, tau, a) - V_med(s, tau, native);
  distribution: INDEPENDENT-quantile pairing (dV = Q_a(u_i) - Q_n(u_j)
        over all pairs of a K-level midpoint grid; the predictive of the
        difference if the two continuation outcomes were independent);
        the COMONOTONE pairing (Q_a(u_i) - Q_n(u_i)) is the reported
        sensitivity.  Quantile functions are piecewise-linear between the
        fitted levels with flat tails (the same tail treatment as the
        CRPS-from-quantiles approximation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAUS_A2_GATING = (30.0, 60.0, 120.0)
TAUS_A3 = (60.0, 120.0, 300.0)
GATING_CONTRASTS = ("damping-native", "pi-native")
REPORT_CONTRASTS = ("damping-native", "pi-native", "switching-native",
                    "fixed-native")
COVERAGE_BAND = (0.75, 0.85)
K_LEVELS = 40
N_BOOT = 2000
BOOT_SEED = 20260930
BOOT_CI = 0.90

# contrast -> (arm sources, native reference source)
CONTRAST_SOURCES = {
    "damping-native": (("base",), "native"),
    "pi-native": (("pi",), "native"),
    "switching-native": (("switching1", "switching2", "switching3"),
                         "native"),
    "fixed-native": (("fixed1", "fixed2"), "native"),
}


# ---------------------------------------------------------- distributions --

def level_grid(k: int = K_LEVELS) -> np.ndarray:
    """K equally weighted midpoint quantile levels."""
    return (np.arange(int(k)) + 0.5) / int(k)


def quantile_samples(qs, preds: dict, u: np.ndarray) -> np.ndarray:
    """[n, K] quantile-function values at levels u from the fitted grid
    (piecewise-linear, flat tails)."""
    qs = sorted(float(q) for q in qs)
    Q = np.column_stack([np.asarray(preds[q], float) for q in qs])
    Q = np.maximum.accumulate(Q, axis=1)          # monotone in q
    out = np.empty((Q.shape[0], len(u)))
    for i in range(Q.shape[0]):
        out[i] = np.interp(u, qs, Q[i])
    return out


def delta_samples(sa: np.ndarray, sn: np.ndarray, pairing: str) -> np.ndarray:
    """[n, M] equally weighted samples of dV = A - N.
    pairing 'independent': all K x K pairs (M = K^2); 'comonotone':
    same-level pairs (M = K)."""
    if pairing == "comonotone":
        return sa - sn
    if pairing != "independent":
        raise ValueError(f"unknown pairing {pairing!r}")
    n, k = sa.shape
    return (sa[:, :, None] - sn[:, None, :]).reshape(n, k * k)


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row CRPS of an equally weighted sample set:
    E|X-y| - 0.5 E|X-X'|, exact for the empirical distribution."""
    S = np.sort(np.asarray(samples, float), axis=1)
    n, m = S.shape
    y = np.asarray(y, float)
    t1 = np.abs(S - y[:, None]).mean(axis=1)
    w = 2.0 * np.arange(1, m + 1) - m - 1.0
    t2 = (S * w[None, :]).sum(axis=1) / (m * m)     # 0.5 * E|X-X'|
    return t1 - t2


# ------------------------------------------------------------- bootstrap --

def world_bootstrap_ci(correct: np.ndarray, worlds, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED, ci: float = BOOT_CI) -> dict:
    """World-grouped bootstrap of the mean of a 0/1 vector: whole worlds
    (all their pairs together) are resampled with replacement.  Returns
    point estimate and the two-sided `ci` interval (lo, hi)."""
    correct = np.asarray(correct, float)
    worlds = np.asarray(worlds).astype(str)
    n = len(correct)
    if n == 0:
        return dict(n=0, acc=float("nan"), lo=float("nan"),
                    hi=float("nan"), n_worlds=0)
    uw, inv = np.unique(worlds, return_inverse=True)
    k = np.bincount(inv, weights=correct, minlength=len(uw))
    m = np.bincount(inv, minlength=len(uw)).astype(float)
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot)
    ok = 0
    for b in range(n_boot):
        draw = rng.integers(0, len(uw), size=len(uw))
        cnt = np.bincount(draw, minlength=len(uw))
        den = float((cnt * m).sum())
        if den <= 0:
            continue
        accs[ok] = float((cnt * k).sum()) / den
        ok += 1
    accs = accs[:ok]
    a = (1.0 - ci) / 2.0
    return dict(n=int(n), n_worlds=int(len(uw)),
                acc=float(correct.mean()),
                lo=float(np.quantile(accs, a)),
                hi=float(np.quantile(accs, 1.0 - a)))


# -------------------------------------------------------------- pairing ---

def build_pairs(rows: pd.DataFrame, splits=("CALIBRATION",),
                contrasts=REPORT_CONTRASTS) -> pd.DataFrame:
    """Contrast pairs at t0 from label rows.

    A pair = (chunk, world, tau) with a label row at t == t0 for the arm
    source AND for the native reference source (the RESTORED native arm of
    chunk 4 / the J3 true-native arm of chunks 2-3), i.e. the same saved
    state.  dY = Y(arm) - Y(native) (negative = the arm lowers delay).
    Auxiliary rows never form pairs."""
    r = rows
    if "aux" in r.columns:
        r = r[~r["aux"].to_numpy(bool)]
    r = r[r["split"].isin(splits) & np.isclose(r["t"], r["t0_s"])]
    keep = ["chunk_id", "world", "tau", "source", "total_delay_ref_s",
            "demand_veh_h", "penetration", "split"]
    r = r.reset_index().rename(columns={"index": "row"})[["row"] + keep]
    out = []
    nat = r[r["source"] == "native"]
    for c in contrasts:
        srcs, ref = CONTRAST_SOURCES[c]
        arm = r[r["source"].isin(srcs)]
        m = arm.merge(nat, on=["chunk_id", "world", "tau"],
                      suffixes=("_a", "_n"))
        for rec in m.itertuples(index=False):
            out.append(dict(
                contrast=c, chunk=rec.chunk_id, world=rec.world,
                key=f"{rec.chunk_id}/{rec.world}", tau=float(rec.tau),
                arm_source=rec.source_a, split=rec.split_a,
                row_a=int(rec.row_a), row_n=int(rec.row_n),
                y_a=float(rec.total_delay_ref_s_a),
                y_n=float(rec.total_delay_ref_s_n),
                dY=float(rec.total_delay_ref_s_a - rec.total_delay_ref_s_n),
                demand=int(rec.demand_veh_h_a),
                penetration=float(rec.penetration_a)))
    cols = ["contrast", "chunk", "world", "key", "tau", "arm_source",
            "split", "row_a", "row_n", "y_a", "y_n", "dY", "demand",
            "penetration"]
    return pd.DataFrame(out, columns=cols)


def train_references(train_pairs: pd.DataFrame) -> dict:
    """TRAIN-side references per (contrast, tau): median |dY|, mean dY per
    demand x penetration cell, pooled mean dY (fallback for a CAL cell
    with no TRAIN pair)."""
    ref: dict = {}
    for (c, tau), g in train_pairs.groupby(["contrast", "tau"]):
        cells = {(int(d), round(float(p), 6)): float(v) for (d, p), v in
                 g.groupby(["demand", "penetration"])["dY"].mean().items()}
        ref[(c, float(tau))] = dict(
            median_abs=float(np.median(np.abs(g["dY"].to_numpy()))),
            cell_mean=cells, pooled_mean=float(g["dY"].mean()),
            n=int(len(g)))
    return ref


def cell_mean_baseline(pairs: pd.DataFrame, ref: dict) -> np.ndarray:
    """Per-pair TRAIN-cell-mean point mass (pooled fallback)."""
    out = np.zeros(len(pairs))
    for i, (c, tau, d, p) in enumerate(zip(
            pairs["contrast"], pairs["tau"], pairs["demand"],
            pairs["penetration"])):
        rr = ref.get((c, float(tau)))
        if rr is None:
            out[i] = 0.0
            continue
        out[i] = rr["cell_mean"].get((int(d), round(float(p), 6)),
                                     rr["pooled_mean"])
    return out


# ------------------------------------------------------------- A3 core ----

def contrast_metrics(pairs: pd.DataFrame, dv_samples_ind: np.ndarray,
                     dv_samples_com: np.ndarray, point_dv: np.ndarray,
                     ref: dict, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> list[dict]:
    """Per (contrast, tau) A3 metrics for the pairs frame (row-aligned
    with the dV sample arrays).  Returns a list of dict records."""
    out = []
    dY = pairs["dY"].to_numpy(float)
    cellm = cell_mean_baseline(pairs, ref)
    mean_ind = dv_samples_ind.mean(axis=1)
    crps_ind = crps_samples(dv_samples_ind, dY)
    crps_com = crps_samples(dv_samples_com, dY)
    crps_zero = np.abs(dY)
    crps_cell = np.abs(dY - cellm)
    for (c, tau), g in pairs.groupby(["contrast", "tau"]):
        idx = g.index.to_numpy()
        rr = ref.get((c, float(tau)))
        thr = rr["median_abs"] if rr else float("nan")
        elig = np.abs(dY[idx]) > thr if rr else np.zeros(len(idx), bool)
        sgn_pred = np.sign(mean_ind[idx])
        correct = (sgn_pred == np.sign(dY[idx])).astype(float)
        sel = idx[elig]
        boot = world_bootstrap_ci(correct[elig], g["key"].to_numpy()[elig],
                                  n_boot=n_boot, seed=seed)
        rec = dict(
            contrast=c, tau=float(tau), n_pairs=int(len(idx)),
            n_worlds=int(g["key"].nunique()),
            train_median_abs_dY=thr, n_eligible=int(elig.sum()),
            sign_acc=boot["acc"], sign_ci_lo=boot["lo"],
            sign_ci_hi=boot["hi"], sign_n_worlds=boot["n_worlds"],
            crps_dv=float(crps_ind[idx].mean()),
            crps_dv_comonotone=float(crps_com[idx].mean()),
            crps_zero=float(crps_zero[idx].mean()),
            crps_cell_mean=float(crps_cell[idx].mean()),
            point_dv_mean=float(np.mean(point_dv[idx])),
            dY_mean=float(dY[idx].mean()),
            mae_point_dv=float(np.mean(np.abs(point_dv[idx] - dY[idx]))),
            mae_zero=float(np.mean(np.abs(dY[idx]))))
        rec["pass_i"] = bool(np.isfinite(rec["sign_ci_lo"])
                             and rec["sign_ci_lo"] > 0.5)
        rec["pass_ii"] = bool(rec["crps_dv"] < rec["crps_zero"]
                              and rec["crps_dv"] < rec["crps_cell_mean"])
        rec["pass_tau"] = bool(rec["pass_i"] and rec["pass_ii"])
        out.append(rec)
    return out


def a3_verdict(records: list[dict]) -> dict:
    """Per gating contrast: passes A3 iff (i) AND (ii) at >= 2 of the 3
    taus in TAUS_A3."""
    res = {}
    for c in GATING_CONTRASTS:
        rs = [r for r in records if r["contrast"] == c
              and r["tau"] in TAUS_A3]
        n_pass = sum(1 for r in rs if r["pass_tau"])
        res[c] = dict(n_taus_evaluated=len(rs), n_taus_pass=int(n_pass),
                      passes=bool(n_pass >= 2),
                      per_tau={f"{r['tau']:.0f}": dict(
                          pass_i=r["pass_i"], pass_ii=r["pass_ii"])
                          for r in rs})
    return res


# ------------------------------------------------------------ A1 / A2 -----

def a1_verdict(per_tau: list[dict], coverage_overall: float) -> dict:
    """per_tau rows carry tau, crps_model, crps_climatology."""
    beats = {f"{r['tau']:.0f}": bool(r["crps_model"] < r["crps_climatology"])
             for r in per_tau}
    cov_ok = COVERAGE_BAND[0] <= coverage_overall <= COVERAGE_BAND[1]
    return dict(crps_beats_climatology=beats,
                all_taus_beat_climatology=bool(all(beats.values())
                                               and len(beats) > 0),
                coverage_overall=float(coverage_overall),
                coverage_in_band=bool(cov_ok), band=list(COVERAGE_BAND),
                passes=bool(all(beats.values()) and len(beats) > 0
                            and cov_ok))


def a2_verdict(acceptance_rows: list[dict]) -> dict:
    """acceptance_rows: baselines_section()['acceptance'] (tau, result)."""
    per = {f"{r['tau']:.0f}": r["result"] for r in acceptance_rows}
    gate = {f"{t:.0f}": per.get(f"{t:.0f}") for t in TAUS_A2_GATING}
    return dict(per_tau=per, gating=gate,
                gating_pass=bool(all(v == "PASS" for v in gate.values())),
                nongating={k: v for k, v in per.items()
                           if k not in gate})


def overall_verdict(a1: dict, a2: dict, a3: dict) -> dict:
    n_c = sum(1 for c in GATING_CONTRASTS if a3.get(c, {}).get("passes"))
    if a1["passes"] and a2["gating_pass"] and n_c == len(GATING_CONTRASTS):
        v = "ACCEPT"
    elif a1["passes"] and a2["gating_pass"] and n_c == 1:
        v = "PARTIAL"
    else:
        v = "FAIL"
    return dict(verdict=v, a1=a1["passes"], a2_gating=a2["gating_pass"],
                a3_contrasts_passing=n_c,
                a3_by_contrast={c: a3.get(c, {}).get("passes", False)
                                for c in GATING_CONTRASTS})
