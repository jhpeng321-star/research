"""WD-01 part 3q shared library: the merge over-count diagnosis + fix
(per-type LC first-class) and the k-fold TRAIN spread refit for the
native Part A emulator (kind idm_merge, p3o levers + the part 3q
levers), everything else inherited from part 3o.

New engine levers (part 3q, all default no-op -- earlier kinds and all
prior-part configs reproduce byte-identically):
  merge_cooldown_ticks  hysteresis on the thinning path: a candidate
                        rejected BY THINNING cannot re-attempt M2 for
                        this many ticks (p3o evidence: per-tick retries
                        make plain merge_accept_prob delay-only).
  merge_early_d /       position-dependent acceptance: a candidate
  merge_early_prob      EARLY in the zone (d = OFFSETS[2] - s >
                        merge_early_d; d falls from 0 at zone entry to
                        -75.4 m at the lane end) draws against
                        merge_early_prob instead of merge_accept_prob.
Plus Rollout.merge_events bookkeeping (tick/vid/s/d/v per executed M2
merge) -- the merge-position/timing diagnosis basis.

Split declaration (brief; the earlier kinds keep their splits):
  TRAIN (this task)   = the 17 non-validation, non-holdout native Part A
                        roots (fit 4 + calibration 4 + remaining 9);
  k-fold              = k=5 BY ROOT, deterministic stratified
                        round-robin over the sorted 17 -> folds f0..f4
                        (sizes 4,4,3,3,3); EVERY window is
                        t0 in 120..480 step 60 x its root;
  fit (development)   = folds f1..f4 (13 roots, 91 windows); fold f0 is
                        held out of ALL parameter fitting and is the
                        only fully out-of-sample fold in the gate table
                        (disclosed per fold);
  gate                = all 5 folds scored with spread factors fitted by
                        CROSS-FITTING (each fold's factors fitted on the
                        other 4 folds' rows only, never on the fold
                        scored);
  frozen spread       = fitted on the FULL 17-root split (119 windows),
                        the production protocol for a POSSIBLE later
                        single validation look (NOT run);
  validation          = the 4 p3l validation roots -- NEVER loaded or
                        scored in part 3q (no validation look);
  holdout             = blocked in p3lib.root_arrays (never passed).

ZERO simulation: no SUMO/TraCI anywhere; the only dynamics is the
part-3 rollout engine (pure torch).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "part3o"), str(HERE.parent / "part3n"),
           str(HERE.parent / "part3k"), str(HERE.parent / "part3l"),
           str(HERE.parent / "part3m"), str(HERE.parent / "part3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import p3o_lib as O  # noqa: E402  (roll + typed-row + fit machinery)
import p3n_lib as N  # noqa: E402  (split + gate + spread machinery)
import decompose_p3k as D  # noqa: E402  (Tee, priority, stats helpers)
import p3lib as P3  # noqa: E402
import p3l_lib as L3  # noqa: E402
import p3m_lib as M  # noqa: E402

RES = HERE / "results"
RAW_DIR = RES / "raw"
PART3O = HERE.parent / "part3o"

HORIZONS = N.HORIZONS
GATE_T0S = N.GATE_T0S
K_TICKS = N.K_TICKS
OUTPUTS = N.OUTPUTS
LC_TYPE_KEYS = O.LC_TYPE_KEYS
LC_BAND = 0.20
ACCEL_END_S = P3.ACCEL_LEN            # 454.10 (== engine.ACCEL_END_S)
DT_S = P3.DT

# --------------------------------------------------------------- folds --

K_FOLD = 5
FOLD_NAMES = tuple(f"f{i}" for i in range(K_FOLD))


def train17():
    """The 17 TRAIN roots of this task (fit + calibration + remaining),
    from the frozen p3l split; asserts disjointness from val/holdout."""
    sp = O.splits()
    tr = sorted(set(sp["fit"]) | set(sp["calibration"]))
    # the remaining TRAIN roots: everything roots_by_split calls train
    # that is neither fit nor calibration nor validation
    all_train = sorted(P3.roots_by_split()["train"])
    extra = [r for r in all_train if r not in tr]
    roots = sorted(tr) + sorted(extra)
    assert len(roots) == 17, roots
    assert not (set(roots) & set(sp["val"]))
    assert not (set(roots) & set(sp["holdout"]))
    return roots


def folds_by_root(roots=None, k=K_FOLD):
    """{fold_name: [roots]}: deterministic stratified round-robin over
    the sorted roots (index i -> fold i % k).  17 roots, k=5 -> sizes
    4,4,3,3,3; every fold mixes load classes (sorted order interleaves
    the cv17/cv19/cv20/cv30/seed families)."""
    roots = sorted(roots if roots is not None else train17())
    out = {f"f{i}": [] for i in range(k)}
    for i, r in enumerate(roots):
        out[f"f{i % k}"].append(r)
    return out


FOLDS = folds_by_root()
DEV_FOLD = "f0"                       # held out of ALL parameter fitting
FIT_FOLDS = tuple(f for f in FOLD_NAMES if f != DEV_FOLD)
FIT_ROOTS = sorted(r for f in FIT_FOLDS for r in FOLDS[f])
FOLD_OF = {r: f for f, rs in FOLDS.items() for r in rs}


def fold_of(root):
    return FOLD_OF[root]


def other_folds(fname):
    return tuple(f for f in FOLD_NAMES if f != fname)


WINDOWS_PER_ROOT = len(GATE_T0S)
ROWS_PER_MEMBER = 2 * WINDOWS_PER_ROOT * 17          # one row per (win, H)


# ------------------------------------------------------------- rolling --

def roll_member_q(root, t0, cfg, dev, K=K_TICKS):
    """One idm_merge rollout from a p3q member config dict (the p3o
    knobs plus the part 3q knobs; all default to the no-op)."""
    import engine as E
    return E.Rollout(root, float(t0), kind="idm_merge",
                     member_seed=int(cfg["seed"]),
                     lc_scale=float(cfg["lc_scale"]),
                     lc_scale_disc=cfg.get("lc_scale_disc"),
                     merge_accept_prob=float(
                         cfg.get("merge_accept_prob", 1.0)),
                     merge_cooldown_ticks=int(
                         cfg.get("merge_cooldown_ticks") or 0),
                     merge_early_d=float(
                         cfg.get("merge_early_d") or 0.0),
                     merge_early_prob=cfg.get("merge_early_prob"),
                     merge_params=dict(cfg["merge_params"]),
                     dawdle_scale=float(cfg.get("dawdle_scale", 1.0)),
                     dev=dev).run(n_ticks=K)


def roll_l0(root, t0, dev, K=K_TICKS):
    import engine as E
    return E.Rollout(root, float(t0), kind="idm_normal",
                     dev=dev).run(n_ticks=K)


def roll_member_rows_q(root, t0, cfg, mi, variant, dev, note, K=K_TICKS):
    r = roll_member_q(root, t0, cfg, dev, K)
    rows = O.window_raw_rows_typed(r, variant, mi, root, t0,
                                   float(cfg.get("dawdle_scale", 1.0)),
                                   note)
    return r, rows


def events_and_stats(r, root, t0, mi):
    """Persistable per-roll merge bookkeeping: the executed-merge event
    records (tick/s/d/v) + the rejection-cause counters."""
    return dict(root=root, t0=int(t0), member=int(mi),
                events=list(r.merge_events),
                stats={k: v for k, v in r.merge_stats.items()
                       if k not in ("breq_hist", "breq_foll_rej")})


def cfg_note(cfg):
    return (f"lc {cfg['lc_scale']} disc {cfg.get('lc_scale_disc')} "
            f"mprob {cfg.get('merge_accept_prob', 1.0)} "
            f"cd {cfg.get('merge_cooldown_ticks', 0)} "
            f"early_d {cfg.get('merge_early_d', 0.0)} "
            f"early_p {cfg.get('merge_early_prob')} "
            f"tau {cfg['merge_params']['tau_accept']} "
            f"b {cfg['merge_params']['b_accept']} "
            f"force {cfg['merge_params']['end_force_s']} "
            f"blend {cfg['merge_params']['v_merge_blend']} "
            f"pcoop {cfg['merge_params']['p_coop_scale']} "
            f"dawdle {cfg.get('dawdle_scale', 1.0)}")


# ------------------------------------------------------ by-type tables --

def lc_by_type(rows, H, roots=None, variant="L6"):
    return O.lc_by_type(rows, H, roots, variant)


def by_type_block(rows_l6, H, folds=FOLD_NAMES):
    """{pooled, f0..f4 -> by-type table} for one horizon."""
    out = dict(pooled=lc_by_type(rows_l6, H))
    for f in folds:
        out[f] = lc_by_type(rows_l6, H, set(FOLDS[f]))
    return out


# ------------------------------------------- merge events (pred + truth) --

_OFF_TO_ACCEL = P3.OFFSETS[1] - P3.OFFSETS[2]   # 2265.56 m


def truth_merge_rows(ra, k0, K):
    """Truth mandatory merges (2->1) with k0 < tick <= k0+K from the
    declared p2C events: [{rel_s (s since window start), s (ACCEL-lane
    frame)}].  lc_s is recorded on the NEW lane's physical coordinate
    (d preserved through the change -- the p3k window_truth_stats
    convention), so s_accel = lc_s - (OFFSETS[1] - OFFSETS[2]) puts the
    truth position in the same frame as the engine's pre-change record.
    The truth side is member-independent."""
    out = []
    for v, t, f, tto, s in zip(ra.lc_vid, ra.lc_tick, ra.lc_from,
                               ra.lc_to, ra.lc_s):
        if not (k0 < t <= k0 + K):
            continue
        if (int(f), int(tto)) != (2, 1):
            continue
        out.append(dict(rel_s=(int(t) - k0) * DT_S,
                        s=float(s) - _OFF_TO_ACCEL))
    return out


def pred_merge_rows(r, H):
    """Pred executed merges within the horizon prefix from the engine
    bookkeeping: [{rel_s, s}] (same schema as truth_merge_rows)."""
    kh = int(round(H / DT_S))
    return [dict(rel_s=e["tick"] * DT_S, s=float(e["s"]))
            for e in r.merge_events if e["tick"] <= kh]


def dist_summary(rows, horizons=HORIZONS):
    """Per horizon: merge-position/timing distribution {n, mean, p10,
    p50, p90} for rel_s (s since t0) and dist_to_end (m to the accel
    lane end, 0 at the end; = ACCEL_END_S - s)."""
    out = {}
    for H in horizons:
        sel = [e for e in rows
               if e["rel_s"] <= H + 1e-9]
        out[H] = _summ([e["rel_s"] for e in sel],
                       [ACCEL_END_S - e["s"] for e in sel])
    return out


def _summ(xs, ys):
    def q(a, p):
        return round(float(np.percentile(a, p)), 2) if len(a) else None
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    return dict(n=int(len(x)),
                rel_mean=round(float(x.mean()), 2) if len(x) else None,
                rel_p10=q(x, 10), rel_p50=q(x, 50), rel_p90=q(x, 90),
                dist_mean=round(float(y.mean()), 2) if len(y) else None,
                dist_p10=q(y, 10), dist_p50=q(y, 50), dist_p90=q(y, 90))


def pool_merge_stats(stats_list):
    """Element-wise sum of the numeric merge_stats counters over rolls
    (breq_hist excluded upstream)."""
    keys = ("checks", "accepted", "rej_lead", "rej_foll", "rej_both",
            "with_foll", "rej_prob", "rej_prob_early", "cd_set",
            "cd_blocked")
    out = {k: 0 for k in keys}
    for st in stats_list:
        for k in keys:
            out[k] += int(st.get(k, 0))
    return out


# ------------------------------------------------- k-fold gate machinery --

def _fit_roots_of(fname, folds):
    other = set().union(*[set(folds[f]) for f in folds if f != fname])
    return other


def gate_rows_kf(rows_l6, folds=FOLDS, horizons=HORIZONS):
    """Criterion-(c) numbers under k-fold CROSS-FITTING: fold f's spread
    factors are fitted on the OTHER folds' rows only (never on the fold
    scored).  Per (output, H): pooled coverage/CRPS/min-max accumulated
    over windows (each scored out-of-fold) + the bias/width breakdown
    (below/in/above + signed truth-minus-centre residuals + band
    widths), and per-fold detail."""
    det = {name: {H: dict(cov=0, n=0, crps=[], width=[], truths=[],
                          resid=[], below=0, above=0, mm=0, per_fold={})
                  for H in horizons}
           for name, _pk, _tk in OUTPUTS}
    for fname in folds:
        score_roots = set(folds[fname])
        fit_roots = _fit_roots_of(fname, folds)
        s_map = N.fit_s_out(rows_l6, fit_roots, horizons)
        for name, pk, tk in OUTPUTS:
            for H in horizons:
                s = s_map[name][H]
                wins = N.member_arrays(rows_l6, "L6", H, score_roots)
                hits = below = above = mm = 0
                crps_v, width_v, truth_v, resid_v = [], [], [], []
                for _k, ws in wins:
                    x = np.array([float(m[pk]) for m in ws])
                    c = float(x.mean())       # mean is widening-invariant
                    y = float(ws[0][tk])
                    if s is not None:
                        x = L3.widen_ensemble(x, s, "mean")
                    lo, hi = L3.band_10_90(x)
                    hit = lo <= y <= hi
                    hits += hit
                    below += y < lo
                    above += y > hi
                    mm += bool(min(x) <= y <= max(x))
                    crps_v.append(L3.crps_ensemble(x, y))
                    width_v.append(hi - lo)
                    truth_v.append(y)
                    resid_v.append(y - c)
                d = det[name][H]
                n_f = len(wins)
                d["cov"] += hits
                d["n"] += n_f
                d["crps"].extend(crps_v)
                d["width"].extend(width_v)
                d["truths"].extend(truth_v)
                d["resid"].extend(resid_v)
                d["below"] += below
                d["above"] += above
                d["mm"] += mm
                d["per_fold"][fname] = dict(
                    s_out=s, n_windows=n_f,
                    cov=round(hits / max(n_f, 1), 4),
                    crps=round(float(np.mean(crps_v)), 3) if crps_v
                    else None,
                    below_frac=round(below / max(n_f, 1), 4),
                    above_frac=round(above / max(n_f, 1), 4),
                    bias_mean=round(float(np.mean(resid_v)), 2)
                    if resid_v else None,
                    width_med=round(float(np.median(width_v)), 2)
                    if width_v else None,
                    minmax=round(mm / max(n_f, 1), 4))
    out = {}
    for name, _pk, _tk in OUTPUTS:
        out[name] = {}
        for H in horizons:
            d = det[name][H]
            tw = np.array(d["truths"])
            rv = np.array(d["resid"]) if d["resid"] else np.array([0.0])
            out[name][H] = dict(
                n_windows=d["n"],
                cov=round(d["cov"] / max(d["n"], 1), 4),
                crps=round(float(np.mean(d["crps"])), 3)
                if d["crps"] else None,
                below_frac=round(d["below"] / max(d["n"], 1), 4),
                in_frac=round((d["n"] - d["below"] - d["above"])
                              / max(d["n"], 1), 4),
                above_frac=round(d["above"] / max(d["n"], 1), 4),
                bias_mean=round(float(rv.mean()), 2),
                resid_med=round(float(np.median(rv)), 2),
                median_band_width=round(float(np.median(d["width"])), 3)
                if d["width"] else None,
                truth_iqr_width=round(float(
                    np.percentile(tw, 90) - np.percentile(tw, 10)), 3),
                minmax=round(d["mm"] / max(d["n"], 1), 4),
                per_fold=d["per_fold"])
    return out


def gate_verdicts_kf(rows_all, horizons=HORIZONS, folds=FOLDS):
    """Full TRAIN k-fold gate: pooled gate v2 (a)-(d) + old gate (each
    window scored with its OUT-OF-FOLD spread factors) plus PER-FOLD
    gate v2 verdicts (fold summaries + that fold's cross-fitted
    coverage).  Reuses the part 3l verdict functions verbatim."""
    import aggregate_p3l as AG
    l0_rows = [r for r in rows_all if r["variant"] == "L0"]
    l6_rows = [r for r in rows_all if r["variant"] == "L6"]
    per_l0 = AG.summarise(l0_rows)
    per_l6 = AG.summarise(l6_rows)
    cov = gate_rows_kf(l6_rows, folds, horizons)
    pooled = {}
    for H in horizons:
        l0_speed = per_l0[("L0", H)]["speed_rmse"]
        l0_cov = dict(delay_phys_mae=round(float(np.mean(
            [abs(float(r["delay_phys_pred"]) -
                 float(r["delay_phys_true"]))
             for r in l0_rows if int(r["H"]) == H])), 2))
        cov_dict = dict(
            ttc3_cov_frac=cov["ttc3"][H]["cov"],
            delay_phys_cov_frac=cov["delay_phys"][H]["cov"],
            crps_phys_mean=cov["delay_phys"][H]["crps"],
            delay_phys_minmax_frac=cov["delay_phys"][H]["minmax"])
        g = AG.gate_v2_verdict(per_l6[("L6", H)], cov_dict, l0_cov,
                               l0_speed)
        g["old"] = AG.old_gate_verdict(per_l6[("L6", H)], cov_dict)
        g["per_fold"] = {name: cov[name][H]["per_fold"]
                         for name, _p, _t in OUTPUTS}
        g["bias_width"] = {name: dict(
            below_frac=cov[name][H]["below_frac"],
            in_frac=cov[name][H]["in_frac"],
            above_frac=cov[name][H]["above_frac"],
            bias_mean=cov[name][H]["bias_mean"],
            resid_med=cov[name][H]["resid_med"],
            median_band_width=cov[name][H]["median_band_width"],
            truth_iqr_width=cov[name][H]["truth_iqr_width"])
            for name in ("delay_phys", "ttc3")}
        pooled[H] = g
    per_fold = {}
    for fname in folds:
        roots_f = set(FOLDS[fname])
        l6_f = [r for r in l6_rows if r["root"] in roots_f]
        l0_f = [r for r in l0_rows if r["root"] in roots_f]
        s6 = AG.summarise(l6_f)
        s0 = AG.summarise(l0_f)
        per_fold[fname] = {}
        for H in horizons:
            l0_speed_f = s0[("L0", H)]["speed_rmse"]
            l0_cov_f = dict(delay_phys_mae=round(float(np.mean(
                [abs(float(r["delay_phys_pred"]) -
                     float(r["delay_phys_true"]))
                 for r in l0_f if int(r["H"]) == H])), 2))
            pf = cov["delay_phys"][H]["per_fold"][fname]
            pf_t = cov["ttc3"][H]["per_fold"][fname]
            cov_dict_f = dict(
                ttc3_cov_frac=pf_t["cov"],
                delay_phys_cov_frac=pf["cov"],
                crps_phys_mean=pf["crps"],
                delay_phys_minmax_frac=pf["minmax"])
            gf = AG.gate_v2_verdict(s6[("L6", H)], cov_dict_f, l0_cov_f,
                                    l0_speed_f)
            gf["old"] = AG.old_gate_verdict(s6[("L6", H)], cov_dict_f)
            gf["fold_s_out"] = dict(delay_phys=pf["s_out"],
                                    ttc3=pf_t["s_out"])
            per_fold[fname][H] = gf
    return dict(pooled=pooled, per_fold=per_fold, cov=cov,
                per_l0=per_l0, per_l6=per_l6)


# ----------------------------------------------------- climatology (kf) --

def climatology_kf(rows_l6, folds=FOLDS, horizons=HORIZONS):
    """Each scored window's TRAIN climatology = the empirical truth
    distribution of the OTHER folds (out-of-fold by construction;
    truth sides are member-independent).  Pooled CRPS accumulates the
    per-window out-of-fold values."""
    truth_map = {}
    for r in rows_l6:
        for name, _pk, tk in OUTPUTS:
            truth_map.setdefault((name, int(r["H"])), {})[
                (r["root"], int(r["t0"]))] = float(r[tk])
    acc = {name: {H: dict(crps=[], meanp=[], n=0) for H in horizons}
           for name, _pk, _tk in OUTPUTS}
    for fname in folds:
        roots_f = set(FOLDS[fname])
        for name, _pk, _tk in OUTPUTS:
            for H in horizons:
                tm = truth_map.get((name, H), {})
                sample_folds = np.array(
                    sorted(v for (rr, _t), v in tm.items()
                           if rr not in roots_f), dtype=np.float64)
                if len(sample_folds) < 2:
                    continue
                crps_v, meanp_v, n = [], [], 0
                for (rr, _t), y in sorted(tm.items()):
                    if rr not in roots_f:
                        continue
                    crps_v.append(L3.crps_ensemble(sample_folds, y))
                    meanp_v.append(abs(float(sample_folds.mean()) - y))
                    n += 1
                a = acc[name][H]
                a["crps"].extend(crps_v)
                a["meanp"].extend(meanp_v)
                a["n"] += n
                a.setdefault("per_fold", {})[fname] = dict(
                    crps=round(float(np.mean(crps_v)), 3),
                    n_windows=n)
    out = {}
    for name, _pk, _tk in OUTPUTS:
        out[name] = {}
        for H in horizons:
            a = acc[name][H]
            out[name][H] = dict(
                n_windows=a["n"],
                clima_crps_empirical=round(float(np.mean(a["crps"])), 3)
                if a["crps"] else None,
                clima_crps_meanpoint=round(float(np.mean(a["meanp"])), 3)
                if a["meanp"] else None,
                per_fold=a.get("per_fold", {}))
    return out


# --------------------------------------------------------- goal package --

def goal_package_q(pooled, rows_l6, folds=FOLD_NAMES):
    """The part 3q goal package per horizon (pooled over the 17 TRAIN
    roots, k-fold cross-fitted coverage):
      b_type   LC by type AND total within +-LC_BAND (merge, disc),
               POOLED (the same pooling as the formal gate-v2 (b))
      b_total  the formal gate-v2 (b) total-LC criterion
      c_gate   the formal gate-v2 (c) coverage+CRPS criterion
      a        collisions <= 1/1000
      d        speed RMSE <= 1.05 x L0 (per-horizon, binding)
    Per-fold by-type tables and v2 verdicts are reported alongside
    (fold f0 is the only fully out-of-sample fold)."""
    out = {}
    for H in HORIZONS:
        g = pooled[H]
        bt = by_type_block(rows_l6, H, folds)
        for name, tab in bt.items():
            if tab["n_windows"] == 0:
                raise ValueError(f"goal package: no {H}-horizon "
                                 f"windows in {name} -- degenerate "
                                 f"gate split")
        checks = dict(
            b_type_pooled=max(abs(bt["pooled"]["dev_merge"]),
                              abs(bt["pooled"]["dev_disc"]),
                              abs(bt["pooled"]["dev_total"])) <= LC_BAND,
            b_total_gate=bool(g["b_lc_pass"]),
            c_gate=bool(g["c_coverage_crps_pass"]),
            a_collisions=bool(g["a_collisions_pass"]),
            d_speed=bool(g["d_per_horizon_pass"]))
        checks["PASS"] = all(checks.values())
        out[H] = dict(checks=checks,
                      by_type=bt,
                      max_type_dev=dict(
                          pooled=max(abs(bt["pooled"]["dev_merge"]),
                                     abs(bt["pooled"]["dev_disc"]),
                                     abs(bt["pooled"]["dev_total"])),
                          **{f: max(abs(bt[f]["dev_merge"]),
                                    abs(bt[f]["dev_disc"]),
                                    abs(bt[f]["dev_total"]))
                             for f in folds}),
                      crps_phys=g["crps_phys"],
                      speed_rmse=g["speed_rmse"],
                      d_threshold=g["d_per_horizon_threshold"],
                      delay_phys_cov=g["delay_phys_cov_frac"],
                      ttc3_cov=g["ttc3_cov_frac"],
                      bias_width=g["bias_width"])
    return out


# --------------------------------------------------------- fit helpers --

def set_field(cfg, field, value):
    cfg = json.loads(json.dumps(cfg))          # deep copy
    if field in ("tau_accept", "b_accept", "end_force_s",
                 "v_merge_blend", "p_coop_scale"):
        cfg["merge_params"] = dict(cfg["merge_params"], **{field: value})
    else:
        cfg[field] = value
    return cfg


def get_field(cfg, field):
    if field in ("tau_accept", "b_accept", "end_force_s",
                 "v_merge_blend", "p_coop_scale"):
        return float(cfg["merge_params"][field])
    v = cfg.get(field)
    if v is None:
        v = {"lc_scale_disc": cfg.get("lc_scale"),
             "merge_accept_prob": 1.0,
             "merge_cooldown_ticks": 0,
             "merge_early_d": 0.0,
             "merge_early_prob": None}.get(field)
    return v


def truth_fit_targets(roots, t0s, K=K_TICKS):
    return O.truth_fit_targets(roots, t0s, K)


def pred_fit_stats_q(root, t0, cfg, dev, K=K_TICKS):
    """O.pred_fit_stats_o mirrored for a p3q member config (the part 3q
    knobs in the roll)."""
    r = roll_member_q(root, t0, cfg, dev, K)
    st = D.window_pred_stats(r, K)
    snaps = D.snaps_to_dicts(r, K)
    births = D.births_from_engine(r, K)
    ra = r.ra
    dp, _dt, _np_, _nt = D.phys_delay_sums(
        snaps, births, D.truth_rows(ra, r.k0, K), D.lc_ticks_from_ra(ra),
        D.birth_states_from_ra(ra, r.k0, K), r.k0, K)
    st["delay"] = dp
    st["lc60"] = int(r.lc_count)
    st["lc30"] = L3.lc_count_between(r.snapshots, 0, K // 2)
    st["ttc3"] = sum(len(M.ttc_pairs(r.snapshots[j]["lane"],
                                     r.snapshots[j]["s"],
                                     r.snapshots[j]["v"]))
                     for j in range(1, K + 1)) * M.DT
    for H, d in N.pred_lc_by_type(r.snapshots, K).items():
        st[f"lcM{H}"] = d["pred_merge"]
        st[f"lcD{H}"] = d["pred_disc"]
    return st


def joint_loss(preds, truths):
    return O.joint_loss(preds, truths)


def softmax_draw(losses, rng, t=0.15):
    return N.softmax_draw(np.asarray(losses, dtype=np.float64), rng, t)


# ------------------------------------------------------------ raw rows --

def read_rows(path):
    return O.read_rows(path)


def member_file_complete(path, expect_rows):
    return O.member_file_complete(path, expect_rows)


def write_rows(path, rows, mode="w"):
    O.write_rows(path, rows, mode)
