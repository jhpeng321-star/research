"""WD-01 part 3n shared library: lane-change BY TYPE (accel->main_right
mandatory merge vs discretionary 0<->1) counting on both sides, the
joint lc_scale x end_force_s loss, the TRAIN-internal 2-fold gate
(split declaration, fold spread fitting, pooled coverage/CRPS), the
climatology/sharpness tables, and the freeze-gate helpers.

Split declaration (brief; the earlier kinds keep their splits):
  fit roots (refit)   = part 3l/3m fit roots, cv17, cv19_4107,
                        cv19_4108, cv20_7001m2000 (t0 120/300, H=60)
  TRAIN-internal gate = the 4 calibration roots, 2 FOLDS:
                        fold A = cv20_7002m2000, cv20_7003m2000
                        fold B = cv20_7004m2000, cv20_7007m2700
                        t0 120..480 step 60 -> 28 windows, M=5, H {30,60}
  validation          = the 4 p3l validation roots x 13 t0 = 52 windows
                        (ONE look, only after TRAIN-internal pass +
                        freeze)
  holdout             = blocked in p3lib.root_arrays (never passed)

2-fold protocol (brief): when scoring fold X, the spread factors s_out
per (output, horizon) are fitted on fold Y's rows (the OTHER fold), so
no window is scored with factors fitted on itself.  The final frozen
factors (for the single validation look) are fitted on the FULL
calibration split, exactly the part 3m protocol.

ZERO simulation: no SUMO/TraCI import anywhere in part 3n; the only
dynamics is the part-3 rollout engine (pure torch).  The engine is
UNTOUCHED in part 3n (end_force_s is already a merge_params knob that
takes any nonnegative distance).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "part3k"), str(HERE.parent / "part3l"),
           str(HERE.parent / "part3m"), str(HERE.parent / "part3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decompose_p3k as D  # noqa: E402  (stats, Tee, delay model)
import p3lib as P3  # noqa: E402
import p3l_lib as L3  # noqa: E402  (splits + spread math, inherited)
import p3m_lib as M  # noqa: E402  (ttc pairs + event ticks, inherited)

# ------------------------------------------------------------- grids -----
LC_GRID_P3N = (0.0005, 0.00075, 0.001, 0.0015, 0.002)   # brief: ..0.002
FORCE_GRID_P3N = (60.0, 100.0, 150.0, 200.0)            # brief (no 0:
# p3m measured force-off 20x worse; the brief's grid starts at 60)
LC_NORM_FLOOR = 5.0               # per-type truth-count floor (veh count)
HORIZONS = (30, 60)
GATE_T0S = tuple(range(120, 481, 60))     # 7 t0 x 4 calibration roots = 28
K_TICKS = 120                     # 60 s rollout covers H <= 60 exactly

FOLD_A = ("cv20_7002m2000", "cv20_7003m2000")
FOLD_B = ("cv20_7004m2000", "cv20_7007m2700")
FOLDS = dict(A=FOLD_A, B=FOLD_B)
OUTPUTS = (("delay_phys", "delay_phys_pred", "delay_phys_true"),
           ("ttc3", "pred_ttc3_exposure", "true_ttc3_exposure"))


def other_fold(fold):
    return "B" if fold == "A" else "A"


# -------------------------------------------------- LC by type (both sides)

def pred_lc_by_type(snaps, K, horizons=HORIZONS):
    """Pred lane-change counts BY TYPE per horizon prefix from the pred
    snapshots: {H: dict(pred_merge=, pred_disc=)}.  merge = the M2
    mandatory accel(2)->main_right(1) change; disc = every other change
    (0<->1).  Uses p3m_lib.pred_event_ticks (per-vid change step
    indices)."""
    ev = M.pred_event_ticks(snaps, K)
    out = {}
    for H in horizons:
        kh = int(round(H / 0.5))
        nm = sum(sum(1 for j in js[0] if j <= kh) for js in ev.values())
        nd = sum(sum(1 for j in js[1] if j <= kh) for js in ev.values())
        out[H] = dict(pred_merge=int(nm), pred_disc=int(nd))
    return out


def truth_lc_by_type(ra, k0, K, horizons=HORIZONS):
    """Truth lane-change counts BY TYPE per horizon prefix from the
    declared p2C events: {H: dict(true_merge=, true_disc=)}."""
    ev_m, ev_l = M.truth_event_ticks(ra, k0, K)
    out = {}
    for H in horizons:
        lim = k0 + int(round(H / 0.5))
        out[H] = dict(
            true_merge=sum(sum(1 for t in ts if t <= lim)
                           for ts in ev_m.values()),
            true_disc=sum(sum(1 for t in ts if t <= lim)
                          for ts in ev_l.values()))
    return out


# ----------------------------------------------------------------- loss --

def truth_fit_stats(root, t0, ra, K=K_TICKS):
    """Truth-side stats for one fit window (superset of the part 3m
    fit target): zone occupancy / profile / merge-d / mr speed (p3m
    stage-1 form), physics delay, TTC<3 exposure over ALL truth rows,
    and LC counts total + BY TYPE at both horizons."""
    k0 = int(round((t0 - 60.0) / 0.5))
    st = D.window_truth_stats(ra, k0, K)
    st["lc60"] = L3.lc_truth_between(ra, k0, k0 + K)
    st["lc30"] = L3.lc_truth_between(ra, k0, k0 + K // 2)
    rows = D.truth_rows(ra, k0, K)
    recs_t, _p, _f = D.truth_records(
        rows, D.lc_ticks_from_ra(ra), D.birth_states_from_ra(ra, k0, K),
        k0, K)
    st["delay"] = sum(r_["delay"] for r_ in recs_t)
    st["ttc3"] = M.truth_ttc3_exposure(ra, k0, K)
    for H, d in truth_lc_by_type(ra, k0, K).items():
        st[f"lcM{H}"] = d["true_merge"]
        st[f"lcD{H}"] = d["true_disc"]
    return st


def pred_fit_stats(root, t0, mp, lc, dev, dw=1.0, K=K_TICKS):
    """Pred-side mirror of truth_fit_stats for one idm_merge rollout
    (the p3m fit's pred_stats + the per-type LC counts)."""
    r = P3_roll(root, t0, mp, lc, dev, dw, K)
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
    for H, d in pred_lc_by_type(r.snapshots, K).items():
        st[f"lcM{H}"] = d["pred_merge"]
        st[f"lcD{H}"] = d["pred_disc"]
    return st


def P3_roll(root, t0, mp, lc, dev, dw=1.0, K=K_TICKS, seed=1):
    """One idm_merge rollout (engine UNCHANGED; end_force_s is a plain
    merge_params knob)."""
    import engine as E
    return E.Rollout(root, float(t0), kind="idm_merge", member_seed=seed,
                     lc_scale=lc, merge_params=dict(mp),
                     dawdle_scale=dw, dev=dev).run(n_ticks=K)


def loss_stage1(preds, truths):
    """Acceptance loss, part 3m form: zone occupancy + pooled accel
    speed profile + merge-point d (verbatim maths)."""
    occ_l = mg_l = 0.0
    pool = {}
    n_w = 0
    prof_bins = tuple(range(-50, 350, 50))
    prof_min_n = 60
    for key, p in preds.items():
        t = truths[key]
        n_w += 1
        occ_l += ((p["occ"] - t["occ"]) / max(t["occ"], 50.0)) ** 2
        if p["merges"] and t["merges"]:
            mg_l += ((float(np.mean(p["merges"])) -
                      float(np.mean(t["merges"]))) / 75.0) ** 2
        elif bool(p["merges"]) != bool(t["merges"]):
            mg_l += 4.0
        for db in prof_bins:
            pp, pt_ = p["prof"].get(db), t["prof"].get(db)
            if pp:
                c = pool.setdefault(db, [0.0, 0, 0.0, 0])
                c[0] += pp[0]
                c[1] += pp[1]
            if pt_:
                c = pool.setdefault(db, [0.0, 0, 0.0, 0])
                c[2] += pt_[0]
                c[3] += pt_[1]
    prof_terms = [((c[0] / c[1] - c[2] / c[3]) / 5.0) ** 2
                  for c in pool.values()
                  if c[1] >= prof_min_n and c[3] >= prof_min_n]
    return (occ_l + sum(prof_terms) / max(len(prof_terms), 1)
            + mg_l) / max(n_w, 1), dict(
                occ=float(occ_l / max(n_w, 1)),
                prof=float(sum(prof_terms) / max(len(prof_terms), 1)),
                merge=float(mg_l / max(n_w, 1)))


def loss_p3n(preds, truths):
    """Joint part 3n loss (brief loss terms: LC count per window BY
    TYPE, TTC<3 exposure, physics delay, merge-zone occupancy):

      stage1 = part 3m stage-1 (occupancy + profile + merge-d guard)
      stage2 = mr guard + physics delay + LC BY TYPE (merge, disc) at
               H=30 (primary) + half weight at H=60 + TTC<3 exposure

    vs part 3m stage 2 the ONLY change is lc30/lc60 split into
    lcM/lcD (norm floor 5 instead of 10 -- the counts are per type
    now)."""
    mr_l = d_l = t3_l = 0.0
    lc_l = {k: 0.0 for k in ("lcM30", "lcD30", "lcM60", "lcD60")}
    n_w = 0
    for key, p in preds.items():
        t = truths[key]
        n_w += 1
        if p["mr"][1] and t["mr"][1]:
            mr_l += ((p["mr"][0] / p["mr"][1] -
                      t["mr"][0] / t["mr"][1]) / 5.0) ** 2
        d_l += ((p["delay"] - t["delay"]) / 200.0) ** 2
        for k in lc_l:
            lc_l[k] += ((p[k] - t[k]) / max(t[k], LC_NORM_FLOOR)) ** 2
        t3_l += ((p["ttc3"] - t["ttc3"]) /
                 max(t["ttc3"], 5.0)) ** 2          # p3m T3_NORM_FLOOR
    det = dict(mr=float(mr_l / max(n_w, 1)),
               delay=float(d_l / max(n_w, 1)),
               ttc3=float(t3_l / max(n_w, 1)))
    det.update({k: float(v / max(n_w, 1)) for k, v in lc_l.items()})
    s2 = (mr_l + d_l + t3_l + lc_l["lcM30"] + lc_l["lcD30"]
          + 0.5 * (lc_l["lcM60"] + lc_l["lcD60"])) / max(n_w, 1)
    return s2, det


def softmax_draw(losses, rng, t=0.15):
    """Index draw with p_i proportional to exp(-(l_i - min)/t) (the p3l
    wider-draw mechanism)."""
    l = np.asarray(losses, dtype=np.float64)
    w = np.exp(-(l - l.min()) / t)
    w = w / w.sum()
    return int(rng.choice(len(l), p=w))


# ------------------------------------------- TRAIN-internal gate machinery

def member_arrays(rows, variant, H, roots=None):
    """[(root, t0)] -> sorted-member value arrays + truth, for one
    (variant, H) over the given roots (default: all rows' roots)."""
    by_win = {}
    for r in rows:
        if r["variant"] != variant or int(r["H"]) != H:
            continue
        if roots is not None and r["root"] not in roots:
            continue
        by_win.setdefault((r["root"], int(r["t0"])), {})[
            int(r["member"])] = r
    out = []
    for key in sorted(by_win):
        memb = by_win[key]
        if len(memb) < 2:
            continue
        out.append((key, [memb[mi] for mi in sorted(memb)]))
    return out


def fit_s_out(rows, roots, horizons=HORIZONS):
    """{name: {H: s_out}} fitted on the given roots' rows ONLY (the
    fold used for FITTING; part 3l spread math)."""
    out = {name: {} for name, _pk, _tk in OUTPUTS}
    for name, pk, tk in OUTPUTS:
        for H in horizons:
            wins = member_arrays(rows, "L6", H, roots)
            samples = [np.array([float(m[pk]) for m in ws])
                       for _k, ws in wins]
            truths = [float(ws[0][tk]) for _k, ws in wins]
            out[name][H] = L3.s_out_for(samples, truths)
    return out


def gate_rows(rows, folds=("A", "B"), horizons=HORIZONS):
    """TRAIN-internal gate criterion-(c) numbers: coverage + ensemble
    CRPS + sharpness pooled over folds, each fold scored with the
    OTHER fold's spread factors (2-fold protocol).  Returns
    dict(name -> {H: dict(...)}) plus per-fold detail."""
    det = {name: {H: dict(cov=0, n=0, crps=[], width=[],
                          truths=[], per_fold={})
                  for H in horizons}
           for name, _pk, _tk in OUTPUTS}
    for fold in folds:
        score_roots = set(FOLDS[fold])
        fit_roots = set(FOLDS[other_fold(fold)])
        s_map = fit_s_out(rows, fit_roots, horizons)
        for name, pk, tk in OUTPUTS:
            for H in horizons:
                s = s_map[name][H]
                wins = member_arrays(rows, "L6", H, score_roots)
                hits = 0
                crps_v, width_v, truth_v = [], [], []
                for _k, ws in wins:
                    x = np.array([float(m[pk]) for m in ws])
                    y = float(ws[0][tk])
                    if s is not None:
                        x = L3.widen_ensemble(x, s, "mean")
                    lo, hi = L3.band_10_90(x)
                    hits += lo <= y <= hi
                    crps_v.append(L3.crps_ensemble(x, y))
                    width_v.append(hi - lo)
                    truth_v.append(y)
                d = det[name][H]
                d["cov"] += hits
                d["n"] += len(wins)
                d["crps"].extend(crps_v)
                d["width"].extend(width_v)
                d["truths"].extend(truth_v)
                d["per_fold"][fold] = dict(
                    s_out=s, n_windows=len(wins),
                    cov=round(hits / max(len(wins), 1), 4),
                    crps=round(float(np.mean(crps_v)), 3) if crps_v
                    else None)
    out = {}
    for name, _pk, _tk in OUTPUTS:
        out[name] = {}
        for H in horizons:
            d = det[name][H]
            tw = np.array(d["truths"])
            out[name][H] = dict(
                n_windows=d["n"],
                cov=round(d["cov"] / max(d["n"], 1), 4),
                crps=round(float(np.mean(d["crps"])), 3)
                if d["crps"] else None,
                median_band_width=round(float(np.median(d["width"])), 3)
                if d["width"] else None,
                truth_iqr_width=round(float(
                    np.percentile(tw, 90) - np.percentile(tw, 10)), 3),
                per_fold=d["per_fold"])
    return out


def minmax_cov(rows, folds=("A", "B"), horizons=HORIZONS):
    """Min-max physics-delay coverage pooled over folds (each fold
    widened with the OTHER fold's delay factor; the old-gate item)."""
    tot = hit = 0
    for fold in folds:
        s_map = fit_s_out(rows, set(FOLDS[other_fold(fold)]), horizons)
        for H in horizons:
            s = s_map["delay_phys"][H]
            for _k, ws in member_arrays(rows, "L6", H, set(FOLDS[fold])):
                x = np.array([float(m["delay_phys_pred"]) for m in ws])
                if s is not None:
                    x = L3.widen_ensemble(x, s, "mean")
                y = float(ws[0]["delay_phys_true"])
                tot += 1
                hit += bool(min(x) <= y <= max(x))
    return round(hit / max(tot, 1), 4)


def gate_verdicts(rows, horizons=HORIZONS, folds=("A", "B")):
    """Full TRAIN-internal gate: (a)-(d) v2 verdicts + old gate for
    L6/L6c (2-fold spread) at each horizon, with L0 on the same split
    as the (d)/CRPS reference.  Reuses the part 3l verdict functions."""
    import aggregate_p3l as AG
    l0_rows = [r for r in rows if r["variant"] == "L0"]
    l6_rows = [r for r in rows if r["variant"] == "L6"]
    per_l0 = AG.summarise(l0_rows)
    per_l6 = AG.summarise(l6_rows)
    cov = gate_rows(l6_rows, folds, horizons)
    mm = minmax_cov(l6_rows, folds, horizons)
    out = {}
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
            delay_phys_minmax_frac=mm)
        g = AG.gate_v2_verdict(per_l6[("L6", H)], cov_dict, l0_cov,
                               l0_speed)
        g["old"] = AG.old_gate_verdict(per_l6[("L6", H)], cov_dict)
        g["per_fold"] = {name: cov[name][H]["per_fold"]
                         for name, _p, _t in OUTPUTS}
        g["sharpness"] = {name: dict(
            median_band_width=cov[name][H]["median_band_width"],
            truth_iqr_width=cov[name][H]["truth_iqr_width"]) for name
            in ("delay_phys", "ttc3")}
        out[H] = g
    return out


# ---------------------------------------------------- climatology tables --

def climatology_from_rows(rows, roots=None, horizons=HORIZONS):
    """TRAIN empirical truth distribution per (output, H): unique
    (root, t0) truth values taken from raw metric rows (the truth side
    is member-independent).  roots=None keeps every root in the file."""
    seen = {}
    for r in rows:
        if roots is not None and r["root"] not in roots:
            continue
        for name, _pk, tk in OUTPUTS:
            if int(r["H"]) in horizons:
                seen.setdefault((name, int(r["H"])), {})[
                    (r["root"], int(r["t0"]))] = float(r[tk])
    return {k: np.array(sorted(d.values()),
                        dtype=np.float64) for k, d in seen.items()}


def climatology_table(rows_l6, clima, horizons=HORIZONS):
    """CRPS of the L6 member ensembles vs the climatology baselines
    (empirical TRAIN distribution + degenerate TRAIN-mean point
    forecast), pooled over the scored windows.  `rows_l6` must be the
    ALREADY-WIDENED scoring context is NOT applied here -- pass rows
    and widen inside (2-fold aware caller)."""
    out = {}
    for name, pk, tk in OUTPUTS:
        for H in horizons:
            samples_t = clima.get((name, H))
            crps_e, crps_m, n = [], [], 0
            for _k, ws in member_arrays(rows_l6, "L6", H):
                y = float(ws[0][tk])
                if samples_t is None or len(samples_t) < 2:
                    continue
                crps_e.append(L3.crps_ensemble(samples_t, y))
                crps_m.append(abs(float(samples_t.mean()) - y))
                n += 1
            out[(name, H)] = dict(
                n_windows=n,
                clima_crps_empirical=round(float(np.mean(crps_e)), 3)
                if crps_e else None,
                clima_crps_meanpoint=round(float(np.mean(crps_m)), 3)
                if crps_m else None,
                clima_samples=len(samples_t) if samples_t is not None
                else 0)
    return out


def widened_l6_rows(rows, s_map):
    """Return member rows whose delay_phys_pred / pred_ttc3_exposure
    are replaced by the widened ensembles under a FIXED {name: {H: s}}
    map (used for the validation look, where the frozen factors apply
    to every window).  The truth sides and all other fields pass
    through."""
    by_win = {}
    for r in rows:
        if r["variant"] != "L6":
            continue
        by_win.setdefault((int(r["H"]), r["root"], int(r["t0"])), {})[
            int(r["member"])] = dict(r)
    out = []
    for (H, root, t0), memb in by_win.items():
        for name, pk, _tk in OUTPUTS:
            s = s_map.get(name, {}).get(H)
            if s is None:
                continue
            mis = sorted(memb)
            xs = np.array([float(memb[mi][pk]) for mi in mis])
            xw = L3.widen_ensemble(xs, s, "mean")
            for pos, mi in enumerate(mis):
                memb[mi][pk] = float(xw[pos])
        out.extend(memb.values())
    return out


# ------------------------------------------------- raw metric row builder

def window_raw_rows(r, variant, mi, root, t0, dw, engine_note,
                    horizons=HORIZONS):
    """Raw metric rows (the eval schema: speed/pos RMSE, collisions,
    LC, both delay metrics, TTC<3, hard brake) for ONE rollout at both
    horizons.  The truth sides are member-independent and are reused by
    the TRAIN climatology."""
    import engine as E
    ra_, k0 = r.ra, r.k0
    snaps = D.snaps_to_dicts(r, K_TICKS)
    births = D.births_from_engine(r, K_TICKS)
    rows = D.truth_rows(ra_, k0, K_TICKS)
    lc_t = D.lc_ticks_from_ra(ra_)
    bstates = D.birth_states_from_ra(ra_, k0, K_TICKS)
    out = []
    for H in horizons:
        kh = int(round(H / 0.5))
        dp, dt_, np_, nt_ = D.phys_delay_sums(
            snaps, births, rows, lc_t, bstates, k0, kh)
        m = E.window_metrics(r.snapshots, ra_, k0, H)
        out.append(dict(
            variant=variant, member=mi, root=root, t0=int(t0), H=H,
            speed_rmse=m["speed_rmse"], pos_rmse=m["pos_rmse"],
            collisions_per_1000=m["collisions_per_1000"],
            pred_lc=m["pred_lc"], true_lc=m["true_lc"],
            delay_pred=m["delay_pred"], delay_true=m["delay_true"],
            pred_vehicle_steps=m["pred_vehicle_steps"],
            matched_vehicle_steps=m["matched_vehicle_steps"],
            delay_phys_pred=dp, delay_phys_true=dt_,
            phys_steps_pred=np_, phys_steps_true=nt_,
            pred_ttc3_exposure=m["pred_ttc3_exposure"],
            true_ttc3_exposure=m["true_ttc3_exposure"],
            pred_hard_brake_s=m["pred_hard_brake_s"],
            true_hard_brake_s=m["true_hard_brake_s"],
            any_ttc3_pred=m["any_ttc3_pred"],
            any_ttc3_true=m["any_ttc3_true"],
            any_hb_pred=m["any_hb_pred"], any_hb_true=m["any_hb_true"],
            lane_frac=m["lane_frac"], dawdle_scale=dw,
            engine=engine_note))
    return out


# ----------------------------------------------------------- freeze gate --

def sha256_text(text):
    """sha256 over the LF text (the part 3m convention: the text is
    written with write_text and re-read with read_text, so the CRLF
    round-trip cancels and the LF hash is stable)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_frozen(path, extra_note):
    """FREEZE: write the sha256 sidecar next to `path` (LF-text hash +
    UTC timestamp).  Returns the sidecar dict."""
    from datetime import datetime, timezone
    text = Path(path).read_text(encoding="utf-8")
    side = dict(sha256=sha256_text(text),
                frozen_utc=datetime.now(timezone.utc).isoformat(),
                file=str(path), note=extra_note)
    Path(str(path) + ".FROZEN.sha256").write_text(
        json.dumps(side, indent=1), encoding="utf-8")
    return side


def check_frozen(path, sidecar_path):
    """FREEZE GATE: `path` must exist AND match its FROZEN sidecar
    hash; raises SystemExit otherwise (eval refuses to run)."""
    path, sidecar_path = Path(path), Path(sidecar_path)
    if not path.exists() or not sidecar_path.exists():
        raise SystemExit(f"FREEZE GATE: {path} / {sidecar_path} missing "
                         f"-- freeze is required BEFORE this step")
    frozen = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sha = sha256_text(path.read_text(encoding="utf-8"))
    if sha != frozen.get("sha256"):
        raise SystemExit(f"FREEZE GATE: {path.name} sha256 {sha} != "
                         f"FROZEN {frozen.get('sha256')} -- modified "
                         f"after freezing")
    return frozen


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
