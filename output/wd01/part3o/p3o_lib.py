"""WD-01 part 3o shared library: the LC over-count fix (gate v2 (b))
of the idm_merge emulator via PER-TYPE hazard levers, everything else
inherited from part 3n.

New engine levers (part 3o, both default no-op -- earlier kinds and
all prior-part configs reproduce byte-identically):
  lc_scale_disc      scales ONLY the idm_merge discretionary 0<->1
                     hazard (M4); None falls back to lc_scale.
  merge_accept_prob  thins gap-accepted M2 merges (1.0 = shipped
                     deterministic acceptance); draws come from a
                     dedicated RNG stream, so flag==1.0 never touches
                     any shared stream.

Split declaration (inherited from p3n, UNCHANGED):
  fit roots (refit)   = cv17, cv19_4107, cv19_4108, cv20_7001m2000
  TRAIN-internal gate = 4 calibration roots, 2 folds
                        fold A = cv20_7002m2000, cv20_7003m2000
                        fold B = cv20_7004m2000, cv20_7007m2700
                        t0 120..480 step 60 -> 28 windows, M=5,
                        H {30, 60}
  validation          = the 4 p3l validation roots -- NEVER loaded or
                        scored in part 3o (no validation look)
  holdout             = blocked in p3lib.root_arrays (never passed)

ZERO simulation: no SUMO/TraCI anywhere; the only dynamics is the
part-3 rollout engine (pure torch).  The 2-fold gate protocol (scored
fold never fits its own spread factors) is N.gate_verdicts, inherited
verbatim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "part3n"), str(HERE.parent / "part3k"),
           str(HERE.parent / "part3l"), str(HERE.parent / "part3m"),
           str(HERE.parent / "part3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import p3n_lib as N  # noqa: E402  (split + gate + spread + freeze machinery)
import decompose_p3k as D  # noqa: E402  (Tee, priority, stats helpers)
import p3lib as P3  # noqa: E402
import p3l_lib as L3  # noqa: E402
import p3m_lib as M  # noqa: E402

RES = HERE / "results"
RAW_DIR = RES / "raw"
PART3N = HERE.parent / "part3n"

HORIZONS = N.HORIZONS
GATE_T0S = N.GATE_T0S
K_TICKS = N.K_TICKS
FOLD_A, FOLD_B, FOLDS = N.FOLD_A, N.FOLD_B, N.FOLDS
OUTPUTS = N.OUTPUTS
LC_TYPE_KEYS = ("pred_lc_merge", "pred_lc_disc",
                "true_lc_merge", "true_lc_disc")
# the p3o goal package: LC by type AND total within this band at BOTH
# horizons on the TRAIN-internal folds (brief: +-20 %, ideally +-10 %)
LC_BAND = 0.20
LC_BAND_IDEAL = 0.10


def splits():
    return N.L3.splits_3l()


def p3n_iter1_reference():
    """Committed p3n iteration-1 TRAIN-internal gate numbers -- the
    'delay CRPS no worse than p3n iter1' reference (read at runtime,
    never hardcoded)."""
    p = PART3N / "results" / "gate_train_internal_iter1.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- the p3n iter1 reference gate")
    g = N.load_json(p)["gate_v2"]
    return {H: dict(crps=g[f"H{H}"]["crps_phys"],
                    speed_rmse=g[f"H{H}"]["speed_rmse"],
                    lc_dev=g[f"H{H}"]["lc_deviation"],
                    delay_cov=g[f"H{H}"]["delay_phys_cov_frac"],
                    ttc3_cov=g[f"H{H}"]["ttc3_cov_frac"])
            for H in HORIZONS}


# ------------------------------------------------------------- rolling --

def roll_member(root, t0, cfg, dev, K=K_TICKS):
    """One idm_merge rollout from a p3o member config dict (the part 3o
    knobs lc_scale_disc / merge_accept_prob default to the no-op)."""
    import engine as E
    return E.Rollout(root, float(t0), kind="idm_merge",
                     member_seed=int(cfg["seed"]),
                     lc_scale=float(cfg["lc_scale"]),
                     lc_scale_disc=cfg.get("lc_scale_disc"),
                     merge_accept_prob=float(
                         cfg.get("merge_accept_prob", 1.0)),
                     merge_params=dict(cfg["merge_params"]),
                     dawdle_scale=float(cfg.get("dawdle_scale", 1.0)),
                     dev=dev).run(n_ticks=K)


def roll_l0(root, t0, dev, K=K_TICKS):
    import engine as E
    return E.Rollout(root, float(t0), kind="idm_normal",
                     dev=dev).run(n_ticks=K)


def window_raw_rows_typed(r, variant, mi, root, t0, dw, engine_note,
                          horizons=HORIZONS):
    """N.window_raw_rows + the LC-BY-TYPE counts on BOTH sides (pred
    and truth) per horizon row -- the per-type gate table's basis."""
    rows = N.window_raw_rows(r, variant, mi, root, t0, dw, engine_note,
                             horizons)
    pred = N.pred_lc_by_type(r.snapshots, K_TICKS, horizons)
    truth = N.truth_lc_by_type(r.ra, r.k0, K_TICKS, horizons)
    for row in rows:
        H = int(row["H"])
        row["pred_lc_merge"] = pred[H]["pred_merge"]
        row["pred_lc_disc"] = pred[H]["pred_disc"]
        row["true_lc_merge"] = truth[H]["true_merge"]
        row["true_lc_disc"] = truth[H]["true_disc"]
    return rows


def roll_member_rows(root, t0, cfg, mi, variant, dev, note,
                     K=K_TICKS):
    r = roll_member(root, t0, cfg, dev, K)
    return window_raw_rows_typed(
        r, variant, mi, root, t0,
        float(cfg.get("dawdle_scale", 1.0)), note)


# ------------------------------------------------------ by-type tables --

def lc_by_type(rows, H, roots=None, variant="L6"):
    """Pooled mean LC per window by type over the selected rows:
    totals derived from the two types; deviations vs truth."""
    sel = [r for r in rows if r["variant"] == variant and
           int(r["H"]) == H and
           (roots is None or r["root"] in roots)]
    n = max(len(sel), 1)
    out = {k: float(np.mean([r[k] for r in sel]))
           for k in LC_TYPE_KEYS}
    out["pred_lc"] = out["pred_lc_merge"] + out["pred_lc_disc"]
    out["true_lc"] = out["true_lc_merge"] + out["true_lc_disc"]
    out["n_windows"] = len(sel)
    for k in LC_TYPE_KEYS:
        out[k] = round(out[k], 3)
    out["pred_lc"] = round(out["pred_lc"], 3)
    out["true_lc"] = round(out["true_lc"], 3)
    out["dev_total"] = round(
        (out["pred_lc"] - out["true_lc"]) / max(out["true_lc"], 1), 4)
    out["dev_merge"] = round(
        (out["pred_lc_merge"] - out["true_lc_merge"]) /
        max(out["true_lc_merge"], 1), 4)
    out["dev_disc"] = round(
        (out["pred_lc_disc"] - out["true_lc_disc"]) /
        max(out["true_lc_disc"], 1), 4)
    return out


def by_type_gate_block(rows_cal):
    """{H: {fold/pooled -> by-type table}} for the gate table."""
    out = {}
    for H in HORIZONS:
        blk = dict(pooled=lc_by_type(rows_cal, H),
                   fold_A=lc_by_type(rows_cal, H, set(FOLDS["A"])),
                   fold_B=lc_by_type(rows_cal, H, set(FOLDS["B"])))
        out[H] = blk
    return out


# ---------------------------------------------------- the goal package --

def goal_package(rows_cal, verdicts, ref):
    """The part 3o goal package per horizon (the brief's success
    criteria, Train-internal only):
      b_type   LC by type AND total within +-LC_BAND (merge, disc),
               POOLED over the TRAIN-internal calibration split (the
               same pooling as the formal gate-v2 (b) criterion)
      b_total  the formal gate-v2 (b) total-LC criterion
      a        collisions <= 1/1000
      d        speed RMSE <= 1.05 x L0 (per-horizon, binding)
      crps     2-fold delay-phys CRPS <= p3n iter1 (no worse)
    The per-FOLD by-type tables are reported alongside (fold_B's truth
    carries systematically fewer LCs, which global hazard levers cannot
    reconcile per-fold -- see REPORT_P3O.md); TTC coverage /
    climatology are diagnostics (owner decision)."""
    out = {}
    for H in HORIZONS:
        g = verdicts[H]
        bt = {name: lc_by_type(rows_cal, H)
              for name in ("pooled",)}
        bt["fold_A"] = lc_by_type(rows_cal, H, set(FOLDS["A"]))
        bt["fold_B"] = lc_by_type(rows_cal, H, set(FOLDS["B"]))
        for name, tab in bt.items():
            if tab["n_windows"] == 0:
                raise ValueError(f"goal package: no {H}-horizon "
                                 f"windows in {name} -- degenerate "
                                 f"gate split")
        dev = {name: max(abs(bt[name]["dev_merge"]),
                         abs(bt[name]["dev_disc"]),
                         abs(bt[name]["dev_total"]))
               for name in bt}
        checks = dict(
            b_type_pooled=dev["pooled"] <= LC_BAND,
            b_total_gate=bool(g["b_lc_pass"]),
            a_collisions=bool(g["a_collisions_pass"]),
            d_speed=bool(g["d_per_horizon_pass"]),
            crps_not_worse=bool(
                g["crps_phys"] is not None and
                ref[H]["crps"] is not None and
                g["crps_phys"] <= ref[H]["crps"]),
        )
        checks["PASS"] = all(checks.values())
        out[H] = dict(checks=checks,
                      max_type_dev=dict(
                          pooled=dev["pooled"],
                          fold_A=dev["fold_A"],
                          fold_B=dev["fold_B"]),
                      by_type=bt,
                      crps_phys=g["crps_phys"],
                      crps_p3n_iter1=ref[H]["crps"],
                      speed_rmse=g["speed_rmse"],
                      d_threshold=g["d_per_horizon_threshold"],
                      ttc3_cov=g["ttc3_cov_frac"],
                      delay_phys_cov=g["delay_phys_cov_frac"])
    return out


# ------------------------------------------------------ fit statistics --

def truth_fit_targets(roots, t0s, K=K_TICKS):
    """{(root, t0): truth stats} on the fit windows -- no roll needed
    (the truth side is member-independent)."""
    import p3lib as P3
    out = {}
    for root in roots:
        ra = P3.root_arrays(root)
        for t0 in t0s:
            out[(root, float(t0))] = N.truth_fit_stats(root, t0, ra, K)
    return out


def pred_fit_stats_o(root, t0, cfg, dev, K=K_TICKS):
    """N.pred_fit_stats mirrored for a p3o member config (the new
    per-type knobs instead of the bare lc_scale x merge_params)."""
    r = roll_member(root, t0, cfg, dev, K)
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
    """p3n stage1 + stage2 (the by-type LC loss), verbatim maths."""
    l1, d1 = N.loss_stage1(preds, truths)
    l2, d2 = N.loss_p3n(preds, truths)
    return l1 + l2, l1, l2, d1, d2


def loss_line(base_cfg, field, grid, keys, truths, dev, K=K_TICKS):
    """One coordinate line: {value: joint loss} rolling ONLY `field`
    away from the base config over the fit windows."""
    out = {}
    for g in grid:
        cfg = dict(base_cfg)
        if field in ("tau_accept", "b_accept", "end_force_s",
                     "v_merge_blend", "p_coop_scale"):
            cfg["merge_params"] = dict(base_cfg["merge_params"],
                                       **{field: g})
        else:
            cfg[field] = g
        preds = {k: pred_fit_stats_o(k[0], k[1], cfg, dev, K)
                 for k in keys}
        out[g] = joint_loss(preds, truths)
    return out


def softmax_draw(losses, rng, t=0.15):
    return N.softmax_draw(np.asarray(losses, dtype=np.float64), rng, t)


# ------------------------------------------------------------ raw rows --

def read_rows(path):
    return [json.loads(ln) for ln in
            Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def member_file_complete(path, expect_rows):
    """Resumability: a per-member raw file is complete when it holds
    expect_rows parsable rows."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        return len(read_rows(p)) >= expect_rows
    except Exception:
        return False


def write_rows(path, rows, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def cfg_note(cfg):
    return (f"lc {cfg['lc_scale']} disc {cfg.get('lc_scale_disc')} "
            f"mprob {cfg.get('merge_accept_prob', 1.0)} "
            f"tau {cfg['merge_params']['tau_accept']} "
            f"b {cfg['merge_params']['b_accept']} "
            f"force {cfg['merge_params']['end_force_s']} "
            f"blend {cfg['merge_params']['v_merge_blend']} "
            f"pcoop {cfg['merge_params']['p_coop_scale']} "
            f"dawdle {cfg.get('dawdle_scale', 1.0)}")
