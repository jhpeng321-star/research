"""T2: rollout-drift measurement on the 21 holdout runs (WD-01 part 1).

Protocol frozen in REGISTRATION.md (hashed in results/registration_record.json
BEFORE this script exists/rns).  Per (run, t0 in {120..480 step 60}): vehicle
set frozen at t0, one autoregressive 120-step rollout per variant (RAW,
CONSTRAINED); leaders recomputed each step from predicted states within the
frozen lane group.  Every window H in {30, 60, 90, 120} is the h<=H prefix of
the same rollout.

Writes results/t2_summary.json, t2_per_horizon.csv, t2_per_run.csv,
t2_per_step.csv.
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import wd1lib as L

H_MAX = 120
VARIANTS = ("RAW", "CONSTRAINED")


def train_group_caps():
    """max v per lane group over the SURVIVING train npz (the model's data)."""
    cap = {g: 0.0 for g in range(4)}
    for f in sorted((L.WM / "tracks" / "train").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        for g in range(4):
            m = z["lane_group"] == g
            if m.any():
                cap[g] = max(cap[g], float(z["v_self"][m].max()))
    return cap


def raw_passthrough(d, g):
    """Geometric pass-through flags on RAW states (counted, not corrected)."""
    ov = np.zeros(len(d), dtype=bool)
    for grp in np.unique(g):
        members = np.where(g == grp)[0]
        order = members[np.argsort(d[members], kind="stable")]
        for j in range(1, len(order)):
            i, lead = int(order[j]), int(order[j - 1])
            if d[i] <= d[lead]:
                ov[i] = True
    return ov


def ttc_ids(d, v, g, w):
    """Indices with closing TTC < 3 s at one tick (in-group leader required)."""
    _, gap, free = w.leader_features(d, v, g)
    v_lead, _, _ = w.leader_features(d, v, g)
    out = []
    for i in range(len(d)):
        if free[i]:
            continue
        closing = v[i] - v_lead[i]
        if closing > 0.0 and gap[i] / closing < 3.0:
            out.append(i)
    return out


class Roll:
    """One autoregressive 120-step rollout in one variant."""

    def __init__(self, d0, v0, g0):
        self.d = [d0.copy()]
        self.v = [v0.copy()]
        self.g0 = g0
        self.ov = [np.zeros(len(d0), dtype=bool)]

    def model_step(self, model, x_mean, x_std, w, aid, dev, variant, v_cap,
                   fwd_timer):
        d_prev, v_prev = self.d[-1], self.v[-1]
        v_lead, gap, free = w.leader_features(d_prev, v_prev, self.g0)
        tmod = L.train_mod()
        x = tmod.features(v_prev, v_lead, gap, self.g0, free,
                          np.full(len(v_prev), aid, dtype=np.int8))
        xn = torch.from_numpy((x - x_mean) / x_std).to(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            dv = model(xn).cpu().numpy()
        fwd_timer[0] += time.perf_counter() - t0
        fwd_timer[1] += len(dv)
        self._integrate(dv, variant, v_cap)

    def baseline_step(self, variant, v_cap):
        self._integrate(np.zeros(len(self.v[-1])), variant, v_cap)

    def _integrate(self, dv, variant, v_cap):
        d_prev, v_prev = self.d[-1], self.v[-1]
        if variant == "RAW":
            d_n, v_n, _ = L.raw_propagate(d_prev, v_prev, dv)
            ov = raw_passthrough(d_n, self.g0)
        else:
            d_n, v_n, ov = L.constrained_propagate(
                d_prev, v_prev, self.g0, dv, v_cap)
        self.d.append(d_n)
        self.v.append(v_n)
        self.ov.append(ov)

    def step_stats(self, h, w):
        """Per-step stats at step h: events, runaway, error components."""
        d, v = self.d[h], self.v[h]
        ttc = set(ttc_ids(d, v, self.g0, w))
        brake = {int(i) for i in np.where(v - self.v[h - 1] < -3.0)[0]}
        return dict(ttc=ttc, brake=brake,
                    capviol=int(sum((v > np.array([VCAP[g] for g in self.g0]))
                                    ).sum()),
                    neg=int((v < 0).sum()),
                    ovl=int(self.ov[h].sum()),
                    geo=int(raw_passthrough(d, self.g0).sum()),
                    vmax=float(v.max()))


def match_events(truth_events, pred_events, tol=2):
    """1-1 greedy matching of (vid, tick) events within +-tol, per vehicle.

    Returns (n_truth_matched, n_truth, n_pred_matched, n_pred)."""
    truth_by_vid, pred_by_vid = {}, {}
    for vid, t in truth_events:
        truth_by_vid.setdefault(vid, []).append(t)
    for vid, t in pred_events:
        pred_by_vid.setdefault(vid, []).append(t)
    n_tm = n_pm = 0
    for vid, tlist in truth_by_vid.items():
        plist = pred_by_vid.get(vid)
        if not plist:
            continue
        pairs = sorted((abs(t - p), t, p) for t in tlist for p in plist)
        used_t, used_p = set(), set()
        for dt, t, p in pairs:
            if dt > tol:
                break
            if t in used_t or p in used_p:
                continue
            used_t.add(t)
            used_p.add(p)
            n_tm += 1
            n_pm += 1
    return n_tm, len(truth_events), n_pm, len(pred_events)


def main():
    t_start = time.time()
    reg = L.load_json(L.RESULTS / "registration_record.json")
    reg_sha = L.sha256_file(L.HERE / "REGISTRATION.md")
    assert reg["sha256"] == reg_sha, \
        "REGISTRATION.md differs from the hash recorded before scoring!"
    print(f"registration guard ok ({reg_sha[:16]}...)")

    dev = L.device()
    w = L.wm()
    tmod = L.train_mod()
    ck = torch.load(L.SURV_CKPT, map_location="cpu", weights_only=False)
    model = tmod.DynamicsMLP(in_dim=ck["arch"]["in_dim"],
                             hidden=tuple(ck["arch"]["hidden"]))
    model.load_state_dict(ck["model_state"])
    model = model.to(dev).eval()
    x_mean = np.asarray(ck["x_mean"], dtype=np.float32)
    x_std = np.asarray(ck["x_std"], dtype=np.float32)

    global VCAP
    cap_base = train_group_caps()
    VCAP = [1.2 * cap_base[g] for g in range(4)]
    print("train group max v:", {g: round(cap_base[g], 2) for g in range(4)},
          "-> v_cap:", [round(c, 2) for c in VCAP], flush=True)

    tmap = L.load_json(L.RESULTS / "tag_run_map.json")
    holdout = [r for r in tmap["rows"] if r["split"] == "holdout"]
    assert len(holdout) == 21

    fwd_timer = [0.0, 0]          # seconds, vehicle-steps (model only)
    step_rows = []                # per (run, t0, variant, h)
    event_store = {}              # (tag, t0) -> per-variant/roth truth events

    for r in holdout:
        tag, aid = r["tag"], r["action_idx"]
        states = L.cache_to_states(L.RESULTS / "states_cache" / "holdout" /
                                   f"{tag}.npz")
        for t0 in L.START_TICKS:
            snap = states.get(t0)
            ids = sorted(snap)
            d0 = np.array([snap[i][0] for i in ids], dtype=np.float64)
            v0 = np.array([snap[i][1] for i in ids], dtype=np.float64)
            g0 = np.array([snap[i][2] for i in ids], dtype=np.int64)
            n0 = len(ids)

            comp = np.zeros((H_MAX + 1, n0), dtype=bool)
            t_d = np.full((H_MAX + 1, n0), np.nan)
            t_v = np.full((H_MAX + 1, n0), np.nan)
            for h in range(H_MAX + 1):
                sn = states.get(t0 + h)
                if not sn:
                    continue
                for i, vid in enumerate(ids):
                    nx = sn.get(vid)
                    if nx is not None and nx[2] == g0[i]:
                        comp[h, i] = True
                        t_d[h, i] = nx[0]
                        t_v[h, i] = nx[1]

            # truth events per step, from the FULL truth snapshot (all truth
            # vehicles, truth lane groups and truth leaders) per REGISTRATION
            # section 3 -- not the comparable subset
            truth_ttc, truth_brake = {}, {}
            for h in range(1, H_MAX + 1):
                sn = states.get(t0 + h)
                if not sn:
                    truth_ttc[h], truth_brake[h] = set(), set()
                    continue
                ids_h = sorted(sn)
                d_h = np.array([sn[i][0] for i in ids_h], dtype=np.float64)
                v_h = np.array([sn[i][1] for i in ids_h], dtype=np.float64)
                g_h = np.array([sn[i][2] for i in ids_h], dtype=np.int64)
                truth_ttc[h] = {ids_h[i] for i in ttc_ids(d_h, v_h, g_h, w)}
                pv = states.get(t0 + h - 1)
                truth_brake[h] = {
                    vid for vid in ids_h
                    if vid in pv and sn[vid][1] - pv[vid][1] < -3.0}

            ev_key = (tag, t0)
            event_store[ev_key] = dict(truth_ttc=truth_ttc,
                                       truth_brake=truth_brake,
                                       pred={})
            for variant in VARIANTS:
                roll = Roll(d0, v0, g0)
                bl = Roll(d0, v0, g0)
                for _ in range(H_MAX):
                    roll.model_step(model, x_mean, x_std, w, aid, dev,
                                    variant, VCAP, fwd_timer)
                for _ in range(H_MAX):
                    bl.baseline_step(variant, VCAP)
                event_store[ev_key]["pred"][variant] = dict(
                    ttc={h: {ids[i] for i in roll.step_stats(h, w)["ttc"]}
                         for h in range(1, H_MAX + 1)},
                    brake={h: {ids[i] for i in roll.step_stats(h, w)["brake"]}
                           for h in range(1, H_MAX + 1)})
                for h in range(1, H_MAX + 1):
                    st = roll.step_stats(h, w)
                    m = comp[h]
                    n_cmp = int(m.sum())
                    if n_cmp:
                        e_v = roll.v[h][m] - t_v[h][m]
                        e_d = roll.d[h][m] - t_d[h][m]
                        sse_v = float((e_v ** 2).sum())
                        sse_d = float((e_d ** 2).sum())
                        bm = bl  # baseline comparable set identical
                        bl_e_v = bl.v[h][m] - t_v[h][m]
                        bl_e_d = bl.d[h][m] - t_d[h][m]
                        bl_sse_v = float((bl_e_v ** 2).sum())
                        bl_sse_d = float((bl_e_d ** 2).sum())
                    else:
                        sse_v = sse_d = bl_sse_v = bl_sse_d = 0.0
                    step_rows.append(dict(
                        run=tag, t0=t0, variant=variant, h=h, n0=n0,
                        n_cmp=n_cmp, sse_v=sse_v, sse_d=sse_d,
                        bl_sse_v=bl_sse_v, bl_sse_d=bl_sse_d,
                        capviol=st["capviol"], neg=st["neg"],
                        ovl=st["ovl"], geo=st["geo"], vmax=st["vmax"],
                        lost=int((~m).sum())))
        print(f"  {tag} done ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------- aggregation per (variant, H) ------------------------
    def evals():
        return sorted({(r["run"], r["t0"]) for r in step_rows})

    all_evals = evals()
    verdicts = {}
    for variant in VARIANTS:
        for H in L.HORIZONS:
            rs = [r for r in step_rows if r["variant"] == variant
                  and r["h"] <= H]
            n_steps = sum(r["n0"] * 1 for r in rs)  # vehicle-steps
            n_cmp_steps = sum(r["n_cmp"] for r in rs)
            # C3 at step H (mean over evals of per-eval RMSE at h == H)
            sp_at, bl_at, ps_at, bl_ps_at = [], [], [], []
            for (tg, tt) in all_evals:
                row = next((r for r in step_rows if r["run"] == tg
                            and r["t0"] == tt and r["variant"] == variant
                            and r["h"] == H), None)
                if row and row["n_cmp"]:
                    sp_at.append((row["sse_v"] / row["n_cmp"]) ** 0.5)
                    bl_at.append((row["bl_sse_v"] / row["n_cmp"]) ** 0.5)
                    ps_at.append((row["sse_d"] / row["n_cmp"]) ** 0.5)
                    bl_ps_at.append((row["bl_sse_d"] / row["n_cmp"]) ** 0.5)
            # C1 over the window
            ttc_t = ttc_p = ttc_m = brk_t = brk_p = brk_m = 0
            for (tg, tt) in all_evals:
                ev = event_store[(tg, tt)]
                pt = {v: h for h, ids_ in ev["pred"][variant]["ttc"].items()
                      if h <= H for v in ids_}
                tt_ = {v: h for h, ids_ in ev["truth_ttc"].items()
                       if h <= H for v in ids_}
                pb = {v: h for h, ids_ in ev["pred"][variant]["brake"].items()
                      if h <= H for v in ids_}
                tb = {v: h for h, ids_ in ev["truth_brake"].items()
                      if h <= H for v in ids_}
                m1 = match_events(list(tt_.items()), list(pt.items()), tol=2)
                m2 = match_events(list(tb.items()), list(pb.items()), tol=2)
                ttc_t += m1[1]
                ttc_p += m1[3]
                ttc_m += m1[2]
                tm_t = m1[0]
                brk_t += m2[1]
                brk_p += m2[3]
                brk_m += m2[2]
            rec_ttc = ttc_m / ttc_t if ttc_t else float("nan")
            prec_ttc = ttc_m / ttc_p if ttc_p else float("nan")
            rec_br = brk_m / brk_t if brk_t else float("nan")
            prec_br = brk_m / brk_p if brk_p else float("nan")
            # C2 over the window
            capviol = sum(r["capviol"] for r in rs)
            neg = sum(r["neg"] for r in rs)
            ovl = sum(r["ovl"] for r in rs)
            geo = sum(r["geo"] for r in rs)
            vmax = max((r["vmax"] for r in rs), default=0.0)
            lost = sum(r["lost"] for r in rs)
            # NOTE: ovl from CONSTRAINED integrator; geo = geometric RAW
            ov_use = ovl if variant == "CONSTRAINED" else geo
            sp = float(np.mean(sp_at)) if sp_at else float("nan")
            bl_sp = float(np.mean(bl_at)) if bl_at else float("nan")
            c1 = bool(rec_ttc >= 0.80 and rec_br >= 0.80)
            c2 = bool(capviol == 0 and ov_use / max(n_steps, 1) * 1000 <= 1.0)
            c3 = bool(sp <= bl_sp)
            verdicts[f"{variant}@H{H}"] = dict(
                n_evals=len(all_evals),
                speed_rmse_at_H_mps=sp, base_speed_rmse_at_H_mps=bl_sp,
                pos_rmse_at_H_m=float(np.mean(ps_at)) if ps_at else None,
                base_pos_rmse_at_H_m=float(np.mean(bl_ps_at)) if bl_ps_at else None,
                ttc_truth=ttc_t, ttc_pred=ttc_p, ttc_matched=ttc_m,
                ttc_recall=rec_ttc, ttc_precision=prec_ttc,
                brake_truth=brk_t, brake_pred=brk_p, brake_matched=brk_m,
                brake_recall=rec_br, brake_precision=prec_br,
                vehicle_steps=n_steps, comparable_vehicle_steps=n_cmp_steps,
                lost_vehicle_step_frac=lost / max(n_steps, 1),
                capviol_steps=capviol, negspeed_steps=neg,
                overlap_events=ov_use,
                overlap_per_1000=ov_use / max(n_steps, 1) * 1000,
                raw_integrator_geo_passthrough=geo if variant == "RAW" else None,
                max_speed_mps=vmax,
                C1_recall_pass=c1, C2_runaway_pass=c2, C3_accuracy_pass=c3,
                H_pass=bool(c1 and c2 and c3),
            )

    passing = [H for H in L.HORIZONS if verdicts[f"CONSTRAINED@H{H}"]["H_pass"]]
    if passing:
        rec_h = max(passing)
        recommendation = dict(
            part_a_horizon_s=rec_h,
            basis="longest registered H with C1&C2&C3 pass in CONSTRAINED",
            constained_passing=passing,
            raw_also_passes_at_recommendation=bool(
                verdicts[f"RAW@H{rec_h}"]["H_pass"]))
    else:
        fails = {}
        for H in L.HORIZONS:
            v = verdicts[f"CONSTRAINED@H{H}"]
            fails[H] = [c for c, k in (("C1", "C1_recall_pass"),
                                       ("C2", "C2_runaway_pass"),
                                       ("C3", "C3_accuracy_pass")) if not v[k]]
        recommendation = dict(
            part_a_horizon_s=None,
            note="no registered horizon passes C1&C2&C3 in CONSTRAINED",
            failing_criteria_by_H={H: fails[H] for H in L.HORIZONS},
            fallback_bound_s=max([H for H in L.HORIZONS if not fails[H]],
                                 default=None))

    summary = dict(
        experiment="WD-01 part 1 T2 rollout drift",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        registration_sha256=reg["sha256"],
        checkpoint=str(L.SURV_CKPT), ckpt_sha256=L.SURV_CKPT_SHA256,
        device=str(dev),
        train_group_max_v={str(g): cap_base[g] for g in range(4)},
        v_cap={str(g): VCAP[g] for g in range(4)},
        n_runs=len(holdout), start_ticks=list(L.START_TICKS),
        horizons=list(L.HORIZONS), n_evals_per_H=len(all_evals),
        model_forward_seconds=round(fwd_timer[0], 2),
        model_forward_vehicle_steps=fwd_timer[1],
        throughput_vehicle_steps_per_s=(fwd_timer[1] / fwd_timer[0])
        if fwd_timer[0] else None,
        rollout_wall_seconds=round(time.time() - t_start, 1),
        verdicts=verdicts, recommendation=recommendation,
    )
    L.dump_json(L.RESULTS / "t2_summary.json", summary)

    with (L.RESULTS / "t2_per_run.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
        cols = list(step_rows[0].keys())
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(step_rows)

    with (L.RESULTS / "t2_per_horizon.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        cols = ["variant", "H", "speed_rmse_at_H_mps",
                "base_speed_rmse_at_H_mps", "pos_rmse_at_H_m",
                "base_pos_rmse_at_H_m", "ttc_recall", "ttc_precision",
                "brake_recall", "brake_precision", "capviol_steps",
                "negspeed_steps", "overlap_events", "overlap_per_1000",
                "max_speed_mps", "lost_vehicle_step_frac", "C1_recall_pass",
                "C2_runaway_pass", "C3_accuracy_pass", "H_pass"]
        wr = csv.writer(fh)
        wr.writerow(cols)
        for variant in VARIANTS:
            for H in L.HORIZONS:
                v = verdicts[f"{variant}@H{H}"]
                wr.writerow([variant, H] + [v[c] for c in cols[2:]])

    with (L.RESULTS / "t2_per_step.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["variant", "h", "mean_speed_rmse_mps",
                     "mean_base_speed_rmse_mps", "mean_pos_rmse_m",
                     "mean_base_pos_rmse_m", "lost_frac", "capviol",
                     "negspeed", "overlaps", "max_speed_mps"])
        for variant in VARIANTS:
            for h in range(1, H_MAX + 1):
                rs = [r for r in step_rows
                      if r["variant"] == variant and r["h"] == h]
                rws = [r for r in rs if r["n_cmp"]]
                sp = float(np.mean([(r["sse_v"] / r["n_cmp"]) ** 0.5
                                    for r in rws])) if rws else ""
                bsp = float(np.mean([(r["bl_sse_v"] / r["n_cmp"]) ** 0.5
                                     for r in rws])) if rws else ""
                ps = float(np.mean([(r["sse_d"] / r["n_cmp"]) ** 0.5
                                    for r in rws])) if rws else ""
                bps = float(np.mean([(r["bl_sse_d"] / r["n_cmp"]) ** 0.5
                                     for r in rws])) if rws else ""
                steps = sum(r["n0"] for r in rs)
                wr.writerow([variant, h, sp, bsp, ps, bps,
                             sum(r["lost"] for r in rs) / max(steps, 1),
                             sum(r["capviol"] for r in rs),
                             sum(r["neg"] for r in rs),
                             sum(r["ovl"] if variant == "CONSTRAINED"
                                 else r["geo"] for r in rs),
                             max((r["vmax"] for r in rs), default=0.0)])

    print(json.dumps(dict(verdicts={k: {kk: vv for kk, vv in v.items()
                                        if not isinstance(vv, dict)}
                                    for k, v in verdicts.items()},
                          recommendation=recommendation,
                          throughput=summary["throughput_vehicle_steps_per_s"]),
                      indent=1, default=str)[:5200])
    print(f"T2 done in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
