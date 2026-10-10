"""WD-01 part 3 REGISTERED holdout evaluation (runs ONCE).

Guards: results/registration_p3_record.json must carry the sha256 of
REGISTRATION_P3.md, and results/holdout_eval.json must not exist yet.
Variants: model members, ensemble mean, B1 persistence, B2 IDM normal,
B3 IDM oracle (vType reference only), B4 part-1 v3 checkpoint at 1 s.
Horizons 30/60/90/120 s are prefixes of one rollout per (window, variant).

Ensemble: per-window mean of scalar metrics; C3 RMSE also computed on the
member-averaged trajectory (s/v averaged per (vid,tick), lane = member
median); U coverage = fraction of windows where the truth metric lies in
the member [min, max] and within mean +/- 2 sd (TTC<3 exposure, delay).
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import p3lib as P  # noqa: E402  (inserts the p2C path first)
import pdl as PD  # noqa: E402
import engine as E  # noqa: E402

T0 = time.time()
BASELINES = ("B1_persistence", "B2_idm_normal", "B3_idm_oracle", "B4_v3_1s")
KINDS = {"B1_persistence": "persistence", "B2_idm_normal": "idm_normal",
         "B3_idm_oracle": "idm_oracle"}
ENS_KEY = "ens_mean"


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def corr(a, b, kind):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    if kind == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def ens_snapshot(member_snaps, j):
    ids = member_snaps[0][j]["vid"]
    M = len(member_snaps)
    S = np.zeros((len(ids), M))
    V = np.zeros((len(ids), M))
    L = np.zeros((len(ids), M), dtype=np.int64)
    ok = np.zeros((len(ids), M), dtype=bool)
    for mi, per_tick in enumerate(member_snaps):
        sn = per_tick[j]
        pos = {int(v): i for i, v in enumerate(sn["vid"])}
        for r, v_ in enumerate(ids):
            i = pos.get(int(v_))
            if i is None:
                continue
            ok[r, mi] = True
            S[r, mi] = sn["s"][i]
            V[r, mi] = sn["v"][i]
            L[r, mi] = sn["lane"][i]
    any_ok = ok.any(1)
    return dict(
        vid=ids[any_ok].copy(),
        lane=np.array([np.median(L[r][ok[r]]) for r in range(len(ids))
                       if any_ok[r]]),
        s=np.array([S[r][ok[r]].mean() for r in range(len(ids))
                    if any_ok[r]]),
        v=np.array([V[r][ok[r]].mean() for r in range(len(ids))
                    if any_ok[r]]),
        a=np.zeros(int(any_ok.sum())),
        collisions=float(np.mean([sn[j]["collisions"]
                                  for sn in member_snaps])),
        capviol=float(np.mean([sn[j]["capviol"] for sn in member_snaps])),
        lc=float(np.mean([sn[j]["lc"] for sn in member_snaps])),
        n_birthed=member_snaps[0][j]["n_birthed"])


def agg_variant(rows):
    n = len(rows)
    out = dict(n_windows=n)
    for k in ("pred_vehicle_steps", "matched_vehicle_steps",
              "pred_collisions", "pred_lc", "true_lc",
              "pred_ttc3_exposure", "true_ttc3_exposure",
              "pred_ttc15_exposure", "true_ttc15_exposure",
              "pred_hard_brake_s", "true_hard_brake_s",
              "delay_pred", "delay_true"):
        out[k] = float(np.mean([r[k] for r in rows]))
    out["collisions_per_1000"] = float(np.mean(
        [r["collisions_per_1000"] for r in rows]))
    out["speed_rmse"] = float(np.sqrt(np.mean(
        [r["speed_rmse"] ** 2 for r in rows])))
    out["pos_rmse"] = float(np.sqrt(np.mean(
        [r["pos_rmse"] ** 2 for r in rows])))
    out["lane_frac"] = float(np.mean([r["lane_frac"] for r in rows]))
    out["capviol_mean"] = float(np.mean([r["capviol"] for r in rows]))
    for f in ("ttc15", "ttc3", "hb"):
        t = [r[f"any_{f}_true"] for r in rows]
        pr = [r[f"any_{f}_pred"] for r in rows]
        tp = sum(1 for x, y in zip(t, pr) if x and y)
        out[f"any_{f}_recall"] = tp / sum(t) if sum(t) else None
        out[f"any_{f}_precision"] = tp / sum(pr) if sum(pr) else None
        out[f"any_{f}_base_rate"] = sum(t) / n
    for k in ("ttc3_exposure", "ttc15_exposure", "hard_brake_s"):
        t = [r[f"true_{k}"] for r in rows]
        pr = [r[f"pred_{k}"] for r in rows]
        out[f"{k}_bias"] = float(np.mean(np.asarray(pr) - np.asarray(t)))
        out[f"{k}_pearson"] = corr(t, pr, "pearson")
        out[f"{k}_spearman"] = corr(t, pr, "spearman")
    pair = [(r["true_min_ttc"], r["pred_min_ttc"]) for r in rows
            if r["true_min_ttc"] is not None and r["pred_min_ttc"] is not None]
    if pair:
        tm = [x for x, _ in pair]
        pm = [y for _, y in pair]
        out["min_ttc_bias"] = float(np.mean(np.asarray(pm) - np.asarray(tm)))
        out["min_ttc_pearson"] = corr(tm, pm, "pearson")
        out["min_ttc_spearman"] = corr(tm, pm, "spearman")
    else:
        out["min_ttc_bias"] = out["min_ttc_pearson"] = None
        out["min_ttc_spearman"] = None
    out["delay_bias"] = out["delay_pred"] - out["delay_true"]
    out["delay_pearson"] = corr([r["delay_true"] for r in rows],
                                [r["delay_pred"] for r in rows], "pearson")
    out["delay_spearman"] = corr([r["delay_true"] for r in rows],
                                 [r["delay_pred"] for r in rows], "spearman")
    return out


def coverage(rows_ens, key, truth_key):
    mm = cov1 = cov2 = 0
    for r in rows_ens:
        vals = r.get(key)
        if not vals:
            continue
        t = r[truth_key]
        mm += 1
        if min(vals) <= t <= max(vals):
            cov1 += 1
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        if mu - 2 * sd <= t <= mu + 2 * sd:
            cov2 += 1
    return (round(cov1 / max(mm, 1), 4), round(cov2 / max(mm, 1), 4))


def main():
    rec_path = HERE / "results" / "registration_p3_v2_record.json"
    if not rec_path.exists():
        print("ABORT: write REGISTRATION_P3_v2.md and record its sha256 "
              "BEFORE the holdout evaluation")
        return 2
    rec = P.load_json(rec_path)
    reg_sha = P.sha256_file(HERE / "REGISTRATION_P3_v2.md")
    if reg_sha != rec["sha256"]:
        print("ABORT: REGISTRATION_P3_v2.md changed after registration")
        return 2
    out_path = HERE / "results" / "holdout_eval.json"
    if out_path.exists():
        print("ABORT: holdout_eval.json already exists (one-shot)")
        return 2
    ck = P.sha256_file(P.CKPT)
    assert ck == P.CKPT_SHA256
    man, man_sha = P.guards()
    log(f"registration ok ({reg_sha[:12]}...); guards ok")

    dev = P.device()
    splits = P.roots_by_split()
    holdout = splits["holdout"]
    members = sorted(f.name for f in (HERE / "weights" / "run2").glob("member*.pt"))
    assert len(members) >= 3, "expected >=3 trained members"
    log(f"holdout {holdout}; members {members}")

    variants = ([f"member{i}" for i in range(len(members))]
                + [ENS_KEY] + list(BASELINES))
    rows = {}
    fwd = dict(steps=0, t=0.0)

    for root in holdout:
        for t0 in P.EVAL_T0S:
            member_snaps, member_mets = [], []
            for w in members:
                t = time.time()
                r = E.Rollout(root, t0, kind="model",
                              weights=HERE / "weights" / "run2" / w,
                              dev=dev).run()
                fwd["steps"] += 240 * len(r.snapshots[0]["vid"])
                fwd["t"] += time.time() - t
                member_snaps.append(r.snapshots)
                member_mets.append({H: E.window_metrics(r.snapshots, r.ra,
                                                        r.k0, H)
                                    for H in P.HORIZONS})
            k05 = int(round((t0 - PD.WINDOW_T0) / 0.5))
            k1 = int(round((t0 - PD.WINDOW_T0) / 1.0))
            ens_snaps = {j: ens_snapshot(member_snaps, j)
                         for j in range(0, 241)}
            log(f"{root} t0={t0}: rollouts done ({time.time() - T0:.0f}s)")
            for variant in variants:
                if variant == ENS_KEY:
                    snaps, kk, dt, stride = ens_snaps, k05, 0.5, 1
                elif variant.startswith("member"):
                    i = int(variant.replace("member", ""))
                    snaps, kk, dt, stride = member_snaps[i], k05, 0.5, 1
                elif variant == "B4_v3_1s":
                    raw = E.V3Baseline(root, t0, dev=dev).run().snapshots
                    snaps = {}
                    for j in range(0, 241):
                        base = raw[min(j // 2, 120)]
                        snaps[j] = dict(base)
                    kk, dt, stride = 2 * k1, 0.5, 1
                else:
                    snaps = E.Rollout(root, t0, kind=KINDS[variant],
                                      dev=dev).run().snapshots
                    kk, dt, stride = k05, 0.5, 1
                ra = P.root_arrays(root)
                for H in P.HORIZONS:
                    m = E.window_metrics(snaps, ra, kk, H, dt=dt,
                                         stride=stride)
                    if variant == ENS_KEY:
                        m["member_ttc3"] = [mm[H]["pred_ttc3_exposure"]
                                            for mm in member_mets]
                        m["member_delay"] = [mm[H]["delay_pred"]
                                             for mm in member_mets]
                    rows.setdefault((variant, H), []).append(m)
            log(f"{root} t0={t0}: metrics done")

    def agg(variant, H):
        rr = rows[(variant, H)]
        out = agg_variant(rr)
        if variant == ENS_KEY:
            c1, c2 = coverage(rr, "member_ttc3", "true_ttc3_exposure")
            out["cov_member_ttc3_minmax"], out["cov_member_ttc3_pm2sd"] = \
                c1, c2
            d1, d2 = coverage(rr, "member_delay", "delay_true")
            out["cov_member_delay_minmax"], out["cov_member_delay_pm2sd"] = \
                d1, d2
        return out

    per_vh = {f"{v}@H{h}": agg(v, h) for v in variants for h in P.HORIZONS}
    result = dict(
        registered_sha256=reg_sha, checkpoint_sha256=ck,
        p2c_manifest_sha256=man_sha, holdout_roots=holdout,
        t0s=list(P.EVAL_T0S), horizons=list(P.HORIZONS), members=members,
        throughput_vehicle_steps_per_s=(fwd["steps"] / fwd["t"]
                                        if fwd["t"] else None),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        per_variant_H=per_vh)
    P.dump_json(out_path, result)

    rec_out = {}
    for H in P.HORIZONS:
        a = per_vh[f"{ENS_KEY}@H{H}"]
        b1 = per_vh[f"B1_persistence@H{H}"]
        b2 = per_vh[f"B2_idm_normal@H{H}"]
        bench = min(b1["speed_rmse"], b2["speed_rmse"])
        c2 = a["collisions_per_1000"] <= 1.0
        c3 = a["speed_rmse"] <= bench
        s3r = (a["any_ttc3_recall"] or 0.0) >= 0.90
        sbr = (a["any_hb_recall"] or 0.0) >= 0.90
        s3p = ((a["any_ttc3_precision"] or 0.0)
               > (a["any_ttc3_base_rate"] or 0.0))
        sbp = ((a["any_hb_precision"] or 0.0)
               > (a["any_hb_base_rate"] or 0.0))
        s_pass = s3r and sbr and s3p and sbp
        failing = [c for c, ok in (("C2", c2), ("C3", c3), ("S", s_pass))
                   if not ok]
        rec_out[str(H)] = dict(
            C2_collisions_per_1000=round(a["collisions_per_1000"], 4),
            C2_pass=c2, C3_speed_rmse=round(a["speed_rmse"], 4),
            C3_bench_min_B1_B2=round(bench, 4), C3_pass=c3,
            S=dict(ttc15_recall=a["any_ttc15_recall"],
                   ttc15_precision=a["any_ttc15_precision"],
                   ttc15_base=round(a["any_ttc15_base_rate"], 4),
                   hb_recall=a["any_hb_recall"],
                   hb_precision=a["any_hb_precision"],
                   hb_base=round(a["any_hb_base_rate"], 4)),
            S_pass=s_pass, failing=failing)
    passing = [H for H in P.HORIZONS if not rec_out[str(H)]["failing"]]
    recommendation = dict(
        primary_variant=ENS_KEY,
        part_a_horizon_s=max(passing) if passing else None,
        horizons_passing=passing,
        first_failing_H=(min(H for H in P.HORIZONS
                             if rec_out[str(H)]["failing"])
                         if not passing else None),
        per_H=rec_out)
    P.dump_json(HERE / "results" / "recommendation.json", recommendation)

    with (HERE / "results" / "holdout_per_H.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        head = ["variant", "H", "speed_rmse", "pos_rmse", "lane_frac",
                "collisions_per_1000", "lc_pred", "lc_true",
                "ttc15_recall", "ttc15_precision", "ttc15_base",
                "ttc3_recall", "ttc3_precision", "ttc3_base",
                "hb_recall", "hb_precision", "hb_base",
                "ttc3_exp_bias", "ttc3_pearson", "ttc3_spearman",
                "hb_s_bias", "min_ttc_bias", "min_ttc_pearson",
                "delay_bias", "delay_pearson",
                "cov_member_ttc3_minmax", "cov_member_ttc3_pm2sd",
                "cov_member_delay_minmax", "cov_member_delay_pm2sd"]
        wr.writerow(head)
        for v in variants:
            for H in P.HORIZONS:
                a = per_vh[f"{v}@H{H}"]
                wr.writerow([v, H] + [("" if a.get(k) is None
                                       else round(a[k], 4)
                                       if isinstance(a[k], float) else a[k])
                                      for k in head[2:]])

    log(f"EVAL DONE in {time.time() - T0:.0f}s; throughput "
        f"{result['throughput_vehicle_steps_per_s']:.0f} veh-steps/s")
    print(json.dumps(recommendation, indent=1, default=str)[:2600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
