"""T1b/T1c evaluator: re-run the verify protocol for a given checkpoint on the
regenerated holdout states (cached), and (optionally) compare against the
logged RESULTS.json.

Protocol details are taken from the source verify module (START_TICKS and
rollout run/start/steps imported from it); the per-step rollout rows are
validated element-wise against RESULTS.json rollout.horizon when which ==
"surviving".  Writes results/identity_eval_<which>.json.

Usage: python identity_eval.py surviving|retrained [ckpt_path]
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import wd1lib as L

TOL = 1e-3


def load_surviving_like(ckpt_path):
    """Load a source-format checkpoint (model_state/x_mean/x_std/arch)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tmod = L.train_mod()
    model = tmod.DynamicsMLP(in_dim=ck["arch"]["in_dim"],
                             hidden=tuple(ck["arch"]["hidden"]))
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, np.asarray(ck["x_mean"], dtype=np.float32), \
        np.asarray(ck["x_std"], dtype=np.float32), ck


def predict_dv(model, x_mean, x_std, v_self, v_leader, gap, lg, free, aid, dev):
    tmod = L.train_mod()
    x = tmod.features(v_self, v_leader, gap, lg, free,
                      np.full(len(v_self), aid, dtype=np.int8))
    xn = torch.from_numpy((x - x_mean) / x_std).to(dev)
    with torch.no_grad():
        out = model(xn)
    return out.cpu().numpy()


def state_arrays(snap):
    ids = sorted(snap)
    d = np.array([snap[i][0] for i in ids], dtype=np.float64)
    v = np.array([snap[i][1] for i in ids], dtype=np.float64)
    g = np.array([snap[i][2] for i in ids], dtype=np.int64)
    return ids, d, v, g


def eval_rollout(model, x_mean, x_std, states, tag, aid, start, steps, dev,
                 w, logged_horizon=None):
    """Autoregressive rollout, leaders from predicted states, clamp=True."""
    _, d0, v0, g0 = state_arrays(states[start])
    w_agg = w
    _, ct_start = w_agg.aggregate_field(d0, v0, g0)
    fixed = ct_start > 0
    d, v, g = d0.copy(), v0.copy(), g0.copy()
    rows = []
    for h in range(steps + 1):
        t_now = start + h
        sp_p, ct_p = w_agg.aggregate_field(d, v, g)
        row = dict(step=h, t=t_now, n_pred_vehicles=int(len(d)),
                   mean_pred_count=float(ct_p.mean()))
        snap = states.get(t_now)
        if snap:
            _, dt, vt, gt = state_arrays(snap)
            sp_t, ct_t = w_agg.aggregate_field(dt, vt, gt)
            m = ct_t > 0
            row["rmse_truth_occupied"] = w_agg.rmse_on(m, sp_p, sp_t)
            row["n_cells"] = int(m.sum())
            row["mean_true_count"] = float(ct_t.mean())
            row["rmse_start_fixed"] = w_agg.rmse_on(m & fixed, sp_p, sp_t)
        rows.append(row)
        if h == steps:
            break
        vl, gap, free = w_agg.leader_features(d, v, g)
        dv = predict_dv(model, x_mean, x_std, v, vl, gap, g, free, aid, dev)
        d, v, _ = L.raw_propagate(d, v, dv)
    out = dict(run=tag, start_t=start, steps=steps, n_start_vehicles=int(len(d0)),
               horizon=rows)
    if logged_horizon is not None and len(logged_horizon) == len(rows):
        devs = []
        for a, b in zip(rows, logged_horizon):
            for k in ("n_pred_vehicles", "mean_pred_count", "n_cells",
                      "mean_true_count", "rmse_truth_occupied", "rmse_start_fixed"):
                if a.get(k) is None or b.get(k) is None:
                    if not (a.get(k) is None and b.get(k) is None):
                        devs.append(f"step{a['step']}:{k} None-mismatch")
                    continue
                dd = abs(float(a[k]) - float(b[k]))
                if dd > TOL:
                    devs.append(f"step{a['step']}:{k} dev={dd:.3e}")
        out["horizon_max_abs_dev"] = max(
            (abs(float(a.get("rmse_truth_occupied") or 0) -
                 float(b.get("rmse_truth_occupied") or 0))
             for a, b in zip(rows, logged_horizon)), default=0.0)
        out["horizon_field_deviations"] = devs[:20]
        out["n_horizon_deviations"] = len(devs)
    return out


def main(which="surviving", ckpt_path=None):
    t0 = time.time()
    dev = L.device()
    w = L.wm()
    vmod = L.load_src_module("verify_vehicle_dynamics", name="wd01_verify_mod")
    tmod = L.train_mod()
    ckpt_path = Path(ckpt_path) if ckpt_path else L.SURV_CKPT
    model, x_mean, x_std, ck = load_surviving_like(ckpt_path)
    model = model.to(dev)

    m = L.load_json(L.RESULTS / "tag_run_map.json")
    holdout = [r for r in m["rows"] if r["split"] == "holdout"]
    logged = L.load_json(L.WM / "RESULTS.json")
    logged_eval = {(e["run"], e["t"]): e for e in logged["per_eval"]}

    per_eval, dv_err, dv_zero, pairs = [], [], [], []
    for r in holdout:
        states = L.cache_to_states(L.RESULTS / "states_cache" / "holdout" /
                                   f"{r['tag']}.npz")
        aid = r["action_idx"]
        for t in vmod.START_TICKS:
            cur, nxt = states.get(t), states.get(t + 1)
            if not cur or not nxt:
                continue
            ids, d, v, g = state_arrays(cur)
            vl, gap, free = w.leader_features(d, v, g)
            dv = predict_dv(model, x_mean, x_std, v, vl, gap, g, free, aid, dev)
            d_m, v_m, _ = L.raw_propagate(d, v, dv)
            sp_m, _ = w.aggregate_field(d_m, v_m, g)
            d_r, v_r, _ = L.raw_propagate(d, v, dv, clamp=False)
            sp_r, _ = w.aggregate_field(d_r, v_r, g)
            sp_p, _ = w.aggregate_field(d - v, v, g)
            _, d1, v1, g1 = state_arrays(nxt)
            sp_t, ct_t = w.aggregate_field(d1, v1, g1)
            sp_0, ct_0 = w.aggregate_field(d, v, g)
            mask = (ct_0 > 0) & (ct_t > 0)
            e_m = w.rmse_on(mask, sp_m, sp_t)
            e_r = w.rmse_on(mask, sp_r, sp_t)
            e_p = w.rmse_on(mask, sp_p, sp_t)
            e_c = w.rmse_on(mask, sp_0, sp_t)
            if e_m is None or e_p is None:
                continue
            per_eval.append(dict(run=r["tag"], t=t, n_vehicles=len(ids),
                                 n_cells=int(mask.sum()), model=e_m, raw=e_r,
                                 persistence=e_p, field_copy=e_c))
            lookup = nxt
            for k, vid in enumerate(ids):
                nx = lookup.get(vid)
                if nx is None or nx[2] != g[k]:
                    continue
                pairs.append((r["tag"], t, vid, float(dv[k]),
                              nx[1] - v[k], v[k], float(vl[k]), float(gap[k])))
                dv_err.append(abs(float(dv[k]) - (nx[1] - v[k])))
                dv_zero.append(abs(nx[1] - v[k]))

    mean_model = float(np.mean([e["model"] for e in per_eval]))
    mean_raw = float(np.mean([e["raw"] for e in per_eval]))
    mean_pers = float(np.mean([e["persistence"] for e in per_eval]))
    mean_copy = float(np.mean([e["field_copy"] for e in per_eval]))
    nc = sum(e["n_cells"] for e in per_eval)
    pooled_model = float(np.sqrt(sum(e["model"] ** 2 * e["n_cells"]
                                     for e in per_eval) / nc))
    pooled_raw = float(np.sqrt(sum(e["raw"] ** 2 * e["n_cells"]
                                   for e in per_eval) / nc))
    pooled_pers = float(np.sqrt(sum(e["persistence"] ** 2 * e["n_cells"]
                                    for e in per_eval) / nc))
    mae_model = float(np.mean(dv_err))
    mae_zero = float(np.mean(dv_zero))

    # logged comparison for the surviving checkpoint
    cmp_block, devs = None, []
    if which == "surviving":
        o = logged["one_step"]
        checks = dict(
            mean_model_rmse=(mean_model, o["mean_model_rmse_mps"]),
            mean_model_rmse_raw=(mean_raw, o["mean_model_rmse_raw_mps"]),
            pooled_model_rmse=(pooled_model, o["pooled_model_rmse_mps"]),
            pooled_model_rmse_raw=(pooled_raw, o["pooled_model_rmse_raw_mps"]),
            mean_persistence_rmse=(mean_pers, o["mean_persistence_rmse_mps"]),
            pooled_persistence_rmse=(pooled_pers, o["pooled_persistence_rmse_mps"]),
            mean_field_copy_rmse=(mean_copy, o["mean_field_copy_rmse_mps"]),
            dv_mae_model=(mae_model, logged["dv_error"]["mean_abs_model_dv_error_mps"]),
            dv_mae_zero=(mae_zero, logged["dv_error"]["mean_abs_zero_dv_baseline_mps"]),
            n_dv_pairs=(float(len(dv_err)),
                        float(logged["dv_error"]["n_pairs"])),
            n_evals=(float(len(per_eval)), float(logged["n_evals"])),
        )
        cmp_block = {k: dict(fresh=a, logged=b, abs_dev=abs(a - b),
                             within_tol=bool(abs(a - b) <= TOL))
                     for k, (a, b) in checks.items()}
        num_devs = []
        for e in per_eval:
            le = logged_eval.get((e["run"], e["t"]))
            if le is None:
                devs.append(f"{e['run']}@{e['t']}: missing in logged per_eval")
                continue
            for k_f, k_l in (("model", "model_rmse"), ("raw", "model_rmse_raw"),
                             ("persistence", "persistence_rmse"),
                             ("field_copy", "field_copy_rmse"),
                             ("n_cells", "n_cells"),
                             ("n_vehicles", "n_vehicles")):
                dd = abs(float(e[k_f]) - float(le[k_l]))
                if dd > TOL:
                    num_devs.append((f"{e['run']}@{e['t']}.{k_l}", dd))
        cmp_block["per_eval_rows_compared"] = len(per_eval)
        cmp_block["per_eval_n_deviations"] = len(num_devs)
        cmp_block["per_eval_max_abs_dev"] = max((dd for _, dd in num_devs),
                                                default=0.0)
        devs += [f"{k}: dev={dd:.3e}" for k, dd in num_devs[:40]]

    # rollout (only meaningful for the surviving ckpt identity check)
    roll_logged = logged.get("rollout") or {}
    roll = None
    if which == "surviving" and roll_logged.get("run"):
        rrow = next((r for r in holdout if r["tag"] == roll_logged["run"]), None)
        if rrow is not None:
            states = L.cache_to_states(L.RESULTS / "states_cache" / "holdout" /
                                       f"{rrow['tag']}.npz")
            roll = eval_rollout(model, x_mean, x_std, states, rrow["tag"],
                                rrow["action_idx"], vmod.ROLL_START,
                                vmod.ROLL_STEPS, dev, w,
                                logged_horizon=roll_logged.get("horizon"))
            rt = [x["rmse_truth_occupied"] for x in roll["horizon"]
                  if x.get("rmse_truth_occupied") is not None]
            rf = [x["rmse_start_fixed"] for x in roll["horizon"]
                  if x.get("rmse_start_fixed") is not None]
            roll["max_rmse_truth_occupied_mps"] = float(np.max(rt))
            roll["max_rmse_start_fixed_mps"] = float(np.max(rf))
            roll["final_step_rmse_truth_occupied_mps"] = float(rt[-1])
            roll["logged_max_rmse_truth_occupied_mps"] = \
                roll_logged.get("max_rmse_truth_occupied_mps")
            roll["logged_max_rmse_start_fixed_mps"] = \
                roll_logged.get("max_rmse_start_fixed_mps")

    out = dict(
        which=which, ckpt=str(ckpt_path), ckpt_sha256=L.sha256_file(ckpt_path),
        device=str(dev), generated_utc=datetime.now(timezone.utc).isoformat(),
        n_evals=len(per_eval), n_runs=len(holdout),
        metrics=dict(mean_model_rmse_mps=mean_model, mean_model_rmse_raw_mps=mean_raw,
                     pooled_model_rmse_mps=pooled_model,
                     pooled_model_rmse_raw_mps=pooled_raw,
                     mean_persistence_rmse_mps=mean_pers,
                     pooled_persistence_rmse_mps=pooled_pers,
                     mean_field_copy_rmse_mps=mean_copy,
                     dv_mae_model=mae_model, dv_mae_zero=mae_zero,
                     n_dv_pairs=len(dv_err)),
        logged_comparison=cmp_block, per_eval_deviations=devs[:40],
        rollout=roll,
        per_eval=per_eval, dv_pairs=pairs,
        seconds=round(time.time() - t0, 1),
    )
    path = L.RESULTS / f"identity_eval_{which}.json"
    L.dump_json(path, out)
    slim = {k: v for k, v in out.items() if k not in ("per_eval", "dv_pairs")}
    slim["per_eval"] = f"<{len(per_eval)} rows in file>"
    print(json.dumps(slim, indent=1, default=str)[:4000])
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "surviving"
    ck = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(main(which, ck))
