"""F2 fidelity diag: recorded vs open-road A11 rollout, per 10 s bin and
per road segment (upstream / merge / downstream), plus head dV error.

TRAIN-only, no SUMO, one CPU process.  Variants:
* ``closed``    -- historical rollout (no exits, no entries, nominal geometry)
* ``exit``      -- physical exits only
* ``open``      -- physical exits + observed entrance counters
* ``open_fid``  -- open + ``sumo_fidelity`` (s_down-frame geometry + SUMO
                   vType-matched IDM)

Writes ``_out/fidelity_diag.json`` and prints a compact per-state table.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output" / "wd01" / "wm_n1"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(6)
from closedloop.a11_predictor import (PLANNER_ACTION_CODES, NONE_CODE,  # noqa: E402
                                      _rollout_mask, batched_rollout,
                                      head_dv_quantiles, load_a11_bundle,
                                      long_plan_from_rollout, tau_index)
from closedloop.live_state import DepartureHistory, live_state_from_row  # noqa: E402
from features.geometry import s_down_of  # noqa: E402

CHUNK = Path(r"E:/research-dense/runs/campaign_stage1_chunk4")
PQ = Path(r"D:/rd-staging/wmn1_feat/labels_fix1_20261003T141730Z/train_outcomes.parquet")
H = 120.0
N = int(round(H / 0.5))
WORLDS = [("w8676m2700p20t360", 360.0), ("w8626m2700p02t300", 300.0),
          ("w8629m2000p02t420", 420.0), ("w8678m2700p20t300", 300.0),
          ("w8677m2700p20t360", 360.0), ("w8680m2600p20t300", 300.0),
          ("w8679m2400p20t540", 540.0), ("w8683m2400p20t540", 540.0)]
OFF = np.asarray([2645.67, 2644.22, 378.66])
# segments in the shared merge-relative s_down coordinate (merge segment
# 1487606179 spans s_down [0, 75.44] on all three lanes).
SEG = {"up": (-1e9, 0.0), "merge": (0.0, 75.44), "down": (75.44, 1e9)}
CH = ("mean_v", "min_v", "mean_a", "frac_dec", "n_cav")


def read_rows(path, n):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def seg_of(s_down):
    if s_down < 0.0:
        return "up"
    if s_down <= 75.44:
        return "merge"
    return "down"


def recorded_bins(rows):
    """[12] per-bin per-segment (count, mean_v, stopped) for recorded rows."""
    bins = []
    for b in range(12):
        lo, hi = b * 20, (b + 1) * 20      # 0.5 s ticks -> 10 s
        segs = {k: {"n": 0, "v": [], "stop": 0} for k in SEG}
        for t in range(lo, min(hi, len(rows))):
            for v in (rows[t].get("vehicles") or {}).values():
                lid = v.get("lane_id")
                if not lid or str(lid).startswith(":"):
                    continue
                try:
                    lp, sd = s_down_of(lid, float(v["lane_position_m"]))
                except Exception:
                    continue
                g = segs[seg_of(sd)]
                g["n"] += 1
                g["v"].append(float(v.get("speed_mps", 0.0)))
                if float(v.get("speed_mps", 0.0)) < 0.5:
                    g["stop"] += 1
        bins.append({k: {"n": g["n"], "mean_v": (float(np.mean(g["v"])) if g["v"] else 0.0),
                         "stop": g["stop"]} for k, g in segs.items()})
    return bins


def rollout_bins(ro, off=OFF, k=None):
    msk, spd, lns, pos = (ro.msk[:, 0], ro.spd[:, 0], ro.lns[:, 0], ro.pos[:, 0])
    bins = []
    for b in range(12):
        lo, hi = b * 20, (b + 1) * 20
        segs = {k: {"n": 0, "v": [], "stop": 0} for k in SEG}
        for t in range(lo, min(hi + 1, msk.shape[0])):
            for s in np.flatnonzero(msk[t]):
                ln = int(lns[t, s])
                if ln < 0:
                    continue
                g = segs[seg_of(float(pos[t, s]))]
                g["n"] += 1
                vv = float(spd[t, s])
                g["v"].append(vv)
                if vv < 0.5:
                    g["stop"] += 1
        bins.append({kk: {"n": g["n"], "mean_v": (float(np.mean(g["v"])) if g["v"] else 0.0),
                          "stop": g["stop"]} for kk, g in segs.items()})
    return bins


def channels(pl):
    return pl[0].detach().cpu().numpy().reshape(12, 5)


def true_dv():
    import pandas as pd
    df = pd.read_parquet(PQ, columns=["world", "source", "t0_s", "t", "tau",
                                      "tau_available", "total_delay_ref_s"])
    out = {}
    for w, t0 in WORLDS:
        sub = df[(df.world == w) & np.isclose(df.t0_s, t0) & np.isclose(df.t, t0)
                 & (df.tau_available > 0) & np.isclose(df.tau, 300.0)]
        a = sub[sub.source == "pi"].total_delay_ref_s
        nn = sub[sub.source == "native"].total_delay_ref_s
        out[w] = float(a.mean() - nn.mean()) if len(a) and len(nn) else None
    return out


def residual_mag(bundle, st):
    kin = torch.as_tensor(np.asarray(st.veh_kin, np.float64)).unsqueeze(0)
    m = torch.as_tensor((np.asarray(st.veh_mask, bool) & _rollout_mask(st))).unsqueeze(0)
    with torch.no_grad():
        h = bundle.hdv
        if hasattr(h, "net"):
            h = h.net
        out = h(kin, m)
    d = out["accel_delta_mps2"].reshape(-1).numpy()
    mm = m.reshape(-1).numpy().astype(bool)
    return float(np.abs(d[mm]).mean())


def main():
    from closedloop.fast_hdv import fast_bundle, install_fast_gap_projection
    install_fast_gap_projection()
    ck = load_a11_bundle()
    b = fast_bundle(ck.bundle)
    ti = tau_index(300.0)
    tdv = true_dv()
    out = {"worlds": {}, "true_dv": tdv}
    var = {"closed": {}, "exit": {"physical_exits": True},
           "open": {"physical_exits": True, "open_entries": True},
           "open_fidg": {"physical_exits": True, "open_entries": True,
                         "sumo_fidelity": True, "sumo_idm": False},
           "open_fid": {"physical_exits": True, "open_entries": True,
                        "sumo_fidelity": True}}
    dv_mat = {k: [] for k in var}
    worlds = WORLDS[:int(os.environ.get("FID_N", len(WORLDS)))]
    for world, t0 in worlds:
        base = CHUNK / f"{world}__base"
        man = json.load(open(base / "saved_state" / "STATE_MANIFEST.json"))
        dep = DepartureHistory(base, t0)
        st = live_state_from_row(man["physics_row_at_t0"], dep)
        cav = st.is_cav & _rollout_mask(st)
        K = len(st.slot_vid)
        codes = np.full((2, K), NONE_CODE)
        codes[0, cav] = PLANNER_ACTION_CODES["pi-external-u"]
        codes[1, cav] = PLANNER_ACTION_CODES["native"]
        rec_rows = read_rows(CHUNK / f"{world}__pi" / "run" / "physics_steps.jsonl", N)
        rbins = recorded_bins(rec_rows)
        rec = {"bins": rbins, "dvs": {}}
        for name, kw in var.items():
            t_ = time.time()
            ro = batched_rollout(b, st, codes, n_steps=N, t0_s=float(t0), **kw)
            pp = channels(long_plan_from_rollout(ro.at(0), st.is_cav))
            dvp = float(head_dv_quantiles(b, st, torch.as_tensor(
                long_plan_from_rollout(ro.at(0), st.is_cav)
                .reshape(1, pp.size)))[0, ti].mean())
            rec["dvs"][name] = dvp
            dv_mat[name].append((tdv[world], dvp))
            rec[name] = {"bins": rollout_bins(ro),
                         "plan_pi": pp.tolist(),
                         "entries": getattr(ro, "entry_summary", None),
                         "sec": round(time.time() - t_)}
        rec["residual_mean_abs_a"] = residual_mag(b, st)
        rec["merge_cav"] = int(cav.sum())
        out["worlds"][world] = rec
        print("=" * 74, flush=True)
        print(f"{world} t0={t0:g} cav0={int(cav.sum())} TRUE={tdv[world]:.0f} "
              f"resid|a|={rec['residual_mean_abs_a']:.3f}", flush=True)
        for name in var:
            pb = np.array(rec[name]["plan_pi"])
            print(f"  {name:9s} dV_pi={rec['dvs'][name]:8.0f} "
                  f"n_cav_bins=[{pb[0,4]:.0f},{pb[6,4]:.0f},{pb[11,4]:.0f}] "
                  f"mean_v={pb[:,0].mean():.1f} min_v={pb[:,1].mean():.1f}", flush=True)
        print(f"  recorded  mean_v={np.mean([x['merge']['mean_v'] for x in rbins]):.1f} "
              f"merge_n/bin={np.mean([x['merge']['n'] for x in rbins]):.1f} "
              f"merge_stop/bin={np.mean([x['merge']['stop'] for x in rbins]):.1f}", flush=True)
        (ROOT / "_out" / "fidelity_diag.json").write_text(json.dumps(out), encoding="utf-8")
    print("\nAccross 8 states: MAE vs TRUE / corr:", flush=True)
    for name, pairs in dv_mat.items():
        a = np.array([p[0] for p in pairs], float)
        pv = np.array([p[1] for p in pairs], float)
        mae = float(np.mean(np.abs(a - pv)))
        corr = float(np.corrcoef(a, pv)[0, 1])
        out.setdefault("summary", {})[name] = {"mae": mae, "corr": corr}
        print(f"  {name:9s} MAE={mae:8.0f} corr={corr:.3f}", flush=True)
    (ROOT / "_out" / "fidelity_diag.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
