"""EXIT_FIX diag: recorded vs A11 rollout plan channels, exits OLD vs NEW.

TRAIN-only, no SUMO, one CPU process.  Extends the pl01-a11 _out/INPUT_DIAG.py
comparison to 8 chunk-4 TRAIN decision states and adds the physical-exit
variant of the A11 rollout.  Writes ``_out/exit_diag.json`` and prints a
compact per-state summary.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output" / "wd01" / "wm_n1"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from closedloop.a11_predictor import (PLANNER_ACTION_CODES, NONE_CODE,  # noqa: E402
                                      _rollout_mask, batched_rollout,
                                      head_dv_quantiles, load_a11_bundle,
                                      long_plan_from_rollout, tau_index)
from closedloop.live_state import DepartureHistory, live_state_from_row  # noqa: E402
from features.idea_a import CAV_ACCEL_SCALE, CAV_TGT_SPEED_SCALE  # noqa: E402
from model import constants as MC  # noqa: E402
from model.paired_effect import long_plan_summary  # noqa: E402

CHUNK = Path(r"E:/research-dense/runs/campaign_stage1_chunk4")
PQ = Path(r"D:/rd-staging/wmn1_feat/labels_fix1_20261003T141730Z/train_outcomes.parquet")
H = 120.0
N = int(round(H / 0.5))
WORLDS = [("w8676m2700p20t360", 360.0),
          ("w8626m2700p02t300", 300.0),
          ("w8629m2000p02t420", 420.0),
          ("w8678m2700p20t300", 300.0),
          ("w8677m2700p20t360", 360.0),
          ("w8680m2600p20t300", 300.0),
          ("w8679m2400p20t540", 540.0),
          ("w8683m2400p20t540", 540.0)]
BINS = 12
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


def recorded_plan(rows, slots, slot_vid):
    """training_plan_long equivalent from recorded physics rows (INPUT_DIAG)."""
    sp = np.zeros((N, 1, len(slots)))
    ac = np.zeros((N, 1, len(slots)))
    m = np.zeros((N, 1, len(slots)))
    alive = {int(s): True for s in slots}
    for t in range(min(N, len(rows))):
        vehs = rows[t].get("vehicles") or {}
        for j, s in enumerate(slots):
            v = slot_vid[int(s)]
            if alive[int(s)] and v in vehs:
                sp[t, 0, j] = vehs[v].get("speed_mps", 0.0)
                ac[t, 0, j] = vehs[v].get("acceleration_mps2", 0.0) or 0.0
                m[t, 0, j] = 1.0
            else:
                alive[int(s)] = False
    sp = (sp / CAV_TGT_SPEED_SCALE).astype(np.float16).astype(np.float32) * CAV_TGT_SPEED_SCALE
    ac = (ac / CAV_ACCEL_SCALE).astype(np.float16).astype(np.float32) * CAV_ACCEL_SCALE
    return long_plan_summary(torch.as_tensor(sp), torch.as_tensor(ac),
                             torch.as_tensor(m), horizon_s=H)


def channels(pl):
    return pl[0].detach().cpu().numpy().reshape(BINS, 5)


def true_dv():
    import pandas as pd
    df = pd.read_parquet(PQ, columns=["world", "source", "t0_s", "t", "tau",
                                      "tau_available", "total_delay_ref_s"])
    out = {}
    for w, t0 in WORLDS:
        sub = df[(df.world == w) & np.isclose(df.t0_s, t0) & np.isclose(df.t, t0)
                 & (df.tau_available > 0) & np.isclose(df.tau, 300.0)]
        a = sub[sub.source == "pi"].total_delay_ref_s
        n = sub[sub.source == "native"].total_delay_ref_s
        out[w] = float(a.mean() - n.mean()) if len(a) and len(n) else None
    return out


def main():
    from closedloop.fast_hdv import fast_bundle, install_fast_gap_projection
    install_fast_gap_projection()
    ck = load_a11_bundle()
    b = fast_bundle(ck.bundle)
    ti = tau_index(300.0)
    tdv = true_dv()
    rec = {}
    for world, t0 in WORLDS:
        base = CHUNK / f"{world}__base"
        man = json.load(open(base / "saved_state" / "STATE_MANIFEST.json"))
        dep = DepartureHistory(base, t0)
        st = live_state_from_row(man["physics_row_at_t0"], dep)
        nrows = read_rows(CHUNK / f"{world}__pi" / "run" / "physics_steps.jsonl", N)
        cav = st.is_cav & _rollout_mask(st)
        slots = np.flatnonzero(cav)
        K = len(st.slot_vid)
        codes = np.full((2, K), NONE_CODE)          # row 0 = PI, row 1 = native
        codes[0, cav] = PLANNER_ACTION_CODES["pi-external-u"]
        codes[1, cav] = PLANNER_ACTION_CODES["native"]
        plan_rec = channels(recorded_plan(nrows, slots, st.slot_vid))
        def dv(pl):
            q = head_dv_quantiles(b, st, torch.as_tensor(pl.reshape(1, BINS * 5)))
            return float(q[0, ti].mean())
        r = {"t0": t0, "n_cav0": int(cav.sum()), "true": tdv[world],
             "rec": plan_rec.tolist(), "dv_scorer_pi": dv(plan_rec)}
        for name, kw in (("old", {}), ("exit", {"physical_exits": True}),
                         ("open", {"physical_exits": True, "open_entries": True})):
            t_ = time.time()
            ro = batched_rollout(b, st, codes, n_steps=N, t0_s=float(t0), **kw)
            pp = channels(long_plan_from_rollout(ro.at(0), st.is_cav))
            pn = channels(long_plan_from_rollout(ro.at(1), st.is_cav))
            r[name] = {"plan_pi": pp.tolist(), "plan_native": pn.tolist(),
                       "dv_pi": dv(pp), "dv_native": dv(pn),
                       "entries": getattr(ro, "entry_summary", None),
                       "present_end": int(ro.msk[-1, 0].sum()), "sec": time.time() - t_}
        rec[world] = r
        print("=" * 78, flush=True)
        print(f"{world} t0={t0:g} ncav0={int(cav.sum())} TRUE={tdv[world]:.0f} scorer={r['dv_scorer_pi']:.0f}")
        for nm in ("old", "exit", "open"):
            x = r[nm]
            print(f"  {nm:5s} dV_pi={x['dv_pi']:8.0f} dV_nat={x['dv_native']:8.0f} "
                  f"n_cav={[round(v, 1) for v in np.array(x['plan_pi'])[:, 4]]} "
                  f"mean_v={np.mean(np.array(x['plan_pi'])[:, 0]):.1f} min_v={np.mean(np.array(x['plan_pi'])[:, 1]):.1f} "
                  f"({x['sec']:.0f}s)", flush=True)
        print(f"  rec   n_cav={[round(v, 1) for v in plan_rec[:, 4]]} "
              f"mean_v={plan_rec[:, 0].mean():.1f} min_v={plan_rec[:, 1].mean():.1f}", flush=True)
        if r["open"]["entries"]:
            print("  entries:", r["open"]["entries"][0], flush=True)
        (ROOT / "_out" / "entry_diag.json").write_text(json.dumps(rec), encoding="utf-8")


if __name__ == "__main__":
    main()
