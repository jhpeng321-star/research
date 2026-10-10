"""Measure recorded TRAIN merge statistics from chunk-4 SUMO physics rows.

TRAIN-only, no SUMO.  For the 8 FIDELITY (+ optionally 12 EXTRA) chunk-4 states
both the ``__pi`` and ``__native`` recorded branches are read (240 rows = 120 s
of 0.5 s physics steps).  We extract, per state:

* on-ramp merges (lane_phys 2 -> 1 transitions): rate (veh/min), merge
  position s_down along the accel lane, accepted lead/lag bumper gaps on the
  target (main_right) lane measured one step BEFORE the merge (the gap the
  driver accepted), and the target-lane follower's acceleration around the
  merge (mainline yielding);
* mainline yielding away from lane 1: lane_phys 1 -> 0 discretionary changes
  inside the merge window;
* exit timing: for every vehicle present at t0, the fraction still present at
  5..120 s and the s_down / speed when it last appears.

Writes ``_out/merge_cal_recorded.json`` and prints a compact table.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output" / "wd01" / "wm_n1"))
import numpy as np  # noqa: E402

from features.geometry import is_junction_lane, s_down_of  # noqa: E402

CHUNK = Path(r"E:/research-dense/runs/campaign_stage1_chunk4")
OUT = ROOT / "_out" / "merge_cal_recorded.json"
N_ROWS = 240                     # 120 s
HORIZONS = (5, 10, 20, 30, 60, 90, 120)
LANE_LEN = {0: 3199.0, 1: 3197.55, 2: 454.1}
OFF = {0: 2645.67, 1: 2644.22, 2: 378.66}
LEN_M = 5.0
MERGE_S0, MERGE_S1 = 0.0, 75.44
YIELD_WINDOW = (-30.0, 150.0)    # s_down window counted as "near the merge"

FIDELITY = ["w8676m2700p20t360", "w8626m2700p02t300", "w8629m2000p02t420",
            "w8678m2700p20t300", "w8677m2700p20t360", "w8680m2600p20t300",
            "w8679m2400p20t540", "w8683m2400p20t540"]
EXTRA = ["w8617m2700p02t300", "w8618m2000p02t480", "w8619m2700p02t360",
         "w8621m2700p02t240", "w8623m2600p02t540", "w8634m2400p02t300",
         "w8635m2700p05t300", "w8638m2700p05t360", "w8640m2000p05t540",
         "w8643m2600p05t240", "w8653m2600p10t420", "w8689m2600p40t360"]


def read_rows(path: Path, n: int) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def state_at(row: dict) -> dict:
    """vid -> (lane_phys, s_down, speed) for one recorded row."""
    out = {}
    for vid, v in (row.get("vehicles") or {}).items():
        lid = v.get("lane_id")
        if not lid or is_junction_lane(lid):
            continue
        try:
            lp, sd = s_down_of(lid, float(v["lane_position_m"]))
        except Exception:
            continue
        out[vid] = (int(lp), float(sd), float(v.get("speed_mps", 0.0)))
    return out


def neighbors_lane(states, lane, s_ego):
    """(lead, lag) (s, speed) on ``lane`` around ``s_ego`` (bumper gaps)."""
    lead = lag = None
    for lp, sd, sp in states.values():
        if lp != lane:
            continue
        if sd > s_ego and (lead is None or sd < lead[0]):
            lead = (sd, sp)
        if sd < s_ego and (lag is None or sd > lag[0]):
            lag = (sd, sp)
    return lead, lag


MERGE_HI = 75.44
STOP_V = 0.5


def analyze(rows: list) -> dict:
    snaps = [state_at(r) for r in rows]
    t0_ids = set(snaps[0]) if snaps else set()
    hz = {str(h): {"present": 0, "n": 0} for h in HORIZONS}
    last_seen = {}
    merges = []
    yield_10 = 0          # lane 1 -> 0 changes near the merge
    merge_v_bins = []
    merge_stop = 0
    for b in range(4):                       # 30 s bins
        vs = []
        for t in range(b * 60, min((b + 1) * 60, len(snaps))):
            for lp, sd, sp in snaps[t].values():
                if 0.0 <= sd <= MERGE_HI:
                    vs.append(sp)
                    if sp < STOP_V:
                        merge_stop += 1
        merge_v_bins.append(float(np.mean(vs)) if vs else 0.0)
    prev = None
    for t, snap in enumerate(snaps):
        if prev is not None:
            for vid, (lp, sd, sp) in snap.items():
                p = prev.get(vid)
                if p is None:
                    continue
                lpp = p[0]
                if lpp == 2 and lp == 1:
                    # accepted gaps: use pre-merge target-lane neighbours
                    lead, lag = neighbors_lane(prev, 1, p[1])
                    gl = (lead[0] - LEN_M - p[1]) if lead else None
                    gg = (p[1] - LEN_M - lag[0]) if lag else None
                    merges.append({"t": t, "s_down_pre": p[1], "s_down": sd,
                                   "v_pre": p[2], "v_post": sp,
                                   "gap_lead": gl, "gap_lag": gg,
                                   "v_lead": (lead[1] if lead else None),
                                   "v_lag": (lag[1] if lag else None)})
                if lpp == 1 and lp == 0 and YIELD_WINDOW[0] <= p[1] <= YIELD_WINDOW[1]:
                    yield_10 += 1
        for vid in t0_ids:
            if vid in snap:
                last_seen[vid] = snap[vid]
        prev = snap
    for h in HORIZONS:
        j = min(int(round(h / 0.5)), len(snaps) - 1)
        present = sum(1 for vid in t0_ids if vid in snaps[j])
        hz[str(h)] = {"present": present, "n": len(t0_ids)}
    exit_sd = [v[1] for v in last_seen.values()]
    exit_v = [v[2] for v in last_seen.values()]
    gl = [m["gap_lead"] for m in merges if m["gap_lead"] is not None]
    gg = [m["gap_lag"] for m in merges if m["gap_lag"] is not None]
    pos = [m["s_down_pre"] for m in merges]
    return {
        "n_merges": len(merges),
        "merge_rate_veh_min": len(merges) / (len(snaps) * 0.5 / 60.0),
        "merge_pos_s_down": _q(pos),
        "accepted_gap_lead_m": _q(gl),
        "accepted_gap_lag_m": _q(gg),
        "yield_1_to_0_near_merge": yield_10,
        "merge_stopped": merge_stop,
        "merge_mean_v_30s": merge_v_bins,
        "n_t0": len(t0_ids),
        "present_frac": {h: hz[h]["present"] / max(1, hz[h]["n"]) for h in hz},
        "last_seen_s_down": _q(exit_sd),
        "last_seen_v": _q(exit_v),
    }


def _q(xs):
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"n": int(a.size), "mean": float(a.mean()), "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def main():
    states_env = os.environ.get("MC_STATES")
    states = ([s for s in states_env.split(",") if s] if states_env
              else FIDELITY + ([] if os.environ.get("MC_FID_ONLY") else EXTRA))
    arms_env = os.environ.get("MC_ARMS", "pi,native")
    arms = [a for a in arms_env.split(",") if a]
    out = {"states": {}, "meta": {"states": states, "arms": arms, "n_rows": N_ROWS}}
    for w in states:
        out["states"][w] = {}
        for arm in arms:
            p = CHUNK / f"{w}__{arm}" / "run" / "physics_steps.jsonl"
            if not p.is_file():
                out["states"][w][arm] = {"missing": str(p)}
                continue
            out["states"][w][arm] = analyze(read_rows(p, N_ROWS))
        s = out["states"][w]
        if "pi" in s and "n_merges" in s["pi"]:
            r = s["pi"]
            print(f"{w:22s} merge/min={r['merge_rate_veh_min']:5.2f} "
                  f"pos50={_fmt(r['merge_pos_s_down'])} "
                  f"lead50={_fmt(r['accepted_gap_lead_m'])} "
                  f"lag50={_fmt(r['accepted_gap_lag_m'])} "
                  f"y10={r['yield_1_to_0_near_merge']:3d} "
                  f"match60={r['present_frac']['60']:.2f} "
                  f"match120={r['present_frac']['120']:.2f} ", flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT)


def _fmt(q):
    return "—" if not q else f"{q['p50']:.1f}"


if __name__ == "__main__":
    main()
