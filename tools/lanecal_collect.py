"""Lane-request acceptance collector (C4-LANECAL).

Runs one SUMO episode (env "direct") with a scripted policy: mild AR(1) random accelerations for all
CAVs and, per CAV and tick, a left/right request with probability --p-request (respecting the 3 s
dwell and the mask geometry, but NOT the gap rule), logging for every request that the env
issues the WM-frame features at request time and whether SUMO executed it within the 1 s step.

    python tools/lanecal_collect.py --out DIR --seed 4107 --demand 2700 --penetration 0.2
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

DWELL_S = 3.0


def neighbours(s, v, lane, present, length, i, target):
    """Nearest front/rear vehicle in ``target`` lane for ego i (bumper gaps, speeds, slots)."""
    idx = np.flatnonzero(present & (lane == target))
    idx = idx[idx != i]
    out = {"front_gap": math.inf, "front_v": float("nan"), "rear_gap": math.inf,
           "rear_v": float("nan"), "front_j": -1, "rear_j": -1}
    for j in idx:
        ds = float(s[j] - s[i])
        if ds >= 0:
            g = ds - float(length[j])
            if g < out["front_gap"]:
                out.update(front_gap=g, front_v=float(v[j]), front_j=int(j))
        else:
            g = -ds - float(length[i])
            if g < out["rear_gap"]:
                out.update(rear_gap=g, rear_v=float(v[j]), rear_j=int(j))
    return out


def run(args):
    from traffic_shepherd.c4.env import CavSumoEnv
    from traffic_shepherd.c4.geometry import MERGE_S0_M, MERGE_S1_M

    rng = np.random.default_rng(args.seed * 1000 + args.demand // 10 + int(args.penetration * 100))
    env = CavSumoEnv(check_parity=False, physics_log=False)
    t0 = time.monotonic()
    env.reset(seed=args.seed, demand_vph=args.demand, penetration=args.penetration,
              warmup_s=args.warmup, horizon_s=args.horizon, controller="direct")
    warm_wall = time.monotonic() - t0
    issued_log = {}
    original = env._issue_lane_commands

    def wrapped(targets, policy_cavs):
        issued, counts = original(targets, policy_cavs)
        issued_log["issued"] = dict(issued)
        issued_log["counts"] = dict(counts)
        return issued, counts
    env._issue_lane_commands = wrapped
    k = env.k
    rule = None
    if args.rule != "none":
        import torch
        from wms.control.lanes import LaneRules, lane_feasibility
        rule = (LaneRules() if args.rule == "new" else
                LaneRules(gap_base_m=6.0, gap_time_s=1.0, closing_time_s=1.0, rear_gap_base_m=None,
                          rear_gap_time_s=None, rear_closing_time_s=None))
    accel = np.zeros(k)
    last_change_t = {}
    last_lane = {}
    rows = []
    n_req_total = 0
    api = env.api
    done = False
    info = {}
    while not done:
        f = env.frame
        t = env.now
        s = np.where(f.present, f.s, 0.0)
        v = np.where(f.present, f.v, 0.0)
        lane = f.lane.astype(int)
        present = f.present & (lane >= 0)
        length = np.full(k, 5.0)
        ids = env.slot_vehicle_ids()
        for slot in np.flatnonzero(present & f.is_cav):
            vid = ids[slot]
            if vid in last_lane and last_lane[vid] != lane[slot]:
                last_change_t[vid] = t
            last_lane[vid] = int(lane[slot])
        accel = 0.85 * accel + 0.15 * rng.normal(0.0, 1.6, k)
        acc = np.clip(accel, -1.0, 0.8)
        lanes = np.full(k, -1.0)
        wanted = {}
        for slot in np.flatnonzero(present & f.is_cav & ~f.junction):
            vid = ids[slot]
            if t - last_change_t.get(vid, -1e9) < DWELL_S - 1e-6:
                continue
            ln = lane[slot]
            if ln == 0:
                tgt = 1
            elif ln == 1:
                tgt = 0
            elif ln == 2 and MERGE_S0_M <= s[slot] <= MERGE_S1_M:
                tgt = 1
            else:
                continue
            if rng.random() > args.p_request:
                continue
            if rule is not None:
                ages = torch.full((1, k), 99.0)
                ok = lane_feasibility(env.state(), ages, rule)[0, slot, 1 if tgt < ln else 2]
                if not bool(ok):
                    continue
            lanes[slot] = tgt
            wanted[vid] = (int(slot), int(ln), tgt)
        pre = {}
        for vid, (slot, ln, tgt) in wanted.items():
            nb = neighbours(s, v, lane, present, length, slot, tgt)
            nb_cur = neighbours(s, v, lane, present, length, slot, ln)
            row = env._table.get(vid) or {}

            def vt(j, name):
                if j < 0:
                    return None
                try:
                    return float(getattr(api.vehicle, name)(ids[j]))
                except Exception:
                    return None
            dist_edge = [b - s[slot] for b in (MERGE_S0_M, MERGE_S1_M) if b - s[slot] >= 0]
            pre[vid] = dict(
                t=t, vid=vid, ego_v=float(v[slot]), ego_s=float(s[slot]), lane=int(ln), target=int(tgt),
                dir="left" if tgt < ln else "right", ramp=bool(ln == 2),
                cur_front_gap=nb_cur["front_gap"], cur_front_v=nb_cur["front_v"],
                tgt_front_gap=nb["front_gap"], tgt_front_v=nb["front_v"],
                tgt_rear_gap=nb["rear_gap"], tgt_rear_v=nb["rear_v"],
                to_edge_m=float(min(dist_edge)) if dist_edge else math.inf,
                ego_len_true=float(f.length_true[slot]),
                rear_len_true=float(f.length_true[nb["rear_j"]]) if nb["rear_j"] >= 0 else None,
                front_len_true=float(f.length_true[nb["front_j"]]) if nb["front_j"] >= 0 else None,
                rear_tau=vt(nb["rear_j"], "getTau"), rear_decel=vt(nb["rear_j"], "getDecel"),
                rear_minGap=vt(nb["rear_j"], "getMinGap"),
                ego_tau=vt(slot, "getTau"), ego_minGap=vt(slot, "getMinGap"),
                edge_lane_pos=float(row.get("lane_position_m", float("nan"))),
            )
        a_in = np.where(present & f.is_cav, acc, np.nan)
        issued_log.clear()
        state, info = env.step(a_in[None], lanes=lanes[None])
        issued = issued_log.get("issued", {})
        f2 = env.frame
        ids2 = env.slot_vehicle_ids()
        now_lane = {ids2[sl]: int(f2.lane[sl]) for sl in np.flatnonzero(f2.present)}
        for vid, (target, _) in issued.items():
            r = pre.get(vid)
            if r is None:
                continue
            r["accepted"] = bool(now_lane.get(vid, -9) == target)
            r["lane_after"] = now_lane.get(vid, None)
            rows.append(r)
        n_req_total += len(wanted)
        done = bool(info["done"])
    summary = info.get("summary", {})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"s{args.seed}_d{args.demand}_p{int(round(args.penetration * 100))}"
    with open(out / f"{name}.jsonl", "w") as fh:
        for r in rows:
            r = {kk: (None if isinstance(vv, float) and not math.isfinite(vv) else vv)
                 for kk, vv in r.items()}
            fh.write(json.dumps(r) + "\n")
    meta = dict(seed=args.seed, demand=args.demand, penetration=args.penetration,
                requested=n_req_total, issued=len(rows),
                collisions=summary.get("collisions"), teleports=summary.get("teleports"),
                lane_commands=summary.get("lane_commands"),
                wall_s=time.monotonic() - t0, warm_wall_s=warm_wall)
    (out / f"{name}.meta.json").write_text(json.dumps(meta))
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--demand", type=int, required=True)
    ap.add_argument("--penetration", type=float, required=True)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--p-request", type=float, default=0.25)
    ap.add_argument("--rule", choices=("none", "old", "new"), default="none",
                    help="only issue requests the given LaneRules mask allows (closed-loop validation)")
    run(ap.parse_args())
