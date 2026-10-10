"""A2 step 2, measurement: how does a commanded target become the
realised next CAV speed in chunk-1 (execution rule per speedMode)?

For every logged command (control tick t, CAV vid) on a DECLARED
deterministic subset of TRAIN branches (every 4th sorted TRAIN world x
all 7 arms, complete receipts only), take the issued target T and the
realised speed at the next physics step (0.5 s later; SUMO keeps the
setSpeed target until the next control tick, so the SECOND step
t+0.5 -> t+1.0 under the same T is measured separately), and compare
the realised v' with candidate execution rules:

  kin       v' = clip(T, v - b*DT, v + a*DT), floor 0      (pure setSpeed
            kinematics with the CAV vType's accel/decel)
  kin_idm   v' = max(0, min(kin, v + idm_acc*DT))          (safe-speed bound
            via the engine IDM with the CAV's own vType; emergency
            braking below maxDecel allowed)
  kin_flat  v' = max(0, min(kin, v + idm_acc*DT)) but with the decel
            floor enforced: max(kin_min, min(kin_max, idm))   (no
            emergency term)

plus diagnostics: realised speed below v - b*DT - eps (emergency), above
v + a*DT + eps (impossible), IDM-bound frequency, lane-change rows
excluded, and residuals per speedMode.

ZERO SIMULATION: chunk-1 jsonl logs read as data only.  TRAIN only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("WD_DEVICE", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import act_lib as L  # noqa: E402

DT = 0.5
WORLD_STRIDE = 4                    # declared deterministic subset
EPS = 1e-6


def lane_table():
    raw = json.loads((HERE.parent / "partA_ctrl" / "results" /
                      "lane_table.json").read_text())
    return {k: tuple(v) for k, v in raw.items()}


def vtypes():
    return L.P3.VTYPES


def idm_acc_np(v, gap, dv, a_max, b, tau, s0, v0):
    v = np.asarray(v, dtype=np.float64)
    gap = np.maximum(np.asarray(gap, dtype=np.float64), 0.3)
    dv = np.asarray(dv, dtype=np.float64)
    v0 = np.maximum(np.asarray(v0, dtype=np.float64), 0.1)
    s_star = np.maximum(s0 + (v * tau + v * dv
                              / (2.0 * np.sqrt(a_max * b))).clip(min=0.0),
                        0.0)
    return a_max * (1.0 - np.minimum(v / v0, 4.0) ** 4
                    - (s_star / gap) ** 2)


class BranchLoader:
    """Physics rows + commands of one branch, with leader search in the
    model-world (lane, s) coordinates (physical lanes span junctions, so
    cross-edge leaders are seen - the same view the engine's neighbour
    builder has).  lane_id -> (lane, s_offset) is memoised; internal
    junction lanes map to (lane=-1, boundary s)."""

    def __init__(self, run_name):
        run = L.refload.CHUNK1_RUNS / run_name / "run"
        self.physics = {}
        self.lc_flag = {}
        self.cav_ids = None
        self._lane_off = {}
        for row in L.iter_jsonl(run / "physics_steps.jsonl"):
            t = round(float(row["time_s"]), 6)
            self.physics[t] = L.row_states_raw(row)
            for vid, v in row["vehicles"].items():
                if v.get("lane_changed"):
                    self.lc_flag[(t, vid)] = True
            if self.cav_ids is None:
                self.cav_ids = set(row["cav_ids"])
        self.commands = {}
        for row in L.iter_jsonl(run / "suffix_commands.jsonl"):
            if row.get("kind") != "command":
                continue
            tick = round(float(row["timestamp_s"]), 6)
            self.commands[(tick, row["vehicle_id"])] = (
                row["owner"], row["issued_target_speed_mps"],
                row["fallback_reason"])

    def _map(self, lane_id, pos):
        key = lane_id
        off = self._lane_off.get(key)
        if off is None:
            lane, s, _inj = L.PD.map_step(lane_id, 0.0)
            off = (lane, float(s), lane < 0)
            self._lane_off[key] = off
        lane, s0, internal = off
        return (lane, s0 if internal else s0 + float(pos))

    def mapped(self, states):
        """{vid: (lane, s, v, type_id)} in model-world coordinates."""
        out = {}
        for vid, st in states.items():
            lane, s = self._map(st[0], st[1])
            out[vid] = (lane, s, st[2], st[3])
        return out

    def leader(self, mapped_states, vid):
        """Same PHYSICAL lane nearest front vehicle: (gap, v_leader) or
        None (bumper gap in the continuous s coordinate)."""
        lane, pos, speed, _typ = mapped_states[vid]
        if lane < 0:
            return None
        best = None
        for other, st in mapped_states.items():
            if other == vid or st[0] != lane:
                continue
            if st[1] >= pos:
                gap = st[1] - L.VEHICLE_LENGTH_M - pos
                if gap >= 0 and (best is None or gap < best[0]):
                    best = (gap, st[2])
        return best


def rule_residuals(rows):
    """Residual quantiles + match rates per rule (+ signed residual and
    leader-present split for the pinned rule)."""
    out = {}
    for rule in ("kin", "kin_idm", "kin_flat"):
        res = np.array([r["v1"] - r[rule] for r in rows])
        ares = np.abs(res)
        exact = float(np.mean(ares < 1e-9)) if len(ares) else None
        tol = float(np.mean(ares < 0.01)) if len(ares) else None
        out[rule] = dict(
            n=int(len(ares)),
            p50=float(np.quantile(ares, 0.5)),
            p95=float(np.quantile(ares, 0.95)),
            p99=float(np.quantile(ares, 0.99)),
            max=float(ares.max()) if len(ares) else None,
            signed_p50=float(np.quantile(res, 0.5)),
            exact_rate=exact, tol01_rate=tol)
    for tag, sel in (("leader", lambda r: r["gap"] < 250.0),
                     ("free", lambda r: r["gap"] >= 250.0),
                     ("idm_bound", lambda r: r["idm_binds"])):
        sel_rows = [r for r in rows if sel(r)]
        if sel_rows:
            res = np.abs(np.array([r["v1"] - r["kin_idm"]
                                   for r in sel_rows]))
            out[f"kin_idm_{tag}"] = dict(
                n=int(len(res)), p50=float(np.quantile(res, 0.5)),
                p95=float(np.quantile(res, 0.95)),
                tol01_rate=float(np.mean(res < 0.01)),
                signed_p50=float(np.quantile(
                    [r["v1"] - r["kin_idm"] for r in sel_rows], 0.5)))
    return out


def main():
    t_start = time.time()
    worlds = L.train_worlds()
    names = sorted(worlds)[::WORLD_STRIDE]
    lanes = lane_table()
    vts = vtypes()
    print("=== A2 execution-rule measurement (TRAIN, ZERO simulation) ===")
    print(f"declared deterministic subset: every {WORLD_STRIDE}th sorted "
          f"TRAIN world -> {len(names)} worlds x 7 arms "
          f"(complete receipts only)")
    print(f"worlds: {names}")

    per_mode_rows = defaultdict(list)     # speedMode -> [row dicts]
    diag = defaultdict(int)
    per_vtype = defaultdict(list)
    n_branch = 0
    for world in names:
        t0 = float(worlds[world]["t0_s"])
        factors = L.speed_factors(world)
        for arm in L.BRANCH_ARMS:
            run_name = f"{world}__{arm}"
            if not L.receipt_ok(run_name):
                diag["incomplete_receipt"] += 1
                continue
            bl = BranchLoader(run_name)
            n_branch += 1
            for (tick, vid), (owner, issued, fallback) in \
                    sorted(bl.commands.items()):
                if issued is None:
                    diag["fallback_tick"] += 1
                    continue
                st0 = bl.physics.get(round(tick, 6))
                st1 = bl.physics.get(round(tick + DT, 6))
                st2 = bl.physics.get(round(tick + 1.0, 6))
                if st0 is None or vid not in st0:
                    diag["missing_row_t"] += 1
                    continue
                lane_id, pos, v0, type_id, speed_mode = (
                    st0[vid][0], st0[vid][1], st0[vid][2], st0[vid][3],
                    31)
                vt = vts[L.TYPE_OF.get(type_id, 1)]
                a_max, b = vt["a_max"], vt["b"]
                lane_len, lane_speed = lanes[lane_id]
                v0free = lane_speed * factors.get(type_id, 1.0)
                map0 = bl.mapped(st0)
                lead0 = bl.leader(map0, vid)
                # ---- step 1: t -> t+0.5 under T
                if st1 is not None and vid in st1:
                    v1 = st1[vid][2]
                    lc = bl.lc_flag.get((round(tick + DT, 6), vid),
                                        st1[vid][0] != lane_id)
                    lead = lead0
                    gap = (lead[0] if lead else 250.0)
                    vlead = (lead[1] if lead else v0)
                    acc_idm = float(idm_acc_np(
                        [v0], [gap], [v0 - vlead], a_max, b, vt["tau"],
                        vt["s0"], v0free)[0])
                    kin_max = v0 + a_max * DT
                    kin_min = max(0.0, v0 - b * DT)
                    kin = float(np.clip(issued, kin_min, kin_max))
                    kin_idm = float(max(0.0, min(kin, v0 + acc_idm * DT)))
                    kin_flat = float(max(kin_min,
                                         min(kin_max, v0 + acc_idm * DT)))
                    row = dict(v1=v1, kin=kin, kin_idm=kin_idm,
                               kin_flat=kin_flat, v0=v0, T=float(issued),
                               acc_idm=acc_idm, lc=lc, gap=gap,
                               emergency=(v1 < v0 - b * DT - EPS),
                               over_accel=(v1 > v0 + a_max * DT + EPS),
                               idm_binds=(kin_idm < kin - 1e-9))
                    per_mode_rows[speed_mode].append(row)
                    per_vtype[type_id].append(row)
                    if lc:
                        diag["lc_step1"] += 1
                    if row["emergency"]:
                        diag["emergency_step1"] += 1
                    if row["over_accel"]:
                        diag["over_accel_step1"] += 1
                    # ---- step 2: t+0.5 -> t+1.0 under the same T
                    if st2 is not None and vid in st2:
                        v0b = v1
                        lane_id_b = st1[vid][0]
                        pos_b = st1[vid][1]
                        v1b = st2[vid][2]
                        lc2 = bl.lc_flag.get((round(tick + 1.0, 6), vid),
                                             st2[vid][0] != lane_id_b)
                        lane_len_b, lane_speed_b = lanes[lane_id_b]
                        v0free_b = lane_speed_b * factors.get(type_id, 1.0)
                        leadb = bl.leader(bl.mapped(st1), vid)
                        gapb = (leadb[0] if leadb else 250.0)
                        vleadb = (leadb[1] if leadb else v0b)
                        acc_idm_b = float(idm_acc_np(
                            [v0b], [gapb], [v0b - vleadb], a_max, b,
                            vt["tau"], vt["s0"], v0free_b)[0])
                        kin_max_b = v0b + a_max * DT
                        kin_min_b = max(0.0, v0b - b * DT)
                        kin_b = float(np.clip(issued, kin_min_b, kin_max_b))
                        kin_idm_b = float(max(0.0, min(
                            kin_b, v0b + acc_idm_b * DT)))
                        kin_flat_b = float(max(
                            kin_min_b, min(kin_max_b,
                                           v0b + acc_idm_b * DT)))
                        rowb = dict(v1=v1b, kin=kin_b, kin_idm=kin_idm_b,
                                    kin_flat=kin_flat_b, v0=v0b,
                                    T=float(issued), acc_idm=acc_idm_b,
                                    lc=lc2, gap=gapb,
                                    emergency=(v1b < v0b - b * DT - EPS),
                                    over_accel=(v1b > v0b + a_max * DT
                                                + EPS),
                                    idm_binds=(kin_idm_b < kin_b - 1e-9))
                        per_mode_rows[speed_mode].append(rowb)
                        per_vtype[type_id].append(rowb)
                        if lc2:
                            diag["lc_step2"] += 1
                        if rowb["emergency"]:
                            diag["emergency_step2"] += 1
                        if rowb["over_accel"]:
                            diag["over_accel_step2"] += 1
                else:
                    diag["vehicle_gone"] += 1

    print(f"\nbranches measured: {n_branch}  "
          f"({time.time()-t_start:.0f}s)")
    result = dict(
        experiment="WD-01 partA_act execution-rule measurement",
        subset=f"every {WORLD_STRIDE}th sorted TRAIN world",
        worlds=names,
        branches=n_branch,
        diag=dict(diag),
        per_speed_mode={}, per_vtype={},
    )
    print("\n--- per speedMode: rule residuals |v'_realised - v'_rule| ---")
    for mode, rows in sorted(per_mode_rows.items()):
        fits = rule_residuals(rows)
        n_em = sum(r["emergency"] for r in rows)
        n_idm = sum(r["idm_binds"] for r in rows)
        n_lc = sum(r["lc"] for r in rows)
        print(f"speedMode {mode}  (n={len(rows)}, lc-excluded "
              f"{n_lc}, emergency {n_em}, idm-bound {n_idm})")
        for rule in ("kin", "kin_idm", "kin_flat"):
            f = fits[rule]
            print(f"  {rule:8s}: p50 {f['p50']:.3e}  p95 {f['p95']:.3e}  "
                  f"p99 {f['p99']:.3e}  max {f['max']:.3e}  "
                  f"exact {f['exact_rate']:.4f}  <0.01 {f['tol01_rate']:.4f}"
                  f"  signed p50 {f['signed_p50']:+.3e}")
        for tag in ("kin_idm_leader", "kin_idm_free", "kin_idm_idm_bound"):
            f = fits[tag]
            print(f"  {tag:16s}: n {f['n']}  p50 {f['p50']:.3e}  "
                  f"p95 {f['p95']:.3e}  <0.01 {f['tol01_rate']:.4f}  "
                  f"signed p50 {f['signed_p50']:+.3e}")
        result["per_speed_mode"][str(mode)] = dict(
            n=len(rows), fits=fits,
            n_lc=n_lc, n_emergency=n_em, n_idm_bound=n_idm)
    print("\n--- per CAV vType (speedMode 31) ---")
    for type_id, rows in sorted(per_vtype.items()):
        fits = rule_residuals(rows)
        f = fits["kin_idm"]
        print(f"{type_id:14s} (n={len(rows)}): kin_idm p50 "
              f"{f['p50']:.3e}  p95 {f['p95']:.3e}  max {f['max']:.3e}  "
              f"exact {f['exact_rate']:.4f}")
        result["per_vtype"][type_id] = fits
    out = HERE / "results" / "exec_rule_fit.json"
    out.write_text(json.dumps(result, indent=1, default=str),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    main()
