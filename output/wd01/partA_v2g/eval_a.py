"""WD-01-PA2G-A (registered analysis, zero simulation): Part A v2
general model error from the TRUE initial state.

REGISTRATION_PA2G_EVAL_A.md (registered 2026-09-29 before any run):

- ANALYSIS on existing logs; no SUMO/TraCI; TRAIN worlds only.  Every
  opened run file is recorded in OPENED_RUN_FILES and asserted to belong
  to a TRAIN world of chunk-2/chunk-3; CAL/VALIDATION/HOLDOUT are never
  opened (asserted at close-out).
- Windows: start every 30 s over (t0, t0+540]; H in {10, 30, 60, 120} s
  (a window is kept for H when the log covers w+H).
- Initial state: the logged vehicles inside the RSU recon zone (the
  default zone extent, params rsu_zone_default) at the window start.
- CAVs: native in these branches.
- Metrics per H over the tracked vehicles: trajectory RMSE (s, v),
  lane-change counts by type, window delay error, TTC<3 exposure error,
  hard-brake error, predicted collisions per 1000 vehicle-steps.
- Missing-arrivals curve: truth delay over the tracked vehicles only vs
  ALL vehicles (including arrivals after w), per H.
- n_free_road_assumed per window (exact truth rule + the predictor-side
  zone-truncation rule).
- Comparators: the v1 p3o engine (kind idm_merge, frozen iter-2
  candidate knobs, replica from the same true state -- declared
  deviations: humans prior-drawn vTypes, CAVs logged types, logged
  arrivals, p3d hazard via WD01_DATA_ROOT) and constant-speed
  persistence.
- Descriptive T/a sensitivity sweep (NO parameter selection):
  T in {1.0, 1.5, 2.0} x a in {0.7, 1.0, 1.5}, H = 60 rows only.

Usage:
    python eval_a.py [--max-worlds N] [--sweep-every K] [--no-p3o]
Raw rows are saved BEFORE aggregation (results/eval_a/raw_rows.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# the p3d weights for the v1 replica live in the populated untracked
# copy of output/wd01 (read-only; part 3k WD01_DATA_ROOT convention)
_P3D_ROOT = os.environ.get("WD01_DATA_ROOT", "").strip() or \
    str(Path("E:/rd-wt/wd01/output/wd01"))
os.environ.setdefault("WD01_DATA_ROOT", _P3D_ROOT)

from partA_v2g import metrics as MX  # noqa: E402
from partA_v2g.engine import (PartAV2GEngine, free_road_flags,
                              load_params, param_set_id)  # noqa: E402

import wdpaths  # noqa: E402

RESULTS = HERE / "results" / "eval_a"
RAW_ROWS = RESULTS / "raw_rows.jsonl"
DT = 0.5
HS = (10, 30, 60, 120)
H_MAX = max(HS)
SWEEP_T = (1.0, 1.5, 2.0)
SWEEP_A = (0.7, 1.0, 1.5)
SWEEP_H = 60
ANALYSIS_ID = "WD-01-PA2G-A"
CHUNKS = {
    2: ("campaign_stage1_chunk2",
        Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" /
        "campaign_stage1_v2" / "results_plan_chunk2" / "WORLD_PLAN.json"),
    3: ("campaign_stage1_chunk3",
        Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" /
        "campaign_stage1_v3" / "results_plan_chunk3" / "WORLD_PLAN.json"),
}

OPENED_RUN_FILES: set = set()
TRAIN_WORLD_NAMES: set = set()


def _open_json(path: Path):
    OPENED_RUN_FILES.add(str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_train_worlds() -> List[dict]:
    """TRAIN worlds of chunk-2 + chunk-3 (splits from the WORLD_PLANs;
    CAL/VALIDATION/HOLDOUT are refused BEFORE any run file is opened)."""
    out = []
    for chunk, (runs, plan_path) in CHUNKS.items():
        plan = _open_json(plan_path)
        for w in plan["worlds"]:
            if w.get("split") != "TRAIN":
                continue
            entry = dict(w)
            entry["chunk"] = chunk
            entry["native_run"] = Path(wdpaths.RESEARCH_DENSE_ROOT) / \
                "runs" / runs / f"{w['world']}__native" / "run" / \
                "physics_steps.jsonl"
            m = re.search(r"w\d+m(\d+)p(\d+)t(\d+)", w["world"])
            entry["demand"] = int(m.group(1)) if m else -1
            entry["penetration"] = int(m.group(2)) / 100.0 if m else \
                float("nan")
            out.append(entry)
    out.sort(key=lambda w: w["world"])
    TRAIN_WORLD_NAMES.update(w["world"] for w in out)
    return out


def assert_train_only():
    """Close-out assertion: no CAL/VALIDATION/HOLDOUT run file opened."""
    bad = []
    for p in OPENED_RUN_FILES:
        m = re.search(r"(w\S+?)__native", p)
        if m and m.group(1) not in TRAIN_WORLD_NAMES:
            bad.append(p)
    assert not bad, f"non-TRAIN run files opened: {bad}"


# ------------------------------------------------- fast model-world mapper

class ModelMapper:
    """(lane_phys, s, in_junction) from SUMO (lane_id, position) via the
    frozen p2C topology (same math as pdl.map_step, cached per lane)."""

    def __init__(self):
        sys.path.insert(0, str(WD01 / "partA_data"))
        import pdl
        topo, _, internal = pdl.topology()
        self.pdl = pdl
        self.segs = {}
        for name, li in pdl.LANE_IDS.items():
            self.segs[li] = [(s.s0, s.lane_id) for s in topo.segments_of(name)]
        self.lane_of = {}
        for li in (0, 1, 2):
            for s0, lid in self.segs[li]:
                self.lane_of[lid] = (li, s0)
        self.internal_s = {}
        for lane_id, (frm, to) in internal.items():
            for li in (0, 1, 2):
                hit = [s0 for s0, lid in self.segs[li] if lid == to]
                if hit:
                    self.internal_s[lane_id] = hit[0]
                    break

    def map(self, lane_id: str, pos: float) -> Tuple[int, float, int]:
        hit = self.lane_of.get(lane_id)
        if hit is not None:
            return hit[0], hit[1] + float(pos), 0
        if lane_id in self.internal_s:
            return -1, self.internal_s[lane_id], 1
        return -1, float("nan"), 1


def retrospective_junction(tick_rows):
    """Junction rows get their next physical lane (the p2C truth
    convention; copy of act_lib.retropective_junction)."""
    per_vid = defaultdict(list)
    for tick in sorted(tick_rows):
        for vid, st in tick_rows[tick].items():
            per_vid[vid].append((tick, st))
    fixed = {t: {} for t in tick_rows}
    for vid, seq in per_vid.items():
        nxt_lane = None
        for tick, st in reversed(seq):
            lane, s, v, a, inj = st
            if inj:
                if nxt_lane is None:
                    nxt_lane = -1
                lane = nxt_lane
            else:
                nxt_lane = lane
            fixed[tick][vid] = (lane, s, v, a, 0)
    return fixed


class NativeLog:
    """One world's native branch: model-world truth rows
    {tick_from_t0: {vid: (lane, s, v, a, type_id)}} + the CAV set.

    WD-01-PA2G-LC1 class-(ii) counting fix: the RAW junction-internal
    (tick, vid) membership is kept in `inj` so the tracked-state rule can
    exclude vehicles that are mid-junction at a window start (the
    retro-fix places them at their destination segment start, which the
    prediction side -- whose snapshots never contain junction states --
    cannot mirror)."""

    def __init__(self, entry: dict):
        self.world = entry["world"]
        self.t0 = float(entry["t0_s"])
        self.path = entry["native_run"]
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        OPENED_RUN_FILES.add(str(self.path))
        mapper = MAPPER
        type_of = mapper.pdl.TYPE_IDS
        raw = {}
        cavs = set()
        types: Dict[str, int] = {}
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                i = line.find('"time_s": ')
                if i < 0:
                    continue
                j = i + len('"time_s": ')
                k = line.find(",", j)
                t = float(line[j:k])
                tick = int(round((t - self.t0) / 0.5))
                if tick < 0:
                    continue
                row = json.loads(line)
                if row.get("cav_ids"):
                    cavs.update(row["cav_ids"])
                mrow = {}
                for vid, v in row["vehicles"].items():
                    li, s, inj = mapper.map(v["lane_id"],
                                            float(v["lane_position_m"]))
                    mrow[vid] = (li, s, float(v["speed_mps"]),
                                 float(v.get("acceleration_mps2", 0.0)),
                                 inj)
                    if vid not in types:
                        types[vid] = type_of.get(v.get("type_id"), 1)
                raw[tick] = mrow
        self.cav_ids = cavs
        self.types = types
        self.inj = {tick: frozenset(vid for vid, st in row.items()
                                    if st[4])
                    for tick, row in raw.items()
                    if any(st[4] for st in row.values())}
        fixed = retrospective_junction(raw)
        self.rows = {t: {vid: st[:4] for vid, st in row.items()}
                     for t, row in fixed.items()}
        self.t_end = self.t0 + 0.5 * max(self.rows)

    def truth_at(self, w: float) -> Dict[str, Tuple[int, float, float, float]]:
        return self.rows[int(round((w - self.t0) / 0.5))]


MAPPER: Optional[ModelMapper] = None


def zone_contains(lane: int, s: float, zone) -> bool:
    lo, hi = zone[lane]
    return (lane >= 0) and (lo <= s < hi)


def tracked_state(truth, params, inj_at=None):
    """The zone-restricted initial state + truth-exact free-road flags
    (the vehicle's true same-lane leader at w is outside the zone).

    WD-01-PA2G-LC1 class-(ii) counting fix: `inj_at` is the set of vids
    whose RAW row at this tick is junction-internal; they are excluded
    from the tracked cohort so truth and prediction are counted
    identically (prediction snapshots contain no junction states)."""
    zone = params["rsu_zone_default"]["per_lane_s"]
    vehicles = {}
    for vid, (lane, s, v, a) in truth.items():
        if inj_at is not None and vid in inj_at:
            continue
        if zone_contains(lane, s, zone):
            vehicles[vid] = dict(lane=lane, s=s, v=v, a=a)
    by_lane = defaultdict(list)
    for vid, (lane, s, _v, _a) in truth.items():
        if lane >= 0:
            by_lane[lane].append((s, vid))
    flags = []
    for vid, veh in vehicles.items():
        ahead = [(s, o) for s, o in by_lane.get(veh["lane"], [])
                 if s > veh["s"] + 1e-9]
        if ahead and min(ahead)[1] not in vehicles:
            flags.append(vid)
    return vehicles, sorted(flags)


def arrivals_between(log: NativeLog, w: float, h: float, tracked) -> List[str]:
    """Vehicles not present at w that appear in (w, w+h] (the untracked
    arrivals of the missing-arrivals curve)."""
    j0 = int(round((w - log.t0) / 0.5))
    j1 = int(round((w + h - log.t0) / 0.5))
    seen = set(log.rows[j0])
    out = []
    for j in range(j0 + 1, min(j1, max(log.rows)) + 1):
        for vid in log.rows.get(j, {}):
            if vid not in seen and vid not in tracked:
                out.append(vid)
            seen.add(vid)
    return out


# ------------------------------------------------------ const-speed model

def const_speed_rollout(params, vehicles, n_steps):
    """Constant-speed persistence: v const, s advances, exits at the
    lane end, no lane changes (the registered comparator)."""
    llen = np.asarray(params["geometry"]["lane_lengths_m"])
    names = sorted(vehicles)
    lane = {n: vehicles[n]["lane"] for n in names}
    s = {n: vehicles[n]["s"] for n in names}
    v = {n: vehicles[n]["v"] for n in names}
    snaps = [{n: (lane[n], s[n], v[n], 0.0) for n in names}]
    live = set(names)
    for _j in range(1, n_steps + 1):
        row = {}
        for n in sorted(live):
            s[n] = s[n] + v[n] * DT
            if s[n] > llen[lane[n]]:
                live.discard(n)
                continue
            row[n] = (lane[n], s[n], v[n], 0.0)
        snaps.append(row)
    return snaps


# ------------------------------------------------------ v1 p3o replica

def crn_uniform(seed, vid, tick, salt):
    """act_lib.crn_uniform copy (deterministic common random numbers)."""
    key = f"{int(seed)}|{vid}|{tick}|{salt}".encode()
    d = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(d, "little") / float(2 ** 64)


P3O_KNOBS = dict(lc_scale=0.002,
                 merge_params=dict(tau_accept=2.4, b_accept=1.0,
                                   p_coop_scale=0.0, end_force_s=60.0,
                                   v_merge_blend=0.0),
                 dawdle_scale=1.0, merge_accept_prob=0.55,
                 lc_scale_disc=0.0016)


class V1P3OReplica:
    """kind 'idm_merge' with the frozen p3o iter-2 knobs, rolled from a
    chunk-2/3 TRUE zone-restricted state.  Declared deviations from the
    chunk-1 p3o rollouts (no p2C arrays exist for these worlds): humans
    prior-drawn vTypes (10/70/20, the act_engine precedent), CAVs their
    logged type, the logged arrivals inserted at their logged ticks,
    p3d hazard weights via WD01_DATA_ROOT."""

    def __init__(self):
        sys.path.insert(0, str(WD01 / "part3"))
        import engine as E  # noqa: W004
        import p3lib as P3  # noqa: W004
        self.E = E
        self.P3 = P3
        self.mp = dict(P3O_KNOBS["merge_params"])
        self._prior_cdf = np.cumsum([0.1, 0.7, 0.2])

    def rollout(self, t0, vehicles, births, n_steps):
        import torch
        E, P3 = self.E, self.P3
        names = sorted(vehicles)

        def col(key):
            return [float(vehicles[n][key]) for n in names]

        lane = torch.tensor([int(vehicles[n]["lane"]) for n in names],
                            dtype=torch.long)
        s = torch.tensor(col("s"), dtype=torch.float64)
        v = torch.tensor(col("v"), dtype=torch.float64)
        vtype = [int(vehicles[n]["_vtype"]) for n in names]
        snaps = {0: {n: (int(lane[i]), float(s[i]), float(v[i]), 0.0)
                     for i, n in enumerate(names)}}
        lc_counts = {"merge": 0, "disc_left": 0, "disc_right": 0}
        collisions_by_step: Dict[int, int] = {}
        vstep_by_step: Dict[int, int] = {}
        ll = P3.lane_lengths()
        ll_t = torch.tensor([ll[0], ll[1], ll[2]], dtype=torch.float64)
        off_t = torch.tensor(P3.OFFSETS, dtype=torch.float64)

        def to_np(x):
            return x.detach().cpu().numpy()

        for j in range(1, n_steps + 1):
            t_now = t0 + (j - 1) * DT
            tick6 = round(t_now, 6)
            prev_row = {n: i for i, n in enumerate(names)}
            for vid, st in (births.get(j) or {}).items():
                lane = torch.cat([lane, torch.tensor([st["lane"]],
                                                     dtype=torch.long)])
                s = torch.cat([s, torch.tensor([float(st["s"])],
                                               dtype=torch.float64)])
                v = torch.cat([v, torch.tensor([float(st["v"])],
                                               dtype=torch.float64)])
                vtype.append(int(st["_vtype"]))
                names.append(vid)
            n_t = len(names)
            if n_t == 0:
                snaps[j] = {}
                continue
            v_prev = v.clone()
            lane_np = lane.numpy()
            s_np = to_np(s)
            v_np = to_np(v)
            vt_np = np.asarray(vtype)
            lim = torch.as_tensor(E.lane_speed_limits(lane_np, s_np),
                                  dtype=torch.float64)
            vcap = 1.2 * lim
            # -- M2 mandatory gap-accepted merge (front-most first)
            merged = self._gap_accept_merge(lane_np, s_np, v_np, vt_np,
                                            tick6, names)
            # -- M4 discretionary 0<->1 hazard
            cav_f = np.zeros(n_t)
            sl = torch.as_tensor(P3.neighbour_slots(lane_np, s_np, v_np,
                                                    cav_f))
            feats = torch.zeros((n_t, 19))
            feats[:, 0] = v.float() / 40.0
            feats[:, 1] = torch.clamp((454.10 - s.float()) / 500.0, 0.0, 1.0)
            feats[:, 2:14] = torch.clamp(
                sl[:, :, 1:3].reshape(n_t, 12).float() / 250.0, -1.0, 1.0)
            feats[:, 14] = ((lane == 2) & (s >= E.MERGE_ZONE_S0)).float()
            with torch.no_grad():
                logits = E._p3d_hazard()(feats)
            pl = torch.sigmoid(logits[:, 0]).numpy()
            pr_ = torch.sigmoid(logits[:, 1]).numpy()
            ul = np.array([crn_uniform(0, str(names[r]), tick6, "lcl")
                           for r in range(n_t)])
            ur = np.array([crn_uniform(0, str(names[r]), tick6, "lcr")
                           for r in range(n_t)])
            fire_l = (lane_np == 1) & (ul < pl * P3O_KNOBS["lc_scale_disc"])
            fire_r = (lane_np == 0) & (ur < pr_ * P3O_KNOBS["lc_scale_disc"])
            new_lane = lane_np.copy()
            new_lane[merged] = 1
            new_lane[fire_l] -= 1
            new_lane[fire_r] += 1
            changed = new_lane != lane_np
            lc_counts["merge"] += int(merged.sum())
            lc_counts["disc_left"] += int((fire_l & (new_lane == 0)).sum())
            lc_counts["disc_right"] += int((fire_r & (new_lane == 1)).sum())
            if changed.any():
                s = torch.where(torch.as_tensor(changed),
                                s + off_t[torch.as_tensor(new_lane)]
                                - off_t[lane], s)
                lane = torch.as_tensor(new_lane, dtype=torch.long)
                lane_np = lane.numpy()
                s_np = to_np(s)
                lim = torch.as_tensor(E.lane_speed_limits(lane_np, s_np),
                                      dtype=torch.float64)
                vcap = 1.2 * lim
            # -- per-type IDM + M1 obstacle + dawdle (CRN)
            leader = np.full(n_t, -1, dtype=np.int64)
            gap = np.full(n_t, P3.GAP_MAX)
            lv = np.zeros(n_t)
            for li in (0, 1, 2):
                idx = np.where(lane_np == li)[0]
                if len(idx) < 2:
                    continue
                o = idx[np.argsort(s_np[idx])]
                for q in range(len(o) - 1):
                    f, l_ = int(o[q]), int(o[q + 1])
                    leader[f] = l_
                    gap[f] = s_np[l_] - 5.0 - s_np[f]
                    lv[f] = v_np[l_]
            a_max = torch.as_tensor([P3.VTYPES[int(t)]["a_max"]
                                     for t in vtype], dtype=torch.float64)
            b = torch.as_tensor([P3.VTYPES[int(t)]["b"] for t in vtype],
                                dtype=torch.float64)
            tau = torch.as_tensor([P3.VTYPES[int(t)]["tau"] for t in vtype],
                                  dtype=torch.float64)
            s0v = torch.as_tensor([P3.VTYPES[int(t)]["s0"] for t in vtype],
                                  dtype=torch.float64)
            sf = torch.as_tensor([P3.VTYPES[int(t)]["sf"] for t in vtype],
                                 dtype=torch.float64)
            v0 = sf * lim
            gap_t = torch.as_tensor(gap)
            dv_t = torch.where(torch.as_tensor(leader >= 0),
                               v - torch.as_tensor(lv), torch.zeros_like(v))
            gap_eff = torch.where(torch.as_tensor(leader >= 0), gap_t,
                                  torch.full_like(gap_t, P3.GAP_MAX))
            acc = P3.idm_acc(v, gap_eff, dv_t, a_max, b, tau, s0v, v0)
            on2 = lane == 2
            if bool(on2.any()):
                acc = torch.where(
                    on2, torch.minimum(acc, E.lane_end_obstacle_acc(
                        v, s, on2, a_max, b, tau, s0v, v0)), acc)
            acc = acc.clamp(-9.0, 3.0)
            dawd = np.array([
                crn_uniform(0, str(names[r]), tick6, "dawdle")
                * E.SIGMA_ACCEL[int(vtype[r])] * DT for r in range(n_t)])
            v_np_new = np.maximum(
                to_np(v) + to_np(acc) * DT
                - dawd * P3O_KNOBS["dawdle_scale"], 0.0)
            v_new = torch.minimum(torch.as_tensor(v_np_new,
                                                  dtype=torch.float64), vcap)
            s_new = s + v_new * DT
            # collisions: same-lane order inversions after the step
            coll = 0
            for i in range(n_t):
                jj = int(leader[i])
                if jj < 0 or lane_np[i] != lane_np[jj]:
                    continue
                if float(s_new[i]) >= float(s_new[jj]) - 1e-9:
                    coll += 1
            if coll:
                collisions_by_step[j] = coll
            vstep_by_step[j] = n_t
            v, s = v_new, s_new
            gone = s > ll_t[lane]
            if bool(gone.any()):
                keep = np.where(~to_np(gone))[0]
                names = [names[r] for r in keep]
                lane = lane[torch.as_tensor(keep, dtype=torch.long)]
                s = s[torch.as_tensor(keep, dtype=torch.long)]
                v = v[torch.as_tensor(keep, dtype=torch.long)]
                vtype = [vtype[r] for r in keep]
            snaps[j] = {n: (int(lane[i]), float(s[i]), float(v[i]),
                            float((v[i] - float(v_prev[prev_row[n]])) / DT
                                  if n in prev_row and
                                  prev_row[n] < len(v_prev) else 0.0))
                        for i, n in enumerate(names)}
        return snaps, lc_counts, collisions_by_step, vstep_by_step

    def _gap_accept_merge(self, lane_np, s_np, v_np, vt_np, tick6, names):
        E, P3 = self.E, self.P3
        out = np.zeros(len(lane_np), dtype=bool)
        cand = np.where((lane_np == 2) & (s_np >= E.MERGE_ZONE_S0))[0]
        if not len(cand):
            return out
        cand = cand[np.argsort(-s_np[cand])]
        on1 = np.where(lane_np == 1)[0]
        occ_d = (P3.OFFSETS[1] - s_np[on1]).astype(np.float64) \
            if len(on1) else np.zeros(0)
        occ_v = v_np[on1].astype(np.float64).copy() if len(on1) \
            else np.zeros(0)
        occ_s0 = np.array([P3.VTYPES[int(vt_np[q])]["s0"] for q in on1]) \
            if len(on1) else np.zeros(0)
        tau_a, b_a = float(self.mp["tau_accept"]), float(self.mp["b_accept"])
        force_s = float(self.mp.get("end_force_s", 0.0))
        p_accept = float(P3O_KNOBS["merge_accept_prob"])
        for i in cand:
            d_i = P3.OFFSETS[2] - s_np[i]
            tau_i, b_i = E.end_force_params(s_np[i], tau_a, b_a, force_s)
            ahead = occ_d < d_i
            gap_lead = (P3.GAP_MAX if not ahead.any()
                        else float(d_i - occ_d[ahead].max()))
            behind = np.where(occ_d > d_i)[0]
            if len(behind):
                jj = behind[np.argmin(occ_d[behind])]
                gap_foll = float(occ_d[jj] - d_i)
                v_f = float(occ_v[jj])
                s0_f = float(occ_s0[jj])
            else:
                gap_foll, v_f, s0_f = P3.GAP_MAX, 0.0, 0.0
            s0_i = P3.VTYPES[int(vt_np[i])]["s0"]
            ok = bool(E.merge_acceptance(
                np.array([gap_lead]), np.array([v_np[i]]),
                np.array([s0_i]), np.array([gap_foll]),
                np.array([v_f]), np.array([s0_f]), tau_i, b_i)[0])
            if ok and p_accept < 1.0:
                u = crn_uniform(0, str(names[int(i)]), tick6, "merge_thin")
                ok = u < p_accept
            if ok:
                out[i] = True
                occ_d = np.append(occ_d, d_i)
                occ_v = np.append(occ_v, float(v_np[i]))
                occ_s0 = np.append(occ_s0, s0_i)
        return out


# ---------------------------------------------------------------- window

def window_rows(params, log: NativeLog, entry, w: float,
                replica: Optional[V1P3OReplica], sweep_params=None,
                emit_hs=HS):
    """All model rows for one window: v2g (or one sweep variant),
    const-speed, optional the v1 p3o replica; plus the truth metrics.
    Per-H lane changes / collisions are sliced from the window, never
    taken from the full rollout."""
    truth_w = log.truth_at(w)
    j0 = int(round((w - log.t0) / 0.5))
    inj = getattr(log, "inj", {})
    vehicles, free_true = tracked_state(truth_w, params,
                                        inj_at=inj.get(j0))
    assert vehicles, f"empty zone state at w={w}"
    vids = sorted(vehicles)
    zone = params["rsu_zone_default"]["per_lane_s"]
    llen = params["geometry"]["lane_lengths_m"]
    free_pred = free_road_flags(vehicles, zone, llen)
    n_roll = int(round(H_MAX / DT))
    rows: List[dict] = []

    def emit(model, snaps, coll_by_step, H):
        n_h = int(round(H / DT))
        pred = _series(model, snaps)
        rmse = pred.rmse(_truth_tr, n_h)
        st_p = pred.stats(n_h, params)
        st_t = _truth_tr.stats(n_h, params)
        st_all = _truth_all.stats(n_h, params)
        arr = arrivals_between(log, w, H, set(vids))
        vsteps = sum(len(snaps[jj]) for jj in range(1, n_h + 1))
        coll = sum(coll_by_step.get(jj, 0) for jj in range(1, n_h + 1))
        rows.append(dict(
            world=entry["world"], chunk=entry["chunk"], w=w, H=H,
            model=model, penetration=entry["penetration"],
            demand=entry["demand"],
            n_tracked=len(vids),
            n_free_road_true=len(free_true),
            n_free_road_pred=len(free_pred),
            n_arrivals=len(arr),
            s_rmse=rmse["s_rmse"], v_rmse=rmse["v_rmse"],
            n_matched=rmse["n_matched"],
            delay_pred=st_p["delay"], delay_true_tracked=st_t["delay"],
            delay_err=st_p["delay"] - st_t["delay"],
            delay_true_all=st_all["delay"],
            ttc3_pred=st_p["expo3"], ttc3_true=st_t["expo3"],
            ttc3_err=st_p["expo3"] - st_t["expo3"],
            hard_pred=st_p["hard"], hard_true=st_t["hard"],
            hard_err=st_p["hard"] - st_t["hard"],
            lc_pred=st_p["lc"], lc_true=st_t["lc"],
            collisions_pred=coll, vehicle_steps_pred=vsteps,
        ))

    # series caches (built once per window): tracked truth, all truth
    j_roll = n_roll
    truth_snaps_all = [log.rows.get(j0 + jj, {}) for jj in range(j_roll + 1)]
    _truth_tr = MX.WindowSeries(truth_snaps_all, vids)
    _truth_all = MX.WindowSeries(truth_snaps_all, None)
    _series_cache: Dict[str, MX.WindowSeries] = {}

    def _series(model, snaps):
        if model not in _series_cache:
            _series_cache[model] = MX.WindowSeries(snaps, vids)
        return _series_cache[model]

    def hs_ok():
        return [H for H in emit_hs if w + H <= log.t_end + 1e-9]

    # ---- v2g (or the sweep variant)
    p = params if sweep_params is None else sweep_params
    eng = PartAV2GEngine(p, t0=w, vehicles=vehicles)
    t_a = time.perf_counter()
    eng.run(n_roll)
    t_engine = time.perf_counter() - t_a
    snaps = MX.snaps_list(eng)
    model_name = "v2g" if sweep_params is None else "v2g_sweep"
    for H in hs_ok():
        emit(model_name, snaps, eng.collisions_by_step, H)
    # ---- constant-speed persistence
    cs = const_speed_rollout(params, vehicles, n_roll)
    for H in hs_ok():
        emit("const_speed", cs, {}, H)
    # ---- v1 p3o replica (with the logged arrivals of the window)
    if replica is not None:
        births: Dict[int, dict] = {}
        seen = set(vids)
        for jj in range(1, n_roll + 1):
            rowj = log.rows.get(j0 + jj, {})
            for vid, st in rowj.items():
                if vid not in seen and st[0] >= 0:
                    births.setdefault(jj, {})[vid] = dict(
                        lane=st[0], s=st[1], v=st[2],
                        is_cav=vid in log.cav_ids,
                        _vtype=log.types.get(vid, 1))
                seen.add(vid)
        v0 = {vid: dict(rec, is_cav=vid in log.cav_ids,
                        _vtype=log.types.get(vid, 1))
              for vid, rec in vehicles.items()}
        # humans: prior vType draws (CRN), CAVs logged types
        for vid, rec in v0.items():
            if not rec["is_cav"]:
                u = crn_uniform(0, vid, 0, "vtype")
                rec["_vtype"] = int(np.searchsorted(
                    np.cumsum([0.1, 0.7, 0.2]), u, "left"))
        snaps_p, _lc, coll_p, _vs = replica.rollout(w, v0, births, n_roll)
        snaps_p = [snaps_p.get(jj, {}) for jj in range(n_roll + 1)]
        for H in hs_ok():
            emit("v1_p3o_replica", snaps_p, coll_p, H)
    return rows, t_engine


# ---------------------------------------------------------------- main

def main():
    global MAPPER
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-worlds", type=int, default=0)
    ap.add_argument("--world-step", type=int, default=1,
                    help="run every k-th TRAIN world from the sorted "
                    "list (a deterministic stratified subset when "
                    "runtime requires fewer worlds; stated in the "
                    "report)")
    ap.add_argument("--sweep-every", type=int, default=4,
                    help="run the T/a sweep on every k-th world (0=off)")
    ap.add_argument("--no-p3o", action="store_true")
    args = ap.parse_args()

    t_start = time.perf_counter()
    params = load_params()
    MAPPER = ModelMapper()
    worlds = load_train_worlds()
    if args.world_step > 1:
        worlds = worlds[::args.world_step]
    if args.max_worlds:
        worlds = worlds[:args.max_worlds]
    print(f"TRAIN worlds total {len(TRAIN_WORLD_NAMES)}, running "
          f"{len(worlds)}; p3o {'ON' if not args.no_p3o else 'OFF'}; "
          f"sweep every {args.sweep_every}", flush=True)

    replica = None if args.no_p3o else V1P3OReplica()
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = open(RAW_ROWS, "w", encoding="utf-8")
    n_windows = 0
    t_eng_total = 0.0
    t_p3o = 0.0
    for wi, entry in enumerate(worlds):
        tw = time.perf_counter()
        try:
            log = NativeLog(entry)
        except FileNotFoundError as e:
            print(f"  {entry['world']}: SKIP ({e})", flush=True)
            continue
        rows_world = 0
        for k in range(1, 19):
            w = log.t0 + 30.0 * k
            if w > log.t_end + 1e-9:
                break
            t0_p3o = time.perf_counter()
            rows, t_engine = window_rows(params, log, entry, w, replica)
            t_eng_total += t_engine
            if replica is not None:
                t_p3o += time.perf_counter() - t0_p3o - t_engine
            for r in rows:
                raw.write(json.dumps(r) + "\n")
            rows_world += len(rows)
            n_windows += 1
        # descriptive T/a sweep (H = 60 only), every k-th world
        if args.sweep_every and (wi % args.sweep_every == 0):
            for T in SWEEP_T:
                for a in SWEEP_A:
                    if abs(T - 1.5) < 1e-9 and abs(a - 1.0) < 1e-9:
                        continue          # the nominal row already exists
                    sp = json.loads(json.dumps(params))
                    sp["idm"]["T_s"] = T
                    sp["idm"]["a_max_mps2"] = a
                    for kk in range(1, 19):
                        w = log.t0 + 30.0 * kk
                        if w > log.t_end + 1e-9:
                            break
                        rows, t_engine = window_rows(
                            params, log, entry, w, None, sweep_params=sp,
                            emit_hs=(SWEEP_H,))
                        t_eng_total += t_engine
                        for r in rows:
                            r["T_s"] = T
                            r["a_max"] = a
                            raw.write(json.dumps(r) + "\n")
        raw.flush()
        print(f"  [{wi + 1}/{len(worlds)}] {entry['world']}: "
              f"{rows_world} rows, {time.perf_counter() - tw:.1f}s",
              flush=True)
    raw.close()

    assert_train_only()

    import pandas as pd
    df = pd.read_json(RAW_ROWS, lines=True)
    summary = aggregate(df)
    out = dict(
        analysis_id=ANALYSIS_ID,
        status="completed",
        n_train_worlds_total=len(TRAIN_WORLD_NAMES),
        world_step=args.world_step,
        n_worlds_run=len(worlds),
        n_windows=n_windows,
        param_set_id=param_set_id(),
        wall_clock_s=time.perf_counter() - t_start,
        t_engine_s=t_eng_total,
        t_p3o_s=t_p3o,
        sweep_every=args.sweep_every,
        opened_run_files=sorted(OPENED_RUN_FILES),
        assert_train_only="passed",
        summary=summary,
    )
    (RESULTS / "summary.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("per-model H rows:")
    for k, v in summary["per_model_h"].items():
        print(" ", k, "n", v["n"], "s_rmse", round(v["s_rmse_mean"], 3),
              "delay_err", round(v["delay_err_mean"], 3))
    print("DONE", ANALYSIS_ID, f"wall {out['wall_clock_s']:.0f}s")


def aggregate(df):
    """Registered aggregations from the RAW rows."""
    base = df[df.model.isin(["v2g", "const_speed", "v1_p3o_replica"])]
    per_mh = {}
    for (m, H), g in base.groupby(["model", "H"]):
        per_mh[f"{m}|H{int(H)}"] = dict(
            n=len(g),
            s_rmse_mean=float(g.s_rmse.mean()),
            v_rmse_mean=float(g.v_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()),
            delay_err_mae=float(g.delay_err.abs().mean()),
            ttc3_err_mean=float(g.ttc3_err.mean()),
            hard_err_mean=float(g.hard_err.mean()),
            lc_merge_pred=int(g.lc_pred.apply(lambda d: d["merge"]).sum()),
            lc_merge_true=int(g.lc_true.apply(lambda d: d["merge"]).sum()),
            lc_disc_pred=int(g.lc_pred.apply(
                lambda d: d["disc_left"] + d["disc_right"]).sum()),
            lc_disc_true=int(g.lc_true.apply(
                lambda d: d["disc_left"] + d["disc_right"]).sum()),
            collisions_pred=int(g.collisions_pred.sum()),
            vehicle_steps_pred=int(g.vehicle_steps_pred.sum()),
            collisions_per_1000=float(
                g.collisions_pred.sum() /
                max(g.vehicle_steps_pred.sum() / 1000.0, 1e-9)),
            n_free_road_true_mean=float(g.n_free_road_true.mean()),
            n_free_road_pred_mean=float(g.n_free_road_pred.mean()),
            n_tracked_mean=float(g.n_tracked.mean()),
        )
    curve = {}
    tr = df[df.model == "v2g"]
    for H, g in tr.groupby("H"):
        curve[f"H{int(H)}"] = dict(
            delay_true_tracked_mean=float(g.delay_true_tracked.mean()),
            delay_true_all_mean=float(g.delay_true_all.mean()),
            arrivals_mean=float(g.n_arrivals.mean()),
            untracked_delay_share=float(
                (g.delay_true_all - g.delay_true_tracked).sum()
                / g.delay_true_all.sum()),
        )
    sweep = []
    sw = df[df.model == "v2g_sweep"]
    if len(sw) and "T_s" in sw:
        for (T, a), g in sw.groupby(["T_s", "a_max"]):
            sweep.append(dict(T_s=float(T), a_max=float(a), n=len(g),
                              s_rmse_mean=float(g.s_rmse.mean()),
                              v_rmse_mean=float(g.v_rmse.mean()),
                              delay_err_mean=float(g.delay_err.mean())))
    by_pen = {}
    for (pen, H), g in tr.groupby(["penetration", "H"]):
        by_pen[f"p{pen}|H{int(H)}"] = dict(
            n=len(g), s_rmse_mean=float(g.s_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()))
    by_dem = {}
    for (dem, H), g in tr.groupby(["demand", "H"]):
        by_dem[f"m{int(dem)}|H{int(H)}"] = dict(
            n=len(g), s_rmse_mean=float(g.s_rmse.mean()),
            delay_err_mean=float(g.delay_err.mean()))
    return dict(per_model_h=per_mh, missing_arrivals=curve, sweep=sweep,
                by_penetration=by_pen, by_demand=by_dem)


if __name__ == "__main__":
    main()
