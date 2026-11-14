"""WD-01 partA_act shared library: chunk-1 TRAIN branch worlds, truth
mapping into the model-world (lane, s) coordinates, action schedules,
common-random-number draws, contract physics delay, and window metrics.

A2 step 2 (action conditioning): the engine kind 'idm_action'
(act_engine.py) rolls the structured Part A engine from a chunk-1 branch
restore state; CAVs follow their per-CAV action through the verified
partA_ctrl emulators with the commanded speed executed by the MEASURED
SUMO execution rule; humans stay the idm_merge model.

ZERO SIMULATION: the chunk-1 jsonl logs and the frozen network are read
as data only.  TRAIN worlds only - HOLDOUT is refused by ActWorld,
CALIBRATION is reserved and never opened here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WDO = HERE.parent
if str(WDO) not in sys.path:                 # wdpaths lives at output/wd01
    sys.path.insert(0, str(WDO))
for _p in (WDO / "partA_ctrl", WDO / "part3", WDO / "part3k",
           WDO / "partA_data", WDO / "part4" / "delay"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import wdpaths  # noqa: E402

# p3d weights (posterior/hazard) live in the untracked data root copy
os.environ.setdefault(
    "WD01_DATA_ROOT", str(Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01"))

import refload  # noqa: E402  (partA_ctrl, read-only reference loader)

refload.load_reference()
import emulators  # noqa: E402
import engine as E  # noqa: E402  (part3 idm_merge dynamics + merge helpers)
import p3lib as P3  # noqa: E402
import pdl as PD  # noqa: E402

DT = 0.5
BRANCH_ARMS = ["native", "pi", "fixed1", "fixed2",
               "switching1", "switching2", "switching3"]
H30_TICKS = 60                      # 30 s commit window at 0.5 s
VEHICLE_LENGTH_M = 5.0
SENSOR_RANGE_M = 100.0
TYPE_OF = PD.TYPE_IDS               # aggressive_car/normal_car/cautious_car


# ------------------------------------------------------------------ guards

def train_worlds():
    """{world: plan entry} for the 49 TRAIN worlds (HOLDOUT refused)."""
    plan = json.load(open(refload.WORLD_PLAN, encoding="utf-8"))
    return {w["world"]: w for w in plan["worlds"] if w["split"] == "TRAIN"}


def is_train_world(world):
    return world in train_worlds()


def penetration_of(world):
    m = re.search(r"p(\d+)t", world)
    return int(m.group(1)) / 100.0 if m else float("nan")


# ------------------------------------------------------------------ draws

def crn_uniform(seed, vid, tick, salt):
    """Uniform [0, 1) common across arms for the same (seed, vid, tick,
    salt) - the DESIGN_PARTS_A_B.md section 2.3 common-random-numbers
    rule (the noise seed is common across plans and native)."""
    key = f"{int(seed)}|{vid}|{tick}|{salt}".encode()
    d = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(d, "little") / float(2 ** 64)


def crn_vtype_birth(seed, vid):
    """Prior vType draw for a birth (10/70/20), common across arms."""
    u = crn_uniform(seed, vid, 0, "vtype")
    return int(np.searchsorted(np.cumsum([0.1, 0.7, 0.2]), u, "left"))


# ------------------------------------------------------------- jsonl io

def iter_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def receipt_ok(run_name):
    """True iff the branch run has a RUN_RECEIPT with exit_status
    complete (same gate as partA_ctrl E4)."""
    p = refload.CHUNK1_RUNS / run_name / "RUN_RECEIPT.json"
    if not p.exists():
        return False
    return json.loads(p.read_text(encoding="utf-8")).get(
        "exit_status") == "complete"


def speed_factors(world):
    """{vType id: speedFactor} of one world's scenario (default 1.0)."""
    route = (refload.CHUNK1_RUNS / f"{world}__base" / "run" / "scenario"
             / "requests.rou.xml")
    factors = {}
    for match in re.finditer(r'<vType\b[^>]*>', route.read_text(encoding="utf-8")):
        tag = match.group(0)
        vid = re.search(r'id="([^"]+)"', tag)
        sf = re.search(r'speedFactor="([^"]+)"', tag)
        if vid:
            factors[vid.group(1)] = float(sf.group(1)) if sf else 1.0
    return factors


def base_row_at_t0(world, t0):
    """The base run's final physics row at t0 (the saveState restore
    source; exact partA_ctrl E4 convention)."""
    path = refload.CHUNK1_RUNS / f"{world}__base" / "run" / \
        "physics_steps.jsonl"
    row = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if abs(float(r["time_s"]) - t0) < 1e-6:
                row = r
    return row


# --------------------------------------------------- model-world mapping

def row_states_raw(row):
    """{vid: (lane_id, position, speed, type_id)} of one raw physics
    row."""
    return {vid: (v["lane_id"], float(v["lane_position_m"]),
                  float(v["speed_mps"]), v["type_id"])
            for vid, v in row["vehicles"].items()}


def map_row(row):
    """One raw physics row -> {vid: (lane, s, v, a, inj)} in model-world
    coordinates via pdl.map_step (junction internal lanes -> lane -1,
    s = traversal boundary)."""
    out = {}
    for vid, v in row["vehicles"].items():
        lane, s, inj = PD.map_step(v["lane_id"],
                                   float(v["lane_position_m"]))
        out[vid] = (lane, float(s), float(v["speed_mps"]),
                    float(v.get("acceleration_mps2", 0.0)), inj)
    return out


def retrospective_junction(tick_rows):
    """Assign junction rows their next physical lane (the documented p2C
    truth convention).  tick_rows: {tick: {vid: (lane, s, v, a, inj)}}.
    Returns the same structure with inj rows replaced by
    (next_lane, s_boundary, v, a, 0)."""
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
                    nxt_lane = -1        # window ends inside a junction
                lane = nxt_lane
            else:
                nxt_lane = lane
            fixed[tick][vid] = (lane, s, v, a, 0)
    return fixed


# ------------------------------------------------------------- ActWorld

class ActWorld:
    """One TRAIN world's branch data for the 30 s commit window.

    Loads (read-only, single filtered pass per branch file) for the
    branch arms with complete receipts:
    - t0 restore state (base run's final row, mapped to model-world);
    - per-arm truth rows t0..t0+30 mapped + junction-retrospective;
    - per-arm births (first appearance after t0, truth (lane, s, v),
      authoritative is_cav from insertions.jsonl, type from the row);
    - per-arm dispatcher action schedule {(tick, vid): action};
    - per-arm logged commands {(tick, vid): (owner, issued, fallback)};
    - truth collisions inside the window;
    - 20-tick history at t0 from the base run (kept for the record;
      the engine draws human vTypes from the chunk-1 prior - see
      act_engine).

    Raises on non-TRAIN worlds (HOLDOUT never opened; CALIBRATION
    reserved).
    """

    def __init__(self, world, t0, arms=BRANCH_ARMS, window_s=30.0):
        worlds = train_worlds()
        if world not in worlds:
            raise ValueError(f"{world} is not a TRAIN world (refused)")
        self.world = world
        self.plan = worlds[world]
        self.t0 = float(t0)
        assert abs(self.plan["t0_s"] - self.t0) < 1e-6
        self.window_s = float(window_s)
        self.k_ticks = int(round(self.window_s / DT))
        self.factors = speed_factors(world)
        self.arms = [a for a in arms if receipt_ok(f"{world}__{a}")]
        base = base_row_at_t0(world, self.t0)
        self.t0_raw = row_states_raw(base)
        self.t0_map = map_row(base)
        self.cav_ids = sorted(set(base["cav_ids"]))
        self._truth_cache = {}
        self._actions_cache = {}
        self._commands_cache = {}
        self._collisions_cache = {}
        self._births_cache = {}
        self.hist20 = self._load_hist20()

    # ------------------------------------------------------------ per arm
    def _scan_branch(self, arm):
        """One filtered pass over the branch's physics_steps.jsonl:
        (truth tick rows mapped, birth candidates, collision records).
        Births: first appearance at tick j in 1..60 with a mapped
        physical lane, carrying the truth (lane, s, v), the row's
        type_id and the authoritative insertions.jsonl is_cav."""
        is_cav = {}
        for j, rows in cav_births(self.world, arm, self.t0,
                                  self.k_ticks).items():
            is_cav.update(rows)
        want = {round(self.t0 + j * DT, 6) for j
                in range(self.k_ticks + 1)}
        raw = {self.t0: self.t0_map}
        coll = []
        seen = set(self.t0_map)
        births = {}
        run = refload.CHUNK1_RUNS / f"{self.world}__{arm}" / "run"
        with open(run / "physics_steps.jsonl", encoding="utf-8") as fh:
            for line in fh:
                i = line.find('"time_s": ')
                if i < 0:
                    continue
                j = i + len('"time_s": ')
                k = line.find(",", j)
                try:
                    t = float(line[j:k])
                except ValueError:
                    continue
                key = round(t, 6)
                ji = int(round((key - self.t0) / DT))
                in_window = key in want
                if not in_window and not (1 <= ji <= self.k_ticks):
                    continue
                row = json.loads(line)
                if in_window:
                    raw[key] = map_row(row)
                    if row.get("colliding_ids"):
                        coll.append((float(row["time_s"]),
                                     list(row["colliding_ids"])))
                mapped = raw.get(key)
                if mapped is None:
                    continue
                for vid, v in row["vehicles"].items():
                    if vid in seen:
                        continue
                    seen.add(vid)
                    if not in_window:
                        continue
                    st = mapped[vid]
                    if st[0] < 0:
                        continue
                    births.setdefault(ji, []).append(
                        (vid, st[0], st[1], st[2],
                         bool(is_cav.get(vid)), v["type_id"]))
        tick_rows = {j: raw[round(self.t0 + j * DT, 6)]
                     for j in range(self.k_ticks + 1)}
        self._truth_raw_cache = getattr(self, "_truth_raw_cache", {})
        self._truth_raw_cache[arm] = {
            j: {vid: st for vid, st in tick_rows[j].items()}
            for j in tick_rows}
        fixed = retrospective_junction(tick_rows)
        truth = {j: {vid: (st[0], st[1], st[2], st[3])
                     for vid, st in fixed[j].items()}
                 for j in fixed}
        self._truth_cache[arm] = truth
        self._collisions_cache[arm] = coll
        self._births_cache[arm] = births

    def truth_rows_teacher(self, arm):
        """Un-retrospected truth rows for teacher forcing: junction rows
        keep lane = -1 (the model world has no junction states, so a
        CAV inside a junction is absent from the teacher state - the
        same rows the E4 liveness gate leaves unchecked)."""
        if arm not in self._truth_cache:
            self._scan_branch(arm)
        return self._truth_raw_cache[arm]

    def truth_rows(self, arm):
        """{j (0..60): {vid: (lane, s, v, a)}}; j=0 is the shared t0
        restore row."""
        if arm not in self._truth_cache:
            self._scan_branch(arm)
        return self._truth_cache[arm]

    def truth_collisions(self, arm):
        """[(t, colliding_ids)] inside the window (raw log)."""
        if arm not in self._truth_cache:
            self._scan_branch(arm)
        return self._collisions_cache[arm]

    def actions(self, arm):
        """{(control tick, vid): action string} - first-substep
        dispatcher stamps back-dated to the control tick (the E4
        convention: stamp t+0.5 carries the decision at tick t; the
        LAST row per (tick, vid) wins, E4's dict semantics)."""
        if arm not in self._actions_cache:
            out = {}
            run = refload.CHUNK1_RUNS / f"{self.world}__{arm}" / "run"
            lo = self.t0 - 0.5 - 1e-9
            hi = self.t0 + self.window_s
            with open(run / "dispatcher_rows.jsonl",
                      encoding="utf-8") as fh:
                for line in fh:
                    if '"timestamp_s"' not in line:
                        continue
                    i = line.find('"timestamp_s": ')
                    j = i + len('"timestamp_s": ')
                    k = line.find(",", j)
                    try:
                        ts = float(line[j:k])
                    except ValueError:
                        continue
                    if abs((ts * 2.0) % 2.0 - 1.0) > 1e-9:
                        continue    # keep the first-substep stamp only
                    tick = round(ts - 0.5, 6)
                    if not (lo <= tick < hi):
                        continue
                    iv = line.find('"vehicle_id": ')
                    kv = line.find('"', iv + len('"vehicle_id": '))
                    kv2 = line.find('"', kv + 1)
                    vid = line[kv + 1:kv2]
                    ia = line.find('"action": ')
                    ka = line.find('"', ia + len('"action": '))
                    ka2 = line.find('"', ka + 1)
                    action = line[ka + 1:ka2]
                    out[(tick, vid)] = action
            self._actions_cache[arm] = out
        return self._actions_cache[arm]

    def commands(self, arm):
        """{(control tick, vid): (owner, issued, fallback)} from
        suffix_commands.jsonl inside the window."""
        if arm not in self._commands_cache:
            out = {}
            run = refload.CHUNK1_RUNS / f"{self.world}__{arm}" / "run"
            lo = self.t0 - 1e-9
            hi = self.t0 + self.window_s
            with open(run / "suffix_commands.jsonl",
                      encoding="utf-8") as fh:
                for line in fh:
                    if '"kind": "command"' not in line:
                        continue
                    i = line.find('"timestamp_s": ')
                    j = i + len('"timestamp_s": ')
                    k = line.find(",", j)
                    tick = round(float(line[j:k]), 6)
                    if not (lo <= tick < hi):
                        continue
                    row = json.loads(line)
                    out[(tick, row["vehicle_id"])] = (
                        row["owner"], row["issued_target_speed_mps"],
                        row["fallback_reason"])
            self._commands_cache[arm] = out
        return self._commands_cache[arm]

    def births(self, arm):
        """{j (1..60): [(vid, lane, s, v, is_cav, type_str)]} - first
        appearance in the branch truth with the truth state (the part-3
        departed-only convention)."""
        if arm not in self._births_cache:
            self._scan_branch(arm)
        return self._births_cache[arm]

    def cav_true_types(self):
        """{vid: vtype int} for the t0 CAV roster from the restore row."""
        return {vid: TYPE_OF.get(self.t0_raw[vid][3], 1)
                for vid in self.cav_ids if vid in self.t0_raw}

    def true_types_t0(self):
        """{vid: vtype int} for ALL t0 vehicles (CAVs keep their logged
        type; ideal sensing)."""
        return {vid: TYPE_OF.get(st[3], 1)
                for vid, st in self.t0_raw.items()}

    def _load_hist20(self):
        """20-tick (v, a, ds) history at t0 from the base run, EXACT
        p3lib.history_at convention: h[-1] = (v, a, 0) at the newest
        tick, ds = forward progress to the next tick, zero-padded at the
        OLD end; every vehicle alive at t0 with >= 1 row gets an entry
        (p3lib convention), so the p3d posterior sees what it was
        trained on."""
        want = {round(self.t0 - k * DT, 6) for k in range(20)}
        rows = {}
        path = (refload.CHUNK1_RUNS / f"{self.world}__base" / "run"
                / "physics_steps.jsonl")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                i = line.find('"time_s": ')
                j = i + len('"time_s": ')
                k = line.find(",", j)
                t = round(float(line[j:k]), 6)
                if t in want:
                    rows[t] = json.loads(line)
        seq = defaultdict(dict)   # vid -> {k(0=oldest..19=t0): (v,a,s)}
        for tick in sorted(rows):
            k = int(round((tick - self.t0) / DT)) + 19
            for vid, (lane, s, v, a, inj) in map_row(rows[tick]).items():
                seq[vid][k] = (v, a, s)
        out = {}
        for vid, cells in seq.items():
            if 19 not in cells:
                continue
            h = np.zeros((20, 3), dtype=np.float32)
            h[19] = (cells[19][0], cells[19][1], 0.0)
            cur = 19
            j = 18
            while j >= 0 and (cur - 1) in cells:
                i = cur - 1
                v_i, a_i, s_i = cells[i]
                v_c, a_c, s_c = cells[cur]
                h[j] = (v_i, a_i, max(s_c - s_i, 0.0))
                cur, j = i, j - 1
            out[vid] = h
        return out


def cav_births(world, arm, t0, k_ticks):
    """{j: {vid: is_cav}} from insertions.jsonl (authoritative is_cav)."""
    out = {}
    path = (refload.CHUNK1_RUNS / f"{world}__{arm}" / "run"
            / "insertions.jsonl")
    for row in iter_jsonl(path):
        t = float(row["inserted_s"])
        j = int(round((t - t0) / DT))
        if 1 <= j <= k_ticks:
            out.setdefault(j, {})[row["vehicle_id"]] = bool(row["is_cav"])
    return out


# --------------------------------------------------- window metrics (A2)

def delay_step(prev_lane, prev_s, cur_lane, cur_s):
    """Contract physics-delay increment for one 0.5 s step (p3k
    step_delay: DT - ds/limit, ds = decrease of d)."""
    lim = E.lane_speed_limits(np.array([prev_lane]),
                              np.array([prev_s], dtype=np.float64))[0]
    ds = (P3.OFFSETS[prev_lane] - prev_s) - (P3.OFFSETS[cur_lane] - cur_s)
    return DT - ds / lim, ds


def phys_delay(snaps, births, truth, k_ticks):
    """Contract physics delay over the window (p3k phys_delay_sums
    convention): (delay_pred, delay_true, n_pred, n_true).

    snaps: {j: {vid: (lane, s, v)}} predicted (j=0..K);
    births: {j: {vid: (lane, s)}} predicted inserts at step start;
    truth: {j: {vid: (lane, s, v, a)}} ALL truth rows (j=0..K).
    """
    dp = dt_ = 0.0
    np_ = nt_ = 0
    state = {}
    for vid, st in snaps.get(0, {}).items():
        state[vid] = (st[0], st[1])
    for j in range(1, k_ticks + 1):
        for vid, bs in births.get(j, {}).items():
            state[vid] = bs
        for vid, st in snaps.get(j, {}).items():
            prev = state.get(vid, (st[0], st[1]))
            if prev[0] >= 0 and st[0] >= 0:
                dly, _ = delay_step(prev[0], prev[1], st[0], st[1])
                dp += dly
                np_ += 1
            state[vid] = (st[0], st[1])
    first = {}
    for j in range(0, k_ticks + 1):
        for vid in truth.get(j, {}):
            first.setdefault(vid, j)
    for j in range(1, k_ticks + 1):
        for vid, st in truth.get(j, {}).items():
            prevrow = truth.get(j - 1, {}).get(vid)
            if prevrow is not None:
                prev = (prevrow[0], prevrow[1])
            elif first.get(vid) == j:
                lim = E.lane_speed_limits(np.array([st[0]]),
                                          np.array([st[1]],
                                                   dtype=np.float64))[0]
                dt_ += DT - (st[2] * DT) / lim
                nt_ += 1
                continue
            else:
                prev = (st[0], st[1])
            if prev[0] < 0 or st[0] < 0:
                continue
            dly, _ = delay_step(prev[0], prev[1], st[0], st[1])
            dt_ += dly
            nt_ += 1
    return dp, dt_, np_, nt_


def window_metrics_act(snaps, truth, k_ticks, cav_names):
    """A2 window metrics for one rollout vs one branch truth (the frozen
    window_metrics conventions on chunk-1 data):
    - id-matched speed RMSE split CAV / human (matched by vehicle NAME);
    - TTC<3 exposure: pred = ALL predicted vehicles, true = MATCHED set
      (predicted names present in the truth tick), pairs per
      engine.ttc_stats, 0.5 s per pair-tick.

    snaps: {j: {name: (lane, s, v)}}; truth: {j: {name: (lane, s, v, a)}}.
    """
    v2c = v2h = nc_ = nh_ = 0
    t3p = t3t = 0
    minp, mint = [], []
    pred_steps = 0
    for j in range(1, k_ticks + 1):
        sn = snaps[j]
        tv = truth[j]
        pred_steps += len(sn)
        lane_p = np.array([st[0] for st in sn.values()], dtype=np.int64)
        s_p = np.array([st[1] for st in sn.values()], dtype=np.float64)
        v_p = np.array([st[2] for st in sn.values()], dtype=np.float64)
        names = list(sn)
        keep = np.array([n in tv and tv[n][0] >= 0 for n in names])
        for r, name in enumerate(names):
            if not keep[r]:
                continue
            dv = float(v_p[r]) - tv[name][2]
            if name in cav_names:
                v2c += dv * dv
                nc_ += 1
            else:
                v2h += dv * dv
                nh_ += 1
        l15p, l3p, mp = E.ttc_stats(lane_p, s_p, v_p)
        t3p += l3p
        minp.append(mp)
        if keep.any():
            tl = np.array([tv[n][0] for n, k in zip(names, keep)
                           if k], dtype=np.int64)
            ts = np.array([tv[n][1] for n, k in zip(names, keep) if k])
            tvv = np.array([tv[n][2] for n, k in zip(names, keep) if k])
            l15t, l3t, mt = E.ttc_stats(tl, ts, tvv)
            t3t += l3t
            mint.append(mt)
    return dict(
        pred_vehicle_steps=pred_steps,
        speed_rmse_cav=float(np.sqrt(v2c / max(nc_, 1))),
        speed_rmse_human=float(np.sqrt(v2h / max(nh_, 1))),
        n_matched_cav=nc_, n_matched_human=nh_,
        pred_ttc3_exposure=t3p * DT, true_ttc3_exposure=t3t * DT,
        pred_min_ttc=(None if not minp or not np.isfinite(minp).any()
                      else float(np.nanmin(minp))),
        true_min_ttc=(None if not mint or not np.isfinite(mint).any()
                      else float(np.nanmin(mint))),
    )


def corr(a, b, kind):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    if kind == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    ra_ = np.argsort(np.argsort(a)).astype(np.float64)
    rb_ = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra_, rb_)[0, 1])
