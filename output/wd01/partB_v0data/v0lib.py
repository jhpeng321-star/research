"""WD-01 Part B round 0: V0 (native-policy value) dataset library.

Builds, per chunk-1 campaign world and source run, the decision-time state
snapshots and the finite-horizon remaining-delay labels of
DESIGN_PARTB_V2.md §4 Step 0 from the 0.5 s physics logs.  ZERO SIMULATION:
run logs, the plan and the network are read as data/XML only.

Sources per world (DESIGN PartB v2 §4 Step 0: "chunk-1 native branches and
bases"):
  base   -- <world>__base, data [0, t0+600]; the campaign incumbent arm
            (fleet-uniform damping per the WM-44 data contract; NO
            controller internals logged).  Decision times [120, t0+600-30].
  native -- <world>__native, data (t0, t0+600]; CAVs on native IDM
            (speedMode 31, pre-fix).  Decision times (t0, t0+600-30].

Labels: contract §4.1 total delay (part4 delay-library conventions) over
[t, t+tau], tau in TAUS, for vehicles present at t or arriving in the
window (entry waiting from the scheduled departures, truncated).  The part4
conventions count, per vehicle, one interval per consecutive record pair
whose END record lies in (t, t+H] (interval-ending; the first record in the
window is an anchor/insertion point), which makes every window quantity an
exact per-tick prefix difference -- equivalence to
delay.core.vehicle_delay + waiting_component + window_totals is tested on
synthetic and real windows.  Samples whose horizon exceeds the end of the
logged data are dropped from the released tables; tau_available is recorded
per decision time.  Every (t, tau) row also carries TTC<3 exposure (veh.s),
hard-brake seconds and collision/teleport flags over the same window (for
the V_exp / V_brake / P_coll heads).

TD tuples: (s_t, D[t,t+30], s_{t+30}, tau) for COMMIT_S = 30 s and every
tau >= 30; within one source realization
label[t, tau] == D[t,t+30] + label[t+30, tau-30]
holds exactly (checked on real data; see td_report.py).

All chunk-1 data is PRE-FIX (cav_speed_mode = 31); cav_speed_mode is
recorded per row.  vType is supervision-only and never featurised.

Lane/s mapping is partA_data.pdl.map_step called exactly as the validated
partA build called it (physical lanes 0/1/2; junction rows lane=-1 with
the shared boundary s), and the geometry is part4 delay wm30_geometry --
the same convention pair the delay library was validated on (26 roots).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "partA_data", HERE.parent / "part4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pdl as PDL  # noqa: E402  (hybrid_state_v1 mapping, imported read-only)
from delay import core  # noqa: E402  (part4 delay library)
from delay.geometry import (ENTRANCE_MAINLINE, ENTRANCE_RAMP,  # noqa: E402
                            wm30_geometry)
from delay.truth import load_schedule  # noqa: E402

DT = 0.5
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
COMMIT_S = 30.0
DECIDE_EVERY_S = 10.0
BASE_DECISION_MIN_T = 120.0
HARD_BRAKE_ACCL = -3.0          # m/s^2 (WM-44 data-contract definition)
TTC_LT3_S = 3.0                 # s
CAV_SPEED_MODE_CHUNK1 = 31      # every chunk-1 world is pre-fix
# v0.3.2 fix #3: the campaign '__native'/'__pi' arms of chunks 1-3 and
# chunk1_fix27 are RANDOM PER-CAV SWITCHING SCHEDULES (per-CAV actions
# come from the per-arm switching schedule / suffix command logs), NOT a
# fleet-wide native-IDM or PI policy; per-source policy labels must not
# claim one.  (True fleet-native pairs exist only in chunk-4
# __native_base, out of scope for this dataset.)
SOURCE_POLICY = {
    "base": "damping fleet (campaign incumbent arm, fleet-uniform damping; "
            "WM-44 DATA_CONTRACT); NO controller internals logged",
    "native": "switching mix (random per-CAV switching schedules from "
              "switch_schedule.json / suffix_commands.jsonl; NOT a "
              "fleet-wide native/PI policy)",
    "pi": "switching mix (random per-CAV switching schedules from "
          "switch_schedule.json / suffix_commands.jsonl; NOT a "
          "fleet-wide PI policy)",
}


# --------------------------------------------------------------- plan ----

def load_plan(plan_path):
    p = Path(plan_path)
    plan = json.loads(p.read_text(encoding="utf-8"))
    worlds = sorted(plan["worlds"], key=lambda w: w["world"])
    return plan, worlds


def receipt_status(run_dir: Path):
    """(complete?, receipt dict|None, reason) for one run directory."""
    p = Path(run_dir) / "RUN_RECEIPT.json"
    if not p.exists():
        return False, None, "RUN_RECEIPT.json missing"
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                       # truncated/corrupt receipt
        return False, None, f"RUN_RECEIPT.json unreadable: {e}"
    if r.get("exit_status") != "complete":
        return False, r, f"exit_status={r.get('exit_status')!r}"
    return True, r, ""


def world_sources(w: dict, runs_root: Path, bases_root: Path | None = None):
    """(t0, {source: arm dir}) for one plan world (receipts not checked).
    bases_root: where the BASE arm dirs live when they are not under
    runs_root (chunk1_fix27: the verify27 mode-27 bases); None = runs_root.
    A plan without base_arm entries (the fix27 plan) uses <world>__base."""
    t0 = float(w["t0_s"])
    base_ref = (w.get("base_arm") or {}).get("run") \
        or f"{w['world']}__base"
    runs = {"base": (runs_root if bases_root is None else bases_root)
            / base_ref,
            "native": runs_root / f"{w['world']}__native"}
    return t0, runs


_GEOM = None


def cached_geometry():
    """wm30_geometry(), parsed once per process (the net XML is frozen)."""
    global _GEOM
    if _GEOM is None:
        _GEOM = wm30_geometry()
    return _GEOM


def decision_grid(t0: float, data_end: float, source: str):
    """Decision times on the physics grid, per the round-0 spec: base over
    [120, t0+570], native over (t0, t0+570]; every 10 s; capped at the end
    of the logged data."""
    lo = BASE_DECISION_MIN_T if source == "base" else t0 + DECIDE_EVERY_S
    hi = t0 + 600.0 - COMMIT_S
    ts = np.arange(lo, hi + 1e-9, DECIDE_EVERY_S)
    return ts[ts <= data_end + 1e-9]


# ------------------------------------------------------- trace building ----

@dataclass
class RunTrace:
    """Compact per-source arrays from one streaming pass over
    physics_steps.jsonl.  Records are per (vehicle, tick), sorted by
    (vid, tick); a PAIR is two consecutive records of one vehicle."""
    world: str
    source: str
    world_t0: float
    t_origin: float                # grid origin (0 for base, t0 for native)
    data_end: float                # last tick time present
    vid_table: np.ndarray          # (n_vehicles,) <U str
    start: np.ndarray              # (n_vehicles,) first record index
    end: np.ndarray                # (n_vehicles,) exclusive last index
    tick: np.ndarray               # (n_records,) int32 grid index
    lane: np.ndarray               # (n_records,) int8 (-1 junction)
    s: np.ndarray                  # (n_records,) float64 physical s
    v: np.ndarray                  # (n_records,) float32
    a: np.ndarray                  # (n_records,) float32
    inj: np.ndarray                # (n_records,) int8 in_junction
    smode: np.ndarray              # (n_records,) float32 (NaN unlogged)
    is_cav: np.ndarray             # (n_records,) int8
    pair_end_k: np.ndarray         # (n_pairs,) int32 end-record tick
    pair_ffr: np.ndarray           # (n_pairs,) float64 free-flow s
    pair_adv: np.ndarray           # (n_pairs,) float64 advance m
    c_ton: np.ndarray              # (n_ticks+1,) cum on-section s (prefix)
    c_ffr: np.ndarray              # (n_ticks+1,) cum free-flow s
    c_trav: np.ndarray             # (n_ticks+1,) cum advance m
    c_ttc: np.ndarray              # (n_ticks+1,) cum TTC<3 follower-steps
    c_hb: np.ndarray               # (n_ticks+1,) cum hard-brake veh-steps
    tick_first: np.ndarray         # (n_ticks+2,) first SNAP-record idx at tick
    tick_last: np.ndarray          # (n_ticks+2,) exclusive (tick-ordered copy)
    snap_vid_of: np.ndarray        # tick-ordered copies for snapshot_at
    snap_lane: np.ndarray
    snap_s: np.ndarray
    snap_v: np.ndarray
    snap_a: np.ndarray
    snap_inj: np.ndarray
    snap_smode: np.ndarray
    snap_is_cav: np.ndarray
    vid_of: np.ndarray             # (n_records,) vid index per record
    n_ticks: int
    t_ins: dict                    # vid -> insertion time (first tick, this run)
    t_ins_pre: dict                # vid -> insertion <= t_origin (shared past)
    sched_t: np.ndarray            # (n_sched,) scheduled depart times
    sched_ent: np.ndarray          # (n_sched,) 0 mainline / 1 ramp
    sched_vid: np.ndarray          # (n_sched,) <U str
    sched_is_cav: np.ndarray       # (n_sched,) int8 (known designations)
    coll_ticks: np.ndarray         # (n,) int32 sorted tick indices
    tele_ticks: np.ndarray         # (n,) int32 sorted tick indices
    coll_sets: list = field(default_factory=list)
    tele_sets: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def snapshot_at(self, t: float):
        """Record slices at decision time t from the tick-ordered copy
        (empty if the tick is absent or t is off-grid)."""
        k = int(round((t - self.t_origin) / DT))
        if not (0 <= k <= self.n_ticks):
            return None
        lo, hi = int(self.tick_first[k]), int(self.tick_last[k])
        if hi <= lo:
            return None
        return dict(k=k, vid=self.snap_vid_of[lo:hi],
                    lane=self.snap_lane[lo:hi], s=self.snap_s[lo:hi],
                    v=self.snap_v[lo:hi], a=self.snap_a[lo:hi],
                    inj=self.snap_inj[lo:hi], smode=self.snap_smode[lo:hi],
                    is_cav=self.snap_is_cav[lo:hi])


def _ttc_counts(tk, ln, ss, vv, n_ticks):
    """Follower-steps with closing TTC < TTC_LT3_S per tick, partA
    convention: same (tick, lane) group, rows ordered by descending s,
    adjacent pair (leader ahead), gap > 0, closing > 0; junction rows form
    their own group (identical to partA_data/build.py ttc_rate)."""
    out = np.zeros(n_ticks + 1, dtype=np.int32)
    order = np.lexsort((-ss, tk))
    tk, ln, ss, vv = tk[order], ln[order], ss[order], vv[order]
    same = (tk[1:] == tk[:-1]) & (ln[1:] == ln[:-1])
    b = np.flatnonzero(same)                     # follower row, leader b+1
    gap = ss[b] - ss[b + 1]
    closing = vv[b] - vv[b + 1]
    ok = (gap > 0) & (closing > 0)
    ttc = np.where(ok, gap / np.maximum(closing, 1e-12), np.inf)
    hit = b[ttc < TTC_LT3_S]
    if len(hit):
        np.add.at(out, tk[hit], 1)
    return out


def open_physics(path):
    """Read a physics log as a UTF-8 text stream (``.gz`` transparent).

    Additive WM-N1-DSPEED support: the direct-speed campaign gzips each
    ``physics_steps.jsonl`` in place (``physics_steps.jsonl.gz``) and drops
    the plain file.  Default uncompressed reads are byte-identical.
    """
    path = Path(path)
    if str(path).endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def build_run_trace(world: str, source: str, arm_dir: Path, t0: float,
                    t_ins_pre: dict | None = None,
                    sched_cache: dict | None = None,
                    bases_root: Path | None = None,
                    base_arm_dir: Path | None = None,
                    physics_path=None) -> RunTrace:
    """One streaming pass over <arm>/run/physics_steps.jsonl.

    arm_dir is the run directory as named in the campaign runs root
    (<world>__base / <world>__native); the physics file and the scenario
    live under arm_dir/run/.  t_ins_pre supplies shared pre-t_origin
    insertion times (the native branch's history before the restore).

    ``physics_path`` (additive WM-N1-DSPEED) overrides the physics log path
    (e.g. a top-level ``physics_steps.jsonl.gz``); ``None`` keeps the frozen
    ``<arm_dir>/run/physics_steps.jsonl`` behaviour and supports ``.gz``."""
    t_start = time.time()
    t_origin = 0.0 if source == "base" else float(t0)
    physics = Path(physics_path) if physics_path is not None \
        else arm_dir / "run" / "physics_steps.jsonl"

    vid_idx: dict = {}
    vids: list = []
    cav_ids: set = set()
    t_ins: dict = {}
    n_missing = 0
    n_pre_origin = 0
    n_ticks_seen = 0
    last_t = None
    coll_by_tick: dict = {}
    tele_by_tick: dict = {}
    tick_blocks: list = []

    with open_physics(physics) as fh:
        for line in fh:
            r = json.loads(line)
            t = float(r["time_s"])
            if source != "base" and t <= t_origin + 1e-9:
                n_pre_origin += 1        # branch grid starts strictly after t0
                continue
            last_t = t
            k = int(round((t - t_origin) / DT))
            if abs(t_origin + k * DT - t) > 1e-6:
                n_missing += 1                   # off-grid row; ignore
                continue
            n_ticks_seen += 1
            for vid_ in r["departed_ids"]:
                if vid_ not in t_ins:
                    t_ins[vid_] = t
            cav_ids.update(r["cav_ids"])
            if r["colliding_ids"]:
                coll_by_tick[k] = set(r["colliding_ids"])
            tp = r.get("starting_teleports") or []
            if tp:
                tele_by_tick[k] = set(tp)
            vehs = r["vehicles"]
            if not vehs:
                continue
            rows = []
            for vid_, vv in vehs.items():
                i = vid_idx.get(vid_)
                if i is None:
                    i = vid_idx[vid_] = len(vids)
                    vids.append(vid_)
                lane, s, inj = PDL.map_step(vv["lane_id"],
                                            vv["lane_position_m"])
                rows.append((i, k, lane, s, float(vv["speed_mps"]),
                             float(vv["acceleration_mps2"]), inj,
                             vv.get("speed_mode", np.nan)))
            rows.sort(key=lambda x: x[0])
            arr = np.asarray(rows, dtype=np.float64)
            tick_blocks.append(arr)

    if last_t is None:
        raise RuntimeError(f"{physics}: no rows")
    n_ticks = int(round((last_t - t_origin) / DT)) + 1

    big = np.concatenate(tick_blocks)
    vid = big[:, 0].astype(np.int32)
    tick = big[:, 1].astype(np.int32)
    lane = big[:, 2].astype(np.int8)
    s = big[:, 3].astype(np.float64)
    v = big[:, 4].astype(np.float32)
    a = big[:, 5].astype(np.float32)
    inj = big[:, 6].astype(np.int8)
    smode = np.where(np.isnan(big[:, 7]), np.nan,
                     big[:, 7]).astype(np.float32)
    del big, tick_blocks

    order = np.lexsort((tick, vid))
    vid, tick, lane, s, v, a, inj, smode = \
        vid[order], tick[order], lane[order], s[order], v[order], a[order], \
        inj[order], smode[order]
    n_vehicles = len(vids)
    starts = np.flatnonzero(np.r_[True, vid[1:] != vid[:-1]])
    ends = np.r_[starts[1:], len(vid)]
    cav_flag = np.isin(vid, [vid_idx[c] for c in cav_ids if c in vid_idx])
    cav = cav_flag.astype(np.int8)

    # pair increments: the library counts every consecutive record pair of a
    # vehicle as one counted interval with free-flow time adv/vlim
    geom = cached_geometry()
    n_pairs = len(vid) - n_vehicles
    pair_end_k = np.zeros(n_pairs, np.int32)
    pair_ffr = np.zeros(n_pairs, np.float64)
    pair_adv = np.zeros(n_pairs, np.float64)
    n_boundary_change = 0
    n_implausible = 0
    j = 0
    v_cap = 1.5 * DT * 45.0
    for vi in range(n_vehicles):
        lo, hi = int(starts[vi]), int(ends[vi])
        tk_v = tick[lo:hi]
        ln_v = lane[lo:hi]
        s_v = s[lo:hi]
        for p in range(hi - lo - 1):
            adv, attr_lane, attr_s, kind = core._pair_advance(
                geom, int(ln_v[p]), float(s_v[p]),
                int(ln_v[p + 1]), float(s_v[p + 1]))
            vlim = (geom.default_speed if attr_lane is None
                    else geom.speed_at(attr_lane, attr_s))
            pair_end_k[j] = tk_v[p + 1]
            pair_ffr[j] = adv / vlim
            pair_adv[j] = adv
            if kind == "boundary_change":
                n_boundary_change += 1
                if not (-core.EPS <= adv <= v_cap):
                    n_implausible += 1
            j += 1

    # per-tick prefix sums; cx[i] == sum(x[:i]) (EXCLUSIVE), so a window
    # [t, t+H] sums ticks in (k0, k1] as cx[k1+1] - cx[k0+1]
    pc = np.bincount(pair_end_k, minlength=n_ticks + 2) \
        if n_pairs else np.zeros(n_ticks + 2, np.int64)
    pf = np.bincount(pair_end_k, weights=pair_ffr, minlength=n_ticks + 2) \
        if n_pairs else np.zeros(n_ticks + 2, np.float64)
    pa = np.bincount(pair_end_k, weights=pair_adv, minlength=n_ticks + 2) \
        if n_pairs else np.zeros(n_ticks + 2, np.float64)
    c_ton = np.concatenate(([0.0], np.cumsum(pc) * DT))
    c_ffr = np.concatenate(([0.0], np.cumsum(pf)))
    c_trav = np.concatenate(([0.0], np.cumsum(pa)))
    ttc_t = _ttc_counts(tick.astype(np.int64), lane.astype(np.int64),
                        s, v.astype(np.float64), n_ticks)
    hb_t = np.bincount(tick[a < HARD_BRAKE_ACCL].astype(np.int64),
                       minlength=n_ticks + 1) \
        if (a < HARD_BRAKE_ACCL).any() else np.zeros(n_ticks + 1, np.int64)
    c_ttc = np.concatenate(([0], np.cumsum(ttc_t)))
    c_hb = np.concatenate(([0], np.cumsum(hb_t)))
    # tick-ordered copies for snapshots (records themselves stay sorted by
    # (vid, tick), which the pair builder relies on)
    idx2 = np.lexsort((vid, tick))
    snap_vid_of = vid[idx2]
    snap_lane = lane[idx2]
    snap_s = s[idx2]
    snap_v = v[idx2]
    snap_a = a[idx2]
    snap_inj = inj[idx2]
    snap_smode = smode[idx2]
    snap_is_cav = cav[idx2]
    tick_first = np.searchsorted(tick[idx2], np.arange(n_ticks + 2), "left")
    tick_last = np.searchsorted(tick[idx2], np.arange(n_ticks + 2), "right")

    # schedule (one shared scenario per world; carried by the base run,
    # which for chunk1_fix27 lives in the separate bases root)
    if source == "base":
        base_arm = arm_dir
    elif base_arm_dir is not None:       # v0.4a: explicit shared base arm
        base_arm = Path(base_arm_dir)     # (arm names are not all __native)
    elif bases_root is not None:
        base_arm = bases_root / arm_dir.name.replace("__native", "__base")
    else:
        base_arm = arm_dir.with_name(
            arm_dir.name.replace("__native", "__base"))
    ck = str(base_arm / "run")
    if sched_cache is not None and ck in sched_cache:
        sched = sched_cache[ck]
    else:
        sched = load_schedule(base_arm / "run")
        if sched_cache is not None:
            sched_cache[ck] = sched
    sched_vids = sorted(sched)
    sched_t = np.array([sched[x][0] for x in sched_vids], np.float64)
    sched_ent = np.array([sched[x][1] for x in sched_vids], np.int8)

    # designated-CAV flags for scheduled vehicles: the physics cav fleet
    # is the designation ground truth.  (v0.3.2 fix #2: the unreliable
    # realised-insertion LOG is never read here or in any label/feature
    # code -- its empty pre-t0 history was the root cause of the w8473
    # label inflation; the designation join is derived from
    # departed_ids/cav_ids alone.)
    sched_is_cav = np.array(
        [1 if x in cav_ids else 0 for x in sched_vids], np.int8)

    smode_seen = smode[~np.isnan(smode)]
    cav_smode_seen = smode[(~np.isnan(smode)) & (cav == 1)]
    n_nan_s = int((~np.isfinite(s)).sum())
    rt = RunTrace(
        world=world, source=source, world_t0=float(t0), t_origin=t_origin,
        data_end=float(last_t),
        vid_table=np.asarray(vids, dtype=np.str_),
        start=starts, end=ends, tick=tick, lane=lane, s=s, v=v, a=a,
        inj=inj, smode=smode, is_cav=cav,
        pair_end_k=pair_end_k, pair_ffr=pair_ffr, pair_adv=pair_adv,
        c_ton=c_ton, c_ffr=c_ffr, c_trav=c_trav, c_ttc=c_ttc, c_hb=c_hb,
        tick_first=tick_first, tick_last=tick_last,
        snap_vid_of=snap_vid_of, snap_lane=snap_lane, snap_s=snap_s,
        snap_v=snap_v, snap_a=snap_a, snap_inj=snap_inj,
        snap_smode=snap_smode, snap_is_cav=snap_is_cav, vid_of=vid,
        n_ticks=n_ticks,
        t_ins=t_ins, t_ins_pre=dict(t_ins_pre or {}),
        sched_t=sched_t, sched_ent=sched_ent,
        sched_vid=np.asarray(sched_vids, dtype=np.str_),
        sched_is_cav=sched_is_cav,
        coll_ticks=np.array(sorted(coll_by_tick), np.int32),
        coll_sets=[coll_by_tick[k] for k in sorted(coll_by_tick)],
        tele_ticks=np.array(sorted(tele_by_tick), np.int32),
        tele_sets=[tele_by_tick[k] for k in sorted(tele_by_tick)],
        stats=dict(
            n_ticks_present=n_ticks_seen, n_offgrid_rows=n_missing,
            n_pre_origin_rows=n_pre_origin,
            n_vehicles=n_vehicles, n_records=int(len(vid)),
            n_pairs=int(n_pairs), n_cav_ids=len(cav_ids),
            n_boundary_change_steps=n_boundary_change,
            n_implausible_boundary=n_implausible,
            n_unmapped_s=n_nan_s,
            smode_rows_logged=int(len(smode_seen)),
            smode_values=sorted(float(x) for x in np.unique(smode_seen)),
            cav_smode_values=sorted(float(x)
                                    for x in np.unique(cav_smode_seen)),
            wall_s=round(time.time() - t_start, 1)),
    )
    return rt


def window_label(rt: RunTrace, t: float, horizon: float,
                 skip_first_tick: bool = False) -> dict:
    """Contract §4.1 totals for [t, t+horizon] (equivalent to
    core.vehicle_delay + waiting_component + window_totals; tested).

    skip_first_tick (v0.4a): drop the interval that ENDS at the first tick
    after t from the on-section / free-flow / travelled sums.  A restored
    branch trace starts at t0 + 0.5 s (no t0 record), so it can never
    contain the pair (t0, t0+0.5]; applying this flag to a base-type trace
    at t = t0 puts the damping arm's t0 label on the SAME interval
    convention as the branch arms, so damping-vs-native contrasts at t0
    are not biased by one tick.  Default False = byte-identical v0.3.2
    labels."""
    t, H = float(t), float(horizon)
    t1 = t + H
    k0 = int(round((t - rt.t_origin) / DT))
    k1 = int(round((t1 - rt.t_origin) / DT))
    k0 = min(max(k0, 0), rt.n_ticks)
    k1 = min(max(k1, 0), rt.n_ticks)
    k0c = min(k0 + 1, rt.n_ticks) if skip_first_tick else k0
    ton = float(rt.c_ton[k1 + 1] - rt.c_ton[k0c + 1])
    ffr = float(rt.c_ffr[k1 + 1] - rt.c_ffr[k0c + 1])
    trav = float(rt.c_trav[k1 + 1] - rt.c_trav[k0c + 1])
    ttc3 = float((rt.c_ttc[k1 + 1] - rt.c_ttc[k0 + 1]) * DT)
    hb = float((rt.c_hb[k1 + 1] - rt.c_hb[k0 + 1]) * DT)

    sched_w = {vid: (td, ent) for vid, td, ent in
               zip(rt.sched_vid, rt.sched_t, rt.sched_ent)
               if td <= t1 + core.EPS}
    ins = dict(rt.t_ins_pre)
    for vid, ti in rt.t_ins.items():
        if ti <= t1 + core.EPS:
            ins[vid] = ti
    waits, wdiag = core.waiting_component(sched_w, ins, t, H)
    w_main = sum(w.wait_s for w in waits if w.entrance == ENTRANCE_MAINLINE)
    w_ramp = sum(w.wait_s for w in waits if w.entrance == ENTRANCE_RAMP)
    n_queued = sum(1 for w in waits if w.queued_at_horizon)

    lo = np.searchsorted(rt.coll_ticks, k0, "right")
    hi = np.searchsorted(rt.coll_ticks, k1, "right")
    coll: set = set()
    for i in range(lo, hi):
        coll |= rt.coll_sets[i]
    lot = np.searchsorted(rt.tele_ticks, k0, "right")
    hit = np.searchsorted(rt.tele_ticks, k1, "right")
    n_tele = sum(len(rt.tele_sets[i]) for i in range(lot, hit))

    kt = int(round((t - rt.t_origin) / DT))
    n_present = 0
    if 0 <= kt <= rt.n_ticks:
        n_present = int(rt.tick_last[kt] - rt.tick_first[kt])
    return dict(
        time_on_section_s=ton, free_flow_ref_s=ffr, traversed_m=trav,
        wait_mainline_s=float(w_main), wait_ramp_s=float(w_ramp),
        total_delay_ref_s=float(ton - ffr + w_main + w_ramp),
        ttc_lt3_veh_s=ttc3, hard_brake_s=hb,
        collision_any=int(bool(coll)), collision_n_veh=len(coll),
        teleports=int(n_tele), n_queued_at_horizon=int(n_queued),
        n_vehicles_present=n_present,
        n_unmatched_insertions=int(wdiag["n_unmatched_sched_id"]),
    )


def queue_lengths(rt: RunTrace, t: float):
    """(q_mainline, q_ramp): scheduled-departed but not inserted at t."""
    t = float(t)
    ins = dict(rt.t_ins_pre)
    for vid, ti in rt.t_ins.items():
        ins.setdefault(vid, ti)
    q = [0.0, 0.0]
    for td, ent, vid in zip(rt.sched_t, rt.sched_ent, rt.sched_vid):
        if td <= t and ins.get(vid, np.inf) > t:
            q[int(ent)] += 1.0
    return q


def build_t_ins_pre(base_rt: RunTrace, t0: float) -> dict:
    """Shared pre-t0 insertion times for the native branch: from the base
    run's realised insertions at or before t0 (identical history by the
    CRN restore; cross-checked in tests)."""
    return {vid: ti for vid, ti in base_rt.t_ins.items() if ti <= t0 + 1e-9}


def sha256_file(path, block=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()
