"""WD-01 part 3j: NUMERIC decomposition of the structured native Part A
emulator's delay under-prediction.  ZERO SIMULATION: no SUMO/TraCI import;
the only dynamics is the part-3 rollout engine (pure torch, read-only
import from the populated RD_WT_ROOT/wd01 copy) that parts 3c-3h already
used.  TRAIN+VALIDATION only: holdout roots stay blocked inside
root_arrays -> holdout_block (this script never passes a holdout root and
never touches the registration gate).

Delay model (per vehicle-step, dt = 0.5 s, pred and truth alike):

    delay_step = dt - ds / limit(lane, position)
    ds         = forward advance over the step
               = DECREASE of d,  d := OFFSETS[lane] - s
  (d decreases downstream; d < 0 is past the merge end; d is continuous
  across lane changes because the engine keeps d constant there and the
  measured SUMO integrator is s' = s + v'*dt.)  limit(lane, s) is the
  per-segment SUMO lane limit from part4/delay/geometry via
  engine.lane_speed_limits (main 31.29 m/s; accel 22.22, merge 31.29),
  evaluated at the START of the step (the state the engine itself
  refreshes limits on).  Birth steps use the logged insertion (lane, s)
  as the step's start state on both sides.

Attribution cells: (lane 0/1/2) x (segment by start-of-step d: d<0
downstream, 0<=d<75 merge_edge, 75<=d<800 rsu_window, d>=800 upstream)
x (t0-present vs born) x (lc_affected = the vehicle changed lane within
the previous 5 s = last 10 steps of the same trajectory, the change step
itself included).  Exit mismatches (one side left the section while the
other is still on it -> count the other side's remaining delay) and
entry waiting (births are inserted at logged times, expected 0) are
accounted separately.

Reconciliation: the p3h "delay" metric is pure exposure
(0.5 s x vehicle count per tick), so this script's
0.5*(n_steps_pred - n_steps_true) per window must reproduce the p3h
per-window delay_pred - delay_true exactly for L0 and for L3 member 0
(same seeds), and the window mean must match the published gate_v2 means
(L0 -461.11, L3 -328.05 veh s over t0 every 30 s; member 0 vs the 5-member
mean may differ by the small member spread).

Outputs (written HERE only): results/decomposition.csv,
results/speed_profile.csv, results/stdout_last.txt, REPORT_P3J.md.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
WDO = HERE.parent                           # output/wd01, THIS tree
sys.path.insert(0, str(WDO))                # wdpaths
import wdpaths  # noqa: E402

# Untracked data/results (partA_data npz, part3h receipts) live in the
# populated RD_WT_ROOT/wd01 tree, READ-ONLY; engine/p3lib/pdl code comes
# from THIS tree (like part3k), pointed at that data via WD01_DATA_ROOT.
POP = Path(wdpaths.RD_WT_ROOT) / "wd01" / "output" / "wd01"
os.environ.setdefault("WD01_DATA_ROOT", str(POP))
for _p in (str(WDO / "partA_data"), str(WDO / "part3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine as E  # noqa: E402
import p3lib as P3  # noqa: E402

DT = 0.5
K_STEPS = 120                                # H = 60 s
LC_WINDOW_STEPS = 10                         # 5 s at 0.5 s
MAIN_T0S = tuple(range(120, 481, 60))        # brief's main grid
REC_T0S = tuple(range(120, 481, 30))         # p3h reconciliation grid
LC_SCALE = 0.003
L3_MEMBER_SEED = 1                           # member 0 = seed m*17+1, m=0
OFFS = tuple(P3.OFFSETS)
P3H_GATE = str(POP / "part3h" / "results" / "gate_v2.json")
P3H_RAW = str(POP / "part3h" / "results" / "raw" / "raw_metrics.jsonl")

SEGS = (("downstream", float("-inf"), 0.0),
        ("merge_edge", 0.0, 75.0),
        ("rsu_window", 75.0, 800.0),
        ("upstream", 800.0, float("inf")))


def seg_bin(d: float) -> str:
    for name, lo, hi in SEGS:
        if lo <= d < hi:
            return name
    return SEGS[-1][0]                       # d == +inf guard; unreachable


def d_of(lane: int, s: float) -> float:
    return OFFS[lane] - s


_LIM_TBL = None


def limit_of(lane: int, s: float) -> float:
    """Per-segment SUMO speed limit (part4 geometry via engine)."""
    global _LIM_TBL
    if _LIM_TBL is None:
        _LIM_TBL = E._seg_table()
    starts, lims = _LIM_TBL[lane]
    i = int(np.searchsorted(starts, s, side="right")) - 1
    return float(lims[max(0, min(i, len(lims) - 1))])


def step_delay(prev_lane: int, prev_s: float, cur_lane: int, cur_s: float):
    """(delay, ds) for one 0.5 s step; ds = decrease of d (may be slightly
    negative on measurement jitter; recorded, never clamped)."""
    ds = d_of(prev_lane, prev_s) - d_of(cur_lane, cur_s)
    return DT - ds / limit_of(prev_lane, prev_s), ds


def lc_hit(change_ticks, abs_tick) -> bool:
    """True if some lane change of this trajectory happened within the
    previous 5 s (the change tick itself included)."""
    for t in change_ticks:
        if 0 <= abs_tick - t <= LC_WINDOW_STEPS:
            return True
    return False


# ----------------------------------------------------------------- core --
# The two recorders are pure: they consume plain per-tick state dicts so
# tests can synthesize trajectories without the engine.

def pred_records(snaps, births, K=K_STEPS):
    """snaps: {j: {vid: (lane, s, v)}} for j = 0..K (snapshot j is AFTER
    step j).  births: {j: {vid: (lane, s)}} inserted at the START of step
    j (the engine keeps d constant through any lane change, so (lane, s)
    pins d).  Returns (records, presence, first_tick) where records are
    per (vehicle, step j in 1..K): {vid, j, delay, ds, lane, seg, cls,
    lc, side}; presence = {vid: sorted steps}; first_tick = {vid: first
    j in 0..K present}."""
    recs, pres = [], defaultdict(list)
    first, cls, changes, state = {}, {}, {}, {}
    for vid, st in snaps.get(0, {}).items():
        state[vid] = (st[0], st[1])
        cls[vid] = "t0"
        changes[vid] = []
        first[vid] = 0
    for j in range(1, K + 1):
        for vid, bs in births.get(j, {}).items():
            state[vid] = bs
            cls[vid] = "born"
            changes.setdefault(vid, [])
            first.setdefault(vid, j)
        for vid, (lane, s, _v) in snaps.get(j, {}).items():
            prev = state.get(vid)
            if prev is None:                 # present without any known
                prev = (lane, s)             # start: degenerate; 0 advance
                first.setdefault(vid, j)
                cls.setdefault(vid, "born")
            if lane != prev[0]:
                changes.setdefault(vid, []).append(j)
            delay, ds = step_delay(prev[0], prev[1], lane, s)
            recs.append(dict(vid=vid, j=j, delay=delay, ds=ds, lane=lane,
                             seg=seg_bin(d_of(prev[0], prev[1])),
                             cls=cls.get(vid, "t0"),
                             lc=lc_hit(changes.get(vid, ()), j),
                             side="pred"))
            pres[vid].append(j)
            state[vid] = (lane, s)
    return recs, dict(pres), first


def truth_records(rows, lc_ticks, birth_states, k0=0, K=K_STEPS):
    """rows: {j: {vid: (lane, s, v)}} for j = 0..K (tick k0+j).
    lc_ticks: {vid: sorted ABSOLUTE lane-change ticks (the first tick on
    the new lane)}.  birth_states: {vid: (lane, s)} logged insertion
    state, or missing to fall back to the integrator identity ds = v*dt
    at the first row (no logged insertion available)."""
    recs, pres, first = [], defaultdict(list), {}
    for j in range(0, K + 1):
        for vid in rows.get(j, {}):
            if vid not in first:
                first[vid] = j
    for j in range(1, K + 1):
        for vid, (lane, s, v) in rows.get(j, {}).items():
            prevrow = rows.get(j - 1, {}).get(vid)
            if prevrow is not None:
                prev = (prevrow[0], prevrow[1])
                born = first[vid] > 0
            elif first[vid] == j:
                bs = birth_states.get(vid)   # birth step
                born = True
                if bs is None:               # no logged insertion: use the
                    delay = DT - (v * DT) / limit_of(lane, s)   # integrator
                    recs.append(dict(vid=vid, j=j, delay=delay, ds=v * DT,
                                     lane=lane, seg=seg_bin(d_of(lane, s)),
                                     cls="born",
                                     lc=lc_hit(lc_ticks.get(vid, ()),
                                               k0 + j), side="true"))
                    pres[vid].append(j)
                    continue
                prev = bs
            else:
                continue                     # gap before the first row
            delay, ds = step_delay(prev[0], prev[1], lane, s)
            recs.append(dict(
                vid=vid, j=j, delay=delay, ds=ds, lane=lane,
                seg=seg_bin(d_of(prev[0], prev[1])),
                cls="born" if born else "t0",
                lc=lc_hit(lc_ticks.get(vid, ()), k0 + j), side="true"))
            pres[vid].append(j)
    return recs, dict(pres), first


def cells_from(recs_p, recs_t):
    """key (lane, seg, cls, lc) -> [delay_pred, delay_true, n_pred, n_true]."""
    out = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for r in recs_p:
        c = out[(r["lane"], r["seg"], r["cls"], r["lc"])]
        c[0] += r["delay"]
        c[2] += 1
    for r in recs_t:
        c = out[(r["lane"], r["seg"], r["cls"], r["lc"])]
        c[1] += r["delay"]
        c[3] += 1
    return dict(out)


def speed_profile_from(seq_p, seq_t):
    """Matched (vid present on both sides at tick j in 1..K) pairs, binned
    by the TRUTH (lane, 50 m d-bin) at tick j.  Returns {(lane, dbin):
    [sum_v_pred, n, sum_v_true, n]} (the two counts are equal)."""
    out = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for vid, js in seq_p.items():
        tt = seq_t.get(vid)
        if not tt:
            continue
        for j, (lane_p, _s_p, v_p) in js.items():
            row = tt.get(j)
            if row is None:
                continue
            lane_t, s_t, v_t = row
            db = int(np.floor(d_of(lane_t, s_t) / 50.0)) * 50
            c = out[(lane_t, db)]
            c[0] += v_p
            c[1] += 1
            c[2] += v_t
            c[3] += 1
    return dict(out)


def aggregate_profiles(windows):
    """Pool per-window speed profiles by variant.  Profile keys are
    (lane, d_bin); the variant lives on the window, NOT in the key."""
    prof_by = {v: defaultdict(lambda: [0.0, 0, 0.0, 0])
               for v in ("L0", "L3")}
    for w in windows:
        tgt = prof_by.get(w["variant"])
        if tgt is None:
            continue
        for key, c in w["profile"].items():
            p = tgt[key]
            for i in range(4):
                p[i] += c[i]
    return prof_by


def exit_mismatch(first_p, pres_p, first_t, pres_t, recs_p, recs_t, K):
    """One side gone from the section while the other remains: count the
    remaining side's delay steps.  Buckets keyed 'pred_early' (a pred
    vehicle vanished while truth was still on section -> truth remaining
    delay) and 'true_early' (mirror).  Each: [n_veh, n_steps, delay]."""
    def exit_j(first, pres):
        """First step j in 1..K where the vehicle was present at j-1 and
        absent at j (presence at 0 == (first == 0)); None if never."""
        if first is None:
            return None
        pset = set(pres)
        p0 = (first == 0)
        for j in range(1, K + 1):
            prev = p0 if j == 1 else (j - 1) in pset
            if prev and j not in pset:
                return j
        return None

    by_p, by_t = defaultdict(list), defaultdict(list)
    for r in recs_p:
        by_p[r["vid"]].append((r["j"], r["delay"]))
    for r in recs_t:
        by_t[r["vid"]].append((r["j"], r["delay"]))

    def remaining(by, vid, from_j):
        # the other side is absent FROM from_j onward (its exit tick), so
        # every step of this side at j >= from_j is unpaired
        steps = [(j, d) for j, d in by.get(vid, ()) if j >= from_j]
        return len(steps), sum(d for _j, d in steps)

    b = {"pred_early": [0, 0, 0.0], "true_early": [0, 0, 0.0]}
    for vid in set(first_p) | set(first_t):
        pe = exit_j(first_p.get(vid), pres_p.get(vid, ()))
        te = exit_j(first_t.get(vid), pres_t.get(vid, ()))
        if pe is None and te is None:
            continue
        if pe is not None and (te is None or pe < te):
            n, dl = remaining(by_t, vid, pe)
            b["pred_early"][0] += 1
            b["pred_early"][1] += n
            b["pred_early"][2] += dl
        elif te is not None:
            n, dl = remaining(by_p, vid, te)
            b["true_early"][0] += 1
            b["true_early"][1] += n
            b["true_early"][2] += dl
    return b


def sum_exit_buckets(windows, variant):
    """Elementwise [n_veh, n_steps, delay] sums of the exit buckets for
    one variant (never use sum() with a list start: that concatenates)."""
    out = {k: [0, 0, 0.0] for k in ("pred_early", "true_early")}
    for w in windows:
        if w["variant"] != variant:
            continue
        for k in out:
            for i in range(3):
                out[k][i] += w["exit"][k][i]
    return out


def entry_check(births_p, first_p, first_t):
    """(total_waiting_steps, n_matched_births, n_pred_only_births,
    n_truth_only_births, n_t0_missing_in_pred).  Waiting expected 0
    because births are inserted at the logged tick on both sides."""
    wait = n = 0
    for vid, j_ins in births_p.items():
        if vid in first_t:
            n += 1
            wait += max(0, j_ins - first_t[vid])   # truth exists, pred not yet
    only_p = sum(1 for vid in births_p if vid not in first_t)
    only_t = sum(1 for vid, f in first_t.items()
                 if f > 0 and vid not in births_p)
    t0_missing = sum(1 for vid, f in first_t.items()
                     if f == 0 and vid not in first_p)
    return wait, n, only_p, only_t, t0_missing


# -------------------------------------------------------------- glue -----

T_START = time.time()


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def rollout_for(variant, root, t0, dev):
    if variant == "L0":
        return E.Rollout(root, float(t0), kind="idm_normal", dev=dev).run()
    return E.Rollout(root, float(t0), kind="idm_sampled",
                     member_seed=L3_MEMBER_SEED, lc_scale=LC_SCALE,
                     dev=dev).run()


def snaps_to_dicts(r, K):
    return {j: {int(v): (int(l), float(ss), float(vv))
                for v, l, ss, vv in zip(r.snapshots[j]["vid"],
                                        r.snapshots[j]["lane"],
                                        r.snapshots[j]["s"],
                                        r.snapshots[j]["v"])}
            for j in range(K + 1)}


def truth_rows(ra, k0, K):
    out = {}
    for j in range(K + 1):
        ids = ra.tick_rows(k0 + j)
        out[j] = {int(v): (int(l), float(ss), float(vv))
                  for v, l, ss, vv in zip(ra.vid[ids], ra.lane_m[ids],
                                          ra.s[ids], ra.v[ids])}
    return out


def births_from_engine(r, K):
    return {j: {int(tvid): (int(l), float(ss))
                for (tvid, l, ss, _vv, _cc) in lst}
            for j, lst in r.births.items() if j <= K and lst}


def birth_states_from_ra(ra, k0, K):
    """{vid: (lane, s)} logged insertion states for births in (k0, k0+K]."""
    out = {}
    m = ra.b_dep & (ra.b_tick > k0) & (ra.b_tick <= k0 + K) & (ra.b_lane >= 0)
    for i in np.where(m)[0]:
        out[int(ra.b_vid[i])] = (int(ra.b_lane[i]), float(ra.b_s[i]))
    return out


def lc_ticks_from_ra(ra):
    out = defaultdict(list)
    for v, t in zip(ra.lc_vid, ra.lc_tick):
        out[int(v)].append(int(t))
    return {v: sorted(ts) for v, ts in out.items()}


def process_window(root, t0, variant, dev, K=K_STEPS):
    """Full per-window result.  Roots come from roots_by_split() only
    (train/val); holdout never passed, holdout_block stays authoritative."""
    r = rollout_for(variant, root, t0, dev)
    ra, k0 = r.ra, r.k0
    snaps = snaps_to_dicts(r, K)
    rows = truth_rows(ra, k0, K)
    n_p = sum(len(snaps[j]) for j in range(1, K + 1))
    n_t = sum(len(rows[j]) for j in range(1, K + 1))
    seq_p, seq_t = defaultdict(dict), defaultdict(dict)
    for j in range(1, K + 1):
        for vid, st in snaps[j].items():
            seq_p[vid][j] = st
        for vid, st in rows[j].items():
            seq_t[vid][j] = st
    births = births_from_engine(r, K)
    recs_p, pres_p, first_p = pred_records(snaps, births, K)
    recs_t, pres_t, first_t = truth_records(
        rows, lc_ticks_from_ra(ra), birth_states_from_ra(ra, k0, K), k0, K)
    cells = cells_from(recs_p, recs_t)
    bmap = {vid: j for j, dd in births.items() for vid in dd}
    wait, n_m, only_p, only_t, t0miss = entry_check(bmap, first_p, first_t)
    return dict(
        root=root, t0=t0, variant=variant, n_p=n_p, n_t=n_t,
        exposure_p=n_p * DT, exposure_t=n_t * DT,
        delay_p=sum(c[0] for c in cells.values()),
        delay_t=sum(c[1] for c in cells.values()),
        cells=cells, profile=speed_profile_from(seq_p, seq_t),
        exit=exit_mismatch(first_p, pres_p, first_t, pres_t,
                           recs_p, recs_t, K),
        entry=(wait, n_m, only_p, only_t, t0miss),
        n_gap_truth=sum(1 for v, js in pres_t.items()
                        if js and (js[-1] - js[0] + 1) != len(js)))


# ---------------------------------------------------------- reporting ----

def agg_cells(windows, variant):
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for w in windows:
        if w["variant"] != variant:
            continue
        for key, c in w["cells"].items():
            a = agg[key]
            for i in range(4):
                a[i] += c[i]
    return agg


ATTR_NAMES = {0: "lane", 1: "segment", 2: "class", 3: "lc_affected"}


def print_attr_breakdown(windows, variant, attr_idx):
    part = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for key, c in agg_cells(windows, variant).items():
        p = part[key[attr_idx]]
        for i in range(4):
            p[i] += c[i]
    nm = ATTR_NAMES[attr_idx]
    print(f"  -- by {nm} ({variant}) --")
    print("  value             n_steps_pred n_steps_true expo_deficit "
          "ff_component delay_deficit")
    for val in sorted(part, key=lambda v: (str(type(v)), v)):
        dp, dt_, npp, ntt = part[val]
        expo = DT * (npp - ntt)
        ff_p, ff_t = DT * npp - dp, DT * ntt - dt_
        label = ("yes" if val else "no") if attr_idx == 3 else str(val)
        print(f"  {nm}={label:<10} {npp:12d} {ntt:12d} {expo:12.2f} "
              f"{ff_t - ff_p:12.2f} {dp - dt_:13.2f}")


def fmt_cell(key):
    return (f"lane={key[0]} seg={key[1]:<10} cls={key[2]:<4} "
            f"lc={'yes' if key[3] else 'no':<3}")


def below_normal_priority():
    if sys.platform == "win32":
        try:
            h = ctypes.windll.kernel32.GetCurrentProcess()
            return bool(ctypes.windll.kernel32.SetPriorityClass(h, 0x4000))
        except Exception:
            return False
    return False


def load_p3h():
    """({(variant, member, root, t0): (delay_pred, delay_true)} at H=60,
    last occurrence kept per key) and the gate_v2 per-(variant,H) table."""
    per = {}
    for line in Path(P3H_RAW).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["H"]) != 60:
            continue
        per[(r["variant"], int(r["member"]), r["root"], int(r["t0"]))] = \
            (float(r["delay_pred"]), float(r["delay_true"]))
    gate = json.loads(Path(P3H_GATE).read_text(encoding="utf-8"))
    return per, gate["per_vh"]


class Tee:
    """stdout -> console + file (the report pastes this file verbatim)."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, m):
        self.file.write(m)
        self.stdout.write(m)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


REPORT_PROSE = """# WD-01 part 3j — numeric decomposition of the structured \
emulator's delay under-prediction

Deliverable: a NUMERIC decomposition of the delay under-prediction of the
structured native Part A emulator (L0 = plain IDM `idm_normal`,
L3 = structured sampled IDM `idm_sampled`, member 0), as measured by the
part-3h gate. ZERO simulation (no SUMO/TraCI anywhere); train+validation
roots only — the holdout block inside `root_arrays` stays authoritative
and was never bypassed.

## Method (every number below is pasted from `decompose.py` stdout)

- Rollouts: `part3/engine.py Rollout` imported READ-ONLY from the
  populated copy `RD_WT_ROOT/wd01/output/wd01/part3`; truth is the p2C
  model view (`root_arrays`); 0.5 s snapshots, H = 60 s (K = 120 steps).
- Delay per vehicle-step: `dt - ds/limit(lane, position)` with
  `ds` = forward advance = DECREASE of `d = OFFSETS[lane] - s` (d
  decreases downstream and is continuous across lane changes), `limit`
  = per-segment SUMO limit from `part4/delay/geometry` via
  `engine.lane_speed_limits` (main 31.29; accel 22.22, merge 31.29),
  evaluated at the START of the step. Birth steps use the logged
  insertion `(lane, s)` as start state on both sides.
- Attribution cells: lane (0/1/2) x segment-by-start-of-step d
  (d<0 downstream; 0<=d<75 merge_edge; 75<=d<800 rsu_window; d>=800
  upstream) x class (t0-present vs born) x lc_affected (lane change
  within the previous 5 s of the same trajectory, change step included).
- Exit mismatch (one side left the section while the other remains:
  the remaining side's delay is counted) and entry waiting (expected 0;
  births are inserted at logged times) are accounted separately.
- Reconciliation: the p3h "delay" metric is pure exposure
  (0.5 s x vehicle count per tick), so per-window
  `0.5*(n_steps_pred - n_steps_true)` must equal the p3h per-window
  `delay_pred - delay_true` exactly (identical rollouts, L0 and L3
  member 0; p3h's published L3 mean is over 5 members), and the window
  mean must match the published gate_v2 values within member spread.
- Identity used throughout: per-vehicle-step physics delay =
  exposure (0.5 s) - free-flow usage (ds/limit), so
  `delay_deficit = expo_deficit + ff_component` where
  `ff_component = ff_true - ff_pred` and `ff = 0.5*n_steps - delay`.
- Files: `results/decomposition.csv` (per-window cells),
  `results/speed_profile.csv` (matched-speed profile),
  `results/stdout_last.txt` (verbatim stdout).

## Full stdout of `python decompose.py` (this run)

```text
"""


def write_report(stdout_path):
    txt = Path(stdout_path).read_text(encoding="utf-8")
    body = REPORT_PROSE + txt + "```\n"
    (HERE / "REPORT_P3J.md").write_text(body, encoding="utf-8")


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    outdir = HERE / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(outdir / "stdout_last.txt")

    print("priority below-normal:", below_normal_priority())
    dev = P3.device()
    splits = P3.roots_by_split()
    val_roots = sorted(splits["val"])
    train4 = sorted(splits["train"])[:4]
    print("=== WD-01 part 3j: delay under-prediction decomposition ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"variants: L0 = kind 'idm_normal'; L3 = kind 'idm_sampled' "
          f"member_seed={L3_MEMBER_SEED} lc_scale={LC_SCALE}")
    print(f"H=60 s (K={K_STEPS} steps of {DT} s); validation roots "
          f"{val_roots}; train check roots {train4}")
    print(f"grids: main t0 every 60 s {MAIN_T0S}; reconciliation t0 every "
          f"30 s {REC_T0S}")
    print(f"d = OFFSETS[lane] - s, OFFSETS={OFFS}; segments "
          f"{[(nm, lo, hi) for nm, lo, hi in SEGS]}")
    print(f"limits via engine.lane_speed_limits: e.g. limit(0,100)="
          f"{limit_of(0, 100.0):.2f} limit(2,0)={limit_of(2, 0.0):.2f} "
          f"limit(2,440)={limit_of(2, 440.0):.2f}")

    # ------- validation windows on the 30 s grid (superset of the 60 s
    # grid): reconciliation on all, decomposition tables on the 60 s subset
    p3h_raw, p3h_gate = load_p3h()
    print("\n== A. reconciliation with p3h (H=60, validation, t0 every "
          "30 s: 4 roots x 13 t0 = 52 windows/variant) ==")
    rec_windows = []
    for root in val_roots:
        for t0 in REC_T0S:
            for variant in ("L0", "L3"):
                w = process_window(root, t0, variant, dev)
                rec_windows.append(w)
                log(f"{variant} {root} t0={t0}: n_p={w['n_p']} "
                    f"n_t={w['n_t']} expo_def="
                    f"{w['exposure_p'] - w['exposure_t']:.1f} "
                    f"delay_def={w['delay_p'] - w['delay_t']:.1f}")
    for variant in ("L0", "L3"):
        ws = [w for w in rec_windows if w["variant"] == variant]
        mp = float(np.mean([w["exposure_p"] for w in ws]))
        mt = float(np.mean([w["exposure_t"] for w in ws]))
        md = mp - mt
        tgt = p3h_gate[f"{variant}@H60"]
        dev_pct = 100.0 * (md - tgt["delay_bias"]) / abs(tgt["delay_bias"])
        print(f"  {variant}: mean exposure_pred={mp:.2f}  "
              f"mean exposure_true={mt:.2f}  mean deficit={md:.2f} veh s  "
              f"| p3h gate_v2 delay_bias={tgt['delay_bias']:.2f} "
              f"(delay_pred={tgt['delay_pred']:.2f} "
              f"delay_true={tgt['delay_true']:.2f})  "
              f"deviation={dev_pct:+.2f}%")
    for variant in ("L0", "L3"):
        diffs, worst = [], None
        for w in rec_windows:
            if w["variant"] != variant:
                continue
            key = (variant, 0, w["root"], w["t0"])
            if key in p3h_raw:
                d = (w["exposure_p"] - w["exposure_t"]) - \
                    (p3h_raw[key][0] - p3h_raw[key][1])
                diffs.append(abs(d))
                if worst is None or abs(d) > abs(worst[0]):
                    worst = (d, key)
        print(f"  {variant}: per-window |diff| vs p3h raw member-0 rows: "
              f"max={max(diffs):.6f} veh s over {len(diffs)} windows "
              f"(worst {worst[1]} diff={worst[0]:+.6f})")

    val_windows = [w for w in rec_windows if w["t0"] in MAIN_T0S]

    # ------- per-window totals on the main grid
    print("\n== B. per-window totals (validation, t0 every 60 s) ==")
    print("  root             t0  variant  n_pred n_true expo_deficit "
          "delay_pred delay_true delay_deficit")
    for w in sorted(val_windows, key=lambda w: (w["root"], w["t0"],
                                                w["variant"])):
        print(f"  {w['root']:<16} {w['t0']:3d}  {w['variant']:>7} "
              f"{w['n_p']:6d} {w['n_t']:6d} "
              f"{w['exposure_p'] - w['exposure_t']:12.2f} "
              f"{w['delay_p']:10.2f} {w['delay_t']:10.2f} "
              f"{w['delay_p'] - w['delay_t']:13.2f}")
    for variant in ("L0", "L3"):
        ws = [w for w in val_windows if w["variant"] == variant]
        mp = float(np.mean([w["exposure_p"] for w in ws]))
        mt = float(np.mean([w["exposure_t"] for w in ws]))
        dp = float(np.mean([w["delay_p"] for w in ws]))
        dt_ = float(np.mean([w["delay_t"] for w in ws]))
        print(f"  {variant} 60s-grid means over {len(ws)} windows: "
              f"expo_def={mp - mt:.2f}  delay_pred={dp:.2f}  "
              f"delay_true={dt_:.2f}  delay_def={dp - dt_:.2f} veh s")

    # ------- attribute breakdowns + top cells
    print("\n== C. deficit by attribute (validation, 60 s grid; units veh s "
          "pooled over 28 windows/variant; expo_deficit = 0.5*(n_pred - "
          "n_true); delay_deficit = delay_pred - delay_true (physics "
          "delay); ff_component = ff_true - ff_pred; identity: "
          "expo + ff = delay deficit) ==")
    for variant in ("L0", "L3"):
        print_attr_breakdown(val_windows, variant, 0)
        print_attr_breakdown(val_windows, variant, 1)
        print_attr_breakdown(val_windows, variant, 2)
        print_attr_breakdown(val_windows, variant, 3)

    print("\n== D. top-10 cells by |delay_deficit| (validation, 60 s "
          "grid, per variant) ==")
    for variant in ("L0", "L3"):
        agg = agg_cells(val_windows, variant)
        top = sorted(agg.items(), key=lambda kv: abs(kv[1][0] - kv[1][1]),
                     reverse=True)[:10]
        print(f"  -- {variant} --")
        print("  cell                                                          "
              " delay_deficit n_pred n_true")
        for key, c in top:
            print(f"  {fmt_cell(key):<60} {c[0] - c[1]:13.2f} "
                  f"{c[2]:6d} {c[3]:6d}")

    # ------- speed profile
    print("\n== E. speed profile, bins with |dv| > 1 m/s (matched "
          "vehicle-steps pooled over validation 60 s windows, binned by "
          "TRUTH (lane, 50 m d-bin) at tick j; dv = v_pred_mean - "
          "v_true_mean m/s) ==")
    prof_by = aggregate_profiles(val_windows)
    print("  variant lane  d_bin  v_pred_mean v_true_mean      n     dv")
    n_sig = 0
    prof_all = {}
    for variant in ("L0", "L3"):
        rows_ = []
        for (lane, db), c in prof_by[variant].items():
            if c[1] == 0:
                continue
            prof_all[(variant, lane, db)] = c
            rows_.append((lane, db, c[0] / c[1], c[2] / c[3],
                          c[1], c[0] / c[1] - c[2] / c[3]))
        for lane, db, vp, vt, n_, dv in sorted(rows_):
            if abs(dv) > 1.0:
                n_sig += 1
                print(f"  {variant}    {lane}  {db:5d}  {vp:11.3f}  "
                      f"{vt:11.3f}  {n_:5d}  {dv:+6.3f}")
    print(f"  ({n_sig} bins with |dv| > 1 m/s)")

    # ------- exit mismatch + entry waiting
    print("\n== F. exit mismatch & entry waiting (validation, 60 s grid) ==")
    for variant in ("L0", "L3"):
        ws = [w for w in val_windows if w["variant"] == variant]
        pe = sum_exit_buckets(val_windows, variant)["pred_early"]
        te = sum_exit_buckets(val_windows, variant)["true_early"]
        wait = sum(w["entry"][0] for w in ws)
        nmb = sum(w["entry"][1] for w in ws)
        onlyp = sum(w["entry"][2] for w in ws)
        onlyt = sum(w["entry"][3] for w in ws)
        t0m = sum(w["entry"][4] for w in ws)
        gaps = sum(w["n_gap_truth"] for w in ws)
        print(f"  {variant}: pred-exited-early: {pe[0]} vehicles, truth "
              f"remained {pe[1]} steps, remaining truth delay={pe[2]:.2f} s "
              f"| truth-exited-early: {te[0]} vehicles, pred remained "
              f"{te[1]} steps, remaining pred delay={te[2]:.2f} s")
        print(f"  {variant}: entry waiting={wait} steps over {nmb} matched "
              f"births (expected 0); pred-only births={onlyp}; "
              f"truth-only births={onlyt}; t0 vehicles missing in pred="
              f"{t0m} (expected 0); truth presence gaps={gaps}")

    # ------- train-root check
    print("\n== G. train-root check (first 4 sorted train roots, 60 s "
          "grid) ==")
    train_windows = []
    for root in train4:
        for t0 in MAIN_T0S:
            for variant in ("L0", "L3"):
                w = process_window(root, t0, variant, dev)
                train_windows.append(w)
                log(f"train {variant} {root} t0={t0}: expo_def="
                    f"{w['exposure_p'] - w['exposure_t']:.1f} "
                    f"delay_def={w['delay_p'] - w['delay_t']:.1f}")
    for variant in ("L0", "L3"):
        ws = [w for w in train_windows if w["variant"] == variant]
        mp = float(np.mean([w["exposure_p"] for w in ws]))
        mt = float(np.mean([w["exposure_t"] for w in ws]))
        dp = float(np.mean([w["delay_p"] for w in ws]))
        dt_ = float(np.mean([w["delay_t"] for w in ws]))
        print(f"  {variant}: mean expo_def={mp - mt:.2f}  mean "
              f"delay_def={dp - dt_:.2f} veh s/window over {len(ws)} "
              f"train windows")

    # ---------------- write results
    csv = ["root,t0,variant,lane,segment,class,lc_affected,delay_pred,"
           "delay_true,n_steps_pred,n_steps_true"]
    for w in val_windows + train_windows:
        rows_ = sorted(w["cells"].items(), key=lambda kv: str(kv[0]))
        for (lane, seg, cl, lc), c in rows_:
            csv.append(f"{w['root']},{w['t0']},{w['variant']},{lane},"
                       f"{seg},{cl},{'yes' if lc else 'no'},"
                       f"{c[0]:.6f},{c[1]:.6f},{c[2]},{c[3]}")
    (outdir / "decomposition.csv").write_text("\n".join(csv) + "\n",
                                              encoding="utf-8")
    csv2 = ["variant,lane,d_bin_50m,v_pred_mean,v_true_mean,n"]
    for key, c in sorted(prof_all.items(),
                         key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        csv2.append(f"{key[0]},{key[1]},{key[2]},{c[0]/c[1]:.6f},"
                    f"{c[2]/c[3]:.6f},{c[1]}")
    (outdir / "speed_profile.csv").write_text("\n".join(csv2) + "\n",
                                              encoding="utf-8")
    log(f"wrote {outdir / 'decomposition.csv'} ({len(csv) - 1} data rows), "
        f"{outdir / 'speed_profile.csv'} ({len(csv2) - 1} data rows), "
        f"{outdir / 'stdout_last.txt'}, REPORT_P3J.md")
    print("\nDONE part 3j decomposition.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout
    write_report(outdir / "stdout_last.txt")


if __name__ == "__main__":
    main()
