"""WD-01-PA2G-LC1 counting audit (registered step 1, run FIRST).

Independent truth lane-change counter from the RAW physics rows of the
32 PA2G-A2 TRAIN worlds, compared against the eval's counter
(WindowSeries.stats on the saved eval_a2 raw rows).

Independence: this module does NOT import partA_data.pdl or the eval's
ModelMapper.  The lane_phys table below was derived directly from the
frozen network (assets/research/highway_merge_shepherd_long/osm.net.xml.gz,
read-only; see diag_lc1_netcheck.py):

    main_left  (0): 1191554636_1 @0, 1487606179_2 @2645.67, 1487606178_1 @2721.11
    main_right (1): 1191554636_0 @0, 1487606179_1 @2644.22, 1487606178_0 @2719.66
    accel      (2): 11553048#1_0 @0, 1487606179_0 @378.66   (DEAD END at
                    junction :13638027693 -- no onward connection)
    internal lanes: :1780739017_{0_0,1_0,1_1}, :13638027693_{0_0,0_1}
                    (same-phys transits only)

Event definition (identical semantics to the eval, built independently):
a vehicle changes lane_phys between two consecutive 0.5 s rows.  A run of
internal rows is resolved to the next physical lane (the eval's
retrospective_junction convention); the event is timestamped at the first
internal tick and counted only when the two physical endpoints differ.

TRAIN only: the plan split is asserted BEFORE any run file is opened; the
opened-file list is recorded and asserted at close-out.  Zero SUMO.

Usage:
    python diag_lc1_audit.py [--workers N]   (N <= 4, CPU-bound parsing)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wdpaths  # noqa: E402

DATA_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT)
PLAN = DATA_ROOT / "output" / "campaign_stage1_v3" / \
    "results_plan_chunk4" / "WORLD_PLAN.json"
RUNS_DIR = "campaign_stage1_chunk4"
ANALYSIS_ID = "WD-01-PA2G-LC1"
DT = 0.5
HS = (10, 30, 60, 120)

# ---- independent lane_phys table (hand-derived from the frozen net) ----
PHYS_OF = {
    "1191554636_1": (0, 0.0),
    "1487606179_2": (0, 2645.67),
    "1487606178_1": (0, 2721.11),
    "1191554636_0": (1, 0.0),
    "1487606179_1": (1, 2644.22),
    "1487606178_0": (1, 2719.66),
    "11553048#1_0": (2, 0.0),
    "1487606179_0": (2, 378.66),
}
INTERNAL_NEXT = {          # internal lane -> next physical lane_id
    ":1780739017_0_0": "1487606179_0",
    ":1780739017_1_0": "1487606179_1",
    ":1780739017_1_1": "1487606179_2",
    ":13638027693_0_0": "1487606178_0",
    ":13638027693_0_1": "1487606178_1",
}
ZONE = {0: (1845.67, 2871.11), 1: (1844.22, 2869.66), 2: (0.0, 454.10)}

RESULTS = HERE / "results" / "lc1"
OPENED: set = set()


def open_json_train(path: Path, train_worlds: set, kind: str = "plan") \
        -> dict:
    OPENED.add(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def load_worlds() -> List[dict]:
    """The 32 A2 worlds; plan split asserted TRAIN before any run open."""
    w32 = json.loads(
        (HERE / "results" / "lc1_a2_worlds32.json").read_text())
    OPENED.add(str(PLAN))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    meta = {w["world"]: w for w in plan["worlds"]}
    out = []
    for name in w32:
        entry = meta[name]
        assert entry.get("split") == "TRAIN", \
            f"{name} split={entry.get('split')} is not TRAIN -- refusing"
        out.append(entry)
    return out


def assert_train_only(names: set) -> None:
    bad = [p for p in OPENED
           if "__native_base" in p and
           re.search(r"([a-z0-9]+)__native_base", p).group(1) not in names]
    assert not bad, f"non-TRAIN run files opened: {bad}"


def phys_state(lane_id: str, pos: float) -> Tuple[int, float, int]:
    """(lane_phys, model_s, is_internal) -- independent mapping."""
    hit = PHYS_OF.get(lane_id)
    if hit is not None:
        return hit[0], hit[1] + float(pos), 0
    if lane_id in INTERNAL_NEXT:
        nxt = PHYS_OF[INTERNAL_NEXT[lane_id]]
        return -1, nxt[1], 1
    return -1, float("nan"), 1


def parse_world(entry: dict) -> dict:
    """One world: raw rows -> vehicle sequences -> events + windows.

    Two tracked-set variants per window:
      - tracked (raw physical rows only; the counting-identity rule the
        prediction side can mirror -- engine snapshots have no junction
        state);
      - tracked_dest (mid-junction vehicles included at their DESTINATION
        segment start, mirroring the eval's retrospective_junction
        placement) -- used to attribute the eval's extra events."""
    world = entry["world"]
    t0 = float(entry["t0_s"])
    path = DATA_ROOT / "runs" / RUNS_DIR / f"{world}__native_base" / \
        "run" / "physics_steps.jsonl"
    OPENED.add(str(path))
    seq: Dict[str, List[Tuple]] = {}   # vid -> [(tick, phys, s, v, flag)]
    dest_of: Dict[Tuple[str, int], float] = {}   # internal rows -> dest s0
    tick_index: Dict[int, Dict[str, Tuple]] = {}
    lc_flag_total = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"time_s": ')
            if i < 0:
                continue
            j = i + len('"time_s": ')
            t = float(line[j:line.find(",", j)])
            if t + 1e-9 < t0:
                continue
            row = json.loads(line)
            tick = int(round((t - t0) / DT))
            trow = {}
            for vid, v in row["vehicles"].items():
                phys, s, internal = phys_state(v["lane_id"],
                                               float(v["lane_position_m"]))
                flag = 1 if v.get("lane_changed") else 0
                lc_flag_total += flag
                spd = float(v["speed_mps"])
                seq.setdefault(vid, []).append((tick, phys, s, spd, flag))
                if internal:
                    dest_of[(vid, tick)] = \
                        PHYS_OF[INTERNAL_NEXT[v["lane_id"]]][1]
                else:
                    trow[vid] = (phys, s, spd)
            if trow:
                tick_index[tick] = trow

    def _ctx(tick: int, phys: int, s: float) -> Tuple[int, float, float]:
        """(n same-lane within +-50 m, their mean speed, gap ahead)."""
        row = tick_index.get(tick)
        if not row:
            return 0, float("nan"), float("nan")
        near = [(s2 - s, v2) for (p2, s2, v2) in row.values()
                if p2 == phys and -50.0 <= s2 - s <= 50.0 and s2 != s]
        ahead = [d for d, _v in ((x[0], x[1]) for x in near) if d > 0]
        return (len(near),
                (sum(v for _d, v in near) / len(near)) if near
                else float("nan"),
                min(ahead) if ahead else float("nan"))
    # ---- events: consecutive rows, internal runs resolved forward ----
    events = []
    n_unresolved = 0
    for vid, rows in seq.items():
        pend = None                    # (from_phys, from_s, from_v, t_tick)
        k = 0
        while k < len(rows):
            tick, phys, s, v, flag = rows[k]
            if phys >= 0:
                resolved = False
                if pend is not None:
                    fp, fs, fv, ft = pend
                    if fp != phys:
                        n_near, mv, ga = _ctx(ft - 1, fp, fs)
                        events.append(dict(
                            world=world, vid=vid, t=t0 + ft * DT,
                            f=fp, t_=phys, s=fs, v=fv, junction=1,
                            raw_flag=flag, n_near=n_near,
                            near_mean_v=mv, gap_ahead=ga))
                    pend = None
                    resolved = True
                if k + 1 < len(rows):
                    ntick, nphys, ns, nv, nflag = rows[k + 1]
                    if nphys >= 0:
                        if nphys != phys and ntick == tick + 1:
                            n_near, mv, ga = _ctx(tick, phys, s)
                            events.append(dict(
                                world=world, vid=vid,
                                t=t0 + ntick * DT, f=phys, t_=nphys,
                                s=s, v=v, junction=0,
                                raw_flag=nflag, n_near=n_near,
                                near_mean_v=mv, gap_ahead=ga))
                    elif ntick == tick + 1:
                        pend = (phys, s, v, ntick)
            elif pend is None and k > 0 and rows[k - 1][1] >= 0 and \
                    rows[k - 1][0] == tick - 1:
                ptick, pphys, ps, pv, _pf = rows[k - 1]
                pend = (pphys, ps, pv, tick)
            elif pend is not None and k == len(rows) - 1:
                n_unresolved += 1      # log ends mid-junction
            k += 1
    # ---- tracked sets + per-window per-H counts ----
    t_end = t0 + DT * max(tk for rows in seq.values() for tk, *_ in rows)

    def row_at(vid: str, j0: int):
        for tick, phys, s, _v, _f in seq[vid]:
            if tick == j0:
                return (tick, phys, s)
            if tick > j0:
                return None
        return None

    def dest_state(vid: str, j0: int):
        """(phys, s) at j0 with mid-junction vehicles placed at their
        destination segment start (the eval retro-fix convention)."""
        hit = row_at(vid, j0)
        if hit is None:
            return None
        _tick, phys, s = hit
        if phys >= 0:
            return phys, s
        # mid-junction: the next physical row after j0 gives the
        # destination phys; the retro-fix s is the destination segment
        # start recorded at parse time
        for tick, phys2, s2, _v, _f in seq[vid]:
            if tick > j0 and phys2 >= 0:
                return phys2, dest_of.get((vid, j0), s2)
        return None

    windows = []
    for kk in range(1, 19):
        w = t0 + 30.0 * kk
        if w > t_end + 1e-9:
            break
        j0 = int(round((w - t0) / DT))
        tracked = set()
        tracked_dest = set()
        for vid in seq:
            hit = row_at(vid, j0)
            if hit is None:
                continue
            _tick, phys, s = hit
            if phys >= 0 and ZONE[phys][0] <= s < ZONE[phys][1]:
                tracked.add(vid)
            ds = dest_state(vid, j0)
            if ds is not None:
                p2, s2 = ds
                if ZONE[p2][0] <= s2 < ZONE[p2][1]:
                    tracked_dest.add(vid)
        ev_all = [e for e in events if e["vid"] in tracked_dest]
        for H in HS:
            if w + H > t_end + 1e-9:
                continue
            counts = {"merge": 0, "disc_left": 0, "disc_right": 0,
                      "other": 0}
            counts_dest = dict(counts)
            for e in ev_all:
                if w < e["t"] <= w + H + 1e-9:
                    key = {(2, 1): "merge", (1, 0): "disc_left",
                           (0, 1): "disc_right"}.get((e["f"], e["t_"]),
                                                     "other")
                    counts_dest[key] += 1
                    if e["vid"] in tracked:
                        counts[key] += 1
            windows.append(dict(world=world, w=w, H=H,
                                n_tracked=len(tracked),
                                n_tracked_dest=len(tracked_dest),
                                **counts,
                                **{f"dest_{k}": v for k, v in
                                   counts_dest.items()}))
    return dict(world=world, events=events, windows=windows,
                n_vehicles=len(seq), lc_flag_total=lc_flag_total,
                n_unresolved=n_unresolved, t_end=t_end)


def _worker(args):
    rank, entries = args
    out = []
    for e in entries:
        t_a = time.perf_counter()
        r = parse_world(e)
        r["wall_s"] = round(time.perf_counter() - t_a, 1)
        out.append(r)
        print(f"  [w{rank}] {e['world']}: {len(r['events'])} events, "
              f"{len(r['windows'])} window rows, {r['wall_s']}s",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n_workers = max(1, min(4, args.workers))
    t_start = time.perf_counter()
    entries = load_worlds()
    names = {e["world"] for e in entries}
    print(f"{ANALYSIS_ID} counting audit: {len(entries)} TRAIN worlds, "
          f"workers {n_workers}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if n_workers == 1:
        results = [_worker((0, entries))]
    else:
        ctx = mp.get_context("spawn")
        chunks = [entries[i::n_workers] for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, [(i, chunks[i])
                                         for i in range(n_workers)])
    all_res = [r for sub in results for r in sub]

    # ---- raw event rows BEFORE aggregation ----
    n_ev = 0
    with open(RESULTS / "truth_events.jsonl", "w", encoding="utf-8") as fh:
        for r in all_res:
            for e in r["events"]:
                fh.write(json.dumps(e) + "\n")
                n_ev += 1
    with open(RESULTS / "truth_windows.jsonl", "w", encoding="utf-8") as fh:
        for r in all_res:
            for wdw in r["windows"]:
                fh.write(json.dumps(wdw) + "\n")

    # ---- compare against the eval's saved lc_true ----
    eval_lc = {}
    with open(HERE / "results" / "eval_a2" / "raw_rows.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("model") != "v2g":
                continue
            eval_lc[(r["world"], round(float(r["w"]), 3), int(r["H"]))] = \
                r["lc_true"]
    row_n = match = match_dest = 0
    mismatches = []
    mismatches_dest = []
    extra_events_by_class: Dict[str, int] = {}
    for wdw in (w for r in all_res for w in r["windows"]):
        key = (wdw["world"], round(wdw["w"], 3), wdw["H"])
        ref = eval_lc.get(key)
        if ref is None:
            continue
        row_n += 1
        mine = {"merge": wdw["merge"], "disc_left": wdw["disc_left"],
                "disc_right": wdw["disc_right"]}
        mine_dest = {"merge": wdw["dest_merge"],
                     "disc_left": wdw["dest_disc_left"],
                     "disc_right": wdw["dest_disc_right"]}
        if mine == ref:
            match += 1
        else:
            for k in ref:
                d = mine_dest[k] - mine[k]
                if d != ref[k] - mine[k]:
                    pass
            mismatches.append(dict(key=list(map(str, key)), eval=ref,
                                   independent=mine))
        if mine_dest == ref:
            match_dest += 1
        else:
            mismatches_dest.append(dict(key=list(map(str, key)), eval=ref,
                                        independent_dest=mine_dest))
        for k in ref:
            extra_events_by_class[k] = extra_events_by_class.get(k, 0) + \
                (ref[k] - mine[k])
    # ---- whole-run global class table ----
    glob: Dict[Tuple, int] = {}
    per_world = {}
    for r in all_res:
        c: Dict[Tuple, int] = {}
        for e in r["events"]:
            k = (e["f"], e["t_"])
            glob[k] = glob.get(k, 0) + 1
            c[k] = c.get(k, 0) + 1
        per_world[r["world"]] = {f"{a}->{b}": n for (a, b), n in
                                 sorted(c.items())}
    # ---- raw lane_changed flag cross-check ----
    flag_total = sum(r["lc_flag_total"] for r in all_res)
    ev_flag = sum(1 for r in all_res for e in r["events"] if e["raw_flag"])
    n_events = sum(len(r["events"]) for r in all_res)
    report = dict(
        analysis_id=ANALYSIS_ID,
        worlds=len(all_res),
        events_total=n_events,
        events_with_raw_lc_flag=ev_flag,
        raw_lc_flag_ticks_total=flag_total,
        unresolved_junction_ends=n_unresolved_total(all_res),
        window_rows_compared=row_n,
        window_rows_equal=match,
        window_mismatches=mismatches[:50],
        n_window_mismatches=len(mismatches),
        window_rows_equal_dest_tracked=match_dest,
        n_window_mismatches_dest_tracked=len(mismatches_dest),
        window_mismatches_dest=mismatches_dest[:10],
        eval_extra_events_by_class=extra_events_by_class,
        global_class_counts={f"{a}->{b}": n for (a, b), n in
                             sorted(glob.items())},
        per_world_class_counts=per_world,
        opened_run_files=sorted(OPENED),
        assert_train_only="passed",
        wall_clock_s=time.perf_counter() - t_start,
    )
    assert_train_only(names)
    (RESULTS / "audit_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_world_class_counts",
                                   "opened_run_files",
                                   "window_mismatches")}, indent=1))
    print("DONE audit", ANALYSIS_ID,
          f"wall {time.perf_counter() - t_start:.0f}s")


def n_unresolved_total(all_res) -> int:
    return sum(r["n_unresolved"] for r in all_res)


if __name__ == "__main__":
    main()
