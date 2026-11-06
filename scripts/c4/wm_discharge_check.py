"""C11 QUEUE-DISCHARGE / ZIPPER check: how the merge area serves the two queues.

Owner-approved analysis (no training, no SUMO): on the recorded TRAIN-val starts of the
C1-C6 fidelity suite (24 fixed starts, 300 s, entries queue) this tool measures, for the
recorded SUMO stream and for WM rollouts of the same starts, how the merge area serves the
ramp queue (lane 2, acceleration lane with the hard stop at 75.44 m) and the mainline
queues (lanes 0/1):

* **served rates** - ramp merges (lane-2 -> lane-1 insertions) per minute and mainline
  passages (mainline-origin crossings of the merge-zone exit s = 75.44 m) per minute;
* **headways at the merge point** - insertion headway (successive ramp merges) and
  discharge headway (successive lane-1 crossings of 75.44 m), median / p10 / p90;
* **ramp queue** - lane-2 roster count over time (mean / p90 / time course) and the ramp
  dwell distribution (merge time - t0 of served ramp cars) incl. the never-served share;
* **zipper alternation** - over successive lane-1 crossing services while BOTH queues are
  present, the share of same-stream successive services and the same-stream run lengths;
* **insertion follower braking** - for each ramp insertion, the nearest lane-1 follower's
  minimum acceleration in a window around the insertion and its gap at the insertion.

Both sides are restricted to the t0 roster (the recorded v5_native streams carry no
post-t0 entrants; WM entrant slots are excluded), so WM and SUMO are scored on the SAME
vehicle set.  Every threshold in the metrics is data-defined: the geometry (75.44 m
merge-zone exit, lanes) comes from the road contract, and the PASS/FLAG verdict is the
episode-bootstrap 95 % band of the SUMO per-start values (bootstrap over starts, fixed
seed); a WM is flagged on a metric when its fleet mean (mean of per-start means, k samples
averaged) falls outside that band.

    python scripts/c4/wm_discharge_check.py \\
        --wm round2=/path/wm_s_best.pt --wm round7b=/path/wm_s_best.pt \\
        --selection sel24_states.json --horizon 300 --k 2 --entries queue \\
        --device cpu --workers 8 --out discharge.json --md DISCHARGE.md

Refuses restricted path tokens (sealed/holdout/heldout/cal_split/chunk5/j4b/inc1) and
evaluation seeds 964000-964199.  torch/wms are imported lazily; the pure helpers run
without torch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shlex
import sys
import time
from pathlib import Path

SCHEMA = "c4_wm_discharge_check/1"
DEFAULT_HORIZON_S = 300.0
DEFAULT_K = 2
DEFAULT_QUEUE_TARGETS = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
MERGE_S1_M = 75.44              # merge-zone exit on the acceleration lane (contract)
ROAD_END_M = 553.33
RAMP_LANE = 2
MAINLINE_LANES = (0, 1)
DEFAULT_LENGTH_M = 5.0
MAX_WORKERS = 16
EVAL_SEED_MIN, EVAL_SEED_MAX = 964000, 964199
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
BOOT_N = 10000
BOOT_SEED = 411
FOLLOWER_PRE_S = 1.0
FOLLOWER_POST_S = 6.0
#: minimum per-start sample count for a metric to enter the fleet band/flag table
MIN_STARTS = 2

CPUN_COMMAND = r"""cd ~/c4/repo_s7
mkdir -p ~/c4/discharge
OMP_NUM_THREADS=1 C4_ALLOWED_ROOTS=$HOME/c4 PYTHONPATH=src:.:tools \
  ~/miniconda3/bin/python scripts/c4/wm_discharge_check.py \
  --wm round2=~/c4/score72/ck_round2/wm_s_best.pt \
  --wm round7b=~/c4/score72/ck_round7b/wm_s_best.pt \
  --selection ~/c4/delay_check/sel24_states.json \
  --horizon 300 --k 2 --entries queue --device cpu --workers 8 \
  --out ~/c4/discharge/discharge_check.json \
  --md ~/c4/discharge/DISCHARGE.md"""

_WORKER: dict = {}


# --------------------------------------------------------------------------- module plumbing
_SUITE = None


def _suite_module():
    """The C1-C6 fidelity suite module (roster / stats / predictor-factory helpers)."""
    global _SUITE
    if _SUITE is None:
        path = Path(__file__).resolve().parent / "wm_fidelity_suite.py"
        spec = importlib.util.spec_from_file_location("wm_discharge_suite_helpers", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SUITE = module
    return _SUITE


def refuse_restricted(path) -> str:
    text = str(path)
    if any(token in text.replace("\\", "/").lower() for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {text}")
    return text


def refused_seed(*values) -> bool:
    """True when any value is an evaluation seed (964000-964199)."""
    for value in values:
        try:
            seed = int(value)
        except (TypeError, ValueError):
            continue
        if EVAL_SEED_MIN <= seed <= EVAL_SEED_MAX:
            return True
    return False


def parse_list(text, cast=float) -> tuple:
    if isinstance(text, (list, tuple)):
        return tuple(cast(v) for v in text)
    parts = str(text).replace(";", ",").split(",")
    return tuple(cast(part) for part in parts if part.strip())


# --------------------------------------------------------------------------- events / queues
def _seq(frames, t0: float, horizon: float, roster) -> dict[str, list[tuple[float, dict]]]:
    """Per-roster-vehicle observation sequence ``(t, row)`` within the window."""
    out: dict[str, list[tuple[float, dict]]] = {vid: [] for vid in roster}
    lo, hi = t0 - 1e-6, t0 + float(horizon) + 1e-6
    for frame in frames:
        t = float(frame.get("time_s", 0.0))
        if t < lo or t > hi:
            continue
        for row in frame.get("vehicles", []):
            if not row.get("present", True):
                continue
            vid = str(row.get("id", ""))
            if vid in out:
                out[vid].append((t, row))
    for vid in out:
        out[vid].sort(key=lambda item: item[0])
    return out


def service_events(frames, t0: float, horizon: float, roster,
                   merge_end_m: float = MERGE_S1_M) -> dict:
    """Merge (insertion) and merge-exit crossing events of the t0 roster.

    * a **merge** is the first lane-2 -> lane-1 lane change of a roster vehicle
      (``t`` = first lane-1 observation, ``s`` = its position);
    * a **crossing** is the first observation with ``s >= merge_end_m`` of a roster
      vehicle that was still upstream of the exit (``origin`` = ``ramp`` when the
      vehicle's first observed lane is 2, else ``mainline``; ``lane`` = lane at the
      crossing).
    """
    seq = _seq(frames, t0, horizon, roster)
    merges, crossings = [], []
    first_lane: dict[str, int] = {}
    for vid in sorted(seq):
        rows = seq[vid]
        if not rows:
            continue
        first_lane[vid] = int(rows[0][1].get("lane", -1))
        prev_lane, crossed = None, False
        for t, row in rows:
            lane = int(row.get("lane", -1))
            if prev_lane == RAMP_LANE and lane == 1:
                merges.append({"id": vid, "t": float(t),
                               "s": float(row.get("s", 0.0))})
                prev_lane = lane
                continue
            if not crossed and float(row.get("s", 0.0)) >= float(merge_end_m):
                crossed = True
                crossings.append({
                    "id": vid, "t": float(t), "s": float(row.get("s", 0.0)),
                    "lane": lane,
                    "origin": ("ramp" if first_lane[vid] == RAMP_LANE else "mainline")})
            prev_lane = lane
    merges.sort(key=lambda e: (e["t"], e["id"]))
    crossings.sort(key=lambda e: (e["t"], e["id"]))
    return {"merges": merges, "crossings": crossings, "first_lane": first_lane,
            "seq": seq}


def queue_series(frames, t0: float, horizon: float, roster, events,
                 merge_end_m: float = MERGE_S1_M) -> list[dict]:
    """Per-frame ramp queue and unserved mainline queue counts.

    ``ramp_q`` = roster vehicles currently on lane 2 (all of them are pre-merge);
    ``mainline_q`` = roster vehicles on lane 0/1 that originate from the mainline and
    have not yet crossed the merge-zone exit.
    """
    first_lane = events["first_lane"]
    out = []
    lo, hi = t0 - 1e-6, t0 + float(horizon) + 1e-6
    for frame in frames:
        t = float(frame.get("time_s", 0.0))
        if t < lo or t > hi:
            continue
        ramp_q = mainline_q = 0
        for row in frame.get("vehicles", []):
            if not row.get("present", True):
                continue
            vid = str(row.get("id", ""))
            if vid not in first_lane:
                continue
            lane, s = int(row.get("lane", -1)), float(row.get("s", 0.0))
            if lane == RAMP_LANE:
                ramp_q += 1
            elif (lane in MAINLINE_LANES and first_lane[vid] != RAMP_LANE
                  and s < float(merge_end_m)):
                mainline_q += 1
        out.append({"t": t, "ramp_q": ramp_q, "mainline_q": mainline_q})
    out.sort(key=lambda row: row["t"])
    return out


def _queue_at(series, t: float, max_gap_s: float = 0.75) -> tuple[int, int]:
    if not series:
        return (0, 0)
    row = min(series, key=lambda item: abs(item["t"] - t))
    if abs(row["t"] - t) > max_gap_s:
        return (0, 0)
    return (int(row["ramp_q"]), int(row["mainline_q"]))


# --------------------------------------------------------------------------- C11 metric block
def headway_stats(times) -> dict:
    """Median / p10 / p90 of successive-event headways (seconds)."""
    _s = _suite_module()
    vals = sorted(b - a for a, b in zip(times, times[1:]) if b > a + 1e-9)
    return {"n": len(vals), "p10": _s._pct(vals, 10), "p50": _s._pct(vals, 50),
            "p90": _s._pct(vals, 90)}


def zipper_stats(crossings, series) -> dict:
    """Alternation of successive lane-1 services while both queues are present.

    The service sequence is the lane-1 crossings sorted by time; a successive pair
    ``(i-1, i)`` counts when BOTH queues still have demand at the later event's time
    (ramp queue >= 1 lane-2 vehicle and >= 1 unserved mainline-origin vehicle upstream
    of the merge-zone exit).  A run is a maximal streak of consecutive counted services
    from the same stream.
    """
    _s = _suite_module()
    lane1 = [e for e in crossings if int(e["lane"]) == 1]
    flags = []
    for event in lane1:
        ramp_q, mainline_q = _queue_at(series, event["t"])
        flags.append(ramp_q >= 1 and mainline_q >= 1)
    same = []
    run_lens = []
    streak = 0
    for index in range(1, len(lane1)):
        if not flags[index]:
            if streak:
                run_lens.append(streak)
            streak = 0
            continue
        pair_same = lane1[index]["origin"] == lane1[index - 1]["origin"]
        same.append(pair_same)
        if pair_same and streak:
            streak += 1
        else:
            if streak:
                run_lens.append(streak)
            streak = 1
    if streak:
        run_lens.append(streak)
    return {
        "n_events_lane1": len(lane1),
        "n_pairs": len(same),
        "same_stream_share": (sum(1 for s in same if s) / len(same)) if same else None,
        "run_len_median": _s._median(run_lens),
        "run_len_p90": _s._pct(run_lens, 90),
        "n_runs": len(run_lens),
    }


def follower_decel(frames_seq, merges, roster,
                   pre_s: float = FOLLOWER_PRE_S, post_s: float = FOLLOWER_POST_S,
                   default_length_m: float = DEFAULT_LENGTH_M) -> dict:
    """Nearest lane-1 follower's braking around each ramp insertion.

    For every merge event the nearest lane-1 roster vehicle behind the insertion
    position at the insertion time is located; the metric is its minimum forward
    acceleration (backward difference on the observation grid) inside
    ``[t_merge - pre_s, t_merge + post_s]`` and its gap to the inserter.
    """
    _s = _suite_module()
    min_accels, gaps = [], []
    for event in merges:
        t_m, s_m = float(event["t"]), float(event["s"])
        follower, f_s = None, None
        for vid in sorted(roster):
            rows = [(t, row) for t, row in frames_seq.get(vid, [])
                    if abs(t - t_m) <= 0.75]
            if not rows:
                continue
            t, row = min(rows, key=lambda item: abs(item[0] - t_m))
            if int(row.get("lane", -1)) != 1:
                continue
            s = float(row.get("s", 0.0))
            if s < s_m and (f_s is None or s > f_s):
                follower, f_s = vid, s
        if follower is None:
            continue
        accels = []
        rows = frames_seq.get(follower, [])
        for (ta, ra), (tb, rb) in zip(rows, rows[1:]):
            dt = tb - ta
            if not (0.0 < dt <= 0.75):
                continue
            if t_m - float(pre_s) - 1e-6 <= tb <= t_m + float(post_s) + 1e-6:
                accels.append((float(rb.get("v", 0.0)) - float(ra.get("v", 0.0))) / dt)
        if not accels:
            continue
        min_accels.append(min(accels))
        gaps.append(s_m - float(default_length_m) - f_s)
    return {
        "n_events": len(min_accels),
        "min_accel_p10": _s._pct(min_accels, 10),
        "min_accel_p50": _s._pct(min_accels, 50),
        "min_accel_p90": _s._pct(min_accels, 90),
        "gap_median_m": _s._median(gaps),
    }


def discharge_block(frames, t0: float, horizon: float, roster,
                    merge_end_m: float = MERGE_S1_M,
                    queue_targets=DEFAULT_QUEUE_TARGETS) -> dict:
    """The full C11 metric block of one stream (recorded SUMO or one WM sample)."""
    _s = _suite_module()
    events = service_events(frames, t0, horizon, roster, merge_end_m)
    series = queue_series(frames, t0, horizon, roster, events, merge_end_m)
    merges, crossings = events["merges"], events["crossings"]
    first_lane = events["first_lane"]
    ramp_ids = {vid for vid, lane in first_lane.items() if lane == RAMP_LANE}
    minutes = float(horizon) / 60.0

    ramp_crossings = [e for e in crossings if e["origin"] == "ramp"]
    main_crossings = [e for e in crossings if e["origin"] == "mainline"]
    lane1_crossings = [e for e in crossings if int(e["lane"]) == 1]

    served = {e["id"]: float(e["t"]) for e in merges}
    dwells = [served[vid] - float(t0) for vid in sorted(ramp_ids) if vid in served]
    never = [vid for vid in sorted(ramp_ids) if vid not in served]

    ramp_q = [row["ramp_q"] for row in series]
    queue_at = {}
    for target in queue_targets:
        tt = float(t0) + float(target)
        near = [row["ramp_q"] for row in series if abs(row["t"] - tt) <= 0.75]
        queue_at[_s._hlabel(target)] = (sum(near) / len(near)) if near else None

    return {
        "n_roster_obs": len(first_lane),
        "n_ramp_t0": len(ramp_ids),
        "rates": {
            "ramp_merges_per_min": len(merges) / minutes,
            "ramp_exit_crossings_per_min": len(ramp_crossings) / minutes,
            "mainline_passages_per_min": len(main_crossings) / minutes,
            "lane1_crossings_per_min": len(lane1_crossings) / minutes,
            "n_ramp_merges": float(len(merges)),
            "n_mainline_passages": float(len(main_crossings)),
        },
        "headways": {
            "insertion": headway_stats([e["t"] for e in merges]),
            "discharge_lane1": headway_stats([e["t"] for e in lane1_crossings]),
            "mainline_passage": headway_stats([e["t"] for e in main_crossings]),
        },
        "queue": {
            "ramp_queue_mean": _s._mean(ramp_q),
            "ramp_queue_p50": _s._pct(ramp_q, 50),
            "ramp_queue_p90": _s._pct(ramp_q, 90),
            "ramp_queue_at": queue_at,
        },
        "dwell": {
            "n_served": len(dwells),
            "never_served_share": (len(never) / len(ramp_ids)) if ramp_ids else None,
            "served_p10": _s._pct(dwells, 10),
            "served_p50": _s._pct(dwells, 50),
            "served_p90": _s._pct(dwells, 90),
        },
        "zipper": zipper_stats(crossings, series),
        "follower": follower_decel(events["seq"], merges, roster),
        "merge_s_median_m": _s._median([e["s"] for e in merges]),
    }


# --------------------------------------------------------------------------- fleet aggregation
METRIC_SPECS = (
    ("rates.ramp_merges_per_min", "served ramp merges / min"),
    ("rates.mainline_passages_per_min", "mainline passages / min"),
    ("rates.lane1_crossings_per_min", "lane-1 crossings / min (zipper outflow)"),
    ("headways.insertion.p50", "insertion headway p50 (s)"),
    ("headways.insertion.p10", "insertion headway p10 (s)"),
    ("headways.insertion.p90", "insertion headway p90 (s)"),
    ("headways.discharge_lane1.p50", "lane-1 discharge headway p50 (s)"),
    ("headways.discharge_lane1.p10", "lane-1 discharge headway p10 (s)"),
    ("headways.discharge_lane1.p90", "lane-1 discharge headway p90 (s)"),
    ("queue.ramp_queue_mean", "ramp queue mean (veh)"),
    ("queue.ramp_queue_p90", "ramp queue p90 (veh)"),
    ("dwell.served_p50", "served ramp dwell p50 (s)"),
    ("dwell.served_p90", "served ramp dwell p90 (s)"),
    ("dwell.never_served_share", "never-served ramp share"),
    ("zipper.same_stream_share", "same-stream successive share (both queues)"),
    ("zipper.run_len_median", "same-stream run length median"),
    ("follower.min_accel_p50", "follower min accel p50 (m/s^2)"),
    ("follower.min_accel_p10", "follower min accel p10 (m/s^2)"),
    ("follower.gap_median_m", "follower gap at insertion median (m)"),
)

#: headline metrics of the per-start table
HEADLINE = ("rates.ramp_merges_per_min", "rates.mainline_passages_per_min",
            "headways.insertion.p50", "queue.ramp_queue_mean",
            "dwell.never_served_share", "zipper.same_stream_share")


def _leaf(block: dict, dotted: str):
    node = block
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool):
        return None
    return (float(node) if isinstance(node, (int, float))
            and math.isfinite(float(node)) else None)


def per_start_values(state: dict, side: str) -> dict[str, float | None]:
    block = state.get(side) or {}
    return {key: _leaf(block, key) for key, _name in METRIC_SPECS}


def bootstrap_mean(values, n_boot: int = BOOT_N, seed: int = BOOT_SEED,
                   ci: float = 95.0) -> dict | None:
    """Episode-bootstrap CI of the mean: resample the per-start values with replacement."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < MIN_STARTS:
        return None
    rng = random.Random(int(seed))
    n = len(vals)
    means = []
    for _ in range(int(n_boot)):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (100.0 - float(ci)) / 2.0

    def _q(q):
        pos = (len(means) - 1) * q / 100.0
        lo_i = int(math.floor(pos))
        hi_i = min(lo_i + 1, len(means) - 1)
        return means[lo_i] + (means[hi_i] - means[lo_i]) * (pos - lo_i)

    return {"n": n, "n_boot": int(n_boot), "seed": int(seed),
            "lo": _q(alpha), "mean": sum(vals) / n, "hi": _q(100.0 - alpha)}


def fleet_rows(states, wm_names, n_boot: int = BOOT_N,
               seed: int = BOOT_SEED) -> dict:
    """Fleet means, SUMO bootstrap bands and per-WM flags for every metric."""
    out = {}
    for key, name in METRIC_SPECS:
        sumo_vals = [per_start_values(state, "sumo")[key] for state in states]
        band = bootstrap_mean(sumo_vals, n_boot=n_boot, seed=seed)
        row = {"name": name, "sumo_values": sumo_vals,
               "sumo": band, "wm": {}}
        for wm in wm_names:
            vals = [per_start_values(state["wm_blocks"][wm], "wm")[key]
                    for state in states]
            wm_band = bootstrap_mean(vals, n_boot=n_boot, seed=seed)
            entry = {"values": vals, "fleet": wm_band}
            if band is not None and wm_band is not None:
                entry["flagged"] = (wm_band["mean"] < band["lo"]
                                    or wm_band["mean"] > band["hi"])
                entry["direction"] = ("below" if wm_band["mean"] < band["lo"]
                                      else "above" if wm_band["mean"] > band["hi"]
                                      else "inside")
            else:
                entry["flagged"] = None
                entry["direction"] = "n/a"
            row["wm"][wm] = entry
        out[key] = row
    return out


# --------------------------------------------------------------------------- per-state driver
def evaluate_state(predictor, payload) -> dict:
    _s = _suite_module()
    record, cfg = payload
    if refused_seed(record.get("seed"), record.get("sumo_seed")):
        raise SystemExit("refusing evaluation seed in a state record")
    t0_frame, roster, cavs = _s._t0_roster(record)
    t0 = float(t0_frame["time_s"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from wms.eval.normalize import recorded_for
    recorded_arms = record.get("recorded") or {}
    arm = str(cfg["arm"])
    if arm not in recorded_arms and isinstance(recorded_arms, dict) and recorded_arms:
        arm = sorted(recorded_arms)[0]
    recorded = recorded_for(record, arm)
    horizon = float(cfg["horizon_s"])
    sumo = discharge_block(recorded, t0, horizon, roster)
    wm_blocks = {}
    for wm in cfg["wm_names"]:
        samples = []
        for sample in range(max(1, int(cfg["k"]))):
            controllers = dict(record.get("cav_controllers") or {})
            controllers["arm"] = arm
            frames = predictor[wm](record, controllers, horizon, sample)
            samples.append(discharge_block(frames, t0, horizon, roster))
        wm_blocks[wm] = {"wm": _s._avg_numeric(samples)}
    return {
        "world": str(record.get("world")),
        "key": record.get("key"),
        "source_arm": record.get("source_arm"),
        "source_set": record.get("source_set"),
        "t0_s": t0,
        "arm": arm,
        "n_roster": len(roster),
        "n_cav": len(cavs),
        "n_ramp_t0": sumo.get("n_ramp_t0"),
        "sumo": sumo,
        "wm_blocks": wm_blocks,
    }


def _init_worker(wm_specs, device, entries) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    _s = _suite_module()
    _WORKER["predictor"] = {
        name: _s._delay_module().PredictorFactory(
            path, device=device, entries=entries)()
        for name, path in wm_specs}


def _work_state(payload):
    return evaluate_state(_WORKER["predictor"], payload)


def _map(payloads, wm_specs, args) -> list[dict]:
    workers = max(1, min(int(args.workers), MAX_WORKERS))
    if workers <= 1:
        _init_worker(wm_specs, args.device, args.entries)
        return [_work_state(payload) for payload in payloads]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(wm_specs, args.device, args.entries)) as pool:
        futures = [pool.submit(_work_state, payload) for payload in payloads]
        return [future.result() for future in futures]


# --------------------------------------------------------------------------- CLI / report
def parse_wm(spec: str) -> tuple[str, str]:
    name, _, path = spec.partition("=")
    name = name.strip() or Path(path).stem
    return name, refuse_restricted(path.strip())


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", action="append", required=True,
                    help="NAME=checkpoint path (repeatable; all scored on the same starts)")
    ap.add_argument("--states", help="states JSON (or directory) with the recorded starts")
    ap.add_argument("--selection", help="fixed selection JSON (as in the C1-C6 suite)")
    ap.add_argument("--select", type=int, default=24,
                    help="number of starts when no fixed selection is given")
    ap.add_argument("--horizon", type=float, default=DEFAULT_HORIZON_S)
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="WM samples per start")
    ap.add_argument("--arm", default="native",
                    help="recorded CAV-teacher arm (falls back to the record's first arm)")
    ap.add_argument("--entries", choices=("off", "on", "queue"), default="queue")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--workers", type=int, default=1,
                    help=f"parallel workers (capped at {MAX_WORKERS})")
    ap.add_argument("--queue-targets", default=",".join(str(int(t)) for t in
                                                        DEFAULT_QUEUE_TARGETS))
    ap.add_argument("--n-boot", type=int, default=BOOT_N)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--md", help="markdown report path")
    return ap


def _resolve_starts(args):
    _s = _suite_module()
    helpers = _s._delay_module()
    if args.selection:
        selection = refuse_restricted(args.selection)
        if Path(selection).exists():
            return helpers.load_selection(selection, "states")
    if not args.states:
        raise SystemExit("--states is required when --selection is absent/empty")
    starts = helpers.select_evenly(helpers.load_records(refuse_restricted(args.states)),
                                   int(args.select))
    if not starts:
        raise SystemExit("no starts selected")
    return starts


def main(argv=None) -> int:
    _s = _suite_module()
    args = parser().parse_args(argv)
    _s._ensure_src_on_path()
    for token in (args.out, args.md, args.states, args.selection, *[w for w in args.wm]):
        if token:
            refuse_restricted(token)
    if not (0 < args.horizon):
        raise SystemExit("--horizon must be positive")
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers must be <= {MAX_WORKERS}")
    if refused_seed(args.boot_seed):
        raise SystemExit("--boot-seed must not be an evaluation seed (964000-964199)")
    wm_specs = [parse_wm(spec) for spec in args.wm]
    wm_names = [name for name, _ in wm_specs]
    if len(set(wm_names)) != len(wm_names):
        raise SystemExit("WM names must be unique")
    starts = _resolve_starts(args)
    queue_targets = parse_list(args.queue_targets)
    cfg = {"arm": args.arm, "k": int(args.k), "horizon_s": float(args.horizon),
           "wm_names": wm_names}

    t_start = time.time()
    states = _map([(record, cfg) for record in starts], wm_specs, args)
    rows = fleet_rows(states, wm_names, n_boot=int(args.n_boot),
                      seed=int(args.boot_seed))

    results = {
        "schema": SCHEMA,
        "wm": {name: path for name, path in wm_specs},
        "arm": args.arm,
        "k": int(args.k),
        "horizon_s": float(args.horizon),
        "entries": args.entries,
        "device": args.device,
        "workers": max(1, min(int(args.workers), MAX_WORKERS)),
        "merge_end_m": MERGE_S1_M,
        "road_end_m": ROAD_END_M,
        "n_starts": len(starts),
        "n_boot": int(args.n_boot),
        "boot_seed": int(args.boot_seed),
        "queue_targets": [_s._hlabel(t) for t in queue_targets],
        "elapsed_s": time.time() - t_start,
        "command_local": "python " + shlex.join(sys.argv),
        "command_cpun": CPUN_COMMAND,
        "fleet": rows,
        "states": states,
        "summary": {
            wm: {"flagged": [key for key in rows if (rows[key]["wm"][wm].get("flagged"))],
                 "inside": [key for key in rows
                            if rows[key]["wm"][wm].get("flagged") is False]}
            for wm in wm_names},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_s._sanitize(results), indent=2, sort_keys=True),
                   encoding="utf-8")
    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(results, queue_targets), encoding="utf-8")
    print(json.dumps({"out": str(out), "md": (str(args.md) if args.md else None),
                      "n_starts": len(starts), "elapsed_s": round(results["elapsed_s"], 1),
                      "summary": results["summary"]}, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- markdown
def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(results: dict, queue_targets) -> str:
    _s = _suite_module()
    wm_names = list(results["wm"].keys())
    rows = results["fleet"]
    states = results["states"]

    def band(key):
        b = rows[key]["sumo"]
        if b is None:
            return "n/a"
        return (f"{_fmt(b['mean'])} [{_fmt(b['lo'])}, {_fmt(b['hi'])}] (n={b['n']})")

    lines = [
        "# C11 - queue discharge / zipper baseline (WM vs recorded SUMO)", "",
        f"- schema `{results['schema']}`; {results['n_starts']} TRAIN-val starts "
        f"(fixed selection); horizon {results['horizon_s']:.0f} s; k={results['k']}; "
        f"entries `{results['entries']}`",
        f"- WM rollouts: {', '.join(f'`{w}`' for w in wm_names)}; workers "
        f"{results['workers']}; elapsed {results['elapsed_s']:.0f} s",
        f"- merge-zone exit s = {results['merge_end_m']:.2f} m (acceleration-lane hard "
        f"stop); road end {results['road_end_m']:.2f} m; ramp = lane 2, mainline = 0/1",
        "- BOTH sides are restricted to the **t0 roster** (the recorded v5_native streams "
          "carry no post-t0 entrants; WM entry slots are excluded), so WM and SUMO are "
          "scored on the same vehicle set over the full 300 s window.",
        "- rates use the fixed window (300 s = 5 min); no behavioural thresholds - every "
          "verdict is the episode-bootstrap 95 % band of the SUMO per-start values "
          f"({results['n_boot']} resamples over starts, seed {results['boot_seed']}); a WM "
          "is FLAGGED on a metric when its fleet mean falls outside the band.",
        "",
        "## Fleet table - SUMO band vs WM fleet means", "",
        "| metric | SUMO mean [95 % band] | " + " | ".join(
            f"{w} mean (verdict)" for w in wm_names) + " |",
        "|---|---|" + "---|" * len(wm_names),
    ]
    for key, name in METRIC_SPECS:
        cells = []
        for wm in wm_names:
            entry = rows[key]["wm"][wm]
            mark = ("**FLAG " + entry["direction"] + "**" if entry.get("flagged")
                    else ("ok" if entry.get("flagged") is False else "n/a"))
            cells.append(f"{_fmt(entry['fleet']['mean'] if entry['fleet'] else None)} "
                         f"({mark}, n={entry['fleet']['n'] if entry['fleet'] else 0})")
        lines.append(f"| {name} | {band(key)} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-start table (headline metrics)", ""]
    header = ["world", "key", "t0 s", "ramp t0", "side"] + [
        rows[k]["name"].split(" (")[0] for k in HEADLINE]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for state in states:
        sides = [("SUMO", state["sumo"])] + [(w, state["wm_blocks"][w]["wm"])
                                                for w in wm_names]
        for side, block in sides:
            vals = per_start_values({"sumo": block}, "sumo")
            cells = [_fmt(vals[k]) for k in HEADLINE]
            lines.append(f"| {state['world']} | {state['key']} | {state['t0_s']:.0f} | "
                         f"{state['n_ramp_t0']} | {side} | " + " | ".join(cells) + " |")

    lines += ["", "## Ramp queue length over time (mean across starts)", "",
              "| t (s) | SUMO | " + " | ".join(wm_names) + " |",
              "|---|---|" + "---|" * len(wm_names)]
    for label in results["queue_targets"]:
        sumo_vals = [state["sumo"]["queue"]["ramp_queue_at"].get(label)
                     for state in states]
        row = [_fmt(_s._mean(sumo_vals))]
        for wm in wm_names:
            wm_vals = [state["wm_blocks"][wm]["wm"]["queue"]["ramp_queue_at"].get(label)
                       for state in states]
            row.append(_fmt(_s._mean(wm_vals)))
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    flagged_any = {wm: results["summary"][wm]["flagged"] for wm in wm_names}
    lines += ["", "## Interpretation", ""]
    for wm in wm_names:
        flags = flagged_any[wm]
        if flags:
            pretty = "; ".join(rows[k]["name"] for k in flags)
            lines.append(f"- `{wm}` falls outside the SUMO 95 % band on: {pretty}.")
        else:
            lines.append(f"- `{wm}` stays inside the SUMO 95 % band on every fleet "
                         "metric with a computable band.")
    sumo_rate = rows["rates.ramp_merges_per_min"]["sumo"]
    zipper = rows["zipper.same_stream_share"]["sumo"]
    if sumo_rate:
        lines.append(
            f"- SUMO baseline: {sumo_rate['mean']:.2f} ramp merges/min "
            f"(band {_fmt(sumo_rate['lo'])}-{_fmt(sumo_rate['hi'])}); "
            f"mainline {_fmt(rows['rates.mainline_passages_per_min']['sumo']['mean'])} "
            "passages/min.")
    if zipper:
        lines.append(
            f"- SUMO zipper: same-stream successive share {zipper['mean']:.2f} "
            f"(band {_fmt(zipper['lo'])}-{_fmt(zipper['hi'])}); 0.0 would be perfect "
            "alternation, ~0.5 random, 1.0 one stream dominating.")
    lines += [
        "",
        "## Proposed C11 gate (data-derived from this run's SUMO bands)", "",
        "A world model passes **C11 queue-discharge/zipper** on this fixed TRAIN-val "
        "selection when its fleet mean stays inside the SUMO episode-bootstrap 95 % band "
        "on ALL of: served ramp merges/min, mainline passages/min, lane-1 discharge "
        "headway p50, ramp queue mean, never-served ramp share, and same-stream "
        "successive share. The numeric bands are the SUMO columns of the fleet table "
        "above (this run: "
        + "; ".join(
            f"{rows[k]['name']}: [{_fmt(rows[k]['sumo']['lo'])}, "
            f"{_fmt(rows[k]['sumo']['hi'])}]"
            for k in ("rates.ramp_merges_per_min", "rates.mainline_passages_per_min",
                      "headways.discharge_lane1.p50", "queue.ramp_queue_mean",
                      "dwell.never_served_share", "zipper.same_stream_share")
            if rows[k]["sumo"]) + ").",
        "",
        "## Commands", "",
        "cpuN (this run):", "",
        "```sh", results["command_cpun"].strip(), "```", "",
        f"Local reproduction: `{results['command_local']}`", "",
        "## Code", "",
        "- `scripts/c4/wm_discharge_check.py` (this tool); unit test "
          "`tests/unit/test_wm_discharge_check.py`.",
        "- worktree commit: `COMMIT_PENDING` (patched after commit).",
        "",
        "## Notes", "",
        "- Same-vehicle-set rule: per the fleet note (delaydisc/DELAY_DISCREPANCY.md) the "
          "recorded streams carry only the t0 roster, so both WM and SUMO metrics are "
          "computed on the t0 roster; WM entrant slots only act as traffic.",
        "- A 'merge' is the lane-2 -> lane-1 lane change (insertion); a 'passage' crosses "
          "the merge-zone exit s = 75.44 m; the zipper sequence is the lane-1 crossings, "
          "filtered to instants where BOTH the ramp queue (lane-2 count) and the unserved "
          "mainline queue (mainline-origin vehicles upstream of the exit) are non-empty.",
        "- Never-served ramp cars are right-censored at the 300 s horizon; their share is "
          "reported next to the served dwell quantiles.",
        "- Insertion-follower braking is the nearest lane-1 roster follower's minimum "
          f"forward acceleration within [t-1 s, t+6 s] of the insertion "
          "(window is the metric definition, not a behavioural threshold).",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
