"""C1-C6 WM-vs-SUMO fidelity suite for fixed TRAIN-val starts (owner 10-20).

One many-core CLI that rolls the WM from recorded TRAIN-val starts (entries queue, ``--k``
samples) and compares the predicted stream with the recorded SUMO stream on six registered
checks:

* **C1 native CAV behaviour** - acceleration distribution, hard-brake share (``a < -2.5``),
  jerk, and the gap / time headway to the same-lane leader of the t0 native CAVs.
* **C2 TTC / near misses** - TTC distribution and near-miss rates (``< 3 s``, ``< 1.5 s``) over
  all roster leader-follower pairs and over CAV-follower pairs.
* **C3 road-end exits** - vehicles beyond the SUMO road end (553.33 m) that are still present,
  plus a static inspection of the rollout code path (does the predictor remove them?).
* **C4 lane share / speed** - lane 0 / 1 HDV share and per-lane mean speed at 0/60/120/300 s.
* **C5 free-flow speed** - per-lane speed distribution of vehicles with no leader within 100 m.
* **C6 wave amplitude** - for leader decel events ``a < -1.5 m/s^2``, the speed drop of
  followers 1..5 (own drop and ratio to the leader's drop).

The SUMO side of the recorded delay-check stream tracks the t0 roster only (no post-t0
entrants), so every per-vehicle check restricts BOTH sides to the t0 roster.  C3 deliberately
uses all WM vehicles (entrants included), because that is where the missing boundary shows up.

Output is a JSON document plus a markdown report with a PASS/FLAG verdict per check.  The
default tolerance is a 25 % relative deviation (with per-metric floors); C3 flags on any
vehicle beyond the road end.

    python scripts/c4/wm_fidelity_suite.py \\
        --wm runs/active/C4-V7FT-20261017/v5m_ref/wm_s_best_ap.pt \\
        --states delay_states.json --select 2 --save-selection fidelity_selection.json \\
        --horizons 300 --k 1 --targets 0,60,120,300 --wave-window-s 10 \\
        --entries queue --device cpu --workers 2 \\
        --out fidelity_suite.json --md FIDELITY_SUITE.md

Deterministic selection: the same input always yields the same starts (evenly spaced across
the sorted ``(world, key, source_arm, t0_s)`` keys).  Torch / wms are imported lazily by the
predictor factory; the metric helpers import and run without torch.

Refuses any path carrying a restricted token (sealed/holdout/heldout/cal_split/chunk5/j4b/inc1).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shlex
import sys
from pathlib import Path

SCHEMA = "c4_wm_fidelity_suite/1"
SELECTION_SCHEMA = "c4_wm_fidelity_suite/selection/1"
DEFAULT_HORIZON_S = 300.0
DEFAULT_K = 1
DEFAULT_TARGETS = (0.0, 60.0, 120.0, 300.0)
DEFAULT_WAVE_WINDOW_S = 10.0
DEFAULT_SELECT = 24
TOLERANCE = 0.25
CPUN_COMMAND = r"""cd ~/c4
OMP_NUM_THREADS=1 ~/miniconda3/bin/python scripts/c4/wm_fidelity_suite.py \
  --wm ~/c4/delay_check/wm_s_best_ap.pt \
  --selection ~/c4/delay_check/sel24_states.json \
  --horizon 300 --targets 0,60,120,300 --wave-window-s 10 \
  --k 2 --entries queue --device cpu --workers 3 \
  --out ~/c4/fidelity_suite/fidelity_suite.json \
  --md ~/c4/fidelity_suite/FIDELITY_SUITE.md"""
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")

ROAD_END_M = 553.33
DEFAULT_LENGTH_M = 5.0
HARD_BRAKE_MPS2 = -2.5
WAVE_EVENT_MPS2 = -1.5
TTC_NEAR_S = 3.0
TTC_CRITICAL_S = 1.5
FREE_FLOW_GAP_M = 100.0
LANE_IDS = (0, 1, 2)

#: Absolute floors under which a ratio is dominated by noise (metric -> floor in metric units).
FLOORS = {
    "accel_mean": 0.25, "accel_p50": 0.25, "accel_p95": 0.25, "accel_min": 0.25,
    "hard_brake_share": 0.01, "jerk_mean": 0.25, "jerk_median": 0.25, "jerk_p95_abs": 0.5,
    "gap_median": 1.0, "gap_p10": 1.0, "thw_median": 0.25, "speed_mean": 1.0,
    "ttc_p05": 0.25, "ttc_p10": 0.25, "ttc_p50": 0.5, "ttc_median": 0.5,
    "near_miss_3_rate": 0.005, "near_miss_15_rate": 0.005,
    "lane0_hdv_share": 0.05, "lane1_hdv_share": 0.05, "ramp_hdv_share": 0.05,
    "lane0_speed": 1.0, "lane1_speed": 1.0, "lane2_speed": 1.0,
    "ff_mean": 1.0, "ff_median": 1.0, "ff_p90": 1.0,
    "leader_drop_median": 0.25, "drop": 0.1, "ratio": 0.05,
}

CHECK_NAMES = {
    "C1": "native CAV behaviour",
    "C2": "TTC distribution and near-miss rate",
    "C3": "road-end exits",
    "C4": "lane share over time",
    "C5": "free-flow speed distribution",
    "C6": "wave amplitude along chains",
}

_WORKER: dict = {}


# --------------------------------------------------------------------------- imports / guards
def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _delay_module():
    """The registered delay-check helpers (selection, loader, predictor factory)."""
    path = Path(__file__).resolve().parent / "wm_delay_check.py"
    spec = importlib.util.spec_from_file_location("wm_fidelity_delay_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refuse_restricted(path) -> str:
    """Return ``str(path)`` unless it carries a restricted token (then raise ``SystemExit``)."""
    text = str(path)
    if any(token in text.replace("\\", "/").lower() for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {text}")
    return text


def parse_list(text, cast=float) -> tuple:
    if isinstance(text, (list, tuple)):
        return tuple(cast(v) for v in text)
    parts = str(text).replace(";", ",").split(",")
    return tuple(cast(part) for part in parts if part.strip())


# --------------------------------------------------------------------------- small stats
def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values) -> float | None:
    vals = [float(v) for v in values if _finite(v)]
    return (sum(vals) / len(vals)) if vals else None


def _median(values) -> float | None:
    vals = sorted(float(v) for v in values if _finite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _pct(values, q: float) -> float | None:
    """Linear-interpolated percentile (``q`` in [0, 100]); ``None`` for no values."""
    vals = sorted(float(v) for v in values if _finite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * max(0.0, min(100.0, float(q))) / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _share(count: int, total: int) -> float:
    return (float(count) / float(total)) if total > 0 else 0.0


def _hlabel(value: float) -> str:
    return str(int(round(float(value))))


def _frame_at(frames, t: float, max_gap_s: float = 0.75) -> dict | None:
    """Nearest frame within ``max_gap_s`` of ``t`` (model / record grid is 0.5 s)."""
    if not frames:
        return None
    best = min(frames, key=lambda f: abs(float(f.get("time_s", 0.0)) - float(t)))
    return best if abs(float(best.get("time_s", 0.0)) - float(t)) <= max_gap_s else None


def _present(frame, ids=None) -> dict[str, dict]:
    out = {}
    for row in (frame or {}).get("vehicles", []):
        if not row.get("present", True):
            continue
        vid = str(row.get("id", ""))
        if ids is not None and vid not in ids:
            continue
        out[vid] = row
    return out


def _t0_roster(record: dict) -> tuple[dict[str, dict], set[str], set[str]]:
    """The t0 frame, the roster ids and the native-CAV ids of one state record."""
    state = record.get("initial_state") or record.get("state") or {}
    rows = [row for row in state.get("vehicles", []) if row.get("present", True)]
    t0 = float(state.get("time_s", record.get("t0_s", 0.0)))
    roster = {str(row["id"]) for row in rows}
    cavs = {str(row["id"]) for row in rows if row.get("is_cav")}
    return {"time_s": t0, "vehicles": rows}, roster, cavs


def _cav_ids_from_frame(frame: dict) -> set[str]:
    return {str(row["id"]) for row in (frame or {}).get("vehicles", [])
            if row.get("present", True) and row.get("is_cav")}


# --------------------------------------------------------------------------- kinematics
def _within(frames, t0: float, horizon: float, include_t0: bool = False) -> list[dict]:
    lo = t0 if include_t0 else t0
    hi = t0 + float(horizon)
    out = []
    for frame in frames:
        t = float(frame.get("time_s", 0.0))
        if t < lo - 1e-6 or t > hi + 1e-6:
            continue
        if not include_t0 and abs(t - t0) <= 1e-6:
            continue
        out.append(frame)
    return out


def _tracks(frames, ids) -> dict[str, list[tuple[float, dict]]]:
    tracks: dict[str, list[tuple[float, dict]]] = {vid: [] for vid in ids}
    for frame in frames:
        t = float(frame.get("time_s", 0.0))
        for vid, row in _present(frame, ids).items():
            tracks.setdefault(vid, []).append((t, row))
    for vid in tracks:
        tracks[vid].sort(key=lambda item: item[0])
    return tracks


def _accel_series(tracks, max_dt_s: float = 0.75) -> dict[str, list[dict]]:
    """Per-vehicle ``{t, a, jerk, lane}`` at 0.5 s intervals (backward differences)."""
    out: dict[str, list[dict]] = {}
    for vid, seq in tracks.items():
        samples: list[dict] = []
        last_a = last_end = None
        for index in range(1, len(seq)):
            t0, row0 = seq[index - 1]
            t1, row1 = seq[index]
            dt = t1 - t0
            if not (0.0 < dt <= max_dt_s):
                last_a = last_end = None
                continue
            accel = (float(row1.get("v", 0.0)) - float(row0.get("v", 0.0))) / dt
            jerk = ((accel - last_a) / dt
                    if last_a is not None and last_end is not None
                    and abs(t0 - last_end) <= 1e-6 else None)
            samples.append({"t": t1, "a": accel, "jerk": jerk,
                            "lane": int(row1.get("lane", -1))})
            last_a, last_end = accel, t1
        out[vid] = samples
    return out


def _leader_map(frame, ids) -> dict[str, tuple[dict, float]]:
    """Follower id -> ``(leader row, gap m)`` for same-lane roster vehicles."""
    rows = _present(frame, ids)
    by_lane: dict[int, list[dict]] = {}
    for row in rows.values():
        lane = int(row.get("lane", -1))
        if lane < 0:
            continue
        by_lane.setdefault(lane, []).append(row)
    out: dict[str, tuple[dict, float]] = {}
    for lane_rows in by_lane.values():
        lane_rows.sort(key=lambda row: float(row.get("s", 0.0)))
        for index in range(len(lane_rows) - 1):
            follower = lane_rows[index]
            leader = lane_rows[index + 1]
            length = float(leader.get("length", DEFAULT_LENGTH_M))
            gap = float(leader.get("s", 0.0)) - length - float(follower.get("s", 0.0))
            out[str(follower["id"])] = (leader, gap)
    return out


def _pairs(frame, ids) -> list[dict]:
    """Adjacent leader-follower pairs on each lane (leader has the larger ``s``)."""
    rows = _present(frame, ids)
    by_lane: dict[int, list[dict]] = {}
    for row in rows.values():
        lane = int(row.get("lane", -1))
        if lane < 0:
            continue
        by_lane.setdefault(lane, []).append(row)
    out: list[dict] = []
    for lane_rows in by_lane.values():
        lane_rows.sort(key=lambda row: float(row.get("s", 0.0)))
        for index in range(len(lane_rows) - 1):
            follower = lane_rows[index]
            leader = lane_rows[index + 1]
            length = float(leader.get("length", DEFAULT_LENGTH_M))
            out.append({"follower": follower, "leader": leader,
                        "gap": float(leader.get("s", 0.0)) - length
                        - float(follower.get("s", 0.0))})
    return out


def _future_min(frames, vid: str, t: float, window: float) -> float | None:
    vals = []
    for frame in frames:
        ft = float(frame.get("time_s", 0.0))
        if ft < t - 1e-6 or ft > t + float(window) + 1e-6:
            continue
        row = _present(frame, {vid}).get(vid)
        if row is not None:
            vals.append(float(row.get("v", 0.0)))
    return min(vals) if vals else None


# --------------------------------------------------------------------------- C1
def cav_behaviour(frames, t0: float, horizon: float, roster, cavs) -> dict:
    """Native-CAV acceleration / jerk / leader-gap / time-headway summaries."""
    tracks = _tracks(frames, cavs)
    series = _accel_series(tracks)
    accels, jerks = [], []
    for vid in cavs:
        for sample in series.get(vid, []):
            if not (t0 + 1e-6 < sample["t"] <= t0 + horizon + 1e-6):
                continue
            accels.append(sample["a"])
            if _finite(sample["jerk"]):
                jerks.append(sample["jerk"])
    gaps, headways, speeds = [], [], []
    for frame in _within(frames, t0, horizon):
        leaders = _leader_map(frame, roster)
        for vid, row in _present(frame, cavs).items():
            speeds.append(float(row.get("v", 0.0)))
            if vid in leaders:
                gap = max(0.0, leaders[vid][1])
                gaps.append(gap)
                headways.append(gap / max(float(row.get("v", 0.0)), 0.5))
    hard = sum(1 for a in accels if a < HARD_BRAKE_MPS2)
    return {
        "n_accel": len(accels),
        "accel_mean": _mean(accels),
        "accel_p05": _pct(accels, 5),
        "accel_p50": _pct(accels, 50),
        "accel_p95": _pct(accels, 95),
        "accel_min": (min(accels) if accels else None),
        "hard_brake_share": _share(hard, len(accels)),
        "jerk_mean": _mean(jerks),
        "jerk_median": _median(jerks),
        "jerk_p95_abs": _pct([abs(j) for j in jerks], 95),
        "gap_median": _median(gaps),
        "gap_p10": _pct(gaps, 10),
        "thw_median": _median(headways),
        "speed_mean": _mean(speeds),
        "n_gap_samples": len(gaps),
    }


# --------------------------------------------------------------------------- C2
def _ttc_block(frames, t0: float, horizon: float, roster, cav_only: bool,
               cavs) -> dict:
    ttc, near3, near15, pair_steps = [], 0, 0, 0
    for frame in _within(frames, t0, horizon):
        for pair in _pairs(frame, roster):
            follower = pair["follower"]
            if cav_only and str(follower["id"]) not in cavs:
                continue
            gap = pair["gap"]
            if gap < 0.0:
                continue
            pair_steps += 1
            dv = float(follower.get("v", 0.0)) - float(pair["leader"].get("v", 0.0))
            if dv <= 1e-6:
                continue
            value = gap / dv
            ttc.append(value)
            near3 += int(value < TTC_NEAR_S)
            near15 += int(value < TTC_CRITICAL_S)
    return {
        "n_pair_steps": pair_steps,
        "n_finite_ttc": len(ttc),
        "ttc_p05": _pct(ttc, 5),
        "ttc_p10": _pct(ttc, 10),
        "ttc_p50": _pct(ttc, 50),
        "ttc_median": _median(ttc),
        "near_miss_3_rate": _share(near3, pair_steps),
        "near_miss_15_rate": _share(near15, pair_steps),
        "min_ttc": (min(ttc) if ttc else None),
    }


def ttc_distribution(frames, t0: float, horizon: float, roster, cavs) -> dict:
    return {
        "all_pairs": _ttc_block(frames, t0, horizon, roster, False, cavs),
        "cav_follower_pairs": _ttc_block(frames, t0, horizon, roster, True, cavs),
    }


# --------------------------------------------------------------------------- C3
def road_end_exits(frames, t0: float, horizon: float, road_end_m=ROAD_END_M) -> dict:
    """Vehicles whose ``s`` is beyond the road end while they are still present."""
    counts: dict[str, int] = {}
    steps = 0
    total_steps = 0
    dt_values = []
    last_t = None
    for frame in _within(frames, t0, horizon, include_t0=True):
        t = float(frame.get("time_s", 0.0))
        if last_t is not None:
            dt_values.append(t - last_t)
        last_t = t
        rows = _present(frame)
        total_steps += len(rows)
        for vid, row in rows.items():
            if float(row.get("s", 0.0)) > float(road_end_m):
                counts[vid] = counts.get(vid, 0) + 1
                steps += 1
    dt = _median(dt_values) or 0.5
    dwells = [n * dt for n in counts.values()]
    return {
        "n_vehicles_beyond": len(counts),
        "beyond_vehicle_steps": steps,
        "beyond_vehicle_seconds": steps * dt,
        "share_of_vehicle_steps": _share(steps, total_steps),
        "mean_dwell_s": _mean(dwells),
        "max_dwell_s": (max(dwells) if dwells else 0.0),
        "n_vehicle_steps": total_steps,
        "dt_s": dt,
        "top_ids": sorted(counts, key=lambda key: counts[key], reverse=True)[:8],
    }


def road_end_code_path(road_end_m=ROAD_END_M) -> dict:
    """Inspect the rollout code: does the predictor remove vehicles at the road end?

    The model step gates presence removal on a state field ``exit_s_m``.  If the predictor
    never supplies it, no road-end removal happens and an over-running vehicle stays present.
    """
    repo = Path(__file__).resolve().parents[2]
    model_path = repo / "src" / "wms" / "model" / "wms.py"
    predictor_path = repo / "src" / "wms" / "eval" / "predictor.py"
    model_text = model_path.read_text(encoding="utf-8") if model_path.exists() else ""
    predictor_text = predictor_path.read_text(encoding="utf-8") if predictor_path.exists() else ""
    model_gate = "next_s <= exit_s" in model_text
    predictor_sets = "exit_s_m" in predictor_text
    removes = bool(model_gate and predictor_sets)
    if removes:
        finding = (
            "Rollouts DO remove vehicles at the road end: wms.model.wms.step gates presence "
            "on state['exit_s_m'] and WmsPredictor.predict_states supplies it."
        )
    else:
        finding = (
            "Rollouts do NOT remove vehicles at the road end. wms.model.wms.step gates removal "
            "on state['exit_s_m'] (src/wms/model/wms.py: `next_present = next_present & "
            "(next_s <= exit_s)`), but WmsPredictor.predict_states (src/wms/eval/predictor.py) "
            "never puts exit_s_m into the initial state. Only the recorded presence sequence "
            "(roster exits) and the entry masks remove slots, so an over-running WM vehicle "
            f"stays present past {road_end_m:.2f} m."
        )
    return {
        "road_end_m": float(road_end_m),
        "model_has_exit_gate": model_gate,
        "predictor_sets_exit_s": predictor_sets,
        "rollouts_remove_at_road_end": removes,
        "finding": finding,
    }


# --------------------------------------------------------------------------- C4
def lane_share_speed(frames, t0: float, horizon: float, roster, targets) -> dict:
    out = {}
    for offset in targets:
        offset = float(offset)
        if offset > float(horizon) + 1e-6:
            continue
        frame = _frame_at(frames, t0 + offset)
        if frame is None:
            continue
        rows = _present(frame, roster)
        hdvs = [row for row in rows.values() if not row.get("is_cav")]
        hdv_lane = {lane: 0 for lane in LANE_IDS}
        for row in hdvs:
            lane = int(row.get("lane", -1))
            if lane in hdv_lane:
                hdv_lane[lane] += 1
        mainline = hdv_lane[0] + hdv_lane[1]
        lane_speed = {}
        for lane in LANE_IDS:
            lane_rows = [row for row in rows.values() if int(row.get("lane", -1)) == lane]
            lane_speed[lane] = _mean([float(row.get("v", 0.0)) for row in lane_rows])
        out[_hlabel(offset)] = {
            "lane0_hdv_share": _share(hdv_lane[0], mainline),
            "lane1_hdv_share": _share(hdv_lane[1], mainline),
            "ramp_hdv_share": _share(hdv_lane[2], len(hdvs)),
            "n_hdv": len(hdvs),
            "lane0_speed": lane_speed[0],
            "lane1_speed": lane_speed[1],
            "lane2_speed": lane_speed[2],
        }
    return out


# --------------------------------------------------------------------------- C5
def free_flow_speed(frames, t0: float, horizon: float, roster,
                    gap_m=FREE_FLOW_GAP_M) -> dict:
    vals: dict[int, list[float]] = {lane: [] for lane in LANE_IDS}
    for frame in _within(frames, t0, horizon):
        leaders = _leader_map(frame, roster)
        for vid, row in _present(frame, roster).items():
            lane = int(row.get("lane", -1))
            if lane not in vals:
                continue
            gap = leaders[vid][1] if vid in leaders else None
            if gap is None or gap > float(gap_m):
                vals[lane].append(float(row.get("v", 0.0)))
    out = {}
    for lane in LANE_IDS:
        values = vals[lane]
        out[str(lane)] = {
            "n": len(values),
            "ff_mean": _mean(values),
            "ff_median": _median(values),
            "ff_p90": _pct(values, 90),
        }
    return out


# --------------------------------------------------------------------------- C6
def _onsets(series: dict[str, list[dict]], t0: float, horizon: float,
            threshold: float) -> list[tuple[str, dict]]:
    events = []
    for vid, samples in series.items():
        previous = None
        for sample in samples:
            if not (t0 + 1e-6 < sample["t"] <= t0 + horizon + 1e-6):
                previous = sample["a"]
                continue
            if sample["a"] < threshold and (previous is None or previous >= threshold):
                events.append((vid, sample))
            previous = sample["a"]
    events.sort(key=lambda item: item[1]["t"])
    return events


def wave_amplitude(frames, t0: float, horizon: float, roster, cavs,
                   window_s: float, road_end_m=ROAD_END_M) -> dict:
    tracks = _tracks(frames, roster)
    series = _accel_series(tracks)
    events = _onsets(series, t0, horizon - float(window_s), WAVE_EVENT_MPS2)
    drops: list[list[float]] = [[] for _ in range(5)]
    ratios: list[list[float]] = [[] for _ in range(5)]
    leader_amps, with_follower = [], 0
    for vid, sample in events:
        t = float(sample["t"])
        frame = _frame_at(frames, t)
        if frame is None:
            continue
        rows = _present(frame, roster)
        leader = rows.get(vid)
        if leader is None:
            continue
        lane = int(leader.get("lane", -1))
        if lane < 0:
            continue
        followers = [row for row in rows.values()
                     if int(row.get("lane", -1)) == lane
                     and float(row.get("s", 0.0)) < float(leader.get("s", 0.0))]
        followers.sort(key=lambda row: float(row.get("s", 0.0)), reverse=True)
        if not followers:
            continue
        v_leader = float(leader.get("v", 0.0))
        min_leader = _future_min(frames, vid, t, window_s)
        if min_leader is None:
            continue
        lead_drop = max(0.0, v_leader - min_leader)
        leader_amps.append(lead_drop)
        with_follower += 1
        for rank, follower in enumerate(followers[:5]):
            fid = str(follower["id"])
            min_f = _future_min(frames, fid, t, window_s)
            if min_f is None:
                continue
            drop = max(0.0, float(follower.get("v", 0.0)) - min_f)
            drops[rank].append(drop)
            ratios[rank].append(drop / max(lead_drop, 0.1))
    return {
        "n_events": len(events),
        "n_events_with_followers": with_follower,
        "leader_drop_median": _median(leader_amps),
        "drops": [(_median(vals) if vals else None) for vals in drops],
        "ratios": [(_median(vals) if vals else None) for vals in ratios],
        "follower_n": [len(vals) for vals in drops],
        "window_s": float(window_s),
        "event_threshold_mps2": WAVE_EVENT_MPS2,
    }


# --------------------------------------------------------------------------- per-state driver
def _metrics(frames, t0, horizon, roster, cavs, targets, wave_window_s) -> dict:
    return {
        "C1": cav_behaviour(frames, t0, horizon, roster, cavs),
        "C2": ttc_distribution(frames, t0, horizon, roster, cavs),
        "C3": road_end_exits(frames, t0, horizon),
        "C4": lane_share_speed(frames, t0, horizon, roster, targets),
        "C5": free_flow_speed(frames, t0, horizon, roster),
        "C6": wave_amplitude(frames, t0, horizon, roster, cavs, wave_window_s),
    }


def _avg_numeric(items):
    """Recursively average numbers / numeric lists / dicts; ``None`` -> ``None``."""
    items = [item for item in items if item is not None]
    if not items:
        return None
    first = items[0]
    if isinstance(first, dict):
        return {key: _avg_numeric([item.get(key) for item in items]) for key in first}
    if isinstance(first, list):
        return [_avg_numeric([item[index] if index < len(item) else None for item in items])
                for index in range(len(first))]
    if isinstance(first, bool):
        return first
    if isinstance(first, (int, float)):
        vals = [float(item) for item in items if _finite(item)]
        return (sum(vals) / len(vals)) if vals else None
    return first


def evaluate_state(predictor, payload) -> dict:
    record, cfg = payload
    t0_frame, roster, cavs = _t0_roster(record)
    t0 = float(t0_frame["time_s"])
    arm = str(cfg["arm"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from wms.eval.normalize import recorded_for
    recorded_arms = record.get("recorded") or {}
    if arm not in recorded_arms and isinstance(recorded_arms, dict) and recorded_arms:
        arm = sorted(recorded_arms)[0]
    recorded = recorded_for(record, arm)
    targets = parse_list(cfg["targets"])
    horizon = float(cfg["horizon_s"])
    wave_window = float(cfg["wave_window_s"])
    sumo = _metrics(recorded, t0, horizon, roster, cavs, targets, wave_window)
    wm_samples = []
    for sample in range(max(1, int(cfg["k"]))):
        controllers = dict(record.get("cav_controllers") or {})
        controllers["arm"] = arm
        frames = predictor(record, controllers, horizon, sample)
        wm_samples.append(_metrics(frames, t0, horizon, roster, cavs, targets,
                                   wave_window))
    wm_avg = _avg_numeric(wm_samples)
    wm_avg["C3"] = {
        **_avg_numeric([block["C3"] for block in wm_samples]),
        "top_ids": wm_samples[0]["C3"].get("top_ids", []),
    }
    return {
        "world": str(record.get("world")),
        "key": record.get("key"),
        "source_arm": record.get("source_arm"),
        "source_set": record.get("source_set"),
        "t0_s": t0,
        "n_roster": len(roster),
        "n_cav": len(cavs),
        "n_samples": len(wm_samples),
        "arm": arm,
        "wm": wm_avg,
        "sumo": sumo,
        "errors": [],
    }


def _init_worker(wm, device, entries) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    _WORKER["predictor"] = _delay_module().PredictorFactory(
        wm, device=device, entries=entries)()


def _work_state(predictor, payload):
    predictor = predictor if predictor is not None else _WORKER["predictor"]
    return evaluate_state(predictor, payload)


def _map(payloads, wm, args) -> list[dict]:
    workers = max(1, int(args.workers))
    if workers <= 1:
        predictor = _delay_module().PredictorFactory(
            wm, device=args.device, entries=args.entries)()
        return [_work_state(predictor, payload) for payload in payloads]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(wm), args.device, args.entries)) as pool:
        futures = [pool.submit(_work_state, None, payload) for payload in payloads]
        return [future.result() for future in futures]


# --------------------------------------------------------------------------- verdict
def _deviation(wm, sumo, floor: float) -> float | None:
    if not _finite(wm) or not _finite(sumo):
        return None
    return abs(float(wm) - float(sumo)) / max(abs(float(sumo)), float(floor))


def _compare(block: str, metric: str, wm, sumo, floor: float | None = None) -> dict:
    floor = FLOORS.get(metric, 0.01) if floor is None else floor
    dev = _deviation(wm, sumo, floor)
    return {"check": block, "metric": metric, "wm": wm, "sumo": sumo,
            "floor": floor, "deviation": dev,
            "flagged": (dev is not None and dev > TOLERANCE)}


def _compare_block(prefix: str, wm: dict, sumo: dict, metrics) -> list[dict]:
    out = []
    for metric in metrics:
        out.append(_compare(prefix, metric, (wm or {}).get(metric),
                            (sumo or {}).get(metric)))
    return out


def _status(comparisons) -> str:
    return "FLAG" if any(row.get("flagged") for row in comparisons) else "PASS"


def _check_payloads(states) -> dict:
    """Average the per-state metric blocks and build the PASS/FLAG comparisons."""
    wm = _avg_numeric([state["wm"] for state in states]) or {}
    sumo = _avg_numeric([state["sumo"] for state in states]) or {}

    c1 = _compare_block("C1", wm.get("C1"), sumo.get("C1"),
                        ("accel_mean", "accel_p50", "accel_p95", "accel_min",
                         "hard_brake_share", "jerk_mean", "jerk_median", "jerk_p95_abs",
                         "gap_median", "gap_p10", "thw_median", "speed_mean"))
    c2 = []
    for pair_kind in ("all_pairs", "cav_follower_pairs"):
        c2 += _compare_block(f"C2/{pair_kind}", (wm.get("C2") or {}).get(pair_kind),
                             (sumo.get("C2") or {}).get(pair_kind),
                             ("ttc_p05", "ttc_p10", "ttc_p50", "ttc_median",
                              "near_miss_3_rate", "near_miss_15_rate"))
    wm3, sumo3 = wm.get("C3") or {}, sumo.get("C3") or {}
    c3 = [
        _compare("C3", "n_vehicles_beyond", wm3.get("n_vehicles_beyond"),
                 sumo3.get("n_vehicles_beyond"), floor=0.5),
        _compare("C3", "beyond_vehicle_seconds", wm3.get("beyond_vehicle_seconds"),
                 sumo3.get("beyond_vehicle_seconds"), floor=0.5),
    ]
    for row in c3:
        if (_finite(row["wm"]) and _finite(row["sumo"])
                and float(row["wm"]) > 0.0 and float(row["sumo"]) <= 0.0):
            row["flagged"] = True
            row["deviation"] = None
            row["note"] = "SUMO road-end count is 0; any WM overrun is a FLAG"

    c4 = []
    for label in sorted((wm.get("C4") or {}).keys(), key=lambda key: int(key)):
        c4 += _compare_block(
            f"C4@{label}", (wm.get("C4") or {}).get(label), (sumo.get("C4") or {}).get(label),
            ("lane0_hdv_share", "lane1_hdv_share", "ramp_hdv_share", "lane0_speed",
             "lane1_speed", "lane2_speed"))

    c5 = []
    for lane in map(str, LANE_IDS):
        wl, sl = (wm.get("C5") or {}).get(lane) or {}, (sumo.get("C5") or {}).get(lane) or {}
        if min(int(wl.get("n") or 0), int(sl.get("n") or 0)) < 20:
            continue
        c5 += _compare_block(f"C5/lane{lane}", wl, sl,
                             ("ff_mean", "ff_median", "ff_p90"))

    c6 = []
    w6, s6 = wm.get("C6") or {}, sumo.get("C6") or {}
    c6.append(_compare("C6", "leader_drop_median", w6.get("leader_drop_median"),
                       s6.get("leader_drop_median")))
    for rank in range(5):
        wn = (w6.get("follower_n") or [0] * 5)[rank]
        sn = (s6.get("follower_n") or [0] * 5)[rank]
        if min(int(wn or 0), int(sn or 0)) < 5:
            continue
        c6.append(_compare("C6", f"drop_rank{rank + 1}",
                           (w6.get("drops") or [None] * 5)[rank],
                           (s6.get("drops") or [None] * 5)[rank], floor=0.1))
        c6.append(_compare("C6", f"ratio_rank{rank + 1}",
                           (w6.get("ratios") or [None] * 5)[rank],
                           (s6.get("ratios") or [None] * 5)[rank], floor=0.05))

    definitions = {
        "C1": ("Native t0 CAVs: acceleration distribution, hard-brake share (a < -2.5 m/s^2), "
               "jerk, gap and time headway to the same-lane leader."),
        "C2": ("Leader-follower TTC (gap / closing speed) distribution and the share of "
               "pair-steps with TTC < 3 s and < 1.5 s; all roster pairs and CAV-follower pairs."),
        "C3": ("Vehicles present with s > 553.33 m (SUMO road end): unique vehicles, "
               "vehicle-seconds beyond the end and dwell; plus the rollout code-path finding."),
        "C4": ("HDV lane share (lane 0 / 1 of mainline HDVs, ramp share) and per-lane mean "
               "speed at 0/60/120/300 s."),
        "C5": ("Per-lane speed distribution of roster vehicles with no same-lane leader "
               "within 100 m."),
        "C6": ("For leader decel onsets a < -1.5 m/s^2: follower 1..5 own speed drop and its "
               "ratio to the leader's drop over the wave window."),
    }
    tolerances = {
        "C1": "<= 25 % relative deviation on every reported scalar (metric floors in JSON).",
        "C2": "<= 25 % relative deviation on TTC quantiles and near-miss rates.",
        "C3": "PASS only when no WM vehicle is present beyond the road end (SUMO = 0).",
        "C4": "<= 25 % relative deviation on lane shares (floor 0.05) and mean speeds (1 m/s).",
        "C5": "<= 25 % relative deviation on per-lane free-flow mean/median/p90 "
              "(lanes with >= 20 samples on both sides).",
        "C6": "<= 25 % relative deviation on follower drops and drop ratios "
              "(ranks with >= 5 events on both sides).",
    }
    return {
        "C1": {"name": CHECK_NAMES["C1"], "definition": definitions["C1"],
               "tolerance": tolerances["C1"], "status": _status(c1),
               "wm": wm.get("C1"), "sumo": sumo.get("C1"), "comparisons": c1},
        "C2": {"name": CHECK_NAMES["C2"], "definition": definitions["C2"],
               "tolerance": tolerances["C2"], "status": _status(c2),
               "wm": wm.get("C2"), "sumo": sumo.get("C2"), "comparisons": c2},
        "C3": {"name": CHECK_NAMES["C3"], "definition": definitions["C3"],
               "tolerance": tolerances["C3"], "status": _status(c3),
               "wm": wm.get("C3"), "sumo": sumo.get("C3"), "comparisons": c3},
        "C4": {"name": CHECK_NAMES["C4"], "definition": definitions["C4"],
               "tolerance": tolerances["C4"], "status": _status(c4),
               "wm": wm.get("C4"), "sumo": sumo.get("C4"), "comparisons": c4},
        "C5": {"name": CHECK_NAMES["C5"], "definition": definitions["C5"],
               "tolerance": tolerances["C5"], "status": _status(c5),
               "wm": wm.get("C5"), "sumo": sumo.get("C5"), "comparisons": c5},
        "C6": {"name": CHECK_NAMES["C6"], "definition": definitions["C6"],
               "tolerance": tolerances["C6"], "status": _status(c6),
               "wm": wm.get("C6"), "sumo": sumo.get("C6"), "comparisons": c6},
    }


# --------------------------------------------------------------------------- CLI / report
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", action="append", required=True,
                    help="WM checkpoint (repeatable; all are scored on the same starts)")
    ap.add_argument("--states", help="states JSON (or directory) with the recorded starts")
    ap.add_argument("--selection", help="fixed selection JSON; its states are used if present")
    ap.add_argument("--save-selection", help="write the deterministic selection here")
    ap.add_argument("--select", type=int, default=DEFAULT_SELECT,
                    help="number of starts selected evenly across the sorted keys")
    ap.add_argument("--horizon", type=float, default=DEFAULT_HORIZON_S,
                    help="rollout horizon in seconds (default 300)")
    ap.add_argument("--targets", default="0,60,120,300",
                    help="C4 lane-share times relative to t0")
    ap.add_argument("--wave-window-s", type=float, default=DEFAULT_WAVE_WINDOW_S,
                    help="C6 follower speed-drop window in seconds")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="WM samples per start")
    ap.add_argument("--arm", default="native",
                    help="recorded CAV-teacher arm (falls back to the record's first arm)")
    ap.add_argument("--entries", choices=("off", "on", "queue"), default="queue")
    ap.add_argument("--road-end-m", type=float, default=ROAD_END_M)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--md", help="optional markdown report path")
    return ap


def _resolve_starts(args):
    helpers = _delay_module()
    selection = Path(args.selection) if args.selection else None
    if selection is not None and selection.exists():
        return helpers.load_selection(selection, "states")
    if not args.states:
        raise SystemExit("--states is required when --selection is absent")
    starts = helpers.select_evenly(helpers.load_records(args.states), int(args.select))
    if not starts:
        raise SystemExit("no starts selected")
    if args.save_selection:
        helpers.write_selection(args.save_selection, starts, "states")
    return starts


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _ensure_src_on_path()
    for token in (args.out, args.md, args.states, args.selection, args.save_selection,
                  *args.wm):
        if token:
            refuse_restricted(token)
    targets = parse_list(args.targets)
    if not targets:
        raise SystemExit("--targets must not be empty")
    if args.horizon <= 0:
        raise SystemExit("--horizon must be positive")
    starts = _resolve_starts(args)
    cfg = {
        "arm": args.arm, "k": int(args.k), "horizon_s": float(args.horizon),
        "wave_window_s": float(args.wave_window_s), "targets": list(targets),
    }
    results = {
        "schema": SCHEMA,
        "wm": [str(wm) for wm in args.wm],
        "arm": args.arm,
        "k": int(args.k),
        "horizon_s": float(args.horizon),
        "targets": [_hlabel(t) for t in targets],
        "wave_window_s": float(args.wave_window_s),
        "entries": args.entries,
        "device": args.device,
        "workers": max(1, int(args.workers)),
        "tolerance": TOLERANCE,
        "road_end_m": float(args.road_end_m),
        "n_starts": len(starts),
        "command_local": "python " + shlex.join(sys.argv),
        "command_cpun": CPUN_COMMAND,
        "starts": [{"world": str(r.get("world")), "key": r.get("key"),
                    "source_set": r.get("source_set"), "t0_s": float(r.get("t0_s", 0.0))}
                   for r in starts],
        "code_path": road_end_code_path(args.road_end_m),
        "wms": {},
    }
    primary = None
    for wm in args.wm:
        payloads = [(record, cfg) for record in starts]
        states = _map(payloads, wm, args)
        checks = _check_payloads(states)
        name = Path(wm).name
        results["wms"][name] = {"wm": str(wm), "checks": checks, "states": states}
        if primary is None:
            primary = {"wm": name, "checks": checks}
    results["primary_wm"] = primary["wm"] if primary else None
    results["summary"] = {
        "PASS": [key for key, block in (primary or {}).get("checks", {}).items()
                 if block["status"] == "PASS"],
        "FLAG": [key for key, block in (primary or {}).get("checks", {}).items()
                 if block["status"] == "FLAG"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize(results), indent=2, sort_keys=True), encoding="utf-8")
    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(results), encoding="utf-8")
    print(json.dumps({
        "out": str(out), "md": (str(args.md) if args.md else None),
        "n_starts": len(starts), "k": int(args.k), "horizon_s": float(args.horizon),
        "summary": results["summary"],
        "c3": results["code_path"]["rollouts_remove_at_road_end"],
    }, indent=2, sort_keys=True))
    return 0


def _sanitize(value):
    """JSON-safe copy: non-finite floats -> None, tuples -> lists."""
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _md_table(rows, metric_cols) -> list[str]:
    header = "| metric | WM | SUMO | relative deviation | verdict |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        verdict = "**FLAG**" if row.get("flagged") else "ok"
        dev = row.get("deviation")
        lines.append(f"| {row['check']} / {row['metric']} | {_fmt(row.get('wm'))} | "
                     f"{_fmt(row.get('sumo'))} | "
                     f"{('n/a' if dev is None else f'{dev:.3f}')} | {verdict} |")
    return lines


def render_markdown(results: dict) -> str:
    lines = [
        "# WM-vs-SUMO fidelity suite (C1-C6)", "",
        f"- schema: `{results['schema']}`",
        f"- WM: `{results['primary_wm']}`; arm `{results['arm']}`; k={results['k']}; "
        f"horizon {_fmt(results['horizon_s'])} s; entries `{results['entries']}`",
        f"- {results['n_starts']} TRAIN-val starts; road end {results['road_end_m']:.2f} m; "
        f"workers {results['workers']}",
        f"- tolerance: <= {results['tolerance'] * 100:.0f} % relative deviation "
        "(per-metric absolute floors in the JSON); C3 flags on any overrun",
        "",
        "## Checks and verdicts", "",
        "| check | definition | tolerance | verdict |",
        "|---|---|---|---|",
    ]
    checks = results["wms"][results["primary_wm"]]["checks"]
    for key in ("C1", "C2", "C3", "C4", "C5", "C6"):
        block = checks[key]
        status = block["status"]
        shown = f"**{status}**" if status == "FLAG" else status
        lines.append(f"| {key} {block['name']} | {block['definition']} | "
                     f"{block['tolerance']} | {shown} |")
    lines += ["", "## C3 road-end code-path finding", "",
              f"**{results['code_path']['finding']}**", "",
              f"- model exit gate present: "
              f"{_fmt(results['code_path']['model_has_exit_gate'])}; "
              f"predictor supplies `exit_s_m`: "
              f"{_fmt(results['code_path']['predictor_sets_exit_s'])}; "
              f"rollouts remove at road end: "
              f"{_fmt(results['code_path']['rollouts_remove_at_road_end'])}", "",
              "## Per-state summary", "",
              "| world | set | arm | t0 s | roster | CAV | C3 WM veh beyond | "
              "C3 WM veh-s | C3 SUMO veh beyond |",
              "|---|---|---|---|---|---|---|---|---|"]
    for state in results["wms"][results["primary_wm"]]["states"]:
        wm3 = state["wm"]["C3"]
        sumo3 = state["sumo"]["C3"]
        lines.append(
            f"| {state['world']} | {state.get('source_set')} | {state.get('arm')} | "
            f"{_fmt(state['t0_s'])} | {state['n_roster']} | {state['n_cav']} | "
            f"{_fmt(wm3.get('n_vehicles_beyond'))} | "
            f"{_fmt(wm3.get('beyond_vehicle_seconds'))} | "
            f"{_fmt(sumo3.get('n_vehicles_beyond'))} |")
    lines.append("")

    for key in ("C1", "C2", "C3", "C4", "C5", "C6"):
        block = checks[key]
        lines += [f"## {key} - {block['name']} ({block['status']})", "",
                  f"{block['definition']}", "",
                  f"Tolerance: {block['tolerance']}", ""]
        if key in ("C3", "C6"):
            wm, sumo = block["wm"] or {}, block["sumo"] or {}
            lines += ["| metric | WM | SUMO |", "|---|---|---|"]
            for metric in sorted(set(wm) | set(sumo)):
                if isinstance(wm.get(metric), (dict, list)):
                    continue
                lines.append(f"| {metric} | {_fmt(wm.get(metric))} | "
                             f"{_fmt(sumo.get(metric))} |")
            if key == "C6":
                lines += ["", "| rank | WM drop | SUMO drop | WM ratio | SUMO ratio |",
                          "|---|---|---|---|---|"]
                for rank in range(5):
                    wd = (wm.get("drops") or [None] * 5)[rank]
                    sd = (sumo.get("drops") or [None] * 5)[rank]
                    wr = (wm.get("ratios") or [None] * 5)[rank]
                    sr = (sumo.get("ratios") or [None] * 5)[rank]
                    lines.append(f"| {rank + 1} | {_fmt(wd)} | {_fmt(sd)} | "
                                 f"{_fmt(wr)} | {_fmt(sr)} |")
            lines.append("")
        else:
            lines += _md_table(block["comparisons"], None)
            lines.append("")

    lines += ["## Command", "",
              "Local (this proof run; laptop, <= 3 workers):", "",
              "```sh", results.get("command_local", "").strip(), "```", "",
              "cpuN production command (24 starts, k=2, 300 s, 3 workers):", "",
              "```sh", results.get("command_cpun", "").strip(), "```", "",
              "## Notes", "",
              "- Per-vehicle metrics (C1, C2, C4-C6) restrict BOTH sides to the t0 roster, "
              "because the recorded delay-check stream carries no post-t0 entrants.",
              "- C3 deliberately uses all WM vehicles (entry slots included).",
              "- CAV motion is teacher-forced from the recorded arm, so C1's CAV acceleration "
              "and jerk are expected to match SUMO; the gap / time-headway part is the "
              "non-trivial part of C1.",
              "- C6 events are onset rows (a < -1.5 m/s^2 after a row at or above it), not "
              "every braking event-second.",
              ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
