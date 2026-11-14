"""Identity-matched car and section metrics for recorded/predicted frame streams."""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable

SECTIONS = {"upstream": (-math.inf, 0.0), "merge": (0.0, 75.44), "downstream": (75.44, math.inf)}
HORIZONS_S = (5.0, 10.0, 30.0, 60.0, 120.0, 180.0, 300.0)
METRICS = ("mean_speed", "density", "stopped_vehicles", "merge_queue_length", "merge_flow", "total_delay")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def frame_at(frames: list[dict], t: float) -> dict | None:
    if not frames:
        return None
    return min(frames, key=lambda item: abs(float(item["time_s"]) - t))


def present_by_id(frame: dict) -> dict[str, dict]:
    return {str(row["id"]): row for row in frame.get("vehicles", []) if row.get("present", True)}


def _summary(errors: list[float]) -> dict:
    return {"median": (float(median(errors)) if errors else None), "p90": percentile(errors, 0.9)}


def _car_group(predicted: dict[str, dict], recorded: dict[str, dict], ids: Iterable[str]) -> dict:
    ids = list(ids)
    ds = [abs(float(predicted[i]["s"]) - float(recorded[i]["s"])) for i in ids]
    dv = [abs(float(predicted[i]["v"]) - float(recorded[i]["v"])) for i in ids]
    lane = [predicted[i].get("lane") == recorded[i].get("lane") for i in ids]
    return {
        "n_recorded": len(ids),
        "n_matched": len(ids),
        "matched_fraction": (len(ids) / len(recorded) if recorded else None),
        "abs_delta_s_m": _summary(ds),
        "abs_delta_v_mps": _summary(dv),
        "lane_agreement": (sum(lane) / len(lane) if lane else None),
    }


def car_metrics(predicted_frames: list[dict], recorded_frames: list[dict], *, t0_s: float = 0.0,
                horizons: tuple[float, ...] = HORIZONS_S) -> dict:
    """Compute car errors at fixed horizons, preserving HDV/CAV strata."""
    result = {}
    for horizon in horizons:
        p = present_by_id(frame_at(predicted_frames, t0_s + horizon))
        r = present_by_id(frame_at(recorded_frames, t0_s + horizon))
        matched = set(p) & set(r)
        strata = {}
        for name, is_cav in (("all", None), ("hdv", False), ("cav", True)):
            ids = [vid for vid in matched if is_cav is None or bool(r[vid].get("is_cav")) == is_cav]
            denom = [vid for vid in r if is_cav is None or bool(r[vid].get("is_cav")) == is_cav]
            group = _car_group(p, r, ids)
            group["n_recorded"] = len(denom)
            group["matched_fraction"] = (len(ids) / len(denom) if denom else None)
            strata[name] = group
        result[str(float(horizon))] = strata
    return result


def _section(s: float) -> str:
    if s < 0.0:
        return "upstream"
    if s <= 75.44:
        return "merge"
    return "downstream"


def _traffic_frame(frame: dict, *, free_speed_mps: float, section_lengths_m: dict, dt_s: float) -> dict:
    rows = present_by_id(frame)
    out = {}
    for name, _bounds in SECTIONS.items():
        section = [row for row in rows.values() if _section(float(row["s"])) == name]
        speeds = [float(row["v"]) for row in section]
        stopped = [row for row in section if float(row["v"]) < 0.5]
        if name == "merge" and stopped:
            queue = max(0.0, max(float(row["s"]) for row in stopped) -
                        min(float(row["s"]) for row in stopped))
        else:
            queue = 0.0
        delay = sum(max(0.0, free_speed_mps - v) / max(free_speed_mps, 1e-9) * dt_s
                    for v in speeds)
        out[name] = {
            "mean_speed": (sum(speeds) / len(speeds) if speeds else None),
            "density": (len(section) / (section_lengths_m[name] / 1000.0)
                         if section_lengths_m[name] > 0 else None),
            "stopped_vehicles": len(stopped),
            "merge_queue_length": queue if name == "merge" else 0.0,
            "merge_flow": 0.0,
            "total_delay": delay,
        }
    return out


def traffic_metrics(frames: list[dict], *, t0_s: float = 0.0, bin_s: float = 30.0,
                    free_speed_mps: float = 13.89, dt_s: float = 0.5,
                    section_lengths_m: dict | None = None) -> dict:
    """Aggregate identical frame operations for predicted and recorded streams.

    Density uses explicit section lengths. Queue length is the span of stopped
    vehicles in the merge; merge flow counts crossings of s_down=75.44.
    """
    lengths = {"upstream": 1000.0, "merge": 75.44, "downstream": 1000.0}
    if section_lengths_m:
        lengths.update({key: float(value) for key, value in section_lengths_m.items()})
    if not frames:
        return {}
    end = max(float(item["time_s"]) for item in frames)
    n_bins = max(1, int(math.ceil(max(0.0, end - t0_s) / bin_s)))
    result = {}
    for idx in range(n_bins):
        lo, hi = t0_s + idx * bin_s, t0_s + (idx + 1) * bin_s
        selected = [item for item in frames if lo <= float(item["time_s"]) < hi]
        per_frame = [_traffic_frame(item, free_speed_mps=free_speed_mps,
                                    section_lengths_m=lengths, dt_s=dt_s) for item in selected]
        block = {}
        for name in SECTIONS:
            block[name] = {}
            for metric in METRICS:
                values = [row[name][metric] for row in per_frame if row[name][metric] is not None]
                if not values:
                    block[name][metric] = None
                elif metric == "total_delay":
                    # v0.4a window_label is an all-vehicle accumulation over
                    # the window, unlike state-like speed/density summaries.
                    block[name][metric] = sum(values)
                else:
                    block[name][metric] = sum(values) / len(values)
        flow = 0
        for before, after in zip(frames, frames[1:]):
            if not (lo <= float(after["time_s"]) < hi):
                continue
            before_ids = {row["id"] for row in before["vehicles"] if row.get("present", True)}
            for row in after["vehicles"]:
                if (row.get("present", True) and row["id"] in before_ids and
                        float(row["s"]) >= 75.44):
                    prev = next((x for x in before["vehicles"] if x["id"] == row["id"]), None)
                    if prev is not None and float(prev["s"]) < 75.44:
                        flow += 1
        block["merge"]["merge_flow"] = flow / (bin_s / 60.0)
        result[str(idx)] = {"start_s": lo, "end_s": hi, "sections": block}
    return result


DOWNSTREAM_BOUNDARY_M = 75.44


def truncate_frames(frames: list[dict], *, t0_s: float, horizon_s: float | None) -> list[dict]:
    """Keep frames inside ``[t0_s, t0_s + horizon_s]`` (inclusive of the end)."""
    if horizon_s is None:
        return list(frames)
    end = float(t0_s) + float(horizon_s)
    return [f for f in frames if float(t0_s) - 1e-9 <= float(f["time_s"]) <= end + 1e-9]


def cumulative_delay(frames: list[dict], *, t0_s: float = 0.0, horizon_s: float | None = None,
                     bin_s: float = 30.0, free_speed_mps: float = 13.89, dt_s: float = 0.5,
                     section_lengths_m: dict | None = None) -> dict:
    """Sum the per-bin normalized delay (vehicle-seconds) up to ``horizon_s``.

    The result carries the three section totals plus their ``network`` sum, so a
    planner can compare candidate actions on a single network-level objective.
    Bins are 30 s aligned by default, so 60/120/300 s horizons never split a bin.
    """
    metrics = traffic_metrics(frames, t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed_mps,
                              dt_s=dt_s, section_lengths_m=section_lengths_m)
    totals = {name: 0.0 for name in SECTIONS}
    for key in sorted(metrics, key=lambda k: int(k)):
        start = float(metrics[key]["start_s"])
        if horizon_s is not None and start - float(t0_s) >= float(horizon_s) - 1e-9:
            continue
        for name in SECTIONS:
            value = metrics[key]["sections"][name].get("total_delay")
            if value:
                totals[name] += float(value)
    totals["network"] = sum(totals[name] for name in SECTIONS)
    return totals


def throughput(frames: list[dict], *, boundary_m: float = DOWNSTREAM_BOUNDARY_M) -> int:
    """Distinct vehicles crossing downstream (``s >= boundary_m``) in the stream.

    This is the merge discharge over the window: a candidate action that gets
    more vehicles past the merge is a higher-throughput action. A vehicle that
    first appears already downstream is not counted as an exit.
    """
    crossed: set[str] = set()
    previous: dict[str, float] = {}
    for frame in frames:
        current = {str(row["id"]): float(row["s"]) for row in frame.get("vehicles", [])
                   if row.get("present", True)}
        for vid, s in current.items():
            if vid in previous and previous[vid] < boundary_m <= s:
                crossed.add(vid)
        previous = current
    return len(crossed)


def paired_difference(predicted: dict, recorded: dict) -> dict:
    out = {}
    for key, pred_bin in predicted.items():
        rec_bin = recorded.get(key, {})
        out[key] = {"start_s": pred_bin.get("start_s"), "end_s": pred_bin.get("end_s"),
                    "sections": {}}
        for section, values in pred_bin.get("sections", {}).items():
            out[key]["sections"][section] = {}
            for metric, value in values.items():
                other = rec_bin.get("sections", {}).get(section, {}).get(metric)
                out[key]["sections"][section][metric] = (
                    None if value is None or other is None else value - other)
    return out
