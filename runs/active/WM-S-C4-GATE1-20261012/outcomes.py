"""Gate 1 outcome metrics on frame streams (recorded SUMO trajectories or WM rollouts).

All outcomes are HDV-focused (CAVs are teacher-forced, so they carry no signal),
restricted to the t0 vehicle roster, at fixed horizons after t0:

* ``hdv_zone_delay_s``      cumulative vehicle-seconds of delay of HDVs inside the
                            merge zone ``0 <= s <= 75.44`` (free speed 27.78 m/s);
* ``hdv_zone_mean_speed_mps`` time-mean mean speed of HDVs inside the zone;
* ``hdv_hard_brake_events``  number of maximal runs of accel < -3.0 m/s^2 per HDV
                            (whole road, 0.5 s frames);
* ``throughput_past_merge``  distinct t0 vehicles crossing s = 75.44 downstream.
"""
from __future__ import annotations

from typing import Mapping, Sequence

ZONE_S0_M = 0.0
ZONE_S1_M = 75.44
FREE_SPEED_MPS = 27.78
HARD_BRAKE_MPS2 = -3.0
DT_S = 0.5
HORIZONS_S = (10.0, 30.0, 60.0)


def roster(frame: Mapping) -> dict[str, tuple[float, int, float, bool]]:
    """id -> (s, lane, v, is_cav) of the t0 snapshot (the analysis roster)."""
    return {str(v["id"]): (float(v["s"]), int(v.get("lane", -1)), float(v["v"]),
                           bool(v.get("is_cav", False)))
            for v in frame["vehicles"] if v.get("present", True)}


def _window(frames: Sequence[Mapping], t0: float, horizon: float) -> list[Mapping]:
    end = t0 + horizon
    return [f for f in frames if t0 - 1e-9 <= float(f["time_s"]) <= end + 1e-9]


def outcomes(frames: Sequence[Mapping], *, t0_s: float, roster_map: Mapping[str, tuple],
             horizons: Sequence[float] = HORIZONS_S) -> dict[str, dict]:
    """Per-horizon outcome dict for one arm's frame stream."""
    out: dict[str, dict] = {}
    for horizon in horizons:
        window = _window(frames, t0_s, float(horizon))
        delay = 0.0
        speed_sum, speed_frames = 0.0, 0
        for frame in window:
            zspeeds = []
            for v in frame["vehicles"]:
                if not v.get("present", True):
                    continue
                vid = str(v["id"])
                info = roster_map.get(vid)
                if info is None or info[3]:
                    continue
                s = float(v["s"])
                if ZONE_S0_M <= s <= ZONE_S1_M:
                    zspeeds.append(float(v["v"]))
                    delay += max(0.0, FREE_SPEED_MPS - float(v["v"])) * DT_S
            if zspeeds:
                speed_sum += sum(zspeeds) / len(zspeeds)
                speed_frames += 1
        brakes = 0
        prev: dict[str, float] = {}
        braking: dict[str, bool] = {}
        for frame in window:
            cur = {str(v["id"]): float(v["v"]) for v in frame["vehicles"]
                   if v.get("present", True)}
            for vid, v in cur.items():
                info = roster_map.get(vid)
                if info is None or info[3] or vid not in prev:
                    braking.pop(vid, None)
                    continue
                hard = (v - prev[vid]) / DT_S < HARD_BRAKE_MPS2
                if hard and not braking.get(vid):
                    brakes += 1
                braking[vid] = hard
            prev = cur
        crossed: set[str] = set()
        previous: dict[str, float] = {}
        for frame in window:
            current = {str(v["id"]): float(v["s"]) for v in frame["vehicles"]
                       if v.get("present", True)}
            for vid, s in current.items():
                if vid in previous and previous[vid] < ZONE_S1_M <= s:
                    crossed.add(vid)
            previous = current
        out[f"{int(horizon)}"] = {
            "hdv_zone_delay_s": round(delay, 3),
            "hdv_zone_mean_speed_mps": (round(speed_sum / speed_frames, 4)
                                        if speed_frames else None),
            "hdv_hard_brake_events": brakes,
            "throughput_past_merge": len(crossed),
        }
    return out


def orient(metric: str, value: float) -> float:
    """Lower-is-better transform (speed/throughput are higher-is-better)."""
    return -value if metric in ("hdv_zone_mean_speed_mps", "throughput_past_merge") else value


METRICS = ("hdv_zone_delay_s", "hdv_zone_mean_speed_mps",
           "hdv_hard_brake_events", "throughput_past_merge")
