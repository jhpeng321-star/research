"""Stage 2 final-generation contract: seeds, hidden demand, scenario, features.

Population: plateau-decline profile — mainline demand M drawn uniformly from
[2200, 2800] veh/h (integer, seeded from the seed number), held 0-1200 s;
three 100 s steps down to 1500 over 1200-1500 s; 1500 for 1500-1800 s; no
departures after 1800. Ramp 400 veh/h for 0-1800 s. Horizon: natural
clearance or 2400 s. Strata: p = 0.20 seeds 8001-8240; p = 0.30 seeds
8241-8360. The drawn demand is a hidden variable: recorded in the manifest,
never a feature. Route files live under scenarios/idm_stage2/.

Arms: D (frozen analytical damping); onset on D = first step with
merge-zone h < 0.15 for 10 consecutive seconds (no arm when no onset by
t = 900); WON_H300 / WON_H450 / WON_H650 = damping + sustained wall engaged
at t_on + 1, held 300/450/650 s, 5 s taper, tau restore (10 m/s, tau +3.0,
200-400 m mainline band, 0.3 m/s^2 clip). Exact prefix identity through
t_on. D is repeated on every 20th seed.

Features (repaired capture; full RSU vector) at t_on, t_on - 60 (when at or
above the capture floor), and the fixed reference t = 150: region counts /
mean speeds / stds, densities, mainline lane imbalance, CAV counts in band
and upstream, 30 s ramp arrivals, 30 s inflow / discharge and their
difference, 120 s backward differences (when history allows), per-lane
geometry with UNCLIPPED nearest-to-band-top (the 1c-2 censoring fix), plus
merge_edge_mean_speed and post_merge_mean_speed at 10/30/60 s lags (the
onset predictors). No M, no absolute time.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from experiments import REPO_ROOT
from experiments.idm.stage1c_release_contract import decline_flow_steps

ROLE = "stage2_generation"
SEEDS_P20 = tuple(range(8001, 8241))
SEEDS_P30 = tuple(range(8241, 8361))
ALL_SEEDS = SEEDS_P20 + SEEDS_P30

SCENARIO_ROOT = REPO_ROOT / "scenarios" / "idm_stage2"
TEMPLATE_ROUTE = REPO_ROOT / "scenarios" / "idm_transition" / "idm_m2200r400.rou.xml"
TEMPLATE_SCENE = REPO_ROOT / "scenarios" / "idm_transition" / "osm.dense.idm_m2200r400.sumocfg"

PLATEAU_END_S = 1200.0
TAIL_DEMAND_VPH = 1500
TAIL_END_S = 1800.0
RAMP_DEMAND_VPH = 400
HORIZON_S = 2400.0
CAV_PENETRATION_P20 = 0.20
CAV_PENETRATION_P30 = 0.30
ONSET_H_THRESHOLD = 0.15
ONSET_SUSTAIN_S = 10.0
ONSET_SEARCH_MAX_STEP = 900
HOLDS_S = (300.0, 450.0, 650.0)
TAPER_DURATION_S = 5.0
CAPTURE_LO_STEP = 30
CAPTURE_HI_STEP = 960  # covers t_on (<= 900) and t_on - 60 and the fixed 150
FIXED_REFERENCE_STEP = 150
LAGS_S = (10, 30, 60)
REPEAT_EVERY = 20

WALL_PARAMS: Dict[str, float] = {"d_lo": 200.0, "d_hi": 400.0, "wall_speed": 10.0,
                                 "wall_tau": 3.0, "a_max_dec": 0.3}


def demand_for_seed(seed: int) -> int:
    """Deterministic per-seed demand draw (uniform integers in [2200, 2800])."""
    rng = np.random.default_rng(int(seed))
    return int(rng.integers(2200, 2800 + 1))


def penetration_for_seed(seed: int) -> float:
    return CAV_PENETRATION_P20 if int(seed) in set(SEEDS_P20) else CAV_PENETRATION_P30


def strate_for_seed(seed: int) -> str:
    return "p20" if int(seed) in set(SEEDS_P20) else "p30"


def route_name_for_seed(seed: int, demand: int) -> str:
    return f"idm_m{demand}_seed{seed}.plateau_decline.rou.xml"


def scene_name_for_seed(seed: int, demand: int) -> str:
    return f"osm.dense.idm_m{demand}_seed{seed}.plateau_decline.sumocfg"


def _flow_block(begin: float, end: float, veh_per_h: int) -> str:
    return (
        f'  <flow id="mainline.{int(begin)}" type="hdv_mix" route="mainline_route" '
        f'begin="{int(begin)}" end="{int(end)}" vehsPerHour="{int(veh_per_h)}" '
        f'departLane="best" departSpeed="max"/>'
    )


def write_seed_scenario(seed: int, demand: int, root=None) -> str:
    """Write the per-seed plateau-decline route and scene; return scene name."""
    import re
    from pathlib import Path

    root = Path(root) if root is not None else SCENARIO_ROOT
    root.mkdir(parents=True, exist_ok=True)
    route_path = root / route_name_for_seed(seed, demand)
    route_src = TEMPLATE_ROUTE.read_text(encoding="utf-8")
    ramp_match = re.search(r'<flow id="ramp"[^>]*/>', route_src)
    ramp_flow = ramp_match.group(0) if ramp_match else ""
    ramp_flow = re.sub(r'vehsPerHour="[0-9]+"', f'vehsPerHour="{RAMP_DEMAND_VPH}"', ramp_flow)
    head = route_src.split("<flow")[0]
    route_out = (
        head
        + "\n".join(_flow_block(b, e, v) for b, e, v in decline_flow_steps(int(demand)))
        + "\n  "
        + ramp_flow
        + "\n</routes>\n"
    )
    # SUMO requires route flows sorted by begin time (unsorted files are
    # silently partially ignored - the Stage 1c ramp-flow bug).
    blocks = re.findall(r"<flow[^>]*/>", route_out)
    blocks.sort(key=lambda b: float(re.search(r'begin="([0-9.]+)"', b).group(1)))
    route_out = route_out.split("<flow")[0] + "\n".join(blocks) + "\n</routes>\n"
    route_path.write_text(route_out, encoding="utf-8")

    scene_path = root / scene_name_for_seed(seed, demand)
    scene_src = TEMPLATE_SCENE.read_text(encoding="utf-8")
    scene_src = re.sub(
        r'<route-files value="[^"]*"/>',
        f'<route-files value="{route_path.name}"/>',
        scene_src,
        count=1,
    )
    scene_path.write_text(scene_src, encoding="utf-8")
    return scene_path.name


def per_lane_unclipped(per_lane_distances: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per-lane CAV band summary with UNCLIPPED nearest-to-band-top.

    ``per_lane_distances`` maps lane id -> list of distances-to-band-top (m).
    nearest_to_band_top_unclipped = min(distance - 400) over the lane's CAVs
    (negative when the nearest CAV sits inside the band) - the 1c-2 fix for
    the censored max(0, ...) column.
    """
    out: Dict[str, Dict[str, float]] = {}
    for lane, distances in per_lane_distances.items():
        if not distances:
            continue
        band = [d for d in distances if 200.0 <= d <= 400.0]
        near = [d for d in distances if 400.0 < d <= 500.0]
        out[lane] = {
            "count_in_band": float(len(band)),
            "count_in_400_500m": float(len(near)),
            "nearest_to_band_top_unclipped": float(min(distances) - 400.0),
            "min_distance": float(min(distances)),
        }
    return out


def stage2_feature_vector(
    history: Mapping[int, Mapping[str, Any]],
    per_lane: Mapping[str, Dict[str, float]],
    t: int,
    h: float,
) -> Dict[str, float]:
    """The full repaired RSU vector at step ``t`` plus onset-predictor lags.

    ``history[t]`` is the fixed-capture per-step record (regions, lane
    imbalance, ramp queue, CAV counts, entries/discharge); ``per_lane`` is
    the unclipped per-lane summary at ``t``. Lag features are undefined (and
    omitted) when history does not reach back far enough.
    """
    out: Dict[str, float] = {}
    now = history[t]
    regions = now.get("regions", {})
    for region in ("upstream", "approach", "near_merge", "merge_edge", "post_merge", "ramp"):
        stats = regions.get(region, {})
        out[f"{region}_count"] = float(stats.get("count", 0.0))
        out[f"{region}_mean_speed"] = float(stats.get("mean_speed", 0.0))
        out[f"{region}_speed_std"] = float(stats.get("speed_std", 0.0))
    out["mainline_lane_imbalance"] = float(now.get("mainline_lane_imbalance", 0.0))
    out["ramp_queue_length"] = float(now.get("ramp_queue_length", 0.0))
    out["cav_count_band"] = float(now.get("cav_count_band", 0.0))
    out["cav_count_upstream"] = float(now.get("cav_count_upstream", 0.0))
    out["ramp_30s_entries"] = sum(
        float(history[s].get("ramp_entries", 0.0))
        for s in range(t - 30, t + 1) if s in history
    )
    inflow = sum(
        float(history[s].get("upstream_entries", 0.0))
        for s in range(t - 30, t + 1) if s in history
    )
    discharge = sum(
        float(history[s].get("merge_discharge", 0.0))
        for s in range(t - 30, t + 1) if s in history
    )
    out["upstream_inflow_30s"] = inflow
    out["merge_discharge_30s"] = discharge
    out["inflow_minus_discharge_30s"] = inflow - discharge
    for region, length_m, lanes in (("merge_edge", 300.0, 3), ("approach", 200.0, 3),
                                    ("upstream", 744.95, 3)):
        out[f"{region}_density"] = float(
            regions.get(region, {}).get("count", 0.0)) / (length_m / 1000.0 * lanes)
    prev = history.get(t - 120)
    if prev is not None:
        merge_now = out["merge_edge_density"]
        merge_prev = float(prev.get("regions", {}).get("merge_edge", {}).get("count", 0.0)) / (
            300.0 / 1000.0 * 3)
        out["merge_density_diff_120s"] = merge_now - merge_prev
        out["approach_std_diff_120s"] = (
            out["approach_speed_std"]
            - float(prev.get("regions", {}).get("approach", {}).get("speed_std", 0.0)))
        inflow_prev = sum(
            float(history[s].get("upstream_entries", 0.0))
            for s in range(t - 120, t - 90 + 1) if s in history)
        discharge_prev = sum(
            float(history[s].get("merge_discharge", 0.0))
            for s in range(t - 120, t - 90 + 1) if s in history)
        out["inflow_minus_discharge_diff_120s"] = (inflow - discharge) - (inflow_prev - discharge_prev)
    # Onset-predictor lags: merge-edge and post-merge mean speeds at 10/30/60 s.
    for region in ("merge_edge", "post_merge"):
        for lag in LAGS_S:
            past = history.get(t - lag)
            key = f"{region}_mean_speed_lag{lag}s"
            if past is not None:
                out[key] = float(
                    past.get("regions", {}).get(region, {}).get("mean_speed", 0.0))
    for lane, values in per_lane.items():
        prefix = f"lane_{lane}_"
        out[f"{prefix}count_in_band"] = float(values["count_in_band"])
        out[f"{prefix}count_in_400_500m"] = float(values["count_in_400_500m"])
        out[f"{prefix}nearest_to_band_top_unclipped"] = float(
            values["nearest_to_band_top_unclipped"])
        out[f"{prefix}min_distance"] = float(values["min_distance"])
    out["h"] = float(h)
    return out


def contract_manifest() -> Dict[str, object]:
    return {
        "schema": "stage2_generation_contract/1",
        "role": ROLE,
        "seeds_p20": [list(SEEDS_P20[:3]), list(SEEDS_P20[-3:])],
        "seeds_p30": [list(SEEDS_P30[:3]), list(SEEDS_P30[-3:])],
        "demand_profile": {
            "draw": "uniform integers [2200, 2800] from the seed number (hidden)",
            "plateau_end_s": PLATEAU_END_S,
            "decline_steps_s": 100.0,
            "decline_end_s": 1500.0,
            "tail_veh_per_h": TAIL_DEMAND_VPH,
            "tail_end_s": TAIL_END_S,
            "ramp_veh_per_h": RAMP_DEMAND_VPH,
        },
        "horizon_s": HORIZON_S,
        "cav_penetration": {"p20": CAV_PENETRATION_P20, "p30": CAV_PENETRATION_P30},
        "arms": {
            "D": "frozen analytical damping, no wall (prefix for everything)",
            **{
                f"WON_H{int(k)}": f"damping + wall engaged at t_on + 1, held {int(k)} s, "
                "5 s taper, tau restore; 10 m/s, tau+3.0, 200-400 m band, 0.3 clip"
                for k in HOLDS_S
            },
        },
        "onset_rule": f"first t with h < {ONSET_H_THRESHOLD} for "
        f"{ONSET_SUSTAIN_S:.0f} consecutive s in the D prefix; no W arms when "
        f"no onset by t = {ONSET_SEARCH_MAX_STEP}",
        "wall_params": WALL_PARAMS,
        "feature_times": "t_on, t_on - 60 (when >= capture floor), and fixed t = 150",
        "capture_step_range": [CAPTURE_LO_STEP, CAPTURE_HI_STEP],
        "per_lane_nearest_unclipped": True,
        "repeat_every": REPEAT_EVERY,
    }
