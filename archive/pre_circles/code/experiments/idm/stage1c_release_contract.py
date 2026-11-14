"""Stage 1c deployable-release contract: plateau-decline demand, seeds, arms.

Demand profile per cell M in {2200, 2400, 2600, 2700}: mainline M veh/h for
0-1200 s; three 100 s steps from M down to 1500 veh/h over 1200-1500 s
(two intermediate steps at integer thirds, then 1500); 1500 veh/h for
1500-1800 s; no departures after 1800. Ramp 400 veh/h for 0-1800 s.
Horizon: natural clearance or 2400 s. p = 0.20. Seeds 7401-7430 (2200),
7431-7460 (2400), 7461-7490 (2600), 7491-7520 (2700).

Arms (all damping-based; W arms branch from D at t_d = 250, exact prefix
identity): D (damping only); W_ORACLE (hold until t = 1200, plateau end —
knows demand, reference only; 5 s taper); W_R1 (release when for 60
consecutive seconds ALL hold: upstream arrival rate over 400-800 m per
30 s <= 0.9 x merge discharge over the same 30 s AND mean speed over
200-800 m >= 0.8 x free-flow speed; min hold 120 s; hard max 900 s);
W_R2 (release when 200-800 m approach density below the treatment-blind
free-flow threshold — 90th percentile of that density over the D prefix
steps 1-100 of the same seed — for 60 consecutive seconds; min hold 120 s;
hard max 900 s); W_HOLD (hold to 900 s; comparability with prep-5).
All walls: 10 m/s, tau 3.0, 200-400 m mainline, 0.3 m/s^2 clip, 5 s taper,
tau restore; no re-engagement. Release-rule inputs (arrival rate, discharge
rate, approach mean speed, approach density) recorded at 1 Hz in every W arm.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Sequence

from experiments import REPO_ROOT

ROLE = "stage1c_release"
CELLS = {2200: tuple(range(7401, 7431)), 2400: tuple(range(7431, 7461)),
         2600: tuple(range(7461, 7491)), 2700: tuple(range(7491, 7521))}
ALL_SEEDS = tuple(s for block in CELLS.values() for s in block)

SCENARIO_ROOT = REPO_ROOT / "scenarios" / "idm_stage1c"
TEMPLATE_ROUTE = REPO_ROOT / "scenarios" / "idm_transition" / "idm_m2200r400.rou.xml"
TEMPLATE_SCENE = REPO_ROOT / "scenarios" / "idm_transition" / "osm.dense.idm_m2200r400.sumocfg"

PLATEAU_END_S = 1200.0
DECLINE_STEPS_S = 100.0
DECLINE_END_S = 1500.0
TAIL_DEMAND_VPH = 1500
TAIL_END_S = 1800.0
RAMP_DEMAND_VPH = 400
HORIZON_S = 2400.0
CAV_PENETRATION = 0.20
WALL_PARAMS: Dict[str, float] = {"d_lo": 200.0, "d_hi": 400.0, "wall_speed": 10.0,
                                 "wall_tau": 3.0, "a_max_dec": 0.3}
DECISION_STEP = 250
MIN_HOLD_S = 120.0
HARD_MAX_HOLD_S = 900.0
TAPER_DURATION_S = 5.0
RELEASE_WINDOW_S = 60.0
R1_ARRIVAL_RATIO = 0.9
R1_SPEED_FRACTION = 0.8
R2_PREFIX_WINDOW_S = 100.0
R2_QUANTILE = 0.90
APPROACH_LO_M = 200.0
APPROACH_HI_M = 800.0
APPROACH_LANES = 3


def decline_flow_steps(mainline_vph: int) -> Sequence[tuple[float, float, int]]:
    """(begin_s, end_s, veh/h) segments for the mainline flow."""
    m = int(mainline_vph)
    step = max(1, round((m - TAIL_DEMAND_VPH) / 3.0))
    v1 = m - step
    v2 = m - 2 * step
    return (
        (0.0, PLATEAU_END_S, m),
        (PLATEAU_END_S, PLATEAU_END_S + DECLINE_STEPS_S, max(TAIL_DEMAND_VPH, v1)),
        (PLATEAU_END_S + DECLINE_STEPS_S, PLATEAU_END_S + 2 * DECLINE_STEPS_S,
         max(TAIL_DEMAND_VPH, v2)),
        (PLATEAU_END_S + 2 * DECLINE_STEPS_S, DECLINE_END_S, TAIL_DEMAND_VPH),
        (DECLINE_END_S, TAIL_END_S, TAIL_DEMAND_VPH),
    )


def route_name_for_seed(seed: int, mainline_vph: int) -> str:
    return f"idm_m{mainline_vph}r400.plateau_decline.seed{seed}.rou.xml"


def scene_name_for_seed(seed: int, mainline_vph: int) -> str:
    return f"osm.dense.idm_m{mainline_vph}r400.plateau_decline.seed{seed}.sumocfg"


def write_seed_scenario(seed: int, mainline_vph: int, root: Path = SCENARIO_ROOT) -> str:
    """Write the per-seed route and scene; return the scene name."""
    root.mkdir(parents=True, exist_ok=True)
    route_path = root / route_name_for_seed(seed, mainline_vph)

    def flow_block(begin: float, end: float, veh_per_h: int) -> str:
        return (
            f'  <flow id="mainline.{int(begin)}" type="hdv_mix" route="mainline_route" '
            f'begin="{int(begin)}" end="{int(end)}" vehsPerHour="{int(veh_per_h)}" '
            f'departLane="best" departSpeed="max"/>'
        )

    route_src = TEMPLATE_ROUTE.read_text(encoding="utf-8")
    # Replace the single mainline flow with the piecewise segments; keep ramp.
    ramp_match = re.search(r'<flow id="ramp"[^>]*/>', route_src)
    ramp_flow = ramp_match.group(0) if ramp_match else ""
    ramp_flow = re.sub(r'vehsPerHour="[0-9]+"', f'vehsPerHour="{RAMP_DEMAND_VPH}"', ramp_flow)
    head = route_src.split("<flow")[0]
    route_out = (
        head
        + "\n".join(flow_block(b, e, v) for b, e, v in decline_flow_steps(mainline_vph))
        + "\n  "
        + ramp_flow
        + "\n</routes>\n"
    )    # SUMO requires route flow entries sorted by begin time (an unsorted
    # file is silently partially ignored - the ramp flow was dropped).
    blocks = re.findall(r"<flow[^>]*/>", route_out)
    blocks.sort(key=lambda b: float(re.search(r'begin="([0-9.]+)"', b).group(1)))
    route_out = route_out.split("<flow")[0] + "\n".join(blocks) + "\n</routes>\n"
    route_path.write_text(route_out, encoding="utf-8")
    scene_path = root / scene_name_for_seed(seed, mainline_vph)
    scene_src = TEMPLATE_SCENE.read_text(encoding="utf-8")
    scene_src = re.sub(
        r'<route-files value="[^"]*"/>',
        f'<route-files value="{route_path.name}"/>',
        scene_src,
        count=1,
    )
    scene_path.write_text(scene_src, encoding="utf-8")
    return scene_path.name


def contract_manifest() -> Dict[str, object]:
    return {
        "schema": "stage1c_release_contract/1",
        "role": ROLE,
        "cells": {str(k): [list(v[:2]), list(v[-2:])] for k, v in CELLS.items()},
        "cav_penetration": CAV_PENETRATION,
        "demand_profile": {
            "plateau_end_s": PLATEAU_END_S,
            "decline_steps_s": DECLINE_STEPS_S,
            "decline_end_s": DECLINE_END_S,
            "tail_veh_per_h": TAIL_DEMAND_VPH,
            "tail_end_s": TAIL_END_S,
            "ramp_veh_per_h": RAMP_DEMAND_VPH,
        },
        "horizon_s": HORIZON_S,
        "wall_params": WALL_PARAMS,
        "decision_step": DECISION_STEP,
        "release_rules": {
            "W_ORACLE": "hold until t=1200 (plateau end), reference only, 5 s taper",
            "W_R1": f"60 s of (arrival_rate(400-800 m, 30 s) <= {R1_ARRIVAL_RATIO} x "
            f"merge_discharge(30 s)) AND (mean speed 200-800 m >= {R1_SPEED_FRACTION} x "
            "free-flow); min hold 120 s; hard max 900 s; 5 s taper",
            "W_R2": f"approach density (200-800 m) below the 90th percentile of the D-prefix "
            "density over steps 1-100 for 60 consecutive seconds; min hold 120 s; hard max "
            "900 s; 5 s taper",
            "W_HOLD": "hold to 900 s, 5 s taper (prep-5 comparability)",
        },
        "min_hold_s": MIN_HOLD_S,
        "hard_max_hold_s": HARD_MAX_HOLD_S,
        "taper_duration_s": TAPER_DURATION_S,
    }
