"""Stage 1b generation contract: seeds, hidden demand draw, scenario writer.

Population: mainline demand per seed drawn uniformly from [2200, 2800]
veh/h (integer), ramp 400 fixed; demand seeded from the seed number
(reproducible). Penetration strata: p = 0.20 (seeds 7001-7240), p = 0.30
(seeds 7241-7360). The drawn demand is a HIDDEN variable: it is recorded in
the per-seed manifest but NEVER enters a feature vector.

Arms per seed (4 branches): D (frozen analytical damping from t = 0, no
wall) and W150 / W250 / W350 (damping + sustained wall engaged at t_d =
150 / 250 / 350, held to 900, 10 m/s, tau 3.0, 200-400 m mainline band,
0.3 m/s^2 clip, 5 s taper, tau restore; damping continues everywhere
outside the band). Each wall branch must have exact prefix identity with D
through its t_d.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments import REPO_ROOT

ROLE = "stage1b_generation"
SEEDS_P20 = tuple(range(7001, 7241))
SEEDS_P30 = tuple(range(7241, 7361))
ALL_SEEDS = SEEDS_P20 + SEEDS_P30
DEMAND_LO = 2200
DEMAND_HI = 2800
RAMP_DEMAND_VPH = 400
CAV_PENETRATION_P20 = 0.20
CAV_PENETRATION_P30 = 0.30

SCENARIO_ROOT = REPO_ROOT / "scenarios" / "idm_stage1b"
TEMPLATE_ROUTE = REPO_ROOT / "scenarios" / "idm_transition" / "idm_m2200r400.rou.xml"
TEMPLATE_SCENE = REPO_ROOT / "scenarios" / "idm_transition" / "osm.dense.idm_m2200r400.sumocfg"

DECISION_TIMES = (150, 250, 350)
HOLD_UNTIL_S = 900.0
TAPER_DURATION_S = 5.0
WALL_PARAMS: Dict[str, float] = {
    "d_lo": 200.0,
    "d_hi": 400.0,
    "wall_speed": 10.0,
    "wall_tau": 3.0,
    "a_max_dec": 0.3,
}
FEATURE_TIMES = (150, 250, 350)
DIFF_BACKWARD_S = 120
REPEAT_EVERY = 20  # every 20th seed repeats D


def demand_for_seed(seed: int) -> int:
    """Deterministic per-seed demand draw (uniform integers in [2200, 2800])."""
    rng = np.random.default_rng(int(seed))
    return int(rng.integers(DEMAND_LO, DEMAND_HI + 1))


def penetration_for_seed(seed: int) -> float:
    return CAV_PENETRATION_P20 if int(seed) in set(SEEDS_P20) else CAV_PENETRATION_P30


def strate_for_seed(seed: int) -> str:
    return "p20" if int(seed) in set(SEEDS_P20) else "p30"


def route_name_for_seed(seed: int, demand: int) -> str:
    return f"idm_m{demand}_seed{seed}.rou.xml"


def scene_name_for_seed(seed: int, demand: int) -> str:
    return f"osm.dense.idm_m{demand}_seed{seed}.sumocfg"


def write_seed_scenario(seed: int, demand: int, root: Path = SCENARIO_ROOT) -> str:
    """Write the per-seed route file and scene file; return the scene name."""
    root.mkdir(parents=True, exist_ok=True)
    route_path = root / route_name_for_seed(seed, demand)
    route_src = TEMPLATE_ROUTE.read_text(encoding="utf-8")
    # Only the mainline flow's vehsPerHour changes.
    route_out, count = re.subn(
        r'(<flow id="mainline"[^>]*?)vehsPerHour="[0-9]+"',
        lambda m: m.group(1) + f'vehsPerHour="{int(demand)}"',
        route_src,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"could not rewrite mainline vehsPerHour in {TEMPLATE_ROUTE}")
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


def contract_manifest() -> Dict[str, object]:
    return {
        "schema": "stage1b_generation_contract/1",
        "role": ROLE,
        "seeds_p20": [list(SEEDS_P20[:3]), list(SEEDS_P20[-3:])],
        "seeds_p30": [list(SEEDS_P30[:3]), list(SEEDS_P30[-3:])],
        "demand_range": [DEMAND_LO, DEMAND_HI],
        "ramp_demand_vph": RAMP_DEMAND_VPH,
        "cav_penetration": {"p20": CAV_PENETRATION_P20, "p30": CAV_PENETRATION_P30},
        "hidden_variable": "seed-drawn mainline demand (recorded, never a feature)",
        "arms": {
            "D": "frozen analytical damping from t=0, no wall",
            **{
                f"W{td}": f"damping + wall engaged at t={td}, held to {HOLD_UNTIL_S}, "
                f"10 m/s, tau 3.0, band 200-400 m, 0.3 clip, 5 s taper"
                for td in DECISION_TIMES
            },
        },
        "wall_params": WALL_PARAMS,
        "feature_times": list(FEATURE_TIMES),
        "diff_backward_s": DIFF_BACKWARD_S,
        "repeat_every": REPEAT_EVERY,
    }
