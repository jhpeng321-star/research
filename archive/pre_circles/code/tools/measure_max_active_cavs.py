"""Measure max concurrent active CAVs per density (for sizing max_cavs_critic).

Runs one full episode per density at a given CAV penetration with no
interventions (empty action dict) and tracks the maximum number of
simultaneously active CAVs across all steps. The MAPPO critic packs every
active CAV, so ``max_cavs_critic`` must cover this maximum.

Usage:
    python tools/measure_max_active_cavs.py --densities light medium dense
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.core import TrafficShepherdMultiAgentEnv  # noqa: E402


def measure_density(
    density: str, penetration: float, max_steps: int | None, seed: int | None
) -> dict:
    env_config = {
        "sumo_config": f"osm.{density}.sumocfg",
        "cav_penetration": penetration,
        "centralized_critic": False,
        "eval_mode": False,
    }
    if seed is not None:
        env_config["cav_selection_seed"] = seed
    env = TrafficShepherdMultiAgentEnv(env_config)
    if max_steps is not None:
        env.max_episode_steps = max_steps
    max_active = 0
    steps = 0
    t0 = time.time()
    try:
        obs, info = env.reset()
        max_active = max(max_active, len(info["active_cavs"]))
        while True:
            obs, rewards, terminated, truncated, info = env.step({})
            steps += 1
            max_active = max(max_active, len(info["active_cavs"]))
            if terminated["__all__"] or truncated["__all__"]:
                break
    finally:
        env.close()
    return {
        "density": density,
        "penetration": penetration,
        "seed": seed,
        "max_concurrent_active_cavs": max_active,
        "steps": steps,
        "wall_time_s": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--densities", nargs="+", default=["light", "medium", "dense"])
    parser.add_argument("--penetration", type=float, default=0.20)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="CAV selection seed (default: config cav_selection_seed=42).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override episode length (quick smoke test only).",
    )
    args = parser.parse_args()

    results = []
    for density in args.densities:
        r = measure_density(density, args.penetration, args.max_steps, args.seed)
        results.append(r)
        print(json.dumps(r), flush=True)

    overall = max(r["max_concurrent_active_cavs"] for r in results)
    # Smallest power of two strictly greater than the observed maximum.
    cap = 1
    while cap <= overall:
        cap *= 2
    summary = {"overall_max": overall, "suggested_max_cavs_critic": cap}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
