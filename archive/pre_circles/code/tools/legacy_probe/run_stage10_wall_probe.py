"""Stage-10 legacy wall probe: w10_t30_h025 and no_op on the long-map IDM scene.

Runs the recovered legacy driver (``_probe_metastable_feasibility.py``,
verbatim from commit 8d993e9) with a reconstructed stage-10 variant table
(see RECOVERED_PROBE.md): w10_t30_h025 = WallHoldPolicy RUNGS[1] (band
200-400 m, 10 m/s, tau 3.0) with the legacy health trigger
arm h < 0.25 / release h > 0.45, and no_op = the legacy NoOpPolicy
(mode-0 coast). Scene: _light/osm.dense.light_m2200r400idm.sumocfg on
D:/research/highway_merge_shepherd_long, --pen 0.2, --steps 1800,
seeds 6001-6030. Rows are written by the driver's own schema-2 row writer
(sustained_collapse, loss, first_collapse_step, ttc_events, dv_mean, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from concurrent.futures import ProcessPoolExecutor, as_completed

LEGACY_PROBE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEGACY_PROBE_DIR))

import _probe_metastable_feasibility as drv  # noqa: E402  (recovered module)

from policies.inflow_control import WallHoldPolicy  # noqa: E402

STAGE10_VARIANTS: Dict[str, tuple] = {
    "no_op": ("no_op", {}),
    "w10_t30_h025": (
        "wall",
        {
            "d_lo": 200.0,
            "d_hi": 400.0,
            "wall_speed": 10.0,
            "wall_tau": 3.0,
            "a_max_dec": 0.3,
            "trigger_h": 0.25,
            "release_h": 0.45,
        },
    ),
}
DEFAULT_LEGACY_TAG = "light_m2200r400idm"
DEFAULT_SCENE_STEPS = 1800
SEEDS = tuple(range(6001, 6031))


def build_policy_stage10(kind: str, params: dict):
    """Stage-10 policy builder: keeps the recovered no_op path; adds a wall
    kind wrapping WallHoldPolicy with the legacy trigger."""
    from policies.metastable_rule import HealthRecordingWrapper

    if kind == "no_op":
        from policies.baselines import NoOpPolicy

        return HealthRecordingWrapper(NoOpPolicy()), "no_op"
    if kind == "wall":
        inner = WallHoldPolicy(
            d_lo=float(params["d_lo"]),
            d_hi=float(params["d_hi"]),
            wall_speed=float(params["wall_speed"]),
            wall_tau=float(params["wall_tau"]),
            a_max_dec=float(params["a_max_dec"]),
            trigger_h=float(params["trigger_h"]),
            release_h=float(params["release_h"]),
        )
        return HealthRecordingWrapper(inner), "wall"
    raise ValueError(f"unknown stage-10 kind {kind!r}")


def _run_one(args: tuple) -> Dict[str, Any]:
    variant, seed, out_dir, tag, steps = args
    noop_cache = Path(out_dir) / "noop_cache"
    # run_variant merges STAGE1..STAGE7_VARIANTS into a local table; extend the
    # module's last stage table (a documented reconstruction of the
    # never-committed stage-10 rows) and patch the global policy builder.
    drv.STAGE7_VARIANTS = {**drv.STAGE7_VARIANTS, **STAGE10_VARIANTS}
    drv._build_policy = build_policy_stage10
    return drv.run_variant(
        variant,
        seed,
        Path(out_dir) / "rows",
        noop_cache,
        tag=tag,
        steps=steps,
        pen=0.2,
        sroot=drv.SUMO_LONG,
        scope="legacy_stage10",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\research-dense\runs\probe\sustained_wall_v1_stage1b_prep4_legacy_probe"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "rows").mkdir(parents=True, exist_ok=True)
    (output / "noop_cache").mkdir(parents=True, exist_ok=True)
    tasks = [
        (variant, seed, str(output), DEFAULT_LEGACY_TAG, DEFAULT_SCENE_STEPS)
        for seed in args.seeds
        for variant in ("no_op", "w10_t30_h025")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            status = row.get("status", "completed")
            print(
                json.dumps({"variant": row.get("variant"), "seed": row.get("seed"), "status": status}),
                flush=True,
            )
    print(json.dumps({"status": "all_workers_done"}), flush=True)


if __name__ == "__main__":
    main()
