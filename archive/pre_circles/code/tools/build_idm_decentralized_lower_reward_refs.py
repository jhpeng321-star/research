"""Build frozen development-only reward scales for lower/transition IDM cells.

The paired treatment effect is always computed from identical abstain and
activation branches.  These references only put severe-braking and
acceleration-energy differences on comparable scales across demand cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from experiments import REPO_ROOT
from experiments.io import atomic_write_json


ROOT = REPO_ROOT
DEMANDS = (1900, 2000, 2100, 2200)
SEEDS = tuple(range(4101, 4107))
SOURCE = ROOT / "runs" / "eval" / "idm_analytical_pilot_v1" / "cases"
OUTPUT = ROOT / "runs" / "eval" / "idm_decentralized_lower_reward_refs_v1.json"


def build_payload() -> dict:
    refs: dict[str, dict] = {}
    for demand in DEMANDS:
        rows = []
        for seed in SEEDS:
            noop = json.loads(
                (SOURCE / f"noop.m{demand}.seed{seed}.json").read_text(encoding="utf-8")
            )
            analytical = json.loads(
                (
                    SOURCE
                    / f"analytical_zero_residual.m{demand}.seed{seed}.json"
                ).read_text(encoding="utf-8")
            )
            noop_metrics = noop["metrics"]
            scale_metrics = analytical["metrics"]
            scale_seconds = max(float(scale_metrics["sim_seconds"]), 1e-9)
            mean_travel_time = float(noop_metrics["travel_time"]["all"]["mean"])
            rows.append(
                {
                    "seed": seed,
                    "delay_rate_ref": float(noop_metrics["mean_time_loss"])
                    / max(mean_travel_time, 1e-9),
                    "flow_q_ref": float(noop_metrics["throughput_veh_per_hour"])
                    / 3600.0,
                    "flow_v_ref": float(
                        noop["trajectory"]["wave"]["mean_premerge_speed_mps"]
                    ),
                    "corridor_severe_braking_count_ref_per_s": float(
                        analytical["trajectory"]["safety"]["severe_braking_events"]
                    )
                    / scale_seconds,
                    "corridor_acceleration_energy_sum_ref_m2ps4": float(
                        analytical["trajectory"]["wave"]["oscillation_energy_m2ps3"]
                    )
                    / scale_seconds,
                }
            )
        refs[f"osm.dense.idm_m{demand}r400.sumocfg"] = {
            "demand_mainline_vph": demand,
            "delay_rate_ref": median(row["delay_rate_ref"] for row in rows),
            "flow_q_ref": median(row["flow_q_ref"] for row in rows),
            "flow_v_ref": median(row["flow_v_ref"] for row in rows),
            "corridor_severe_braking_count_ref_per_s": median(
                row["corridor_severe_braking_count_ref_per_s"] for row in rows
            ),
            "corridor_acceleration_energy_sum_ref_m2ps4": median(
                row["corridor_acceleration_energy_sum_ref_m2ps4"] for row in rows
            ),
            "source_rows": rows,
        }
    return {
        "schema": "idm_decentralized_lower_reward_refs/1",
        "source": str(SOURCE),
        "source_role": (
            "development-only seeds 4101-4106; no-op supplies delay/flow "
            "references and analytical-zero supplies safety normalization scales"
        ),
        "seeds": list(SEEDS),
        "demands_vph": list(DEMANDS),
        "paired_effect_baseline": (
            "identical abstain branch; references are scales, not treatment controls"
        ),
        "refs": refs,
    }


def main() -> None:
    payload = build_payload()
    atomic_write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
