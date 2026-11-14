"""Build Phase-5 IDM reward references for the balanced 2200/2700 curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    2200: ROOT / "runs" / "eval" / "idm_analytical_pilot_v1" / "cases",
    2700: ROOT / "runs" / "eval" / "idm_transition_scan_v1" / "cases",
}
OUT = ROOT / "runs" / "eval" / "idm_phase5_reward_refs_v1.json"
SEEDS = tuple(range(4101, 4107))
SCALE_ARMS = {
    2200: "analytical_zero_residual",
    2700: "analytical_disturbance_reference_zero_residual",
}


def main() -> None:
    refs = {}
    source_manifest = {}
    for demand, source in SOURCES.items():
        rows = []
        source_manifest[str(demand)] = str(source)
        for seed in SEEDS:
            path = source / f"noop.m{demand}.seed{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            scale_path = source / f"{SCALE_ARMS[demand]}.m{demand}.seed{seed}.json"
            scale_payload = json.loads(scale_path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            scale_seconds = max(float(scale_payload["metrics"]["sim_seconds"]), 1e-9)
            mean_tt = float(metrics["travel_time"]["all"]["mean"])
            mean_loss = float(metrics["mean_time_loss"])
            throughput = float(metrics["throughput_veh_per_hour"])
            premerge_speed = float(payload["trajectory"]["wave"]["mean_premerge_speed_mps"])
            rows.append(
                {
                    "seed": seed,
                    "delay_rate_ref": mean_loss / max(mean_tt, 1e-9),
                    "flow_q_ref": throughput / 3600.0,
                    "flow_v_ref": premerge_speed,
                    "mean_time_loss_s_per_vehicle": mean_loss,
                    "mean_travel_time_s": mean_tt,
                    "corridor_severe_braking_count_ref_per_s": (
                        float(scale_payload["trajectory"]["safety"]["severe_braking_events"])
                        / scale_seconds
                    ),
                    "corridor_acceleration_energy_sum_ref_m2ps4": (
                        float(scale_payload["trajectory"]["wave"]["oscillation_energy_m2ps3"])
                        / scale_seconds
                    ),
                }
            )
        config = f"osm.dense.idm_m{demand}r400.sumocfg"
        refs[config] = {
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
    payload = {
        "schema": "idm_phase5_reward_refs/2",
        "sources": source_manifest,
        "source_role": (
            "development-only no-op IDM cases 4101-4106; excluded from "
            "checkpoint evaluation and validation"
        ),
        "seeds": list(SEEDS),
        "curriculum_demands_vph": sorted(SOURCES),
        "formulas": {
            "delay_rate_ref": "mean_time_loss_s / mean_travel_time_s",
            "flow_q_ref": "throughput_veh_per_hour / 3600",
            "flow_v_ref": "mean pre-merge speed m/s",
            "corridor_severe_braking_count_ref_per_s": (
                "analytical-zero severe-braking events / simulated seconds"
            ),
            "corridor_acceleration_energy_sum_ref_m2ps4": (
                "analytical-zero corridor oscillation energy / simulated seconds"
            ),
            "aggregation": "median across six development seeds",
        },
        "refs": refs,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
