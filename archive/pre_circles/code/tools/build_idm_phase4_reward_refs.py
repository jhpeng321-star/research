"""Derive Phase-4 IDM reward references from the frozen no-op pilot cases."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "eval" / "idm_analytical_pilot_v1" / "cases"
OUT = ROOT / "runs" / "eval" / "idm_phase4_reward_refs_v1.json"
DEMANDS = (1900, 2000, 2100, 2200)
SEEDS = tuple(range(4101, 4107))


def main() -> None:
    refs = {}
    for demand in DEMANDS:
        rows = []
        for seed in SEEDS:
            path = SOURCE / f"noop.m{demand}.seed{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
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
                }
            )
        config = f"osm.dense.idm_m{demand}r400.sumocfg"
        refs[config] = {
            "demand_mainline_vph": demand,
            "delay_rate_ref": median(row["delay_rate_ref"] for row in rows),
            "flow_q_ref": median(row["flow_q_ref"] for row in rows),
            "flow_v_ref": median(row["flow_v_ref"] for row in rows),
            "source_rows": rows,
        }
    payload = {
        "schema": "idm_phase4_reward_refs/1",
        "source": str(SOURCE),
        "source_role": "development-only no-op IDM pilot; excluded from validation",
        "seeds": list(SEEDS),
        "formulas": {
            "delay_rate_ref": "mean_time_loss_s / mean_travel_time_s",
            "flow_q_ref": "throughput_veh_per_hour / 3600",
            "flow_v_ref": "mean pre-merge speed m/s",
            "aggregation": "median across six development seeds",
        },
        "refs": refs,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
