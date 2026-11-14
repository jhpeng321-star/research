"""Artifact-only natural-clearance reconciliation for the frozen v5.7.1 Gate A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments import REPO_ROOT
from experiments.idm.wall_v5_7_1_original_corridor_contract import SEEDS
from experiments.io import atomic_write_json


ROOT = REPO_ROOT / "runs" / "probe" / "idm_v5_7_1_original_corridor_expanded_zone_p30_s4401_4402_4404_4405"


def terminal_observation(trace: Mapping[str, Any], outcomes: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only a requested horizon or a fully evidenced natural clearance."""
    metadata = dict(trace.get("controller_metadata") or {})
    final = dict(trace.get("final") or {})
    reason = metadata.get("termination_reason")
    reached_horizon = not bool(trace.get("terminated_early"))
    natural_checks = {
        "termination_reason_terminated": reason == "terminated",
        "zero_remaining_vehicles": final.get("vehicle_count") == 0,
        "zero_expected_or_pending_vehicles": metadata.get("terminal_expected_vehicle_count") == 0,
        "no_collision_termination": int(outcomes.get("collision_unique_vehicle_count", -1)) == 0,
        "no_teleport_termination": int(outcomes.get("starting_teleport_unique_vehicle_count", -1)) == 0,
        "no_gridlock_termination": reason != "gridlock",
        "no_simulator_or_controller_error": reason in {"terminated", "horizon_reached"},
        "service_accounting_finalized": "merge_discharge_count" in outcomes and "incremental_delay_s" in outcomes,
    }
    natural_clearance = all(natural_checks.values())
    return {
        "reached_requested_horizon": reached_horizon,
        "natural_clearance": natural_clearance,
        "scientifically_complete": reached_horizon or natural_clearance,
        "termination_reason": reason,
        "final_sim_time_s": final.get("sim_time"),
        "checks": natural_checks,
    }


def reconcile_row(row: Mapping[str, Any]) -> dict[str, Any]:
    control = terminal_observation(row["control_trace"], row["outcomes"])
    repeat = terminal_observation(row["repeat_trace"], row["repeat_outcomes"])
    pair_terminal_identity = (
        control["scientifically_complete"]
        and repeat["scientifically_complete"]
        and control["natural_clearance"] == repeat["natural_clearance"]
        and control["termination_reason"] == repeat["termination_reason"]
        and control["final_sim_time_s"] == repeat["final_sim_time_s"]
    )
    congestion = dict(row["congestion"])
    return {
        "seed": int(row["seed"]),
        "damping_replay_identity": bool(row["damping_replay_identity"]["passed"]),
        "terminal": {"control": control, "repeat": repeat, "pair_terminal_identity": pair_terminal_identity},
        "general_congestion_headroom": bool(congestion["predicate_duration_s"] > 0),
        "general_congestion_duration_s": congestion["predicate_duration_s"],
        "sustained_collapse_headroom": bool(congestion["sustained_event"]),
        "sustained_collapse_duration_s": congestion["sustained_duration_s"],
    }


def reconcile(root: Path = ROOT) -> dict[str, Any]:
    rows = [json.loads((root / "baseline" / f"seed{seed}.json").read_text()) for seed in SEEDS]
    reconciled = [reconcile_row(row) for row in rows]
    gates = {
        "original_corridor_only": all(not row["derived_network_used"] for row in rows),
        "frozen_t880": all(row["decision_absolute_s"] == 880 for row in rows),
        "damping_replay_identity": all(row["damping_replay_identity"] for row in reconciled),
        "scientifically_complete_terminal": all(row["terminal"]["pair_terminal_identity"] for row in reconciled),
        "at_least_three_general_congestion_headroom": sum(row["general_congestion_headroom"] for row in reconciled) >= 3,
    }
    return {
        "schema": "idm_v5_7_1_gate_a_terminal_reconciliation/1",
        "source_gate_a_rerun": False,
        "terminal_rule": "requested_horizon_or_verified_identical_natural_clearance",
        "gates": gates,
        "gate_a_passed_under_natural_clearance_semantics": all(gates.values()),
        "rows": reconciled,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reconcile(args.root.resolve())
    output = args.output or args.root / "gate_a_terminal_reconciliation.json"
    atomic_write_json(output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
