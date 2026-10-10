"""Recapture only control prefixes for the ten completed v5.2 wall cases.

This creates no treatment branch and no outcome. Exact equality of every
stored v1 trace-derived feature is required before observation-v2 values are
accepted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from counterfactual.actionability import dose_qualified
from counterfactual.idm_predictor_contract import (
    IDM_DECISION_FEATURES_V1,
    OBSERVATION_V2_FEATURES,
    feature_contract_manifest,
    preflight_features,
)
from counterfactual.outcomes import OutcomeDefinition
from counterfactual.pilot import features_from_trace
from counterfactual.twin_replay import run_replay
from experiments import REPO_ROOT
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.wall_v5_2_contract import overlay_overrides
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from policies.idm_analytical import build_frozen_idm_analytical_policy


SOURCE = REPO_ROOT / "runs" / "eval" / "idm_wall_v5_2_prospective_p30_s4301_4312"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "probe" / "idm_observation_v2_capture_v5_2_completed"


def _completed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((SOURCE / "seeds").glob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(row for row in payload["cases"] if dose_qualified(row))
    return rows


def _capture_seed(seed: int, rows: list[dict[str, Any]], sumo_binary: str) -> list[dict[str, Any]]:
    decisions = sorted(int(row["decision_step"]) for row in rows)
    trace = run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=sumo_binary,
        sumo_config=transition_scenario_config(2700).name,
        seed=seed,
        decision_step=max(decisions),
        horizon_steps=1,
        treatment_steps=None,
        cav_penetration=0.30,
        sumo_step_length=0.5,
        supervisor_overrides=supervisor_overrides(),
        digits=6,
        base_policy_factory=build_frozen_idm_analytical_policy,
        intervention_schedule=lambda _step: 0,
        overlay_overrides=overlay_overrides(),
        controller_metadata={"scientific_role": "observation_v2_control_prefix_only"},
    )
    definition = OutcomeDefinition()
    captured = []
    treatment = overlay_overrides()
    for source in rows:
        scan = source["timing_scan"]
        features = features_from_trace(
            trace,
            decision_step=int(source["decision_step"]),
            definition=definition,
            dose_context={
                "proposed_wall_speed_mps": treatment["wall_params"]["wall_speed"],
                "release_boundary_m": treatment["compact_release_boundary_m"],
                "selected_member_ids": (scan.get("formation") or {}).get(
                    "controlled_cav_ids", ()
                ),
            },
        )
        features.update(preflight_features(scan))
        mismatches = []
        for name in IDM_DECISION_FEATURES_V1:
            if name.startswith("context_"):
                continue
            old = source["features"].get(name)
            new = features.get(name)
            if old != new:
                mismatches.append({"feature": name, "stored": old, "recaptured": new})
        captured.append(
            {
                "schema": "idm_observation_v2_capture_row/1",
                "seed": seed,
                "decision_step": int(source["decision_step"]),
                "source_case_id": source["case_id"],
                "source_feature_contract_version": source["feature_contract_version"],
                "feature_contract_version": feature_contract_manifest()["contract_version"],
                "identity": {
                    "stored_v1_feature_count_checked": len(IDM_DECISION_FEATURES_V1) - 2,
                    "mismatch_count": len(mismatches),
                    "passed": not mismatches,
                    "mismatches": mismatches[:20],
                },
                "features": {name: features[name] for name in OBSERVATION_V2_FEATURES},
            }
        )
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    args = parser.parse_args()
    rows = _completed_rows()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    captured: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_capture_seed, seed, seed_rows, args.sumo_binary): seed
            for seed, seed_rows in grouped.items()
        }
        for future in as_completed(futures):
            result = future.result()
            captured.extend(result)
            print(json.dumps({"seed": futures[future], "rows": len(result)}), flush=True)
    captured.sort(key=lambda row: (row["seed"], row["decision_step"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in captured),
        encoding="utf-8",
    )
    atomic_write_json(
        args.output / "summary.json",
        {
            "schema": "idm_observation_v2_capture_summary/1",
            "source": str(SOURCE),
            "case_count": len(captured),
            "seed_count": len({row["seed"] for row in captured}),
            "all_identity_pass": all(row["identity"]["passed"] for row in captured),
            "new_treatment_branches": 0,
            "new_outcomes": 0,
            "feature_contract": feature_contract_manifest(),
        },
    )


if __name__ == "__main__":
    main()
