"""Analyze the frozen 15-case IDM-to-P30 projected-blend authority screen."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.io import atomic_write_json
from idm.decentralized_damping import (
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
    ACTIVATION_SPEED_CEILING_ACTION_SET,
)
from idm.projected_blend import IDM_P30_PROJECTED_BLEND_V1
from tools.analyze_idm_modified_followerstopper_screen import (
    _cohort_ids,
    _local_safety,
)


SCHEMA = "idm_p30_projected_blend_authority/1"
COMMAND_DELTA_MPS2 = 0.05
MINIMUM_DISTINCT_COMMAND_STEPS = 2
HARD_TTC_FLOOR_S = 4.0
SEVERE_BRAKING_THRESHOLD_MPS2 = -3.0
HARD_BRAKING_THRESHOLD_MPS2 = -4.5
P30_IDENTITY_TOLERANCE_MPS2 = 1e-9
IDEMPOTENCE_TOLERANCE_MPS2 = 1e-9
STRENGTH_BY_MODE = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
EXPECTED_CASE_IDS = frozenset(
    {
        "m2000.seed4123.t61",
        "m2000.seed4127.t76",
        "m2000.seed4129.t354",
        "m2000.seed4130.t55",
        "m2000.seed4131.t382",
        "m2000.seed4132.t49",
        "m2000.seed4123.t146",
        "m2000.seed4127.t200",
        "m2000.seed4129.t436",
        "m2000.seed4130.t192",
        "m2000.seed4132.t291",
        "m2000.seed4127.t289",
        "m2000.seed4129.t528",
        "m2000.seed4130.t331",
        "m2000.seed4132.t574",
    }
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _effect_summary(effect: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": effect.get("mode"),
        "mode_index": int(effect["mode_index"]),
        "classification": str(effect["classification"]),
        "score": float(effect["safety_composite_reference_units"]),
        "constraints_pass": bool(effect["constraints_pass"]),
        "constraints": dict(effect["constraints"]),
        "severe_effect": float(effect["severe_braking_effect_reference_units"]),
        "wave_effect": float(effect["wave_effect_reference_units"]),
        "delay_rate_delta": float(effect["delay_rate_delta"]),
        "arrival_delta": int(effect["arrival_delta"]),
        "merge_discharge_count_delta": int(effect["merge_discharge_count_delta"]),
        "low_ttc_exposure_step_delta": int(effect["low_ttc_exposure_step_delta"]),
        "target_command_step_count": int(effect["target_command_step_count"]),
        "treatment_safety_abort_count": int(
            effect.get("treatment_safety_abort_count", 0)
        ),
    }


def _effect_map(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    summary = _read(root / "summary.json")
    result = {}
    for case in summary.get("cases", []):
        for effect in case.get("effects", []):
            if int(effect.get("candidate_slot") or 0) != 1:
                continue
            key = (str(case["case_id"]), int(effect["mode_index"]))
            if key in result:
                raise ValueError(f"duplicate effect {key} in {root}")
            result[key] = dict(effect)
    return result


def _branch_map(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for path in sorted((root / "branches").glob("*.json")):
        row = _read(path)
        if row.get("is_control") or int(row.get("candidate_slot") or 0) != 1:
            continue
        key = (str(row["case_id"]), int(row["mode_index"]))
        if key in result:
            raise ValueError(f"duplicate branch {key} in {root}")
        result[key] = row
    return result


def _discovery_map(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((root / "discoveries").glob("*.json")):
        row = _read(path)
        decision = row.get("decision")
        if decision is None:
            continue
        case_id = (
            f"m{int(row.get('demand', 2000))}.seed{int(row['seed'])}."
            f"t{int(decision['decision_step'])}"
        )
        result[case_id] = row
    return result


def _target_commands(branch: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    target = str(branch.get("target_vehicle_id"))
    return {
        int(row["relative_step"]): row
        for row in branch.get("command_rows", [])
        if str(row.get("vehicle_id")) == target
    }


def _command_contrast(
    reference: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    lhs, rhs = _target_commands(reference), _target_commands(treatment)
    shared = sorted(set(lhs) & set(rhs))
    deltas = [
        abs(
            float(rhs[step]["projected_acceleration_mps2"])
            - float(lhs[step]["projected_acceleration_mps2"])
        )
        for step in shared
    ]
    diagnostics = [rhs[step].get("projected_blend") for step in sorted(rhs)]
    idempotence = [
        abs(float(row.get("final_projection_idempotence_error_mps2", math.inf)))
        for row in diagnostics
        if row is not None
    ]
    projected = [
        float(rhs[step]["projected_acceleration_mps2"]) for step in shared
    ]
    jerk = [
        abs(projected[index] - projected[index - 1])
        for index in range(1, len(projected))
    ]
    return {
        "reference_command_steps": len(lhs),
        "treatment_command_steps": len(rhs),
        "shared_command_steps": len(shared),
        "all_treatment_commands_compared": bool(rhs) and set(rhs).issubset(lhs),
        "distinct_command_steps": sum(
            delta + 1e-12 >= COMMAND_DELTA_MPS2 for delta in deltas
        ),
        "meaningfully_distinct": sum(
            delta + 1e-12 >= COMMAND_DELTA_MPS2 for delta in deltas
        )
        >= MINIMUM_DISTINCT_COMMAND_STEPS,
        "maximum_absolute_acceleration_delta_mps2": max(deltas, default=0.0),
        "all_blend_diagnostics_present": bool(rhs)
        and len(diagnostics) == len(rhs)
        and all(row is not None for row in diagnostics),
        "maximum_idempotence_error_mps2": max(idempotence, default=math.inf),
        "any_projection_conflict": any(
            bool(row.get("treatment_infeasible", False)) for row in rhs.values()
        ),
        "maximum_command_jerk_mps3": max(jerk, default=0.0),
    }


def _finite_at_least(value: Any, lower: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number + 1e-12 >= lower


def _ttc_at_least_or_nonclosing(value: Any, lower: float) -> bool:
    """Treat an absent finite TTC as non-closing when samples are present.

    Phase-response rows store TTC only for closing pairs.  Consequently, the
    minimum over a populated cohort is ``None`` when every sampled pair is
    non-closing; this represents infinite TTC rather than missing telemetry.
    """
    if value is None:
        return True
    return _finite_at_least(value, lower)


def _local_safety_complete(local: Mapping[str, Any]) -> bool:
    stable = local["stable_upstream_cohort"]
    immediate = local["dynamic_immediate_follower"]
    return (
        bool(local["decision_cohort_vehicle_ids"])
        and int(stable["vehicle_step_count"]) > 0
        and int(immediate["vehicle_step_count"]) > 0
        and _ttc_at_least_or_nonclosing(
            stable["minimum_ttc_s"], HARD_TTC_FLOOR_S
        )
        and _ttc_at_least_or_nonclosing(
            immediate["minimum_ttc_s"], HARD_TTC_FLOOR_S
        )
        and _finite_at_least(
            stable["minimum_acceleration_mps2"], HARD_BRAKING_THRESHOLD_MPS2
        )
        and _finite_at_least(
            immediate["minimum_acceleration_mps2"], HARD_BRAKING_THRESHOLD_MPS2
        )
    )


def _distinct_level_count(values: Sequence[float]) -> int:
    levels: list[float] = []
    for value in sorted(float(item) for item in values):
        if not levels or value - levels[-1] + 1e-12 >= COMMAND_DELTA_MPS2:
            levels.append(value)
    return len(levels)


def _case_has_three_levels(branches: Mapping[int, Mapping[str, Any]]) -> bool:
    commands = {mode: _target_commands(branch) for mode, branch in branches.items()}
    shared = set.intersection(*(set(rows) for rows in commands.values()))
    qualifying_steps = 0
    for step in shared:
        values = [
            float(commands[mode][step]["projected_acceleration_mps2"])
            for mode in sorted(commands)
        ]
        if max(values) - min(values) + 1e-12 < 0.10:
            continue
        if _distinct_level_count(values) >= 3:
            qualifying_steps += 1
    return qualifying_steps >= 2


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [float(row["effect"]["score"]) for row in rows]
    classifications = [str(row["effect"]["classification"]) for row in rows]
    return {
        "case_count": len(rows),
        "classification_counts": dict(Counter(classifications)),
        "harmful_case_count": classifications.count("harmful"),
        "below_minus_0p05_count": sum(score < -0.05 for score in scores),
        "below_minus_0p20_count": sum(score < -0.20 for score in scores),
        "mean_effect": float(np.mean(scores)) if scores else None,
        "p10_effect": float(np.percentile(scores, 10.0)) if scores else None,
    }


def analyze(
    p30_roots: Sequence[Path], blend_roots: Sequence[Path]
) -> dict[str, Any]:
    if len(p30_roots) != len(blend_roots):
        raise ValueError("--p30-runs and --blend-runs must have equal lengths")
    rows: list[dict[str, Any]] = []
    source_rows = []
    case_branches: dict[str, dict[int, dict[str, Any]]] = {}
    p30_effects_by_case: dict[str, dict[str, Any]] = {}
    replay_exact = []
    for p30_root, blend_root in zip(p30_roots, blend_roots):
        p30_summary = _read(p30_root / "summary.json")
        blend_summary = _read(blend_root / "summary.json")
        p30_protocol = dict(p30_summary["protocol"])
        blend_protocol = dict(blend_summary["protocol"])
        if p30_protocol.get("action_set") != ACTIVATION_SPEED_CEILING_ACTION_SET:
            raise ValueError(f"{p30_root} is not frozen P30")
        if (
            blend_protocol.get("action_set")
            != ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET
        ):
            raise ValueError(f"{blend_root} is not the frozen projected blend")
        expected_contract = json.loads(
            json.dumps(IDM_P30_PROJECTED_BLEND_V1.to_dict())
        )
        if blend_protocol.get("blend_contract") != expected_contract:
            raise ValueError(f"{blend_root} blend contract drifted")
        frozen_keys = (
            "demand",
            "candidate_slots",
            "option_steps",
            "minimum_dose_steps",
            "cav_penetration",
            "decision_index",
            "candidate_ranking",
            "candidate_count",
            "safety_horizon_steps",
            "service_horizon_steps",
            "landmark",
            "reward_refs_sha256",
        )
        field_identity = {
            key: p30_protocol.get(key) == blend_protocol.get(key)
            for key in frozen_keys
        }
        source_rows.append(
            {
                "p30_root": str(p30_root),
                "blend_root": str(blend_root),
                "protocol_field_identity": field_identity,
                "abstain_replay_exact": bool(
                    (blend_summary.get("smoke_gates") or {}).get(
                        "abstain_replay_exact", False
                    )
                ),
            }
        )
        replay_exact.append(source_rows[-1]["abstain_replay_exact"])
        p30_effects = _effect_map(p30_root)
        blend_effects = _effect_map(blend_root)
        p30_branches = _branch_map(p30_root)
        blend_branches = _branch_map(blend_root)
        discoveries = _discovery_map(blend_root)
        case_ids = {case_id for case_id, _mode in blend_effects}
        for case_id in sorted(case_ids):
            p30_key = next(
                (key for key in p30_effects if key[0] == case_id), None
            )
            p30_branch_key = next(
                (key for key in p30_branches if key[0] == case_id), None
            )
            if p30_key is None or p30_branch_key is None or case_id not in discoveries:
                raise ValueError(f"incomplete P30 pairing for {case_id}")
            required = {(case_id, mode) for mode in STRENGTH_BY_MODE}
            missing = required - set(blend_effects) | required - set(blend_branches)
            if missing:
                raise ValueError(f"missing blend arms for {case_id}: {sorted(missing)}")
            p30_branch = p30_branches[p30_branch_key]
            p30_effects_by_case[case_id] = p30_effects[p30_key]
            case_branches[case_id] = {
                mode: blend_branches[(case_id, mode)] for mode in STRENGTH_BY_MODE
            }
            cohort = _cohort_ids(case_branches[case_id][4])
            p30_local = _local_safety(p30_branch, cohort)
            decision = discoveries[case_id]["decision"]
            matched = decision.get("matched_decision_source") or {}
            for mode, strength in STRENGTH_BY_MODE.items():
                effect = _effect_summary(blend_effects[(case_id, mode)])
                branch = case_branches[case_id][mode]
                local = _local_safety(branch, cohort)
                contrast = _command_contrast(
                    case_branches[case_id][4], branch
                )
                diag_strengths = {
                    float((row.get("projected_blend") or {}).get("strength"))
                    for row in _target_commands(branch).values()
                    if (row.get("projected_blend") or {}).get("strength") is not None
                }
                rows.append(
                    {
                        "case_id": case_id,
                        "seed": int(branch["seed"]),
                        "decision_step": int(branch["decision_step"]),
                        "target_vehicle_id": branch.get("target_vehicle_id"),
                        "same_target_as_p30": branch.get("target_vehicle_id")
                        == p30_branch.get("target_vehicle_id"),
                        "mode_index": mode,
                        "strength": strength,
                        "diagnostic_strengths": sorted(diag_strengths),
                        "matched_source_identity": {
                            key: bool(matched.get(key, False))
                            for key in (
                                "raw_decision_context_identity",
                                "nonclassic_observation_identity",
                                "treatment_difference_expected",
                            )
                        },
                        "effect": effect,
                        "command_contrast_vs_p30": contrast,
                        "frozen_p30_command_contrast": _command_contrast(
                            p30_branch, branch
                        ),
                        "local_safety": local,
                        "local_safety_complete_and_bounded": (
                            _local_safety_complete(local)
                        ),
                        "p30_local_safety": p30_local,
                    }
                )

    case_ids = {row["case_id"] for row in rows}
    by_case = {
        case_id: sorted(
            (row for row in rows if row["case_id"] == case_id),
            key=lambda row: row["strength"],
        )
        for case_id in sorted(case_ids)
    }
    p30_rows = [
        {
            "case_id": case_id,
            "seed": case_rows[0]["seed"],
            "effect": _effect_summary(p30_effects_by_case[case_id]),
        }
        for case_id, case_rows in by_case.items()
    ]
    p30_by_case = {row["case_id"]: row for row in p30_rows}
    strength_rows = {
        strength: [row for row in rows if row["strength"] == strength]
        for strength in STRENGTH_BY_MODE.values()
    }
    fixed_metrics = {
        str(strength): _arm_metrics(group)
        for strength, group in strength_rows.items()
    }
    p30_metrics = _arm_metrics(p30_rows)

    oracle_rows = []
    for case_id, case_rows in by_case.items():
        admissible = [row for row in case_rows if row["effect"]["constraints_pass"]]
        active_winner = max(
            admissible,
            key=lambda row: (row["effect"]["score"], -row["strength"]),
        )
        p30_score = float(p30_by_case[case_id]["effect"]["score"])
        deployable_score = max(0.0, float(active_winner["effect"]["score"]))
        deployable_strength = (
            float(active_winner["strength"])
            if float(active_winner["effect"]["score"]) > 0.0
            else 0.0
        )
        endpoint_best = max(0.0, p30_score)
        intermediate = [row for row in admissible if row["strength"] < 1.0]
        intermediate_best = max(
            intermediate,
            key=lambda row: (row["effect"]["score"], -row["strength"]),
        )
        oracle_rows.append(
            {
                "case_id": case_id,
                "seed": int(case_rows[0]["seed"]),
                "p30_effect": p30_score,
                "active_oracle_strength": float(active_winner["strength"]),
                "active_oracle_effect": float(active_winner["effect"]["score"]),
                "deployable_oracle_strength": deployable_strength,
                "deployable_oracle_effect": deployable_score,
                "blend_increment_over_endpoints": deployable_score - endpoint_best,
                "intermediate_best_strength": float(intermediate_best["strength"]),
                "intermediate_best_effect": float(intermediate_best["effect"]["score"]),
                "p30_harm_rescued_by_intermediate": (
                    p30_score < -0.02
                    and float(intermediate_best["effect"]["score"]) >= -0.02
                ),
                "p30_harm_made_helpful_by_intermediate": (
                    p30_score < -0.02
                    and float(intermediate_best["effect"]["score"]) > 0.02
                ),
            }
        )

    active_scores = [row["active_oracle_effect"] for row in oracle_rows]
    deployable_scores = [row["deployable_oracle_effect"] for row in oracle_rows]
    best_constant_mean = max(
        [0.0] + [float(metrics["mean_effect"]) for metrics in fixed_metrics.values()]
    )
    distinct_case_counts = {
        str(strength): sum(
            row["command_contrast_vs_p30"]["meaningfully_distinct"]
            for row in strength_rows[strength]
        )
        for strength in (0.25, 0.50, 0.75)
    }
    three_level_cases = sum(
        _case_has_three_levels(case_branches[case_id]) for case_id in by_case
    )
    actionable = [
        row for row in oracle_rows if row["blend_increment_over_endpoints"] > 0.02
    ]
    active_helpful_seeds = {
        row["seed"] for row in oracle_rows if row["active_oracle_effect"] > 0.02
    }
    active_nonbeneficial_seeds = {
        row["seed"] for row in oracle_rows if row["active_oracle_effect"] <= 0.02
    }
    p30_identity_rows = strength_rows[1.0]

    identity_gates = {
        "exact_fifteen_case_set": case_ids == EXPECTED_CASE_IDS and len(rows) == 60,
        "all_protocol_fields_match": all(
            all(source["protocol_field_identity"].values()) for source in source_rows
        ),
        "all_abstain_replays_exact": bool(replay_exact) and all(replay_exact),
        "all_matched_source_identities": all(
            all(row["matched_source_identity"].values()) for row in rows
        ),
        "all_targets_match_p30": all(row["same_target_as_p30"] for row in rows),
        "all_diagnostic_strengths_exact": all(
            row["diagnostic_strengths"] == [row["strength"]] for row in rows
        ),
        "full_p30_command_identity": all(
            row["frozen_p30_command_contrast"][
                "maximum_absolute_acceleration_delta_mps2"
            ]
            <= P30_IDENTITY_TOLERANCE_MPS2
            and row["frozen_p30_command_contrast"]["shared_command_steps"] == 20
            for row in p30_identity_rows
        ),
        "p30_outcome_identity_within_replay_noise": all(
            abs(
                row["effect"]["score"]
                - p30_by_case[row["case_id"]]["effect"]["score"]
            )
            <= 0.02
            for row in p30_identity_rows
        ),
    }
    mechanics_safety_gates = {
        "all_branches_constraints_pass": all(
            row["effect"]["constraints_pass"] for row in rows
        ),
        "all_active_arms_deliver_twenty_commands": all(
            row["effect"]["target_command_step_count"] == 20 for row in rows
        ),
        "all_commands_have_blend_diagnostics": all(
            row["command_contrast_vs_p30"]["all_blend_diagnostics_present"]
            for row in rows
        ),
        "all_treatment_commands_compared": all(
            row["command_contrast_vs_p30"]["all_treatment_commands_compared"]
            for row in rows
        ),
        "zero_projection_conflicts": not any(
            row["command_contrast_vs_p30"]["any_projection_conflict"]
            for row in rows
        ),
        "projection_idempotence": all(
            row["command_contrast_vs_p30"]["maximum_idempotence_error_mps2"]
            <= IDEMPOTENCE_TOLERANCE_MPS2
            for row in rows
        ),
        "all_local_safety_complete_and_bounded": all(
            row["local_safety_complete_and_bounded"] for row in rows
        ),
        "all_command_jerk_within_0p5": all(
            row["command_contrast_vs_p30"]["maximum_command_jerk_mps3"]
            <= 0.5 + 1e-9
            for row in rows
        ),
        "aggregate_severe_exposure_not_worse_than_p30": all(
            sum(
                row["local_safety"]["stable_upstream_cohort"][
                    "severe_braking_exposure_steps"
                ]
                + row["local_safety"]["dynamic_immediate_follower"][
                    "severe_braking_exposure_steps"
                ]
                for row in strength_rows[strength]
            )
            <= sum(
                row["p30_local_safety"]["stable_upstream_cohort"][
                    "severe_braking_exposure_steps"
                ]
                + row["p30_local_safety"]["dynamic_immediate_follower"][
                    "severe_braking_exposure_steps"
                ]
                for row in strength_rows[strength]
            )
            for strength in STRENGTH_BY_MODE.values()
        ),
    }
    action_credit_gates = {
        "at_least_two_intermediate_strengths_distinct_in_five_cases": sum(
            count >= 5 for count in distinct_case_counts.values()
        )
        >= 2,
        "three_projected_levels_in_five_cases": three_level_cases >= 5,
    }
    oracle_gates = {
        "blend_increment_in_three_cases": len(actionable) >= 3,
        "blend_increment_spans_two_seeds": len({row["seed"] for row in actionable})
        >= 2,
        "actionable_winners_use_two_strengths": len(
            {row["deployable_oracle_strength"] for row in actionable}
        )
        >= 2,
        "actionable_winners_span_three_seeds": len(
            {row["seed"] for row in actionable}
        )
        >= 3,
        "oracle_mean_beats_best_constant_by_0p02": (
            mean(deployable_scores) > best_constant_mean + 0.02
        ),
        "two_p30_harms_rescued_by_intermediate": sum(
            row["p30_harm_rescued_by_intermediate"] for row in oracle_rows
        )
        >= 2,
        "one_p30_harm_made_helpful_by_intermediate": any(
            row["p30_harm_made_helpful_by_intermediate"] for row in oracle_rows
        ),
        "active_oracle_harmful_count_at_most_three": sum(
            score < -0.02 for score in active_scores
        )
        <= 3,
        "active_oracle_p10_not_below_p30": float(
            np.percentile(active_scores, 10.0)
        )
        >= float(p30_metrics["p10_effect"]),
        "active_oracle_below_minus_0p05_at_most_three": sum(
            score < -0.05 for score in active_scores
        )
        <= 3,
        "active_oracle_none_below_minus_0p20": not any(
            score < -0.20 for score in active_scores
        ),
        "active_oracle_helpful_in_three_seeds": len(active_helpful_seeds) >= 3,
        "active_oracle_nonbeneficial_in_three_seeds": len(
            active_nonbeneficial_seeds
        )
        >= 3,
    }
    gates = {
        **identity_gates,
        **mechanics_safety_gates,
        **action_credit_gates,
        **oracle_gates,
    }
    return {
        "schema": SCHEMA,
        "development_only": True,
        "action_set": ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
        "blend_contract": IDM_P30_PROJECTED_BLEND_V1.to_dict(),
        "case_count": len(by_case),
        "case_ids": sorted(case_ids),
        "source_rows": source_rows,
        "p30_metrics": p30_metrics,
        "fixed_strength_metrics": fixed_metrics,
        "distinct_case_counts": distinct_case_counts,
        "three_level_case_count": three_level_cases,
        "oracle_metrics": {
            "active_harmful_count": sum(score < -0.02 for score in active_scores),
            "active_mean_effect": mean(active_scores),
            "active_p10_effect": float(np.percentile(active_scores, 10.0)),
            "deployable_mean_effect": mean(deployable_scores),
            "best_constant_mean_effect": best_constant_mean,
            "blend_increment_case_count": len(actionable),
            "blend_increment_seeds": sorted({row["seed"] for row in actionable}),
            "rescued_p30_harm_count": sum(
                row["p30_harm_rescued_by_intermediate"] for row in oracle_rows
            ),
            "helpful_rescued_p30_harm_count": sum(
                row["p30_harm_made_helpful_by_intermediate"]
                for row in oracle_rows
            ),
        },
        "identity_gates": identity_gates,
        "mechanics_safety_gates": mechanics_safety_gates,
        "action_credit_gates": action_credit_gates,
        "oracle_gates": oracle_gates,
        "authority_pass": bool(rows) and all(gates.values()),
        "observation_reward_audit_authorized": bool(rows) and all(gates.values()),
        "rl_training_authorized": False,
        "case_strength_rows": rows,
        "oracle_rows": oracle_rows,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_outputs(output: Path, payload: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", dict(payload))
    _write_jsonl(output / "case_strengths.jsonl", payload["case_strength_rows"])
    _write_jsonl(output / "oracle_cases.jsonl", payload["oracle_rows"])
    with (output / "command_diversity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "seed",
                "strength",
                "distinct_command_steps",
                "maximum_absolute_acceleration_delta_mps2",
                "meaningfully_distinct",
            ),
        )
        writer.writeheader()
        for row in payload["case_strength_rows"]:
            contrast = row["command_contrast_vs_p30"]
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "seed": row["seed"],
                    "strength": row["strength"],
                    "distinct_command_steps": contrast["distinct_command_steps"],
                    "maximum_absolute_acceleration_delta_mps2": contrast[
                        "maximum_absolute_acceleration_delta_mps2"
                    ],
                    "meaningfully_distinct": contrast["meaningfully_distinct"],
                }
            )
    failed = [
        key
        for group in (
            payload["identity_gates"],
            payload["mechanics_safety_gates"],
            payload["action_credit_gates"],
            payload["oracle_gates"],
        )
        for key, value in group.items()
        if not value
    ]
    summary_md = (
        "# IDM-to-P30 projected-blend authority screen\n\n"
        f"- Cases: {payload['case_count']}\n"
        f"- Authority pass: **{payload['authority_pass']}**\n"
        f"- RL training authorized: **{payload['rl_training_authorized']}**\n"
        f"- Failed gates: {', '.join(failed) if failed else 'none'}\n"
    )
    (output / "summary.md").write_text(summary_md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p30-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--blend-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        [path.resolve() for path in args.p30_runs],
        [path.resolve() for path in args.blend_runs],
    )
    _write_outputs(args.output.resolve(), payload)
    failed = [
        key
        for group in (
            payload["identity_gates"],
            payload["mechanics_safety_gates"],
            payload["action_credit_gates"],
            payload["oracle_gates"],
        )
        for key, value in group.items()
        if not value
    ]
    print(
        json.dumps(
            {
                "case_count": payload["case_count"],
                "authority_pass": payload["authority_pass"],
                "p30_metrics": payload["p30_metrics"],
                "fixed_strength_metrics": payload["fixed_strength_metrics"],
                "oracle_metrics": payload["oracle_metrics"],
                "failed_gates": failed,
                "rl_training_authorized": payload["rl_training_authorized"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
