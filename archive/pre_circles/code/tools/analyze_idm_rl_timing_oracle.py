"""Retrospective oracle headroom for P30 timing/candidate selection.

Reads the state-aligned RL option library and computes the frozen baselines
(never, always candidate-1, always candidate-2, fixed 5-10% rule, best fixed
recurrence, best fixed rank rule) and the constrained state-wise oracle, then
applies the frozen oracle-headroom gates of
docs/IDM_RL_TIMING_CANDIDATE_HEADROOM_PROTOCOL.md.  Oracle bounds are
unattainable retrospective evidence only; they never authorize training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.io import atomic_write_json
from idm.rl_option_library import ACTIONS, RLOptionRow, load_rl_library

DEADBAND = 0.02
MINIMUM_DOSE_STEPS = 16
ORACLE_GAIN_GATE = 0.02
MEANINGFUL_SEED_MARGIN = 0.02


def action_outcome(row: RLOptionRow, action: str) -> Mapping[str, Any]:
    return row.actions[action]


def policy_metrics(rows: list[RLOptionRow], selected: list[str]) -> dict[str, Any]:
    rewards = np.asarray([row.actions[action].reward for row, action in zip(rows, selected)], dtype=float)
    effects = np.asarray(
        [row.actions[action].physical_utility for row, action in zip(rows, selected)],
        dtype=float,
    )
    counts = Counter(
        row.actions[action].classification for row, action in zip(rows, selected)
    )
    failures = [
        row.sample_id
        for row, action in zip(rows, selected)
        if action != "abstain" and not row.actions[action].constraints_pass
    ]
    actions_used = Counter(action for action in selected)
    nonabstain = Counter(action for action in selected if action != "abstain")
    return {
        "mean_policy_reward": float(np.mean(rewards)),
        "mean_policy_physical_effect": float(np.mean(effects)),
        "helpful": int(counts["helpful"]),
        "harmful": int(counts["harmful"]),
        "neutral": int(counts["neutral"]),
        "nonabstain_activations": int(sum(nonabstain.values())),
        "harmful_fraction_of_activations": (
            float(counts["harmful"] / max(sum(nonabstain.values()), 1))
        ),
        "actions_used": dict(actions_used),
        "safety_service_failures": failures,
        "constraint_failure_count": len(failures),
    }


def fixed_rule_selection(rows: list[RLOptionRow], action: str) -> list[str]:
    return [
        action if row.actions[action].feasible else "abstain" for row in rows
    ]


def oracle_selection(rows: list[RLOptionRow], *, constrained: bool, dose_limited: bool) -> list[str]:
    selected = []
    for row in rows:
        best, best_score = "abstain", 0.0
        for action in ("candidate_1", "candidate_2"):
            outcome = row.actions[action]
            if not outcome.feasible:
                continue
            if constrained and not outcome.constraints_pass:
                continue
            if dose_limited and outcome.dose_steps < MINIMUM_DOSE_STEPS:
                continue
            if float(outcome.reward) > best_score:
                best, best_score = action, float(outcome.reward)
        selected.append(best)
    return selected


def best_fixed_recurrence(
    train: list[RLOptionRow], validation: list[RLOptionRow]
) -> tuple[list[str], dict[str, float]]:
    """Per-cell recurrence rule selected on training rows only."""
    per_cell: dict[tuple[int, float], dict[int, list[float]]] = {}
    for row in train:
        key = (row.demand, round(row.penetration, 6))
        per_cell.setdefault(key, {}).setdefault(row.recurrence, []).append(
            float(row.actions["candidate_1"].reward)
        )
    rule: dict[tuple[int, float], int] = {}
    scores: dict[str, float] = {}
    for key, recurrences in per_cell.items():
        best_recurrence = max(
            recurrences, key=lambda r: float(np.mean(recurrences[r]))
        )
        rule[key] = best_recurrence
        scores[f"m{key[0]}_p{key[1]:.2f}"] = float(
            np.mean(recurrences[best_recurrence])
        )
    selection = []
    for row in validation:
        key = (row.demand, round(row.penetration, 6))
        if key in rule and row.recurrence == rule[key]:
            selection.append("candidate_1" if row.actions["candidate_1"].feasible else "abstain")
        else:
            selection.append("abstain")
    return selection, scores


def analyze(rows: list[RLOptionRow], train_seeds: set[int], validation_seeds: set[int]) -> dict[str, Any]:
    train = [row for row in rows if row.seed in train_seeds]
    validation = [row for row in rows if row.seed in validation_seeds]
    baselines: dict[str, dict[str, Any]] = {
        "never_activate": policy_metrics(validation, ["abstain"] * len(validation)),
        "always_candidate_1": policy_metrics(
            validation, fixed_rule_selection(validation, "candidate_1")
        ),
        "always_candidate_2": policy_metrics(
            validation, fixed_rule_selection(validation, "candidate_2")
        ),
        "fixed_5_10_rule": policy_metrics(validation, fixed_rule_selection(validation, "candidate_1")),
    }
    candidate_2_qualified = sum(
        1 for row in validation if row.actions["candidate_2"].feasible
    )
    oracles = {
        "state_wise_unconstrained": oracle_selection(validation, constrained=False, dose_limited=False),
        "constrained": oracle_selection(validation, constrained=True, dose_limited=False),
        "constrained_adequate_dose": oracle_selection(validation, constrained=True, dose_limited=True),
    }
    fixed_best_action = max(
        ("candidate_1", "candidate_2"),
        key=lambda action: np.mean(
            [
                row.actions[action].reward
                for row in train
                if row.actions[action].feasible
            ]
        )
        if any(row.actions[action].feasible for row in train)
        else float("-inf"),
    )
    fixed_best = policy_metrics(
        validation, fixed_rule_selection(validation, fixed_best_action)
    )
    baselines["best_fixed_rank_rule"] = fixed_best
    recurrence_rule, recurrence_scores = best_fixed_recurrence(train, validation)
    baselines["best_fixed_recurrence"] = policy_metrics(validation, recurrence_rule)
    baselines["recurrence_scores"] = recurrence_scores

    def _best_fixed():
        return max(
            [
                baselines["fixed_5_10_rule"],
                baselines["best_fixed_rank_rule"],
                baselines["best_fixed_recurrence"],
            ],
            key=lambda row: row["mean_policy_reward"],
        )

    best_fixed = _best_fixed()
    oracle_reports: dict[str, dict[str, Any]] = {}
    for name, selection in oracles.items():
        metrics = policy_metrics(validation, selection)
        gain = float(metrics["mean_policy_reward"] - best_fixed["mean_policy_reward"])
        per_seed: dict[int, float] = {}
        seen_seeds = sorted({row.seed for row in validation})
        for seed in seen_seeds:
            rows_seed = [row for row in validation if row.seed == seed]
            sel_seed = [a for row, a in zip(validation, selection) if row.seed == seed]
            mean_seed = float(
                np.mean(
                    [
                        row.actions[a].reward
                        for row, a in zip(rows_seed, sel_seed)
                    ]
                )
            )
            mean_rule_seed = float(
                np.mean(
                    [
                        row.actions["candidate_1"].reward
                        for row in rows_seed
                        if row.actions["candidate_1"].feasible
                    ]
                )
            )
            per_seed[seed] = mean_seed - mean_rule_seed
        meaningful_seeds = sorted(
            seed
            for seed, delta in per_seed.items()
            if delta >= MEANINGFUL_SEED_MARGIN
        )
        options_used = {
            action: count
            for action, count in Counter(selection).items()
            if action != "abstain"
        }
        top_seed = max(per_seed, key=per_seed.get) if per_seed else None
        gain_without_top = (
            gain - per_seed[top_seed] if top_seed is not None and gain >= 0 else None
        )
        helpful = metrics["helpful"]
        harmful = metrics["harmful"]
        neutral = metrics["neutral"]
        oracle_reports[name] = {
            **metrics,
            "gain_over_best_fixed_rule": gain,
            "gain_contributing_seeds": {
                str(seed): round(value, 5)
                for seed, value in sorted(per_seed.items())
                if value > 1e-9
            },
            "meaningful_improvement_seeds": meaningful_seeds,
            "top_contributing_seed": top_seed,
            "gain_without_top_seed": gain_without_top,
            "nonabstain_options_used": len(options_used),
            "options": options_used,
            "outcome_diversity": {"helpful": helpful, "harmful": harmful, "neutral": neutral},
            "gates": {
                "improves_best_fixed_rule_by_0p02": gain >= ORACLE_GAIN_GATE,
                "meaningful_improvement_in_at_least_three_seeds": len(meaningful_seeds) >= 3,
                "uses_more_than_one_nonabstain_option": len(options_used) > 1,
                "not_single_outlier": (
                    gain_without_top is None or gain_without_top >= 0.0
                ),
                "adequate_action_and_outcome_diversity": (
                    len(options_used) > 1 and helpful >= 3 and harmful >= 3
                ),
                "preserves_hard_constraints": metrics["constraint_failure_count"] == 0,
            },
        }
        oracle_reports[name]["gates"]["effects_exceed_replay_noise"] = _replay_noise_ok(
            rows, train_seeds, validation_seeds
        )
        oracle_reports[name]["passes"] = all(oracle_reports[name]["gates"].values())
    return {
        "schema": "idm_rl_timing_oracle_headroom/1",
        "protocol": "docs/IDM_RL_TIMING_CANDIDATE_HEADROOM_PROTOCOL.md",
        "row_count": len(rows),
        "train_row_count": len(train),
        "validation_row_count": len(validation),
        "candidate_2_feasible_validation_rows": candidate_2_qualified,
        "best_fixed_deployable_rule": fixed_best_action,
        "baselines": baselines,
        "best_fixed": {
            "name": (
                "fixed_5_10_rule"
                if best_fixed is baselines["fixed_5_10_rule"]
                else "best_fixed_rank_rule"
            ),
            **best_fixed,
        },
        "oracles": oracle_reports,
        "headroom_passes": any(report["passes"] for report in oracle_reports.values()),
    }


def _replay_noise_ok(rows: list[RLOptionRow], train_seeds: set[int], validation_seeds: set[int]) -> bool:
    sources = {str(row.source) for row in rows}
    for source in sources:
        path = Path(source) / "summary.json"
        if not path.exists():
            return False
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not bool(summary.get("smoke_gates", {}).get("abstain_replay_exact", False)):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = load_rl_library(args.library.resolve())
    result = analyze(rows, set(int(s) for s in manifest["train_seeds"]), set(int(s) for s in manifest["validation_seeds"]))
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True)[:5000])


if __name__ == "__main__":
    main()
