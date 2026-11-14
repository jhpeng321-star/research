"""State-aligned RL timing/candidate option library records.

One row per frozen predecision state; per-state action outcomes for abstain,
candidate-1 P30 and candidate-2 P30 (where feasible), with causal candidate
features and the penetration-free local observation.  No outcome, dose,
reward, release reason or branch identity enters the actor feature set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


RL_LIBRARY_SCHEMA = "idm_rl_timing_option_library/1"
PHYSICAL_OBSERVATION_DIM = 134
ACTIONS = ("abstain", "candidate_1", "candidate_2")
ACTION_BY_SLOT = {1: "candidate_1", 2: "candidate_2"}


def _infeasible(action: str) -> ActionOutcome:
    return ActionOutcome(
        action=action,
        feasible=False,
        reward=0.0,
        physical_utility=0.0,
        constraints_pass=True,
        classification="neutral",
        dose_steps=0,
        vehicle_id=None,
    )


@dataclass(frozen=True)
class ActionOutcome:
    action: str
    feasible: bool
    reward: float
    physical_utility: float
    constraints_pass: bool
    classification: str
    dose_steps: int
    vehicle_id: str | None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}")
        if not isfinite(float(self.reward)) or not isfinite(float(self.physical_utility)):
            raise ValueError(f"{self.action}: non-finite outcome")


@dataclass(frozen=True)
class RLOptionRow:
    sample_id: str
    source: str
    seed: int
    penetration: float
    demand: int
    recurrence: int
    decision_step: int
    observation: tuple[float, ...]
    candidate_features: dict[str, dict[str, Any]]
    pattern_context: dict[str, float]
    actions: dict[str, ActionOutcome]

    def __post_init__(self) -> None:
        if len(self.observation) != PHYSICAL_OBSERVATION_DIM:
            raise ValueError("local observation must be 134 physical fields")
        if not np.all(np.isfinite(np.asarray(self.observation, dtype=np.float64))):
            raise ValueError("observation contains non-finite values")
        if set(self.actions) != set(ACTIONS):
            raise ValueError("a row must define all three actions")
        if not self.actions["abstain"].feasible:
            raise ValueError("abstain is always feasible")

    @property
    def state_id(self) -> tuple[int, float, int, int, int]:
        return (self.seed, round(self.penetration, 6), self.demand, self.recurrence, self.decision_step)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = RL_LIBRARY_SCHEMA
        payload["observation"] = list(self.observation)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RLOptionRow":
        if payload.get("schema") != RL_LIBRARY_SCHEMA:
            raise ValueError(f"unexpected row schema {payload.get('schema')!r}")
        values = dict(payload)
        values.pop("schema", None)
        values["observation"] = tuple(float(value) for value in values["observation"])
        values["actions"] = {
            key: ActionOutcome(**row) for key, row in values["actions"].items()
        }
        return cls(**values)


def rl_rows_from_raw_records(
    records: Iterable[Any], manifest: Mapping[str, Any]
) -> list[RLOptionRow]:
    """Group single-action paired records (abstain is implicit, reward zero)
    into per-state rows with candidate_1/candidate_2 actions."""
    states: dict[tuple[int, float, int, int, int], dict[str, Any]] = {}
    for record in records:
        key = (
            int(record.seed),
            round(float(record.penetration), 6),
            int(record.demand),
            int(record.decision_index),
            int(record.decision_step),
        )
        slot = int(record.sample_id.rsplit("slot", 1)[-1])
        action = ACTION_BY_SLOT[slot]
        entry = states.setdefault(key, {"record": record, "actions": {}})
        entry["actions"][action] = ActionOutcome(
            action=action,
            feasible=True,
            reward=float(record.activate_reward),
            physical_utility=float(record.physical_utility),
            constraints_pass=bool(record.constraints_pass),
            classification=str(record.classification),
            dose_steps=int(record.dose_steps),
            vehicle_id=str(record.vehicle_id),
        )
    rows = []
    for key, entry in states.items():
        record = entry["record"]
        actions = {
            "abstain": ActionOutcome(
                action="abstain",
                feasible=True,
                reward=0.0,
                physical_utility=0.0,
                constraints_pass=True,
                classification="neutral",
                dose_steps=0,
                vehicle_id=None,
            ),
            "candidate_1": entry["actions"].get("candidate_1"),
            "candidate_2": entry["actions"].get("candidate_2") or _infeasible("candidate_2"),
        }
        if actions["candidate_1"] is None:
            raise ValueError(f"state {key} lacks a candidate-1 branch")
        features = {}
        pattern: dict[str, float] = {}
        for slot in (1, 2):
            if ACTION_BY_SLOT[slot] not in entry["actions"]:
                continue
            candidate, context = features_from_source(record.source, slot, record)
            features[ACTION_BY_SLOT[slot]] = candidate
            if slot == 1 and context is not None:
                pattern = dict(context)
        rows.append(
            RLOptionRow(
                sample_id=(
                    f"m{record.demand}.p{record.penetration:.3f}.seed{record.seed}."
                    f"r{record.decision_index}"
                ),
                source=str(record.source),
                seed=int(record.seed),
                penetration=float(record.penetration),
                demand=int(record.demand),
                recurrence=int(record.decision_index),
                decision_step=int(record.decision_step),
                observation=tuple(float(value) for value in record.observation),
                candidate_features=features,
                pattern_context=dict(pattern),
                actions=actions,
            )
        )
    rows.sort(key=lambda row: row.sample_id)
    return rows


def features_from_source(
    source: str, slot: int, record: Any
) -> tuple[dict[str, Any], dict[str, float] | None]:
    """Extract causal candidate features from the frozen discovery decision."""
    root = Path(source)
    discovery = json.loads(
        (root / "discoveries" / f"m{record.demand}.seed{record.seed}.json").read_text(
            encoding="utf-8"
        )
    )
    decision = discovery.get("decision") or {}
    candidates = decision.get("candidates") or []
    candidate = next(
        (row for row in candidates if int(row.get("slot", -1)) == slot), None
    )
    if candidate is None:
        raise ValueError(f"state {record.sample_id} has no candidate slot {slot}")
    previews = candidate.get("terminal_speed_previews") or {}
    p30 = previews.get("p30") or {}
    separation = None
    nominal = candidate.get("preview_nominal_acceleration_mps2")
    desired = p30.get("desired_acceleration_mps2")
    if nominal is not None and desired is not None:
        separation = float(desired) - float(nominal)
    return (
        {
            "slot": int(candidate.get("slot", slot)),
            "runway_m": candidate.get("runway_m"),
            "predicted_dose_s": candidate.get("predicted_dose_s"),
            "front_gap_m": candidate.get("front_gap_m"),
            "rear_gap_m": candidate.get("rear_gap_m"),
            "front_ttc_s": candidate.get("front_ttc_s"),
            "rear_ttc_s": candidate.get("rear_ttc_s"),
            "front_secure_gap_m": candidate.get("front_secure_gap_m"),
            "rear_secure_gap_m": candidate.get("rear_secure_gap_m"),
            "prospective_rear_headroom_m": candidate.get("prospective_rear_headroom_m"),
            "disturbance_distance_to_merge_m": candidate.get(
                "disturbance_distance_to_merge_m"
            ),
            "distance_to_merge_m": candidate.get("distance_to_merge_m"),
            "p30_reference_speed_mps": p30.get("reference_speed_mps"),
            "p30_projected_acceleration_mps2": p30.get("projected_acceleration_mps2"),
            "command_separation_mps2": separation,
            "geometric_rank": candidate.get("geometric_rank"),
            "lane_change_state": candidate.get("lane_change_state"),
        },
        dict(decision.get("pattern_context") or {}),
    )


def load_rl_library(path: Path) -> tuple[list[RLOptionRow], dict[str, Any]]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(RLOptionRow.from_dict(json.loads(line)))
    return records, manifest


def write_rl_library(
    output: Path,
    rows: Sequence[RLOptionRow],
    cores: Mapping[str, Any],
    *,
    train_seeds: Iterable[int],
    validation_seeds: Iterable[int],
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = {int(seed) for seed in train_seeds}
    validation = {int(seed) for seed in validation_seeds}
    if train & validation:
        raise ValueError("training and validation seeds must be disjoint")
    for row in rows:
        if row.seed not in (train | validation):
            raise ValueError(f"row {row.sample_id} belongs to no frozen split")
    lines = [json.dumps(row.to_dict()) for row in rows]
    records_bytes = "\n".join(lines).encode("utf-8")
    records_sha256 = hashlib.sha256(records_bytes).hexdigest()
    core = {
        "schema": RL_LIBRARY_SCHEMA,
        "train_seeds": sorted(train),
        "validation_seeds": sorted(validation),
        "record_count": len(rows),
        "records_sha256": records_sha256,
        "actor_excludes": [
            "seed",
            "demand",
            "penetration",
            "recurrence",
            "decision_step",
            "vehicle_id",
            "dose_steps",
            "constraints",
            "classification",
            "reward",
            "release reason",
            "future/postdecision measurements",
        ],
    }
    manifest = {**cores, **core, "dataset_sha256": hashlib.sha256(
        json.dumps(core, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    (output / "records.jsonl").write_text(records_bytes.decode("utf-8"), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest
