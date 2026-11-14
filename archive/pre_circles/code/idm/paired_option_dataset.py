"""Frozen records for the decentralized IDM binary option problem.

Each row contains a treatment-independent 134-field actor observation and the
paired reward of applying the P30 option.  Demand, seed, penetration and
outcomes are retained for grouped evaluation, but are deliberately excluded
from :attr:`PairedOptionRecord.observation`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from idm.option_reward import IDM_BINARY_OPTION_REWARD_V1, binary_option_reward


SCHEMA = "idm_paired_option_dataset/1"
RECORD_SCHEMA = "idm_paired_option_record/1"
PHYSICAL_OBSERVATION_DIM = 134
MASKED_OBSERVATION_DIM = 136
EXPECTED_ACTION_SET = "activation_speed_ceiling_v1"
OPTION_CONTEXT_NAMES = (
    "configured_penetration_over_0p30",
    "predicted_dose_over_20s",
    "runway_over_1000m",
    "distance_to_merge_over_2000m",
    "disturbance_distance_to_merge_over_2000m",
    "front_secure_headroom_over_100m",
    "rear_secure_headroom_over_100m",
    "prospective_rear_headroom_over_100m",
    "front_ttc_capped_60s_over_30s",
    "rear_ttc_capped_60s_over_30s",
    "reference_speed_over_35mps",
    "nominal_acceleration_over_3mps2",
    "lane_index_over_two",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def _reference_by_demand(payload: Mapping[str, Any]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in (payload.get("refs") or payload).values():
        demand = int(row["demand_mainline_vph"])
        result[demand] = {
            "severe": float(row["corridor_severe_braking_count_ref_per_s"]),
            "energy": float(row["corridor_acceleration_energy_sum_ref_m2ps4"]),
        }
    return result


def option_context_from_candidate(
    candidate: Mapping[str, Any], penetration: float
) -> tuple[float, ...]:
    """Return prespecified causal option-feasibility fields.

    Every input exists before the action and is already computed by candidate
    admission or the analytical preview.  No realized dose or branch outcome
    is admitted.
    """

    def number(name: str, default: float = 0.0) -> float:
        value = candidate.get(name)
        return float(default if value is None else value)

    def ttc(name: str) -> float:
        value = candidate.get(name)
        return min(max(float(value), 0.0), 60.0) / 30.0 if value is not None else 2.0

    lane_id = str(candidate.get("lane_id", ""))
    try:
        lane_index = int(lane_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        lane_index = 0
    front_headroom = number("front_gap_m") - number("front_secure_gap_m")
    rear_headroom = number("rear_gap_m") - number("rear_secure_gap_m")
    values = (
        float(penetration) / 0.30,
        number("predicted_dose_s") / 20.0,
        number("runway_m") / 1000.0,
        number("distance_to_merge_m") / 2000.0,
        number("disturbance_distance_to_merge_m") / 2000.0,
        front_headroom / 100.0,
        rear_headroom / 100.0,
        number("prospective_rear_headroom_m") / 100.0,
        ttc("front_ttc_s"),
        ttc("rear_ttc_s"),
        number("preview_reference_speed_mps") / 35.0,
        float(np.clip(number("preview_nominal_acceleration_mps2") / 3.0, -2.0, 2.0)),
        float(lane_index) / 2.0,
    )
    if len(values) != len(OPTION_CONTEXT_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("option context is incomplete or non-finite")
    return values


@dataclass(frozen=True)
class PairedOptionRecord:
    sample_id: str
    source: str
    demand: int
    seed: int
    penetration: float
    decision_index: int
    decision_step: int
    vehicle_id: str
    observation: tuple[float, ...]
    activate_reward: float
    physical_utility: float
    constraints_pass: bool
    classification: str
    dose_steps: int

    def __post_init__(self) -> None:
        if len(self.observation) != PHYSICAL_OBSERVATION_DIM:
            raise ValueError(
                f"{self.sample_id}: expected {PHYSICAL_OBSERVATION_DIM} physical fields, "
                f"got {len(self.observation)}"
            )
        if not np.all(np.isfinite(np.asarray(self.observation, dtype=np.float64))):
            raise ValueError(f"{self.sample_id}: observation contains non-finite values")
        if not math.isfinite(self.activate_reward) or not math.isfinite(self.physical_utility):
            raise ValueError(f"{self.sample_id}: reward/effect must be finite")
        if not 0.0 < self.penetration <= 1.0:
            raise ValueError(f"{self.sample_id}: penetration must lie in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = RECORD_SCHEMA
        payload["observation"] = list(self.observation)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedOptionRecord":
        if payload.get("schema") != RECORD_SCHEMA:
            raise ValueError(f"unexpected record schema {payload.get('schema')!r}")
        values = dict(payload)
        values.pop("schema", None)
        values["observation"] = tuple(float(value) for value in values["observation"])
        return cls(**values)


def _branch(root: Path, case_id: str, arm: str) -> dict[str, Any]:
    path = root / "branches" / f"{case_id}.{arm}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return _read(path)


def records_from_roots(
    roots: Sequence[Path], refs_path: Path
) -> tuple[list[PairedOptionRecord], dict[str, Any]]:
    """Reconstruct paired training rows from authoritative SUMO branches."""

    refs_path = Path(refs_path).resolve()
    refs_sha256 = _sha256_bytes(refs_path.read_bytes())
    references = _reference_by_demand(_read(refs_path))
    records: list[PairedOptionRecord] = []
    source_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_root in roots:
        root = Path(raw_root).resolve()
        summary = _read(root / "summary.json")
        protocol = dict(summary["protocol"])
        demand = int(protocol["demand"])
        if demand not in references:
            raise ValueError(f"{root}: no reward reference for demand {demand}")
        required = {
            "action_set": protocol.get("action_set") == EXPECTED_ACTION_SET,
            "option_steps": int(protocol.get("option_steps", -1))
            == IDM_BINARY_OPTION_REWARD_V1.option_steps,
            "safety_horizon_steps": int(protocol.get("safety_horizon_steps", -1))
            == IDM_BINARY_OPTION_REWARD_V1.horizon_steps,
            "service_horizon_steps": int(protocol.get("service_horizon_steps", -1))
            == IDM_BINARY_OPTION_REWARD_V1.horizon_steps,
            "reward_reference": protocol.get("reward_refs_sha256") == refs_sha256,
        }
        if not all(required.values()):
            raise ValueError(f"{root}: incompatible paired protocol {required}")
        penetration = float(protocol["cav_penetration"])
        refs = references[demand]
        source_rows.append(
            {
                "root": str(root),
                "demand": demand,
                "penetration": penetration,
                "case_count": len(summary["cases"]),
                "protocol_identity": required,
            }
        )

        for case in summary["cases"]:
            if not bool(case.get("all_branch_decision_identity")) or not bool(
                case.get("all_discovery_identity")
            ):
                raise ValueError(f"{root}: branch identity failed for {case['case_id']}")
            seed = int(case["seed"])
            discovery = _read(root / "discoveries" / f"m{demand}.seed{seed}.json")
            decision = discovery.get("decision")
            if decision is None or int(decision["decision_step"]) != int(
                case["decision_step"]
            ):
                raise ValueError(f"{root}: missing matching discovery for seed {seed}")
            control = _branch(root, str(case["case_id"]), "abstain")

            for effect in case["effects"]:
                arm = str(effect["arm"])
                action = _branch(root, str(case["case_id"]), arm)
                slot = int(effect["candidate_slot"])
                candidate = decision["candidates"][slot - 1]
                vehicle_id = str(candidate["vehicle_id"])
                masked = tuple(
                    float(value) for value in decision["candidate_observations"][vehicle_id]
                )
                if len(masked) != MASKED_OBSERVATION_DIM:
                    raise ValueError(
                        f"{root}: expected {MASKED_OBSERVATION_DIM} fields, got {len(masked)}"
                    )
                if tuple(masked[-2:]) != (1.0, 1.0):
                    raise ValueError(
                        f"{root}: stored decision {case['case_id']} was not binary eligible"
                    )
                computed = binary_option_reward(
                    control,
                    action,
                    severe_reference_per_step=refs["severe"],
                    energy_reference_per_step=refs["energy"],
                )
                stored_effect = float(effect["safety_composite_reference_units"])
                if abs(float(computed["physical_utility"]) - stored_effect) > 1e-12:
                    raise ValueError(f"{root}: effect reproduction failed for {arm}")
                if computed["constraints"] != effect["constraints"]:
                    raise ValueError(f"{root}: constraint reproduction failed for {arm}")
                sample_id = (
                    f"m{demand}.p{penetration:.3f}.seed{seed}."
                    f"d{int(decision.get('decision_index', 1))}.slot{slot}"
                )
                if sample_id in seen:
                    raise ValueError(f"duplicate paired sample {sample_id}")
                seen.add(sample_id)
                records.append(
                    PairedOptionRecord(
                        sample_id=sample_id,
                        source=str(root),
                        demand=demand,
                        seed=seed,
                        penetration=penetration,
                        decision_index=int(decision.get("decision_index", 1)),
                        decision_step=int(decision["decision_step"]),
                        vehicle_id=vehicle_id,
                        observation=masked[:PHYSICAL_OBSERVATION_DIM],
                        activate_reward=float(computed["activate_reward"]),
                        physical_utility=float(computed["physical_utility"]),
                        constraints_pass=bool(computed["constraints_pass"]),
                        classification=str(effect["classification"]),
                        dose_steps=int(effect["target_command_step_count"]),
                    )
                )

    records.sort(key=lambda row: row.sample_id)
    manifest_core = {
        "schema": SCHEMA,
        "reward_contract": IDM_BINARY_OPTION_REWARD_V1.to_dict(),
        "reward_refs_path": str(refs_path),
        "reward_refs_sha256": refs_sha256,
        "physical_observation_dim": PHYSICAL_OBSERVATION_DIM,
        "actor_excludes": [
            "demand",
            "seed",
            "penetration",
            "decision_index",
            "decision_step",
            "vehicle_id",
            "dose_steps",
            "constraints",
            "classification",
            "outcomes",
        ],
        "sources": source_rows,
        "record_count": len(records),
        "record_hashes": [_canonical_hash(row.to_dict()) for row in records],
    }
    manifest = {**manifest_core, "dataset_sha256": _canonical_hash(manifest_core)}
    return records, manifest


def write_dataset(
    output: Path,
    records: Sequence[PairedOptionRecord],
    manifest: Mapping[str, Any],
    *,
    train_seeds: Iterable[int],
    validation_seeds: Iterable[int],
) -> dict[str, Any]:
    """Write an immutable JSONL dataset with disjoint grouped seed splits."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = {int(seed) for seed in train_seeds}
    validation = {int(seed) for seed in validation_seeds}
    if not train or not validation or train & validation:
        raise ValueError("train/validation seeds must be non-empty and disjoint")
    observed = {row.seed for row in records}
    if not observed <= train | validation:
        raise ValueError(f"records contain unassigned seeds {sorted(observed - train - validation)}")
    if not (observed & train) or not (observed & validation):
        raise ValueError("both grouped splits must contain records")

    lines = []
    counts = {"train": 0, "validation": 0}
    for row in records:
        payload = row.to_dict()
        payload["split"] = "train" if row.seed in train else "validation"
        counts[payload["split"]] += 1
        lines.append(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    records_bytes = "".join(lines).encode("utf-8")
    records_path = output / "records.jsonl"
    temporary = output / f"records.jsonl.tmp"
    temporary.write_bytes(records_bytes)
    temporary.replace(records_path)
    payload = {
        **dict(manifest),
        "records_sha256": _sha256_bytes(records_bytes),
        "train_seeds": sorted(train),
        "validation_seeds": sorted(validation),
        "split_counts": counts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_dataset(path: Path, split: str | None = None) -> tuple[list[PairedOptionRecord], dict]:
    root = Path(path)
    manifest = _read(root / "manifest.json")
    records_bytes = (root / "records.jsonl").read_bytes()
    if _sha256_bytes(records_bytes) != manifest["records_sha256"]:
        raise ValueError("paired option dataset record hash mismatch")
    records = []
    for line in records_bytes.decode("utf-8").splitlines():
        payload = json.loads(line)
        row_split = payload.pop("split")
        if split is None or row_split == split:
            records.append(PairedOptionRecord.from_dict(payload))
    if not records:
        raise ValueError(f"paired option dataset has no records for split {split!r}")
    return records, manifest
