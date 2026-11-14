"""Isolated v5.8 draft contract for a future two-wall coordinated treatment.

Status: draft, nonexecuted, no feasibility or outcome claim
-------------------------------------------------------------

This module mirrors the shape of the frozen v5.7.1 contract only syntactically.
It is intentionally NOT an executable scientific contract: the v5.6 Candidate F
multiwall feasibility audit established only that the four development cases
contain enough CAVs numerically.  Two spatially separated full-lane cohorts,
safe front/rear gaps, minimum inter-wall spacing, storage capacity between
walls, independent release safety and noninteracting treatment effects are all
unestablished.  Consequently the scientific fields below are kept unresolved
and must NOT be guessed from stored outcomes.

Mechanically defined (fixed by this skeleton):

- maximum of two active cohorts;
- disjoint membership (never two walls simultaneously);
- damping-preserving action composition;
- independent lifecycle state per cohort;
- hard-safety priority over HOLD duration, spacing and taper preferences;
- deterministic, JSON-serializable diagnostics;
- no simultaneous CAV ownership (primary or reserve);
- atomic reserve transfer.

Scientifically unresolved (must be bound only after a treatment-blind
feasibility audit):

- minimum inter-wall spacing;
- assembly zones;
- HOLD zones;
- activation offset;
- per-wall authority;
- combined authority budget;
- storage threshold;
- release order;
- minimum release separation;
- candidate-selection rule;
- number of CAVs per lane;
- treatment duration.

``draft_manifest()`` is serializable but never executable.  Only
``executable_profile()`` with every scientific field bound and validated
returns a profile carrying a deterministic SHA-256 fingerprint; an unresolved
profile is never fingerprinted and never described as a scientific treatment.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping


CONTRACT_VERSION = "idm_multiwall/5.8-draft"
DRAFT_PROFILE_ID = "idm_v5_8_multiwall_draft"
EXECUTABLE_PROFILE_ID = "idm_v5_8_multiwall_executable_profile"

MAX_ACTIVE_COHORTS = 2

# Mechanically defined fields: the skeleton fixes these regardless of the
# scientific choices that remain unresolved.
MECHANICAL_SPEC: dict[str, Any] = {
    "max_active_cohorts": MAX_ACTIVE_COHORTS,
    "action_composition": "damping_preserving_copy_then_override_active_members",
    "nonmember_actions": "identical_supplied_damping_payloads",
    "membership": "disjoint_primary_disjoint_reserve_never_both",
    "lifecycle": ["INACTIVE", "ASSEMBLING", "HOLDING", "TAPERING", "RELEASED", "ABORTED"],
    "lifecycle_independence": "per_cohort_independent_state",
    "hard_safety_priority": "above_hold_duration_spacing_and_taper_preferences",
    "diagnostics": "deterministic_json_serializable",
    "reserve_transfer": "atomic",
    "authority_accounting": "requested_and_realized_per_wall_and_combined",
    "release_mechanism": "validated_enum_and_configuration_fields",
    "scientific_equivalence_claim": "none_a_two_wall_budget_split_is_not_claimed_safe_or_equivalent_to_one_stronger_wall",
}

# Every field in this list must be bound by an executable profile and is
# explicitly unresolved in the draft manifest.  None may be inferred from
# existing outcomes or stored artifacts.
SCIENTIFIC_UNRESOLVED_FIELDS: tuple[str, ...] = (
    "min_inter_wall_spacing_m",        # numerical separation between wall spans
    "assembly_zone_m",                 # upstream band where each cohort assembles
    "hold_zone_m",                     # band where each cohort holds
    "activation_offset_s",             # second cohort activation delay relative to the first
    "per_wall_authority_mps2",         # deceleration magnitude per wall (unresolved values)
    "combined_authority_budget_mps2",  # shared total-authority budget across both walls
    "storage_threshold_m",             # minimum inter-wall storage capacity
    "release_order",                   # downstream_first / upstream_first / external
    "min_release_separation_s",        # minimum time between release initiations
    "candidate_selection_rule",        # which CAVs become members/reserves
    "cavs_per_lane",                   # number of CAVs per occupied lane
    "treatment_duration_s",            # requested treatment duration
)

_RESOLVED_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "min_inter_wall_spacing_m": ("positive",),
    "assembly_zone_m": ("nonnegative",),
    "hold_zone_m": ("nonnegative",),
    "activation_offset_s": ("nonnegative",),
    "per_wall_authority_mps2": ("positive",),
    "combined_authority_budget_mps2": ("positive",),
    "storage_threshold_m": ("positive",),
    "release_order": ("enum",),
    "min_release_separation_s": ("nonnegative",),
    "candidate_selection_rule": ("nonempty_string",),
    "cavs_per_lane": ("positive_int",),
    "treatment_duration_s": ("positive",),
}

_VALID_RELEASE_ORDERS = ("downstream_first", "upstream_first", "external")


class UnresolvedFieldsError(ValueError):
    """Raised when an executable profile is requested while scientific fields
    remain unresolved."""

    def __init__(self, missing: tuple[str, ...], invalid: tuple[str, ...]) -> None:
        self.missing = tuple(missing)
        self.invalid = tuple(invalid)
        problems = " ".join(
            [f"missing={self.missing}" if self.missing else "", f"invalid={self.invalid}" if self.invalid else ""]
        )
        super().__init__(f"multiwall executable profile refused: {problems}".strip())


def draft_manifest() -> dict[str, Any]:
    """Serialize the draft/contract proposal; never executable.

    The manifest carries ``executable: False``, the full list of unresolved
    scientific fields, and an explicit ``profile_sha256: None`` because an
    unresolved draft is not fingerprinted as a scientific treatment.
    """
    return {
        "schema": "idm_v5_8_multiwall_draft_manifest/1",
        "contract_version": CONTRACT_VERSION,
        "profile_id": DRAFT_PROFILE_ID,
        "status": "draft",
        "executable": False,
        "outcome_claim": False,
        "feasibility_claim": False,
        "base_control_arm": "frozen_decentralized_idm_damping_future_freezing_required",
        "mechanical_spec": MECHANICAL_SPEC,
        "scientifically_unresolved_fields": {
            name: _RESOLVED_REQUIREMENTS[name] for name in SCIENTIFIC_UNRESOLVED_FIELDS
        },
        "fresh_seeds_authorized": False,
        "predictor_data_authorized": False,
        "rl_training_authorized": False,
        "sumo_execution_authorized": False,
        "profile_sha256": None,
    }


def _validate_resolved(fields: Mapping[str, Any]) -> tuple[str, ...]:
    missing = tuple(
        name for name in SCIENTIFIC_UNRESOLVED_FIELDS if name not in fields
    )
    invalid: list[str] = []
    for name in SCIENTIFIC_UNRESOLVED_FIELDS:
        if name not in fields:
            continue
        value = fields[name]
        requirement = _RESOLVED_REQUIREMENTS[name]
        try:
            if "positive" in requirement:
                number = float(value)
                if not math.isfinite(number) or number <= 0.0:
                    invalid.append(f"{name}={value!r}")
            elif "nonnegative" in requirement:
                number = float(value)
                if not math.isfinite(number) or number < 0.0:
                    invalid.append(f"{name}={value!r}")
            elif "positive_int" in requirement:
                number = int(value)
                if number <= 0:
                    invalid.append(f"{name}={value!r}")
            elif "nonempty_string" in requirement:
                if not isinstance(value, str) or not value.strip():
                    invalid.append(f"{name}={value!r}")
            elif "enum" in requirement:
                if str(value) not in _VALID_RELEASE_ORDERS:
                    invalid.append(f"{name}={value!r}")
        except (TypeError, ValueError):
            invalid.append(f"{name}={value!r}")
    return tuple(invalid)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def executable_profile(resolved_scientific: Mapping[str, Any]) -> dict[str, Any]:
    """Return an executable (still unexecuted) profile once every scientific
    field is bound and validated.

    Raises ``UnresolvedFieldsError`` when any required scientific field is
    missing or invalid; nothing is fingerprinted in that case.  A fingerprint
    is produced only for a complete, validated profile.
    """
    missing = tuple(
        name for name in SCIENTIFIC_UNRESOLVED_FIELDS if name not in resolved_scientific
    )
    invalid = _validate_resolved(resolved_scientific)
    if missing or invalid:
        raise UnresolvedFieldsError(missing, invalid)
    payload: dict[str, Any] = {
        "schema": "idm_v5_8_multiwall_executable_profile/1",
        "contract_version": CONTRACT_VERSION,
        "profile_id": EXECUTABLE_PROFILE_ID,
        "status": "executable_mechanics_only",
        "executable": True,
        "outcome_claim": False,
        "feasibility_claim": False,
        "scientific_values_bound": True,
        "mechanical_spec": MECHANICAL_SPEC,
        "resolved_scientific": {
            name: resolved_scientific[name] for name in SCIENTIFIC_UNRESOLVED_FIELDS
        },
        "fresh_seeds_authorized": False,
        "predictor_data_authorized": False,
        "rl_training_authorized": False,
        "sumo_execution_authorized": False,
        "profile_sha256": None,
    }
    payload["profile_sha256"] = profile_fingerprint(payload)
    return payload


def profile_fingerprint(profile: dict[str, Any]) -> str:
    """Stable SHA-256 of a fully specified profile dict.

    The ``profile_sha256`` key itself is excluded so the helper is idempotent
    on a fingerprinted profile.
    """
    payload = {key: value for key, value in profile.items() if key != "profile_sha256"}
    return _fingerprint(payload)
