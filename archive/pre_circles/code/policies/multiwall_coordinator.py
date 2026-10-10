"""Pure state-management and command-composition skeleton for a future two-wall
coordinated moving-bottleneck treatment.

Preparatory engineering only
----------------------------

This module is a pure-Python state machine and action-composition layer for a
hypothetical future two-cohort coordinated moving-bottleneck treatment.  It
makes no feasibility, safety, or efficacy claim about such a treatment.  The
prior multiwall audit (docs/IDM_V5_6_MULTIWALL_FEASIBILITY_AUDIT.md)
established only that four development cases contain enough CAVs numerically;
it did not establish spatially separated cohorts, safe gaps, inter-wall
storage, or noninteracting releases.  None of those parameters are guessed
here.

Boundaries
----------

- The coordinated layer *never* computes the decentralized damping action.
  It receives an already-computed damping action map and returns a copy of it,
  replacing only the actions of explicitly active wall members with
  externally supplied wall-command payloads.
- Damping and wall-command payloads are opaque: arbitrary, safely copied and
  never mutated.  No single command representation is assumed.
- The passive counterfactual policy (``policies/counterfactual_escalation.py``)
  is not imported.  TraCI, SUMO, Ray and RLlib are not imported.
- The active v5.7.1 Gate A experiment, its contract, runner and output
  directory are out of scope and untouched.

Scientific versus mechanical
-----------------------------

The *mechanical* rules below are fixed by this skeleton: at most two active
cohorts, disjoint ownership, damping-preserving composition, independent
lifecycle per cohort, hard-safety priority, deterministic JSON diagnostics,
and no simultaneous CAV ownership.

Inter-wall spacing, storage threshold, per-wall and combined authority,
release order and release separation are *scientifically unresolved*
configuration fields.  Each is ``None`` by default (reported as "unresolved"
in diagnostics) and may only be bound by an executable profile whose
feasibility has been established (see the v5.8 draft contract).  The skeleton
expresses the future hypothesis that two mild walls share a bounded authority
budget while making no claim that this is safe or equivalent to one stronger
wall.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CohortLifecycle",
    "ReleaseOrder",
    "SafeAbortPolicy",
    "SpatialGeometry",
    "SafetyStatus",
    "LaneCoverage",
    "CohortState",
    "MultiwallConfig",
    "ComposeResult",
    "MultiwallCoordinator",
    "MultiwallError",
    "OwnershipError",
    "AuthorityError",
    "LifecycleTransitionError",
    "SpatialConflictError",
    "ReleaseConflictError",
]

_SMALL_EPS = 1.0e-9

_ACTIVE_STATES = frozenset(
    {"ASSEMBLING", "HOLDING", "TAPERING"}
)


class CohortLifecycle(str, Enum):
    """One cohort lifecycle state, independent per wall cohort."""

    INACTIVE = "INACTIVE"
    ASSEMBLING = "ASSEMBLING"
    HOLDING = "HOLDING"
    TAPERING = "TAPERING"
    RELEASED = "RELEASED"
    ABORTED = "ABORTED"


class ReleaseOrder(str, Enum):
    """Mechanism for ordering release initiation; the scientific choice is a
    config value, not frozen here."""

    DOWNSTREAM_FIRST = "downstream_first"
    UPSTREAM_FIRST = "upstream_first"
    EXTERNAL = "external"


class SafeAbortPolicy(str, Enum):
    """What a hard-safety failure does to the other cohort.

    ``ABORT_ONLY_COHORT`` (default) keeps the other cohort unchanged;
    ``COORDINATED_ALL`` aborts every active cohort because a declared
    interaction rule requires it.
    """

    ABORT_ONLY_COHORT = "abort_only_cohort"
    COORDINATED_ALL = "coordinated_all"


@dataclass(frozen=True)
class SpatialGeometry:
    """Longitudinal center and span of one cohort.

    ``center_m`` is the longitudinal center expressed as distance to the
    downstream merge point, so a *larger* center is *upstream*.  ``span_m`` is
    the cohort's longitudinal extent, so its downstream edge is
    ``center - span/2`` and its upstream edge is ``center + span/2``.
    """

    center_m: float
    span_m: float = 0.0

    @property
    def downstream_edge_m(self) -> float:
        return float(self.center_m) - float(self.span_m) / 2.0

    @property
    def upstream_edge_m(self) -> float:
        return float(self.center_m) + float(self.span_m) / 2.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "center_m": float(self.center_m),
            "span_m": float(self.span_m),
            "downstream_edge_m": self.downstream_edge_m,
            "upstream_edge_m": self.upstream_edge_m,
        }


_SAFETY_FIELDS: Tuple[str, ...] = (
    "front_safe",
    "rear_safe",
    "ttc_safe",
    "lane_coverage_safe",
    "compactness_safe",
    "member_identity_safe",
    "storage_safe",
    "release_safe",
)


@dataclass(frozen=True)
class SafetyStatus:
    """Externally computed per-cohort safety.

    The coordinator never computes gaps, TTC, or compactness itself.  Every
    field must be passed in by the caller; unknown status is deliberately not
    representable, so the coordinator fails closed (hard-safety abort) when an
    active cohort's safety status is absent.  Hard safety always has priority
    over HOLD duration, spacing objectives, and taper/jerk preferences.
    """

    front_safe: bool
    rear_safe: bool
    ttc_safe: bool
    lane_coverage_safe: bool
    compactness_safe: bool
    member_identity_safe: bool
    storage_safe: bool
    release_safe: bool

    def unsafe_fields(self) -> Tuple[str, ...]:
        return tuple(name for name in _SAFETY_FIELDS if not bool(getattr(self, name)))

    @property
    def is_safe(self) -> bool:
        return not self.unsafe_fields()

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: bool(getattr(self, name)) for name in _SAFETY_FIELDS
        } | {"is_safe": self.is_safe, "unsafe_fields": list(self.unsafe_fields())}


def _safe_status(is_safe: bool) -> SafetyStatus:
    return SafetyStatus(
        front_safe=is_safe,
        rear_safe=is_safe,
        ttc_safe=is_safe,
        lane_coverage_safe=is_safe,
        compactness_safe=is_safe,
        member_identity_safe=is_safe,
        storage_safe=is_safe,
        release_safe=is_safe,
    )


@dataclass(frozen=True)
class LaneCoverage:
    """Lane-coverage report for one cohort, computed from declared members."""

    required_lanes: Tuple[str, ...]
    covered_lanes: Tuple[str, ...]
    coverage_fraction: float
    missing_lanes: Tuple[str, ...]
    duplicate_assignments: Tuple[str, ...]
    full_coverage_retained: bool

    @classmethod
    def compute(
        cls,
        required_lanes: Sequence[str],
        members_by_lane: Mapping[str, Sequence[str]],
    ) -> "LaneCoverage":
        required = tuple(sorted(str(v) for v in required_lanes))
        loaded = {str(v): tuple(str(x) for x in ids) for v, ids in members_by_lane.items()}
        covered = tuple(sorted(str(v) for v, ids in loaded.items() if ids))
        missing = tuple(v for v in required if v not in set(covered))
        seen: Dict[str, str] = {}
        duplicates: list[str] = []
        for lane in sorted(loaded):
            for cav_id in loaded[lane]:
                if cav_id in seen and seen[cav_id] != lane:
                    duplicates.append(cav_id)
                seen.setdefault(cav_id, lane)
        return cls(
            required_lanes=required,
            covered_lanes=covered,
            coverage_fraction=(
                len(tuple(v for v in required if v in set(covered))) / len(required)
                if required
                else 1.0
            ),
            missing_lanes=missing,
            duplicate_assignments=tuple(sorted(set(duplicates))),
            full_coverage_retained=not missing,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_lanes": list(self.required_lanes),
            "covered_lanes": list(self.covered_lanes),
            "coverage_fraction": float(self.coverage_fraction),
            "missing_lanes": list(self.missing_lanes),
            "duplicate_assignments": list(self.duplicate_assignments),
            "full_coverage_retained": bool(self.full_coverage_retained),
        }


@dataclass
class CohortState:
    """Full per-cohort lifecycle state (one per wall; independent)."""

    cohort_id: str
    members_by_lane: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    reserves_by_lane: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    lifecycle: CohortLifecycle = CohortLifecycle.INACTIVE
    activation_step: Optional[int] = None
    activation_time_s: Optional[float] = None
    assembly_elapsed_s: float = 0.0
    hold_elapsed_s: float = 0.0
    taper_elapsed_s: float = 0.0
    release_started_step: Optional[int] = None
    requested_duration_s: Optional[float] = None
    assembly_duration_s: Optional[float] = None
    hold_duration_s: Optional[float] = None
    taper_duration_s: Optional[float] = None
    target_speed_mps: Optional[float] = None
    requested_authority_mps2: Optional[float] = None
    realized_authority_mps2: Optional[float] = None
    authority_decel_limit_mps2: Optional[float] = None
    authority_accel_limit_mps2: Optional[float] = None
    required_lanes: Tuple[str, ...] = ()
    spatial: Optional[SpatialGeometry] = None
    spatial_step: Optional[int] = None
    handoff_count: int = 0
    abort_reason: Optional[str] = None
    release_reason: Optional[str] = None
    last_defer_reason: Optional[str] = None
    events: list[Dict[str, Any]] = field(default_factory=list)
    member_changes: list[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.lifecycle.value in _ACTIVE_STATES

    def active_elapsed_s(self) -> float:
        """Elapsed lifecycle duration (assembly + HOLD + taper)."""
        return (
            float(self.assembly_elapsed_s)
            + float(self.hold_elapsed_s)
            + float(self.taper_elapsed_s)
        )

    def lane_coverage(self) -> LaneCoverage:
        return LaneCoverage.compute(self.required_lanes, self.members_by_lane)

    def to_dict(self) -> Dict[str, Any]:
        coverage = self.lane_coverage().to_dict()
        return {
            "cohort_id": str(self.cohort_id),
            "lifecycle": self.lifecycle.value,
            "members_by_lane": {
                str(k): list(v) for k, v in sorted(self.members_by_lane.items())
            },
            "reserves_by_lane": {
                str(k): list(v) for k, v in sorted(self.reserves_by_lane.items())
            },
            "member_ids": list(self.member_ids()),
            "reserve_ids": list(self.reserve_ids()),
            "assembly_elapsed_s": float(self.assembly_elapsed_s),
            "hold_elapsed_s": float(self.hold_elapsed_s),
            "taper_elapsed_s": float(self.taper_elapsed_s),
            "release_started_step": self.release_started_step,
            "realized_active_s": self.active_elapsed_s(),
            "requested_duration_s": self.requested_duration_s,
            "assembly_duration_s": self.assembly_duration_s,
            "hold_duration_s": self.hold_duration_s,
            "taper_duration_s": self.taper_duration_s,
            "activation_step": self.activation_step,
            "activation_time_s": self.activation_time_s,
            "target_speed_mps": self.target_speed_mps,
            "requested_authority_mps2": self.requested_authority_mps2,
            "realized_authority_mps2": self.realized_authority_mps2,
            "authority_decel_limit_mps2": self.authority_decel_limit_mps2,
            "authority_accel_limit_mps2": self.authority_accel_limit_mps2,
            "lane_coverage": coverage,
            "center_m": None if self.spatial is None else float(self.spatial.center_m),
            "span_m": None if self.spatial is None else float(self.spatial.span_m),
            "spatial_step": self.spatial_step,
            "handoff_count": int(self.handoff_count),
            "abort_reason": self.abort_reason,
            "release_reason": self.release_reason,
            "events": list(self.events),
            "member_changes": list(self.member_changes),
        }

    def member_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                str(cav_id)
                for ids in self.members_by_lane.values()
                for cav_id in ids
            )
        )

    def reserve_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                str(cav_id)
                for ids in self.reserves_by_lane.values()
                for cav_id in ids
            )
        )


@dataclass(frozen=True)
class MultiwallConfig:
    """Mechanism configuration; unresolved scientific values stay ``None``.

    ``min_inter_wall_spacing_m``, ``storage_threshold_m``,
    ``combined_authority_budget_mps2`` and ``min_release_separation_s`` are
    unresolved scientific parameters and default to ``None`` ("unresolved" in
    diagnostics).  ``release_order`` is a mechanism enum validated here; the
    *value* chosen for a profile is a scientific decision.  ``spacing_must_be
    positive when bound`` and ``storage_threshold`` likewise; zero separation
    or zero storage is expressed by leaving the field unresolved.
    """

    max_active_cohorts: int = 2
    min_inter_wall_spacing_m: Optional[float] = None
    storage_threshold_m: Optional[float] = None
    combined_authority_budget_mps2: Optional[float] = None
    allow_simultaneous_release: bool = False
    min_release_separation_s: Optional[float] = None
    release_order: ReleaseOrder = ReleaseOrder.EXTERNAL
    safe_abort_policy: SafeAbortPolicy = SafeAbortPolicy.ABORT_ONLY_COHORT
    spatial_missing_behavior: str = "abort"
    max_handoffs_per_cohort: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_active_cohorts not in (1, 2):
            raise ValueError(
                "the first multiwall skeleton supports at most two active cohorts "
                f"(got max_active_cohorts={self.max_active_cohorts!r})"
            )
        for name in (
            "min_inter_wall_spacing_m",
            "storage_threshold_m",
            "combined_authority_budget_mps2",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)) or (
                value is not None and float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be a positive number or None (unresolved), got {value!r}")
        sep = self.min_release_separation_s
        if sep is not None and (not math.isfinite(float(sep)) or float(sep) < 0.0):
            raise ValueError(
                f"min_release_separation_s must be a nonnegative number or None, got {sep!r}"
            )
        order = self.release_order
        if not isinstance(order, ReleaseOrder):
            try:
                object.__setattr__(self, "release_order", ReleaseOrder(str(order)))
            except ValueError as exc:
                raise ValueError(f"unknown release_order: {order!r}") from exc
        policy = self.safe_abort_policy
        if not isinstance(policy, SafeAbortPolicy):
            try:
                object.__setattr__(self, "safe_abort_policy", SafeAbortPolicy(str(policy)))
            except ValueError as exc:
                raise ValueError(f"unknown safe_abort_policy: {policy!r}") from exc
        if self.spatial_missing_behavior not in ("abort", "diagnose"):
            raise ValueError(
                "spatial_missing_behavior must be 'abort' (fail closed) or 'diagnose'; "
                f"got {self.spatial_missing_behavior!r}"
            )
        if self.max_handoffs_per_cohort is not None and self.max_handoffs_per_cohort < 0:
            raise ValueError("max_handoffs_per_cohort must be nonnegative or None")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_active_cohorts": int(self.max_active_cohorts),
            "min_inter_wall_spacing_m": self.min_inter_wall_spacing_m,
            "storage_threshold_m": self.storage_threshold_m,
            "combined_authority_budget_mps2": self.combined_authority_budget_mps2,
            "allow_simultaneous_release": bool(self.allow_simultaneous_release),
            "min_release_separation_s": self.min_release_separation_s,
            "release_order": self.release_order.value,
            "safe_abort_policy": self.safe_abort_policy.value,
            "spatial_missing_behavior": str(self.spatial_missing_behavior),
            "max_handoffs_per_cohort": self.max_handoffs_per_cohort,
        }


@dataclass(frozen=True)
class ComposeResult:
    """Output action map plus deterministic JSON-serializable diagnostics."""

    actions: Dict[str, Any]
    diagnostics: Dict[str, Any]


class MultiwallError(ValueError):
    """Base class for deterministic multiwall state-transition rejections."""


class OwnershipError(MultiwallError):
    pass


class AuthorityError(MultiwallError):
    pass


class LifecycleTransitionError(MultiwallError):
    pass


class SpatialConflictError(MultiwallError):
    pass


class ReleaseConflictError(MultiwallError):
    pass


def _copy_payload(value: Any) -> Any:
    """Safely copy an opaque action payload without interpreting it."""
    return copy.deepcopy(value)


def _event(step: int, time_s: float, kind: str, **details: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "step": int(step),
        "time_s": float(time_s),
        "event": str(kind),
    }
    for key in sorted(details):
        record[str(key)] = _jsonable(details[key])
    return record


def _jsonable(value: Any) -> Any:
    """Recursively normalize to JSON-safe primitives (deterministic)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _validate_spatial(geometry: SpatialGeometry, context: str) -> None:
    if not (
        math.isfinite(float(geometry.center_m)) and math.isfinite(float(geometry.span_m))
    ):
        raise SpatialConflictError(f"{context}: non-finite spatial geometry")
    if float(geometry.span_m) < 0.0:
        raise SpatialConflictError(f"{context}: negative span")


class MultiwallCoordinator:
    """Pure two-cohort lifecycle state and damping-preserving composition.

    Lifecycle per cohort: ``register_cohort`` (INACTIVE) →
    ``activate_cohort`` (ASSEMBLING) → advance to ``HOLDING`` (assembly
    elapsed) → ``TAPERING`` (release requested or HOLD expired) → ``RELEASED``
    (taper elapsed).  ``abort_cohort`` (hard safety or explicit request) moves
    a cohort to ``ABORTED`` from any active state.  Overrides are removed from
    the action map exactly as soon as a member is released, aborted, or
    removed, so the supplied damping action is restored immediately.
    """

    def __init__(self, config: Optional[MultiwallConfig] = None) -> None:
        self.config = config if config is not None else MultiwallConfig()
        self.reset()

    def reset(self) -> None:
        """Clear all memberships, events, lifecycle state and diagnostics."""
        self._cohorts: Dict[str, CohortState] = {}
        self._member_owner: Dict[str, str] = {}
        self._reserve_owner: Dict[str, str] = {}
        self._step = 0
        self._time_s: Optional[float] = None
        self._last_release: Optional[Dict[str, Any]] = None
        self._pair_order: Dict[Tuple[str, str], str] = {}
        self._pair_previous_gap: Dict[Tuple[str, str], float] = {}
        self._spatial_conflicts: list[Dict[str, Any]] = []
        self._release_conflicts: list[Dict[str, Any]] = []
        self._authority_conflicts: list[Dict[str, Any]] = []
        self._events: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Registration and ownership
    # ------------------------------------------------------------------

    def register_cohort(
        self,
        cohort_id: str,
        *,
        members_by_lane: Optional[Mapping[str, Sequence[str]]] = None,
        reserves_by_lane: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        """Register one cohort in the INACTIVE lifecycle state.

        Ownership is global and disjoint: a CAV may be a primary member of at
        most one cohort, a reserve of at most one pool, and never both at the
        same time, across all registered cohorts.
        """
        cohort_id = str(cohort_id)
        if not cohort_id:
            raise OwnershipError("cohort_id must be a nonempty string")
        if cohort_id in self._cohorts:
            raise OwnershipError(f"cohort already registered: {cohort_id}")
        members = self._normalize_membership(members_by_lane or {})
        reserves = self._normalize_membership(reserves_by_lane or {})
        self._validate_ownership(cohort_id, members, reserves)
        state = CohortState(cohort_id=cohort_id)
        state.members_by_lane = dict(members)
        state.reserves_by_lane = dict(reserves)
        self._cohorts[cohort_id] = state
        for lane, ids in members.items():
            for cav_id in ids:
                self._member_owner[cav_id] = cohort_id
        for lane, ids in reserves.items():
            for cav_id in ids:
                self._reserve_owner[cav_id] = cohort_id

    def _normalize_membership(
        self, by_lane: Mapping[str, Sequence[str]]
    ) -> Dict[str, Tuple[str, ...]]:
        normalized: Dict[str, Tuple[str, ...]] = {}
        for lane in sorted(str(v) for v in by_lane):
            ids = tuple(sorted(str(x) for x in by_lane[lane]))
            if not ids:
                continue
            if len(set(ids)) != len(ids):
                raise OwnershipError(f"duplicate_member_assignment:lane={lane}")
            normalized[lane] = ids
        return normalized

    def _validate_ownership(
        self,
        cohort_id: str,
        members: Mapping[str, Sequence[str]],
        reserves: Mapping[str, Sequence[str]],
    ) -> None:
        member_set = {str(cav_id) for ids in members.values() for cav_id in ids}
        reserve_set = {str(cav_id) for ids in reserves.values() for cav_id in ids}
        overlap = member_set & reserve_set
        if overlap:
            raise OwnershipError(
                f"member_reserve_overlap:{cohort_id}:{','.join(sorted(overlap))}"
            )
        for cav_id in sorted(member_set):
            if cav_id in self._member_owner:
                raise OwnershipError(
                    f"cross_wall_member:{cav_id}:{self._member_owner[cav_id]}->{cohort_id}"
                )
            if cav_id in self._reserve_owner:
                raise OwnershipError(
                    f"cross_wall_reserve:{cav_id}:{self._reserve_owner[cav_id]}->{cohort_id}"
                )
        for cav_id in sorted(reserve_set):
            if cav_id in self._member_owner:
                raise OwnershipError(
                    f"cross_wall_member:{cav_id}:{self._member_owner[cav_id]}->{cohort_id}"
                )
            if cav_id in self._reserve_owner:
                raise OwnershipError(
                    f"cross_wall_reserve:{cav_id}:{self._reserve_owner[cav_id]}->{cohort_id}"
                )

    def cohort_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._cohorts))

    def cohort(self, cohort_id: str) -> Optional[CohortState]:
        return self._cohorts.get(str(cohort_id))

    def active_cohort_ids(self) -> Tuple[str, ...]:
        return tuple(
            sorted(cid for cid, state in self._cohorts.items() if state.is_active)
        )

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate_cohort(
        self,
        cohort_id: str,
        *,
        step: int,
        time_s: float,
        required_lanes: Sequence[str],
        requested_duration_s: float,
        target_speed_mps: float,
        assembly_duration_s: float,
        hold_duration_s: float,
        taper_duration_s: float,
        authority_request_mps2: float,
        authority_decel_limit_mps2: float,
        authority_accel_limit_mps2: float,
        spatial: Optional[SpatialGeometry],
    ) -> Dict[str, Any]:
        """Move INACTIVE -> ASSEMBLING with full mechanical validation.

        Fail-closed rejections (no state change): too many active cohorts,
        unresolved duration schedule, authority limits or combined budget
        exceeded, missing required-lane coverage, missing spatial data, or an
        inter-wall spatial conflict (overlap, spacing, ordering, storage).
        """
        cohort_id = str(cohort_id)
        self._valid_step_time(step, time_s)
        state = self._cohort(cohort_id)
        if state.lifecycle != CohortLifecycle.INACTIVE:
            raise LifecycleTransitionError(
                f"cohort_not_inactive:{cohort_id}:{state.lifecycle.value}"
            )
        active = self.active_cohort_ids()
        if len(active) >= self.config.max_active_cohorts:
            raise LifecycleTransitionError(
                "max_active_cohorts_exceeded:"
                f"active={','.join(active)};max={self.config.max_active_cohorts}"
            )
        self._validate_duration_schedule(
            requested_duration_s, assembly_duration_s, hold_duration_s, taper_duration_s
        )
        if not math.isfinite(float(target_speed_mps)) or float(target_speed_mps) < 0.0:
            raise MultiwallError(f"target_speed_mps must be finite and nonnegative, got {target_speed_mps!r}")
        required = tuple(sorted(str(v) for v in required_lanes))
        if not required:
            raise MultiwallError("required_lanes must be non-empty for an activating wall")
        coverage = LaneCoverage.compute(required, state.members_by_lane)
        if not coverage.full_coverage_retained:
            raise MultiwallError(
                "lane_coverage_missing:"
                + ",".join(coverage.missing_lanes)
                + f"(cohort={cohort_id})"
            )
        self._validate_authority(
            cohort_id,
            authority_request_mps2,
            authority_decel_limit_mps2,
            authority_accel_limit_mps2,
        )
        if spatial is None:
            raise SpatialConflictError(f"spatial_data_missing:{cohort_id}")
        _validate_spatial(spatial, f"cohort={cohort_id}")
        for other_id in active:
            self._check_activation_spatial(cohort_id, spatial, other_id)
        state.lifecycle = CohortLifecycle.ASSEMBLING
        state.activation_step = int(step)
        state.activation_time_s = float(time_s)
        state.assembly_elapsed_s = 0.0
        state.hold_elapsed_s = 0.0
        state.taper_elapsed_s = 0.0
        state.release_started_step = None
        state.requested_duration_s = float(requested_duration_s)
        state.assembly_duration_s = float(assembly_duration_s)
        state.hold_duration_s = float(hold_duration_s)
        state.taper_duration_s = float(taper_duration_s)
        state.target_speed_mps = float(target_speed_mps)
        state.requested_authority_mps2 = float(authority_request_mps2)
        state.realized_authority_mps2 = float(authority_request_mps2)
        state.authority_decel_limit_mps2 = float(authority_decel_limit_mps2)
        state.authority_accel_limit_mps2 = float(authority_accel_limit_mps2)
        state.required_lanes = required
        state.spatial = spatial
        state.spatial_step = int(step)
        state.handoff_count = 0
        state.abort_reason = None
        state.release_reason = None
        state.last_defer_reason = None
        self._record_global_event(step, time_s, "cohort_activated", cohort_id=cohort_id)
        for other_id in active:
            self._update_pair_order(cohort_id, other_id, step, time_s)
            self._update_pair_gap(cohort_id, other_id, step, time_s)
        return state.to_dict()

    def _validate_duration_schedule(
        self,
        requested_duration_s: float,
        assembly_duration_s: float,
        hold_duration_s: float,
        taper_duration_s: float,
    ) -> None:
        if not math.isfinite(float(requested_duration_s)) or float(requested_duration_s) <= 0.0:
            raise MultiwallError(
                f"requested_duration_s must be positive, got {requested_duration_s!r}"
            )
        parts = []
        for name, value in (
            ("assembly_duration_s", assembly_duration_s),
            ("hold_duration_s", hold_duration_s),
            ("taper_duration_s", taper_duration_s),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise MultiwallError(f"{name} must be finite and nonnegative, got {value!r}")
            parts.append(float(value))
        if sum(parts) > float(requested_duration_s) + _SMALL_EPS:
            raise MultiwallError(
                "duration_schedule_exceeds_requested:"
                f"assembly+hold+taper={sum(parts):.3f}>requested={float(requested_duration_s):.3f}"
            )

    def _validate_authority(
        self,
        cohort_id: str,
        authority_request_mps2: float,
        authority_decel_limit_mps2: float,
        authority_accel_limit_mps2: float,
    ) -> None:
        for name, value in (
            ("authority_request_mps2", authority_request_mps2),
            ("authority_decel_limit_mps2", authority_decel_limit_mps2),
            ("authority_accel_limit_mps2", authority_accel_limit_mps2),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise AuthorityError(
                    f"{name} must be finite and nonnegative, got {value!r}"
                )
        if float(authority_request_mps2) > float(authority_decel_limit_mps2) + _SMALL_EPS:
            raise AuthorityError(
                "per_wall_authority_limit_exceeded:"
                f"request={float(authority_request_mps2):.3f}>"
                f"decel_limit={float(authority_decel_limit_mps2):.3f}"
            )
        budget = self.config.combined_authority_budget_mps2
        if budget is not None:
            combined = float(authority_request_mps2) + sum(
                float(self._cohorts[cid].requested_authority_mps2 or 0.0)
                for cid in self.active_cohort_ids()
            )
            if combined > float(budget) + _SMALL_EPS:
                raise AuthorityError(
                    "combined_authority_budget_exceeded:"
                    f"combined={combined:.3f}>budget={float(budget):.3f}"
                )

    # ------------------------------------------------------------------
    # Spatial state and interaction
    # ------------------------------------------------------------------

    def update_spatial(
        self,
        cohort_id: str,
        spatial: SpatialGeometry,
        *,
        step: int,
        time_s: float,
    ) -> list[Dict[str, Any]]:
        """Externally refreshed longitudinal center and span.

        The update is applied (positions are facts) and every inter-wall
        interaction check is recomputed; hard conflicts (overlap, spacing,
        ordering reversal, insufficient storage) are returned and recorded,
        never silently swallowed.  Ordering reversal is detected against the
        last known upstream/downstream ordering.
        """
        cohort_id = str(cohort_id)
        self._valid_step_time(step, time_s)
        state = self._cohort(cohort_id)
        if not state.is_active:
            raise LifecycleTransitionError(
                f"spatial_update_invalid_state:{cohort_id}:{state.lifecycle.value}"
            )
        _validate_spatial(spatial, f"cohort={cohort_id}")
        conflicts = self._pair_conflicts_for_update(cohort_id, spatial, step, time_s)
        state.spatial = spatial
        state.spatial_step = int(step)
        self._spatial_conflicts = conflicts
        self._update_pair_order(cohort_id, None, step, time_s)
        return list(conflicts)

    def _pair_key(self, a_id: str, b_id: str) -> Tuple[str, str]:
        return tuple(sorted((a_id, b_id)))  # type: ignore[return-value]

    def _pair_source(self, a_id: str) -> Optional[SpatialGeometry]:
        state = self._cohorts.get(a_id)
        return None if state is None else state.spatial

    def _ordered_pair(
        self, a_id: str, b_id: str
    ) -> Tuple[Optional[str], Optional[str], Optional[SpatialGeometry], Optional[SpatialGeometry]]:
        a_spatial = self._pair_source(a_id)
        b_spatial = self._pair_source(b_id)
        if a_spatial is None or b_spatial is None:
            return None, None, a_spatial, b_spatial
        if float(a_spatial.center_m) >= float(b_spatial.center_m):
            return a_id, b_id, a_spatial, b_spatial
        return b_id, a_id, b_spatial, a_spatial

    def _ordered_pair_with(
        self, a_id: str, a_spatial: SpatialGeometry, b_id: str
    ) -> Tuple[Optional[str], Optional[str], Optional[SpatialGeometry], Optional[SpatialGeometry]]:
        b_spatial = self._pair_source(b_id)
        if b_spatial is None:
            return None, None, a_spatial, b_spatial
        if float(a_spatial.center_m) >= float(b_spatial.center_m):
            return a_id, b_id, a_spatial, b_spatial
        return b_id, a_id, b_spatial, a_spatial

    def _edge_gap(
        self, upstream: SpatialGeometry, downstream: SpatialGeometry
    ) -> float:
        return (
            float(upstream.downstream_edge_m) - float(downstream.upstream_edge_m)
        )

    def _pair_conflicts_for_update(
        self, cohort_id: str, spatial: SpatialGeometry, step: int, time_s: float
    ) -> list[Dict[str, Any]]:
        conflicts: list[Dict[str, Any]] = []
        for other_id in self.active_cohort_ids():
            if other_id == cohort_id:
                continue
            other_spatial = self._pair_source(other_id)
            if other_spatial is None:
                conflicts.append(
                    self._conflict_record(
                        "spatial_data_missing",
                        cohort_id,
                        other_id,
                        step,
                        time_s,
                        severity="hard",
                    )
                )
                continue
            # Ordering reversal is checked against the stored pair order
            # (which post-activation reflects the last known geometry).
            stored = self._pair_order.get(self._pair_key(cohort_id, other_id))
            up, dn, up_g, dn_g = self._ordered_pair_with(cohort_id, spatial, other_id)
            if stored is not None and up is not None and dn is not None and stored != up:
                conflicts.append(
                    self._conflict_record(
                        "order_reversal",
                        cohort_id,
                        other_id,
                        step,
                        time_s,
                        severity="hard",
                        upstream_cohort_id=up,
                        downstream_cohort_id=dn,
                        previous_upstream_cohort_id=stored,
                    )
                )
            gap = self._edge_gap(up_g, dn_g) if up_g is not None and dn_g is not None else None
            if gap is not None and gap < -_SMALL_EPS:
                conflicts.append(
                    self._conflict_record(
                        "inter_wall_overlap",
                        cohort_id,
                        other_id,
                        step,
                        time_s,
                        severity="hard",
                        upstream_cohort_id=up,
                        downstream_cohort_id=dn,
                        inter_wall_gap_m=float(gap),
                    )
                )
            if gap is not None and self.config.min_inter_wall_spacing_m is not None:
                spacing = float(self.config.min_inter_wall_spacing_m)
                if gap < spacing - _SMALL_EPS:
                    conflicts.append(
                        self._conflict_record(
                            "minimum_spacing_violation",
                            cohort_id,
                            other_id,
                            step,
                            time_s,
                            severity="hard",
                            upstream_cohort_id=up,
                            downstream_cohort_id=dn,
                            inter_wall_gap_m=float(gap),
                            min_inter_wall_spacing_m=spacing,
                        )
                    )
            if gap is not None and self.config.storage_threshold_m is not None:
                threshold = float(self.config.storage_threshold_m)
                if gap < threshold - _SMALL_EPS:
                    conflicts.append(
                        self._conflict_record(
                            "insufficient_storage",
                            cohort_id,
                            other_id,
                            step,
                            time_s,
                            severity="hard",
                            upstream_cohort_id=up,
                            downstream_cohort_id=dn,
                            inter_wall_gap_m=float(gap),
                            storage_threshold_m=threshold,
                        )
                    )
            previous = self._pair_previous_gap.get(self._pair_key(cohort_id, other_id))
            if (
                gap is not None
                and previous is not None
                and previous >= 0.0
                and gap < previous - _SMALL_EPS
            ):
                conflicts.append(
                    self._conflict_record(
                        "upstream_catching_downstream",
                        cohort_id,
                        other_id,
                        step,
                        time_s,
                        severity="warning",
                        upstream_cohort_id=up,
                        downstream_cohort_id=dn,
                        inter_wall_gap_m=float(gap),
                        previous_gap_m=float(previous),
                    )
                )
            self._pair_previous_gap[self._pair_key(cohort_id, other_id)] = gap
        return conflicts

    def _check_activation_spatial(
        self, new_id: str, new_spatial: SpatialGeometry, other_id: str
    ) -> None:
        other_spatial = self._pair_source(other_id)
        if other_spatial is None:
            raise SpatialConflictError(f"spatial_data_missing:{other_id}")
        up, dn, up_g, dn_g = self._ordered_pair_with(new_id, new_spatial, other_id)
        if up is None or dn is None:
            raise SpatialConflictError(
                f"indistinct_cohort_centers:{','.join(sorted((new_id, other_id)))}"
            )
        gap = self._edge_gap(up_g, dn_g)
        if gap < -_SMALL_EPS:
            raise SpatialConflictError(
                self._conflict_reason(
                    "inter_wall_overlap", new_id, other_id, gap
                )
            )
        spacing = self.config.min_inter_wall_spacing_m
        if spacing is not None and gap < float(spacing) - _SMALL_EPS:
            raise SpatialConflictError(
                self._conflict_reason(
                    "minimum_spacing_violation", new_id, other_id, gap
                )
            )
        threshold = self.config.storage_threshold_m
        if threshold is not None and gap < float(threshold) - _SMALL_EPS:
            raise SpatialConflictError(
                self._conflict_reason("insufficient_storage", new_id, other_id, gap)
            )

    def _conflict_reason(self, kind: str, a_id: str, b_id: str, gap: float) -> str:
        return (
            f"{kind}:cohorts={','.join(sorted((a_id, b_id)))}:"
            f"inter_wall_gap_m={float(gap):.3f}"
        )

    def _conflict_record(
        self,
        kind: str,
        a_id: str,
        b_id: str,
        step: int,
        time_s: float,
        *,
        severity: str,
        **extra: Any,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "kind": str(kind),
            "severity": str(severity),
            "cohort_a": str(a_id),
            "cohort_b": str(b_id),
            "reason": (
                f"{kind}:cohorts={','.join(sorted((a_id, b_id)))}"
                + (
                    f":inter_wall_gap_m={float(extra['inter_wall_gap_m']):.3f}"
                    if "inter_wall_gap_m" in extra
                    else ""
                )
            ),
            "step": int(step),
            "time_s": float(time_s),
        }
        for key in sorted(extra):
            record[str(key)] = _jsonable(extra[key])
        return record

    def _update_pair_order(self, a_id: str, b_id: Optional[str], step: int, time_s: float) -> None:
        if b_id is None:
            for other_id in self.active_cohort_ids():
                if other_id != a_id:
                    self._record_pair_order(a_id, other_id)
            return
        self._record_pair_order(a_id, b_id)

    def _record_pair_order(self, a_id: str, b_id: str) -> None:
        up, dn, _up_g, _dn_g = self._ordered_pair(a_id, b_id)
        if up is not None:
            self._pair_order[self._pair_key(a_id, b_id)] = up

    def _update_pair_gap(self, a_id: str, b_id: str, step: int, time_s: float) -> None:
        up, dn, up_g, dn_g = self._ordered_pair(a_id, b_id)
        if up_g is not None and dn_g is not None:
            gap = self._edge_gap(up_g, dn_g)
            self._pair_previous_gap[self._pair_key(a_id, b_id)] = gap

    def evaluate_spatial_conflicts(self) -> list[Dict[str, Any]]:
        """Deterministic current-interaction report for the active cohort pair."""
        active = self.active_cohort_ids()
        if len(active) < 2:
            return []
        conflicts: list[Dict[str, Any]] = []
        a_id, b_id = active[0], active[1]
        a_spatial = self._pair_source(a_id)
        b_spatial = self._pair_source(b_id)
        if a_spatial is None or b_spatial is None:
            conflicts.append(
                self._conflict_record(
                    "spatial_data_missing", a_id, b_id, self._step, self._time_s or 0.0,
                    severity="hard",
                )
            )
            return conflicts
        up, dn, up_g, dn_g = self._ordered_pair(a_id, b_id)
        gap = self._edge_gap(up_g, dn_g)
        if up is None or dn is None:
            conflicts.append(
                self._conflict_record(
                    "indistinct_cohort_centers", a_id, b_id, self._step, self._time_s or 0.0,
                    severity="hard",
                )
            )
            return conflicts
        if gap < -_SMALL_EPS:
            conflicts.append(
                self._conflict_record(
                    "inter_wall_overlap", a_id, b_id, self._step, self._time_s or 0.0,
                    severity="hard", upstream_cohort_id=up, downstream_cohort_id=dn,
                    inter_wall_gap_m=float(gap),
                )
            )
        spacing = self.config.min_inter_wall_spacing_m
        if spacing is not None and gap < float(spacing) - _SMALL_EPS:
            conflicts.append(
                self._conflict_record(
                    "minimum_spacing_violation", a_id, b_id, self._step, self._time_s or 0.0,
                    severity="hard", upstream_cohort_id=up, downstream_cohort_id=dn,
                    inter_wall_gap_m=float(gap), min_inter_wall_spacing_m=float(spacing),
                )
            )
        threshold = self.config.storage_threshold_m
        if threshold is not None and gap < float(threshold) - _SMALL_EPS:
            conflicts.append(
                self._conflict_record(
                    "insufficient_storage", a_id, b_id, self._step, self._time_s or 0.0,
                    severity="hard", upstream_cohort_id=up, downstream_cohort_id=dn,
                    inter_wall_gap_m=float(gap), storage_threshold_m=float(threshold),
                )
            )
        return conflicts

    def spatial_status(self) -> Dict[str, Any]:
        """Snapshot of inter-wall geometry with spacing/storage status."""
        active = self.active_cohort_ids()
        row: Dict[str, Any] = {
            "active_cohort_ids": list(active),
            "conflicts": self.evaluate_spatial_conflicts(),
        }
        if len(active) == 2:
            a_id, b_id = active[0], active[1]
            up, dn, up_g, dn_g = self._ordered_pair(a_id, b_id)
            if up is None or dn is None:
                row["status"] = "spatial_data_missing"
                return row
            gap = self._edge_gap(up_g, dn_g)
            spacing_status = "unresolved"
            if self.config.min_inter_wall_spacing_m is not None:
                spacing_status = (
                    "conflict" if gap < float(self.config.min_inter_wall_spacing_m) - _SMALL_EPS
                    else "satisfied"
                )
            storage_status = "unresolved"
            if self.config.storage_threshold_m is not None:
                storage_status = (
                    "insufficient" if gap < float(self.config.storage_threshold_m) - _SMALL_EPS
                    else "sufficient"
                )
            row.update(
                {
                    "status": "ok",
                    "upstream_cohort_id": str(up),
                    "downstream_cohort_id": str(dn),
                    "center_distance_m": float(up_g.center_m) - float(dn_g.center_m),
                    "edge_gap_m": float(gap),
                    "min_inter_wall_spacing_m": self.config.min_inter_wall_spacing_m,
                    "spacing_status": spacing_status,
                    "storage_threshold_m": self.config.storage_threshold_m,
                    "storage_status": storage_status,
                }
            )
        return row

    # ------------------------------------------------------------------
    # Lifecycle advance, release, abort
    # ------------------------------------------------------------------

    def advance(self, *, step: int, time_s: float) -> Dict[str, str]:
        """Advance active-cohort timers and apply automatic transitions.

        Automatic transitions: ASSEMBLING -> HOLDING on assembly elapsed;
        HOLDING -> TAPERING on HOLD elapsed when the configured release rules
        permit (otherwise the release is deferred and the reason recorded);
        TAPERING -> RELEASED on taper elapsed, which restores damping.
        """
        self._valid_step_time(step, time_s)
        dt = self._delta_time(step, time_s)
        transitions: Dict[str, str] = {}
        for cohort_id in self.active_cohort_ids():
            state = self._cohorts[cohort_id]
            before = state.lifecycle.value
            if state.lifecycle == CohortLifecycle.ASSEMBLING:
                state.assembly_elapsed_s += dt
                if (
                    state.assembly_duration_s is not None
                    and state.assembly_elapsed_s >= float(state.assembly_duration_s) - _SMALL_EPS
                ):
                    state.lifecycle = CohortLifecycle.HOLDING
                    self._record_global_event(step, time_s, "hold_started", cohort_id=cohort_id)
            elif state.lifecycle == CohortLifecycle.HOLDING:
                state.hold_elapsed_s += dt
                if (
                    state.hold_duration_s is not None
                    and state.hold_elapsed_s >= float(state.hold_duration_s) - _SMALL_EPS
                ):
                    allowed, reason = self._release_rules_check(
                        cohort_id, step, time_s, raising=False
                    )
                    if allowed:
                        self._start_release(cohort_id, step, time_s, "hold_elapsed")
                    elif state.last_defer_reason != reason:
                        state.last_defer_reason = reason
                        state.events.append(
                            _event(step, time_s, "release_deferred", cohort_id=cohort_id, reason=reason)
                        )
            elif state.lifecycle == CohortLifecycle.TAPERING:
                state.taper_elapsed_s += dt
                if (
                    state.taper_duration_s is not None
                    and state.taper_elapsed_s >= float(state.taper_duration_s) - _SMALL_EPS
                ):
                    state.lifecycle = CohortLifecycle.RELEASED
                    self._record_global_event(step, time_s, "cohort_released", cohort_id=cohort_id)
            if state.lifecycle.value != before:
                transitions[cohort_id] = state.lifecycle.value
        return transitions

    def request_release(
        self,
        cohort_id: str,
        *,
        step: int,
        time_s: float,
        reason: str = "requested",
    ) -> Dict[str, Any]:
        """Request HOLDING -> TAPERING under the configured release rules.

        Fail-closed conflicts (no state change): invalid lifecycle state,
        simultaneous release when prohibited, minimum release separation, or
        a release-order violation.  ``EXTERNAL`` release order means the
        external caller decides the order and the coordinator checks only
        separation constraints.
        """
        cohort_id = str(cohort_id)
        self._valid_step_time(step, time_s)
        state = self._cohort(cohort_id)
        if state.lifecycle != CohortLifecycle.HOLDING:
            raise ReleaseConflictError(
                f"release_request_invalid_state:{cohort_id}:{state.lifecycle.value}"
            )
        allowed, reason = self._release_rules_check(cohort_id, step, time_s, raising=True)
        assert allowed
        self._start_release(cohort_id, step, time_s, reason or str(reason))
        return state.to_dict()

    def _release_rules_check(
        self, cohort_id: str, step: int, time_s: float, *, raising: bool
    ) -> Tuple[bool, Optional[str]]:
        state = self._cohorts[cohort_id]
        if not self.config.allow_simultaneous_release:
            for other_id in self.active_cohort_ids():
                other = self._cohorts[other_id]
                if (
                    other_id != cohort_id
                    and other.lifecycle == CohortLifecycle.TAPERING
                    and other.release_started_step == int(step)
                ):
                    if raising:
                        raise ReleaseConflictError(
                            f"simultaneous_release_prohibited:{cohort_id}:while={other_id}:step={int(step)}"
                        )
                    return False, "simultaneous_release_prohibited"
        last = self._last_release
        if (
            last is not None
            and last["cohort_id"] != cohort_id
            and self.config.min_release_separation_s is not None
        ):
            separation = float(time_s) - float(last["time_s"])
            if separation < float(self.config.min_release_separation_s) - _SMALL_EPS:
                if raising:
                    raise ReleaseConflictError(
                        "minimum_release_separation:"
                        f"request={float(time_s):.3f};last={float(last['time_s']):.3f};"
                        f"min={float(self.config.min_release_separation_s):.3f}"
                    )
                return False, "minimum_release_separation"
        order = self.config.release_order
        if order != ReleaseOrder.EXTERNAL:
            for other_id in self.active_cohort_ids():
                if other_id == cohort_id:
                    continue
                other = self._cohorts[other_id]
                # A cohort that already began its taper (or finished/aborted)
                # does not block the configured release order.
                if other.lifecycle not in (
                    CohortLifecycle.ASSEMBLING,
                    CohortLifecycle.HOLDING,
                ):
                    continue
                up, dn, _up_g, _dn_g = self._ordered_pair(cohort_id, other_id)
                if up is None or dn is None:
                    if raising:
                        raise ReleaseConflictError("release_order_undetermined:spatial_data_missing")
                    return False, "release_order_undetermined"
                if order == ReleaseOrder.DOWNSTREAM_FIRST and up == cohort_id and dn == other_id:
                    if raising:
                        raise ReleaseConflictError(
                            f"release_order_violation:downstream_first:{cohort_id}:still_active={other_id}"
                        )
                    return False, "release_order_violation:downstream_first"
                if order == ReleaseOrder.UPSTREAM_FIRST and dn == cohort_id and up == other_id:
                    if raising:
                        raise ReleaseConflictError(
                            f"release_order_violation:upstream_first:{cohort_id}:still_active={other_id}"
                        )
                    return False, "release_order_violation:upstream_first"
        return True, None

    def _start_release(self, cohort_id: str, step: int, time_s: float, reason: str) -> None:
        state = self._cohorts[cohort_id]
        state.lifecycle = CohortLifecycle.TAPERING
        state.taper_elapsed_s = 0.0
        state.release_started_step = int(step)
        state.release_reason = str(reason)
        self._last_release = {
            "cohort_id": cohort_id,
            "step": int(step),
            "time_s": float(time_s),
        }
        self._record_global_event(step, time_s, "release_started", cohort_id=cohort_id, reason=str(reason))

    def abort_cohort(
        self,
        cohort_id: str,
        *,
        step: int,
        time_s: float,
        reason: str = "manual_abort",
    ) -> Dict[str, Any]:
        """Move an active cohort to ABORTED and remove its overrides.

        Use this for explicit calls; hard-safety failures during compose call
        it internally with a deterministic reason.
        """
        cohort_id = str(cohort_id)
        self._valid_step_time(step, time_s)
        state = self._cohort(cohort_id)
        if not state.is_active:
            raise LifecycleTransitionError(
                f"abort_invalid_state:{cohort_id}:{state.lifecycle.value}"
            )
        state.lifecycle = CohortLifecycle.ABORTED
        state.abort_reason = str(reason)
        self._record_global_event(step, time_s, "cohort_aborted", cohort_id=cohort_id, reason=str(reason))
        return state.to_dict()

    # ------------------------------------------------------------------
    # Membership changes
    # ------------------------------------------------------------------

    def promote_reserve(
        self,
        cohort_id: str,
        lane_id: str,
        reserve_id: str,
        *,
        step: int,
        time_s: float,
    ) -> Dict[str, Any]:
        """Atomically transfer one reserve into primary membership.

        On failure nothing changes; on success the reserve id is removed from
        each lane list and added to the primary members of its lane.
        """
        cohort_id = str(cohort_id)
        self._valid_step_time(step, time_s)
        state = self._cohort(cohort_id)
        if state.lifecycle in (CohortLifecycle.RELEASED, CohortLifecycle.ABORTED):
            raise LifecycleTransitionError(
                f"promote_invalid_state:{cohort_id}:{state.lifecycle.value}"
            )
        lane_id = str(lane_id)
        reserve_id = str(reserve_id)
        reserves = list(state.reserves_by_lane.get(lane_id, ()))
        if reserve_id not in reserves:
            raise OwnershipError(f"reserve_not_in_cohort_pool:{cohort_id}:{lane_id}:{reserve_id}")
        new_reserves = tuple(sorted(v for v in reserves if v != reserve_id))
        members = list(state.members_by_lane.get(lane_id, ()))
        new_members = tuple(sorted(set(members) | {reserve_id}))
        if new_members == tuple(sorted(members)):
            raise OwnershipError(f"reserve_already_member:{cohort_id}:{lane_id}:{reserve_id}")
        # Commit atomically.
        if new_reserves:
            state.reserves_by_lane[lane_id] = new_reserves
        else:
            state.reserves_by_lane.pop(lane_id, None)
        state.members_by_lane[lane_id] = new_members
        self._reserve_owner.pop(reserve_id, None)
        self._member_owner[reserve_id] = cohort_id
        change = {
            "event": "reserve_promoted",
            "cohort_id": cohort_id,
            "lane_id": lane_id,
            "reserve_id": reserve_id,
            "step": int(step),
            "time_s": float(time_s),
        }
        state.member_changes.append(change)
        state.events.append(_event(step, time_s, "reserve_promoted", cohort_id=cohort_id, lane_id=lane_id, reserve_id=reserve_id))
        return state.to_dict()

    def handoff(
        self,
        cohort_id: str,
        lane_id: str,
        to_id: str,
        *,
        step: int,
        time_s: float,
        from_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deterministic same-lane handoff; removed member returns to damping.

        The removable member is ``from_id``, or the single member of the lane
        if exactly one exists.  The reserve promotion (out of the reserve
        pool) and the member replacement are one atomic state change.
        """
        cohort_id = str(cohort_id)
        self._valid_step_time(step, time_s)
        state = self._cohort(cohort_id)
        if state.lifecycle not in (CohortLifecycle.ASSEMBLING, CohortLifecycle.HOLDING):
            raise LifecycleTransitionError(
                f"handoff_invalid_state:{cohort_id}:{state.lifecycle.value}"
            )
        lane_id = str(lane_id)
        to_id = str(to_id)
        members = list(state.members_by_lane.get(lane_id, ()))
        if not members:
            raise OwnershipError(f"handoff_no_members:{cohort_id}:{lane_id}")
        if from_id is None:
            if len(members) != 1:
                raise OwnershipError(
                    f"handoff_needs_from_id:{cohort_id}:{lane_id}:members={','.join(members)}"
                )
            from_id = members[0]
        from_id = str(from_id)
        if from_id not in members:
            raise OwnershipError(f"handoff_from_not_member:{cohort_id}:{lane_id}:{from_id}")
        if to_id not in state.reserves_by_lane.get(lane_id, ()):
            raise OwnershipError(f"handoff_to_not_reserve:{cohort_id}:{lane_id}:{to_id}")
        if self.config.max_handoffs_per_cohort is not None and (
            state.handoff_count >= int(self.config.max_handoffs_per_cohort)
        ):
            raise LifecycleTransitionError(
                f"handoff_limit_reached:{cohort_id}:{state.handoff_count}"
            )
        replacements = tuple(sorted(v for v in members if v != from_id) + [to_id])
        new_reserves = tuple(
            sorted(v for v in state.reserves_by_lane.get(lane_id, ()) if v != to_id)
        )
        # Commit atomically.
        state.members_by_lane[lane_id] = replacements
        if new_reserves:
            state.reserves_by_lane[lane_id] = new_reserves
        else:
            state.reserves_by_lane.pop(lane_id, None)
        self._member_owner.pop(from_id, None)
        self._reserve_owner.pop(to_id, None)
        self._member_owner[to_id] = cohort_id
        state.handoff_count += 1
        change = {
            "event": "handoff",
            "cohort_id": cohort_id,
            "lane_id": lane_id,
            "from_id": from_id,
            "to_id": to_id,
            "handoff_count": int(state.handoff_count),
            "step": int(step),
            "time_s": float(time_s),
        }
        state.member_changes.append(change)
        state.events.append(
            _event(
                step, time_s, "member_handoff", cohort_id=cohort_id, lane_id=lane_id,
                from_id=from_id, to_id=to_id, count=int(state.handoff_count),
            )
        )
        return state.to_dict()

    # ------------------------------------------------------------------
    # Command composition
    # ------------------------------------------------------------------

    def compose(
        self,
        damping_actions: Mapping[str, Any],
        wall_commands: Mapping[str, Any],
        *,
        step: int,
        time_s: float,
        safety_by_cohort: Optional[Mapping[str, SafetyStatus]] = None,
        member_speeds: Optional[Mapping[str, float]] = None,
    ) -> ComposeResult:
        """Compose the action map: copy damping, override active members only.

        ``damping_actions`` is an already-computed decentralized damping map
        (vehicle id -> opaque payload).  The output starts as an exact copy;
        then, for every explicitly active wall member with a supplied command,
        that member's payload is replaced.  Nonmembers, released members,
        aborted members and removed members always retain the identical
        supplied damping payload.  The input mappings are never mutated.

        Hard safety is evaluated first: an active cohort whose safety status
        is absent or not fully safe is aborted with an exact recorded reason;
        its overrides are removed and its members return to damping this step.
        """
        self._valid_step_time(step, time_s)
        dt = self._delta_time(step, time_s)
        cohort_map = dict(damping_actions)
        output = {str(k): _copy_payload(v) for k, v in cohort_map.items()}
        if safety_by_cohort is None:
            safety_by_cohort = {}
        # 1. Hard safety first (priority over HOLD duration and taper).
        for cohort_id in self.active_cohort_ids():
            status = safety_by_cohort.get(cohort_id)
            if status is None:
                self._hard_safety_abort(cohort_id, step, time_s, "safety_status_missing")
                continue
            unsafe = status.unsafe_fields()
            if unsafe:
                if self.config.safe_abort_policy == SafeAbortPolicy.COORDINATED_ALL:
                    for other_id in list(self.active_cohort_ids()):
                        self._hard_safety_abort(
                            other_id, step, time_s, f"coordinated_abort:{cohort_id}:{unsafe[0]}"
                        )
                else:
                    self._hard_safety_abort(
                        cohort_id, step, time_s, f"hard_safety:{unsafe[0]}"
                    )
        # 2. Fail closed when spatial data are missing for an active cohort.
        for cohort_id in self.active_cohort_ids():
            state = self._cohorts[cohort_id]
            if state.spatial is None:
                if self.config.spatial_missing_behavior == "abort":
                    self._hard_safety_abort(cohort_id, step, time_s, "spatial_data_missing")
                else:
                    self._spatial_conflicts.append(
                        self._conflict_record(
                            "spatial_data_missing", cohort_id, cohort_id, step, time_s,
                            severity="hard",
                        )
                    )
        # 3. Advance timers and automatic lifecycle transitions.
        for cohort_id in self.active_cohort_ids():
            self._advance_one(cohort_id, step, time_s, dt)
        # 4. Apply wall overrides only to active members.
        payload_missing: list[str] = []
        member_missing: list[str] = []
        for cohort_id in self.active_cohort_ids():
            state = self._cohorts[cohort_id]
            for cav_id in state.member_ids():
                if cav_id not in output:
                    member_missing.append(cav_id)
                    continue
                if cav_id not in wall_commands:
                    payload_missing.append(cav_id)
                    continue
                output[cav_id] = _copy_payload(wall_commands[cav_id])
        if payload_missing:
            self._record_global_event(
                step, time_s, "wall_payload_missing", cohort_ids=sorted(set(payload_missing))
            )
        if member_missing:
            self._record_global_event(
                step, time_s, "member_missing_from_action_map", cohort_ids=sorted(set(member_missing))
            )
        diagnostics = self._diagnostics(
            step,
            time_s,
            damping_actions,
            output,
            safety_by_cohort,
            member_speeds or {},
            tuple(sorted(set(payload_missing))),
            tuple(sorted(set(member_missing))),
        )
        return ComposeResult(actions=output, diagnostics=diagnostics)

    def _advance_one(self, cohort_id: str, step: int, time_s: float, dt: float) -> None:
        state = self._cohorts[cohort_id]
        if state.lifecycle == CohortLifecycle.ASSEMBLING:
            state.assembly_elapsed_s += dt
            if (
                state.assembly_duration_s is not None
                and state.assembly_elapsed_s >= float(state.assembly_duration_s) - _SMALL_EPS
            ):
                state.lifecycle = CohortLifecycle.HOLDING
                self._record_global_event(step, time_s, "hold_started", cohort_id=cohort_id)
        elif state.lifecycle == CohortLifecycle.HOLDING:
            state.hold_elapsed_s += dt
            if (
                state.hold_duration_s is not None
                and state.hold_elapsed_s >= float(state.hold_duration_s) - _SMALL_EPS
            ):
                allowed, reason = self._release_rules_check(cohort_id, step, time_s, raising=False)
                if allowed:
                    self._start_release(cohort_id, step, time_s, "hold_elapsed")
                elif state.last_defer_reason != reason:
                    state.last_defer_reason = reason
                    state.events.append(
                        _event(step, time_s, "release_deferred", cohort_id=cohort_id, reason=reason)
                    )
        elif state.lifecycle == CohortLifecycle.TAPERING:
            state.taper_elapsed_s += dt
            if (
                state.taper_duration_s is not None
                and state.taper_elapsed_s >= float(state.taper_duration_s) - _SMALL_EPS
            ):
                state.lifecycle = CohortLifecycle.RELEASED
                self._record_global_event(step, time_s, "cohort_released", cohort_id=cohort_id)

    def _hard_safety_abort(self, cohort_id: str, step: int, time_s: float, reason: str) -> None:
        state = self._cohorts[cohort_id]
        state.lifecycle = CohortLifecycle.ABORTED
        state.abort_reason = str(reason)
        self._record_global_event(
            step, time_s, "hard_safety_abort", cohort_id=cohort_id, reason=str(reason)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _diagnostics(
        self,
        step: int,
        time_s: float,
        damping_actions: Mapping[str, Any],
        output: Mapping[str, Any],
        safety_by_cohort: Mapping[str, SafetyStatus],
        member_speeds: Mapping[str, float],
        payload_missing_ids: Tuple[str, ...],
        member_missing_ids: Tuple[str, ...],
    ) -> Dict[str, Any]:
        overridden: list[str] = []
        returned: list[str] = []
        for cohort_id in self.cohort_ids():
            state = self._cohorts[cohort_id]
            member_tuple = state.member_ids()
            if state.is_active:
                overridden.extend(
                    cid
                    for cid in member_tuple
                    if cid in output and cid not in payload_missing_ids
                )
            else:
                returned.extend(member_tuple)
        nonmember_total = 0
        all_nonmember_identical = True
        for cav_id in sorted(damping_actions):
            if cav_id in overridden:
                continue
            nonmember_total += 1
            if not (output.get(cav_id) == damping_actions[cav_id]):
                all_nonmember_identical = False
        cohorts: Dict[str, Any] = {}
        for cohort_id in self.cohort_ids():
            state = self._cohorts[cohort_id]
            member_tuple = state.member_ids()
            cohorts[cohort_id] = {
                "lifecycle": state.lifecycle.value,
                "members_by_lane": {
                    str(k): list(v) for k, v in sorted(state.members_by_lane.items())
                },
                "reserves_by_lane": {
                    str(k): list(v) for k, v in sorted(state.reserves_by_lane.items())
                },
                "member_ids": list(member_tuple),
                "reserve_ids": list(state.reserve_ids()),
                "wall_override_ids": [
                    cid
                    for cid in member_tuple
                    if state.is_active and cid in output and cid not in payload_missing_ids
                ],
                "damping_preservation": {
                    "nonmember_identical": bool(all_nonmember_identical),
                    "returned_to_damping_ids": (
                        [] if state.is_active else list(member_tuple)
                    ),
                    "wall_payload_missing_ids": [
                        cid for cid in payload_missing_ids if cid in member_tuple
                    ],
                    "member_missing_ids": [
                        cid for cid in member_missing_ids if cid in member_tuple
                    ],
                },
                "requested_duration_s": state.requested_duration_s,
                "realized_duration_s": state.active_elapsed_s(),
                "assembly_elapsed_s": float(state.assembly_elapsed_s),
                "hold_elapsed_s": float(state.hold_elapsed_s),
                "taper_elapsed_s": float(state.taper_elapsed_s),
                "release_started_step": state.release_started_step,
                "target_speed_mps": state.target_speed_mps,
                "requested_authority_mps2": state.requested_authority_mps2,
                "realized_authority_mps2": state.realized_authority_mps2,
                "authority_decel_limit_mps2": state.authority_decel_limit_mps2,
                "authority_accel_limit_mps2": state.authority_accel_limit_mps2,
                "lane_coverage": state.lane_coverage().to_dict(),
                "center_m": None if state.spatial is None else float(state.spatial.center_m),
                "span_m": None if state.spatial is None else float(state.spatial.span_m),
                "speed_spread_mps": self._speed_spread(state, member_speeds),
                "handoff_count": int(state.handoff_count),
                "abort_reason": state.abort_reason,
                "release_reason": state.release_reason,
                "events": list(state.events),
                "member_changes": list(state.member_changes),
            }
        safety_rows: Dict[str, Any] = {}
        for cohort_id in self.active_cohort_ids():
            status = safety_by_cohort.get(cohort_id)
            safety_rows[cohort_id] = (
                {"status": "safe", "unsafe_fields": []}
                if status is not None and status.is_safe
                else (
                    {"status": "missing"}
                    if status is None
                    else {"status": "unsafe", "unsafe_fields": list(status.unsafe_fields())}
                )
            )
        inter_wall = self.spatial_status()
        authority = {
            "combined_requested_mps2": sum(
                float(self._cohorts[cid].requested_authority_mps2 or 0.0)
                for cid in self.active_cohort_ids()
            ),
            "combined_realized_mps2": sum(
                float(self._cohorts[cid].realized_authority_mps2 or 0.0)
                for cid in self.active_cohort_ids()
            ),
            "combined_budget_mps2": self.config.combined_authority_budget_mps2,
            "status": (
                "unresolved"
                if self.config.combined_authority_budget_mps2 is None
                else (
                    "within_budget"
                    if all(
                        float(self._cohorts[cid].requested_authority_mps2 or 0.0)
                        <= float(self.config.combined_authority_budget_mps2) + _SMALL_EPS
                        for cid in self.active_cohort_ids()
                    )
                    else "conflict"
                )
            ),
            "conflicts": list(self._authority_conflicts),
        }
        diagnostics: Dict[str, Any] = {
            "schema": "idm_v5_8_multiwall_coordinator_diagnostics/1",
            "step": int(step),
            "time_s": float(time_s),
            "config": self.config.to_dict(),
            "active_cohort_ids": list(self.active_cohort_ids()),
            "cohorts": cohorts,
            "inter_wall": inter_wall,
            "release": {
                "last_release": dict(self._last_release) if self._last_release is not None else None,
                "allow_simultaneous_release": bool(self.config.allow_simultaneous_release),
                "min_release_separation_s": self.config.min_release_separation_s,
                "release_order": self.config.release_order.value,
                "conflicts": list(self._release_conflicts),
            },
            "authority": authority,
            "safety": {"statuses": safety_rows},
            "nonmember_damping_preservation": {
                "nonmember_count": nonmember_total,
                "all_identical": bool(all_nonmember_identical),
            },
            "returned_to_damping_ids": sorted(set(returned)),
            "wall_payload_missing_ids": list(payload_missing_ids),
            "member_missing_ids": list(member_missing_ids),
            "spatial_conflicts": list(self._spatial_conflicts),
            "conflicts": (
                list(self._spatial_conflicts)
                + list(self._release_conflicts)
                + list(self._authority_conflicts)
            ),
            "events": list(self._events),
            "diagnostic_sha256": None,
        }
        encoded = json.dumps(diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False)
        diagnostics["diagnostic_sha256"] = sha256(encoded.encode()).hexdigest()
        return diagnostics

    def _speed_spread(
        self, state: CohortState, member_speeds: Mapping[str, float]
    ) -> Optional[float]:
        values = [
            float(member_speeds[v])
            for v in state.member_ids()
            if v in member_speeds
        ]
        if not values:
            return None
        return max(values) - min(values)

    def snapshot(self) -> Dict[str, Any]:
        """Deterministic full JSON-serializable state snapshot."""
        return {
            "schema": "idm_v5_8_multiwall_coordinator_snapshot/1",
            "config": self.config.to_dict(),
            "step": int(self._step),
            "time_s": self._time_s,
            "cohorts": {cid: self._cohorts[cid].to_dict() for cid in self.cohort_ids()},
            "member_ownership": dict(sorted(self._member_owner.items())),
            "reserve_ownership": dict(sorted(self._reserve_owner.items())),
            "last_release": dict(self._last_release) if self._last_release is not None else None,
            "pair_order": {list(k): v for k, v in sorted(self._pair_order.items())},
            "spatial_conflicts": list(self._spatial_conflicts),
            "events": list(self._events),
        }

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _cohort(self, cohort_id: str) -> CohortState:
        state = self._cohorts.get(cohort_id)
        if state is None:
            raise LifecycleTransitionError(f"unknown_cohort:{cohort_id}")
        return state

    def _valid_step_time(self, step: int, time_s: float) -> None:
        if int(step) != step or int(step) < 0:
            raise ValueError(f"step must be a nonnegative integer, got {step!r}")
        if not math.isfinite(float(time_s)) or float(time_s) < 0.0:
            raise ValueError(f"time_s must be finite and nonnegative, got {time_s!r}")

    def _delta_time(self, step: int, time_s: float) -> float:
        # The implicit time origin is t=0; the first compose/advance call
        # carries the elapsed simulation time since activation or reset.
        baseline = self._time_s if self._time_s is not None else 0.0
        dt = max(0.0, float(time_s) - float(baseline))
        self._step = int(step)
        self._time_s = float(time_s)
        return dt

    def _record_global_event(self, step: int, time_s: float, kind: str, **details: Any) -> None:
        self._events.append(_event(step, time_s, kind, **details))
