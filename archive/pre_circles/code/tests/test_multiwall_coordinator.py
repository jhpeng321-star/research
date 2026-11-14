"""Focused pure-Python tests for the isolated v5.8 multiwall skeleton.

All inputs are synthetic.  No SUMO is launched, no TraCI connection is opened
and no active experiment code is imported here.
"""

from __future__ import annotations

import copy
import json

import pytest

from policies.multiwall_coordinator import (
    AuthorityError,
    CohortLifecycle,
    LifecycleTransitionError,
    MultiwallConfig,
    MultiwallCoordinator,
    MultiwallError,
    OwnershipError,
    ReleaseConflictError,
    ReleaseOrder,
    SafetyStatus,
    SpatialConflictError,
    SpatialGeometry,
)
from experiments.idm.wall_v5_8_multiwall_contract import (
    SCIENTIFIC_UNRESOLVED_FIELDS,
    UnresolvedFieldsError,
    draft_manifest,
    executable_profile,
    profile_fingerprint,
)


def _safe() -> SafetyStatus:
    return SafetyStatus(
        front_safe=True,
        rear_safe=True,
        ttc_safe=True,
        lane_coverage_safe=True,
        compactness_safe=True,
        member_identity_safe=True,
        storage_safe=True,
        release_safe=True,
    )


def _unsafe(*names: str) -> SafetyStatus:
    base = _safe()
    return SafetyStatus(
        front_safe=base.front_safe and "front_safe" not in names,
        rear_safe=base.rear_safe and "rear_safe" not in names,
        ttc_safe=base.ttc_safe and "ttc_safe" not in names,
        lane_coverage_safe=base.lane_coverage_safe and "lane_coverage_safe" not in names,
        compactness_safe=base.compactness_safe and "compactness_safe" not in names,
        member_identity_safe=base.member_identity_safe and "member_identity_safe" not in names,
        storage_safe=base.storage_safe and "storage_safe" not in names,
        release_safe=base.release_safe and "release_safe" not in names,
    )


def _activate(
    coord: MultiwallCoordinator,
    cohort_id: str,
    *,
    center_m: float,
    span_m: float = 20.0,
    step: int = 0,
    time_s: float = 0.0,
    lanes: tuple = ("L0",),
    requested_duration_s: float = 30.0,
    assembly_duration_s: float = 1.0,
    hold_duration_s: float = 4.0,
    taper_duration_s: float = 1.0,
    authority_mps2: float = 0.1,
    decel_limit_mps2: float = 0.3,
) -> None:
    coord.activate_cohort(
        cohort_id,
        step=step,
        time_s=time_s,
        required_lanes=lanes,
        requested_duration_s=requested_duration_s,
        target_speed_mps=10.0,
        assembly_duration_s=assembly_duration_s,
        hold_duration_s=hold_duration_s,
        taper_duration_s=taper_duration_s,
        authority_request_mps2=authority_mps2,
        authority_decel_limit_mps2=decel_limit_mps2,
        authority_accel_limit_mps2=1.0,
        spatial=SpatialGeometry(center_m=center_m, span_m=span_m),
    )


def _register(coord: MultiwallCoordinator, cohort_id: str, members, reserves=()) -> None:
    coord.register_cohort(
        cohort_id,
        members_by_lane={"L0": list(members)},
        reserves_by_lane={"L0": list(reserves)} if reserves else {},
    )


def _complete_scientific_fields() -> dict:
    return {
        "min_inter_wall_spacing_m": 30.0,
        "assembly_zone_m": 900.0,
        "hold_zone_m": 600.0,
        "activation_offset_s": 10.0,
        "per_wall_authority_mps2": 0.3,
        "combined_authority_budget_mps2": 0.6,
        "storage_threshold_m": 50.0,
        "release_order": "downstream_first",
        "min_release_separation_s": 30.0,
        "candidate_selection_rule": "synthetic_placeholder",
        "cavs_per_lane": 3,
        "treatment_duration_s": 58.0,
    }


# 1 ---------------------------------------------------------------------
def test_01_two_disjoint_cohorts_can_be_registered() -> None:
    coord = MultiwallCoordinator()
    coord.register_cohort(
        "wall_a",
        members_by_lane={"L0": ["m1", "m2"]},
        reserves_by_lane={"L0": ["r1"]},
    )
    coord.register_cohort(
        "wall_b",
        members_by_lane={"L0": ["m3"]},
        reserves_by_lane={"L0": ["r2"]},
    )
    assert coord.cohort_ids() == ("wall_a", "wall_b")
    a, b = coord.cohort("wall_a"), coord.cohort("wall_b")
    assert a.lifecycle == CohortLifecycle.INACTIVE
    assert a.member_ids() == ("m1", "m2")
    assert a.reserve_ids() == ("r1",)
    assert b.member_ids() == ("m3",)
    assert b.reserve_ids() == ("r2",)


# 2 ---------------------------------------------------------------------
def test_02_third_active_cohort_is_rejected() -> None:
    coord = MultiwallCoordinator()
    for cid, center in (("wall_a", 400.0), ("wall_b", 200.0), ("wall_c", 50.0)):
        _register(coord, cid, [f"{cid}-m"])
    _activate(coord, "wall_a", center_m=400.0)
    _activate(coord, "wall_b", center_m=200.0)
    with pytest.raises(LifecycleTransitionError, match="max_active_cohorts_exceeded"):
        _activate(coord, "wall_c", center_m=50.0)
    assert coord.active_cohort_ids() == ("wall_a", "wall_b")


# 3 ---------------------------------------------------------------------
def test_03_duplicate_membership_across_walls_is_rejected() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1"])
    with pytest.raises(OwnershipError, match="cross_wall_member"):
        _register(coord, "wall_b", ["m1"])
    # A CAV that is already a primary member cannot become another wall's reserve.
    with pytest.raises(OwnershipError, match="cross_wall_member"):
        _register(coord, "wall_b", [], reserves=["m1"])
    coord2 = MultiwallCoordinator()
    coord2.register_cohort("wall_a", reserves_by_lane={"L0": ["r1"]})
    with pytest.raises(OwnershipError, match="cross_wall_reserve"):
        coord2.register_cohort("wall_b", reserves_by_lane={"L0": ["r1"]})


# 4 ---------------------------------------------------------------------
def test_04_primary_reserve_overlap_is_rejected() -> None:
    coord = MultiwallCoordinator()
    with pytest.raises(OwnershipError, match="member_reserve_overlap"):
        coord.register_cohort(
            "wall_a",
            members_by_lane={"L0": ["m1"]},
            reserves_by_lane={"L0": ["m1"]},
        )
    # A CAV that is already a reserve cannot also become a primary member.
    coord2 = MultiwallCoordinator()
    coord2.register_cohort("wall_a", reserves_by_lane={"L0": ["r1"]})
    with pytest.raises(OwnershipError, match="cross_wall_reserve"):
        _register(coord2, "wall_b", ["r1"])


# 5 ---------------------------------------------------------------------
def test_05_reserve_transfer_is_atomic() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1"], reserves=["r1"])
    coord.promote_reserve("wall_a", "L0", "r1", step=0, time_s=0.0)
    a = coord.cohort("wall_a")
    assert "r1" in a.member_ids()
    assert "r1" not in a.reserve_ids()
    assert all("r1" not in coord.cohort(cid).reserve_ids() for cid in coord.cohort_ids())
    # A failing transfer must not change anything.
    with pytest.raises(OwnershipError, match="reserve_not_in_cohort_pool"):
        coord.promote_reserve("wall_a", "L0", "missing", step=0, time_s=0.0)
    a = coord.cohort("wall_a")
    assert a.member_ids() == ("m1", "r1")
    assert a.reserve_ids() == ()


# 6 ---------------------------------------------------------------------
def test_06_nonmember_damping_actions_remain_exactly_unchanged() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1", "m2"])
    _register(coord, "wall_b", ["m3"])
    _activate(coord, "wall_a", center_m=200.0)
    damping = {
        "m1": {"tau": 3.0, "speed": 10.0},
        "m2": {"tau": 2.0, "speed": 9.0},
        "m3": {"tau": 1.5, "speed": 8.0},
        "other": {"tau": 0.5, "speed": 7.0},
    }
    wall = {"m1": {"tau": 0.5, "speed": 5.0}, "m2": {"tau": 0.8, "speed": 5.5}}
    result = coord.compose(damping, wall, step=1, time_s=0.5, safety_by_cohort={"wall_a": _safe()})
    assert result.actions["m3"] == damping["m3"]
    assert result.actions["other"] == damping["other"]
    assert result.actions["m1"] == wall["m1"]
    assert result.actions["m2"] == wall["m2"]
    diag = result.diagnostics
    assert diag["nonmember_damping_preservation"]["all_identical"] is True


# 7 ---------------------------------------------------------------------
def test_07_wall_override_applies_only_to_active_members() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1", "m2"])
    _register(coord, "wall_b", ["m3"])
    _activate(coord, "wall_a", center_m=200.0)
    damping = {"m1": {"a": 0.0}, "m2": {"a": 0.1}, "m3": {"a": 0.2}}
    wall = {"m1": {"a": -0.3}, "m2": {"a": -0.1}}
    result = coord.compose(damping, wall, step=1, time_s=0.5, safety_by_cohort={"wall_a": _safe()})
    assert result.actions["m1"] == {"a": -0.3}
    assert result.actions["m2"] == {"a": -0.1}
    assert result.actions["m3"] == {"a": 0.2}
    assert result.diagnostics["cohorts"]["wall_a"]["wall_override_ids"] == ["m1", "m2"]
    assert result.diagnostics["cohorts"]["wall_a"]["lifecycle"] == "ASSEMBLING"
    # The inactive cohort's member was never overridden.
    assert "m3" not in result.diagnostics["cohorts"]["wall_b"]["wall_override_ids"]


# 8 ---------------------------------------------------------------------
def test_08_released_members_return_exactly_to_damping() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1", "m2"])
    _activate(coord, "wall_a", center_m=200.0, requested_duration_s=10.0, hold_duration_s=4.0)
    damping = {"m1": {"a": 0.0}, "m2": {"a": 0.1}, "other": {"a": 0.2}}
    wall = {"m1": {"a": -0.3}, "m2": {"a": -0.3}}
    for step in range(1, 9):
        result = coord.compose(
            damping, wall, step=step, time_s=float(step), safety_by_cohort={"wall_a": _safe()}
        )
    assert coord.cohort("wall_a").lifecycle == CohortLifecycle.RELEASED
    assert result.actions["m1"] == damping["m1"]
    assert result.actions["m2"] == damping["m2"]
    assert result.diagnostics["returned_to_damping_ids"] == ["m1", "m2"]


# 9 ---------------------------------------------------------------------
def test_09_aborted_members_return_exactly_to_damping() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1", "m2"])
    _activate(coord, "wall_a", center_m=200.0)
    damping = {"m1": {"a": 0.0}, "m2": {"a": 0.1}}
    wall = {"m1": {"a": -0.3}, "m2": {"a": -0.3}}
    coord.compose(damping, wall, step=1, time_s=0.5, safety_by_cohort={"wall_a": _safe()})
    coord.abort_cohort("wall_a", step=2, time_s=1.0, reason="pilot_reason")
    result = coord.compose(damping, wall, step=3, time_s=1.5)
    assert result.actions["m1"] == damping["m1"]
    assert result.actions["m2"] == damping["m2"]
    assert coord.cohort("wall_a").lifecycle == CohortLifecycle.ABORTED
    assert coord.cohort("wall_a").abort_reason == "pilot_reason"


# 10 --------------------------------------------------------------------
def test_10_one_wall_can_abort_without_changing_the_other() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["a1", "a2"])
    _register(coord, "wall_b", ["b1", "b2"])
    _activate(coord, "wall_a", center_m=400.0)
    _activate(coord, "wall_b", center_m=100.0, step=1, time_s=1.0)
    damping = {"a1": {"a": 0.0}, "a2": {"a": 0.0}, "b1": {"a": 0.1}, "b2": {"a": 0.1}}
    wall = {"a1": {"a": -0.3}, "a2": {"a": -0.3}, "b1": {"a": -0.2}, "b2": {"a": -0.2}}
    coord.compose(
        damping, wall, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe(), "wall_b": _safe()}
    )
    coord.abort_cohort("wall_a", step=3, time_s=3.0, reason="collision")
    result = coord.compose(
        damping, wall, step=4, time_s=4.0, safety_by_cohort={"wall_b": _safe()}
    )
    assert result.actions["a1"] == damping["a1"]
    assert result.actions["a2"] == damping["a2"]
    assert result.actions["b1"] == wall["b1"]
    assert result.actions["b2"] == wall["b2"]
    assert coord.cohort("wall_b").lifecycle == CohortLifecycle.HOLDING
    assert coord.cohort("wall_b").events == []
    assert coord.cohort("wall_a").abort_reason == "collision"


# 11 --------------------------------------------------------------------
def test_11_missing_required_lane_coverage_fails_safely() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1"])
    with pytest.raises(MultiwallError, match="lane_coverage_missing"):
        _activate(coord, "wall_a", center_m=200.0, lanes=("L0", "L1"))
    assert coord.cohort("wall_a").lifecycle == CohortLifecycle.INACTIVE


# 12 --------------------------------------------------------------------
def test_12_inter_wall_overlap_is_detected() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["a1"])
    _register(coord, "wall_b", ["b1"])
    _activate(coord, "wall_a", center_m=200.0, span_m=50.0)
    with pytest.raises(SpatialConflictError, match="inter_wall_overlap"):
        _activate(coord, "wall_b", center_m=210.0, span_m=50.0, step=1, time_s=1.0)
    # Positions move over time; an update that overlaps must surface a conflict.
    _activate(coord, "wall_b", center_m=400.0, span_m=50.0, step=1, time_s=1.0)
    conflicts = coord.update_spatial(
        "wall_b", SpatialGeometry(center_m=230.0, span_m=50.0), step=2, time_s=2.0
    )
    assert any(c["kind"] == "inter_wall_overlap" for c in conflicts)


# 13 --------------------------------------------------------------------
def test_13_minimum_spacing_violation_is_detected() -> None:
    coord = MultiwallCoordinator(config=MultiwallConfig(min_inter_wall_spacing_m=50.0))
    _register(coord, "wall_a", ["a1"])
    _register(coord, "wall_b", ["b1"])
    _activate(coord, "wall_a", center_m=200.0, span_m=20.0)
    with pytest.raises(SpatialConflictError, match="minimum_spacing_violation"):
        _activate(coord, "wall_b", center_m=260.0, span_m=20.0, step=1, time_s=1.0)
    _activate(coord, "wall_b", center_m=300.0, span_m=20.0, step=1, time_s=1.0)
    conflicts = coord.update_spatial(
        "wall_b", SpatialGeometry(center_m=245.0, span_m=20.0), step=2, time_s=2.0
    )
    assert any(c["kind"] == "minimum_spacing_violation" for c in conflicts)


# 14 --------------------------------------------------------------------
def test_14_catch_up_and_order_reversal_are_detected() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["a1"])
    _register(coord, "wall_b", ["b1"])
    _activate(coord, "wall_a", center_m=400.0, span_m=20.0)
    _activate(coord, "wall_b", center_m=100.0, span_m=20.0, step=1, time_s=1.0)
    assert coord.spatial_status()["upstream_cohort_id"] == "wall_a"
    conflicts = coord.update_spatial(
        "wall_a", SpatialGeometry(center_m=50.0, span_m=20.0), step=2, time_s=2.0
    )
    kinds = {c["kind"] for c in conflicts}
    assert "order_reversal" in kinds
    assert "upstream_catching_downstream" in kinds
    assert coord.spatial_status()["upstream_cohort_id"] == "wall_b"


# 15 --------------------------------------------------------------------
def test_15_missing_spatial_information_fails_closed() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1"])
    with pytest.raises(SpatialConflictError, match="spatial_data_missing"):
        coord.activate_cohort(
            "wall_a",
            step=0,
            time_s=0.0,
            required_lanes=("L0",),
            requested_duration_s=30.0,
            target_speed_mps=10.0,
            assembly_duration_s=1.0,
            hold_duration_s=4.0,
            taper_duration_s=1.0,
            authority_request_mps2=0.1,
            authority_decel_limit_mps2=0.3,
            authority_accel_limit_mps2=1.0,
            spatial=None,
        )
    assert coord.cohort("wall_a").lifecycle == CohortLifecycle.INACTIVE


# 16 --------------------------------------------------------------------
def test_16_insufficient_storage_is_detected() -> None:
    coord = MultiwallCoordinator(config=MultiwallConfig(storage_threshold_m=30.0))
    _register(coord, "wall_a", ["a1"])
    _register(coord, "wall_b", ["b1"])
    _activate(coord, "wall_a", center_m=200.0, span_m=20.0)
    _activate(coord, "wall_b", center_m=260.0, span_m=20.0, step=1, time_s=1.0)
    conflicts = coord.update_spatial(
        "wall_b", SpatialGeometry(center_m=243.0, span_m=20.0), step=2, time_s=2.0
    )
    assert any(c["kind"] == "insufficient_storage" for c in conflicts)
    assert coord.spatial_status()["storage_status"] == "insufficient"


# 17 --------------------------------------------------------------------
def test_17_combined_authority_budget_is_enforced() -> None:
    coord = MultiwallCoordinator(config=MultiwallConfig(combined_authority_budget_mps2=0.5))
    _register(coord, "wall_a", ["a1"])
    _register(coord, "wall_b", ["b1"])
    _activate(coord, "wall_a", center_m=400.0, authority_mps2=0.3, decel_limit_mps2=0.3)
    with pytest.raises(AuthorityError, match="combined_authority_budget_exceeded"):
        _activate(
            coord, "wall_b", center_m=100.0, step=1, time_s=1.0,
            authority_mps2=0.3, decel_limit_mps2=0.3,
        )
    # A within-budget combination is accepted.
    coord2 = MultiwallCoordinator(config=MultiwallConfig(combined_authority_budget_mps2=0.5))
    _register(coord2, "wall_a", ["a1"])
    _register(coord2, "wall_b", ["b1"])
    _activate(coord2, "wall_a", center_m=400.0, authority_mps2=0.2, decel_limit_mps2=0.3)
    _activate(
        coord2, "wall_b", center_m=100.0, step=1, time_s=1.0,
        authority_mps2=0.3, decel_limit_mps2=0.3,
    )


# 18 --------------------------------------------------------------------
def test_18_per_wall_authority_limits_are_enforced() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1"])
    with pytest.raises(AuthorityError, match="per_wall_authority_limit_exceeded"):
        _activate(coord, "wall_a", center_m=200.0, authority_mps2=0.4, decel_limit_mps2=0.3)


# 19 --------------------------------------------------------------------
def test_19_simultaneous_release_can_be_prohibited() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["a1", "a2"])
    _register(coord, "wall_b", ["b1", "b2"])
    _activate(coord, "wall_a", center_m=400.0, hold_duration_s=1.0)
    _activate(coord, "wall_b", center_m=100.0, hold_duration_s=1.0, step=1, time_s=1.0)
    damping = {"a1": {}, "a2": {}, "b1": {}, "b2": {}}
    wall = {"a1": {}, "a2": {}, "b1": {}, "b2": {}}
    coord.compose(
        damping, wall, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe(), "wall_b": _safe()}
    )
    assert coord.cohort("wall_a").lifecycle == CohortLifecycle.HOLDING
    assert coord.cohort("wall_b").lifecycle == CohortLifecycle.HOLDING
    coord.request_release("wall_a", step=3, time_s=3.0, reason="planned")
    with pytest.raises(ReleaseConflictError, match="simultaneous_release_prohibited"):
        coord.request_release("wall_b", step=3, time_s=3.0, reason="planned")
    assert coord.cohort("wall_b").lifecycle == CohortLifecycle.HOLDING


# 20 --------------------------------------------------------------------
def test_20_minimum_release_separation_is_enforced() -> None:
    config = MultiwallConfig(allow_simultaneous_release=True, min_release_separation_s=5.0)
    coord = MultiwallCoordinator(config=config)
    _register(coord, "wall_a", ["a1", "a2"])
    _register(coord, "wall_b", ["b1", "b2"])
    _activate(coord, "wall_a", center_m=400.0, hold_duration_s=1.0)
    _activate(coord, "wall_b", center_m=100.0, hold_duration_s=1.0, step=1, time_s=1.0)
    damping = {"a1": {}, "a2": {}, "b1": {}, "b2": {}}
    wall = {"a1": {}, "a2": {}, "b1": {}, "b2": {}}
    coord.compose(
        damping, wall, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe(), "wall_b": _safe()}
    )
    coord.request_release("wall_a", step=3, time_s=3.0, reason="planned")
    with pytest.raises(ReleaseConflictError, match="minimum_release_separation"):
        coord.request_release("wall_b", step=4, time_s=4.0, reason="planned")
    assert coord.cohort("wall_b").lifecycle == CohortLifecycle.HOLDING
    # After the separation window passes, the release is allowed.
    coord.request_release("wall_b", step=9, time_s=9.0, reason="planned")
    assert coord.cohort("wall_b").lifecycle == CohortLifecycle.TAPERING


# 21 --------------------------------------------------------------------
def test_21_configurable_release_ordering_is_respected() -> None:
    downstream_first = MultiwallCoordinator(config=MultiwallConfig(release_order=ReleaseOrder.DOWNSTREAM_FIRST))
    _register(downstream_first, "wall_a", ["a1"])
    _register(downstream_first, "wall_b", ["b1"])
    _activate(downstream_first, "wall_a", center_m=400.0, hold_duration_s=1.0)
    _activate(downstream_first, "wall_b", center_m=100.0, hold_duration_s=1.0, step=1, time_s=1.0)
    small = {"a1": {}, "b1": {}}
    downstream_first.compose(
        small, small, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe(), "wall_b": _safe()}
    )
    # Upstream (wall_a) cannot release while downstream (wall_b) is still held.
    with pytest.raises(ReleaseConflictError, match="release_order_violation:downstream_first"):
        downstream_first.request_release("wall_a", step=3, time_s=3.0)
    downstream_first.request_release("wall_b", step=3, time_s=3.0)
    downstream_first.request_release("wall_a", step=4, time_s=4.0)

    upstream_first = MultiwallCoordinator(config=MultiwallConfig(release_order=ReleaseOrder.UPSTREAM_FIRST))
    _register(upstream_first, "wall_a", ["a1"])
    _register(upstream_first, "wall_b", ["b1"])
    _activate(upstream_first, "wall_a", center_m=400.0, hold_duration_s=1.0)
    _activate(upstream_first, "wall_b", center_m=100.0, hold_duration_s=1.0, step=1, time_s=1.0)
    upstream_first.compose(
        small, small, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe(), "wall_b": _safe()}
    )
    # Downstream (wall_b) cannot release while upstream (wall_a) is still held.
    with pytest.raises(ReleaseConflictError, match="release_order_violation:upstream_first"):
        upstream_first.request_release("wall_b", step=3, time_s=3.0)
    upstream_first.request_release("wall_a", step=3, time_s=3.0)
    upstream_first.request_release("wall_b", step=4, time_s=4.0)


# 22 --------------------------------------------------------------------
def test_22_hard_safety_overrides_hold_and_taper_preferences() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["a1", "a2"])
    _register(coord, "wall_b", ["b1", "b2"])
    # HOLD is short: at the next compose both cohorts are release-eligible.
    _activate(coord, "wall_a", center_m=400.0, hold_duration_s=1.0, taper_duration_s=5.0)
    _activate(coord, "wall_b", center_m=100.0, hold_duration_s=1.0, taper_duration_s=5.0, step=1, time_s=1.0)
    damping = {"a1": {}, "a2": {}, "b1": {}, "b2": {}}
    wall = {"a1": {"x": 1}, "a2": {"x": 1}, "b1": {"x": 2}, "b2": {"x": 2}}
    coord.compose(
        damping, wall, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe(), "wall_b": _safe()}
    )
    assert coord.cohort("wall_a").lifecycle == CohortLifecycle.HOLDING
    assert coord.cohort("wall_a").hold_elapsed_s == 0.0
    # TTC safety fails exactly when wall_a is about to be released: it must
    # abort rather than begin its taper, while wall_b is untouched by the abort.
    result = coord.compose(
        damping, wall, step=3, time_s=3.0, safety_by_cohort={"wall_a": _unsafe("ttc_safe"), "wall_b": _safe()}
    )
    a = coord.cohort("wall_a")
    assert a.lifecycle == CohortLifecycle.ABORTED
    assert a.abort_reason == "hard_safety:ttc_safe"
    assert not any(event["event"] == "release_started" for event in a.events)
    assert result.actions["a1"] == damping["a1"]
    assert result.actions["a2"] == damping["a2"]
    assert coord.cohort("wall_b").lifecycle == CohortLifecycle.TAPERING
    assert result.actions["b1"] == wall["b1"]


# 23 --------------------------------------------------------------------
def test_23_diagnostics_are_deterministic_and_json_serializable() -> None:
    def script(coord: MultiwallCoordinator) -> dict:
        _register(coord, "wall_a", ["m1", "m2"])
        _register(coord, "wall_b", ["m3"])
        _activate(coord, "wall_a", center_m=400.0)
        _activate(coord, "wall_b", center_m=100.0, step=1, time_s=1.0)
        damping = {"m1": {"a": 0.0}, "m2": {"a": 0.0}, "m3": {"a": 0.1}}
        wall = {"m1": {"a": -0.3}, "m2": {"a": -0.3}}
        result = coord.compose(
            damping, wall, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe(), "wall_b": _safe()}
        )
        return result.diagnostics

    first = script(MultiwallCoordinator())
    second = script(MultiwallCoordinator())
    assert first == second
    assert first["diagnostic_sha256"] == second["diagnostic_sha256"]
    encoded = json.dumps(first, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == first


# 24 --------------------------------------------------------------------
def test_24_state_transitions_are_deterministic() -> None:
    def script(coord: MultiwallCoordinator) -> dict:
        _register(coord, "wall_a", ["m1"], reserves=["r1"])
        _activate(coord, "wall_a", center_m=200.0)
        coord.promote_reserve("wall_a", "L0", "r1", step=1, time_s=1.0)
        damping = {"m1": {"a": 0.0}, "r1": {"a": 0.0}}
        wall = {"m1": {"a": -0.3}, "r1": {"a": -0.3}}
        coord.compose(damping, wall, step=2, time_s=2.0, safety_by_cohort={"wall_a": _safe()})
        coord.abort_cohort("wall_a", step=3, time_s=3.0, reason="determinism")
        return coord.snapshot()

    assert script(MultiwallCoordinator()) == script(MultiwallCoordinator())


# 25 --------------------------------------------------------------------
def test_25_unresolved_draft_contract_cannot_become_executable() -> None:
    manifest = draft_manifest()
    assert manifest["executable"] is False
    assert manifest["profile_sha256"] is None
    assert isinstance(SCIENTIFIC_UNRESOLVED_FIELDS, tuple)
    assert set(manifest["scientifically_unresolved_fields"]) >= set(SCIENTIFIC_UNRESOLVED_FIELDS)
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    with pytest.raises(UnresolvedFieldsError):
        executable_profile({})
    with pytest.raises(UnresolvedFieldsError):
        executable_profile({name: None for name in SCIENTIFIC_UNRESOLVED_FIELDS})


# 26 --------------------------------------------------------------------
def test_26_complete_synthetic_contract_receives_a_stable_fingerprint() -> None:
    fields = _complete_scientific_fields()
    first = executable_profile(fields)
    second = executable_profile(dict(fields))
    assert first["status"] == "executable_mechanics_only"
    assert first["executable"] is True
    assert first["profile_sha256"] == second["profile_sha256"]
    assert len(first["profile_sha256"]) == 64
    # Any change to a resolved scientific value changes the fingerprint.
    changed = executable_profile({**fields, "treatment_duration_s": 59.0})
    assert changed["profile_sha256"] != first["profile_sha256"]
    assert profile_fingerprint(first) == first["profile_sha256"]


# 27 --------------------------------------------------------------------
def test_27_reset_clears_all_memberships_events_and_lifecycle() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1"])
    _activate(coord, "wall_a", center_m=200.0)
    damping = {"m1": {"a": 0.0}}
    wall = {"m1": {"a": -0.3}}
    coord.compose(damping, wall, step=1, time_s=1.0, safety_by_cohort={"wall_a": _safe()})
    assert coord.active_cohort_ids() == ("wall_a",)
    coord.reset()
    assert coord.cohort_ids() == ()
    assert coord.active_cohort_ids() == ()
    result = coord.compose(damping, wall, step=0, time_s=0.0)
    assert result.actions == {"m1": {"a": 0.0}}
    assert result.diagnostics["active_cohort_ids"] == []
    assert result.diagnostics["events"] == []


# 28 --------------------------------------------------------------------
def test_28_input_damping_action_mappings_are_not_mutated() -> None:
    coord = MultiwallCoordinator()
    _register(coord, "wall_a", ["m1"])
    _activate(coord, "wall_a", center_m=200.0)
    damping = {"m1": {"tau": 3.0, "meta": ["a", "b"]}, "peer": {"tau": 1.0}}
    wall = {"m1": {"tau": 0.5, "meta": ("x", 1)}}
    before = (copy.deepcopy(damping), copy.deepcopy(wall))
    result = coord.compose(damping, wall, step=1, time_s=1.0, safety_by_cohort={"wall_a": _safe()})
    assert damping == before[0]
    assert wall == before[1]
    # The returned map is an independent copy: mutating it cannot alter the
    # supplied damping payloads.
    result.actions["m1"]["tau"] = 99.0
    result.actions["peer"]["tau"] = 99.0
    assert damping == before[0]
    assert wall == before[1]
