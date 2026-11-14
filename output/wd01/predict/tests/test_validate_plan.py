"""§3.3 plan rules — positive and negative tests for every rule.

Fixture controller_state (from the conformance kit): wall W1 in assemble
(members cav_a0/cav_a1, locked), wall W2 in hold (cav_h0, locked), wall W3 in
release (cav_r0, locked), free CAVs cav_n0/cav_n1.
"""
import dataclasses
import math

import pytest

from predict import (
    PHASE_ASSEMBLE,
    PHASE_RELEASE,
    ControllerStateEntry,
    PlanCandidate,
    PlanViolation,
    WallSpec,
    build_cav_ids,
    build_controller_state,
    build_plans,
    native_reference,
    native_unlocked_reference,
    validate_plan,
)

CS = build_controller_state()
IDS = build_cav_ids()


def free_plan(per_cav=None, walls=None):
    """A valid minimal plan: exactly the §5 native reference encoding."""
    native, _ = native_reference(CS, IDS)
    if per_cav is None:
        per_cav = dict(native.per_cav)
    if walls is None:
        walls = native.walls
    return PlanCandidate(1, per_cav, tuple(walls))


# --- positives ---------------------------------------------------------------

@pytest.mark.parametrize(
    "plan",
    build_plans(CS),
    ids=["native", "unlocked", "hold", "new-wall"],
)
def test_conformance_fixture_plans_pass(plan):
    validate_plan(plan, CS)


def test_native_and_unlocked_references_pass():
    validate_plan(native_reference(CS, IDS)[0], CS)
    validate_plan(native_unlocked_reference(CS, IDS), CS)


def test_raw_dict_plan_passes():
    native, _ = native_reference(CS, IDS)
    raw = {
        "plan_version": 1,
        "per_cav": dict(native.per_cav),
        "walls": [
            {
                "wall_id": w.wall_id,
                "members": list(w.members),
                "speed_mps": w.speed_mps,
                "phase": w.phase,
                "release_at_s": w.release_at_s,
                "hold_s": w.hold_s,
            }
            for w in native.walls
        ],
    }
    validate_plan(raw, CS)


def test_indefinite_hold_inf_is_acceptable():
    """Native kept walls encode 'assemble until a safety abort ends it' as
    hold_s=inf (spec question; see REPORT_PREDICT.md)."""
    native, kept = native_reference(CS, IDS)
    assert kept == ("W1",)
    w1 = next(w for w in native.walls if w.wall_id == "W1")
    assert w1.phase == PHASE_ASSEMBLE
    assert w1.hold_s == math.inf
    validate_plan(native, CS)


# --- negatives, one per §3.3 rule ---------------------------------------------

def test_plan_version_must_be_1():
    p = dataclasses.replace(free_plan(), plan_version=2)
    with pytest.raises(PlanViolation, match="plan_version"):
        validate_plan(p, CS)


def test_per_cav_must_be_the_full_map():
    per_cav = dict(free_plan().per_cav)
    del per_cav["cav_n1"]
    with pytest.raises(PlanViolation, match="FULL map"):
        validate_plan(free_plan(per_cav=per_cav), CS)


def test_per_cav_rejects_unknown_cavs():
    per_cav = dict(free_plan().per_cav)
    per_cav["cav_ghost"] = "native"
    with pytest.raises(PlanViolation, match="not non-member CAVs"):
        validate_plan(free_plan(per_cav=per_cav), CS)


def test_per_cav_rejects_bad_action():
    per_cav = dict(free_plan().per_cav)
    per_cav["cav_n0"] = "bananas"
    with pytest.raises(PlanViolation, match="bananas"):
        validate_plan(free_plan(per_cav=per_cav), CS)


def test_wall_member_must_not_appear_in_per_cav():
    per_cav = dict(free_plan().per_cav)
    per_cav["cav_h0"] = "native"  # member of release-encoded wall W2
    with pytest.raises(PlanViolation, match="never appear in per_cav"):
        validate_plan(free_plan(per_cav=per_cav), CS)


def test_locked_cav_may_not_stand_in_per_cav():
    native, _ = native_reference(CS, IDS)
    # drop wall W1 (members cav_a0/cav_a1 are locked) and set cav_a0 native
    walls = tuple(w for w in native.walls if w.wall_id != "W1")
    per_cav = dict(native.per_cav)
    per_cav["cav_a0"] = "native"
    per_cav["cav_a1"] = "native"  # full map: both freed members must appear
    with pytest.raises(PlanViolation, match="locked"):
        validate_plan(PlanCandidate(1, per_cav, walls), CS)


def test_locked_cav_may_not_switch_walls():
    native, _ = native_reference(CS, IDS)
    walls = [w for w in native.walls if w.wall_id != "W2"]
    walls.append(
        WallSpec("W9", ("cav_h0",), 6.0, PHASE_ASSEMBLE, None, 5.0)  # wrong wall_id
    )
    with pytest.raises(PlanViolation, match="locked"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), tuple(walls)), CS)


def test_wall_member_must_be_a_known_cav():
    native, _ = native_reference(CS, IDS)
    walls = list(native.walls)
    walls[-1] = dataclasses.replace(walls[-1], members=("cav_ghost",))
    with pytest.raises(PlanViolation, match="known CAV"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), tuple(walls)), CS)


@pytest.mark.parametrize(
    "release_at_s,hold_s",
    [(0.0, 10.0), (None, None)],
    ids=["both-set", "neither-set"],
)
def test_exactly_one_of_release_at_s_hold_s(release_at_s, hold_s):
    native, _ = native_reference(CS, IDS)
    w = WallSpec("W5", ("cav_n0",), 7.0, PHASE_ASSEMBLE, release_at_s, hold_s)
    with pytest.raises(PlanViolation, match="exactly one of"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), (w,)), CS)


def test_phase_must_be_known():
    native, _ = native_reference(CS, IDS)
    w = dataclasses.replace(native.walls[0], phase="paused")
    walls = (w,) + native.walls[1:]
    with pytest.raises(PlanViolation, match="phase"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), walls), CS)


def test_wall_memberships_must_be_disjoint():
    native, _ = native_reference(CS, IDS)
    walls = (
        WallSpec("W5", ("cav_n0",), 7.0, PHASE_ASSEMBLE, None, 5.0),
        WallSpec("W6", ("cav_n0", "cav_n1"), 7.0, PHASE_ASSEMBLE, None, 5.0),
    )
    with pytest.raises(PlanViolation, match="disjoint"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), walls), CS)


def test_wall_members_must_be_non_empty():
    native, _ = native_reference(CS, IDS)
    w = WallSpec("W5", (), 7.0, PHASE_ASSEMBLE, None, 5.0)
    with pytest.raises(PlanViolation, match="non-empty"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), (w,)), CS)


def test_wall_must_not_duplicate_a_member():
    native, _ = native_reference(CS, IDS)
    w = WallSpec("W5", ("cav_n0", "cav_n0"), 7.0, PHASE_ASSEMBLE, None, 5.0)
    with pytest.raises(PlanViolation, match="already belongs"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), (w,)), CS)


def test_wall_ids_must_be_unique():
    native, _ = native_reference(CS, IDS)
    w1 = WallSpec("W5", ("cav_n0",), 7.0, PHASE_ASSEMBLE, None, 5.0)
    w2 = WallSpec("W5", ("cav_n1",), 7.0, PHASE_ASSEMBLE, None, 5.0)
    with pytest.raises(PlanViolation, match="duplicate wall_id"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), (w1, w2)), CS)


@pytest.mark.parametrize(
    "speed",
    [0.0, -1.0, float("nan"), float("inf")],
    ids=["zero", "negative", "nan", "inf"],
)
def test_wall_speed_must_be_finite_positive(speed):
    native, _ = native_reference(CS, IDS)
    w = WallSpec("W5", ("cav_n0",), speed, PHASE_ASSEMBLE, None, 5.0)
    with pytest.raises(PlanViolation, match="speed_mps"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), (w,)), CS)


def test_release_at_s_must_be_finite_non_negative():
    native, _ = native_reference(CS, IDS)
    w = WallSpec("W5", ("cav_n0",), 7.0, PHASE_RELEASE, -1.0, None)
    with pytest.raises(PlanViolation, match="release_at_s"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), (w,)), CS)


def test_hold_s_must_be_non_negative():
    native, _ = native_reference(CS, IDS)
    w = WallSpec("W5", ("cav_n0",), 7.0, PHASE_ASSEMBLE, None, -5.0)
    with pytest.raises(PlanViolation, match="hold_s"):
        validate_plan(PlanCandidate(1, dict(native.per_cav), (w,)), CS)


# --- controller-state hygiene (§3.2), checked inside validate_plan ------------

@pytest.mark.parametrize(
    "entry",
    [
        ControllerStateEntry("wall", None, PHASE_ASSEMBLE, 0.0, 5.0, False),
        ControllerStateEntry("wall", "W1", None, 0.0, 5.0, False),
        ControllerStateEntry("yoyo"),
        ControllerStateEntry("native", "W1"),
        ControllerStateEntry("native", None, None, -1.0, None, False),
        ControllerStateEntry("native", None, None, 0.0, None, 1),  # locked must be bool
    ],
    ids=[
        "wall-without-wall-id",
        "wall-without-phase",
        "unknown-action",
        "native-with-wall-id",
        "negative-time-in-phase",
        "locked-not-bool",
    ],
)
def test_bad_controller_entries_are_rejected(entry):
    cs = dict(CS)
    cs["cav_x9"] = entry
    with pytest.raises(PlanViolation):
        validate_plan(free_plan(), cs)
