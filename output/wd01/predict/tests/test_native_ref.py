"""§5 native reference: ASSEMBLE kept, HOLD dissolved, RELEASE dissolved,
native_kept_walls, unlocked variant. Fixture: W1 assemble (cav_a0+cav_a1),
W2 hold (cav_h0), W3 release (cav_r0), free cav_n0/cav_n1."""
import math

import pytest

from predict import (
    PHASE_ASSEMBLE,
    PHASE_RELEASE,
    ControllerStateEntry,
    NativeReferenceError,
    PlanCandidate,
    build_cav_ids,
    build_controller_state,
    native_reference,
    native_unlocked_reference,
    validate_plan,
)

CS = build_controller_state()
IDS = build_cav_ids()


def walls_by_id(plan):
    return {w.wall_id: w for w in plan.walls}


def test_assemble_wall_is_kept():
    native, kept = native_reference(CS, IDS)
    assert kept == ("W1",)
    w1 = walls_by_id(native)["W1"]
    assert w1.phase == PHASE_ASSEMBLE
    assert w1.members == ("cav_a0", "cav_a1")  # deterministic (sorted) order
    assert w1.hold_s == math.inf  # continues until a safety abort ends it
    assert w1.speed_mps == 5.0  # consensus of the controller entries


def test_hold_wall_is_dissolved():
    """Dissolution is model semantics ('members hand back to native', §5); the
    v1 plan encoding is a release-now wall: locked members may not move into
    per_cav (§3.3), so phase='release' + release_at_s=0.0 is 'dissolve at t0'."""
    native, kept = native_reference(CS, IDS)
    assert "W2" not in kept
    w2 = walls_by_id(native)["W2"]
    assert w2.phase == PHASE_RELEASE
    assert w2.release_at_s == 0.0
    assert w2.hold_s is None
    assert w2.members == ("cav_h0",)


def test_release_wall_is_dissolved():
    native, kept = native_reference(CS, IDS)
    assert "W3" not in kept
    w3 = walls_by_id(native)["W3"]
    assert w3.phase == PHASE_RELEASE
    assert w3.release_at_s == 0.0
    assert w3.members == ("cav_r0",)


def test_all_other_cavs_run_native():
    native, _ = native_reference(CS, IDS)
    assert native.plan_version == 1
    assert native.per_cav == {"cav_n0": "native", "cav_n1": "native"}
    # locked members never appear in per_cav
    for m in ("cav_a0", "cav_a1", "cav_h0", "cav_r0"):
        assert m not in native.per_cav


def test_native_candidate_is_a_valid_plan():
    assert validate_plan(native_reference(CS, IDS)[0], CS) is None


def test_native_kept_walls_lists_only_assemble_walls():
    _, kept = native_reference(CS, IDS)
    assert kept == ("W1",)
    assert "W2" not in kept and "W3" not in kept


def test_unlocked_variant_dissolves_every_wall():
    p = native_unlocked_reference(CS, IDS)
    validate_plan(p, CS)  # members stay wall members: they are locked
    assert set(walls_by_id(p)) == {"W1", "W2", "W3"}
    assert all(w.phase == PHASE_RELEASE and w.release_at_s == 0.0 for w in p.walls)
    assert p.per_cav == {"cav_n0": "native", "cav_n1": "native"}


def test_two_assemble_walls_both_kept_sorted():
    cs = dict(CS)
    cs["cav_n0"] = ControllerStateEntry("wall", "W7", PHASE_ASSEMBLE, 2.0, 6.0, True)
    cs["cav_n1"] = ControllerStateEntry("wall", "W7", PHASE_ASSEMBLE, 2.0, 6.0, True)
    native, kept = native_reference(cs, IDS)
    assert kept == ("W1", "W7")
    assert walls_by_id(native)["W7"].speed_mps == 6.0
    validate_plan(native, cs)


def test_no_walls_at_all():
    cs = {"cav_x": ControllerStateEntry("native")}
    native, kept = native_reference(cs, ["cav_x"])
    assert kept == () and native.walls == ()
    assert native.per_cav == {"cav_x": "native"}
    validate_plan(native, cs)


def test_members_disagreeing_on_phase_is_rejected():
    cs = dict(CS)
    cs["cav_a1"] = ControllerStateEntry("wall", "W1", None, 12.0, 5.0, True)
    with pytest.raises(NativeReferenceError, match="wall_phase"):
        native_reference(cs, IDS)


def test_assemble_wall_without_speed_is_rejected():
    cs = dict(CS)
    cs["cav_a0"] = ControllerStateEntry("wall", "W1", PHASE_ASSEMBLE, 12.0, None, True)
    with pytest.raises(NativeReferenceError, match="wall_speed_mps"):
        native_reference(cs, IDS)


def test_cav_ids_must_cover_controller_state():
    with pytest.raises(NativeReferenceError, match="outside"):
        native_reference(CS, ["cav_n0"])  # controller has entries outside cav_ids
    with pytest.raises(NativeReferenceError, match="not present"):
        native_reference(CS, IDS + ("cav_ghost",))


def test_native_member_order_is_deterministic():
    """controller_state carries no lane ids; member order is sorted(cav_id)
    (spec question, REPORT_PREDICT.md)."""
    cs = dict(CS)
    # same wall W1, members entered in reverse id order
    native, _ = native_reference(cs, IDS)
    assert walls_by_id(native)["W1"].members == ("cav_a0", "cav_a1")
