"""Independent scorer contracts; these synthetic fixtures are not traffic evidence."""
from copy import deepcopy
import json

import pytest

from traffic_shepherd.formation_evidence import (
    join_physics_and_phases, score_formation, score_formation_files,
)


def proposal():
    return dict(start_s=0., required_lanes=["L0", "L1"],
        members=[dict(vehicle_id="a", lane_id="L0"), dict(vehicle_id="b", lane_id="L1")],
        target_speed_mps=10., profile=dict(control_dt_s=1., assembly_deadline_s=5.,
            hold_s=2., taper_s=2., formed_span_m=50., formed_speed_spread_mps=1.,
            target_speed_tolerance_mps=1., readiness_samples=3))


def fixture(*, native=False, tail=False):
    terminal = 4. if native else 6.
    end = terminal+2 if tail else terminal
    samples, telemetry = [], []
    for tick in range(int(end*2)+1):
        now = tick*.5
        phase = "assembly" if now < 2 else "hold" if now < 4 else "taper" if now < terminal else "release"
        reason = "identical_damping_requests_native_explicit_handback" if native and now >= terminal else "fixture"
        samples.append(dict(time_s=now, phase=phase, reason=reason, members=[
            dict(id="a", lane="L0", present=True, distance_m=1000.-now*10, speed_mps=10.),
            dict(id="b", lane="L1", present=True, distance_m=1010.-now*10, speed_mps=10.)]))
        if now.is_integer():
            for member in ("a", "b"):
                wall = now < terminal
                telemetry.append(dict(kind="issued_command", vehicle_id=member, timestamp_s=now,
                    owner="wall" if wall else "native" if native else "damping",
                    wall_request_mps2=0. if wall else None,
                    issued_acceleration_mps2=None if native and not wall else 0.,
                    issued_target_speed_mps=None if native and not wall else 10.,
                    fallback_reason="missing_front_detection" if native and not wall else None,
                    released_to_native=bool(native and now == terminal)))
    return samples, telemetry


def test_measured_readiness_hold_taper_and_ownership_are_scored_independently():
    result = score_formation(*fixture(), proposal=proposal())
    assert result["delivered"] and not result["safety_established"]
    assert result["assembly_complete_at_s"] == 2.
    assert result["compact_hold_seconds"] == 2.
    assert result["longest_contiguous_compact_hold_seconds"] == 2.
    assert result["taper_seconds"] == 2. and result["full_taper_completed"]
    assert result["release_ownership_verified"] and not result["failure_reasons"]


def test_half_second_samples_cannot_be_counted_as_one_second_readiness_ticks():
    rows, commands = fixture()
    for row in rows:
        if row["time_s"] < 1.:
            for member in row["members"]:
                member["speed_mps"] = 12.
    result = score_formation(rows, commands, proposal=proposal())
    assert result["readiness_trace"][-1]["consecutive_control_samples"] == 2
    assert not result["delivered"]
    assert "entry_without_current_target_settled_readiness" in result["failure_reasons"]


def test_matching_speed_and_compactness_without_target_settling_is_not_readiness():
    rows, commands = fixture()
    for row in rows:
        for member in row["members"]:
            member["speed_mps"] = 15.
    result = score_formation(rows, commands, proposal=proposal())
    assert result["first_measured_readiness_s"] is None
    assert not result["assembly_complete"] and not result["delivered"]


@pytest.mark.parametrize("failure", ["span", "lane", "missing"])
def test_a_half_second_hold_failure_is_visible_even_when_control_ticks_are_good(failure):
    rows, commands = fixture()
    row = next(r for r in rows if r["time_s"] == 2.5)
    if failure == "span":
        row["members"][1]["distance_m"] += 100.
    elif failure == "lane":
        row["members"][1]["lane"] = "L0"
    else:
        row["members"][1] = dict(id="b", present=False)
    result = score_formation(rows, commands, proposal=proposal())
    assert not result["delivered"] and not result["compact_hold_delivered"]
    assert "noncompact_or_missing_hold_boundary" in result["failure_reasons"]


def test_hold_end_boundary_is_measured_before_taper_and_cannot_hide_loss():
    rows, commands = fixture()
    next(r for r in rows if r["time_s"] == 4.)["members"][1]["distance_m"] += 100.
    assert not score_formation(rows, commands, proposal=proposal())["compact_hold_delivered"]


def test_shortened_native_handback_is_explicit_and_not_called_full_taper():
    result = score_formation(*fixture(native=True), proposal=proposal())
    assert result["delivered"] and result["explicit_native_handback"]
    assert result["taper_seconds"] == 0. and not result["full_taper_completed"]
    assert result["terminal_ownership"]["native_members"] == ["a", "b"]


def test_native_release_must_show_actual_release_of_the_previous_wall_override():
    rows, commands = fixture(native=True)
    commands[-1]["released_to_native"] = False
    result = score_formation(rows, commands, proposal=proposal())
    assert not result["delivered"] and not result["release_ownership_verified"]
    assert "native_override_not_released" in result["failure_reasons"]


def test_favorable_phase_string_cannot_replace_actual_terminal_command_owner():
    rows, commands = fixture()
    commands[-1].update(owner="wall", wall_request_mps2=0.)
    result = score_formation(rows, commands, proposal=proposal())
    assert result["compact_hold_delivered"] and not result["delivered"]
    assert "wall_command_retained_at_terminal" in result["failure_reasons"]


def test_departed_members_after_release_do_not_break_complete_wall_scoring():
    rows, commands = fixture(tail=True)
    for row in rows:
        if row["time_s"] > 6.:
            row["members"] = [dict(id="a", present=False), dict(id="b", present=False)]
    result = score_formation(rows, commands, proposal=proposal())
    assert result["delivered"] and result["ignored_postterminal_geometry_samples"] == 4


def test_postterminal_wall_command_cannot_be_hidden_by_geometry_trimming():
    rows, commands = fixture(tail=True)
    commands[-1].update(owner="wall", wall_request_mps2=0.)
    result = score_formation(rows, commands, proposal=proposal())
    assert not result["delivered"] and "wall_command_after_terminal" in result["failure_reasons"]


def test_absent_member_abort_is_retained_and_other_member_handback_is_verified():
    rows, commands = fixture()
    rows = [r for r in rows if r["time_s"] <= 1.]
    rows[-1].update(phase="abort", reason="missing_member_report")
    rows[-1]["members"][1] = dict(id="b", present=False)
    commands = [r for r in commands if r["timestamp_s"] < 1. or
                (r["timestamp_s"] == 1. and r["vehicle_id"] == "a")]
    commands[-1].update(owner="native", wall_request_mps2=None, issued_acceleration_mps2=None,
                        issued_target_speed_mps=None, released_to_native=True)
    result = score_formation(rows, commands, proposal=proposal())
    assert not result["delivered"] and result["terminal_phase"] == "abort"
    assert result["abort_ownership_verified"]
    assert result["terminal_ownership"]["absent_members"] == ["b"]
    assert result["terminal_reason"] == "missing_member_report"


def test_missing_measurement_abort_is_an_outcome_not_a_keyerror():
    rows, commands = fixture()
    rows = rows[:3]
    rows[-1].update(phase="abort", reason="measurement unavailable")
    rows[-1]["members"][1] = dict(id="b", present=True, measurement_unavailable="sensor overlap")
    commands = [r for r in commands if r["timestamp_s"] <= 1.]
    for row in commands[-2:]:
        row.update(owner="native", wall_request_mps2=None, released_to_native=True)
    result = score_formation(rows, commands, proposal=proposal())
    assert result["abort_ownership_verified"] and not result["delivered"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "invalid"])
def test_incomplete_physics_sampling_is_not_silently_imputed(kind):
    rows, commands = fixture()
    if kind == "missing":
        rows.pop(5)
    elif kind == "duplicate":
        rows[5]["time_s"] = rows[4]["time_s"]
    else:
        rows[5]["time_s"] = float("nan")
    with pytest.raises(ValueError, match="sample|clock"):
        score_formation(rows, commands, proposal=proposal())


def test_phase_cannot_change_at_an_unregistered_half_second_boundary():
    rows, commands = fixture()
    next(r for r in rows if r["time_s"] == 1.5)["phase"] = "hold"
    result = score_formation(rows, commands, proposal=proposal())
    assert "phase_changed_off_control_grid" in result["failure_reasons"]
    assert not result["delivered"]


def test_no_terminal_sample_has_no_invented_handback():
    rows, commands = fixture()
    result = score_formation(rows[:-1], commands, proposal=proposal())
    assert not result["delivered"] and not result["complete_observation_horizon"]
    assert "terminal_not_observed" in result["failure_reasons"]


def test_assembly_only_is_never_counted_as_completed_wall_hold():
    rows, commands = fixture()
    rows = [r for r in rows if r["time_s"] <= 4.]
    for row in rows:
        if row["time_s"] >= 2.:
            row["phase"] = "taper" if row["time_s"] < 4. else "release"
    commands = [r for r in commands if r["timestamp_s"] <= 4.]
    for row in commands[-2:]:
        row.update(owner="damping", wall_request_mps2=None)
    result = score_formation(rows, commands, proposal=proposal(), arm="assembly_only")
    assert result["assembly_only_delivered"] and not result["delivered"]
    assert not result["compact_hold_delivered"]


def test_joined_offline_physics_does_not_use_future_control_phases():
    rows, _ = fixture()
    controls = [dict(time_s=r["time_s"], phase=r["phase"], reason=r["reason"])
                for r in rows if r["time_s"].is_integer()]
    physics = [dict(time_s=r["time_s"], members=deepcopy(r["members"])) for r in rows]
    physics[1]["members"].pop()
    joined = join_physics_and_phases(physics, controls, start_s=0., member_ids=["a", "b"])
    assert joined[3]["phase"] == "assembly" and joined[3]["command_time_s"] == 1.
    assert joined[1]["members"][1] == dict(id="b", present=False)
    with pytest.raises(ValueError, match="missing controlling"):
        join_physics_and_phases(physics, controls[1:], start_s=0., member_ids=["a", "b"])


def test_file_adapter_binds_both_raw_inputs(tmp_path):
    rows, commands = fixture()
    paths = [tmp_path / "formation.jsonl", tmp_path / "telemetry.jsonl"]
    for path, values in zip(paths, (rows, commands)):
        path.write_text("".join(json.dumps(r)+"\n" for r in values))
    result = score_formation_files(*paths, proposal=proposal())
    assert result["delivered"] and set(result["source_sha256"]) == {"formation", "telemetry"}
    assert all(len(value) == 64 for value in result["source_sha256"].values())
