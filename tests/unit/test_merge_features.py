"""T2F CPU-only observation, latency, locality and additive-block contracts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from tools.coopmwm import merge_features as mf

ON = mf.MergeFeaturesConfig(merge_features=True)


def vehicle(vid, s=-20.0, lane=1, speed=10.0, length=5.0):
    return mf.ObservedVehicle(vid, s, lane, speed, length)


def report(vid="ego", s=-20.0, lane=1, speed=10.0, *, time=10.0,
           covered=(0, 1, 2), rows=()):
    return mf.SensorReport(time, vehicle(vid, s, lane, speed), covered, tuple(rows))


def features(own=None, reports=None, *, partners=(), now=10.0):
    return mf.merge_features(own, "ego", reports, now_s=now, partner_ids=partners, cfg=ON)


def values(block):
    return dict(zip(mf.MERGE_FEATURES, block.tolist()))


def test_shape_layout_and_existing_constant_sources():
    from wms.control import contract
    from wms.control.rsu_field import QUEUE_SPEED_MPS
    from tools.coopmwm.v2v_obs import V2V_K, V2V_RANGE_M

    for name in ("SENSOR_RANGE_M", "CONTROL_DT_S", "MERGE_S0_M", "MERGE_S1_M", "N_LANES"):
        assert getattr(mf, name) == getattr(contract, name)
    assert mf.QUEUE_SPEED_MPS == QUEUE_SPEED_MPS
    assert mf.V2V_K == V2V_K and mf.V2V_RANGE_M == V2V_RANGE_M
    assert mf.MERGE_DIM == len(mf.MERGE_FEATURES) == 29
    assert len(set(mf.MERGE_FEATURES)) == mf.MERGE_DIM
    assert mf.MERGE_FEATURES[:8] == tuple(f"lane_0_{x}" for x in mf.MERGE_LANE_FEATURES)
    out = features(report())
    assert out.shape == (mf.MERGE_DIM,) and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert mf.MergeFeaturesConfig().merge_features is False


class Unreadable:
    def __getattribute__(self, name):
        raise AssertionError(f"disabled feature accessed input: {name}")


def test_default_off_reads_nothing_and_preserves_original_mapping_bytes():
    base = {"own": np.arange(13, dtype=np.float32), "v2v": np.arange(48, dtype=np.float32)}
    before = {key: array.tobytes() for key, array in base.items()}
    bomb = Unreadable()
    out = mf.merge_features(bomb, "ego", bomb, now_s=float("nan"), partner_ids=bomb)
    assert out.tobytes() == np.zeros(mf.MERGE_DIM, np.float32).tobytes()
    result = mf.with_merge_features(base, bomb, "ego", bomb, now_s=float("nan"),
                                   partner_ids=bomb)
    assert result is base
    assert {key: array.tobytes() for key, array in base.items()} == before
    assert "merge_features" not in base


def test_enabled_adds_only_separate_block_without_mutating_base_or_packets():
    base = {"own": np.arange(13, dtype=np.float32), "v2v": np.zeros(48, np.float32)}
    own = report(rows=(vehicle("rear", -40),))
    original = repr(own)
    result = mf.with_merge_features(base, own, "ego", now_s=10, cfg=ON)
    assert result is not base and set(result) == {*base, "merge_features"}
    for key in base:
        assert result[key] is base[key]
    np.testing.assert_array_equal(result["merge_features"], features(own))
    assert "merge_features" not in base and repr(own) == original


def test_local_queue_counts_speeds_and_rear_bumper_geometry():
    own = report(rows=(
        vehicle("ramp_queue", -40, 2, 0), vehicle("ramp_slow", -10, 2, 4),
        vehicle("ramp_front", 20, 2, 12), vehicle("ramp_downstream", 76, 2, 0),
        vehicle("main_rear", -60, 1, 8), vehicle("left", -30, 0, 3),
    ))
    v = values(features(own))
    assert v["lane_2_queue_length_veh"] == 2
    assert v["lane_2_approaching_count"] == 2
    assert v["lane_2_approaching_mean_speed_mps"] == 2
    assert v["lane_2_rear_gap_m"] == 15 and v["lane_2_rear_speed_mps"] == 0
    assert v["lane_1_approaching_count"] == 2  # includes ego, not just neighbours
    assert v["lane_1_approaching_mean_speed_mps"] == 9
    assert v["lane_1_rear_gap_m"] == 35 and v["lane_1_rear_speed_mps"] == 8
    assert v["lane_0_queue_length_veh"] == 1
    assert v["visible_vehicle_count"] == 7
    assert v["own_missing"] == 0 and v["partner_reports_missing"] == 1


def test_empty_covered_lane_is_distinct_from_unobserved_lane():
    v = values(features(report(covered=(1,))))
    assert v["lane_1_lane_missing"] == 0
    assert v["lane_0_lane_missing"] == v["lane_2_lane_missing"] == 1
    assert v["lane_0_approaching_count"] == v["lane_2_approaching_count"] == 0
    assert v["lane_2_approaching_speed_missing"] == v["lane_2_rear_missing"] == 1
    covered = values(features(report(covered=(1, 2))))
    assert covered["lane_2_lane_missing"] == 0
    assert covered["lane_2_approaching_speed_missing"] == covered["lane_2_rear_missing"] == 1


def test_missing_ego_does_not_authorize_any_partner():
    v = values(features(None, {"p": report("p", time=9)}, partners=("p",)))
    assert v["own_missing"] == v["partner_reports_missing"] == 1
    assert v["visible_vehicle_count"] == v["partner_report_count"] == 0
    for lane in range(3):
        assert all(v[f"lane_{lane}_{key}"] == 1 for key in
                   ("lane_missing", "approaching_speed_missing", "rear_missing"))


@pytest.mark.parametrize("time", [8.0, 8.9, 9.1, 9.9, 10.0, 11.0, np.nan])
def test_only_exact_t_minus_one_partner_publications_are_visible(time):
    own = report(covered=(1,))
    baseline = features(own)
    p = report("p", -50, 2, 1, time=time, covered=(2,))
    np.testing.assert_array_equal(features(own, {"p": p}, partners=("p",)), baseline)


def test_valid_delayed_packet_and_fractional_time_without_integer_truncation():
    own = report(time=10.25, covered=(1,))
    p = report("p", -50, 2, 0, time=9.25, covered=(2,),
               rows=(vehicle("hdv", -80, 2, 2),))
    v = values(features(own, {"p": p}, partners=("p",), now=10.25))
    assert v["partner_report_count"] == v["partner_report_age_s"] == 1
    assert v["partner_reports_missing"] == 0
    assert v["lane_2_queue_length_veh"] == v["lane_2_approaching_count"] == 2
    assert v["lane_2_lane_missing"] == 0 and v["visible_vehicle_count"] == 3
    invalid = replace(p, timestamp_s=9.75)
    np.testing.assert_array_equal(
        features(own, {"p": invalid}, partners=("p",), now=10.25),
        features(own, now=10.25),
    )


@pytest.mark.parametrize("time", [9.0, 10.1, 11.0, np.nan])
def test_own_sensing_must_be_current(time):
    assert values(features(report(time=time)))["own_missing"] == 1


def test_explicit_partner_membership_identity_range_and_no_self_reports():
    own = report()
    p = report("p", -120, 2, 1, time=9)
    baseline = features(own)
    np.testing.assert_array_equal(features(own, {"p": p}), baseline)
    v = values(features(own, {"p": p}, partners=("p", "p")))
    assert v["partner_report_count"] == 1  # exactly at radio range, deduplicated
    for invalid in (replace(p, source=vehicle("impostor", -50, 2, 1)),
                    replace(p, source=vehicle("p", -120.001, 2, 1))):
        np.testing.assert_array_equal(features(own, {"p": invalid}, partners=("p",)), baseline)
    np.testing.assert_array_equal(
        features(own, {"ego": replace(own, timestamp_s=9)}, partners=("ego",)), baseline,
    )


def test_rows_limited_to_each_publishers_sensor_range_and_declared_lanes():
    own = report(covered=(1,), rows=(vehicle("local_too_far", -120.001, 1, 1),
                                   vehicle("uncovered", -30, 2, 1)))
    p = report("p", 70, 2, 10, time=9, covered=(2,), rows=(
        vehicle("at_limit", 170, 2, 1), vehicle("too_far", 170.001, 2, 1),
        vehicle("uncovered_remote", 80, 0, 1),
    ))
    v = values(features(own, {"p": p}, partners=("p",)))
    assert v["visible_vehicle_count"] == 3  # ego, publisher, boundary sensor row
    assert v["lane_0_lane_missing"] == 1
    assert v["lane_2_queue_length_veh"] == 0  # visible row beyond merge end


def test_duplicates_use_local_then_nearest_publisher_and_no_double_count():
    own = report(rows=(vehicle("shared", -30, 2, 2),))
    near = report("near", 0, 1, 10, time=9, rows=(vehicle("shared", -30, 2, 4),
                                                           vehicle("remote", -40, 2, 1)))
    far = report("far", 30, 1, 10, time=9, rows=(vehicle("shared", -30, 2, 0),
                                                          vehicle("remote", -40, 2, 3)))
    v = values(features(own, {"far": far, "near": near}, partners=("far", "near")))
    assert v["lane_2_approaching_count"] == 2
    assert v["lane_2_approaching_mean_speed_mps"] == 1.5
    assert v["visible_vehicle_count"] == 5


def test_partner_cap_and_determinism_ignore_mapping_and_packet_order():
    own = report()
    reports = {f"p{i}": report(f"p{i}", -20 + i * 10, 2, i, time=9) for i in range(1, 7)}
    ids = tuple(reports)
    out = features(own, reports, partners=ids)
    v = values(out)
    assert v["partner_report_count"] == 4 and v["visible_vehicle_count"] == 5
    assert v["lane_2_queue_length_veh"] == 4
    np.testing.assert_array_equal(out, features(own, dict(reversed(list(reports.items()))),
                                              partners=tuple(reversed(ids))))
    rows = (vehicle("b", -30, 2, 2), vehicle("a", -30, 2, 3))
    np.testing.assert_array_equal(features(report(rows=rows)), features(report(rows=rows[::-1])))
    assert values(features(report(rows=rows)))["lane_2_rear_speed_mps"] == 3  # ID tie


def test_publisher_ties_resolve_by_id_not_input_order():
    own = report()
    a = report("a", -10, time=9, rows=(vehicle("shared", -30, 2, 1),))
    b = report("b", -30, time=9, rows=(vehicle("shared", -30, 2, 3),))
    v = values(features(own, {"b": b, "a": a}, partners=("b", "a")))
    assert v["lane_2_approaching_mean_speed_mps"] == 1


def test_queue_merge_and_speed_boundaries_and_bumper_overlap():
    own = report(s=20, rows=(
        vehicle("exact_speed", -1, 2, 5), vehicle("below_speed", 0, 2, 4.999),
        vehicle("end", mf.MERGE_S1_M, 2, 0),
        vehicle("past_end", mf.MERGE_S1_M + 0.001, 2, 0),
        vehicle("overlap", 18, 1, 1),
    ))
    v = values(features(own))
    assert v["lane_2_queue_length_veh"] == 2
    assert v["lane_2_approaching_count"] == 1
    assert v["lane_2_approaching_mean_speed_mps"] == 5
    assert v["lane_1_rear_gap_m"] == 0 and v["lane_1_rear_missing"] == 0


@pytest.mark.parametrize("bad_row", [
    vehicle("bad", s=np.nan), vehicle("bad", speed=np.inf), vehicle("bad", speed=-1),
    vehicle("bad", lane=-1), vehicle("bad", lane=3), vehicle("bad", length=0),
    vehicle("bad", length=np.nan), vehicle(""), {"s_m": -20, "v_mps": 0},
])
def test_malformed_atomic_packets_are_missing_instead_of_fabricated_empty(bad_row):
    own = report()
    p = report("p", time=9, rows=(bad_row,))
    np.testing.assert_array_equal(features(own, {"p": p}, partners=("p",)), features(own))
    assert values(features(report(rows=(bad_row,))))["own_missing"] == 1


def test_duplicate_ids_and_invalid_lane_coverage_invalidate_atomic_packet():
    for invalid in (report(rows=(vehicle("same"), vehicle("same"))),
                    report(covered=(1, 3))):
        assert values(features(invalid))["own_missing"] == 1


class SelectedReportsOnly(Mapping):
    """Any iteration/global scan or unselected source access is a test failure."""

    def __init__(self, packet):
        self.packet = packet

    def __getitem__(self, key):
        assert key == "p"
        return self.packet

    def __iter__(self):
        raise AssertionError("global report scan")

    def __len__(self):
        return 1


def test_no_global_scan_and_unselected_unobservable_traffic_cannot_change_block():
    own = report()
    p = report("p", -40, 2, 0, time=9)
    expected = features(own, {"p": p}, partners=("p",))
    np.testing.assert_array_equal(
        features(own, SelectedReportsOnly(p), partners=("p",)), expected,
    )
    np.testing.assert_array_equal(
        features(own, {"p": p, "global_queue": Unreadable(), "unselected": Unreadable()},
                 partners=("p",)), expected,
    )
    # A state-like mapping is not accepted as an observable packet.
    assert values(features({"ego": own.source, "all_vehicles": Unreadable()}))["own_missing"] == 1


def test_no_cached_report_survives_into_next_step_or_new_episode():
    own = report()
    p = report("p", -40, 2, 0, time=9)
    assert values(features(own, {"p": p}, partners=("p",)))["partner_report_count"] == 1
    next_own = replace(own, timestamp_s=11)
    np.testing.assert_array_equal(features(next_own, {"p": p}, partners=("p",), now=11),
                                  features(next_own, now=11))
    np.testing.assert_array_equal(features(own, {}, partners=("p",)), features(own))


@pytest.mark.parametrize("now", [np.nan, np.inf, -np.inf])
def test_enabled_requires_finite_decision_time(now):
    with pytest.raises(ValueError, match="now_s must be finite"):
        features(report(), now=now)
