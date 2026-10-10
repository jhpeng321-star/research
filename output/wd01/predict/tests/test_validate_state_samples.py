"""§3.1 state-sample rules."""
import dataclasses

import pytest

from predict import (
    Arrival,
    ArrivalDraw,
    SampleMeta,
    StateSample,
    StateSamplesViolation,
    build_state_samples,
    validate_state_samples,
)

SAMPLES = build_state_samples(k=3)


def test_valid_samples_pass_and_are_returned():
    out = validate_state_samples(SAMPLES)
    assert isinstance(out, tuple) and len(out) == 3
    assert [s.meta.sample_id for s in out] == ["s0", "s1", "s2"]


def test_k_must_be_at_least_one():
    with pytest.raises(StateSamplesViolation, match="K >= 1"):
        validate_state_samples([])


def test_all_samples_share_time_s():
    late = dataclasses.replace(
        SAMPLES[1],
        meta=dataclasses.replace(SAMPLES[1].meta, sample_id="sX", time_s=SAMPLES[1].meta.time_s + 1.0),
    )
    with pytest.raises(StateSamplesViolation, match="time_s"):
        validate_state_samples([SAMPLES[0], late])


def test_sample_ids_must_be_unique():
    dup = dataclasses.replace(
        SAMPLES[1], meta=dataclasses.replace(SAMPLES[1].meta, sample_id=SAMPLES[0].meta.sample_id)
    )
    with pytest.raises(StateSamplesViolation, match="unique"):
        validate_state_samples([SAMPLES[0], dup])


@pytest.mark.parametrize("coverage", [1.5, -0.01, float("nan")], ids=["high", "negative", "nan"])
def test_coverage_must_be_in_unit_interval(coverage):
    bad = dataclasses.replace(
        SAMPLES[0], meta=dataclasses.replace(SAMPLES[0].meta, coverage=coverage)
    )
    with pytest.raises(StateSamplesViolation, match="coverage"):
        validate_state_samples([bad])


def test_merge_observed_must_be_bool():
    bad = dataclasses.replace(
        SAMPLES[0], meta=dataclasses.replace(SAMPLES[0].meta, merge_observed="yes")
    )
    with pytest.raises(StateSamplesViolation, match="merge_observed"):
        validate_state_samples([bad])


def test_hybrid_must_be_present():
    bad = dataclasses.replace(SAMPLES[0], hybrid=None)
    with pytest.raises(StateSamplesViolation, match="hybrid"):
        validate_state_samples([bad])


def test_arrival_draw_is_accepted():
    s = dataclasses.replace(SAMPLES[0], arrivals=ArrivalDraw(demand_model_id="wm30-base", seed=7))
    validate_state_samples([s])


def test_empty_arrival_list_is_accepted():
    s = dataclasses.replace(SAMPLES[0], arrivals=())
    validate_state_samples([s])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_s": -1.0},
        {"speed_mps": -3.0},
        {"entrance": ""},
        {"is_cav": "yes"},
        {"vehicle_type": ""},
        {"t_s": float("nan")},
    ],
    ids=["neg-t", "neg-speed", "empty-entrance", "is-cav-not-bool", "empty-vtype", "nan-t"],
)
def test_bad_arrival_rows_are_rejected(kwargs):
    row = Arrival(t_s=10.0, entrance="ramp", speed_mps=20.0, is_cav=False, vehicle_type="car")
    bad_row = dataclasses.replace(row, **kwargs)
    s = dataclasses.replace(SAMPLES[0], arrivals=(bad_row,))
    with pytest.raises(StateSamplesViolation):
        validate_state_samples([s])


def test_raw_mapping_samples_are_accepted():
    raw = {
        "hybrid": {"t0": 0.0, "section_len_m": 1000.0, "tracked": {}},
        "arrivals": (),
        "meta": {"sample_id": "sX", "time_s": 0.0, "merge_observed": False, "coverage": 0.5},
    }
    out = validate_state_samples([raw])
    assert isinstance(out[0], StateSample)


def test_single_sample_call_is_valid():
    validate_state_samples(SAMPLES[:1])
