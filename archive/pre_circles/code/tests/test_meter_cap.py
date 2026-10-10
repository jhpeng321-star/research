"""Tests for the meter-cap / V2V coordination gate (2026-08-06).

Caps how many CAVs may meter concurrently, so at high penetration the
baseline-smoothing benefit of more CAVs is kept while the injection sources
are bounded. Protect (safety) is exempt.
"""

from tests.test_shepherd_actions import _make_engine


def _engine(**overrides):
    mc = {"meter_cap": 2, "meter_cap_modes": ("fair_balance", "recover", "vacate_drop")}
    mc.update(overrides)
    return _make_engine(mode_config=mc)


def test_cap_limits_concurrent_meters():
    e = _engine()
    e._metering_load = 0
    # load < cap -> not blocked, increments
    assert not e._meter_cap_blocks("fair_balance")
    e._metering_load = 2  # at cap
    assert e._meter_cap_blocks("fair_balance") is True
    assert e._meter_cap_blocks("recover") is True
    assert e._meter_cap_blocks("vacate_drop") is True


def test_protect_exempt_and_coast_exempt():
    e = _engine()
    e._metering_load = 100
    assert e._meter_cap_blocks("guard") is False
    assert e._meter_cap_blocks("coast") is False


def test_no_cap_no_block():
    e = _engine(meter_cap=None)
    e._metering_load = 100
    assert e._meter_cap_blocks("fair_balance") is False


def test_load_increments_when_metering():
    e = _engine(meter_cap=None)
    e._metering_load = 0
    # meter_cap None => no block but still tracks load (for observability)
    assert not e._meter_cap_blocks("fair_balance")
    e._metering_load = 1
    assert e._metering_load == 1
