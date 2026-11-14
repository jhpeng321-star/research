"""Analytical regressions for the failed OE-28 positive-acceleration taper."""
from dataclasses import replace

import pytest

from traffic_shepherd.gentle_wall import GentleProfile
from traffic_shepherd.recovery_release import positive_speed_rise, resolve_command
from traffic_shepherd.sensing import SensorFrame


def test_taper_reserves_more_than_one_step_speed_gain():
    # a=.30 then .15 then 0 adds .45 m/s at a 1-second command cadence.
    assert positive_speed_rise(.3, .15, 1.) == pytest.approx(.45)
    frame = SensorFrame('m', 0., 26.7, .15, 27., 0)
    command, evidence = resolve_command(frame, .3, GentleProfile(), previous=.15)
    # a + (a-.15) = .30 permits at most .225 m/s².
    assert command == pytest.approx(.225)
    assert evidence['positive_taper_speed_rise_mps'] <= .3+1e-10


def test_unavoidable_overspeed_is_not_hidden_by_a_stronger_taper():
    frame = SensorFrame('m', 0., 26.9, .6, 27., 0)
    command, _ = resolve_command(frame, 0., GentleProfile(), previous=.6)
    assert command is None


def test_forward_command_response_reaches_limit_without_stranding_acceleration():
    p = GentleProfile()
    frame = SensorFrame('m', 0., 24., 0., 27., 0)
    previous = 0.
    for tick in range(30):
        command, _ = resolve_command(frame, .6*(27.-frame.speed_mps), p, previous=previous)
        assert command is not None
        assert -.3-1e-10 <= command <= 1.5+1e-10
        assert abs(command-previous) <= .15+1e-10
        speed = frame.speed_mps+command
        assert speed <= 27.+1e-10
        previous = command
        frame = replace(frame, speed_mps=speed, acceleration_mps2=command, timestamp_s=tick+1.)
    assert frame.speed_mps >= 26.9
