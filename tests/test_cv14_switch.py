"""Pure CV14 one-boundary switching checks; no SUMO."""
import pytest

from traffic_shepherd.fleet_switch import FleetSwitchSession, SingleSwitchSpec


def test_spec_rejects_repeat_and_invalid_modes():
    assert SingleSwitchSpec("damping", "damping", 1.).prefix == "damping"
    with pytest.raises(ValueError):
        SingleSwitchSpec("bad", "native", 1.)


def test_native_to_damping_keeps_owner_and_switches_once():
    calls = []
    circles = object()
    class Owner:
        controller = circles
        cav_ids = frozenset({"a"})
        def step(self, frames, **kwargs):
            calls.append((self.controller, kwargs["native_ids"]))
            return []
    owner = Owner()
    session = FleetSwitchSession(SingleSwitchSpec("native", "damping", 2.),
                                 circles_controller=circles, pi_session=None, cav_ids={"a"})
    session.step(owner, {}, now_s=1., live_ids={"a"})
    session.step(owner, {}, now_s=2., live_ids={"a"})
    assert len(calls) == 2 and calls[0][1] == {"a"} and calls[1][1] == set()
    session.step(owner, {}, now_s=3., live_ids={"a"})
    assert len(calls) == 3 and calls[-1][1] == set()
