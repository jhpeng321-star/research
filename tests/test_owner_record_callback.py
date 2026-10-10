"""Per-vehicle acknowledgement must reflect successful writes, not batch intent."""
from types import SimpleNamespace
from pathlib import Path
import importlib.util
import sys

import pytest

from traffic_shepherd.config import load_config
from traffic_shepherd.damping import DampingDecision
from traffic_shepherd.ownership import CommandOwnership


class Controller:
    config = load_config()

    def decide(self, frame, *, now_s):
        return DampingDecision(command_acceleration_mps2=1., target_speed_mps=11.)


class API:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def getSpeedMode(self, vehicle):
        return 31

    def slowDown(self, vehicle, target, duration):
        if vehicle == self.fail:
            raise RuntimeError('write failure')
        self.calls.append(('slowDown', vehicle, target, duration))

    def setSpeed(self, vehicle, speed):
        self.calls.append(('setSpeed', vehicle, speed))


def make(fail=None):
    api = API(fail)
    return CommandOwnership(SimpleNamespace(vehicle=api), {'a','b','c'}, Controller()), api


def test_successful_first_write_receipted_before_second_failure():
    owner, api = make('b')
    receipts = []

    def receive(row):
        assert owner._last[row.vehicle_id] is row
        assert api.calls[-1][1] == row.vehicle_id
        receipts.append(row)

    with pytest.raises(RuntimeError, match='write failure'):
        owner.step({}, now_s=0, live_ids={'a','b','c','hdv'}, record_callback=receive)
    assert [r.vehicle_id for r in receipts] == ['a']
    assert [c[1] for c in api.calls] == ['a']
    assert set(owner._last) == {'a'}


def test_callback_exception_preserves_already_written_record():
    owner, api = make()

    def fail(row):
        raise RuntimeError('receipt failure')

    with pytest.raises(RuntimeError, match='receipt failure'):
        owner.step({}, now_s=0, live_ids={'a','b'}, record_callback=fail)
    assert list(owner._last) == ['a']
    assert [c[1] for c in api.calls] == ['a']


def test_none_callback_and_recording_callback_have_same_commands():
    first, a = make()
    second, b = make()
    received = []
    for t, native in [(0,()), (1,('a',)), (2,())]:
        left = first.step({}, now_s=t, live_ids={'a'}, native_ids=native)
        right = second.step({}, now_s=t, live_ids={'a'}, native_ids=native, record_callback=received.append)
        assert left == right
    assert a.calls == b.calls
    assert [r.owner for r in received] == ['damping','native','damping']
    assert received[1].released_to_native


def test_noncallable_callback_rejected_before_write():
    owner, api = make()
    with pytest.raises(TypeError, match='record_callback'):
        owner.step({}, now_s=0, live_ids={'a'}, record_callback=17)
    assert not api.calls


def test_default_trace_matches_preserved_owner_before_callback_seam():
    source = Path(__file__).resolve().parents[1] / 'output/controller_verification_v7/source_preservation/ownership.py'
    name = 'traffic_shepherd._cv07_preserved_owner'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        old_api, new_api = API(), API()
        old = module.CommandOwnership(SimpleNamespace(vehicle=old_api), {'a'}, Controller())
        new = CommandOwnership(SimpleNamespace(vehicle=new_api), {'a'}, Controller())
        for t, live, native in [(0,{'a','hdv'},()), (1,{'a'},('a',)), (2,{'a'},()), (3,{'hdv'},())]:
            kwargs = dict(now_s=t, live_ids=live, native_ids=native)
            assert [r.as_dict() for r in old.step({}, **kwargs)] == [r.as_dict() for r in new.step({}, **kwargs)]
        assert old_api.calls == new_api.calls
    finally:
        sys.modules.pop(name, None)
