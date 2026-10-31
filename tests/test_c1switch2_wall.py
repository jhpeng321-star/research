"""Actual frozen-law continuation, rather than a lifecycle stub."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.c1switch2.wall import Wall
from traffic_shepherd.c4.switchable_wall import SwitchableWallController


def fixture():
    frame = SimpleNamespace(s=np.array([-1000., -1010., -1100.]), v=np.array([20., 20., 20.]),
        lane=np.array([0, 1, 0]), lane_carry=np.array([0, 1, 0]),
        present=np.ones(3, bool), is_cav=np.array([True, True, False]),
        junction=np.zeros(3, bool), vid_index=np.array([0, 1, 2]))
    builder = SimpleNamespace(vehicle_ids=['cav-a', 'cav-b', 'unobservable-id'])
    return frame, builder


def test_real_wall_memory_remaps_slots_and_preserves_future_requests():
    frame, builder = fixture()
    wall = SwitchableWallController()
    wall.start(now_s=300)
    for t in range(300, 320):
        wall.commands(frame, builder, now_s=t, live_cavs={'cav-a', 'cav-b'})
    env = SimpleNamespace(now=320, frame=frame, builder=builder, cav_ids={'cav-a', 'cav-b'},
        api=SimpleNamespace(vehicle=SimpleNamespace(getIDList=lambda: builder.vehicle_ids)))
    memory = Wall.capture_memory(wall, env)
    assert 'unobservable-id' not in str(memory)
    restored = Wall.restore(memory, slots={'cav-a': 1, 'cav-b': 0}, absent_slots=[], k=3)
    perm = [1, 0, 2]
    moved = SimpleNamespace(**{k: v[perm].copy() for k, v in vars(frame).items()})
    moved.vid_index = np.arange(3)
    anonymous = SimpleNamespace(vehicle_ids={0: '0', 1: '1', 2: '2'})
    for t in range(320, 380):
        expected = wall.commands(frame, builder, now_s=t, live_cavs={'cav-a', 'cav-b'})
        actual = restored.commands(moved, anonymous, now_s=t, live_cavs={'0', '1'})
        assert actual == {str({'cav-a': 1, 'cav-b': 0}[vid]): a for vid, a in expected.items()}
        assert (wall.law.executor is None) == (restored.law.executor is None)


def test_memory_rejects_hdv_extras_and_future_clocks():
    memory = dict(schema='c1switch2.wall/1', started_s=300., last_s=319., cooldown_s=None,
        members=['cav-a', 'cav-b'], speed_mps=17., phase=0,
        assembly_since_s=300., assembled_at_s=None)
    Wall.validate_memory(memory, {'cav-a', 'cav-b'}, 320)
    for key, value in [('members', ['cav-a', 'hdv']), ('last_s', 320.),
                       ('hidden_extra', 1), ('speed_mps', float('nan'))]:
        invalid = copy.deepcopy(memory)
        invalid[key] = value
        with pytest.raises(ValueError):
            Wall.validate_memory(invalid, {'cav-a', 'cav-b'}, 320)


def test_absent_member_aborts_group_without_leaking_or_reassigning_identity():
    memory = dict(schema='c1switch2.wall/1', started_s=300., last_s=319., cooldown_s=None,
        members=['cav-a', None], speed_mps=17., phase=0,
        assembly_since_s=300., assembled_at_s=None)
    Wall.validate_memory(memory, {'cav-a'}, 320)
    restored = Wall.restore(memory, slots={'cav-a': 0}, absent_slots=[1], k=3)
    frame, _ = fixture()
    frame.present[1] = False
    frame.vid_index = np.arange(3)
    restored.commands(frame, SimpleNamespace(vehicle_ids={i: str(i) for i in range(3)}),
                      now_s=320, live_cavs={'0'})
    assert restored.law.executor is None
    assert restored.law.cooldown_until == 350
