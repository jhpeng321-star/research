"""W2 gradual-release wall macro: shape, registered ranges, executor and executable plans (CPU)."""
import numpy as np
import pytest
import torch

from . import moves, wall_release as wr
from .moves import Move


def wall(shape='linear', release=10, onset=2, hold=20, depth=.8, members=('a', 'b', 'c')):
    return Move('wall', members, onset, hold, depth, release if shape != 'step' else 0, shape)


@pytest.mark.parametrize('shape', wr.RELEASE_SHAPES)
def test_release_profile_hold_then_monotone_ramp_back_to_base(shape):
    m = wall(shape, 10)
    series = [moves.offsets(m, t)['a'] for t in range(40)]
    assert series[:2] == [0., 0.]
    assert series[2:22] == [-.8]*20                                  # unchanged hold box
    ramp = series[22:32]
    assert all(-.8 < x < 0 for x in ramp)                            # strictly inside (0, depth)
    assert all(a < b for a, b in zip(ramp, ramp[1:]))                # monotone back to base
    assert np.mean(ramp) == pytest.approx(-.4)                       # midpoint-sampled ramp, mean depth/2
    assert series[32:] == [0.]*8
    if shape == 'linear':
        assert ramp == pytest.approx([-.8*(1-(k+.5)/10) for k in range(10)])
    else:
        assert ramp == pytest.approx([-.8*.5*(1+np.cos(np.pi*(k+.5)/10)) for k in range(10)])
    assert wr.duration(m) == 32
    # Every member of the wall releases together; release follows the merge-zone guard too.
    assert moves.offsets(m, 25) == {v: series[25] for v in m.members}
    assert moves.offsets(m, 25, state=dict(s_m={'b': moves._ZONE_S0})) == dict(a=0., b=0., c=0.)


def test_step_is_the_legacy_box_bytes_and_dicts():
    legacy = Move('wall', ('a', 'b'), 3, 25, .7)
    assert legacy == wall('step', members=('a', 'b'), onset=3, hold=25, depth=.7)
    assert legacy.to_dict() == dict(family='wall', members=['a', 'b'], onset=3, hold=25, depth=.7)
    assert Move.from_dict(legacy.to_dict()) == legacy
    ramp = wall('cosine', 15)
    assert ramp.to_dict()['release_s'] == 15 and ramp.to_dict()['release_shape'] == 'cosine'
    assert Move.from_dict(ramp.to_dict()) == ramp
    assert wr.descriptor(None) == [0.]*4 and wr.descriptor(legacy) == [0., 1., 0., 0.]
    assert wr.descriptor(ramp) == [.5, 0., 0., 1.]


@pytest.mark.parametrize('kwargs', [
    dict(family='gap_open', members=('a', 'b')), dict(family='ease_off', members=('a',)),
    dict(shape='square'), dict(release=4), dict(release=31), dict(release=10.5),
    dict(shape='step', release=5)])
def test_release_outside_registered_ranges_is_refused(kwargs):
    family = kwargs.pop('family', 'wall')
    members = kwargs.pop('members', ('a', 'b'))
    shape, release = kwargs.pop('shape', 'linear'), kwargs.pop('release', 10)
    with pytest.raises(ValueError):
        Move(family, members, 0, 20, .5, release, shape)
    with pytest.raises(ValueError):
        Move('wall', ('a', 'b'), 0, 20, 1.2, 10, 'linear')           # depth outside variety window


def test_executable_plan_binds_exactly_and_runs_through_the_existing_executors():
    from . import bank_breadth, group_lane
    from .plans import acceleration_at
    m = wall('linear', 12)
    policy = [[.1*i+.01*t for t in range(wr.duration(m))] for i in range(3)]
    plan = wr.plan(m, policy)
    # Same G5 executable fields as lane macros: profile = policy reference + macro offsets.
    fields = group_lane.executable_fields(m, wr.duration(m), policy)
    assert {k: plan[k] for k in fields} == fields
    assert moves.from_plan(plan) == m and bank_breadth.mined_move(plan) == m
    for t in range(plan['duration_s']):
        executed = acceleration_at(plan, t)
        assert [executed[v] for v in m.members] == pytest.approx(
            [policy[i][t]+moves.offsets(m, t)[v] for i, v in enumerate(m.members)])
    # Exact binding: a tampered profile, macro or duration is refused (never projected).
    for key, change in (('profile', lambda p: p['profile'][0].__setitem__(30, 9.)),
                        ('wall_macro', lambda p: p['wall_macro'].__setitem__('release_s', 13)),
                        ('duration_s', lambda p: p.__setitem__('duration_s', 40)),
                        ('knots', lambda p: p.__setitem__('knots', [[0, 0]]*3))):
        bad = {k: (v if not isinstance(v, (list, dict)) else __import__('copy').deepcopy(v))
               for k, v in plan.items()}
        change(bad)
        with pytest.raises(ValueError):
            moves.from_plan(bad)
    # A legacy box plan without wall_macro keeps the legacy projection path.
    box = Move('wall', ('a', 'b', 'c'), 2, 20, .8)
    legacy = dict(id='x', members=list(box.members), duration_s=22,
                  **group_lane.executable_fields(box, 22, [[0.]*22]*3))
    projected = moves.from_plan(legacy)
    assert (projected.family, projected.members, projected.onset, projected.hold) == \
        (box.family, box.members, box.onset, box.hold)
    assert projected.depth == pytest.approx(box.depth)
    assert projected.release_s == 0


def test_move_executor_keeps_the_commitment_through_the_release():
    from .track2_backend import MoveAPI, TriggerMove
    from .trigger_policy import MoveExecutor
    m = wall('cosine', 10, onset=0, hold=20, members=('a', 'b'))
    executor = MoveExecutor(moves_api=MoveAPI())
    assert executor.start(TriggerMove(m), ego='a', observable_cav_ids=['a', 'b'], now=100,
                          until=100+wr.duration(m))
    seen = []
    for t in range(100, 132):
        out = executor.requests(dict(a=.5, b=.5), dict(a=0., b=0.), now=t, eligible_cav_ids=['a', 'b'],
                                feasible_bounds=dict(a=(-3., 2.), b=(-3., 2.)))
        seen.append(round(out['a']['requested_raw']-.5, 10) if 'a' in out else 0.)
    assert seen[:20] == [-.8]*20 and all(-.8 < x < 0 for x in seen[20:30]) and seen[30:] == [0., 0.]


def test_decide_and_start_commits_onset_hold_and_release():
    from . import feasible_proposals as fp
    from .track2_backend import MoveAPI, TriggerMove
    from .trigger_policy import MoveExecutor, TriggerBatch
    m = wall('linear', 15, onset=3, hold=25, members=('a', 'b'))

    class Trigger:
        families = moves.FAMILIES

        def decide(self, context, **kwargs):
            return TriggerBatch(context, ((None, TriggerMove(m)),), torch.zeros(1, 2, 1),
                                torch.ones(1, 2, dtype=torch.bool), torch.tensor([1]),
                                torch.zeros(1), torch.zeros(1))
    executor = MoveExecutor(moves_api=MoveAPI())
    rows = [dict(vid='a', s_m=-400., v_mps=10., lane=0, is_cav=True),
            dict(vid='b', s_m=-405., v_mps=10., lane=1, is_cav=True)]
    _, accepted, _ = fp.decide_and_start(Trigger(), dict(members=['a'], x=[[0.]]),
        obs_states={'a': dict(self='a', rows=rows)}, observable_cavs={'a': ['a', 'b']},
        executor=executor, now=300, rng=np.random.default_rng(0), generator=torch.Generator())
    assert accepted == {'a': True}
    assert executor.active[0]['until'] == 300+3+25+15


def test_release_proposals_are_registered_feasible_and_arm_free():
    from . import feasible_proposals as fp
    rows = [dict(vid='a', s_m=-500., v_mps=10., lane=0, is_cav=True),
            dict(vid='b', s_m=-505., v_mps=10., lane=1, is_cav=True),
            dict(vid='c', s_m=-520., v_mps=10., lane=1, is_cav=True)]
    state = dict(self='a', rows=rows, wall_release=True)
    walls = [m for seed in range(40) for m in fp.candidates(state, np.random.default_rng(seed), 6)
             if m.family == 'wall']
    shapes = {m.release_shape for m in walls}
    assert shapes == {'step', 'linear', 'cosine'}
    by = {r['vid']: r for r in rows}
    for m in walls:
        arrival = min((moves._ZONE_S0-by[v]['s_m'])/by[v]['v_mps'] for v in m.members)
        assert wr.duration(m) <= arrival                          # the ramp ends before the zone
        if m.release_s:
            assert wr.RELEASE_WINDOW_S[0] <= m.release_s <= wr.RELEASE_WINDOW_S[1]
    # OFF (absent flag) proposals are exactly the legacy slate; the flag carries no arm input.
    legacy = fp.candidates(dict(self='a', rows=rows), np.random.default_rng(3), 6)
    assert all(m.release_s == 0 for m in legacy)
    with pytest.raises(ValueError):
        fp.candidates(dict(state, wall_release='yes'), np.random.default_rng(3), 6)
    # Near the zone there is no room for a registered ramp: the wall stays a step wall.
    near = [dict(r, s_m=r['s_m']+380.) for r in rows]
    for seed in range(20):
        for m in fp.candidates(dict(self='a', rows=near, wall_release=True), np.random.default_rng(seed), 6):
            if m.family == 'wall':
                assert m.release_s == 0 or wr.duration(m) <= min(
                    (moves._ZONE_S0-r['s_m'])/r['v_mps'] for r in near if r['vid'] in m.members)


def test_live_policy_with_release_commits_full_duration():
    from .test_wall_w2_cpu import base_cfg, run_policy
    p, samples = run_policy(dict(base_cfg(), wall_release=True, wall_pressure_features=True), ticks=4)
    for entry in p.executor.active:
        assert entry['until']-entry['start'] == wr.duration(entry['move'].move)
    tables = [m for s in samples if 'track2_trigger' in s for t in s['track2_trigger']['tables']
              for m in t if m]
    assert tables and all('family' in m for m in tables)
    width = len(p.trigger.families)+p.trigger.partner_slots+3+wr.DESCRIPTOR_DIM
    assert all(len(d) == width for s in samples if 'track2_trigger' in s
               for row in s['track2_trigger']['descriptors'] for d in row)
