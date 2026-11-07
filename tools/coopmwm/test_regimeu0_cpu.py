"""S-REGIMEU0 rule port: CPU contracts (no SUMO)."""
import pytest

from . import regimeu0 as ru


def cav(vid, s, wm_lane, v=25.):
    return dict(vid=vid, s_m=s, v_mps=v, lane=wm_lane, is_cav=True)


class Ctl:
    def __init__(self):
        self.g5_lane_targets = {}

    def apply(self, rows, commands, now):
        self.g5_lane_targets = {}
        return {}


def rows_wall():
    return [cav('a', -1000., 0), cav('b', -950., 1), cav('c', -900., 2, v=10.), cav('far', -100., 1),
            dict(vid='h', s_m=-990., v_mps=20., lane=0, is_cav=False)]


def test_pick_wall_geometry():
    w = ru.pick_wall(rows_wall()[:4])
    assert sorted(c['vid'] for c in w) == ['a', 'b', 'c']
    assert ru.pick_wall([cav('a', -1000., 0), cav('b', -950., 1)]) is None          # a lane missing
    assert ru.pick_wall([cav('a', -1000., 0), cav('b', -950., 1), cav('c', -700., 2)]) is None   # spread 300
    assert ru.pick_wall([cav('a', -200., 0), cav('b', -210., 1), cav('c', -220., 2)]) is None    # outside zone


@pytest.mark.parametrize('strategy,members', [('wall_h_solo2', ['a']), ('wall_h_solo0', ['c']),
                                              ('wall_h', ['a', 'b', 'c'])])
def test_rule_hook_activation_law_lane_hold_and_release(strategy, members):
    ctl = Ctl()
    hook = ru.RuleHook(strategy)
    hook(ctl, {}, {})
    rows = rows_wall()
    cmds = {r['vid']: .2 for r in rows if r['is_cav']}
    out = ctl.apply(rows, cmds, 300)
    assert sorted(out) == members and sorted(hook.acts[0]['wall']) == ['a', 'b', 'c'] and hook.next_ok == 390
    for vid in members:
        v = next(r['v_mps'] for r in rows if r['vid'] == vid)
        assert out[vid]['requested_raw'] == pytest.approx(max(-1.5, min(1.0, 15.-v)))
        assert ctl.g5_lane_targets[vid] == next(r['lane'] for r in rows if r['vid'] == vid)
    assert 'h' not in out and 'far' not in out
    out = ctl.apply(rows, cmds, 301)                                    # cooldown: no new group
    assert sorted(out) == members and len(hook.acts) == 1
    moved = [dict(r, s_m=80.) if r['vid'] == members[0] else r for r in rows]
    out = ctl.apply(moved, cmds, 302)                                   # past the drop -> released
    assert members[0] not in out and hook.acts[0]['release'][members[0]][1] == 'at_drop'
    out = ctl.apply(rows, cmds, 390)                                    # hold end -> released, new wall
    assert len(hook.acts) == 2


def test_base_arm_is_untouched():
    ctl = Ctl()
    original = ctl.apply
    hook = ru.RuleHook(None)
    hook(ctl, {}, {})
    assert ctl.apply == original


def test_wall_g_parameters_and_cooldown():
    ctl = Ctl()
    hook = ru.RuleHook('wall_g')
    hook(ctl, {}, {})
    rows = rows_wall()
    cmds = {r['vid']: .2 for r in rows if r['is_cav']}
    out = ctl.apply(rows, cmds, 300)
    assert sorted(out) == ['a', 'b', 'c'] and hook.next_ok == 420
    assert out['c']['requested_raw'] == pytest.approx(1.0)            # v=10 -> clip(22-10, -1, 1)
    assert out['a']['requested_raw'] == pytest.approx(-1.0)           # v=25 -> clip(22-25, -1, 1)


@pytest.mark.parametrize('strategy,members,law_v,next_ok', [('wall_m_solo2', ['a'], 18., 390),
                                                            ('wall_g_duo12', ['a', 'b'], 22., 420),
                                                            ('wall_g_solo2_gate0', ['a'], 22., 420)])
def test_multicrit_strategies(strategy, members, law_v, next_ok):
    ctl = Ctl()
    hook = ru.RuleHook(strategy)
    hook(ctl, {}, {})
    rows = rows_wall()
    cmds = {r['vid']: .2 for r in rows if r['is_cav']}
    out = ctl.apply(rows, cmds, 300)
    assert sorted(out) == members and hook.next_ok == next_ok and hook.gate_refusals == 0
    assert hook.acts[0]['sumo_lanes'] == [ru.sumo_lane(next(r['lane'] for r in rows if r['vid'] == m)) for m in hook.acts[0]['members']]
    for vid in members:
        v = next(r['v_mps'] for r in rows if r['vid'] == vid)
        assert out[vid]['requested_raw'] == pytest.approx(max(-1.0, min(1.0, law_v-v)))


def test_gate0_refuses_when_lane_occupied_behind():
    ctl = Ctl()
    hook = ru.RuleHook('wall_g_solo2_gate0')
    hook(ctl, {}, {})
    rows = rows_wall()+[dict(vid='h2', s_m=-1060., v_mps=28., lane=0, is_cav=False),     # 60 m behind 'a', same lane
                        dict(vid='h3', s_m=-1050., v_mps=28., lane=1, is_cav=False)]     # other lane: ignored
    cmds = {r['vid']: .2 for r in rows if r['is_cav']}
    out = ctl.apply(rows, cmds, 300)
    assert out == {} and hook.acts == [] and hook.gate_refusals == 1 and hook.next_ok == -1
    far = [dict(r, s_m=-1120.) if r['vid'] == 'h2' else r for r in rows]                 # 120 m behind: allowed
    out = ctl.apply(far, cmds, 301)
    assert sorted(out) == ['a'] and hook.next_ok == 421
    assert ru.behind_count(rows, rows[0], 100.) == 1


def test_seed_range_registered():
    from . import map_scenarios as ms
    w = ru.world_of(991000, 2400, 0.6)
    assert w['seed'] == 991000 and w['cell_key'] == 'D2400_p60' and w['repeat'] == 0
    assert ru.world_of(991099, 2800, 0.4)['cell_key'] == 'D2800_p40'
    with pytest.raises(ValueError):
        ru.world_of(991200, 2400, 0.6)
    assert ms.guard(w, 'train') == w


def _r2_rows(drop_v=8.0, l2_v=28.0, l1_v=12.0):
    rows = [cav('x', -1000., 0, v=27.), cav('y', -600., 0, v=27.), cav('z', -900., 1, v=12.),
            cav('far', -100., 1, v=drop_v)]
    rows += [dict(vid=f'h{i}', s_m=-500.+40*i, v_mps=drop_v, lane=1, is_cav=False) for i in range(10)]
    rows += [dict(vid='a2', s_m=-450., v_mps=l2_v, lane=0, is_cav=False), dict(vid='a1', s_m=-800., v_mps=l1_v, lane=1, is_cav=False)]
    return rows


def test_r2_picks_most_downstream_lane2_cav_when_situation_holds():
    ctl = Ctl()
    hook = ru.RuleHook('r2_solo2')
    hook(ctl, {}, {})
    rows = _r2_rows()
    cmds = {r['vid']: .2 for r in rows if r['is_cav']}
    out = ctl.apply(rows, cmds, 300)
    assert sorted(out) == ['y'] and hook.next_ok == 420 and hook.acts[0]['sumo_lanes'] == [2]
    assert out['y']['requested_raw'] == pytest.approx(-1.0) and hook.acts[0]['r2']['drop_speed'] < 15
    slow_l2 = [dict(r, v_mps=9.0) if r['lane'] == 0 else r for r in _r2_rows()]   # lane 2 not faster than lane 1
    for bad in (_r2_rows(drop_v=20.0), slow_l2):
        ctl2 = Ctl()
        h2 = ru.RuleHook('r2_solo2')
        h2(ctl2, {}, {})
        assert ctl2.apply(bad, {r['vid']: .2 for r in bad if r['is_cav']}, 300) == {} and h2.acts == []
