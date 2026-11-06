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
