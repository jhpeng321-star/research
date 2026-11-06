"""clean5b CPU contracts: FULLPOL slot templates for LCEXEC's per-CAV initiate/join protocol.

Real lc_program FreeProgram / MemberProgram validation, registry check and MemberExecutor on the
generated templates (slot ids resolved to concrete vehicles); no SUMO. The ProgramOption shape
follows the LCEXEC contract (lc_families, ~21:3x) through a local test stub until it lands.
"""
from collections import Counter
from importlib import import_module

import numpy as np
import pytest

from . import fullpol_head as fh, fullpol_programs as fp, fullpol_templates as ft, lc_program as lc
from .lane_drop import MapView
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)

VIEW = MapView('wz3to2-stub', 0., (0, 1, 2), 2)


def row(vid, s, lane, cav=True, v=20.):
    return dict(vid=vid, s_m=float(s), lane=lane, v_mps=v, is_cav=cav)


def scene_rows(own, sparse=False):
    s = float(own['s_m'])
    cavs = [row('a', s+30, 0), row('b', s-20, 1)]+([] if sparse else [row('c', s+10, 2)])
    return cavs+[row('h1', s+50, 1, cav=False), row('h2', s-60, 2, cav=False), row('h3', s+5, own['lane'], cav=False)]


def draws(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        own = row('ego', -(150.+i % 23*90.), i % 3)
        for t in ft.propose(dict(frame=VIEW, rows=scene_rows(own, sparse=bool(i % 2))), own, rng):
            out.append((own, t))
    return out


def test_hook_path_and_every_template_is_a_valid_registered_lcexec_template():
    module, name = 'tools.coopmwm.fullpol_programs:propose'.split(':')
    hook = getattr(import_module(module), name)
    assert hook.__name__ == 'propose' and hook.__module__.endswith('fullpol_programs')
    found = draws()
    assert found
    for own, t in found:
        assert isinstance(t, lc.FreeProgram) and t.is_template and t.members[0] == '@0'
        assert lc.FreeProgram.from_dict(t.to_dict()) == t
        assert lc.check_registered(t) == []
        assert t.join_rules[0] is None and all(abs(x) <= 100. for r in t.join_rules[1:] for x in r['ds'])
        for p in t.programs:
            refs = set(p.lon.refs) | {p.anchor, p.partner} - {None}
            assert refs <= set(t.members)                    # slot references only, never vehicle ids
            if p.lane is not None:
                assert p.lane.target in VIEW.lane_ids


def test_coverage_of_classes_families_primitives_and_lane_types():
    found = draws()
    fams = Counter(t.source.get('family') for _, t in found)
    assert set(fams) >= {'abreast_wall', 'closing_pair', 'solo_hold', 'wall_fill', 'gap_open_merge',
                         'lane_redistribute', 'random'}
    procs = {t.source.get('procedure') for _, t in found if t.source.get('family') == 'random'}
    assert procs == {'lcm_R_v1', 'fullpol_prim_v1'}
    progs = [p for _, t in found for p in t.programs]
    assert {p.lon.kind for p in progs} == set(fp.LON_KINDS)
    assert {p.tau for p in progs} >= {None, .6, 2.0}
    assert {p.start_el for p in progs} >= {0, 20, 40, 60}
    assert {len(t.members) for _, t in found} >= {1, 2, 3, 4}
    lanes = set()
    for own, t in found:
        rules = [None]+list(t.join_rules[1:])
        for k, p in enumerate(t.programs):
            if p.lane is None:
                continue
            src = own['lane'] if k == 0 else rules[k]['lane']
            lanes.add('into_closing' if p.lane.target == 2 else ('out_of_closing' if src == 2 else
                      ('left' if p.lane.target < src else 'right')))
    assert lanes == {'left', 'right', 'into_closing', 'out_of_closing'}


def test_none_guard_and_observables_only():
    rng = lambda: np.random.default_rng(5)
    far = row('ego', -3000., 1)                                  # outside every registered zone
    out = ft.propose(dict(frame=VIEW, rows=scene_rows(far), fullpol=dict(classes=('structured', 'random_r'))),
                     far, rng())
    assert out == []                                             # infeasible draws contribute nothing
    own = row('ego', -900., 1)
    a = ft.propose(dict(frame=VIEW, rows=scene_rows(own)), own, rng())
    hidden = [dict(r, driver_type='x', demand=3200, penetration=.6) for r in scene_rows(own)]
    b = ft.propose(dict(frame=VIEW, rows=hidden), dict(own, regime='D3200_p60'), rng())
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_verified_templates_are_used_as_registered_and_executor_runs_them():
    t = lc.FreeProgram(('@0', '@1'), (
        lc.MemberProgram(lc.Lon('speed', v_w=15., amax=1., dmax=1.5), lc.LaneReq(2, giveup_s=-300.), hold_s=90,
                         release_s=0.),
        lc.MemberProgram(lc.Lon('track', refs=('@0',), off=-25., dmax=1.5, amax=1.), hold_s=90, release_s=0.,
                         partner='@0', partner_lane=2, releases=('hold_end', 'release_s', 'partner_left'))),
        duration_s=90, source=dict(family='gap_open_merge', verification_id='v1'),
        join_rules=(None, dict(lane=1, ds=(-60., 0.))))
    own = row('ego', -900., 1)
    got = ft.propose(dict(frame=VIEW, rows=[], fullpol=dict(classes=('verified',), templates=[t.to_dict()])),
                     own, np.random.default_rng(0))
    assert got == [t]
    # resolve slots to concrete CAVs and run LCEXEC's executor on own observations
    prog = t.programs[0]
    ex = lc.MemberExecutor('ego', prog)
    obs = lc.Obs(0, lc.View('ego', -900., 1, 20.))
    cmds = ex.step(obs)
    assert ('changeLane', (2, 2.0)) in [(k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in cmds]
    assert any(k == 'accel' for k, _ in cmds)


def test_template_descriptor_width_zero_for_others_and_flags_own_slot():
    own = row('ego', -900., 1)
    rng = np.random.default_rng(2)
    t = ft.random_template('random_r', VIEW, own, rng, 100.)
    while t is None or len(t.members) < 2:
        t = ft.random_template('random_r', VIEW, own, rng, 100.)
    init = dict(family='free_program_initiate', template=t.to_dict(), cap=6, member_count=1, gap_m=0., range_m=100.)
    join = dict(init, family='free_program_join', slot=1, member_count=2, gap_m=40.)
    d0, d1 = fh.template_descriptor(init, VIEW), fh.template_descriptor(join, VIEW)
    n = len(fh.TEMPLATE_SLOT_FEATURES)
    own_col = fh.TEMPLATE_SLOT_FEATURES.index('is_own_slot')
    assert len(d0) == len(d1) == fh.template_width()
    assert d0[own_col] == 1. and d0[n+own_col] == 0. and d1[own_col] == 0. and d1[n+own_col] == 1.
    assert fh.template_descriptor(None, VIEW) == [0.]*fh.template_width()
    assert fh.template_descriptor(dict(family='abreast_wall'), VIEW) == [0.]*fh.template_width()
    assert ft.labels(t) == ('random',) and 'D3200' not in repr(d0)


@pytest.mark.parametrize('cls', ['random_r', 'random_prim'])
def test_random_template_slot_lanes_make_lane_targets_adjacent(cls):
    rng = np.random.default_rng(9)
    own = row('ego', -900., 1)
    for _ in range(300):
        t = ft.random_template(cls, VIEW, own, rng, 100.)
        if t is None:
            continue
        lanes = [own['lane']]+[r['lane'] for r in t.join_rules[1:]]
        for lane, p in zip(lanes, t.programs):
            if p.lane is not None:
                assert abs(p.lane.target-lane) == 1


def test_program_extension_zero_init_families_and_off_identity(cfg):
    import torch
    from .test_lane_drop_cpu import ld_cfg, make
    base_cfg = ld_cfg(cfg, v2v=False)
    off, on = make(base_cfg), make(dict(base_cfg, lc_families=dict(enabled=True)))
    assert off.trigger.option_families[-2:] != fh.PROGRAM_FAMILIES
    assert on.trigger.option_families[-2:] == fh.PROGRAM_FAMILIES
    assert on.trigger.descriptor_width == off.trigger.descriptor_width+fh.template_width()
    w_on, w_off = on.trigger.trigger_head[0].weight, off.trigger.trigger_head[0].weight
    hidden = on.trigger.base_policy.actor[-1].in_features
    start = hidden+off.trigger.descriptor_width
    block = w_on[:, start:start+fh.template_width()]
    assert torch.equal(block, torch.zeros_like(block))
    assert torch.equal(torch.cat((w_on[:, :start], w_on[:, start+fh.template_width():]), 1), w_off)
    own = row('ego', -900., 0)
    t = ft.structured('solo_hold', VIEW, own, [], np.random.default_rng(0), 100.)
    option = dict(family='free_program_initiate', members=['ego'], template=t.to_dict(), cap=1, member_count=1,
                  gap_m=0., range_m=100.)
    desc = on.trigger.extension.descriptor(option)
    assert len(desc) == on.trigger.extension.descriptor_width and any(desc[-fh.template_width():])
    assert ft.labels(t) == ('wall_g_solo2', 'solo_hold') or ft.labels(t) == ('wall_h_solo2', 'solo_hold')
