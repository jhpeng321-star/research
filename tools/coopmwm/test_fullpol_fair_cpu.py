"""clean5b-F CPU contracts: fairness-filtered EV-BC, gate0 / duo12 slots, wall_h ban, fairness reward.

Real EpisodePolicy / feasible slates / ArmRuntime.train paths; synthetic MULTICRIT-schema analyses and N6
vehicle-log reports (plus the real MULTICRIT files when present). No SUMO.
"""
# ruff: noqa: F811, E731 (imported pytest fixtures are test parameters; small test lambdas)
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from . import fullpol_ev as fe, fullpol_fair as ff, fullpol_learn as fl, lane_drop as ld, track2_backend as tb
from .test_fullpol_ev_cpu import advance, canon, ev_cfg, make, scene, tick
from .test_lane_drop_cpu import row
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)

PROV = 'clean5b-F CPU fixture (owner REGISTRATION 2026-11-07 04:0x)'
MULTICRIT = Path('E:/research-dense/runs/active/C4-V7FT-20261017/multicrit')


def metric(diff, lo=None, hi=None):
    return dict(diff=diff, lo=diff-1. if lo is None else lo, hi=diff+1. if hi is None else hi)


def analysis(arms, cell='D3200_p60'):
    """arms: {name: (J diff, J hi, {group: increase_s})} -> MULTICRIT-schema analysis."""
    out = {}
    for name, (dj, hi, inc) in arms.items():
        m = {g: metric(inc.get(g, -1.)) for g in ff.GROUP_KEYS}
        m['J'] = metric(dj, hi=hi)
        out[name] = dict(n=24, metrics=m)
    return dict(cells={cell: dict(u0={g: 50. for g in ff.GROUP_KEYS}, arms=out)})


def fair_cfg(cfg, rows=None):
    base = ev_cfg(cfg)
    rows = rows if rows is not None else ff.combine(ff.multicrit_rows(analysis({
        'wall_g_solo2': (-9000., -6000., {'L2_mean': 6.}),
        'wall_g_solo2_gate0': (-8000., -5000., {'L2_mean': 3.}),
        'wall_g_duo12': (-7000., -4000., {'L2_mean': 9.}),
        'wall_h_solo2': (-15000., -12000., {'L2_mean': 19.})})))
    g = fl.guidance_doc(rows, sources=[dict(name='fixture', sha256='0'*64, baseline='mappo')], provenance=PROV)
    return dict(base, **fe.with_fairness({fe.KEY: base[fe.KEY]}, regime_guidance=g))


def test_multicrit_rows_flag_group_fairness_ban_and_combine_conservatively(cfg):
    rows = ff.multicrit_rows(analysis({'a': (-500., -100., {'L0_mean': 10.}), 'b': (-500., -100., {'cav_mean': 10.5}),
                                       'wall_h_solo2': (-900., -800., {})}))
    by = {r['family']: r for r in rows}
    assert by['a']['fair'] and not by['b']['fair'] and by['wall_h_solo2']['banned'] and not by['wall_h_solo2']['fair']
    assert by['a']['lb95_veh_s'] == 100. and by['a']['mean_veh_s'] == 500.          # gain = -(arm - u0)
    other = ff.multicrit_rows(analysis({'a': (-400., -50., {'L1_mean': 12.})}))
    merged = {r['family']: r for r in ff.combine(rows+other)}
    assert not merged['a']['fair'] and merged['a']['lb95_veh_s'] == 50.               # fair in EVERY source


def test_real_multicrit_files_give_the_registered_eligibility(cfg):
    paths = [MULTICRIT/'confirm'/'MULTICRIT_confirm.json', MULTICRIT/'MULTICRIT_taskb.json']
    if not all(p.exists() for p in paths):
        pytest.skip('MULTICRIT outputs not present')
    doc = ff.build_guidance(paths, provenance=PROV)
    by = {(r['regime'], r['family']): r for r in doc['rows']}
    assert not by[('D3200_p60', 'wall_h_solo2')]['fair']                                # banned + >10 s
    assert by[('D3200_p60', 'wall_g_solo2')]['fair'] and by[('D3200_p60', 'wall_g_solo2')]['lb95_veh_s'] > 0
    on = dict(ev_cfg(cfg), **fe.with_fairness({fe.KEY: ev_cfg(cfg)[fe.KEY]}, regime_guidance=doc))
    table, receipt = fl.ev_table(on, 'B')
    assert 'wall_h_solo2' in receipt['dropped_unfair']
    assert fl.qualifying(table.get('D2400_p60', {})) == []
    assert fl.qualifying(table['D3200_p60'])[0] == 'wall_g_solo2'


def test_off_identical_and_slate_bans_wall_h_offers_duo_and_gate0(cfg):
    a, b = make(ev_cfg(cfg)), make(dict(ev_cfg(cfg), clean5_ev=dict(ev_cfg(cfg)['clean5_ev'], fair=None)))
    assert canon(tick(a, scene(), 300, seed=3)) == canon(tick(b, scene(), 300, seed=3))
    p = make(fair_cfg(cfg))
    seen = []
    for t in range(300, 330):
        seen += [m for table in tick(p, advance(scene(), t-300), t, deterministic=False, seed=t)['track2_trigger']['tables']
                 for m in table if m]
        p.last_decision, p.executor.active = None, []
    solos = [m for m in seen if m['family'] == 'group_hold' and len(m['members']) == 1]
    assert solos and all(fe.variant_of(m) == 'wall_g' for m in solos)                 # firm 15 m/s solo never offered
    view = ld.MapView.from_descriptor(fair_cfg(cfg)['lane_drop']['map'])
    duos = [m for m in seen if fe.labels(m, view, fe.settings(fair_cfg(cfg)))[0] == 'wall_g_duo12']
    assert duos and all(sorted(m['lanes']) == [0, 1] and m['target_mps'] == 22. for m in duos)
    own = row('e', -1200, 0)
    assert fe._gate0(own, [own, row('x', -1250, 0, cav=False)]) is False             # vehicle 50 m behind
    assert fe._gate0(own, [own, row('x', -1350, 0, cav=False), row('y', -1250, 1)]) is True
    gated = fe._solo(view, own, np.random.default_rng(0), fe.settings(fair_cfg(cfg)), [own])
    assert gated.to_dict()['gate0'] and tb._option_from_dict(gated.to_dict()).to_dict()['gate0']
    assert fe.labels(gated.to_dict(), view)[:2] == ('wall_g_solo2_gate0', 'wall_g_solo2')


def test_ev_targets_only_fair_eligible_strategies(cfg):
    rows = ff.combine(ff.multicrit_rows(analysis({
        'wall_g_solo2': (-9000., 500., {'L2_mean': 6.}),          # lb95 < 0 -> not a target
        'wall_g_solo2_gate0': (-8000., -5000., {'L2_mean': 3.}),  # fair, lb95 > 0
        'wall_h_solo2': (-15000., -12000., {'L2_mean': 19.})})))  # unfair / banned
    on = fair_cfg(cfg, rows)
    table, _ = fl.ev_table(on, 'B')
    view = ld.MapView.from_descriptor(on['lane_drop']['map'])
    solo = lambda gate, ln=0: dict(fe._solo(view, row('e', -1200, ln), np.random.default_rng(0), fe.settings(on),
                                            [row('e', -1200, ln)] + ([] if gate else [row('x', -1250, ln)])).to_dict())
    rows_ = table['D3200_p60']

    def target(entry):
        trig = dict(context=dict(members=['e']), tables=[[None, entry]], mask=[[True, True]])
        return fl.ev_targets(on, trig, rows_)[0][0]
    assert target(solo(True)) == 1 and target(solo(False)) == 0 and target(solo(True, ln=2)) == 0


def test_excess_harm_distribution_and_fail_closed(cfg, tmp_path):
    b = {'origin:L0': dict(n=10, mean_s=40.), 'class:hdv': dict(n=5, mean_s=30.), 'origin:L2': dict(n=4, mean_s=10.)}
    a = {'origin:L0': dict(n=10, mean_s=25.), 'class:hdv': dict(n=5, mean_s=25.), 'origin:L2': dict(n=4, mean_s=20.)}
    harm, parts = ff.excess_harm(b, a)
    assert harm == pytest.approx(10*5.) and parts['class:hdv'] == 0. and parts['origin:L2'] == 0.
    targets = {('e1', 300): [-1., -1.], ('e1', 301): [-1., -1.], ('e2', 300): [-2.]}
    out = ff.apply(targets, {'e1': 8., 'e2': 0.})
    assert sum(sum(v) for k, v in out.items() if k[0] == 'e1') == pytest.approx(-4.-8.)  # one-for-one in veh-s
    assert out[('e2', 300)] == [-2.]
    report = dict(cav=3, hdv=2, delay_by_class=dict(cav_mean_s=5., hdv_mean_s=7.), origins={'L0': 2, 'L2': 3},
                  delay_by_origin_s={'L0': 9., 'L2': 1.})
    assert ff.summary_from_report(report) == {'class:cav': dict(n=3, mean_s=5.), 'class:hdv': dict(n=2, mean_s=7.),
                                              'origin:L0': dict(n=2, mean_s=9.), 'origin:L2': dict(n=3, mean_s=1.)}
    eps = [dict(id='e1', world=dict(seed=989220), vehicle_groups=b), dict(id='p', practice=True, world=dict(seed=1))]
    harm, receipt = ff.penalties(fair_cfg(cfg), 'B', eps, {'989220': a})
    assert harm == {'e1': pytest.approx(50.)} and receipt['skipped'][0]['episode_id'] == 'p'
    with pytest.raises(ValueError, match='fail closed'):
        ff.penalties(fair_cfg(cfg), 'B', eps, {})
    with pytest.raises(ValueError, match='fail closed'):
        ff.reference(dict(fair_cfg(cfg), run_root=str(tmp_path)), 1)


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_train_adds_the_fairness_cost_for_b_and_r_identically(cfg, monkeypatch, tmp_path, arm):
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    from .test_track2_backend_cpu import fake_load
    fake_load(monkeypatch)
    out = {}
    update = tb.ppo_update
    for mode in ('off', 'on'):
        on = fair_cfg(dict(cfg, run_root=str(tmp_path)))
        on['clean5_ev'] = dict(on['clean5_ev'], fair=dict(on['clean5_ev']['fair'], reward=mode == 'on'))
        runtime = tb.build_arm_runtime(on, arm)['runtime']
        runtime.safety_checked = True
        p = runtime.policy
        c = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
        vehicles = [dict(r, v_mps=2.) for r in scene()]
        p.observe(vehicles, 300, native_commands={r['vid']: 0. for r in vehicles if r['is_cav']})
        c.apply(vehicles, {r['vid']: 0. for r in vehicles if r['is_cav']}, 300)
        s = c.events[-1]['sample']
        reg = on['track2_reward']
        rows = []
        for vid in s['context']['members']:
            nb = {v: -1. for v in s['track2']['reward_neighbourhoods'][vid]}
            rows.append(dict(episode_id='ep1', time_s=300, cav_id=vid, source='sumo', own_reward=-1., neighbour_rewards=nb,
                             reward=-1.+reg['weight']*(sum(nb.values())/len(nb) if nb else 0.), weight=reg['weight'],
                             weight_provenance=reg['provenance']))
        (tmp_path/'fair').mkdir(exist_ok=True)
        ff.reference_path(on, 1).write_text(json.dumps({'989220': {'origin:L0': dict(n=10, mean_s=20.)}}))
        captured = []

        def capture(model, optimizer, batch, advantages, returns, **kw):
            captured.append(list(returns))
            return update(model, optimizer, batch, advantages, returns, **kw)
        monkeypatch.setattr(tb, 'ppo_update', capture)
        monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
        episode = dict(id='ep1', arm=arm, steps=[dict(time_s=300, sample=s)], world=dict(seed=989220),
                       vehicle_groups={'origin:L0': dict(n=10, mean_s=40.)})
        rep = runtime.train([episode], rewards=dict(schema=tb.REWARD_SCHEMA, arm=arm, update=1, rows=rows))
        out[mode] = (captured[0], rep)
    assert 'fullpol_fair' not in out['off'][1] and out['on'][1]['fullpol_fair']['total_harm_veh_s'] == 100.
    n = len(out['on'][0])
    delta = [a-b for a, b in zip(out['on'][0], out['off'][0])]
    assert sum(delta) == pytest.approx(-100./tb.UNIT) and all(d == pytest.approx(-100./tb.UNIT/n) for d in delta)
