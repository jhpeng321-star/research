"""clean5a CPU contracts: speed-only EV mode on N2 families (no SUMO, no LCEXEC dependency).

Real EpisodePolicy / TriggerPolicy / feasible_proposals / N2 executor / PPO / BC code paths;
fakes: the N1 map stub of test_lane_drop_cpu (drop at s = 0, WM lanes 0..2, closing lane 2) and
synthetic observation rows / S-REGIME-format episode rows.
"""
# ruff: noqa: F811, E731 (imported pytest fixtures are test parameters; small test lambdas)
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from . import fullpol_ev as fe, fullpol_learn as fl, lane_drop as ld, track2_backend as tb
from .test_lane_drop_cpu import advance, canon, ld_cfg, make, row
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)
from .trigger_policy import ppo_update
from tools.coopnative.policy import encode

PROV = 'clean5a CPU fixture (MAIN 2026-11-06 ~19:3x)'
PARITY = ['wall_g', 'wall_h']+[f'wall_{v}_solo{k}' for v in 'gh' for k in (0, 1, 2)]


def ev_cfg(cfg, *, learn=None, guidance=None, parity=PARITY, families=('abreast_wall', 'group_hold'),
           sizes=(1, 2), v2v=False):
    out = ld_cfg(cfg, v2v=v2v, slots=4)
    out.update(fe.config_keys(provenance=PROV, families=families, group_sizes=sizes, learn=learn,
                              regime_guidance=guidance, parity_families=parity))
    return out


def scene():
    """Zone CAVs: a lane 0 (left), b lane 1, c closing lane 2 (wall geometry), d behind; HDVs."""
    return [row('a', -1200, 0), row('b', -1190, 1), row('c', -1185, 2), row('d', -1250, 0),
            row('h1', -1150, 1, cav=False), row('h2', -1210, 2, cav=False)]


def tick(p, vehicles, now, deterministic=True, seed=1, native=.3):
    p.observe(vehicles, now, native_commands={r['vid']: native for r in vehicles if r['is_cav']})
    context = encode(vehicles)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    sample, _ = p.sample(context, torch.Generator().manual_seed(seed), deterministic)
    return sample


def entries(sample):
    trig = sample['track2_trigger']
    return [m for table in trig['tables'] for m in table if m]


def guidance(rows):
    return fl.guidance_doc(rows, sources=[dict(name='S-REGIME fixture', sha256='0'*64, baseline='native')],
                           provenance=PROV)


def grow(family, lb, mean=None, baseline='native', regime='D3200_p60', arm='B'):
    return dict(arm=arm, regime=regime, family=family, baseline=baseline, n=24,
                mean_veh_s=float(mean if mean is not None else lb+500.), lb95_veh_s=float(lb))


# ------------------------------------------------------------------ settings / OFF

def test_settings_off_validation_and_requires_n2_moves(cfg):
    assert fe.settings(cfg) is None and fe.settings(dict(cfg, clean5_ev=False)) is None
    on = ev_cfg(cfg)
    assert fe.settings(on)['families'] == ('abreast_wall', 'group_hold')
    with pytest.raises(ValueError):
        fe.settings(dict(on, clean5_ev=dict(on['clean5_ev'], group_sizes=[3])))
    with pytest.raises(ValueError):
        fe.settings(dict(cfg, **fe.config_keys(provenance=PROV)))       # no lane_drop moves
    p = make(ld_cfg(cfg, v2v=False))
    p.observe(scene(), 300, native_commands={})
    assert 'clean5_ev' not in p.local_states['a']


def test_off_identical_with_clean5_ev_absent_or_false(cfg):
    outs = []
    for value in (ld_cfg(cfg, v2v=False), dict(ld_cfg(cfg, v2v=False), clean5_ev=False)):
        p = make(value)
        samples = [tick(p, advance(scene(), t-300), t, deterministic=False, seed=t) for t in range(300, 303)]
        outs.append(canon(samples)+canon({k: v.tolist() for k, v in p.worker_state()['trigger'].items()}))
    assert outs[0] == outs[1]


# ------------------------------------------------------------------ slate: N2 only, solo + pair

def test_slate_is_n2_only_with_solo_and_pair_group_holds(cfg):
    p = make(ev_cfg(cfg))
    seen = []
    for t in range(300, 340):
        seen += entries(tick(p, advance(scene(), t-300), t, deterministic=False, seed=t))
        p.last_decision = None          # every tick is a periodic decision (test only)
        p.executor.active = []
    fams = {m['family'] for m in seen}
    assert fams <= {'abreast_wall', 'group_hold'}, fams
    sizes = {len(m['members']) for m in seen if m['family'] == 'group_hold'}
    assert sizes == {1, 2}
    solos = [m for m in seen if m['family'] == 'group_hold' and len(m['members']) == 1]
    assert {fe.variant_of(m) for m in solos} == {'wall_g', 'wall_h'}
    assert all(row_['is_cav'] for row_ in scene() if row_['vid'] in {v for m in seen for v in m['members']})
    walls = [m for m in seen if m['family'] == 'abreast_wall']
    assert walls and {fe.variant_of(m) for m in walls} <= {'wall_g', 'wall_h'}


def test_solo_hold_roundtrip_labels_and_trigger_descriptor(cfg):
    view = ld.MapView.from_descriptor(ld_cfg(cfg)['lane_drop']['map'])
    solo = fe.SoloHold('group_hold', ('a',), 0, 90, 1.5, 15., 0., False, (0,))
    data = solo.to_dict()
    assert tb._option_from_dict(data) == solo and isinstance(tb._option_from_dict(data), fe.SoloHold)
    assert fe.labels(data, view)[0] == 'wall_h_solo2'          # WM lane 0 = SUMO lane 2 (left)
    assert fe.labels(dict(data, lanes=[2]), view)[0] == 'wall_h_solo0'
    assert fe.labels(dict(data, hold=120, target_mps=22., depth=1.), view)[0] == 'wall_g_solo2'
    pair = ld.LaneDropMove('group_hold', ('a', 'b'), 0, 90, 1.5, 15., 0., False, (0, 1)).to_dict()
    assert fe.labels(pair, view) == ('group_hold_pair', 'group_hold')
    wall = ld.LaneDropMove('abreast_wall', ('a', 'b', 'c'), 0, 120, 1., 22., 0., False, (0, 1, 2)).to_dict()
    assert fe.labels(wall, view) == ('wall_g', 'abreast_wall')
    with pytest.raises(ValueError):
        fe.SoloHold('group_hold', ('a', 'b'), 0, 90, 1.5, 15., 0., False, (0, 1))
    p = make(ev_cfg(cfg))
    desc = p.trigger._descriptor(solo, ['a', 'b'])
    assert len(desc) == p.trigger.descriptor_width


def test_selected_solo_hold_runs_the_n2_law_for_its_cav_only(cfg):
    p = make(ev_cfg(cfg))
    vehicles = scene()
    solo = None
    for t in range(300, 340):
        sample = tick(p, advance(vehicles, t-300), t, deterministic=False, seed=t)
        trig = sample.get('track2_trigger')
        if trig is None:
            continue
        for i, vid in enumerate(trig['context']['members']):
            m = trig['tables'][i][trig['actions'][i]]
            if m and m['family'] == 'group_hold' and len(m['members']) == 1 and trig['accepted'][vid]:
                solo = (t, vid)
                break
        if solo:
            break
        p.last_decision = None
    assert solo is not None
    t0, vid = solo
    laws = []
    for t in range(t0+1, t0+5):
        sample = tick(p, advance(vehicles, t-300), t)
        laws.append(sample['track2']['lane_drop']['law_commands'])
    assert all(set(c) == {vid} or vid in c for c in laws)
    assert not any(v.startswith('h') for c in laws for v in c)


# ------------------------------------------------------------------ EV guidance

def test_guidance_rows_reproduce_the_registered_analysis_sign_and_ci():
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(989700, 989712):
        base = 30000.+rng.normal(0, 800)
        rows.append(dict(status='ok', cell='D3200_p60', seed=seed, arm='base', J_total=base))
        rows.append(dict(status='ok', cell='D3200_p60', seed=seed, arm='wall_h_solo2', J_total=base-900.+rng.normal(0, 300)))
        rows.append(dict(status='ok', cell='D3200_p60', seed=seed, arm='wall_g', J_total=base+200.+rng.normal(0, 300)))
    g = {r['family']: r for r in fl.guidance_rows(rows)}
    # rg_analyze.boot_mean on d = strategy - base, then gain = -d
    for arm, row_ in g.items():
        d = np.array([r['J_total'] for r in rows if r['arm'] == arm])-np.array(
            [r['J_total'] for r in rows if r['arm'] == 'base'])
        idx = np.random.default_rng(20261106).integers(0, len(d), (10000, len(d)))
        m = d[idx].mean(1)
        assert row_['mean_veh_s'] == pytest.approx(-d.mean())
        assert row_['lb95_veh_s'] == pytest.approx(-float(np.quantile(m, .975)))
    assert g['wall_h_solo2']['lb95_veh_s'] > 0 > g['wall_g']['mean_veh_s']


def test_mappo_relative_rows_win_over_native_and_parity_filters(cfg):
    rows = [grow('wall_h_solo2', 300., baseline='native'), grow('wall_h_solo2', -50., baseline='mappo'),
            grow('wall_g', 400., baseline='native'), grow('zipper', 900.)]
    on = ev_cfg(cfg, guidance=guidance(rows), learn=dict(ev_bc=True))
    rows.append(grow('wall_h', 50., regime='D2400_p60'))
    on = ev_cfg(cfg, guidance=guidance(rows), learn=dict(ev_bc=True))
    table, receipt = fl.ev_table(on, 'B')
    # a cell with MAPPO-relative (S-REGIMEU0) rows uses ONLY those; native rows only for other cells
    assert set(table['D3200_p60']) == {'wall_h_solo2'}
    assert table['D3200_p60']['wall_h_solo2']['baseline'] == 'mappo'
    assert table['D2400_p60']['wall_h']['baseline'] == 'native'
    assert receipt['regime_baseline'] == {'D2400_p60': 'native', 'D3200_p60': 'mappo'}
    assert fl.qualifying(table['D3200_p60']) == [] and fl.qualifying(table['D2400_p60']) == ['wall_h']
    native_only = ev_cfg(cfg, guidance=guidance([r for r in rows if r['baseline'] == 'native']))
    t2, r2 = fl.ev_table(native_only, 'B')
    assert 'zipper' not in t2['D3200_p60'] and r2['dropped_without_parity'] == {'zipper': 1}
    assert fl.qualifying(t2['D3200_p60']) == ['wall_g', 'wall_h_solo2']


def _episode(sample, eid='e1', seed=989220):
    return dict(id=eid, arm='B', world=dict(seed=seed, demand=3200, penetration=.6),
                steps=[dict(time_s=300, sample=sample)])


def _decisions(p, n=6):
    out = []
    for t in range(300, 300+n):
        sample = tick(p, advance(scene(), t-300), t, deterministic=False, seed=t)
        p.last_decision = None
        p.executor.active = []
        if 'track2_trigger' in sample:
            out.append(sample)
    return out


def test_ev_bc_targets_best_offered_qualifying_family_else_none(cfg):
    on = ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', 300.), grow('wall_g', 100.)]), learn=dict(ev_bc=True))
    p = make(on)
    eps = [_episode(s, f'e{i}') for i, s in enumerate(_decisions(p))]
    records, receipt = fl.ev_bc_records(on, 'B', eps)
    assert records and receipt['rows'].get('act', 0) > 0
    view = ld.MapView.from_descriptor(on['lane_drop']['map'])
    for r in records:
        b = r['batch']
        for i, vid in enumerate(b.context['members']):
            if not float(r['row_weight'][i]):
                continue
            table = b.tables[i]
            labels = [fe.labels(m.to_dict(), view) if m is not None else () for m in table]
            offered = {f for ls in labels for f in ls}
            t = int(b.actions[i])
            if 'wall_h_solo2' in offered:
                assert labels[t][0] == 'wall_h_solo2'
                assert all(bool(r['equivalent'][i][j]) == (labels[j][:1] == ('wall_h_solo2',))
                           for j in range(len(table)))
            elif 'wall_g' in offered:
                assert 'wall_g' in labels[t]
            else:
                assert t == 0
    # nothing with lb95 > 0 -> every labelled row targets none
    neg = ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', -10.)]), learn=dict(ev_bc=True))
    records, _ = fl.ev_bc_records(neg, 'B', eps)
    assert records and all(int(r['batch'].actions[i]) == 0 for r in records for i in range(len(r['batch'].actions)))


def test_r_without_matched_sweep_gets_no_terms_and_b_r_code_paths_match(cfg):
    on = ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', 300.)]), learn=dict(ev_bc=True, ev_credit=True))
    p = make(on)
    eps = [_episode(s, f'e{i}') for i, s in enumerate(_decisions(p, 3))]
    for e in eps:
        e['arm'] = 'R'
    assert fl.ev_bc_records(on, 'R', eps)[0] == []
    assert fl.ev_credit_terms(on, 'R', eps, eligible={(e['id'], 300) for e in eps})[0] == {}
    # identical guidance given to R -> identical records (only the guidance source differs by arm)
    both = ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', 300.), grow('wall_h_solo2', 300., arm='R')]),
                  learn=dict(ev_bc=True))
    rb, _ = fl.ev_bc_records(both, 'B', eps)
    rr, _ = fl.ev_bc_records(both, 'R', eps)
    assert [r['batch'].actions.tolist() for r in rb] == [r['batch'].actions.tolist() for r in rr]


def test_no_regime_label_or_guidance_reaches_policy_inputs(cfg):
    outs = []
    for value in (ev_cfg(cfg), ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', 300.)]),
                                       learn=dict(ev_bc=True, ev_credit=True))):
        p = make(value)
        samples = []
        for t in range(300, 304):
            clean = advance(scene(), t-300)
            p.observe([dict(r, demand=3200, penetration=.6, regime='D3200_p60') for r in clean], t,
                      native_commands={r['vid']: .3 for r in clean if r['is_cav']})
            context = encode(clean)
            context['v2v'] = {v: [0.]*48 for v in context['members']}
            samples.append(p.sample(context, torch.Generator().manual_seed(t), True)[0])
        flat = canon(samples)+canon(p.local_states)
        assert 'D3200_p60' not in flat and 'penetration' not in flat and 'lb95' not in flat
        outs.append(canon([s.get('track2_trigger') for s in samples]))
    assert outs[0] == outs[1]


def test_supervise_ev_bc_kl_guard_and_report_only_receipts(cfg):
    on = ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', 300.), grow('wall_g', 100.)]),
                learn=dict(ev_bc=True, report_kl_unlabelled=True, report_valsplit_auc=True))
    on['imitation_epochs'] = 20
    p = make(on)
    samples = _decisions(p, 8)
    eps = [_episode(s, f'e{i}', seed=989220+i) for i, s in enumerate(samples)]
    eps.append(dict(_episode(samples[0], 'unl', seed=989299), world=dict(seed=989299, demand=2400, penetration=.6)))
    before = deepcopy(p.trigger.state_dict())
    out = fl.supervise(SimpleNamespace(cfg=on, arm='B', policy=p), episodes=eps)
    assert out['epochs'] == 20 and out['kl_budget'] == .045
    assert all(ps['kl'] <= .045 for ps in out['fit']['passes'] if ps['accepted'])
    assert out['fit']['final_ce'] < out['fit']['initial_ce']
    assert out['report_bc_kl']['accepted_steps'] >= 1
    assert out['report_kl_unlabelled']['contexts'] == 1          # the D2400 episode has no regime rows
    v = out['report_valsplit_auc']
    assert v['seeds'] >= 2 and 'shadow' in v['rule']
    assert any(not torch.equal(before[k], t) for k, t in p.trigger.state_dict().items())
    assert fl.supervise(SimpleNamespace(cfg=on, arm='A', policy=p), episodes=eps) is None
    off = ev_cfg(cfg)
    assert fl.supervise(SimpleNamespace(cfg=off, arm='B', policy=make(off)), episodes=eps)['records'] == 0


def test_ppo_on_recorded_solo_decision_and_ev_credit(cfg):
    on = ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', 300., mean=3000.)]), learn=dict(ev_credit=True))
    p = make(on)
    for sample in _decisions(p, 10):
        trig = sample['track2_trigger']
        picks = [i for i, (table, a) in enumerate(zip(trig['tables'], trig['actions']))
                 if table[a] and table[a]['family'] == 'group_hold' and len(table[a]['members']) == 1]
        if picks:
            break
    else:
        pytest.skip('no solo selected in this stream')
    terms, receipt = fl.ev_credit_terms(on, 'B', [_episode(sample)], eligible={('e1', 300)})
    assert receipt['enabled'] and all(abs(v) <= 1. for v in terms.values())
    batch = tb._batch(trig)
    adv = [terms.get(('e1', 300, v), 0.) for v in trig['context']['members']]
    metrics = ppo_update(p.trigger, p.trigger_optimizer, batch, adv, [0.]*len(adv), **on['trigger_ppo'])
    assert np.isfinite(metrics['loss'])


def test_g4ld_credits_pair_holds_and_skips_solo_holds(cfg):
    from . import sparse_synergy_credit as g4
    on = ev_cfg(cfg)
    p = make(on)
    view = ld.MapView.from_descriptor(on['lane_drop']['map'])
    solo = fe.SoloHold('group_hold', ('a',), 0, 90, 1.5, 15., 0., False, (0,))
    pair = ld.LaneDropMove('group_hold', ('a', 'b'), 0, 90, 1.5, 15., 0., False, (0, 1))
    sample = tick(p, scene(), 300, deterministic=False, seed=3)
    trig = deepcopy(sample['track2_trigger'])
    i = trig['context']['members'].index('a')
    trig['tables'][i] = [None, solo.to_dict(), pair.to_dict()]+[None]*(len(trig['tables'][i])-3)
    trig['mask'][i] = [True, True, True]+[False]*(len(trig['mask'][i])-3)
    ext = dict(enabled=True, rule=g4.EXTENSION_RULE, families=['abreast_wall', 'group_hold'], provenance=PROV)
    found = {}
    for action in (1, 2):
        trig['actions'][i] = action
        trig['accepted'] = {v: v == 'a' for v in trig['context']['members']}
        step = dict(time_s=300, sample=dict(sample, track2_trigger=deepcopy(trig)))
        found[action] = g4.opportunities([dict(id='e1', steps=[step])], ext)
    assert found[1] == []                                            # a solo hold is not a joint start
    assert [r['g4ld']['members'] for r in found[2]] == [['a', 'b']]
    assert fe.labels(found[2][0]['move'], view) == ('group_hold_pair', 'group_hold')


def test_guidance_builder_cli_and_recommended_recipe(cfg, tmp_path):
    rng = np.random.default_rng(1)
    lines = []
    for seed in range(989700, 989706):
        base = 30000.+rng.normal(0, 500)
        lines += [dict(status='ok', cell='D3200_p60', seed=seed, arm='base', J_total=base),
                  dict(status='ok', cell='D3200_p60', seed=seed, arm='wall_h_solo2', J_total=base-2000.)]
    path = tmp_path/'REGIME_RESULTS.jsonl'
    path.write_text('\n'.join(json.dumps(x) for x in lines))
    out = tmp_path/'G.json'
    fl.main(['guidance', '--native', str(path), '--provenance', PROV, '--out', str(out)])
    doc = json.loads(out.read_text())
    assert doc['rows'][0]['family'] == 'wall_h_solo2' and doc['rows'][0]['mean_veh_s'] == pytest.approx(2000.)
    with pytest.raises(SystemExit):
        fl.main(['guidance', '--native', str(path), '--provenance', PROV, '--out', str(out)])
    on = dict(ld_cfg(cfg, v2v=False), **fe.recommended(provenance=PROV, regime_guidance=doc))
    assert fe.settings(on) and fl.learn_settings(on)['ev_bc'] and not fl.learn_settings(on)['ev_credit']
    table, _ = fl.ev_table(on, 'B')
    assert fl.best_family(table['D3200_p60']) == 'wall_h_solo2'


def test_u0_results_base_arm_and_analysis_summary_inputs(tmp_path):
    rows = []
    for seed in range(989850, 989856):
        rows += [dict(status='ok', cell='D3200_p60', seed=seed, arm='u0', J_total=50000.),
                 dict(status='ok', cell='D3200_p60', seed=seed, arm='wall_h_solo2', J_total=40000.+seed % 3)]
    path = tmp_path/'U0.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    doc = fl.build_guidance(mappo=str(path), provenance=PROV)
    assert [(r['family'], r['baseline']) for r in doc['rows']] == [('wall_h_solo2', 'mappo')]
    assert doc['rows'][0]['mean_veh_s'] == pytest.approx(10000.-1.)
    summary = dict(cells={'D2800_p40': {'wall_h': dict(n=16, mean_diff=-900., ci95=[-1500., -300.], harm_rate=.1)}})
    spath = tmp_path/'ANALYSIS.json'
    spath.write_text(json.dumps(summary))
    doc = fl.build_guidance(native=str(spath), provenance=PROV)
    assert doc['rows'][0]['lb95_veh_s'] == 300. and doc['rows'][0]['mean_veh_s'] == 900.


def test_solo_hold_ev_target_is_lane_specific(cfg):
    """MAIN 22:1x: a solo-hold entry carries the EV of the solo strategy of THAT member's current
    SUMO lane; walls / pairs use their own rows; missing or lb95 <= 0 -> none."""
    rows = {r['family']: r for r in [grow('wall_h_solo2', 5000., baseline='mappo'),
                                     grow('wall_h_solo0', -900., baseline='mappo'),
                                     grow('wall_h', 3000., baseline='mappo')]}
    on = ev_cfg(cfg)
    solo = lambda vid, wm: fe.SoloHold('group_hold', (vid,), 0, 90, 1.5, 15., 0., False, (wm,)).to_dict()
    wall = ld.LaneDropMove('abreast_wall', ('s0', 'x', 'y'), 0, 90, 1.5, 15., 0., False, (0, 1, 2)).to_dict()
    pair = ld.LaneDropMove('group_hold', ('s1', 'x'), 0, 90, 1.5, 15., 0., False, (1, 0)).to_dict()

    def target(vid, table):
        trig = dict(context=dict(members=[vid]), tables=[table], mask=[[m is not None or j == 0
                                                                        for j, m in enumerate(table)]])
        return fl.ev_targets(on, trig, rows)[0][0]
    assert target('s2', [None, solo('s2', 0)]) == 1               # SUMO lane 2 (left, WM 0): solo2 > 0
    assert target('s0', [None, solo('s0', 2)]) == 0               # SUMO lane 0 (closing, WM 2): solo0 n.s.
    assert target('s1', [None, solo('s1', 1)]) == 0               # SUMO lane 1: no solo1 row -> none
    assert target('s0', [None, solo('s0', 2), wall]) == 2         # walls use their own (positive) row
    assert target('s1', [None, solo('s1', 1), pair]) == 0         # pairs: no group_hold_pair row -> none
    view = ld.MapView.from_descriptor(on['lane_drop']['map'])
    assert fe.labels(solo('s0', 2), view)[0] == 'wall_h_solo0' and fe.labels(solo('s2', 0), view)[0] == 'wall_h_solo2'
    # a solo's ONLY label is its lane-specific strategy: a generic solo/group row can never match it
    for wm in (0, 1, 2):
        assert fe.labels(solo('s', wm), view) == (f'wall_h_solo{2-wm}',)
    generic = dict(rows, group_hold=grow('group_hold', 9000.), solo_hold=grow('solo_hold', 9000.))
    trig = dict(context=dict(members=['s0']), tables=[[None, solo('s0', 2)]], mask=[[True, True]])
    assert fl.ev_targets(on, trig, generic)[0][0] == 0


U0_ANALYSIS = Path('E:/research-dense/runs/active/C4-V7FT-20261017/regimeu0/ANALYSIS_p60_p40.json')


def test_real_u0_analysis_excludes_base_and_derived_entries(cfg):
    if not U0_ANALYSIS.exists():
        pytest.skip('S-REGIMEU0 ANALYSIS not present in this checkout')
    doc = fl.build_guidance(mappo=str(U0_ANALYSIS), provenance=PROV)
    fams = {(r['regime'], r['family']) for r in doc['rows']}
    assert {f for _, f in fams} == {'wall_h', 'wall_h_solo0', 'wall_h_solo2'}
    assert not any(f in ('u0', 'group_minus_best_solo') for _, f in fams)
    raw = json.loads(U0_ANALYSIS.read_text())
    for r in doc['rows']:
        v = raw['cells'][r['regime']][r['family']]
        assert r['mean_veh_s'] == pytest.approx(-v['mean_diff']) and r['lb95_veh_s'] == pytest.approx(-v['ci95'][1])
        assert r['baseline'] == 'mappo' and r['n'] == v['n']
    on = ev_cfg(cfg, guidance=doc, learn=dict(ev_bc=True))
    table, _ = fl.ev_table(on, 'B')
    for cell in ('D3200_p60', 'D3200_p40'):
        assert fl.qualifying(table[cell])[0] == 'wall_h_solo2' and 'wall_h_solo0' not in fl.qualifying(table[cell])
    with pytest.raises(ValueError, match='unrecognised'):
        fl.guidance_rows_from_analysis(dict(cells={'D1_p1': {'odd': dict(point=1.)}}), baseline='mappo')


def test_production_path_clean5a_episode_ppo_ev_bc_metrics_and_g4(cfg, monkeypatch):
    """Real worker (episode_components + V2V controller), real ArmRuntime.train PPO on the recorded
    N2-only slates incl. solo holds, EV-BC supervise, P3 update metrics and G4LD opportunities."""
    from . import launch_production as launch, sparse_synergy_credit as g4
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    from .test_track2_backend_cpu import fake_load
    fake_load(monkeypatch)
    on = ev_cfg(cfg, guidance=guidance([grow('wall_h_solo2', 5000.), grow('wall_h', 3000.)]),
                learn=dict(ev_bc=True, report_kl_unlabelled=True))
    on['decision_interval_s'] = 10
    runtime = tb.build_arm_runtime(on, 'B')['runtime']
    runtime.safety_checked = True
    p = runtime.policy
    job = dict(track2_backend=True, track2_config=on, arm='B', track2_state=p.worker_state(), noise_seed=7)
    worker, factory = tb.episode_components(job, p.layer.base, controller_class(Controller))
    controller = factory(worker, torch.Generator().manual_seed(9), False)
    slow = [dict(r, v_mps=2.) for r in scene()]
    steps, reward_rows = [], []
    reg = on['track2_reward']
    for t in range(300, 361):
        vehicles = advance(slow, t-300, speed=2.)
        controller.apply(vehicles, {r['vid']: 0. for r in vehicles if r['is_cav']}, t)
        s = controller.events[-1]['sample']
        steps.append(dict(time_s=t, sample=s, reward=-1.))
        outcomes = {r['vid']: -1. for r in vehicles}
        for vid in s['context']['members']:
            nb = {v: outcomes[v] for v in s['track2']['reward_neighbourhoods'][vid]}
            reward_rows.append(dict(episode_id='ep1', time_s=t, cav_id=vid, source='sumo', own_reward=-1.,
                                    neighbour_rewards=nb, reward=-1.+reg['weight']*(sum(nb.values())/len(nb) if nb else 0.),
                                    weight=reg['weight'], weight_provenance=reg['provenance']))
    picked = [m for st in steps if 'track2_trigger' in st['sample']
              for i, m in enumerate(st['sample']['track2_trigger']['tables'])
              for m in [m[st['sample']['track2_trigger']['actions'][i]]] if m]
    assert picked and {m['family'] for m in picked} <= {'abreast_wall', 'group_hold'}
    episode = dict(id='ep1', arm='B', steps=steps, actual_sim_s=361,
                   world=dict(seed=989220, demand=3200, penetration=.6, cell_key='D3200_p60'))
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    report = runtime.train([episode], rewards=dict(schema=tb.REWARD_SCHEMA, arm='B', update=1, rows=reward_rows))
    assert report['trigger'] and all(np.isfinite(m['loss']) for m in report['trigger'])
    out = fl.supervise(runtime, [episode])
    assert out['ev_bc']['records'] > 0 and out['fit']['initial_ce'] is not None
    metrics = launch.update_metrics('B', [dict(episode, totals=dict(collisions=0, teleports=0, hard_brake=0),
                                               ttc_events=0)],
                                    dict(report, bank_move_ce=None), dict(receipt={}), augmented_examples=0)
    assert set(metrics['trigger_counts']) >= {'none'} and metrics['decisions'] > 0
    ext = dict(enabled=True, rule=g4.EXTENSION_RULE, families=['abreast_wall', 'group_hold'], provenance=PROV)
    rows_g4 = g4.opportunities([dict(id='ep1', steps=steps)], ext)
    assert all(len(r['g4ld']['members']) >= 2 for r in rows_g4)


U0_FINAL = Path('E:/research-dense/runs/active/C4-V7FT-20261017/regimeu0')


@pytest.mark.parametrize('name', ['ANALYSIS.json', 'REGIMEU0_RESULTS.jsonl'])
def test_final_u0_inputs_agree_and_light_cells_target_none(cfg, name):
    path = U0_FINAL/name
    if not path.exists():
        pytest.skip('final S-REGIMEU0 files not present in this checkout')
    doc = fl.build_guidance(mappo=str(path), provenance=PROV)
    rows = {(r['regime'], r['family']): r for r in doc['rows']}
    assert not any(f in ('u0', 'group_minus_best_solo') for _, f in rows)
    if name.endswith('.jsonl'):
        assert {r['j_field'] for r in doc['rows']} == {'J_300_1200_vs'}
        raw = json.loads((U0_FINAL/'ANALYSIS.json').read_text()) if (U0_FINAL/'ANALYSIS.json').exists() else None
        if raw is not None:
            for (cell, fam), r in rows.items():
                assert r['mean_veh_s'] == pytest.approx(-raw['cells'][cell][fam]['mean_diff'], rel=1e-6, abs=1e-3)
    on = ev_cfg(cfg, guidance=doc, learn=dict(ev_bc=True))
    table, _ = fl.ev_table(on, 'B')
    assert fl.qualifying(table['D3200_p60'])[0] == 'wall_h_solo2'
    for cell in ('D2400_p60', 'D2400_p40'):
        assert fl.qualifying(table.get(cell, {})) == []       # light traffic: no metering -> none


def test_recipe_target_table_solo_only_lane2_meters_heavy_cells(cfg):
    """MAIN ~20:4x: EV-BC targets = lane-specific solos only; walls / pairs stay offered."""
    rows = []
    for cell, solo2, wall in (('D3200_p60', 15169., 11684.), ('D3200_p40', 17739., 12855.), ('D2800_p60', 1836., -2749.),
                              ('D2800_p40', -3308., -6262.), ('D2400_p60', -5763., -10702.), ('D2400_p40', -7542., -10413.)):
        rows += [grow('wall_h_solo2', solo2, regime=cell, baseline='mappo'),
                 grow('wall_h', wall, regime=cell, baseline='mappo')]
        if cell in ('D3200_p60', 'D3200_p40'):
            rows.append(grow('wall_h_solo0', -2772., regime=cell, baseline='mappo'))
    on = dict(ld_cfg(cfg, v2v=False), **fe.recommended(provenance=PROV, regime_guidance=guidance(rows)))
    assert set(fe.settings(on)['families']) == {'abreast_wall', 'group_hold'}      # still offered
    table, receipt = fl.ev_table(on, 'B')
    assert receipt['dropped_without_parity'] == {'wall_h': 6}
    solo = lambda vid, wm: fe.SoloHold('group_hold', (vid,), 0, 90, 1.5, 15., 0., False, (wm,)).to_dict()
    wall = ld.LaneDropMove('abreast_wall', ('e', 'x', 'y'), 0, 90, 1.5, 15., 0., False, (0, 1, 2)).to_dict()
    pair = ld.LaneDropMove('group_hold', ('e', 'x'), 0, 90, 1.5, 15., 0., False, (1, 0)).to_dict()

    def target(cell, wm):
        table_row = [None, solo('e', wm), wall, pair]
        trig = dict(context=dict(members=['e']), tables=[table_row], mask=[[True]*4])
        return fl.ev_targets(on, trig, table.get(cell, {}))[0][0]
    for cell in ('D3200_p60', 'D3200_p40', 'D2800_p60'):
        assert target(cell, 0) == 1                       # left lane (SUMO 2): solo
        assert target(cell, 1) == 0 and target(cell, 2) == 0   # middle / closing: none, even with a wall offered
    for cell in ('D2800_p40', 'D2400_p60', 'D2400_p40'):
        assert all(target(cell, wm) == 0 for wm in (0, 1, 2))   # light traffic: none


def test_bc_kl_cap_is_a_mean_over_labelled_decision_rows():
    """The registered .045 cap is a mean over supervised CONTEXTS (one CAV decision row each);
    a multi-CAV EV step must not be capped by its row-SUM (that would freeze EV-BC)."""
    a = torch.zeros((3, 4))
    b = torch.tensor([[2., 0., 0., 0.]]*3)
    kl_row = float(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=a[0]),
                                                     torch.distributions.Categorical(logits=b[0])))
    assert fl._mean_kl([a], [b]) == pytest.approx(kl_row)
    assert fl._mean_kl([a], [b], [torch.tensor([1., 1., 0.])]) == pytest.approx(kl_row)
    assert fl._mean_kl([a, a[:1]], [b, b[:1]], [torch.tensor([0., 0., 0.]), None]) == pytest.approx(kl_row)


def test_g2_coordination_partner_never_reopens_legacy_families_under_clean5a(cfg):
    """N11 F10: a G2 pair ego gets the N2-only slate (pair offered as a group_hold pair)."""
    from . import feasible_proposals as fpp
    p = make(ev_cfg(cfg))
    p.observe(scene(), 300, native_commands={r['vid']: .3 for r in scene() if r['is_cav']})
    state = dict(p.local_states['a'], acquirable_cav_ids=['a', 'b'], coordination_partner='b')
    out = fpp.candidates(state, np.random.default_rng(0), 8)
    fams = {o.family for o in out}
    assert out and fams <= {'abreast_wall', 'group_hold'}, fams
    assert any(o.family == 'group_hold' and set(o.members) == {'a', 'b'} for o in out)
    from . import coordination_opportunities as co
    calls = []
    real = co.candidates
    try:
        co.candidates = lambda state, rng, n: calls.append(state['self']) or []
        fpp.candidates(state, np.random.default_rng(0), 8)
        assert calls == []                                             # clean5a: G2 legacy slate bypassed
        off = make(ld_cfg(cfg, v2v=False))
        off.observe(scene(), 300, native_commands={r['vid']: .3 for r in scene() if r['is_cav']})
        fpp.candidates(dict(off.local_states['a'], coordination_partner='b'), np.random.default_rng(0), 8)
        assert calls == ['a']                                          # OFF: G2 path unchanged
    finally:
        co.candidates = real


def _wide_reference(rows=150, seed=0):
    from .trigger_policy import TriggerBatch
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn((rows, 9), generator=gen)
    after = logits+.06*torch.randn((rows, 9), generator=gen)     # a small per-CAV policy move
    batch = TriggerBatch(dict(members=[f'c{i}' for i in range(rows)], x=[[0.]]*rows), (), torch.zeros((rows, 9, 1)),
                         torch.ones((rows, 9), dtype=torch.bool), torch.zeros(rows, dtype=torch.long),
                         torch.zeros(rows), torch.zeros(rows))
    return batch, logits, after


@pytest.mark.parametrize('mode', ['row_sum', 'row_mean'])
def test_trigger_ppo_kl_guard_row_sum_vs_registered_row_mean_on_150_rows(cfg, monkeypatch, mode):
    """MAIN ~21:1x: with ~150 CAV rows the inherited row-SUM guard exceeds 0.045 for a tiny per-CAV
    move (revert every step); the clean5a row-MEAN guard admits it. OFF = row sum (unchanged)."""
    from .test_track2_backend_cpu import fake_load
    fake_load(monkeypatch)
    value = ev_cfg(cfg) if mode == 'row_sum' else dict(ev_cfg(cfg), clean5_ev=dict(ev_cfg(cfg)['clean5_ev'],
                                                                                      ppo_kl='row_mean'))
    runtime = tb.build_arm_runtime(value, 'B')['runtime']
    batch, logits, after = _wide_reference()
    monkeypatch.setattr(runtime.policy.trigger, 'distribution', lambda c, d, m: (after, torch.zeros(len(after))))
    kl = runtime._trigger_kl([(batch, logits)])
    per_row = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=logits),
                                                torch.distributions.Categorical(logits=after))
    if mode == 'row_sum':
        assert kl == pytest.approx(float(per_row.sum())) and kl > .045       # would revert every step
    else:
        assert kl == pytest.approx(float(per_row.mean())) and kl < .045      # admitted
    plain = tb.build_arm_runtime(ld_cfg(cfg, v2v=False), 'B')['runtime']   # clean5a OFF entirely
    monkeypatch.setattr(plain.policy.trigger, 'distribution', lambda c, d, m: (after, torch.zeros(len(after))))
    assert plain._trigger_kl([(batch, logits)]) == pytest.approx(float(per_row.sum()))


def test_recipe_registers_ev_bc_steps_60_and_row_mean_ppo_kl(cfg):
    on = dict(ld_cfg(cfg, v2v=False), **fe.recommended(provenance=PROV, regime_guidance=guidance([grow('wall_h_solo2', 9.)])))
    s = fe.settings(on)
    assert s['ev_bc_steps'] == 60 and s['ppo_kl'] == 'row_mean'
    with pytest.raises(ValueError):
        fe.settings(dict(on, clean5_ev=dict(on['clean5_ev'], ppo_kl='other')))
    with pytest.raises(ValueError):
        fe.settings(dict(on, clean5_ev=dict(on['clean5_ev'], ev_bc_steps=0)))
    p = make(on)
    eps = [_episode(s_, f'e{i}', seed=989220+i) for i, s_ in enumerate(_decisions(make(on), 3))]
    out = fl.supervise(SimpleNamespace(cfg=dict(on, imitation_epochs=20), arm='B', policy=p), episodes=eps)
    assert out['epochs'] == 60 and len(out['fit']['passes']) <= 60


def test_production_train_reports_ppo_kl_only_when_clean5a_row_mean(cfg, monkeypatch):
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    from .test_track2_backend_cpu import fake_load
    fake_load(monkeypatch)
    reports = {}
    for mode in ('row_sum', 'row_mean'):
        on = ev_cfg(cfg)
        on['clean5_ev'] = dict(on['clean5_ev'], ppo_kl=mode)
        runtime = tb.build_arm_runtime(on, 'B')['runtime']
        runtime.safety_checked = True
        p = runtime.policy
        controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
        vehicles = [dict(r, v_mps=2.) for r in scene()]
        p.observe(vehicles, 300, native_commands={r['vid']: 0. for r in vehicles if r['is_cav']})
        controller.apply(vehicles, {r['vid']: 0. for r in vehicles if r['is_cav']}, 300)
        s = controller.events[-1]['sample']
        reg = on['track2_reward']
        rows = []
        for vid in s['context']['members']:
            nb = {v: -1. for v in s['track2']['reward_neighbourhoods'][vid]}
            rows.append(dict(episode_id='ep1', time_s=300, cav_id=vid, source='sumo', own_reward=-1., neighbour_rewards=nb,
                             reward=-1.+reg['weight']*(sum(nb.values())/len(nb) if nb else 0.), weight=reg['weight'],
                             weight_provenance=reg['provenance']))
        monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
        reports[mode] = runtime.train([dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s)])],
                                      rewards=dict(schema=tb.REWARD_SCHEMA, arm='B', update=1, rows=rows))
    assert 'fullpol_ppo_kl' not in reports['row_sum']
    r = reports['row_mean']['fullpol_ppo_kl']
    assert r['rule'] == 'row_mean' and r['kl_budget'] == .045 and r['accepted']+r['rejected'] >= 1


def _solo_run(value, ticks=40):
    """Tick a scene until a CAV's solo hold is accepted; return (policy, controller, events)."""
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    p = make(value)
    c = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    out = []
    for t in range(300, 300+ticks):
        vehicles = [dict(r, v_mps=2.) for r in advance(scene(), t-300, speed=2.)]
        p.observe(vehicles, t, native_commands={r['vid']: 0. for r in vehicles if r['is_cav']})
        c.apply(vehicles, {r['vid']: 0. for r in vehicles if r['is_cav']}, t)
        out.append((t, vehicles, c.events[-1]['sample'], dict(c.g5_lane_targets)))
    return p, c, out


def test_hold_lane_pin_off_is_identical_and_on_pins_active_hold_members_to_their_own_lane(cfg):
    base = ev_cfg(cfg)
    runs = {}
    for name, value in (('absent', base), ('false', dict(base, clean5_ev=dict(base['clean5_ev'], hold_lane_pin=False))),
                        ('on', dict(base, clean5_ev=dict(base['clean5_ev'], hold_lane_pin=True)))):
        runs[name] = _solo_run(value)
    canon_run = lambda r: canon([s for _, _, s, _ in r[2]])
    assert canon_run(runs['absent']) == canon_run(runs['false'])
    assert all('hold_lane_pins' not in s for _, _, s, _ in runs['absent'][2])
    pinned = 0
    for t, vehicles, s, g5 in runs['on'][2]:
        lanes = {r['vid']: r['lane'] for r in vehicles}
        law = (s['track2'].get('lane_drop') or {}).get('law_commands') or {}
        pins = s.get('hold_lane_pins') or {}
        assert set(pins) == set(law) - set(s.get('lane_drop_lane_requests') or {})   # exactly the active N2 members
        for vid, lane in pins.items():
            assert lane == lanes[vid]                                           # own CURRENT lane, no change
            assert g5.get(vid) == lane                                          # enters the single lane writer
            assert not vid.startswith('h')                                      # HDVs never pinned
            pinned += 1
    assert pinned > 0
    with pytest.raises(ValueError):
        fe.settings(dict(base, clean5_ev=dict(base['clean5_ev'], hold_lane_pin='yes')))
    assert fe.hold_pins({'a', 'b', 'z'}, {'a': 0, 'b': 2}, zip_members={'b'}) == {'a': 0}
