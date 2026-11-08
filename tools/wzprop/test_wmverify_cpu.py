"""S-WMVERIFY CPU tests (no SUMO, no GPU): spec validation, seed range, executor parity on the fake plant (solo2 core
== S-REGIMEU0 RuleHook; hold path == WM-C make_hook), LCEXEC policy executor on the fake plant, CAV-only guard,
S-REGIMEU0 analysis round trip on the frozen c0b578f8 ANALYSIS, analysis == regimeu0.analyze, end-to-end
plan -> worker (regimeu0.worker, fake episode) -> analyze -> FULLPOL reader, registration text, dry run."""
from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path

import pytest

from tools.coopmwm import imitreplay as ir
from tools.coopmwm import map_descriptor as md
from tools.coopmwm import regimeu0 as R
from tools.wzprop import strategy_hook as SH
from tools.wzprop import strategy_search as SS
from tools.wzprop import wmverify as V
from tools.wzprop import wmverify_fake as F
from tools.wzprop import wmverify_hook as VH

FIX = Path(__file__).with_name('fixtures') / 'wmverify' / 'regimeu0_ANALYSIS_c0b578f8.json.gz'
SECONDS = 700                       # fake episodes: 300 s warmup + 400 s control (plumbing only)


def solo2():
    """The BANNED firm hold (owner 2026-11-07): executable as a comparator, rejected as a candidate."""
    return dict(SS.SOLO2, gate=None)


def inc():
    """The gentle incumbent wall_g_solo2 (spec v3)."""
    return dict(SS.INCUMBENT)


def wall3():
    """A one-CAV-per-lane gentle wall (N2 abreast_wall)."""
    return dict(SS.INCUMBENT, name='wall_g3', roles=[dict(SS.INCUMBENT['roles'][0], lane=ln) for ln in (0, 1, 2)])


def lane_spec(name='wmv_lc_closing', target=0, zone='far'):
    """A two-role lane program: lane-2 solo2 hold + a middle-lane CAV changing into the closing lane (SUMO 0)."""
    return dict(name=name, trigger='lanes', zone=zone, spread_m=300.0, cooldown_s=90, slots=1, gate=None,
                roles=[dict(lane=2, rank=0, v_w=22.0, dmax=1.0, hold_s=120, onset_s=0, lc_target=None),
                       dict(lane=1, rank=0, v_w=18.0, dmax=1.5, hold_s=90, onset_s=0, lc_target=target)])


@pytest.fixture
def wz(monkeypatch):
    from tools.coopmwm.lc_writer_harness import MAP_DIR
    monkeypatch.delenv(md.ENV, raising=False)
    md._reset_for_tests()
    desc = md.bind(md.load(MAP_DIR))
    yield desc
    md._reset_for_tests()


# ------------------------------------------------------------------------------------------ validation / seeds
def test_check_spec_registered_only():
    assert V.check_spec(dict(inc(), name='g_twin'))['name'] == 'g_twin'
    assert V.check_spec(lane_spec())['roles'][1]['lc_target'] == 0
    for name in ('wall_g_alt21_gate0', 'g2_gapopen12', 'wall_g_relay21'):
        assert V.check_spec(next(x for x in SS.seeds() if x['name'] == name))
    with pytest.raises(ValueError, match='banned'):
        V.check_spec(dict(solo2(), name='firm'))                                  # owner: 15 m/s banned
    bad = []
    s = inc(); s['extra'] = 1; bad.append(s)                                      # noqa: E702
    s = copy.deepcopy(inc()); s['roles'][0]['v_w'] = 16.0; bad.append(s)          # noqa: E702
    s = copy.deepcopy(inc()); s['roles'][0]['vid'] = 'c1'; bad.append(s)          # noqa: E702 (vehicle ids)
    s = copy.deepcopy(inc()); s['roles'][0]['lane'] = 2.0; bad.append(s)          # noqa: E702
    s = dict(inc(), spread_m=250.0); bad.append(s)                                # noqa: E702
    s = dict(inc(), slots=2); bad.append(s)                                       # noqa: E702
    s = dict(inc(), cooldown_s=300); bad.append(s)                                # noqa: E702
    s = dict(inc(), gate=7.0); bad.append(s)                                      # noqa: E702
    s = dict(inc(), start_gate='empty50'); bad.append(s)                          # noqa: E702
    s = dict(wall3(), phases=[[0], [0, 1, 2]]); bad.append(s)                     # noqa: E702
    s = dict(wall3(), phases=[[0, 1, 2]]); bad.append(s)                          # noqa: E702
    for name in ('u0', 'base', 'wall_h_solo2', 'wall_g_solo2', 'inc_core', 'a_minus_b', '9x', 'x y'):
        bad.append(dict(inc(), name=name))
    for s in bad:
        with pytest.raises((ValueError, KeyError, TypeError)):
            V.check_spec(s)
    with pytest.raises(ValueError, match='closing-lane'):
        V.check_spec(lane_spec(), lane_executor='plant')
    assert V.check_spec(lane_spec(target=2), lane_executor='plant')


def test_seed_range_and_cells():
    cells = V.parse_cells(['D3200_p60:989950-989953', 'D2800_p60:989954-989955'])
    assert [c[0] for c in cells] == ['D3200_p60', 'D2800_p60'] and cells[0][2] == 0.6 and len(cells[0][3]) == 4
    for bad in (['D3200_p60:989949-989950'], ['D3200_p60:989900-989901'], ['D3200_p60:989600-989601'],
                ['D3200_p60:989998-990000'], ['D3200_p60:989950-989951', 'D2800_p60:989951-989952']):
        with pytest.raises(ValueError):
            V.parse_cells(bad)
    from tools.coopmwm import map_scenarios as ms
    assert ms.seed_scope(989950) == 'train' and ms.seed_scope(989999) == 'train'
    assert ms.seed_scope(991140) == 'train' and ms.seed_scope(991199) == 'train'
    assert len(V.parse_cells(['D3200_p20:991140-991151'])[0][3]) == 12
    assert ms.seed_scope(991200) == 'train' and ms.seed_scope(991299) == 'train'
    assert ms.seed_scope(991300) == 'train' and ms.seed_scope(991347) == 'train'
    assert ms.seed_scope(991348) == 'train' and ms.seed_scope(991399) == 'train'
    assert ms.seed_scope(991400) == 'train' and ms.seed_scope(991419) == 'train'
    assert ms.seed_scope(991420) == 'train' and ms.seed_scope(991499) == 'train'
    assert ms.seed_scope(991500) == 'train' and ms.seed_scope(991599) == 'train' and ms.seed_scope(991600) is None
    assert len(V.parse_cells(['D2800_p20:991420-991467'])[0][3]) == 48
    assert len(V.parse_cells(['D2800_p20:991396-991419'])[0][3]) == 24
    assert len(V.parse_cells(['D2800_p20:991348-991371'])[0][3]) == 24
    assert len(V.parse_cells(['D2800_p20:991300-991347'])[0][3]) == 48
    assert len(V.parse_cells(['D3200_p20:991188-991211'])[0][3]) == 24
    for bad in (['D3200_p20:991139-991140'], ['D3200_p20:991599-991600'], ['D3200_p20:991100-991101']):
        with pytest.raises(ValueError):
            V.parse_cells(bad)
    assert ms.seed_scope(989900) is None and ms.seed_scope(989949) is None


def test_candidates_groups_dedupe_and_incumbent_drop():
    a, b = dict(inc(), name='g_v18', roles=[dict(SS.INCUMBENT['roles'][0], v_w=18.0)]), lane_spec()
    c = next(x for x in SS.seeds() if x['name'] == 'wall_g_alt21')
    doc = {'D3200_p60': dict(wm=[a, dict(inc(), name='wm_inc')], wm_bounded=[dict(a, name='dup_of_a'), b],
                             wm_fairer=[], reference=[], structured=[], random=[], best=dict(delay=c, fairer=None),
                             incumbent=inc(), banned_comparator=solo2())}
    got = V.candidates(doc, 'D3200_p60', V.GROUPS + V.SINGLE_GROUPS)
    assert [(src, s['name']) for src, s in got] == [('wm', 'g_v18'), ('wm_bounded', 'wmv_lc_closing'),
                                                     ('best:delay', 'wall_g_alt21')]
    assert V.candidates({'c': dict(wm=[dict(solo2(), name='firm')])}, 'c') == []     # = the banned arm: dropped
    firm120 = dict(solo2(), name='firm120', cooldown_s=120, roles=[dict(SS.SOLO2['roles'][0], hold_s=120)])
    with pytest.raises(ValueError, match='banned'):
        V.candidates({'c': dict(wm=[firm120])}, 'c')
    with pytest.raises(ValueError, match='two different'):
        V.candidates({'c': dict(wm=[a, dict(b, name='g_v18')])}, 'c')


# ------------------------------------------------------------------------------------------ executors (fake plant)
def _trace_eq(a, b):
    return [x['t'] for x, y in zip(a['trace'], b['trace']) if x['desired'] != y['desired'] or x['g5'] != y['g5']]


@pytest.mark.parametrize('strategy,spec', [('wall_g_solo2', inc), ('wall_h_solo2', solo2)])
def test_incumbent_core_reproduces_regimeu0_rulehook(strategy, spec):
    rule = R.RuleHook(strategy)
    core = VH.make_hook(dict(spec(), name='inc_core'), executor='plant')
    a = F.run_episode(rule, seed=989950, seconds=SECONDS, record=True)
    b = F.run_episode(core, seed=989950, seconds=SECONDS, record=True)
    assert len(rule.acts) >= 2 and sum(1 for x in a['trace'] if x['desired']) > 50
    assert _trace_eq(a, b) == [] and a['totals'] == b['totals']
    assert [(x['t'], x['members']) for x in rule.acts] == [(x['t'], x['members']) for x in core.acts]
    assert all(x['release'] == y['release'] for x, y in zip(rule.acts, core.acts))


@pytest.mark.parametrize('name', ['wall_g_duo12', 'g2_stagger40', 'wall_g_alt21_gate0'])
def test_plant_executor_is_wmc_make_hook(name):
    spec = next(s for s in SS.seeds() if s['name'] == name)
    a = F.run_episode(SH.make_hook(spec), seed=989951, seconds=SECONDS, record=True)
    h = VH.make_hook(spec, executor='plant')
    b = F.run_episode(h, seed=989951, seconds=SECONDS, record=True)
    assert _trace_eq(a, b) == [] and a['totals'] == b['totals'] and h.receipts()['executor'] == 'plant'


def test_default_executors():
    assert VH.resolve_executor(inc()) == 'plant' and VH.resolve_executor(lane_spec()) == 'lcexec'
    assert VH.resolve_executor(None) == 'none'
    with pytest.raises(ValueError):
        VH.resolve_executor(lane_spec(), lane_executor='nope')


def test_lcexec_policy_executor_runs_lane_programs(wz):
    h = VH.make_hook(lane_spec(), executor='lcexec')
    assert h.needs_lc_runtime
    res = F.run_episode(h, seed=989952, seconds=SECONDS, record=True)
    rc = h.receipts()
    assert rc['activations'] >= 2 and rc['protocol']['sessions'] == rc['activations'] - rc['refused_activations']
    assert rc['protocol']['n_refusals'] == 0 and rc['lc_ticks'] > 0
    st = rc['lc_vehicle_status']
    assert st.get('issued', 0) + st.get('held', 0) > 0                      # lane programs actually executed
    assert any(t['lc'] for t in res['trace'])
    assert not any(t['desired'] for t in res['trace'])                      # members go through LCEXEC, not the plant
    # every session's members are the StrategyCore selection (same select_members / roles / timing)
    assert [s['members'] for s in rc['protocol']['session_log']] == [a['members'] for a in h.acts
                                                                      if not a.get('executor_refused')]


def test_lcexec_policy_executor_holds_too(wz):
    h = VH.make_hook(solo2(), executor='lcexec')
    F.run_episode(h, seed=989953, seconds=SECONDS)
    rc = h.receipts()
    assert rc['activations'] >= 2 and rc['protocol']['sessions'] >= 2 and rc['lc_ticks'] > 0


def test_lcexec_needs_the_lcexec_env(wz):
    h = VH.make_hook(lane_spec(zone='full'), executor='lcexec')
    with pytest.raises(RuntimeError, match='LCEXEC env'):
        F.run_episode(h, seed=989952, seconds=SECONDS, lc_runtime=False)


def test_cav_only_guard(monkeypatch):
    h = VH.make_hook(solo2(), executor='plant')
    monkeypatch.setattr(h.core, 'step', lambda rows, commands, t: ({next(r['vid'] for r in rows
                                                                          if not r['is_cav']): -1.0}, {}))
    with pytest.raises(RuntimeError, match='non-CAV'):
        F.run_episode(h, seed=989950, seconds=SECONDS)


def test_family_template_and_offer_condition(wz):
    from tools.coopmwm import lc_families as lf, lc_program as lp
    frame = lp.frame_from_descriptor(wz)
    for s in SS.seeds() + [lane_spec()]:
        for ph in range(VH.n_phases(s)):
            d, sha = VH.family_template(s, frame, phase=ph)
            tpl = lf.template_of(d, cap=6, frame=frame, n_lanes=3)
            assert tpl.members == tuple(f'@{k}' for k in range(len(SS.active_roles(s, ph))))
            assert sha == lf.template_sha256(d)
        oc = VH.offer_condition(s)
        assert oc['own_wm_lanes'] == sorted({2 - r['lane'] for r in s['roles']})
        assert oc['zone_upstream_m'][0] >= 400.0 and (oc['gate'] is None) == (s.get('gate') is None)
        assert (oc['start_gate'] is None) == (s.get('start_gate') is None)
        assert (oc['phases'] is None) == (s.get('phases') is None)


# ------------------------------------------------------------------------------------------ analysis
def _fullpol_rows(analysis):
    """Verbatim copy of fullpol_learn.guidance_rows_from_analysis (c4/fullpol-5a e36ece92) + its skip lists."""
    BASE_ENTRIES, DERIVED_ENTRIES = ('base', 'u0'), ('group_minus_best_solo',)
    out = []
    for cell, arms in sorted(analysis['cells'].items()):
        for a, v in sorted(arms.items()):
            if a in BASE_ENTRIES or a in DERIVED_ENTRIES or 'minus' in a:
                continue
            if not isinstance(v, dict) or not {'mean_diff', 'ci95'} <= set(v) or not {'n', 'n_paired'} & set(v):
                raise ValueError(f'unrecognised ANALYSIS entry {cell}/{a}: strategy rows need mean_diff, ci95, n')
            lo, hi = v['ci95']
            out.append(dict(arm='B', regime=cell, family=a, baseline='mappo', n=int(v.get('n_paired', v.get('n'))),
                            mean_veh_s=-float(v['mean_diff']), lb95_veh_s=-float(hi), ub95_veh_s=-float(lo),
                            harm_rate=float(v.get('harm_rate', float('nan')))))
    return out


def test_round_trip_regimeu0_c0b578f8():
    raw = gzip.decompress(FIX.read_bytes())
    import hashlib
    assert hashlib.sha256(raw).hexdigest().startswith('c0b578f8')
    ref = json.loads(raw)
    got = V.cells_from_rows(ref['seeds'], list(R.ALL_ARMS))
    assert set(got) == set(ref['cells'])
    for cell, arms in ref['cells'].items():
        for arm, v in arms.items():
            if arm == 'group_minus_best_solo':      # c0b578f8 predates the group/beats_all_solos_fraction keys
                assert {k: got[cell][arm][k] for k in v} == v
            else:
                assert got[cell][arm] == v, (cell, arm)
    rows = _fullpol_rows(dict(cells=got))
    solo = {(r['regime'], r['family']): r for r in rows if r['family'] == 'wall_h_solo2'}
    assert solo[('D3200_p60', 'wall_h_solo2')]['mean_veh_s'] == -ref['cells']['D3200_p60']['wall_h_solo2']['mean_diff']


def _summary(hook, res, job):
    return dict(schema=R.SCHEMA + '.summary', id=job['id'], arm=job['arm'], strategy=job['strategy'],
                world=job['world'], totals=res['totals'], actual_sim_s=res['actual_sim_s'],
                series_keys=list(ir.SERIES_KEYS), series=hook.series, activations=hook.acts, issued=len(hook.issued),
                lane_held_steps=hook.lane_held, lane_outcome={}, base_sha256='x', error=None, wall_s=0.)


def _fake_root(tmp_path, arms=('u0', 'wall_g_solo2', 'wall_g', 'wall_g_solo0'), seeds=(989950, 989951)):
    """A regimeu0-format root with fake-plant SUMMARY files (identity + paired groups)."""
    root = tmp_path / 'root'
    (root / 'jobs').mkdir(parents=True)
    groups = [dict(stage='identity', jobs=[dict(id=V.IDENTITY_JOB, arm='u0', strategy=None, sim_s=1200,
                                                world=R.world_of(989599))])]
    for s in seeds:
        w = R.world_of(s)
        groups.append(dict(stage='paired', cell=w['cell_key'],
                           jobs=[dict(id=f's{s}_{a}', arm=a, strategy=None if a == 'u0' else a, world=w, sim_s=1200)
                                 for a in arms]))
    for g in groups:
        for j in g['jobs']:
            hook = R.RuleHook(j['strategy'])
            res = F.run_episode(hook, seed=j['world']['seed'], seconds=SECONDS)
            (root / 'jobs' / j['id']).mkdir()
            with gzip.open(root / 'jobs' / j['id'] / 'SUMMARY.json.gz', 'wt', encoding='utf-8') as f:
                json.dump(_summary(hook, res, j), f)
    (root / 'PLAN.json').write_text(json.dumps(dict(schema=V.SCHEMA + '.plan', groups=groups, plan_sha256='p',
                                                    inputs={})))
    (root / 'REGISTRATION.json').write_text(json.dumps(dict(cap_sim_s=10 ** 6)))
    return root


def test_analysis_equals_regimeu0_analyze(tmp_path):
    root = _fake_root(tmp_path)
    ref = R.analyze([root])
    got = V.analysis([root])
    assert json.loads(json.dumps(got, sort_keys=True, default=str)) == json.loads(
        json.dumps(ref, sort_keys=True, default=str))
    assert set(got) == set(ref) == {'schema', 'roots', 'spent', 'plan_sha256', 'cells', 'seeds', 'identity'}


# ------------------------------------------------------------------------------------------ fairness (N6 port)
def _vlog(path, heads, ticks):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for h in heads:
            f.write(json.dumps(h) + '\n')
        for t in ticks:
            f.write(json.dumps(t) + '\n')


def test_per_vehicle_first_logged_row_and_depart_delay(tmp_path, monkeypatch):
    """nm_finaleval.vehicle_log_report rules: origin = WM lane of the first row at t >= 300 if upstream of the drop
    (else 'D' = undefined); delay = window speed deficit + depart delay only if first seen after 300 s."""
    from tools.wzprop import wmverify_fair as WF
    monkeypatch.setattr(WF, 'HORIZON_S', 303)
    heads = [dict(veh=0, id='a', cav=False, origin='up', depart_delay_s=4.0, first_t=300),     # on road at 300
             dict(veh=1, id='b', cav=True, origin='up', depart_delay_s=2.5, first_t=301),      # inserted later
             dict(veh=2, id='c', cav=False, origin='up', depart_delay_s=0.0, first_t=300)]     # first seen past drop
    v = 31.29 / 2
    ticks = [dict(t=300, k=[0, 2], s=[-500.0, 80.0], v=[v, 31.29], a=[0, 0], lane=[0, 1]),
             dict(t=301, k=[0, 1, 2], s=[-480.0, -2600.0, 100.0], v=[v, 31.29, 31.29], a=[0, 0, 0], lane=[1, 2, 1]),
             dict(t=302, k=[0, 1], s=[-460.0, -2580.0], v=[v, 31.29], a=[0, 0], lane=[1, 2])]
    _vlog(tmp_path / 'v.jsonl.gz', heads, ticks)
    pv, problems = WF.per_vehicle(tmp_path / 'v.jsonl.gz')
    assert problems == []
    assert pv['a'] == [1.5, 'L2', False]          # first row lane 0 (WM) -> L2; depart delay NOT added (first_t 300)
    assert pv['b'] == [2.5, 'L0', True]           # inserted after 300: depart delay added; WM lane 2 -> L0
    assert pv['c'] == [0.0, 'D', False]           # first logged row downstream of the drop -> undefined
    assert WF._labels(pv)['c'] == ('class:hdv', 'origin:undefined')


def test_co_primary_groups_labels_from_u0_by_vehicle_id():
    """Amendment 13: labels from u0 applied by vehicle id (an arm's own insertion lane is ignored), undefined
    reported only, missing vehicles counted both ways, ALLOWED needs all five groups present and passing."""
    from tools.wzprop import wmverify_fair as WF
    u0, arm = {}, {}
    for w in (1, 2):
        u0[w] = {'h0': [10.0, 'L0', False], 'h1': [10.0, 'L1', False], 'c2': [10.0, 'L2', True],
                 'x': [10.0, 'D', False], 'gone': [10.0, 'L1', False]}
        # same vehicles, but h0 entered on another lane in the arm (its own label would be L2) and +5 s for all
        arm[w] = {'h0': [15.0, 'L2', False], 'h1': [15.0, 'L1', False], 'c2': [15.0, 'L2', True],
                  'x': [15.0, 'D', False], 'new': [99.0, 'L0', True]}
    r = WF.co_primary_groups({'u0': u0, 'arm': arm}, [1, 2], 'arm', 'u0', reference='u0')
    g = r['groups']
    assert g['origin:L0']['increase_mean_s'] == pytest.approx(5.0)          # h0 stays in L0 (u0 label)
    assert g['origin:undefined']['increase_mean_s'] == pytest.approx(5.0) and not g['origin:undefined']['registered']
    assert r['vehicles_missing']['arm'] == dict(not_in_reference=2, reference_not_in_arm=2)
    assert r['all_pass'] is True
    own = r['sensitivity_own_arm_labels']['groups']
    assert own['origin:L0']['increase_mean_s'] != pytest.approx(5.0)       # own-arm labels differ (sensitivity)
    # a co-primary group absent -> not allowed (fail closed)
    for w in (1, 2):
        u0[w].pop('c2')
        arm[w].pop('c2')
        arm[w].pop('new')
    r2 = WF.co_primary_groups({'u0': u0, 'arm': arm}, [1, 2], 'arm', 'u0', reference='u0')
    assert r2['groups']['class:cav']['status'] == 'unavailable' and r2['all_pass'] is False
    # one world only -> every group unavailable -> not allowed
    assert WF.co_primary_groups({'u0': u0, 'arm': arm}, [1], 'arm', 'u0', reference='u0')['all_pass'] is False


# ------------------------------------------------------------------------------------------ registration gate
def test_registration_gate_exact_block(tmp_path):
    cells = V.parse_cells(['D3200_p60:989950-989951'])
    specs = {'D3200_p60': [('wm', dict(inc(), name='g_v18', roles=[dict(SS.INCUMBENT['roles'][0], v_w=18.0)]))]}
    text, planned = V.registration_text(cells=cells, specs_by_cell=specs, cap=40_000, parity_seeds=1)
    _ir, _R = V._harness()
    gr = V.build_groups(cells, specs, R.world_of, parity_seeds=1)
    sha, mods = V.jobs_digest(gr), V.module_shas()
    assert V.registration_check(text, cap=40_000, planned=planned, jobs_sha256=sha, arms_sha256=None,
                                module_sha256=mods)
    stray = ('S-WMVERIFY wm_verify 989950-989999 40,000 D3200_p60 ' + text.split('\n', 1)[1])   # no exact header
    with pytest.raises(ValueError, match='exact header'):
        V.registration_check(stray, cap=40_000, planned=planned, jobs_sha256=sha, arms_sha256=None, module_sha256=mods)
    for kw in (dict(jobs_sha256='0' * 64), dict(cap=50_000), dict(planned=planned + 1200), dict(arms_sha256='ab'),
               dict(module_sha256=dict(mods, wmverify='x'))):
        args = dict(cap=40_000, planned=planned, jobs_sha256=sha, arms_sha256=None, module_sha256=mods)
        args.update(kw)
        with pytest.raises(ValueError, match='no S-WMVERIFY registration block'):
            V.registration_check(text, **args)


def test_holdpin_is_cancelled(tmp_path):
    from tools.wzprop import holdpin as HP
    with pytest.raises(ValueError, match='CANCELLED'):
        HP.plan(tmp_path / 'hp', regimeu0_root=tmp_path, registration_md=tmp_path / 'R.md')
    with pytest.raises(ValueError, match='CANCELLED'):
        HP.run(tmp_path / 'hp')


# ------------------------------------------------------------------------------------------ end to end (fake episode)
def _registration(tmp_path, cells, specs, cap, parity_seeds=1, banned_seeds=0, arms=None):
    text, planned = V.registration_text(cells=cells, specs_by_cell=specs, cap=cap, regimeu0_u0_sha256='u0sha',
                                        parity_seeds=parity_seeds, banned_seeds=banned_seeds,
                                        arms_sha256=None if arms is None else ir.sha_file(arms))
    p = tmp_path / 'REGISTRATION.md'
    p.write_text('# run registrations\nS-WMVERIFY wm_verify mention elsewhere\n\n' + text, encoding='utf-8')
    return p, planned


def _e2e_inputs(tmp_path):
    reg_root = tmp_path / 'regimeu0'
    (reg_root / 'inputs').mkdir(parents=True)
    (reg_root / 'inputs' / 'CONFIG.json').write_text('{}')
    (reg_root / 'inputs' / 'u0.pt').write_bytes(b'u0')
    (reg_root / 'PLAN.json').write_text(json.dumps(dict(inputs={'u0.pt': dict(sha256=ir.sha_file(
        reg_root / 'inputs' / 'u0.pt'))})))
    arms = tmp_path / 'ARMS.json'
    arms.write_text(json.dumps({'D3200_p60': dict(wm=[lane_spec(zone='far'),
                                                      dict(inc(), name='g_v18',
                                                           roles=[dict(SS.INCUMBENT['roles'][0], v_w=18.0)])],
                                                  structured=[], random=[], incumbent=inc(),
                                                  banned_comparator=solo2())}))
    return reg_root, arms


def _fake_physics(monkeypatch):
    import torch
    from tools.coopfree import backend as cfb
    from tools.coopmwm import accel_plant, coop_warmstart, v2v_runtime
    from tools.wzprop import wmverify_fair as WF
    monkeypatch.setenv(md.ENV, 'bound-by-fixture')
    monkeypatch.setenv('COOPMWM_G5_RUNTIME', '1')
    monkeypatch.setattr(v2v_runtime, 'load_checkpoint', lambda *a, **k: (torch.nn.Linear(1, 1), None, None, None))
    monkeypatch.setattr(coop_warmstart, 'state_sha256', lambda m: 'fake')
    monkeypatch.setattr(cfb, 'tick', F.fake_tick)
    monkeypatch.setattr(WF, 'HORIZON_S', SECONDS)          # fake episodes stop at SECONDS (the log window too)
    episode = F.fake_episode_factory()

    def fake_episode(job, **kw):
        return episode(dict(job, stop=SECONDS), **kw)
    monkeypatch.setattr(accel_plant, 'episode', fake_episode)


def _run_all(root, body):
    for j in [j for g in body['groups'] for j in g['jobs']]:
        assert ir.charge(root, [(j['id'], j['sim_s'], 'paired')])
        s = V.worker(root, j['id'])
        assert not s['error'], s['error']


def test_end_to_end_plan_worker_analyze(tmp_path, monkeypatch, wz):
    reg_root, arms = _e2e_inputs(tmp_path)
    cells = V.parse_cells(['D3200_p60:989950-989951'])
    specs = {'D3200_p60': V.candidates(json.loads(arms.read_text()), 'D3200_p60')}
    cap = 40_000
    reg, planned = _registration(tmp_path, cells, specs, cap, banned_seeds=1, arms=arms)
    with pytest.raises(ValueError, match='no S-WMVERIFY registration block'):
        V.plan(tmp_path / 'bad', regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap + 1,
               registration_md=reg, parity_seeds=1, banned_seeds=1)
    with pytest.raises(ValueError, match='no S-WMVERIFY registration block'):     # budget differs from the block
        V.plan(tmp_path / 'bad2', regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap,
               registration_md=reg, parity_seeds=1, banned_seeds=0)
    root = tmp_path / 'run'
    body = V.plan(root, regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap, registration_md=reg,
                  parity_seeds=1, banned_seeds=1)
    assert body['planned_sim_s'] == planned == 1200 * (1 + 2 * 4 + 1 + 1)
    assert body['groups'][0]['stage'] == 'identity'                      # mandatory, no bypass
    fam = body['families']
    assert fam['wmv_lc_closing']['executor'] == 'lcexec' and fam['g_v18']['executor'] == 'plant'
    assert fam['wmv_lc_closing']['lcexec_parity']['id'] == 'S-LCEXEC-SMOKE' and fam['g_v18']['lcexec_parity'] is None
    assert fam['wmv_lc_closing']['n2_parity'] is None and fam['g_v18']['n2_parity']['identical']
    jobs = [j for g in body['groups'] for j in g['jobs']]
    assert [j['arm'] for j in jobs[:7]] == ['u0', 'u0', 'wall_g_solo2', 'wall_h_solo2', 'inc_core', 'wmv_lc_closing',
                                            'g_v18']
    with pytest.raises(ValueError, match='prior root'):
        V.plan(tmp_path / 'run2', regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap, registration_md=reg,
               prior_roots=[root], parity_seeds=1, banned_seeds=1)
    _fake_physics(monkeypatch)
    _run_all(root, body)
    assert ir.spent(root) == body['planned_sim_s']
    assert all((root / V.VEHICLE_LOG_DIR / j['id'] / 'VEHICLES.jsonl.gz').exists() for j in jobs)
    # ---- the fake plant cannot reproduce the identity J -> FAIL CLOSED: no FULLPOL-facing file
    blocked = tmp_path / 'blocked'
    rep0, ver0 = V.analyze([root], blocked)
    assert rep0 is None and not ver0['analysis_written']
    assert sorted(p.name for p in blocked.iterdir()) == ['CHECK.json', 'VERIFY.json']
    assert 'identity_u0_989599' in ver0['summary']['failed_checks']
    with pytest.raises(SystemExit):
        V.main(['analyze', '--root', str(root), '--out', str(tmp_path / 'blocked_cli')])
    # ---- explicit test-only override of the identity reference (the fake identity episode's own J)
    idj = ir._read(root / 'jobs' / V.IDENTITY_JOB / 'SUMMARY.json.gz')['totals']['team_delay_vs']
    monkeypatch.setattr(R, 'IDENTITY', dict(R.IDENTITY, expected_team_delay_vs=idj))
    out = tmp_path / 'analysis'
    rep, ver = V.analyze([root], out)
    assert ver['integrity']['ok'], [c for c in ver['integrity']['checks'] if not c['ok']]
    assert set(rep) == {'schema', 'roots', 'spent', 'plan_sha256', 'cells', 'seeds', 'identity'}
    assert rep['schema'] == R.SCHEMA + '.analysis'
    cell = rep['cells']['D3200_p60']
    assert set(cell) == {'u0', 'wall_g_solo2', 'wmv_lc_closing', 'g_v18'}   # parity + banned arms are not families
    rows = _fullpol_rows(json.loads((out / 'ANALYSIS.json').read_text()))
    assert {r['family'] for r in rows} == {'wall_g_solo2', 'wmv_lc_closing', 'g_v18'}
    assert all(r['n'] == 2 for r in rows)
    vs = ver['vs_incumbent']['D3200_p60']
    assert set(vs) == {'wmv_lc_closing', 'g_v18', 'wall_h_solo2'} and vs['g_v18']['n'] == 2
    assert vs['wall_h_solo2']['n'] == 1 and 'banned' in vs['wall_h_solo2']['label']
    for a in ('wmv_lc_closing', 'g_v18'):
        v = vs[a]
        d = [rep['seeds'][i]['arms'][a]['J'] - rep['seeds'][i]['arms']['wall_g_solo2']['J'] for i in range(2)]
        assert v['mean_diff'] == pytest.approx(sum(d) / 2) and math.isclose(v['better_share'], sum(x < 0 for x in d) / 2)
    # per-group fairness: N6 amendment 13 (labels from u0), ALLOWED filter
    fair = ver['fairness']['D3200_p60']
    for a in ('wall_g_solo2', 'wmv_lc_closing', 'g_v18'):
        f0 = fair[a]['fairness']['u0']
        assert f0['reference'] == 'u0' and f0['n_worlds'] == 2 and fair[a]['allowed'] == f0['all_pass']
        assert set(f0['groups']) == {'origin:L0', 'origin:L1', 'origin:L2', 'class:hdv', 'class:cav',
                                     'origin:undefined'}
        assert 'sensitivity_own_arm_labels' in f0 and set(f0['vehicles_missing']) == {a, 'u0'}
    assert fair['wall_h_solo2']['label'] == 'banned comparator'
    allowed = json.loads((out / 'ANALYSIS_ALLOWED.json').read_text())
    keep = {a for a in ('wall_g_solo2', 'wmv_lc_closing', 'g_v18') if fair[a]['allowed']}
    assert set(allowed['cells']['D3200_p60']) == {'u0'} | keep and allowed['schema'] == rep['schema']
    assert {r['family'] for r in _fullpol_rows(allowed)} == keep
    assert ver['summary']['D3200_p60']['allowed'] == sorted(a for a in ('wmv_lc_closing', 'g_v18') if fair[a]['allowed'])
    chk = ver['integrity']
    names = {c['name']: c['ok'] for c in chk['checks']}
    assert names['executor_parity_solo2_core_equals_incumbent'] and names['crn_no_divergence_before_first_activation']
    assert names['vehicle_log_present_every_episode'] and names['identity_u0_989599']
    assert chk['lane_receipts']['wmv_lc_closing']['sessions'] > 0
    assert ver['families']['wmv_lc_closing']['template']['schema'] == 'lcexec-free-program-v1'
    res = [json.loads(x) for x in (out / 'RESULTS.jsonl').read_text().splitlines()]
    assert {r['arm'] for r in res} == {'u0', 'wall_g_solo2', 'wmv_lc_closing', 'g_v18'}    # no banned rows
    assert all('J_300_1200_vs' in r for r in res)
    with pytest.raises(FileExistsError):
        V.analyze([root], out)
    # ---- verified-plan bank (clean5c): allowed families only, never banned / parity arms
    from tools.wzprop import wmverify_bank as BK
    bank = BK.build(out, [root])
    assert {r['family'] for r in bank} <= keep and all(r['family'] not in ('wall_h_solo2', 'inc_core') for r in bank)
    assert all(r['members'] and r['template'] for r in bank) and all(r['seed'] in (989950, 989951) for r in bank)
    if keep - {'wall_g_solo2'}:
        assert bank
    with pytest.raises(ValueError, match='fail closed'):
        BK.build(tmp_path / 'blocked', [root])
    # ---- a missing vehicle log also fails closed
    (root / V.VEHICLE_LOG_DIR / jobs[3]['id'] / 'VEHICLES.jsonl.gz').unlink()
    rep2, ver2 = V.analyze([root], tmp_path / 'blocked2')
    assert rep2 is None and 'vehicle_log_present_every_episode' in ver2['summary']['failed_checks']
    assert not (tmp_path / 'blocked2' / 'ANALYSIS_ALLOWED.json').exists()


def test_dryrun_reports_incumbent_parity(tmp_path, wz):
    arms = tmp_path / 'ARMS.json'
    arms.write_text(json.dumps({'D3200_p60': dict(wm=[lane_spec()], structured=[], random=[])}))
    rep = V.dryrun(tmp_path / 'dry', arms_path=arms, cells=V.parse_cells(['D3200_p60:989950-989951']),
                   fake_seconds=SECONDS)
    c = rep['cells']['D3200_p60']
    assert c['incumbent_parity']['identical'] and c['incumbent_parity']['incumbent_command_ticks'] > 0
    assert c['arms']['wmv_lc_closing']['executor'] == 'lcexec'
    body = json.loads((tmp_path / 'dry' / 'PLAN.json').read_text())
    assert body['dry_run'] is True
    with pytest.raises(ValueError, match='dry-run'):
        V.run(tmp_path / 'dry')


def _wmsearch_root(tmp_path, schema, specs, name='ws'):
    root = tmp_path / name
    (root / 'jobs').mkdir(parents=True)
    groups, rows = [], []
    for seed in (989770, 989771):
        w = R.world_of(seed)
        jobs = [dict(id=f'D3200_p60_s{seed}_{arm}', arm=arm, strategy=None if sp is None else sp['name'],
                     strategy_spec=sp, world=w, sim_s=1200) for arm, sp in specs.items()]
        groups.append(dict(stage='paired', cell='D3200_p60', jobs=jobs))
        rec = dict(seed=seed, J={})
        for j in jobs:
            hook = SH.make_hook(j['strategy_spec'])
            res = F.run_episode(hook, seed=seed, seconds=SECONDS)
            (root / 'jobs' / j['id']).mkdir()
            summ = _summary(hook, res, j)
            with gzip.open(root / 'jobs' / j['id'] / 'SUMMARY.json.gz', 'wt', encoding='utf-8') as f:
                json.dump(summ, f)
            rec['J'][j['arm']] = R.control_j(summ)
        rows.append(rec)
    (root / 'PLAN.json').write_text(json.dumps(dict(schema=schema, groups=groups, specs=specs,
                                                    plan_sha256='w', inputs={'u0.pt': dict(sha256='u0')})))
    (root / 'REGISTRATION.json').write_text(json.dumps(dict(cap_sim_s=10 ** 6)))
    return root, rows


def test_wmsearch_roots_banned_specs_never_fullpol_facing_and_fail_closed(tmp_path):
    """S-WMSEARCH v1/v2 roots: banned-spec arms (v1 structured solo2_v15_h120, the solo2 comparator) are excluded
    from ANALYSIS / RESULTS / families by SPEC; the conversion still agrees with wmsearch_u0.analyze_rows; without
    identity episode / vehicle logs analyze FAILS CLOSED (no ANALYSIS files)."""
    from tools.wzprop import wmsearch_u0 as WS
    a = dict(inc(), name='g_v18', roles=[dict(SS.INCUMBENT['roles'][0], v_w=18.0)])
    v15 = dict(solo2(), name='solo2_v15_h120', cooldown_s=120, roles=[dict(SS.SOLO2['roles'][0], hold_s=120)])
    specs = dict(u0=None, solo2=solo2(), str_0=v15, wm_0=a)            # v1: solo2 incumbent, pre-ban 15 m/s spec
    root, rows = _wmsearch_root(tmp_path, V.WMSEARCH_SCHEMA, specs)
    col = V.collect([root])
    assert set(col['banned']) == {'wall_h_solo2', 'solo2_v15_h120'}
    rep = V.analysis([root], col)
    assert set(rep['cells']['D3200_p60']) == {'u0', 'g_v18'}
    assert all(not ({'wall_h_solo2', 'solo2_v15_h120'} & set(r['arms'])) for r in rep['seeds'])
    assert {r['arm'] for r in V.results_rows(col['rows'], set(V.NOT_FAMILIES) | set(col['banned']))} == {'u0', 'g_v18'}
    assert set(V.wmsearch_families(json.loads((root / 'PLAN.json').read_text()))) == {'g_v18'}
    _r, ver = V.analyze([root], tmp_path / 'out_v1')
    assert _r is None and not (tmp_path / 'out_v1' / 'ANALYSIS.json').exists()
    assert {'identity_u0_989599', 'vehicle_log_present_every_episode'} <= set(ver['summary']['failed_checks'])
    assert 'solo2_v15_h120' not in ver['families']
    # v2: inc -> wall_g_solo2, solo2 -> banned comparator; contrasts agree with analyze_rows
    b = next(s for s in SS.seeds() if s['name'] == 'g2_stagger40')
    specs2 = dict(u0=None, inc=inc(), solo2=solo2(), wm_0=a, str_0=b)
    root2, rows2 = _wmsearch_root(tmp_path, 'wmc.wmsearch_u0.v2.plan', specs2, name='ws2')
    col2 = V.collect([root2])
    assert set(V.analysis([root2], col2)['cells']['D3200_p60']) == {'u0', 'wall_g_solo2', 'g_v18', 'g2_stagger40'}
    order = ['u0'] + [x for x in col2['arms'] if x != 'u0']
    vs = V.contrasts(col2['rows'], order)['D3200_p60']
    ref = WS.analyze_rows(rows2, specs2, n_boot=200)['arms']
    for arm, name in (('wm_0', 'g_v18'), ('str_0', 'g2_stagger40')):
        assert vs[name]['mean_diff'] == pytest.approx(ref[arm]['vs_inc']['mean'])


def test_analyze_refuses_foreign_roots(tmp_path):
    root = tmp_path / 'hp'
    root.mkdir()
    (root / 'PLAN.json').write_text(json.dumps(dict(schema='wmverify.holdpin.v1.plan', groups=[])))
    with pytest.raises(ValueError, match='refused'):
        V.collect([root])


def test_n2_parity_hold_strategies():
    """Policy N2 hold execution (FULLPOL e36ece92 lane_drop.member_command / SoloHold) vs the plant path through the
    real env writer on fixture states: speed writes identical; lane-mode writes differ only by the plant path's
    G5 own-lane pin (513); the n2 executor (no pin) is TraCI-identical; staggered onsets > 30 s are not N2."""
    from tools.wzprop import wmverify_n2 as N2
    r = N2.parity(solo2(), seconds=SECONDS, n2_pin=False, both=False)        # FULLPOL e36ece92 (no pin)
    assert r['verdict'] == 'speed_identical_lane_mode_differs' and r['speed_writes_identical']
    assert r['write_counts']['speed_plant'] == r['write_counts']['speed_n2'] > 50 and r['write_counts']['lane_n2'] == 0
    assert all(d['side'] == 'plant_only' and d['write'][1] == 'setLaneChangeMode' for d in r['differences'])
    assert r['n2_moves'][0][0]['family'] == 'group_hold' and len(r['n2_moves'][0][0]['members']) == 1
    p = N2.parity(solo2(), seconds=SECONDS)                                   # FULLPOL 5cbd616e hold_lane_pin ON
    assert p['identical'] is True and p['n2_hold_lane_pin'] and p['pin_off']['verdict'] == r['verdict']
    g = N2.parity(dict(solo2(), name='solo2_gate10', gate=10.0), seconds=SECONDS, both=False)
    assert g['identical'] is True and 'ungated twin' in g['evaluated_on']
    n = N2.parity(solo2(), executor='n2', seconds=SECONDS, n2_pin=False, both=False)   # unpinned twins
    assert n['identical'] is True and n['verdict'] == 'identical'
    w = N2.parity(wall3(), seconds=SECONDS, both=False)
    assert w['identical'] and w['n2_moves'][0][0]['family'] == 'abreast_wall'
    stag = next(s for s in SS.seeds() if s['name'] == 'g2_stagger40')
    alt = N2.parity(next(s for s in SS.seeds() if s['name'] == 'wall_g_alt21'), seconds=SECONDS, both=False)
    assert alt['identical'] and all(len(m[0]['members']) == 1 for m in alt['n2_moves'])   # alternating solo holds
    assert N2.parity(stag, seconds=SECONDS, both=False)['verdict'] == 'not_representable'
    with pytest.raises(ValueError, match='n2 executor'):
        VH.make_hook(lane_spec(), executor='n2')
    a = F.run_episode(VH.make_hook(solo2(), executor='plant'), seed=989950, seconds=SECONDS, record=True)
    b = F.run_episode(VH.make_hook(solo2(), executor='n2'), seed=989950, seconds=SECONDS, record=True)
    assert [x['desired'] for x in a['trace']] == [x['desired'] for x in b['trace']]   # fake plant ignores lcMode
    assert any(x['g5'] for x in a['trace']) and not any(x['g5'] for x in b['trace'])


def test_dense_option1_hold_speeds_18_to_22_only():
    """MAIN 2026-11-08: the calibrated option-1 picks use v_w 19 / 21 (WM-C DENSE 18-22 m/s); anything else stays
    rejected (15 banned, 16/17/23/24 unregistered)."""
    base = dict(inc(), name='g_dense')
    for v in (18.0, 19.0, 20.0, 21.0, 22.0):
        V.check_spec(dict(base, roles=[dict(SS.INCUMBENT['roles'][0], v_w=v)]))
    for v in (14.0, 15.0, 16.0, 17.0, 17.5, 22.5, 23.0, 24.0):
        with pytest.raises(ValueError):
            V.check_spec(dict(base, roles=[dict(SS.INCUMBENT['roles'][0], v_w=v)]))
    assert 19.0 not in SS.grids(search=True)['v_w'] and 15.0 not in SS.grids(search=True)['v_w']


def test_fairx_worst_group_and_cvar90_paired():
    from tools.wzprop import wmverify_fairx as FX
    u0 = {str(i): [10.0 + i, 'L0' if i % 3 == 0 else ('L1' if i % 3 == 1 else 'L2'), i % 5 == 0] for i in range(20)}
    arm = {k: [d + (5.0 if o == 'L2' else 0.0), o, c] for k, (d, o, c) in u0.items()}
    rep = FX.report({'D2400_p20': {1: {'u0': u0, 'x': arm}, 2: {'u0': u0, 'x': arm}}}, n_boot=200)
    r = rep['D2400_p20']['x']['vs_u0']
    assert r['worst_group'] == 'origin:L2' and abs(r['worst_group_inc'] - 5.0) < 1e-9
    # worst 2 of 20 vehicles: u0 ids 19, 18 (29, 28); arm adds 5 to L2 (i%3==2): 29 -> 29? id 19 is L1, 18 is L0
    per = sorted((d for d, _, _ in arm.values()), reverse=True)
    assert abs(r['cvar90_diff'] - (sum(per[:2]) / 2 - 28.5)) < 1e-9
    assert FX.headline(rep, 'D2400')[0].startswith('D2400_p20 x: worst origin:L2 +5.0')
    assert r['worst_group_cvar90'] == 'origin:L2' and abs(r['worst_group_cvar90_diff'] - 5.0) < 1e-9
    assert abs(r['group_cvar90']['origin:L0']['mean']) < 1e-9


# ------------------------------------------------------------------------------------------ PAIR arms (2026-11-09)
def _pair_doc():
    from tools.wzprop import strategy_search as SS2
    duo = SS2._spec('duo12_v20', 'wall', [SS2._role(1, dict(v_w=20.0, dmax=1.0, hold_s=120)),
                                           SS2._role(2, dict(v_w=20.0, dmax=1.0, hold_s=120))])
    solo = SS2._spec('mut_solo0', 'lanes', [SS2._role(0, dict(v_w=19.0, dmax=1.0, hold_s=90))])
    return {'D2800_p20': dict(pairs=[dict(name='duo_alone', specs=[duo]),
                                     dict(name='pair_duo_solo0', specs=[duo, solo])])}


def test_pair_oracle_is_the_reviewed_hook():
    from tools.coopmwm import oracle_plan as OP
    from tools.wzprop import wmverify_oracle as WO
    raw = (Path(OP.__file__)).read_bytes().replace(b'\r\n', b'\n')
    assert WO.git_blob_sha(raw) == WO.ORACLE_BLOB and WO.check_oracle() == WO.ORACLE_DIGEST


def test_pair_minibank_rebuilds_and_plans_pair_jobs():
    from tools.wzprop import wmverify_oracle as WO
    pb = V.pairs_of(_pair_doc(), [('D2800_p20', 2800, 0.2, [991372])])
    names = [n for n, _ in pb['D2800_p20']]
    assert names == ['duo_alone', 'pair_duo_solo0']
    from tools.coopmwm import oracle_plan as OP
    plans, refused = OP.bank_plans(pb['D2800_p20'][1][1], 'D2800_p20')
    assert [p['family'] for p in plans] == ['duo12_v20', 'mut_solo0'] and not refused
    assert {p['row_class'] for p in plans} == {'unconditional'}
    gr = V.build_groups([('D2800_p20', 2800, 0.2, [991372])], {'D2800_p20': []}, R.world_of, parity_seeds=0,
                        pairs_by_cell=pb)
    jobs = gr[1]['jobs']
    assert [j['arm'] for j in jobs] == ['u0', 'wall_g_solo2', 'duo_alone', 'pair_duo_solo0']
    assert jobs[3]['executor'] == WO.EXECUTOR and jobs[3]['strategy_spec'] is None and jobs[3]['oracle_bank']
    with pytest.raises(ValueError):
        WO.mini_bank('D2800_p20', [dict(_pair_doc()['D2800_p20']['pairs'][0]['specs'][0], roles=[
            dict(lane=2, rank=0, v_w=20.0, dmax=1.0, hold_s=120, onset_s=0, lc_target=1)])])


def test_pair_recorder_runs_concurrent_families_on_fake_plant():
    from tools.wzprop import wmverify_oracle as WO
    pb = V.pairs_of(_pair_doc(), [('D2800_p20', 2800, 0.2, [991372])])
    job = dict(executor=WO.EXECUTOR, oracle_bank=pb['D2800_p20'][1][1], world=dict(cell_key='D2800_p20'))
    hook = V.make_job_hook(job, R)
    res = F.run_episode(hook, seed=991372, seconds=SECONDS, share=0.2)
    rec = hook.receipts()
    assert rec['oracle_sha256'] == WO.ORACLE_DIGEST and rec['oracle']['refused_counts'] == {}
    assert rec['oracle']['activations'] == len(hook.acts) and res['totals'] is not None
    hook2 = V.make_job_hook(job, R)
    res2 = F.run_episode(hook2, seed=991372, seconds=SECONDS, share=0.2)
    assert res2['totals'] == res['totals'] and hook2.receipts() == rec           # deterministic
