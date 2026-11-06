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
    (root / 'PLAN.json').write_text(json.dumps(dict(groups=groups, plan_sha256='p', inputs={})))
    (root / 'REGISTRATION.json').write_text(json.dumps(dict(cap_sim_s=10 ** 6)))
    return root


def test_analysis_equals_regimeu0_analyze(tmp_path):
    root = _fake_root(tmp_path)
    ref = R.analyze([root])
    got = V.analysis([root])
    assert json.loads(json.dumps(got, sort_keys=True, default=str)) == json.loads(
        json.dumps(ref, sort_keys=True, default=str))
    assert set(got) == set(ref) == {'schema', 'roots', 'spent', 'plan_sha256', 'cells', 'seeds', 'identity'}


# ------------------------------------------------------------------------------------------ end to end (fake episode)
def _registration(tmp_path, cells, specs, cap, parity_seeds=1):
    text, planned = V.registration_text(cells=cells, specs_by_cell=specs, cap=cap, regimeu0_u0_sha256='u0sha',
                                        parity_seeds=parity_seeds)
    p = tmp_path / 'REGISTRATION.md'
    p.write_text('# run registrations\n' + text, encoding='utf-8')
    return p, planned


def test_end_to_end_plan_worker_analyze(tmp_path, monkeypatch, wz):
    import torch
    from tools.coopfree import backend as cfb
    from tools.coopmwm import accel_plant, coop_warmstart, v2v_runtime
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
    cells = V.parse_cells(['D3200_p60:989950-989951'])
    specs = {'D3200_p60': V.candidates(json.loads(arms.read_text()), 'D3200_p60')}
    cap = 40_000
    reg, planned = _registration(tmp_path, cells, specs, cap)
    with pytest.raises(ValueError, match='lacks'):
        V.plan(tmp_path / 'bad', regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap + 1,
               registration_md=reg)
    root = tmp_path / 'run'
    body = V.plan(root, regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap, registration_md=reg,
                  parity_seeds=1, banned_seeds=1)
    assert body['planned_sim_s'] == planned + 1200 == 1200 * (1 + 2 * 4 + 1 + 1)
    fam = body['families']
    assert fam['wmv_lc_closing']['executor'] == 'lcexec' and fam['g_v18']['executor'] == 'plant'
    assert fam['wmv_lc_closing']['lcexec_parity']['id'] == 'S-LCEXEC-SMOKE' and fam['g_v18']['lcexec_parity'] is None
    assert fam['wmv_lc_closing']['n2_parity'] is None and fam['g_v18']['n2_parity']['identical']
    jobs = [j for g in body['groups'] for j in g['jobs']]
    assert [j['arm'] for j in jobs[:7]] == ['u0', 'u0', 'wall_g_solo2', 'wall_h_solo2', 'inc_core', 'wmv_lc_closing',
                                            'g_v18']
    with pytest.raises(ValueError, match='prior root'):
        V.plan(tmp_path / 'run2', regimeu0_root=reg_root, arms_path=arms, cells=cells, cap=cap, registration_md=reg,
               prior_roots=[root], identity=False)
    # ---- fake physics through regimeu0.worker
    monkeypatch.setenv(md.ENV, 'bound-by-fixture')
    monkeypatch.setenv('COOPMWM_G5_RUNTIME', '1')
    monkeypatch.setattr(v2v_runtime, 'load_checkpoint', lambda *a, **k: (torch.nn.Linear(1, 1), None, None, None))
    monkeypatch.setattr(coop_warmstart, 'state_sha256', lambda m: 'fake')
    monkeypatch.setattr(cfb, 'tick', F.fake_tick)
    episode = F.fake_episode_factory()

    def fake_episode(job, **kw):
        return episode(dict(job, stop=SECONDS), **kw)
    monkeypatch.setattr(accel_plant, 'episode', fake_episode)
    for j in jobs:
        assert ir.charge(root, [(j['id'], j['sim_s'], 'paired')])
        s = V.worker(root, j['id'])
        assert not s['error'], s['error']
        side = json.loads((root / 'jobs' / j['id'] / 'WMVERIFY.json').read_text())
        assert side['executor'] == j['executor']
    assert ir.spent(root) == body['planned_sim_s']
    out = tmp_path / 'analysis'
    rep, ver = V.analyze([root], out)
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
    # per-group fairness (fake per-vehicle records), ALLOWED filter
    fair = ver['fairness']['D3200_p60']
    for a in ('wall_g_solo2', 'wmv_lc_closing', 'g_v18', 'wall_h_solo2'):
        f0 = fair[a]['fairness']['road']['u0']
        assert f0['n'] == (1 if a == 'wall_h_solo2' else 2) and fair[a]['allowed'] == f0['allowed_point']
        assert set(f0['groups']) <= {'lane0', 'lane1', 'lane2', 'hdv', 'cav'} and 'cvar90' in f0
        assert 'wall_g_solo2' in fair[a]['fairness']['road'] or a == 'wall_g_solo2'
    allowed = json.loads((out / 'ANALYSIS_ALLOWED.json').read_text())
    keep = {a for a in ('wall_g_solo2', 'wmv_lc_closing', 'g_v18') if fair[a]['allowed']}
    assert set(allowed['cells']['D3200_p60']) == {'u0'} | keep and allowed['schema'] == rep['schema']
    assert {r['family'] for r in _fullpol_rows(allowed)} == keep
    assert ver['summary']['D3200_p60']['allowed'] == sorted(a for a in ('wmv_lc_closing', 'g_v18') if fair[a]['allowed'])
    assert ver['summary']['D3200_p60']['incumbent_allowed'] == fair['wall_g_solo2']['allowed']
    chk = ver['integrity']
    names = {c['name']: c['ok'] for c in chk['checks']}
    assert names['executor_parity_solo2_core_equals_incumbent'] and names['crn_no_divergence_before_first_activation']
    assert names['perveh_present_every_episode']
    assert names['ledger_within_cap_no_uncharged_jobs'] and names['inputs_u0_identical_and_equal_regimeu0']
    assert names['identity_u0_989599'] is False             # the fake plant cannot reproduce 53,269.33 veh-s
    assert chk['lane_receipts']['wmv_lc_closing']['sessions'] > 0
    assert ver['families']['wmv_lc_closing']['template']['schema'] == 'lcexec-free-program-v1'
    res = [json.loads(x) for x in (out / 'RESULTS.jsonl').read_text().splitlines()]
    assert {r['arm'] for r in res} == {'u0', 'wall_g_solo2', 'wall_h_solo2', 'wmv_lc_closing', 'g_v18'}
    assert all('J_300_1200_vs' in r for r in res)
    with pytest.raises(FileExistsError):
        V.analyze([root], out)


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


def test_wmsearch_root_converts_to_regimeu0_schema(tmp_path):
    """S-WMSEARCH v2 (wmsearch_u0) roots -> the same ANALYSIS schema, arms renamed to spec names, inc ->
    wall_g_solo2, the banned solo2 arm never a family; strategy - incumbent contrasts agree with analyze_rows."""
    from tools.wzprop import wmsearch_u0 as WS
    a = dict(inc(), name='g_v18', roles=[dict(SS.INCUMBENT['roles'][0], v_w=18.0)])
    b = next(s for s in SS.seeds() if s['name'] == 'g2_stagger40')
    specs = dict(u0=None, inc=inc(), solo2=solo2(), wm_0=a, str_0=b)
    root = tmp_path / 'ws'
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
    (root / 'PLAN.json').write_text(json.dumps(dict(schema='wmc.wmsearch_u0.v2.plan', groups=groups, specs=specs,
                                                    plan_sha256='w', inputs={'u0.pt': dict(sha256='u0')})))
    (root / 'REGISTRATION.json').write_text(json.dumps(dict(cap_sim_s=10 ** 6)))
    rep, ver = V.analyze([root], tmp_path / 'out')
    assert set(rep['cells']['D3200_p60']) == {'u0', 'wall_g_solo2', 'g_v18', 'g2_stagger40'}
    ref = WS.analyze_rows(rows, specs, n_boot=200)['arms']
    for arm, name in (('wm_0', 'g_v18'), ('str_0', 'g2_stagger40')):
        assert ver['vs_incumbent']['D3200_p60'][name]['mean_diff'] == pytest.approx(ref[arm]['vs_inc']['mean'])
        assert ver['families'][name]['source_arm'] == arm and ver['families'][name]['executor'] == 'plant'
    assert {r['family'] for r in _fullpol_rows(rep)} == {'wall_g_solo2', 'g_v18', 'g2_stagger40'}
    assert 'wall_h_solo2' in ver['vs_incumbent']['D3200_p60']
    names = {c['name']: c['ok'] for c in ver['integrity']['checks']}
    assert names['inputs_u0_identical_and_equal_regimeu0'] and names['executor_parity_solo2_core_equals_incumbent']
    assert 'identity_u0_989599' not in names


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


def test_holdpin_plan_and_analysis(tmp_path):
    """S-HOLDPIN: plan = identity + 72 unpinned solo2 episodes on the S-REGIMEU0 worlds (87,600 sim-s, executor n2);
    analysis pairs them with REUSED regimeu0-format u0 / pinned summaries (CRN, diffs, lane leaving)."""
    from tools.wzprop import holdpin as HP
    reg_root = tmp_path / 'regimeu0'
    (reg_root / 'inputs').mkdir(parents=True)
    (reg_root / 'inputs' / 'CONFIG.json').write_text('{}')
    (reg_root / 'inputs' / 'u0.pt').write_bytes(b'u0')
    (reg_root / 'PLAN.json').write_text(json.dumps(dict(inputs={'u0.pt': dict(sha256=ir.sha_file(
        reg_root / 'inputs' / 'u0.pt'))})))
    text, planned = HP.registration_text('x')
    reg = tmp_path / 'REGISTRATION.md'
    reg.write_text(text, encoding='utf-8')
    root = tmp_path / 'hp'
    body = HP.plan(root, regimeu0_root=reg_root, registration_md=reg)
    jobs = [j for g in body['groups'] for j in g['jobs']]
    assert body['planned_sim_s'] == planned == 87_600 and len(jobs) == 73
    assert {j['executor'] for j in jobs[1:]} == {'n2'} and jobs[0]['id'] == V.IDENTITY_JOB
    assert {(j['world']['cell_key'], j['world']['seed']) for j in jobs[1:]} == {
        (c, s) for c, _d, _p, seeds in HP.CELLS for s in seeds}
    # fake summaries: reused comparators (regimeu0 format) + treatment with the n2 hook's lane receipts
    ref = tmp_path / 'ref'
    (ref / 'jobs').mkdir(parents=True)
    groups = []
    for seed in (989850, 989851):
        w = R.world_of(seed)
        js = [dict(id=f's{seed}_{a}', arm=a, strategy=None if a == 'u0' else a, world=w, sim_s=1200)
              for a in ('u0', 'wall_h_solo2')]
        groups.append(dict(stage='paired', cell=w['cell_key'], jobs=js))
        for j in js:
            hook = R.RuleHook(j['strategy'])
            res = F.run_episode(hook, seed=seed, seconds=SECONDS)
            (ref / 'jobs' / j['id']).mkdir()
            with gzip.open(ref / 'jobs' / j['id'] / 'SUMMARY.json.gz', 'wt', encoding='utf-8') as f:
                json.dump(_summary(hook, res, j), f)
        tj = next(j for j in jobs if j['world']['seed'] == seed and j['world']['cell_key'] == 'D3200_p60')
        hook = VH.make_hook(tj['strategy_spec'], executor='n2')
        res = F.run_episode(hook, seed=seed, seconds=SECONDS)
        (root / 'jobs' / tj['id']).mkdir(parents=True)
        with gzip.open(root / 'jobs' / tj['id'] / 'SUMMARY.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(_summary(hook, res, tj), f)
        (root / 'jobs' / tj['id'] / 'WMVERIFY.json').write_text(json.dumps(dict(receipts=hook.receipts())))
    (ref / 'PLAN.json').write_text(json.dumps(dict(groups=groups)))
    out = HP.analyze([root], [ref])
    c = out['cells']['D3200_p60']
    assert c['n'] == 2 and not c['crn_violations'] and c['same_first_activation_as_pinned'] == 2
    # the fake plant ignores laneChangeMode: unpinned == pinned exactly, so the diff is 0 and the vs-u0 diff equals
    # the pinned effect
    assert c['unpinned_minus_pinned']['mean_diff'] == 0.0
    assert c['unpinned_minus_u0']['mean_diff'] == pytest.approx(c['pinned_minus_u0_same_seeds']['mean_diff'])
    assert c['lane_leaving']['holds'] > 0 and c['lane_leaving']['held_seconds'] > 0
    assert out['integrity']['crn_ok'] and out['cells']['D2400_p60']['n'] == 0
