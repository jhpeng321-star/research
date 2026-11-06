"""WM-C CPU tests (fakes only: StubWM + fake SUMO core; no SUMO, GPU or ssh).

    C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/wzprop/test_wzprop_cpu.py
"""
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tools.wzprop import compare as C
from tools.wzprop import fakes
from tools.wzprop import families as F
from tools.wzprop import geometry as G
from tools.wzprop import n3
from tools.wzprop import pipeline
from tools.wzprop import proposer as PR
from tools.wzprop import runner as RN
from tools.wzprop import verify_plugin as V
from tools.wzprop import wm_backend as W


def moment(seed=989400, pen=0.6, t0=450, sid=None):
    cavs = fakes.fake_cavs(seed, pen, t0)
    return dict(cell='D3200_p60', demand=3200, penetration=pen, seed=seed, window=0, t0=t0, valid=True,
                sid=sid or f'wmc_D3200_p60_s{seed}_t{t0}', inserted_t0=100 + t0, approach_speed=12.0,
                closing_lane_slow=3, n_zone_cavs=len(G.zone_cavs(cavs)), variants={}, union=[],
                dither_ids={}, all_cavs_t0=cavs)


SMALL = dict(n_screen=24, n_synergy=10, refine_rounds=2, n_parents=4, n_children=6, finalists=8)


@pytest.fixture(scope='module')
def P():
    return n3.modules()[0]


@pytest.fixture(scope='module')
def dry(tmp_path_factory):
    out = tmp_path_factory.mktemp('wzprop_dry')
    return pipeline.dry_run(out / 'run', cfg=SMALL)


# ------------------------------------------------------------------------------------ geometry / guard
def test_wm_frame_projection_matches_n1_descriptor():
    r = G.wm_row(dict(id='a', edge='up', lane=0, x=G.X_B, v=10.0))
    assert r['lane'] == 2 and abs(r['s_m'] - 75.44) < 1e-9          # closing lane -> WM 2, drop at s = 75.44
    assert G.wm_row(dict(id='b', edge='up', lane=2, x=1000.0, v=1.0))['lane'] == 0
    assert G.wm_row(dict(id='c', edge='up', lane=1, x=1000.0, v=1.0))['lane'] == 1
    assert G.wm_row(dict(id='d', edge='wz', lane=0, x=3000.0, v=1.0))['lane'] == 1   # up_1 -> wz_0
    assert G.wm_row(dict(id='e', edge='wz', lane=1, x=3000.0, v=1.0))['lane'] == 0   # up_2 -> wz_1


def test_guard_refuses_heldout_eval_and_non_train_seeds():
    for s in (989000, 989200, 989400, 989599, 989700, 989899):
        assert G.check_seed(s) == s
    for s in (989600, 989650, 989699, 964000, 964199, 962123, 977001, 986000, 989100, 989199, 989900, 'x'):
        with pytest.raises(G.ForbiddenSeedError):
            G.check_seed(s)
    for bad in ('/a/holdout/M.json', 'E:/x/HELDOUT.json', '/c/cal_split/m', '/s/sealed/x', '/t/hetdata_truth.json',
                '/v/vType_params.xml', '/j/j4b/x', '/i/inc1/y', '/k/chunk5/z'):
        with pytest.raises(G.ForbiddenPathError):
            G.check_path(bad)
    m = moment(seed=989650)
    with pytest.raises(G.ForbiddenSeedError):
        W.moment_input(m)


# ------------------------------------------------------------------------------------ vendored N3
def test_vendored_pipeline_is_pinned_and_binds_without_edits(tmp_path, P):
    got = n3.verify()
    assert set(got) == {'pg_plan.py', 'pg_analyze.py', 'nm2_core.py', 'population.py'}
    bad = tmp_path / 'v'
    bad.mkdir()
    for name in n3.PINS:
        (bad / name).write_bytes((n3.VENDOR / name).read_bytes())
    (bad / 'pg_plan.py').write_bytes((n3.VENDOR / 'pg_plan.py').read_bytes() + b'\n# drift\n')
    with pytest.raises(n3.VendorDriftError):
        n3.verify(bad)
    root, src = tmp_path / 'root', tmp_path / 'src'
    P2, A = n3.bind(root, natives_root=src, cap=1234.0)
    assert P2 is P and A.P is P
    assert P.RUNS == root.resolve() / 'runs' and P.CAP == 1234.0
    assert P.native_dir(3200, 0.6, 989400) == src.resolve() / 'runs' / 'D3200_p60' / 's989400' / 'native'
    assert P.MAPSPEC == src.resolve() / 'maps' / 'wz3to2' / 'MAP_SPEC.json'
    n3.bind(root)
    assert P.native_dir(3200, 0.6, 989400) == root.resolve() / 'runs' / 'D3200_p60' / 's989400' / 'native'
    assert (P.X_B, P.ZONE, P.PAIR_DX, P.WALL_SPREAD) == (G.X_B, G.ZONE, G.PAIR_DX, G.WALL_SPREAD)


def test_member_specs_and_structured_variants_equal_pg_plan(P):
    m = moment()
    snap = dict(cavs=m['all_cavs_t0'])
    pair, wall = P.pick_pair(snap), P.pick_wall(snap)
    moves = F.structured_moves(m['all_cavs_t0'])
    assert set(moves) == set(F.STRUCTURED)
    for vid, mv in moves.items():
        fam, prm = P.VARIANTS[vid]
        assert fam == mv['family'] and prm == mv['params']
        mem = pair if fam == 'closing_pair' else wall
        ref = [P.member_spec(c, prm, anchor=pair[0]['id'] if fam == 'closing_pair' else None) for c in mem]
        if fam == 'abreast_wall':
            ref = [P.member_spec(c, prm) for c in mem]
        assert mv['members'] == ref
    assert F.REL == P.REL and F.EARLY_RELEASE_M == G.X_B - P.VARIANTS['pair_t'][1]['anchor_release_x']


# ------------------------------------------------------------------------------------ candidates
def test_candidate_geometry_grid_and_keys():
    cavs = moment()['all_cavs_t0']
    pool = F.generate(cavs)
    n_grid = len(list(F.grid_points(F.DEFAULT_GRID)))
    sets = F.member_sets(cavs)
    assert len(pool) == n_grid * (len(sets['abreast_wall']) + len(sets['closing_pair']))
    assert len({p['key'] for p in pool}) == len(pool)
    by = {c['id']: c for c in cavs}
    for mv in pool:
        xs = [m['x'] for m in mv['members']]
        assert all(m['role'] == 'hold' for m in mv['members'])
        assert all(by[c['id']]['edge'] == 'up' and G.ZONE[0] <= c['x'] <= G.ZONE[1] for c in mv['members'])
        if mv['family'] == 'abreast_wall':
            assert sorted(m['lane'] for m in mv['members']) == [0, 1, 2] and max(xs) - min(xs) <= G.WALL_SPREAD
        else:
            assert [m['lane'] for m in mv['members']] == [0, 1] and max(xs) - min(xs) <= G.PAIR_DX
            if mv['knob']['release'] == 'early':
                assert all(m['params']['anchor'] == mv['members'][0]['id'] for m in mv['members'])
    mv = next(p for p in pool if p['family'] == 'abreast_wall')
    for nb in F.neighbors(mv, cavs):
        assert nb['key'] != mv['key'] and nb['family'] == mv['family']
        changed = [k for k in mv['knob'] if nb['knob'][k] != mv['knob'][k]]
        assert len(changed) + len(set(nb['member_ids']) ^ set(mv['member_ids'])) // 2 == 1


def test_open_loop_profile_follows_hold_law():
    c = dict(id='m', edge='up', lane=0, x=1000.0, v=25.0)
    spec = F.member_spec(c, dict(v_w=15.0, hold_s=30, dmax=1.5, start_s=5))
    prof = F.open_loop_profile([spec], 300)['m']
    assert prof[:5] == [None] * 5 and len(prof) == 35
    assert prof[5] == -1.5 and min(a for a in prof if a is not None) >= -1.5


# ------------------------------------------------------------------------------------ WM contract
def test_stub_wm_draws_are_crn_consistent_across_calls():
    wm = W.StubWM()
    m = moment()
    st = wm.observe(W.moment_input(m))
    pool = F.generate(m['all_cavs_t0'])
    a = [W.action('base', 'base', None, 'base', [], 300)] + [
        W.action(f'g{i}', 'group', p['family'], p['key'], p['members'], 300) for i, p in enumerate(pool[:3])]
    b = [W.action('zz', 'group', pool[1]['family'], pool[1]['key'], pool[1]['members'], 300),
         W.action('b2', 'base', None, 'base', [], 300)]
    ra, rb = wm.evaluate(st, a, 4, 300, 7), wm.evaluate(st, b, 4, 300, 7)
    assert ra['g1'] == rb['zz'] and ra['base'] == rb['b2']
    assert wm.evaluate(st, a, 4, 300, 8)['base'] != ra['base']


def test_normalize_accepts_wm_formats_and_rejects_contract_violations():
    acts = [dict(id='base'), dict(id='g0')]
    exp = {'base': [1.0, 2.0], 'g0': [0.5, 1.5]}
    assert W.normalize(exp, acts, 2, 300) == exp
    assert W.normalize({'300': {k: [{'team_delay': x} for x in v] for k, v in exp.items()}}, acts, 2, 300) == exp
    assert W.normalize({'scores': {'300': exp}}, acts, 2, 300) == exp
    with pytest.raises(W.WMContractError):
        W.normalize({'base': [1.0, 2.0]}, acts, 2, 300)
    with pytest.raises(W.WMContractError):
        W.normalize({'base': [1.0], 'g0': [1.0]}, acts, 2, 300)
    with pytest.raises(W.WMContractError):
        W.normalize({'base': [1.0, float('nan')], 'g0': [1.0, 2.0]}, acts, 2, 300)


def test_wma_binding_from_ready_file(tmp_path, monkeypatch):
    mod = tmp_path / 'fake_wma_api.py'
    mod.write_text(
        "CALLS = []\n"
        "def load(scale=1.0):\n    return dict(scale=scale)\n"
        "def observe(h, moment):\n    CALLS.append(sorted(moment))\n    return dict(moment_id=moment['moment_id'], wm_rows=[])\n"
        "def evaluate(h, state, actions, K, H, seed):\n"
        "    return {str(H): {a['id']: [{'team_delay': h['scale'] * (100.0 - 10.0 * len(a['members']) + k)}"
        " for k in range(K)] for a in actions}}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    ready = tmp_path / 'WMA_READY.json'
    ready.write_text(json.dumps(dict(api=dict(load='fake_wma_api:load', observe='fake_wma_api:observe',
                                              evaluate='fake_wma_api:evaluate'), load_kwargs=dict(scale=2.0),
                                     commits=['abc'])))
    wm = W.load_backend(f'wma:{ready}')
    assert wm.meta['ready_sha256'] and wm.meta['wma_commits'] == ['abc'] and wm.handle == dict(scale=2.0)
    row = PR.propose_moment(wm, moment(), PR.config(dict(SMALL, k=3)))
    assert len(row['variants']) == 3 and row['backend']['backend'] == 'wma'
    import fake_wma_api
    keys = fake_wma_api.CALLS[0]
    assert not {'sid', 'cell', 'demand', 'penetration', 'seed', 'approach_speed', 'closing_lane_slow'} & set(keys)
    monkeypatch.setenv(W.READY_ENV, str(tmp_path / 'missing.json'))
    monkeypatch.chdir(tmp_path / '..')
    if W.find_ready() is None:
        with pytest.raises(FileNotFoundError):
            W.load_backend('auto')
    ready2 = tmp_path / 'WMA_READY_noapi.json'
    ready2.write_text(json.dumps(dict(commits=[])))
    with pytest.raises(ValueError):
        W.WMABackend(ready2)
    wm2 = W.WMABackend(ready2, api=dict(evaluate='fake_wma_api:evaluate'))
    assert wm2.handle is None


# ------------------------------------------------------------------------------------ proposer
def test_proposer_finds_the_planted_optimum_and_beats_structured(P):
    m = moment()
    wm = W.StubWM()
    cfg = PR.config(dict(k=5, families=['abreast_wall', 'closing_pair']))
    row = PR.propose_moment(wm, m, cfg)
    assert not row['abstain'] and len(row['variants']) == 5
    top = row['variants']['wm0']
    assert top['family'] == 'abreast_wall'
    assert (top['params']['v_w'], top['params']['hold_s'], top['params'].get('start_s')) == (18.0, 120, 10)
    world = fakes.PlantedWorld()
    best_struct = max(world.effect(mv['members']) for mv in F.structured_moves(m['all_cavs_t0']).values())
    assert world.effect(top['members']) > 2 * best_struct
    per = {}
    for v in row['variants'].values():
        per[F.set_key(v['family'], [x['id'] for x in v['members']])] = per.get(
            F.set_key(v['family'], [x['id'] for x in v['members']]), 0) + 1
        assert v['wm']['M']['lcb'] <= v['wm']['M']['mean']
        c = {x['id']: x for x in m['all_cavs_t0']}
        assert v['members'] == [P.member_spec(c[x['id']], v['params'], anchor=x['params'].get('anchor'))
                                for x in v['members']]
    assert max(per.values()) <= cfg['max_per_set']
    ranks = [v['wm']['score'] for v in row['variants'].values()]
    assert ranks == sorted(ranks, reverse=True)
    assert row['ledger']['rollouts'] > 0 and row['n_evaluated'] >= cfg['n_synergy']
    assert PR.propose_moment(wm, m, cfg)['variants'] == row['variants']          # deterministic


def test_proposer_abstains_only_when_asked():
    m = moment()
    null = W.StubWM(world=fakes.PlantedWorld(group_gain=0.0, closing_solo=0.0, through_cost=90.0))
    row = PR.propose_moment(null, m, PR.config(dict(SMALL, abstain_below=0.0)))
    assert row['abstain'] and row['variants'] == {} and row['best_score'] <= 0
    row = PR.propose_moment(null, m, PR.config(dict(SMALL, k=2)))
    assert not row['abstain'] and len(row['variants']) == 2


def test_uncertainty_score_prefers_reliable_candidates():
    s_hi = PR.stat([120, -80, 280, -50], 1.645)
    s_lo = PR.stat([60, 55, 65, 58], 1.645)
    assert s_hi['mean'] > s_lo['mean'] and s_lo['lcb'] > s_hi['lcb']
    assert PR.score_of(dict(T=s_hi, M=s_lo), 'lcb')[0] == s_lo['lcb']
    assert PR.score_of(dict(T=s_hi), 'mean')[0] == s_hi['mean']


def test_n9_harmonize_rows_when_module_available(P, monkeypatch):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Param:
        key: str
        lo: float
        hi: float

    class Spec:
        cap = (2, 6)

        def param(self, k):
            return dict(row_speed_mps=Param(k, 15.0, 22.0), row_spacing_m=Param(k, 30.0, 90.0))[k]

        def constant(self, k):
            return dict(row_span_m=60.0, zone_far_m=1800.0, release_m=400.0)[k]

    nf = types.SimpleNamespace(SPECS={'harmonize_rows': Spec()}, DEPTH=(0.3, 1.2), RULE='n9-test',
                               FAMILY_NAMES=('platoon_headway', 'harmonize_rows', 'zipper_order'))
    st = F.n9_status(nf)
    assert st['executable'] == ['harmonize_rows'] and 'zipper_order' in st['pending_executor']
    cavs = moment()['all_cavs_t0']
    pool = F.generate(cavs, families=('harmonize_rows',), nf=nf)
    assert pool
    c = {x['id']: x for x in cavs}
    for mv in pool[:20]:
        assert mv['params']['release_x'] == G.X_B - 400.0 and mv['params']['amax'] == mv['params']['dmax']
        assert mv['members'] == [P.member_spec(c[x['id']], mv['params']) for x in mv['members']]
    monkeypatch.setitem(sys.modules, 'tools.coopmwm.nm_families', nf)
    import tools.coopmwm
    monkeypatch.setattr(tools.coopmwm, 'nm_families', nf, raising=False)
    row = PR.propose_moment(W.StubWM(), moment(), PR.config(dict(SMALL, k=3)))
    assert 'harmonize_rows' in row['families_searched'] and row['n9']['available']


# ------------------------------------------------------------------------------------ plug-in
def test_plugin_refuses_stub_tampered_specs_and_heldout(tmp_path, dry):
    ref = Path(dry['ref'])
    geo = ref / 'MOMENTS_GEOMETRY.json'
    root = tmp_path / 'w'
    V.init(root, cap=1e6, natives_root=ref)
    with pytest.raises(ValueError, match='StubWM'):
        V.build_moments(root, geo, dry['proposals'])
    rows = [json.loads(x) for x in Path(dry['proposals']).read_text().splitlines()]
    v = next(iter(rows[0]['variants'].values()))
    v['members'][0]['params']['v_w'] = 9.0
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    root2 = tmp_path / 'w2'
    V.init(root2, cap=1e6, natives_root=ref)
    with pytest.raises(ValueError, match='member_spec'):
        V.build_moments(root2, geo, bad, allow_stub=True)
    ms = json.loads(geo.read_text())
    ms[0]['seed'] = 989601
    hgeo = tmp_path / 'G.json'
    hgeo.write_text(json.dumps(ms))
    root3 = tmp_path / 'w3'
    V.init(root3, cap=1e6, natives_root=ref)
    with pytest.raises(G.ForbiddenSeedError):
        V.build_moments(root3, hgeo, dry['proposals'], allow_stub=True)
    with pytest.raises(FileExistsError):
        V.init(root3, cap=1e6, natives_root=ref)


def test_dither_targets_reused_unless_a_member_is_one(tmp_path, dry, P):
    ref = Path(dry['ref'])
    geo = json.loads((ref / 'MOMENTS_GEOMETRY.json').read_text())
    m = next(x for x in geo if x['valid'])
    target = m['dither_ids']['1']
    c = next(x for x in m['all_cavs_t0'] if x['id'] == target)
    if c['edge'] != 'up':
        pytest.skip('dither target not on up')
    other = next(x for x in m['all_cavs_t0'] if x['edge'] == 'up' and x['id'] != target)
    mv = F.make_move('closing_pair', [dict(c, lane=0), dict(other, lane=1)], dict(v_w=18.0, hold_s=120, dmax=1.5,
                                                                                  start_s=0, release='drop'))
    if (c['lane'], other['lane']) != (0, 1):
        mv = F.make_move('abreast_wall', [c, other], dict(v_w=18.0, hold_s=120, dmax=1.5, start_s=0, release='drop'))
    prop = dict(sid=m['sid'], t0=m['t0'], abstain=False, backend=dict(backend='test'),
                variants={'wm0': dict(F.to_variant(mv), wm=None)})
    pf = tmp_path / 'p.jsonl'
    pf.write_text(json.dumps(prop) + '\n')
    root = tmp_path / 'd'
    V.init(root, cap=1e6, natives_root=ref)
    rec = V.build_moments(root, ref / 'MOMENTS_GEOMETRY.json', pf)
    out = json.loads((root / 'MOMENTS.json').read_text())
    assert rec['counts']['dither_recomputed'] == 1 and out[0]['dither_source'] == 'recomputed'
    assert target not in out[0]['dither_ids'].values()
    assert not set(out[0]['dither_ids'].values()) & set(out[0]['union'])


def test_end_to_end_dry_run_same_funnel_and_rows(dry):
    from tools.coopmwm.group_lane_bank import MINED_BINDING_FIELDS
    wm_root, ref_root = Path(dry['wm']), Path(dry['ref'])
    assert dry['wm_funnel']['wm']['admitted_variants'] >= 1
    assert set(dry['ref_funnel']) == {'structured', 'random'}
    rows = [json.loads(x) for x in (wm_root / 'results' / 'WMC_MOVES_B.jsonl').read_text().splitlines()]
    assert rows and all(r['proposer'] == 'wm' and r['schema'] == 'coopmwm-mined-move-v1' for r in rows)
    for r in rows:
        assert all(k in r for k in MINED_BINDING_FIELDS)
        assert r['n_reps'] == 3 and len(r['profile']) == len(r['members'])
        assert all(len(p) == r['duration_s'] for p in r['profile'])
        assert r['verification_id'].startswith('wmcv_') and r['wm_prediction']['backend'] == 'stub'
        G.check_seed(r['world']['seed'])
    summ = json.loads((wm_root / 'results' / 'WMC_SUMMARY.json').read_text())
    adm = [v for v in summ['verdicts'] if v['admitted']]
    for v in adm:
        reps = v['reps']
        assert len(reps) == 3 and all(r['T_lb'] > 0 and r['S_lb'] > 0 and r['S'] >= 1 and r['safe'] for r in reps.values())
        assert reps['1']['T_ext'] > 0
    # same moments, same natives (read-only) and the WM root never wrote into the reference root
    wm_m = json.loads((wm_root / 'MOMENTS.json').read_text())
    ref_m = json.loads((ref_root / 'MOMENTS.json').read_text())
    assert {m['sid'] for m in wm_m} == {m['sid'] for m in ref_m}
    job = json.loads(next((wm_root / 'runs').rglob('JOB.json')).read_text())
    assert str(ref_root) in job['native_dir'] and str(wm_root) in job['out_dir']
    # the standalone cost model equals the WM root's own ledger (WM-only root)
    cmp = dry['compare']
    assert cmp['metrics']['wm']['sim_s'] == pytest.approx(RN.charged(wm_root))
    assert set(cmp['metrics']) == {'wm', 'structured', 'random'} and cmp['n_common_moments'] == len(wm_m)
    assert cmp['metrics']['wm']['n_candidates'] == cmp['metrics']['structured']['n_candidates']     # matched k
    for arm in cmp['metrics'].values():
        assert arm['verified_per_ksim'] == pytest.approx(1000 * arm['admitted_variants'] / arm['sim_s'], abs=1e-4)
    assert cmp['paired']['wm_minus_structured'] is not None and cmp['wm_diagnostics']['n'] > 0


def test_runner_precharges_refuses_over_cap_and_skips_existing(tmp_path):
    root = tmp_path / 'r'
    (root / 'runs').mkdir(parents=True)
    jobs = [dict(kind='native', job_id=f'n{i}', seed=989400 + i, penetration=0.6, t_end=900, save_times=[450],
                 out_dir=str(root / 'runs' / f'n{i}')) for i in range(3)]
    res = RN.run(jobs, root, 2000.0, core='fake', inprocess=True)
    assert res['launched'] == 2 and res['refused'] == 1 and RN.charged(root) == 1800.0
    assert (root / 'runs' / 'n2' / 'REFUSED_BUDGET.json').exists()
    res = RN.run(jobs[:2], root, 1e9, core='fake', inprocess=True)
    assert res['skipped_existing'] == 2 and res['launched'] == 0
    with pytest.raises(ValueError, match='outside the run root'):
        RN.run([dict(jobs[0], out_dir=str(tmp_path / 'elsewhere'))], root, 1e9, core='fake', inprocess=True)
    with pytest.raises(ValueError, match='fake core'):
        RN.run([dict(jobs[0], job_id='x', out_dir=str(root / 'runs' / 'x'))], root, 1e9, core='nm2', inprocess=True)


def test_cost_model_by_stage():
    def vd(stage, n=3, solos=True):
        return dict(stage=stage, members=['a'] * n, S_r0=1.0 if solos else None)
    assert C.moment_cost([vd('screen', solos=False)] * 5) == 300 * 6
    assert C.moment_cost([vd('solos'), vd('screen', solos=False)]) == 300 * 3 + 900
    assert C.moment_cost([vd('r1', 2)]) == 300 * 2 + 600 + (300 + 300 * 3)
    assert C.moment_cost([vd('admitted', 3)]) == 300 * 2 + 900 + (300 + 1200) + (420 + 1200 + 120)
    assert C.wilson(0, 0) == [None, None] and C.wilson(3, 6)[0] < 0.5 < C.wilson(3, 6)[1]
    assert C.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_compare_fixed_budget_prefix_and_arm_suffix(tmp_path, dry):
    ref = Path(dry['ref']) / 'results' / 'WMC_SUMMARY.json'
    cmp = C.compare([ref, ref], cap=1.0)
    assert {'structured', 'random', 'structured@1', 'random@1'} <= set(cmp['metrics'])
    assert all(v['moments'] == 0 for v in cmp['fixed_budget'].values())
    full = C.compare([ref], arms=['structured'])
    assert full['fixed_budget']['structured']['sim_s'] == pytest.approx(full['metrics']['structured']['sim_s'])


# ------------------------------------------------------------------------------------ W3 wall-fit
def test_wallfit_labels_load_only_under_the_wz_map_scope(tmp_path, dry, monkeypatch):
    from tools.wallfit import dataset, guard
    from tools.wzprop import wallfit_bridge as WB
    ref, wmr = Path(dry['ref']), Path(dry['wm'])
    rec = WB.build_labels([ref / 'results' / 'WMC_SUMMARY.json', wmr / 'results' / 'WMC_SUMMARY.json'],
                          [ref / 'MOMENTS.json', wmr / 'MOMENTS.json'], tmp_path / 'lab')
    assert rec['pairs'] > 0 and rec['pairs_admitted'] >= 1 and rec['pairs_with_wm'] >= 1
    lab = tmp_path / 'lab' / 'LABELS.jsonl'
    monkeypatch.delenv(guard.MAP_ENV, raising=False)
    with pytest.raises(guard.ForbiddenSeedError):
        dataset.load_pairs([lab])
    monkeypatch.setenv(guard.MAP_ENV, 'wz3to2')
    pairs = dataset.load_pairs([lab])
    assert pairs and all(989200 <= p['world_seed'] <= 989599 for p in pairs)
    adm = [p for p in pairs if p['labels']['admitted_any']]
    assert adm and all(p['labels']['n_measured'] == 3 for p in adm)
    with pytest.raises(guard.ForbiddenSeedError):
        guard.check_seed(989650)
    with pytest.raises(guard.ForbiddenSeedError):
        guard.check_seed(986000)                       # one process = one map
    monkeypatch.setenv(guard.MAP_ENV, 'nope')
    with pytest.raises(guard.ForbiddenSeedError):
        guard.check_seed(989300)
    monkeypatch.delenv(guard.MAP_ENV)
    assert guard.check_seed(986000) == 986000
    with pytest.raises(guard.ForbiddenSeedError):
        guard.check_seed(989650)


def test_wallfit_head_rerank_is_optional_and_wm_ordered_at_init():
    from tools.wallfit import head as hd
    m = moment()
    wm = W.StubWM()
    cfg = PR.config(dict(SMALL, k=4))
    plain = PR.propose_moment(wm, m, cfg)
    bundle = hd.untrained()
    ranked = PR.propose_moment(wm, m, cfg, head=bundle)
    assert set(plain) == set(ranked) and len(ranked['variants']) == 4
    heads = [v['wm']['head']['score'] for v in ranked['variants'].values()]
    syn = [v['wm']['S']['mean'] for v in ranked['variants'].values()]
    assert heads == sorted(heads, reverse=True)
    assert np.argsort(-np.asarray(heads)).tolist() == np.argsort(-np.asarray(syn), kind='stable').tolist()
    assert all('head' not in v['wm'] for v in plain['variants'].values())
    assert math.isfinite(heads[0])


def test_zero_sumo_rank_check_and_labels_with_wm_scores(tmp_path, dry):
    from tools.wzprop import wallfit_bridge as WB
    ref = Path(dry['ref'])
    sc = tmp_path / 'SCORES.jsonl'
    rec = PR.score_moments(W.StubWM(), ref / 'MOMENTS.json', sc, K=4)
    n_var = sum(len(m['variants']) for m in json.loads((ref / 'MOMENTS.json').read_text()))
    assert rec['n_variants'] == n_var and rec['new_sumo_sim_s'] == 0
    res = C.rank_check([ref / 'results' / 'WMC_SUMMARY.json'], sc)
    assert res['n_scored'] == res['n_verdicts'] == n_var
    assert res['all']['spearman_T'] > 0.5 and set(res['per_arm']) == {'structured', 'random'}
    lab = WB.build_labels([ref / 'results' / 'WMC_SUMMARY.json'], [ref / 'MOMENTS.json'], tmp_path / 'lab',
                          wm_scores=[sc])
    assert lab['pairs_with_wm'] == lab['pairs'] > 0


def test_cli_propose_and_compare(tmp_path, dry):
    from tools.wzprop.__main__ import main
    geo = Path(dry['ref']) / 'MOMENTS_GEOMETRY.json'
    out = tmp_path / 'P.jsonl'
    assert main(['propose', '--moments', str(geo), '--out', str(out), '--wm', 'stub', '--k', '2', '--limit', '1',
                 '--config', json.dumps(SMALL)]) in (None, 0)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 1 and len(rows[0]['variants']) == 2
    assert json.loads((tmp_path / 'P_RECEIPT.json').read_text())['new_sumo_sim_s'] == 0
    with pytest.raises(FileExistsError):
        main(['propose', '--moments', str(geo), '--out', str(out), '--wm', 'stub'])
    cmp_out = tmp_path / 'C.json'
    main(['compare', '--summary', str(Path(dry['ref']) / 'results' / 'WMC_SUMMARY.json'), '--summary',
          str(Path(dry['wm']) / 'results' / 'WMC_SUMMARY.json'), '--out', str(cmp_out)])
    assert set(json.loads(cmp_out.read_text())['metrics']) == {'wm', 'structured', 'random'}


def test_compare_excludes_budget_stranded_moments(tmp_path):
    def vd(sid, arm, stage, adm=False):
        return dict(sid=sid, seed=989300, arm=arm, family='abreast_wall' if arm == 'B' else 'R_random',
                    variant='v', stage=stage, admitted=adm, members=['a', 'b'], S_r0=None, T_r0=1.0)
    verdicts = [vd('m1', 'B', 'admitted', True), vd('m1', 'R', 'screen'), vd('m2', 'B', 'screen_missing'),
                vd('m2', 'R', 'screen_missing'), vd('m3', 'B', 'r1'), vd('m3', 'R', 'r2_missing')]
    p = tmp_path / 'S.json'
    p.write_text(json.dumps(dict(verdicts=verdicts)))
    res = C.compare([p])
    assert res['incomplete_moments'] == ['m2', 'm3'] and res['n_common_moments'] == 1
    assert res['metrics']['structured']['admitted_variants'] == 1
    assert C.compare([p], exclude_incomplete=False)['n_common_moments'] == 3


def _write_tape(path, seed=989300, cell='D3200_p60'):
    # 3 steps at t = 0.5, 1.0, 1.5; vehicles: c1 (CAV, up lane 1), h_near (HDV 50 m ahead same lane),
    # h_far (HDV 500 m away), h_cl (HDV closing lane next to c1), c1 appears at 0.5, others later
    rows = [(0, 'c1', 'up', 1, 1000.0), (1, 'c1', 'up', 1, 1010.0), (1, 'h_near', 'up', 1, 1050.0),
            (2, 'c1', 'up', 1, 1020.0), (2, 'h_near', 'up', 1, 1070.0), (2, 'h_far', 'up', 2, 1520.0),
            (2, 'h_cl', 'up', 0, 990.0), (1, 'h_cl', 'up', 0, 980.0)]
    ids = ['c1', 'h_near', 'h_far', 'h_cl']
    edges = ['up', ':n1_0', 'wz']
    meta = dict(schema='wzqual_native_tape/1', seed=seed, cell=cell, t_end=1.5, split='WMQUAL')
    np.savez_compressed(path, time_s=np.array([0.5, 1.0, 1.5]), step=np.array([r[0] for r in rows], np.int32),
                        vid=np.array([ids.index(r[1]) for r in rows], np.int32),
                        edge=np.array([edges.index(r[2]) for r in rows], np.int16),
                        lane=np.array([r[3] for r in rows], np.int16), x=np.array([r[4] for r in rows]),
                        v=np.full(len(rows), 20.0), a=np.zeros(len(rows)), backlog=np.zeros(3, np.int32),
                        inserted=np.zeros(3, np.int32), ids=np.array(ids), is_cav=np.array([True, False, False, False]),
                        depart_s=np.zeros(4), edges=np.array(edges), meta_json=np.array(json.dumps(meta)))


def test_tape_enrichment_keeps_only_observable_content(tmp_path):
    from tools.wzprop import tapes
    root = tmp_path / 'tapes'
    p = root / 'runs' / 'D3200_p60' / 's989300' / 'prefix'
    p.mkdir(parents=True)
    _write_tape(p / 'TAPE.npz')
    m = dict(cell='D3200_p60', demand=3200, penetration=0.6, seed=989300, t0=1.5, valid=True, sid='x',
             all_cavs_t0=[dict(id='c1', edge='up', lane=1, x=1020.0, v=20.0)], variants={})
    src = tmp_path / 'M.json'
    src.write_text(json.dumps([m]))
    rep = tapes.enrich(src, root, tmp_path / 'O.json')
    assert rep['enriched'] == 1 and not rep['identity_fail']
    o = json.loads((tmp_path / 'O.json').read_text())[0]['observed']
    assert {v['id'] for v in o['vehicles']} == {'h_near', 'h_cl'}          # h_far: 500 m away, never reported
    assert o['coverage']['undetected'] == 1
    # all entries are inside bin 0 (t0 - 10 s, t0]: through c1 (0.5), h_near (1.0), h_far (1.5); closing h_cl (1.0)
    assert o['counter'][0][0] == 3 and o['counter'][1][0] == 1 and sum(o['counter'][0]) == 3
    mi = W.moment_input(json.loads((tmp_path / 'O.json').read_text())[0])
    assert {v['id'] for v in mi['vehicles']} == {'h_near', 'h_cl'} and mi['counter'] == o['counter']
    m2 = dict(m, all_cavs_t0=[dict(id='c1', edge='up', lane=1, x=1030.0, v=20.0)])
    src.write_text(json.dumps([m2]))
    rep = tapes.enrich(src, root, tmp_path / 'O2.json')
    assert rep['enriched'] == 0 and rep['identity_fail'][0]['max_dx'] == pytest.approx(10.0)
    m3 = dict(m, seed=989650)
    src.write_text(json.dumps([m3]))
    with pytest.raises(G.ForbiddenSeedError):
        tapes.enrich(src, root, tmp_path / 'O3.json')


def test_rank_metrics_perfect_and_reversed_rankers(tmp_path):
    from tools.wzprop import rankcheck as RC
    verdicts, scores = [], []
    rng = np.random.default_rng(0)
    for w in range(6):
        for mo in range(2):
            sid = f'pg_s{989300 + w}_t{mo}'
            for i in range(5):
                T = float(rng.normal(0, 100))
                S = T - 50.0
                adm = i == 0 and mo == 0
                if adm:
                    T, S = 900.0, 400.0
                verdicts.append(dict(sid=sid, seed=989300 + w, arm='B', family='abreast_wall', variant=f'v{i}',
                                     stage='admitted' if adm else 'solos', admitted=adm, members=['a'],
                                     T_r0=T, T_lb_r0=T - 10, S_r0=S, S_lb_r0=S - 10))
                st = lambda x: dict(mean=x, lcb=x - 1.0, sd=1.0, p_pos=1.0, draws=[x])  # noqa: E731
                scores.append(dict(sid=sid, variant=f'v{i}', wm=dict(T=st(T), S=st(S), M=st(min(T, S)))))
    sp = tmp_path / 'S.json'
    sp.write_text(json.dumps(dict(verdicts=verdicts)))
    sc = tmp_path / 'SC.jsonl'
    sc.write_text(''.join(json.dumps(s) + '\n' for s in scores))
    res = RC.run([sp], sc, label='oracle')
    assert res['spearman_T']['value'] == 1.0 and res['spearman_S']['value'] == 1.0
    assert res['precision_T_pos']['value'] == 1.0 and res['top1_T_pos']['value'] == 1.0
    assert res['admitted_hit_at_k']['M_lcb']['1']['value'] == 1.0
    assert res['admitted_hit_random']['1']['value'] == pytest.approx(0.2)
    assert res['spearman_T']['ci95'] is not None and res['n_moments'] == 12
    rev = [dict(s, wm={k: dict(v, mean=-v['mean'], lcb=-v['lcb']) for k, v in s['wm'].items()}) for s in scores]
    sc2 = tmp_path / 'SC2.jsonl'
    sc2.write_text(''.join(json.dumps(s) + '\n' for s in rev))
    res2 = RC.run([sp], sc2, label='reversed')
    assert res2['spearman_T']['value'] == -1.0 and res2['admitted_hit_at_k']['T_mean']['1']['value'] == 0.0
    assert '| Spearman' in RC.markdown(res2)


# ------------------------------------------------------------------------------------ LCMOVES families
def _lc_moment():
    m = moment()
    up = [c for c in m['all_cavs_t0'] if c['edge'] == 'up']
    m['observed'] = dict(vehicles=[dict(id=f'h{i}', edge='up', lane=c['lane'], x=c['x'] + 15.0, v=c['v'], is_cav=False)
                                   for i, c in enumerate(up)], counter=[[1] * 30, [1] * 30])
    return m


def test_lc_families_are_the_lcmoves_builders():
    from tools.wzprop import families_lc as FL
    FL.verify()
    m = _lc_moment()
    moves = FL.lc_moves(m)
    assert moves and {mv['family'] for mv in moves} <= set(FL.LC_FAMILIES)
    ids = {c['id'] for c in m['all_cavs_t0'] if c['edge'] == 'up'}
    for mv in moves:
        assert all(mm['role'] == 'prog' and mm['id'] in ids for mm in mv['members'])
        FL.validate_prog(m, mv['key'], F.to_variant(mv))
    import lcm_families
    snap = FL.snapshot(m)
    for vid, _fam, _fn in FL.BUILDERS:
        ref = lcm_families.propose(snap, vid)
        mine = next((mv for mv in moves if mv['origin'] == f'{vid}#0'
                     and set(mv['knob']) == {'builder', 'rank', 'origin'}), None)
        assert (ref is None) == (mine is None), vid
        if ref is not None:
            assert mine['members'] == ref, vid
    mv = next(mv for mv in moves if any(mm['params'].get('lc_target') is not None for mm in mv['members']))
    bad = F.to_variant(mv)
    j = next(i for i, mm in enumerate(bad['members']) if mm['params'].get('lc_target') is not None)
    bad['members'][j]['params']['lc_target'] = 7
    with pytest.raises(ValueError, match='registered ranges'):
        FL.validate_prog(m, 'bad', bad)


def test_lc_families_only_with_a_backend_that_executes_programs():
    m = _lc_moment()
    fams = ['wall_fill', 'rolling_wall', 'gap_open_merge', 'lane_redistribute', 'free_program']
    row = PR.propose_moment(W.StubWM(), m, PR.config(dict(SMALL, k=4, families=fams)))
    assert row['variants'] and all(mm['role'] == 'prog' for v in row['variants'].values() for mm in v['members'])
    assert not row['families_unsupported_by_wm']

    class HoldOnly:
        name = 'hold_only'
        meta = dict(backend='hold_only')

        def __init__(self):
            self.inner = W.StubWM()

        def observe(self, mo):
            return self.inner.observe(mo)

        def evaluate(self, st, acts, K, H, seed):
            assert all(mm.get('role', 'hold') == 'hold' for a in acts for mm in a['members'])
            return self.inner.evaluate(st, acts, K, H, seed)
    wm = W._Factory(HoldOnly())
    row = PR.propose_moment(wm, m, PR.config(dict(SMALL, k=3, families=['abreast_wall'] + fams)))
    assert set(row['families_unsupported_by_wm']) == set(fams)
    assert all(v['family'] == 'abreast_wall' for v in row['variants'].values())


def test_regime_benefit_agreement_on_synthetic_inputs(tmp_path):
    from tools.wzprop import regime_wm as RG
    strategies = ['wall_g', 'wall_h', 'wall_h_solo2', 'pair_t']
    truth = {'wall_g': 300.0, 'wall_h': 900.0, 'wall_h_solo2': 1500.0, 'pair_t': -400.0}
    states, scores, results = [], [], []
    for cell in ('D3200_p60', 'D3200_p40'):
        for seed in range(989300, 989306):
            for t0 in (450, 650):
                sid = f'rgwm_{cell}_s{seed}_t{t0}'
                states.append(dict(cell=cell, seed=seed, t0=t0, sid=sid, valid=True,
                                   variants={s: {} for s in strategies}, feasible={s: True for s in strategies}))
                for s in strategies:
                    mean = truth[s] / 4 + (seed - 989303) * 10
                    scores.append(dict(sid=sid, variant=s, wm=dict(T=dict(mean=mean, lcb=mean - 1))))
            base = 100000.0
            results.append(dict(status='ok', cell=cell, seed=seed + 600, arm='base', J_total=base))
            for s in strategies + ['lane_balance']:
                results.append(dict(status='ok', cell=cell, seed=seed + 600, arm=s,
                                    J_total=base - truth.get(s, 50.0) + (seed % 3) * 30))
    sp, sc, rp = tmp_path / 'st.json', tmp_path / 'sc.jsonl', tmp_path / 'res.jsonl'
    sp.write_text(json.dumps(states))
    sc.write_text(''.join(json.dumps(x) + '\n' for x in scores))
    rp.write_text(''.join(json.dumps(x) + '\n' for x in results))
    res = RG.run(sp, sc, [rp], label='synthetic')
    assert res['point']['spearman_pooled'] > 0.9 and res['point']['sign_accuracy'] == 1.0
    assert res['point']['n_pairs'] == 8 and 'lane_balance' in res['not_evaluable']
    assert res['ci95']['sign_accuracy'][0] <= 1.0 and '| regime |' in RG.markdown(res)


def test_tape_stage_and_trimmed_geometry(tmp_path, dry):
    from tools.wzprop import runner as RN2
    RN2.tape_verify()
    ref = Path(dry['ref'])
    out = tmp_path / 'tapes.jsonl'
    res = V.plan_tapes(ref, out)
    jobs = RN2.read_jobs(out)
    assert res['jobs'] == len(jobs) == len(json.loads((ref / 'WORLDS.json').read_text()))
    for j in jobs:
        assert j['kind'] == 'prefix' and j['t_end'] == 900.0 and RN2.planned(j) == 900.0
        assert j['out_dir'].replace('\\', '/').endswith(f"tapes/runs/{j['cell']}/s{j['seed']}/prefix")
        G.check_seed(j['seed'])
    geo = json.loads((ref / 'MOMENTS_GEOMETRY.json').read_text())
    P = n3.modules()[0]
    for m in geo:
        if not m.get('valid'):
            continue
        r_ids = sorted(v for v in m['variants'] if v.startswith('R'))
        assert r_ids == [f'R{i}' for i in range(len(r_ids))]
        snap = dict(cavs=m['all_cavs_t0'])
        excl = {frozenset(mm['id'] for mm in v['members']) for k, v in m['variants'].items() if not k.startswith('R')}
        four = P.r_proposals.__globals__['N_R']
        assert four == 5
        P.N_R = 4
        try:
            r4 = P.r_proposals(snap, 'pg_' + m['sid'][4:], set(excl))
        finally:
            P.N_R = 5
        assert [x[0] for x in r4] == r_ids[:len(r4)]
        for vid, mem, prm in r4:
            assert [mm['id'] for mm in m['variants'][vid]['members']] == [c['id'] for c in mem]


# ------------------------------------------------------------------------------------ open search
OS_SMALL = dict(n_init=24, generations=2, n_elite=4, n_children=12, finalists=8, k=5)


def test_open_search_roams_archives_and_splits_slots(P):
    from tools.wzprop import open_search as OS
    m = _lc_moment()
    row, archive, evaluated = OS.search(W.StubWM(), m, OS.config(OS_SMALL))
    assert len(row['variants']) == 5 and row['archive_cells'] >= 4 and len(evaluated) > 24
    slots = [v['wm']['open_search']['slot'] for v in row['variants'].values()]
    assert slots[0] == 'top' and 'diverse' in slots
    cells = [tuple(v['wm']['open_search']['cell']) for v in row['variants'].values()]
    div = [c for c, s in zip(cells, slots) if s == 'diverse']
    assert len(set(div)) == len(div) and not set(div) & {cells[0]}
    roles = {mm['role'] for v in row['variants'].values() for mm in v['members']}
    assert roles <= {'hold', 'prog'}
    for vid, v in row['variants'].items():
        assert 2 <= len(v['members']) <= 6 or v['family'] != 'free_program'
        V.validate_variant(P, m, vid, v)                   # the SUMO plug-in accepts every proposal
    for r in evaluated.values():                           # disagreement-penalised LCB
        st = r.get('M') or r['T']
        assert r['score'] <= st['mean'] - 0.5 * st['sd'] + 1e-9
    assert OS.search(W.StubWM(), m, OS.config(OS_SMALL))[0]['variants'] == row['variants']


def test_open_search_with_a_hold_only_wm_stays_executable(P):
    from tools.wzprop import open_search as OS

    class HoldOnly:
        name = 'hold_only'
        meta = dict(backend='hold_only')

        def __init__(self):
            self.inner = W.StubWM()

        def observe(self, mo):
            return self.inner.observe(mo)

        def evaluate(self, st, acts, K, H, seed):
            assert all(mm.get('role', 'hold') == 'hold' for a in acts for mm in a['members'])
            return self.inner.evaluate(st, acts, K, H, seed)
    m = _lc_moment()
    row, _a, _e = OS.search(W._Factory(HoldOnly()), m, OS.config(OS_SMALL))
    assert row['variants'] and all(mm['role'] == 'hold' for v in row['variants'].values() for mm in v['members'])
    het = [v for v in row['variants'].values() if v['family'] == 'free_program']
    for v in het:
        V.validate_variant(P, m, 'x', v)
    bad = json.loads(json.dumps(next(iter(row['variants'].values()))))
    bad['params'] = {}
    bad['members'][0]['params']['v_w'] = 40.0
    with pytest.raises(ValueError, match='registered ranges'):
        V.validate_variant(P, m, 'bad', bad)


def test_descriptor_cells_select_and_feedback(dry):
    from tools.wzprop import open_search as OS
    c = [dict(id='a', edge='up', lane=0, x=1500.0, v=20.0), dict(id='b', edge='up', lane=1, x=1560.0, v=24.0)]
    hold = dict(members=[F.member_spec(x, dict(v_w=15.0, hold_s=90, dmax=1.5)) for x in c])
    cell = OS.descriptor(hold, {x['id']: x for x in c})
    assert cell == (2, 2, 0, OS._bin(-7.0, OS.DV_BINS), OS._bin(G.X_B - 1530.0, OS.DROP_BINS), 0)
    Pl = __import__('tools.wzprop.families_lc', fromlist=['lcm']).lcm()
    prog = dict(members=[Pl.prog(c[0], lc_target=1, lc_from=0, speed='accel', dmax=1.0, hold_s=60)])
    assert OS.descriptor(prog, {})[1:3] == (1, 1)

    def rec(key, mean, sd, cell):
        return dict(move=dict(key=key), score=mean - sd, cell=cell, T=dict(mean=mean, sd=sd), K=8)
    final = [rec('a', 100, 10, (1,)), rec('b', 90, 5, (1,)), rec('c', 50, 5, (2,)), rec('d', 20, 80, (3,)),
             rec('e', 10, 1, (4,))]
    picks = OS.select(final, 4, 0.5, 0.25)
    assert [s for s, _ in picks] == ['top', 'top', 'diverse', 'uncertain']
    assert [r['move']['key'] for _, r in picks] == ['a', 'b', 'c', 'd']
    summ = json.loads((Path(dry['wm']) / 'results' / 'WMC_SUMMARY.json').read_text())
    moms = json.loads((Path(dry['wm']) / 'MOMENTS.json').read_text())
    fb = OS.feedback_rows(summ, moms)
    assert fb and all(r['measured']['T_r0'] is not None and r['action']['members'] for r in fb)
    cmp = C.compare([Path(dry['ref']) / 'results' / 'WMC_SUMMARY.json', Path(dry['wm']) / 'results' / 'WMC_SUMMARY.json'],
                    moments=[Path(dry['ref']) / 'MOMENTS.json', Path(dry['wm']) / 'MOMENTS.json'])
    vp = cmp['verified_plans']
    assert vp and all(r['cell'] is not None and r['origin'] for r in vp)


# ------------------------------------------------------------------------------------ strategy-level search
def _strategy_states(n=6, seed0=989300):
    out = []
    for i in range(n):
        m = moment(seed=seed0 + i, t0=450 + 10 * (i % 2))
        m['sid'] = f"st_{m['sid']}_{i}"
        out.append(m)
    return out


def test_solo2_is_the_regime_rule_and_strategies_validate(P):
    from tools.wzprop import strategy_search as SS
    m = moment()
    wall = P.pick_wall(dict(cavs=m['all_cavs_t0']))
    ref = [P.member_spec(c, P.VARIANTS['wall_h'][1]) for c in wall if c['lane'] == 2]
    assert SS.activation(SS.SOLO2, m['all_cavs_t0']) == ref
    for s in SS.seeds():
        SS.validate(s)
    rng = __import__('random').Random(3)
    rs = [SS.random_strategy(rng) for _ in range(50)]
    assert all(1 <= len(s['roles']) <= 6 for s in rs) and len({SS.strategy_key(s) for s in rs}) > 40
    for s in rs:
        spec = SS.activation(s, m['all_cavs_t0'])
        if spec:
            roles = {x['role'] for x in spec}
            assert len(roles) == 1 and all(x['id'] in {c['id'] for c in m['all_cavs_t0']} for x in spec)
            if roles == {'prog'}:
                from tools.wzprop import families_lc as FL
                for x in spec:
                    assert not FL.out_of_range(x)
            else:
                for x in spec:
                    assert not F.hold_out_of_range(x)
    with pytest.raises(ValueError):
        SS.validate(dict(SS.SOLO2, roles=[dict(SS.SOLO2['roles'][0], v_w=40.0)]))
    with pytest.raises(ValueError):
        SS.validate(dict(SS.SOLO2, roles=[dict(SS.SOLO2['roles'][0], lc_target=0)]))
    kids = [SS.mutate(rng, s, 0.5) for s in rs[:20]]
    assert any(k is not None for k in kids)


def test_strategy_search_beats_or_reports_against_solo2_in_imagination():
    from tools.wzprop import strategy_search as SS
    states = _strategy_states(6)
    per, n = SS.evaluate(W.StubWM(), states, [SS.SOLO2] + SS.seeds()[1:3], 4, 300)
    s2 = SS.summarize(per[SS.strategy_key(SS.SOLO2)], 0.5)
    assert s2['gain_vs_solo2'] == 0.0 and s2['n_states'] == 6 and n > 0
    res = SS.search_cell(W.StubWM(), states, dict(n_init=10, generations=1, n_children=8, n_screen_states=4,
                                                    finalists=5, K_final=4))
    keys = [r['key'] for r in res['final']]
    assert SS.strategy_key(SS.SOLO2) in keys and res['archive_cells'] >= 3
    assert all(r['gain_vs_solo2_lcb'] <= r['gain_vs_solo2'] + 1e-9 for r in res['final'])
    picks = SS.select_for_sumo(res, 3)
    assert len({tuple(SS.descriptor(p)) for p in picks}) == len(picks)
    assert all(SS.strategy_key(p) != SS.strategy_key(SS.SOLO2) for p in picks)
    assert len(SS.structured_arm()) == 4 and len(SS.random_arm()) == 4
    if res['best_beating_solo2']:
        assert res['best_beating_solo2']['gain_vs_solo2_lcb'] > 0


def test_strategy_core_executes_the_rule_on_plant_rows():
    from tools.wzprop import strategy_hook as SH
    from tools.wzprop import strategy_search as SS
    cav = [('A', 1600.0, 2), ('B', 1580.0, 1), ('C', 1400.0, 0), ('D', 1200.0, 2)]
    rows = [dict(vid=v, s_m=x - SH.S_OFF, v_mps=22.0, lane=2 - ln, is_cav=True) for v, x, ln in cav]
    rows.append(dict(vid='hdv1', s_m=1610.0 - SH.S_OFF, v_mps=20.0, lane=0, is_cav=False))
    cmds = {r['vid']: 0.0 for r in rows if r['is_cav']}
    s = dict(name='t', trigger='lanes', zone='full', spread_m=300.0, cooldown_s=60, slots=1,
             roles=[dict(lane=2, rank=0, v_w=15.0, dmax=1.5, hold_s=90, onset_s=0, lc_target=None),
                    dict(lane=1, rank=0, v_w=18.0, dmax=1.0, hold_s=90, onset_s=20, lc_target=2)])
    core = SH.StrategyCore(s, warmup_s=300)
    assert core.step(rows, cmds, 299) == ({}, {})
    acc, lanes = core.step(rows, cmds, 300)
    mem = SS.select_members(s, [SH.to_sumo_row(r) for r in rows if r['is_cav']])
    assert mem and set(core.active) == {c['id'] for c in mem} and 'hdv1' not in acc
    assert list(acc) == [mem[0]['id']]                     # second role starts at onset 20 s
    for t in range(301, 321):
        acc, lanes = core.step(rows, cmds, t)
    assert mem[1]['id'] in acc and lanes[mem[1]['id']] == SH.wm_lane_of(2) and core.lane_requests >= 1
    assert len(core.acts) == 1                             # cooldown 60 s: no second activation yet
    for t in range(321, 430):
        core.step(rows, cmds, t)
    assert all(v[1] in ('hold_end', 'at_drop', 'left') for a in core.acts for v in a['release'].values())
    assert len(core.acts) >= 2


def test_wmsearch_u0_plan_and_paired_analysis(tmp_path):
    from tools.wzprop import strategy_search as SS
    from tools.wzprop import wmsearch_u0 as WU
    arms = dict(D3200_p60=dict(incumbent=SS.SOLO2, wm=SS.random_arm(4, seed=1), structured=SS.structured_arm(4),
                                random=SS.random_arm(4, seed=2)))
    specs = WU.arm_specs(arms, 'D3200_p60')
    assert list(specs)[:2] == ['u0', 'solo2'] and len(specs) == 14
    groups = WU.plan_jobs('D3200_p60', range(989770, 989786), specs,
                          lambda s, d, p: dict(seed=s, demand=d, penetration=p, cell_key='D3200_p60'))
    assert sum(j['sim_s'] for g in groups for j in g['jobs']) == 16 * 14 * 1200 <= WU.CAP
    with pytest.raises(G.ForbiddenSeedError):
        WU.plan_jobs('D3200_p60', [989650], specs, lambda s, d, p: {})
    rows = []
    for s in range(16):
        J = dict(u0=50000.0 + 100 * s, solo2=40000.0 + 100 * s)
        for a in specs:
            if a.startswith('wm_'):
                J[a] = J['solo2'] - 2000.0 + 50 * (s % 3)
            elif a.startswith(('str_', 'rnd_')):
                J[a] = J['solo2'] + 500.0 - 40 * (s % 4)
        rows.append(dict(seed=989770 + s, J=J))
    res = WU.analyze_rows(rows, specs)
    assert res['summary']['wm']['n_beat_solo2_ci'] == 4 and res['summary']['random']['n_beat_solo2_ci'] == 0
    assert res['arms']['solo2']['vs_u0']['mean'] == pytest.approx(-10000.0)


def test_parallel_wm_equals_sequential():
    from tools.wzprop import strategy_search as SS
    states = _strategy_states(4)
    strats = [SS.SOLO2] + SS.seeds()[1:4]
    seq, _ = SS.evaluate(W.StubWM(), states, strats, 2, 300)
    pw = W.ParallelWM('stub', 2)
    try:
        par, _ = SS.evaluate(pw, states, strats, 2, 300)
    finally:
        pw.close()
    assert dict(seq) == dict(par) and pw.meta['parallel_workers'] == 2


def test_condition_gate_is_observable_and_shared_by_wm_and_sumo():
    from tools.wzprop import strategy_hook as SH
    from tools.wzprop import strategy_search as SS
    m = moment()
    cavs = m['all_cavs_t0']
    gated = dict(SS.SOLO2, name='g', gate=10.0)
    slow = [dict(id='h1', edge='up', lane=0, x=G.X_B - 100.0, v=4.0, is_cav=False)]
    fast = [dict(id='h1', edge='up', lane=0, x=G.X_B - 100.0, v=25.0, is_cav=False)]
    base = SS.activation(SS.SOLO2, cavs)
    assert base and SS.activation(gated, cavs, slow) == base
    assert SS.activation(gated, cavs, fast) is None
    assert SS.activation(gated, [c for c in cavs if not (c['edge'] == 'up' and c['lane'] == 0 and
                                                      c['x'] >= SS.FIELD_X[0])], []) is None   # nothing observed: closed
    with pytest.raises(ValueError):
        SS.validate(dict(SS.SOLO2, gate=7.0))
    rows = [dict(vid='c', s_m=-50.0, v_mps=10.0, lane=1, is_cav=True),
            dict(vid='h_near', s_m=0.0, v_mps=3.0, lane=2, is_cav=False),
            dict(vid='h_far', s_m=-400.0, v_mps=3.0, lane=2, is_cav=False),
            dict(vid='h_lane', s_m=-40.0, v_mps=3.0, lane=0, is_cav=False)]
    ids = {r['id'] for r in SH.observed_rows(rows)}
    assert ids == {'c', 'h_near', 'h_lane'}


def test_fairness_groups_archives_and_trace():
    from tools.wzprop import strategy_search as SS
    states = _strategy_states(4)
    wm = W.StubWM()
    mi = W.moment_input(states[0])
    st = wm.observe(mi)
    acts = [W.action('base', 'base', None, 'base', [], 300)] + [
        W.action(f'c{i}', 'group', 's', f'k{i}', SS.activation(s, states[0]['all_cavs_t0']), 300)
        for i, s in enumerate(SS.seeds()[:3]) if SS.activation(s, states[0]['all_cavs_t0'])]
    g = wm.evaluate_groups(st, acts, 2, 300, 1)
    for aid, gg in g.items():
        for k in range(2):
            assert sum(gg[f'lane{L}'][k] for L in (0, 1, 2)) == pytest.approx(gg['all'][k])
            assert gg['hdv'][k] + gg['cav'][k] == pytest.approx(gg['all'][k])
    res = SS.search_cell(wm, states, dict(n_init=6, generations=1, n_children=6, n_screen_states=3, finalists=5,
                                          K_final=2, K_screen=2))
    assert res['fairness'] == 'available' and all(r['fair'] is not None for r in res['final'])
    for r in res['final']:
        if r['fair']:
            assert r['gain_vs_base_lcb'] > 0 and min(r['group_gain_vs_base_lcb'].values()) >= 0
    assert all(f['worst_group_lcb'] >= 0 for f in res['fair_archive'])
    # planapi-style trace -> per-group delays
    import numpy as np
    tr = dict(rows=[(0, 0), (1, 0)], speeds=[np.array([[[31.29, 0.0], [15.645, 0.0]], [[0.0, 31.29], [0.0, 31.29]]])],
              masks=[np.array([[[1, 1], [1, 1]], [[1, 1], [1, 0]]])], is_cav=[np.array([[1, 0], [1, 0]])],
              origin_lane=[np.array([[0, 2], [0, 2]])])
    out = W.groups_from_trace(tr, [dict(id='a'), dict(id='b')], 1)
    assert out['a']['lane2'][0] == pytest.approx(0.25) and out['a']['lane0'][0] == pytest.approx(1.0)
    assert out['b']['cav'][0] == pytest.approx(1.0) and out['b']['hdv'][0] == pytest.approx(0.0)
    with pytest.raises(W.FairnessUnavailable):
        W.groups_from_trace(dict(rows=[], speeds=[]), [], 1)
