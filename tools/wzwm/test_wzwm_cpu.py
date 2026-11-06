"""WM-C CPU tests (fakes only: StubWM + fake SUMO core; no SUMO, GPU or ssh).

    C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/wzwm/test_wzwm_cpu.py
"""
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tools.wzwm import compare as C
from tools.wzwm import fakes
from tools.wzwm import families as F
from tools.wzwm import geometry as G
from tools.wzwm import n3
from tools.wzwm import pipeline
from tools.wzwm import proposer as PR
from tools.wzwm import runner as RN
from tools.wzwm import verify_plugin as V
from tools.wzwm import wm_backend as W


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
    out = tmp_path_factory.mktemp('wzwm_dry')
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
    for s in (989200, 989400, 989599):
        assert G.check_seed(s) == s
    for s in (989600, 989650, 989699, 964000, 964199, 962123, 977001, 986000, 989199, 989700, 'x'):
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
    cfg = PR.config(dict(k=5))
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
    from tools.wzwm import wallfit_bridge as WB
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
    from tools.wzwm import wallfit_bridge as WB
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
    from tools.wzwm.__main__ import main
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
