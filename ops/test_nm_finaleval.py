"""CPU-only tests for the clean4 wz3to2 final-evaluation launcher. No simulator, no process pool."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nm_finaleval as nm  # noqa: E402
import nm_map_stub  # noqa: E402
from tools.coopmwm import rerun_cpus as rc  # noqa: E402
from tools.coopmwm.exact_grade import _seal  # noqa: E402

ASSETS = Path(nm_map_stub.DEFAULT_ASSETS)/nm.MAP


@pytest.fixture(scope='module')
def panel():
    return nm.load_panel()


def make(tmp_path, panel, base='mappo_u0', register=True, **register_kw):
    root, cfg = nm.build_fake_root(tmp_path/'dry', panel, base)
    runtime = nm.FakeRuntime()
    final_root = tmp_path/'dry'/'final_root'
    plan = nm.register(root=root, final_root=final_root, runtime=runtime, panel=panel, **register_kw) if register else None
    return dict(root=root, cfg=cfg, runtime=runtime, final_root=final_root, plan=plan, panel=panel)


def write_config(f, cfg):
    (Path(f['root'])/'CONFIG.json').write_text(json.dumps(cfg))
    return cfg


def charge(root, **row):
    with (Path(root)/'CHARGES.jsonl').open('a') as fh:
        fh.write(json.dumps(row)+'\n')


def test_registered_panel_is_pinned_and_follows_the_registration(panel):
    assert panel['panel_sha256'] == nm.PANEL_SHA256
    seeds = [r['world']['seed'] for r in panel['worlds']]
    assert seeds == list(range(989600, 989696)) and len(set(seeds)) == 96
    assert all(r['world_id'] == f'wz3to2_{r["world"]["seed"]}' and r['world']['map'] == 'wz3to2' for r in panel['worlds'])
    assert not any(989200 <= s <= 989599 for s in seeds)
    assert panel['cell']['demand_vph'] == {'main': 2800} and panel['cell']['penetration'] == 0.3
    assert {(r['world']['demand'], r['world']['penetration'], r['world']['profile']) for r in panel['worlds']} == {(2800, 0.3, 'peak')}
    contract = panel['contract']
    assert {k: contract['bootstrap'][k] for k in ('reps', 'level', 'seed')} == \
        {k: rc.FINAL_EVALUATION['bootstrap'][k] for k in ('reps', 'level', 'seed')}
    assert contract['horizon_s'] == 1200 and contract['window'] == dict(start_s=0, end_s=1200)
    assert contract['safety_counters'] == list(nm.SAFETY) and contract['comparisons'] == ['B-A', 'R-A', 'B-R']
    assert contract['primary']['contrast'] == 'B-BASE'
    assert contract['decisions']['worlds'] == [f'wz3to2_{989600+4*k}' for k in range(24)]
    assert contract['budget'] == dict(contract['budget'], phase='v2v_final', allocation_sim_s=690_000,
                                      quote_mappo_u0=662_400, quote_native=547_200)
    assert panel['seeds_reserved_unused'] == [989696, 989699]
    assert panel['power']['chosen'] == 96 and panel['power']['mde_sd_units']['96'] == pytest.approx(2.8016/math.sqrt(96), abs=1e-3)


def test_panel_tampering_is_refused(panel, tmp_path):
    edited = deepcopy(panel)
    edited['worlds'][0]['world']['seed'] = 989601
    path = tmp_path/'p.json'
    path.write_text(json.dumps(edited))
    with pytest.raises(nm.Blocked, match='seal'):
        nm.load_panel(path)
    path.write_text(json.dumps(_seal({k: v for k, v in edited.items() if k != 'panel_sha256'}, 'panel_sha256')))
    with pytest.raises(nm.Blocked):
        nm.load_panel(path, pinned=None)
    resealed = deepcopy(panel)
    resealed['cell']['penetration'] = 0.4
    for r in resealed['worlds']:
        r['world']['penetration'] = 0.4
    path.write_text(json.dumps(_seal({k: v for k, v in resealed.items() if k != 'panel_sha256'}, 'panel_sha256')))
    assert nm.load_panel(path, pinned=None)['cell']['penetration'] == 0.4     # internally consistent ...
    with pytest.raises(nm.Blocked, match='pinned'):
        nm.load_panel(path)                                                    # ... but not the registration


@pytest.mark.skipif(not (ASSETS/'MAP_SPEC.json').is_file(), reason='wz3to2 map files not on this machine')
def test_panel_rebuilds_from_the_map_files_only(panel):
    rebuilt = nm.build_panel(ASSETS, panel['registered_utc'])
    assert rebuilt == panel and rebuilt['panel_sha256'] == nm.PANEL_SHA256
    report = nm.map_check(nm.bind_map_api('nm_map_stub'), panel)
    assert report['ok'] and report['stub'] and not report['mismatches']


def test_map_check_flags_an_api_that_disagrees_with_the_panel(panel):
    fake = types.ModuleType('fake_n1_disagree')
    fake.STUB = False
    fake.seed_ranges = lambda name: dict(train=(989200, 989599), eval=(989600, 989650))
    fake.descriptor = lambda name: dict(net_sha256='0'*64, stream_namespace='other')
    fake.world = lambda name, seed, cell, repeat: dict(nm_map_stub.world(name, seed, cell, repeat), demand=3400)
    fake.check_world = nm_map_stub.check_world
    sys.modules['fake_n1_disagree'] = fake
    report = nm.map_check(nm.bind_map_api('fake_n1_disagree'), panel)
    assert not report['ok'] and not report['stub'] and len(report['mismatches']) == 4
    with pytest.raises(nm.Blocked, match='lacks'):
        nm.bind_map_api('json')


@pytest.mark.parametrize('base', ['mappo_u0', 'native'])
def test_dry_run_end_to_end_is_zero_physics(panel, tmp_path, monkeypatch, base):
    import concurrent.futures

    class Forbidden:
        def __init__(self, *a, **k):
            raise AssertionError('dry run must never open a process pool')
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Forbidden)
    result = nm.dry_run(out=tmp_path/'out', panel=panel, base=base)
    assert result['status'] == 'PASS' and result['new_sim_s'] == 0 and result['physics'] is False
    assert result['present'] == ['B', 'A', 'R', 'BASE', 'NATIVE']
    runs = [c for c in result['calls'] if c[0] == 'run']
    assert runs[0] == ('run', 96*(5 if base == 'mappo_u0' else 4), 24)
    assert runs[1][1] <= 2*24*(2+nm.MAX_MEMBERS)
    assert {c[1] for c in result['calls'] if c[0] == 'reserve'} == {'v2v_final'}
    assert result['fake_v2v_final_sim_s'] <= result['quote']['total_ceiling_sim_s'] <= 690_000
    final_root = Path(result['final_root'])
    tables = json.loads((final_root/'FINAL_TABLES.json').read_text())
    nm.verify(tables, 'analysis_sha256', 'tables')
    assert tables['dry_run'] is True and tables['missing'] == {} and tables['panel_sha256'] == nm.PANEL_SHA256
    t = tables['tables']
    assert set(t['registered_B_A_R']['rows']) == {'B-A', 'R-A', 'B-R'}
    assert t['total_benefit']['base_is_native'] is (base == 'native')
    assert set(t['total_benefit']['vs_NATIVE']['rows']) == ({'B', 'A', 'R', 'BASE'} if base == 'mappo_u0' else {'B', 'A', 'R'})
    for key in ('B-BASE', 'B-NATIVE'):
        assert t['synergy']['panel_decomposition'][key]['max_identity_error_veh_s'] < 1e-9
    pdx = t['synergy']['per_decision']
    assert pdx['registered_worlds'] == 24
    for arm in ('B', 'R'):
        assert pdx[arm]['decisions']+pdx[arm]['worlds_without_decision'] == 24
        assert pdx[arm]['max_identity_error_veh_s'] < 1e-9
    assert t['synergy']['registered_u24_train_states']['verdict'] == 'pass'
    assert 'row' in t['primary'] and isinstance(t['primary']['meets_rule'], bool)
    text = (final_root/'FINAL_TABLES.md').read_text()
    assert 'FAKE PHYSICS' in text and '## T3 Synergy decomposition' in text and 'Per-decision decomposition' in text
    status = json.loads((final_root/'STATUS.json').read_text())
    assert status['step'] == 'TABLES' and status['status'] == 'PASS'


def test_rerun_reuses_published_waves_and_charges_nothing(panel, tmp_path):
    result = nm.dry_run(out=tmp_path/'out', panel=panel)
    root, final_root = Path(result['out'])/'clean4_root', Path(result['final_root'])
    before = (root/'CHARGES.jsonl').read_text()
    runtime = nm.FakeRuntime()
    value = nm.execute(root=root, final_root=final_root, runtime=runtime, panel=panel, poll=0, dry_run=True)
    assert value['analysis_sha256'] == result['analysis_sha256']
    assert not [c for c in runtime.calls if c[0] in ('run', 'reserve')] and (root/'CHARGES.jsonl').read_text() == before


def test_gate_waits_for_u24_and_blocks_on_safety_or_tampering(panel, tmp_path):
    f = make(tmp_path, panel)
    root, cfg, runtime = f['root'], f['cfg'], f['runtime']
    binding = nm.clean4_binding(cfg, panel)
    assert nm.gate(root, cfg, runtime, binding)['status'] == 'PASS'
    (root/'B_checkpoint_024.pt').rename(root/'B_hidden.pt')
    with pytest.raises(nm.NotReady, match='u24 not sealed'):
        nm.gate(root, cfg, runtime, binding)
    (root/'B_hidden.pt').rename(root/'B_checkpoint_024.pt')
    (root/'PAUSE_SAFETY.json').write_text('{}')
    with pytest.raises(nm.NotReady, match='health'):
        nm.gate(root, cfg, runtime, binding)
    (root/'PAUSE_SAFETY.json').unlink()
    path = root/'LEARNING_MONITOR_u011.json'
    row = json.loads(path.read_text())
    row['arms'][0]['metrics']['safety']['teleports'] = 1
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    with pytest.raises(nm.Blocked, match='safety not clean'):
        nm.gate(root, cfg, runtime, binding)
    row['arms'][0]['metrics']['safety']['teleports'] = 0
    row['arms'][2]['action'] = 'PAUSE_SAFETY'
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    with pytest.raises(nm.Blocked, match='pauses'):
        nm.gate(root, cfg, runtime, binding)
    row['arms'][2]['action'] = 'CONTINUE'
    row['config_sha256'] = '2'*64
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    with pytest.raises(nm.Blocked, match='not bound'):
        nm.gate(root, cfg, runtime, binding)
    row['config_sha256'] = nm.digest(cfg)
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    (root/'A_checkpoint_000.pt').write_bytes(b'other')
    with pytest.raises(nm.Blocked, match='common MAPPO u0'):
        nm.gate(root, cfg, runtime, binding)
    assert nm.gate(root, cfg, runtime, dict(binding, base='native'))['status'] == 'PASS'   # u0 irrelevant for native


def test_clean4_binding_refuses_mismatches_and_reports_unbound_keys(panel, tmp_path):
    f = make(tmp_path, panel, register=False)
    cfg = f['cfg']
    assert nm.clean4_binding(cfg, panel)['unbound'] == []
    for change, match in ((dict(map='weave'), 'map'), (dict(eval_panel_sha256='1'*64), 'panel'),
                          (dict(cell=dict(demand_vph={'main': 3400}, penetration=0.3, profile='peak')), 'cell'),
                          (dict(cell=dict(demand_vph={'main': 2800}, penetration=0.4)), 'cell'),
                          (dict(base_controller='lane_on'), 'base')):
        bad = deepcopy(cfg)
        bad['newmap'].update(change)
        with pytest.raises(nm.Blocked, match=match):
            nm.clean4_binding(bad, panel)
    assert nm.clean4_binding(dict(cfg, newmap=dict(cfg['newmap'], cell=dict(demand=2800, penetration=0.3))), panel)
    bare = {k: v for k, v in cfg.items() if k != 'newmap'}
    with pytest.raises(nm.Blocked, match='base controller'):
        nm.clean4_binding(bare, panel)
    out = nm.clean4_binding(bare, panel, declared_base='native')
    assert out['base'] == 'native' and set(out['unbound']) == {'map', 'eval_panel_sha256', 'cell', 'base_controller'}
    with pytest.raises(nm.Blocked, match='declared base'):
        nm.clean4_binding(cfg, panel, declared_base='native')
    with pytest.raises(nm.Blocked, match='common_checkpoint'):
        nm.clean4_binding(dict(cfg, common_checkpoint_sha256=None), panel)


def test_register_records_an_operator_declared_base_for_an_unbound_config(panel, tmp_path):
    f = make(tmp_path, panel, register=False)
    cfg = deepcopy(f['cfg'])
    for k in ('base_controller', 'cell'):
        cfg['newmap'].pop(k)
    write_config(f, cfg)
    plan = nm.register(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel, base='native')
    assert plan['binding']['base'] == 'native' and set(plan['binding']['unbound']) == {'base_controller', 'cell'}
    assert plan['quote']['total_ceiling_sim_s'] == 547_200 and plan['budget']['allocated'] == 690_000
    _, binding, _ = nm.preflight(f['root'], f['final_root'], f['runtime'], panel)
    assert binding['base'] == 'native'
    with pytest.raises(nm.Blocked, match='never overwrite'):
        nm.register(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel, base='native')


def test_budget_is_debited_from_clean4_v2v_final_only_within_its_allocation(panel, tmp_path):
    f = make(tmp_path, panel, register=False)
    cfg = f['cfg']
    report = nm.budget_check(f['root'], cfg, 'mappo_u0')
    assert report['source'] == 'clean4_budget' and report['allocated'] == 690_000 and report['remaining'] == 690_000
    small = dict(cfg, clean4_budget=_seal(dict(phases=dict(v2v_final=600_000)), 'receipt_sha256'))
    with pytest.raises(nm.Blocked, match='allocation'):
        nm.budget_check(f['root'], small, 'mappo_u0')
    edited = deepcopy(cfg)
    edited['clean4_budget']['phases']['v2v_final'] = 10**7          # edited after sealing
    with pytest.raises(nm.Blocked, match='seal'):
        nm.budget_check(f['root'], edited, 'mappo_u0')
    with pytest.raises(nm.Blocked, match='allocation'):
        nm.budget_check(f['root'], {k: v for k, v in cfg.items() if k != 'clean4_budget'}, 'mappo_u0')
    legacy = dict({k: v for k, v in cfg.items() if k != 'clean4_budget'}, clean3_budget=cfg['clean4_budget'])
    assert nm.budget_check(f['root'], legacy, 'mappo_u0')['source'] == 'clean3_budget'
    charge(f['root'], id='paired_u024_0000', phase='v2v_final', sim_s=1200)
    with pytest.raises(nm.Blocked, match='foreign'):
        nm.budget_check(f['root'], cfg, 'mappo_u0')


def test_sub_cap_refuses_a_wave_that_would_exceed_690000(panel, tmp_path):
    f = make(tmp_path, panel)
    charge(f['root'], id='nmfinal_old_attempt_0000', phase='v2v_final', sim_s=600_000)
    specs = nm.episode_specs('NATIVE', panel, f['cfg'], f['root'], 'mappo_u0')
    with pytest.raises(nm.Blocked, match='sub-cap'):
        nm.run_waves([('NATIVE', 'episode', specs)], root=f['root'], final_root=f['final_root'], cfg=f['cfg'],
                     runtime=f['runtime'], workers=8, base='mappo_u0')
    assert not f['runtime'].calls


def test_exposure_check_refuses_evaluation_seeds_in_clean4(panel, tmp_path):
    f = make(tmp_path, panel, register=False)
    root, cfg = f['root'], f['cfg']
    ok = dict(cfg, seed_ranges=dict(train=[989200, 989599], eval=[989600, 989699]), digest_x='ab989612cd'*6+'0000',
              newmap=dict(cfg['newmap'], eval_seed_range=[989600, 989699]))
    assert nm.exposure_check(root, ok)['status'] == 'PASS'
    with pytest.raises(nm.Blocked, match='exposed'):
        nm.exposure_check(root, dict(cfg, train_worlds=[dict(seed=989200), dict(seed=989650)]))
    with pytest.raises(nm.Blocked, match='exposed'):
        nm.exposure_check(root, dict(cfg, note='interim on wz3to2_989601'))
    charge(root, id='clean4_interim_B_989612', phase='v2v_interim', sim_s=1200)
    with pytest.raises(nm.Blocked, match='ledger'):
        nm.exposure_check(root, cfg)


def test_physics_is_refused_with_the_stub_map_api_or_without_a_runner(panel, tmp_path):
    f = make(tmp_path, panel)

    class Physical(nm.FakeRuntime):
        physics = True
    with pytest.raises(nm.Blocked, match='map API not qualified'):
        nm.preflight(f['root'], f['final_root'], Physical(), panel)
    real = types.ModuleType('fake_n1_real')
    real.STUB = False
    for name in ('seed_ranges', 'descriptor', 'world', 'check_world'):
        setattr(real, name, getattr(nm_map_stub, name))
    sys.modules['fake_n1_real'] = real
    if (ASSETS/'MAP_SPEC.json').is_file():
        with pytest.raises(nm.Blocked, match='runner unavailable'):
            nm.preflight(f['root'], f['final_root'], Physical(runner=dict(ok=False, reason='absent')), panel,
                         map_module='fake_n1_real')
    status = nm.Runtime('tools.coopmwm.no_such_nm_runner').runner_status()
    assert status['ok'] is False and 'not importable' in status['reason']
    with pytest.raises(nm.Blocked, match='RUN_PLAN'):
        nm.execute(root=f['root'], final_root=tmp_path/'unregistered', runtime=f['runtime'], panel=panel, poll=0)


def test_missing_runner_capabilities_require_explicit_exclusion(panel, tmp_path):
    f = make(tmp_path, panel)
    runtime = nm.FakeRuntime(runner=dict(decisions=False))
    kw = dict(root=f['root'], final_root=f['final_root'], panel=panel, poll=0, dry_run=True)
    with pytest.raises(nm.Blocked, match='DECISIONS'):
        nm.execute(runtime=runtime, **kw)
    assert not nm.fe.ledger_rows(f['root'])[3:]          # nothing charged beyond the fake training rows
    value = nm.execute(runtime=runtime, allow_missing=('DECISIONS',), **kw)
    assert value['tables']['synergy']['per_decision']['status'] == 'unavailable'
    assert set(value['missing']) >= {'DECISIONS', 'replicated_u024'}
    assert not [r for r in nm.fe.ledger_rows(f['root']) if r['id'].startswith('nmdec_')]


def test_interrupted_waves_are_never_replayed(panel, tmp_path):
    f = make(tmp_path, panel)
    charge(f['root'], id='nmfinal_native_0000', phase='v2v_final', sim_s=1200)
    with pytest.raises(nm.Blocked, match='never replay'):
        nm.execute(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel, poll=0, dry_run=True)
    assert not [c for c in f['runtime'].calls if c[0] in ('run', 'reserve')]


def test_published_waves_are_reused_not_rerun(panel, tmp_path):
    f = make(tmp_path, panel)
    specs = nm.episode_specs('NATIVE', panel, f['cfg'], f['root'], 'mappo_u0')
    kw = dict(root=f['root'], final_root=f['final_root'], cfg=f['cfg'], runtime=f['runtime'], workers=8, base='mappo_u0')
    out, _ = nm.run_waves([('NATIVE', 'episode', specs)], **kw)
    calls = list(f['runtime'].calls)
    again, sources = nm.run_waves([('NATIVE', 'episode', specs)], **kw)
    assert again == out and f['runtime'].calls == calls and sources['NATIVE']['source'].endswith('WAVE_NATIVE.json')
    with pytest.raises(nm.Blocked, match='registered specs'):
        nm.run_waves([('NATIVE', 'episode', specs[1:])], **kw)


def test_episode_and_decision_records_are_validated(panel, tmp_path):
    f = make(tmp_path, panel)
    specs = nm.episode_specs('B', panel, f['cfg'], f['root'], 'mappo_u0')
    assert specs[0]['job_id'] == 'nmfinal_b_u024_0000' and specs[0]['sim_s'] == 1200
    assert specs[0]['noise_seed'] == 989600 and specs[0]['capture']['max_t_s'] == 840
    assert [s['capture'] is not None for s in specs[:5]] == [True, False, False, False, True]
    good = nm.fake_episode(specs[0])
    good['decision'] = dict(t_s=600, move='abreast_wall', members=['main.1', 'main.2', 'main.3'],
                            state_sha256='a'*64, state_ref='states/x.xml')
    assert nm.check_episode(good, specs[0])['J_veh_s'] == good['J_veh_s']
    for mutate, match in ((lambda r: r.update(J_veh_s=float('nan')), 'finite'),
                          (lambda r: r.update(window=dict(start_s=0, end_s=1800)), 'identity'),
                          (lambda r: r['world'].update(seed=989601), 'identity'),
                          (lambda r: r['safety'].update(teleports=-1), 'safety'),
                          (lambda r: r['safety'].pop('hard_brakes'), 'safety'),
                          (lambda r: r.update(status='failed'), 'ungraded'),
                          (lambda r: r.pop('decision'), 'capture missing'),
                          (lambda r: r['decision'].update(t_s=900), 'capture rule'),
                          (lambda r: r['decision'].update(t_s=120), 'capture rule'),
                          (lambda r: r['decision'].update(members=['a', 'b', 'c', 'd']), 'capture rule'),
                          (lambda r: r['decision'].update(members=['a', 'a']), 'capture rule'),
                          (lambda r: r['decision'].update(state_sha256='x'), 'capture rule')):
        record = deepcopy(good)
        mutate(record)
        with pytest.raises(nm.Blocked, match=match):
            nm.check_episode(record, specs[0])
    outside = dict(nm.fake_episode(specs[1]), decision=good['decision'])
    with pytest.raises(nm.Blocked, match='outside the registered'):
        nm.check_episode(outside, specs[1])
    native = nm.episode_specs('NATIVE', panel, f['cfg'], f['root'], 'mappo_u0')[0]
    assert native['capture'] is None
    with pytest.raises(nm.Blocked, match='outside the registered'):
        nm.check_episode(dict(nm.fake_episode(native), decision=good['decision']), native)


def test_decision_branches_follow_the_registered_decomposition(panel, tmp_path):
    from tools.coopmwm import policy_synergy as ps
    f = make(tmp_path, panel)
    specs = nm.episode_specs('B', panel, f['cfg'], f['root'], 'mappo_u0')
    records = {}
    for i, s in enumerate(specs):
        r = nm.fake_episode(s)
        if s['capture'] is not None:
            r['decision'] = None if i % 8 == 0 else dict(t_s=400+i, move='closing_pair', members=[f'main.{i}', f'main.{i+1}'],
                                                         state_sha256=f'{i:064x}', state_ref=f'states/{i}.xml')
        records[s['world_id']] = nm.check_episode(r, s)
    branches = nm.decision_specs('B', records, panel)
    assert len(branches) == 4*12 and {b['world_id'] for b in branches} <= set(panel['contract']['decisions']['worlds'])
    first = [b for b in branches if b['world_id'] == specs[4]['world_id']]
    assert [b['label'] for b in first] == ['all', 'background', 'solo0', 'solo1']
    assert [b['new_members'] for b in first] == [['main.4', 'main.5'], [], ['main.4'], ['main.5']]
    assert all(b['window'] == dict(start_s=404, end_s=764) and b['sim_s'] == 360 and b['background'] == 'BASE' for b in first)
    J = dict(all=30_000.0, background=33_000.0, solo0=31_000.0, solo1=32_500.0)
    rows = {b['label']: nm.check_branch(dict(nm.fake_branch(b), J_veh_s=J[b['label']]), b) for b in first}
    mine = nm.decision_metrics(rows)
    ref = ps.metrics(J['background'], [dict(status='graded', J_veh_s=J[k]) for k in ('all', 'solo0', 'solo1')],
                     ['main.4', 'main.5'])
    assert mine['delta_all_new_veh_s'] == ref['delta_all_new_veh_s'] == 3_000
    assert mine['policy_cooperation_gain_veh_s'] == ref['policy_cooperation_gain_veh_s'] == 1_000
    assert mine['delta_only_member_new_veh_s'] == ref['delta_only_member_new_veh_s']
    assert mine['identity_error_veh_s'] == 0
    with pytest.raises(nm.Blocked, match='identity'):
        nm.check_branch(dict(nm.fake_branch(first[0]), new_members=[]), first[0])
    incomplete = [nm.check_branch(nm.fake_branch(b), b) for b in first[:3]]
    with pytest.raises(nm.Blocked, match='incomplete'):
        nm.analyse_decisions(dict(B=incomplete), panel['contract']['bootstrap'])


def test_budget_quote_fits_the_clean4_v2v_final_allocation():
    q = nm.quote('mappo_u0')
    assert q['episodes_sim_s'] == 5*96*1200 == 576_000 and q['decisions_ceiling_sim_s'] == 2*24*5*360 == 86_400
    assert q['total_ceiling_sim_s'] == 662_400 and q['slack_sim_s'] == 27_600 and q['fits']
    n = nm.quote('native')
    assert n['episodes_sim_s'] == 4*96*1200 and n['total_ceiling_sim_s'] == 547_200 and n['fits']
    assert 'episodes_BASE' not in n['need_ceiling']
    with pytest.raises(nm.Blocked):
        nm.quote('lane_on')


def test_forbidden_paths_are_refused_and_never_used(tmp_path):
    for token in ('heldout', 'holdout', 'sealed', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth'):
        with pytest.raises(nm.Blocked):
            nm.guard(tmp_path/token/'x.json')
    for path in (nm.PANEL_FILE, Path(nm.__file__), Path(nm.ROOT), Path(nm.FINAL_ROOT)):
        nm.guard(path)


def test_map_stub_matches_the_registered_ranges(panel):
    assert nm_map_stub.seed_ranges('wz3to2') == dict(train=nm.TRAIN_RANGE, eval=nm.EVAL_RANGE)
    for row in panel['worlds'][:3]:
        w = row['world']
        assert nm_map_stub.world('wz3to2', w['seed'], panel['cell'], w['repeat']) == w
        assert nm_map_stub.check_world(w, 'eval')
        with pytest.raises(ValueError):
            nm_map_stub.check_world(w, 'train')
