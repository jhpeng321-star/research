"""CPU-only tests for the clean4 wz3to2 final-evaluation launcher. No simulator, no process pool."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nm_finaleval as nm  # noqa: E402
from tools.coopmwm import map_scenarios as ms, rerun_cpus as rc  # noqa: E402
from tools.coopmwm.exact_grade import _seal  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO/nm.MAP_SPEC_REL
STAGED_NET = SPEC.parent.is_dir() and hashlib.sha256((SPEC.parent/'wz3to2.net.xml').read_bytes()).hexdigest() == nm.NET_SHA256


@pytest.fixture(scope='module')
def panel():
    return nm.load_panel()


def make(tmp_path, panel, base='mappo_u0', register=True, mappo=True, **register_kw):
    root, cfg = nm.build_fake_root(tmp_path/'dry', panel, base)
    runtime = nm.FakeRuntime()
    final_root = tmp_path/'dry'/'final_root'
    plan = nm.register(root=root, final_root=final_root, runtime=runtime, panel=panel, **register_kw) if register else None
    bound = nm.bind_fake_mappo(tmp_path/'dry', final_root) if register and mappo else None
    return dict(root=root, cfg=cfg, runtime=runtime, final_root=final_root, plan=plan, panel=panel, mappo=bound)


def write_config(f, cfg):
    (Path(f['root'])/'CONFIG.json').write_text(json.dumps(cfg))
    return cfg


def charge(root, **row):
    with (Path(root)/'CHARGES.jsonl').open('a') as fh:
        fh.write(json.dumps(row)+'\n')


def fake_api(name, **override):
    """A map-API module with the documented interface (defaults: N1 values)."""
    n1 = nm.N1MapAPI(SPEC)
    module = types.ModuleType(name)
    module.STUB = False
    module.seed_ranges = n1.seed_ranges
    module.descriptor = lambda m: dict(name=m, net_sha256=nm.NET_SHA256, stream_namespace=m)
    module.world = n1.world
    module.check_world = n1.check_world
    for k, v in override.items():
        setattr(module, k, v)
    sys.modules[name] = module
    return name


def test_registered_panel_v2_is_the_clean4_cell_on_n1_worlds(panel):
    assert panel['panel_sha256'] == nm.PANEL_SHA256 == '3acb2b565fcb61f4978834b3ba47f0b574fe2d5da7ca028210ef75830c19a130'
    seeds = [r['world']['seed'] for r in panel['worlds']]
    assert seeds == list(range(989600, 989696))
    assert all(r['world'] == ms.world(r['world']['seed'], 3200, 0.6) and r['world_id'] == f'wz3to2_{r["world"]["seed"]}'
               for r in panel['worlds'])
    assert {(r['world']['cell'], r['world']['cell_key']) for r in panel['worlds']} == {(8, 'D3200_p60')}
    assert all(ms.guard(r['world'], 'final_test') for r in panel['worlds'][:3])
    assert panel['cell']['demand_vph'] == {'main': 3200} and panel['cell']['penetration'] == 0.6
    assert panel['map_binding']['net_sha256'] == nm.NET_SHA256 and nm.NET_SHA256.startswith('b6277898')
    contract = panel['contract']
    assert {k: contract['bootstrap'][k] for k in ('reps', 'level', 'seed')} == \
        {k: rc.FINAL_EVALUATION['bootstrap'][k] for k in ('reps', 'level', 'seed')}
    assert contract['horizon_s'] == 1200 and contract['window'] == dict(start_s=0, end_s=1200)
    assert contract['comparisons'] == ['B-A', 'R-A', 'B-R'] and contract['primary']['contrast'] == 'B-BASE'
    assert contract['decisions']['worlds'] == [f'wz3to2_{989600+12*k}' for k in range(8)]
    assert contract['budget']['quote_mappo_u0'] == 672_000 and contract['budget']['quote_native'] == 556_800
    assert panel['supersedes'] == [dict(panel_sha256='b5e1cd2a1d068c5e2bb66bb44005b2dd954e0419b4a6c99778af304aeae6dbc1',
                                        reason=nm.SUPERSEDED['b5e1cd2a1d068c5e2bb66bb44005b2dd954e0419b4a6c99778af304aeae6dbc1'])]


def test_v1_panel_is_superseded_by_a_sealed_note(panel):
    note = json.loads(nm.SUPERSEDED_FILE.read_text())
    nm.verify(note, 'note_sha256', 'supersession note')
    assert note['superseded']['panel_sha256'] in nm.SUPERSEDED and note['superseded_by']['panel_sha256'] == nm.PANEL_SHA256
    assert note['physics_on_superseded'] == 0
    with pytest.raises(nm.Blocked, match='SUPERSEDED'):
        nm.load_panel(nm.HERE/'nm_eval_panel_wz3to2.json', pinned=None)


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
    resealed['map_binding']['net_sha256'] = '0'*64
    path.write_text(json.dumps(_seal({k: v for k, v in resealed.items() if k != 'panel_sha256'}, 'panel_sha256')))
    with pytest.raises(nm.Blocked, match='staged probe network'):
        nm.load_panel(path, pinned=None)


@pytest.mark.skipif(not STAGED_NET, reason='staged probe network bytes (CRLF copy) not in this checkout')
def test_panel_rebuilds_from_n1_and_the_map_check_passes(panel):
    rebuilt = nm.build_panel(SPEC, panel['registered_utc'])
    assert rebuilt == panel and rebuilt['panel_sha256'] == nm.PANEL_SHA256
    report = nm.map_check(nm.N1MapAPI(SPEC), panel)
    assert report['ok'] and not report['stub'] and not report['mismatches'] and report['net_sha256'] == nm.NET_SHA256
    with pytest.raises(nm.Blocked, match='registered N1'):
        nm.build_panel(SPEC, panel['registered_utc'], demand=3000)


def test_map_check_flags_an_api_that_disagrees_with_the_panel(panel):
    assert nm.map_check(nm.bind_map_api(fake_api('fake_n1_agree')), panel)['ok']
    name = fake_api('fake_n1_disagree', seed_ranges=lambda m: dict(train=(989200, 989599), eval=(989600, 989650)),
                    descriptor=lambda m: dict(net_sha256='0'*64, stream_namespace='other'),
                    world=lambda m, seed, cell, repeat: dict(ms.world(seed, 3200, 0.6), cell_key='x'))
    report = nm.map_check(nm.bind_map_api(name), panel)
    assert not report['ok'] and len(report['mismatches']) == 4
    report = nm.map_check(nm.bind_map_api(fake_api('fake_n1_signature', world=lambda m, seed, cell: None)), panel)
    assert not report['ok'] and 'TypeError' in report['mismatches'][0]
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
    assert result['present'] == ['B', 'A', 'R', 'MAPPO', 'NATIVE']
    runs = [c for c in result['calls'] if c[0] == 'run']
    assert runs[0] == ('run', 96*5, 24)
    assert runs[1][1] <= 2*8*(2+nm.MAX_MEMBERS)
    assert {c[1] for c in result['calls'] if c[0] == 'reserve'} == {'v2v_final'}
    assert result['fake_v2v_final_sim_s'] <= result['quote']['total_ceiling_sim_s'] <= 690_000
    final_root = Path(result['final_root'])
    tables = json.loads((final_root/'FINAL_TABLES.json').read_text())
    nm.verify(tables, 'analysis_sha256', 'tables')
    assert tables['dry_run'] is True and tables['missing'] == {} and tables['panel_sha256'] == nm.PANEL_SHA256
    t = tables['tables']
    assert set(t['registered_B_A_R']['rows']) == {'B-A', 'R-A', 'B-R'}
    alias = 'MAPPO' if base == 'mappo_u0' else 'NATIVE'
    assert t['total_benefit']['base_is_native'] is (base == 'native') and t['base']['alias_of'] == alias
    assert t['base']['mappo_comparator']['kind'] == 'retrained_wz3to2'
    assert set(t['total_benefit']['vs_NATIVE']['rows']) == {'B', 'A', 'R', 'MAPPO'}
    assert set(t['total_benefit']['vs_BASE']['rows']) == {'B', 'A', 'R', 'MAPPO', 'NATIVE'}-{alias}
    assert t['arms'][alias]['label'].endswith('= BASE')
    for key in ('B-BASE', 'B-NATIVE'):
        assert t['synergy']['panel_decomposition'][key]['max_identity_error_veh_s'] < 1e-9
    pdx = t['synergy']['per_decision']
    assert pdx['registered_worlds'] == 8
    for arm in ('B', 'R'):
        assert pdx[arm]['decisions']+pdx[arm]['worlds_without_decision'] == 8
        assert pdx[arm]['max_identity_error_veh_s'] < 1e-9
        act = t['synergy']['decision_activity'][arm]
        assert act['worlds'] == 96 and act['subset_captures'] == pdx[arm]['decisions']
        assert act['cooperative_moves'] == 2*act['worlds_with_cooperative_move']
    assert t['synergy']['registered_u24_train_states']['verdict'] == 'pass'
    assert 'row' in t['primary'] and isinstance(t['primary']['meets_rule'], bool)
    text = (final_root/'FINAL_TABLES.md').read_text()
    assert 'FAKE PHYSICS' in text and 'Per-decision decomposition' in text and 'Decision activity' in text
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


def test_runner_without_decision_branches_never_blocks_the_evaluation(panel, tmp_path):
    f = make(tmp_path, panel)
    value = nm.execute(root=f['root'], final_root=f['final_root'], runtime=nm.FakeRuntime(runner=dict(decisions=False)),
                       panel=panel, poll=0, dry_run=True)
    assert value['present'] == ['B', 'A', 'R', 'MAPPO', 'NATIVE']
    assert value['tables']['synergy']['per_decision']['status'] == 'unavailable'
    assert value['tables']['synergy']['decision_activity']['B']['worlds'] == 96
    assert 'DECISIONS' in value['missing']
    assert not [r for r in nm.fe.ledger_rows(f['root']) if r['id'].startswith('nmdec_')]


def test_the_real_runner_module_satisfies_the_contract():
    status = nm.Runtime().runner_status()
    assert status == dict(ok=True, native=True, decisions=False, reason=None)
    status = nm.Runtime('tools.coopmwm.no_such_nm_runner').runner_status()
    assert status['ok'] is False and 'not importable' in status['reason']


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


def test_clean4_binding_refuses_mismatches_superseded_panels_and_reports_unbound_keys(panel, tmp_path):
    f = make(tmp_path, panel, register=False)
    cfg = f['cfg']
    assert nm.clean4_binding(cfg, panel)['unbound'] == []
    for change, match in ((dict(map='weave'), 'map'), (dict(eval_panel_sha256='1'*64), 'panel'),
                          (dict(eval_panel_sha256=next(iter(nm.SUPERSEDED))), 'SUPERSEDED'),
                          (dict(cell=dict(demand_vph={'main': 2800}, penetration=0.3, profile='peak')), 'cell'),
                          (dict(cell='D3200_p40'), 'cell'), (dict(cell=5), 'cell'), (dict(cell='D9999_p99'), 'unreadable'),
                          (dict(cell=dict(demand_vph={'main': 3200}, penetration=0.6, profile='flat')), 'unreadable'),
                          (dict(base_controller='lane_on'), 'base')):
        bad = deepcopy(cfg)
        bad['newmap'].update(change)
        with pytest.raises(nm.Blocked, match=match):
            nm.clean4_binding(bad, panel)
    for cell in ('D3200_p60', 8, dict(demand=3200, penetration=0.6), dict(demand_vph={'main': 3200}, penetration=0.6)):
        assert nm.clean4_binding(dict(cfg, newmap=dict(cfg['newmap'], cell=cell)), panel)['unbound'] == []
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
    assert plan['quote']['total_ceiling_sim_s'] == 672_000 and plan['budget']['allocated'] == 690_000
    assert plan['amendments'][0]['base_alias'] == 'NATIVE' and 'MAPPO' in plan['amendments'][0]['arms']
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


def test_physics_needs_the_map_selection_a_matching_map_api_and_a_runner(panel, tmp_path, monkeypatch):
    f = make(tmp_path, panel)

    class Physical(nm.FakeRuntime):
        physics = True
    monkeypatch.delenv('COOPMWM_MAP_SPEC', raising=False)
    with pytest.raises(nm.Blocked, match='COOPMWM_MAP_SPEC'):
        nm.preflight(f['root'], f['final_root'], Physical(), panel, map_module=fake_api('fake_n1_ok'))
    monkeypatch.setenv('COOPMWM_MAP_SPEC', str(SPEC))
    stub = fake_api('fake_n1_stub', STUB=True)
    with pytest.raises(nm.Blocked, match='map API not qualified'):
        nm.preflight(f['root'], f['final_root'], Physical(), panel, map_module=stub)
    with pytest.raises(nm.Blocked, match='map API not qualified'):
        nm.preflight(f['root'], f['final_root'], Physical(), panel,
                     map_module=fake_api('fake_n1_net', descriptor=lambda m: dict(net_sha256='0'*64)))
    with pytest.raises(nm.Blocked, match='runner unavailable'):
        nm.preflight(f['root'], f['final_root'], Physical(runner=dict(ok=False, reason='absent')), panel,
                     map_module=fake_api('fake_n1_ok'))
    _, _, report = nm.preflight(f['root'], f['final_root'], Physical(), panel, map_module=fake_api('fake_n1_ok'))
    assert report['map_api']['ok']
    with pytest.raises(nm.Blocked, match='RUN_PLAN'):
        nm.execute(root=f['root'], final_root=tmp_path/'unregistered', runtime=f['runtime'], panel=panel, poll=0)


def test_missing_native_requires_explicit_exclusion(panel, tmp_path):
    f = make(tmp_path, panel)
    runtime = nm.FakeRuntime(runner=dict(native=False))
    kw = dict(root=f['root'], final_root=f['final_root'], panel=panel, poll=0, dry_run=True)
    with pytest.raises(nm.Blocked, match='NATIVE'):
        nm.execute(runtime=runtime, **kw)
    assert not nm.fe.ledger_rows(f['root'])[3:]          # nothing charged beyond the fake training rows
    value = nm.execute(runtime=runtime, allow_missing=('NATIVE',), **kw)
    assert value['present'] == ['B', 'A', 'R', 'MAPPO'] and value['tables']['total_benefit']['vs_NATIVE']['status'] == 'unavailable'


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
    assert [s['capture'] is not None for s in specs[:13]] == [True]+[False]*11+[True]
    good = nm.fake_episode(specs[0])
    good['decision'] = dict(t_s=600, move='abreast_wall', members=['main.1', 'main.2', 'main.3'],
                            state_sha256='a'*64, state_ref='states/x.pt')
    good['decision_log'] = [dict(t_s=600, move='abreast_wall', members=['main.1', 'main.2', 'main.3'])]
    assert nm.check_episode(good, specs[0])['decision_log'] == good['decision_log']
    stateless = dict(good, decision=dict(good['decision'], state_sha256=None, state_ref=None))
    assert nm.check_episode(stateless, specs[0])['decision']['state_sha256'] is None
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
                          (lambda r: r['decision'].update(state_sha256='x'), 'capture rule'),
                          (lambda r: r['decision'].update(state_ref=None), 'capture rule'),
                          (lambda r: r.update(decision_log=[dict(t_s=1, move='wall', members=['a'])]), 'decision_log'),
                          (lambda r: r.update(decision_log='x'), 'decision_log')):
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
    with pytest.raises(nm.Blocked, match='only B/R'):
        nm.check_episode(dict(nm.fake_episode(native), decision_log=good['decision_log']), native)


def test_decision_branches_follow_the_registered_decomposition(panel, tmp_path):
    from tools.coopmwm import policy_synergy as ps
    f = make(tmp_path, panel)
    specs = nm.episode_specs('B', panel, f['cfg'], f['root'], 'mappo_u0')
    records = {}
    for i, s in enumerate(specs):
        r = nm.fake_episode(s)
        if s['capture'] is not None:
            r['decision'] = None if i % 24 == 0 else dict(t_s=400+i, move='closing_pair', members=[f'main.{i}', f'main.{i+1}'],
                                                          state_sha256=f'{i:064x}', state_ref=f'states/{i}.pt')
        records[s['world_id']] = nm.check_episode(r, s)
    branches = nm.decision_specs('B', records, panel)
    assert len(branches) == 4*4 and {b['world_id'] for b in branches} <= set(panel['contract']['decisions']['worlds'])
    first = [b for b in branches if b['world_id'] == specs[12]['world_id']]
    assert [b['label'] for b in first] == ['all', 'background', 'solo0', 'solo1']
    assert [b['new_members'] for b in first] == [['main.12', 'main.13'], [], ['main.12'], ['main.13']]
    assert all(b['window'] == dict(start_s=412, end_s=772) and b['sim_s'] == 772 and b['background'] == 'BASE' for b in first)
    J = dict(all=30_000.0, background=33_000.0, solo0=31_000.0, solo1=32_500.0)
    rows = {b['label']: nm.check_branch(dict(nm.fake_branch(b), J_veh_s=J[b['label']]), b) for b in first}
    mine = nm.decision_metrics(rows)
    ref = ps.metrics(J['background'], [dict(status='graded', J_veh_s=J[k]) for k in ('all', 'solo0', 'solo1')],
                     ['main.12', 'main.13'])
    assert mine['delta_all_new_veh_s'] == ref['delta_all_new_veh_s'] == 3_000
    assert mine['policy_cooperation_gain_veh_s'] == ref['policy_cooperation_gain_veh_s'] == 1_000
    assert mine['delta_only_member_new_veh_s'] == ref['delta_only_member_new_veh_s'] and mine['identity_error_veh_s'] == 0
    with pytest.raises(nm.Blocked, match='identity'):
        nm.check_branch(dict(nm.fake_branch(first[0]), new_members=[]), first[0])
    incomplete = [nm.check_branch(nm.fake_branch(b), b) for b in first[:3]]
    with pytest.raises(nm.Blocked, match='incomplete'):
        nm.analyse_decisions(dict(B=incomplete), panel['contract']['bootstrap'])
    stateless = dict(records)
    stateless[specs[12]['world_id']] = dict(records[specs[12]['world_id']], decision=dict(
        records[specs[12]['world_id']]['decision'], state_sha256=None, state_ref=None))
    with pytest.raises(nm.Blocked, match='no branchable state'):
        nm.decision_specs('B', stateless, panel)


def test_budget_quote_fits_the_clean4_v2v_final_allocation():
    for base in nm.BASES:
        q = nm.quote(base)
        assert q['episodes_sim_s'] == 5*96*1200 == 576_000 and q['decisions_ceiling_sim_s'] == 2*8*5*1200 == 96_000
        assert q['total_ceiling_sim_s'] == 672_000 and q['slack_sim_s'] == 18_000 and q['fits']
        assert set(q['need_ceiling']) == {f'episodes_{a}' for a in nm.EPISODE_ARMS} | {'decisions_B', 'decisions_R'}
    with pytest.raises(nm.Blocked):
        nm.quote('lane_on')


def test_forbidden_paths_are_refused_and_never_used(tmp_path):
    for token in ('heldout', 'holdout', 'sealed', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth'):
        with pytest.raises(nm.Blocked):
            nm.guard(tmp_path/token/'x.json')
    for path in (nm.PANEL_FILE, nm.SUPERSEDED_FILE, Path(nm.__file__), Path(nm.ROOT), Path(nm.FINAL_ROOT)):
        nm.guard(path)


def test_mappo_comparator_is_a_bound_parameter_verified_at_load(panel, tmp_path):
    f = make(tmp_path, panel, mappo=False)
    fake = tmp_path/'n7_mappo.pt'
    fake.write_bytes(b'n7 stand-in')
    sha = nm.file_sha(fake)
    kw = dict(final_root=f['final_root'], path=fake, authority='N7: retrain done')
    for bad, match in ((dict(sha256='0'*64, kind='retrained_wz3to2'), 'does not hash'),
                       (dict(sha256=sha[:12], kind='retrained_wz3to2'), 'full lowercase'),
                       (dict(sha256=sha, kind='laneon'), 'kind'),
                       (dict(sha256=sha, kind='u0_transferred_not_retrained'), 'd9042cc5')):
        with pytest.raises(nm.Blocked, match=match):
            nm.bind_mappo(**kw, **bad)
    with pytest.raises(nm.Blocked, match='authority'):
        nm.bind_mappo(final_root=f['final_root'], path=fake, sha256=sha, kind='retrained_wz3to2', authority=' ')
    first = nm.bind_mappo(**kw, sha256=sha, kind='retrained_wz3to2')
    assert nm.mappo_binding(f['final_root']) == first and first['label'] == nm.MAPPO_KINDS['retrained_wz3to2']
    second = nm.bind_mappo(**kw, sha256=sha, kind='retrained_wz3to2')
    assert (f['final_root']/'MAPPO_COMPARATOR_superseded_001.json').is_file() and nm.mappo_binding(f['final_root']) == second
    spec = nm.episode_specs('MAPPO', panel, f['cfg'], f['root'], 'mappo_u0', second)[0]
    assert spec['job_id'] == 'nmfinal_mappo_0000' and spec['checkpoint']['executor_sha256'] == sha
    assert spec['checkpoint']['path'].endswith('A_checkpoint_000.pt') and spec['checkpoint']['update'] == 0
    assert nm.MAPPO_KINDS['u0_transferred_not_retrained'].endswith('(u0 transferred, not retrained)')
    nm.publish(f['final_root']/'RESERVED_MAPPO.json', dict(jobs=[]))
    with pytest.raises(nm.Blocked, match='already started'):
        nm.bind_mappo(**kw, sha256=sha, kind='retrained_wz3to2')


def test_unbound_mappo_never_blocks_the_other_arms_and_is_added_later(panel, tmp_path):
    f = make(tmp_path, panel, mappo=False)
    kw = dict(root=f['root'], final_root=f['final_root'], panel=panel, poll=0, dry_run=True)
    value = nm.execute(runtime=f['runtime'], **kw)
    assert value['present'] == ['B', 'A', 'R', 'NATIVE'] and set(value['missing']) >= {'MAPPO', 'BASE'}
    assert value['tables']['primary']['status'] == 'unavailable'
    assert value['tables']['total_benefit']['vs_BASE']['status'] == 'unavailable'
    charged = len(nm.fe.ledger_rows(f['root']))
    nm.bind_fake_mappo(tmp_path/'dry', f['final_root'])
    runtime = nm.FakeRuntime()
    value = nm.execute(runtime=runtime, **kw)
    assert value['present'] == ['B', 'A', 'R', 'MAPPO', 'NATIVE'] and 'MAPPO' not in value['missing']
    assert [c for c in runtime.calls if c[0] == 'run'] == [('run', 96, 24)]          # only the MAPPO wave
    assert len(nm.fe.ledger_rows(f['root'])) == charged+96
    assert (f['final_root']/'FINAL_TABLES.json').is_file() and len(list(f['final_root'].glob('FINAL_TABLES_*.json'))) == 1
