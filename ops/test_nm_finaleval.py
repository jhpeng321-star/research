"""CPU-only tests for the clean4 wz3to2 final-evaluation launcher. No simulator, no process pool."""
from copy import deepcopy
import hashlib
import json
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
    bound = ({a: nm.bind_fake_mappo(tmp_path/'dry', final_root, a, residual=root/'A_checkpoint_000.pt')
              for a in nm.BOUND_ARMS}
             if register and mappo else None)
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
    assert contract['budget']['allocation_sim_s'] == nm.PANEL_ALLOCATION_SIM_S == 690_000   # as registered
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
    # Only the informational descriptor digest moved (N1_READY extended the descriptor record after the
    # registration: f22a6ce2 -> 6fc6be65); worlds, cell, raw/semantic net and spec digests are identical.
    assert panel['map_binding']['descriptor_sha256_at_registration'].startswith('f22a6ce2')
    assert rebuilt['map_binding']['descriptor_sha256_at_registration'] == nm.N1MapAPI(SPEC).descriptor('wz3to2')['descriptor_sha256']
    rebuilt['map_binding']['descriptor_sha256_at_registration'] = panel['map_binding']['descriptor_sha256_at_registration']
    rebuilt = _seal({k: v for k, v in rebuilt.items() if k != 'panel_sha256'}, 'panel_sha256')
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


@pytest.mark.parametrize('base', ['mappo_frozen', 'native'])
def test_dry_run_end_to_end_is_zero_physics(panel, tmp_path, monkeypatch, base):
    import concurrent.futures

    class Forbidden:
        def __init__(self, *a, **k):
            raise AssertionError('dry run must never open a process pool')
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Forbidden)
    result = nm.dry_run(out=tmp_path/'out', panel=panel, base=base)
    assert result['status'] == 'PASS' and result['new_sim_s'] == 0 and result['physics'] is False
    assert set(result['analyses']) == {'claim3a', 'claim3b_MAPPO_SPEED', 'claim3b_MAPPO_LANE', 'claim3c'}
    assert result['present']['claim3a'] == ['B', 'A', 'R', 'RULE', 'RULE_FAIR', 'U0', 'NATIVE']
    assert result['present']['claim3b_MAPPO_SPEED'] == ['B', 'A', 'R', 'RULE', 'RULE_FAIR', 'U0', 'MAPPO_SPEED', 'NATIVE']
    assert result['present']['claim3c'] == ['B', 'R', 'RULE', 'RULE_FAIR', 'U0', 'NATIVE']
    assert set(result['secondary']) == set(result['analyses'])                    # u24 SECONDARY, same panels
    runs = [c for c in result['calls'] if c[0] == 'run']
    # pre-run NATIVE/U0/RULE (3a + 3c) in chunks of 2 x 16; u12 PRIMARY: only B/R (+ 3c B/R) - A is the frozen U0
    # alias - then each comparator's F11 parity + its wave; u24 SECONDARY: only B/R again (everything else reused)
    assert runs[0] == ('run', 2, 2)                                   # the one-episode logcheck (NATIVE, RULE)
    runs = runs[1:]
    k = result['prerun']['rounds']
    assert result['prerun']['status'] == 'PASS' and k == -(-4*(96+24)//32)
    acc, i = 0, 0
    while acc < 4*(96+24):
        acc, i = acc+runs[i][1], i+1
    pre, runs = runs[:i], runs[i:]
    # MAIN process rule: every dispatch = one canary job per arm (own processes) verified, then the rest (16 workers)
    assert acc == 4*(96+24) and all(r[2] == 16 for r in pre[1::2]) and all(r[1] == r[2] <= 4 for r in pre[0::2])
    assert runs == ([('run', 2, 2), ('run', 2*96+2*24-2, 24)]
                    + [('run', 1, 1), ('run', 3, 24), ('run', 1, 1), ('run', 95, 24)]*2
                    + [('run', 2, 2), ('run', 2*96+2*24-2, 24)])
    assert {c[1] for c in result['calls'] if c[0] == 'reserve'} == {'v2v_final', 'v2v_final_prerun', 'v2v_final_logcheck'}
    assert result['fake_prerun_sim_s'] == nm.prerun_cap() == 576_000 == result['quote']['prerun_sim_s']
    assert (result['fake_v2v_final_sim_s']+result['fake_prerun_sim_s'] == result['quote']['total_ceiling_sim_s']
            == 1_385_760 <= nm.ALLOCATION_SIM_S == 1_600_000)
    final_root = Path(result['final_root'])
    alias = dict(mappo_frozen='A', mappo_u0='U0', native='NATIVE')[base]
    claim3a = json.loads((final_root/'CLAIM3A_TABLES_u012.json').read_text())
    nm.verify(claim3a, 'analysis_sha256', 'claim3a')
    assert claim3a['scope'] == 'claim3a' and claim3a['dry_run'] is True and claim3a['panel_sha256'] == nm.PANEL_SHA256
    assert claim3a['update'] == 12 and claim3a['evaluation'] == 'PRIMARY (u12)' and claim3a['a_alias_of_u0'] is True
    assert set(claim3a['missing']) == {'MAPPO_SPEED', 'MAPPO_LANE', 'DECISIONS'} and 'MAPPO_SPEED' not in claim3a['present']
    assert claim3a['missing']['DECISIONS'].startswith('unavailable (not built)')
    assert claim3a['sources']['A']['alias_of'] == 'U0' and 'B_u012' in claim3a['sources']
    secondary = json.loads((final_root/'CLAIM3A_TABLES_u024.json').read_text())
    assert secondary['update'] == 24 and secondary['evaluation'] == 'SECONDARY (u24)' and 'B_u024' in secondary['sources']
    assert secondary['tables']['arms']['B']['label'].startswith('B u24')
    t = claim3a['tables']
    arms = set(nm.CLAIM3A_ARMS)
    assert t['base']['alias_of'] == alias and t['arms'][alias]['label'].endswith('= BASE')
    assert set(t['total_benefit']['vs_NATIVE']['rows']) == arms-{'NATIVE'}
    assert set(t['total_benefit']['vs_BASE']['rows']) == arms-{alias}
    assert t['total_benefit']['vs_MAPPO_SPEED']['status'] == 'unavailable'      # no comparator data in 3a
    assert set(t['registered_B_A_R']['rows']) == {'B-A', 'R-A', 'B-R'}
    for key in ('B-BASE', 'B-NATIVE'):
        assert t['synergy']['panel_decomposition'][key]['max_identity_error_veh_s'] < 1e-9
    assert t['synergy']['per_decision']['status'] == 'unavailable'                 # F7 not built (MAIN)
    assert all(t['synergy']['decision_activity'][arm]['worlds'] == 96 for arm in ('B', 'R'))
    assert t['arms']['A']['label'].startswith('A = U0')
    cp = t['co_primary']                                               # owner fairness decision (04:0x)
    assert set(cp['groups']['groups']) == set(nm.REPORTED_GROUPS) and cp['groups']['reference'] == 'U0'
    assert cp['groups']['vehicles_missing']['B'] == dict(not_in_reference=0, reference_not_in_arm=0)
    assert not cp['groups']['groups']['origin:undefined'].get('registered', False)
    assert set(cp['groups']['sensitivity_own_arm_labels']['groups']) == set(nm.REPORTED_GROUPS)
    assert cp['total_B_minus_A']['diff'] == pytest.approx(t['registered_B_A_R']['rows']['B-A']['diff'])
    assert cp['meets'] == (cp['total_lower'] and cp['groups']['all_pass'])
    assert 'RULE_FAIR' in t['arms'] and t['arms']['RULE_FAIR']['label'].startswith('RULE_FAIR')
    sec = t['secondary']                                              # SECONDARY report-only (amendment 5)
    assert sec['status'].startswith('report-only') and sec['bootstrap']['reps'] == 200      # dry run: fewer reps
    assert set(sec['metrics']) == {m for _, m, _ in nm.SECONDARY_METRICS} and set(sec['delay_by_origin_s']) == {'main_in', 'ramp_in'}
    gini = sec['metrics']['delay_gini']
    assert set(gini['vs']) == {'BASE', 'NATIVE', 'U0', 'RULE', 'RULE_FAIR'} and len(gini['per_world']['B']) == 96
    assert set(gini['vs']['NATIVE']) == set(t['arms'])-{'NATIVE'} and 'mean_ci' in gini['vs']['NATIVE']['B']
    assert t['total_benefit']['vs_RULE']['rows']['A'] == t['total_benefit']['vs_RULE']['rows']['U0']   # A is U0
    ps = t['per_seed']
    assert ps['comparators']['MAPPO_SPEED'] == dict(status='unavailable')
    for c in ('NATIVE', 'U0', 'RULE'):
        rows = ps['comparators'][c]
        assert set(rows) == arms-{c}
        for r in rows.values():
            assert r['n'] == 96 and len(r['sorted']) == 96 and r['frac_benefit']+r['frac_harm']+r['frac_tie'] == pytest.approx(1)
            assert [x['diff_veh_s'] for x in r['sorted']] == sorted(x['diff_veh_s'] for x in r['sorted'])
    assert ps['B-R']['mean_ci']['diff'] == pytest.approx(t['registered_B_A_R']['rows']['B-R']['diff'])
    assert len(ps['table_B']['rows']) == 96
    rg = t['regimes']
    assert rg['report_only'] and rg['demand_cell']['D3200_p60']['n'] == 96
    assert sum(b['n'] for b in rg['closing_lane_speed_tercile']['terciles'].values()) == 96
    text = (final_root/'CLAIM3A_TABLES_u012.md').read_text()
    assert 'FAKE PHYSICS' in text and 'T5 Per-seed view' in text and 'random' not in text
    speed = json.loads((final_root/'CLAIM3B_MAPPO_SPEED_TABLES_u012.json').read_text())
    assert speed['scope'] == 'claim3b_MAPPO_SPEED' and speed['analysis_sha256'] != claim3a['analysis_sha256']
    ts = speed['tables']
    assert set(ts['total_benefit']['vs_MAPPO_SPEED']['rows']) == arms
    assert ts['base']['comparators']['MAPPO_SPEED']['kind'] == 'retrained_wz3to2_it168'
    assert 'S-MAPPOWZ it168' in ts['arms']['MAPPO_SPEED']['label']
    assert set(ts['per_seed']['comparators']['MAPPO_SPEED']) == arms
    assert ts['registered_B_A_R'] == t['registered_B_A_R']                       # 3a rows unchanged in 3b
    status = json.loads((final_root/'STATUS.json').read_text())
    assert status['step'] == 'DONE_u024' and status['status'] == 'PASS'


def test_rerun_reuses_published_waves_and_charges_nothing(panel, tmp_path):
    result = nm.dry_run(out=tmp_path/'out', panel=panel)
    root, final_root = Path(result['out'])/'clean4_root', Path(result['final_root'])
    before = (root/'CHARGES.jsonl').read_text()
    runtime = nm.FakeRuntime()
    value = nm.execute(root=root, final_root=final_root, runtime=runtime, panel=panel, panel_3c=nm.load_panel_3c(),
                       poll=0, dry_run=True)
    assert {k: v['analysis_sha256'] for k, v in value.items()} == result['analyses']
    assert not [c for c in runtime.calls if c[0] in ('run', 'reserve')] and (root/'CHARGES.jsonl').read_text() == before


def test_runner_without_decision_branches_never_blocks_the_evaluation(panel, tmp_path):
    f = make(tmp_path, panel)
    value = nm.execute(root=f['root'], final_root=f['final_root'], runtime=nm.FakeRuntime(runner=dict(decisions=False)),
                       panel=panel, poll=0, dry_run=True)['claim3a']
    assert value['present'] == ['B', 'A', 'R', 'RULE', 'RULE_FAIR', 'U0', 'NATIVE']
    assert value['tables']['synergy']['per_decision']['status'] == 'unavailable'
    assert value['tables']['synergy']['decision_activity']['B']['worlds'] == 96
    assert value['missing']['DECISIONS'].startswith('unavailable (not built): MAIN decision')
    assert not [r for r in nm.fe.ledger_rows(f['root']) if r['id'].startswith('nmdec_')]


def test_the_real_runner_module_satisfies_the_contract():
    status = nm.Runtime().runner_status()
    assert status == dict(ok=True, native=True, decisions=False, lane=True, rule=True, rule_fair=True, parity=True, reason=None)
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
    assert nm.clean4_binding(dict(cfg, newmap=dict(cfg['newmap'], base_controller='mappo_frozen')), panel)['base'] == 'mappo_frozen'
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
    assert plan['quote']['total_ceiling_sim_s'] == 1_385_760 and plan['budget']['sub_cap'] == 1_600_000
    assert plan['amendments'][0]['base_alias'] == 'NATIVE' and plan['amendments'][0]['arms'] == list(nm.EPISODE_ARMS)
    _, binding, _ = nm.preflight(f['root'], f['final_root'], f['runtime'], panel)
    assert binding['base'] == 'native'
    with pytest.raises(nm.Blocked, match='never overwrite'):
        nm.register(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel, base='native')


def test_budget_is_debited_from_clean4_v2v_final_only_within_its_allocation(panel, tmp_path):
    f = make(tmp_path, panel, register=False)
    cfg = f['cfg']
    report = nm.budget_check(f['root'], cfg, 'mappo_u0')
    assert report['source'] == 'clean4_budget' and report['sub_cap'] == 1_600_000 and report['remaining'] == 1_600_000
    small = dict(cfg, clean4_budget=_seal(dict(phases=dict(v2v_final=1_340_000)), 'receipt_sha256'))   # clean5a r3 value
    report = nm.budget_check(f['root'], small, 'mappo_u0')       # reported, not binding: N6 accounts in its own root
    assert report['allocated'] == 1_340_000 and report['sub_cap'] == 1_600_000
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


def test_sub_cap_refuses_a_wave_that_would_exceed_the_allocation(panel, tmp_path):
    f = make(tmp_path, panel)
    charge(f['root'], id='nmfinal_old_attempt_0000', phase='v2v_final', sim_s=1_500_000)
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
    value = nm.execute(runtime=runtime, allow_missing=('NATIVE',), **kw)['claim3a']
    assert value['present'] == ['B', 'A', 'R', 'RULE', 'RULE_FAIR', 'U0']
    assert value['tables']['total_benefit']['vs_NATIVE']['status'] == 'unavailable'


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


def test_budget_quote_fits_the_raised_v2v_final_allocation():
    for base in nm.BASES:
        q = nm.quote(base)                                        # clean5a: A = the frozen U0 alias
        need = q['need_ceiling']
        assert q['prerun_sim_s'] == 4*(96+24)*1200 == 576_000 and q['decisions_ceiling_sim_s'] == need['decisions'] == 0
        assert need['u012_B'] == need['u024_R'] == 96*1200 and need['u012_C3_B'] == need['u024_C3_R'] == 24*1200
        assert 'u012_A' not in need and need['episodes_MAPPO_SPEED'] == 96*1200 and q['parity_sim_s'] == 2*2*2*420 == 3_360
        assert q['total_ceiling_sim_s'] == 576_000+2*2*(96+24)*1200+2*96*1200+3_360 == 1_385_760
        assert q['allocation_sim_s'] == 1_600_000 and q['fits'] and q['margin_over_total'] == pytest.approx(.1546, abs=1e-4)
        assert nm.ALLOCATION_SIM_S >= q['total_ceiling_sim_s']
        full = nm.quote(base, a_alias=False)                      # A simulated at both updates
        assert full['total_ceiling_sim_s'] == 1_385_760+2*96*1200 and not full['fits']
    with pytest.raises(nm.Blocked):
        nm.quote('lane_on')


def test_forbidden_paths_are_refused_and_never_used(tmp_path):
    for token in ('heldout', 'holdout', 'sealed', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth'):
        with pytest.raises(nm.Blocked):
            nm.guard(tmp_path/token/'x.json')
    for path in (nm.PANEL_FILE, nm.SUPERSEDED_FILE, Path(nm.__file__), Path(nm.ROOT), Path(nm.FINAL_ROOT)):
        nm.guard(path)


def export(path, *, iteration=168, lane_head=False, trainer='mappo'):
    import torch
    policy = {'actor.0.weight': torch.zeros(1)}
    if lane_head:
        policy['lane_head.weight'] = torch.zeros(1)
    torch.save(dict(format='c4_dreamer/3', trainer=trainer, iteration=iteration, lane_head=lane_head, policy=policy), path)
    return nm.file_sha(path)


def test_comparators_follow_the_registered_binding(panel, tmp_path):
    f = make(tmp_path, panel, mappo=False)
    it168, it84 = tmp_path/'it168.pt', tmp_path/'it84.pt'
    lane, lane_less = tmp_path/'lane_final.pt', tmp_path/'speed_only.pt'
    sha, sha84 = export(it168), export(it84, iteration=84)
    sha_lane, sha_less = export(lane, iteration=500, lane_head=True), export(lane_less, iteration=500)
    kw = dict(final_root=f['final_root'], authority='N7: it168 done')
    speed = dict(arm='MAPPO_SPEED', kind='retrained_wz3to2_it168')
    for bad, match in ((dict(path=it168, sha256='0'*64, **speed), 'does not hash'),
                       (dict(path=it168, sha256=sha[:12], **speed), 'full lowercase'),
                       (dict(path=it84, sha256=sha84, **speed), 'it168 only'),                 # no it84 substitution
                       (dict(path=it168, sha256=sha, arm='MAPPO_SPEED', kind='retrained_wz3to2'), 'kind'),
                       (dict(path=lane, sha256=sha_lane, **speed), 'it168 only'),
                       (dict(path=lane_less, sha256=sha_less, arm='MAPPO_LANE', kind='lane_retrained_wz3to2_final'),
                        'lane head'),
                       (dict(path=it168, sha256=sha, arm='MAPPO_X', kind='x'), 'comparator arm')):
        with pytest.raises(nm.Blocked, match=match):
            nm.bind_mappo(**kw, **bad)
    for arm, prefix in (('MAPPO_SPEED', 'd9042cc5'), ('MAPPO_LANE', 'aecd7d70'), ('MAPPO_LANE', 'd9042cc5')):
        with pytest.raises(nm.Blocked, match='never substitutes'):
            nm.check_export(arm, prefix+'0'*56, dict(trainer='mappo', iteration=168, lane_head=arm == 'MAPPO_LANE'))
    with pytest.raises(nm.Blocked, match='frozen d9042cc5 executor only'):
        nm.bind_mappo(**kw, path=it168, sha256=sha, arm='U0', kind='u0_transferred_d9042cc5',
                      residual_path=f['root']/'A_checkpoint_000.pt', residual_sha256=f['cfg']['common_checkpoint_sha256'])
    with pytest.raises(nm.Blocked, match='authority'):
        nm.bind_mappo(final_root=f['final_root'], path=it168, sha256=sha, authority=' ', **speed)
    first = nm.bind_mappo(**kw, path=it168, sha256=sha, **speed)
    assert first['export']['iteration'] == 168 and nm.mappo_binding(f['final_root']) == first
    second = nm.bind_mappo(**kw, path=it168, sha256=sha, **speed)
    assert (f['final_root']/'BOUND_MAPPO_SPEED_superseded_001.json').is_file() and nm.mappo_binding(f['final_root']) == second
    with pytest.raises(nm.Blocked, match='needs the U0 binding'):
        nm.episode_specs('MAPPO_SPEED', panel, f['cfg'], f['root'], 'mappo_u0', second)
    bound_u0 = nm.bind_fake_mappo(tmp_path/'dry', f['final_root'], 'U0', residual=f['root']/'A_checkpoint_000.pt')
    spec = nm.episode_specs('MAPPO_SPEED', panel, f['cfg'], f['root'], 'mappo_u0', second, bound_u0)[0]
    assert spec['job_id'] == 'nmfinal_mappo_speed_0000' and spec['checkpoint']['executor_sha256'] == sha
    assert spec['checkpoint']['path'].endswith('A_checkpoint_000.pt') and spec['checkpoint']['update'] == 0
    with pytest.raises(nm.Blocked, match='needs the U0 binding'):
        nm.episode_specs('U0', panel, f['cfg'], f['root'], 'mappo_frozen')
    assert bound_u0['dry_run'] is True and bound_u0['export']['dry_run_stand_in'] is True
    assert bound_u0['residual_sha256'] == f['cfg']['common_checkpoint_sha256']
    u0 = nm.episode_specs('U0', panel, f['cfg'], f['root'], 'mappo_frozen', None, bound_u0)[0]
    assert u0['job_id'] == 'nmfinal_u0_0000' and u0['checkpoint']['kind'] == 'u0'
    bound_lane = nm.bind_mappo(**kw, path=lane, sha256=sha_lane, arm='MAPPO_LANE', kind='lane_retrained_wz3to2_final')
    spec = nm.episode_specs('MAPPO_LANE', panel, f['cfg'], f['root'], 'mappo_u0', bound_lane)[0]
    assert spec['job_id'] == 'nmfinal_mappo_lane_0000' and spec['checkpoint']['kind'] == 'mappo_lane'
    nm.publish(f['final_root']/'RESERVED_MAPPO_SPEED.json', dict(jobs=[]))
    with pytest.raises(nm.Blocked, match='already started'):
        nm.bind_mappo(**kw, path=it168, sha256=sha, **speed)
    with pytest.raises(nm.Blocked, match='is bound'):
        nm.declare_unavailable(final_root=f['final_root'], arm='MAPPO_LANE', authority='MAIN', reason='x')


def test_claim3a_opens_first_and_each_claim3b_only_after_its_comparator_is_bound(panel, tmp_path):
    f = make(tmp_path, panel, mappo=False)
    kw = dict(root=f['root'], final_root=f['final_root'], panel=panel, poll=0, dry_run=True)
    with pytest.raises(nm.NotReady, match='pending U0'):                    # U0 is part of claim 3a
        nm.execute(runtime=f['runtime'], **kw)
    assert not [c for c in f['runtime'].calls if c[0] in ('run', 'reserve')]
    nm.bind_fake_mappo(tmp_path/'dry', f['final_root'], 'U0', residual=f['root']/'A_checkpoint_000.pt')
    with pytest.raises(nm.NotReady, match='pending MAPPO_SPEED'):
        nm.execute(runtime=f['runtime'], **kw)
    claim3a = json.loads((f['final_root']/'CLAIM3A_TABLES_u012.json').read_text())   # 3a opened, sealed on its own
    assert claim3a['present'] == ['B', 'A', 'R', 'RULE', 'RULE_FAIR', 'U0', 'NATIVE']
    assert not list(f['final_root'].glob('WAVE_MAPPO_*.json')) and not list(f['final_root'].glob('CLAIM3B_*'))
    with pytest.raises(nm.Blocked, match='authority and reason'):
        nm.declare_unavailable(final_root=f['final_root'], arm='MAPPO_LANE', authority='MAIN', reason=' ')
    nm.declare_unavailable(final_root=f['final_root'], arm='MAPPO_LANE', authority='MAIN 2026-11-08 (test)',
                           reason='S-MAPPOWZL not delivered')
    with pytest.raises(nm.Blocked, match='declared unavailable'):
        nm.bind_fake_mappo(tmp_path/'dry', f['final_root'], 'MAPPO_LANE')
    nm.bind_fake_mappo(tmp_path/'dry', f['final_root'], 'MAPPO_SPEED')
    runtime = nm.FakeRuntime()
    value = nm.execute(runtime=runtime, **kw)
    # only the new comparator: its F11 parity (4 traces) and its wave
    assert [c for c in runtime.calls if c[0] == 'run'] == [('run', 1, 1), ('run', 3, 24), ('run', 1, 1), ('run', 95, 24)]
    assert value['claim3a']['analysis_sha256'] == claim3a['analysis_sha256']     # 3a reproduces from its receipts
    assert value['claim3b_MAPPO_SPEED']['present'] == ['B', 'A', 'R', 'RULE', 'RULE_FAIR', 'U0', 'MAPPO_SPEED', 'NATIVE']
    lane = json.loads((f['final_root']/'CLAIM3B_MAPPO_LANE_UNAVAILABLE.json').read_text())
    assert lane['status'] == 'declared unavailable' and 'S-MAPPOWZL' in lane['declaration']['reason']
    published = nm._analyse_published(f['root'], f['final_root'], runtime, panel)
    assert set(published) == {'claim3a', 'claim3b_MAPPO_SPEED'}


def test_a_b_r_are_bound_by_sha_before_the_gate(panel, tmp_path):
    f = make(tmp_path, panel)
    root, cfg, final_root = f['root'], f['cfg'], f['final_root']
    shas = {a: nm.file_sha(root/f'{a}_checkpoint_012.pt') for a in ('B', 'R')}
    with pytest.raises(nm.Blocked, match='sha256 required'):
        nm.bind_arms(final_root=final_root, shas=dict(shas, R='x'), authority='MAIN', update=12)
    with pytest.raises(nm.Blocked, match='sha256 required'):
        nm.bind_arms(final_root=final_root, shas=dict(B=shas['B']), authority='MAIN', update=12)   # R required
    with pytest.raises(nm.Blocked, match='one of'):
        nm.bind_arms(final_root=final_root, shas=shas, authority='MAIN', update=18)
    nm.bind_arms(final_root=final_root, shas=shas, authority='MAIN 23:10: bind clean5a B/R by sha at u12', update=12)
    binding = nm.clean4_binding(cfg, panel)
    gated = nm.gate(root, cfg, f['runtime'], binding, final_root, 12)
    assert {a: gated['checkpoint_sha256'][a] for a in ('B', 'R')} == shas and gated['update'] == 12
    assert gated['a_alias_of_u0']['identical'] is True and gated['monitors'] == 12
    with pytest.raises(nm.Blocked, match='never rebound'):
        nm.bind_arms(final_root=final_root, shas=shas, authority='again', update=12)
    nm.bind_arms(final_root=final_root, shas={a: nm.file_sha(root/f'{a}_checkpoint_024.pt') for a in nm.ARMS},
                 authority='MAIN: u24 SECONDARY', update=24)                                      # A optional, separate file
    (root/'B_checkpoint_012.pt').write_bytes(b'another B')
    with pytest.raises(nm.Blocked, match='differ from the bound'):
        nm.gate(root, cfg, f['runtime'], binding, final_root, 12)
    (root/'A_COMPLETE.json').unlink()
    with pytest.raises(nm.NotReady, match='A_COMPLETE'):                                     # COMPLETE only at u24
        nm.gate(root, cfg, f['runtime'], binding, final_root, 24)
    f['runtime'].frozen_a_identical = False
    with pytest.raises(nm.Blocked, match='frozen control arm A differs'):
        nm.gate(root, cfg, f['runtime'], binding, None, 12)


def test_bound_lane_comparator_without_runner_support_needs_explicit_exclusion(panel, tmp_path):
    f = make(tmp_path, panel)
    runtime = nm.FakeRuntime(runner=dict(lane=False))
    kw = dict(root=f['root'], final_root=f['final_root'], panel=panel, poll=0, dry_run=True)
    with pytest.raises(nm.Blocked, match='MAPPO_LANE'):
        nm.execute(runtime=runtime, **kw)
    assert (f['final_root']/'CLAIM3A_TABLES_u012.json').is_file()      # 3a and 3b-speed were not held back
    value = nm.execute(runtime=runtime, allow_missing=('MAPPO_LANE',), **kw)
    assert 'claim3b_MAPPO_LANE' not in value and 'claim3b_MAPPO_SPEED' in value


# ---------------------------------------------------------------- RULE arm and the sealed pre-run

def test_rule_arm_is_the_registered_s_regimeu0_rule(panel, tmp_path):
    from tools.coopmwm import nm_eval_runner as runner, regimeu0
    assert runner.rule_digest() == nm.RULE_SHA256 and regimeu0.STRATEGIES[nm.RULE_STRATEGY] == 2
    f = make(tmp_path, panel)
    u0 = f['mappo']['U0']
    with pytest.raises(nm.Blocked, match='needs the U0 binding'):
        nm.episode_specs('RULE', panel, f['cfg'], f['root'], 'mappo_u0')
    spec = nm.episode_specs('RULE', panel, f['cfg'], f['root'], 'mappo_u0', None, u0)[0]
    ck = spec['checkpoint']
    assert spec['job_id'] == 'nmfinal_rule_0000' and spec['capture'] is None
    assert (ck['kind'], ck['rule'], ck['rule_sha256'], ck['arm'], ck['update']) == ('rule', 'wall_h_solo2', nm.RULE_SHA256, 'A', 0)
    assert ck['executor_sha256'] == u0['sha256'] and ck['file_sha256'] == f['cfg']['common_checkpoint_sha256']
    record = nm.fake_episode(spec)
    assert nm.check_episode(record, spec)['rule']['strategy'] == 'wall_h_solo2'
    with pytest.raises(nm.Blocked, match='rule actually loaded'):
        nm.check_episode(dict(record, rule_sha256_loaded='0'*64), spec)
    with pytest.raises(nm.Blocked, match='never silent'):
        nm.check_episode(dict(record, executor_sha256_loaded='0'*64), spec)
    u0spec = nm.episode_specs('U0', panel, f['cfg'], f['root'], 'mappo_u0', None, u0)[0]
    with pytest.raises(nm.Blocked, match='rule telemetry'):
        nm.check_episode(dict(nm.fake_episode(u0spec), rule={}), u0spec)
    assert 'RULE' in nm.CLAIM3A_ARMS and 'RULE' in nm.CLAIM3C_ARMS and 'RULE' in nm.PER_SEED_COMPARATORS
    runtime = nm.FakeRuntime(runner=dict(rule=False))
    kw = dict(root=f['root'], final_root=f['final_root'], panel=panel, poll=0, dry_run=True)
    with pytest.raises(nm.Blocked, match='RULE'):
        nm.execute(runtime=runtime, **kw)
    value = nm.execute(runtime=runtime, allow_missing=('RULE',), **kw)['claim3a']
    assert 'RULE' not in value['present'] and 'S-REGIMEU0' in value['missing']['RULE']


def test_u0_binding_needs_the_common_u0_residual(panel, tmp_path):
    f = make(tmp_path, panel, mappo=False)
    path = tmp_path/'u0.pt'
    kw = dict(final_root=f['final_root'], path=path, sha256=export(path, iteration=None), arm='U0',
              kind=next(iter(nm.MAPPO_KINDS['U0'])), authority='test', dry_run=True)
    with pytest.raises(nm.Blocked, match='zero-residual'):
        nm.bind_mappo(**kw)
    with pytest.raises(nm.Blocked, match='does not hash'):
        nm.bind_mappo(**kw, residual_path=f['root']/'A_checkpoint_000.pt', residual_sha256='0'*64)
    bound = nm.bind_mappo(**kw, residual_path=f['root']/'A_checkpoint_000.pt',
                          residual_sha256=f['cfg']['common_checkpoint_sha256'])
    assert bound['residual_path'] == str(f['root']/'A_checkpoint_000.pt')


def prerun_fixture(tmp_path, panel, bind=True):
    root, cfg = nm.build_fake_root(tmp_path/'dry', panel, 'mappo_frozen')
    runtime, final_root, panel_3c = nm.FakeRuntime(), tmp_path/'dry'/'final_root', nm.load_panel_3c()
    plan = nm.prerun_register(final_root=final_root, plant_config=root/'CONFIG.json', u0_path=root/'A_checkpoint_000.pt',
                              u0_sha256=cfg['common_checkpoint_sha256'], runtime=runtime, panel=panel,
                              panel_3c=panel_3c, authority='test')
    if bind:
        nm.bind_fake_mappo(tmp_path/'dry', final_root, 'U0', residual=final_root/nm.PRERUN_PLANT/'A_checkpoint_000.pt')
        assert nm.prerun_logcheck(final_root=final_root, runtime=runtime, panel=panel)['verdict'] == 'PASS'
    return dict(root=root, cfg=cfg, runtime=runtime, final_root=final_root, plan=plan, panel=panel, panel_3c=panel_3c)


def run_prerun(f, runtime=None, **kw):
    return nm.prerun_execute(final_root=f['final_root'], runtime=runtime or f['runtime'], panel=f['panel'],
                             panel_3c=f['panel_3c'], **kw)


def test_prerun_runs_native_u0_rule_sealed_in_its_own_ledger(panel, tmp_path):
    f = prerun_fixture(tmp_path, panel, bind=False)
    plan, final_root = f['plan'], f['final_root']
    assert plan['arms'] == ['NATIVE', 'U0', 'RULE', 'RULE_FAIR'] and plan['cap_sim_s'] == 576_000
    assert plan['phase'] == 'v2v_final_prerun'
    assert plan['rule']['rule_sha256'] == nm.RULE_SHA256 and plan['panel_3c_sha256'] == nm.PANEL_3C_SHA256
    plant = json.loads((final_root/nm.PRERUN_PLANT/'CONFIG.json').read_text())
    source = json.loads((f['root']/'CONFIG.json').read_text())
    assert plant == source and nm.digest(plant) == plan['plant']['config_sha256']   # the clean CONFIG, unchanged
    with pytest.raises(nm.Blocked, match='never overwrite'):
        nm.prerun_register(final_root=final_root, plant_config=f['root']/'CONFIG.json', u0_path=f['root']/'A_checkpoint_000.pt',
                           u0_sha256=f['cfg']['common_checkpoint_sha256'], runtime=f['runtime'], panel=panel,
                           panel_3c=f['panel_3c'], authority='again')
    with pytest.raises(nm.Blocked, match='prerun-logcheck first'):
        run_prerun(f)
    with pytest.raises(nm.Blocked, match='bind-mappo --arm U0'):
        nm.prerun_logcheck(final_root=final_root, runtime=f['runtime'], panel=panel)
    nm.bind_fake_mappo(tmp_path/'dry', final_root, 'U0', residual=f['root']/'A_checkpoint_000.pt')
    with pytest.raises(nm.Blocked, match='not the pre-run u0 copy'):
        nm.prerun_logcheck(final_root=final_root, runtime=f['runtime'], panel=panel)
    nm.bind_fake_mappo(tmp_path/'dry', final_root, 'U0', residual=final_root/nm.PRERUN_PLANT/'A_checkpoint_000.pt')
    check = nm.prerun_logcheck(final_root=final_root, runtime=f['runtime'], panel=panel)
    assert check['verdict'] == 'PASS' and set(check['arms']) == {'NATIVE', 'RULE'} and check['sim_s'] == 2*360
    rule = check['arms']['RULE']['check']
    assert rule['ticks'] == 60 and rule['cav'] == 3 and rule['hdv'] == 3 and rule['origin_coverage'] == 1.0
    assert rule['delay']['mean_s'] > 0 and rule['ttc']['pairs'] > 0 and rule['accel']['var'] > 0
    assert nm.prerun_logcheck(final_root=final_root, runtime=nm.FakeRuntime(), panel=panel) == check   # sealed, reused
    out = run_prerun(f, workers=44)
    assert out['status'] == 'PASS' and out['sealed'] and out['chunk'] == 88 and out['charged_sim_s'] == 576_000+720
    assert set(out['coverage']) == {'NATIVE', 'U0', 'RULE', 'RULE_FAIR', 'C3_NATIVE', 'C3_U0', 'C3_RULE', 'C3_RULE_FAIR'}
    assert all(v['published'] == v['registered'] for v in out['coverage'].values())
    assert 'J_veh_s' not in json.dumps(out) and not list(final_root.glob('CLAIM*'))          # nothing opened (F12)
    assert not [r for r in nm.fe.ledger_rows(f['root']) if r['id'].startswith(nm.PREFIXES)]   # clean ledger untouched
    assert {r['phase'] for r in nm.fe.ledger_rows(final_root)} == {'v2v_final_prerun', 'v2v_final_logcheck'}
    again = nm.FakeRuntime()
    assert run_prerun(f, again)['charged_sim_s'] == 576_000+720
    assert not [c for c in again.calls if c[0] in ('run', 'reserve')]                     # never re-run
    with pytest.raises(nm.Blocked, match='already started'):
        nm.bind_fake_mappo(tmp_path/'dry', final_root, 'U0', residual=final_root/nm.PRERUN_PLANT/'A_checkpoint_000.pt')
    with pytest.raises(nm.Blocked, match='v2v_final_prerun cap'):
        nm.PrerunRuntime(f['runtime'], final_root).reserve(None, [dict(id='nmfinal_extra_0000', sim_s=1)])


def test_prerun_stops_at_a_chunk_boundary_and_resumes(panel, tmp_path):
    f = prerun_fixture(tmp_path, panel)
    final_root = f['final_root']
    first = nm.FakeRuntime()
    original = first.run

    def run_then_stop(items, cfg, workers):           # the operator drops STOP_PRERUN during the 2nd chunk
        out = original(items, cfg, workers)
        if len([c for c in first.calls if c[0] == 'run']) == 4:  # 2 chunks x (canary + rest)
            (final_root/nm.STOP_PRERUN).write_text('N4: clean5a launch')
        return out
    first.run = run_then_stop
    out = run_prerun(f, first, workers=8, chunk=20)
    assert out['status'] == 'STOPPED' and out['rounds'] == 2 and out['charged_sim_s'] == 40*1200+720
    assert out['coverage']['NATIVE'] == dict(published=40, registered=96) and out['coverage']['U0']['published'] == 0
    assert sorted(p.name for p in final_root.glob('WAVE_*.json')) == ['WAVE_NATIVE__0000-0019.json',
                                                                      'WAVE_NATIVE__0020-0039.json']
    (final_root/nm.STOP_PRERUN).unlink()
    second = nm.FakeRuntime()                         # e.g. cpuV, same absolute paths, fewer workers
    out = run_prerun(f, second, workers=16)
    assert out['status'] == 'PASS' and out['charged_sim_s'] == 576_000+720
    assert [c[1] for c in second.calls if c[0] == 'run'][:2] == [1, 31]          # canary, then the rest of 32
    assert (final_root/'WAVE_NATIVE__0040-0071.json').is_file()


def test_interrupted_prerun_chunk_is_never_replayed(panel, tmp_path):
    f = prerun_fixture(tmp_path, panel)
    nm.PrerunRuntime(f['runtime'], f['final_root']).reserve(None, [dict(id='nmfinal_native_0000', sim_s=1200)])
    with pytest.raises(nm.Blocked, match='interrupted wave'):
        run_prerun(f)


def test_main_run_reuses_the_prerun_only_on_the_same_plant_and_u0(panel, tmp_path):
    f = prerun_fixture(tmp_path, panel)
    run_prerun(f, workers=16, chunk=50)
    other = dict(f['cfg'], warm_history='another warm-up')
    write_config(f, other)
    with pytest.raises(nm.Blocked, match='plant differs'):
        nm.register(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel)
    write_config(f, f['cfg'])
    plan = nm.register(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel)
    assert plan['prerun']['status'] == 'compatible' and plan['prerun']['same_config'] is True
    for arm in ('MAPPO_SPEED', 'MAPPO_LANE'):
        nm.bind_fake_mappo(tmp_path/'dry', f['final_root'], arm)
    runtime = nm.FakeRuntime()
    value = nm.execute(root=f['root'], final_root=f['final_root'], runtime=runtime, panel=panel,
                       panel_3c=f['panel_3c'], poll=0, dry_run=True)
    assert value['claim3a']['present'] == ['B', 'A', 'R', 'RULE', 'RULE_FAIR', 'U0', 'NATIVE']
    assert value['claim3c']['present'] == ['B', 'R', 'RULE', 'RULE_FAIR', 'U0', 'NATIVE']
    assert not [r for r in nm.fe.ledger_rows(f['root']) if r['id'].startswith(nm.PREFIXES)]     # clean root never written
    clean = [r['id'] for r in nm.fe.ledger_rows(f['final_root']) if r['phase'] == 'v2v_final']
    assert not [i for i in clean if i.split('_')[1] in ('native', 'u0', 'rule')]          # reused, not re-run
    tables = json.loads((f['final_root']/'CLAIM3A_TABLES_u012.json').read_text())
    assert len(tables['sources']['RULE']['receipt_sha256']) >= 2                            # chunked pre-run receipts
    total = sum(r['sim_s'] for r in nm.fe.ledger_rows(f['final_root']) if r['phase'] == 'v2v_final')
    assert total+432_000 <= nm.quote('mappo_frozen')['total_ceiling_sim_s'] <= nm.ALLOCATION_SIM_S
    again = nm._analyse_published(f['root'], f['final_root'], runtime, panel, f['panel_3c'])
    for k in ('claim3a', 'claim3c'):                  # zero-physics re-tabulation from the chunked receipts
        strip = lambda x: {n: v for n, v in x.items() if n not in ('secondary', 'co_primary')}  # noqa: E731  (dry: 200 reps)
        assert strip(again[k]['tables']) == strip(value[k]['tables']) and again[k]['present'] == value[k]['present']


# ---------------------------------------------------------------- F11 parity

def test_f11_parity_runs_before_each_comparator_wave_and_is_reported(panel, tmp_path):
    result = nm.dry_run(out=tmp_path/'out', panel=panel)
    final_root = Path(result['final_root'])
    for arm in nm.COMPARATOR_ARMS:
        report = nm.verify(json.loads((final_root/f'PARITY_{arm}.json').read_text()), 'receipt_sha256', 'parity')
        assert report['verdict'] == 'IDENTICAL' and report['descriptive'] is True and set(report['worlds']) == {'989598', '989599'}
        c = report['worlds']['989598']['compare']
        assert c['ticks_common'] == 120 and c['agree_fraction'] == 1.0 and c['first_divergence_t'] is None
        wave = json.loads((final_root/f'WAVE_PARITY_{arm}.json').read_text())
        assert [r['job_id'] for r in wave['records']] == [f'nmparity_{arm.lower()}_{p}_{s}' for s in (989598, 989599)
                                                         for p in ('plant', 'train')]
        assert all(r['world']['seed'] in (989598, 989599) and r['world']['demand'] == 3200 for r in wave['records'])
        tables = json.loads((final_root/f'CLAIM3B_{arm}_TABLES_u012.json').read_text())
        assert tables['sources'][f'PARITY_{arm}_REPORT']['verdict'] == 'IDENTICAL'
    rows = [r for r in nm.fe.ledger_rows(final_root) if r['id'].startswith('nmparity_')]
    assert len(rows) == 8 and sum(r['sim_s'] for r in rows) == nm.PARITY_SIM_S == 3_360


def test_parity_compare_reports_the_first_divergence(panel, tmp_path):
    f = make(tmp_path, panel)
    specs = nm.parity_specs('MAPPO_SPEED', panel, f['cfg'], f['root'], 'mappo_u0', f['mappo']['MAPPO_SPEED'], f['mappo']['U0'])
    assert [s['path'] for s in specs] == ['plant', 'train']*2 and specs[0]['checkpoint']['kind'] == 'mappo_comparator'
    plant, train = (nm.check_parity(nm.fake_parity(s), s) for s in specs[:2])
    assert nm.parity_compare(plant, train)['verdict'] == 'IDENTICAL'
    shifted = deepcopy(train)
    shifted['trace'][50]['cav']['main.2'][3] += 0.01                       # one command differs at t = 350
    c = nm.parity_compare(plant, shifted)
    assert c['verdict'] == 'DIFFERS' and c['first_divergence_t'] == 350 and c['max_abs']['command_mps2'] == pytest.approx(.01)
    assert c['agree_pairs'] == c['pairs']-1
    gone = deepcopy(train)
    del gone['trace'][10]['cav']['main.4']
    assert nm.parity_compare(plant, gone)['cav_set_mismatch_ticks'] == 1
    with pytest.raises(nm.Blocked, match='never silent'):
        nm.check_parity(dict(nm.fake_parity(specs[0]), executor_sha256_loaded='0'*64), specs[0])
    with pytest.raises(nm.Blocked, match='train-config'):
        nm.parity_specs('MAPPO_SPEED', panel, f['cfg'], f['root'], 'mappo_u0',
                        dict(f['mappo']['MAPPO_SPEED'], train_config=None), f['mappo']['U0'])
    with pytest.raises(nm.Blocked, match='no F11 parity'):
        nm.bind_mappo(final_root=f['final_root'], path=f['mappo']['U0']['path'], sha256=f['mappo']['U0']['sha256'],
                      kind=f['mappo']['U0']['kind'], arm='U0', authority='t', dry_run=True,
                      residual_path=f['root']/'A_checkpoint_000.pt', residual_sha256=f['cfg']['common_checkpoint_sha256'],
                      train_config=f['root']/'CONFIG.json')


def test_comparator_without_train_config_runs_but_notes_missing_parity(panel, tmp_path):
    f = make(tmp_path, panel, mappo=False)
    nm.bind_fake_mappo(tmp_path/'dry', f['final_root'], 'U0', residual=f['root']/'A_checkpoint_000.pt')
    path = tmp_path/'it168.pt'
    nm.bind_mappo(final_root=f['final_root'], path=path, sha256=export(path), kind='retrained_wz3to2_it168',
                  arm='MAPPO_SPEED', authority='N7 (no config given)')
    nm.declare_unavailable(final_root=f['final_root'], arm='MAPPO_LANE', authority='MAIN (test)', reason='not delivered')
    value = nm.execute(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel, poll=0, dry_run=True)
    assert 'without --train-config' in value['claim3b_MAPPO_SPEED']['missing']['PARITY_MAPPO_SPEED']
    assert not list(f['final_root'].glob('PARITY_*.json'))


# ---------------------------------------------------------------- own-root accounting (check_pins)

def test_prerun_keeps_the_clean_config_and_accounts_in_its_own_root(panel, tmp_path):
    root, cfg = nm.build_fake_root(tmp_path/'first', panel, 'mappo_frozen')
    runtime = nm.FakeRuntime()
    rerooted = dict(cfg, run_root=str(tmp_path/'first'/'final_root'/nm.PRERUN_PLANT))      # the 829c7303 plant copy
    with pytest.raises(ValueError, match='OWN root'):                                     # IMITREPLAY's refusal
        runtime.plant_view(rerooted)
    bad = tmp_path/'bad_config.json'
    bad.write_text(json.dumps(rerooted))
    final_root = tmp_path/'first'/'final_root'
    with pytest.raises(ValueError, match='OWN root'):
        nm.prerun_register(final_root=final_root, plant_config=bad, u0_path=root/'A_checkpoint_000.pt',
                           u0_sha256=cfg['common_checkpoint_sha256'], runtime=runtime, panel=panel,
                           panel_3c=nm.load_panel_3c(), authority='test')
    assert not final_root.exists()                                                        # no partial root
    f = prerun_fixture(tmp_path, panel)
    run_prerun(f, workers=16, chunk=60)
    assert [r for r in nm.fe.ledger_rows(f['root']) if r['phase'] != 'v2v_train'] == []   # clean ledger untouched
    nm.register(root=f['root'], final_root=f['final_root'], runtime=f['runtime'], panel=panel)
    for arm in ('MAPPO_SPEED', 'MAPPO_LANE'):
        nm.bind_fake_mappo(tmp_path/'dry', f['final_root'], arm)
    nm.execute(root=f['root'], final_root=f['final_root'], runtime=nm.FakeRuntime(), panel=panel,
               panel_3c=f['panel_3c'], poll=0, dry_run=True)
    assert [r for r in nm.fe.ledger_rows(f['root']) if r['phase'] != 'v2v_train'] == []   # main run too
    own = nm.fe.ledger_rows(f['final_root'])
    assert {r['phase'] for r in own} == {'v2v_final', 'v2v_final_prerun', 'v2v_final_logcheck'}
    assert sum(r['sim_s'] for r in own) <= nm.ALLOCATION_SIM_S
    with pytest.raises(nm.Blocked, match='duplicate'):
        nm.own_reserve(f['final_root'], [dict(id=own[0]['id'], sim_s=1)], 'v2v_final')
    with pytest.raises(nm.Blocked, match='allocation'):
        nm.own_reserve(f['final_root'], [dict(id='nmfinal_extra_0000', sim_s=nm.ALLOCATION_SIM_S)], 'v2v_final')



def test_vehicle_log_report_refuses_an_empty_or_partial_log(tmp_path):
    import gzip
    empty = tmp_path/'empty.jsonl.gz'
    with gzip.open(empty, 'wt'):
        pass                                                          # the 316461a4 NATIVE failure: 40-byte gzip
    r = nm.vehicle_log_report(empty, t0=300, t1=360)
    assert r['ok'] is False and any('ticks' in x for x in r['problems']) and 'no vehicles' in r['problems']
    good = tmp_path/'good'
    nm.fake_logcheck(dict(sim_s=360, arm='NATIVE', world_id='w', world={}, checkpoint={}, window={}, panel_sha256='x'),
                     good)
    r = nm.vehicle_log_report(good/nm.VEHICLE_LOG, t0=300, t1=360)
    assert r['ok'] and r['crossings_drop'] >= 1 and r['delay']['gini'] is not None and r['ttc']['min_s'] is not None
    lines = gzip.open(good/nm.VEHICLE_LOG, 'rt').read().splitlines()
    one_class = tmp_path/'one_class.jsonl.gz'
    with gzip.open(one_class, 'wt') as f:
        f.write('\n'.join(x.replace('"cav": false', '"cav": true') for x in lines)+'\n')
    assert any('both fleet classes' in x for x in nm.vehicle_log_report(one_class, t0=300, t1=360)['problems'])



# ---------------------------------------------------------------- 01:09 failure: no NATIVE -> policy process reuse

def test_runtime_runs_each_arm_in_its_own_fresh_pool():
    """Regression (3c chunk 01:09): with ONE worker the old pool reused the NATIVE process for U0 and the G5
    binding refused; the probe runner raises exactly like that."""
    items = [('episode', dict(job_id=f'j{i}', arm=arm), f'w{i}') for i, arm in enumerate(['NATIVE']*3+['U0']*2+['RULE'])]
    assert nm.process_groups(items) == [[0, 1, 2], [3, 4], [5]]
    out = nm.Runtime('tools.coopmwm._n6_pool_probe').run(items, cfg={}, workers=1)
    assert [r['job_id'] for r in out] == [f'j{i}' for i in range(6)]
    pids = {r['arm']: {x['pid'] for x in out if x['arm'] == r['arm']} for r in out}
    assert not pids['NATIVE'] & pids['U0'] and not pids['NATIVE'] & pids['RULE']
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:   # the old pool
        futures = [pool.submit(nm._worker, k, sp, {}, w, 'tools.coopmwm._n6_pool_probe') for k, sp, w in items]
        with pytest.raises(ValueError, match='G5 runtime binding'):
            [f.result() for f in futures]


def test_failed_chunk_is_replayed_once_only_by_authorisation_and_the_cap_amendment(panel, tmp_path):
    f = prerun_fixture(tmp_path, panel)
    final_root = f['final_root']
    first = nm.FakeRuntime()

    def fail_second(items, cfg, workers):
        if len([c for c in first.calls if c[0] == 'run']) == 1:
            first.calls.append(('run', len(items), workers))
            raise ValueError('traffic_shepherd.c4.env was imported before the G5 runtime binding')
        return nm.FakeRuntime.run(first, items, cfg, workers)
    first.run = fail_second
    with pytest.raises(ValueError, match='G5'):
        run_prerun(f, first, workers=8, chunk=20)                   # chunk 1: canary ok, the rest fails after charging
    assert nm.interrupted_jobs(final_root) == [f'nmfinal_native_{i:04}' for i in range(20)]
    with pytest.raises(nm.Blocked, match='never replay'):
        run_prerun(f, nm.FakeRuntime(), workers=8, chunk=20)
    with pytest.raises(nm.Blocked, match='authority and reason'):
        nm.authorize_replay(final_root=final_root, authority='MAIN', reason=' ')
    rep = nm.authorize_replay(final_root=final_root, authority='MAIN 2026-11-07: re-run 3c after the pool fix',
                              reason='pool-worker G5 failure, nothing published or opened')
    assert rep['jobs'] == [f'nmfinal_native_{i:04}' for i in range(20)] and rep['results_opened'] is False
    assert set(rep['rounds'].values()) == {1}
    with pytest.raises(nm.Blocked, match='v2v_final_prerun cap'):
        run_prerun(f, nm.FakeRuntime(), workers=16)                 # 432,000 cannot hold the replay + the rest
    with pytest.raises(nm.Blocked, match='cap in'):
        nm.amend_prerun_cap(final_root=final_root, cap_sim_s=700_000, authority='MAIN')
    nm.amend_prerun_cap(final_root=final_root, cap_sim_s=620_000, authority='MAIN 2026-11-07: sub-cap raise')
    out = run_prerun(f, nm.FakeRuntime(), workers=16)
    assert out['status'] == 'PASS' and out['cap_sim_s'] == 620_000
    assert all(v['published'] == v['registered'] for v in out['coverage'].values())
    rows = [r['id'] for r in nm.fe.ledger_rows(final_root)]
    assert sum(r.endswith('#r1') for r in rows) == 20 and len(list(final_root.glob('RESERVED_NATIVE__0000-*.replay1.json'))) == 1
    assert all((final_root/'evaluation'/f'nmfinal_native_{i:04}__r1'/nm.VEHICLE_LOG).is_file() for i in range(20))
    wave = json.loads((final_root/'WAVE_NATIVE__0000-0031.json').read_text())
    assert wave['records'][0]['work_dir'] == 'nmfinal_native_0000__r1' and 'work_dir' not in wave['records'][25]
    with pytest.raises(nm.Blocked, match='no interrupted part'):
        nm.authorize_replay(final_root=final_root, authority='MAIN', reason='again')


def test_seqcheck_runs_native_u0_rule_in_one_dispatch(panel, tmp_path):
    f = prerun_fixture(tmp_path, panel)
    value = nm.prerun_seqcheck(final_root=f['final_root'], runtime=f['runtime'], panel=panel)
    assert value['verdict'] == 'PASS' and value['order'] == ['NATIVE', 'U0', 'RULE'] and value['sim_s'] == 1_080
    assert len(set(value['pids'])) == 3 and all(a['check']['ok'] for a in value['arms'].values())
    assert {r['phase'] for r in nm.fe.ledger_rows(f['final_root'])} >= {'v2v_final_seqcheck'}



def test_a_failed_replay_round_needs_a_new_authorisation_and_a_fresh_dir(panel, tmp_path):
    f = prerun_fixture(tmp_path, panel)
    final_root = f['final_root']
    bad = nm.FakeRuntime()

    def always_fail_rest(items, cfg, workers):
        bad.calls.append(('run', len(items), workers))
        if len(items) > 1:
            raise FileExistsError('vehicles.jsonl.gz')
        return nm.FakeRuntime.run(nm.FakeRuntime(), items, cfg, workers)
    bad.run = always_fail_rest
    with pytest.raises(FileExistsError):
        run_prerun(f, bad, workers=8, chunk=10)
    nm.authorize_replay(final_root=final_root, authority='MAIN 1', reason='first failure')
    with pytest.raises(FileExistsError):
        run_prerun(f, bad, workers=8, chunk=10)                      # round 1 fails too
    with pytest.raises(nm.Blocked, match='never replay without a MAIN authorisation'):
        run_prerun(f, nm.FakeRuntime(), workers=8, chunk=10)
    rep = nm.authorize_replay(final_root=final_root, authority='MAIN 2', reason='round-1 replay failed (fresh dirs)')
    assert set(rep['rounds'].values()) == {2} and len(rep['jobs']) == 10
    nm.amend_prerun_cap(final_root=final_root, cap_sim_s=640_000, authority='MAIN: 640,000')
    out = run_prerun(f, nm.FakeRuntime(), workers=8, chunk=10)
    assert out['status'] == 'PASS'
    assert (final_root/'evaluation'/'nmfinal_native_0001__r2'/nm.VEHICLE_LOG).is_file()
    assert sum(r['id'].endswith('#r2') for r in nm.fe.ledger_rows(final_root)) == 10



def test_group_labels_mirror_the_clean5b_f_training_convention(tmp_path):
    """Amendment 13: origin = WM lane of the FIRST logged row if upstream of the drop (else undefined); class groups
    hold every U0-paired vehicle; labels come from U0 by id; missing vehicles are counted both ways."""
    import gzip

    def log(path, rows):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            for k, (vid, cav) in enumerate([('v0', True), ('v1', False), ('v2', False), ('v3', True)]):
                f.write(json.dumps(dict(veh=k, id=vid, cav=cav, origin='e', depart_delay_s=0., first_t=300))+'\n')
            for t in range(300, 302):
                f.write(json.dumps(dict(t=t, **rows(t)))+'\n')
    # v0 first seen upstream in WM lane 2 (L0); v1 upstream WM 0 (L2); v2 first seen PAST the drop (undefined);
    # v3 only in the reference arm
    log(tmp_path/'u0.gz', lambda t: dict(k=[0, 1, 2, 3], s=[-50., -80., 90., -10.], v=[10., 20., 30., 31.],
                                         a=[0., .1, 0., -.1], lane=[2, 0, 1, 1]))
    log(tmp_path/'b.gz', lambda t: dict(k=[0, 1, 2], s=[-50., -80., 90.], v=[5., 20., 30.], a=[0., .1, 0.],
                                        lane=[1, 0, 1]))
    u0, b = (nm.vehicle_log_report(tmp_path/n, t0=300, t1=302) for n in ('u0.gz', 'b.gz'))
    assert u0['per_vehicle']['v0'][1] == 'L0' and u0['per_vehicle']['v1'][1] == 'L2' and u0['per_vehicle']['v2'][1] == 'D'
    labels = nm._labels(u0['per_vehicle'])
    assert labels['v0'] == ('class:cav', 'origin:L0') and labels['v2'] == ('class:hdv', 'origin:undefined')
    reports = dict(U0={'w': u0}, A={'w': u0}, B={'w': b})
    boot = dict(reps=50, seed=1, level=.95)
    out = nm.co_primary_groups({k: {**v, 'w2': v['w']} for k, v in reports.items()}, ['w', 'w2'], boot)
    assert out['vehicles_missing']['B'] == dict(not_in_reference=0, reference_not_in_arm=2)   # v3 x 2 worlds
    assert out['groups']['origin:L0']['increase_mean_s'] > 0 and out['groups']['origin:undefined']['registered'] is False


def test_eval_per_vehicle_equals_the_clean5b_f_training_table(tmp_path):
    """Where the clean5b-F logger exists (1caf5c1c): same delay and origin per vehicle, same A-labelled groups."""
    fvl = pytest.importorskip('tools.coopmwm.fair_vehicle_log', reason='clean5b-F logger not in this tree')
    nm.fake_logcheck(dict(sim_s=360, arm='NATIVE', world_id='w', world={}, checkpoint={}, window={}, panel_sha256='x'),
                     tmp_path)
    path = tmp_path/nm.VEHICLE_LOG
    ours = nm.vehicle_log_report(path, t0=300, t1=360)['per_vehicle']
    theirs = fvl.per_vehicle(path, t0=300, t1=360)
    assert set(ours) == set(theirs)
    for vid, (cav, origin, delay) in theirs.items():
        assert ours[vid][0] == pytest.approx(delay, abs=1e-3) and ours[vid][1] == origin and ours[vid][2] == cav
    groups, _ = fvl.groups(theirs, fvl.labels(theirs))
    mine = nm._group_means(ours, nm._labels(ours))
    assert {g: v['mean_s'] for g, v in groups.items()} == pytest.approx({g: mine[g] for g in groups}, abs=1e-3)
