"""CPU-only tests for the clean3 final-evaluation launcher. No simulator, no process pool."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean3_finaleval as fe  # noqa: E402
from tools.coopmwm import planner_eval as pe, protocol, rerun_cpus as rc  # noqa: E402
from tools.coopmwm.clean3_restart import LANE_ON_BASELINE  # noqa: E402
from tools.coopmwm.exact_grade import _seal  # noqa: E402


def registration_bytes(extra=None):
    worlds = protocol.worlds('train', 0)+protocol.worlds('train', 1)[:24]
    if extra is not None:
        worlds = worlds[:-1]+[extra]
    value = dict(final_evaluation=dict(role='train', worlds=[dict(world_id=str(w['seed']), world=w) for w in worlds]))
    return json.dumps(value, sort_keys=True).encode()


def config(reg, **phases):
    budget = dict(v2v_final=531_600, isolated_replicated=165_600, v2v_interim=398_400)
    budget.update(phases)
    return dict(final_evaluation=deepcopy(rc.FINAL_EVALUATION), final_paired_branches=False,
                isolated_registration_sha256=hashlib.sha256(reg).hexdigest(),
                extra_final_baselines=[dict(LANE_ON_BASELINE)],
                clean3_budget=_seal(dict(phases=budget), 'receipt_sha256'),
                common_checkpoint_sha256='0'*64, runtime='/home/ubuntu/e3final', run_root='placeholder')


@pytest.fixture
def files(tmp_path):
    reg = registration_bytes()
    (tmp_path/'REG.json').write_bytes(reg)
    (tmp_path/'CONFIG.json').write_text(json.dumps(config(reg)))
    return tmp_path/'CONFIG.json', tmp_path/'REG.json'


@pytest.fixture
def fake(tmp_path):
    reg = registration_bytes()
    root, cfg, registration = fe.build_fake_root(tmp_path/'dry', config(reg), reg)
    runtime = fe.FakeRuntime()
    final_root = tmp_path/'dry'/'final_root'
    budget = fe.register(root=root, final_root=final_root, runtime=runtime)
    return dict(root=root, cfg=cfg, registration=registration, runtime=runtime, final_root=final_root,
                budget=budget, tmp=tmp_path)


def approve(final_root, budget, **override):
    value = dict(approved=True, budget_sha256=budget['receipt_sha256'], authority='test approval', dry_run=True)
    value.update(override)
    (Path(final_root)/'OWNER_APPROVAL.json').write_text(json.dumps(value))


def laneon_stub(tmp):
    path = Path(tmp)/'laneon_stub.pt'
    path.write_bytes(b'stub')
    return dict(laneon_checkpoint=path, laneon_sha=fe.file_sha(path))


def test_dry_run_end_to_end_is_zero_physics(files, tmp_path, monkeypatch):
    import concurrent.futures

    class Forbidden:
        def __init__(self, *a, **k):
            raise AssertionError('dry run must never open a process pool')
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Forbidden)
    result = fe.dry_run(config=files[0], registration_file=files[1], out=tmp_path/'out')
    assert result['status'] == 'PASS' and result['new_sim_s'] == 0 and result['physics'] is False
    assert result['present'] == ['B', 'A', 'R', 'MAPPO_U0', 'LANEON', 'NATIVE']
    reserves = sorted((c[2], c[3]) for c in result['calls'] if c[0] == 'reserve')
    assert reserves == [('final_mappo_u0', 115_200), ('final_native', 115_200), ('v2v_final', 115_200)]
    assert ('run', 288, 24) in result['calls']
    final_root = Path(result['final_root'])
    tables = json.loads((final_root/'FINAL_TABLES.json').read_text())
    fe.verify(tables, 'analysis_sha256', 'tables')
    assert tables['dry_run'] is True and tables['missing'] == {}
    t = tables['tables']
    assert t['registered_B_A_R']['matches_paired_u024_analysis'] is True
    assert set(t['total_benefit_vs_mappo']['rows']) == {'B', 'A', 'R', 'LANEON', 'NATIVE'}
    assert t['synergy']['panel_decomposition']['max_identity_error_veh_s'] < 1e-9
    assert t['synergy']['registered_isolated_u24']['verdict'] == 'pass'
    assert set(t['by_penetration']['rows']) == {'0.05', '0.1', '0.2', '0.4'}
    text = (final_root/'FINAL_TABLES.md').read_text()
    assert 'FAKE PHYSICS' in text and '## T3 Synergy decomposition' in text
    status = json.loads((final_root/'STATUS.json').read_text())
    assert status['step'] == 'TABLES' and status['status'] == 'PASS'


def test_registration_check_passes_registered_shapes_and_flags_tampering(fake):
    root, cfg = fake['root'], fake['cfg']
    report = fe.registration_check(root, cfg, final_root=fake['final_root'])
    assert report['status'] == 'PASS', report
    assert report['extra_baseline_budget']['status'] == 'REGISTERED_UNAPPROVED'
    assert all(p['margin'] >= fe.MARGIN_MIN for p in report['extra_baseline_budget']['phases'].values())
    bad = deepcopy(cfg)
    bad['final_evaluation']['bootstrap']['reps'] = 5000
    assert fe.registration_check(root, bad)['status'] == 'FAIL'
    bad = deepcopy(cfg)
    bad['final_paired_branches'] = True
    assert fe.registration_check(root, bad)['status'] == 'FAIL'
    bad = deepcopy(cfg)
    bad['isolated_registration_sha256'] = '1'*64
    assert fe.registration_check(root, bad)['status'] == 'FAIL'
    bad = deepcopy(cfg)
    bad['clean3_budget'] = _seal(dict(phases=dict(v2v_final=460_800)), 'receipt_sha256')
    assert fe.registration_check(root, bad)['status'] == 'FAIL'   # no 15% margin
    bad = deepcopy(cfg)
    bad['clean3_budget']['phases']['v2v_final'] = 10**7          # edited after sealing
    assert fe.registration_check(root, bad)['status'] == 'FAIL'
    bad = deepcopy(cfg)
    bad['extra_final_baselines'] = []
    assert fe.registration_check(root, bad)['status'] == 'FAIL'


def test_panel_outside_registered_train_universe_fails(tmp_path):
    reg = registration_bytes(extra=dict(seed=973000, cell=0, repeat=0, penetration=0.05, demand=1900))
    root, cfg, _ = fe.build_fake_root(tmp_path, config(reg), reg)
    report = fe.registration_check(root, cfg)
    failed = [c['name'] for c in report['checks'] if not c['ok']]
    assert failed == ['panel_is_registered_train_universe']


def test_budget_outstanding_need_counts_existing_paired_charges(fake):
    root, cfg = fake['root'], fake['cfg']
    with (root/'CHARGES.jsonl').open('a') as f:
        for i in range(288):
            f.write(json.dumps(dict(id=f'paired_u024_{i:04}', phase='v2v_final', sim_s=1200))+'\n')
    report = fe.registration_check(root, cfg)
    row = [c for c in report['checks'] if c['name'].startswith('clean3_final_phase')][0]
    assert row['ok'] and row['phases']['v2v_final']['outstanding_need'] == 115_200
    with (root/'CHARGES.jsonl').open('a') as f:
        f.write(json.dumps(dict(id='unexplained', phase='v2v_final', sim_s=1200))+'\n')
    assert fe.registration_check(root, cfg)['status'] == 'FAIL'


def test_gate_waits_for_u24_and_blocks_on_safety_or_tampering(fake):
    root, cfg, runtime = fake['root'], fake['cfg'], fake['runtime']
    assert fe.gate(root, cfg, runtime)['status'] == 'PASS'
    monitor = root/'LEARNING_MONITOR_u024.json'
    saved = monitor.read_text()
    monitor.unlink()
    with pytest.raises(fe.NotReady):
        fe.gate(root, cfg, runtime)
    monitor.write_text(saved)
    (root/'FAULT.json').write_text('{}')
    with pytest.raises(fe.NotReady, match='health'):
        fe.gate(root, cfg, runtime)
    (root/'FAULT.json').unlink()
    path = root/'LEARNING_MONITOR_u005.json'
    row = json.loads(path.read_text())
    row['arms'][1]['metrics']['safety']['collisions'] = 1
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    with pytest.raises(fe.Blocked, match='safety not clean'):
        fe.gate(root, cfg, runtime)
    row['arms'][1]['metrics']['safety']['collisions'] = 0
    row['arms'][2]['action'] = 'PAUSE_SAFETY'
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    with pytest.raises(fe.Blocked, match='pauses'):
        fe.gate(root, cfg, runtime)
    row['arms'][2]['action'] = 'CONTINUE'
    row['config_sha256'] = '2'*64
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    with pytest.raises(fe.Blocked, match='not bound'):
        fe.gate(root, cfg, runtime)
    row['config_sha256'] = fe.digest(cfg)
    path.write_text(json.dumps(dict(row, update=6)))
    with pytest.raises(fe.Blocked):
        fe.gate(root, cfg, runtime)
    path.write_text(json.dumps(_seal({k: v for k, v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')))
    update = root/'B_UPDATE_024.json'
    update.write_text(json.dumps(dict(json.loads(update.read_text()), arm='edited')))
    with pytest.raises(fe.Blocked, match='u24 update'):
        fe.gate(root, cfg, runtime)


def test_gate_refuses_wrong_common_checkpoint(fake):
    root, cfg, runtime = fake['root'], fake['cfg'], fake['runtime']
    (root/'A_checkpoint_000.pt').write_bytes(b'other')
    with pytest.raises(fe.Blocked, match='common MAPPO u0'):
        fe.gate(root, cfg, runtime)


def test_paired_stats_reproduce_the_registered_estimator():
    rng = np.random.default_rng(3)
    c = rng.normal(50_000, 5_000, 96)
    t = c+rng.normal(-300, 400, 96)
    boot = dict(rc.FINAL_EVALUATION['bootstrap'])
    stats = fe.paired_stats(t, c, boot)
    ref = pe._paired_ci(t-c, boot['reps'], boot['seed'], 1-boot['level'])
    assert (stats['diff'], stats['ci95'], stats['p_two_sided']) == (ref['mean_delta_vs'], ref['ci95'], ref['p_two_sided'])
    assert stats['pct'] == pytest.approx(100*(t.mean()-c.mean())/c.mean())
    assert stats['pct_ci95'][0] < stats['pct'] < stats['pct_ci95'][1]
    assert stats['worlds_lower']+stats['worlds_higher'] <= 96


def test_extra_baselines_need_owner_approval(fake):
    kw = dict(root=fake['root'], final_root=fake['final_root'], runtime=fake['runtime'], poll=0,
              on_gate=lambda: fe.fake_monitor(fake['root'], fake['cfg'], fake['registration']), dry_run=True)
    with pytest.raises(fe.Blocked, match='MAPPO_U0'):
        fe.execute(**kw, **laneon_stub(fake['tmp']))
    approve(fake['final_root'], fake['budget'], budget_sha256='3'*64)
    assert fe.approval(fake['final_root'])['status'] == 'REGISTERED_UNAPPROVED'
    value = fe.execute(**kw, allow_missing=('MAPPO_U0', 'NATIVE'), **laneon_stub(fake['tmp']))
    assert value['present'] == ['B', 'A', 'R', 'LANEON']
    assert set(value['missing']) == {'MAPPO_U0', 'NATIVE'}
    t = value['tables']
    assert t['total_benefit_vs_mappo']['status'] == 'unavailable'
    assert t['total_benefit_vs_mappo']['fallback_comparator'] == 'A'
    assert 'components' not in t['synergy']['panel_decomposition']
    assert not fe.ledger_rows(fake['final_root'])        # nothing charged to the unapproved ledger


def test_interrupted_waves_are_never_replayed(fake):
    approve(fake['final_root'], fake['budget'])
    with (fake['final_root']/'CHARGES.jsonl').open('a') as f:
        f.write(json.dumps(dict(id='finaleval_native_0000', phase='final_native', sim_s=1200))+'\n')
    kw = dict(root=fake['root'], final_root=fake['final_root'], runtime=fake['runtime'], poll=0, dry_run=True,
              on_gate=lambda: fe.fake_monitor(fake['root'], fake['cfg'], fake['registration']))
    with pytest.raises(fe.Blocked, match='never replay'):
        fe.execute(**kw, **laneon_stub(fake['tmp']))
    assert not [c for c in fake['runtime'].calls if c[0] == 'run']


def test_published_baseline_is_reused_not_rerun(fake):
    approve(fake['final_root'], fake['budget'])
    kw = dict(root=fake['root'], final_root=fake['final_root'], cfg=fake['cfg'], registration=fake['registration'],
              runtime=fake['runtime'], workers=8)
    out, _ = fe.run_baselines(['MAPPO_U0'], **kw)
    calls = list(fake['runtime'].calls)
    again, sources = fe.run_baselines(['MAPPO_U0'], **kw)
    assert again == out and fake['runtime'].calls == calls
    assert sources['MAPPO_U0']['source'].endswith('BASELINE_MAPPO_U0.json')


def test_laneon_is_adopted_from_the_monitor_receipt(fake):
    root, registration = fake['root'], fake['registration']
    specs = fe.baseline_specs('LANEON', registration, fake['cfg'], root)
    results = [{k: v for k, v in fe.fake_record(s, 'mappo_laneon_aecd7d70').items() if k != 'checkpoint'}
               for s in specs]
    fe.publish(root/fe.LANEON_MONITOR_FILE, _seal(dict(update=24, results=results,
               registration=registration), 'receipt_sha256'))
    assert fe.laneon_mode(root, fake['cfg'], fake['runtime']) == 'adopt'
    value = fe.execute(root=root, final_root=fake['final_root'], runtime=fake['runtime'], poll=0, dry_run=True,
                       baselines=('LANEON',), on_gate=lambda: fe.fake_monitor(root, fake['cfg'], registration))
    assert value['present'] == ['B', 'A', 'R', 'LANEON']
    assert not [c for c in fake['runtime'].calls if c[0] in ('run', 'reserve')]
    assert value['sources']['LANEON']['source'].endswith('LANEON_u024.json')


def test_laneon_unavailable_requires_explicit_exclusion(fake):
    runtime = fe.FakeRuntime(laneon=False)
    with pytest.raises(fe.Blocked, match='LANEON'):
        fe.execute(root=fake['root'], final_root=fake['final_root'], runtime=runtime, poll=0,
                   baselines=('LANEON',))


def test_tables_refuse_tampered_or_incomplete_receipts(fake):
    root, cfg, registration = fake['root'], fake['cfg'], fake['registration']
    fe.fake_monitor(root, cfg, registration)
    path = root/fe.PAIRED_FILE
    paired = json.loads(path.read_text())
    kw = dict(root=root, final_root=fake['final_root'], cfg=cfg, registration=registration, runtime=fake['runtime'],
              baselines={}, sources={}, missing={}, dry_run=True)
    edited = deepcopy(paired)
    edited['results'][0]['J_veh_s'] += 1
    path.write_text(json.dumps(edited))
    with pytest.raises(fe.Blocked, match='PAIRED_u024'):
        fe.tables(**kw)
    resealed = _seal({k: v for k, v in edited.items() if k != 'receipt_sha256'}, 'receipt_sha256')
    path.write_text(json.dumps(resealed))
    with pytest.raises(fe.Blocked, match='reproduce'):
        fe.tables(**kw)
    path.write_text(json.dumps(paired))
    value = fe.tables(**kw)
    assert value['present'] == ['B', 'A', 'R'] and value['tables']['registered_B_A_R']['matches_paired_u024_analysis']


def test_record_validation_refuses_bad_records(fake):
    registration = fake['registration']
    specs = fe.baseline_specs('NATIVE', registration, fake['cfg'], fake['root'])
    good = [fe.fake_record(s, 'NATIVE') for s in specs]
    assert len(fe.complete(good, registration, 'NATIVE')) == 96
    for mutate, match in ((lambda r: r.update(J_veh_s=float('nan')), 'finite'),
                          (lambda r: r.update(window=dict(start_s=0, end_s=900)), '1200'),
                          (lambda r: r['safety'].update(collisions=-1), 'safety'),
                          (lambda r: r['safety'].pop('ttc_events'), 'safety'),
                          (lambda r: r.update(status='failed'), 'ungraded'),
                          (lambda r: r['world'].update(demand=9999), 'panel')):
        records = deepcopy(good)
        mutate(records[0])
        with pytest.raises(fe.Blocked, match=match):
            fe.complete(records, registration, 'NATIVE')
    with pytest.raises(fe.Blocked, match='exactly once'):
        fe.complete(good[1:], registration, 'NATIVE')
    with pytest.raises(fe.Blocked, match='duplicate'):
        fe.complete(good+[good[0]], registration, 'NATIVE')


def test_budget_constants_cover_every_arm_with_margin():
    episodes = fe.EPISODES*fe.HORIZON_S
    assert all(v/episodes-1 >= fe.MARGIN_MIN for v in fe.FINAL_PHASES.values())
    assert 531_600/(4*episodes)-1 >= fe.MARGIN_MIN      # clean3 v2v_final: A/B/R + lane-on
    assert fe.BASELINES['LANEON']['ledger'] == 'clean3'
    assert {fe.BASELINES[n]['phase'] for n in ('MAPPO_U0', 'NATIVE')} == set(fe.FINAL_PHASES)


def test_forbidden_paths_are_refused(tmp_path):
    for token in ('holdout', 'cal_split', 'chunk5', 'hetdata_truth', 'vehicle_params'):
        with pytest.raises(fe.Blocked):
            fe.guard(tmp_path/token/'x.json')
