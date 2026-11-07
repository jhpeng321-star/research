"""clean5b-F8P recovery contracts (MAIN 2026-11-07): resumed failed episode -> recorded replacement (never a retry),
owner/start records kept on resume, disk floor waits, consumed G4 data pruned with hashes recorded first."""
import gzip
import json
from types import SimpleNamespace

import pytest

from . import accel_runner as ar, disk_prune as dp, launch_training as lt, runner


def _failed(tmp_path, empty=True):
    d = tmp_path/'jobs'/'t2c_warm_u001_R_989224'
    d.mkdir(parents=True)
    (d/'CLAIM.json').write_text('{}')
    (d/'EPISODE_FAILED.json').write_text('' if empty else '{"id": "x"}')
    return dict(id='t2c_warm_u001_R_989224', path=str(d), arm='R', world=dict(seed=989224, cell_key='D2800_p40'),
                noise_seed=1)


def test_resumed_failure_is_replaced_once_and_never_retried(tmp_path, monkeypatch):
    job = _failed(tmp_path)
    monkeypatch.delenv(ar.RESUME_ENV, raising=False)
    assert not ar.resume_failed_episode(tmp_path/'jobs'/job['id'])          # flag off: the original refusal stands
    monkeypatch.setenv(ar.RESUME_ENV, '1')
    assert ar.resume_failed_episode(tmp_path/'jobs'/job['id'])
    seen = []
    monkeypatch.setattr(ar, 'replacement_world', lambda w, used, taken, arm, ident: seen.append((used, taken)) or
                        dict(seed=989230, cell_key=w['cell_key']))
    new = ar.resumed_replacement(job, set(range(989220, 989228)), {job['id']}, 't2c_warm_u001')
    assert new['id'] == 't2c_warm_u001_R_989230' and new['replaces'] == job['id'] and new['world']['seed'] == 989230
    assert job['id'] in seen[0][1] and 989224 in seen[0][0]
    receipt = json.loads((tmp_path/'jobs'/job['id']/'EPISODE_FAILED_RESUMED.json').read_text())
    assert receipt['claims_replayed'] is False and receipt['original_receipt_bytes'] == 0
    monkeypatch.setattr(ar, 'replacement_world', lambda *a: pytest.fail('second resume must reuse the receipt'))
    assert ar.resumed_replacement(job, set(), set(), 't2c_warm_u001')['id'] == new['id']
    (tmp_path/'jobs'/job['id']/'RESULT.json').write_text('{}')
    assert not ar.resume_failed_episode(tmp_path/'jobs'/job['id'])          # a completed job is never "resumed"


def test_resume_keeps_dead_owner_records(tmp_path, monkeypatch):
    p = tmp_path/'owners'/'R.json'
    p.parent.mkdir()
    p.write_text('{}')
    monkeypatch.delenv(ar.RESUME_ENV, raising=False)
    assert lt._resume_name(p) == p
    monkeypatch.setenv(ar.RESUME_ENV, '1')
    q = lt._resume_name(p)
    assert q != p and q.name.startswith('R.resume_') and q.suffix == '.json'
    assert lt._resume_name(tmp_path/'owners'/'B.json').name == 'B.json'


def test_disk_floor_waits(monkeypatch):
    import tools.coopnative.runner as prior
    monkeypatch.setattr(prior, 'admission', lambda b: None)
    free = iter([10*1024**3, 61*1024**3])
    monkeypatch.setattr(runner, '_disk_free_bytes', lambda path: next(free))
    slept = []
    monkeypatch.setattr(runner.time, 'sleep', lambda s: slept.append(s))
    monkeypatch.setenv('COOPMWM_DISK_MIN_GB', '60')
    runner.admission(dict(load_max=40, run_root='.'))
    assert slept == [30]
    monkeypatch.setenv('COOPMWM_DISK_MIN_GB', '5')
    with pytest.raises(ValueError):
        runner.admission(dict(load_max=40, run_root='.'))


def test_prune_consumed_g4_data_records_hashes_first(tmp_path):
    root = tmp_path
    cap = root/'jobs'/'t2c_warm_u001_B_989220'/'g4_captures'
    cap.mkdir(parents=True)
    (cap/'c0.bin').write_bytes(b'state'*100)
    (root/'jobs'/'t2c_warm_u001_B_989220'/'RESULT.json').write_text('{}')
    (root/'jobs'/'t2c_warm_u002_B_989230'/'g4_captures').mkdir(parents=True)
    (root/'jobs'/'t2c_warm_u002_B_989230'/'g4_captures'/'c1.bin').write_bytes(b'x')
    (root/'g4').mkdir()
    (root/'g4'/'B_INPUT_001.json').write_text('{"a": 1}')
    with pytest.raises(ValueError, match='not consumed'):
        dp.prune_update(root, 1)
    (root/'LEARNING_MONITOR_u001.json').write_text('{}')
    rec = dp.prune_update(root, 1)
    assert [c['path'] for c in rec['captures']] == ['jobs/t2c_warm_u001_B_989220/g4_captures/c0.bin'.replace('/', __import__('os').sep)]
    assert rec['captures'][0]['sha256'] == __import__('hashlib').sha256(b'state'*100).hexdigest()
    assert not (cap/'c0.bin').exists() and (root/'jobs'/'t2c_warm_u001_B_989220'/'RESULT.json').exists()
    assert json.loads(gzip.open(root/'g4'/'B_INPUT_001.json.gz').read()) == {'a': 1}
    assert not (root/'g4'/'B_INPUT_001.json').exists() and (root/'pruned'/'PRUNE_u001.json').exists()
    assert (root/'jobs'/'t2c_warm_u002_B_989230'/'g4_captures'/'c1.bin').exists()        # u2 untouched
    assert dp.prune_update(root, 1) is None                                                # idempotent
    _ = SimpleNamespace


def test_replacement_world_is_receipted_out_of_the_fairness_term(tmp_path):
    """B = R: an episode on a world outside the update's registered worlds (failure replacement) has no same-world
    A episode; it is marked and skipped with a receipt, never failing closed after its sim-s are spent."""
    from . import fair_vehicle_log as fv, fullpol_fair as ff
    (tmp_path/'fair').mkdir()
    fv.a_labels_path(tmp_path, 1).write_text(json.dumps({'989220': {'v1': [False, 'L0']}}))
    table = {'v1': [False, 'L0', 12.0]}
    log = dict(ticks=900)
    eps = [dict(id='R1', world=dict(seed=989220), vehicle_table=table, vehicle_log=log),
           dict(id='R2', world=dict(seed=989230), vehicle_table=table, vehicle_log=log)]
    out = fv.attach(tmp_path, 1, eps, registered_seeds={989220, 989221})
    assert out['replacement_skipped'] == 1 and 'vehicle_groups' in eps[0] and eps[1]['fair_unpaired_replacement']
    with pytest.raises(ValueError, match='fail closed'):
        fv.attach(tmp_path, 1, [dict(eps[1], world=dict(seed=989221))], registered_seeds={989220, 989221})
    _, receipt = ff.penalties({}, 'R', eps, {'989220': eps[0]['vehicle_groups']})
    assert [s['episode_id'] for s in receipt['skipped']] == ['R2'] and len(receipt['episodes']) == 1


def test_resume_counts_escalation_limits(tmp_path, monkeypatch):
    for seed in (1, 2, 3):
        d = tmp_path/'jobs'/f't2c_warm_u001_R_{seed}'
        d.mkdir(parents=True)
        (d/'EPISODE_FAILED.json').write_text('')
    monkeypatch.setattr(ar, 'replacement_world', lambda *a: dict(seed=99))
    job = dict(id='t2c_warm_u001_R_1', path=str(tmp_path/'jobs'/'t2c_warm_u001_R_1'), arm='R', world=dict(seed=1),
               noise_seed=1)
    with pytest.raises(ar.EpisodeFailureEscalation):
        ar.resumed_replacement(job, set(range(8)), set(), 't2c_warm_u001')


def test_disk_floor_alerts_then_times_out(tmp_path, monkeypatch):
    import tools.coopnative.runner as prior
    monkeypatch.setattr(prior, 'admission', lambda b: None)
    monkeypatch.setattr(runner, '_disk_free_bytes', lambda path: 0)
    clock = iter(range(0, 10**6, 3600))
    monkeypatch.setattr(runner.time, 'time', lambda: next(clock))
    monkeypatch.setattr(runner.time, 'sleep', lambda s: None)
    monkeypatch.setenv('COOPMWM_DISK_MIN_GB', '60')
    with pytest.raises(RuntimeError, match='timeout'):
        runner.admission(dict(load_max=40, run_root=str(tmp_path)))
    assert list(tmp_path.glob('DISK_FLOOR_WAIT_*.json'))


def test_interrupted_prune_resumes_safely(tmp_path):
    (tmp_path/'g4').mkdir()
    src = tmp_path/'g4'/'R_INPUT_001.json'
    src.write_text('{"b": 2}')
    (tmp_path/'g4'/'R_INPUT_001.json.gz.tmp').write_bytes(b'partial garbage')        # crash mid-gzip
    (tmp_path/'LEARNING_MONITOR_u001.json').write_text('{}')
    rec = dp.prune_update(tmp_path, 1)
    assert rec['inputs'] and not src.exists()
    assert json.loads(gzip.open(tmp_path/'g4'/'R_INPUT_001.json.gz').read()) == {'b': 2}
    src.write_text('{"c": 3}')                                                       # stale gz must not unlink it
    with pytest.raises(ValueError, match='original kept'):
        dp.prune_update(tmp_path, 1)
    assert src.exists()
