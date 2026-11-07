"""Migration resume (COOPMWM_MIGRATION_RESUME=<tag>) CPU contracts: no SUMO, explicit test doubles only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from . import accel_runner, fair_vehicle_log, fullpol_fair, g4_runtime as runtime, launch_training as launch
from . import migration_resume as mr, track2_backend as tb
from .exact_grade import _seal
from .test_g4_runtime_cpu import toy  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(mr.ENV, raising=False)
    monkeypatch.delenv(accel_runner.RESUME_ENV, raising=False)


def plain(tmp_path, **extra):
    return dict(run_root=str(tmp_path), job_namespace='t2c',
                worlds_by_update={str(u): [dict(seed=11), dict(seed=12)] for u in range(1, 25)}, **extra)


def charge(root, *rows):
    with (Path(root)/'CHARGES.jsonl').open('a', encoding='utf-8') as stream:
        for ident, sim_s in rows:
            stream.write(json.dumps(dict(id=ident, phase='track2_train', sim_s=sim_s))+'\n')


def job(root, ident, *names):
    path = Path(root)/'jobs'/ident
    path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (path/name).write_text('{}\n', encoding='utf-8')


def listing(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*'))


# ---------------------------------------------------------------- tag / suffix / absent env

@pytest.mark.parametrize('value', ['', 'W1', 'w-1', 'w_1', 'w 1', 'a'*13, 'inc1', 'xdriver', 'j4bx', 'sealed'])
def test_tag_validation_fails_closed(monkeypatch, value):
    assert mr.env_tag() is None
    monkeypatch.setenv(mr.ENV, value)
    with pytest.raises(ValueError, match='COOPMWM_MIGRATION_RESUME'):
        mr.env_tag()


@pytest.mark.parametrize('value', ['w1', '7', 'abc123', 'z'*12])
def test_valid_tags(monkeypatch, value):
    monkeypatch.setenv(mr.ENV, value)
    assert mr.env_tag() == value


def test_absent_env_keeps_every_id_path_and_noise_seed(tmp_path, monkeypatch):
    c = plain(tmp_path)
    assert mr.prepare(c, 'B', 3, 24) is None and listing(tmp_path) == []   # no write, nothing receipted
    assert mr.ident(c, 'B', 3) == 't2c_warm_u003' and mr.g4_suffix(c, 3) == ''
    assert runtime.input_path(c, 'R', 3) == tmp_path/'g4'/'R_INPUT_003.json'
    assert runtime._envelope_path(c, 3) == tmp_path/'g4'/'ENVELOPE_003.json'
    assert fair_vehicle_log.a_labels_path(str(tmp_path), 3) == tmp_path/'fair'/'A_LABELS_u003.json'
    assert fair_vehicle_log.a_groups_path(str(tmp_path), 3) == tmp_path/'fair'/'A_GROUPS_u003.json'
    assert fullpol_fair.reference_path(c, 3) == tmp_path/'fair'/'A_GROUPS_u003.json'
    assert mr.update_marker(c, 'B', 3) is None
    assert mr.seed_ident('t2c_warm_u003mw1') == 't2c_warm_u003mw1'          # identity without the variable
    record = tmp_path/'owners'/'B.json'
    record.parent.mkdir(parents=True)
    record.write_text('{}')
    assert launch._resume_name(record) == record                          # original no-overwrite contract
    monkeypatch.setenv(mr.ENV, 'w1')
    # set, but no receipt for this update (e.g. updates after the re-attempt): still the original ids/paths
    assert mr.ident(c, 'B', 4) == 't2c_warm_u004' and mr.g4_suffix(c, 4) == ''
    assert launch._resume_name(record).name.startswith('B.resume_')        # dead owner record kept, new beside it


def test_seed_ident_strips_only_this_tag(monkeypatch):
    monkeypatch.setenv(mr.ENV, 'w1')
    assert mr.seed_ident('t2c_warm_u003mw1') == 't2c_warm_u003'
    assert mr.seed_ident('t2c_warm_u003mw1_practice') == 't2c_warm_u003_practice'
    assert mr.seed_ident('t2c_warm_u003mw1_B_practiceref') == 't2c_warm_u003_B_practiceref'
    assert mr.seed_ident('t2c_warm_u003mw12') == 't2c_warm_u003mw12'
    assert mr.seed_ident('t2c_warm_u003') == 't2c_warm_u003'
    failed = dict(id='t2c_warm_u003mw1_B_11', arm='B', path='/r/jobs/t2c_warm_u003mw1_B_11', world=dict(seed=11))
    tagged = accel_runner.replacement_job(failed, dict(seed=13), 't2c_warm_u003mw1')
    monkeypatch.delenv(mr.ENV)
    original = accel_runner.replacement_job(dict(failed, id='t2c_warm_u003_B_11'), dict(seed=13), 't2c_warm_u003')
    assert tagged['id'] == 't2c_warm_u003mw1_B_13' and original['id'] == 't2c_warm_u003_B_13'
    assert tagged['noise_seed'] == original['noise_seed']                 # same CRN as the original update


# ---------------------------------------------------------------- receipt

def abandoned_root(tmp_path):
    root = tmp_path
    charge(root, ('t2c_warm_u002_B_11', 1200), ('t2c_warm_u003_A_11', 1200),          # completed / other arm
           ('t2c_warm_u003_B_11', 1200), ('t2c_warm_u003_B_12', 1200),
           ('t2c_warm_u003_p01_B_40', 1200), ('t2c_warm_u003_B_practiceref_pa01_A_40', 1200),
           ('t2c_warm_u003_R_11', 1200))
    job(root, 't2c_warm_u003_B_11', 'CLAIM.json', 'RESULT.json', 'CLOSE.json')
    job(root, 't2c_warm_u003_B_12', 'CLAIM.json')
    job(root, 't2c_warm_u003_p01_B_40', 'CLAIM.json', 'EPISODE_FAILED.json')
    (root/'updates').mkdir()
    (root/'updates'/'B_003_STARTED.json').write_text('{}')
    return root


def test_receipt_contents_and_idempotency(tmp_path, monkeypatch):
    c = plain(tmp_path)
    root = abandoned_root(tmp_path)
    monkeypatch.setenv(mr.ENV, 'w1')
    receipt = mr.prepare(c, 'B', 3, 24)
    path = root/'MIGRATION_RESUME_B_003_w1.json'
    assert json.loads(path.read_text()) == receipt
    assert receipt['reason'] == 'spot reclaim migration' and receipt['tag'] == 'w1'
    assert receipt['namespace'] == dict(reattempt='t2c_warm_u003mw1', abandoned=['t2c_warm_u003'])
    jobs = {j['id']: j for j in receipt['abandoned']['jobs']}
    assert set(jobs) == {'t2c_warm_u003_B_11', 't2c_warm_u003_B_12', 't2c_warm_u003_p01_B_40',
                         't2c_warm_u003_B_practiceref_pa01_A_40'}
    assert {i: j['state'] for i, j in jobs.items()} == {
        't2c_warm_u003_B_11': 'RESULT', 't2c_warm_u003_B_12': 'CLAIM', 't2c_warm_u003_p01_B_40': 'CLAIM',
        't2c_warm_u003_B_practiceref_pa01_A_40': 'none'}
    assert jobs['t2c_warm_u003_p01_B_40']['episode_failed'] and jobs['t2c_warm_u003_B_11']['close']
    assert all(j['charged_sim_s'] == 1200 and j['ledger'] for j in jobs.values())
    assert receipt['abandoned']['charged_sim_s'] == 4800 and receipt['abandoned']['g4_plans'] == []
    assert receipt['abandoned']['process_records'] == ['updates/B_003_STARTED.json']
    ledger = (root/'CHARGES.jsonl').read_bytes()
    # idempotent: a second resume with the same tag reuses the sealed receipt (no rewrite, ledger untouched)
    stamp = path.stat().st_mtime_ns
    assert mr.prepare(c, 'B', 3, 24) == receipt and path.stat().st_mtime_ns == stamp
    assert (root/'CHARGES.jsonl').read_bytes() == ledger
    # the re-attempt namespace is tagged; the abandoned ids are never reused or adopted
    assert mr.ident(c, 'B', 3) == 't2c_warm_u003mw1' and mr.suffix(root, 'B', 3) == 'mw1'
    assert mr.ident(c, 'B', 4) == 't2c_warm_u004'                              # later updates are normal
    marker = mr.update_marker(c, 'B', 3)
    assert marker['tag'] == 'w1' and marker['receipt_sha256'] == receipt['receipt_sha256']
    # without the variable an in-flight receipted update fails closed (it is never silently run untagged)
    monkeypatch.delenv(mr.ENV)
    with pytest.raises(ValueError, match='requires COOPMWM_MIGRATION_RESUME'):
        mr.suffix(root, 'B', 3)
    with pytest.raises(ValueError, match='resume it with'):
        mr.prepare(c, 'B', 3, 24)
    # another runtime config never reuses the receipt
    monkeypatch.setenv(mr.ENV, 'w1')
    with pytest.raises(ValueError, match='another runtime config'):
        mr.prepare(dict(c, training_seed=1), 'B', 3, 24)
    # an edited receipt fails its seal
    edited = dict(receipt, reason='edited')
    path.unlink()
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match='receipt_sha256 mismatch'):
        mr.suffix(root, 'B', 3)


def test_completed_reattempt_resolves_from_its_sealed_update(tmp_path, monkeypatch):
    c = plain(tmp_path)
    monkeypatch.setenv(mr.ENV, 'w1')
    mr.prepare(c, 'A', 3, 24)
    assert fair_vehicle_log.a_labels_path(str(tmp_path), 3).name == 'A_LABELS_u003mw1.json'
    marker = mr.update_marker(c, 'A', 3)
    (tmp_path/'A_UPDATE_003.json').write_text(json.dumps(_seal(dict(arm='A', migration_resume=marker),
                                                              'update_sha256')))
    # B/R (or a later resume under another tag, or none) read A's re-attempt namespace of u3
    monkeypatch.setenv(mr.ENV, 'w2')
    assert mr.ident(c, 'A', 3) == 't2c_warm_u003mw1'
    monkeypatch.delenv(mr.ENV)
    assert mr.ident(c, 'A', 3) == 't2c_warm_u003mw1'
    assert fullpol_fair.reference_path(c, 3).name == 'A_GROUPS_u003mw1.json'
    (tmp_path/'A_UPDATE_003.json').unlink()
    (tmp_path/'A_UPDATE_003.json').write_text(json.dumps(dict(arm='A')))   # receipted but completed untagged
    with pytest.raises(ValueError, match='disagrees with its migration receipts'):
        mr.ident(c, 'A', 3)


def test_different_tag_after_completed_tagged_reattempt_fails_closed(tmp_path, monkeypatch):
    c = plain(tmp_path)
    root = abandoned_root(tmp_path)
    monkeypatch.setenv(mr.ENV, 'w1')
    mr.prepare(c, 'B', 3, 24)
    charge(root, ('t2c_warm_u003mw1_B_11', 1200), ('t2c_warm_u003mw1_B_12', 1200))
    job(root, 't2c_warm_u003mw1_B_11', 'CLAIM.json', 'RESULT.json')
    job(root, 't2c_warm_u003mw1_B_12', 'CLAIM.json', 'RESULT.json')
    monkeypatch.setenv(mr.ENV, 'w2')
    with pytest.raises(ValueError, match='completed its collection'):
        mr.prepare(c, 'B', 3, 24)
    assert not (root/'MIGRATION_RESUME_B_003_w2.json').exists()


def test_different_tag_after_partial_tagged_reattempt_chains_both_abandoned_attempts(tmp_path, monkeypatch):
    c = plain(tmp_path)
    root = abandoned_root(tmp_path)
    monkeypatch.setenv(mr.ENV, 'w1')
    first = mr.prepare(c, 'B', 3, 24)
    charge(root, ('t2c_warm_u003mw1_B_11', 1200), ('t2c_warm_u003mw1_B_12', 1200))
    job(root, 't2c_warm_u003mw1_B_11', 'CLAIM.json', 'RESULT.json')
    job(root, 't2c_warm_u003mw1_B_12', 'CLAIM.json')                       # reclaimed again mid-collection
    monkeypatch.setenv(mr.ENV, 'w2')
    second = mr.prepare(c, 'B', 3, 24)
    assert second['namespace']['abandoned'] == ['t2c_warm_u003', 't2c_warm_u003mw1']
    assert second['previous_receipts'] == [dict(tag='w1', receipt='MIGRATION_RESUME_B_003_w1.json',
                                                receipt_sha256=first['receipt_sha256'])]
    assert {'t2c_warm_u003mw1_B_11', 't2c_warm_u003mw1_B_12'} <= {j['id'] for j in second['abandoned']['jobs']}
    assert second['abandoned']['charged_sim_s'] == 4800+2400
    assert mr.ident(c, 'B', 3) == 't2c_warm_u003mw2'


@pytest.mark.parametrize('anomaly', ['later_ledger', 'later_started', 'later_update', 'own_checkpoint',
                                     'tag_namespace_used'])
def test_unexpected_state_fails_closed_before_any_receipt(tmp_path, monkeypatch, anomaly):
    c = plain(tmp_path)
    root = abandoned_root(tmp_path)
    if anomaly == 'later_ledger':
        charge(root, ('t2c_warm_u004_B_11', 1200))
    elif anomaly == 'later_started':
        (root/'updates'/'B_004_STARTED.json').write_text('{}')
    elif anomaly == 'later_update':
        (root/'B_UPDATE_005.json').write_text('{}')
    elif anomaly == 'own_checkpoint':
        (root/'B_checkpoint_003.pt').write_bytes(b'x')
    else:
        charge(root, ('t2c_warm_u003mw1_B_11', 1200))
    monkeypatch.setenv(mr.ENV, 'w1')
    with pytest.raises(ValueError, match='fail closed'):
        mr.prepare(c, 'B', 3, 24)
    assert not list(root.glob('MIGRATION_RESUME_*'))


def test_br_reattempt_requires_completed_a_payloads(tmp_path, monkeypatch):
    c = plain(tmp_path)
    (tmp_path/'A_UPDATE_003.json').write_text(json.dumps(dict(arm='A')))
    for seed in (11, 12):
        job(tmp_path, f't2c_warm_u003_A_{seed}', 'CLAIM.json')
        (tmp_path/'jobs'/f't2c_warm_u003_A_{seed}'/'RESULT.json').write_text(json.dumps(
            dict(encoding='coopmwm-lossless-gzip-v1', file='RESULT.json.gz', sha256='0'*64)))
    (tmp_path/'jobs'/'t2c_warm_u003_A_11'/'RESULT.json.gz').write_bytes(b'x')
    monkeypatch.setenv(mr.ENV, 'w1')
    with pytest.raises(ValueError, match='t2c_warm_u003_A_12/RESULT.json.gz'):
        mr.prepare(c, 'R', 3, 24)
    (tmp_path/'jobs'/'t2c_warm_u003_A_12'/'RESULT.json.gz').write_bytes(b'x')
    assert mr.prepare(c, 'R', 3, 24)['a_reference_namespace'] == 't2c_warm_u003'


# ---------------------------------------------------------------- G4 pairing (real Coordinator)

def g4_setup(cfg, monkeypatch):
    from .test_sparse_synergy_credit_cpu import planned, captures
    pure, episodes, _, _ = planned()
    cfg.update(pure)
    cfg['worlds_by_update']['2'] = cfg['worlds_by_update']['1']
    cfg['budget'] = dict(total=100000, phases=dict(g4_synergy_credit=60000, train=10000))
    monkeypatch.setattr(tb, '_approved', lambda cfg: None)

    def named(suffix, update=1):
        out = deepcopy(episodes)
        for arm, es in out.items():
            for e in es:
                e['id'] = f'clean1_warm_u{update:03}{suffix}_{arm}_7'
        for arm, es in out.items():
            es[0]['g4_captures'] = captures(out)[arm]
        return out
    return named


def failing(job, cfg):
    raise RuntimeError('reclaimed mid G4 (test double)')


def graded(job, cfg):
    from .test_sparse_synergy_credit_cpu import measured
    return measured(job)                     # explicit 'sumo' receipt test double


def test_g4_reattempt_chains_after_abandoned_plan_and_refuses_duplicates(cfg, monkeypatch):
    named = g4_setup(cfg, monkeypatch)
    root = Path(cfg['run_root'])
    ledger = root/'CHARGES.jsonl'
    original = named('')
    with pytest.raises(RuntimeError, match='reclaimed'):
        runtime.Coordinator(cfg, worker=failing).measure(1, original)
    abandoned_bytes = ledger.read_bytes()
    abandoned = next(json.loads(line)['g4_plan'] for line in abandoned_bytes.splitlines() if b'g4_plan' in line)
    with pytest.raises(ValueError, match='G4 reserved attempt incomplete'):   # today's rule, unchanged
        runtime.Coordinator(cfg, worker=graded).measure(1, original)
    monkeypatch.setenv(mr.ENV, 'w1')
    receipt_b = mr.prepare(cfg, 'B', 1, 24)
    mr.prepare(cfg, 'R', 1, 24)
    [plan_row] = receipt_b['abandoned']['g4_plans']
    assert plan_row['plan_sha256'] == abandoned['plan_sha256'] and plan_row['envelope_present'] is False
    assert plan_row['charged_sim_s'] == abandoned['quote_sim_s'] == sum(j['charged_sim_s'] for j in plan_row['jobs'])
    assert runtime.input_path(cfg, 'B', 1).name == 'B_INPUT_001mw1.json'
    assert runtime._envelope_path(cfg, 1).name == 'ENVELOPE_001mw1.json'
    again = named('mw1')
    envelope = runtime.Coordinator(cfg, worker=graded).measure(1, again)
    plan = envelope['plan']
    marker = plan['migration_resume']
    assert marker['tag'] == 'w1' and marker['supersedes_plan_sha256'] == abandoned['plan_sha256']
    assert plan['reservation_before'] == dict(abandoned['reservation_after'],
                                              last_update=abandoned['reservation_before']['last_update'])
    assert plan['reservation_after']['reserved_sim_s'] == 2*abandoned['quote_sim_s']    # the abandoned stays spent
    assert ledger.read_bytes().startswith(abandoned_bytes)                             # append-only, continuous
    assert runtime._envelope_path(cfg, 1).exists() and not (root/'g4'/'ENVELOPE_001.json').exists()
    # adoption of the persisted tagged envelope; never a second reservation for the tagged attempt
    before = ledger.read_bytes()
    assert runtime.Coordinator(cfg, worker=failing).measure(1, again) == envelope
    assert ledger.read_bytes() == before
    # the abandoned episodes can never be credited by the tagged attempt (their opportunities stay reserved)
    with pytest.raises(ValueError, match='G4 restart inputs changed|G4 opportunity already reserved'):
        runtime.Coordinator(cfg, worker=graded).measure(1, original)
    assert mr.check_pair(cfg, 1) == 'mw1'                                      # B and R share the tag
    # without the variable the receipted in-flight update fails closed
    monkeypatch.delenv(mr.ENV)
    with pytest.raises(ValueError, match='COOPMWM_MIGRATION_RESUME'):
        runtime.Coordinator(cfg, worker=graded).measure(1, again)
    # the next (untagged) update chains after the tagged plan; the walk accepts the marked history
    later = runtime.Coordinator(cfg, worker=graded).measure(2, named('', update=2))
    assert 'migration_resume' not in later['plan']
    assert later['plan']['reservation_before'] == plan['reservation_after']


def test_g4_tagged_reservation_without_envelope_is_never_retried(cfg, monkeypatch):
    named = g4_setup(cfg, monkeypatch)
    with pytest.raises(RuntimeError):
        runtime.Coordinator(cfg, worker=failing).measure(1, named(''))
    monkeypatch.setenv(mr.ENV, 'w1')
    mr.prepare(cfg, 'B', 1, 24)
    mr.prepare(cfg, 'R', 1, 24)
    with pytest.raises(RuntimeError):
        runtime.Coordinator(cfg, worker=failing).measure(1, named('mw1'))
    ledger = (Path(cfg['run_root'])/'CHARGES.jsonl').read_bytes()
    with pytest.raises(ValueError, match='G4 reserved attempt incomplete'):
        runtime.Coordinator(cfg, worker=graded).measure(1, named('mw1'))
    assert (Path(cfg['run_root'])/'CHARGES.jsonl').read_bytes() == ledger
    assert mr.prepare(cfg, 'B', 1, 24)['tag'] == 'w1'                     # same tag: the receipt is reused
    # reclaimed again before any w1 collection result: a w2 re-attempt chains after BOTH abandoned plans
    plans = [json.loads(line)['g4_plan'] for line in ledger.splitlines() if b'g4_plan' in line]
    monkeypatch.setenv(mr.ENV, 'w2')
    receipt = mr.prepare(cfg, 'B', 1, 24)
    mr.prepare(cfg, 'R', 1, 24)
    assert [p['plan_sha256'] for p in receipt['abandoned']['g4_plans']] == [p['plan_sha256'] for p in plans]
    assert [p['tag'] for p in receipt['abandoned']['g4_plans']] == [None, 'w1']
    envelope = runtime.Coordinator(cfg, worker=graded).measure(1, named('mw2'))
    marker = envelope['plan']['migration_resume']
    assert marker['tag'] == 'w2' and marker['supersedes_plan_sha256'] == plans[1]['plan_sha256']
    assert marker['rewound_last_update'] == plans[0]['reservation_before']['last_update'] == 0
    assert envelope['plan']['reservation_before']['reserved_sim_s'] == sum(p['quote_sim_s'] for p in plans)


def test_g4_no_abandoned_plan_reserves_plainly_chained(cfg, monkeypatch):
    named = g4_setup(cfg, monkeypatch)
    monkeypatch.setenv(mr.ENV, 'w1')
    mr.prepare(cfg, 'B', 1, 24)
    mr.prepare(cfg, 'R', 1, 24)
    envelope = runtime.Coordinator(cfg, worker=graded).measure(1, named('mw1'))
    marker = envelope['plan']['migration_resume']
    assert marker['supersedes_plan_sha256'] is None and marker['rewound_last_update'] is None


def test_g4_partner_completed_fails_closed(cfg, monkeypatch):
    g4_setup(cfg, monkeypatch)
    (Path(cfg['run_root'])/'R_UPDATE_001.json').write_text('{}')
    monkeypatch.setenv(mr.ENV, 'w1')
    with pytest.raises(ValueError, match='completed G4-paired update 1'):
        mr.prepare(cfg, 'B', 1, 24)


def test_g4_envelope_of_another_attempt_is_refused_by_r(cfg, monkeypatch):
    named = g4_setup(cfg, monkeypatch)
    envelope = runtime.Coordinator(cfg, worker=graded).measure(1, named(''))     # an untagged envelope
    monkeypatch.setenv(mr.ENV, 'w1')
    mr.prepare(cfg, 'R', 1, 24)
    mr.prepare(cfg, 'B', 1, 24)
    episodes = named('mw1')['R']
    with pytest.raises(ValueError, match='not the migration re-attempt'):
        runtime.paired_envelope(cfg, 'R', 1, episodes, wait=lambda path: envelope)


# ---------------------------------------------------------------- end to end: real launcher, toy physics (no SUMO)

def test_real_launcher_reattempt_after_reclaim_mid_g4(toy, monkeypatch):
    """B and R abandon update 1 after collection + G4 reservation (worker dies, no envelope); only the essentials
    survive (no g4/). With the tag, both owners receipt and re-attempt update 1 from scratch: new job ids, tagged G4
    input/envelope, the G4 plan chained after the abandoned one, and sealed UPDATEs that carry the marker."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    cfg, root = toy.cfg, Path(toy.cfg['run_root'])
    toy.fail_worker[0] = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(toy.update, a) for a in ('B', 'R')]:
            with pytest.raises((RuntimeError, ValueError)):
                future.result(timeout=180)
    abandoned_bytes = (root/'CHARGES.jsonl').read_bytes()
    abandoned = [json.loads(line) for line in abandoned_bytes.splitlines()]
    shutil.rmtree(root/'g4')                              # not part of the essentials backup
    toy.fail_worker[0] = False
    # untagged, the abandoned update can never complete (today's rules: here the dead start record already blocks)
    with pytest.raises(OSError):
        launch.update(cfg, 'B', 1)
    monkeypatch.setenv(mr.ENV, 'w1')
    receipts = {a: mr.prepare(cfg, a, 1, 24) for a in ('B', 'R')}    # the arm owners, before the re-attempt
    assert {j['id'] for j in receipts['B']['abandoned']['jobs']} == {'clean1_warm_u001_B_7'}
    assert {j['id'] for j in receipts['R']['abandoned']['jobs']} == {'clean1_warm_u001_R_7'}
    [g4_plan] = receipts['B']['abandoned']['g4_plans']
    assert g4_plan['charged_sim_s'] == sum(r['sim_s'] for r in abandoned if r['id'].startswith('g4_'))
    import time

    def wait(path, root):                              # a fresh barrier (the toy's records the abandoned failure)
        deadline = time.monotonic()+120
        while not Path(path).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(.01)
        return launch.read(path)
    monkeypatch.setattr(launch, 'wait_for', wait)

    def resume(arm):
        import torch
        torch.set_num_threads(1)
        return launch.update(cfg, arm, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {a: pool.submit(resume, a) for a in ('B', 'R')}
        updates = {a: future.result(timeout=180) for a, future in futures.items()}
    ledger = (root/'CHARGES.jsonl').read_bytes()
    assert ledger.startswith(abandoned_bytes)                 # abandoned rows stay, byte-for-byte
    rows = [json.loads(line) for line in ledger.splitlines()][len(abandoned):]
    # (the toy wave runs episodes without charging them; the G4 batch is charged by the real Coordinator)
    assert all((root/'jobs'/f'clean1_warm_u001mw1_{a}_7'/'RESULT.json').exists() for a in ('B', 'R'))
    plan = next(r['g4_plan'] for r in rows if 'g4_plan' in r)
    assert plan['migration_resume']['supersedes_plan_sha256'] == g4_plan['plan_sha256']
    assert (root/'g4'/'B_INPUT_001mw1.json').exists() and (root/'g4'/'ENVELOPE_001mw1.json').exists()
    for arm in ('B', 'R'):
        value = updates[arm]
        assert value['migration_resume']['tag'] == 'w1'
        assert value['migration_resume']['receipt_sha256'] == receipts[arm]['receipt_sha256']
        assert [e['id'] for e in value['physical']] == [f'clean1_warm_u001mw1_{arm}_7']
        assert value['learning']['g4_synergy_credit']['plan_sha256'] == plan['plan_sha256']
        assert launch.read(root/f'{arm}_UPDATE_001.json') == value
        assert any((root/'updates').glob(f'{arm}_001_STARTED.resume_*.json'))   # dead start record kept
    # completed: resolution comes from the sealed UPDATEs, with or without the variable
    monkeypatch.delenv(mr.ENV)
    assert mr.g4_suffix(cfg, 1) == 'mw1' and mr.ident(cfg, 'R', 1) == 'clean1_warm_u001mw1'
