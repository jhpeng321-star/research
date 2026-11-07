"""resume3 (operator, after REV-5BF 04108d37): a second resume adopts the recorded replacement (no re-submit, no new
charge); a recorded replacement with a CLAIM but no RESULT fails closed; ONE receipt-derived marker drives the
safety / fairness / G4 exclusions (seed AND cell checked); a restarted B adopts its published G4 input by streamed
sha256 (never re-loaded or copied); with COOPMWM_RESUME_FAILED_EPISODES unset every path is unchanged."""
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import accel_runner as ar, fair_vehicle_log as fv, g4_runtime as g4, launch_training as lt
from . import sparse_synergy_credit as credit, track2_backend as tb
from .test_g4_runtime_cpu import toy  # noqa: F401
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401

REAL_SAFETY = tb.ArmRuntime.safety          # captured before the toy fixture stubs it
IDENT = 't2c_warm_u001'
SEEDS = range(989220, 989228)
FAILED, NEXT = f'{IDENT}_R_989224', f'{IDENT}_R_989228'


def world(seed, cell=5):
    return dict(seed=seed, cell=cell, repeat=seed % 400, penetration=.6, demand=3200, map='wz3to2',
                cell_key='D3200_p60' if cell == 5 else f'cell{cell}')


REG = [world(s) for s in SEEDS]


def result(job_id, w, arm='R'):
    return dict(id=job_id, world=w, arm=arm, steps=[], metrics={'900': dict(hard_brake=0.)}, actual_sim_s=1200,
                g4_source='ordinary_training', g4_captures={})


def next_world(failed, used, taken, arm, ident):
    """accel_runner.replacement_world's rule on a CPU registry: next unused same-cell TRAIN seed."""
    seed = next(s for s in range(989220, 989600) if s not in used and f'{ident}_{arm}_{s}' not in taken)
    return world(seed, failed['cell'])


class Inline:
    """accel_runner.wave's ProcessPoolExecutor, run inline (completed futures)."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, job):
        future = Future()
        try:
            future.set_result(fn(job))
        except BaseException as error:
            future.set_exception(error)
        return future

    def map(self, fn, jobs):
        return [fn(j) for j in jobs]


@pytest.fixture
def wave_env(tmp_path, monkeypatch):
    """R u1 on disk as on cpuZ: 7 completed RESULTs, 989224 claimed + truncated EPISODE_FAILED (disk full)."""
    from . import accel_plant, policy_wm, runner
    monkeypatch.setenv(ar.RESUME_ENV, '1')
    charges = [dict(id=f'{IDENT}_R_{s}', sim_s=1200, phase='track2_train') for s in SEEDS]
    reserved, calls, fail = [], [], set()
    monkeypatch.setattr(ar, 'ledger', lambda root: list(charges))

    def reserve(root, jobs, phase):
        reserved.extend(j['id'] for j in jobs)
        charges.extend(dict(id=j['id'], sim_s=j['sim_s'], phase=phase) for j in jobs)
    monkeypatch.setattr(ar, 'reserve', reserve)
    monkeypatch.setattr(runner, 'admission', lambda b: None)
    monkeypatch.setattr(ar, 'ProcessPoolExecutor', Inline)
    monkeypatch.setattr(policy_wm, 'actor_state', lambda policy: {})
    monkeypatch.setattr(ar, 'replacement_world', next_world)

    def episode(job):                        # accel_plant.episode's CLAIM-first / RESULT-last contract
        calls.append(job['id'])
        path = Path(job['path'])
        ar.write(path/'CLAIM.json', dict(id=job['id']))
        if job['id'] in fail:
            raise RuntimeError('No space left on device (CPU fixture)')
        out = result(job['id'], job['world'], job['arm'])
        ar.write(path/'RESULT.json', out)
        return out
    monkeypatch.setattr(accel_plant, 'episode', episode)
    for s in SEEDS:
        d = tmp_path/'jobs'/f'{IDENT}_R_{s}'
        if s == 989224:
            d.mkdir(parents=True)
            (d/'CLAIM.json').write_text('{}')
            (d/'EPISODE_FAILED.json').write_text('')
        else:
            ar.write(d/'RESULT.json', result(d.name, world(s)))
    b = dict(runtime='rt', checkpoint='ck', accel_bounds=[-3., 2.6], gpu_deadline=None, workers=1,
             worlds_by_update={'1': REG})

    def run(ws=REG):
        return ar.wave(tmp_path, b, {'R': SimpleNamespace(state_dict=dict)}, ws, 'train', IDENT, 'track2_train',
                       False, dict(update=1))
    return SimpleNamespace(root=tmp_path, run=run, calls=calls, reserved=reserved, fail=fail, charges=charges)


def test_first_resume_runs_the_replacement_once_and_the_second_resume_adopts_it(wave_env):
    first = wave_env.run()
    by = {e['id']: e for e in first}
    assert wave_env.calls == [NEXT] and wave_env.reserved == [NEXT] and FAILED not in by and len(first) == 8
    marker = by[NEXT][ar.RESUMED_KEY]
    assert marker == ar.resumed_marker(wave_env.root/'jobs'/FAILED, NEXT) and marker['world']['seed'] == 989228
    assert [e['id'] for e in first if ar.RESUMED_KEY in e] == [NEXT]
    second = wave_env.run()                  # the update crashed later (e.g. old G4) and is resumed again
    assert wave_env.calls == [NEXT] and wave_env.reserved == [NEXT]          # no re-submit, no new charge
    assert json.dumps(second, sort_keys=True) == json.dumps(first, sort_keys=True)
    assert not (wave_env.root/'jobs'/NEXT/'EPISODE_FAILED.json').exists()     # no spurious failure
    assert not list((wave_env.root/'jobs').glob('*989229*'))                 # no second replacement


def test_recorded_replacement_is_never_rereplaced_and_claim_without_result_fails_closed(wave_env):
    wave_env.fail.add(NEXT)
    with pytest.raises(ar.EpisodeFailureEscalation, match='never re-replaced'):
        wave_env.run()
    assert wave_env.calls == [NEXT] and wave_env.reserved == [NEXT]
    assert ar.read(wave_env.root/'jobs'/NEXT/'EPISODE_FAILED.json')['replacement'] is None
    with pytest.raises(ValueError, match='never retried'):
        wave_env.run()                       # CLAIM, no RESULT: fail closed, nothing submitted or charged
    assert wave_env.calls == [NEXT] and wave_env.reserved == [NEXT]


def test_resume_replacement_never_takes_or_keeps_a_registered_seed(wave_env):
    out = wave_env.run(REG[:6])              # practice rows: the normal wave is a prefix of the registered worlds
    assert [e['world']['seed'] for e in out if ar.RESUMED_KEY in e] == [989228]       # not 989226 / 989227
    receipt = wave_env.root/'jobs'/FAILED/'EPISODE_FAILED_RESUMED.json'
    value = ar.read(receipt)
    receipt.unlink()
    ar.write(receipt, dict(value, replacement_world=world(989226)))   # a receipt recorded by the old rule
    with pytest.raises(ValueError, match='aliases a registered world seed'):
        wave_env.run(REG[:6])
    assert wave_env.calls == [NEXT]                                    # failed closed before any physics


def test_flag_off_paths_unchanged(wave_env, monkeypatch):
    monkeypatch.delenv(ar.RESUME_ENV)
    with pytest.raises(ValueError, match='attempted job cannot be retried'):
        wave_env.run()
    eps = [result(f'{IDENT}_R_{s}', world(s)) for s in SEEDS]
    assert lt._resume_view(dict(run_root=str(wave_env.root), worlds_by_update={'1': REG}), 1, eps) is None
    ordinary = dict(B=eps, R=eps)
    assert g4._resume_exclusions(dict(worlds_by_update={'1': REG}), 1, ordinary) is ordinary
    assert wave_env.calls == [] and wave_env.reserved == []


def test_restarted_monitor_keeps_the_dead_record(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, 'release', lambda c: None)
    monkeypatch.setattr(lt, 'registered_last_update', lambda c: 0)
    (tmp_path/'owners').mkdir()
    (tmp_path/'owners'/'monitor.json').write_text('{"pid": 307512}')
    cfg_ = dict(run_root=str(tmp_path), registered_stop='CPU fixture')
    monkeypatch.delenv(ar.RESUME_ENV, raising=False)
    with pytest.raises(FileExistsError):                  # flag off: unchanged (a restart would fault)
        lt.run_monitor(cfg_)
    monkeypatch.setenv(ar.RESUME_ENV, '1')
    lt.run_monitor(cfg_)
    assert len(list((tmp_path/'owners').glob('monitor.resume_*.json'))) == 1
    assert (tmp_path/'MONITOR_COMPLETE.json').exists()


def _resumed_episodes(env):
    eps = env.run()
    for e in eps:
        e.update(vehicle_table={'v1': [False, 'L0', 12.0]}, vehicle_log=dict(ticks=900))
    return eps


def test_marker_is_verified_against_its_receipt_and_seed_cell_aliasing_fails_closed(wave_env):
    eps = _resumed_episodes(wave_env)
    root = wave_env.root
    kept, out, receipts = ar.resumed_split(root, eps, REG)
    assert [e['id'] for e in out] == [NEXT] and len(kept) == 7
    assert receipts == [dict(episode_id=NEXT, world_seed=989228, cell=5, replaces=FAILED,
                             receipt=f'jobs/{FAILED}/EPISODE_FAILED_RESUMED.json', reason=ar.RESUMED_RULE)]
    rep = next(e for e in eps if e['id'] == NEXT)
    bad = [dict(rep, world=world(989229)),                                           # world not the recorded one
           dict(rep, **{ar.RESUMED_KEY: dict(rep[ar.RESUMED_KEY], replaces='t2c_warm_u001_R_989223')}),
           dict(rep, **{ar.RESUMED_KEY: dict(rep[ar.RESUMED_KEY], replaces='../x')}),
           dict(rep, practice=dict(slot='p01'))]
    for e in bad:
        with pytest.raises(ValueError, match='receipt'):
            ar.resumed_split(root, [e], REG)
    alias = dict(eps[0], world=world(989220, cell=6))       # registered seed, another cell (seed-keyed pairing)
    with pytest.raises(ValueError, match='aliases registered seed'):
        ar.resumed_split(root, [alias], REG, strict=False)
    with pytest.raises(ValueError, match='aliases registered world seed'):            # replacement on a registered seed
        ar.resumed_split(root, [rep], REG+[world(989228, cell=6)])
    stray = dict(eps[0], id=f'{IDENT}_R_989300', world=world(989300))                # unregistered, no receipt
    with pytest.raises(ValueError, match='no resume receipt'):
        ar.resumed_split(root, [stray], REG)
    assert ar.resumed_split(root, [stray], REG, strict=False)[0] == [stray]          # G4: its own rule decides
    practice = dict(eps[0], practice=dict(slot='p01'), world=world(989400, cell=9))
    assert ar.resumed_split(root, [practice], REG)[0] == [practice]


def test_r_update_with_one_receipted_replacement_passes_fairness_and_safety(wave_env, monkeypatch):
    eps = _resumed_episodes(wave_env)
    root = wave_env.root
    cfg_ = dict(run_root=str(root), job_namespace='t2c', worlds_by_update={'1': REG})
    resume = lt._resume_view(cfg_, 1, eps)
    assert [e['id'] for e in resume['replacements']] == [NEXT]
    (root/'fair').mkdir()
    fv.a_labels_path(root, 1).write_text(json.dumps({str(s): {'v1': [False, 'L0']} for s in SEEDS}))
    fair_log = fv.attach(root, 1, eps, registered_seeds={w['seed'] for w in REG})     # = update_step's call
    lt._resume_fair_check(resume, 'R', fair_log, eps)
    assert fair_log['replacement_skipped'] == 1 and fair_log['episodes'] == 7
    with pytest.raises(ValueError, match='fairness replacement skip disagrees'):
        lt._resume_fair_check(dict(resume, replacements=[]), 'R', fair_log, eps)
    for s in SEEDS:                                       # frozen A references exist only for registered worlds
        ar.write(root/'jobs'/f'{IDENT}_A_{s}'/'RESULT.json', result(f'{IDENT}_A_{s}', world(s), 'A'))
    runtime = SimpleNamespace(arm='R', cfg=cfg_, update=1)
    with pytest.raises(FileNotFoundError):                # 04108d37 behaviour: the replacement has no A reference
        REAL_SAFETY(runtime, eps)
    out = REAL_SAFETY(runtime, resume['kept'])
    assert out['status'] == 'checked' and out['normal_references'] == 7 and runtime.safety_checked
    gone = root/'jobs'/f'{IDENT}_A_989221'
    for p in gone.iterdir():
        p.unlink()
    gone.rmdir()
    with pytest.raises(FileNotFoundError):                # a registered world without its A reference: fail closed
        REAL_SAFETY(SimpleNamespace(arm='R', cfg=cfg_, update=1), resume['kept'])


def _g4cfg(cfg):
    return dict(cfg, g4_synergy_credit=dict(enabled=True, registration=credit.registration(),
                phase_id='g4-production-cpu', first_update=1), worlds_by_update={'1': REG})


def test_g4_excludes_receipted_replacements_and_still_raises_for_other_unregistered(wave_env, cfg):
    eps = _resumed_episodes(wave_env)
    gcfg = dict(_g4cfg(cfg), run_root=str(wave_env.root))
    b_eps = [dict(e, arm='B', id=e['id'].replace('_R_', '_B_')) for e in eps if ar.RESUMED_KEY not in e]
    b_eps.append(dict(b_eps[0], id=f'{IDENT}_B_989224', world=world(989224)))
    out = g4._resume_exclusions(gcfg, 1, dict(B=b_eps, R=eps))
    assert out['B'] == b_eps and [e['id'] for e in out['R']] == [e['id'] for e in eps if e['id'] != NEXT]
    stray = dict(eps[0], id=f'{IDENT}_R_989300', world=world(989300))
    with pytest.raises(ValueError, match='G4 requires complete registered ordinary TRAIN episodes'):
        credit.plan_update(gcfg, 1, g4._resume_exclusions(gcfg, 1, dict(B=b_eps, R=eps+[stray])), {},
                           credit.initial_state(gcfg))
    with pytest.raises(ValueError, match='G4 requires complete registered ordinary TRAIN episodes'):
        credit.plan_update(gcfg, 1, dict(B=b_eps, R=eps), {}, credit.initial_state(gcfg))   # 04108d37: FAULT


def test_published_sha_is_the_publish_bytes():
    value = dict(arm='B', binding=dict(z=1, a=[]), update=1, episodes=[
        dict(id='e1', steps=[dict(t=1, x=.1+.2, s='\u00e9\u4e2d', k=(1, 2), n=None, b=True), {}],
             ints={10: 'x', 3: 'y'}, nested=dict(deep=dict(deeper=[1.5e-300, -0.0, 10**20]))), {}, []])
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    assert g4.published_sha(value) == hashlib.sha256(raw).hexdigest()


def test_restarted_b_adopts_its_published_input_without_loading_or_copying(cfg, tmp_path, monkeypatch):
    gcfg = _g4cfg(cfg)
    eps = [result(f'{IDENT}_B_{s}', world(s), 'B') for s in SEEDS]
    path = g4.input_path(gcfg, 'B', 1)
    g4.publish(path, dict(binding=credit.settings(gcfg), update=1, arm='B', episodes=eps))
    before, listing = path.stat(), sorted(os.listdir(path.parent))
    real_read = g4.read
    monkeypatch.setattr(g4, 'read', lambda p: pytest.fail('the published input must not be re-loaded'))

    class Waiting(Exception):
        pass

    def wait(p):
        raise Waiting(Path(p).name)
    monkeypatch.setenv(ar.RESUME_ENV, '1')
    adopted = json.loads(json.dumps(eps))              # as re-derived from the RESULT files after a restart
    with pytest.raises(Waiting, match='R_INPUT_001'):
        g4.paired_envelope(gcfg, 'B', 1, adopted, wait=wait)
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino)
    assert sorted(os.listdir(path.parent)) == listing                       # nothing written, no temp copy
    monkeypatch.setattr(g4, 'read', real_read)
    with pytest.raises(ValueError, match='G4 published collection changed'):
        g4.paired_envelope(gcfg, 'B', 1, [dict(eps[0], ttc_events=99)]+eps[1:], wait=wait)
    monkeypatch.delenv(ar.RESUME_ENV)                  # flag off: the original semantic comparison
    seen = []
    monkeypatch.setattr(g4, 'read', lambda p: seen.append(p) or real_read(p))
    with pytest.raises(Waiting):
        g4.paired_envelope(gcfg, 'B', 1, adopted, wait=wait)
    assert seen == [path]


def test_restarted_b_and_resumed_r_through_the_real_launcher(toy, monkeypatch):  # noqa: F811
    """B is killed while waiting for R_INPUT (after publishing B_INPUT); a fresh B update (flag set) adopts its
    RESULTs and its published input, R resumes with one receipted replacement, B's coordinator excludes it."""
    cfg_, root = toy.cfg, Path(toy.cfg['run_root'])
    w7 = cfg_['worlds_by_update']['1'][0]
    w17, w27 = dict(w7, seed=17), dict(w7, seed=27)
    cfg_['worlds_by_update']['1'] = [w7, w17]
    ident = cfg_.get('job_namespace', 'clean1')+'_warm_u001'
    for w in (w7, w17):
        ar.write(root/'jobs'/f'{ident}_A_{w["seed"]}'/'RESULT.json', result(f'{ident}_A_{w["seed"]}', w, 'A'))
    monkeypatch.setattr(tb.ArmRuntime, 'safety', REAL_SAFETY)
    toy_wave, toy_wait = tb._wave, lt.wait_for
    monkeypatch.setattr(ar, 'replacement_world', lambda *args: w27)

    def wave(c, policies, worlds, ident_, extras, **kwargs):
        (arm,) = policies
        out, todo = [], []
        for w in worlds:
            path = root/'jobs'/f'{ident_}_{arm}_{w["seed"]}'
            if (path/'RESULT.json').exists():
                out.append(ar.read(path/'RESULT.json'))         # accel_runner.wave_jobs adoption
            elif arm == 'R' and w['seed'] == 17:                # disk-full failure -> recorded resume replacement
                path.mkdir(parents=True)
                (path/'CLAIM.json').write_text('{}')
                (path/'EPISODE_FAILED.json').write_text('')
                eight = {7, 17} | set(range(1001, 1007))       # escalation share counted over an 8-world wave
                new = ar.resumed_replacement(dict(id=path.name, path=str(path), arm='R', world=w, noise_seed=9),
                                             eight, set(), ident_, exclude={7, 17})
                done = toy_wave(c, policies, [new['world']], ident_, extras, **kwargs)[0]
                out.append(dict(done, **{ar.RESUMED_KEY: ar.resumed_marker(path, new['id'])}))
            else:
                todo.append(w)
        out += toy_wave(c, policies, todo, ident_, extras, **kwargs) if todo else []
        return sorted(out, key=lambda e: (e['world']['seed'], e['arm']))
    monkeypatch.setattr(tb, '_wave', wave)

    class Killed(BaseException):
        pass
    phase = [1]

    def wait(path, root_, **kwargs):
        if phase[0] == 1 and Path(path).name == 'R_INPUT_001.json':
            raise Killed()
        return toy_wait(path, root_)
    monkeypatch.setattr(lt, 'wait_for', wait)
    monkeypatch.delenv(ar.RESUME_ENV, raising=False)
    with pytest.raises(Killed):
        lt.update(cfg_, 'B', 1)                              # the old B u1 child, stopped at the R_INPUT wait
    b_input = g4.input_path(cfg_, 'B', 1)
    before, listing, physics = b_input.stat(), set(os.listdir(b_input.parent)), len(toy.calls)
    real_read = g4.read
    monkeypatch.setattr(g4, 'read', lambda p: pytest.fail('B input re-loaded') if Path(p) == b_input
                        else real_read(p))
    phase[0] = 2
    monkeypatch.setenv(ar.RESUME_ENV, '1')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {a: pool.submit(toy.update, a) for a in ('B', 'R')}
        receipts = {a: f.result(timeout=300) for a, f in futures.items()}
    rid = f'{ident}_R_27'
    assert not [p for p in toy.calls[physics:] if Path(p).name.startswith(f'{ident}_B_')]   # B never re-ran
    assert sum(Path(p).name == rid for p in toy.calls) == 1
    after = b_input.stat()
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (before.st_size, before.st_mtime_ns, before.st_ino)
    assert set(os.listdir(b_input.parent))-listing <= {'R_INPUT_001.json', 'ENVELOPE_001.json'}
    assert list((root/'updates').glob('B_001_STARTED.resume_*.json'))
    r = receipts['R']
    assert r['safety']['replacement_references_skipped'] == 1 and r['safety']['normal_references'] == 1
    assert r['resume_replacements']['count'] == 1 and r['resume_replacements']['episodes'][0]['episode_id'] == rid
    assert 'resume_replacements' not in receipts['B'] and 'replacement_receipts' not in receipts['B']['safety']
    assert receipts['B']['safety']['normal_references'] == 2
    r_input = g4.read(g4.input_path(cfg_, 'R', 1))['episodes']
    assert [e['id'] for e in r_input if e.get(ar.RESUMED_KEY)] == [rid]
    plan = next(json.loads(line)['g4_plan'] for line in (root/'CHARGES.jsonl').read_text().splitlines()
                if 'g4_plan' in json.loads(line))
    assert plan['jobs'] and all(j['start']['episode_id'] != rid for j in plan['jobs'])   # measured, R_27 excluded
    assert credit.opportunities([e for e in r_input if e['id'] == rid])       # it had candidates of its own
    assert all(receipts[a]['learning']['g4_synergy_credit']['plan_sha256'] == plan['plan_sha256'] for a in 'BR')
