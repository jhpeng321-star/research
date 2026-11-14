"""A102TS (time-space trace re-run; prepared 2026-11-14, NOT launched by its author).

Re-runs exactly ONE world, 988700 (the first A94 panel world, fixed in advance), for the two A94 panel jobs
a94c_mappo60c_0000 (MAPPO60C alone) and a94c_bank_searched_0000 (A75_BANK on the S5 bank 577a0564 over MAPPO60C
2abed317), with raw output KEPT (no drop-raw): RESULT.json.gz, vehicles.jsonl.gz, observable.jsonl.gz stay under
<root>/evaluation/<job>/.

THIN WRAPPER over the unchanged A94 driver ops/a94_bank.py (and through it the unchanged A81 driver). Specs are built by
the A94 function a94_bank.arm_specs with the A94 tags (job ids a94c_mappo60c_0000 / a94c_bank_searched_0000), then
sliced to index 0 (= world 988700). Changes vs A94, all bookkeeping: own root (/root/a102ts/root, HOST_BASE
/root/a102ts), own ledger phase 'tstrace_a102', phase cap 2,400 sim-s (2 x 1,200; enforced by the A42 runtime and
again here), own log A102TS_LOG.jsonl, no mechanics gate (identity is checked after the run by a102ts_check.py against
the stored A94 records), no drop-raw, workers <= 2.

Commands (cwd = $SRC; env from env_a102ts.sh; A94_HAND_SHA256=none, A94_BANK_SHA256=577a0564...):
  prepare     zero SUMO: write-once bank copies into <root>/a94_banks (a94_bank.prepare)
  dry         zero SUMO: build the two specs; identity vs the stored A94 panel records (checkpoint / world / window /
              arm / job_id / panel_sha256); prints the field diff (expected: only the bank path prefix)
  prephysics  zero SUMO: the runner pre-physics checks of a75_d2forms.execute (plant CONFIG vs PRERUN_PLAN, runner
              status / arms, map check, U0 binding), executor binding sha, a75 digest, reservation prefix + cap
  run         SUMO: the two episodes once (2,400 sim-s), workers <= 2, no drop-raw
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

SRC = Path(os.environ.get('SRC', '/root/a102ts/src'))
sys.path[:0] = [str(SRC/'ops'), str(SRC)]
if os.environ.get('A94_HAND_SHA256') != 'none':
    raise SystemExit('A102TS: set A94_HAND_SHA256=none (only BANK_SEARCHED among the bank arms is built)')

import a94_bank as A  # noqa: E402  (unchanged A94 driver)
import a81_bigconfirm as B  # noqa: E402

SEED = 988700
WORLD_ID = f'wz3to2_{SEED}'
ARMS = ('MAPPO60C', 'BANK_SEARCHED')
JOBS = {'MAPPO60C': 'a94c_mappo60c_0000', 'BANK_SEARCHED': 'a94c_bank_searched_0000'}
EPISODES = len(ARMS)
CAP = EPISODES*B.HORIZON                       # 2,400 sim-s
assert CAP == 2_400
PHASE = 'tstrace_a102'
MAPPO60C_SHA256 = '2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8'
S5_BANK_SHA256 = '577a0564944b2062e441d7ee603e7fe8090bede52a5dd6e8774d41ed9cb2d8b6'
A94_PANEL_SHA256 = 'd85999ddaaf3eb583775d66dbabca168b4432966bc01e7193dfc79f244f7946c'
A94_ROOT = Path('/root/a94/root')               # READ ONLY (stored panel records)
STORED_WAVES = {'MAPPO60C': ('WAVE_A94C_MAPPO60C__0000-0095.json',
                             'e126789b682c391159555d42a4d4bc0f6e134e26fd8765b087a0b12d0d4d626b'),
                'BANK_SEARCHED': ('WAVE_A94C_BANK_SEARCHED__0000-0095.json',
                                  '44951b2a6a9a737ffe37be3ad8d8fc223e82fc7c574ddefebd6635e17a689cc3')}
LOG_FILE = 'A102TS_LOG.jsonl'
MAX_WORKERS = 2
#: the only spec-vs-stored-record difference allowed: the bank copy lives in this root (same bytes, sha checked)
ALLOWED_DIFF = {'checkpoint.a75_bank.path'}

# bookkeeping overrides BEFORE a94_bank.point() / core() read them (a94_bank.core reads PHASE / CAP_SIM_S at call time)
A.PHASE = PHASE
A.CAP_SIM_S = CAP
A.CONFIRM_CAP = CAP
A.point()
B.CAP_SIM_S = CAP
B.CONFIRM_CAP = CAP
B.STAGE_CAP = dict(mech=0, confirm=CAP)
B.LOG_FILE = LOG_FILE


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _setup(root):
    os.environ['A81_PANEL_SHA256'] = A94_PANEL_SHA256
    D, panel = B.setup('c', root)
    if panel is None or panel['panel_sha256'] != A94_PANEL_SHA256:
        raise SystemExit('A102TS: panel missing or not d85999dd')
    if panel['worlds'][0]['world']['seed'] != SEED:
        raise SystemExit('A102TS: panel world 0 is not 988700')
    if D.A42_PHASE != PHASE or D.A42_CAP_SIM_S != CAP or D.JOB_PREFIX != 'a94':
        raise SystemExit(f'A102TS: core overrides not applied ({D.A42_PHASE}, {D.A42_CAP_SIM_S}, {D.JOB_PREFIX})')
    return D, panel


def waves(D, panel, root, a66_root):
    """[(wave, 'episode', [spec])]: the A94 specs of each arm (A94 tags), index 0 only."""
    out = []
    for arm in ARMS:
        specs = A.arm_specs(D, arm, panel, root, a66_root, f'{A.PREFIX["c"]}{A.TAG[arm]}')
        s = specs[0]
        if s['job_id'] != JOBS[arm] or s['world']['seed'] != SEED or s['world_id'] != WORLD_ID:
            raise SystemExit(f'A102TS: {arm} spec 0 is {s["job_id"]} / {s["world"]["seed"]}')
        out.append((f'A94C_{arm}', 'episode', [s]))
    return out


def _flat(x, path=''):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(_flat(v, f'{path}.{k}' if path else str(k)))
        return out
    return {path: x}


def identity(ws, root):
    """Spec fields the runner copies into the episode record vs the stored A94 panel record of the same job."""
    out, ok = {}, True
    for (_, _, (s,)), arm in zip(ws, ARMS):
        name, pin = STORED_WAVES[arm]
        p = A94_ROOT/name
        if _sha(p) != pin:
            raise SystemExit(f'A102TS: stored {name} is not {pin[:8]}')
        rec = [r for r in json.loads(p.read_text())['records'] if r['job_id'] == JOBS[arm]]
        if len(rec) != 1:
            raise SystemExit(f'A102TS: {JOBS[arm]} not in {name}')
        rec = rec[0]
        keys = ('checkpoint', 'world', 'window', 'world_id', 'job_id', 'panel_sha256')
        mine = _flat({k: s.get(k) for k in keys})
        theirs = _flat({k: rec.get(k) for k in keys})
        diff = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        arm_ok = set(diff) <= ALLOWED_DIFF
        for k in diff:
            if k == 'checkpoint.a75_bank.path':
                arm_ok = arm_ok and (str(theirs[k]).replace('/root/a94/root/', f'{root}/') == str(mine[k])
                                     and Path(mine[k]).is_file() and _sha(mine[k]) == S5_BANK_SHA256
                                     and mine.get('checkpoint.a75_bank.file_sha256') == S5_BANK_SHA256)
        ok = ok and arm_ok
        out[arm] = dict(job_id=s['job_id'], seed=s['world']['seed'], spec_arm=s['arm'], record_arm=rec['arm'],
                        executor_sha256=s['checkpoint'].get('executor_sha256'), compared_fields=len(mine),
                        diff={k: [theirs.get(k), mine.get(k)] for k in diff}, ok=arm_ok,
                        stored_receipt=json.loads(p.read_text())['receipt_sha256'],
                        spec_keys=sorted(s), sim_s=s.get('sim_s'),
                        phase_in_spec=[k for k, v in _flat(s).items() if isinstance(v, str) and
                                       ('searchedbank_a94' in v or PHASE in v)])
        if s['checkpoint'].get('executor_sha256') != MAPPO60C_SHA256:
            ok = False
    return ok, out


def prephysics(D, panel, root, a66_root, ws):
    """The pre-physics part of a75_d2forms.execute (no episode), plus executor / digest / reservation checks."""
    N = B._mods()[0]
    D2 = B._D2()
    import a32_ablation as A32
    from tools.coopmwm import nm_eval_runner as R
    from tools.coopmwm import a75_holdforms as H
    out = {}
    out['a75_digest'] = H.a75_digest()
    assert out['a75_digest'] == D2.A75_SHA256, 'a75_holdforms digest is not d1af7e0f'
    runtime = D.A42Runtime(N.Runtime(N.RUNNER), root)
    assert runtime.physics and runtime.phase == PHASE, (runtime.physics, runtime.phase)
    cfg = runtime.config(root/N.PRERUN_PLANT)
    out['plant_config'] = 'CONFIG == PRERUN_PLAN'
    assert os.environ.get('COOPMWM_MAP_SPEC'), 'COOPMWM_MAP_SPEC unset'
    status = runtime.runner_status()
    assert status['ok'] and status.get('learned'), status
    assert getattr(R, 'A75_ARMS', None) == ('A75_PLAN', 'A75_BANK') and getattr(R, 'A37_ARMS', None) == D.C37.A37_ARMS
    assert tuple(getattr(R, 'A94_RANGES', ())) == (A.BLOCK,)
    N.map_check(N.bind_map_api(None), panel)
    out['runner'] = dict(module=N.RUNNER, ok=status['ok'], learned=status['learned'], map_check='ok')
    plan = N.prerun_plan(root)
    u0, _ = A32.bindings(root)
    assert (u0['residual_path'], u0['residual_sha256']) == (plan['u0']['path'], plan['u0']['sha256'])
    out['u0_binding'] = 'residual == PRERUN_PLAN u0'
    exe = A.B._mods()[3].executor(a66_root)
    out['executor'] = {k: exe.get(k) for k in ('sha256', 'binding_sha256', 'path') if k in exe}
    assert exe['sha256'] == MAPPO60C_SHA256
    for _, _, (s,) in ws:
        ck = s['checkpoint']
        assert ck['executor_sha256'] == MAPPO60C_SHA256 and _sha(ck['executor_path']) == MAPPO60C_SHA256
    out['executor_file_sha256'] = MAPPO60C_SHA256
    jobs = [dict(id=s['job_id'], sim_s=s['sim_s']) for _, _, (s,) in ws]
    assert all(str(j['id']).startswith(D.JOB_PREFIX) for j in jobs), 'reservation prefix'
    out['reservation'] = dict(prefix=D.JOB_PREFIX, ids=[j['id'] for j in jobs], sim_s=sum(j['sim_s'] for j in jobs),
                              charged=D.charged(root))
    runtime.check_capacity(sum(j['sim_s'] for j in jobs))              # must pass
    try:
        runtime.check_capacity(sum(j['sim_s'] for j in jobs)+B.HORIZON)
        raise AssertionError('cap not enforced')
    except N.Blocked as e:
        out['cap_enforced'] = str(e)[:160]
    out['published_before'] = {w: N.published(root, w, ss, D2.check_d2)[2] for w, _, ss in ws}
    out['cfg_keys'] = len(cfg)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('prepare', 'dry', 'prephysics', 'run'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--a66-root', required=True)
    p.add_argument('--a75d2-root', required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args(argv)
    N = B._mods()[0]
    D2 = B._D2()
    root = Path(a.final_root)
    if root.resolve() == A94_ROOT.resolve() or not str(root).startswith('/root/a102ts/'):
        raise SystemExit('A102TS: the final root is /root/a102ts/root only')
    if os.environ.get('A94_BANK_SHA256') != S5_BANK_SHA256:
        raise SystemExit('A102TS: A94_BANK_SHA256 must be the S5 bank 577a0564...')
    D, panel = _setup(root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    if a.command == 'prepare':
        print(json.dumps(A.prepare(D, root, a.a75d2_root), default=str))
        return
    ws = waves(D, panel, root, a.a66_root)
    if a.command == 'dry':
        ok, rows = identity(ws, root)
        print(json.dumps(dict(stage='dry', phase=PHASE, cap=CAP, panel_sha256=panel['panel_sha256'],
                              waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[0]['world']['seed']) for n, _, s in ws],
                              episodes=EPISODES, sim_s=sum(s[0]['sim_s'] for _, _, s in ws), charged=D.charged(root),
                              identity=rows, IDENTITY_OK=ok), indent=1, default=str))
        if not ok:
            sys.exit(6)
        return
    if a.command == 'prephysics':
        print(json.dumps(dict(PREPHYSICS_OK=True, **prephysics(D, panel, root, a.a66_root, ws)), indent=1, default=str))
        return
    # run
    import fcntl
    if a.workers < 1 or a.workers > MAX_WORKERS:
        raise SystemExit(f'A102TS: workers must be 1..{MAX_WORKERS}')
    ok, rows = identity(ws, root)
    if not ok:
        raise SystemExit(f'A102TS: spec identity failed: {json.dumps(rows, default=str)[:1500]}')
    with (root/'OWNER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
        pre = prephysics(D, panel, root, a.a66_root, ws)
        pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in ws)
        if pending != EPISODES or D.charged(root) != 0:
            raise SystemExit(f'A102TS: pending {pending} charged {D.charged(root)}: runs once only, never a retry')
        if D.charged(root)+pending*B.HORIZON > CAP:
            raise SystemExit('A102TS: cap')
        log(dict(start='run', phase=PHASE, episodes=EPISODES, workers=a.workers, pid=os.getpid(), drop_raw=False,
                 charged_before=D.charged(root), jobs=[s[0]['job_id'] for _, _, s in ws], executor=pre['executor']))
        res = D2.execute(D, root, ws, workers=a.workers, chunk=EPISODES, log=log, panel=panel)
        log(dict(done='run', **res))
        if res.get('status') != 'PASS':
            sys.exit(8)


if __name__ == '__main__':
    main()
