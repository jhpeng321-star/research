"""Option A cross-host parity gate (operator; run on cpuZ AFTER the REGISTRATION entry, before any switch).

ONE identical arm-R TRAIN job dict runs twice, charged through the run's normal ledger (2 x 1200 sim-s):
  * locally on cpuZ through the same spawned ProcessPoolExecutor path as accel_runner.wave, and
  * on cpuV through RemotePool (remote_wave.py: replica identity check, admission, detached nice-19 worker,
    rsync landing + sha manifest, RESULT pointer last).
The job dict is built by the REAL builder (accel_runner.wave_jobs) from R's runtime for a COMPLETED update K
(R's normal-wave policies/extras are captured from ArmRuntime.collect with _wave intercepted; nothing is written),
on a fresh operator-chosen TRAIN world under a fresh parity ident. Both runs use job id P; only the local copy's
directory differs (jobs/P__local, ledger row P__local; the remote copy is jobs/P, ledger row P), so every
id-bearing byte (RESULT payload, g4 capture payloads) must be identical.

Compared (sha256): RESULT.json pointer + RESULT.json.gz, every g4_captures/*.pt, the DECOMPRESSED observable tape
(gzip headers carry a wall-clock mtime), and every other file except CLAIM.json (pid). The remote worker also
records every file it opened outside its job dir (audit hook); those files are hashed on both hosts.
Exit 0 + OPTIONA_PARITY_<tag>.json verdict PASS only when everything matches; otherwise FAIL (do not switch).

Safety: refuses seeds 964000-964199, 962xxx, 977xxx, 9906xx, 989600-989699, any seed registered in the run's
worlds_by_update and any existing job dir for the seed. A hard event in the parity episode triggers the same
registered safety pause as any R episode (run it while the owners are between updates).

Usage (same env as the R owner: env.sh + run-chain COOPMWM_* vars + COOPMWM_REMOTE_R/_KEY/_KNOWN_HOSTS):
  python -B ops/optiona_parity.py --config <run>/CONFIG.json --update K --tag T \
      --world-json '{"seed": S, "cell": C, ...}' [--dry-run]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FORBIDDEN_SEEDS = (range(964000, 964200), range(962000, 963000), range(977000, 978000), range(990600, 990700),
                   range(989600, 989700))


class _Captured(Exception):
    def __init__(self, cfg, policies, worlds, ident, extras):
        super().__init__('captured')
        self.args_ = (cfg, policies, worlds, ident, extras)


def check_world(cfg, root, world):
    seed = int(world['seed'])
    if any(seed in r for r in FORBIDDEN_SEEDS):
        raise SystemExit(f'seed {seed} is in a reserved/forbidden range')
    registered = {int(w['seed']) for ws in cfg['worlds_by_update'].values() for w in ws}
    if seed in registered:
        raise SystemExit(f'seed {seed} is a registered training world of this run')
    if any((root/'jobs').glob(f'*_{seed}')) or any((root/'jobs').glob(f'*_{seed}__local')):
        raise SystemExit(f'seed {seed} already has a job directory in this run')


def build_job(cfg, native, update, world, tag):
    """The real wave_jobs dict for R at update K (policies/extras captured from collect; read-only)."""
    from tools.coopmwm import accel_runner, g4_runtime as g4, track2_backend as tb
    from tools.coopmwm import practice_starts as ps
    from tools.coopmwm.storage import ledger
    from tools.coopmwm.v2v_runner import patched
    root = Path(native['run_root'])
    if not (root/f'R_UPDATE_{update:03}.json').exists():
        raise SystemExit(f'update {update} must be COMPLETED by R (read-only capture of its wave inputs)')
    if ps.enabled(native, 'R') and not ps.plan_path(native['run_root'], 'R', update).exists():
        raise SystemExit('R practice plan of that update is missing (the capture would write one)')

    def capture(cfg_, policies, worlds, ident, extras, **kwargs):
        raise _Captured(cfg_, policies, worlds, ident, extras)

    def plan_count(cfg_, u):   # read-only: never parses B_INPUT, never writes the early receipt
        return len(g4.read(g4.b_plan_path(cfg_, u))['rows'])
    hooks = tb.build_arm_runtime(native, 'R')
    try:
        with patched(tb, dict(_wave=capture)), patched(g4, dict(paired_practice_count=plan_count)):
            hooks['collect_update']()
    except _Captured as captured:
        cfg_, policies, _, ident, extras = captured.args_
    else:
        raise SystemExit('collect did not reach a wave')
    extras = {k: v for k, v in extras.items() if k not in ('practice_batch', 'practice', 'branch')}
    ident = cfg.get('job_namespace', 'clean1')+f'_optiona_parity_{tag}'
    phase = native.get('phase_map', {}).get('train', 'track2_train')
    accounting = native.get('accounting_root', native['run_root'])
    with patched(accel_runner, dict(ledger=lambda r: ledger(accounting))):
        jobs, results = accel_runner.wave_jobs(root, cfg_, policies, [world], 'train', ident, phase, False, extras)
    if results or len(jobs) != 1 or jobs[0]['arm'] != 'R':
        raise SystemExit('parity job must be one fresh arm-R job')
    return jobs[0], phase


def _tree(directory):
    """{name: sha256}; *.gz compared by decompressed bytes; CLAIM.json (pid) skipped."""
    out = {}
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file() or path.name == 'CLAIM.json':
            continue
        name = path.relative_to(directory).as_posix()
        if path.suffix == '.gz' and path.name != 'RESULT.json.gz':
            with gzip.open(path, 'rb') as stream:
                out[name+'#decompressed'] = hashlib.sha256(stream.read()).hexdigest()
        else:
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--update', type=int, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--world-json', required=True)
    parser.add_argument('--phase')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.tag.replace('-', '').isalnum():
        raise SystemExit('tag must be alphanumeric')
    from tools.coopmwm import launch_training as lt, launch_production as production
    from tools.coopmwm import track2_backend as tb, remote_wave, remote_worker as rw
    from tools.coopmwm.accel_plant import episode
    from tools.coopmwm.accel_storage import read
    from tools.coopmwm.runner import admission
    cfg = lt.checked_config(args.config)
    root = Path(cfg['run_root'])
    world = json.loads(args.world_json)
    check_world(cfg, root, world)
    native = production.backend_config(cfg)
    native['update'] = args.update
    job, phase = build_job(cfg, native, args.update, world, args.tag)
    phase = args.phase or phase
    ident = job['id']
    local_job = dict(job, path=str(root/'jobs'/(ident+'__local')))
    report = dict(tag=args.tag, update=args.update, world=world, job_id=ident, local_dir=f'jobs/{ident}__local',
                  remote_dir=f'jobs/{ident}', phase=phase, quote_sim_s=2*job['sim_s'],
                  differing_job_keys=sorted(k for k in job if job[k] is not local_job[k]))   # ['path'] only
    if args.dry_run:
        print(json.dumps(dict(report, dry_run=True), sort_keys=True, indent=1))
        return 0
    if not remote_wave.enabled():
        raise SystemExit('COOPMWM_REMOTE_R (+ key/known_hosts) must be set')
    lt.health(root)
    pool = remote_wave.RemotePool(root, native, [job], audit_open=True)   # replica identity BEFORE any charge
    tb._reserve(native, [dict(id=ident, sim_s=job['sim_s']), dict(id=ident+'__local', sim_s=job['sim_s'])], phase)
    admission(native)
    started = time.time()
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as local:
        with pool:
            remote_future = pool.submit(episode, job)
            local_future = local.submit(episode, local_job)
            local_error = remote_error = None
            try:
                local_future.result()
            except BaseException as error:   # noqa: BLE001 - reported, never retried
                local_error = repr(error)
            try:
                remote_future.result()
            except BaseException as error:   # noqa: BLE001
                remote_error = repr(error)
    report.update(seconds=time.time()-started, local_error=local_error, remote_error=remote_error)
    if local_error is None and remote_error is None:
        left, right = _tree(root/'jobs'/(ident+'__local')), _tree(root/'jobs'/ident)
        report['files'] = dict(local=left, remote=right)
        report['mismatched'] = sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))
        pointers = {side: root/'jobs'/name/'RESULT.json' for side, name in (('local', ident+'__local'),
                                                                             ('remote', ident))}
        read(pointers['local']), read(pointers['remote'])          # both gzip payloads verify against their pointer
        report['result_payload_sha256'] = {side: json.loads(p.read_text())['sha256'] for side, p in pointers.items()}
        landed = json.loads((root/rw.STAGE/ident/'LANDED.json').read_text())
        opened = sorted(p for p in (landed.get('opened') or ()) if Path(p).is_absolute())
        payload = dict(source_root=str(pool.source_root), inputs=opened, envsh=pool.envsh)
        here = rw.op_check(root, payload, replica=False)['inputs']
        there = pool.transport.call('check', root, payload=payload)['inputs']
        report['opened_files'] = len(opened)
        report['opened_mismatched'] = sorted(p for p in opened if here.get(p) != there.get(p))
    passed = (local_error is None and remote_error is None and report.get('mismatched') == []
              and report.get('opened_mismatched') == [] and len(report['files']['local']) > 0)
    report['verdict'] = 'PASS' if passed else 'FAIL'
    report['identity'] = pool.identity['remote']
    rw._publish_bytes(root/f'OPTIONA_PARITY_{args.tag}.json', rw._json_bytes(report))
    print(json.dumps(dict(verdict=report['verdict'], mismatched=report.get('mismatched'),
                          opened_mismatched=report.get('opened_mismatched'), local_error=local_error,
                          remote_error=remote_error), sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
