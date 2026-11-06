"""clean5b-F logger identity canary (MAIN 2026-11-07): the per-vehicle logger must be READ-ONLY on the plant.

One registered TRAIN world, frozen A arm (u0 base, same noise seed), run twice through accel_plant.episode in fresh
nice-19 processes: OFF = the b58daad8 plant (no logger code), ON = this head with job['vehicle_log']=True. PASS iff
the decompressed observable trace bytes, the per-second prefix fingerprints, the final fingerprint, totals and every
recorded step are identical (the ON result only adds vehicle_log / vehicle_table). Charges 2 x 1200 sim-s to its own
ledger (CANARY_LEDGER.jsonl in the output dir; carried as prior in the clean5b-F quote).

Usage (on the host, env.sh of --src-on sourced, G5/LC/MAP env set):
  python ops/clean5bf_logger_identity.py --config <prepared clean5b-F CONFIG.json> --src-off <b58daad8 tree>
         --src-on <this tree> --out <new dir>
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

CHILD = r'''
import json, pickle, sys
from pathlib import Path
job = pickle.loads(Path(sys.argv[1]).read_bytes())
from tools.coopmwm import accel_plant
result = accel_plant.episode(job)
print(json.dumps(dict(id=result['id'], vehicle_log=result.get('vehicle_log'))))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--src-off', required=True)
    p.add_argument('--src-on', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=False)
    from tools.coopmwm import launch_production as production, track2_backend as tb, accel_runner
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
    native = production.backend_config(cfg)
    world = cfg['worlds_by_update']['1'][0]
    runtime = tb.build_arm_runtime(native, 'A')['runtime']
    extras = dict(update=1, v2v=True, v2v_dim=cfg['v2v_dim'], b_replay=False, health_guards=False)
    jobs = {}
    for label, vlog in (('off', False), ('on', True)):
        root = out/label
        root.mkdir()
        x = dict(extras, **(dict(vehicle_log=True) if vlog else {}))
        (job,), _ = accel_runner.wave_jobs(root, native, {'A': runtime.base}, [world], 'train', 'canary_u001',
                                          'clean5bf_canary', False, x)
        jobs[label] = job
    common = {k for k in jobs['off'] if k not in ('path', 'queue_root', 'vehicle_log')}
    if any(repr(jobs['off'][k]) != repr(jobs['on'][k]) for k in common if k != 'weights') or \
            any(not jobs['off']['weights'][k].equal(jobs['on']['weights'][k]) for k in jobs['off']['weights']):
        raise ValueError('canary jobs differ beyond path/vehicle_log')
    ledger = out/'CANARY_LEDGER.jsonl'
    for label, src in (('off', a.src_off), ('on', a.src_on)):
        job_file = out/f'{label}.job.pkl'
        job_file.write_bytes(pickle.dumps(jobs[label]))
        with ledger.open('a') as f:
            f.write(json.dumps(dict(id=f'clean5bf_logger_identity_{label}', phase='clean5bf_canary', sim_s=1200,
                                    unix=time.time()))+'\n')
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([f'{src}/pytools', f'{src}/runtime_src', f'{src}/WM_PACKAGE',
                   '/home/ubuntu/e3final/runtime/src', '/home/ubuntu/e3final/runtime']))
        with (out/f'{label}.log').open('x') as log:
            code = subprocess.run(['nice', '-n', '19', '/home/ubuntu/miniconda3/bin/python', '-B', '-c', CHILD,
                                   str(job_file)], cwd='/home/ubuntu', env=env, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL).returncode
        if code:
            raise SystemExit(f'{label} episode failed ({code}); see {out}/{label}.log')
    res = {k: json.loads((Path(jobs[k]['path'])/'RESULT.json').read_text()) for k in ('off', 'on')}

    def trace(k):
        return gzip.open(Path(jobs[k]['path'])/'observable.jsonl.gz').read()
    t_off, t_on = trace('off'), trace('on')
    checks = dict(
        observable_trace=t_off == t_on,
        prefixes=res['off']['prefixes'] == res['on']['prefixes'],
        final=res['off']['final'] == res['on']['final'],
        totals=res['off']['totals'] == res['on']['totals'],
        steps=json.dumps(res['off']['steps'], sort_keys=True) == json.dumps(res['on']['steps'], sort_keys=True),
        result_keys=set(res['on'])-set(res['off']) == {'vehicle_log', 'vehicle_table'},
        logger=res['on']['vehicle_log']['ticks'] == 900 and len(res['on']['vehicle_table']) > 0)
    verdict = dict(schema='coopmwm.clean5bf.logger_identity.v1', status='PASS' if all(checks.values()) else 'FAIL',
                   checks=checks, world=world, src_off=a.src_off, src_on=a.src_on,
                   trace_sha256=dict(off=hashlib.sha256(t_off).hexdigest(), on=hashlib.sha256(t_on).hexdigest()),
                   vehicles_logged=len(res['on']['vehicle_table']), sim_s=2400, unix=time.time())
    (out/'IDENTITY.json').write_text(json.dumps(verdict, indent=1, sort_keys=True))
    print(json.dumps(dict(status=verdict['status'], checks=checks)))
    if verdict['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
