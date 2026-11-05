"""SHA-bound owner safety recovery over the unchanged qualified launch config.

The only numerical-runtime patches are the explicit B/R safety actions. Original smoke,
gate, config, models, completed updates and physics retain their exact bindings.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

from . import launch_training as training, composite_health
from .exact_grade import verify_sealed
from .protocol import digest

PATCHED = {'pytools/tools/coopmwm/composite_health.py',
           'pytools/tools/coopmwm/warm_health.py'}


def checked_recovery(config_path):
    cfg=training.checked_config(config_path)
    training.release(cfg)
    root=Path(cfg['run_root'])
    b_amended=(root/composite_health.B_RULE).exists()
    authorization=training.read(root/(composite_health.B_RULE if b_amended else composite_health.R_RULE))
    verify_sealed(authorization,'receipt_sha256')
    if not composite_health.r_report_only(root,'R'):
        raise ValueError('owner R-only amendment required')
    if b_amended and not composite_health.report_only(root,'B'):
        raise ValueError('owner B-only amendment required')
    source=Path(__file__).resolve().parents[3]
    manifest=training.read(source/'SOURCE_MANIFEST.json')
    if (authorization.get('config_sha256')!=digest(cfg)
            or authorization.get('recovery_source_manifest_sha256')!=digest(manifest)
            or authorization.get('recovery_source_root')!=str(source)):
        raise ValueError('recovery source/config authorization mismatch')
    original=training.read(Path(cfg['source_root'])/'SOURCE_MANIFEST.json')
    for name,sha in manifest['files'].items():
        if training.rc.sha(source/name)!=sha:
            raise ValueError('immutable recovery source changed: '+name)
        if name.endswith('.py') and original['files'].get(name)!=sha:
            allowed=(name in PATCHED or name=='pytools/tools/coopmwm/launch_recovery.py'
                     or Path(name).name.startswith('test_'))
            if not allowed:raise ValueError('unexpected executable recovery change: '+name)
    if not set(original['files']).issubset(manifest['files']):
        raise ValueError('recovery removed qualified source files')
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['arm','monitor','check'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--arm',choices=['A','B','R'])
    args=parser.parse_args()
    cfg=checked_recovery(args.config)
    if args.action=='check':
        print(json.dumps(dict(status='RECOVERY_SOURCE_CONFIG_VERIFIED',config_sha256=digest(cfg))))
        return
    root=Path(cfg['run_root'])
    import fcntl
    # Full-lifetime role locks protect against duplicate recovery dispatches.
    with (root/('RECOVERY_1230_'+(args.arm or 'monitor')+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.action=='arm':training.run_arm(cfg,args.arm)
            else:training.run_monitor(cfg)
        except BaseException as error:
            fault=dict(unix=time.time(),pid=training.os.getpid(),arm=args.arm,
                       error=repr(error),traceback=traceback.format_exc())
            for name in ('FAULT.json','STOP.json'):
                if not (root/name).exists():training.write(root/name,fault)
            raise


if __name__=='__main__':main()
