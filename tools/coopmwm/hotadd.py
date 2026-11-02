"""Outcome-blind admission of an additional, independently qualified GPU worker."""
import argparse
import json
import os
from pathlib import Path
import time

GATES=('STARTUP_WARM_BATCH_SINGLE_K4_H120.json','STARTUP_WARM_BATCH_SINGLE_K8_H300.json',
       'STARTUP_PARITY_K4_H120.json','STARTUP_PARITY_K8_H300.json',
       'STARTUP_WARM_PARITY_K4_H120.json','STARTUP_WARM_PARITY_K8_H300.json',
       'STARTUP_COMMITTED_K4_H300.json','STARTUP_COMMITTED_K8_H420.json')


def qualified(status,expected_owner=None):
    if status['failed']:raise ValueError('new worker failed; preserve, never replay')
    owner=status['receipts'].get('SPOOL_OWNER.json')
    if expected_owner is not None and owner!=expected_owner:raise ValueError('hot-add owner changed')
    if not status['alive']:raise ValueError('hot-add worker exited')
    if not status['ready']:return False
    for name in GATES:
        gate=status['receipts'].get(name,{})
        if not gate.get('pass_') or gate.get('max_cost_error')!=0.:
            raise ValueError('missing/nonexact new-worker gate '+name)
    return True


def binding(original,hashes,host):
    """Operational batch/memory changes only; the numerical source is untouched."""
    from .warm_service import NUMERICAL
    for name in NUMERICAL:
        if hashes.get(name)!=original['hashes'][name]:raise ValueError('changed numerical source '+name)
    result=dict(original,hashes=hashes,memory_fraction=.42 if host=='gpuA' else .34,
        batch_width_by_K={'4':18 if host=='gpuA' else 4,'8':10 if host=='gpuA' else 4})
    # Fresh workers must run every gate; no inherited per-process qualification.
    result.pop('adopted_startup_gates',None)
    result.pop('batch_widths',None);result.pop('batch_width',None)
    assert result['deadline_unix']==original['deadline_unix']
    return result


def receipt_label(config):
    label=str(config.get('receipt_label',config['host']))
    if not label.replace('_','').replace('-','').isalnum():raise ValueError('hot-add receipt label must be alphanumeric')
    return label


def run(config_path):
    import fcntl
    import signal
    from .warm_broker import Remote
    from .warm_spool import immutable
    from .warm_health import assert_running
    from .warm_transport import retry
    config_path=Path(config_path);config=json.loads(config_path.read_text())
    root=Path(config['run_root'])
    # Receipt label: default keeps the original one-per-host names; a replacement
    # for a dead hot-added worker needs its own label (the old OWNER/ADMISSION/
    # FAILED receipts are immutable and would otherwise conflict forever).
    host=receipt_label(config)
    lock=(root/f'HOTADD_{host}.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
    immutable(root/f'HOTADD_{host}_OWNER.json',owner)
    remote=Remote(root,config['endpoint'])
    def verify():qualified(remote.raw(dict(action='status')),config['expected_owner'])
    def event(value):
        from .protocol import digest
        value=dict(value,unix=time.time())
        immutable(root/('HOTADD_'+host+'_TRANSPORT_'+digest(value)+'.json'),value)
    try:
        while time.time()<config['deadline_unix']:
            assert_running(root,'B')
            status=retry(lambda:remote.raw(dict(action='status')),verify,event)
            if qualified(status,config['expected_owner']):
                remote.verify()
                original=json.loads((root/'WARM_REMOTE_BINDING.json').read_text())
                if config['search_sha256']!=original['search_sha256']:raise ValueError('search hash differs')
                if config['deadline_unix']!=original['deadline_unix']:raise ValueError('deadline differs')
                if config['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():
                    raise ValueError('unregistered hot-add')
                value=dict(amendment_sha256=config['amendment_sha256'],
                    search_sha256=config['search_sha256'],endpoints=[config['endpoint']],
                    gates={n:status['receipts'][n] for n in GATES},owner=config['expected_owner'],
                    unix=time.time(),new_sumo_sim_s=0)
                immutable(root/f'WARM_REMOTE_ENDPOINT_ADMISSION_{host}_SECOND.json',value)
                print(json.dumps(dict(admitted=True,slot=config['endpoint']['slot'],unix=time.time())),flush=True)
                return
            time.sleep(30)
        raise ValueError('original GPU deadline reached before hot-add qualification')
    except BaseException as error:
        # A candidate worker failure cannot stop the already-running B search service.
        immutable(root/f'HOTADD_{host}_FAILED.json',dict(error=repr(error),unix=time.time(),
            existing_workers_unchanged=True,inference_retry=False,new_sumo_sim_s=0))
        raise
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path)
    run(parser.parse_args().config)
