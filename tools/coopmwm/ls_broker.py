"""Collection-only durable delivery; no training, physics or global STOP writes."""
import json
import os
from pathlib import Path
import threading
import time
from .warm_spool import immutable
from .warm_broker import Remote
from .protocol import digest


def owner_matches(status,expected):
    return status['alive'] and status['ready'] and not status['failed'] and all(
        status['receipts'][name].get(k)==expected[k]
        for name in ('SPOOL_OWNER.json','SERVICE_OWNER.json')
        for k in ('pid','start_ticks','boot_id'))


def drain_request(root,slot,owner):
    """Append-only generation-specific drains supersede the historical slot alias."""
    from .ls_fast_worker import drain_applies,owner_key
    root=Path(root)
    key=digest(dict(slot=slot,owner=owner_key(owner)))
    paths=(root/'gpu_drains_v2'/(key+'.json'),root/'gpu_drains'/(digest(slot)+'.json'))
    for path in paths:
        if path.exists() and drain_applies(json.loads(path.read_text()),owner):
            parent='gpu_drained_v2' if path.parent.name=='gpu_drains_v2' else 'gpu_drained'
            return root/parent/path.name
    return None


def run(root,receipt_suffix=None):
    import fcntl,signal
    root=Path(root);cfg=json.loads((root/'LS_RECOVERY_CONFIG.json').read_text())
    if (root/'LS_VALIDITY_CONFIG.json').exists():
        cfg.update(json.loads((root/'LS_VALIDITY_CONFIG.json').read_text()))
    main=Path(cfg['run_root']);suffix=receipt_suffix or cfg['receipt_suffix']
    lock=(root/'BROKER.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    immutable(root/('BROKER_OWNER'+suffix+'.json'),dict(pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time()))
    for n in ('broker_assignments','broker_completed','broker_failures'):(root/n).mkdir(exist_ok=True)
    mutex=threading.Lock();busy=set();stop=threading.Event()
    def worker(e,expected):
        remote=Remote(root,e);slot=e['slot'];key=None
        try:
            status=remote.call(dict(action='status'))
            if not owner_matches(status,expected):raise ValueError('collection GPU gate/owner mismatch')
            remote.verify()
            while not stop.is_set():
                key=None
                # An immutable per-owner drain is checked only between searches.
                # Finish/deliver the current inference; never withdraw or replay it.
                target=drain_request(root,slot,expected)
                if target is not None:
                    target.parent.mkdir(exist_ok=True)
                    immutable(target,dict(slot=slot,owner=expected,unix=time.time(),inference_replayed=False))
                    return
                with mutex:
                    for p in sorted((root/'requests').glob('*.json'),key=lambda p:p.stat().st_mtime):
                        k=p.stem;assigned=root/'broker_assignments'/p.name
                        if k in busy or (main/'responses'/p.name).exists() or (root/'broker_failures'/p.name).exists():continue
                        prior=json.loads(assigned.read_text()) if assigned.exists() else None
                        if prior and prior['slot']!=slot:continue
                        record=json.loads(p.read_text())
                        immutable(assigned,dict(slot=slot,key=k,record_sha256=digest(record)))
                        busy.add(k);key=k;break
                if key is None:stop.wait(2);continue
                begin=time.time()
                remote.call(dict(action='submit',record=record))
                while not stop.is_set():
                    value=remote.call(dict(action='forecast',key=key))
                    if value['forecast'] is not None:break
                    if not owner_matches(value['status'],expected):raise ValueError('worker died; inference never replayed')
                    stop.wait(3)
                if stop.is_set():return
                payload=value['forecast']
                immutable(main/'responses'/(key+'.json'),payload['result'])
                immutable(root/'broker_completed'/(key+'.json'),dict(slot=slot,key=key,
                    inference_wall_s=payload['wall_s'],delivery_wall_s=time.time()-begin,unix=time.time()))
                with mutex:busy.remove(key)
        except Exception as error:
            import traceback
            value=dict(slot=slot,key=key,error=repr(error),traceback=traceback.format_exc(),
                unix=time.time(),inference_replayed=False)
            immutable(root/'broker_failures'/(digest(value)+'.json'),value)
            print('ALERT collection_slot_failed '+json.dumps(value),flush=True)
            # Assigned claims stay assigned. Other workers keep collecting.
    threads={}
    while not (root/'MAIN_STOP.json').exists():
        if time.time()>cfg['gpu_deadline']:break
        path=root/'LS_ENDPOINTS.json'
        if path.exists():
            v=json.loads(path.read_text())
            assert v['amendment_sha256']==cfg['amendment_sha256']
            for x in v['workers']:
                slot=x['endpoint']['slot']
                from .ls_fast_worker import owner_key
                if drain_request(root,slot,x['owner']) is not None:continue
                if slot not in threads or (not threads[slot][0].is_alive() and threads[slot][1]!=owner_key(x['owner'])):
                    t=threading.Thread(target=worker,args=(x['endpoint'],x['owner']),daemon=True)
                    threads[slot]=(t,owner_key(x['owner']));t.start()
        stop.wait(3)
    stop.set()


if __name__=='__main__':
    import sys
    run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
