"""MAIN03:19: serial new-attempt regrades; never rerun completed WM searches."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


GENERATION='SUMOPATH1'


def eligible(label):
    v=label['verification']
    return (v['status']=='failed' and v.get('actual_sim_s')==0
            and not v.get('branches') and 'FileNotFoundError' in v.get('reason',''))


def select(labels):
    complete={x['search_id'] for x in labels if x['verification']['status'] in ('accepted','rejected')}
    return sorted((x for x in labels if eligible(x) and x['search_id'] not in complete),
                  key=lambda x:(x['world_id'],x['search_id']))


def run(root):
    import fcntl
    from .warm_spool import immutable
    from .warm_protocol import install_accounting
    from .ls_collection import label
    from .ls_label_runtime import sumo_environment
    from .accel_storage import read,charged,ledger
    from .protocol import digest
    root=Path(root)
    reg=json.loads((root/'SAVED_REGRADE_REGISTRATION_0319.json').read_text())
    config=reg['config'];main=Path(config['run_root'])
    install_accounting(main,config['budget'])
    from . import accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    accel_storage._reserve.__globals__['CAP']=min(accel_storage._reserve.__globals__['CAP'],6000000-SUPERSEDED_CHARGE)
    os.environ.update(sumo_environment(config['sumo_home']))
    attempt=root/'saved_regrade_0319';attempt.mkdir(exist_ok=True)
    with (attempt/'OWNER.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import signal
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        immutable(attempt/'OWNER.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time()))
        charges={x['id']:x for x in ledger(main)}
        for item in reg['items']:
            key=item['search_id'];done=attempt/(key+'.json');claim=attempt/(key+'.claim.json')
            if done.exists():continue
            if claim.exists():raise ValueError('interrupted regrade claim: never replay '+key)
            target=root/'labels'/(key+'_'+GENERATION+'.json')
            if target.exists():raise ValueError('existing attempt label without completion; adopt manually '+key)
            immutable(claim,dict(unix=time.time(),key=key,original_label_sha256=item['original_label_sha256']))
            try:
                original=json.loads((root/'labels'/item['file']).read_text())
                assert digest(original)==item['original_label_sha256'] and eligible(original)
                assert not (root/'labels'/(key+'.json')).exists(),'old environment retry uses another recovery'
                assert charges['lslabel_'+key+'_'+GENERATION]['sim_s']==item['sim_s']
                envelope=json.loads(gzip.decompress(Path(item['pending_path']).read_bytes()))
                response=read(main/'responses'/(key+'.json'))
                assert digest(response)==item['response_sha256']
                assert envelope['record']['state_id']==key
                result=label(root,envelope,response,dict(config,label_generation=GENERATION))
                immutable(done,dict(unix=time.time(),status=result['status'],key=key,
                    receipt_sha256=result['receipt_sha256'],actual_sim_s=result.get('actual_sim_s'),
                    charged_total_sim_s=charged(main),original_charges_retained=True,new_wm_searches=0))
                print(json.dumps(json.loads(done.read_text())),flush=True)
            except BaseException as error:
                immutable(done,dict(unix=time.time(),status='failed',key=key,error=repr(error),
                    traceback=traceback.format_exc(),charges_retained=True))
                print(done.read_text(),flush=True)
        immutable(attempt/'COMPLETE.json',dict(unix=time.time(),count=len(reg['items']),new_wm_searches=0))


if __name__=='__main__':
    import sys
    run(sys.argv[1])
