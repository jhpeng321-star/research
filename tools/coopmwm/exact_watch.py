"""Receipt-only observer and one-use gated launcher; remote training survives its exit."""
import argparse
from datetime import datetime,timezone
import json
import subprocess
import sys
import time
from . import ops,exact_ops
from .storage import read,write

DISPATCHES={'grader':'EXACT_GRADER_DISPATCH.json','broker':'WARM_REMOTE_BROKER_DISPATCH.json',
            'B':'WARM_B_DISPATCH_V4.json','R':'WARM_R_DISPATCH_V4.json'}
TAG='EXACT_OBSERVER_V2'


def adopt(cpu):
    """Reconcile launch receipts after SSH loss BEFORE considering any dispatch."""
    for name,entry in cpu['owners'].items():
        if name not in DISPATCHES.values():continue
        path=ops.EVIDENCE/name
        if path.exists():
            assert read(path)==entry['owner'],'remote owner differs from local dispatch'
        else:write(path,entry['owner'])


def launch_ready(gpus,cpu):
    adopt(cpu)
    if cpu['stops']:raise ValueError('registered remote STOP/pause; no automatic restart')
    if not all(v['alive'] and v['ready'] and not v['failed'] and
               len([k for k in v['receipts'] if k.startswith('STARTUP_')])==8 for v in gpus.values()):
        return False
    if not (ops.EVIDENCE/'EXACT_RUNTIME_REGISTRATION.json').exists():exact_ops.register()
    for component,name in DISPATCHES.items():
        if not (ops.EVIDENCE/name).exists():exact_ops.dispatch(component)
    return True


def announce():
    path=ops.EVIDENCE/'EXACT_TRAINING_STARTED.json'
    if path.exists():return
    value=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dispatches={k:read(ops.EVIDENCE/v) for k,v in DISPATCHES.items()},
        source_root=exact_ops.CPU,qualification='WARM MEMORY QUALIFIED retained; concurrent source8EXACT0 gates per host',
        endpoints=dict(A=48,B=48,R=24),N_exec=10,grade_source='exact marginal SUMO only',
        ETA='first executed-plan throughput/acceptance measurement pending; no fabricated completion ETA')
    write(path,value)
    doc=ops.LOCAL/'docs/paper_results/COOPMWM.md'
    with doc.open('a',encoding='utf-8') as f:
        f.write('\n## TRAINING STARTED B/R — '+value['timestamp_utc']+'\n'
            '- MAPPOu0, immutable exactcpu5; B/R and exact grader/broker detached on cpuQ.\n'
            '- WARM MEMORY QUALIFIED retained; both concurrent GPU workers pass8source-bound EXACT0 gates.\n'
            '- K4/H300 screen, K8/Hfinal, Nexec10; no look-ahead reduction. Exact grades only for trust/imitation/library, asynchronous.\n'
            '- R centre requests retain lower priority until B finishes; no candidate scoring/library for R.\n'
            '- First-update executed-plan rate/ETA pending. Owners: `coopmwm/EXACT_TRAINING_STARTED.json`.\n')


def commit():
    from .warm_v3_watch import commit as checked_commit
    patterns=('EXACT_*.json','CONCURRENT_STATUS_*.json','warm_live/*.json')
    paths=[str(p.relative_to(ops.LOCAL)) for pattern in patterns for p in ops.EVIDENCE.glob(pattern)]
    paths.append('docs/paper_results/COOPMWM.md')
    try:
        checked_commit(paths)
    except ValueError as error:
        if str(error)!='other staged work; preserve it':raise
        # Observation must survive an operator's independent commit. Leave the
        # index untouched and publish all pending immutable receipts next poll.
        write(ops.EVIDENCE/f'EXACT_COMMIT_DEFERRED_{time.time_ns()}.json',dict(
            unix=time.time(),reason=str(error),index_untouched=True,physics_retry=False))


def run():
    import msvcrt
    from .warm_publish import publish
    from .warm_transport import retry
    lock=(ops.EVIDENCE/(TAG+'.claim')).open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    binding=read(ops.EVIDENCE/(TAG+'_BINDING.json'))
    previous={}
    def record(event):write(ops.EVIDENCE/f'EXACT_OBSERVER_TRANSPORT_{time.time_ns()}.json',event)
    def verify():
        g,c=exact_ops.status()
        for host,v in previous.get('gpu',{}).items():
            assert g[host]['receipts']['SPOOL_OWNER.json']==v['receipts']['SPOOL_OWNER.json']
            for name,r in v['receipts'].items():assert g[host]['receipts'].get(name)==r,'immutable GPU receipt changed'
        for name,v in previous.get('cpu',{}).get('owners',{}).items():assert c['owners'][name]['owner']==v['owner']
    try:
        while time.time()<binding['deadline_unix']:
            gpus,cpu=retry(exact_ops.status,verify,record)
            previous.update(gpu=gpus,cpu=cpu)
            if any(v['failed'] for v in gpus.values()):raise ValueError('source-bound GPU failure; preserve/no inference replay')
            if TAG in ('EXACT_OBSERVER_V6','EXACT_OBSERVER_V7'):
                adopt(cpu);launched=False  # existing repaired remote owners only
            else:launched=launch_ready(gpus,cpu)
            if launched:announce()
            retry(publish,verify,record);commit()
            if cpu['stops']:raise ValueError('health/engineering pause; diagnose before registered release')
            time.sleep(60)
        raise ValueError('registered GPU observation deadline reached')
    except BaseException as error:
        path=ops.EVIDENCE/f'EXACT_OBSERVER_EXIT_{int(time.time())}.json'
        write(path,dict(unix=time.time(),error=repr(error),remote_training_survives=True,automatic_physics_retry=False))
        commit();raise
    finally:lock.close()


def dispatch():
    target=ops.EVIDENCE/(TAG+'_DISPATCH.json');assert not target.exists()
    if TAG=='EXACT_OBSERVER_V7':
        from .protocol import checked,REGISTRY,sha
        receipt=ops.EVIDENCE/'EXACT_OBSERVER_TIMING_RECOVERY.json';assert not receipt.exists()
        old=read(ops.EVIDENCE/'EXACT_OBSERVER_EXIT_1793571681.json');assert 'TIMING' in old['error']
        value=dict(previous_exit=old,previous_observer_pid=41116,previous_pid_verified_absent=True,
            repair='numeric3digit update receipt glob, metadata never parsed as update; adopt QUOTA owners, publish repaired exact twins and Q1 faults, suppress invalid adopted-u1 ETA',
            tests=6,new_sumo_sim_s=0,dispatches_training=False,original_gpu_deadline_retained=True,
            status_sha256=sha(ops.LOCAL/'tools/coopmwm/exact_ops.py'),publisher_sha256=sha(ops.LOCAL/'tools/coopmwm/warm_publish.py'))
        write(receipt,value)
        line='\nS-COOPMWM OBSERVER UPDATE-METADATA REPAIR REGISTERED amendmentSHA'+sha(receipt)+': '+json.dumps(value,sort_keys=True)+'\n'
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    if TAG=='EXACT_OBSERVER_V6':
        from .protocol import checked,REGISTRY,sha
        receipt=ops.EVIDENCE/'EXACT_OBSERVER_CAPACITY_RECOVERY.json';assert not receipt.exists()
        write(receipt,dict(previous_observer_pid=41484,previous_observer_verified_absent=True,
            purpose='observe repaired V6 B/R and V2 broker/grader; preserve historical owner receipts and SHA-released engineering STOP',
            status_sha256=sha(ops.LOCAL/'tools/coopmwm/exact_ops.py'),publisher_sha256=sha(ops.LOCAL/'tools/coopmwm/warm_publish.py'),
            tests=6,new_sumo_sim_s=0,dispatches_training=False))
        line='\nS-COOPMWM MAIN0508 OBSERVATION RECOVERY REGISTERED amendmentSHA'+sha(receipt)+': V5local41484 verifiedabsent; V6 observe-only existing repaired remote owners, SHA-gated historicalSTOP release, newSTOP_V2/FAILED_V5 public; no launch/retry, originalGPUdeadline preserved; CPU6PASS/zeroSUMO.\n'
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    if TAG=='EXACT_OBSERVER_V3':
        from .protocol import checked,REGISTRY,sha
        receipt=ops.EVIDENCE/'EXACT_OBSERVER_STAGING_RECOVERY.json'
        assert not receipt.exists()
        value=dict(previous_exit=read(ops.EVIDENCE/'EXACT_OBSERVER_EXIT_1793564486.json'),
            observer_source_sha256=sha(ops.LOCAL/'tools/coopmwm/exact_watch.py'),
            purpose='restore receipt observation after operator staging conflict; defer only this commit conflict',
            tests=2,new_sumo_sim_s=0,automatic_physics_retry=False)
        write(receipt,value)
        line='\nS-COOPMWM OBSERVER STAGING RECOVERY REGISTERED amendmentSHA'+sha(receipt)+': local observer43408 exited on other-staged-work guard; verified absent; V3 restores observation/adopts existing remote owners, does not repeat launches, defers index-conflict commit until next poll; all physics/inference and original GPU deadline unchanged; CPU2PASS/zeroSUMO.\n'
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    if TAG=='EXACT_OBSERVER_V4':
        from .protocol import checked,REGISTRY,sha
        receipt=ops.EVIDENCE/'EXACT_OBSERVER_PUBLICATION_RECOVERY.json';assert not receipt.exists()
        write(receipt,dict(previous_exit=read(ops.EVIDENCE/'EXACT_OBSERVER_EXIT_1793566613.json'),
            publisher_sha256=sha(ops.LOCAL/'tools/coopmwm/warm_publish.py'),tests=4,new_sumo_sim_s=0,
            repair='strict TRAIN receipt whitelist; cache existence is not publication; adopt identical cached values and append missing markers; metadata excluded; no overwrite/replay'))
        line='\nS-COOPMWM TRAIN PUBLICATION SCHEMA RECOVERY REGISTERED amendmentSHA'+sha(receipt)+': observer34356 exited KeyError(viability) on reporter metadata matched by broad glob; verified absent. CPU4PASS, strict TRAIN/health receipt whitelist, retain metadata/cached files; identical already-fetched unpublished A receipts adopted and missing paper markers restored, no overwrite/deletion/physics/inference replay. V4 observes existing owners only, originaldeadline retained.\n'
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    if TAG=='EXACT_OBSERVER_V5':
        from .protocol import checked,REGISTRY,sha
        receipt=ops.EVIDENCE/'EXACT_OBSERVER_PAYLOAD_RECOVERY.json';assert not receipt.exists()
        write(receipt,dict(previous_exit=read(ops.EVIDENCE/'EXACT_OBSERVER_EXIT_1793567134.json'),
            publisher_sha256=sha(ops.LOCAL/'tools/coopmwm/warm_publish.py'),tests=4,new_sumo_sim_s=0,
            repair='correct remote payload indentation; CPU tests compile actual transport payload; real one-shot publication recovered4receipts without overwrite/replay'))
        line='\nS-COOPMWM PUBLICATION PAYLOAD REPAIR REGISTERED amendmentSHA'+sha(receipt)+': observer39060 exited before remote receipt read on IndentationError; verified absent; fixed generated Python indentation and added actual-payload compilation to CPU4PASS. Real one-shot fetch/publication succeeded, recovered A48/check/dev/pause from existing immutable data. V5 restores observation only, original remote owners/deadline preserved, zero new physics/inference.\n'
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    binding=dict(deadline_unix=read(ops.EVIDENCE/'WARM_A48_BINDING.json')['gpu_deadline'],
        contract='observe/publish only; repaired owners already dispatched; never launch physics/inference' if TAG in ('EXACT_OBSERVER_V6','EXACT_OBSERVER_V7') else 'observe/publish; adopt existing owners; once8EXACT0 gates per host, register and launch grader/broker/B/R once; no physics retry')
    write(ops.EVIDENCE/(TAG+'_BINDING.json'),binding)
    with (ops.EVIDENCE/(TAG+'.log')).open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.exact_watch','run','--tag',TAG],cwd=ops.LOCAL,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    value=dict(pid=p.pid,unix=time.time(),binding=binding);write(target,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','dispatch'))
    p.add_argument('--tag',choices=('EXACT_OBSERVER_V2','EXACT_OBSERVER_V3','EXACT_OBSERVER_V4','EXACT_OBSERVER_V5','EXACT_OBSERVER_V6','EXACT_OBSERVER_V7'),default=TAG)
    args=p.parse_args();TAG=args.tag;globals()[args.mode]()
