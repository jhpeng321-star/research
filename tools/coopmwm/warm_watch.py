"""One bounded, receipt-driven continuation through the early warm viability pause.

Never retries a physics/inference attempt. Adopts existing qualifiers/services,
releases only after all gates, publishes each check, and keeps durable backups.
"""
import argparse
from datetime import datetime,timezone
import json
import subprocess
import sys
import time
from . import ops
from .storage import read,write
from .protocol import checked,sha
from .warm_deploy import ROOT,dispatch as dispatch_arm
from . import warm_release_ops as release_ops
from .warm_publish import publish
from .warm_storage import backup

BRIEF=checked('E:/agent-work/_sol/coopmwm_prompt.md')
OWN_PATTERNS=('WARM_RELEASE_STATE_*.json','WARM_WORKER_STATE_*.json',
    'WARM_SERVICE_STAGE_gpuR.json','WARM_SERVICE_ADMISSION_gpuR.json','WARM_GPU_*_READY.json','WARM_GPU_*_SSH.json',
    'WARM_RELEASE_AMENDMENT.json','WARM_RELEASE_REGISTRATION.json','WARM_MEMORY_QUALIFIED.json',
    'WARM_TRAINING_STARTED.json','WARM_B_DISPATCH2.json','WARM_R_DISPATCH2.json','WARM_BACKUP_*.json',
    'WARM_WATCH_*.json','warm_live/*.json')


def state():
    return json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,shutil,sys
r=pathlib.Path({ROOT!r});assert r.is_dir();owners={{}}
source='/home/ubuntu/coopmwm20261102_exactcpu5'
for name in ('MAIN0548_RUNTIME_BINDING_V2.json','MAIN0548_RUNTIME_BINDING.json','MAIN0508_REPAIR_BINDING_V2.json'):
 if (r/name).exists():source=json.loads((r/name).read_text())['source_root'];break
sys.path.insert(0,source+'/pytools')
from tools.coopmwm.warm_health import assert_running,HealthPause
for arm in ('A','B','R'):
 names=[f'QUOTA_{{arm}}_DISPATCH.json']+[f'WARM_{{arm}}_DISPATCH{{suffix}}.json' for suffix in ('_V6','_V4','_V3','2','')]
 p=next((r/name for name in names if (r/name).exists()),None)
 if p is None:continue
 d=json.loads(p.read_text());q=pathlib.Path('/proc')/str(d['pid']);s=(q/'stat').read_text().split() if (q/'stat').exists() else None
 assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'owner boot changed'
 assert not s or s[21]==d['start_ticks'],'owner PID reused'
 owners[arm]=dict(owner=d,state=s[2] if s else 'exited',dispatch_receipt=p.name)
stop=None
try:assert_running(r,'B')
except (ValueError,HealthPause) as error:stop=dict(error=str(error),source='current registered warm_health guard')
pauses=sorted(p.name for p in r.glob('PAUSE_*.json'))
failures={{p.name:json.loads(p.read_text()) for pattern in ('WARM_*_FAILED_Q1.json','WARM_*_FAILED_V5.json') for p in r.glob(pattern)}}
if failures and stop is None:stop=dict(error='current training supervisor failure',receipts=failures)
charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
print(json.dumps(dict(unix=time.time(),owners=owners,stop=stop,
   updates={{a:sorted(int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json')) for a in ('A','B','R')}},
  paused={{a:(r/f'WARM_{{a}}_PAUSED.json').exists() or (a in ('B','R') and bool(pauses)) for a in ('A','B','R')}},
  pause_receipts=pauses,failures=failures,
  historical_transport_stop=json.loads((r/'STOP.json').read_text()) if (r/'STOP.json').exists() else None,
  charged_sim_s=811640+sum(c['sim_s'] for c in charges),disk_free=shutil.disk_usage(r).free,
  ram_free=shutil.disk_usage('/dev/shm').free)))''',timeout=30))


def register_state_repair():
    from .protocol import REGISTRY
    target=ops.EVIDENCE/'WARM_STATE_CURRENT_OWNERS_REPAIR.json'
    assert not target.exists(),'adopt existing observation repair receipt'
    live=state()
    value=dict(unix=time.time(),state=live,tests=11,new_sumo_sim_s=0,
        source_sha256=sha(ops.LOCAL/'tools/coopmwm/warm_watch.py'),
        repair='numeric update indices; current dispatch precedence with boot/start identity; active registered health guard instead of historical released transport STOP; PAUSE and current supervisor failures exposed',
        training_or_inference_changes=False,original_deadline_retained=True)
    write(target,value)
    line='\nS-COOPMWM MAIN0628 GUARD STATE COMPATIBILITY REPAIR REGISTERED amendmentSHA'+sha(target)+': read-only state now adopts current A/V3,B-R/QUOTA owners, boot/start identity; released historical SSH STOP is separate from active stop, PAUSE and current supervisor faults visible; CPU11PASS including actual generated payload, real cpuQ status PASS; zero SUMO/inference/PID control, existing owners/observers and original deadline retained.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    print(json.dumps(value))


def commit():
    paths=['docs/paper_results/COOPMWM.md','HANDOFF_COOPMWM.md']
    paths += [str(p.relative_to(ops.LOCAL)) for pattern in OWN_PATTERNS for p in ops.EVIDENCE.glob(pattern)]
    paths=sorted(set(paths))
    def git(*args):return subprocess.check_output(['git',*args],cwd=ops.LOCAL,text=True,encoding='utf-8')
    assert not git('diff','--cached','--name-only').strip(),'another writer has staged work; preserve it'
    print(git('status','--short','--untracked-files=no'),flush=True)
    print(git('diff','--stat','--',*paths),flush=True)
    print(git('log','--oneline','-10'),flush=True)
    subprocess.run(['git','add','--',*paths],cwd=ops.LOCAL,check=True)
    if git('diff','--cached','--name-only').strip():
        subprocess.run(['git','commit','-m','coopmwm: publish warm qualification and live checks'],cwd=ops.LOCAL,check=True)


def backups():
    local=ops.EVIDENCE/'warm_live'
    for path in sorted(local.glob('*.json')):
        parts=path.stem.split('_')
        if len(parts)!=3 or parts[0] not in ('A','B','R') or parts[1] not in ('UPDATE','CHECK','DEV'):continue
        arm,kind,u=parts[0],parts[1],int(parts[2])
        if arm=='A' and u<7:continue  # Those original disk paths stay preserved.
        phase={'UPDATE':'train','CHECK':'check','DEV':'dev'}[kind]
        suffix='' if phase=='train' else phase.upper()+'_'
        if not (ops.EVIDENCE/f'WARM_BACKUP_{arm}_{suffix}{u:03}.json').exists():backup(arm,u,phase)


def announce_restart():
    target=ops.EVIDENCE/'WARM_TRAINING_STARTED.json'
    if target.exists():return
    value=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        B=read(ops.EVIDENCE/'WARM_B_DISPATCH2.json'),R=read(ops.EVIDENCE/'WARM_R_DISPATCH2.json'),
        qualification=read(ops.EVIDENCE/'WARM_RELEASE_REGISTRATION.json'),
        runtime='CPU8 / MAPPO u0 / Nexec10 / full dimension-scaled medium MPPI',
        R_contract='same policy centre, zero candidate WM scores; GPU helper waits until B-u4 pause',
        pause='B/Ru4 early viability review; A continues fixed48; final panel remains unopened')
    write(target,value)
    doc=checked(ops.LOCAL/'docs/paper_results/COOPMWM.md')
    text=('\n## TRAINING STARTED — warm B/R restart '+value['timestamp_utc']+'\n'
        f'- B ownedPID{value["B"]["pid"]}, R ownedPID{value["R"]["pid"]}; MAPPOu0, immutable CPU8.\n'
        '- WARM MEMORY QUALIFIED is recorded above. Nexec10; full medium search and source-bound both-GPU gates retained.\n'
        '- R uses the identical current-policy centre extractor, zero candidate scoring; baseline-only helper waits for B-u4 pause.\n'
        '- ETA remains acceptance-dependent; publish measured executed-plan throughput after the first completed update.\n'
        '- Registered first viability pauseB/Ru4, A continues48; no final-panel outcome access.\n')
    with doc.open('a',encoding='utf-8') as f:f.write(text)
    handoff=checked(ops.LOCAL/'HANDOFF_COOPMWM.md')
    with handoff.open('a',encoding='utf-8') as f:f.write('\nLATEST AUTOMATED WARM RESTART: '+value['timestamp_utc']+
        f'; B{value["B"]["pid"]}/R{value["R"]["pid"]}, CPU8u0; WARM_TRAINING_STARTED.json pins owners; '
        'use warm_watch state/status receipts, never duplicate dispatch.\n')


def run():
    import msvcrt
    claim=(ops.EVIDENCE/'WARM_WATCH.claim').open('a+b');claim.seek(0)
    if not claim.read(1):claim.write(b'1');claim.flush()
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    binding=read(ops.EVIDENCE/'WARM_WATCH_BINDING.json')
    assert sha(BRIEF)==binding['brief_sha256']
    try:
        while time.time()<binding['deadline_unix']:
            assert sha(BRIEF)==binding['brief_sha256'],'owner brief changed; fresh reading required before dispatch'
            for name,h in binding['helper_hashes'].items():assert sha(ops.LOCAL/name)==h,'observer helper changed; fresh recovery required'
            live=state();print(json.dumps(live),flush=True)
            publish();backups();commit()
            if live['stop']:raise ValueError('durable warm STOP: '+str(live['stop']))
            if all(live['paused'].values()):
                write(ops.EVIDENCE/'WARM_WATCH_COMPLETE.json',dict(live,reason='A48 and early B/Ru4 receipts published; owner review; no final test'))
                commit();return
            if not (ops.EVIDENCE/'WARM_SERVICE_ADMISSION_gpuR.json').exists():
                data=release_ops.collect();r=data['gpuR']
                if 'FAILED.json' in data['gpuA']['receipts']:raise ValueError('GPU A group6 qualifier failed; no automatic retry')
                if 'FAILED.json' in r['receipts']:raise ValueError('warm qualifier failed; no automatic retry: '+str(r['receipts']['FAILED.json']))
                if r['state'] in ('exited','Z'):
                    assert 'SEARCH_GROUP4.json' in r['receipts']
                    if not (ops.EVIDENCE/'WARM_SERVICE_STAGE_gpuR.json').exists():release_ops.stage_worker('gpuR')
                    release_ops.admit('gpuR');commit()
            if (ops.EVIDENCE/'WARM_SERVICE_ADMISSION_gpuR.json').exists() and not (ops.EVIDENCE/'WARM_RELEASE_REGISTRATION.json').exists():
                workers={h:release_ops.worker_status(h) for h in ('gpuA','gpuR')}
                data=release_ops.collect()
                if 'FAILED.json' in data['gpuA']['receipts']:raise ValueError('GPU A group6 qualifier failed; no automatic retry')
                if all(len(v['receipts'])==7 for v in workers.values()) and 'SEARCH_GROUP6.json' in data['gpuA']['receipts'] and data['gpuA']['state'] in ('exited','Z'):
                    release_ops.release();commit()
            if (ops.EVIDENCE/'WARM_RELEASE_REGISTRATION.json').exists():
                for arm in ('B','R'):
                    if not (ops.EVIDENCE/f'WARM_{arm}_DISPATCH2.json').exists():dispatch_arm(arm)
                announce_restart();commit()
            time.sleep(60)
        raise ValueError('registered observer deadline; no automatic extension')
    except BaseException as error:
        write(ops.EVIDENCE/'WARM_WATCH_FAILED.json',dict(unix=time.time(),error=repr(error),
            disposition='no retries or process control; preserve all live PIDs/receipts and resume from disk'))
        with checked(ops.LOCAL/'docs/paper_results/COOPMWM.md').open('a',encoding='utf-8') as f:
            f.write('\n## Warm observer stopped\n- '+repr(error)+'; no automatic retry; inspect immutable WARM_WATCH_FAILED.json and live owners.\n')
        commit();raise


def dispatch():
    dest=ops.EVIDENCE/'WARM_WATCH_DISPATCH.json';assert not dest.exists()
    b=read(ops.EVIDENCE/'WARM_A48_BINDING.json')
    binding=dict(brief_sha256=sha(BRIEF),deadline_unix=b['gpu_deadline'],new_sumo_sim_s=0,
        helper_hashes={f'tools/coopmwm/{name}.py':sha(ops.LOCAL/f'tools/coopmwm/{name}.py') for name in
            ('warm_watch','warm_release_ops','warm_publish','warm_report','warm_deploy','warm_storage',
             'resume_gate','warm_transport','warm_bridge')},
        contract='receipt-only observer; qualify/register before B/R dispatch once; publish/backup each check; no final test/retry/PID control')
    write(ops.EVIDENCE/'WARM_WATCH_BINDING.json',binding)
    with (ops.EVIDENCE/'WARM_WATCH.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-m','tools.coopmwm.warm_watch','run'],cwd=ops.LOCAL,stdout=log,stderr=subprocess.STDOUT)
    value=dict(pid=p.pid,unix=time.time(),binding=binding)
    write(dest,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('dispatch','run','state','register_state_repair'));a=p.parse_args()
    if a.mode=='state':print(json.dumps(state()))
    else:globals()[a.mode]()
