"""MAIN20:33 focus-share compliance; metadata-only safe-boundary handover."""
import argparse
import json
from . import ls_label_ops as labels, ops
from .ls_collect_ops import COLLECTION, ROOT
from .ls_validity_ops import remote
from .protocol import REGISTRY, checked, sha
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_lsfocuscpu1'


def qualify():
    labels.BASE=labels.SOURCE
    labels.SOURCE=SOURCE
    labels.FILES=('ls_collection','test_ls_collection_cpu')
    labels.qualify()


def result():
    labels.SOURCE=SOURCE
    labels.result()


def register():
    cpu=read(ops.EVIDENCE/('LS_LABEL_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    target=ops.EVIDENCE/'LS_FOCUS_REGISTRATION.json'
    line=('\nS-COOPMWM MAIN20:33 FOCUS-ALLOCATION ENGINEERING CORRECTION: collection admissions '
        'repay the existing focus deficit, then allocate at least50% to demand2400/2700 '
        'and penetration.2/.4 while candidates are available. Unchanged new-world/time/group '
        'coverage order WITHIN each stratum; no state dropped. Uses only registered world '
        'metadata/counts, never search scores or exact grades. Adopt all72 completed '
        'episodes, original claims/queries/labels/charges and four GPU owners. Replace ONLY '
        'metadata supervisor at a verified idle grading boundary; additive immutable config. '
        'No new episode, numerical WM/search/horizon/K change, budget increase, mirror '
        'activation or PPO update. Existing conservative6M guard remains.\n')
    if not target.exists():
        write(target,dict(line=line,cpu=cpu,source_root=SOURCE,focus_fraction=.5,
            authority='MAIN2026-11-02 20:33 verified-plan-bank focus>=50%'))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(target)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(target),cpu_tests=cpu['tests'])))


def deploy():
    regpath=ops.EVIDENCE/'LS_FOCUS_REGISTRATION.json';reg=read(regpath);h=sha(regpath)
    v=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,time,signal,subprocess,hashlib
c=Path({COLLECTION!r});r=Path({ROOT!r});s=Path({SOURCE!r})
dest=c/'SUPERVISOR_DISPATCH_SUMOFOCUS1.json'
if dest.exists():print(dest.read_text());raise SystemExit
reg=json.loads({json.dumps(reg)!r})
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
sys=__import__('sys');sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.ls_label_runtime import sumo_environment
old=json.loads((c/'SUPERVISOR_DISPATCH_SUMODISK1.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert p.exists() and (p/'stat').read_text().split()[21]==old['start_ticks']
argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
assert 'tools.coopmwm.ls_collection' in argv and str(c) in argv
status=json.loads((c/'LS_COLLECT_STATUS.json').read_text())
if status['grading_inflight']:
 print(json.dumps(dict(wait=True,reason='adopt active grades; wait for idle boundary',grading_inflight=status['grading_inflight'])));raise SystemExit
os.kill(old['pid'],signal.SIGSTOP)
children=(p/'task'/str(old['pid'])/'children').read_text().split()
for child in children:
 q=Path('/proc',child);a=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 if 'multiprocessing.resource_tracker' not in a:
  os.kill(old['pid'],signal.SIGCONT);print(json.dumps(dict(wait=True,reason='live branch child; adopted')));raise SystemExit
status2=json.loads((c/'LS_COLLECT_STATUS.json').read_text())
if status2['grading_inflight']:
 os.kill(old['pid'],signal.SIGCONT);print(json.dumps(dict(wait=True,reason='grading boundary advanced; adopted')));raise SystemExit
unsettled=[q.stem for q in (c/'admissions').glob('*.json') if (r/'responses'/q.name).exists() and not (c/'labels'/(q.stem+'_SUMOENV1.json')).exists() and not (c/'label_failures'/q.name).exists()]
if unsettled or any((r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R')):
 os.kill(old['pid'],signal.SIGCONT);print(json.dumps(dict(wait=True,reason='unsettled answers or training boundary changed; adopted',unsettled=unsettled)));raise SystemExit
os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise ValueError('old supervisor did not exit')
immutable(c/'SUPERVISOR_RETIRED_SUMOFOCUS1.json',dict(owner=old,unix=time.time(),no_gpu_broker_or_physics_signals=True))
config=dict(source_root=str(s),receipt_suffix='_SUMOFOCUS1',focus_fraction=.5,amendment_sha256={h!r})
immutable(c/'LS_FOCUS_CONFIG.json',config)
with (r/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+chr(10))
env=sumo_environment('/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env.update(PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'SUPERVISOR_SUMOFOCUS1.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_collection',str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(c),source_root=str(s),unix=time.time(),amendment_sha256={h!r})
immutable(dest,d);print(json.dumps(d))''',timeout=60))
    if not v.get('wait'):
        target=ops.EVIDENCE/'LS_FOCUS_DISPATCH.json'
        if not target.exists():write(target,v)
    print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','result','register','deploy'))
    globals()[p.parse_args().action]()
