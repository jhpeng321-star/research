"""MAIN04:06 bank-only qualification, registration and durable cpuQ dispatch."""
import argparse
import base64
import json
import subprocess
import time
import zlib
from . import ops
from .ls_collect_ops import ROOT, COLLECTION
from .ls_validity_ops import remote
from .protocol import checked, REGISTRY, sha
from .storage import read, write

SOURCE = '/home/ubuntu/coopmwm20261103_bankcpu1'
BASE = '/home/ubuntu/coopmwm20261103_lspathcpu1'
CONTROL = COLLECTION+'/bank_builder_0406'
REVISION = '4f090b76'


def qualify():
    names = ('replay_bank', 'sumo_verify', 'accel_protocol',
             'test_replay_bank_confirm_cpu', 'test_replay_bank_lesson_cpu')
    files = {n: subprocess.check_output(['git', 'show',
        f'{REVISION}:tools/coopmwm/{n}.py'], cwd=ops.LOCAL).decode('utf-8') for n in names}
    for n in ('bank_builder', 'test_bank_builder_cpu'):
        files[n] = checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8')
    packed = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    print(remote('cpuQ', f'''from pathlib import Path
import json,os,shutil,hashlib,subprocess,time,base64,zlib
r=Path({SOURCE!r});base=Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 r.mkdir();shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):shutil.copy2(base/name,r/name)
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=['test_replay_bank_confirm_cpu','test_replay_bank_lesson_cpu','test_bank_builder_cpu','test_sumo_verify_cpu','test_exact_grade_cpu','test_awr_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('x') as f:json.dump(tests,f)
for n,text in files.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
 with (r/'LS_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_cpu_test',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))''', timeout=60))


def cpu_result():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json
r=Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
print(p.read_text() if p.exists() else json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-8000:])))'''))
    if not value.get('pending'):
        path = ops.EVIDENCE/'BANK_BUILDER_CPU_0406.json'
        if not path.exists():write(path, value)
        print(json.dumps({k: value[k] for k in ('pass_', 'tests', 'skips', 'sim_s', 'wall_s', 'root')}))
        if not value['pass_']:
            print(remote('cpuQ', f'from pathlib import Path;print(Path({SOURCE!r},"LS_CPU.log").read_text()[-20000:])'))
    else:print(json.dumps(value))


def plan():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,sys,time
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm import replay_bank as rb,sumo_verify as sv
from tools.coopmwm.bank_builder import quoted_jobs
from tools.coopmwm.accel_storage import charged
c=Path({COLLECTION!r});r=Path({ROOT!r})
cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text())
for n in ('LS_LABEL_CONFIG','LS_FOCUS_CONFIG'):
 p=c/(n+'.json')
 if p.exists():cfg.update(json.loads(p.read_text()))
cap=json.loads((r/'BANK_VERIFICATION_CAP_2232.json').read_text())
items=[]
for x in rb.bank_decisions(r,'B',sources=[str(c)]):
 if x['status']!='accepted' or not (x['grade_veh_s'] or 0)>0:continue
 snap=rb._decision_snapshot(r,'B',x,{{}})
 jobs=quoted_jobs(snap,x['chosen_plan'],sv.build_branches,rb.solo_candidate)
 items.append(dict(verification_id=x['verification_id'],world=x['episode']['world'],t0=x['t0'],members=x['chosen_plan']['members'],duration_s=x['chosen_plan']['duration_s'],jobs=jobs,upper_sim_s=sum(j['sim_s'] for j in jobs)))
print(json.dumps(dict(unix=time.time(),items=items,total_upper_sim_s=sum(x['upper_sim_s'] for x in items),charged_before=charged(r),operational_cap=6000000,budget=cap['budget'],sumo_home=cfg['sumo_home'],runtime=cfg['runtime'])))''', timeout=60))
    path = ops.EVIDENCE/'BANK_BUILDER_PLAN_0406.json'
    if not path.exists():write(path, value)
    else:assert read(path)['items'] == value['items'], 'adopt existing plan; do not replace'
    print(json.dumps({k: value[k] for k in ('unix', 'total_upper_sim_s', 'charged_before', 'operational_cap')}))
    print(json.dumps(dict(initial_positive_decisions=len(value['items']))))


def register():
    cpu = read(ops.EVIDENCE/'BANK_BUILDER_CPU_0406.json')
    quote = read(ops.EVIDENCE/'BANK_BUILDER_PLAN_0406.json')
    assert cpu['pass_'] and cpu['skips'] == cpu['sim_s'] == 0
    assert quote['charged_before']+quote['total_upper_sim_s'] <= quote['operational_cap']
    path = ops.EVIDENCE/'BANK_BUILDER_REGISTRATION_0406.json'
    if not path.exists():
        line = ('\nS-COOPMWM MAIN2026-11-03 04:06 OFFLINE BANK BUILDER: delegated '
            'replay_bank/confirmation/synergy from4f090b76 (f023f10c,b58ca5b2,62b80987), '
            'CPU-qualified isolated source; NO full B-REPLAY activation or GPU/collection '
            'handover. Existing B/u3 TRAIN LS labels only; confirm second CRN seed, '
            'extended min(end,t0+D+600), no safety regression (collision/teleport/hard '
            'brake/trace TTC<1.5); admitted multi-member plans get every solo. '
            'All replication/extended/conditional-solo pairs precharged atomically in '
            'bank_verification_ext BEFORE physics; unused conditional charges retained, '
            'no refund or job replay. Shared ledger lock and fresh operational6M hardcap '
            'retained; previously registered phase cap2.6M. New accepted labels processed '
            'continuously by the SAME owned remote builder; budget shortage waits before '
            'claim, no partial group. Existing owner/GPU/PPO jobs untouched. '
            f'Initial{len(quote["items"])} decisions quoted upper{quote["total_upper_sim_s"]} '
            'sim-s; exact grades and synergy published BANK_QUALITY_B.json; no sealed access.\n')
        write(path, dict(line=line, cpu=cpu, quote=quote, revision=REVISION,
            sources=[COLLECTION], run_root=ROOT, source_root=SOURCE,
            initial_verification_ids=[x['verification_id'] for x in quote['items']],
            operational_cap=quote['operational_cap'], budget=quote['budget'],
            sumo_home=quote['sumo_home'], runtime=quote['runtime'],
            source_hashes=cpu['source_hashes'], branch_timeout_s=3600, poll_s=30,
            deadline_unix=1793894340, authority='MAIN2026-11-03 04:06'))
        with checked(REGISTRY).open('a', encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(path)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(path), initial_upper_sim_s=quote['total_upper_sim_s'])))


def launch():
    config = read(ops.EVIDENCE/'BANK_BUILDER_REGISTRATION_0406.json')
    h = sha(ops.EVIDENCE/'BANK_BUILDER_REGISTRATION_0406.json')
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,sys,os,time,subprocess,hashlib
s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir()
if c.exists() and (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
cfg=json.loads({json.dumps(config)!r});cfg['amendment_sha256']={h!r}
for n,h in cfg['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
c.mkdir(exist_ok=True)
with (c/'BINDING.json').open('x') as f:json.dump(cfg,f)
with (Path(cfg['run_root'])/'AUTHORITY.md').open('a') as f:f.write(cfg['line']+'amendmentSHA'+cfg['amendment_sha256']+chr(10))
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+cfg['runtime']+'/runtime/src:'+cfg['runtime']+'/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'BUILDER.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.bank_builder',str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256=cfg['amendment_sha256'])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''', timeout=60))
    path = ops.EVIDENCE/'BANK_BUILDER_DISPATCH_0406.json'
    if not path.exists():write(path, value)
    print(json.dumps(value))


def status():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,time
c=Path({CONTROL!r});r=Path({ROOT!r});d=json.loads((c/'DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
st=(p/'stat').read_text().split() if p.exists() else None
assert st is None or st[21]==d['start_ticks']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
q=r/'replay_bank/B/BANK_QUALITY.json';first=c/'FIRST_BATCH_COMPLETE.json'
attempts=[json.loads(p.read_text()) for p in (c/'attempts').glob('*.json') if not p.name.endswith('.claim.json')]
print(json.dumps(dict(unix=time.time(),owner=d,state=st[2] if st else 'ABSENT',quality=json.loads(q.read_text()) if q.exists() else None,first_batch_complete=json.loads(first.read_text()) if first.exists() else None,attempts=attempts,log_tail=(c/'BUILDER.log').read_text()[-6000:])))''', timeout=60))
    if value['quality'] is not None:
        target = ops.EVIDENCE/'BANK_QUALITY_B.json'
        tmp = target.with_name(f'.BANK_QUALITY_B.{time.time_ns()}.publishing')
        with tmp.open('w', encoding='utf-8') as f:json.dump(value['quality'], f)
        __import__('os').replace(tmp, target)
    write(ops.EVIDENCE/f'BANK_BUILDER_STATUS_{time.time_ns()}.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=('qualify','cpu_result','plan','register','launch','status'))
    globals()[p.parse_args().action]()
