"""Targeted Linux qualification and read-only preview of MAIN19:00 bank v2."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .storage import write,read
from .protocol import checked,sha,REGISTRY

SOURCE='/home/ubuntu/coopmwm20261103_bankboot1900_v5'


def qualify():
    files={n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8') for n in
        ('bank_statistics.py','bank_imitation.py','test_bank_boot_cpu.py','test_imitfix2_cpu.py','policy_synergy.py',
         'test_bank_imitation_cpu.py','test_exact_grade_cpu.py','test_bank_safety_cpu.py','test_replay_bank_v2v_cpu.py',
         'test_policy_synergy_cpu.py','test_policy_synergy_isolated_cpu.py')}
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,shutil,subprocess,os,time
r=Path({SOURCE!r});base=Path('/home/ubuntu/coopmwm20261103_v2vcpu8')
assert r.parent.is_dir();r.mkdir(exist_ok=True)
if not (r/'pytools').exists():shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
for n,text in json.loads({json.dumps(files)!r}).items():
 p=r/'pytools/tools/coopmwm'/n
 marker=r/(n+'.STAGED')
 if marker.exists():assert p.read_text()==text
 else:p.write_text(text);marker.touch()
p=r/'CPU_RESULT.json'
if not p.exists():
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':'+str(base/'runtime_src')+':/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 paths=[str(r/'pytools/tools/coopmwm'/n) for n in ('test_bank_boot_cpu.py','test_bank_amendment_cpu.py','test_policy_synergy_cpu.py','test_policy_synergy_isolated_cpu.py','test_bank_imitation_cpu.py','test_imitfix2_cpu.py','test_exact_grade_cpu.py','test_bank_safety_cpu.py','test_replay_bank_v2v_cpu.py')]
 start=time.time();x=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-m','pytest',*paths,'-q'],env=env,cwd=str(r),capture_output=True,text=True,timeout=90)
 v=dict(pass_=x.returncode==0,stdout=x.stdout,stderr=x.stderr,wall_s=time.time()-start,new_sim_s=0)
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())''',timeout=110))
    p=ops.EVIDENCE/'BANK_BOOT_PAIRED_WINDOW_LINUX_CPU_1910.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def preview():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,subprocess,os,time
r=Path({SOURCE!r});assert json.loads((r/'CPU_RESULT.json').read_text())['pass_'];out={{}}
env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/coopmwm20261103_v2vcpu8/runtime_src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for arm in ('B','R'):
 p=r/('BANK_BOOT_PREVIEW_'+arm+'.json');done=r/('PREVIEW_'+arm+'_RESULT.json')
 if not done.exists():
  command=['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-m','tools.coopmwm.bank_statistics','preview','--root','/home/ubuntu/coopmwm20261102_warm1','--arm',arm,'--out',str(p)]
  if arm=='B':command+=['--source','/home/ubuntu/coopmwm20261102_lscollect1']
  x=subprocess.run(command,env=env,cwd=str(r),capture_output=True,text=True,timeout=70)
  v=dict(pass_=x.returncode==0,stdout=x.stdout,stderr=x.stderr,new_sim_s=0,unix=time.time())
  with done.open('x') as f:json.dump(v,f)
 out[arm]=json.loads(done.read_text())
print(json.dumps(out))''',timeout=160))
    p=ops.EVIDENCE/'BANK_BOOT_PAIRED_WINDOW_PREVIEW_1910.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    cpu=read(ops.EVIDENCE/'BANK_BOOT_PAIRED_WINDOW_LINUX_CPU_1910.json');assert cpu['pass_']
    record=ops.EVIDENCE/'BANK_BOOT_PAIRED_WINDOW_REGISTRATION_1910.json'
    line=('\nS-COOPMWM MAIN19:00/19:10 engineering integration: bank-boot92d4deca + '
        'imit-fix2128516c9; boot_v2 remains default. Schema guard checks only '
        'boot_v2-required verification/replication/later-harm/safety keys; missing '
        'second-seed solo evidence is single_seed, not a t1727 schema exception. '
        'Legacy first top8 group tape can end later than the chosen-plan solo; '
        'synergy uses their exact shared recorded interval (same start, min(end)), '
        'records both original/used windows, never unmatched totals/missing seconds. '
        'No outcome-selected threshold or new SUMO. 96 targeted Linux tests PASS '
        'including exact_update/full+empty bank/Adam rollback/schema/synergy. '
        'V2V remains HELD; full source qualification and intent/fix gate still required.\n')
    if not record.exists():write(record,dict(line=line,cpu=cpu,new_sim_s=0,source=SOURCE,
        hashes={n:sha(ops.LOCAL/'tools/coopmwm'/n) for n in ('bank_statistics.py','bank_imitation.py','test_bank_boot_cpu.py','test_imitfix2_cpu.py')}))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(registered=True,sha256=h,new_sim_s=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','preview','register'))
    globals()[p.parse_args().action]()
