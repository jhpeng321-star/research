"""Explicit A environment repair; preserve and recharge all failed reservations."""
import argparse
import base64
import json
from . import ops
from .protocol import REGISTRY, checked, sha
from .storage import read, write
from .warm_a_ops import ROOT

SOURCE = '/home/ubuntu/coopmwm20261101_warmA5'


def prepare():
    code = checked(ops.LOCAL/'tools/coopmwm/warm_a.py').read_text(encoding='utf-8')
    code = code.replace('A_WARM3', 'A_WARM5').replace('A_WARM_CONTINUATION', 'A_WARM5_CONTINUATION')
    old = "jobs.append(dict(id=key, path=str(root/'jobs'/key), world=w,"
    new = """key = key + '_environment5'
                    from tools.coopmwm.accel_storage import reserve
                    reserve(root, [dict(id=key,sim_s=1200)], 'train')
                    jobs.append(dict(id=key, path=str(root/'jobs'/key), world=w,"""
    # Failed A4 claims are never adopted or overwritten. Only closed results reuse.
    code = code.replace("if adoptable(root, key, charges) == 'complete':", "if (root/'jobs'/key/'RESULT.json').exists():")
    assert old in code
    code = code.replace(old, new)
    compile(code, 'warm_a5.py', 'exec')
    staged = json.loads(ops.stage('cpuQ', SOURCE, {'warm_a.py': code}))
    value = json.loads(ops.ssh.remote('cpuQ', f'''import pathlib,json,shutil,sumo,os,time
r=pathlib.Path({ROOT!r});d=json.loads((r/'A_WARM4_DISPATCH.json').read_text())
assert not (pathlib.Path('/proc')/str(d['pid'])).exists()
assert 'SUMO binary not found on PATH' in (r/'A_WARM4.log').read_text()
assert not list(r.glob('A_UPDATE_*.json'))
home=str(pathlib.Path(sumo.__file__).parent);binary=shutil.which('sumo',path=home+'/bin:'+os.environ.get('PATH',''))
assert binary and pathlib.Path(binary).is_file()
jobs=[]
for p in sorted((r/'jobs').glob('u001_A_*')):
 if (p/'RESULT.json').exists():assert (p/'CLOSE.json').exists();continue
 jobs.append(dict(id=p.name,claimed=(p/'CLAIM.json').exists(),closed=(p/'CLOSE.json').exists()))
assert len(jobs)==12
print(json.dumps(dict(pass_=True,sim_s=0,sumo_home=home,sumo_binary=binary,failed_reservations=jobs,unix=time.time())))
''', timeout=25))
    amendment = dict(arm='A', stop_update=4, source=staged, qualification=value,
        preserved_stop_sha256=read(ops.EVIDENCE/'WARM_A4_AMENDMENT.json')['preserved_stop_sha256'],
        full_allocation_sim_s=4324040, cap_sim_s=6000000, max_new_sim_s=115200,
        extra_failed_reservation_charge_sim_s=14400,
        reason='A4 missing SUMO PATH before initialization; preserve all claims, new environment5 job IDs/precharges; reuse twelve completed episodes',
        B_R_held=True)
    write(ops.EVIDENCE/'WARM_A5_AMENDMENT.json', amendment)
    h = sha(ops.EVIDENCE/'WARM_A5_AMENDMENT.json')
    line = f'\nS-COOPMWM A-ONLY ENVIRONMENT REPAIR REGISTERED amendmentSHA{h}: A4 exited SUMO binary not found before initialization; all failed reservations retained. Set SUMO_HOME/PATH exactly as original accel_deploy; verified binary {value["sumo_binary"]}. Reuse twelve closed A episodes; twelve new environment5 IDs reserve another14400sim-s, never overwrite/retry old claims. A u1..4/check then pause; B/R held. Full allocation4324040/cap6000000/free1675960; maxnew115200sim-s, no sealed access.\n'
    with checked(REGISTRY).open('a', encoding='utf-8') as f:
        f.write(line)
    write(ops.EVIDENCE/'WARM_A5_REGISTRATION.json', dict(line=line, amendment_sha256=h))
    print(json.dumps(amendment))


def dispatch():
    a = read(ops.EVIDENCE/'WARM_A5_AMENDMENT.json')
    reg = read(ops.EVIDENCE/'WARM_A5_REGISTRATION.json')
    files = {n:base64.b64encode(checked(ops.EVIDENCE/n.replace('A_WARM5','WARM_A5')).read_bytes()).decode()
             for n in ('A_WARM5_AMENDMENT.json','A_WARM5_REGISTRATION.json')}
    value = json.loads(ops.ssh.remote('cpuQ', f'''import pathlib,json,os,sys,subprocess,time,base64,hashlib,shutil,sumo
r=pathlib.Path({ROOT!r});src=pathlib.Path({SOURCE!r});assert r.is_dir() and src.is_dir()
assert not (r/'A_WARM5_DISPATCH.json').exists()
for n,h in {a['source']['hashes']!r}.items():assert hashlib.sha256((src/n).read_bytes()).hexdigest()==h
for n,v in {files!r}.items():
 with (r/n).open('xb') as f:f.write(base64.b64decode(v))
with (r/'A_WARM5_AUTHORITY.md').open('x') as f:f.write({checked(REGISTRY).read_text(encoding='utf-8')!r})
d=json.loads((r/'TRAIN_DISPATCH.json').read_text());home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''),PYTHONPATH=d['source']+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',COOPMWM_COST_SOURCE=d['source']+'/FROZEN_COSTS.json')
assert shutil.which('sumo',path=env['PATH'])=={a['qualification']['sumo_binary']!r}
preflight="import importlib.util,pathlib;s=importlib.util.spec_from_file_location('warm_a',"+repr(str(src/'warm_a.py'))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.verify_a(pathlib.Path("+repr(str(r))+"))"
subprocess.run([sys.executable,'-B','-c',preflight],env=env,check=True,capture_output=True,text=True)
with (r/'A_WARM5.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-B','-u',str(src/'warm_a.py'),str(r)],cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=p.pid,start_ticks=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source=str(src),unix=time.time())
with (r/'A_WARM5_DISPATCH.json').open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''', timeout=25))
    write(ops.EVIDENCE/'WARM_A5_DISPATCH.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','dispatch'))
    {'prepare':prepare,'dispatch':dispatch}[p.parse_args().mode]()
