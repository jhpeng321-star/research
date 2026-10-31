"""Use an allocated t540 branch to gate cross-host post-MAPPO trunk parity."""
import json
from .dispatch import EVIDENCE,read,once,remote,HOSTS,canonical


def launch_late_gates():
    plan=read(EVIDENCE/'SHARDS.json')
    for host,shard in plan['shards'].items():
        root=shard['root'];config=HOSTS[host]
        code=f'''import pathlib,json,os,time,traceback
from tools.coopbandit.runner import init,branch,completed,job_id
from tools.coopbandit.storage import read,write,stop
r=pathlib.Path({root!r})
write(r/'OWNED_TRUNK540.json',dict(pid=os.getpid(),unix=time.time()))
try:
 init(str(r));job=next(j for j in read(r/'BRANCH_JOBS.json') if j[1]==540 and j[2]=='O1' and j[3] is None)
 if completed(r,job_id(job)) is None: branch(job)
 write(r/'TRUNK540_COMPLETE.json',dict(job_id=job_id(job),bitwise=True,unix=time.time()))
except BaseException:
 stop(r,traceback.format_exc());raise
'''
        marker=EVIDENCE/f'DISPATCH_TRUNK540_{host}.json'
        if marker.exists():
            print(canonical(read(marker)));continue
        output=remote(host,f'''import pathlib,subprocess,os,sys,json
r=pathlib.Path({root!r});assert (r/'BRANCH_CANARY_COMPLETE.json').exists();sys.path[:0]={config['deps']!r}
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=':'.join([str(r/'pytools')]+{config['deps']!r}),SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
log=(r/'SUPERVISOR_trunk540.log').open('x')
p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-c',{code!r}],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps(dict(host={host!r},root=str(r),pid=p.pid,phase='trunk540',extra_budget=0)))
''')
        result=json.loads(output);once(marker,result);print(canonical(result))


if __name__=='__main__': launch_late_gates()
