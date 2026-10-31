"""Own-root, CPU-only reporting launch/status with short sequential SSH calls."""
import argparse
import base64
import json
import subprocess

HOSTS={"gpuR":("/root/miniconda3/bin/python","/root/coopbehav20261031_analysis"),
       "cpuQ":("/home/ubuntu/miniconda3/bin/python","/home/ubuntu/coopbehav20261031_analysis")}


def execute(host,code,timeout=30):
    python,_=HOSTS[host]
    encoded=base64.b64encode(code.encode()).decode()
    command=f"echo {encoded} | base64 -d | {python}"
    result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",host,command],
                          capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def launch(host,views,label):
    from .io import safe
    safe(views); safe(label)
    if not label.replace("_","").replace("-","").isalnum(): raise ValueError("Invalid run label")
    python,root=HOSTS[host]
    code=f'''import json,os,subprocess
from pathlib import Path
root=Path({root!r}); assert root.is_dir()
receipt=root/{(label+"_PID.json")!r}
assert not receipt.exists(), "Existing launch retained; use status"
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
command=[{python!r},"-m","tools.coopbehav.report","--views",{views!r},"--out",str(root/{label!r}),"--label",{label!r}]
log=(root/{(label+".log")!r}).open("ab")
p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
record=dict(pid=p.pid,start_ticks=Path(f"/proc/{{p.pid}}/stat").read_text().split()[21],command=command,root=str(root))
receipt.write_text(json.dumps(record,indent=2)); print(json.dumps(record))
'''
    return execute(host,code)


def status(host,label):
    _,root=HOSTS[host]
    if not label.replace("_","").replace("-","").isalnum(): raise ValueError("Invalid run label")
    code=f'''import json
from pathlib import Path
root=Path({root!r}); receipt=root/{(label+"_PID.json")!r}
if not receipt.exists(): print(json.dumps(dict(status="not_launched")))
else:
 r=json.loads(receipt.read_text()); proc=Path(f"/proc/{{r['pid']}}/stat")
 live=proc.exists() and proc.read_text().split()[21]==r['start_ticks']
 done=root/{label!r}/"COMPLETE.json"
 output=dict(owned_process_live=live,complete=done.exists(),receipt=r)
 if done.exists(): output['result']=json.loads(done.read_text())
 log=root/{(label+".log")!r}
 output['log_tail']=log.read_text()[-3000:] if log.exists() else ''
 print(json.dumps(output))
'''
    return execute(host,code)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["launch","status"]); p.add_argument("--host",choices=HOSTS,required=True)
    p.add_argument("--label",required=True); p.add_argument("--views")
    a=p.parse_args()
    if a.action=="launch" and not a.views: p.error("launch requires --views")
    print(launch(a.host,a.views,a.label) if a.action=="launch" else status(a.host,a.label),end="")
