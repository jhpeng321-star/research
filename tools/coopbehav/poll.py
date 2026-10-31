"""Bounded, read-only handoff polling; reports run only on complete explicit indices.

python -m tools.coopbehav.poll --config tools/coopbehav/handoffs.json --out ...
Use --watch to keep polling (>=60 s); an own-root STOP file exits cooperatively.
The poller records design-document changes and consumes ready pair indices.
No recursive search, simulation, process signals, or arbitrary document commands.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .io import read_json, safe
from .report import run_report, write_json

REMOTE_INDEX=re.compile(r"\b(gpuR|cpuQ):(/[A-Za-z0-9_./-]*COOPBEHAV_INDEX\.json)")


def publish(config,entry):
    """Copy only completed own-root derived reports; optionally step-commit them."""
    from .remote import HOSTS
    from .verify import verify_report
    if entry.get("state")!="complete" or not config.get("publish_dir"): return
    sha=entry["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}",sha): raise ValueError("Invalid report SHA")
    destination=safe(config["publish_dir"]); destination.mkdir(parents=True,exist_ok=True)
    target=destination/("paired_"+sha)
    host=entry["host"]; _,root=HOSTS[host]
    if not (target/"COMPLETE.json").is_file():
        result=subprocess.run(["scp","-r","-o","BatchMode=yes","-o","ConnectTimeout=15",
            f"{host}:{root}/paired_{sha}",str(destination)],capture_output=True,text=True,timeout=120)
        if result.returncode: raise RuntimeError(result.stderr)
    verification=verify_report(target)
    entry["published"]=str(target); entry["verified"]=verification["verified"]
    if not config.get("commit_published"): return
    repo=Path(__file__).resolve().parents[2]
    relative=target.resolve().relative_to(repo).as_posix()
    def git(*args):
        r=subprocess.run(["git",*args],cwd=repo,capture_output=True,text=True,timeout=120)
        if r.returncode: raise RuntimeError(r.stderr or r.stdout)
        return r.stdout
    # Observe status/diff/history before every step commit; never stage peer files.
    if not git("status","--porcelain","--",relative).strip(): return
    git("status","--short"); git("diff","--",relative); git("log","--oneline","-10")
    if git("diff","--cached","--name-only").strip():
        entry["commit_state"]="waiting_for_clean_index"; return
    git("add","--",relative); git("diff","--cached","--check")
    staged=git("diff","--cached","--name-only").splitlines()
    if not all(p.startswith(relative+"/") for p in staged):
        raise RuntimeError("Unexpected staged path; no commit performed")
    git("commit","-m",f"S-COOPBEHAV: paired trace report {sha[:12]}")
    entry["commit"]=git("rev-parse","HEAD").strip()


def remote_poll(host,path):
    from .remote import HOSTS, execute
    safe(path)
    python,root=HOSTS[host]
    # Read only the explicit handoff; native_release pairing and restricted-path
    # validation are performed again by report.py before any trace is opened.
    code=f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,{root!r})
from tools.coopbehav.io import safe
p=safe({path!r}); root=Path({root!r})
result=dict(path=str(p),exists=p.is_file())
if p.is_file():
 data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest(); index=json.loads(data)
 result.update(sha256=sha,ready=index.get('ready') is True)
 if result['ready']:
  target=root/('paired_'+sha); receipt=root/('paired_'+sha+'_PID.json')
  if (target/'COMPLETE.json').is_file(): result['state']='complete'
  elif receipt.is_file():
   r=json.loads(receipt.read_text()); proc=Path(f"/proc/{{r['pid']}}/stat")
   live=proc.exists() and proc.read_text().split()[21]==r['start_ticks']
   result.update(state='running' if live else 'failed',receipt=r)
  else:
   assert root.is_dir()
   active=root/'PAIR_ACTIVE.json'
   if active.is_file():
    owned=json.loads(active.read_text()); proc=Path(f"/proc/{{owned['pid']}}/stat")
    if proc.exists() and proc.read_text().split()[21]==owned['start_ticks']:
     result.update(state='waiting_for_owned_report',receipt=owned)
     print(json.dumps(result)); raise SystemExit(0)
   frozen=root/('index_'+sha+'.json')
   if not frozen.exists():
    document=dict(index,source_index_sha256=sha)
    for pair in document.get('pairs',[]):
     for name in ('factual','reference'):
      q=Path(pair[name]); pair[name]=str(safe(q if q.is_absolute() else p.parent/q))
    frozen.write_text(json.dumps(document,indent=2))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   if (root/'deps').is_dir(): env['PYTHONPATH']=str(root/'deps')
   command=[{python!r},'-m','tools.coopbehav.report','--pairs',str(frozen),'--out',str(target),'--label','paired']
   log=(root/('paired_'+sha+'.log')).open('ab')
   child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
   r=dict(pid=child.pid,start_ticks=Path(f'/proc/{{child.pid}}/stat').read_text().split()[21],command=command)
   receipt.write_text(json.dumps(r)); result.update(state='launched',receipt=r)
   active.write_text(json.dumps(r))
print(json.dumps(result))
'''
    return json.loads(execute(host,code))


def poll_once(config,out):
    out=safe(out); out.mkdir(parents=True,exist_ok=True)
    status=dict(timestamp=time.time(),pid=os.getpid(),documents=[],indices=[],reports=[],remote_indices=[],sources=[])
    remotes={(r["host"],r["path"]) for r in config.get("remote_indices",[])}
    for raw in config.get("documents",[]):
        path=safe(raw)
        entry=dict(path=str(path),exists=path.is_file())
        if path.is_file():
            data=path.read_bytes()
            entry.update(sha256=hashlib.sha256(data).hexdigest(),
                         trace_handoff_lines=[line for line in data.decode("utf-8").splitlines()
                                              if "trace" in line.lower() or "COOPBEHAV_INDEX" in line])
            remotes.update(REMOTE_INDEX.findall(data.decode("utf-8")))
        status["documents"].append(entry)
    for raw in config.get("indices",[]):
        path=safe(raw)
        entry=dict(path=str(path),exists=path.is_file())
        status["indices"].append(entry)
        if not path.is_file(): continue
        data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        entry["sha256"]=digest
        index=json.loads(data)
        # Producers atomically publish ready=true only after closing/hashing all traces.
        if index.get("ready") is not True:
            entry["state"]="producer_not_ready"; continue
        target=out/("paired_"+digest)
        if (target/"COMPLETE.json").is_file():
            entry["state"]="complete"; continue
        try:
            report=run_report([],target,path,label=index.get("source","paired"))
            entry["state"]="complete"
            status["reports"].append(dict(path=str(target),episodes=report["episodes"]))
        except Exception as exc:
            entry.update(state="blocked",reason=f"{type(exc).__name__}: {exc}")
    for host,path in sorted(remotes):
        try:
            status["remote_indices"].append(dict(host=host,**remote_poll(host,path)))
        except Exception as exc:
            status["remote_indices"].append(dict(host=host,path=path,state="blocked",reason=str(exc)))
    for source in config.get("remote_sources",[]):
        from .remote import HOSTS,execute
        host,kind,root=source["host"],source["kind"],source["root"]
        safe(root)
        _,ownroot=HOSTS[host]
        code=f'''import json,sys
sys.path.insert(0,{ownroot!r})
from tools.coopbehav.sources import build_source_index
print(json.dumps(build_source_index({kind!r},{root!r},{(ownroot+'/source_'+kind)!r})))
'''
        try:
            entry=json.loads(execute(host,code)); entry["host"]=host
            status["sources"].append(entry)
            status["remote_indices"].append(dict(host=host,**remote_poll(host,entry["index"])))
        except Exception as exc:
            status["sources"].append(dict(host=host,root=root,state="blocked",reason=str(exc)))
    for entry in status["remote_indices"]:
        try: publish(config,entry)
        except Exception as exc: entry["publish_error"]=str(exc)
    write_json(out/"POLL_STATUS.json",status)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True); p.add_argument("--out",required=True)
    p.add_argument("--watch",action="store_true"); p.add_argument("--interval",type=float,default=120.)
    a=p.parse_args()
    if a.interval<60: p.error("Polling interval must be >=60 seconds")
    out=safe(a.out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/"PID.json",dict(pid=os.getpid(),start_time=time.time(),command="coopbehav.poll"))
    while not (out/"STOP").exists():
        print(json.dumps(poll_once(read_json(a.config),out)),flush=True)
        if not a.watch: break
        time.sleep(a.interval)


if __name__=="__main__": main()
