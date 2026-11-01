"""Read-only ownership and GPU-drain status; never dispatch or signal anything."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from tools.rb2int.audit import ROOT
from tools.rb2int.drain_ops import remote_code
from tools.rb2int.hold_ops import PROCESSES
from tools.rb2int.interim_watch import cache_state, hold_state
from tools.rb2int.resume_ci import alive
from tools.rbint.remote import ssh
from tools.rbeval.core import load, save_new


def host_status(host):
    helpers = remote_code(host).split("process=load(root/'SUPERVISOR_PROCESS.json')")[0]
    code = helpers + '''
hold=load(root/'OWNER_1755_HOLD_APPLIED.json')
paused=[]
for item in hold['paused']:
 identity=item['process'];pid=identity['pid']
 assert incarnation(pid)==identity
 assert state(pid)=='T', 'new admission owner is no longer paused'
 paused.append(dict(process=identity,state=state(pid),stage=item['stage']))
active=[]
claims=[]
for path in sorted((root/'jobs').glob('*/PROCESS.json')):
 process=load(path);claims.append(path.parent.name)
 try:
  identity={k:process[k] for k in ('pid','start_ticks','boot_id')}
  if incarnation(process['pid'])==identity and state(process['pid'])!='Z':
   active.append(dict(job=path.parent.name,process=identity,state=state(process['pid'])))
 except FileNotFoundError: pass
print(json.dumps(dict(time=datetime.now().astimezone().isoformat(),paused=paused,
 active_evaluations=active,claimed_jobs=claims,new_gpu_admissions=0,
 gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip(),
 inference_dispatched=False,signals_sent=False)))
'''
    return json.loads(ssh(host, code, timeout=90))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    out = Path(args.receipt)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('exclusive receipt in existing parent required')
    local = []
    for name, pid, created in (
        ('rb2int_finish_v3',25240,134379970282244431),
        ('rb2int_interim_v3',44988,134379973214244749),
        ('rb2int_interim_hold_v1',23664,134380020249390235),
        ('rb2int_late_wait_v1',9724,134380004640417822),
    ):
        folder = ROOT/'runs'/name
        record = load(folder/'PROCESS.json')
        if (record['pid'],record['create_filetime']) != (pid,created):
            raise ValueError('local ownership changed')
        local.append(dict(root=str(folder),process=record,alive=alive(record),
            terminal={p.name:load(p) for p in folder.glob('*.json')
                      if p.name in ('STOP.json','COMPLETE.json','INCOMPLETE_AT_DEADLINE.json')},
            log_tail=(folder/'watch.log').read_text(encoding='utf-8')[-4000:]))
    hosts = {host:host_status(host) for host in PROCESSES}
    worlds, count = cache_state()
    result = dict(time=datetime.now().astimezone().isoformat(),local=local,hosts=hosts,
                  complete_collected_world_jobs=count,matched_v5_v4_worlds=worlds,
                  hold_children_all_collected=hold_state(),inference_dispatched=False,
                  signals_sent=False)
    save_new(out,result)
    print(json.dumps(result,indent=1))


if __name__ == '__main__':
    main()
