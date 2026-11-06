"""Hash an explicit controller source allowlist and read only loop receipts."""
import json, hashlib, socket, os
from pathlib import Path
host=socket.gethostname()
gpu=os.geteuid()==0
runtime=Path('/root/autodl-tmp/c4/v5/mainstart/mainr16/restart/runtime') if gpu else Path('/home/ubuntu/c4/main14/mainr16/runtime')
out=Path('/root/c0diag_s1') if gpu else Path('/home/ubuntu/c4/c0diag_s1')
names=['src/wms/control/policy.py','src/wms/control/obs.py','src/wms/control/contract.py',
       'src/wms/control/safety.py','src/wms/control/lanes.py','src/wms/control/bridge.py',
       'src/wms/control/policy14.py','src/wms/control/policy14_cli.py',
       'traffic_shepherd/c4/policy_pairs.py','traffic_shepherd/c4/env.py',
       'traffic_shepherd/ownership.py','traffic_shepherd/damping.py',
       'configs/c4_policy14_main14_nativelane.toml']
result={'hostname':host,'source':{n:hashlib.sha256((runtime/n).read_bytes()).hexdigest() for n in names}}
if gpu:
    arm=runtime.parent
    cfg=json.loads((arm/'CONFIG.json').read_text())
    result['configuration']={k:cfg.get(k) for k in ('fast','fastloop','policy','collect')}
    result['policy_config_text']=(runtime/'configs/c4_policy14_main14_nativelane.toml').read_text()
    result['state']=json.loads(Path('/root/main14_loop/LOOP14_STATE.json').read_text())
    result['world_journal']=json.loads(Path('/root/main14_loop/worlds_c00.json').read_text())
else:
    seed=968129 if 'cpu-p' in host.lower() else 968130
    for seed in (968129,968130):
        ck=runtime.parent/f'work/w{seed}/C4-LOOP14-COOPON-20261026/c00/inputs/policy.pt'
        if ck.is_file():result['collected_policy_sha']=hashlib.sha256(ck.read_bytes()).hexdigest()
(out/'provenance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result)[:10000])
