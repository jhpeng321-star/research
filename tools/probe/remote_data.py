"""Read only named accepted artifacts on gpuA; stream to cpuP:~/probe/data.

No remote enumeration, simulator, process discovery or training is performed.
"""
import json
import subprocess
import sys

SCRIPT = r'''
import json
from pathlib import Path
root = Path('/root/autodl-tmp/c4/restart20261026/arm')
state = json.loads((root/'loop/LOOP14_STATE.json').read_text())
cfg = json.loads((root/'CONFIG.json').read_text())
import torch
policy = torch.load(state['policy']['checkpoint'], map_location='cpu', weights_only=True)
print(json.dumps({'accepted': state['accepted_wm'], 'policy': state['policy'],
                 'native_reference': policy.get('counterfactual')}))
'''
result = subprocess.run(["ssh", "gpuA", "/root/miniconda3/bin/python -"], input=SCRIPT, text=True, capture_output=True)
if result.returncode:
    raise SystemExit(result.stderr)
print(result.stdout)
if "--copy" in sys.argv:
    bindings = json.loads(result.stdout)
    script = r'''
import sys, tarfile
from pathlib import Path
root = Path('/root/autodl-tmp/c4/restart20261026/arm/loop')
files = [(root/'wm/c01'/n, 'wm/'+n) for n in ('wm_s.pt', 'physics.json', 'TRAIN_RECEIPT.json', 'config.resolved.json')]
files.append((root/'policy/c01.a3/policy_latest.pt', 'policy.pt'))
ep = root/'worlds/c01/w968157.a4/artifact/pairs/episodes/w968157m1900p05t300__loop14_pairs_b0360_native'
files += [(ep/(n+'.npy'), 'episode/'+n+'.npy') for n in ('veh_kin','veh_mask','target_mask','vid','is_cav','time_s','counter')]
files.append((ep/'episode.json', 'episode/episode.json'))
head = HEAD_PATH
if head:
    files.append((Path(head), 'native_head.json'))
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:
    for path, name in files:
        if any(x in str(path).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')):
            raise ValueError('restricted path')
        tar.add(path, arcname=name, recursive=False)
'''
    head = (bindings.get("native_reference") or {}).get("native_cav_head_path")
    script = script.replace("HEAD_PATH", repr(head))
    subprocess.run(["ssh", "-n", "cpuP", "test -d ~/probe && mkdir -p ~/probe/data"], check=True)
    read = subprocess.Popen(["ssh", "gpuA", "/root/miniconda3/bin/python -"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    read.stdin.write(script.encode())
    read.stdin.close()
    write = subprocess.run(["ssh", "cpuP", "tar -xf - -C ~/probe/data"], stdin=read.stdout)
    read.stdout.close()
    if read.wait() or write.returncode:
        raise SystemExit("artifact streaming failed")
    print("Named cycle-1 observable state and checkpoints streamed to cpuP:~/probe/data")
