"""S-C0DIAG-1 allowlisted, offline-only artifact inventory and training audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/root/main14_loop')
OUT = Path('/root/c0diag_s1')
assert OUT.is_dir()
torch.set_num_threads(2)
m = json.loads((ROOT/'data/pairs_c00.a1/C4_EPISODES_MANIFEST.json').read_text())
print('MANIFEST', str(m)[:6000])
s = json.loads((ROOT/'policy/c00.a2/RUN_SUMMARY.json').read_text())
print('SUMMARY', json.dumps({k: v for k, v in s.items() if k not in ('updates', 'history', 'metrics')})[:18000])
u = [json.loads(l) for l in (ROOT/'policy/c00.a2/updates.jsonl').read_text().splitlines()]
print('UPDATE_KEYS', list(u[0]))
print('UPDATES', json.dumps([{k:r.get(k) for k in ('update','action_stats','independent_J','returns','B','D_all','critic_real_loss','critic_imagined_loss','entropy','lagrange_multiplier')} for r in u if r['update'] in (1,10,50,100,150)]))
p = torch.load(ROOT/'policy/c00.a2/policy_latest.pt', map_location='cpu', weights_only=True)
print('CHECKPOINT', {k:v for k,v in p.items() if k not in ('policy','group_encoder','critic','target_critic','actor_optimizer','critic_optimizer','rng','sampler','optimizers') and not isinstance(v,(dict,list,torch.Tensor))})
print('CK_KEYS', list(p))
entries = m.get('episodes',m.get('entries',[]))
if entries:
    e=entries[0]
    d=ROOT/'data/pairs_c00.a1'/e['dir']
    print('EPISODE_FILES', [x.name for x in d.iterdir()])
result={'summary':{k:v for k,v in s.items() if k not in ('updates','history','metrics')},
        'selected_updates':[{k:r.get(k) for k in ('update','action_stats','independent_J','returns','B','D_all','critic_real_loss','critic_imagined_loss','entropy','lagrange_multiplier')} for r in u if r['update'] in (1,10,50,100,150)],
        'checkpoint_keys':list(p), 'policy_sha256':hashlib.sha256((ROOT/'policy/c00.a2/policy_latest.pt').read_bytes()).hexdigest()}
(OUT/'audit.json').write_text(json.dumps(result,indent=2))
