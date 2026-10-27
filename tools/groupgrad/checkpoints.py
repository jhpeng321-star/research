"""CPU checkpoint census; reconstruct native-init with its registered RNG seed."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path.home() / 'groupgrad'
sys.path[:0] = [str(ROOT / 'runtime/src'), str(ROOT / 'runtime')]
import torch
from wms.control.policy import CavPolicy, RsuGroupEncoder
from wms.control.train_cli import GroupSignal
from wms.control.real_returns14 import ScaledTeamCritic


def delta(a, b):
    rows = []
    for key in a:
        if key not in b or a[key].shape != b[key].shape:
            rows.append(dict(name=key, shape=list(a[key].shape), changed=True,
                             max_abs=0., l2=0., comparison='new key/shape; not a numeric delta'))
            continue
        d = a[key].double() - b[key].double()
        rows.append(dict(name=key, shape=list(d.shape), changed=not torch.equal(a[key], b[key]),
                         max_abs=float(d.abs().max()), l2=float(d.norm())))
    return dict(n=len(rows), changed=sum(r['changed'] for r in rows), missing=sorted(b.keys()-a.keys()),
                max_abs=max((r['max_abs'] for r in rows), default=0),
                l2=sum(r['l2']**2 for r in rows)**.5, tensors=rows)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(965140)
    policy = CavPolicy(lane_head=False, residual_on_native=True, accel_max=1.2)
    group = GroupSignal(RsuGroupEncoder())
    critic = ScaledTeamCritic(entry_backlog=True, value_scale=10000.)
    init = dict(policy=policy.state_dict(), group_encoder=group.state_dict(),
                critic=critic.state_dict(), target_critic=critic.state_dict())
    payloads = {label: torch.load(ROOT / f'checkpoints/{label}.pt', map_location='cpu',
                                weights_only=False) for label in ('c00','c01_on','c01_off','c02')}
    # Verify the RNG reconstruction, rather than assuming a random-feature match.
    assert all(torch.equal(v, payloads['c00']['policy'][k]) for k, v in init['policy'].items()
               if not k.startswith('residual_head.')), 'init reconstruction failed'
    assert all(torch.equal(v, payloads['c00']['group_encoder'][k])
               for k, v in init['group_encoder'].items()), 'group reconstruction failed'
    result = dict(init=dict(seed=965140, reconstruction='CLI reseeds before actor/group/critic construction; '
                           'all 15 non-residual actor tensors and 26 group tensors verified exact; '
                           'critic is constructor init, before real-return/longcritic warmups'), checkpoints={})
    for label, p in payloads.items():
        mods = {k: delta({n.removeprefix('base.') if k.endswith('critic') else n:v
                          for n,v in p[k].items()}, init[k])
                for k in ('policy','group_encoder','critic','target_critic')}
        for name, prefix in [('residual','residual_head.'),('features','own_encoder.'),
                             ('unused_absolute_head','head.'),('lane','lane_head.')]:
            mods[name] = delta({k:v for k,v in p['policy'].items() if k.startswith(prefix)},
                               {k:v for k,v in init['policy'].items() if k.startswith(prefix)})
        result['checkpoints'][label] = dict(update=p['update_count'], modules=mods,
            sha256=hashlib.sha256((ROOT / f'checkpoints/{label}.pt').read_bytes()).hexdigest(),
            actor_optimizer_groups=p['actor_optimizer']['param_groups'],
            actor_optimizer_steps={str(k):float(v['step']) for k,v in p['actor_optimizer']['state'].items()},
            config=p['config'],counterfactual=p['counterfactual'],coopw=p['coopw'])
    result['on_minus_off'] = {k:delta(payloads['c01_on'][k], payloads['c01_off'][k])
                             for k in ('policy','group_encoder','critic','target_critic')}
    out = ROOT/'checkpoint_deltas.json'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{m:{n:v for n,v in d.items() if n!='tensors'} for m,d in x['modules'].items()}
                      for k,x in result['checkpoints'].items()},indent=2))
    print('CPU checkpoint census PASS', out)


if __name__ == '__main__':
    main()
