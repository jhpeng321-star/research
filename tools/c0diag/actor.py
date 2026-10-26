"""Fixed observed start-state actor statistics across saved updates and seeded initialization."""
import json
from pathlib import Path
import numpy as np
import torch
from wms.control.policy import CavPolicy, RsuGroupEncoder, load_policy_components
from wms.control.obs import build_observations
from wms.control.rsu_field import build_rsu_field

torch.set_num_threads(2)
root=Path('/root/main14_loop'); data=root/'data/pairs_c00.a1'; out=Path('/root/c0diag_s1')
es=json.loads((data/'C4_EPISODES_MANIFEST.json').read_text())['episodes']
torch.manual_seed(965140)
initial=CavPolicy(accel_max=1.2,init_zero_accel=False)
initial_encoder=RsuGroupEncoder()
models=[('seeded_initialization',initial,initial_encoder)]
for name in ('policy_000100.pt','policy_latest.pt'):
    pol,enc,_,_=load_policy_components(root/'policy/c00.a2'/name)
    models.append((name,pol,enc))
rows=[]
for name,pol,enc in models:
    raw=[]; latent=[]; sig=[]; logits=[]
    for e in es:
        if e['branch']['kind']!='POLICY':continue
        kin=np.load(data/e['dir']/'veh_kin.npy',mmap_mode='r',allow_pickle=False)[120].copy()
        pre=torch.tensor(np.load(data/e['dir']/'veh_mask.npy',mmap_mode='r')[120].copy()).bool()[None]
        cav=torch.tensor(np.load(data/e['dir']/'is_cav.npy',mmap_mode='r')[120].copy()).bool()[None]
        k=torch.tensor(kin)[None]; lane=torch.where(k[:,:,1:4].sum(-1)>0,k[:,:,1:4].argmax(-1),-1)
        st=dict(s_m=k[:,:,0],v_mps=k[:,:,4],a_mps2=k[:,:,5],length_m=k[:,:,6],lane=lane,present=pre,is_cav=cav)
        mask=pre&cav&(lane>=0)
        hook=pol.head.register_forward_hook(lambda _m,_i,o:latent.append(o[mask].detach()))
        with torch.no_grad():
            obs=build_observations(st); group=enc(obs,st['s_m'],build_rsu_field(st))
            a,_=pol(obs,group,deterministic=True)
            raw.append(a[mask])
        hook.remove()
    a=torch.cat(raw); l=torch.cat(latent)
    rows.append({'checkpoint':name,'n':len(a),'deterministic_mean':float(a.mean()),'deterministic_std':float(a.std()),
                 'deterministic_min':float(a.min()),'deterministic_max':float(a.max()),'negative_fraction':float((a<0).float().mean()),
                 'latent_mean':float(l[:,0].mean()),'latent_sigma_mean':float(l[:,1].clamp(-5,1).exp().mean())})
print(json.dumps(rows,indent=2))
(out/'actor.json').write_text(json.dumps(rows,indent=2))
