"""Observable tensor equivalent of the residual actor for imagined continuations."""
import torch
from .acceleration import Policy


def actor_state(policy):
    return {k:v.detach().cpu().tolist() for k,v in policy.state_dict().items()
            if k.startswith(('actor.', 'broadcast.'))}


def load_actor(state, device):
    policy=Policy().to(device).eval()
    own=policy.state_dict()
    required={k for k in own if k.startswith(('actor.', 'broadcast.'))}
    if set(state)!=required:raise ValueError('residual actor state whitelist')
    own.update({k:torch.as_tensor(v,device=device,dtype=own[k].dtype) for k,v in state.items()})
    policy.load_state_dict(own)
    policy.zero_actor=not bool(policy.actor[-1].weight.count_nonzero() or policy.actor[-1].bias.count_nonzero())
    return policy


def features(state, ids):
    """Match encode(): same sensed radius, nearest-neighbour and identity tie rule."""
    present=state['present'];s=state['s_m'];v=state['v_mps'];lane=state['lane'];cav=state['is_cav']
    live=present.any(0).nonzero().flatten()
    # Lexical identity order supplies the inherited nearest-distance tie break.
    order=sorted(live.tolist(),key=lambda i:ids[i] if i<len(ids) else 'new:'+str(i))
    ix=torch.tensor(order,device=s.device,dtype=torch.long)
    ss=s[:,ix];vv=v[:,ix];ll=lane[:,ix];pp=present[:,ix];cc=cav[:,ix]
    x=torch.stack((ss/1000,vv/30,*[(ll==l).to(s) for l in range(3)]),-1)
    ds=ss[:,None,:]-ss[:,:,None]
    extras=[]
    for l in range(3):
        for direction in (-1,1):
            valid=pp[:,None,:] & (ll[:,None,:]==l) & (ds.abs()<=100) & (direction*ds>0)
            distance=torch.where(valid,ds.abs(),torch.inf)
            gap,j=distance.min(-1);ok=torch.isfinite(gap)
            gather=lambda value:value.gather(1,j)
            f=torch.stack((gather(ss)-ss,gather(vv),gather(cc).to(s),ok.to(s)),-1)
            f=f/torch.tensor([100,30,1,1],device=s.device)
            extras.append(torch.where(ok[...,None],f,0.))
    x=torch.cat((x,*extras),-1)
    rsu=[]
    for l in range(3):
        mask=pp & (ll==l) & (ss>=-100) & (ss<=75.44)
        n=mask.sum(-1);den=n.clamp_min(1)
        rsu.extend((n/10,(vv*mask).sum(-1)/den/30,(ss*mask).sum(-1)/den/100,(cc*mask).sum(-1)/10))
    return ix,x,torch.stack(rsu,-1),pp & cc


def residuals(policy,state,ids):
    out=torch.zeros_like(state['s_m'])
    if policy.zero_actor:return out
    ix,x,rsu,mask=features(state,ids)
    latent=policy.broadcast(rsu)[:,None,:].expand(-1,x.shape[1],-1)
    delta=policy.actor(torch.cat((x,latent),-1)).squeeze(-1)
    out[:,ix]=torch.where(mask,delta,0.)
    return out
