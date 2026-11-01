"""Acceleration decoding at the inherited WMTEACH pre-safety boundary only."""
import ast
from contextlib import contextmanager
from contextlib import nullcontext
import inspect
from types import SimpleNamespace
import torch
from .plans import acceleration_at
from .wm import adapted_futures, executor_adapter
from .fused import FusedModel
from .warm_runtime import adapter as warm_adapter


def profile_table(actions, ids, device, horizon=300, dynamic_policy=False):
    """Compile once, avoiding per-step CPU rosters or per-vehicle GPU reads."""
    raw=torch.zeros((len(actions),int(horizon),len(ids)),device='cpu')
    mask=torch.zeros_like(raw,dtype=torch.bool)
    residual=torch.zeros_like(mask)
    slots={v:i for i,v in enumerate(ids)}
    for b,a in enumerate(actions):
        for t in range(int(horizon)):
            # All unchanged CAVs retain the SAME own-action continuation; after
            # commitment the selected group rejoins that continuation as well.
            background={} if dynamic_policy else a.get('background_residuals',{})
            for v,x in background.items():
                if v in slots:
                    raw[b,t,slots[v]]=x;mask[b,t,slots[v]]=True;residual[b,t,slots[v]]=True
            if 'knots' in a or 'profile' in a:
                values=acceleration_at(a,t);is_residual=False
            else:
                values=a.get('residuals',{}) if t<a.get('window',0) and not dynamic_policy else {};is_residual=True
            for v,x in values.items():
                if v not in slots:continue
                raw[b,t,slots[v]]=x;mask[b,t,slots[v]]=True;residual[b,t,slots[v]]=is_residual
            from .committed import background as committed_background
            for v,x in committed_background(a.get('committed_plans',[]),t).items():
                if v in slots:
                    raw[b,t,slots[v]]=x;mask[b,t,slots[v]]=True;residual[b,t,slots[v]]=False
    return raw.to(device),mask.to(device),residual.to(device)


def profile_apply(accel,tick,table,bounds):
    raw,mask,residual=table
    t=int(tick*.5)
    if t>=raw.shape[1]:return accel
    n=raw.shape[-1]
    out=accel.clone()
    request=torch.where(residual[:,t],accel[:,:n]+raw[:,t],raw[:,t])
    out[:,:n]=torch.where(mask[:,t],request.clamp(*bounds),accel[:,:n])
    return out


@contextmanager
def plan_executor(horizon=300, policy=None, trace=None, final_chain=False):
    """Keep original bridge, memory, lane mask, cap, and safe_accel implementation."""
    from tools.wmteach import runtime
    original=runtime.option_controller
    tree=ast.parse(inspect.getsource(original));matches=0
    outer=tree.body[0]
    # Inherited fixed targets are irrelevant to acceleration profiles.
    for i,node in enumerate(outer.body):
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='fixed' for x in node.targets):
            outer.body[i]=ast.parse("table=profile_table(candidate_actions,ids,next(mp.policy.parameters()).device,horizon,dynamic_policy=policy is not None)").body[0]
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name=='injected':
            matches+=1
            node.body=ast.parse("raw=decode(state,accel,tick,table,(mp.policy.accel_min,mp.policy.accel_max),candidate_actions,ids)\nreturn post_safety(state,raw,tick,safe_accel,kw)").body
    if matches!=1:raise ValueError('inherited pre-safety ABI changed')
    ast.fix_missing_locations(tree)
    held=[None]
    from .command_chain import SlowDown
    chain=SlowDown() if final_chain else None
    def post_safety(state,raw,tick,safety,kw):
        if chain is None:return safety(state,raw,**kw)
        from wms.control.sumo_idm import allowed_speed
        from wms.control.alwayscmd import inferred_cap
        # SUMO computes safe_accel/CommandOwnership once per control decision,
        # then executes the slowDown timeline under speedMode27 every half-step.
        safe=safety(state,raw,**kw) if tick%2==0 else raw
        return chain.apply(state,safe,tick,allowed_speed(state['lane'],state['s_m']),inferred_cap(state))
    def decode(state,accel,tick,table,bounds,actions,ids):
        if policy is not None:
            from .policy_wm import residuals
            if tick%2==0 or held[0] is None:held[0]=residuals(policy,state,ids)
            own=torch.tensor([a['id']!='MAPPO' for a in actions],device=accel.device)[:,None]
            accel=accel+held[0]*own
        raw=profile_apply(accel,tick,table,bounds)
        if trace is not None and tick%2==0:trace.append(raw.detach().clone())
        return raw
    ns=dict(original.__globals__,profile_table=profile_table,horizon=horizon,policy=policy,decode=decode,post_safety=post_safety)
    exec(compile(tree,'<coopmwm-acceleration-profile-decoder>','exec'),ns)
    runtime.option_controller=ns['option_controller']
    try:yield
    finally:runtime.option_controller=original


def evaluate(rt,record,actions,K,H,chunk_actions=4,trim=True,pad_actions=False,horizons=None,speed2=False,memory=None,trace=None,transfers=False,final_chain=False):
    """Same source draws, CRN keys, insertions, departure loop and J reducer.

    A single compact-forward proxy retains full virgin-slot controller memory.
    Candidate chunks bound memory without changing future IDs or source capacity.
    """
    hs=tuple(horizons or (H,))
    output={str(h):{a['id']:[] for a in actions} for h in hs}
    proxy=FusedModel(rt.model,1,trim_absent=trim or speed2);proxy.local.index=0
    from .speed2_runtime import ledger,compact_geometry
    runtime=SimpleNamespace(**vars(rt));runtime.model=proxy
    if transfers and not speed2:raise ValueError('transfer batching requires qualified speed2 geometry')
    from .speed2_transfers import fast_transfers
    policy=None
    if record.get('policy_state') is not None:
        from .policy_wm import load_actor
        policy=load_actor(record['policy_state'],next(rt.mp.policy.parameters()).device)
    with (memory.context(rt,record) if memory is not None else warm_adapter(rt,record)),(compact_geometry(proxy) if speed2 else nullcontext()),(fast_transfers() if transfers else nullcontext()),executor_adapter(),plan_executor(H,policy,trace,final_chain=final_chain):
        for start in range(0,len(actions),chunk_actions):
            subset=actions[start:start+chunk_actions]
            subset=[dict(a,background_residuals=record['policy_residuals']) if ('knots' in a or 'profile' in a) else a for a in subset]
            if record.get('committed_plans'):
                subset=[dict(a,committed_plans=a.get('committed_plans',record['committed_plans'])) for a in subset]
            real_ids=[a['id'] for a in subset]
            if pad_actions:
                subset=list(subset)
                width=4 if pad_actions is True else int(pad_actions)
                while len(subset)<width:
                    subset.append(dict(subset[-1],id=f'_pad{len(subset)}'))
            ns=adapted_futures(hs);ns['_actions']=subset
            if speed2:
                proxy.live_slots=max(1,len(record['rows']))
            with (ledger(proxy) if speed2 else nullcontext()),torch.no_grad():
                result=ns['futures'](runtime,dict(record,setting=0),indices=tuple(range(K)))
            if not result['population_audit']['finite'] or not result['population_audit']['conserved']:
                raise ValueError('imagined population accounting failed')
            for h in hs:
                for a in real_ids:output[str(h)][a]=result['futures'][str(h)][a]
    return output if horizons is not None else output[str(H)]
