"""Transport an already prepared causal warm query without changing its sampler.

Posterior construction belongs to the warm-memory owner. This adapter accepts
the native FULL-v4 fitted-state result and genuine observable history, rekeys
its identities to WMTEACH's existing virtual-future/packed-slot convention, and
keeps the inherited encoder, draws, source prior, lane continuation and costs.
"""
import ast
import copy
from contextlib import contextmanager
import inspect
import time
from unittest.mock import patch

import torch


class WarmQuery:
    def __init__(self, source, posteriors_by_slot, t0_s, receipt):
        self.source=source
        self.posteriors=posteriors_by_slot
        self.t0_s=t0_s
        self.receipt=receipt
        self.encoder_seconds=[]

    @classmethod
    def from_record(cls,rt,record):
        """Consume owner-prepared observable histories/posteriors, never refit them.

        Rows determine slot identity. The producer must validate its causal60s
        observation buffer before serializing these history features and native
        finite-support distributions. An empty payload is exactly the cold path.
        """
        from tools.wmteach import runtime
        payload=record.get('warm_memory')
        if not payload:return None
        allowed={'t0_s','history','history_mask','driver_history','driver_history_mask','posteriors_by_vid'}
        if set(payload)-allowed or payload.get('t0_s')!=record['time_s']:
            raise ValueError('warm query observable/time whitelist mismatch')
        source=runtime.diagnostic_source(rt,record,0)
        n=len(record['rows']);device=source.state['s_m'].device
        histories={}
        for key in ('history','driver_history'):
            if key not in payload:continue
            value=torch.as_tensor(payload[key],dtype=torch.float32,device=device)
            mask=torch.as_tensor(payload[key+'_mask'],dtype=torch.bool,device=device)
            if value.ndim!=4 or value.shape[:2]!=(1,n) or mask.shape!=value.shape[:-1] or not torch.isfinite(value).all():
                raise ValueError('invalid identity-aligned observable warm history')
            histories.update({key:value,key+'_mask':mask})
        if not histories:raise ValueError('nonempty warm payload requires observable history')
        slots={r['vid']:i for i,r in enumerate(record['rows'])}
        posteriors=payload.get('posteriors_by_vid',{})
        if set(posteriors)-set(slots):raise ValueError('posterior for unobserved identity')
        if not any(bool(v.any()) for k,v in histories.items() if k.endswith('_mask')):
            if posteriors:raise ValueError('empty history cannot supply a driver posterior')
            return None
        source.rollout_kwargs.update(histories)
        return cls(source,{slots[v]:p for v,p in posteriors.items()},record['time_s'],
                   dict(posterior_source='caller_prepared_native_causal_motion',new_sumo_sim_s=0))

    @contextmanager
    def context(self, rt, record):
        from tools.wmteach import runtime,marginal
        assert record['time_s']==self.t0_s
        for name in ('s_m','v_mps','lane','is_cav','length_m'):
            values=self.source.state[name][0,:len(record['rows'])].detach().cpu().tolist()
            expected=[r.get(name,5.) for r in record['rows']]
            # Compare at the source tensor's precision, not JSON double precision.
            expected=torch.as_tensor(expected,dtype=self.source.state[name].dtype).tolist()
            if values!=expected:raise ValueError('warm source belongs to a different observable roster: '+name)
        original=marginal.plans
        tree=ast.parse(inspect.getsource(original));count=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='with_motion_posteriors':
                assert isinstance(node.args[0],ast.Dict) and not node.args[0].keys
                node.args[0]=ast.parse('warm_posteriors(row_keys)',mode='eval').body
                count+=1
        assert count==1,'frozen warm posterior binding ABI changed'
        ast.fix_missing_locations(tree)
        def posteriors(keys):
            return {(int(key),slot):distribution for key in keys for slot,distribution in self.posteriors.items()}
        ns=dict(original.__globals__,warm_posteriors=posteriors)
        exec(compile(tree,'<speed2-native-warm-posterior-transport>','exec'),ns)
        def source(runtime_, record_, future):
            assert record_['state_id']==record['state_id']
            return copy.deepcopy(self.source)
        model=rt.model
        encode=model.encode_drivers
        def timed(*args,**kw):
            device=args[0]['s_m'].device
            if device.type=='cuda':torch.cuda.synchronize(device)
            start=time.perf_counter()
            out=encode(*args,**kw)
            if device.type=='cuda':torch.cuda.synchronize(device)
            self.encoder_seconds.append(time.perf_counter()-start)
            return out
        with patch.object(runtime,'diagnostic_source',source),patch.object(marginal,'plans',ns['plans']),patch.object(model,'encode_drivers',timed):
            yield


def native_fixture(rt):
    """Fixed FIT-only observable fixture; native posterior construction, no physics.

    The frozen loader admits969420..969439; explicitly pin969420 (within the
    bootstrap's969400..969431 FIT allowlist) before it opens any world path.
    """
    from tools.rbint.observations import load_motion_history
    from tools.rbint.recipe import fitted_state_at_start
    from .protocol import digest
    start=time.perf_counter()
    seed=969420
    assert 969400<=seed<969432
    observed,raw=load_motion_history(seed)
    source=copy.deepcopy(rt.batches[seed-969400])
    selected=observed['state']['present'][0].bool()
    ids=torch.as_tensor(raw['vid_index'][-1])[selected].long().tolist()
    n=len(ids)
    assert int(source.state['present'].sum())==n
    for name in ('s_m','v_mps','lane','is_cav','length_m'):
        expected=observed['state'][name][:,selected].to(source.state[name])
        assert torch.equal(source.state[name][:,:n],expected),name
    # Read only observed kinematics/identity and causal history fields.
    allowed={'history','history_mask','driver_history','driver_history_mask'}
    source.rollout_kwargs={k:v for k,v in source.rollout_kwargs.items() if k in allowed}
    assert 'driver_history' in source.rollout_kwargs,'native start must have genuine driver history'
    mask=source.rollout_kwargs['driver_history_mask']
    assert mask.any(),'warm fixture has empty history'
    fitted,native_receipt=fitted_state_at_start(rt.model,seed)
    by_slot={slot:copy.deepcopy(fitted.posterior['drivers'][f'{seed}:{vid}'])
             for slot,vid in enumerate(ids) if f'{seed}:{vid}' in fitted.posterior['drivers']}
    assert by_slot,'native warm fixture must have per-driver motion posteriors'
    state={k:v[0,:n].detach().cpu().tolist() for k,v in source.state.items()
           if k in ('s_m','v_mps','lane','is_cav','length_m')}
    rows=[dict(vid=f'observed.{vid}',**{k:state[k][slot] for k in state}) for slot,vid in enumerate(ids)]
    cavs=[r['vid'] for r in rows if r['is_cav']]
    record=dict(world_id=seed,time_s=300,rows=rows,group=cavs[:2],
                commands={v:0. for v in cavs},policy_residuals={v:-.25 for v in cavs})
    record['state_id']=digest(dict(world=seed,time_s=300,rows=rows,memory='native_observable60s'))
    receipt=dict(native_receipt,world_id=seed,history_shape=list(source.rollout_kwargs['driver_history'].shape),
                 valid_history_samples=int(mask.sum()),posterior_drivers=len(by_slot),
                 preparation_wall_s=time.perf_counter()-start,new_sumo_sim_s=0)
    return record,WarmQuery(source,by_slot,300,receipt)
