"""FULL-v4 causal history encoder + learned motion-posterior marginal adapter.

No saved identity lookup. Motion likelihood, support, residual scale and shrinkage
are the frozen FULL-v4 implementation. Empty history delegates byte-for-byte to
the qualified cold producer. Runtime conditioning is performed once per query.
"""
import ast
from contextlib import contextmanager
import inspect
from types import SimpleNamespace
import time
from .warm_history import arrays


def condition(model, record):
    import numpy as np
    import torch
    from tools.rbint import motion_posterior as frozen
    raw = arrays(record)
    if raw is None:
        return None
    if not hasattr(model,'_wmphys_motion_fit'):
        raise ValueError('FULL-v4 frozen motion inference unavailable')
    # Generalize ONLY the original registered clock, retaining its selector,
    # physical likelihood, support order and learned shrinkage implementation.
    tree = ast.parse(inspect.getsource(frozen.condition_population))
    class Clock(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is float and node.value in (240.,300.):
                return ast.copy_location(ast.Constant(record['time_s']-(60. if node.value==240. else 0.)),node)
            return node
    tree=Clock().visit(tree);ast.fix_missing_locations(tree)
    ns=dict(frozen.condition_population.__globals__)
    exec(compile(tree,'<coopmwm-causal-moving-clock>','exec'),ns)
    state=dict(present=torch.as_tensor(raw['mask'][-1:]),is_cav=torch.as_tensor(raw['is_cav'][None]))
    begin=time.monotonic()
    fitted,receipt=ns['condition_population'](model._wmphys_fitted_state,
        model._wmphys_motion_fit,model._wmphys_shrinkage_policy,
        dict(t0=record['time_s'],seed=record['world_id'],state=state),raw)
    receipt.update(history_frames=len(raw['time_s']),history_start_s=float(raw['time_s'][0]),
        history_end_s=float(raw['time_s'][-1]),conditioning_wall_s=time.monotonic()-begin,
        observed_only=True,history_encoder=True)
    return fitted,raw,receipt


def encoded_history(model, raw, initial):
    """Use the frozen feature builder in the observed slot order, mask all entrants."""
    import numpy as np
    import torch
    from wms.data.history import long_history_features
    times=raw['time_s']
    if len(times)<2:
        return {}
    dt=float(times[1]-times[0])
    if not np.allclose(np.diff(times),dt,rtol=0,atol=1e-8):
        raise ValueError('history encoder requires genuinely uniform observed ticks')
    kin=torch.as_tensor(raw['kin']);mask=torch.as_tensor(raw['mask'])
    features,valid=long_history_features(kin,mask,torch.as_tensor(raw['target_mask']),mask[-1],
        history_s=60.,history_stride_s=max(1.,dt),dt_s=dt)
    n=raw['current_slots'];capacity=initial['present'].shape[1]
    encoder=getattr(model,'driver_encoder',None)
    width=9 if getattr(encoder,'long_history',False) else 6
    history=torch.zeros((1,capacity,features.shape[1],width),device=initial['present'].device)
    history_mask=torch.zeros(history.shape[:-1],dtype=torch.bool,device=history.device)
    history[0,:n]=features[:n,:,:width].to(history)
    history_mask[0,:n]=valid[:n].to(history.device)
    return dict(driver_history=history.expand(initial['present'].shape[0],-1,-1,-1).clone(),
        driver_history_mask=history_mask.expand(initial['present'].shape[0],-1,-1).clone())


class Proxy:
    def __init__(self, base, **overrides):
        self.base=base;self.__dict__.update(overrides)
    def __getattr__(self,name):return getattr(self.base,name)


@contextmanager
def adapter(rt, record):
    from tools.wmteach import marginal
    result=condition(rt.model,record)
    if result is None:
        yield
        return
    fitted,raw,receipt=result
    original=marginal.plans
    def plans(runtime,initial,kwargs,**kw):
        source_world=record['world_id']
        drivers={}
        for key,value in fitted.posterior['drivers'].items():
            world,vid=map(int,key.split(':'))
            if world!=source_world:raise ValueError('posterior world mismatch')
            for row in set(kw['row_keys']):drivers[int(row),vid]=value
        base=rt.model._wmphys_fitted_state
        state_proxy=Proxy(base,with_motion_posteriors=lambda unused,*,t0_s:base.with_motion_posteriors(drivers,t0_s=t0_s))
        model=Proxy(runtime.model,_wmphys_fitted_state=state_proxy)
        local=SimpleNamespace(**vars(runtime));local.model=model
        kwargs=dict(kwargs,**encoded_history(runtime.model,raw,initial))
        output=original(local,initial,kwargs,**kw)
        for plan in output:
            plan.receipt.update(receipt,observation_contract='observable_rolling60s_per_driver',
                causal_motion_conditioning=True,saved_ID_lookup=False,
                controller_memory='cold; driver history warm',source_counter_conditioning=False)
        return output
    marginal.plans=plans
    try:yield
    finally:marginal.plans=original
