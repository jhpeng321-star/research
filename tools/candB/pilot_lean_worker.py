"""Read-only executed telemetry transported from owned parallel rollout workers."""
import atexit
import json
import os
from pathlib import Path
import sys

_state = dict(totals=None, minima=None, registered=False)


def flush():
    if _state['totals'] is None:
        return
    root, index = Path(sys.argv[-2]), sys.argv[-1]
    if 'EXPLORATORY-LEAN' not in str(root) or root.parent.name != 'future_workers' or not index.isdigit():
        raise ValueError('worker telemetry requires its owned lean pool directory')
    count, nonzero, brake = _state['totals'].cpu().tolist()
    minima = _state['minima'].cpu().tolist()
    with (root/index/'executed.json').open('x') as stream:
        json.dump(dict(label='EXPLORATORY-LEAN',count=count,nonzero=nonzero,braking_sum=brake,
            fraction_nonzero=nonzero/max(1,count),mean_braking=brake/max(1,count),
            min_delta_mps2=minima[0],min_braking_magnitude_mps2=minima[1],
            min_executed_acceleration_mps2=minima[2]),stream,indent=1)


def observe(frame, result, act_mask):
    if 'WMS_SPEED_FUTURE_PARENT_PID' not in os.environ:
        return
    import torch
    from wms.control.execution import executed_residual
    if not _state['registered']:
        atexit.register(flush)
        _state['registered']=True
    with torch.no_grad():
        zeros=torch.zeros_like(frame['s_m'])
        reference=executed_residual(frame,zeros,zeros,residual_reference='ctrl0')
        reference=((frame['v_mps']+.5*reference).clamp_min(0.)-frame['v_mps'])/.5
        mask=frame['present'] & frame['is_cav'] & (frame['lane']>=0)
        if act_mask is not None:
            mask=mask & act_mask
        delta=(result[...,3]-reference)[mask]
        if delta.numel():
            totals=torch.stack((delta.new_tensor(delta.numel()),(delta!=0).sum(),(-delta).clamp_min(0).sum())).double()
            minima=torch.stack((delta.min(),(-delta).clamp_min(0).min(),result[...,3][mask].min()))
            _state['totals']=totals if _state['totals'] is None else _state['totals']+totals
            _state['minima']=minima if _state['minima'] is None else torch.minimum(_state['minima'],minima)


def aggregate(pools):
    rows=[w['executed'] for pool in pools if pool['role']=='unperturbed_forward'
          for w in pool['workers'] if w.get('executed')]
    if not rows:
        return None
    count=sum(r['count'] for r in rows)
    return dict(scope='unperturbed K8 training forwards; FINAL acting-CAV physics ticks across counterfactual rows',
        count=count,fraction_nonzero=sum(r['nonzero'] for r in rows)/max(1,count),
        mean_braking=sum(r['braking_sum'] for r in rows)/max(1,count),
        **{k:min(r[k] for r in rows) for k in
           ('min_delta_mps2','min_braking_magnitude_mps2','min_executed_acceleration_mps2')})
