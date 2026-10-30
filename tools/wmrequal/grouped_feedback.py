"""Batch identical frozen entrant policies across futures; independent row memory.

Uses the original bridge/TimelineControllers. No action approximation or policy
change; only removes RowControllers' one-Python-callback-per-future dispatch.
"""
import torch


class Grouped:
    def __init__(self,groups):
        self.groups=groups

    def subset(self,state,indices):
        batch=state['s_m'].shape[0]
        return {k:v[indices] if torch.is_tensor(v) and v.ndim and v.shape[0]==batch else v
                for k,v in state.items()}

    def __call__(self,state):
        from wms.model.cav import WM_MERGE_KEY
        result=state['s_m'].new_zeros((*state['s_m'].shape,4))
        delegated=torch.zeros_like(state['present'])
        for indices,callback in self.groups:
            st=self.subset(state,indices)
            result[indices]=callback(st)
            delegated[indices]=st.get(WM_MERGE_KEY,torch.zeros_like(st['present']))
        state[WM_MERGE_KEY]=delegated
        return result

    def native_lane_mask(self,state):
        result=torch.zeros_like(state['present'])
        for indices,callback in self.groups:
            st=self.subset(state,indices)
            fn=getattr(callback,'native_lane_mask',None)
            if fn is not None:
                result[indices]=fn(st)
        return result


def feedback(specs,arm_names,model,timeline,original_slots,device,*,branches=None):
    from wms.control.entrant import policy_feedback,inferred_timeline
    from wms.train.cav_drive import TimelineControllers
    from labels import safe
    import hashlib
    groups=[]
    for arm in sorted(set(arm_names)):
        indices=[i for i,a in enumerate(arm_names) if a==arm]
        if branches and any((branches[i] or {}).get('policy_params') for i in indices):
            raise ValueError('grouped feedback supports the registered E3/PANEL120 policies only')
        if arm in specs:
            spec=dict(specs[arm]);path=safe(spec.pop('checkpoint'))
            expected=spec.pop('sha256')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
            callback=policy_feedback(path,device=device,**spec)
        else:
            row={k:v[indices] if torch.is_tensor(v) else v for k,v in timeline.items()}
            row['controller']=inferred_timeline(row,original_slots).expand_as(row['controller'])
            callback=TimelineControllers(model,row)
        groups.append((indices,callback))
    return Grouped(groups)
