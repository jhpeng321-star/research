"""One registered five-branch TRAIN decision and live-continuation smoke."""
import copy
import json
from pathlib import Path


def controller_class(original, t0, proposal):
    from .acceleration import Controller
    from .exact_grade import fast_forward
    by_time={int(s['time_s']):s for s in original['steps']}

    class SmokeController(Controller):
        def __init__(self,policy,generator,deterministic=False,guide=None,beta=1.,replay=None,branch=None):
            def cached(rows,commands,now,sample):
                if int(now)!=t0:return None
                # One query draw, as in the original B guide, before trust.
                import torch
                torch.rand((),generator=generator)
                p=copy.deepcopy(proposal)
                for name in ('quota_reservation','random_match','parallel_queries'):
                    p.pop(name,None)
                p['behaviour']=dict(route='wm',sample_probability=1.,mixture_mass=1.,
                    eligible_query=True,regime=p.get('regime','ordinary'),executed=False,
                    proposal_accepted=True,option=p['option'])
                return p
            cached.prime=lambda frames:None
            super().__init__(policy,generator,False,cached,1.,None,None)
            fast_forward(generator,[by_time[t] for t in range(300,t0)],original['arm'])

        def apply(self,rows,commands,now):
            t=int(now)
            if t<t0:
                self.replay={str(t):by_time[t]['sample']}
                try:return super().apply(rows,commands,now)
                finally:self.replay=None
            if t==t0:
                rec=by_time[t]['sample']
                self.memory=dict(zip(rec['context']['members'],[float(v[0]) for v in rec['previous_noise']]))
                self.active_plans=[dict(plan=copy.deepcopy(c['plan']),start_s=t0-c['elapsed_s'])
                                   for c in rec.get('committed_plans',[])]
            result=super().apply(rows,commands,now)
            if t==t0 and self.events[-1]['sample']['own_action']!=by_time[t]['sample']['own_action']:
                raise ValueError('smoke own-action CRN does not match the registered original decision')
            return result
    return SmokeController


def run(root):
    import torch
    from . import accel_plant
    from .accel_storage import read,reserve
    from .accel_learner import load
    from .protocol import sha
    from .warm_protocol import install_accounting
    from .warm_spool import immutable
    from .sumo_verify import check_live_continuation
    r=Path(root);v=read(r/'VERIFICATION_RUNTIME_BINDING.json')
    assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
    assert not (r/'VERIFICATION_SMOKE_STARTED.json').exists(),'adopt original smoke; never repeat physics'
    for name,h in v['cpu']['source_hashes'].items():assert sha(Path(v['source_root'])/name)==h
    spec=v['smoke'];original=read(r/'jobs'/spec['original_episode']/'RESULT.json')
    step=next(s for s in original['steps'] if s['time_s']==spec['time_s'])
    proposal=step['proposal'];assert len(proposal['topk'])==4
    assert sha(r/'jobs'/spec['original_episode']/'RESULT.json')==spec['original_result_sha256']
    policy,_,_,payload=load(r/'B_checkpoint_002.pt');assert payload['update']==2
    b=read(r/'EXACT_OPERATIONAL_BINDING.json');install_accounting(r)
    job=dict(id=spec['id'],path=str(r/'jobs'/spec['id']),world=original['world'],role='train',arm='B',
        weights=policy.state_dict(),runtime=b['runtime'],checkpoint=b['checkpoint'],
        noise_seed=spec['noise_seed'],deterministic=False,guided=False,sim_s=spec['stop'],stop=spec['stop'],
        warm_history=True,accel_bounds=b['accel_bounds'],queue_root=str(r),gpu_deadline=b['gpu_deadline'],
        branch=dict(time_s=spec['time_s'],takeover_s=spec['time_s'],revert=[],
                    prefixes={str(t):original['prefixes'][str(t)] for t in range(300,spec['time_s']+1)}),
        sumo_verify=dict(root=str(r/'sumo_verification_smoke'),run_root=str(r),phase='sumo_verification',k=4,workers=5,timeout_s=3600))
    reserve(r,[job],'sumo_verification')
    immutable(r/'VERIFICATION_SMOKE_STARTED.json',dict(id=job['id'],spec=spec))
    old=accel_plant.Controller;accel_plant.Controller=controller_class(original,spec['time_s'],proposal)
    try:live=accel_plant.episode(job)
    finally:accel_plant.Controller=old
    receipts=list((r/'sumo_verification_smoke'/'verifications').glob('*.json'))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text());assert len(receipt['branches'])==5
    assert all(x['status']=='graded' and (x.get('audit') or {}).get('crn_verified_t0') for x in receipt['branches'])
    passed=check_live_continuation(receipt,live)
    immutable(r/'VERIFICATION_SMOKE_PASSED.json',dict(passed,branch_count=5,new_sim_s=spec['quote_sim_s'],
        source_root=v['source_root'],amendment_sha256=v['amendment_sha256']))
    return passed


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    print(json.dumps(run(p.parse_args().root)))
