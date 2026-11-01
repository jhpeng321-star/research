"""Integration checks for long MPPI execution and the exact transfer hot-swap."""
import unittest
from unittest.mock import patch
import torch
from .plan_wm import profile_table
from .speed2_transfers import vector_table


class LongProfileTests(unittest.TestCase):
    def test_MPPI_trust_columns_never_alias_the_own_policy_controls(self):
        from .exploration import trust
        records=[]
        for world in range(4):
            for regime,treated,control in (('ordinary',30.,10.),('merge-dense',-10.,10.)):
                records.extend([dict(world=world,executed=True,q=.5,p=.5,regime=regime,option='MPPI',route='wm',advantage=treated),
                    dict(world=world,executed=False,q=0.,p=.5,regime=regime,option=None,route='policy',advantage=control)])
        result=trust(1.,records,bootstraps=200)
        self.assertEqual(result['mean'],0.)
        self.assertGreater(result['buckets']['MPPI|ordinary']['mean'],0.)
        self.assertLess(result['buckets']['MPPI|merge-dense']['mean'],0.)
        self.assertEqual(result['buckets']['MPPI|merge-dense']['beta'],0.)
        self.assertEqual(result['buckets']['MPPI|ordinary']['beta'],1.)

    def test_R_uses_the_same_policy_centre_and_never_scores_candidates(self):
        import tempfile
        from pathlib import Path
        from .plan_client import guide
        from .storage import write
        from .acceleration import encode
        from .mppi_search import sample_unscored
        observed=[dict(vid=v,s_m=s,v_mps=10.,lane=lane,is_cav=True,length_m=5.)
            for v,s,lane in [('a',-20.,1),('b',-40.,2)]]
        centre=[[i/1000 for i in range(300)],[-i/1000 for i in range(300)]]
        sample=dict(context=encode(observed),own_action=[[0.],[0.]])
        with tempfile.TemporaryDirectory() as tmp,patch('tools.coopmwm.centre_client.trajectory',return_value=(centre,dict(K=1,H=300))) as rpc,patch(
                'tools.coopmwm.plan_client.sample_unscored',wraps=sample_unscored) as sampled,patch(
                'tools.coopmwm.mppi_search.search',side_effect=AssertionError('R candidate WM scoring forbidden')):
            q=Path(tmp)/'quota.json'
            write(q,dict(sampling={'ordinary':.95,'merge-dense':.95},tickets=[dict(id=0,regime='ordinary',group_size=2,duration_s=300)]))
            fn=guide(dict(id='R1',world=dict(seed=986000),queue_root=tmp,accel_bounds=[-3.,2.6],arm='R',update=1,
                random_quota=str(q),gpu_deadline=1e12,policy_state={}),torch.Generator().manual_seed(5))
            fn.observe(observed,300)
            result=fn(observed,{'a':2.6,'b':2.6},300,sample)
            self.assertTrue(result['accepted']);self.assertEqual(sampled.call_args.args[1],centre)
            self.assertEqual(rpc.call_count,1);self.assertEqual(result['plan']['duration_s'],300)
            self.assertNotIn('search_learning',rpc.call_args.args[1])

    def test_centre_service_calls_only_the_existing_comparator_trajectory_extractor(self):
        from .centre_service import result
        record=dict(state_id='r',rows=[],group=[])
        with patch('tools.coopmwm.mppi_runtime.imagined_centres',return_value=([[.1]*300],[[0.]*300],dict(K=1,H=300))) as centre,patch(
                'tools.coopmwm.mppi_search.search',side_effect=AssertionError('no scoring')):
            value=result(None,record,None)
        self.assertEqual(centre.call_count,1)
        self.assertEqual(value['candidate_scores'],0)
        self.assertEqual(value['policy_traj'],[[.1]*300])

    def test_imagined_nonzero_actor_matches_observable_plant_actor(self):
        from .acceleration import Policy,encode
        from .policy_wm import actor_state,load_actor,residuals
        from tools.coopnative.test_cpu import rows
        observed=rows();torch.manual_seed(12);policy=Policy()
        with torch.no_grad():policy.actor[-1].weight.normal_(0,.1)
        state={k:torch.tensor([[r[k] for r in observed]]) for k in ('s_m','v_mps','lane','is_cav')}
        state['present']=torch.ones_like(state['is_cav'],dtype=torch.bool)
        ids=[r['vid'] for r in observed]
        expected,_=policy(encode(observed))
        actual=residuals(load_actor(actor_state(policy),'cpu'),state,ids)
        positions=[ids.index(v) for v in encode(observed)['members']]
        self.assertTrue(torch.allclose(actual[0,positions],expected[:,0],atol=1e-6,rtol=0))

    def test_A_continuation_keeps_u4_resume_and_fixed_u48_without_final_access(self):
        import tempfile
        from pathlib import Path
        from . import warm_runner as runner
        from .storage import read
        from .acceleration import Policy
        policy=Policy();rng=torch.Generator().manual_seed(1)
        binding=dict(accel_bounds=[-3.,2.6],gpu_deadline=1e12)
        eps=[dict(steps=[dict(time_s=700,reward=-1.,sample=dict(value=0.)),
                        dict(time_s=701,reward=-2.,sample=dict(value=0.))])]
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'verify',return_value=binding),patch.object(
                runner,'load',return_value=(policy,None,rng,dict(update=4))),patch.object(
                runner,'wave',return_value=eps),patch.object(runner,'routing',return_value={}),patch.object(
                runner,'evidence',return_value=[]),patch.object(runner,'viability',return_value={}),patch.object(
                runner,'update',return_value=[]),patch.object(runner,'save') as saved,patch.object(
                runner,'charged',return_value=988040),patch.object(runner,'check') as checked,patch.object(runner,'panel_report'):
            root=Path(tmp)
            from .storage import write
            write(root/'MAPPO_DEV.json',dict(records=[]))
            runner.run_arm(root,'A')
            self.assertEqual(read(root/'WARM_A_STARTED.json')['first_update'],5)
            self.assertEqual(read(root/'WARM_A_PAUSED.json')['update'],48)
            self.assertEqual(saved.call_count,44)
            self.assertEqual([call.args[-1] for call in checked.call_args_list],list(range(8,49,4)))
            self.assertFalse((root/'FINAL_TEST_STARTED.json').exists())

    def test_warmup_observation_hook_keeps_real_ticks_and_no_duplicate_terminal(self):
        from .warm_plant import observed_backend
        from types import SimpleNamespace
        class Base:
            def __init__(self):pass
            def observe(self,env):pass
        b=observed_backend(Base)()
        row=dict(vid='a',s_m=1.,v_mps=2.,lane=0,is_cav=True,length_m=5.)
        with patch('tools.coopfree.backend.observations',return_value=[row]):
            for i in range(478,602):b.observe(SimpleNamespace(now=i*.5))
            b.observe(SimpleNamespace(now=300.))
        self.assertEqual([f['time_s'] for f in b.causal_history.snapshot(300)],list(range(240,301)))

    def test_transfer_table_exact_for_long_profiles_and_dynamic_policy(self):
        actions=[dict(id='p',members=['a','b'],duration_s=300,
            profile=[[.25]*300,[-.5]*300],background_residuals={'a':.1,'b':.2,'c':-.25}),
            dict(id='policy',window=600,residuals={'a':.3}),
            dict(id='MAPPO',window=0,residuals={})]
        for dynamic in (False,True):
            old=profile_table(actions,['a','b','c'],'cpu',420,dynamic)
            new=vector_table(actions,['a','b','c'],'cpu',420,dynamic)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(old,new)))
            self.assertEqual(float(new[0][0,299,0]),.25)
            self.assertEqual(bool(new[1][0,300,0]),not dynamic)
            if not dynamic:self.assertEqual(float(new[0][0,419,0]),float(torch.tensor(.1)))

    def test_member_reversion_preserves_profile_and_background(self):
        from .accel_diagnostic import evaluate
        plan=dict(members=['a','b','c'],duration_s=300,profile=[[.1]*300,[.2]*300,[.3]*300])
        record=dict(plan=plan,policy_residuals={},rows=[])
        captured=[]
        def futures(rt,rec,actions,*args,**kw):
            captured.extend(actions)
            return {}
        with patch('tools.coopmwm.plan_wm.evaluate',futures):evaluate(None,record)
        self.assertEqual(len(captured),5)
        self.assertEqual(captured[2]['members'],['a','c'])
        self.assertEqual(captured[2]['profile'],[plan['profile'][0],plan['profile'][2]])
        self.assertEqual(captured[-1]['profile'],[])


if __name__=='__main__':unittest.main()
