"""CPU gates: real parity boundary, correlated likelihood and committed profiles."""
import copy
import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
from unittest.mock import patch
from tools.coopnative.test_cpu import rows
from .acceleration import Policy, Controller, requests, encode, PHI
from .plans import acceleration_at, validate, thompson, population
from .plan_wm import profile_table, profile_apply


class AccelerationCPU(unittest.TestCase):
    def test_acceleration_modules_import_and_long_credit(self):
        from . import accel_runner,accel_plant,plan_bridge,plan_service,plan_client,accel_deploy,accel_diagnostic
        from .accel_learner import GAMMA,LAMBDA
        self.assertGreaterEqual(-1/np.log(GAMMA*LAMBDA),300.)

    def test_viability_has_a_fixed_pre_final_endpoint(self):
        from .accel_runner import update_limit
        self.assertEqual(update_limit(dict(viability_updates=4)),4)
        self.assertEqual(update_limit({}),24)
        with self.assertRaises(ValueError):update_limit(dict(viability_updates=3))

    def test_R_sampling_rate_excludes_committed_steps_and_unavailable_windows(self):
        from .accel_runner import routing
        eligible=dict(route='wm',sample_probability=.95,eligible_query=True,regime='ordinary')
        continuation=dict(route='plan',executed=True,plan_start=False,regime='ordinary')
        unavailable=dict(route='policy',sample_probability=0.,regime='ordinary')
        steps=[dict(sample=dict(behaviour=b)) for b in [eligible]+[continuation]*59+[unavailable]*10]
        result=routing([dict(steps=steps)])
        self.assertEqual(result['sampling']['ordinary'],.95)
        self.assertEqual(result['eligible_query_windows'],1)

    def test_random_guide_returns_to_policy_route_when_matched_quota_is_exhausted(self):
        from .plan_client import guide
        from .storage import write
        g=torch.Generator().manual_seed(5);sample={}
        with tempfile.TemporaryDirectory() as tmp,patch('tools.coopmwm.plan_client.group',return_value=['a','b']),patch('tools.wmteach.labels.actions',return_value=[dict(id='O1',members=['a','b'])]):
            root=Path(tmp);q=root/'quota.json'
            write(q,dict(sampling=dict(ordinary=.95,**{'merge-dense':.95}),tickets=[]))
            fn=guide(dict(queue_root=tmp,accel_bounds=[-3.,2.6],arm='R',update=1,random_quota=str(q)),g)
            self.assertIsNone(fn(rows(),{},300,sample))
            self.assertEqual(sample['behaviour']['route'],'policy')
            self.assertEqual(sample['behaviour']['sample_probability'],0.)

    def test_control_observes_same_eligible_windows_without_queries_or_rng_change(self):
        from .plan_client import guide
        g=torch.Generator().manual_seed(5);sample={};before=g.get_state().clone()
        with tempfile.TemporaryDirectory() as tmp,patch('tools.coopmwm.plan_client.group',return_value=['a','b']),patch('tools.wmteach.labels.actions',return_value=[dict(id='O1',members=['a','b'])]):
            fn=guide(dict(queue_root=tmp,accel_bounds=[-3.,2.6],arm='A'),g)
            self.assertIsNone(fn(rows(),{},300,sample))
            self.assertEqual(sample['behaviour']['sample_probability'],0.)
            self.assertEqual(sample['behaviour']['route'],'policy')
            self.assertTrue(torch.equal(before,g.get_state()))
            self.assertFalse((Path(tmp)/'requests').exists())

    def test_viability_runner_pauses_all_arms_without_touching_final_panel(self):
        from . import accel_runner as runner
        from .accel_storage import read,write
        waves=[]
        def wave(root,b,policies,ws,role,ident,phase,*args):
            self.assertNotEqual(role,'final_test');waves.append((ident,tuple(policies)))
            return [dict(arm=a,steps=[dict(time_s=700,reward=-1.,sample=dict(value=0.)),
                dict(time_s=701,reward=-2.,sample=dict(value=0.))]) for a in policies]
        route=dict(executed=10,durations=[dict(regime='ordinary',duration_s=20)]*10,
            sampling=dict(ordinary=.5,**{'merge-dense':.5}))
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'verify',return_value=dict(viability_updates=4,N_exec=10,eta_primary_hours=None)),patch.object(runner,'wave',side_effect=wave),patch.object(runner,'routing',return_value=route),patch.object(runner,'evidence',return_value=[]),patch.object(runner,'viability',return_value={}),patch.object(runner,'update',return_value=[]),patch.object(runner,'save'),patch.object(runner,'panel_report'),patch.object(runner,'final',side_effect=AssertionError('sealed panel forbidden during viability')):
            root=Path(tmp);write(root/'SMOKE.json',dict(pass_=True));write(root/'MAPPO_CHECK.json',dict(records=[]))
            runner.run(root)
            self.assertEqual(read(root/'VIABILITY_PAUSED.json')['update'],4)
            self.assertEqual(len(list(root.glob('*_UPDATE_*.json'))),12)
            self.assertFalse((root/'FINAL_TEST_STARTED.json').exists())
            self.assertFalse((root/'COMPLETE.json').exists())
            self.assertTrue(any(arms==('R',) for _,arms in waves))

    def test_lossless_episode_storage_digest_and_no_overwrite(self):
        from .accel_storage import read,write
        from tools.coopnative.storage import read as plain
        value=dict(steps=[dict(reward=-3.125,action=[0.,-1.],context='repeat'*100)]*100)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'RESULT.json';write(p,value)
            self.assertEqual(read(p),value)
            self.assertEqual(plain(p)['encoding'],'coopmwm-lossless-gzip-v1')
            self.assertLess((p.parent/'RESULT.json.gz').stat().st_size,plain(p)['uncompressed_bytes'])
            with self.assertRaises(FileExistsError):write(p,value)
            self.assertEqual(read(p),value)

    def test_diagnostic_pairing_uses_actual_profile_and_mappo_background(self):
        from .accel_diagnostic import evaluate,agreement
        from types import SimpleNamespace
        plan=dict(id='p',members=['a','b'],duration_s=60,knots=[[-1.]*7,[1.]*7])
        r=dict(plan=plan,policy_residuals={'a':1.},state_id='fixture')
        def fake(rt,record,acts,K,H,**kw):
            self.assertEqual(record['policy_residuals'],{})
            self.assertEqual([len(a['members']) for a in acts],[2,1,1,0])
            self.assertEqual(kw['horizons'],(60,120,300,600))
            return {str(h):{a['id']:[dict(draw_id=str(k),team_delay=v+k) for k in range(8)]
                for a,v in zip(acts,[50.,80.,70.,110.])} for h in kw['horizons']}
        with patch('tools.coopmwm.plan_wm.evaluate',fake):result=evaluate(SimpleNamespace(),r)
        self.assertEqual(result['credits']['300'],dict(D_i=[30.,20.],D_G=60.,C_G=10.))
        self.assertEqual(result['noise']['300']['paired_credit_variance'],0.)
        self.assertGreater(result['noise']['300']['team_return_advantage_variance'],0.)
        rs=[dict(world=dict(seed=i),wm=result['credits'],exact=result['credits']) for i in range(3)]
        summary=agreement(rs,100)
        self.assertEqual(summary['horizons']['300']['scale'],1.)
        self.assertEqual(summary['weights']['300'],.25)
        self.assertFalse(agreement(rs[:1],10)['horizons']['300']['pass_'])

    def test_one_second_episode_orchestration_and_sumo_only_reward(self):
        from .accel_plant import episode
        class Backend:
            def __init__(self,*args):
                self.actual=300;self.totals=dict(team_delay_vs=0.,collisions=0.,teleports=0.,ttc_below_3_vs=0.,hard_brake=0.)
            def fingerprint(self):return dict(clock=self.actual)
            def close(self):pass
        def tick(b,controller,trace):
            controller.apply(rows(),{r['vid']:0. for r in rows() if r['is_cav']},b.actual)
            b.actual+=1;b.totals['team_delay_vs']+=7.
            return []
        from .accel_protocol import worlds
        with tempfile.TemporaryDirectory() as tmp,patch('tools.leverprobe.backend.setup'),patch('tools.leverprobe.backend.Backend',Backend),patch('tools.coopfree.backend.tick',tick),patch('tools.coopmwm.accel_plant.install_seed_scope'),patch('tools.coopmwm.accel_plant.guard'):
            e=episode(dict(path=tmp,id='fixture',sim_s=1200,runtime='',checkpoint='',world=worlds()[0],role='train',
                weights=Policy().state_dict(),noise_seed=1,deterministic=True,arm='A'))
            self.assertEqual(len(e['steps']),900)
            self.assertEqual({s['reward'] for s in e['steps']},{-7.})
            self.assertEqual(e['metrics']['600']['team_delay_vs'],4200.)
            self.assertEqual(e['metrics']['900']['team_delay_vs'],6300.)
            self.assertEqual(e['actual_sim_s'],1200)

    def test_duplicate_cem_profiles_keep_complete_separate_k8_futures(self):
        from .plans import search,SEARCH
        seed=dict(id='O1',members=['a','b'],knots=[[0.]*7]*2,duration_s=20,family='O1',origin='inherited')
        record=dict(rows=[],group=['a','b'],commands={},policy_residuals={})
        def fake(acts,k,h):
            self.assertEqual(len({a['id'] for a in acts}),len(acts))
            return {a['id']:[dict(team_delay=100. if a['id'] in ('policy','MAPPO') else 0.) for _ in range(k)] for a in acts}
        with patch('tools.coopmwm.plans.family_seeds',return_value=[seed]),patch('tools.coopmwm.plans.population',side_effect=lambda r,g,b,s,n,*args:[dict(seed,id='same') for _ in range(n)]):
            result=search(record,(-3.,2.6),fake)
        self.assertTrue(result['accepted']);self.assertEqual(result['candidates_searched'],SEARCH['population']*SEARCH['iterations'])

    def test_checkpoint_restores_acceleration_optimizer_and_AR_generator(self):
        from .accel_learner import save,load,update
        p=Policy();opt=torch.optim.Adam(p.parameters(),lr=3e-4);g=torch.Generator().manual_seed(1)
        sample,_=p.sample(encode(rows()),g)
        ep=dict(steps=[dict(sample=sample,reward=-123.)])
        update(p,opt,[ep],g,epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'accel.pt';save(path,p,opt,g,1,811640)
            restored,other,rg,_=load(path)
            update(p,opt,[ep],g,epochs=1);update(restored,other,[ep],rg,epochs=1)
            for a,b in zip(p.parameters(),restored.parameters()):self.assertTrue(torch.equal(a,b))

    def test_singular_plan_ratio_never_invents_gaussian_mass(self):
        from .accel_learner import ratio,imitation
        p=Policy();s,_=p.sample(encode(rows()),torch.Generator().manual_seed(9))
        lp,_=p.score(s);s['behaviour']=dict(route='wm',executed=True,beta=1.,proposal_members=s['context']['members'])
        self.assertEqual(float(ratio(lp,s)),0.)
        s['action']=[[1.] for _ in s['action']]
        loss=imitation(p,s);loss.backward()
        self.assertGreater(float(p.actor[-1].bias.grad.abs().sum()),0.)
        s['behaviour']=dict(route='policy',sample_probability=.25)
        self.assertAlmostEqual(float(ratio(lp,s)),1/.75,places=5)

    def test_real_frozen_mappo_acceleration_executor_parity(self):
        import os
        from wms.control.policy import load_policy_fn
        from tools.coopnative.test_cpu import state,env
        from tools.coopfree.execution import inject
        a,b=[load_policy_fn(os.environ['COOPMWM_MAPPO'],device='cpu') for _ in range(2)]
        a.reset();b.reset();sensor=rows();e=env(sensor)
        s=state(sensor);s['a_mps2']=torch.zeros_like(s['v_mps'])
        controller=Controller(Policy(),torch.Generator().manual_seed(1),True)
        for t in range(300,310):
            expected,_=a(s,with_lanes=True)
            with inject(b,e,sensor,controller,t,[]):actual,_=b(s,with_lanes=True)
            self.assertEqual(expected.tobytes(),actual.tobytes())
        self.assertEqual(len(controller.events),10)

    def test_96_panel_roles_dual_seed_guards_and_full_budget(self):
        from .accel_protocol import worlds,budget,guard,install_seed_scope
        panel=worlds('final_test');self.assertEqual(len(panel),96)
        self.assertEqual(len({w['seed'] for w in panel}),96)
        for c in range(12):self.assertEqual(sum(w['cell']==c for w in panel),8)
        install_seed_scope()
        for w in panel:guard(w,'final_test')
        with self.assertRaises(ValueError):guard(panel[-1],'train')
        for seed in (964000,962001,977001):
            with self.assertRaises(ValueError):guard(dict(panel[0],seed=seed),'final_test')
        b=budget();self.assertEqual(b['allocated']+b['unallocated'],6000000)
        self.assertGreater(b['with_seed2_total'],6000000)

    def test_zero_update_is_exact_mappo_before_unchanged_safety(self):
        sensor=rows();commands={r['vid']:2.59707 for r in sensor if r['is_cav']}
        p=Policy();s,_=p.sample(encode(sensor),torch.Generator().manual_seed(1),True)
        self.assertEqual(requests(s,commands),{})
        self.assertTrue(torch.equal(p.broadcast[-1].weight,torch.zeros_like(p.broadcast[-1].weight)))

    def test_conditional_ar_density_and_no_leakage(self):
        p=Policy();g=torch.Generator().manual_seed(20);context=encode(rows())
        first,m=p.sample(context,g);second,_=p.sample(context,g,memory=m)
        self.assertAlmostEqual(float(p.score(second)[0]),second['logp'],places=5)
        self.assertTrue(any(x[0]!=0 for x in second['previous_noise']))
        with self.assertRaises(ValueError):encode([dict(rows()[0],world_id=1)])
        self.assertGreater(1/(-np.log(PHI)),4.99)

    def test_profiles_change_futures_and_release_at_duration(self):
        plan=dict(id='x',members=['a','b'],duration_s=60,knots=[[-2.]*7,[1.]*7])
        validate(plan,(-3.,2.6));table=profile_table([plan],['a','b'],'cpu')
        x=torch.zeros((1,2));v=torch.tensor([[20.,20.]]);baseline=v.clone();position=v*0
        for tick in range(120):
            a=profile_apply(x,tick,table,(-3.,2.6));v=(v+.5*a).clamp_min(0);position+=.5*v
        self.assertGreater(float((v-baseline).abs().max()),1.)
        self.assertNotEqual(float(position[0,0]),float(position[0,1]))
        self.assertEqual(acceleration_at(plan,60),{})
        self.assertTrue(torch.equal(profile_apply(x,120,table,(-3.,2.6)),x))

    def test_policy_residual_and_plan_absolute_raw_are_distinct(self):
        acts=[dict(id='policy',window=60,residuals={'a':.5}),dict(id='MAPPO',window=0),
            dict(id='p',members=['a','b'],knots=[[-1.]*7,[2.]*7],duration_s=20)]
        out=profile_apply(torch.full((3,2),.7),0,profile_table(acts,['a','b'],'cpu'),(-3.,2.6))
        self.assertTrue(torch.allclose(out,torch.tensor([[1.2,.7],[.7,.7],[-1.,2.]])))

    def test_unchanged_cavs_and_postcommit_follow_policy_continuation(self):
        plan=dict(id='p',members=['a','b'],duration_s=20,knots=[[-1.]*7,[2.]*7],background_residuals={'a':.1,'b':.2,'c':.3})
        table=profile_table([plan],['a','b','c'],'cpu');raw=torch.full((1,3),.5)
        self.assertTrue(torch.allclose(profile_apply(raw,0,table,(-3.,2.6)),torch.tensor([[-1.,2.,.8]])))
        self.assertTrue(torch.allclose(profile_apply(raw,40,table,(-3.,2.6)),torch.tensor([[.6,.7,.8]])))
        self.assertTrue(torch.allclose(profile_apply(raw,598,table,(-3.,2.6)),torch.tensor([[.6,.7,.8]])))

    def test_thompson_uses_same_crn_indices_and_beats_both_comparators(self):
        plans=[dict(id='p',family='O1',origin='sampled')]
        fs={a:[dict(team_delay=x) for x in values] for a,values in
            dict(policy=[20.]*8,MAPPO=[10.]*8,p=[11.]*8).items()}
        r=thompson({},plans,fs,np.random.default_rng(1));self.assertFalse(r['accepted'])
        fs['p']=[dict(team_delay=0.)]*8
        r=thompson({},plans,fs,np.random.default_rng(1));self.assertTrue(r['accepted'])
        self.assertEqual(r['predicted_rank'],1);self.assertEqual(len(r['bootstrap_indices']),8)

    def test_plan_is_committed_and_replay_is_immutable(self):
        sensor=rows();members=[r['vid'] for r in sensor if r['is_cav']][:2]
        self.assertEqual(len(members),2)
        plan=dict(id='p',members=members,duration_s=20,knots=[[-1.]*7,[1.]*7],family='O1')
        calls=[]
        def guide(rs,cmd,t,s):
            calls.append(t)
            return dict(accepted=True,plan=plan,option='O1',behaviour=dict(route='wm',regime='ordinary',executed=False))
        c=Controller(Policy(),torch.Generator().manual_seed(2),guide=guide)
        cmd={r['vid']:0. for r in sensor if r['is_cav']}
        for t in range(300,320):
            result=c.apply(sensor,cmd,t)
            self.assertEqual(result[members[0]]['requested_raw'],-1.)
            b=c.events[-1]['sample']['behaviour']
            self.assertTrue(b['executed'])
            self.assertEqual(b['route'],'wm' if t==300 else 'plan')
            from .accel_learner import ratio
            self.assertEqual(float(ratio(torch.tensor(0.),c.events[-1]['sample'])),0.)
        self.assertEqual(calls,[300])
        saved=copy.deepcopy(c.events[0]['sample']);snapshot=copy.deepcopy(saved)
        r=Controller(Policy(),None,replay={'300':saved})
        r.apply(sensor,cmd,300);self.assertEqual(saved,snapshot)

if __name__=='__main__':unittest.main()
