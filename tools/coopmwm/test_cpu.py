"""Physics-forbidden CPU tests for MAPPO parity and member-reversion credit."""
import copy
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import numpy as np
from tools.coopnative.test_cpu import rows,state,env
from tools.coopnative.learner import GAMMA,LAMBDA
from tools.coopfree.execution import inject
from tools.coopnative.policy import targets
from .policy import Policy,Controller,encode,group
from .credit import algebra,paired,ramp,bonus,calibration,multihorizon,multibonus,noise
from .wm import candidates,validate
from .protocol import budget,guard,install_seed_scope,worlds,registry_conflicts


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(31)

    def test_MAPPO_zero_residual_actual_presafety_bitwise(self):
        from wms.control import safety
        from types import SimpleNamespace
        r=rows();e=env(r);raw=torch.tensor([[.13,-.75,1.2,.3]])
        exe=SimpleNamespace(accel_min=-3.,accel_max=1.5)
        ctrl=Controller(Policy(),torch.Generator(),True)
        original=safety.safe_accel;expected=original(e.state(),raw,accel_max=1.5)
        for now in range(300,320):
            audit=[]
            with inject(exe,e,r,ctrl,now,audit):
                actual=safety.safe_accel(e.state(),raw,accel_max=1.5)
            self.assertTrue(torch.equal(expected,actual))
            self.assertEqual(expected.numpy().tobytes(),actual.numpy().tobytes())
            self.assertEqual(audit,[])
        self.assertEqual(len(ctrl.events),2)
        self.assertTrue(torch.equal(Policy().broadcast(torch.zeros(12)),torch.zeros(4)))

    def test_frozen_MAPPO_checkpoint_parity_actual_executor(self):
        """Real d9042cc5 loader, tensor ABI, memory and command comparison, no SUMO."""
        from wms.control.policy import load_policy_fn
        from tools.coopnative.comparator import MAPPO_SHA
        from .protocol import sha
        path=Path(os.environ['COOPMWM_MAPPO'])
        self.assertEqual(sha(path),MAPPO_SHA)
        a,b=[load_policy_fn(str(path),device='cpu') for _ in range(2)]
        a.reset();b.reset()
        from wms.control.safety import safe_accel
        # Executor consumes normal kinematics/history; mirror the production ABI.
        s=state(rows());s['a_mps2']=torch.zeros_like(s['v_mps'])
        e=env(rows());ctrl=Controller(Policy(),torch.Generator(),True)
        for now in range(300,303):
            x,lanes=a(s,with_lanes=True)
            with inject(b,e,rows(),ctrl,now,[]):y,other=b(s,with_lanes=True)
            self.assertIsNone(lanes);self.assertIsNone(other)
            self.assertTrue(np.array_equal(np.isnan(x),np.isnan(y)))
            self.assertEqual(x.tobytes(),y.tobytes())

    def test_reversion_algebra_not_solo_active(self):
        c=algebra(60.,[90.,80.],110.)
        self.assertEqual(c,dict(D_i=[30.,20.],D_G=50.,C_G=0.))
        self.assertEqual(algebra(100.,[95.,98.],80.)['C_G'],-13.)
        with self.assertRaises(ValueError):algebra(1.,[],3.)
        with self.assertRaises(ValueError):algebra(float('nan'),[1.],2.)

    def test_WM_exact_fixture_same_algebra_CRN(self):
        fs=[{k:dict(draw_id=str(i),team_delay=v) for k,v in
             [('joint',60.),('revert:0',90.),('revert:1',80.),('all',115.)]} for i in range(8)]
        self.assertEqual(paired(fs),algebra(60.,[90.,80.],115.))
        fs[0]['all']['draw_id']='other'
        with self.assertRaises(ValueError):paired(fs)

    def test_ramp_fixed_and_anti_gaming(self):
        self.assertEqual([ramp(u) for u in (1,2,5,9,80)],[0.,.125,.5,1.,1.])
        self.assertEqual(bonus(algebra(60.,[90.,80.],115.),1,2.),0.)
        self.assertEqual(bonus(algebra(60.,[90.,80.],115.),9,2.),105.)
        with self.assertRaises(ValueError):bonus({},1,-1.)
        self.assertGreaterEqual(-10/math.log(GAMMA*LAMBDA),300.)

    def test_learned_scale_sign_gate(self):
        rs=[dict(world=dict(cell=i),wm=dict(D_i=[float(i+1),2.],C_G=float(i+3)),
                 exact=dict(D_i=[float(2*i+2),4.],C_G=float(2*i+6))) for i in range(12)]
        c=calibration(rs,100)
        self.assertEqual(c['scale'],2.);self.assertTrue(c['pass_'])
        for r in rs:r['exact']['C_G']*=-1
        self.assertFalse(calibration(rs,100)['pass_'])

    def test_no_leakage_observations_replay_and_group(self):
        r=rows();commands={v['vid']:.2 for v in r if v['is_cav']}
        c=encode(r,commands)
        self.assertEqual(c,encode(list(reversed(r)),commands))
        for k in ('seed','demand','driver_type','exact','world'):
            with self.assertRaises(ValueError):encode([dict(r[0],**{k:1}),*r[1:]],commands)
        record=dict(state_id='cpu',world_id=986000,time_s=300,rows=r,
                    group=['ramp','upstream'],targets={'ramp':dict(speed=8.,tau=1.)})
        validate(record)
        with self.assertRaises(ValueError):validate(dict(record,exact=1))
        actions=candidates(record)
        self.assertEqual(actions[-1]['targets'],{})
        self.assertEqual(actions[1]['targets'],{})

    def test_exact_branch_reverts_entire_window_and_releases(self):
        r=rows();cmd={v['vid']:.2 for v in r if v['is_cav']}
        sample=Policy().sample(encode(r,cmd),torch.Generator(),deterministic=True)
        sample['action']=[[1.,.1]]*3
        ctrl=Controller(samples={'300':sample,'310':sample},branch=dict(time_s=300,revert=['ramp']))
        self.assertNotIn('ramp',ctrl.apply(r,cmd,300))
        self.assertNotIn('ramp',ctrl.apply(r,cmd,309))
        self.assertEqual(ctrl.apply(r,cmd,310),{})

    def test_budget_guards_and_reassigned_unused_seeds(self):
        self.assertEqual(sum(budget()['phases'].values())+budget()['unallocated'],6_000_000)
        install_seed_scope()
        for role in ('train','check','dev'):
            for w in worlds(role):guard(w,role)
        self.assertEqual(registry_conflicts('S-COOPNATIVE DESIGN READY 986000..986083'),[])
        self.assertEqual(registry_conflicts('another task986000..986083'),[1])
        with self.assertRaises(ValueError):guard(worlds('check')[0],'train')

    def test_random_guide_budget_final_panel_and_scope(self):
        from .protocol import UPDATES,TRAINING_SEEDS,batch_worlds,training_plan
        b=budget();self.assertEqual(UPDATES,24)
        self.assertEqual(b['phases']['train'],5*UPDATES*24*1200)
        self.assertEqual(TRAINING_SEEDS,(20261101,20261102))
        self.assertEqual(training_plan(),((0,20261101,('A','B')),(0,20261101,('R',)),(1,20261102,('A','B'))))
        self.assertEqual(b['phases']['final_test'],4*36*1200)
        self.assertEqual(b['unallocated'],1081200)
        self.assertEqual(b['phases']['replication_test'],86400)
        self.assertEqual(b['phases']['random_guide_test'],43200)
        self.assertEqual(b['phases']['checks'],5*6*12*1200+12*1200)
        self.assertEqual(b['phases']['dev'],(3+5)*24*1200)
        install_seed_scope();seen=set()
        for i in range(2):
            panel=worlds('train',i)
            self.assertFalse(seen&{w['seed'] for w in panel});seen|={w['seed'] for w in panel}
            for w in panel:guard(w,'train')
            self.assertEqual(len(batch_worlds(1,i)),24)
        with self.assertRaises(ValueError):guard(worlds('train',2)[0],'train')
        final=worlds('final_test')
        self.assertEqual(len(final),36);self.assertFalse(seen&{w['seed'] for w in final})
        for w in final:
            guard(w,'final_test')
            with self.assertRaises(ValueError):guard(w,'train')
        with self.assertRaises(ValueError):guard(dict(final[0],seed=964000),'final_test')
        self.assertEqual(registry_conflicts('other 989599..989636'),[1])

    def test_final_test_refuses_early_and_no_fallback_checkpoint(self):
        import tempfile
        from .runner import final_test,compare
        from .storage import write,read
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,'before this seed training'):
                final_test(root,{}, {})
            self.assertFalse((root/'FINAL_TEST_STARTED.json').exists())
            for i in range(3):write(root/f'S{i}_TRAINING_COMPLETE.json',dict(active_arms=['A']))
            final_test(root,{}, {})
            self.assertTrue(read(root/'FINAL_TEST_NOT_RUN.json')['no_fallback'])
            self.assertFalse((root/'FINAL_TEST_STARTED.json').exists())
        metrics={h:dict(team_delay_vs=10.,collisions=0.,teleports=0.,ttc_below_3_vs=2.,hard_brake=1.) for h in ('600','900')}
        records=[dict(world=w,arm=a,metrics=copy.deepcopy(metrics)) for w in worlds('final_test') for a in ('A','B')]
        result=compare(records,[],'B','A','final_test')
        self.assertEqual(result['worlds'],36)
        self.assertEqual(result['metrics']['900_team_delay_vs']['ci95'],[0.,0.])

    def test_frozen_batched_K8_adapter_multihorizon_structure(self):
        from .wm import adapted_futures
        ns=adapted_futures()
        self.assertIn('futures',ns)
        self.assertIn(1200,ns['futures'].__code__.co_consts)
        self.assertIn((60,120,300,600),ns['futures'].__code__.co_consts)
        from .wm import executor_adapter
        from tools.wmteach import runtime
        original=runtime.prepare_batch
        with executor_adapter():
            self.assertIn(600,runtime.prepare_batch.__code__.co_consts)
            self.assertIn(1600,runtime.prepare_batch.__code__.co_consts)
        self.assertIs(runtime.prepare_batch,original)

    def test_multihorizon_selection_scale_and_CRN_noise(self):
        rs=[dict(world=dict(cell=i,seed=986000+i),wm={},exact={}) for i in range(12)]
        for r in rs:
            for h in ('60','120','300','600'):
                r['wm'][h]=algebra(10.,[15.,16.],25.)
                r['exact'][h]={k:([2*x for x in v] if isinstance(v,list) else 2*v) for k,v in r['wm'][h].items()}
            r['exact']['600']['C_G']*=-1
        gate=multihorizon(rs,100)
        self.assertEqual(set(gate['weights']),{'60','120','300'})
        self.assertEqual(gate['primary_horizon_s'],300)
        self.assertAlmostEqual(sum(gate['weights'].values()),1.)
        self.assertEqual(multibonus(rs[0]['wm'],1,gate),0.)
        self.assertEqual(multibonus(rs[0]['wm'],9,gate),26.)
        fs=[{k:dict(draw_id=str(i),team_delay=v+1000*i) for k,v in
            [('joint',10.),('revert:0',15.),('revert:1',16.),('all',25.)]} for i in range(8)]
        n=noise(fs)
        self.assertGreater(n['team_return_advantage_variance'],0)
        self.assertEqual(n['paired_credit_variance'],0.)
        for i,f in enumerate(fs):f['all']['team_delay']+=i
        n=noise(fs)
        self.assertEqual(n['paired_credit_variance'],6.)
        self.assertEqual(n['D_G_variance'],6.)
        self.assertEqual(n['D_i_variance'],[0.,0.])

    def test_qualified_cost_source_binds_exec_function_without_linecache(self):
        import os,json
        from .wm import cost_adapter
        source='def reducer(x):\n return {h:x*h for h in (90,120,300,600)}\n'
        ns={};exec(compile(source,'<shared-frozen-reference>','exec'),ns)
        adapted=cost_adapter(ns['reducer'],source)
        self.assertEqual(adapted(2),{60:120,120:240,300:600,600:1200})
        with self.assertRaisesRegex(ValueError,'bytecode mismatch'):
            cost_adapter(ns['reducer'],source.replace('x*h','x+h'))
        path=os.environ.get('COOPMWM_COST_SOURCE')
        self.assertIsNotNone(path,'explicit qualified reducer receipt required')
        from pathlib import Path
        frozen=json.loads(Path(path).read_text())['code'];bound={}
        exec(compile(frozen,'<actual-frozen-cost-reference>','exec'),bound)
        fn=next(v for v in bound.values() if callable(v) and hasattr(v,'__code__'))
        actual=cost_adapter(fn,frozen)
        self.assertEqual(actual.__name__,fn.__name__)

    def test_dual_GPU_parity_checks_all_credit_coordinates(self):
        import copy
        from .dual import credit_difference
        a=dict(credits={str(h):dict(D_i=[1.,-1.],D_G=3.,C_G=3.) for h in (60,120,300,600)})
        b=copy.deepcopy(a);self.assertEqual(credit_difference(a,b),0.)
        b['credits']['600']['D_i'][1]+=2.
        self.assertEqual(credit_difference(a,b),2.)

    def test_throughput_sampling_logs_exact_mixture_mass(self):
        from .exploration import route,mixture_ratio
        from unittest.mock import patch
        import math
        sample=dict(logp=0.)
        with patch('tools.coopmwm.exploration.regime',return_value='ordinary'):
            route(sample,[],1.,torch.Generator().manual_seed(123),dict(ordinary=.125))
        self.assertEqual(sample['behaviour']['mixture_mass'],.125)
        sample['behaviour']['route']='policy'
        self.assertAlmostEqual(float(mixture_ratio(torch.tensor(0.),sample)),1/.875,places=6)

    def test_concurrent_diagnostic_and_train_reservations_are_once_only(self):
        from concurrent.futures import ThreadPoolExecutor
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .storage import reserve,ledger
        with TemporaryDirectory() as d:
            root=Path(d)
            def charge(i):reserve(root,[dict(id=str(i),sim_s=1200)],'train' if i%2 else 'calibration')
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(charge,range(12)))
            rows=ledger(root)
            self.assertEqual(len(rows),12);self.assertEqual(sum(r['sim_s'] for r in rows),14400)
            with self.assertRaisesRegex(ValueError,'duplicate charge'):charge(0)

    def test_numeric_sampling_requires_matching_prospective_payload_registration(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from .protocol import digest
        from .runner import registered_sampling
        with TemporaryDirectory() as d:
            root=Path(d);payload=dict(probabilities={'merge-dense':.02,'ordinary':.005});h=digest(payload)
            (root/'SAMPLING_1755.json').write_text(json.dumps(dict(payload=payload,registration_sha256_of_payload=h)))
            auth=root/'SAMPLING_AUTHORITY_1755.md';auth.write_text('missing registration')
            with self.assertRaisesRegex(ValueError,'registration absent'):registered_sampling(root)
            auth.write_text('S-COOPMWM 17:55 NUMERIC SAMPLING REGISTERED '+h)
            self.assertEqual(registered_sampling(root),payload['probabilities'])
            payload['probabilities']['ordinary']=.01
            (root/'SAMPLING_1755.json').write_text(json.dumps(dict(payload=payload,registration_sha256_of_payload=h)))
            with self.assertRaisesRegex(ValueError,'binding mismatch'):registered_sampling(root)

    def test_random_window_selection_uses_only_eligibility_not_credit(self):
        from .runner import select_calibration
        episodes=[dict(world=dict(cell=i,seed=986000+i),steps=[
            dict(time_s=300+j*10,group=['a','b'],wm_credit=(-1)**j*10000) for j in range(5)]) for i in range(12)]
        first,missing=select_calibration(episodes)
        self.assertFalse(missing)
        changed=copy.deepcopy(episodes)
        for ep in changed:
            for s in ep['steps']:s['wm_credit']*=-1
        second,_=select_calibration(changed)
        self.assertEqual([s['time_s'] for ep,s in first],[s['time_s'] for ep,s in second])

    def test_runner_CPU_fake_episode_and_exact_prefix(self):
        import tempfile
        from types import SimpleNamespace
        from .plant import episode
        from tools.leverprobe import backend as exact
        from tools.coopfree import backend as cf
        r=rows()
        class Fake:
            def __init__(self,*args):
                self.env=SimpleNamespace(now=300);self.actual=300.
                self.totals=dict.fromkeys(('team_delay_vs','road_delay_vs','source_wait_vs','collisions',
                    'teleports','ttc_below_3_vs','hard_brake'),0.)
            def fingerprint(self):return dict(time_s=self.env.now,J=self.totals['team_delay_vs'])
            def close(self):pass
        def tick(b,c,trace):
            c.apply(r,{x['vid']:.2 for x in r if x['is_cav']},b.env.now)
            b.env.now+=1;b.actual+=1
            b.totals['team_delay_vs']+=5
            b.totals['road_delay_vs']+=2;b.totals['source_wait_vs']+=3
            return []
        with tempfile.TemporaryDirectory() as temp:
            job=dict(id='cpu',path=temp,world=worlds()[0],role='train',arm='A',
                weights=Policy().state_dict(),runtime='unused',checkpoint='unused',noise_seed=1,
                deterministic=True,sim_s=1200)
            with patch.object(exact,'setup'),patch.object(exact,'Backend',Fake),patch.object(cf,'tick',tick):
                result=episode(job)
            self.assertEqual(result['actual_sim_s'],1200)
            self.assertEqual(len(result['steps']),90)
            self.assertEqual(sum(s['reward'] for s in result['steps']),-4500.)
            self.assertEqual(result['metrics']['600']['team_delay_vs'],3000.)

    def test_mixture_deterministic_atom_has_zero_policy_mass(self):
        from .exploration import mixture_ratio
        lp=torch.tensor(-3.,requires_grad=True)
        own=dict(logp=-3.,behaviour=dict(route='policy',mixture_mass=.25))
        atom=dict(logp=-3.,behaviour=dict(route='wm',mixture_mass=.25))
        self.assertAlmostEqual(float(mixture_ratio(lp,own)),4/3,places=6)
        self.assertEqual(float(mixture_ratio(lp,atom)),0.)
        # Exact enumeration of the route measure integrates the target to one.
        integral=.75*mixture_ratio(lp,own)+.25*mixture_ratio(lp,atom)
        self.assertAlmostEqual(float(integral),1.,places=6)
        integral.backward();self.assertAlmostEqual(float(lp.grad),1.,places=6)

    def test_executed_proposal_imitation_and_SUMO_only_update(self):
        from .exploration import apply_proposal,update,imitation
        r=rows();p=Policy();g=torch.Generator().manual_seed(4)
        s=p.sample(encode(r,{v['vid']:.2 for v in r if v['is_cav']}),g)
        s['behaviour']=dict(route='wm',beta=1.,mixture_mass=.1,executed=False)
        proposal=dict(accepted=True,option='O1',targets={'upstream':dict(speed=3.,tau=2.)},expected_saving=1.e9)
        actual=apply_proposal(s,proposal)
        for k in ('speed','tau'):
            self.assertAlmostEqual(targets(actual)['upstream'][k],proposal['targets']['upstream'][k])
        loss=imitation(p,actual);loss.backward()
        self.assertGreater(float(p.actor[-1].bias.grad.abs().sum()),0.)
        episodes=[dict(steps=[dict(sample=actual,reward=-100.)])]
        before=copy.deepcopy(episodes)
        update(p,torch.optim.Adam(p.parameters(),lr=3e-4),episodes,g,epochs=1)
        self.assertEqual(episodes,before)
        # A forecast cannot leak through the real reward/advantage ABI.
        other=copy.deepcopy(episodes);other[0]['steps'][0]['sample']['behaviour']['expected_saving']=-1.e9
        from tools.coopnative.learner import advantages
        self.assertEqual(advantages(episodes[0]['steps']),advantages(other[0]['steps']))

    def test_adaptive_beta_scale_free_monotone_and_zero_floor(self):
        from .exploration import trust
        def data(sign):
            return [dict(world=i,route=route,executed=route=='wm',q=.25,
                         advantage=sign*100. if route=='wm' else 0.)
                    for i in range(12) for route in ('wm','policy','policy','policy')]
        negative=data(-1);v=trust(1.,negative,100)
        self.assertEqual(v['beta'],0.);self.assertLess(v['ci95'][1],0.)
        self.assertEqual(trust(1.,data(1),100)['beta'],1.)
        self.assertEqual(trust(.3,data(1),100)['beta'],.3)
        self.assertEqual(trust(0.,data(1),100)['beta'],0.)
        scaled=copy.deepcopy(negative)
        for r in scaled:r['advantage']*=1000
        self.assertEqual(trust(1.,scaled,100)['beta'],v['beta'])
        self.assertEqual(trust(1.,[],100)['beta'],1.)

    def test_proposal_targets_reuse_inherited_executors_and_whitelist(self):
        from .wm import proposal_candidates
        r=rows()
        # Observable two-mainline group and a ramp: enables multiple inherited options.
        r[0].update(vid='a',s_m=-20.,lane=0,is_cav=True)
        r[1].update(vid='b',s_m=-40.,lane=1,is_cav=True)
        record=dict(state_id='cpu',world_id=986000,time_s=300,rows=r,
            group=['a','b'],targets={},mode='proposal',setting=0)
        validate(record);acts=proposal_candidates(record)
        self.assertEqual([a['id'] for a in acts[:2]],['policy','MAPPO'])
        self.assertIn('O1',[a['id'] for a in acts])
        for a in acts[2:]:
            self.assertEqual(a['window'],10)
            self.assertLessEqual(set(a['proposal_targets']),{'a','b'})
        with self.assertRaises(ValueError):validate(dict(record,driver_type=1))

    def test_B_deterministic_evaluation_never_queries_teacher(self):
        r=rows();cmd={v['vid']:.2 for v in r if v['is_cav']}
        def forbidden(*a):raise AssertionError('teacher forbidden in evaluation')
        ctrl=Controller(Policy(),torch.Generator(),True,guide=forbidden,beta=1.)
        self.assertEqual(ctrl.apply(r,cmd,300),{})
        self.assertNotIn('behaviour',ctrl.events[0]['sample'])

    def test_diagnostic_cannot_gate_B_or_add_reward(self):
        import inspect
        from .runner import run
        source=inspect.getsource(run)
        self.assertNotIn('multibonus',source)
        self.assertNotIn("step['reward']",source)
        self.assertNotIn('B_STOP',source)
        self.assertIn('final_test(root,b,policies,si)',source)

    def test_primary_final_runs_before_replications_and_controls_are_reused(self):
        import tempfile
        from .runner import final_test
        from .storage import write,read
        calls=[]
        metrics={h:dict(team_delay_vs=10.,collisions=0.,teleports=0.,ttc_below_3_vs=2.,hard_brake=1.) for h in ('600','900')}
        def fake(root,b,p,ws,arm,role,ident,phase,*args):
            calls.append((arm,phase))
            return [dict(world=w,arm=arm,metrics=copy.deepcopy(metrics)) for w in ws]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'S0_TRAINING_COMPLETE.json',{})
            for a in ('A','B'):(root/f'S0_{a}_checkpoint_024.pt').write_bytes(b'fixture')
            with patch('tools.coopmwm.runner.wave',fake),patch('tools.coopmwm.runner.discover'):
                final_test(root,{'authority':[]},{'A':Policy(),'B':Policy()})
                self.assertEqual(calls,[('mappo','final_test'),('native','final_test'),('A','final_test'),('B','final_test')])
                self.assertTrue((root/'FINAL_TEST_COMPLETE.json').exists())
                final_test(root,{}, {}) # Completed panel has no physical rerun.
                self.assertEqual(len(calls),4)
                write(root/'S1_TRAINING_COMPLETE.json',{})
                for a in ('A','B'):(root/f'S1_{a}_checkpoint_024.pt').write_bytes(b'fixture')
                final_test(root,{}, {'A':Policy(),'B':Policy()},1)
                self.assertEqual(calls[-2:],[('A','replication_test'),('B','replication_test')])
                self.assertEqual(read(root/'S1_FINAL_TEST_COMPLETE.json')['worlds'],36)

    def test_wave_noise_labels_are_fresh_and_paired(self):
        from .protocol import digest
        seeds=[int(digest(dict(training_seed=20261101,world=986000,wave=f's0_u{u:03}'))[:15],16) for u in range(1,25)]
        self.assertEqual(len(set(seeds)),24)

    def test_bucket_EB_trust_separates_winners_losers_and_is_scale_free(self):
        from .exploration import trust
        data=[]
        for w in range(12):
            data.append(dict(world=w,route='policy',executed=False,p=.01,q=0.,
                             option=None,regime='merge-dense',advantage=-100.))
            for o,gain in [('O1',40.),('O2',-40.)]:
                data.append(dict(world=w,route='wm',executed=True,p=.01,q=.01,
                    option=o,regime='merge-dense',advantage=-100.+gain))
        v=trust(1.,data,300)
        self.assertAlmostEqual(v['buckets']['O1|merge-dense']['beta'],1.)
        self.assertEqual(v['buckets']['O2|merge-dense']['beta'],0.)
        self.assertGreater(v['between_bucket_variance'],0.)
        self.assertEqual(v['buckets']['O4|ordinary']['shrinkage_to_local'],0.)
        scaled=copy.deepcopy(data)
        for r in scaled:r['advantage']*=1000
        vs=trust(1.,scaled,300)
        for k,b in v['buckets'].items():
            self.assertAlmostEqual(vs['buckets'][k]['beta'],b['beta'])
        later=trust(v,data,300)
        self.assertEqual(later['buckets']['O2|merge-dense']['beta'],0.)
        self.assertAlmostEqual(trust(v,[],100)['buckets']['O1|merge-dense']['beta'],1.)

    def test_query_then_bucket_execution_preserves_actual_behaviour_measure(self):
        from .exploration import execute,route,regime,rate,BUCKETS
        r=rows();p=Policy();g=torch.Generator().manual_seed(8)
        s=p.sample(encode(r,{v['vid']:.2 for v in r if v['is_cav']}),g)
        prior=dict(beta=1.,buckets={k:dict(beta=0. if k.startswith('O1|') else 1.) for k in BUCKETS})
        with patch('tools.coopmwm.exploration.torch.rand',return_value=torch.tensor(0.)):
            self.assertTrue(route(s,r,prior,g))
        self.assertEqual(s['behaviour']['mixture_mass'],rate(r))
        proposal=dict(accepted=True,option='O1',targets={'upstream':dict(speed=3.,tau=2.)},expected_saving=10.)
        stopped=execute(copy.deepcopy(s),proposal,prior,g)
        self.assertFalse(stopped['behaviour']['executed'])
        self.assertEqual(stopped['behaviour']['beta'],0.)
        proposal['option']='O2';actual=execute(copy.deepcopy(s),proposal,prior,g)
        self.assertTrue(actual['behaviour']['executed'])
        self.assertEqual(actual['behaviour']['q'],rate(r))
        self.assertEqual(regime(list(reversed(r))),regime(r))
        # No seed/demand/share/truth is passed to the observable regime rule.
        self.assertNotIn('demand',__import__('inspect').getsource(regime))

    def test_random_proposer_uniform_available_mask_and_zero_WM(self):
        from .client import random_guide
        from .exploration import execute,mixture_ratio
        r=rows();r[0].update(vid='a',s_m=-20.,lane=0,is_cav=True)
        r[1].update(vid='b',s_m=-40.,lane=1,is_cav=True)
        commands={v['vid']:.2 for v in r if v['is_cav']}
        p=Policy();g=torch.Generator().manual_seed(13)
        s=p.sample(encode(r,commands),g)
        s['behaviour']=dict(route='wm',beta=0.,regime='ordinary',sample_probability=.01,mixture_mass=.01,executed=False)
        from .wm import proposal_candidates
        record=dict(rows=r,targets=__import__('tools.coopmwm.policy',fromlist=['targets']).targets(s),group=['a','b'],setting=2)
        available=proposal_candidates(record)[2:]
        self.assertGreater(len(available),1)
        seen=set()
        for i,option in enumerate(available):
            with patch('tools.coopmwm.client.request',side_effect=AssertionError('R cannot query WM')),patch('torch.randint',return_value=torch.tensor(i)) as draw:
                proposal=random_guide(g)(r,commands,300,s)
            self.assertEqual(draw.call_args.args[0],len(available))
            self.assertEqual(proposal['option'],option['id']);seen.add(proposal['option'])
            self.assertEqual(proposal['targets'],option['proposal_targets'])
            self.assertEqual(proposal['option_probability'],1/len(available))
            self.assertIsNone(proposal['expected_saving'])
            actual=execute(copy.deepcopy(s),proposal,1.,g)
            self.assertTrue(actual['behaviour']['executed'])
            self.assertEqual(actual['behaviour']['q'],.01/len(available))
            self.assertEqual(float(mixture_ratio(torch.tensor(s['logp']),actual)),0.)
        self.assertEqual(seen,{a['id'] for a in available})
        with patch('tools.coopmwm.wm.proposal_candidates',return_value=available[:0]):
            self.assertFalse(random_guide(g)(r,commands,300,s)['accepted'])

    def test_random_final_test_once_controls_reused_and_B_minus_R(self):
        import tempfile
        from .runner import random_final_test
        from .storage import write,read
        calls=[]
        metrics=lambda value:{h:dict(team_delay_vs=value,collisions=0.,teleports=0.,ttc_below_3_vs=0.,hard_brake=0.) for h in ('600','900')}
        def fake(root,b,p,ws,arm,role,ident,phase,*args):
            calls.append((arm,phase))
            return [dict(world=w,arm=arm,metrics=metrics(12.)) for w in ws]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,'requires primary test'):random_final_test(root,{},Policy())
            write(root/'FINAL_TEST_COMPLETE.json',{});write(root/'R0_TRAINING_COMPLETE.json',{})
            with self.assertRaisesRegex(ValueError,'checkpoint absent'):random_final_test(root,{},Policy())
            (root/'S0_R_checkpoint_024.pt').write_bytes(b'fixture')
            for w in worlds('final_test'):
                for a in ('A','B','mappo','native'):
                    write(root/'jobs'/f's0_final_{a}_{w["seed"]}'/'RESULT.json',dict(world=w,arm=a,metrics=metrics(10.)))
            with patch('tools.coopmwm.runner.wave',fake):
                random_final_test(root,{},Policy());random_final_test(root,{},Policy())
            self.assertEqual(calls,[('R','random_guide_test')])
            result=read(root/'R_FINAL_TEST_COMPLETE.json')
            self.assertEqual(result['comparisons']['B_minus_R']['metrics']['900_team_delay_vs']['ci95'],[-2.,-2.])

    def test_completed_wave_adoption_never_reserves_or_runs_SUMO(self):
        import tempfile
        from .runner import wave
        from .storage import write
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);w=worlds()[0]
            record=dict(world=w,arm='A',id=f's0_u001_A_{w["seed"]}')
            write(root/'jobs'/record['id']/'RESULT.json',record)
            with patch('tools.coopmwm.runner.reserve',side_effect=AssertionError('duplicate physics')):
                self.assertEqual(wave(root,{},Policy(),[w],'A','train','s0_u001','train'),[record])


if __name__=='__main__':unittest.main()
