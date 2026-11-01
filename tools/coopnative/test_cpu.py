"""CPU contracts; suite must run under an outer subprocess.Popen prohibition."""
import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .policy import Policy, encode, targets
from .execution import Controller
from .learner import advantages, GAMMA, LAMBDA, update, save, load
from .protocol import approval_line, registry_conflicts, worlds, batch_worlds, team_delay, CAP, budget
from .storage import reserve, reconciled, write


def rows():
    return [dict(vid=v,s_m=s,v_mps=speed,lane=lane,is_cav=cav,length_m=5.) for v,s,speed,lane,cav in (
        ('upstream',-400.,20.,0,True),('downstream',300.,22.,1,True),
        ('ramp',-20.,10.,2,True),('lead',-380.,5.,0,False))]


def state(r):
    return {k:torch.tensor([[row[field] for row in r]],dtype=dtype) for k,field,dtype in (
        ('s_m','s_m',torch.float32),('v_mps','v_mps',torch.float32),
        ('lane','lane',torch.long),('is_cav','is_cav',torch.bool),('length_m','length_m',torch.float32))} | dict(present=torch.ones((1,len(r)),dtype=torch.bool))


def env(r):
    return SimpleNamespace(slot_vehicle_ids=lambda:[x['vid'] for x in r],state=lambda:state(r))


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_zero_mean_native_all_road_all_window(self):
        p, g, c = Policy(), torch.Generator().manual_seed(2), Controller()
        sample = p.sample(encode(rows()),g,deterministic=True)
        self.assertEqual(sample['action'],[[0.,0.]]*3)
        self.assertEqual(targets(sample),{})
        c.hold(sample,300)
        for t in range(300,311):
            changed = [dict(r,v_mps=r['v_mps']+(t-300)*.3) for r in rows()]
            command,audit = c.command(env(changed),changed,t)
            self.assertTrue(torch.isnan(command).all())
            self.assertEqual(audit,[])

    def test_zero_native_actual_commandownership_parity_and_release(self):
        from traffic_shepherd.ownership import CommandOwnership
        from traffic_shepherd.config import load_config
        from traffic_shepherd.damping import DampingDecision
        class Vehicle:
            def __init__(self): self.calls=[]
            def getSpeedMode(self,v): return 31
            def slowDown(self,*a): self.calls.append(('slowDown',*a))
            def setSpeed(self,*a): self.calls.append(('setSpeed',*a))
        controller = SimpleNamespace(config=load_config(),decide=lambda *a,**kw:DampingDecision(fallback_reason='none'))
        v1,v2 = Vehicle(),Vehicle()
        a,b = [CommandOwnership(SimpleNamespace(vehicle=v),{'upstream'},controller) for v in (v1,v2)]
        c = Controller()
        c.hold(Policy().sample(encode(rows()),torch.Generator(),deterministic=True),300)
        for t in range(300,310):
            commands,_ = c.command(env(rows()),rows(),t)
            direct = {r['vid']:float(commands[0,i]) for i,r in enumerate(rows()) if math.isfinite(float(commands[0,i]))}
            one = a.step({},now_s=t,live_ids={'upstream'},native_ids={'upstream'})
            two = b.step({},now_s=t,live_ids={'upstream'},wall_commands=direct,native_ids={'upstream'}-direct.keys())
            self.assertEqual(one,two)
        self.assertEqual(v1.calls,v2.calls)
        self.assertEqual(v1.calls,[])

    def test_setpoint_safety_roadwide_hold_expiry_env_actuator(self):
        from wms.control.safety import safe_accel
        r = rows()
        sample = Policy().sample(encode(r),torch.Generator(),deterministic=True)
        sample['action'] = [[5.,0.],[0.,.5],[5.,0.]]
        c = Controller()
        c.hold(sample,300)
        e = env(r)
        with patch('wms.control.safety.safe_accel',wraps=safe_accel) as safety:
            command,audit = c.command(e,r,300)
        safety.assert_called_once()
        self.assertEqual({a['vid'] for a in audit},{'upstream','downstream','ramp'})
        self.assertLess(float(command[0,0]),0.)  # Real stopping-distance bound, slow leader.
        held = copy.deepcopy(c.targets)
        c.command(e,[dict(x,v_mps=x['v_mps']+1) for x in r],309)
        self.assertEqual(c.targets,held)
        self.assertTrue(torch.isnan(c.command(e,r,310)[0]).all())
        e.step = unittest.mock.Mock(return_value=(None,dict(interval={})))
        c.step(e,r,300)
        self.assertEqual(e.step.call_args.kwargs,dict(lanes=None,fallback='native'))

    def test_team_reward_exact_comparator_J_including_source(self):
        from .plant import accumulate, METRICS
        totals = dict.fromkeys(METRICS,0.)
        reward = 0.
        for road,source in ((2.25,8.5),(0.,11.),(7.125,0.)):
            interval = dict(speed_deficit_vehicle_seconds=road,source_wait_vehicle_seconds=source)
            reward -= team_delay(interval)
            accumulate(totals,interval)
        self.assertEqual(-reward,totals['road_delay_vs']+totals['source_wait_vs'])
        self.assertEqual(-reward,totals['team_delay_vs'])

    def test_observation_leakage_guard_and_local_noninterference(self):
        r = rows()
        for extra in ('seed','world','demand','true_share','driver_type'):
            with self.assertRaisesRegex(ValueError,'whitelist'):
                encode([dict(r[0],**{extra:999}),*r[1:]])
        a = encode(r)
        self.assertEqual(a,encode(list(reversed(r))))
        altered = [dict(x,v_mps=99.) if x['vid']=='lead' else x for x in r]
        b = encode(altered)
        i = a['members'].index('downstream')
        self.assertEqual(a['x'][i],b['x'][i])
        self.assertEqual(a['rsu'],b['rsu'])

    def test_joint_density_matches_correlated_mvn(self):
        from tools.coopfree.policy import gaussian_log_prob
        n=5
        mean=torch.randn(n,2,dtype=torch.float64,requires_grad=True)
        action=torch.randn(n,2,dtype=torch.float64)
        sigma=[4.5,.5]
        cov=.5*torch.eye(n,dtype=mean.dtype)+.5*torch.ones(n,n,dtype=mean.dtype)
        expected=sum(torch.distributions.MultivariateNormal(mean[:,d],covariance_matrix=cov*sigma[d]**2).log_prob(action[:,d]) for d in (0,1))
        actual=gaussian_log_prob(action,mean,.5,sigma)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-10))
        self.assertTrue(torch.allclose(torch.autograd.grad(actual,mean,retain_graph=True)[0],torch.autograd.grad(expected,mean)[0]))

    def test_long_credit_gae_reaches_300s(self):
        self.assertGreaterEqual(-10/math.log(GAMMA*LAMBDA),300.)
        steps=[dict(reward=10000. if i==30 else 0.,sample=dict(value=0.)) for i in range(31)]
        self.assertAlmostEqual(advantages(steps)[0][0],math.exp(-300/400))

    def test_checkpoint_resume_exact_optimizer_rng_next_update(self):
        p, g = Policy(),torch.Generator().manual_seed(7)
        optimizer=torch.optim.Adam(p.parameters(),lr=3e-4)
        def batch(policy,generator):
            return [dict(steps=[dict(sample=policy.sample(encode(rows()),generator),reward=-float(i+1)) for i in range(12)])]
        update(p,optimizer,batch(p,g),g,epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            save(path,p,optimizer,g,1,1200)
            ep=batch(p,g)
            update(p,optimizer,ep,g,epochs=1)
            q,opt,h,payload=load(path)
            eq=batch(q,h)
            self.assertEqual(ep,eq)
            update(q,opt,eq,h,epochs=1)
            for key,value in p.state_dict().items(): self.assertTrue(torch.equal(value,q.state_dict()[key]),key)
            for key,value in optimizer.state_dict()['state'].items():
                for field,tensor in value.items(): self.assertTrue(torch.equal(tensor,opt.state_dict()['state'][key][field]))
            self.assertEqual(payload['charged'],1200)
            self.assertTrue(torch.equal(g.get_state(),h.get_state()))

    def test_no_mapppo_initialization_loader_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'foreign.pt'
            torch.save(dict(schema='mappo'),path)
            with self.assertRaises(ValueError): load(path)

    def test_approval_blocks_before_any_plant(self):
        from .runner import run
        for text in ('S-COOPNATIVE BUDGET APPROVED',"MAIN until 'S-COOPNATIVE BUDGET APPROVED'",'MAIN NO SUMO until S-COOPNATIVE BUDGET APPROVED'):
            with self.assertRaises(ValueError): approval_line(text)
        self.assertIn('APPROVED',approval_line('2026-11-01 MAIN: S-COOPNATIVE BUDGET APPROVED cap 4664400'))
        with tempfile.TemporaryDirectory() as directory:
            registry=Path(directory)/'REGISTRATION.md'
            registry.write_text('MAIN pending approval')
            with patch('tools.coopnative.plant.episode',side_effect=AssertionError('plant forbidden')) as plant:
                with self.assertRaises(ValueError): run(Path(directory),registry)
                plant.assert_not_called()

    def test_world_roles_ranges_and_whole_batch_budgets(self):
        self.assertEqual(len(worlds()),72)
        self.assertFalse({w['seed'] for w in worlds()} & {w['seed'] for w in worlds('check')})
        for u in (1,2,3,144):
            batch=batch_worlds(u)
            self.assertEqual(len(batch),24)
            self.assertEqual({w['cell'] for w in batch},set(range(12)))
        self.assertEqual(registry_conflicts('old 986000–986099'),[1])
        self.assertEqual(registry_conflicts('old 986000..083'),[1])
        self.assertEqual(registry_conflicts('sha af986000be'),[])
        self.assertLessEqual(sum(budget()['phases'].values()),CAP)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            reserve(root,[dict(id='one')],'train')
            with self.assertRaises(ValueError): reserve(root,[dict(id='one')],'train')
            with self.assertRaises(ValueError): reconciled(root)
            write(root/'jobs/one/RESULT.json',dict(actual_sim_s=1200))
            self.assertEqual(reconciled(root),1200)
            with self.assertRaises(ValueError): reserve(root,[dict(id='large',sim_s=CAP)],'train')

    def test_actual_frozen_dual_seed_guards_all_roles(self):
        from .protocol import install_seed_scope,guard
        install_seed_scope()
        for role in ('train','check','dev'):
            for world in worlds(role): guard(world,role)
        for seed in (964000,962001,977001):
            with self.assertRaises(ValueError): guard(dict(worlds()[0],seed=seed),'train')
        with self.assertRaises(ValueError): guard(worlds('check')[0],'train')

    def test_episode_orchestration_CPU_fake_physics_no_simulator(self):
        import sys
        from .plant import episode
        clock=[0.]
        def simulation_step(): clock[0]+=1
        traci=SimpleNamespace(simulationStep=simulation_step,simulation=SimpleNamespace(getTime=lambda:clock[0]))
        r=rows()
        class FakeEnv:
            def __init__(self,**kwargs):
                self.history=SimpleNamespace(raw=lambda n:{k:np.array([clock[0]]*3) for k in ('kin','mask','vid_index')})
            def reset(self,**kwargs):
                clock[0]=0.
                for _ in range(300):
                    traci.simulationStep()
                    kwargs['warmup_hook'](self)
            def slot_vehicle_ids(self): return [row['vid'] for row in r]
            def state(self): return state(r)
            def step(self,cmd,**kwargs):
                traci.simulationStep()
                return None,dict(interval=dict(speed_deficit_vehicle_seconds=2.,source_wait_vehicle_seconds=3.),done=clock[0]==1200,lane_commands={})
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            job=dict(id='cpu',path=directory,runtime='unused',world=worlds()[0],role='train',arm='learned',
                     weights=Policy().state_dict(),noise_seed=1,
                     approval_line='MAIN: S-COOPNATIVE BUDGET APPROVED CPU fake-physics fixture')
            with patch('tools.coopnative.plant.setup'),patch('tools.coopnative.plant.guard'),patch('tools.coopnative.plant.observations',side_effect=lambda env:copy.deepcopy(r)),patch('wms.control.policy.load_policy_fn',side_effect=AssertionError('MAPPO loader forbidden in learning')),patch.dict(sys.modules,{'traci':traci,'traffic_shepherd.c4.env':SimpleNamespace(CavSumoEnv=FakeEnv)}):
                result=episode(job)
            self.assertEqual(result['actual_sim_s'],1200.)
            self.assertEqual(result['metrics']['600']['team_delay_vs'],3000.)
            self.assertEqual(result['metrics']['900']['team_delay_vs'],4500.)
            self.assertEqual(sum(s['reward'] for s in result['steps']),-4500.)
            self.assertEqual(len(result['steps']),90)

    def test_existing_behaviour_detector_schema(self):
        from tools.coopbehav.io import Tape,Vehicle
        from tools.coopbehav.detectors import detect
        tape=Tape(np.array([300.,301.]),[{r['vid']:Vehicle(r['s_m'],r['v_mps'],r['length_m'],r['lane'],r['is_cav'],None,'native',None) for r in rows()} for _ in range(2)],
                  meta=dict(seed=986072),coverage=dict(trace_scope='full_observable_fleet'))
        result=detect(tape)
        self.assertIn('events',result)
        self.assertIn('coverage',result)

    def test_paired_world_bootstrap_and_complete_panel(self):
        from .evaluation import paired,exact_credit
        records=[]
        for w in worlds('check'):
            for arm in ('learned','native'):
                records.append(dict(world=w,arm=arm,metrics={str(h):dict(team_delay_vs=w['seed']+(arm=='learned')*2,
                    collisions=0,teleports=0,ttc_below_3_vs=0,hard_brake=0) for h in (600,900)}))
        result=paired(records,'native','check')
        self.assertEqual(result['metrics']['600_team_delay_vs'],dict(mean=2.,ci95=[2.,2.]))
        with self.assertRaises(ValueError): paired(records[:-1],'native','check')
        with self.assertRaises(ValueError): paired(records+[records[0]],'native','check')
        self.assertEqual(exact_credit(100,60,[110,105]),dict(D_G=40,D_i=[-10,-5],C_G=55))


if __name__ == '__main__':
    unittest.main()
