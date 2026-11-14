"""Meaningful CPU contracts on the actual frozen MAPPO executor, with no SUMO."""
import copy
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .policy import SetPointPolicy, encode, requests, decode, gaussian_log_prob, SIGMA
from .execution import Controller, inject
from .credit import credit, update, next_sigma
from .protocol import worlds, guard, group, install_seed_scope, budget


def rows():
    return [dict(vid=v,s_m=s,v_mps=speed,lane=lane,is_cav=cav,length_m=5.)
            for v,s,lane,speed,cav in [('a',-50,0,20.,True),('b',-48,1,18.,True),
                                      ('lead',-10,1,15.,False),('ramp',-5,2,5.,True)]]


class ScientificContracts(unittest.TestCase):
    def test_binary_version_banner(self):
        from .protocol import sumo_version_ok
        self.assertTrue(sumo_version_ok('Eclipse SUMO sumo 1.27.1\n Build features: Linux'))
        self.assertTrue(sumo_version_ok('Eclipse SUMO sumo Version 1.27.1\n'))
        self.assertFalse(sumo_version_ok('Eclipse SUMO sumo 1.27.10\n'))
        self.assertFalse(sumo_version_ok('Eclipse SUMO sumo 1.26.0\n'))

    def test_approval_and_binding_fail_before_plant(self):
        from .protocol import approval_line
        from .runner import run
        for text in ('awaiting S-COOPFREE BUDGET APPROVED',
                     "MAIN must append 'S-COOPFREE BUDGET APPROVED'",'S-COOPBANDIT BUDGET APPROVED'):
            with self.assertRaises(ValueError): approval_line(text)
        self.assertEqual(approval_line('- MAIN S-COOPFREE BUDGET APPROVED cap 2,000,000'),
                         '- MAIN S-COOPFREE BUDGET APPROVED cap 2,000,000')
        with tempfile.TemporaryDirectory() as d:
            with patch('tools.coopfree.backend.create',side_effect=AssertionError('unexpected plant')) as plant:
                with self.assertRaises(FileNotFoundError): run(Path(d),'screen')
                plant.assert_not_called()

    def test_all_lane_whole_zone_groups_and_no_truth_features(self):
        r = rows()
        self.assertEqual(group(r),['a','b','ramp'])
        cmd = dict(a=.1,b=-.2,ramp=0.)
        a = encode(r,cmd)
        altered = [dict(x,hidden_type=99,seed=964000,unobserved_setting=8) for x in r]
        self.assertEqual(a,encode(altered,cmd))
        self.assertEqual(a,encode(list(reversed(r)),cmd))
        self.assertEqual(len(a['x'][0]),43)

    def test_zero_mean_broadcast_and_representable_first_command(self):
        p = SetPointPolicy()
        cmd = dict(a=.1,b=-.2,ramp=0.)
        context = encode(rows(),cmd)
        mean,_ = p(context)
        self.assertTrue(torch.equal(mean,torch.tensor(context['anchors'])))
        self.assertTrue(torch.equal(p.broadcast(torch.randn(12)),torch.zeros(4)))
        sample = p.sample(context,torch.Generator().manual_seed(1),deterministic=True)
        issued = requests(decode(sample),rows())
        for vid in cmd:
            self.assertAlmostEqual(issued[vid]['requested_raw'],cmd[vid],places=5)

    def test_fixed_absolute_targets_expiry_exit_and_solo_context(self):
        context = encode(rows(),dict(a=0.,b=0.,ramp=0.))
        sample = SetPointPolicy().sample(context,torch.Generator().manual_seed(2))
        a = Controller(sample=sample,start=300)
        b = Controller(sample=sample,start=300,members=['b'])
        first = a.apply(rows(),dict(a=0.,b=0.,ramp=0.),300)
        self.assertEqual(first['b'],b.apply(rows(),dict(a=0.,b=0.,ramp=0.),300)['b'])
        targets = copy.deepcopy(a.targets)
        a.apply([dict(r,v_mps=r['v_mps']+1) for r in rows()],dict(a=9.,b=9.,ramp=9.),301)
        self.assertEqual(a.targets,targets)
        self.assertEqual(a.apply(rows(),dict(a=0.,b=0.,ramp=0.),300+sample['window']),{})
        exited = [dict(r,s_m=80.) if r['vid']=='b' else r for r in rows()]
        self.assertNotIn('b',a.apply(exited,dict(a=0.,b=0.,ramp=0.),301))

    def test_correlated_density_and_gradient_matches_mvn(self):
        for n in (1,2,9):
            for rho in (0.,.4,.94):
                mean = torch.randn(n,2,dtype=torch.float64,requires_grad=True)
                action = torch.randn(n,2,dtype=torch.float64)
                sigma = [4.5,.5]
                ours = gaussian_log_prob(action,mean,rho,sigma)
                cov = (1-rho)*torch.eye(n,dtype=mean.dtype)+rho*torch.ones(n,n,dtype=mean.dtype)
                expected = sum(torch.distributions.MultivariateNormal(mean[:,d],covariance_matrix=cov*sigma[d]**2)
                               .log_prob(action[:,d]) for d in (0,1))
                self.assertTrue(torch.allclose(ours,expected,atol=1e-8))
                g = torch.autograd.grad(ours,mean,retain_graph=True)[0]
                h = torch.autograd.grad(expected,mean)[0]
                self.assertTrue(torch.allclose(g,h,atol=1e-8))

    def test_credit_algebra_team_sharing_and_anti_gaming(self):
        got = credit(100.,60.,[110.,105.])
        self.assertEqual((got['D_G'],got['D_i'],got['C_G']), (40.,[-10.,-5.],55.))
        self.assertEqual(got['objective'],80.)
        self.assertTrue(all(v > 0 for v in got['member_team_credit']))
        for solos in ([1.,2.],[1000.,2000.]):
            self.assertEqual(credit(100.,60.,solos)['objective'],80.)
        with self.assertRaises(ValueError): credit(100.,60.,[70.],w_coop=2.)
        with self.assertRaises(ValueError): credit(100.,60.,[])

    def test_update_uses_team_objective_from_first_batch_and_enforces_floor(self):
        p = SetPointPolicy()
        optim = torch.optim.Adam(p.parameters(),lr=.001)
        context = encode(rows(),dict(a=0.,b=0.,ramp=0.))
        gen = torch.Generator().manual_seed(4)
        records = [dict(world=dict(seed=i%12),sample=p.sample(context,gen),
                        credit={str(h):credit(100.,80.,[110.,110.,110.]) for h in (120,300,600)},
                        execution=dict(motion_changed=True,issued_changed=True,requests=3,overrides=1))
                   for i in range(156)]
        before = p.mean.weight.detach().clone()
        with self.assertRaises(ValueError): update(p,optim,records[:30])
        report = update(p,optim,records)
        self.assertFalse(torch.equal(before,p.mean.weight))
        self.assertEqual(report['positive_C_G_share'],1.)
        self.assertEqual(report['executed_vs_issued_share'],1.)
        self.assertAlmostEqual(report['safety_override_share'],1/3)

    def test_execution_scaling_and_physical_not_command_detection(self):
        from .backend import motion_changed
        self.assertFalse(motion_changed({'a':[1.,2.]},{'a':[1.,2.]}))
        self.assertTrue(motion_changed({'a':[1.,2.]},{'a':[1.,2.01]}))
        self.assertTrue(motion_changed({'a':[1.,2.]},{}))
        sigma,count = next_sigma(SIGMA,.1,0)
        self.assertEqual(count,1)
        self.assertAlmostEqual(sigma[0],SIGMA[0]*math.sqrt(2))
        self.assertEqual(next_sigma(sigma,.9,1),(sigma,1))
        self.assertEqual(next_sigma(sigma,0.,2),(sigma,2))

    def test_seed_scope_budget_and_restricted_paths(self):
        # Load the real frozen aliases BEFORE installing scope, as a resumed host may do.
        import traffic_shepherd.c4.env as frozen_env
        self.assertEqual(budget()['total'],2000000)
        install_seed_scope()
        for world in worlds()+worlds(True):
            guard(world)
            frozen_env.check_case(world['seed'],world['demand'])
        for seed in (964000,962100,977100,981000,982048):
            with self.assertRaises(ValueError): guard(dict(seed=seed,demand=2400))
        with self.assertRaises(ValueError): guard(dict(worlds()[0],demand=2700))
        with self.assertRaises(ValueError): frozen_env.check_case(worlds()[0]['seed'],2700)
        from .protocol import checked
        with self.assertRaises(ValueError): checked('/sealed/example')

    def test_whole_batch_precharge_and_unfinished_resume(self):
        from .storage import reserve,ledger,completed
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reserve(root,[dict(id='one',sim_s=900,phase='train')])
            with self.assertRaises(ValueError): reserve(root,[dict(id='one',sim_s=900,phase='train')])
            with self.assertRaises(ValueError): reserve(root,[dict(id='two',sim_s=1650000,phase='train')])
            self.assertEqual(len(ledger(root)),1)
            self.assertIsNone(completed(root,'one'))
            (root/'jobs'/'one').mkdir(parents=True)
            with self.assertRaises(ValueError): completed(root,'one')

    def test_prefix_mismatch_stops_before_treatment_and_closes(self):
        from .runner import branch
        fake = SimpleNamespace(env=SimpleNamespace(now=300),fingerprint=lambda:dict(wrong=True),
                               close=lambda:closed.append(True))
        closed = []
        sample = dict(context=encode(rows(),dict(a=0.,b=0.,ramp=0.)),action=[[0.,0.]]*3,window=20)
        with tempfile.TemporaryDirectory() as d:
            with patch('tools.coopfree.runner.fresh',return_value=(fake,Path(d))), \
                 patch('tools.coopfree.backend.advance') as advance, \
                 patch('tools.coopfree.backend.tick',side_effect=AssertionError('treatment after failed gate')):
                with self.assertRaisesRegex(ValueError,'BEFORE'):
                    branch(Path(d),{},'new',worlds()[0],300,dict(prefix={},rows=rows()),sample,['a'])
                self.assertEqual(advance.call_count,1)
                self.assertEqual(closed,[True])

    def test_train_orchestration_checkpoint_resume_and_complete_budget(self):
        from .runner import train
        from .storage import write,ledger
        context = encode(rows(),dict(a=0.,b=0.,ramp=0.))
        state = dict(context=context,prefix={},rows=rows(),features=dict(cav_count=3,density=10,ramp=True))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for w in worlds():
                path = root/'jobs'/f's{w["seed"]}_screen'
                write(path/'RESULT.json',dict(world=w,states={'300':state},actual_sim_s=1140,
                    metrics={str(t):dict(team_delay_vs=100.) for t in (300,420,600,900)},
                    motions={str(t):{'a':[1.,2.]} for t in range(300,331)}))
                write(path/'CLOSE.json',dict(actual_sim_s=1140,charged_sim_s=1140))
            calls = []
            def fake_branch(root,binding,ident,world,t0,state,sample,members):
                calls.append(ident)
                return dict(metrics={str(h):dict(team_delay_vs=-40. if len(members)>1 else 10.)
                                     for h in (120,300,600)},motions={'301':{'a':[2.,2.]}},
                            execution=dict(requests=3,overrides=0,issued_changed=True))
            with patch('tools.coopfree.runner.branch',side_effect=fake_branch):
                report = train(root,{},1)
                self.assertEqual(report['windows'],156)
                self.assertEqual(report['worlds'],24)
                self.assertEqual(len(calls),156*4)
                self.assertEqual(sum(r['sim_s'] for r in ledger(root)),156*4*900)
                self.assertEqual(report['C_G']['mean'],70.)
                self.assertEqual(report['executed_change_share'],1.)
                resumed = train(root,{},1)
                self.assertEqual(len(calls),156*4)
                self.assertEqual(resumed['checkpoint_sha256'],report['checkpoint_sha256'])
                with self.assertRaisesRegex(ValueError,'budget'):
                    # Whole second batch would exceed 1.65M after three batches; no partial charge.
                    from .storage import reserve
                    reserve(root,[dict(id='over',sim_s=1650000,phase='train')])

    def test_observable_selection_and_world_bootstrap(self):
        from .selection import select
        from .evaluation import paired_summary
        context = encode(rows(),dict(a=0.,b=0.,ramp=0.))
        screens = [dict(world=w,states={'300':dict(context=context,features=dict(cav_count=3,density=10,ramp=True)),
                                      '360':dict(context=context,features=dict(cav_count=3,density=2,ramp=False))})
                   for w in worlds()]
        chosen,selection = select(screens,np.random.default_rng(1))
        self.assertEqual(len(chosen),156)
        self.assertEqual(selection['worlds'],24)
        self.assertGreater(sum(r['dense'] for r in chosen),len(chosen)/2)
        records = [dict(world=w,arm=arm,metrics={str(h):dict.fromkeys(
            ('team_delay_vs','collisions','teleports','ttc_below_3_vs','hard_brake'),value) for h in (600,900)})
                   for w in worlds(True) for arm,value in (('mappo',10.),('bandit',9.),('learned',8.))]
        summary = paired_summary(records)
        self.assertEqual(summary['worlds'],24)
        self.assertEqual(summary['learned_minus_comparator']['mappo']['600_team_delay_vs']['ci95'],[-2.,-2.])

    def test_detector_adapter_and_train_prior_protection(self):
        import gzip
        from .analysis import tape
        from .prior import fit
        from .protocol import canonical
        from tools.coopbehav.detectors import detect
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root/'observable.jsonl.gz'
            with gzip.open(path,'wt') as f:
                for t in (300,301,302):
                    f.write(canonical(dict(time_s=t,vehicles=rows()))+'\n')
            trace = tape(path,982024)
            detected = detect(trace)
            self.assertIsInstance(detected['events'],list)
            self.assertEqual(detected['coverage']['trace_scope'],'full_observable_fleet')
            (root/'states.jsonl').write_text(canonical(dict(world_id=982024))+'\n')
            (root/'labels.csv').write_text('state_id,D,horizon_s,available\n')
            with self.assertRaisesRegex(ValueError,'TRAIN'):
                fit(root)


class FrozenExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wms.control.policy import load_policy_fn
        checkpoint = os.environ['COOPFREE_CPU_CHECKPOINT']  # No skips: real fixture mandatory.
        cls.load = staticmethod(lambda:load_policy_fn(checkpoint,device='cpu'))
        cls.state = dict(s_m=torch.tensor([[-50.,-48.,-10.,-5.]]),v_mps=torch.tensor([[20.,18.,15.,5.]]),
                         a_mps2=torch.zeros(1,4),lane=torch.tensor([[0,1,1,2]]),
                         present=torch.ones(1,4,dtype=torch.bool),is_cav=torch.tensor([[True,True,False,True]]),
                         length_m=torch.full((1,4),5.),junction=torch.zeros(1,4,dtype=torch.bool))
        cls.env = SimpleNamespace(slot_vehicle_ids=lambda:['a','b','lead','ramp'])

    def test_off_zone_exact_mappo_commands_and_memory(self):
        from tools.leverprobe.backend import exact
        a,b = self.load(),self.load()
        for t in range(12):
            state = copy.deepcopy(self.state)
            state['s_m'] += 200+t
            r = [dict(x,s_m=x['s_m']+200+t) for x in rows()]
            x,_ = a(state,with_lanes=True)
            with inject(b,self.env,r,Controller(policy=SetPointPolicy(),deterministic=True),t,[]):
                y,_ = b(state,with_lanes=True)
            self.assertTrue(np.array_equal(x.view(np.uint64),y.view(np.uint64)))
            self.assertEqual(exact(vars(a.memory)),exact(vars(b.memory)))

    def test_actual_mappo_zero_update_first_command_parity(self):
        a,b = self.load(),self.load()
        x,_ = a(copy.deepcopy(self.state),with_lanes=True)
        controller = Controller(policy=SetPointPolicy(),deterministic=True)
        audit = []
        with inject(b,self.env,rows(),controller,300,audit):
            y,_ = b(copy.deepcopy(self.state),with_lanes=True)
        self.assertTrue(np.allclose(x,y,atol=1e-6,rtol=0.,equal_nan=True))
        self.assertEqual(len(audit),3)
        # A held absolute target is intentionally not recalibrated to the next raw MAPPO command.
        targets = copy.deepcopy(controller.targets)
        controller.apply(rows(),dict(a=2.,b=2.,ramp=2.),301)
        self.assertEqual(controller.targets,targets)

    def test_real_safety_memory_then_command_ownership(self):
        from traffic_shepherd.ownership import CommandOwnership
        from traffic_shepherd.config import load_config
        from wms.control.safety import safe_accel
        exe = self.load()
        context = encode(rows(),dict(a=0.,b=0.,ramp=0.))
        sample = dict(context=context,action=[[0.,0.]]*3,window=20)
        audit = []
        with inject(exe,self.env,rows(),Controller(sample=sample,start=300),300,audit):
            cmd,lanes = exe(copy.deepcopy(self.state),with_lanes=True)
        self.assertIsNone(lanes)
        self.assertEqual(len(audit),3)
        expected_raw = torch.zeros(1,4)
        for row in audit:
            i = self.env.slot_vehicle_ids().index(row['vid'])
            expected_raw[0,i] = row['issued_raw']
        expected = safe_accel(self.state,expected_raw,accel_max=exe.accel_max)
        for row in audit:
            i = self.env.slot_vehicle_ids().index(row['vid'])
            self.assertEqual(cmd[0,i],expected[0,i].item())
            self.assertEqual(exe.memory.accel[0,i].item(),cmd[0,i])
        calls = []
        vehicle = SimpleNamespace(getSpeedMode=lambda vid:27,
                                  slowDown=lambda *args:calls.append(args))
        owner = CommandOwnership(SimpleNamespace(vehicle=vehicle),{'a','b','ramp'},
                                 SimpleNamespace(config=load_config()),cav_speed_mode=27)
        frames = {r['vid']:SimpleNamespace(vehicle_id=r['vid'],timestamp_s=300.,speed_mps=r['v_mps'],
                    allowed_speed_mps=30.,front_coverage_m=100.,rear_coverage_m=100.,front=None,rear=None)
                  for r in rows() if r['is_cav']}
        commands = {vid:float(cmd[0,self.env.slot_vehicle_ids().index(vid)]) for vid in frames}
        result = owner.step(frames,now_s=300.,live_ids=set(self.env.slot_vehicle_ids()),wall_commands=commands)
        self.assertEqual(len(result),3)
        self.assertEqual({c[0] for c in calls},{'a','b','ramp'})
        for r in result:
            self.assertIsNone(r.fallback_reason)
            self.assertAlmostEqual(r.issued_acceleration_mps2,commands[r.vehicle_id])


if __name__ == '__main__':
    unittest.main()
