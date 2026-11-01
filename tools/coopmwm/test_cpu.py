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
from .credit import algebra,paired,ramp,bonus,calibration
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

    def test_frozen_batched_K8_adapter_H90_structure(self):
        from .wm import adapted_futures
        ns=adapted_futures()
        self.assertIn('futures',ns)
        self.assertIn(180,ns['futures'].__code__.co_consts)
        self.assertIn((90,),ns['futures'].__code__.co_consts)

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


if __name__=='__main__':unittest.main()
