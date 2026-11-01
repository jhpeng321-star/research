import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import numpy as np
import torch
from tools.coopnative.test_cpu import rows
from .acceleration import requests,Controller,Policy
from . import diagnostic_repair as d
from .accel_storage import write,read


class DiagnosticRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def sample(self):return dict(context=dict(members=['live','departed']),action=[[1.],[2.]],own_action=[[1.],[2.]])

    def test_only_counterfactual_replay_may_ignore_departed_commands(self):
        sample=self.sample()
        with self.assertRaises(KeyError):requests(sample,{'live':3.})
        result=requests(sample,{'live':3.},ignore_departed=True)
        self.assertEqual(result,dict(live=dict(requested_raw=4.,acceleration_residual=1.)))

    def test_common_roster_requests_identical_with_guard_enabled(self):
        sample=self.sample();sample['absolute_accels']=dict(live=-2.,departed=1.)
        commands=dict(live=3.,departed=4.)
        self.assertEqual(requests(sample,commands),requests(sample,commands,ignore_departed=True))

    def test_controller_replay_is_strict_before_branch_and_safe_after(self):
        branch=dict(time_s=300,group=['live','departed'],revert=['departed'],until=360,mappo_background=True)
        c=Controller(Policy(),torch.Generator(),replay={'299':self.sample(),'300':self.sample()},branch=branch)
        with self.assertRaises(KeyError):c.apply(rows(),dict(live=0.),299)
        self.assertEqual(c.apply(rows(),dict(live=0.),300)['live']['requested_raw'],1.)

    def test_complete_group_has_n_plus_two_branches(self):
        for n in range(2,7):
            group=[str(i) for i in range(n)];bs=d.branches(group)
            self.assertEqual(len(bs),n+2);self.assertEqual(bs[1],('all',group))
            self.assertEqual([v for _,v in bs[2:]],[[v] for v in group])

    def fixture(self):
        step=dict(time_s=300,sample=self.sample(),proposal=dict(plan=dict(members=['live','departed'],duration_s=60)))
        ep=dict(id='warm_u001_B_986001',world=dict(seed=986001),prefixes={'300':'same'},steps=[step])
        branch=dict(world=ep['world'],prefixes={'300':'same'},actual_sim_s=900,
            horizon_totals={str(h):dict(team_delay_vs=1.) for h in (60,120,300,600)})
        return ep,step,branch

    def test_completed_branch_adoption_needs_all_horizons_and_exact_prefix(self):
        ep,_,branch=self.fixture();self.assertTrue(d.valid_adoption(branch,ep,300))
        bad=copy.deepcopy(branch);bad['prefixes']['300']='different'
        with self.assertRaises(ValueError):d.valid_adoption(bad,ep,300)
        bad=copy.deepcopy(branch);del bad['horizon_totals']['600']
        with self.assertRaises(ValueError):d.valid_adoption(bad,ep,300)

    def test_completed_branches_adopted_without_physics_or_duplicate_charge(self):
        ep,step,branch=self.fixture()
        write(self.root/'jobs/planexact000_joint_986001/RESULT.json',branch)
        b=dict(runtime='runtime',checkpoint='checkpoint',accel_bounds=[-3.,2.6],gpu_deadline=999.)
        todo,adopted,manifest=d.jobs(self.root,b,Mock(state_dict=lambda:{}),ep,step,0)
        self.assertEqual(len(todo),3);self.assertEqual(set(adopted),{'joint'})
        self.assertEqual(sum(j['sim_s'] for j in todo),2700)
        self.assertEqual(len(manifest),4)
        write(Path(todo[0]['path'])/'CLAIM.json',dict(pid=1))
        with self.assertRaises(AssertionError):d.jobs(self.root,b,Mock(state_dict=lambda:{}),ep,step,0)

    def test_selection_is_identical_to_registered_training_rng_and_ignores_grade(self):
        eps=[dict(steps=[dict(time_s=300+i,sample=dict(behaviour=dict(plan_start=True)),grade=i-5) for i in range(10)])]
        rng=np.random.default_rng(20261101)
        expected=np.random.default_rng(20261101).permutation(10)[:3].tolist()
        actual=[s['time_s']-300 for _,s in d.selected(eps,rng,3)]
        self.assertEqual(actual,expected)
        for s in eps[0]['steps']:s['grade']*=-100
        self.assertEqual([s['time_s']-300 for _,s in d.selected(eps,np.random.default_rng(20261101),3)],expected)

    def test_whole_missing_group_blocked_before_any_charge_or_launch(self):
        ep,step,_=self.fixture();write(self.root/'ACCEL_EXACT_PLAN_000_FAILED.json',dict(error="KeyError('departed')"))
        b=dict(budget=dict(phases=dict(discovery=200000)))
        with patch('tools.coopmwm.accel_learner.load',return_value=(Mock(),None,None,dict(update=0))),patch.object(d,'jobs',return_value=([dict(sim_s=900)]*2,{},[])),patch.object(d,'ledger',return_value=[dict(phase='discovery',sim_s=199000)]),patch.object(d,'reserve') as reserve:
            self.assertFalse(d.repair(self.root,b,ep,step,0,1));reserve.assert_not_called()
        self.assertFalse(read(self.root/'DIAGNOSTIC_REPAIR_000_BLOCKED.json')['partial_group_spending'])


if __name__=='__main__':unittest.main()
