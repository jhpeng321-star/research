"""Independent bootstrap and complete-panel negative/corruption checks."""
from copy import deepcopy
import unittest

import numpy as np

from tools.coopplan.predictor import ObservableState
from tools.coopplan.planner import TIMES
from .data import read
from .test_live import fixture
from .portable import Student
from .live import panel
from .report import build, ci, credit_ci


def zero_panel(policy):
    records,charges = [],[]
    metrics = dict(team_delay_vs=0.,collisions=0.,teleports=0.,ttc_below_3_vs=0.,hard_brake=0.)
    clocks = {'600':metrics.copy(),'900':metrics.copy()}
    for w in panel():
        seed = w['seed'];model = Student(policy);history = []
        for t in TIMES:
            e = model.step(ObservableState(t,tuple(),f's{seed}_t{t}'))
            e.update(proposed_action=deepcopy(e['executed_action']),budget_pruned=False)
            history.append(e)
        base = dict(world=w,clocks=deepcopy(clocks),actual_sim_s=1200,final_prefix=dict(seed=seed))
        student = dict(**deepcopy(base),history=history,credits=[],warmup_pass=True,
                       all_O0_bitwise_pass=True,credit_sim_s=0,requests=0,overrides=0)
        records.append(dict(world=w,baseline=base,student=student))
        for arm in ('mappo','student'):
            charges.append(dict(id=f's{seed}_{arm}',sim_s=1200,kind='paired',seed=seed))
    charges.append(dict(id='mechanics_all_O0',sim_s=1200,kind='mechanics',seed=panel()[0]['seed']))
    closed = [dict(id=r['id'],actual_sim_s=1200,charged_sim_s=1200) for r in charges]
    return dict(complete=True,stopped=False,incomplete=[],worlds=24,mechanics=dict(pass_=True),
                records=records,actual=58800,charged=58800,charges=charges,closed=closed,
                branch_results={},reservations={})


class Tests(unittest.TestCase):
    def setUp(self):
        self.policy = read(fixture()/'POLICY.json')
        self.receipt = zero_panel(self.policy)

    def test_zero_effect_complete_panel(self):
        summary,rows,credits = build(self.receipt,self.policy)
        self.assertEqual(summary['savings']['600']['ci95'],[0.,0.])
        self.assertEqual(summary['savings']['900']['mean'],0.)
        self.assertEqual(summary['executed_actions'],{'O0':288})
        self.assertIsNone(summary['credit'])
        self.assertEqual(summary['all_O0_bitwise_worlds'],24)
        self.assertEqual(len(rows),24);self.assertEqual(credits,[])

    def test_no_partial_panel_headline(self):
        self.receipt['worlds'] = 23
        with self.assertRaisesRegex(ValueError,'complete24'):
            build(self.receipt,self.policy)

    def test_no_missing_or_unclosed_job(self):
        self.receipt['closed'].pop()
        with self.assertRaisesRegex(ValueError,'unclosed'):
            build(self.receipt,self.policy)

    def test_zero_policy_clock_drift_rejected(self):
        self.receipt['records'][0]['student']['clocks']['600']['team_delay_vs'] = 1.
        with self.assertRaisesRegex(ValueError,'bitwise'):
            build(self.receipt,self.policy)

    def test_altered_student_action_rejected(self):
        self.receipt['records'][0]['student']['history'][0]['executed_action'] = dict(id='O1',members=['a','b'])
        with self.assertRaisesRegex(ValueError,'frozen observable'):
            build(self.receipt,self.policy)

    def test_unspent_reservation_rejected(self):
        self.receipt['reservations']['unexpected'] = dict(cost=1680)
        with self.assertRaisesRegex(ValueError,'outstanding'):
            build(self.receipt,self.policy)

    def test_bootstrap_independently_recomputed(self):
        x = np.arange(24.)-12.
        got = ci(x)
        rng = np.random.default_rng(20261101)
        independent = np.array([x[rng.integers(0,24,24)].sum()/24 for _ in range(10000)])
        np.testing.assert_array_equal(got['ci95'],np.quantile(independent,[.025,.975]))
        self.assertEqual(got['mean'],x.mean())

    def test_credit_cluster_includes_zero_action_worlds(self):
        events = [[dict(C_G=-1.),dict(C_G=3.)],[],[dict(C_G=5.)]]
        got = credit_ci(events)
        self.assertEqual(got['mean'],7/3)
        self.assertEqual(got['positive'],2)
        self.assertEqual(got['active_worlds'],2)
        self.assertGreater(got['zero_event_draws'],0)


if __name__ == '__main__':unittest.main()
