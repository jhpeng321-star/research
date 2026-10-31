"""Analysis checks for correlated actions, complete coverage and unchanged strict-win gate."""
from copy import deepcopy
import unittest

from .extension_report import credit_ci,analyze
from .planner import TIMES


def record():
    history=[]
    for t in TIMES:
        group=['a','b'] if t==300 else []
        name='O3' if group else 'O0'
        decision=dict(selected=name,eligible=['O3'] if group else [],
                      expected_delay={'O0':10.,'O3':5.},probability_beats={'O0':0.,'O3':1.})
        history.append(dict(time_s=t,group=group,executed_action=dict(id=name,members=group),decision=decision))
    clocks={k:dict(team_delay_vs=50.,collisions=0,teleports=0,ttc_below_3_vs=2.,hard_brake=1.) for k in ('600','900')}
    return dict(world=dict(seed=1,cell=0,penetration=.05,demand=1900),baseline=clocks,planner=deepcopy(clocks),
                history=history,credits=[dict(time_s=300,action=history[0]['executed_action'],members=['a','b'],
                                             C_G=3.,D_G=5.,member_alone_D=[1.,1.])],
                causal_behaviours=[dict(time_s=300,events=[])],observational_behaviours=dict(events=[]))


class ExtensionAnalysisTests(unittest.TestCase):
    def test_credit_bootstrap_resamples_worlds_not_actions(self):
        records=[dict(credits=[dict(C_G=0.)]*100),dict(credits=[dict(C_G=100.)])]
        c=credit_ci(records)
        self.assertEqual(c['mean'],100/101)
        self.assertEqual(c['lower'],0.)
        self.assertEqual(c['upper'],100.)
        self.assertEqual(c['worlds'],2)
        self.assertEqual(c['events'],101)

    def test_no_action_worlds_stay_in_resampling_frame(self):
        c=credit_ci([dict(credits=[]),dict(credits=[dict(C_G=3.)])])
        self.assertEqual(c['worlds'],2)
        self.assertEqual(c['active_worlds'],1)
        self.assertGreater(c['zero_event_bootstrap_draws'],0)
        self.assertEqual(c['mean'],3.)

    def test_all_group_coverage_and_credit_identity(self):
        r=record();rows,credits,s=analyze([r])
        self.assertEqual(s['all_group_credits'],1)
        self.assertEqual(s['executed_actions'],dict(O3=1,O0=11))
        self.assertEqual(s['savings']['900']['mean'],0.)
        r['credits']=[]
        with self.assertRaises(AssertionError):analyze([r])

    def test_duplicate_credit_is_rejected(self):
        r=record();r['credits']*=2
        with self.assertRaises(AssertionError):analyze([r])

    def test_nonpositive_joint_saving_is_not_execution_evidence(self):
        r=record();r['credits'][0].update(D_G=0.,C_G=-2.)
        with self.assertRaises(AssertionError):analyze([r])

    def test_causal_detector_coverage_is_complete(self):
        r=record();r['causal_behaviours']=[]
        with self.assertRaises(AssertionError):analyze([r])


if __name__=='__main__':unittest.main()
