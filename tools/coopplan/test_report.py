"""Analysis ABI/whole-world checks; no simulator calls or filesystem writes."""
from copy import deepcopy
import unittest

from .protocol import worlds
from .report import build


def complete_panel():
    records=[]
    for world in worlds():
        def metrics(j):
            return dict(team_delay_vs=j,collisions=0.,teleports=0.,ttc_below_3_vs=0.,hard_brake=0.)
        records.append(dict(world=world,baseline={c:metrics(10.) for c in ('600','900')},
            planner={c:metrics(8.) for c in ('600','900')},
            history=[dict(executed_action=dict(id='O0')) for _ in range(12)],credits=[],
            # Real COOPBEHAV returns events + coverage/exposure, never a bare event list.
            observational_behaviours=dict(events=[dict(behaviour='yield_to_merge')],coverage={}),
            causal_behaviours=[dict(events=[dict(behaviour='gap_creation',evidence='paired_policy_mean_support')])]))
    return [dict(records=records,complete=True,stopped=False,incomplete=[],branches=[],
                 design_sha256='53b0edf4ea8e6fc72165691727e923505ed6a805af9c0294efd63f2ab3389a5f',
                 source_sha256s={'test-source':'bound'},charged=0,actual=0,requests=0,overrides=0)]


class ReportTests(unittest.TestCase):
    def test_structured_detector_output_and_paired_world_ci(self):
        _,rows,credits,summary=build(complete_panel())
        self.assertEqual(len(rows),12)
        self.assertEqual(credits,[])
        self.assertEqual(summary['observational_behaviours'],{'yield_to_merge':12})
        self.assertEqual(summary['sampled_behaviour_evidence'],{'gap_creation/paired_policy_mean_support':12})
        self.assertEqual(summary['savings']['900'],dict(n=12,mean=2.,lower=2.,upper=2.))

    def test_incomplete_panel_cannot_be_reported_as_complete(self):
        panel=complete_panel()
        for kind in ('missing_world','stopped','missing_close'):
            p=deepcopy(panel)
            if kind=='missing_world':p[0]['records'].pop()
            elif kind=='stopped':p[0]['stopped']=True
            else:p[0]['incomplete']=['unfinished-owned-attempt']
            with self.assertRaises(ValueError):build(p)


if __name__ == '__main__':
    unittest.main()
