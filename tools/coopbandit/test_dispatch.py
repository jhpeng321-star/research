"""Partition accounting and restricted transport boundaries, without SSH or SUMO."""
import unittest
from unittest.mock import patch
from .dispatch import make_plan, fetch
from .protocol import HIGH, worlds, DATA_CAP


class DispatchTests(unittest.TestCase):
    def test_global_unique_exact_allocation(self):
        selected=[]
        for cell in range(12):
            selected.extend(dict(world=w) for w in [w for w in worlds() if w['cell']==cell][:(9 if cell in HIGH else 3)])
        solo=[e['world']['seed'] for c in HIGH for e in [e for e in selected if e['world']['cell']==c][:2]]
        plan=make_plan(dict(worlds=selected,solo_worlds=solo))
        attempts=[(h,k,v) for h,s in plan['shards'].items() for k,v in s['attempts'].items()]
        self.assertEqual(len(attempts),1360)
        self.assertEqual(len({k for h,k,v in attempts}),1360)
        self.assertEqual(sum(v['sim_s'] for h,k,v in attempts)+102000,DATA_CAP)
        self.assertLessEqual(plan['conservative_total'],1600000)
        owners={}
        for h,s in plan['shards'].items():
            for w in s['worlds']:
                self.assertNotIn(w['seed'],owners)
                owners[w['seed']]=h
        self.assertEqual(len(owners),60)
        self.assertEqual(len(plan['shards']['gpuR']['worlds']),2)

    def test_forbidden_transport_rejected_before_ssh(self):
        with patch('tools.coopbandit.dispatch.remote',side_effect=AssertionError('SSH disabled')):
            for name in ('../RESULT.json','jobs/holdout/RESULT.json','jobs/s981000_screen/scenarios.xml'):
                with self.assertRaises(ValueError): fetch('cpuQ','/tmp',[name])


if __name__=='__main__': unittest.main()
