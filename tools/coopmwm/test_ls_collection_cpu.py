import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import ls_collection as ls
from .ls_gpu import capture_search


class CollectionTests(unittest.TestCase):
    def test_logging_wrapper_preserves_scores_calls_and_search_choice(self):
        calls=[]
        actions=[dict(id=str(i),profile=[[i]],duration_s=30) for i in range(8)]
        def evaluate(record,acts,k,h):
            calls.append((copy.deepcopy(acts),k,h))
            return {a['id']:[dict(team_delay=8-int(a['id']))]*k for a in acts}
        def search(record,group,policy,bounds,evaluate,regime):
            scores=evaluate(record,actions,8,420)
            return dict(accepted=True,plan=actions[-1],scores=scores)
        baseline=search({},None,None,None,evaluate,'ordinary');first=copy.deepcopy(calls);calls.clear()
        result=capture_search(search)({},None,None,None,evaluate,'ordinary')
        self.assertEqual(result.pop('collection_top8'),list(reversed(actions)))
        self.assertEqual(result,baseline);self.assertEqual(calls,first)

    def test_snapshot_guide_never_waits_or_executes_and_covers_all_groups(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sample=dict(context=dict(members=['a','b']),own_action=[[0.],[0.]])
            controller=SimpleNamespace(active_plans=[],guide=None)
            job=dict(id='lscollect1_B3_986000',world=dict(seed=986000),policy_state={},
                path=str(root/'episode'),ls_collect=dict(root=str(root)))
            snap=dict(samples={},prefixes={},t0=300)
            with patch('tools.coopmwm.mppi_search.group_components',return_value=[dict(members=['a','b']),dict(members=['c','d'])]), \
                 patch('tools.coopmwm.sumo_verify.capture',return_value=snap), \
                 patch('tools.coopmwm.exploration.regime',return_value='ordinary'), \
                 patch('time.sleep',side_effect=AssertionError('collection blocked')):
                ls.install(controller,job,{},torch.Generator())
                self.assertIsNone(controller.guide([],{},300,sample))
            records=[json.loads(gzip.decompress(p.read_bytes())) for p in (root/'pending').glob('*.gz')]
            self.assertEqual(len(records),2);self.assertEqual({r['group_index'] for r in records},{0,1})
            self.assertTrue(all(r['record']['history']==[] for r in records))
            self.assertFalse(sample['behaviour']['executed']);self.assertEqual(controller.active_plans,[])

    def test_queue_order_spreads_worlds_before_later_groups(self):
        xs=[dict(time_s=300,group_index=i,world_id=w,record=dict(state_id=str(w)+str(i)))
            for w in (986002,986001) for i in (1,0)]
        self.assertEqual([(x['group_index'],x['world_id']) for x in sorted(xs,key=ls.envelope_order)],
            [(0,986001),(0,986002),(1,986001),(1,986002)])

    def test_collection_hook_is_opt_in_and_precedes_history_priming(self):
        from . import accel_plant
        import inspect
        source=inspect.getsource(accel_plant.episode)
        self.assertIn("if job.get('ls_collect'):",source)
        self.assertLess(source.index('install(controller,job,prefixes,generator)'),source.index('controller.guide.prime'))


if __name__=='__main__':unittest.main()
