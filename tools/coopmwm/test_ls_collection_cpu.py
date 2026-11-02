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
    def test_gpu_exit_race_never_signals_empty_argv_or_reused_pid(self):
        from .ls_gpu import live_identity
        old=dict(boot_id='boot',start_ticks='123',argv='python worker ')
        stat=['0']*22;stat[2]='S';stat[21]='123'
        self.assertTrue(live_identity('boot',stat,old['argv'],old))
        self.assertFalse(live_identity('boot',stat,'',old))
        self.assertFalse(live_identity('boot',None,'',old))
        with self.assertRaises(AssertionError):live_identity('newboot',stat,'',old)
        stat[21]='124'
        with self.assertRaises(AssertionError):live_identity('boot',stat,'',old)

    def test_collection_broker_requires_same_live_qualified_owner_on_both_receipts(self):
        from .ls_broker import owner_matches
        owner=dict(pid=1,start_ticks='12',boot_id='boot')
        status=dict(alive=True,ready=True,failed=False,receipts={
            'SPOOL_OWNER.json':dict(owner),'SERVICE_OWNER.json':dict(owner)})
        self.assertTrue(owner_matches(status,owner))
        status['receipts']['SERVICE_OWNER.json']['pid']=2
        self.assertFalse(owner_matches(status,owner))

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

    def test_queue_metadata_drops_large_payload_and_coverage_rotates_worlds_times(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'q.json.gz'
            ls.compressed_json(p,dict(world_id=986001,time_s=600,group_index=0,
                episode_path=d,record=dict(state_id='q',rows=['large'],history=['large'])))
            e=ls.queue_metadata(p)
            self.assertNotIn('record',e);self.assertNotIn('history',e)
            self.assertEqual(e['key'],'q')
        xs=[dict(world_id=w,time_s=t,group_index=0,key=f'{w}_{t}')
            for w in (986000,986001) for t in (300,600,900)]
        counts={};chosen=[]
        for _ in range(6):
            e=min(xs,key=lambda e:ls.coverage_order(e,counts));xs.remove(e)
            chosen.append((e['world_id'],e['time_s']))
            counts[e['world_id']]=counts.get(e['world_id'],0)+1
        self.assertEqual(chosen,[(986000,300),(986001,300),(986000,600),
            (986001,600),(986000,900),(986001,900)])

    def test_label_list8_retains_negative_predictions_without_extra_wm_calls(self):
        from .mppi_search import search,group_components
        rows=[dict(vid=str(i),s_m=20*i,v_mps=10.,lane=i,is_cav=True) for i in range(2)]
        group=group_components(rows)[0];calls=[]
        def evaluate(record,acts,k,h):
            calls.append((copy.deepcopy(acts),k,h))
            return {a['id']:[100. if a['id'] in ('policy','MAPPO') else 110.+j]*k
                for j,a in enumerate(acts)}
        args=({},group,[[0.]*300]*2,(-3.,2.),evaluate,'ordinary')
        search(*args,seed=1);baseline=copy.deepcopy(calls);calls.clear()
        result=capture_search(search,label_list8=True)(*args,seed=1)
        self.assertEqual(calls,baseline)
        self.assertEqual(len(result['topk']),8)
        self.assertTrue(all(p['saving_vs_mappo']<0 for p in result['collection_top8']))
        self.assertEqual(result['config']['top_k'],8)
        self.assertEqual(result['final'],dict(K=8,H=420))

    def test_collection_hook_is_opt_in_and_precedes_history_priming(self):
        from . import accel_plant
        import inspect
        source=inspect.getsource(accel_plant.episode)
        self.assertIn("if job.get('ls_collect'):",source)
        self.assertLess(source.index('install(controller,job,prefixes,generator)'),source.index('controller.guide.prime'))


if __name__=='__main__':unittest.main()
