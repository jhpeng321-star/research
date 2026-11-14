import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
from . import parallel_groups as pg,quota_admission as q
from .warm_history import ObservableHistory
from .warm_spool import immutable
from .protocol import digest


class ParallelGroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('requests','responses','guided_starts/u002'):(self.root/name).mkdir(parents=True)
        self.job=dict(queue_root=str(self.root),id='warm_u002_B_986000',update=2,
            N_exec=10,arm='B',world=dict(seed=986000),gpu_deadline=time.time()+10)
        self.groups=[dict(members=['a','b']),dict(members=['c','d']),dict(members=['e','f'])]
        self.sample=dict(context=dict(members=['a','b','c','d','e','f']),own_action=[[0.]]*6)

    def execute(self,accepts=(False,True,True)):
        seen=[]
        def publish(path,value):
            immutable(path,value)
            if path.parent.name!='requests':return
            seen.append(value)
            if len(seen)==len(self.groups):
                # Reverse completion order; control choice must remain stable.
                for i,r in reversed(list(enumerate(seen))):
                    result=dict(record_sha256=digest(r),accepted=accepts[i],candidates_searched=100+i)
                    if accepts[i]:result.update(plan=dict(id=str(i),members=r['group'],duration_s=60),option='MPPI',expected_saving=1000*i)
                    immutable(self.root/'responses'/(r['state_id']+'.json'),result)
        with patch.object(pg,'immutable',side_effect=publish),patch.object(pg,'assert_running'),patch.object(pg.time,'sleep',side_effect=AssertionError('all groups must publish before waiting')):
            result=pg.query(self.job,torch.Generator().manual_seed(0),ObservableHistory(),[],{},300,self.sample,self.groups,'ordinary',[1,2,3,4])
        return result,seen

    def test_submits_every_group_before_wait_and_selects_by_inherited_order(self):
        result,seen=self.execute()
        self.assertEqual([r['group'] for r in seen],[g['members'] for g in self.groups])
        self.assertEqual(result['plan']['id'],'1')
        self.assertEqual([r['selected'] for r in result['parallel_queries']],[False,True,False])
        self.assertEqual(result['behaviour']['parallel_query_counts'],dict(queries=3,accepted=2,withdrawn_pending=0,candidates_searched=303))
        self.assertEqual(q.state(self.root,2),dict(executed=0,inflight=1))
        q.resolved(result['quota_reservation'],True)
        immutable(self.root/'guided_starts/u002'/(result['record_sha256']+'.json'),dict(job=self.job['id']))
        self.assertEqual(q.state(self.root,2),dict(executed=1,inflight=0))

    def test_all_declines_release_window_without_phantom_executions(self):
        result,_=self.execute((False,False,False))
        self.assertFalse(result['accepted'])
        self.assertEqual(q.state(self.root,2),dict(executed=0,inflight=0))
        q.resolved(result['quota_reservation'],False)

    def test_batch_admissions_are_real_blocking_queries(self):
        result,seen=self.execute()
        for r in seen:
            admission=json.loads((self.root/'quota_admissions'/(r['state_id']+'.json')).read_text())
            self.assertTrue(admission['blocking'])
        self.assertEqual(result['behaviour']['sample_probability'],.95)

    def test_executed_minimum_uses_unchanged_path_without_rng_draw(self):
        for i in range(10):immutable(self.root/'guided_starts/u002'/f'{i}.json',dict(job=self.job['id']))
        with patch('torch.rand',side_effect=AssertionError('must not consume draw')):
            self.assertIsNone(pg.query(self.job,None,None,[],{},300,self.sample,self.groups,'ordinary',[1]))

    def test_one_query_draw_preserves_original_exact_grade_schedule(self):
        from .exact_grade import guide_draws
        result,_=self.execute()
        self.assertEqual(guide_draws(dict(sample=self.sample,proposal=result),'B'),['rand','rand'])
        expected=torch.Generator().manual_seed(0);torch.rand((),generator=expected)
        actual=torch.Generator().manual_seed(0)
        # Another instant, all requests declined, still one draw for all groups.
        calls=[]
        def draw():calls.append(1);return float(torch.rand((),generator=actual))
        q.choose_group_batch(self.root,self.job,310,draw,0,4)
        self.assertEqual(calls,[1]);self.assertTrue(torch.equal(actual.get_state(),expected.get_state()))

    def test_requests_retain_warm_history_policy_library_and_other_commitments(self):
        self.job.update(policy_state=dict(x=[1]),search_learning=dict(arm='B',seed=20261101))
        self.sample['committed_plans']=[dict(plan=dict(id='other',members=['z']),elapsed_s=5)]
        before=copy.deepcopy(self.sample['committed_plans']);result,seen=self.execute()
        for r in seen:
            self.assertEqual(r['committed_plans'],before)
            self.assertEqual(r['history'],[])
            self.assertEqual(r['policy_state'],self.job['policy_state'])
            self.assertEqual(r['search_learning'],self.job['search_learning'])
        self.assertEqual(self.sample['committed_plans'],before)

    def test_routing_counts_queries_but_only_one_execution_and_one_wait(self):
        from .accel_runner import routing
        result,_=self.execute();b=result['behaviour'];b.update(executed=True,plan_start=True)
        report=routing([dict(steps=[dict(sample=dict(behaviour=b))])])
        self.assertEqual((report['queries'],report['accepted'],report['executed']),(3,2,1))
        self.assertEqual(len(report['durations']),1)
        self.assertEqual(report['candidates_searched'],303)
        self.assertEqual(report['query_wait_wall_s'],result['query_wall_s'])

    def test_first_accepted_group_returns_without_waiting_for_later_groups(self):
        """HARDEN-3 stall: an accepted inherited-order answer must reach verification at once."""
        seen=[]
        def publish(path,value):
            immutable(path,value)
            if path.parent.name!='requests':return
            seen.append(value)
            if len(seen)==1:
                result=dict(record_sha256=digest(value),accepted=True,candidates_searched=7,
                    plan=dict(id='0',members=value['group'],duration_s=60),option='MPPI',expected_saving=5.)
                immutable(self.root/'responses'/(value['state_id']+'.json'),result)
        with patch.object(pg,'immutable',side_effect=publish),patch.object(pg,'assert_running'),                patch.object(pg.time,'sleep',side_effect=AssertionError('must not wait for later groups')):
            result=pg.query(self.job,torch.Generator().manual_seed(0),ObservableHistory(),[],{},300,
                self.sample,self.groups,'ordinary',[1,2,3,4])
        self.assertTrue(result['accepted']);self.assertEqual(result['plan']['id'],'0')
        self.assertEqual([r['pending'] for r in result['parallel_queries']],[False,True,True])
        self.assertEqual(result['behaviour']['parallel_query_counts'],
            dict(queries=3,accepted=1,withdrawn_pending=2,candidates_searched=7))
        gone=sorted(p.stem for p in (self.root/'request_withdrawals').glob('*.json'))
        self.assertEqual(gone,sorted(r['state_id'] for r in seen[1:]))
        # The single window reservation stays potentially executable until resolved.
        self.assertEqual(q.state(self.root,2),dict(executed=0,inflight=1))
        q.resolved(result['quota_reservation'],False)
        self.assertEqual(q.state(self.root,2),dict(executed=0,inflight=0))

    def test_broker_orders_breadth_first_and_declines_withdrawn(self):
        from . import capacity_broker as cb
        from . import exploration
        records=[]
        for world in (986002,986001):
            job=dict(self.job,id=f'warm_u002_B_{world}',world=dict(seed=world))
            _,_,_,res=q.choose_group_batch(self.root,job,300,lambda:0.,0,4)
            batch=[]
            for g in self.groups:
                r=dict(world_id=world,time_s=300,rows=[],group=g['members']);r['state_id']=digest(dict(record=r,job=job['id']))
                batch.append(r)
            q.bind_group_batch(res,batch)
            for r in batch:immutable(self.root/'requests'/(r['state_id']+'.json'),r)
            records.append(batch)
            time.sleep(.01)
        with patch.object(exploration,'regime',return_value='ordinary'):
            order=[r['state_id'] for _,r in cb.pending(self.root)]
        self.assertEqual(order,[batch[i]['state_id'] for i in range(3) for batch in records])
        pg.withdraw(self.root,records[0][2],'warm_u002_B_986002')
        items=[('plan',r) for batch in records for r in batch]
        blocking={r['state_id'] for batch in records for r in batch}
        declines,_=cb.capacity_declines(items,{},['s1','s2'],blocking,False,cb.withdrawn(self.root))
        self.assertEqual([r['state_id'] for r in declines],[records[0][2]['state_id']])


if __name__=='__main__':unittest.main()
