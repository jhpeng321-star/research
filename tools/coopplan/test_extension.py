"""CPU-only budget obligations, complete twin coverage, and unchanged decision semantics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .protocol import extension_worlds,extension_budget
from .predictor import ObservableState,SumoOraclePredictor
from .planner import Planner
from .storage import write,charge,read,ledger_lock
from .quota import outstanding,reserve_actions,release_credit
from .test_cpu import ROWS


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'jobs').mkdir()
        write(self.root/'ALLOCATION.json',dict(cap=10000,reserved=dict(s1_mappo=1200,s1_planner=1200)))
        self.verify = patch('tools.coopplan.storage.verify',return_value={})
        self.verify.start()
        self.state = ObservableState(300,tuple(ROWS),'s1_t300')
        self.actions = [dict(id=n,members=[] if n=='O0' else ['a','b']) for n in ('O0','O1','O2')]

    def tearDown(self):
        self.verify.stop();self.temp.cleanup()

    def test_fresh_panel_and_cap(self):
        ws = extension_worlds()
        self.assertEqual(len(ws),24)
        self.assertEqual(len({w['seed'] for w in ws}),24)
        self.assertEqual({w['seed'] for w in ws},set(range(983200,983224)))
        self.assertEqual([sum(w['cell']==c for w in ws) for c in range(12)],[2]*12)
        self.assertEqual(220980+sum(extension_budget().values()),800000)

    def test_reserve_both_solos_before_predicting(self):
        actions,receipt=reserve_actions(self.root,self.state,self.actions)
        self.assertEqual(len(actions),3)
        pending=outstanding(self.root)
        self.assertEqual(pending['s1_t300_CG'],840)
        self.assertEqual(sum(pending.values()),2400+5*420)
        self.assertEqual(receipt['pruned'],[])
        for a in actions:charge(self.root,'s1_t300_'+a['id'],420)
        # Each member consumes exactly half the pre-reserved generic winning-group pool.
        charge(self.root,'s1_t300_O1_solo0',420)
        charge(self.root,'s1_t300_O1_solo1',420)
        self.assertEqual(outstanding(self.root)['s1_t300_CG'],0)
        with self.assertRaises(ValueError):charge(self.root,'s1_t300_O2_solo0',420)

    def test_no_unreserved_physics_or_duplicate_attempt(self):
        with self.assertRaises(ValueError):charge(self.root,'s1_t300_O0',420)
        charge(self.root,'s1_mappo',1200)
        with self.assertRaises(ValueError):charge(self.root,'s1_mappo',1200)
        self.assertEqual(sum(outstanding(self.root).values()),1200)

    def test_rejected_group_releases_solo_obligation(self):
        reserve_actions(self.root,self.state,self.actions)
        for a in self.actions:charge(self.root,'s1_t300_'+a['id'],420)
        release_credit(self.root,'s1_t300_CG')
        self.assertEqual(sum(outstanding(self.root).values()),2400)
        # A later decision may use freed solos; charged physics is never refunded.
        charge(self.root,'s1_mappo',1200)
        self.assertEqual(sum(outstanding(self.root).values()),1200)

    def test_hard_cap_preserves_all_future_pairs(self):
        with patch('tools.coopplan.quota.read',side_effect=lambda p:dict(cap=4079,reserved=read(p)['reserved']) if p.name=='ALLOCATION.json' else read(p)):
            actions,receipt=reserve_actions(self.root,self.state,self.actions)
        self.assertEqual(actions,[])
        self.assertEqual(receipt['reason'],'budget_pruned')
        self.assertEqual(sum(outstanding(self.root).values()),2400)

    def test_affordable_prefix_prunes_only_candidates(self):
        with patch('tools.coopplan.quota.read',side_effect=lambda p:dict(cap=4080,reserved=read(p)['reserved']) if p.name=='ALLOCATION.json' else read(p)):
            actions,receipt=reserve_actions(self.root,self.state,self.actions)
        self.assertEqual([a['id'] for a in actions],['O0','O1'])
        self.assertEqual(receipt['pruned'],['O2'])
        self.assertEqual(sum(outstanding(self.root).values()),4080)

    def test_skip_default_does_not_call_predictor(self):
        def fail(*args):raise AssertionError('no physics for a provably default-only decision')
        planner=Planner(SumoOraclePredictor(fail),admit=lambda s,a:([],dict(reason='budget_pruned')))
        event,prediction=planner.step(self.state)
        self.assertIsNone(prediction)
        self.assertEqual(event['executed_action']['id'],'O0')
        self.assertFalse(event['group'])

    def test_full_admission_preserves_selection(self):
        def run(s,a,h):return dict(team_delay=10. if a['id']=='O0' else 9.,breakdown=False)
        plain=Planner(SumoOraclePredictor(run))
        bounded=Planner(SumoOraclePredictor(run),admit=lambda s,a:(a,dict(reason='full')))
        a,p=plain.step(self.state);b,q=bounded.step(self.state)
        self.assertEqual(a['executed_action'],b['executed_action'])
        self.assertEqual(a['decision'],b['decision'])

    def test_windows_lock_does_not_write_before_acquisition(self):
        from types import SimpleNamespace
        calls=[]
        def locking(fd,mode,size):
            self.assertEqual((self.root/'ledger.lock').stat().st_size,0)
            calls.append((mode,size))
        windows=SimpleNamespace(LK_LOCK=1,LK_UNLCK=2,locking=locking)
        with patch('tools.coopplan.storage.os',SimpleNamespace(name='nt')):
            with patch.dict('sys.modules',msvcrt=windows):
                with ledger_lock(self.root):
                    self.assertEqual(calls,[(windows.LK_LOCK,1)])
        self.assertEqual(calls,[(windows.LK_LOCK,1),(windows.LK_UNLCK,1)])

    def test_parallel_worlds_cannot_steal_reserved_twins_or_pairs(self):
        from concurrent.futures import ThreadPoolExecutor
        states=[ObservableState(300,tuple(ROWS),f's{i}_t300') for i in range(4)]
        with ThreadPoolExecutor(4) as pool:
            results=list(pool.map(lambda s:reserve_actions(self.root,s,self.actions),states))
        pending=outstanding(self.root)
        self.assertLessEqual(sum(pending.values()),10000)
        self.assertEqual(pending['s1_mappo'],1200)
        self.assertEqual(pending['s1_planner'],1200)
        for state,(actions,receipt) in zip(states,results):
            if actions:
                self.assertEqual(pending[state.replay_key+'_CG'],840)
        self.assertTrue(any(r['reason']=='budget_pruned' for _,r in results))


if __name__=='__main__':unittest.main()
