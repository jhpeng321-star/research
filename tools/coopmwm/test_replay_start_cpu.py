"""Start-gate, one-use dispatch, fresh namespace and shared-queue CPU checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import replay_start as rs
from .protocol import digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value))


class StartGateTests(unittest.TestCase):
    def test_threshold_release_and_budget_are_joint_gates(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);c=r/'collection';v=dict(collection_root=str(c), budget={'total':14500000})
            put(r/'replay_bank/B/BANK_QUALITY.json', dict(admitted=9,cooperative=3))
            put(c/'LS_ENDPOINTS.json', dict(workers=[]))
            with patch('tools.coopmwm.warm_health.assert_running'):
                self.assertFalse(rs.prereqs(r,v,False)['ready'])
                put(r/'replay_bank/B/BANK_QUALITY.json', dict(admitted=10,cooperative=3))
                put(c/'LS_ENDPOINTS.json', dict(workers=[dict(owner={'generation':'NICHENULL3'})]*4))
                self.assertFalse(rs.prereqs(r,v,False)['ready'])
                put(c/'LS_NULLFIX_CONFIG.json', dict(nullfix_active=True,budget=v['budget']))
                self.assertTrue(rs.prereqs(r,v,False)['ready'])
                put(c/'LS_NULLFIX_CONFIG.json', dict(nullfix_active=True,budget={'total':6000000}))
                with self.assertRaises(AssertionError):rs.prereqs(r,v,False)

    def test_health_pause_is_never_bypassed(self):
        with patch('tools.coopmwm.warm_health.assert_running', side_effect=RuntimeError('pause')):
            with self.assertRaisesRegex(RuntimeError,'pause'):rs.prereqs(Path('.'),{},False)

    def test_fresh_job_namespace_preserves_closed_attempts(self):
        for a in ('B','R'):
            self.assertNotEqual(rs.job_namespace('warm_u004',a),'warm_u004')
            self.assertEqual(rs.job_namespace('warm_u004',a),'replay0735_warm_u004')

    def test_interim_fixed_checkpoint_phase_and_exact_quote(self):
        panel=[dict(seed=990000+i) for i in range(96)]
        specs=rs.interim_specs(panel,['A','B','R'])
        self.assertEqual(sum(s['sim_s'] for s in specs),345600)
        self.assertEqual({s['checkpoint_update'] for s in specs},{8})
        self.assertEqual({s['phase'] for s in specs},{'interim_u8_test'})
        self.assertEqual(len({s['id'] for s in specs}),288)

    def test_shared_broker_bridge_is_idempotent_and_centre_only(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);c=r/'collection';c.mkdir();v=dict(collection_root=str(c))
            for a in ('B','R'):
                q=dict(state_id=a,group=['a','b'])
                if a=='R':q['mode']='centre'
                put(r/'replay_bank'/a/'snapshots'/(a+'.json'),dict(record_sha256=digest(q)))
                put(r/('requests' if a=='B' else 'centre_requests')/(a+'.json'),q)
            put(r/'responses/R.json',dict(record_sha256='test',policy_traj=[]))
            self.assertEqual(rs.bridge_once(r,v),2)
            self.assertEqual(rs.bridge_once(r,v),2)
            self.assertEqual(json.loads((c/'requests/R.json').read_text())['mode'],'centre')
            self.assertTrue((r/'centre_responses/R.json').exists())
            q=json.loads((r/'requests/B.json').read_text());q['group']=['changed']
            put(r/'requests/B.json',q)
            with self.assertRaises(AssertionError):rs.bridge_once(r,v)

    def test_spawn_adopts_live_receipt_and_refuses_dead_child(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);v=dict(control_root=str(r));put(r/'grade_B_DISPATCH.json',dict(pid=123))
            with patch.object(rs,'alive',return_value=True),patch.object(rs.subprocess,'Popen') as p:
                self.assertEqual(rs.spawn(r,v,'grade','B')['pid'],123);p.assert_not_called()
            with patch.object(rs,'alive',return_value=False):
                with self.assertRaisesRegex(AssertionError,'exited'):rs.spawn(r,v,'grade','B')

    def test_old_endpoint_preserved_by_sha_and_final_path_released(self):
        from .protocol import sha
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=r/'WARM_R_PAUSED.json';put(p,dict(update=1))
            expected=dict(sha256=sha(p))
            rs.retire_endpoint(r,'R',expected);rs.retire_endpoint(r,'R',expected)
            self.assertFalse(p.exists())
            self.assertEqual(sha(r/'WARM_R_PAUSED_BEFORE_REPLAY0735.json'),expected['sha256'])

    def test_bounded_loop_and_detached_dispatch_compile(self):
        # Validate the exact operator payload via its pure helper, no SSH/SUMO.
        from .replay_start_ops import qualification_payload
        compile(qualification_payload({}, 'test'),'<qualification>','exec')

    def test_collection_ignores_retired_queue_and_alternates_qualified_configs(self):
        from .replay_bank import collector_guide
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            for i in range(9):put(r/'requests'/f'closed{i}.json',dict(closed=True))
            job=dict(id='replay0735_warm_u004_B_986000',arm='B',queue_root=str(r),
                world=dict(seed=986000),collector_policy_update=3,
                collect=dict(slots=4,fanout_factor=2,owned_pending_only=True,
                    demand_phase_end_s=1200,search_config=dict(iterations=2,screen_K=2,niches=1),
                    niche_alternation=True))
            sample=dict(context=dict(members=['a','b']),own_action=[[0.],[0.]])
            with patch('tools.coopmwm.mppi_search.group_components',return_value=[dict(members=['a','b'])]), \
                 patch('tools.coopmwm.exploration.regime',return_value='ordinary'), \
                 patch('tools.coopmwm.sumo_verify.capture',return_value={}), \
                 patch('torch.save'):
                guide=collector_guide(job,None,None,{})
                self.assertIsNone(guide([],dict(a=0.,b=0.),300,sample))
                self.assertIsNone(guide([],dict(a=0.,b=0.),310,sample))
            metas=[json.loads(p.read_text()) for p in (r/'replay_bank/B/snapshots').glob('*.json')]
            self.assertEqual(len(metas),2)
            configs=[json.loads(p.read_text())['search_config']['niches'] for p in sorted((r/'replay_bank/B/collection_admissions').glob('*.json'))]
            self.assertEqual(configs,[1,4])
            self.assertEqual({m['policy_update'] for m in metas},{3})

    def test_R_centre_receives_its_current_policy_but_no_scoring_or_library(self):
        from .replay_bank import collector_guide
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            job=dict(id='R_current',arm='R',queue_root=str(r),world=dict(seed=986001),
                collector_policy_update=5,policy_state={'R_current_weights':[1.,2.]},
                search_learning={'must_not_reach_R':True},
                collect=dict(slots=4,owned_pending_only=True,demand_phase_end_s=1200))
            sample=dict(context=dict(members=['a','b']),own_action=[[0.],[0.]])
            with patch('tools.coopmwm.mppi_search.group_components',return_value=[dict(members=['a','b'])]), \
                 patch('tools.coopmwm.exploration.regime',return_value='ordinary'), \
                 patch('tools.coopmwm.sumo_verify.capture',return_value={}), \
                 patch('torch.save'):
                guide=collector_guide(job,None,None,{})
                self.assertIsNone(guide([],dict(a=0.,b=0.),300,sample))
            reqs=list((r/'centre_requests').glob('*.json'));self.assertEqual(len(reqs),1)
            q=json.loads(reqs[0].read_text())
            self.assertEqual(q['policy_state'],job['policy_state'])
            self.assertEqual(q['mode'],'centre')
            self.assertNotIn('search_learning',q)
            self.assertFalse((r/'requests').exists())



if __name__=='__main__':unittest.main()
