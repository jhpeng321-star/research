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
            r=Path(d);c=r/'collection';s=r/'source'
            v=dict(collection_root=str(c),source_root=str(s),budget={'total':14500000,'phases':{'train':3008400}},start_admitted_min=8)
            put(s/'LS_CPU_RESULT.json',dict(pass_=True,sim_s=0,skips=0))
            put(r/'B_REPLAY_PHASE_TRANSFER_0706.json',dict(funding_complete=True,budget=v['budget']))
            put(r/'replay_bank/B/BANK_QUALITY.json', dict(admitted=7,cooperative=3))
            with patch('tools.coopmwm.warm_health.assert_running'):
                self.assertFalse(rs.prereqs(r,v,False)['ready'])
                put(r/'replay_bank/B/BANK_QUALITY.json', dict(admitted=8,cooperative=3))
                # No endpoint/collector activation files exist; no GPU RPC needed.
                self.assertTrue(rs.prereqs(r,v,False)['ready'])
                self.assertEqual(rs.prereqs(r,v,False)['start_admitted_min'],8)
                put(s/'LS_CPU_RESULT.json',dict(pass_=False,sim_s=0,skips=0))
                with self.assertRaises(AssertionError):rs.prereqs(r,v,False)
                put(s/'LS_CPU_RESULT.json',dict(pass_=True,sim_s=0,skips=0))
                put(r/'B_REPLAY_PHASE_TRANSFER_0706.json',dict(funding_complete=False,budget=v['budget']))
                with self.assertRaises(AssertionError):rs.prereqs(r,v,False)

    def test_collection_config_follows_independent_activation(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);v=dict(collection_root=str(r),demand_phase_end_s=1200)
            before=rs.collection_options(v)
            self.assertFalse(before['niche_alternation'])
            self.assertEqual(before['search_config'],dict(iterations=2,screen_K=2))
            put(r/'LS_NULLFIX_CONFIG.json',dict(nullfix_active=True))
            after=rs.collection_options(v)
            self.assertTrue(after['niche_alternation'])
            self.assertEqual(after['search_config']['niches'],1)

    def test_health_pause_is_never_bypassed(self):
        with patch('tools.coopmwm.warm_health.assert_running', side_effect=RuntimeError('pause')):
            with self.assertRaisesRegex(RuntimeError,'pause'):rs.prereqs(Path('.'),{},False)

    def test_cooperative_config_survives_real_admission_and_R_stays_unscored(self):
        from .replay_bank import collector_guide
        from .niche_collect import search_config
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);v=dict(collection_root=str(r),demand_phase_end_s=1200)
            before=rs.collection_options(v)
            self.assertEqual(before,dict(slots=4,fanout_factor=2,owned_pending_only=True,
                niche_alternation=False,search_config=dict(iterations=2,screen_K=2),
                demand_phase_end_s=1200))
            put(r/'LS_NULLFIX_CONFIG.json',dict(nullfix_active=True))
            coop=dict(coop_objective=True,coop_lambda=1.,coop_solo_cap=16,coop_diversity=True)
            put(r/'LS_COOP_SEARCH_CONFIG.json',dict(coop_search=coop,coop_alternation=True))
            cfg=rs.collection_options(v)
            (r/'requests').mkdir()
            sample=dict(context=dict(members=['a','b']),own_action=[[0.],[0.]])
            def job(arm):
                return dict(id=arm+'_current',arm=arm,queue_root=str(r),world=dict(seed=986000),
                    collector_policy_update=2,collect=cfg)
            with patch('tools.coopmwm.mppi_search.group_components',return_value=[dict(members=['a','b'])]), \
                 patch('tools.coopmwm.exploration.regime',return_value='ordinary'), \
                 patch('tools.coopmwm.sumo_verify.capture',return_value={}),patch('torch.save'):
                guide=collector_guide(job('B'),None,None,{})
                for t in (300,310,320,330):guide([],dict(a=0.,b=0.),t,sample)
                collector_guide(job('R'),None,None,{})([],dict(a=0.,b=0.),300,sample)
            admissions=sorted((r/'replay_bank/B/collection_admissions').glob('*.json'))
            configs=[json.loads(p.read_text())['search_config'] for p in admissions]
            self.assertEqual(configs,[search_config(i,coop=coop,alternate=True) for i in range(4)])
            self.assertEqual([c.get('coop_objective',False) for c in configs],[True,True,False,False])
            requests=[json.loads(p.read_text()) for p in (r/'requests').glob('*.json')]
            self.assertEqual({json.dumps(q['search_config'],sort_keys=True) for q in requests},
                {json.dumps(c,sort_keys=True) for c in configs})
            random=json.loads(next((r/'centre_requests').glob('*.json')).read_text())
            self.assertEqual(random['mode'],'centre')
            self.assertNotIn('search_config',random)

    def test_cooperative_config_cannot_override_admission_or_silently_lose_alternation(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);v=dict(collection_root=str(r),demand_phase_end_s=1200)
            cfg=dict(coop_search=dict(coop_objective=True),coop_alternation=True)
            put(r/'LS_COOP_SEARCH_CONFIG.json',cfg)
            with self.assertRaisesRegex(ValueError,'qualified niche'):rs.collection_options(v)
            put(r/'LS_NULLFIX_CONFIG.json',{})
            put(r/'LS_COOP_SEARCH_CONFIG.json',dict(cfg,slots=100))
            with self.assertRaisesRegex(ValueError,'config keys'):rs.collection_options(v)

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

    def test_adopted_grader_completion_is_read_from_its_original_control(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);old=r/'old';current=r/'new'
            put(old/'grade_B_COMPLETE.json',dict(done=True))
            receipt=dict(component='grade',arm='B',completion_root=str(old))
            self.assertTrue(rs.completed(current,receipt))
            self.assertFalse(rs.completed(current,dict(component='grade',arm='R')))

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
            for i in range(8):
                put(r/'replay_bank/B/snapshots'/f'pending_B{i}.json',dict(arm='B'))
                put(r/'requests'/f'pending_B{i}.json',dict(arm='B'))
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
            self.assertEqual(len(list((r/'requests').glob('*.json'))),8)  # R never scores candidates



if __name__=='__main__':unittest.main()
