"""Recovery tests: no physics, retained attempts, exact scope and reservation gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from .incident_recovery import recovery_budget,episode_plan,retired_key
from .protocol import digest


class RecoveryTests(unittest.TestCase):
    def test_budget_conservative_liabilities(self):
        old=dict(total=7500000,allocated=7495040,unallocated=4960,
            phases=dict(train=3477600,discovery=200000,exact_plan_grading=1000000,
                final_test=460800,paired_branch=691200,other=1661440))
        # Canonical phase sum, including all retained historical charges.
        old['phases']['other']=old['allocated']-sum(v for k,v in old['phases'].items() if k!='other')
        before=copy.deepcopy(old);b=recovery_budget(old)
        self.assertEqual(old,before);self.assertEqual(b['allocated'],7028840)
        self.assertEqual(b['unallocated'],471160);self.assertEqual(b['phases']['discovery'],179000)
        self.assertEqual(b['updates'],dict(A=48,B=24,R=24))

    def test_completed_adopted_unclaimed_precharged_failed_replaced(self):
        train=[dict(seed=986000+i,cell=0,repeat=i) for i in range(6)]
        original=train[:3];jobs={f'warm_u003_B_{w["seed"]}':dict(closed=i==0,claimed=i!=2) for i,w in enumerate(original)}
        v=episode_plan(original,train,jobs,'B',3)
        self.assertEqual([w['seed'] for w in v['worlds']],[986000,986002,986003])
        self.assertEqual(v['precharged_unattempted'],['warm_u003_B_986002'])
        self.assertEqual(v['additional_sim_s'],1200)

    def test_record_retirement_is_job_bound(self):
        r=dict(world_id=986000,time_s=300,group=['a','b'])
        r['state_id']=digest(dict(record=r,job='warm_u003_B_986000'))
        self.assertTrue(retired_key(r,['warm_u003_B_986000']))
        self.assertFalse(retired_key(r,['warm_u002_B_986000']))

    def test_failed_job_does_not_satisfy_quota(self):
        from .quota_admission import state
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);q=r/'quota_requests/u003';q.mkdir(parents=True)
            g=r/'guided_starts/u003';g.mkdir(parents=True)
            (r/'INCIDENT_RETIRED_JOBS.json').write_text(json.dumps(dict(jobs=['failed'])))
            for job in ('failed','complete'):
                (g/(job+'.json')).write_text(json.dumps(dict(job=job)))
                (q/(job+'_RESERVE.json')).write_text(json.dumps(dict(token=job,job=job)))
            self.assertEqual(state(r,3),dict(executed=1,inflight=1))

    def test_retired_R_ticket_released_without_overwrite(self):
        from .plan_client import used_tickets
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);m=r/'random_match/u002';m.mkdir(parents=True)
            (m/'failed_START.json').write_text('{"ticket":0}')
            (m/'good_START.json').write_text('{"ticket":1}')
            marker=r/'INCIDENT_RETIRED_JOBS.json'
            marker.write_text(json.dumps(dict(R_ticket_keys=['failed'])))
            self.assertEqual(used_tickets(m),{1})
            self.assertTrue((m/'failed_START.json').exists())

    def test_stop_release_is_hash_bound_and_fresh_failure_stops(self):
        from .warm_health import assert_running
        from .protocol import sha
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);p=r/'WARM_GUIDED_STOP_V2.json'
            p.write_text(json.dumps(dict(error="ValueError('GPU failed; no inference retry')")))
            (r/'AUTHORITY.md').write_text('registered-hash')
            with self.assertRaises(ValueError):assert_running(r,'B')
            (r/'INCIDENT_GUIDED_RELEASE.json').write_text(json.dumps(dict(original_stop_sha256=sha(p),amendment_sha256='registered-hash')))
            assert_running(r,'B')
            (r/'WARM_GUIDED_STOP_V3.json').write_text('{}')
            with self.assertRaises(ValueError):assert_running(r,'B')

    def test_retired_query_not_resubmitted(self):
        from .capacity_broker import pending
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'requests').mkdir()
            (r/'requests/old.json').write_text('{}')
            (r/'INCIDENT_RETIRED_JOBS.json').write_text('{"query_keys":["old"]}')
            self.assertEqual(pending(r),[])

    def test_dead_slot_excluded_before_endpoint_identity_validation(self):
        from .warm_broker import worker_endpoints
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);(r/'INCIDENT_RETIRED_JOBS.json').write_text('{"disabled_slots":["gpuA@0"]}')
            self.assertEqual(worker_endpoints(r,dict(endpoints=[dict(slot='gpuA@0')]),{'gpuA@0':dict(slot='gpuA@0')}),{})


if __name__=='__main__':unittest.main()
