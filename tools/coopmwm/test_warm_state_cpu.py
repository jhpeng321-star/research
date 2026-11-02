"""Execute the guard's real transport payload without SSH or SUMO."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from . import warm_watch as w
from . import warm_health  # Bind its real pathlib.Path before the transport stub.


class WarmStateTests(unittest.TestCase):
    def test_fast_resume_owners_override_closed_dns_and_surface_new_failure(self):
        files={'FAST_B_DISPATCH_1844.json':self.owner(),
            'DNS_B_DISPATCH_1651.json':self.owner(22),
            'FAST_R_DISPATCH_1844.json':self.owner(),
            'DNS_R_DISPATCH_1651.json':self.owner(22),
            'FAST_BROKER_DISPATCH_1844.json':self.owner(),
            'GPU_RESOURCE_BROKER_DISPATCH_1651_V2.json':self.owner(22)}
        v=self.collect(files)
        self.assertEqual(v['owners']['B']['dispatch_receipt'],'FAST_B_DISPATCH_1844.json')
        self.assertEqual(v['owners']['R']['dispatch_receipt'],'FAST_R_DISPATCH_1844.json')
        self.assertEqual(v['service_owners']['broker']['owner']['pid'],77)
        self.assertIsNone(v['stop'])
        files['WARM_B_FAILED_FAST1844.json']={'error':'fresh fault'}
        self.assertIsNotNone(self.collect(files)['stop'])

    def test_dns_recovery_owner_precedence_and_new_failures_remain_visible(self):
        files={'DNS_B_DISPATCH_1651.json':self.owner(),
            'BETA_B_DISPATCH.json':self.owner(22),
            'DNS_GRADER_DISPATCH_1651.json':self.owner(),
            'VERIFIED_GRADER_DISPATCH.json':self.owner(22),
            'GPU_RESOURCE_BROKER_DISPATCH_1651.json':self.owner(),
            'DNS_BROKER_DISPATCH_1651.json':self.owner(22),
            'DNS_RECOVERY_STARTED_1651.json':{},'WARM_R_FAILED_SB1.json':{'error':'old DNS stop'}}
        v=self.collect(files)
        self.assertEqual(v['owners']['B']['dispatch_receipt'],'DNS_B_DISPATCH_1651.json')
        self.assertEqual(v['service_owners']['broker']['dispatch_receipt'],'GPU_RESOURCE_BROKER_DISPATCH_1651.json')
        self.assertEqual(v['service_owners']['grader']['dispatch_receipt'],'DNS_GRADER_DISPATCH_1651.json')
        self.assertIsNone(v['stop'])
        files['WARM_B_FAILED_DNS1651.json']={'error':'fresh fault'}
        self.assertIsNotNone(self.collect(files)['stop'])

    def collect(self, files, *, boot='fixture-boot', start='123'):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proc=root/'proc';proc.mkdir()
            for name,value in files.items():
                (root/name).write_text(json.dumps(value))
            (root/'CHARGES.jsonl').write_text('{"sim_s":1200}\n')
            (proc/'sys/kernel/random').mkdir(parents=True)
            (proc/'sys/kernel/random/boot_id').write_text(boot)
            (proc/'77').mkdir()
            stat=['77','(python)','S']+['0']*18+[start]
            (proc/'77/stat').write_text(' '.join(stat))
            paths=types.ModuleType('pathlib')
            def path(*args):
                if args==(w.ROOT,):return root
                p=Path(*args)
                if p==Path('/proc') or Path('/proc') in p.parents:
                    return proc/p.relative_to('/proc')
                return p
            paths.Path=path
            def remote(host,source,timeout):
                self.assertEqual(host,'cpuQ');compile(source,'<actual warm state payload>','exec')
                out=io.StringIO();saved=sys.path[:]
                try:
                    with patch.dict(sys.modules,{'pathlib':paths}),patch('shutil.disk_usage',return_value=types.SimpleNamespace(free=123456)),contextlib.redirect_stdout(out):
                        exec(source,{})
                finally:sys.path[:]=saved
                return out.getvalue()
            with patch.object(w.ops.ssh,'remote',side_effect=remote):return w.state()

    def owner(self, pid=77):
        return dict(pid=pid,start_ticks='123',boot_id='fixture-boot')

    def test_current_owner_precedence_and_numeric_updates(self):
        v=self.collect({'WARM_B_DISPATCH_V4.json':self.owner(22),
            'QUOTA_B_DISPATCH.json':self.owner(),
            'WARM_A_DISPATCH_V3.json':self.owner(),
            'B_UPDATE_001.json':{},'B_UPDATE_001_TIMING.json':{},
            'B_UPDATE_other.json':{},'A_UPDATE_048.json':{}})
        self.assertEqual(v['updates'],dict(A=[48],B=[1],R=[]))
        self.assertEqual(v['owners']['B']['owner']['pid'],77)
        self.assertEqual(v['owners']['A']['dispatch_receipt'],'WARM_A_DISPATCH_V3.json')
        self.assertIsNone(v['stop'])
        self.assertEqual(v['charged_sim_s'],812840)

    def test_only_sha_registered_transport_release_clears_stop(self):
        old=dict(error='Connection closed by 1.15.42.207 port 22\n',
            source='owned warm bridge; no automatic retry')
        # sha is file bytes, not the canonical JSON digest.
        import hashlib
        h=hashlib.sha256(json.dumps(old).encode()).hexdigest()
        files={'STOP.json':old,'WARM_TRANSPORT_RELEASE.json':dict(
            original_stop_sha256=h,amendment_sha256='registered-observation-fixture')}
        # AUTHORITY is read as text; a JSON string is sufficient for this fixture.
        files['AUTHORITY.md']='registered-observation-fixture'
        v=self.collect(files)
        self.assertIsNone(v['stop']);self.assertEqual(v['historical_transport_stop'],old)
        files['WARM_TRANSPORT_RELEASE.json']['original_stop_sha256']='wrong'
        self.assertIsNotNone(self.collect(files)['stop'])

    def test_new_stop_and_health_pause_visible(self):
        v=self.collect({'WARM_GUIDED_STOP_V2.json':dict(error='new fault')})
        self.assertIn('new durable guided',v['stop']['error'])
        v=self.collect({'PAUSE_STARVATION.json':dict(rule='STARVATION')})
        self.assertIsNotNone(v['stop']);self.assertTrue(v['paused']['B']);self.assertTrue(v['paused']['R'])
        self.assertEqual(v['pause_receipts'],['PAUSE_STARVATION.json'])

    def test_tail_owner_takes_precedence_and_tail_failure_is_visible(self):
        v=self.collect({'TAIL_B_DISPATCH.json':self.owner(),
            'QUOTA_B_DISPATCH.json':self.owner(22),'WARM_B_FAILED_T1.json':dict(error='tail fault'),
            'TAIL_DRAIN_BROKER_DISPATCH.json':self.owner(),'CENTRE_FAIR_BROKER_DISPATCH.json':self.owner(22)})
        self.assertEqual(v['owners']['B']['dispatch_receipt'],'TAIL_B_DISPATCH.json')
        self.assertEqual(v['service_owners']['broker']['dispatch_receipt'],'TAIL_DRAIN_BROKER_DISPATCH.json')
        self.assertIsNotNone(v['stop'])
        self.assertIn('WARM_B_FAILED_T1.json',v['failures'])

    def test_current_supervisor_failure_visible(self):
        v=self.collect({'WARM_R_FAILED_Q1.json':dict(error='executor fault')})
        self.assertEqual(v['stop']['error'],'current training supervisor failure')

    def test_recovery_owners_and_only_authorised_old_failures_adopted(self):
        import hashlib
        old=dict(error="ValueError('GPU failed; no inference retry')")
        files={'INCIDENT_RECOVERY_BINDING.json':dict(source_root='/fixture/recovery'),
            'INCIDENT_GUIDED_RELEASE.json':dict(original_stop_sha256=hashlib.sha256(json.dumps(old).encode()).hexdigest(),amendment_sha256='registered'),
            'AUTHORITY.md':'registered','WARM_GUIDED_STOP_V2.json':old,
            'WARM_B_FAILED_T1.json':dict(error='old incident'),
            'WARM_R_FAILED_Q1.json':dict(error='old incident'),
            'INCIDENT_B_DISPATCH.json':self.owner(),
            'TAIL_B_DISPATCH.json':self.owner(22),
            'INCIDENT_BROKER_DISPATCH.json':self.owner(),
            'INCIDENT_GRADER_DISPATCH.json':self.owner()}
        v=self.collect(files)
        self.assertIsNone(v['stop']);self.assertEqual(v['failures'],{})
        self.assertEqual(v['owners']['B']['dispatch_receipt'],'INCIDENT_B_DISPATCH.json')
        self.assertEqual(v['service_owners']['grader']['dispatch_receipt'],'INCIDENT_GRADER_DISPATCH.json')
        files['WARM_B_FAILED_I1.json']=dict(error='fresh recovery fault')
        v=self.collect(files)
        self.assertIsNotNone(v['stop']);self.assertIn('WARM_B_FAILED_I1.json',v['failures'])
        del files['WARM_B_FAILED_I1.json']
        files['INCIDENT_GUIDED_RELEASE.json']['original_stop_sha256']='wrong'
        self.assertIsNotNone(self.collect(files)['stop'])

    def test_fair_broker_and_latest_grader_override_historical_owners(self):
        v=self.collect({'CENTRE_FAIR_BROKER_DISPATCH.json':self.owner(),
            'QUOTA_broker_DISPATCH.json':self.owner(22),
            'EXACT_GRADER_DISPATCH_V2.json':self.owner(),
            'EXACT_GRADER_DISPATCH.json':self.owner(23)})
        self.assertEqual(v['service_owners']['broker']['dispatch_receipt'],'CENTRE_FAIR_BROKER_DISPATCH.json')
        self.assertEqual(v['service_owners']['grader']['dispatch_receipt'],'EXACT_GRADER_DISPATCH_V2.json')
        self.assertTrue(all(x['state']=='S' for x in v['service_owners'].values()))

    def test_exited_new_broker_not_hidden_by_live_predecessor(self):
        v=self.collect({'CENTRE_FAIR_BROKER_DISPATCH.json':self.owner(22),
            'QUOTA_broker_DISPATCH.json':self.owner()})
        self.assertEqual(v['service_owners']['broker']['state'],'exited')
        self.assertEqual(v['service_owners']['broker']['owner']['pid'],22)

    def test_service_pid_reuse_never_reported_as_healthy(self):
        with self.assertRaisesRegex(AssertionError,'service PID reused'):
            self.collect({'CENTRE_FAIR_BROKER_DISPATCH.json':self.owner()},start='456')

    def test_identity_mismatch_never_reports_healthy_owner(self):
        for options in (dict(boot='changed-boot'),dict(start='456')):
            with self.subTest(options=options),self.assertRaises(AssertionError):
                self.collect({'QUOTA_B_DISPATCH.json':self.owner()},**options)


if __name__=='__main__':unittest.main()
