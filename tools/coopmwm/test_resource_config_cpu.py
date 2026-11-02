"""Immutable clocks, exact DNS release and fresh-attempt adoption contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from .resource_config import clocks,extension,retired,NAME
from .protocol import sha


class ResourceCPU(unittest.TestCase):
    def test_extension_does_not_mutate_original_and_is_authority_gated(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);old=dict(gpu_deadline=1,operational=dict(H=300,deadline_unix=1),budget=9500000)
            (r/'WARM_BINDING.json').write_text(json.dumps(old));h=sha(r/'WARM_BINDING.json')
            self.assertEqual(clocks(r,old),old)
            (r/NAME).write_text(json.dumps(dict(gpu_deadline=1793894340.,amendment_sha256='registered')))
            (r/'AUTHORITY.md').write_text('')
            with self.assertRaises(AssertionError):extension(r)
            (r/'AUTHORITY.md').write_text('registered')
            changed=clocks(r,old)
            self.assertEqual(changed['operational'],dict(H=300,deadline_unix=1793894340.))
            self.assertEqual(old['gpu_deadline'],1);self.assertEqual(sha(r/'WARM_BINDING.json'),h)

    def test_retirement_is_additive_not_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            for name,key in [('INCIDENT_RETIRED_JOBS.json','old'),('DNS_RETIRED_JOBS_1651.json','dns')]:
                (r/name).write_text(json.dumps(dict(jobs=[key],query_keys=[key+'q'])))
            self.assertEqual(retired(r),dict(jobs=['old','dns'],query_keys=['oldq','dnsq']))

    def test_dns_release_is_sha_scoped_and_fresh_stop_remains_fatal(self):
        from .warm_health import assert_running
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);p=r/'WARM_GUIDED_STOP_V4.json'
            p.write_text(json.dumps(dict(source='capacity broker',error="RuntimeError('ssh: Could not resolve hostname connect.westb.seetacloud.com: Temporary failure in name resolution')")))
            (r/'AUTHORITY.md').write_text('registered')
            with self.assertRaises(ValueError):assert_running(r,'B')
            (r/'DNS_GUIDED_RELEASE_1651.json').write_text(json.dumps(dict(original_stop_sha256=sha(p),amendment_sha256='registered')))
            assert_running(r,'B');assert_running(r,'R')
            (r/'WARM_GUIDED_STOP_V5.json').write_text('{}')
            with self.assertRaises(ValueError):assert_running(r,'B')
