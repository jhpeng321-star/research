import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .capacity_broker import drain_active,capacity_declines


class TailDrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.config=dict(boundary_drain_until='TAIL_HANDOVER_COMPLETE.json',boundary_drain_update=2,boundary_drain_minimum=10)

    def test_unmet_executed_quota_never_drained_even_if_inflight_reachable(self):
        with patch('tools.coopmwm.quota_admission.state',return_value=dict(executed=9,inflight=20)):
            self.assertFalse(drain_active(self.root,self.config))

    def test_exact_quota_enables_one_time_boundary_and_marker_restores_admission(self):
        with patch('tools.coopmwm.quota_admission.state',return_value=dict(executed=10,inflight=2)):
            self.assertTrue(drain_active(self.root,self.config))
        (self.root/'TAIL_HANDOVER_COMPLETE.json').write_text('{}')
        with patch('tools.coopmwm.quota_admission.state',side_effect=AssertionError('new wave must not inspect old quota')):
            self.assertFalse(drain_active(self.root,self.config))

    def test_boundary_preserves_assigned_and_nuisance_and_diagnostic_queries(self):
        items=[('plan',dict(state_id='owned')),('plan',dict(state_id='new')),
            ('centre',dict(state_id='centre')),('plan',dict(state_id='e2',mode='diagnostic'))]
        declined,_=capacity_declines(items,dict(owned=dict(slot='gpuR@1')),['gpuA@0','gpuR@1'],drain=True)
        self.assertEqual([r['state_id'] for r in declined],['new'])

    def test_original_drain_and_default_are_unchanged(self):
        self.assertFalse(drain_active(self.root,{}))
        self.assertTrue(drain_active(self.root,dict(drain_until_handover=True)))
        (self.root/'MAIN0548_HANDOVER_COMPLETE.json').write_text('{}')
        self.assertFalse(drain_active(self.root,dict(drain_until_handover=True)))


if __name__=='__main__':unittest.main()
