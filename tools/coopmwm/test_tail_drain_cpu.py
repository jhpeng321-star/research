import json
from pathlib import Path
import tempfile
import signal
import unittest
from unittest.mock import patch,call
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

    def test_broker_replacement_never_signals_changed_identity(self):
        from . import tail_drain_ops as d
        with patch.object(d,'broker_state',side_effect=AssertionError('wrong bound source')),patch.object(d.os,'kill') as kill:
            with self.assertRaises(AssertionError):d.retire_broker(dict(pid=123))
            kill.assert_not_called()

    def test_verified_broker_term_is_pid_only_and_poll_keeps_start_identity(self):
        from . import tail_drain_ops as d
        with patch.object(d,'broker_state',side_effect=['S','exited']) as state,patch.object(d.os,'kill') as kill:
            owner=dict(pid=123);d.retire_broker(owner)
            self.assertEqual(kill.call_args_list,[call(123,signal.SIGTERM)])
            self.assertEqual(state.call_args_list,[call(owner),call(owner,verify_command=False)])

    def test_rootless_serve_identity_requires_exact_argv_boot_start_and_source(self):
        from . import tail_drain_ops as d
        proc=self.root/'proc/123';proc.mkdir(parents=True)
        fields=['123','(python)','S']+['0']*19;fields[21]='original'
        (proc/'stat').write_text(' '.join(fields))
        argv=['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.centre_fair_ops','serve']
        (proc/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
        source='/home/ubuntu/coopmwm20261102_centrefaircpu1'
        (proc/'environ').write_bytes(('PYTHONPATH='+source+'/pytools:/runtime\0').encode())
        boot=self.root/'boot';boot.write_text('boot')
        def path(*parts):
            if parts==('/proc/sys/kernel/random/boot_id',):return boot
            if parts==('/proc',):return self.root/'proc'
            return Path(*parts)
        owner=dict(pid=123,start_ticks='original',boot_id='boot',root=d.ROOT,source=source)
        with patch.object(d,'Path',side_effect=path):
            self.assertEqual(d.broker_state(owner),'S')
            for changed in (dict(start_ticks='reused'),dict(boot_id='newboot'),dict(source='/other')):
                with self.assertRaises(AssertionError):d.broker_state(dict(owner,**changed))
            (proc/'environ').write_bytes(b'PYTHONPATH=/other\0')
            with self.assertRaises(AssertionError):d.broker_state(owner)


if __name__=='__main__':unittest.main()
