import json
from pathlib import Path
import tempfile
import signal
import time
import unittest
from unittest.mock import patch,call
import numpy as np
from . import tail_runtime as t
from .warm_spool import immutable as _immutable


def immutable(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    return _immutable(path,value)


class TailRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_resume_preserves_trust_library_and_diagnostic_draw_schedule(self):
        rng=np.random.default_rng(20261101)
        for u,n in ((1,3),(2,10)):
            ep=dict(world=dict(seed=986000),arm='B',steps=[dict(time_s=300+i,sample=dict(behaviour=dict(plan_start=True))) for i in range(n)])
            immutable(self.root/'jobs'/f'warm_u{u:03}_B_986000'/'RESULT.json',ep)
            rng.permutation(n)
        immutable(self.root/'B_UPDATE_002.json',dict(trust=dict(beta=.7,graded=13)))
        immutable(self.root/'B_SEARCH_LEARNING_002.json',dict(arm='B',seed=20261101,library=['qualified']))
        with patch('tools.coopmwm.accel_learner.evidence',side_effect=lambda episodes:[len(episodes[0]['steps'])]):
            v=t.restore_progress(self.root,'B',2,lambda u:[dict(seed=986000)],20261101)
        self.assertEqual(v['running'],[3,10]);self.assertEqual(v['beta'],dict(beta=.7,graded=13))
        self.assertEqual(v['learned']['library'],['qualified']);self.assertEqual(v['exact_count'],12)
        self.assertEqual(v['exact_rng_state'],rng.bit_generator.state)

    def test_restore_uses_real_lossless_compressed_result_reader(self):
        from .accel_storage import write
        ep=dict(world=dict(seed=986000),arm='B',steps=[dict(time_s=300,sample=dict(behaviour=dict(plan_start=True)))])
        write(self.root/'jobs/warm_u001_B_986000/RESULT.json',ep)
        self.assertIn('encoding',json.loads((self.root/'jobs/warm_u001_B_986000/RESULT.json').read_text()))
        immutable(self.root/'B_UPDATE_001.json',dict(trust=dict(beta=.9)))
        immutable(self.root/'B_SEARCH_LEARNING_001.json',dict(arm='B',library=['verified']))
        with patch('tools.coopmwm.accel_learner.evidence',return_value=['diagnostic']):
            v=t.restore_progress(self.root,'B',1,lambda u:[dict(seed=986000)],20261101)
        self.assertEqual(v['exact_count'],1);self.assertEqual(v['running'],['diagnostic'])

    def test_busy_intervals_are_clipped_unioned_per_slot_with_pending_censored(self):
        immutable(self.root/'quota_requests/u002/a_REQUEST.json',dict(records=[dict(state_id=k) for k in ('a','b','c')]))
        immutable(self.root/'broker_completed/a.json',dict(slot='gpuA@0',unix=110,inference_wall_s=20))
        immutable(self.root/'broker_completed/b.json',dict(slot='gpuA@0',unix=115,inference_wall_s=10))
        v=t.busy_report(self.root,2,100,120,['gpuA@0','gpuR@0'])
        self.assertEqual(v['slots']['gpuA@0'],dict(completed_inference_s=15,busy_fraction=.75))
        self.assertEqual(v['slots']['gpuR@0']['busy_fraction'],0)
        self.assertEqual(v['pending_queries'],1)

    def test_original_single_request_busy_receipt_still_supported(self):
        immutable(self.root/'quota_requests/u002/a_REQUEST.json',dict(state_id='a'))
        immutable(self.root/'broker_completed/a.json',dict(slot='gpuA@0',unix=110,inference_wall_s=20))
        self.assertEqual(t.busy_report(self.root,2,100,105,['gpuA@0'])['slots']['gpuA@0']['busy_fraction'],1.)

    def setup_transition(self):
        from . import tail_transition as transition
        immutable(self.root/'QUOTA_B_DISPATCH.json',dict(pid=123,start_ticks='456',boot_id='boot',root=str(self.root)))
        jobs=[self.root/'jobs'/f'job{i}' for i in range(24)]
        for p in jobs:immutable(p/'CLAIM.json',dict(pid=456))
        (self.root/'B_checkpoint_000.pt').write_bytes(b'original checkpoint')
        return transition,jobs

    def test_freeze_only_original_parent_after_all_claims(self):
        transition,jobs=self.setup_transition()
        with patch.object(transition,'paths',return_value=jobs),patch.object(transition,'state',side_effect=['S','T']),patch.object(transition.os,'kill') as kill:
            frozen=transition.freeze(self.root)
        self.assertEqual(kill.call_args_list,[call(123,signal.SIGSTOP)])
        self.assertEqual((frozen['resume_update'],frozen['adopt_update']),(0,1))
        self.assertEqual(len(frozen['existing_claims']),24)

    def test_boundary_race_unfreezes_without_launch_or_checkpoint_replay(self):
        transition,jobs=self.setup_transition()
        def state(*args):
            if state.called:
                (self.root/'B_checkpoint_001.pt').write_bytes(b'new checkpoint')
                return 'T'
            state.called=True;return 'S'
        state.called=False
        with patch.object(transition,'paths',return_value=jobs),patch.object(transition,'state',side_effect=state),patch.object(transition.os,'kill') as kill:
            self.assertIsNone(transition.freeze(self.root))
        self.assertEqual(kill.call_args_list,[call(123,signal.SIGSTOP),call(123,signal.SIGCONT)])
        self.assertFalse((self.root/'TAIL_PARENT_FROZEN.json').exists())

    def test_partial_wave_never_frozen_and_identity_fault_never_signals(self):
        transition,jobs=self.setup_transition()
        with patch.object(transition,'paths',return_value=jobs+[self.root/'unclaimed']),patch.object(transition,'state',return_value='S'),patch.object(transition.os,'kill') as kill:
            self.assertIsNone(transition.freeze(self.root));kill.assert_not_called()
        with patch.object(transition,'state',side_effect=AssertionError('reused PID')),patch.object(transition.os,'kill') as kill:
            with self.assertRaises(AssertionError):transition.freeze(self.root)
            kill.assert_not_called()

    def test_incomplete_wave_and_existing_dispatch_never_terminated_or_launched(self):
        transition,jobs=self.setup_transition()
        with patch.object(transition,'paths',return_value=jobs),patch.object(transition,'terminate') as terminate,patch.object(transition.subprocess,'Popen') as launch:
            with self.assertRaises(AssertionError):transition.handover(self.root,dict(adopt_update=1),{})
            terminate.assert_not_called();launch.assert_not_called()
        immutable(self.root/'TAIL_B_DISPATCH.json',dict(pid=999))
        with patch.object(transition,'terminate') as terminate,patch.object(transition.subprocess,'Popen') as launch:
            with self.assertRaises(AssertionError):transition.handover(self.root,dict(adopt_update=1),{})
            terminate.assert_not_called();launch.assert_not_called()

    def test_repair_controller_adopts_existing_frozen_wave_without_refreezing(self):
        from . import tail_transition as transition
        frozen=dict(owner=dict(pid=123),adopt_update=2,resume_update=1)
        immutable(self.root/'TAIL_PARENT_FROZEN.json',frozen)
        immutable(self.root/'TAIL_PLAN_BINDING_V2.json',dict(amendment_sha256='registered',cpu=dict(source_hashes={}),source_root='/unused'))
        immutable(self.root/'WARM_REMOTE_BINDING.json',dict(deadline_unix=time.time()+60))
        (self.root/'AUTHORITY.md').write_text('registered')
        with patch.object(transition,'freeze',side_effect=AssertionError('already frozen')),patch.object(transition,'completed',return_value=True),patch.object(transition,'state',return_value='T'),patch.object(transition,'assert_running'),patch.object(transition.signal,'signal'),patch.object(transition,'handover',return_value=dict(pid=999)) as handover:
            self.assertEqual(transition.watch(self.root,'TAIL_PLAN_BINDING_V2.json'),dict(pid=999))
            self.assertEqual(handover.call_args.args[1],frozen)


if __name__=='__main__':unittest.main()
