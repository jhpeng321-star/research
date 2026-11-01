import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from . import quota_admission as q
from .capacity_broker import capacity_declines
from .warm_spool import immutable
from .protocol import digest


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('responses','guided_starts/u001','guided_starts/u002'):
            (self.root/name).mkdir(parents=True,exist_ok=True)

    def job(self,i=0):return dict(id=f'warm_u002_B_{986000+i}',update=2,N_exec=10)

    def test_queues_until_quota_potential_reachable(self):
        self.assertEqual(q.probability(2,7,10,100,4),(.95,True))
        self.assertEqual(q.probability(2,8,10,100,4),(0.,False))
        self.assertEqual(q.probability(10,0,10,1,4),(.95,False))

    def test_concurrent_clients_do_not_overreserve_quota(self):
        with ThreadPoolExecutor(max_workers=24) as pool:
            out=list(pool.map(lambda i:q.choose(self.root,self.job(i),300,lambda:0.,99,4),range(24)))
        self.assertEqual(sum(v[1] for v in out),10)
        self.assertEqual(q.state(self.root,2),dict(executed=0,inflight=10))

    def test_rng_draw_once_even_when_no_slot(self):
        draws=[]
        def draw():draws.append(1);return .99
        self.assertFalse(q.choose(self.root,self.job(),300,draw,99,4)[1])
        self.assertEqual(len(draws),1)
        self.assertEqual(q.state(self.root,2)['inflight'],0)

    def test_rejected_response_releases_potential(self):
        _,_,_,reservation=q.choose(self.root,self.job(),300,lambda:0.,99,4)
        record=dict(state_id='test',rows=[]);q.bind(reservation,record)
        immutable(self.root/'responses/test.json',dict(accepted=False))
        self.assertEqual(q.state(self.root,2)['inflight'],0)

    def test_trust_rejection_releases_accepted_query(self):
        _,_,_,reservation=q.choose(self.root,self.job(),300,lambda:0.,99,4)
        q.bind(reservation,dict(state_id='test'))
        immutable(self.root/'responses/test.json',dict(accepted=True))
        q.resolved(reservation,False)
        self.assertEqual(q.state(self.root,2)['inflight'],0)

    def test_real_start_not_double_counted_with_pending_resolution(self):
        _,_,_,reservation=q.choose(self.root,self.job(),300,lambda:0.,99,4)
        record=dict(state_id='test');q.bind(reservation,record);q.resolved(reservation,True)
        self.assertEqual(q.state(self.root,2),dict(executed=0,inflight=1))
        immutable(self.root/'guided_starts/u002'/(digest(record)+'.json'),dict(job=self.job()['id']))
        self.assertEqual(q.state(self.root,2),dict(executed=1,inflight=0))

    def test_superseded_u1_starts_excluded(self):
        immutable(self.root/'MAIN0508_REPAIR_BINDING.json',dict(repair_plan=dict(worlds=[dict(seed=986008)])))
        for i in (986000,986008):immutable(self.root/'guided_starts/u001'/f'{i}.json',dict(job=f'warm_u001_B_{i}'))
        self.assertEqual(q.state(self.root,1)['executed'],1)

    def test_broker_preserves_quota_queue_and_prior_claims(self):
        items=[('plan',dict(state_id=k)) for k in ('old','quota1','quota2','overflow')]
        assigned={'old':dict(slot='gpuA@0')}
        declined,busy=capacity_declines(items,assigned,['gpuA@0'],{'quota1','quota2'})
        self.assertEqual([r['state_id'] for r in declined],['overflow'])
        self.assertEqual(busy,{'gpuA@0'})

    def test_boundary_drain_adopts_claims_but_does_not_extend_old_wave(self):
        items=[('plan',dict(state_id=k)) for k in ('old','new')]
        declined,_=capacity_declines(items,{'old':dict(slot='gpuA@0')},['gpuA@0','gpuR@0'],drain=True)
        self.assertEqual([r['state_id'] for r in declined],['new'])

    def test_binding_keeps_gpu_request_unmodified(self):
        _,_,_,reservation=q.choose(self.root,self.job(),300,lambda:0.,99,4)
        record=dict(state_id='test',rows=[]);before=digest(record);q.bind(reservation,record)
        self.assertEqual(before,digest(record))
        self.assertTrue(json.loads((self.root/'quota_admissions/test.json').read_text())['blocking'])

    def test_only_registered_u1_shortfall_is_waived(self):
        with self.assertRaises(ValueError):q.first_update_deviation(self.root,'B',1,4,10)
        immutable(self.root/'MAIN0548_RUNTIME_BINDING.json',dict(plan_amendment_sha256='registered'))
        (self.root/'AUTHORITY.md').write_text('registered')
        self.assertTrue(q.first_update_deviation(self.root,'B',1,4,10)['engineering_deviation'])
        with self.assertRaises(ValueError):q.first_update_deviation(self.root,'B',2,4,10)
        self.assertIsNone(q.first_update_deviation(self.root,'B',2,10,10))

    def test_frozen_parent_exit_is_identity_checked_and_continued(self):
        from unittest.mock import patch,call
        import signal
        from . import quota_transition as t
        with patch.object(t,'state',side_effect=['T','exited']) as state,patch.object(t.os,'kill') as kill:
            t.terminate(dict(pid=123),'owned-module')
            self.assertEqual(kill.call_args_list,[call(123,signal.SIGTERM),call(123,signal.SIGCONT)])
            self.assertEqual(state.call_count,2)

    def test_parent_identity_failure_never_signals(self):
        from unittest.mock import patch
        from . import quota_transition as t
        with patch.object(t,'state',side_effect=AssertionError('identity changed')),patch.object(t.os,'kill') as kill:
            with self.assertRaises(AssertionError):t.terminate(dict(pid=123),'owned-module')
            kill.assert_not_called()

    def test_process_exit_transient_does_not_abort_handover(self):
        from unittest.mock import patch,call
        import signal
        from . import quota_transition as t
        with patch.object(t,'state',side_effect=['T','exiting','exited']),patch.object(t.os,'kill') as kill,patch.object(t.time,'sleep'):
            t.terminate(dict(pid=123),'owned-module')
            self.assertEqual(kill.call_args_list,[call(123,signal.SIGTERM),call(123,signal.SIGCONT)])

    def test_already_exiting_owner_is_only_observed(self):
        from unittest.mock import patch
        from . import quota_transition as t
        with patch.object(t,'state',side_effect=['exiting','exited']),patch.object(t.os,'kill') as kill:
            t.terminate(dict(pid=123),'owned-module');kill.assert_not_called()

    def test_empty_cmdline_still_requires_original_boot_and_start(self):
        from unittest.mock import patch
        from . import quota_transition as t
        proc=self.root/'proc/123';proc.mkdir(parents=True)
        fields=['123','(python)','R']+['0']*19;fields[21]='original'
        (proc/'stat').write_text(' '.join(fields));(proc/'cmdline').write_bytes(b'')
        boot=self.root/'boot';boot.write_text('boot')
        def path(*parts):
            if parts==('/proc/sys/kernel/random/boot_id',):return boot
            if parts==('/proc',):return self.root/'proc'
            return Path(*parts)
        with patch.object(t,'Path',side_effect=path):
            owner=dict(pid=123,start_ticks='original',boot_id='boot',root='/owned')
            self.assertEqual(t.state(owner,'module'),'exiting')
            with self.assertRaises(AssertionError):t.state(dict(owner,start_ticks='reused'),'module')

    def test_existing_dispatch_never_relaunched(self):
        from unittest.mock import patch
        from . import quota_transition as t
        immutable(self.root/'QUOTA_B_DISPATCH.json',dict(pid=1))
        with patch.object(t.subprocess,'Popen') as launch:
            with self.assertRaises(AssertionError):t.launch(self.root,'B','/unused')
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
