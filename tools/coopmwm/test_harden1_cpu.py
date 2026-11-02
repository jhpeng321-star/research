"""HARDEN-1 regressions for the 2026-11-02 incident history (CPU only, no SUMO/GPU).

Each test reproduces one incident path with mocks; it fails on aadee535 and
passes with the HARDEN-1 fixes. See HARDEN_1_REPORT.md.
"""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


def fake_family_modules(members):
    """Inherited option machinery (tools.wmteach/leverprobe) reduced to one option."""
    labels=types.ModuleType('tools.wmteach.labels')
    labels.actions=lambda rows,setting:[dict(id='O1',members=list(members),window=30)]
    intervention=types.ModuleType('tools.leverprobe.intervention')
    intervention.targets=lambda option,rows:{}
    intervention.requests=lambda option,current,fixed:({},None)
    wmteach=types.ModuleType('tools.wmteach');leverprobe=types.ModuleType('tools.leverprobe')
    return {'tools.wmteach':wmteach,'tools.wmteach.labels':labels,
            'tools.leverprobe':leverprobe,'tools.leverprobe.intervention':intervention}


class FamilySeedsTests(unittest.TestCase):
    """09:41 (concurrent1) and 16:29 (second1) gpuA worker deaths."""

    def rows(self,n):
        return [dict(vid=str(i),s_m=-10.*i,v_mps=10.,is_cav=True) for i in range(n)]

    def test_four_to_six_member_groups_skip_prototypes_never_raise(self):
        from .plans import family_seeds
        for n in (4,5,6):
            members=[str(i) for i in range(n)]
            with self.subTest(n=n),patch.dict(sys.modules,fake_family_modules(members)):
                self.assertEqual(family_seeds(self.rows(n),members,{v:0. for v in members},(-3.,2.6)),[])

    def test_two_three_member_prototypes_unchanged(self):
        from .plans import family_seeds
        for n in (2,3):
            members=[str(i) for i in range(n)]
            with self.subTest(n=n),patch.dict(sys.modules,fake_family_modules(members)):
                seeds=family_seeds(self.rows(n),members,{v:0. for v in members},(-3.,2.6))
            self.assertEqual(len(seeds),4)
            self.assertEqual({s['family'] for s in seeds},{'O1'})
            self.assertTrue(all(len(s['knots'])==n for s in seeds))


class QueryErrorTests(unittest.TestCase):
    """MAIN17:26: a per-query error is a recorded decline, never a worker death."""

    def test_group_size_validation_error_is_a_decline(self):
        from .query_errors import execute,validate_response
        record=dict(state_id='a'*64)
        def fault():raise ValueError('complete observable group of two or three required')
        value=execute(record,fault);validate_response(record,value)
        self.assertFalse(value['accepted']);self.assertIn('two or three',value['query_error']['error'])
        def fault2():raise ValueError('group does not match registered observable components')
        self.assertFalse(execute(record,fault2)['accepted'])

    def test_large_group_cuda_oom_is_released_and_declined(self):
        from .query_errors import execute
        OutOfMemoryError=type('OutOfMemoryError',(RuntimeError,),{})
        def fault():raise OutOfMemoryError('CUDA out of memory. Tried to allocate 1.01 GiB')
        self.assertFalse(execute(dict(state_id='b'*64),fault)['accepted'])

    def test_named_invariants_and_other_runtime_errors_stay_fatal(self):
        from .query_errors import execute
        for error in (ValueError('persistent worker batch parity failed'),AssertionError('immutable source changed'),
                      ValueError('registered GPU deadline'),ValueError('duplicate inference attempt'),
                      RuntimeError('CUDA error: device-side assert'),OSError('disk')):
            def fault(error=error):raise error
            with self.subTest(error=repr(error)),self.assertRaises(type(error)):execute({},fault)


class PlanServiceTests(unittest.TestCase):
    """A malformed plan request must not kill the persistent worker loop."""

    def test_invalid_request_declines_and_next_request_is_served(self):
        from . import plan_service as service
        from .protocol import digest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'forecasts').mkdir()
            binding=dict(hashes={},qualification=dict(pass_=True),deadline_unix=4e9)
            (root/'GPU_BINDING.json').write_text(json.dumps(binding))
            (root/'FROZEN_COSTS.json').write_text(json.dumps(dict(code='x')))
            rows=[dict(vid='a',is_cav=True),dict(vid='b',is_cav=True)]
            bad=dict(state_id='1'*64,world_id=1,time_s=300,rows=rows,group=['a','b'],commands={},policy_residuals={})
            good=dict(bad,state_id='2'*64)
            runtime=types.ModuleType('tools.wmteach.runtime');runtime.bootstrap=lambda:None
            rt=types.SimpleNamespace(frozen=types.SimpleNamespace(costs=None))
            calls=[]
            def memory(rt_,r,b):
                calls.append(r['state_id'])
                if r['state_id']==bad['state_id']:raise ValueError('observable plan request whitelist')
            real=Path
            class Stat:
                def __init__(self,text):self.text=text
                def read_text(self):return self.text
            def path(p,*a):
                if str(p).startswith('/proc/sys'):return Stat('boot')
                if str(p).startswith('/proc'):return Stat(' '.join(['0']*30))
                return real(p,*a)
            stdin=io.StringIO(json.dumps(bad)+'\n'+json.dumps(good)+'\n')
            with patch.dict(sys.modules,{'tools.wmteach':types.ModuleType('tools.wmteach'),'tools.wmteach.runtime':runtime}), \
                    patch.object(service,'configured_runtime',return_value=rt),patch.object(service,'request_memory',side_effect=memory), \
                    patch.object(service,'Path',side_effect=path),patch('tools.coopmwm.wm.cost_adapter',lambda c,*a,**k:c), \
                    patch('tools.coopnative.policy.encode'),patch('tools.coopmwm.mppi_runtime.plan',
                        side_effect=lambda rt_,r,m,b:dict(accepted=False,reason='no candidate beats policy',record_sha256=digest(r))), \
                    patch.object(sys,'stdin',stdin),patch('builtins.print'):
                service.main(root)
            self.assertEqual(calls,[bad['state_id'],good['state_id']])
            first=json.loads((root/'forecasts'/(bad['state_id']+'.json')).read_text())['result']
            second=json.loads((root/'forecasts'/(good['state_id']+'.json')).read_text())['result']
            self.assertFalse(first['accepted']);self.assertIn('whitelist',first['query_error']['error'])
            self.assertEqual(second['reason'],'no candidate beats policy')


class HotAddGateTests(unittest.TestCase):
    """14:24 gpuA hot-add stall: 'missing/nonexact new-worker gate STARTUP_COMMITTED_K4_H300.json'."""

    def test_spool_status_exposes_all_gates_required_for_hot_add(self):
        from .warm_rpc import status
        from .hotadd import GATES,qualified
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in GATES:(root/name).write_text(json.dumps(dict(pass_=True,max_cost_error=0.)))
            (root/'SERVICE_OWNER.json').write_text(json.dumps(dict(pid=1)))
            s=status(root)
            self.assertTrue(s['ready'])
            s.update(alive=True,failed=False)  # no SPOOL_OWNER/proc on this CPU fixture
            self.assertTrue(qualified(s))
            (root/'STARTUP_COMMITTED_K4_H300.json').write_text(json.dumps(dict(pass_=False,max_cost_error=1.)))
            s=dict(status(root),alive=True,failed=False)
            with self.assertRaisesRegex(ValueError,'STARTUP_COMMITTED_K4_H300'):qualified(s)


class BrokerWorkerTests(unittest.TestCase):
    """A worker found dead while re-verifying (e.g. after a DNS/transport retry)
    quarantines its slot; it must not write a global STOP."""

    def test_unready_worker_on_verify_quarantines_slot_without_global_stop(self):
        from . import capacity_broker as cb
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'AUTHORITY.md').write_text('registered')
            (root/'WARM_REMOTE_BINDING.json').write_text(json.dumps(dict(deadline_unix=4e9,endpoints=[])))
            config=root/'CONFIG.json';config.write_text(json.dumps(dict(amendment_sha256='registered',deadline_unix=4e9,source_root='x')))
            for name in ('requests','responses','quota_admissions'):(root/name).mkdir()
            dead=dict(alive=False,failed=True,ready=False,receipts={'SPOOL_FAILED.json':dict(error="ValueError('complete observable group of two or three required')")})
            from .warm_rpc import GATES
            gates={n:dict(pass_=True,max_cost_error=0.) for n in (*GATES,'SERVICE_OWNER.json','SPOOL_OWNER.json')}
            ready=dict(alive=True,failed=False,ready=True,receipts=gates)
            answers=iter([ready,dead])  # start status, then the endpoint re-verification
            from . import warm_broker
            quarantined=threading.Event()
            real=cb.settle_unavailable
            def settle(*a):
                quarantined.set();return real(*a)
            class Sentinel(Exception):pass
            def drain(root_,config_):
                if quarantined.wait(5):raise Sentinel('test end')
                return False
            real_path=Path
            class Stat:
                def __init__(self,text):self.text=text
                def read_text(self):return self.text
            def path(p,*a):
                if str(p).startswith('/proc/sys'):return Stat('boot')
                if str(p).startswith('/proc'):return Stat(' '.join(['0']*30))
                return real_path(p,*a)
            with patch.object(warm_broker.Remote,'raw',side_effect=lambda value:next(answers)),patch.object(cb,'worker_endpoints',return_value={'gpuA@0':dict(slot='gpuA@0')}), \
                    patch.object(cb,'settle_unavailable',side_effect=settle),patch.object(cb,'drain_active',side_effect=drain), \
                    patch.object(cb,'Path',side_effect=path),patch('signal.signal'),patch('signal.SIGHUP',1,create=True), \
                    patch('tools.coopmwm.exploration.regime',return_value='ordinary'):
                with self.assertRaises(ValueError) as caught:cb.run(root,config)
            self.assertEqual(eval(str(caught.exception)),["Sentinel('test end')"])  # only the test's own end
            self.assertEqual(len(list((root/'worker_restart_requests').glob('*.json'))),1)
            stop=json.loads((root/'WARM_GUIDED_STOP.json').read_text())
            self.assertEqual(stop['source'],'capacity broker control')  # the injected end, not the worker


class ExtensionRecordTests(unittest.TestCase):
    """16:34 FileExistsError on immutable WARM_BINDING.json / KeyError 'gpu_deadline'."""

    def test_superseded_in_place_binding_rewrite_refuses_before_any_io(self):
        from . import verification_beta_ops as ops
        trap=types.ModuleType('tools.coopmwm.ops')
        def touched(name):raise RuntimeError('extend_register touched ops.'+name)
        trap.__getattr__=touched
        with patch.dict(sys.modules,{'tools.coopmwm.ops':trap}):
            with self.assertRaisesRegex(ValueError,'superseded by additive RESOURCE_EXTENSION_1651'):
                ops.extend_register()

    def test_gpu_spool_reads_additive_extension_clock(self):
        from . import warm_spool
        from .resource_config import NAME
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            binding=dict(deadline_unix=1793614022.4598482,memory_mode='warm',startup_qualification=True,speed2=True,speed2_transfers=True)
            (root/'GPU_BINDING.json').write_text(json.dumps(binding))
            (root/'AUTHORITY.md').write_text('amend1651')
            (root/NAME).write_text(json.dumps(dict(amendment_sha256='amend1651',gpu_deadline=1793894340.0)))
            seen={}
            class Spool:
                def __init__(self,root_,deadline):seen['deadline']=deadline
            class Stat:
                def __init__(self,text):self.text=text
                def read_text(self):return self.text
            real=Path
            def path(p,*a):
                if str(p).startswith('/proc/sys'):return Stat('boot')
                if str(p).startswith('/proc'):return Stat(' '.join(['0']*30))
                return real(p,*a)
            with patch.object(warm_spool,'SpoolInput',Spool),patch.object(warm_spool,'Path',side_effect=path), \
                    patch('signal.signal'),patch('signal.SIGHUP',1,create=True),patch('tools.coopmwm.warm_service.main'):
                warm_spool.main(root)
            self.assertEqual(seen['deadline'],1793894340.0)
            self.assertEqual(json.loads((root/'GPU_BINDING.json').read_text()),binding)  # immutable binding untouched


class RunnerBindingTests(unittest.TestCase):
    """Same class as the 05:26 CAPACITY_REPAIR TypeError ('multiple values for keyword')."""

    def test_operational_override_may_carry_awr_without_type_error(self):
        from . import warm_runner as wr
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'AUTHORITY.md').write_text('scope release runtime awr')
            src=Path(d,'src').resolve();fake_file=str(src/'pytools/tools/coopmwm/warm_runner.py')
            files={'WARM_BINDING.json':dict(budget='B',cpu=dict(pass_=True,skips=[],sim_s=0),hashes={},
                    scope_amendment_sha256='scope',release_amendment_sha256='release'),
                'WARM_RELEASE.json':dict(pass_=True,N_exec=10,release_amendment_sha256='release'),
                'WARM_B_RUNTIME_BINDING_V3.json':dict(cpu=dict(pass_=True,skips=[]),amendment_sha256='runtime',
                    source_root=str(src),hashes={},operational=dict(workers=24,awr=dict(coefficient=0.))),
                'AWR_BINDING.json':dict(amendment_sha256='awr',config=dict(coefficient=1.))}
            for name in files:(root/name).write_text('{}')
            with patch.object(wr,'read',side_effect=lambda p:files[Path(p).name]),patch.object(wr,'budget',return_value='B'), \
                    patch.object(wr,'assert_running'),patch.object(wr,'install_accounting'),patch.object(wr,'__file__',fake_file):
                b=wr.verify(root,'B')
            self.assertEqual(b['awr'],dict(coefficient=1.));self.assertEqual(b['workers'],24)


if __name__=='__main__':unittest.main()
