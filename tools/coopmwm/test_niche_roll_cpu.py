import hashlib
import json
from . import niche_roll as nr


def test_one_qualifier_across_hosts_and_failed_owner_blocks_roll():
    import pytest
    def worker(host,suffix,generation='FASTK2V5',ready=True,state='R',failure=None):
        return dict(owner=dict(root='/root/coopmwm20261102_'+suffix,generation=generation),
            ready=ready,state=state,failure=failure)
    gpus={'gpuA':dict(workers=[worker('gpuA','incident1','NICHENULL2',False),worker('gpuA','deadlineA2')]),
          'gpuR':dict(workers=[worker('gpuR','concurrent3'),worker('gpuR','deadlineR2')])}
    assert nr.roll_target(gpus) is None
    gpus['gpuA']['workers'][0]['ready']=True
    assert nr.roll_target(gpus)==('gpuA','/root/coopmwm20261102_deadlineA2')
    gpus['gpuA']['workers'][0].update(ready=False,state='exited',failure={'error':'fault'})
    with pytest.raises(ValueError,match='diagnose'):nr.roll_target(gpus)


def test_activation_requires_all_four_ready_and_never_controls_them(monkeypatch,tmp_path):
    from . import ls_validity_ops as vo
    monkeypatch.setattr(nr.ops,'EVIDENCE',tmp_path)
    monkeypatch.setattr(vo,'gpu_status',lambda host:dict(workers=[dict(ready=False,owner=dict(generation='NICHENULL2'))]*2))
    def forbidden(*args,**kwargs):raise AssertionError('no collector transport before all GPU gates')
    monkeypatch.setattr(nr,'remote',forbidden)
    assert nr.activate() is False


def test_activation_payload_compiles_and_waits_on_any_incomplete_claim(monkeypatch,tmp_path):
    from . import ls_validity_ops as vo
    monkeypatch.setattr(nr.ops,'EVIDENCE',tmp_path)
    monkeypatch.setattr(vo,'gpu_status',lambda host:dict(workers=[dict(ready=True,owner=dict(generation='NICHENULL2'))]*2))
    (tmp_path/'LS_LABEL_CPU_coopmwm20261103_nichenullcpu6.json').write_text('{}')
    (tmp_path/'NICHE_COLLECTOR_ACTIVATION_REGISTRATION.json').write_text('{}')
    def remote(host,code,**kwargs):
        assert host=='cpuQ'
        compile(code,'<collector-no-grade-boundary>','exec')
        assert "open_claims" in code and "signal.SIGCONT" in code
        assert "label_attempts/SUMOENV1/verification/claims" in code
        assert "resource_tracker" in code
        assert "operational_spend_cap=6000000" in code
        return '{"wait":true,"open_claims":["original-grade"]}'
    monkeypatch.setattr(nr,'remote',remote)
    assert nr.activate() is False
    assert not (tmp_path/'NICHE_NULLFIX_COLLECTOR_ACTIVE.json').exists()


def test_worker_checks_additive_dependency_before_import(tmp_path):
    import sys
    import types
    from unittest.mock import patch
    worker=types.ModuleType('tools.coopmwm.niche_worker')
    calls=[]
    worker.run=lambda root,generation:calls.append((root,generation))
    (tmp_path/'geometry.py').write_bytes(b'pure geometry')
    (tmp_path/'NICHENULL_DEPENDENCY_HASHES.json').write_text(json.dumps(
        {'geometry.py':hashlib.sha256(b'pure geometry').hexdigest()}))
    with patch.dict(sys.modules,{'tools.coopmwm.niche_worker':worker}):
        exec(nr.worker_command(str(tmp_path)),{})
        (tmp_path/'geometry.py').write_bytes(b'changed')
        import pytest
        with pytest.raises(AssertionError):exec(nr.worker_command(str(tmp_path)),{})
    assert calls==[(str(tmp_path),'NICHENULL1')]


def test_broker_payload_compiles_without_process_control(monkeypatch,tmp_path):
    monkeypatch.setattr(nr.ops,'EVIDENCE',tmp_path)
    (tmp_path/nr.REG).write_text(json.dumps(dict(cpu=dict(source_hashes={}),line='test')))
    def remote(host,code,**kwargs):
        assert host=='cpuQ'
        compile(code,'<broker-handover>','exec')
        assert 'os.kill(old' in code and 'signal.SIGTERM' in code
        return '{}'
    monkeypatch.setattr(nr,'remote',remote)
    nr.broker()


def test_gpu_dispatch_payload_compiles_and_preserves_original_binding(monkeypatch,tmp_path):
    monkeypatch.setattr(nr.ops,'EVIDENCE',tmp_path)
    names=('mppi_search','mppi_runtime','plan_service','ls_gpu','niche_collect','feasible',
           'sumo_verify','nullfix_geometry','niche_worker')
    hashes={f'pytools/tools/coopmwm/{n}.py':hashlib.sha256(
        (nr.ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8').encode()).hexdigest() for n in names}
    hashes['FROZEN_GEOMETRY_SOURCE.py']=hashlib.sha256(
        (nr.ops.LOCAL/'traffic_shepherd/c4/geometry.py').read_text(encoding='utf-8').encode()).hexdigest()
    hashes['FROZEN_NETWORK.net.xml.gz']=hashlib.sha256(
        (nr.ops.LOCAL/'assets/research/highway_merge_shepherd_long/osm.net.xml.gz').read_bytes()).hexdigest()
    (tmp_path/nr.REG).write_text(json.dumps(dict(cpu=dict(source_hashes=hashes))))
    (tmp_path/'NICHE_DEPENDENCY_RECOVERY2.json').write_text(json.dumps(dict(source_hashes={})))
    seen=[]
    def remote(host,code,**kwargs):
        compile(code,'<gpu-dispatch>','exec');seen.append(host)
        if host=='cpuQ':return json.dumps(dict(pid=1,start_ticks='10',boot_id='b'))
        assert 'GPU_BINDING_NICHENULL2.json' in code and 'PRE_NICHENULL1' in code
        assert 'actual_exit' not in code or 'SPOOL_WORKER.claim' in code
        return json.dumps(dict(pid=2,start_ticks='20',root='/root/coopmwm20261102_incident1',host=host))
    monkeypatch.setattr(nr,'remote',remote)
    nr.gpu('gpuA','/root/coopmwm20261102_incident1')
    assert seen==['cpuQ','gpuA']
