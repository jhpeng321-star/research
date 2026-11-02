import hashlib
import json
from . import niche_roll as nr


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
    seen=[]
    def remote(host,code,**kwargs):
        compile(code,'<gpu-dispatch>','exec');seen.append(host)
        if host=='cpuQ':return json.dumps(dict(pid=1,start_ticks='10',boot_id='b'))
        assert 'GPU_BINDING_NICHENULL1.json' in code and 'PRE_NICHENULL1' in code
        assert 'actual_exit' not in code or 'SPOOL_WORKER.claim' in code
        return json.dumps(dict(pid=2,start_ticks='20',root='/root/coopmwm20261102_incident1',host=host))
    monkeypatch.setattr(nr,'remote',remote)
    nr.gpu('gpuA','/root/coopmwm20261102_incident1')
    assert seen==['cpuQ','gpuA']
