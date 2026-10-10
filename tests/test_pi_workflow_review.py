"""Independent supported PI workflow tests; fake physics, real ownership."""
import json
import sys
from types import SimpleNamespace

import pytest
from tests.multi_cav_runtime import multi_cav_runtime as _multi_cav_runtime

from traffic_shepherd import __main__ as cli, workflow
from traffic_shepherd.sensing import SensorFrame, NeighborMeasurement

multi_cav_runtime = _multi_cav_runtime


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def capture(vehicle, *, now_s):
    if vehicle == 'b' and now_s == 2.:
        raise ValueError('missing local frame')
    speed = 10. if vehicle == 'a' else 12.
    return SensorFrame(vehicle, now_s, speed, 0., 30., 0,
        front=NeighborMeasurement(50., 0.), front_coverage_m=100., rear_coverage_m=100.)


@pytest.mark.parametrize('u', [None, 30., float('nan'), float('inf'), -1.])
def test_pi_rejects_nonregistered_u_before_start(tmp_path, multi_cav_runtime, u):
    with pytest.raises(ValueError):
        workflow.evaluate(seed=4107, loading=2000, penetration=.2, controller_name='pi-external-u',
                          desired_speed_mps=u, output=tmp_path/'invalid', duration=3)
    assert not multi_cav_runtime['writes']
    assert not multi_cav_runtime['closed']

def test_actual_cli_multicav_raw_memory_and_missing_reset(tmp_path, multi_cav_runtime, monkeypatch):
    monkeypatch.setattr(workflow,'SumoLocalSensor',lambda *a: SimpleNamespace(capture=capture))
    output=tmp_path/'cli'
    cli.main(['evaluate','--seed','4107','--loading','2000','--penetration','.2',
              '--controller','pi-external-u','--desired-speed-mps','31.29','--duration','4','--output',str(output)])
    telemetry=rows(output/'telemetry.jsonl')
    evidence={(x['vehicle_id'],x['timestamp_s']):x for x in telemetry if x['kind']=='pi_source_evidence'}
    commands={(x['vehicle_id'],x['timestamp_s']):x for x in telemetry if x['kind']=='command'}
    assert len(commands)==6 and len(evidence)==5
    previous={}
    for tick in (1.,2.,3.):
        for vehicle in ('a','b'):
            if (vehicle,tick)==('b',2.):
                assert commands[vehicle,tick]['owner']=='native'
                assert commands[vehicle,tick]['fallback_reason']=='missing_observation'
                previous.pop(vehicle)
                continue
            ego=10. if vehicle=='a' else 12.
            prior=previous.get(vehicle,ego)
            raw=.5*32.29+.5*prior
            item=evidence[vehicle,tick]
            assert item['previous_raw_command_mps']==pytest.approx(prior)
            assert item['raw_proposal_mps']==pytest.approx(raw)
            assert item['issued_target_speed_mps']==pytest.approx(ego+1.5)
            assert item['owner_record']=={k:v for k,v in commands[vehicle,tick].items() if k!='kind'}
            assert commands[vehicle,tick]['network_output_mps2'] is None
            previous[vehicle]=raw
    assert len([x for x in telemetry if x['kind']=='response'])==12
    assert multi_cav_runtime['closed']


def test_empty_roster_fails_before_sumo_start(tmp_path,multi_cav_runtime,monkeypatch):
    from dataclasses import replace
    original=workflow.build_scenario
    starts=[]
    def build(*args,**kwargs):
        scene,requests,metadata=original(*args,**kwargs)
        return scene,[replace(r,is_cav=False) for r in requests],metadata
    monkeypatch.setattr(workflow,'build_scenario',build)
    monkeypatch.setattr(workflow,'start_sumo',lambda *a,**k: starts.append(1))
    with pytest.raises(ValueError):
        workflow.evaluate(seed=4107,loading=2000,penetration=0.,controller_name='pi-external-u',
                          desired_speed_mps=31.29,output=tmp_path/'empty',duration=3)
    assert starts==[]


def test_partial_writer_failure_retains_successful_raw_commit(tmp_path,multi_cav_runtime,monkeypatch):
    api=sys.modules['traci']
    write=api.vehicle.slowDown
    def fail_b(vehicle,*args):
        if vehicle=='b':
            raise RuntimeError('second vehicle write failed')
        return write(vehicle,*args)
    api.vehicle.slowDown=fail_b
    output=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='second vehicle write'):
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name='pi-external-u',
                          desired_speed_mps=31.29,output=output,duration=3)
    telemetry=rows(output/'telemetry.jsonl')
    committed=[x for x in telemetry if x['kind']=='pi_owner_receipt' and x['status']=='committed_raw_proposal']
    assert [(x['vehicle_id'],x['timestamp_s']) for x in committed]==[('a',1.)]
    evidence={x['vehicle_id']:x for x in telemetry if x['kind']=='pi_source_evidence'}
    assert set(evidence)=={'a','b'}
    assert evidence['a']['owner_record'] is not None
    assert evidence['b']['owner_record'] is None
    assert multi_cav_runtime['closed']
    assert json.loads((output/'manifest.json').read_text())['status']=='failed'
    assert not (output/'summary.json').exists()


def test_halfstep_exit_and_id_reuse_resets_prior(tmp_path,multi_cav_runtime,monkeypatch):
    api=sys.modules['traci']
    api.vehicle.getIDList=lambda: ([] if multi_cav_runtime['time']<1 else ['a'] if multi_cav_runtime['time']==1.5 else ['a','b'])
    api.simulation.getArrivedIDList=lambda: []
    api.simulation.getDepartedIDList=lambda: ['a','b'] if multi_cav_runtime['time']==1 else []
    monkeypatch.setattr(workflow,'SumoLocalSensor',lambda *a: SimpleNamespace(capture=lambda vehicle,now_s:
        SensorFrame(vehicle,now_s,10.,0.,30.,0,front=NeighborMeasurement(50.,0.),front_coverage_m=100.,rear_coverage_m=100.)))
    output=tmp_path/'reuse'
    workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name='pi-external-u',
                      desired_speed_mps=31.29,output=output,duration=4)
    evidence={(x['vehicle_id'],x['timestamp_s']):x for x in rows(output/'telemetry.jsonl') if x['kind']=='pi_source_evidence'}
    assert evidence['a',2.]['previous_raw_command_mps']==pytest.approx(21.145)
    assert evidence['b',2.]['previous_raw_command_mps']==10.
    assert evidence['b',2.]['generation']>evidence['b',1.]['generation']

    assert evidence['b',3.]['previous_raw_command_mps']==pytest.approx(evidence['b',2.]['raw_proposal_mps'])
    assert evidence['b',3.]['generation']==evidence['b',2.]['generation']


def test_midphysics_failure_keeps_pi_evidence_and_actual_progress(tmp_path,multi_cav_runtime,monkeypatch):
    prior=workflow.ttc_exposure
    def fail(api,*args):
        if multi_cav_runtime['time']==1.5:
            raise RuntimeError('diagnostic failed after physics')
        return prior(api,*args)
    monkeypatch.setattr(workflow,'ttc_exposure',fail)
    progress=[]
    output=tmp_path/'physics_failure'
    with pytest.raises(RuntimeError,match='diagnostic failed'):
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name='pi-external-u',
                          desired_speed_mps=31.29,output=output,duration=3,progress_callback=progress.append)
    assert progress==[.5,1.,1.5]
    assert len([x for x in rows(output/'telemetry.jsonl') if x['kind']=='pi_source_evidence'])==2
    assert multi_cav_runtime['closed']


def test_manifest_pins_actual_transitive_pi_sources(tmp_path,multi_cav_runtime):
    import hashlib
    from pathlib import Path
    output=tmp_path/'identities'
    workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name='pi-external-u',
                      desired_speed_mps=31.29,output=output,duration=2)
    manifest=json.loads((output/'manifest.json').read_text())
    sources=manifest['pi_transitive_source_sha256']
    assert {Path(p).name for p in sources}>={'pi_owner_adapter.py','pi_external_u.py','pi_saturation_v1.py'}
    for path,digest in sources.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest


def test_actual_telemetry_flush_failure_preserves_pending_evidence(tmp_path,multi_cav_runtime,monkeypatch):
    from pathlib import Path
    original=Path.open
    class BrokenTelemetry:
        def __init__(self,stream):
            self.stream=stream
            self.has_pi=False
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self,*args):
            return self.stream.__exit__(*args)
        def write(self,text):
            self.has_pi |= 'pi_source_evidence' in text
            return self.stream.write(text)
        def flush(self):
            if self.has_pi:
                raise OSError('telemetry flush failed')
            return self.stream.flush()
    def open_file(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        if path.name=='telemetry.jsonl' and args and args[0]=='x':
            return BrokenTelemetry(stream)
        return stream
    monkeypatch.setattr(Path,'open',open_file)
    output=tmp_path/'sink_failure'
    with pytest.raises(OSError,match='telemetry flush failed'):
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name='pi-external-u',
                          desired_speed_mps=31.29,output=output,duration=3)
    rescue=json.loads((output/'pi_evidence_rescue.json').read_text())
    receipts=[x for x in rescue['rows'] if x['kind']=='pi_owner_receipt' and x['status']=='committed_raw_proposal']
    assert {x['vehicle_id'] for x in receipts}=={'a','b'}
    assert json.loads((output/'manifest.json').read_text())['status']=='failed'
    assert multi_cav_runtime['closed']

def test_nonpi_writer_failure_preserves_original_exception(tmp_path,multi_cav_runtime):
    def fail(*args):
        raise RuntimeError('original nonPI writer failure')
    sys.modules['traci'].vehicle.slowDown=fail
    with pytest.raises(RuntimeError,match='original nonPI writer failure'):
        workflow.evaluate(seed=4107,loading=2000,penetration=.2,controller_name='damping',
                          output=tmp_path/'nonpi_failure',duration=3)
    assert multi_cav_runtime['closed']
