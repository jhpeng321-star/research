"""CV15 independent saved-evidence mutations of real workflow output; no SUMO."""
import json

import pytest

from output.controller_verification_v15 import independent_review as oracle
from tests.test_cv15_workflow import test_actual_multicav_workflow_passes_independent_fullhorizon_oracle as produce


def write(path, value):
    path.write_text(json.dumps(value),encoding='utf-8')


def lines(path, data):
    path.write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')


def expected(base,suffix):
    run=base/'actual'
    m=oracle.read(run/'manifest.json')
    req=base/'frozen/requests.json'
    e={key:m[key] for key in ('config_sha256','model_sha256','sumo_binary_sha256',
                              'scenario_identity','active_source_sha256')}
    e.update(seed=4108,loading=2700,penetration=.2,duration_s=2400,desired_speed_mps=31.29,
        model_path=m['config']['model_path'],suffix=suffix,fleet_switch=m['fleet_switch'],
        requests_path=str(req),requests_sha256=oracle.sha(req),
        scene_sha256={p.name:oracle.sha(p) for p in (base/'frozen/scenario').iterdir()},
        pi_transitive_source_sha256=m.get('pi_transitive_source_sha256',{}))
    return e


@pytest.fixture
def saved(tmp_path,monkeypatch):
    produce(tmp_path,monkeypatch,'pi-external-u')
    return tmp_path/'actual',expected(tmp_path,'pi-external-u')


def refresh(root):
    m=oracle.read(root/'manifest.json')
    for name in ('summary','telemetry','intervals'):
        m[name+'_sha256']=oracle.sha(root/(name+('.json' if name=='summary' else '.jsonl')))
    write(root/'manifest.json',m)


@pytest.mark.parametrize('fault',['raw','network','missing_writer','hdv_writer','clock','census','U','schedule','service'])
def test_saved_corruption_refused(saved,fault):
    root,e=saved
    if fault in ('raw','network'):
        data=list(oracle.rows(root/'telemetry.jsonl'))
        if fault=='raw':
            next(r for r in data if r['kind']=='pi_source_evidence')['raw_proposal_mps']+=1
        else:
            next(r for r in data if r['kind']=='command' and r['timestamp_s']==599)['network_output_mps2']+=1
        lines(root/'telemetry.jsonl',data)
    elif fault in ('missing_writer','hdv_writer'):
        data=list(oracle.rows(root/'writer_trace.jsonl'))
        if fault=='missing_writer':
            data.pop(0)
        else:
            data[0]['args'][0]='hdv'
        lines(root/'writer_trace.jsonl',data)
    elif fault in ('clock','census'):
        data=list(oracle.rows(root/'physics_steps.jsonl'))
        if fault=='clock':
            data[0]['time_s']=.75
        else:
            next(r for r in data if r['time_s']==600)['vehicles'].pop('a')
        lines(root/'physics_steps.jsonl',data)
    elif fault in ('U','schedule'):
        m=oracle.read(root/'manifest.json')
        if fault=='U':
            m['fleet_controller_identities']['pi']['u_mps']=30.
        else:
            m['fleet_switch']['switch_time_s']=601.
        write(root/'manifest.json',m)
    else:
        s=oracle.read(root/'summary.json')
        s['system_seconds_per_requested_vehicle']+=1
        write(root/'summary.json',s)
    refresh(root)
    result=oracle.verify_run(root,e)
    assert not result['integrity_pass'],result


def test_safety_separate_from_integrity(saved):
    root,e=saved
    data=list(oracle.rows(root/'intervals.jsonl'))
    data[600]['collision_involvements']+=1
    lines(root/'intervals.jsonl',data)
    s=oracle.read(root/'summary.json')
    s['totals']['collision_involvements']+=1
    write(root/'summary.json',s)
    refresh(root)
    result=oracle.verify_run(root,e)
    assert result['integrity_pass'] and not result['safety_pass'],result


def test_three_valid_prefixes_then_boundary_mutants(tmp_path,monkeypatch):
    paths={}
    for arm in ('damping','native','pi-external-u'):
        base=tmp_path/arm
        base.mkdir()
        produce(base,monkeypatch,arm)
        paths[arm]=base/'actual'
    for arm in ('native','pi-external-u'):
        result=oracle.verify_prefix(paths['damping'],paths[arm])
        assert result['integrity_pass'],result
    root=paths['pi-external-u']
    original=list(oracle.rows(root/'telemetry.jsonl'))
    changed=json.loads(json.dumps(original))
    next(r for r in changed if r['kind']=='cav_report' and r['frame']['timestamp_s']==600)['frame']['speed_mps']+=.01
    lines(root/'telemetry.jsonl',changed)
    assert not oracle.verify_prefix(paths['damping'],root)['integrity_pass']
    lines(root/'telemetry.jsonl',original)
    physics=list(oracle.rows(root/'physics_steps.jsonl'))
    next(r for r in physics if r['time_s']==600)['vehicles']['a']['lane_position_m']+=.01
    lines(root/'physics_steps.jsonl',physics)
    # A valid but different boundary position passes each arm's own arithmetic.
    # The independent matching gate must still refuse the causal comparison.
    result=oracle.verify_run(root,expected(tmp_path/'pi-external-u','pi-external-u'))
    assert result['integrity_pass'],result
    assert not oracle.verify_prefix(paths['damping'],root)['integrity_pass']
