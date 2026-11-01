import json

import numpy as np
import pytest
import torch

from tools.rb2int.inputs import RegisteredObservations, validate_panel
from tools.rbeval.core import digest


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def panel_fixture(root):
    reg = root/'registration.json'
    write(reg, dict(final_comparison_worlds=[969550], fitting_worlds=[969420]))
    inputs = root/'inputs'
    write(inputs/'expected_hashes.json', {})
    write(inputs/'data/design.json', dict(worlds=[dict(seed=969550)]))
    real = root/'real.json'
    write(real, dict(episodes=[]))
    release = root/'release.json'
    write(release, dict(status='READY', registration_sha256=digest(reg), exposure_audit_passed=True,
                       execution_purposes=['final_comparison']))
    panel = root/'panel.json'
    write(panel, dict(schema='rb2int-panel-inputs/1', registration_path=str(reg),
        registration_sha256=digest(reg), observations_root=str(inputs), worlds=[969550],
        observation_registry_sha256=digest(inputs/'expected_hashes.json'),
        real_observables=str(real), real_observables_sha256=digest(real),
        release_path=str(release), release_sha256=digest(release)))
    return panel


@pytest.mark.parametrize('defect', ['unregistered', 'development', 'registry_tamper', 'no_clearance'])
def test_changed_release_or_inputs_cannot_enter_final_evaluator(tmp_path, defect):
    path = panel_fixture(tmp_path)
    panel = json.loads(path.read_text())
    assert validate_panel(path)['worlds'] == [969550]
    if defect == 'unregistered':
        panel['worlds'] = [969551]
    elif defect == 'development':
        panel['worlds'] = [969420]
    elif defect == 'registry_tamper':
        write(tmp_path/'inputs/expected_hashes.json', dict(changed=True))
    else:
        release = json.loads((tmp_path/'release.json').read_text())
        release['exposure_audit_passed'] = False
        write(tmp_path/'release.json', release)
        panel['release_sha256'] = digest(tmp_path/'release.json')
    write(path, panel)
    with pytest.raises(ValueError):
        validate_panel(path)


def recorded_start(root, *, queues=True):
    job = root/'data/jobs/s969550_ctrl0'
    job.mkdir(parents=True)
    start = dict(t0=300, seed=969550, initial_state_sha256='initial',
        state=dict(present=torch.tensor([[True, False, True]])),
        vehicle_ids=['a', 'absent', 'b'], past_entries=[dict(time=1, lane=0)])
    if queues:
        start['observed_queue_history'] = [[300, 0, 0]]
    torch.save(start, job/'start.pt')
    np.savez(job/'start_history.npz', vid_index=np.tile([2, 1, 0], (241, 1)),
             kin=np.zeros((241, 3, 7)), mask=np.ones((241, 3)))
    write(job/'result.json', dict(metrics=dict(initial_state_sha256='initial'),
        output_hashes={n: digest(job/n) for n in ('start.pt', 'start_history.npz')}))
    write(root/'expected_hashes.json', {f'jobs/s969550_ctrl0/{n}': digest(job/n)
        for n in ('start.pt', 'start_history.npz', 'result.json')})
    return job


def test_new_registered_world_uses_numeric_vid_map_not_packed_ordinals(tmp_path):
    job = recorded_start(tmp_path)
    reader = RegisteredObservations(tmp_path, [969550])
    _, ids, _ = reader.load(969550)
    assert ids.tolist() == [2, 0]
    _, motion = reader.motion(969550)
    assert motion['time_s'][0] == 180 and motion['time_s'][-1] == 300
    with pytest.raises(ValueError, match='explicit observable registry'):
        reader.load(969551)
    # A fresh reader cannot silently accept bytes changed after registry creation.
    with (job/'start_history.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='input hash differs'):
        RegisteredObservations(tmp_path, [969550]).load(969550)


def test_missing_source_sensing_is_not_replaced_by_zero_queue(tmp_path):
    recorded_start(tmp_path, queues=False)
    with pytest.raises(ValueError, match='observed source counters'):
        RegisteredObservations(tmp_path, [969550]).load(969550)


def test_development_bridge_release_cannot_be_used_for_final_forecasts(tmp_path):
    path = panel_fixture(tmp_path)
    panel = json.loads(path.read_text())
    release = json.loads((tmp_path/'release.json').read_text())
    release['execution_purposes'] = ['development_bridge']
    write(tmp_path/'release.json', release)
    panel['release_sha256'] = digest(tmp_path/'release.json')
    write(path, panel)
    with pytest.raises(ValueError, match='execution purpose'):
        validate_panel(path)
