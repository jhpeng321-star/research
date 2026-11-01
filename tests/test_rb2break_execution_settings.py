from pathlib import Path
from types import SimpleNamespace

import torch

from wms.control.execution import load_learned_driver
from wms.control.execution_settings import bind_execution_settings
from wms.control.hidden_state import BirthMix, Population
from wms.model.driver_encoder import PARAMETER_NAMES


def test_admission_uses_same_cav_vector_as_executor_without_changing_hdv_draws(monkeypatch):
    monkeypatch.setenv('C4_EXECUTION_PARITY', '1')  # legacy learned-fit path, explicitly NOT mode2
    artifact = Path(__file__).resolve().parents[1]/'configs/execparity_cav_idm_v1.json'
    pool = torch.arange(1., len(PARAMETER_NAMES)+1)[None]
    population = Population([pool.clone()], [pool.clone()], [torch.ones(1)])
    mix = BirthMix(population, 190, [969420])
    hdv = mix.candidate(0, 0, 13, cav=False)
    state = dict(present=torch.tensor([[True, True, False]]),
                 is_cav=torch.tensor([[True, False, True]]))
    params = {name:torch.full((1, 3), float(i+1)) for i,name in enumerate(PARAMETER_NAMES)}
    original = {name:value.clone() for name,value in params.items()}
    receipt = bind_execution_settings(dict(driver_params=params),
        SimpleNamespace(process=SimpleNamespace(birth_mix=mix)), state, artifact)
    expected = load_learned_driver(str(artifact))
    for name,value in expected.items():
        assert params[name][0, 0] == torch.tensor(value, dtype=params[name].dtype)
        assert torch.equal(params[name][:, 1:], original[name][:, 1:])
        assert abs(mix.candidate(0, 0, 13, cav=True)[name]-value) < 2e-6
    assert mix.candidate(0, 0, 13, cav=False) == hdv
    assert mix.counts == [[0, 0]] and mix.assigned == [0]
    assert receipt['external_executor_changed'] is False


def test_mode2_uses_actual_shared_actuator_not_retired_learned_mirror(monkeypatch, tmp_path):
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    from wms.control import sumo_idm
    pool = torch.ones(1, len(PARAMETER_NAMES))
    mix = BirthMix(Population([pool.clone()], [pool.clone()], [torch.ones(1)]), 190, [969420])
    state = dict(present=torch.tensor([[True, True]]), is_cav=torch.tensor([[True, False]]))
    params = {name:torch.full((1, 2), 9.) for name in PARAMETER_NAMES}
    receipt = bind_execution_settings(dict(driver_params=params),
        SimpleNamespace(process=SimpleNamespace(birth_mix=mix)), state,
        tmp_path/'nonexistent_retired_fit.json')  # must not even open the obsolete artifact
    for name in ('accel', 'decel', 'time_headway', 'min_gap'):
        assert params[name][0, 0] == torch.tensor(sumo_idm.CAV_PARAMS[name])
        assert params[name][0, 1] == 9.
    assert params['v0'][0, 0] == torch.tensor(sumo_idm.allowed_speed_scalar(0, 0.))
    assert receipt['retired_learned_driver_used'] is False
    assert receipt['cav_settings_source'] == 'registered_exact_mode2_executor'
