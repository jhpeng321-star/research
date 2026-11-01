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


def test_cav_settings_precede_population_and_replay_without_changing_latent_streams(monkeypatch, tmp_path):
    import numpy as np
    from tests.unit.test_hidden_state_futures import fixture, plan
    from wms.model.entries import build_entrance_history
    from wms.control.execution_settings import active_execution_settings
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    model, state, history, original = fixture()
    histories = [build_entrance_history(counter=np.ones((2, 30)))]
    common = dict(n_samples=2, seed=190, row_keys=[969420], entrance_histories=histories)
    late = plan(model, state, history, **common)
    for future in late:
        bind_execution_settings(future.kwargs, SimpleNamespace(process=future.kwargs['entries']),
                                state, tmp_path/'unused.json')
    model._wmphys_match_cav_execution_settings = True
    observed = []
    original_population = Population.from_motion.__func__
    def capture(cls, params, s, mask):
        observed.append(float(params['v0'][0, 3]))
        return original_population(cls, params, s, mask)
    monkeypatch.setattr(Population, 'from_motion', classmethod(capture))
    early = plan(model, state, history, **common)
    settings, _ = active_execution_settings()
    assert observed == [torch.tensor(settings['v0']).item()]
    assert original['v0'][0, 3] == 30.  # encoder-owned tensors were not mutated
    for a, b in zip(late, early):
        for name in PARAMETER_NAMES:
            torch.testing.assert_close(a.kwargs['driver_params'][name], b.kwargs['driver_params'][name], rtol=0, atol=0)
        for cav in (False, True):
            assert a.kwargs['entries'].birth_mix.candidate(0, 0, 11, cav) == b.kwargs['entries'].birth_mix.candidate(0, 0, 11, cav)
        np.testing.assert_array_equal(a.kwargs['entries'].t_next, b.kwargs['entries'].t_next)


def test_known_actuator_supports_future_cav_without_observed_cav_donor(monkeypatch, tmp_path):
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    pool = torch.ones(1, len(PARAMETER_NAMES))
    mix = BirthMix(Population([pool.clone()], [pool[:0].clone()], [torch.ones(1)]), 190, [969420])
    state = dict(present=torch.ones(1, 1, dtype=torch.bool), is_cav=torch.zeros(1, 1, dtype=torch.bool))
    params = {name:torch.ones(1, 1) for name in PARAMETER_NAMES}
    receipt = bind_execution_settings(dict(driver_params=params),
        SimpleNamespace(process=SimpleNamespace(birth_mix=mix)), state, tmp_path/'unused.json')
    assert receipt['cav_known_fallback_rows'] == [0]
    assert mix.candidate(0, 0, 0, cav=True)['accel'] == torch.tensor(1.2).item()
    assert mix.candidate(0, 0, 0, cav=False)['accel'] == 1.
