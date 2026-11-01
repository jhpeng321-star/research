import json
from types import SimpleNamespace

import pytest
import torch

from wms.control.entrant_population import bind_entrant_population
from wms.control.fitted_hidden import IDM_ORDER
from wms.control.hidden_state import BirthMix, Population
from wms.model.driver_encoder import PARAMETER_NAMES


def setup(tmp_path, improved=True):
    support = [[30., 1.2, 1.2, 2., 2.], [34., .8, 2., 3., 1.]]
    pool = torch.ones(2, len(PARAMETER_NAMES))
    for i, name in enumerate(IDM_ORDER):
        pool[:, PARAMETER_NAMES.index(name)] = torch.tensor(support)[:, i]
    mix = BirthMix(Population([pool], [pool[:1].clone()], [torch.tensor([.9, .1])]), 190, [969420])
    artifact = dict(schema='wm_phys.entrant_population.v1', parameter_order=IDM_ORDER,
        source_population_sha256='source', support=support, weights=[.1, .9],
        motion_validation_improved=improved, fit_worlds=list(range(969420,969432)),
        conditioning_end_s=300, per_driver_motion_window_s=10.)
    path = tmp_path/'entrant_population.json'
    path.write_text(json.dumps(artifact))
    return mix, path


def test_entrant_reweighting_preserves_joint_support_cavs_and_sampling_counters(tmp_path):
    mix, path = setup(tmp_path)
    original = mix.pools[0].clone()
    cav = mix.candidate(0, 0, 11, cav=True)
    receipt = bind_entrant_population(SimpleNamespace(process=SimpleNamespace(birth_mix=mix)), path, 'source')
    torch.testing.assert_close(mix.weights[0], torch.tensor([.1, .9], dtype=torch.float64))
    assert torch.equal(mix.pools[0], original)
    assert mix.candidate(0, 0, 11, cav=True) == cav
    assert mix.counts == [[0, 0]] and mix.assigned == [0]
    assert receipt['current_driver_posteriors_changed'] is False
    assert receipt['arrival_process_changed'] is False


def test_failed_motion_gate_or_mismatched_support_cannot_be_deployed(tmp_path):
    mix, path = setup(tmp_path, improved=False)
    entries = SimpleNamespace(process=SimpleNamespace(birth_mix=mix))
    with pytest.raises(ValueError, match='unqualified'):
        bind_entrant_population(entries, path, 'source')
    mix, path = setup(tmp_path, improved=True)
    mix.pools[0][0, 0] += 1
    with pytest.raises(ValueError, match='support differs'):
        bind_entrant_population(SimpleNamespace(process=SimpleNamespace(birth_mix=mix)), path, 'source')
