import numpy as np
import pytest
import torch

from tools.rbint import endpoints


def test_predicted_endpoint_uses_exact60s_and_preserves_observed_ids_not_slots(monkeypatch):
    start = dict(state=dict(present=torch.tensor([[False, True, True, False]]),
                           is_cav=torch.tensor([[False, False, True, False]])),
        vehicle_ids=['unused', 'ramp.9', 'unused2', 'mainline.27'], initial_state_sha256='observed')
    monkeypatch.setattr(endpoints, 'load_observations',
        lambda seed: (start, torch.tensor([3, 1]), {'start.pt':'start', 'start_history.npz':'past'}))
    positions = np.broadcast_to(np.arange(1200)[None, :, None], (2, 1200, 4)).astype(float).copy()
    masks = np.ones_like(positions, dtype=bool)
    masks[1, 20, 0] = False  # exited old ID, then slot reused before60s
    episodes = [dict(seed=969432, initial_sha256='observed') for _ in range(2)]
    endpoints.attach_endpoints60(episodes, dict(positions=positions, masks=masks), 969432, 2)
    a, b = [e['endpoint60'] for e in episodes]
    assert a['vehicle_ids'] == ['mainline.27', 'ramp.9']
    assert a['is_hdv'] == [True, False]
    assert a['s_m'] == [119., 119.]  # after120halfsteps, not index120
    assert b['survived'] == [False, True] and b['s_m'] == [None, 119.]
    assert not a['factual_future_used']


def test_endpoint_export_rejects_another_start_before_attaching_data(monkeypatch):
    start = dict(state=dict(present=torch.tensor([[True]]), is_cav=torch.tensor([[False]])),
                 vehicle_ids=['a'], initial_state_sha256='one')
    monkeypatch.setattr(endpoints, 'load_observations', lambda seed:
        (start, torch.tensor([0]), {'start.pt':'a', 'start_history.npz':'b'}))
    arrays = dict(positions=np.ones((1, 120, 1)), masks=np.ones((1, 120, 1), dtype=bool))
    with pytest.raises(ValueError, match='start identity'):
        endpoints.attach_endpoints60([dict(seed=969432, initial_sha256='another')], arrays, 969432, 1)
