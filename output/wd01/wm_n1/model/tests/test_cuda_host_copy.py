"""Tiny synthetic CPU/CUDA rollout regression for arrival host conversion."""

from __future__ import annotations

import pytest
import torch

from model import constants as C
from model.net import WmN1Net
from model.rollout import rollout
from synth_model import make_batch, small_cfg


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_rollout_arrival_host_copy_keeps_device_and_seed(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    torch.manual_seed(17)
    device = torch.device(device_name)
    net = WmN1Net(small_cfg()).to(device).eval()
    batch = {name: value.to(device) for name, value in
             make_batch(B=1, K=6, W=2, seed=23).items()}
    assert len(batch) == 6

    first = rollout(net, batch, n_steps=2, n_samples=1, seed=41)
    second = rollout(net, batch, n_steps=2, n_samples=1, seed=41)
    expected = {
        "positions": (1, 3, 1, 6),
        "speeds": (1, 3, 1, 6),
        "lanes": (1, 3, 1, 6),
        "masks": (1, 3, 1, 6),
        "accel": (1, 2, 1, 6),
        "arrivals": (1, 2, 1, C.N_ENTRANCES),
        "insertions": (1, 2, 1, C.N_ENTRANCES),
    }
    for name, shape in expected.items():
        value = first[name]
        assert tuple(value.shape) == shape, name
        assert value.device.type == device_name, name
        assert bool(torch.isfinite(value.float()).all()), name
        assert torch.equal(value, second[name]), name
    assert first["validity"] == second["validity"]
