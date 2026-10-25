"""CUDA allocator guard before the trainer's first model allocation (CPU-only tests)."""
from types import SimpleNamespace

import pytest
import torch

from wms.train.settings import TrainSettings
from wms.train.trainer import WmSTrainer


@pytest.mark.parametrize("device,fraction,expected_index", [
    ("cuda", "0.41", 2),
    ("cuda:1", "0.41", 1),
    ("cuda", None, None),
    ("cpu", "0.41", None),
])
def test_allocator_cap_resolves_device_before_model_allocation(
        tmp_path, monkeypatch, device, fraction, expected_index):
    settings = SimpleNamespace(uses_hard_cap=False, device=device, threads=1, seed=3)
    monkeypatch.setattr(TrainSettings, "from_config", lambda _cfg: settings)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "set_num_threads", lambda _n: None)
    monkeypatch.setattr(torch, "manual_seed", lambda _seed: None)
    if fraction is None:
        monkeypatch.delenv("C4_GPU_MEM_FRACTION", raising=False)
    else:
        monkeypatch.setenv("C4_GPU_MEM_FRACTION", fraction)
    caps, queries = [], []

    def current_device():
        queries.append(True)
        return 2

    def cap(value, index):
        # Match PyTorch's allocator contract without allocating GPU memory.
        assert isinstance(index, int) or (isinstance(index, torch.device) and index.index is not None)
        caps.append((value, index))

    class BeforeAllocation(Exception):
        pass

    def allocate(target):
        assert target == torch.device(device)
        assert caps == ([] if expected_index is None else [(0.41, expected_index)])
        assert len(queries) == int(device == "cuda" and fraction is not None)
        raise BeforeAllocation

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", cap)
    with pytest.raises(BeforeAllocation):
        WmSTrainer(None, SimpleNamespace(to=allocate), None, out_dir=tmp_path)
