"""Free-memory gate (mocked probe): pause while low, raise when stuck."""
import pytest

from corpus import memory


def test_default_threshold_is_10gb():
    assert memory.FREE_VIRTUAL_GB == 10.0


def test_gate_pauses_then_resumes():
    samples = iter([
        {"free_virtual_gb": 2.0, "free_ram_gb": 8.0},
        {"free_virtual_gb": 12.5, "free_ram_gb": 8.0},
    ])
    slept = []
    out = memory.require_memory(
        10.0, None, probe=lambda: next(samples), pause_s=1.0,
        max_wait_s=100.0, sleep=lambda s: slept.append(s))
    assert out["free_virtual_gb"] == 12.5
    assert slept == [1.0]


def test_gate_raises_when_stuck():
    stuck = {"free_virtual_gb": 1.0, "free_ram_gb": 1.0}
    with pytest.raises(memory.MemoryGateError):
        memory.require_memory(10.0, probe=lambda: dict(stuck), max_wait_s=0.0,
                              sleep=lambda s: None)


def test_wait_for_memory_alias():
    out = memory.wait_for_memory(
        10.0, probe=lambda: {"free_virtual_gb": 99.0, "free_ram_gb": 99.0},
        max_wait_s=1.0, sleep=lambda s: None)
    assert out["free_virtual_gb"] == 99.0
