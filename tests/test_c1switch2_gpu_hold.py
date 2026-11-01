"""The owner hold blocks every new GPU admission before remote work or claims."""
import pytest

from tools.c1switch2 import bbcontinue, gpuhold, gpuops, modelops, supervise


@pytest.mark.parametrize('operation', [
    lambda: gpuops.start('gpuA', '/root/c1switch2_gpu_bb_unused'),
    lambda: bbcontinue.start(),
    lambda: modelops.dispatch('gpuR', '/root/c1switch2_run_unused', 'bb', 1),
])
def test_gpu_hold_refuses_before_any_remote_or_claim(tmp_path, monkeypatch, operation):
    flag = tmp_path/'GPU_HOLD.json'
    flag.write_text('{}')
    monkeypatch.setattr(gpuhold, 'HOLD', flag)
    for module in (gpuops, bbcontinue, modelops):
        monkeypatch.setattr(module, 'ssh', lambda *a, **k: pytest.fail('remote work during hold'))
    with pytest.raises(ValueError, match='GPU HOLD'):
        operation()
    assert list(tmp_path.iterdir()) == [flag]


def test_supervisor_hold_disarms_even_unused_follower(tmp_path, monkeypatch):
    flag = tmp_path/'GPU_HOLD.json'
    flag.write_text('{}')
    monkeypatch.setattr(gpuhold, 'HOLD', flag)
    monkeypatch.setattr(supervise, 'ssh', lambda *a, **k: pytest.fail('hold must short circuit'))
    assert supervise.admit()['status'] == 'OWNER2014_GPU_HOLD_NO_ADMISSION'


def test_no_flag_preserves_original_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(gpuhold, 'HOLD', tmp_path/'absent.json')
    gpuhold.require_available()
