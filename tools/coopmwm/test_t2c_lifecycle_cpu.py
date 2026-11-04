import json
from pathlib import Path

import pytest

from tools.coopmwm.test_t2c_runtime_cpu import prepare, CPUBackend
from tools.coopmwm.rerun_tracks_runtime import Runtime, state_sha
from tools.coopmwm.rerun_tracks_monitor import read_monitor


def test_smoke_then_full_update_one_same_approval_root_and_ledger(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    smoke = Runtime(cfg, reg, backend, smoke=True)
    original = {a: state_sha(m) for a, m in smoke.models.items()}
    smoke.run()
    cache_bytes = [(smoke.root/f'T2C_AUGMENT_{a}.json').read_bytes() for a in ('B', 'R')]
    smoke_checkpoint = (smoke.output_root/'B_checkpoint_001.pt').read_bytes()
    fresh_backend = CPUBackend()
    def no_repeat(*a, **kw): pytest.fail('once-only augmentation must not repeat after smoke')
    fresh_backend.verify_augmentation = no_repeat
    full = Runtime(cfg, reg, fresh_backend)
    assert {a: state_sha(m) for a, m in full.models.items()} == original
    previous_augmentation_spend = full.spend['augment_verify']
    for a in ('A', 'B', 'R'):
        full.update(a, 1, episodes=2)
    assert full.spend['track2_smoke'] == cfg['smoke']['total_sim_s']
    assert full.spend['augment_verify'] == previous_augmentation_spend > 0
    assert read_monitor(full.root)['updates'] == dict(A=1, B=1, R=1)
    assert read_monitor(smoke.output_root)['updates'] == dict(A=1, B=1, R=1)
    assert (smoke.output_root/'B_checkpoint_001.pt').read_bytes() == smoke_checkpoint
    assert [(full.root/f'T2C_AUGMENT_{a}.json').read_bytes() for a in ('B', 'R')] == cache_bytes
    charges = [json.loads(x) for x in (full.root/'SYNTHETIC_CHARGES.jsonl').read_text().splitlines()]
    ids = [r['id'] for r in charges]
    assert len(ids) == len(set(ids))
    assert any(i.startswith('t2c_smoke_u001') for i in ids) and any(i.startswith('t2c_u001') for i in ids)
    assert sum(r['sim_s'] for r in charges) == sum(full.spend.values()) <= cfg['budget']['total']
    with pytest.raises(FileExistsError): Runtime(cfg, reg, CPUBackend())
    with pytest.raises((FileExistsError, ValueError)): Runtime(cfg, reg, CPUBackend(), smoke=True)


def test_interrupted_smoke_cannot_be_replayed_or_transition_to_full(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    Runtime(cfg, reg, backend, smoke=True)  # claim persisted, no completion receipt
    with pytest.raises(FileExistsError): Runtime(cfg, reg, CPUBackend(), smoke=True)
    with pytest.raises(ValueError, match='interrupted smoke'): Runtime(cfg, reg, CPUBackend())


def test_smoke_transition_rejects_different_approval_binding(tmp_path, monkeypatch):
    from tools.coopmwm.exact_grade import _seal
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    smoke = Runtime(cfg, reg, backend, smoke=True)
    # A valid self-hash alone cannot authorise a transition from another config.
    receipt = _seal(dict(status='CPU_PLANT_SMOKE_PASS', synthetic=True,
        config_sha256='0'*64, registration_sha256=smoke.registration_sha), 'receipt_sha256')
    smoke._write('T2C_SMOKE.json', receipt)
    with pytest.raises(ValueError, match='approval/receipt mismatch'):
        Runtime(cfg, reg, CPUBackend())
    assert not (smoke.root/'T2C_RUNTIME.claim').exists()
