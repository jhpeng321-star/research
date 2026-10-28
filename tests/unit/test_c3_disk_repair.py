"""Lossless disk repair and collect-only continuation must not replay policy work."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name):
    path = Path(__file__).resolve().parents[2] / f'tools/longint/{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('apply', [False, True])
def test_dedup_preserves_paths_and_every_byte(tmp_path, monkeypatch, apply):
    mod = load('disk_repair_c3')
    work = tmp_path / 'work'
    files = []
    for seed in (968183, 968184):
        path = work / f'w{seed}/C4-LOOP14-COOPON-20261026/c03/inputs/wm/wm_s.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'identical frozen checkpoint')
        files.append(path)
        array = path.parents[2] / 'artifact/pairs/episodes/one/observed.npy'
        array.parent.mkdir(parents=True)
        array.write_bytes(b'identical observable array')
        files.append(array)
    receipt = tmp_path / 'autonomy/receipt.json'
    receipt.parent.mkdir()
    original = {p: (p.read_bytes(), p.stat().st_ino) for p in files}
    monkeypatch.setattr(mod, 'WORK', work)
    monkeypatch.setattr(mod, 'RECEIPT', receipt)
    mod.dedup(apply)
    assert all(p.is_file() and p.read_bytes() == data for p, (data, _) in original.items())
    assert receipt.exists() == apply
    if apply:
        assert files[0].stat().st_ino == files[2].stat().st_ino
        assert files[1].stat().st_ino == files[3].stat().st_ino
    else:
        assert all(p.stat().st_ino == inode for p, (_, inode) in original.items())


def test_disk_continuation_never_retrains_or_finetunes(tmp_path, monkeypatch):
    if os.name != 'posix':
        pytest.skip('remote coordinator uses POSIX locks')
    mod = load('continue_c3')
    monkeypatch.setattr(mod, 'ARM', tmp_path)
    (tmp_path / 'loop').mkdir()
    (tmp_path / 'loop/STOP_DRIVER').touch()
    phases = {'wm_ft': {'status': 'pending'}, 'convert': {'receipt': {'pairs': 'converted'}}}
    state = {'cycles': [{'cycle': 3, 'phases': phases}], 'sim_s_charged': 250339, 'policy': {}}
    calls, statuses = [], []

    def run_phase(cfg, phase):
        calls.append(phase)
        if phase == 'collect':
            state['sim_s_charged'] += 14205
        return 0

    loop = SimpleNamespace(load_config=lambda p: {}, load_state=lambda c: state, run_phase=run_phase)
    monkeypatch.setattr(mod.base, 'setup', lambda: loop)
    monkeypatch.setattr(mod.base, 'locks', lambda: [object()])
    monkeypatch.setattr(mod.base, 'status', lambda l, s, **kw: statuses.append(s))
    monkeypatch.setattr(mod, 'install', lambda l: None)
    assert mod.drive() == 0
    assert calls == ['collect', 'convert']
    assert statuses[-1] == 'WAIT_MAIN_C3_WMFT'
    assert state['sim_s_charged'] == 264544
    assert phases['wm_ft']['status'] == 'pending'
