import json
from pathlib import Path

import pytest

from tools.rbint.panel_monitor import digest, verify_completion


def make_panel(root):
    for i in range(3):
        shard = root/f'shard{i}'
        shard.mkdir()
        (shard/'IDENTITY.json').write_text(json.dumps({'shard': i, 'scientific_identity': 'synthetic'}))
        hashes = {}
        for j, seed in enumerate(range(969420, 969440)):
            if j % 3 != i:
                continue
            for k in range(8):
                path = shard/f'w{seed}_k{k}.json'
                path.write_text(json.dumps({'seed': seed, 'future': k}))
                hashes[path.name] = digest(path)
        (shard/'COMPLETE.json').write_text(json.dumps(dict(tasks=len(hashes),
            hashes=hashes, identity_sha256=digest(shard/'IDENTITY.json'))))


def test_completion_requires_all160_original_hashes_and_identities(tmp_path):
    make_panel(tmp_path)
    assert verify_completion(tmp_path)
    path = tmp_path/'shard0/w969432_k0.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='task changed'):
        verify_completion(tmp_path)


def test_fake_complete_marker_does_not_certify_missing_futures(tmp_path):
    make_panel(tmp_path)
    path = tmp_path/'shard1/COMPLETE.json'
    value = json.loads(path.read_text())
    value['hashes'].pop(next(iter(value['hashes'])))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='noncanonical'):
        verify_completion(tmp_path)
    assert not verify_completion(Path(str(tmp_path)+'_missing'))


@pytest.mark.parametrize('action', ['canary', 'continue', 'monitor', 'status'])
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_remote_orchestration_is_valid_python_without_dispatch(monkeypatch, action, device):
    from tools.rbint import panel_ops
    seen = []
    def inspect_only(host, code, **kwargs):
        compile(code, '<INT remote orchestration>', 'exec')
        seen.append((host, code))
        return 'compile-only, no remote process'
    monkeypatch.setattr(panel_ops, 'ssh', inspect_only)
    monkeypatch.setattr('sys.argv', ['panel_ops', action, '--runtime',
        '/root/rbeval20261031/phys_code_4015b6412cdf', '--name', 'WM-PHYS-FULL-test',
        '--device', device, '--recipe', 'wm_phys_cpu.toml' if device=='cpu' else 'wm_phys.toml'])
    panel_ops.main()
    assert len(seen) == 1 and seen[0][0] == 'gpuA'


def test_int_dispatch_cannot_claim_peer_variants(monkeypatch):
    from tools.rbint import panel_ops
    monkeypatch.setattr('sys.argv', ['panel_ops', 'canary', '--runtime',
        '/root/rbeval20261031/phys_code_4015b6412cdf', '--name', 'WM-PHYS-A2'])
    monkeypatch.setattr(panel_ops, 'ssh', lambda *a, **k: pytest.fail('must refuse before SSH'))
    with pytest.raises(ValueError, match='INT owns FULL'):
        panel_ops.main()
