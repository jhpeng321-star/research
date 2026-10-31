import json
import sys
from pathlib import Path

import pytest

from tools.rbint.panel_monitor import available_memory, digest, verify_completion


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


@pytest.mark.parametrize('action', ['canary', 'continue', 'monitor', 'status', 'stop-canary'])
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
        '--device', device, '--recipe', 'wm_phys_cpu.toml' if device=='cpu' else 'wm_phys.toml',
        '--reason', 'synthetic compile-only operational stop'])
    panel_ops.main()
    assert len(seen) == 1 and seen[0][0] == 'gpuA'


def test_int_dispatch_cannot_claim_peer_variants(monkeypatch):
    from tools.rbint import panel_ops
    monkeypatch.setattr('sys.argv', ['panel_ops', 'canary', '--runtime',
        '/root/rbeval20261031/phys_code_4015b6412cdf', '--name', 'WM-PHYS-A2'])
    monkeypatch.setattr(panel_ops, 'ssh', lambda *a, **k: pytest.fail('must refuse before SSH'))
    with pytest.raises(ValueError, match='INT owns FULL'):
        panel_ops.main()


def test_cpu_headroom_respects_container_limit_not_only_host_ram(tmp_path):
    proc, cgroup = tmp_path/'proc', tmp_path/'cgroup'
    proc.mkdir(); cgroup.mkdir()
    (proc/'meminfo').write_text('MemAvailable: 20971520 kB\n')  #20GiB
    (cgroup/'memory.max').write_text(str(16*2**30))
    (cgroup/'memory.current').write_text(str(14*2**30))
    assert available_memory(proc, cgroup) == 2*2**30
    (cgroup/'memory.max').write_text('max')
    assert available_memory(proc, cgroup) == 20*2**30


@pytest.mark.parametrize('action', ['canary', 'continue', 'monitor', 'status'])
def test_endpoint_protocol_has_exact_own_namespace_and_original_model_runtime(monkeypatch, action):
    from tools.rbint import panel_ops
    seen = []
    def inspect_only(host, code, **kwargs):
        compile(code, '<endpoint orchestration>', 'exec')
        assert "baseline_endpoint=True" in code
        assert '/root/autodl-tmp/c4/setupaudit20261030/fix' in code
        assert 'evaluate_WM2_60.py' in code
        seen.append(code)
        return 'compile-only'
    monkeypatch.setattr(panel_ops, 'ssh', inspect_only)
    monkeypatch.setattr(sys, 'argv', ['panel_ops', action, '--runtime',
        '/root/rbeval20261031/phys_code_f378ac603307', '--name', 'WM2-HDV60',
        '--baseline-endpoint', '--host', 'gpuR'])
    panel_ops.main()
    assert len(seen) == 1
