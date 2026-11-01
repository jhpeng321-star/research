import json
import sys

import pytest

from tools.rb2int import smoke_ops


def test_owned_smoke_launch_is_exclusive_and_remote_program_compiles(tmp_path, monkeypatch):
    calls = []

    def remote(host, code, timeout):
        compile(code, '<owned-development-smoke>', 'exec')
        calls.append(code)
        return json.dumps(dict(pid=123, start_ticks='456', boot_id='example'))

    monkeypatch.setattr(smoke_ops, 'ssh', remote)
    receipt = tmp_path/'launch.json'
    monkeypatch.setattr(sys, 'argv', ['smoke_ops', 'launch', '--host', 'gpuA',
        '--runtime', '/root/autodl-tmp/c4/rb2int20261101/phys_code_123456abcdef',
        '--recipe', 'wm_phys_v5.toml', '--kind', 'fivearm600', '--device', 'cuda',
        '--receipt', str(receipt)])
    smoke_ops.main()
    assert json.loads(receipt.read_text())['pid'] == 123
    with pytest.raises(ValueError, match='new local launch receipt'):
        smoke_ops.main()
    assert len(calls) == 1


def test_status_program_is_read_only_and_requires_exact_stage_namespace(monkeypatch):
    calls = []

    def remote(host, code, timeout):
        compile(code, '<owned-development-status>', 'exec')
        calls.append(code)
        assert 'subprocess.Popen(' not in code
        assert 'os.kill(' not in code
        return '{}'

    monkeypatch.setattr(smoke_ops, 'ssh', remote)
    argv = ['smoke_ops', 'status', '--host', 'gpuR', '--runtime',
        '/root/autodl-tmp/c4/rb2int20261101/phys_code_123456abcdef', '--recipe', 'wm_phys_v5_lp.toml']
    monkeypatch.setattr(sys, 'argv', argv)
    smoke_ops.main()
    argv[argv.index('--runtime')+1] = '/root/some-peer/runtime'
    with pytest.raises(ValueError, match='immutable INT stage'):
        smoke_ops.main()
    assert len(calls) == 1
