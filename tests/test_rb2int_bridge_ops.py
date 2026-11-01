import json
import sys

import pytest

from tools.rb2int import bridge_ops


@pytest.mark.parametrize('action', ['launch', 'status', 'verify'])
def test_bridge_remote_program_compiles_and_operation_receipt_is_exclusive(tmp_path, monkeypatch, action):
    calls = []

    def remote(host, code, timeout):
        compile(code, '<baseline-development-bridge>', 'exec')
        if action == 'status':
            assert 'subprocess.Popen(' not in code
        if action == 'launch':
            assert '--development-direct' in code
        calls.append(code)
        return json.dumps(dict(passed=True))

    monkeypatch.setattr(bridge_ops, 'ssh', remote)
    receipt = tmp_path/'receipt.json'
    monkeypatch.setattr(sys, 'argv', ['bridge_ops', action, '--host', 'gpuR', '--model', 'WM2',
        '--control', '/root/autodl-tmp/c4/rb2int20261101/phys_code_123456abcdef', '--receipt', str(receipt)])
    bridge_ops.main()
    assert json.loads(receipt.read_text())['passed'] is True
    with pytest.raises(ValueError, match='exclusive local'):
        bridge_ops.main()
    assert len(calls) == 1
