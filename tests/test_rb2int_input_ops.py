import json
import sys

import pytest

from tools.rb2int import input_ops


def test_export_program_is_parseable_and_local_claim_blocks_duplicate_job(tmp_path, monkeypatch):
    calls = []

    def remote(host, code, timeout):
        compile(code, '<causal-export-launch>', 'exec')
        calls.append(code)
        return json.dumps(dict(pid=123, host=host, action='export'))

    monkeypatch.setattr(input_ops, 'ssh', remote)
    receipt = tmp_path/'export.json'
    monkeypatch.setattr(sys, 'argv', ['input_ops', 'export', '--host', 'gpuR', '--receipt', str(receipt)])
    input_ops.main()
    assert json.loads(receipt.read_text())['pid'] == 123
    with pytest.raises(ValueError, match='exclusive local'):
        input_ops.main()
    assert len(calls) == 1


def test_unknown_source_host_and_prohibited_tag_cannot_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['input_ops', 'export', '--host', 'gpuA', '--receipt', str(tmp_path/'a.json')])
    with pytest.raises(ValueError, match='source is gpuR'):
        input_ops.main()
    monkeypatch.setattr(sys, 'argv', ['input_ops', 'export', '--host', 'gpuR', '--tag', 'sealed', '--receipt', str(tmp_path/'b.json')])
    with pytest.raises(ValueError, match='safe own export tag'):
        input_ops.main()
