import json
import sys

import pytest

from tools.rb2int import target_ops


@pytest.mark.parametrize('action',['launch','status'])
def test_scoring_target_program_is_cpu_only_and_one_use(tmp_path,monkeypatch,action):
    calls = []

    def remote(host,code,timeout):
        compile(code,'<scoring-only-target-extraction>','exec')
        assert host == 'gpuR'
        if action == 'launch':
            assert "CUDA_VISIBLE_DEVICES=''" in code
        else:
            assert 'subprocess.Popen(' not in code
        calls.append(code)
        return json.dumps(dict(passed=True))

    monkeypatch.setattr(target_ops,'ssh',remote)
    receipt = tmp_path/'target.json'
    monkeypatch.setattr(sys,'argv',['target_ops',action,'--receipt',str(receipt)])
    target_ops.main()
    with pytest.raises(ValueError,match='exclusive target'):
        target_ops.main()
    assert len(calls) == 1
