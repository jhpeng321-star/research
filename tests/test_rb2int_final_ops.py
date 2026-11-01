import json
import sys

import pytest

from tools.rb2int import final_ops


def test_deployment_has_only_causal_registered_rosters_and_fixed_source_release():
    for host in ('gpuA','gpuR'):
        files = final_ops.deployment_files(host,'/root/autodl-tmp/c4/rb2int20261101/phys_code_123456abcdef')
        release = json.loads(files['RELEASE.json'])
        assert release['status'] == 'READY'
        assert release['original_model_config']['sha256']
        for name,raw in files.items():
            if not name.startswith('panels/'):
                continue
            panel = json.loads(raw)
            assert panel['purpose'] == 'final_comparison'
            assert set(panel['worlds']).isdisjoint(range(969400,969432))
            assert panel['real_observables'].endswith('/prefix_observables.json')
            assert len(panel['worlds']) == (50 if name in ('panels/WM2.json','panels/FULL-v4.json') else 58)


@pytest.mark.parametrize('action',['deploy','launch','status'])
def test_final_operation_program_is_parseable_and_local_receipt_is_exclusive(tmp_path,monkeypatch,action):
    calls = []

    def remote(host,code,timeout):
        compile(code,'<registered-final-supervisor-operation>','exec')
        if action == 'status':
            assert 'subprocess.Popen(' not in code
        calls.append(code)
        return json.dumps(dict(passed=True))

    monkeypatch.setattr(final_ops,'ssh',remote)
    monkeypatch.setattr(final_ops,'deployment_files',lambda *a: {'RELEASE.json':b'{}','RUN_PLAN.json':b'{}'})
    receipt = tmp_path/'operation.json'
    monkeypatch.setattr(sys,'argv',['final_ops',action,'--host','gpuA',
        '--control','/root/autodl-tmp/c4/rb2int20261101/phys_code_123456abcdef','--receipt',str(receipt)])
    final_ops.main()
    assert json.loads(receipt.read_text())['passed'] is True
    with pytest.raises(ValueError,match='exclusive local'):
        final_ops.main()
    assert len(calls) == 1
