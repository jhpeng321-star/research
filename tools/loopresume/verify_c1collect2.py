"""Verify paused handoff integrity; never rerun the failed preflight or collectors."""
import ast
import hashlib
import json
from pathlib import Path

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import ROOT, remote


def main():
    doc = json.loads((ROOT/'tools/loopresume/receipts_c1collect2/RESULT.json').read_text())
    assert doc['status'] == 'OPERATIONAL_PAUSED'
    assert doc['charged_total'] + doc['remaining'] == doc['total_cap'] == 210000
    assert not doc['canary_dispatched'] and doc['additional_charged'] == 0
    for name, sha in doc['source'].items():
        data = (ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, name
        ast.parse(data, filename=name)
    remote('gpuA', PRELUDE+f'''
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
assert loop.sha256(r/'C1COLLECT2_RESULT.json')=={doc['remote_result_sha256']!r}
assert loop.sha256(cfg['registration'])==state['registration_sha256']=={doc['registration_sha256']!r}
assert state['sim_s_charged']==139540
assert loop.read_json(cfg['registration'])['budget']['cycle_attempt_allowances']['1']==73866
assert loop.read_json(a/'OPERATIONAL_PAUSED.json')['task']=='S-C1COLLECT2'
assert not previous.live(loop.read_json(a/'PID.json'))
assert loop.read_json(a/'loop/worlds_c01.json')['charged']==36933
assert loop.read_json(r/'C1COLLECT2_RESULT.json')['preflight']['cpuQ']=='NOT_RUN_AFTER_cpuP_FAILURE'
print('PASS: local source syntax/SHA, remote result/register/state SHA, unchanged charges, paused driver; no preflight or simulation rerun')
''')


if __name__ == '__main__':
    main()
