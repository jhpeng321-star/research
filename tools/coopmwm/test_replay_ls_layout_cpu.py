import json
import pytest
from . import replay_bank as rb, exact_grade as xg


@pytest.mark.parametrize('generation', ['', 'SUMOENV1', 'SUMOPATH1'])
def test_live_label_generation_resolves_its_original_tape_and_snapshot(tmp_path, monkeypatch, generation):
    src = tmp_path/'collection';(src/'labels').mkdir(parents=True)
    key = 'a'*64
    receipt = xg._seal(dict(status='accepted', episode=dict(arm='B'),
        timing=dict(finished_unix=1), verification_id='decision'), 'receipt_sha256')
    suffix = '_'+generation if generation else ''
    (src/'labels'/(key+suffix+'.json')).write_text(json.dumps(
        dict(search_id=key, verification=receipt)))
    branches = src/'label_attempts'/generation/'branches' if generation else src/'branches'
    tape = branches/'decision/cand1/RESULT.json';tape.parent.mkdir(parents=True)
    tape.write_text('{}')
    rows = rb.bank_decisions(tmp_path/'run', 'B', sources=[src])
    assert len(rows) == 1
    row = rows[0]
    assert row['_branches'] == str(branches) and row['_source'] == str(src)
    assert (rb.Path(row['_branches'])/'decision/cand1/RESULT.json').exists()
    monkeypatch.setattr(rb, '_ls_snapshot', lambda root, search_id: (root, search_id))
    assert rb._decision_snapshot(tmp_path/'run', 'B', row, {}) == (src, key)
    assert rb.bank_decisions(tmp_path/'run', 'R', sources=[src]) == []
