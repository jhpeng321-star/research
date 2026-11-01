import copy
import hashlib
import json

import pytest

from tools.rb2break.summarize import tasks, verify_identical_arms


def test_native_intervention_cannot_hide_a_change_in_a_commanded_arm():
    baseline = {(969420, 0):dict(episodes=[dict(arm=arm, value=0) for arm in
        ('native','ctrl0','mappo','pi','circles')])}
    candidate = copy.deepcopy(baseline)
    candidate[969420, 0]['episodes'][0]['value'] = 3
    verify_identical_arms(candidate, baseline, ['ctrl0','mappo','pi','circles'])
    candidate[969420, 0]['episodes'][1]['value'] = 1
    with pytest.raises(ValueError, match='nonintervened arm changed'):
        verify_identical_arms(candidate, baseline, ['ctrl0','mappo','pi','circles'])


def test_task_is_bound_to_unchanged_source_identity(tmp_path):
    raw = json.dumps(dict(worlds=[969420])).encode()
    (tmp_path/'IDENTITY.json').write_bytes(raw)
    task = dict(seed=969420, future=0, identity_sha256=hashlib.sha256(raw).hexdigest(),
        future_sampling=dict(population_audit=dict(conserved=True)))
    (tmp_path/'w969420_k0.json').write_text(json.dumps(task))
    assert len(tasks(tmp_path)) == 1
    (tmp_path/'IDENTITY.json').write_text(json.dumps(dict(worlds=[969420], changed=True)))
    with pytest.raises(AssertionError, match='task identity changed'):
        tasks(tmp_path)


def test_completion_marker_cannot_turn_missing_future_into_complete_data(tmp_path):
    raw = json.dumps(dict(worlds=[969420])).encode()
    (tmp_path/'IDENTITY.json').write_bytes(raw)
    task = dict(seed=969420, future=0, identity_sha256=hashlib.sha256(raw).hexdigest(),
                future_sampling=dict(population_audit=dict(conserved=True)))
    (tmp_path/'w969420_k0.json').write_text(json.dumps(task))
    (tmp_path/'COMPLETE.json').write_text(json.dumps(dict(hashes={
        'w969420_k0.json':'unused', 'w969420_k1.json':'missing'})))
    with pytest.raises(AssertionError, match='declared complete tasks missing'):
        tasks(tmp_path, require_complete=True)
