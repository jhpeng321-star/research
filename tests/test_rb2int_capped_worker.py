"""Claim-preservation/priority checks for the owner operational continuation."""
import json

import pytest

from tools.rb2int.capped_worker import CAPS, priority, recovery_plan


def job(seed, model='FULL-v5'):
    return dict(id=model+'_'+str(seed), seed=seed, model=model, host='gpuA')


def claim(root, entry, *, live=False, complete=True, exited=None):
    directory = root/'jobs'/entry['id']
    directory.mkdir(parents=True)
    (directory/'CLAIM_ONCE.json').write_text(json.dumps(dict(job=entry)))
    (directory/'PROCESS.json').write_text(json.dumps(dict(job=entry, live=live)))
    tasks = directory/'tasks'
    tasks.mkdir()
    if complete:
        (tasks/'COMPLETE.json').write_text('{}')
    if exited is not None:
        (directory/'EXIT.json').write_text(json.dumps(dict(complete_K8=exited)))
    return directory


def complete(directory, seed):
    return (directory/'COMPLETE.json').exists()


def test_completed_and_running_claims_never_become_pending(tmp_path):
    entries = [job(969550), job(969551), job(969432), job(969433), job(969550, 'FULL-v5-LP')]
    claim(tmp_path, entries[0], exited=True)
    claim(tmp_path, entries[1], live=True, complete=False)
    active, recovered, pending, claimed = recovery_plan(tmp_path, entries, complete, lambda p: p['live'])
    assert active == [entries[1]['id']]
    assert recovered == []
    assert {j['id'] for j in pending}.isdisjoint(claimed)
    assert pending == entries[2:]
    assert claimed == sorted([entry['id'] for entry in entries[:2]])


def test_exited_complete_claim_recovered_without_rerun(tmp_path):
    entry = job(969436)
    directory = claim(tmp_path, entry)
    active, recovered, pending, claimed = recovery_plan(tmp_path, [entry], complete, lambda p: p['live'])
    assert not active and not pending
    assert [row[0] for row in recovered] == [directory]
    assert claimed == [entry['id']]
    assert not (directory/'EXIT.json').exists()  # validation does not mutate claims


@pytest.mark.parametrize('complete_claim,exited', [(False, None), (True, False)])
def test_failed_or_incomplete_exited_claim_stops_without_retry(tmp_path, complete_claim, exited):
    entry = job(969436)
    claim(tmp_path, entry, complete=complete_claim, exited=exited)
    with pytest.raises(ValueError, match='never retry'):
        recovery_plan(tmp_path, [entry], complete, lambda p: p['live'])


def test_mismatched_claim_stops_before_mutation(tmp_path):
    entry = job(969436)
    directory = claim(tmp_path, entry)
    (directory/'PROCESS.json').write_text(json.dumps(dict(job=job(969437), live=False)))
    with pytest.raises(ValueError, match='differs'):
        recovery_plan(tmp_path, [entry], complete, lambda p: p['live'])
    assert not (directory/'EXIT.json').exists()


def test_duplicate_or_unregistered_claim_stops(tmp_path):
    entry = job(969436)
    claim(tmp_path, entry)
    with pytest.raises(ValueError, match='duplicate'):
        recovery_plan(tmp_path, [entry, entry], complete, lambda p: p['live'])
    with pytest.raises(ValueError, match='unregistered'):
        recovery_plan(tmp_path, [job(969437)], complete, lambda p: p['live'])


def test_owner_cap_and_stable_reserved_priority():
    entries = [job(969550, 'WM2'), job(969439), job(969550), job(969432), job(969432, 'FULL-v5-LP')]
    assert sorted(entries, key=priority) == [entries[i] for i in (1, 3, 0, 2, 4)]
    assert CAPS == {'gpuA': 2, 'gpuR': 1}
