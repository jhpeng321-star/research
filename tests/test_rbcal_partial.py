"""Recovery admission must never turn stopped/duplicate futures into K8."""
import pytest

from tools.rbcal.partial import inventory, matched_observations


def test_admit_complete_world_exclude_partial_and_never_open_reserved(tmp_path):
    shard = tmp_path/'shard0'
    shard.mkdir()
    for k in range(8):
        (shard/f'w969420_k{k}.json').write_text('{}')
    (shard/'w969421_k0.json').write_text('{}')
    # Malformed reserved data must never be decoded or hashed/admitted.
    (shard/'w969432_k0.json').write_text('reserved data not admitted')
    result = inventory(tmp_path)
    assert result['complete_worlds'] == [969420]
    assert result['retained_tasks'] == 9
    assert result['scored_tasks'] == 8
    assert result['excluded_incomplete_tasks'] == 1
    assert all(item['world'] < 969432 for item in result['tasks'])


def test_duplicate_device_future_refused(tmp_path):
    for shard in ('shard0', 'shard1'):
        (tmp_path/shard).mkdir()
        (tmp_path/shard/'w969420_k0.json').write_text('{}')
    with pytest.raises(ValueError, match='duplicate'):
        inventory(tmp_path)


def test_no_complete_world_refused(tmp_path):
    with pytest.raises(ValueError, match='no complete'):
        inventory(tmp_path)


def test_matched_saved_baseline_requires_same_dev_worlds():
    data = {'events': [{'seed': 969420}, {'seed': 969421}],
            'distributions': [{'seed': 969420}, {'seed': 969421}]}
    result = matched_observations(data, [969420])
    assert result == {key: [{'seed': 969420}] for key in data}
    with pytest.raises(ValueError, match='incomplete'):
        matched_observations(data, [969420, 969422])
    with pytest.raises(ValueError, match='development-only'):
        matched_observations(data, [969432])
    data['events'].append({'seed': 969432})
    with pytest.raises(ValueError, match='non-development'):
        matched_observations(data, [969420])
