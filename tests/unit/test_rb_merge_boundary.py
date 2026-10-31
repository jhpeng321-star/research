import pytest

from tools.rbeval.summarize_full_fleet import augment_boundary_events


def fixture():
    audit = dict(target_sha256='target', missing_from_helper=[dict(seed=969400, vid=12,
        time_s=300.5, position_m=50., episode_sha256='episode')])
    return audit, [dict(seed=969400, episode_sha256='episode')]


def test_first_step_omission_is_added_once_without_changing_existing_events():
    audit, rows = fixture()
    events = [dict(seed='969400', arm='native', vid='11', t_merge_s=305., s_merge_m=75.)]
    assert augment_boundary_events(events, audit, rows, 'target') == [dict(seed=969400, vid=12)]
    assert len(events) == 2 and events[0]['vid'] == '11'
    assert augment_boundary_events(events, audit, rows, 'target') == []


def test_a_different_episode_or_target_cannot_relabel_validation():
    audit, rows = fixture()
    with pytest.raises(ValueError, match='another helper'):
        augment_boundary_events([], audit, rows, 'changed')
    rows[0]['episode_sha256'] = 'different'
    with pytest.raises(ValueError, match='identity mismatch'):
        augment_boundary_events([], audit, rows, 'target')


def test_nonboundary_event_is_not_accepted_as_a_target_omission():
    audit, rows = fixture()
    audit['missing_from_helper'][0]['time_s'] = 301.
    with pytest.raises(ValueError, match='identity mismatch'):
        augment_boundary_events([], audit, rows, 'target')
