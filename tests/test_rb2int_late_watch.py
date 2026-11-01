"""Prevent overlapping or unintended delayed-panel ownership transfers."""
import json

import pytest

from tools.rb2int.late_watch import old_owners_ready


@pytest.fixture
def owners(tmp_path):
    folders = [tmp_path/name for name in ('collector', 'publisher')]
    for index, folder in enumerate(folders):
        folder.mkdir()
        (folder/'PROCESS.json').write_text(json.dumps(dict(pid=index+1)))
    return folders


def test_no_takeover_until_both_natural_deadlines_and_actual_exits(owners):
    collector, publisher = owners
    assert old_owners_ready(*owners, lambda p: False) == 'wait'
    for folder in owners:
        (folder/'INCOMPLETE_AT_DEADLINE.json').write_text('{}')
    assert old_owners_ready(*owners, lambda p: p['pid'] == 1) == 'wait'
    assert old_owners_ready(*owners, lambda p: p['pid'] == 2) == 'wait'
    assert old_owners_ready(*owners, lambda p: False) == 'ready'


def test_unexpected_source_stop_requires_explicit_recovery(owners):
    (owners[0]/'STOP.json').write_text(json.dumps(dict(error='unexpected source guard')))
    with pytest.raises(ValueError, match='unexpected prior STOP'):
        old_owners_ready(*owners, lambda p: False)


def test_exact_final_guard_waits_for_original_publisher(owners):
    error = "ValueError('progress report changed during collection; preserve edits and assemble explicitly')"
    (owners[0]/'STOP.json').write_text(json.dumps(dict(error=error)))
    assert old_owners_ready(*owners, lambda p: False) == 'wait'


@pytest.mark.parametrize('index', [0, 1])
def test_previously_completed_final_is_never_republished(owners, index):
    (owners[index]/'COMPLETE.json').write_text('{}')
    assert old_owners_ready(*owners, lambda p: True) == 'complete'
