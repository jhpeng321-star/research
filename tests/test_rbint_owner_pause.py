"""Collector/owner handover must not race publication or signal workers."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.rbint.continue_noon import owner_pause


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_ack_precedes_owner_edit_and_resume_keeps_receipts(tmp_path):
    request = tmp_path/'PAUSE_edit.json'
    write(request, dict(owner='S-RB-INT', reason='engineering intake'))
    def owner_edits(_):
        ack = json.loads((tmp_path/'PAUSED_edit.json').read_text())
        assert ack['between_publications'] and ack['pid'] == os.getpid()
        assert len(ack['request_sha256']) == 64
        write(tmp_path/'RESUME_edit.json', dict(owner='S-RB-INT'))
    with patch('tools.rbint.continue_noon.time.sleep', side_effect=owner_edits):
        owner_pause(tmp_path)
    assert len(list(tmp_path.glob('*.json'))) == 3
    original = (tmp_path/'PAUSED_edit.json').read_bytes()
    owner_pause(tmp_path)
    assert (tmp_path/'PAUSED_edit.json').read_bytes() == original


def test_pause_cannot_extend_hard_deadline(tmp_path):
    write(tmp_path/'PAUSE_edit.json', dict(owner='S-RB-INT', reason='intake'))
    with pytest.raises(RuntimeError, match='publication deadline'):
        owner_pause(tmp_path, deadline=0)
    assert (tmp_path/'PAUSED_edit.json').is_file()


def test_foreign_request_is_not_acknowledged(tmp_path):
    write(tmp_path/'PAUSE_edit.json', dict(owner='peer', reason='intake'))
    with pytest.raises(AssertionError):
        owner_pause(tmp_path)
    assert not (tmp_path/'PAUSED_edit.json').exists()


def test_foreign_resume_does_not_authorize_collection(tmp_path):
    write(tmp_path/'PAUSE_edit.json', dict(owner='S-RB-INT', reason='intake'))
    write(tmp_path/'RESUME_edit.json', dict(owner='peer'))
    with pytest.raises(AssertionError):
        owner_pause(tmp_path)
