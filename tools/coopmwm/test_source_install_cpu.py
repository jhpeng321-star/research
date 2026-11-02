import hashlib
import os
import pytest
from .source_install import install_source


def test_install_detaches_shared_source_and_preserves_original(tmp_path):
    peer=tmp_path/'serving.py';peer.write_bytes(b'qualified peer source')
    slot=tmp_path/'slot.py'
    if os.name=='posix':slot.symlink_to(peer)
    else:os.link(peer,slot)
    original=hashlib.sha256(peer.read_bytes()).hexdigest()
    backup=tmp_path/'backup/source.py'
    row=install_source(slot,backup,b'new qualified source',original)
    assert peer.read_bytes()==backup.read_bytes()==b'qualified peer source'
    assert slot.read_bytes()==b'new qualified source'
    assert row['original_sha256']==original and not slot.is_symlink()


def test_existing_approved_incoming_bytes_adopted_without_overwriting_backup(tmp_path):
    target=tmp_path/'slot.py';target.write_bytes(b'approved incoming')
    backup=tmp_path/'old.py';backup.write_bytes(b'older original')
    install_source(target,backup,b'approved incoming','older-binding-sha')
    assert backup.read_bytes()==b'older original'
    target.write_bytes(b'unknown')
    with pytest.raises(AssertionError):install_source(target,backup,b'approved incoming','older-binding-sha')
    assert target.read_bytes()==b'unknown'
