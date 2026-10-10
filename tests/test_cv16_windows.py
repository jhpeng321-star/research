"""Real Windows sharing denial; temporary accounting files, no simulator."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path

import pytest
from traffic_shepherd import budget_guard as module

pytestmark=pytest.mark.skipif(os.name!='nt',reason='real Windows sharing semantics')


def deny_delete(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
        wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.CloseHandle.restype=wintypes.BOOL
    handle=create(str(path),0x80000000,0x1|0x2,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    closed=False
    def close():
        nonlocal closed
        if not closed:
            assert kernel.CloseHandle(handle)
            closed=True
    return close


def ledger(tmp_path):
    path=tmp_path/'ledger.json'
    data=dict(schema='campaign_budget_ledger/1',campaigns={},reservations=[
        dict(id='test:1',campaign='test',category='fixture',amount_s=2400,completed_simulated_s=190.,status='reserved')])
    path.write_text(json.dumps(data),encoding='utf-8')
    return path


def test_real_reader_denial_recovers_without_relaunch(tmp_path,monkeypatch):
    path=ledger(tmp_path)
    close=deny_delete(path)
    original=module.os.replace
    errors=[]
    delays=[]
    def replace(src,dst):
        try:
            return original(src,dst)
        except PermissionError as exc:
            errors.append(exc.winerror)
            raise
    def sleep(delay):
        delays.append(delay)
        # Release the real held OS handle only after an actual replace denial.
        assert errors
        close()
    monkeypatch.setattr(module.os,'replace',replace)
    monkeypatch.setattr(module.time,'sleep',sleep)
    try:
        module.BudgetGuard(path).mark_progress('test:1',190.5)
    finally:
        close()
    assert len(errors)==1 and errors[0] in (5,32,33)
    assert delays==[.05]
    data=json.loads(path.read_text())
    assert len(data['reservations'])==1
    assert data['reservations'][0]['amount_s']==2400
    assert data['reservations'][0]['completed_simulated_s']==190.5
    assert not path.with_suffix('.tmp').exists()


def test_real_persistent_reader_denial_is_bounded_and_preserves_old(tmp_path,monkeypatch):
    path=ledger(tmp_path)
    original_bytes=path.read_bytes()
    close=deny_delete(path)
    original=module.os.replace
    calls=[]
    delays=[]
    def replace(src,dst):
        calls.append((Path(src),Path(dst)))
        return original(src,dst)
    monkeypatch.setattr(module.os,'replace',replace)
    monkeypatch.setattr(module.time,'sleep',delays.append)
    try:
        with pytest.raises(PermissionError):
            module.BudgetGuard(path).mark_progress('test:1',190.5)
        assert len(calls)==5 and delays==[.05]*4
        assert path.read_bytes()==original_bytes
        assert json.loads(path.with_suffix('.tmp').read_text())['reservations'][0]['completed_simulated_s']==190.5
    finally:
        close()
