import io
import json
import shlex
import tarfile
from types import SimpleNamespace

import pytest

from tools.rb2int import collect


def test_read_only_empty_snapshot_is_verified_and_does_not_dispatch(tmp_path,monkeypatch):
    state = dict(complete=[],pending=['wm2_969550'],hashes={},read_only=True)
    buffer = io.BytesIO()
    raw = json.dumps(state).encode()
    with tarfile.open(fileobj=buffer,mode='w:gz') as stream:
        member=tarfile.TarInfo('COLLECTION.json');member.size=len(raw)
        stream.addfile(member,io.BytesIO(raw))
    def remote(argv,**kwargs):
        code=shlex.split(argv[-1])[0]
        compile(code,'<read-only final snapshot>','exec')
        assert 'subprocess.Popen(' not in code and 'os.kill(' not in code
        return SimpleNamespace(returncode=0,stdout=buffer.getvalue(),stderr=b'')
    monkeypatch.setattr(collect.subprocess,'run',remote)
    result=collect.collect_host('gpuA',tmp_path/'cache',tmp_path/'receipt.json')
    assert result['complete_world_jobs']==0 and result['pending_world_jobs']==1
    with pytest.raises(ValueError,match='exclusive receipt'):
        collect.collect_host('gpuA',tmp_path/'cache',tmp_path/'receipt.json')
