import json
from . import ls_validity_ops as vo


def test_same_second_admissions_preserve_both_receipts(monkeypatch, tmp_path):
    monkeypatch.setattr(vo.ops, 'EVIDENCE', tmp_path)
    (tmp_path/'LS_VALIDITY_REGISTRATION.json').write_text('{}')
    for host in ('gpuA', 'gpuR'):
        (tmp_path/f'LS_VALIDITY_GPU_{host}_1.json').write_text(json.dumps(
            dict(workers=[dict(ready=True, owner={}, gates=8)])))
    ticks = iter([1000000001, 1000000002])
    monkeypatch.setattr(vo.time, 'time_ns', lambda: next(ticks))
    def remote(host, code, **kwargs):
        compile(code, '<serialized-endpoint-admission>', 'exec')
        assert "LS_ENDPOINT_ADMISSION.claim" in code and 'fcntl.LOCK_EX' in code
        assert "str(os.getpid())+'.'+str(time.time_ns())" in code
        return json.dumps(dict(unix=1.25, admitted_slots=['qualified']))
    monkeypatch.setattr(vo, 'remote', remote)
    vo.admit(); vo.admit()
    paths = sorted(tmp_path.glob('LS_VALIDITY_ADMISSION_*.json'))
    assert len(paths) == 2
    assert paths[0].read_bytes() == paths[1].read_bytes()
