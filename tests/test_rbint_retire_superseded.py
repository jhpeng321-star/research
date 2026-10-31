import copy
import pytest

from tools.rbint.retire_superseded import validate_owned


def owned():
    root, runtime, source = '/own/panel', '/own/runtime', 'frozen-source'
    spec = dict(owner='S-RB-INT', variant='FULL', runtime=runtime, stage=dict(commit=source))
    workers = [dict(pid=10+i, boot_id='boot', start_ticks='100', argv=[
        runtime+'/tools/rbint/evaluate_FULL.py', '--out', root+f'/shard{i}']) for i in range(3)]
    return spec, workers, root, runtime, source


def test_exact_owned_three_shard_binding():
    validate_owned(*owned())


@pytest.mark.parametrize('bad', ['owner', 'source', 'runtime', 'output', 'duplicate', 'entry'])
def test_foreign_or_reused_receipt_is_never_authorized(bad):
    args = copy.deepcopy(owned())
    spec, workers, *_ = args
    if bad == 'owner': spec['owner'] = 'S-RB-RUNA'
    if bad == 'source': spec['stage']['commit'] = 'new-source'
    if bad == 'runtime': spec['runtime'] = '/peer/runtime'
    if bad == 'output': workers[0]['argv'][-1] = '/peer/shard0'
    if bad == 'duplicate': workers[1] = workers[0]
    if bad == 'entry': workers[0]['argv'][0] = '/peer/evaluate_FULL.py'
    with pytest.raises(AssertionError): validate_owned(*args)


def test_retirement_remote_program_compiles_without_execution(monkeypatch):
    from tools.rbint import retire_superseded
    seen = []
    def inspect_only(host, code, **kwargs):
        assert host == 'gpuA'
        compile(code, '<own retirement>', 'exec')
        assert 'pidfd_send_signal' in code and 'range(969432,969440)' in code
        seen.append(code)
        return 'compile-only'
    monkeypatch.setattr(retire_superseded, 'ssh', inspect_only)
    monkeypatch.setattr('sys.argv', ['retire', '--name', 'WM-PHYS-FULL-v1'])
    retire_superseded.main()
    assert len(seen) == 1
