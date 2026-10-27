"""Budget, scientific no-retry, and detached/owner-priority safety regressions."""
import ast
import os
from pathlib import Path
import sys
import types
import json
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('AUTONOMY_SOURCE_DIR', str(ROOT / 'tools/autonomy')))


def module(name):
    # fcntl is unavailable on the Windows transport host; only pure helpers here.
    sys.modules.setdefault('fcntl', types.SimpleNamespace())
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, SOURCE / f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_scientific_failures_do_not_retry():
    remote = module('remote')
    for text in ('early signal gate: independently rescored J did not decrease',
                 'native smoke harmful', 'nonfinite update', '60s scientific horizon gate failed',
                 'WM acceptance REJECT', 'native-relative gate WM/config provenance mismatch'):
        assert remote.scientific(text)
    assert not remote.scientific('SSH connection reset by peer')
    assert not remote.scientific('CUDA out of memory')


def test_pause_precedes_any_wait_or_launch():
    source = (SOURCE / 'remote.py').read_text()
    tree = ast.parse(source)
    drive = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'drive')
    assert isinstance(drive.body[0], ast.If)
    assert ast.unparse(drive.body[0].test) == 'OFF'
    assert isinstance(drive.body[0].body[0], ast.Raise)


def test_no_name_signals_and_detached_child():
    source = (SOURCE / 'remote.py').read_text()
    assert 'start_new_session=True' in source
    assert 'subprocess.DEVNULL' in source
    assert 'pkill' not in source and 'pgrep' not in source
    assert "assert live(child)" in source  # timeouts may signal only this recorded child


def test_preflight_stages_exact_native_gate_sidecar(tmp_path, monkeypatch):
    remote = module('remote')
    remote.R = tmp_path
    remote.status = Mock()
    copied = []
    fake = types.SimpleNamespace(host=lambda cfg, box: {'python': 'python', 'code_root': '/runtime', 'ssh': ['ssh', box]},
        sync=lambda cfg, src_host, src, box, dst, log: copied.append((str(src), box, dst)),
        write_atomic=Mock())
    remote.loop = fake
    monkeypatch.setattr(remote.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps({'status': 'PASS', 'charged_sim_s': 0}), stderr=''))
    boxes = remote.preflight({'collect': {'work_root': '/collector/work'}},
        {'policy': {'checkpoint': '/phase/policy_latest.pt', 'sha256': 'p'}}, [])
    assert boxes == ['cpuP', 'cpuQ']
    assert sum(src == '/phase/native_gate_result.json' for src, _, _ in copied) == 2
    assert all(dst.endswith('native_gate_result.json') for src, _, dst in copied if src.endswith('native_gate_result.json'))


@pytest.fixture
def collection(tmp_path, monkeypatch):
    native = types.ModuleType('wms.control.nativeinit')
    native.artifact_world_delta = lambda *a: -1
    native.smoke_guard = lambda a: {'verdict': 'inconclusive', 'passed': False}
    monkeypatch.setitem(sys.modules, 'wms.control.nativeinit', native)
    reg = {'budget': {'sim_s_total': 280000, 'cycle_attempt_allowances': {'2': 73866}},
           'collection': {'cycle_long_attempt_allowances': {'2': 15600}},
           'stop': {'max_branch_incidents_per_cycle': 0}}
    cfg = {'collect': {'worlds_per_cycle': 13, 'branch_points': 2,
                       'long_group': {'enabled': True}, 'long_single_hold': {'enabled': True}},
           'fastloop': {'workers_per_box': 7}}
    state = {'cycles': [{'cycle': 2}], 'policy': {'sha256': 'p'},
             'accepted_wm': {'sha256': 'w'}, 'sim_s_charged': 176473, 'seeds_used': []}
    seeds = list(range(968170, 968183))
    calls = []
    Refused = type('Refused', (Exception,), {})
    def write(path, obj):
        path.write_text(json.dumps(obj))
    def append(path, obj):
        with path.open('a') as stream:
            stream.write(json.dumps(obj) + '\n')
    def world(l, c, s, p, r, item, box, attempt):
        # The append-only charge and saved state MUST precede the world runner.
        ledger = [json.loads(x) for x in (tmp_path / 'seed_ledger.jsonl').read_text().splitlines()]
        assert any(x.get('seed') == item['seed'] and x.get('host') == box for x in ledger)
        calls.append((item['seed'], box, attempt))
        return {'artifact': 'valid', 'case': item, 'collect': {'branch_collisions': 0, 'branch_teleports': 0}}
    fast = types.SimpleNamespace(cases=lambda c, s: [
        {'index': i, 'seed': seed, 'box': ('cpuP', 'cpuQ')[i % 2]} for i, seed in enumerate(s)],
        artifact_valid=lambda a: a == 'valid', smoke_stops=lambda c, k, s: s['verdict'] == 'harmful',
        _world=world)
    loop = types.SimpleNamespace(fast_module=lambda: fast, Refused=Refused,
        check_registration=lambda c: reg, allocate_seeds=lambda *a: seeds,
        collect_sim_s=lambda c: 36933, long_group_sim_s=lambda c: 7800,
        write_atomic=write, append=append, read_json=lambda p: json.loads(p.read_text()),
        save_state=lambda c, s: write(tmp_path / 'state.json', s))
    run = lambda boxes=('cpuP', 'cpuQ'): module('collector').collect(
        loop, cfg, state, {'attempt': 1}, tmp_path, lambda *a: list(boxes), Mock(), Mock())
    return types.SimpleNamespace(run=run, reg=reg, cfg=cfg, state=state, fast=fast,
                                 calls=calls, root=tmp_path, loop=loop, native=native)


def test_budget_refuses_whole_cohort_before_any_charge(collection):
    f = collection
    f.reg['budget']['sim_s_total'] = 213405  # exactly one second short
    with pytest.raises(f.loop.Refused, match='remaining collection'):
        f.run()
    assert not f.calls
    assert f.state['sim_s_charged'] == 176473


def test_failed_first_canary_never_charges_remainder(collection):
    f = collection
    def fail(*a):
        raise RuntimeError('missing deployment asset')
    f.fast._world = fail
    with pytest.raises(RuntimeError, match='missing deployment asset'):
        f.run()
    assert f.state['sim_s_charged'] == 176473 + 2841
    rows = [json.loads(x) for x in (f.root / 'seed_ledger.jsonl').read_text().splitlines()]
    assert sum(x.get('sim_s_charged', 0) for x in rows) == 2841


def test_surviving_box_canary_then_full_cohort_inconclusive_continues(collection):
    f = collection
    result = f.run(('cpuQ',))
    assert len(result['worlds']) == 13
    assert all(box == 'cpuQ' for _, box, _ in f.calls)
    assert f.state['sim_s_charged'] == 213406
    assert result['sim_s_charged'] == 36933


def test_harmful_four_world_guard_blocks_remaining_nine(collection):
    f = collection
    f.native.smoke_guard = lambda a: {'verdict': 'harmful', 'passed': False}
    with pytest.raises(f.loop.Refused, match='harmful'):
        f.run()
    assert len(f.calls) == 4
    assert f.state['sim_s_charged'] == 176473 + 4 * 2841
    assert f.state['stopped'] == 'native smoke harmful'


def test_no_available_box_never_charges(collection):
    f = collection
    with pytest.raises(RuntimeError, match='SPOT_UNAVAILABLE'):
        f.run(())
    assert not f.calls and f.state['sim_s_charged'] == 176473


def test_completed_artifacts_retained_on_retry(collection):
    f = collection
    f.run()
    charge = f.state['sim_s_charged']
    f.calls.clear()
    f.run()
    assert not f.calls and f.state['sim_s_charged'] == charge
