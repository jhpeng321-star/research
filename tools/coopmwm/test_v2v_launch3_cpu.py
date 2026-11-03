"""Fail-closed launch3 metadata checks; no dispatcher, real /proc or physics."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from . import v2v_launch3 as launch


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    root, control, source, mover = [tmp_path / n for n in ('run', 'control', 'source', 'mover')]
    for path in (root, control, source, mover):
        path.mkdir()
    files = {}
    for name in launch.REQUIRED_SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# qualified source\n', encoding='utf-8')
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(launch, '__file__', str(source / 'tools/coopmwm/v2v_launch3.py'))
    pin = 'a' * 40
    manifest = source / 'SOURCE_PROVENANCE.json'
    manifest_hash = write_json(manifest, dict(commit=pin, complete_source_inventory=True, files=files))
    cfg = dict(launch.INTENT_OPTIONS, **launch.BANK_OPTIONS,
        budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES, train=12000000)),
        imitation_epochs=5, diagnostic_schedule=copy.deepcopy(launch.DIAGNOSTIC_SCHEDULE),
        active_arms=['A', 'B', 'R'], final_paired_branches=False,
        run_root=str(root), source_root=str(source), accounting_root=str(root),
        phase_map=dict(train='v2v_train'), diagnostic_phases=dict(lessons='v2v_lessons', isolated='v2v_isolated'),
        launch_authorized=True, phase_funding_authorized=True)
    runtime = root / 'V2V_RUNTIME.json'
    runtime_hash = write_json(runtime, dict(config=cfg))
    gate = control / 'COOPFIX_GATE.json'
    gate_hash = write_json(gate, dict(verdict='PASS', commit=pin, unix=1000,
        runtime_sha256=runtime_hash, source_manifest_sha256=manifest_hash))
    policy = dict(required_mem_available_bytes=7 * 1024**3,
        memory_per_arm_bytes=dict(A=1024**3, B=2*1024**3, R=2*1024**3))
    binding = dict(schema=launch.SCHEMA, qualified=True, pinned_commit=pin, source_root=str(source),
        gate_path=str(gate), gate_sha256=gate_hash, runtime_sha256=runtime_hash,
        gate_not_before_unix=900, gate_max_age_s=200, resource_policy=policy,
        source_provenance=dict(path=str(manifest), sha256=manifest_hash))
    path = control / 'LAUNCH_BINDING.json'
    binding_hash = write_json(path, binding)
    (root / 'CHARGES.jsonl').write_text('', encoding='utf-8')
    (root / 'ACCOUNTING.claim').touch()
    monkeypatch.setattr(launch.time, 'time', lambda: 1100)
    observer = launch.resource_gate
    # Package fixture: no Git checkout; identity tests override this boundary.
    monkeypatch.setattr(launch.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(returncode=128, stdout=b'', stderr=b'not a git repository'))
    monkeypatch.setattr(launch, 'resource_gate', lambda *args: launch.decision(
        launch.REQUIRED_FREE_BYTES, True, False, 8 * 1024**3, policy))
    # Windows has no flock; exercise real canonical charged() with a read-only
    # ledger snapshot adapter, preserving its canonical historical-charge logic.
    from . import storage, accel_storage
    def ledger(r):
        return [json.loads(line) for line in (r / 'CHARGES.jsonl').read_text().splitlines()]
    monkeypatch.setattr(storage, 'ledger', ledger)
    monkeypatch.setattr(accel_storage, 'ledger', ledger)
    return SimpleNamespace(root=root, control=control, source=source, mover=mover, cfg=cfg,
        runtime=runtime, manifest=manifest, gate=gate, binding=binding, path=path, observer=observer,
        kwargs=dict(binding_sha256=binding_hash, root=root, control=control, mover=mover))


def check(d):
    return launch.preflight(d.path, **d.kwargs)


def rebind(d):
    d.kwargs['binding_sha256'] = write_json(d.path, d.binding)


def test_ready_is_read_only_and_uses_canonical_accounting(deployed):
    d = deployed
    before = {p: p.read_bytes() for base in (d.root, d.control, d.source) for p in base.rglob('*') if p.is_file()}
    receipt = check(d)
    from .accel_storage import charged
    assert receipt['ready'] and receipt['check_only'] and receipt['new_sim_s'] == 0
    assert receipt['budget']['charged_to_date_sim_s'] == charged(d.root)
    assert receipt['budget']['intent_extra_sim_s'] == 0
    after = {p: p.read_bytes() for base in (d.root, d.control, d.source) for p in base.rglob('*') if p.is_file()}
    assert before == after
    assert not (d.root / 'V2V_STARTED.json').exists()
    assert not hasattr(launch, 'start')


def test_binding_sha_is_external_and_pin_is_explicit(deployed):
    d = deployed
    d.binding['pinned_commit'] = 'b' * 40
    write_json(d.path, d.binding)
    with pytest.raises(launch.Refused, match='SHA256 mismatch'):
        check(d)
    rebind(d)
    with pytest.raises(launch.Refused, match='provenance'):
        check(d)


def test_git_absent_never_trusts_supplied_commit_alone(deployed):
    d = deployed
    del d.binding['source_provenance']
    rebind(d)
    with pytest.raises(launch.Refused, match='git absent'):
        check(d)


@pytest.mark.parametrize('failure', ['wrong_head', 'git_missing', 'git_error', 'dirty_source'])
def test_git_must_verify_exact_actual_head_and_deployed_bytes(deployed, monkeypatch, failure):
    d = deployed
    (d.source / '.git').mkdir()
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:3] == ['git', '-C', str(d.source)]
        if failure == 'git_missing':
            raise FileNotFoundError('git')
        arg = argv[-1]
        if arg == 'HEAD':
            return SimpleNamespace(returncode=int(failure == 'git_error'),
                stdout=((('b' * 40) if failure == 'wrong_head' else 'a' * 40) + '\n').encode())
        if arg == '--show-toplevel':
            return SimpleNamespace(returncode=0, stdout=str(d.source).encode())
        raw = (d.source / arg[len('HEAD:'):]).read_bytes()
        h = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        return SimpleNamespace(returncode=0, stdout=('0'*40 if failure == 'dirty_source' else h).encode())
    monkeypatch.setattr(launch.subprocess, 'run', run)
    with pytest.raises(launch.Refused):
        check(d)
    assert calls and all('merge-base' not in argv for argv in calls)


def test_git_success_checks_actual_source_root_not_launcher_parent(deployed, monkeypatch):
    d = deployed
    (d.source / '.git').mkdir()
    del d.binding['source_provenance']
    # Git gates need no packaged-manifest field.
    rebind(d)
    def run(argv, **kwargs):
        assert argv[1:3] == ['-C', str(d.source)]
        arg = argv[-1]
        if arg == 'HEAD':
            raw = ('a' * 40).encode()
        elif arg == '--show-toplevel':
            raw = str(d.source).encode()
        else:
            data = (d.source / arg[len('HEAD:'):]).read_bytes()
            raw = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest().encode()
        return SimpleNamespace(returncode=0, stdout=raw)
    monkeypatch.setattr(launch.subprocess, 'run', run)
    assert check(d)['commit']['method'] == 'exact_git_HEAD'


def test_enclosing_checkout_cannot_be_disguised_as_packaged_source(deployed, monkeypatch):
    d = deployed
    (d.source.parent / '.git').mkdir()
    def run(argv, **kwargs):
        value = str(d.source.parent) if argv[-1] == '--show-toplevel' else 'a' * 40
        return SimpleNamespace(returncode=0, stdout=value.encode())
    monkeypatch.setattr(launch.subprocess, 'run', run)
    with pytest.raises(launch.Refused, match='actual deployed HEAD'):
        check(d)


def test_package_manifest_works_when_git_executable_is_absent(deployed, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('git')
    monkeypatch.setattr(launch.subprocess, 'run', missing)
    assert check(deployed)['commit']['method'] == 'qualified_source_manifest'


@pytest.mark.parametrize('mutation', ['bytes', 'missing', 'escape', 'prohibited'])
def test_package_requires_qualified_complete_safe_source_hashes(deployed, mutation):
    d = deployed
    manifest = json.loads(d.manifest.read_text())
    if mutation == 'bytes':
        (d.source / launch.REQUIRED_SOURCES[1]).write_text('# drift')
    else:
        if mutation == 'missing':
            manifest['files'].pop(launch.REQUIRED_SOURCES[1])
        else:
            name = '../outside.py' if mutation == 'escape' else 'sealed/never_read.py'
            manifest['files'][name] = '0' * 64
        d.binding['source_provenance']['sha256'] = write_json(d.manifest, manifest)
        rebind(d)
    with pytest.raises(launch.Refused):
        check(d)


@pytest.mark.parametrize('patch', [dict(verdict='FAIL'), dict(commit='b'*40), dict(unix=899),
    dict(unix=1101), dict(unix=100), dict(runtime_sha256='b'*64),
    dict(source_manifest_sha256='b'*64), dict(unix=float('nan'))])
def test_gate_must_be_pass_sha_bound_fresh_for_exact_commit(deployed, patch):
    d = deployed
    gate = json.loads(d.gate.read_text())
    gate.update(patch)
    d.binding['gate_sha256'] = write_json(d.gate, gate)
    rebind(d)
    with pytest.raises(launch.Refused):
        check(d)


def test_gate_replacement_without_requalification_refuses(deployed):
    d = deployed
    gate = json.loads(d.gate.read_text())
    gate['unix'] += 1
    write_json(d.gate, gate)
    with pytest.raises(launch.Refused, match='SHA256 mismatch'):
        check(d)


def test_gate_with_matching_commit_and_sha_still_expires(deployed):
    d = deployed
    d.binding['gate_max_age_s'] = 99
    rebind(d)
    with pytest.raises(launch.Refused, match='stale'):
        check(d)


def test_missing_runtime_fails_and_binding_is_mandatory(deployed):
    d = deployed
    d.runtime.unlink()
    with pytest.raises(launch.Refused, match='V2V_RUNTIME'):
        check(d)


@pytest.mark.parametrize('patch', [dict(v2v=False), dict(v2v_dim=24), dict(v2v_intent=False),
    dict(v2v_intent_horizon=4), dict(v2v_intent_loss_weight=.1), dict(v2v_intent_weight=.1),
    dict(bank_statistics='rule_v2'), dict(bank_statistics='boot_v2'), dict(bank_imitation=None), dict(imitation_epochs=4),
    dict(diagnostic_schedule=dict(A={}, B={}, R={})), dict(active_arms=['A', 'B']),
    dict(phase_map=dict(train='train'))])
def test_real_runtime_config_options_fail_closed(deployed, patch):
    cfg = copy.deepcopy(deployed.cfg)
    cfg.update(patch)
    with pytest.raises(launch.Refused):
        launch.bind_runtime(dict(config=cfg), root=deployed.root, source_root=deployed.source)


@pytest.mark.parametrize('name', ['CHARGES.jsonl', 'ACCOUNTING.claim'])
def test_authoritative_ledger_and_lock_are_mandatory(deployed, name):
    d = deployed
    (d.root / name).unlink()
    with pytest.raises(launch.Refused, match='mandatory shared authoritative'):
        check(d)


@pytest.mark.parametrize('entry', [dict(id='bad', phase='train', sim_s=-1),
    dict(id='bad', phase='train', sim_s=True), dict(id='bad', phase='unfunded', sim_s=1),
    dict(id='bad', phase='v2v_train', sim_s=1)])
def test_malformed_unfunded_and_phase_exhausted_ledger_refuse(deployed, entry):
    d = deployed
    (d.root / 'CHARGES.jsonl').write_text(json.dumps(entry) + '\n')
    with pytest.raises(launch.Refused):
        check(d)


def test_budget_total_shortfall_uses_canonical_charges(deployed):
    d = deployed
    cfg = copy.deepcopy(d.cfg)
    cfg['budget']['phases']['train'] = launch.CEILING_SIM_S - sum(launch.V2V_PHASES.values())
    from .accel_protocol import SUPERSEDED_CHARGE
    amount = cfg['budget']['phases']['train'] - SUPERSEDED_CHARGE + 1
    # Keep the training phase individually valid but exceed total by one sim-s.
    (d.root / 'CHARGES.jsonl').write_text(json.dumps(dict(id='past', phase='train', sim_s=amount)) + '\n')
    with pytest.raises(launch.Refused, match='total shortfall'):
        launch.requote(d.root, cfg)


def test_ledger_change_during_snapshot_refuses(deployed, monkeypatch):
    d = deployed
    from . import accel_storage
    original = accel_storage.charged
    def changed(root):
        amount = original(root)
        (root / 'CHARGES.jsonl').write_text(json.dumps(dict(id='concurrent', phase='train', sim_s=1)) + '\n')
        return amount
    monkeypatch.setattr(accel_storage, 'charged', changed)
    with pytest.raises(launch.Refused, match='ledger changed'):
        check(d)


def test_parent_can_check_exact_continuation_cost_without_recharging_smokes(deployed):
    d = deployed
    smoke = 3 * 24 * 1200
    (d.root / 'CHARGES.jsonl').write_text(json.dumps(dict(id='smokes', phase='v2v_train', sim_s=smoke)) + '\n')
    started = d.root / 'V2V_STARTED.json'
    started.write_text('durable supervisor owns this receipt\n')
    remaining = launch.V2V_PHASES['v2v_train'] - smoke
    receipt = launch.before_launch(d.path, **d.kwargs, phase_costs=dict(v2v_train=remaining))
    assert receipt['budget']['quote_total_sim_s'] == remaining
    assert started.read_text() == 'durable supervisor owns this receipt\n'
    with pytest.raises(launch.Refused, match='phase shortfall'):
        launch.before_launch(d.path, **d.kwargs, phase_costs=dict(v2v_train=remaining+1))


@pytest.mark.parametrize('costs', [{}, dict(train=1200), dict(v2v_train=0), dict(v2v_train=-1)])
def test_dispatch_quote_must_be_positive_explicit_and_registered(deployed, costs):
    with pytest.raises(launch.Refused):
        launch.before_launch(deployed.path, **deployed.kwargs, phase_costs=costs)


def test_memavailable_is_required_and_memory_estimates_are_configurable(deployed):
    policy = deployed.binding['resource_policy']
    assert not launch.decision(launch.REQUIRED_FREE_BYTES, True, False, 0, policy)['ready']
    ready = launch.decision(launch.REQUIRED_FREE_BYTES, True, False,
        policy['required_mem_available_bytes'], policy)
    assert ready['ready'] and ready['memory_per_arm_bytes'] == policy['memory_per_arm_bytes']
    for alive, stopped in ((False, False), (True, True)):
        assert not launch.decision(launch.REQUIRED_FREE_BYTES, alive, stopped, 99*1024**3, policy)['ready']
    with pytest.raises(launch.Refused):
        launch.decision(launch.REQUIRED_FREE_BYTES, True, False, 99*1024**3, {})


def test_missing_linux_memavailable_cannot_fall_back_to_memfree(deployed, monkeypatch):
    monkeypatch.setattr(launch, '_bytes', lambda path: b'MemFree: 99999999 kB\n')
    with pytest.raises(launch.Refused, match='MemAvailable'):
        deployed.observer(deployed.root, deployed.mover, deployed.binding['resource_policy'])


def test_linux_observer_uses_memavailable_and_reports_per_arm_estimates(deployed, monkeypatch):
    d = deployed
    monkeypatch.setattr(launch, '_bytes', lambda path: b'MemFree: 0 kB\nMemAvailable: 8388608 kB\n')
    monkeypatch.setattr(launch, 'mover_alive', lambda path: True)
    monkeypatch.setattr(launch.shutil, 'disk_usage', lambda path: SimpleNamespace(free=launch.REQUIRED_FREE_BYTES))
    result = d.observer(d.root, d.mover, d.binding['resource_policy'])
    assert result['ready'] and result['mem_available_bytes'] == 8*1024**3
    assert result['memory_per_arm_bytes'] == d.binding['resource_policy']['memory_per_arm_bytes']


@pytest.mark.parametrize('base', ['root', 'control'])
def test_hold_present_at_start_refuses_without_reading_content(deployed, base):
    (getattr(deployed, base) / launch.HOLD_NAME).mkdir()
    with pytest.raises(launch.Refused, match='HOLD'):
        check(deployed)


def test_hold_appearing_during_resource_observation_refuses(deployed, monkeypatch):
    d = deployed
    observer = launch.resource_gate
    def resource(*args):
        value = observer(*args)
        (d.control / launch.HOLD_NAME).touch()
        return value
    monkeypatch.setattr(launch, 'resource_gate', resource)
    with pytest.raises(launch.Refused, match='HOLD'):
        check(d)


def test_hold_rechecked_immediately_after_wait(deployed, monkeypatch):
    d = deployed
    monkeypatch.setattr(launch, 'resource_gate', lambda *args: dict(ready=False))
    slept = []
    def sleep(seconds):
        slept.append(seconds)
        (d.control / launch.HOLD_NAME).touch()
    with pytest.raises(launch.Refused, match='HOLD'):
        launch.wait_for_preflight(d.path, **d.kwargs, wait_seconds=3, sleep=sleep)
    assert slept == [3]


def test_wait_rechecks_full_preflight_then_parent_checks_again_before_launch(deployed, monkeypatch):
    d = deployed
    resource = launch.resource_gate
    polls = []
    def observation(*args):
        polls.append(1)
        return dict(ready=False) if len(polls) == 1 else resource(*args)
    monkeypatch.setattr(launch, 'resource_gate', observation)
    assert launch.wait_for_preflight(d.path, **d.kwargs, wait_seconds=1, sleep=lambda s: None)['ready']
    (d.control / launch.HOLD_NAME).touch()
    with pytest.raises(launch.Refused, match='HOLD'):
        launch.before_launch(d.path, **d.kwargs)
    assert len(polls) == 2


def test_hold_appearing_during_budget_check_is_caught_at_final_boundary(deployed, monkeypatch):
    d = deployed
    quote = launch.requote
    def with_hold(*args, **kwargs):
        value = quote(*args, **kwargs)
        (d.root / launch.HOLD_NAME).touch()
        return value
    monkeypatch.setattr(launch, 'requote', with_hold)
    with pytest.raises(launch.Refused, match='HOLD'):
        launch.before_launch(d.path, **d.kwargs)


def test_wait_never_retries_nonresource_refusal(deployed):
    d = deployed
    d.runtime.unlink()
    def unexpected_sleep(seconds):
        pytest.fail('nonresource failure must not wait')
    with pytest.raises(launch.Refused):
        launch.wait_for_preflight(d.path, **d.kwargs, wait_seconds=100, sleep=unexpected_sleep)
