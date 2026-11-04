"""Read-only launch3 checks for integration by the durable V2V supervisor.

Nothing here dispatches, reserves physics, or writes V2V_STARTED.json. A successful
check is a snapshot, not a launch receipt. The supervisor must call before_launch
immediately before EACH dispatch (including after a resource wait), retain its
existing release/admission checks and shared-ledger reservation, and write started
only after it owns actual durable training children.

The parent qualifies an additive JSON binding, then supplies its SHA256 separately
to preflight. Required binding keys are schema, qualified, pinned_commit,
source_root, gate_path, gate_sha256, runtime_sha256, gate_not_before_unix,
gate_max_age_s, and resource_policy. No code embeds its own future commit hash.
resource_policy requires required_mem_available_bytes and memory_per_arm_bytes
(A/B/R); required_free_bytes can increase the inherited 6 GiB disk minimum.

Packaged sources require source_provenance={path, sha256}. That externally
qualified manifest declares commit, complete_source_inventory=True and files
(relative deployed code paths -> SHA256). Its qualification must cover the entire
executable inventory, including inherited dependencies; a claimed commit alone
is never provenance. Gate metadata is PASS, commit, unix, runtime_sha256 and,
for packages, source_manifest_sha256. Freshness limits are explicit parent policy.
Only explicit safe source paths are hashed; no outcome/bank/checkpoint is opened.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time

from .protocol import checked

SCHEMA = 'coopmwm-v2v-launch3-v1'
CEILING_SIM_S = 17471840
REQUIRED_FREE_BYTES = 6 * 1024 ** 3
HOLD_NAME = 'V2V_LAUNCH3_HOLD.json'
ARMS = ('A', 'B', 'R')
INTENT_OPTIONS = dict(v2v=True, v2v_dim=48, v2v_intent=True,
    v2v_intent_horizon=5, v2v_intent_loss_weight=1.0,v2v_intent_detach_trunk=True)
BANK_OPTIONS = dict(bank_imitation='separate_full_pass',
    bank_statistics='per_seed_v1',   # owner 2026-11-03 21:45 + audit: per_seed_v1 only (B and R)
    # owner 2026-11-04 00:55: B and R imitate ONLY within-seed cooperative examples (A has no bank)
    imitation_bank_filter='cooperative_only',
    # c4/imit-strength: 'fixed' stays byte-identical; 'kl_targeted' only at a registered update
    # boundary, for B-v2v and R-v2v together. OWNER 2026-11-04 08:52 ("yes, switch on stronger
    # imitation now"): the pinned value is 'kl_targeted' (KL limit .045 and every check unchanged).
    # The bank intent term switch defaults True = current behaviour.
    imitation_step_mode='kl_targeted', v2v_intent_bank_term=True,
    # c4/imit-kl-option: the runtime's separate-imitation KL limit. Default .045 = every earlier
    # receipt (byte-identical); prepare_intent may pin a REGISTERED alternative (.06/.09) for a
    # runtime, for B and R together; bind_runtime refuses any unregistered value or B/R mismatch.
    # (fix round: the stale 'imitation_kl_cap' label is gone; bind_runtime refuses any runtime
    # that still carries it with a value different from the runtime's own imitation_kl_limit.)
    imitation_kl_limit=.045)
IMITATION_BANK_FILTERS = ('all_admitted', 'cooperative_only')
IMITATION_STEP_MODES = ('fixed', 'kl_targeted')
IMITATION_KL_LIMITS = (.045, .06, .09)   # c4/imit-kl-option: registered imitation_kl_limit values
DIAGNOSTIC_SCHEDULE = dict(A=dict(lessons=[], isolated=[]),
    B=dict(lessons=[6, 12, 18, 24], isolated=[6, 12, 18, 24]),
    R=dict(lessons=[], isolated=[12, 24]))
V2V_PHASES = dict(v2v_train=2073600, v2v_interim=345600, v2v_final=345600,
    v2v_lessons=21840, v2v_isolated=185200)
# Minimum executable coverage, not a replacement for qualification's full inventory.
REQUIRED_SOURCES = tuple('tools/coopmwm/' + name + '.py' for name in (
    'v2v_launch3', 'v2v_live', 'v2v_runner', 'v2v_runtime', 'v2v_model',
    'v2v_obs', 'v2v_integrate', 'bank_v2v', 'bank_imitation', 'bank_statistics',
    'accel_storage', 'accel_protocol', 'storage'))


class Refused(ValueError):
    """A missing, stale, inconsistent or unverifiable prerequisite."""


class ResourceNotReady(Refused):
    """The only refusal that wait_for_preflight may retry."""


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(path):
    try:
        return checked(path).read_bytes()
    except (OSError, ValueError) as error:
        raise Refused('cannot read required file ' + str(path)) from error


def _json(path, expected_sha256=None):
    raw = _bytes(path)
    if expected_sha256 is not None:
        _hex(expected_sha256, 64, 'SHA256')
        if _hash(raw) != expected_sha256:
            raise Refused('SHA256 mismatch: ' + str(path))
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise Refused('invalid JSON: ' + str(path)) from error
    if not isinstance(value, dict):
        raise Refused('JSON object required: ' + str(path))
    return value, _hash(raw)


def _hex(value, length, label):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{' + str(length) + '}', value):
        raise Refused('full lowercase ' + label + ' required')
    return value


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise Refused('positive finite ' + label + ' required')
    return value


def _amount(value, label):
    if type(value) is not int or value < 0:
        raise Refused('nonnegative integer ' + label + ' required')
    return value


def _absolute(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise Refused('absolute ' + label + ' required')
    try:
        return checked(value).resolve()
    except (OSError, ValueError) as error:
        raise Refused('unsafe ' + label) from error


def read_binding(path, sha256):
    """Trust anchor SHA must come from the parent's qualified additive record."""
    _hex(sha256, 64, 'binding SHA256')
    value, _ = _json(path, sha256)
    if value.get('schema') != SCHEMA or value.get('qualified') is not True:
        raise Refused('qualified additive launch3 binding required')
    _hex(value.get('pinned_commit'), 40, 'pinned commit')
    for key in ('source_root', 'gate_path'):
        _absolute(value.get(key), key)
    for key in ('gate_sha256', 'runtime_sha256'):
        _hex(value.get(key), 64, key)
    _positive(value.get('gate_not_before_unix'), 'gate_not_before_unix')
    _positive(value.get('gate_max_age_s'), 'gate_max_age_s')
    _resource_policy(value.get('resource_policy'))
    return value


def _git(source, *args):
    try:
        return subprocess.run(['git', '-C', str(source), *args],
            capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as error:
        raise Refused('deployed git identity unavailable; explicit qualified provenance required') from error


def _source_path(source, name):
    if not isinstance(name, str):
        raise Refused('executable source path must be a string')
    rel = PurePosixPath(name)
    if (rel.is_absolute() or '..' in rel.parts
            or '\\' in name or ':' in name or rel.suffix not in ('.py', '.so', '.pyd')):
        raise Refused('unsafe executable source path: ' + str(name))
    try:
        target = checked(source / name)
    except (OSError, ValueError) as error:
        raise Refused('unsafe executable source path: ' + name) from error
    if not target.resolve().is_relative_to(source.resolve()):
        raise Refused('source path escapes deployed root: ' + name)
    return target


def verify_commit(binding):
    """Exact deployed HEAD, or externally SHA-qualified complete package hashes.

    A present Git checkout always takes precedence: provenance cannot bypass a
    wrong/unreadable HEAD. The loaded checker must belong to the deployed source.
    """
    source = _absolute(binding.get('source_root'), 'source_root')
    pin = _hex(binding.get('pinned_commit'), 40, 'pinned commit')
    loaded = Path(__file__).resolve()
    try:
        checker = loaded.relative_to(source).as_posix()
    except ValueError as error:
        raise Refused('checker is not loaded from the deployed source_root') from error
    required = set(REQUIRED_SOURCES)
    # Packaged releases usually have a pytools/ prefix; all tool paths share it.
    suffix = 'tools/coopmwm/v2v_launch3.py'
    if not checker.endswith(suffix):
        raise Refused('unexpected deployed checker path')
    prefix = checker[:-len(suffix)]
    required = {prefix + name for name in required}
    # Query from the deployment, even when .git is in a parent directory. A
    # broken marker or unavailable Git in a checkout must never enable fallback.
    markers = [base / '.git' for base in (source, *source.parents)]
    has_git = any(path.exists() or path.is_symlink() for path in markers)
    try:
        top = _git(source, 'rev-parse', '--show-toplevel')
    except Refused:
        if has_git:
            raise
        top = None
    if top is not None and top.returncode == 0:
        head = _git(source, 'rev-parse', '--verify', 'HEAD')
        if head.returncode:
            raise Refused('deployed HEAD cannot be verified')
        actual = head.stdout.decode().strip()
        if Path(top.stdout.decode().strip()).resolve() != source or actual != pin:
            raise Refused('actual deployed HEAD differs from pinned commit: ' + actual)
        # HEAD alone must not qualify modified executable bytes in the checkout.
        for name in sorted(required):
            blob = _git(source, 'rev-parse', 'HEAD:' + name)
            if blob.returncode:
                raise Refused('required source absent from pinned commit: ' + name)
            raw = _bytes(_source_path(source, name))
            actual_blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            if actual_blob != blob.stdout.decode().strip():
                raise Refused('deployed source differs from pinned commit: ' + name)
        return dict(commit=actual, method='exact_git_HEAD', checked_sources=sorted(required))
    if has_git:
        raise Refused('deployed checkout identity unavailable; package provenance cannot bypass Git')
    provenance = binding.get('source_provenance')
    if not isinstance(provenance, dict):
        raise Refused('git absent: explicit SHA-verified source provenance required')
    manifest, h = _json(_absolute(provenance.get('path'), 'provenance path'),
        _hex(provenance.get('sha256'), 64, 'provenance SHA256'))
    files = manifest.get('files')
    if (manifest.get('commit') != pin or manifest.get('complete_source_inventory') is not True
            or not isinstance(files, dict) or not required <= set(files)):
        raise Refused('complete qualified source provenance for pinned commit required')
    for name, expected in files.items():
        _hex(expected, 64, 'source SHA256')
        if _hash(_bytes(_source_path(source, name))) != expected:
            raise Refused('packaged source SHA256 mismatch: ' + name)
    return dict(commit=pin, method='qualified_source_manifest', manifest_sha256=h,
        checked_sources=sorted(files))


def read_gate(binding, deployed, *, now=None):
    gate, h = _json(binding['gate_path'], binding['gate_sha256'])
    if gate.get('verdict') != 'PASS' or gate.get('commit') != deployed['commit']:
        raise Refused('gate must PASS for the exact deployed commit')
    if gate.get('runtime_sha256') != binding['runtime_sha256']:
        raise Refused('gate runtime SHA256 differs')
    if (deployed.get('manifest_sha256') is not None
            and gate.get('source_manifest_sha256') != deployed['manifest_sha256']):
        raise Refused('gate source provenance SHA256 differs')
    stamp = _positive(gate.get('unix'), 'gate unix')
    now = time.time() if now is None else now
    if (stamp < binding['gate_not_before_unix'] or stamp > now
            or now - stamp > binding['gate_max_age_s']):
        raise Refused('gate is stale, predates qualification, or is future-dated')
    return dict(verdict='PASS', commit=deployed['commit'], sha256=h, unix=stamp)


def check_hold(control, root):
    for base in (Path(control), Path(root)):
        path = checked(base / HOLD_NAME)
        # Broken symlinks also represent a HOLD; never open its contents.
        if path.exists() or path.is_symlink():
            raise Refused('HOLD file present: ' + str(path))
    return dict(held=False)


def imitation_filter_for(cfg, arm):
    """The imitation bank filter of arm B or R. One key serves both arms; an optional per-arm
    mapping cfg['imitation_bank_filter_by_arm'] is accepted only when B == R == the key."""
    have = cfg.get('imitation_bank_filter', 'all_admitted')
    if have not in IMITATION_BANK_FILTERS:
        raise Refused('unknown imitation_bank_filter ' + repr(have))
    by_arm = cfg.get('imitation_bank_filter_by_arm')
    if by_arm is not None:
        if not isinstance(by_arm, dict) or by_arm.get('B') != by_arm.get('R') or by_arm.get('B') != have:
            raise Refused('imitation_bank_filter must be identical for B and R: ' + repr(by_arm))
    if arm not in ('B', 'R'):
        raise Refused('only arms B and R have an imitation bank')
    return have


def imitation_step_mode_for(cfg, arm):
    """The imitation step mode of arm B or R (c4/imit-strength; 'kl_targeted' only ever at a
    registered update boundary, for B and R together). One key serves both arms; an optional
    per-arm mapping cfg['imitation_step_mode_by_arm'] is accepted only when B == R == the key."""
    have = cfg.get('imitation_step_mode', 'fixed')
    if have not in IMITATION_STEP_MODES:
        raise Refused('unknown imitation_step_mode ' + repr(have))
    by_arm = cfg.get('imitation_step_mode_by_arm')
    if by_arm is not None:
        if not isinstance(by_arm, dict) or by_arm.get('B') != by_arm.get('R') or by_arm.get('B') != have:
            raise Refused('imitation_step_mode must be identical for B and R: ' + repr(by_arm))
    if arm not in ('B', 'R'):
        raise Refused('only arms B and R have an imitation bank')
    return have


def imitation_kl_limit_for(cfg, arm):
    """The separate-imitation KL limit of arm B or R (c4/imit-kl-option). The runtime pins ONE
    registered value (default .045 = every earlier receipt; a pre-option runtime without the key
    keeps the .045 default, byte-identical). One key serves both arms; an optional per-arm mapping
    cfg['imitation_kl_limit_by_arm'] is accepted only when B == R == the key."""
    have = cfg.get('imitation_kl_limit', .045)
    if have not in IMITATION_KL_LIMITS:
        raise Refused('unknown imitation_kl_limit ' + repr(have))
    by_arm = cfg.get('imitation_kl_limit_by_arm')
    if by_arm is not None:
        if not isinstance(by_arm, dict) or by_arm.get('B') != by_arm.get('R') or by_arm.get('B') != have:
            raise Refused('imitation_kl_limit must be identical for B and R: ' + repr(by_arm))
    if arm not in ('B', 'R'):
        raise Refused('only arms B and R have an imitation bank')
    return have


def intent_bank_term_for(cfg, arm):
    """Whether the bank intent term trains arm B or R (default True = current behaviour). One key
    serves both arms; an optional per-arm mapping cfg['v2v_intent_bank_term_by_arm'] is accepted
    only when B == R == the key."""
    have = cfg.get('v2v_intent_bank_term', True)
    if not isinstance(have, bool):
        raise Refused('v2v_intent_bank_term must be a bool: ' + repr(have))
    by_arm = cfg.get('v2v_intent_bank_term_by_arm')
    if by_arm is not None:
        if (not isinstance(by_arm, dict) or not all(isinstance(by_arm.get(a), bool) for a in ('B', 'R'))
                or by_arm.get('B') is not by_arm.get('R') or by_arm.get('B') is not have):
            raise Refused('v2v_intent_bank_term must be identical for B and R: ' + repr(by_arm))
    if arm not in ('B', 'R'):
        raise Refused('only arms B and R have an imitation bank')
    return have


def bind_runtime(value, *, root, source_root):
    """Validate deployed values, not newly manufactured desired arm configs."""
    cfg = value.get('config')
    if not isinstance(cfg, dict):
        raise Refused('V2V_RUNTIME config object required')
    for key, want in dict(INTENT_OPTIONS, **BANK_OPTIONS).items():
        have = cfg.get(key)
        if key == 'imitation_kl_limit':
            # c4/imit-kl-option: the runtime may pin any REGISTERED alternative (chosen by
            # prepare_intent); everything else refuses. A pre-option runtime without the key
            # keeps the registered default .045 (byte-identical). B == R is enforced below.
            if 'imitation_kl_limit' not in cfg:
                have = want
            if have not in IMITATION_KL_LIMITS:
                raise Refused('runtime option imitation_kl_limit differs: ' + repr(have))
            continue
        if have != want or isinstance(have, bool) != isinstance(want, bool):
            raise Refused('runtime option ' + key + ' differs: ' + repr(have))
    # fix round (c4/imit-kl-option): the stale imitation_kl_cap label left BANK_OPTIONS; a
    # runtime that still carries it must agree with its own imitation_kl_limit (the default
    # .045 when that key is absent). A stale .045 claim next to a larger registered limit
    # refuses, so the pinned cap can never disagree with the runtime's real trust region.
    if 'imitation_kl_cap' in cfg:
        limit = cfg['imitation_kl_limit'] if 'imitation_kl_limit' in cfg else BANK_OPTIONS['imitation_kl_limit']
        cap = cfg['imitation_kl_cap']
        if cap != limit or isinstance(cap, bool) != isinstance(limit, bool):
            raise Refused('stale runtime imitation_kl_cap label differs from imitation_kl_limit: '
                          + repr(cap) + ' vs ' + repr(limit))
    imitation_filter_for(cfg, 'B'), imitation_filter_for(cfg, 'R')   # refuses B != R
    imitation_step_mode_for(cfg, 'B'), imitation_step_mode_for(cfg, 'R')   # refuses B != R
    imitation_kl_limit_for(cfg, 'B'), imitation_kl_limit_for(cfg, 'R')   # refuses B != R
    intent_bank_term_for(cfg, 'B'), intent_bank_term_for(cfg, 'R')   # refuses B != R
    if 'v2v_intent_weight' in cfg:
        raise Refused('obsolete runtime v2v_intent_weight is not an intent loss binding')
    passes = cfg.get('imitation_epochs')
    if type(passes) is not int or passes < 5:
        raise Refused('full-pass separate imitation requires at least five requested epochs')
    if cfg.get('diagnostic_schedule') != DIAGNOSTIC_SCHEDULE:
        raise Refused('runtime diagnostic_schedule differs from registered plan')
    if cfg.get('active_arms') != list(ARMS) or cfg.get('final_paired_branches') is not False:
        raise Refused('all three runtime arms and no final paired branches required')
    for key, want in (('run_root', root), ('source_root', source_root)):
        if _absolute(cfg.get(key), key) != Path(want).resolve():
            raise Refused('runtime ' + key + ' differs from deployed binding')
    if cfg.get('phase_map', {}).get('train') != 'v2v_train' or cfg.get('diagnostic_phases') != dict(
            lessons='v2v_lessons', isolated='v2v_isolated'):
        raise Refused('runtime phase mapping differs')
    if not cfg.get('launch_authorized') or not cfg.get('phase_funding_authorized'):
        raise Refused('runtime launch and phase funding authority required')
    budget = cfg.get('budget', {})
    phases = budget.get('phases')
    if budget.get('total') != CEILING_SIM_S or not isinstance(phases, dict):
        raise Refused('runtime registered budget total/phases required')
    for phase, cap in phases.items():
        _amount(cap, 'phase allocation ' + phase)
    if sum(phases.values()) > CEILING_SIM_S:
        raise Refused('runtime phase allocations exceed total cap')
    for phase, cost in V2V_PHASES.items():
        if phases.get(phase, -1) < cost:
            raise Refused('runtime phase allocation missing/insufficient: ' + phase)
    return cfg


def _resource_policy(policy):
    if not isinstance(policy, dict):
        raise Refused('explicit MAIN20:28 resource_policy required')
    mem = _amount(policy.get('required_mem_available_bytes'), 'required MemAvailable bytes')
    arms = policy.get('memory_per_arm_bytes')
    if mem <= 0 or not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise Refused('positive MemAvailable threshold and explicit A/B/R memory estimates required')
    for arm, amount in arms.items():
        if _amount(amount, 'memory per arm ' + arm) <= 0:
            raise Refused('positive memory per arm required: ' + arm)
    disk = _amount(policy.get('required_free_bytes', REQUIRED_FREE_BYTES), 'disk threshold')
    if disk < REQUIRED_FREE_BYTES:
        raise Refused('disk threshold below inherited 6 GiB minimum')
    return dict(required_mem_available_bytes=mem, memory_per_arm_bytes=dict(arms),
        required_free_bytes=disk)


def decision(free, alive, stopped, mem_available, policy):
    policy = _resource_policy(policy)
    _amount(free, 'disk free bytes')
    _amount(mem_available, 'MemAvailable bytes')
    return dict(policy, ready=(free >= policy['required_free_bytes'] and alive is True
        and stopped is False and mem_available >= policy['required_mem_available_bytes']),
        disk_free_bytes=free, mem_available_bytes=mem_available,
        mover_alive=alive, mover_stopped=stopped,
        rule='MAIN16:47 disk/live mover plus MAIN20:28 MemAvailable; per-arm estimates are qualified policy')


def mover_alive(mover):
    try:
        record, _ = _json(Path(mover) / 'OWNER.json')
        pid = record['pid']
        if type(pid) is not int or pid <= 0:
            return False
        # comm can contain spaces/parentheses: split only after its closing ')'.
        fields = _bytes(Path('/proc') / str(pid) / 'stat').decode().rsplit(')', 1)[1].split()
        return (fields[0] not in ('Z', 'X', 'T', 't') and fields[19] == record['start_ticks']
            and _bytes('/proc/sys/kernel/random/boot_id').decode().strip() == record['boot_id'])
    except (Refused, KeyError, IndexError, UnicodeError):
        return False


def resource_gate(root, mover, policy):
    try:
        rows = _bytes('/proc/meminfo').decode().splitlines()
        available = [line.split() for line in rows if line.startswith('MemAvailable:')]
        if len(available) != 1 or len(available[0]) != 3 or available[0][2] != 'kB':
            raise Refused('Linux MemAvailable in kB is mandatory')
        mem = int(available[0][1]) * 1024
        stop = checked(Path(mover) / 'STOP_GPU_SPACE.json')
        return decision(shutil.disk_usage(checked(root)).free, mover_alive(mover),
            stop.exists() or stop.is_symlink(), mem, policy)
    except (OSError, ValueError, UnicodeError) as error:
        raise Refused('resource observation unavailable: ' + str(error)) from error


def requote(root, cfg, *, phase_costs=None):
    """Canonical accounting, read-only; reservations remain the runner's duty.

    The registered frozen-slate quote includes zero extra intent sim-s. Future
    bank growth still needs the existing per-diagnostic whole-slate prequote.
    For a later child/wave the parent supplies its exact positive phase_costs;
    charging the entire initial run again after smokes would block continuation.
    """
    from . import accel_storage, storage
    costs = dict(V2V_PHASES) if phase_costs is None else phase_costs
    if not isinstance(costs, dict) or not costs or not set(costs) <= set(V2V_PHASES):
        raise Refused('explicit nonempty registered phase_costs required')
    for phase, amount in costs.items():
        if _amount(amount, 'planned sim_s ' + phase) <= 0:
            raise Refused('positive planned phase cost required: ' + phase)
    root = checked(root)
    accounting = _absolute(cfg.get('accounting_root'), 'accounting_root')
    for name in ('CHARGES.jsonl', 'ACCOUNTING.claim'):
        here, there = checked(root / name), checked(accounting / name)
        if not here.is_file() or not there.is_file() or here.resolve() != there.resolve():
            raise Refused('mandatory shared authoritative ledger AND lock: ' + name)
    before = _hash(_bytes(root / 'CHARGES.jsonl'))
    try:
        entries = storage.ledger(root)
        charged = accel_storage.charged(root)
    except (OSError, ValueError, KeyError, ImportError) as error:
        raise Refused('canonical budget/ledger unavailable or malformed') from error
    if before != _hash(_bytes(root / 'CHARGES.jsonl')):
        raise Refused('ledger changed during preflight; recheck required')
    _amount(charged, 'canonical charged sim_s')
    seen, spent = set(), {}
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get('id'), str)
                or not entry['id'] or entry['id'] in seen or not isinstance(entry.get('phase'), str)):
            raise Refused('malformed/duplicate ledger charge')
        seen.add(entry['id'])
        amount = _amount(entry.get('sim_s'), 'ledger sim_s')
        spent[entry['phase']] = spent.get(entry['phase'], 0) + amount
    phases = cfg['budget']['phases']
    for phase, amount in spent.items():
        if phase not in phases or amount > phases[phase]:
            raise Refused('ledger phase unallocated/overdrawn: ' + phase)
    for phase, cost in costs.items():
        if spent.get(phase, 0) + cost > phases[phase]:
            raise Refused('budget phase shortfall: ' + phase)
    quote = sum(costs.values())
    if charged + quote > cfg['budget']['total']:
        raise Refused('budget total shortfall_sim_s=' + str(charged + quote - cfg['budget']['total']))
    return dict(charged_to_date_sim_s=charged, ceiling_sim_s=cfg['budget']['total'],
        quote_total_sim_s=quote, phase_costs=dict(costs), ledger_sha256=before,
        intent_extra_sim_s=0, fits=True, reservation_sim_s=0,
        is_full_run_upper_bound=False, ledger_rule='accel_storage.charged + storage.ledger; no fallback')


def preflight(binding_path, *, binding_sha256, root, control, mover, phase_costs=None):
    """Side-effect-free check snapshot; mandatory runtime, identity, gate and budget."""
    root = checked(root)
    check_hold(control, root)
    binding = read_binding(binding_path, binding_sha256)
    deployed = verify_commit(binding)
    runtime, runtime_hash = _json(root / 'V2V_RUNTIME.json', binding['runtime_sha256'])
    cfg = bind_runtime(runtime, root=root, source_root=binding['source_root'])
    gate = read_gate(binding, deployed)
    resources = resource_gate(root, mover, binding['resource_policy'])
    check_hold(control, root)  # required AFTER a resource observation/wait
    if resources.get('ready') is not True:
        raise ResourceNotReady('resource gate not ready: ' + json.dumps(resources, sort_keys=True))
    budget = requote(root, cfg, phase_costs=phase_costs)
    # Revalidate inputs potentially changed while observing resources/accounting.
    if verify_commit(binding) != deployed:
        raise Refused('deployed identity changed during preflight')
    read_gate(binding, deployed)
    _json(root / 'V2V_RUNTIME.json', runtime_hash)
    read_binding(binding_path, binding_sha256)
    check_hold(control, root)  # final operation before returning dispatch check
    return dict(ready=True, check_only=True, new_sim_s=0, unix=time.time(),
        binding_sha256=binding_sha256, commit=deployed, gate=gate,
        runtime_sha256=runtime_hash, resource=resources, budget=budget)


def wait_for_preflight(binding_path, *, binding_sha256, root, control, mover,
                       phase_costs=None, wait_seconds=0, poll_seconds=120, sleep=time.sleep):
    """Retry resources only, rechecking HOLD and all qualifications on every pass."""
    _amount(wait_seconds, 'wait_seconds')
    _positive(poll_seconds, 'poll_seconds')
    while True:
        try:
            return preflight(binding_path, binding_sha256=binding_sha256,
                root=root, control=control, mover=mover, phase_costs=phase_costs)
        except ResourceNotReady:
            check_hold(control, root)
            if wait_seconds <= 0:
                raise
            interval = min(poll_seconds, wait_seconds)
            sleep(interval)
            wait_seconds -= interval
            check_hold(control, root)


def before_launch(binding_path, **kwargs):
    """Parent calls synchronously immediately before every actual child dispatch.

    Never reuse an earlier ready receipt. This API deliberately cannot launch or
    claim started; the durable supervisor owns dispatch, ledger locking and races
    between the final HOLD check and process creation. For continuation/test
    children pass their exact phase_costs, rather than the default full-run quote.
    Adoption/duplicate-child checks and ownership of V2V_STARTED stay with the
    supervisor; checking an existing run must not claim it has been relaunched.
    """
    return preflight(binding_path, **kwargs)
