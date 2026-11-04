"""PREPARE ONLY (owner 2026-11-04, c4/rerun-cpus): NEW standalone A/B/R-v2v experiment on cpuS
with ITS OWN root and ITS OWN single ledger. Supersedes the c4/rerun_clean prep (which kept the
coop3 A as a remote control): here ALL THREE arms A, B and R run from MAPPO u0 with identical
code in ONE new root - B and R with the full fix set (V2V intent, per_seed_v1 cooperative_only
bank, imitation_step_mode kl_targeted, practice_starts 0.25/cap 2, imitation_target_space
'effective'), A the same runtime minus bank/practice (as in coop3) - so the B/R safety
composites read A's results from the SAME new root and no remote A read ever happens.

Nothing here dispatches, reserves, charges or runs SUMO. The new root owns CHARGES.jsonl +
ACCOUNTING.claim (lock) + AUTHORITY.md; it is created only if absent (refused otherwise), and
every charge is refused unless it lands in THIS root's own ledger (never a symlink or another
root's ledger). A read-only staging manifest lists every bank/tape/snapshot/MAPPO-checkpoint
file the run reads - resolved through the REAL bank readers (bank_decisions, the per_seed_v1
confirmations, temperature, practice links) - relative to /home/ubuntu so an operator can
rsync+sha them to cpuS. The dry preflight refuses unless OWNER_APPROVAL.json binds the budget
request SHA, the owner quote SHA and the verdict file SHA, and every manifest file SHA matches
the staged copies. Effective-imitation is pinned to f35b193a.
"""
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

UPDATES = 24
EPISODE_SIM_S = 1200
EPISODES_PER_UPDATE = 24
PRACTICE_REFS_PER_UPDATE = 6          # one matched A-policy reference per practice episode (.25 of 24)
GATE_SIM_S = 35820                    # cooperative_only exact gate (7 states / 55 branches), measured x5
LESSON_SIM_S_MEASURED = 5280          # B u6 V2V lesson, measured
ISOLATED_SIM_S_MEASURED = 32610       # isolated synergy, measured
TEST_EPISODES = 96                    # u8 interim + u24 final fixed episode tests, ALL THREE arms
SCHEDULE = dict(A=dict(lessons=[], isolated=[]),
                B=dict(lessons=[6, 12, 18, 24], isolated=[6, 12, 18, 24]),
                R=dict(lessons=[], isolated=[12, 24]))
PINS = dict(v2v=True, v2v_dim=48, v2v_intent=True, v2v_intent_horizon=5, v2v_intent_loss_weight=1.0,
            v2v_intent_detach_trunk=True, bank_imitation='separate_full_pass', imitation_kl_cap=.045,
            bank_statistics='per_seed_v1', imitation_bank_filter='cooperative_only',
            imitation_step_mode='kl_targeted', v2v_intent_bank_term=True, practice_starts=True,
            practice_fraction=.25, practice_repeat_cap=2, imitation_target_space='effective',
            stop_after_update=UPDATES, A_stop_after_update=UPDATES)
EFFECTIVE_IMITATION_COMMIT = 'f35b193aa539c7ddec1bb20acaa1f6b13ffb3d31'
HOME = '/home/ubuntu'                 # every staged path is listed relative to this prefix
CHECKPOINT_000_SOURCE = 'B_checkpoint_000.pt'      # the common MAPPO u0 in the original run
ISOLATED_REFERENCE_CHECKPOINTS = ('B_checkpoint_003.pt', 'R_checkpoint_003.pt')  # fresh legacy-u3 refs
RULE_FILE = 'SAFETY_RULE_0957.json'   # composite MAIN09:57 rule, copied into the new root
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')
_TRUTH = re.compile(r'(vtype|driver|vehicle_params)[^/]*truth|truth[^/]*(vtype|driver|vehicle_params)',
                    re.IGNORECASE)


def quote():
    """Full-run sim-s quote for ALL THREE arms (24 updates each) including the exact gate,
    with the u8/u24 test numbers written out. A trains plain PPO (no bank, no practice); B and R
    each add 6 matched A-policy practice reference episodes per update."""
    a_train = UPDATES*EPISODES_PER_UPDATE*EPISODE_SIM_S
    br_train = 2*UPDATES*(EPISODES_PER_UPDATE+PRACTICE_REFS_PER_UPDATE)*EPISODE_SIM_S
    lessons = len(SCHEDULE['B']['lessons'])*LESSON_SIM_S_MEASURED
    isolated = (len(SCHEDULE['B']['isolated'])+len(SCHEDULE['R']['isolated']))*ISOLATED_SIM_S_MEASURED
    test_per_arm = TEST_EPISODES*EPISODE_SIM_S
    out = dict(a_train_sim_s=a_train, b_train_sim_s=br_train//2, r_train_sim_s=br_train//2,
               v2v_train=a_train+br_train, v2v_lessons=lessons, v2v_isolated=isolated,
               gate_bank_verification_ext=GATE_SIM_S,
               interim_u8_sim_s=3*test_per_arm, final_u24_sim_s=3*test_per_arm,
               interim_u8=dict(per_arm=test_per_arm, arms=3, updates=8, episodes_per_arm=TEST_EPISODES),
               final_u24=dict(per_arm=test_per_arm, arms=3, updates=24, episodes_per_arm=TEST_EPISODES))
    out['total_without_tests'] = a_train+br_train+lessons+isolated+GATE_SIM_S
    out['total'] = out['total_without_tests']+out['interim_u8_sim_s']+out['final_u24_sim_s']
    return out


def budget():
    """The standalone experiment's own registered budget (one ledger, one total): the phase
    allocations cover the quote with the registered v2v_lessons floor (21,840) kept."""
    q = quote()
    phases = dict(v2v_train=q['v2v_train'], v2v_lessons=21840, v2v_isolated=q['v2v_isolated'],
                  gate_bank_verification_ext=GATE_SIM_S, v2v_interim=q['interim_u8_sim_s'],
                  v2v_final=q['final_u24_sim_s'])
    total = sum(phases.values())
    return dict(total=total, phases=phases, allocated=total, unallocated=0, updates=UPDATES,
                episodes_per_update=EPISODES_PER_UPDATE, arms=['A', 'B', 'R'],
                quote_total_sim_s=q['total'], single_ledger=True,
                rule='rerun-cpus standalone: own root, own CHARGES.jsonl; never the coop3 ledger')


def wall_clock(pace_s=1800, measured_update_wall_s=1800, diagnostic_extra_s=900,
               test_episode_wall_s=300, test_workers=2):
    """Paced updates (measured u7-u10 walls ~1,800 s each per arm, A the cheapest); B's
    diagnostics add ~diagnostic_extra_s each; R trails B by ~one update; tests run 3 arms."""
    per_update = max(pace_s, measured_update_wall_s)
    b = UPDATES*per_update+len(SCHEDULE['B']['isolated'])*diagnostic_extra_s
    r = b+per_update
    tests = 2*3*TEST_EPISODES*test_episode_wall_s/test_workers      # u8 + u24, three arms
    return dict(per_update_s=per_update, A_s=UPDATES*per_update, B_s=b, R_s=r, tests_s=tests,
                total_h=round((r+tests)/3600, 1))


# ---------------------------------------------------------------- own single ledger

AUTHORITY_LINE = ('S-COOPMWM RERUN-CPUS 2026-11-04: NEW STANDALONE A/B/R-v2v EXPERIMENT. This '
                  'root owns ONE SINGLE LEDGER (CHARGES.jsonl + ACCOUNTING.claim lock, real files '
                  'in THIS directory, never a symlink into another root) and its own registered '
                  'budget; every charge must land HERE. All three arms A/B/R run from the common '
                  'MAPPO u0 with identical code; B/R safety composites read A results from this '
                  'same root. No remote A reads, no shared accounting with any legacy root.\n')


def init_root(root, *, amendment_shas=()):
    """Create the NEW standalone run root with its own ledger; refuses if the root exists."""
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise ValueError('REFUSED: run root already exists: '+str(root))
    root.mkdir(parents=True)
    for name in ('CHARGES.jsonl', 'ACCOUNTING.claim'):
        with (root/name).open('x'):
            pass
    text = AUTHORITY_LINE+'effective-imitation commit '+EFFECTIVE_IMITATION_COMMIT+'\n'
    for amendment in amendment_shas:
        text += 'amendmentSHA'+amendment+'\n'
    with (root/'AUTHORITY.md').open('x', encoding='utf-8') as f:
        f.write(text)
    return own_ledger(root)


def own_ledger(root):
    """The OWN single-ledger binding: CHARGES.jsonl, ACCOUNTING.claim and AUTHORITY.md must be
    REAL files inside this root (a symlink or a path resolving elsewhere is another root's
    ledger and is refused), and AUTHORITY.md must carry the standalone rule."""
    root_resolved = Path(root).resolve()
    files = {}
    for name in ('CHARGES.jsonl', 'ACCOUNTING.claim', 'AUTHORITY.md'):
        p = Path(root)/name
        if p.is_symlink() or not p.is_file():
            raise ValueError('REFUSED: not this root\'s own single ledger: '+str(p))
        if p.resolve().parent != root_resolved:
            raise ValueError('REFUSED: ledger file resolves outside the run root: '+str(p))
        if p.stat().st_nlink > 1:
            raise ValueError('REFUSED: ledger file shared with another directory entry '
                             '(another root\'s ledger): '+str(p))
        files[name] = p
    text = files['AUTHORITY.md'].read_text(encoding='utf-8')
    if 'SINGLE LEDGER' not in text:
        raise ValueError('REFUSED: AUTHORITY.md is not the standalone single-ledger rule: '+str(root))
    if EFFECTIVE_IMITATION_COMMIT not in text:
        raise ValueError('REFUSED: AUTHORITY.md does not pin the effective-imitation commit')
    return dict(root=str(root_resolved), **files, foreign=False)


def bind_ledger(root, cfg):
    """Standalone accounting hook (replaces v2v_live.accounting): binds accel_storage's reserve
    to THIS root's own ledger and the standalone budget; refuses any foreign ledger first."""
    own_ledger(root)
    import types
    from tools.coopnative import storage
    from . import accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    b = cfg['budget']
    accel_storage._reserve = types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__, CAP=b['total']-SUPERSEDED_CHARGE, CAPS=b['phases']))
    return dict(root=str(root), total=b['total'], phases=dict(b['phases']),
                rule='own single ledger of this root only; no other root\'s ledger accepted')


def charge(root, jobs, phase, *, budget):
    """Reserve jobs against THIS root's own ledger (refuses any foreign ledger)."""
    bind_ledger(root, dict(budget=budget))
    from . import accel_storage
    return accel_storage.reserve(root, jobs, phase)


# ---------------------------------------------------------------- 3-arm runtime config

def runtime_config(base, *, root, reference_root, source_root, common_checkpoint_sha256):
    """Registered producer of the standalone runtime: v2v_runner.prepare_intent with every B/R
    fix pinned from u0 (kl_targeted, practice_starts 0.25/cap 2, imitation_target_space
    'effective'), the 24-update cap, and the standalone bindings: accounting_root = run_root =
    the NEW root (its own ledger), legacy_root = the staged legacy reference dir (only the
    fresh legacy-u3 isolated reference checkpoints live there), control arm A in the SAME new
    root (no remote A reads), own budget. Refuses while the effective-imitation pin differs."""
    from . import v2v_runner
    root, reference_root, source_root = (v2v_runner.remote_name(p) for p in
                                         (root, reference_root, source_root))
    if root == reference_root:
        raise ValueError('the run root must not be the staged legacy reference root')
    if root == source_root:
        raise ValueError('the run root must not be the deployed source root')
    if len(common_checkpoint_sha256) != 64:
        raise ValueError('common original MAPPO u0 checkpoint hash required')
    base = dict(base)
    base['budget'] = budget()
    cfg = v2v_runner.prepare_intent(base, root=root, legacy_root=reference_root,
        source_root=source_root, common_checkpoint_sha256=common_checkpoint_sha256,
        imitation_step_mode='kl_targeted', practice_starts=True, practice_fraction=.25,
        practice_repeat_cap=2, imitation_target_space='effective')
    cfg.update(run_root=root, legacy_root=reference_root, accounting_root=root,
               bank_source_root=root, source_root=source_root, standalone=True, rerun_cpus=True,
               rerun_arms=['A', 'B', 'R'],
               control_arm=dict(arm='A', root=root, update=UPDATES),
               effective_imitation_commit=EFFECTIVE_IMITATION_COMMIT,
               single_ledger=dict(root=root, charges='CHARGES.jsonl', lock='ACCOUNTING.claim',
                                  authority='AUTHORITY.md', foreign_refused=True),
               budget=budget(),
               a_results=dict(root=root, rule='B/R safety composites read A_UPDATE_<u> and the '
                              'same-world A references from THIS root; no remote A reads'))
    check_pins(cfg)
    return cfg


def check_pins(cfg):
    """Pins for all three arms (identical code; A differs only by having no bank/practice) plus
    the standalone single-ledger and same-root A bindings."""
    bad = {k: cfg.get(k) for k, v in PINS.items() if cfg.get(k) != v}
    if bad:
        raise ValueError('rerun-cpus pin mismatch: '+json.dumps(bad, sort_keys=True))
    if cfg.get('accounting_root') != cfg.get('run_root'):
        raise ValueError('REFUSED: the standalone experiment must account in its OWN root')
    if cfg.get('legacy_root') == cfg.get('run_root') or cfg.get('source_root') == cfg.get('run_root'):
        raise ValueError('REFUSED: the run root must not double as the reference/source root')
    if cfg.get('control_arm') != dict(arm='A', root=cfg['run_root'], update=UPDATES):
        raise ValueError('REFUSED: control arm A must run in the SAME new root (no remote A reads)')
    if list(cfg.get('rerun_arms', ())) != ['A', 'B', 'R']:
        raise ValueError('REFUSED: all three arms A/B/R rerun from MAPPO u0')
    if cfg.get('effective_imitation_commit') != EFFECTIVE_IMITATION_COMMIT:
        raise ValueError('REFUSED: effective-imitation commit not pinned to '+EFFECTIVE_IMITATION_COMMIT)
    if cfg.get('budget', {}).get('total') != budget()['total']:
        raise ValueError('REFUSED: standalone budget differs from the registered 3-arm quote')
    from .practice_starts import enabled
    if enabled(cfg, 'A') or not (enabled(cfg, 'B') and enabled(cfg, 'R')):
        raise ValueError('REFUSED: practice_starts applies to B and R only; A stays plain PPO')
    if cfg.get('diagnostic_schedule') != SCHEDULE:
        raise ValueError('REFUSED: diagnostic schedule differs from the registered plan')
    return True


# ---------------------------------------------------------------- read-only staging manifest

def guard_path(path):
    """HARD RULES: never open sealed/ or any path containing holdout/heldout/cal_split/chunk5/
    j4b/inc1; never read hetdata_truth or vType/driver/vehicle_params truth files."""
    s = str(path).replace('\\', '/').lower()
    hit = [t for t in FORBIDDEN if t in s]
    if hit:
        raise ValueError('REFUSED: forbidden path token '+repr(hit)+': '+str(path))
    if _TRUTH.search(s):
        raise ValueError('REFUSED: truth file path refused: '+str(path))
    return path


def sha(path):
    guard_path(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class _ReadRecorder:
    """Records every file the REAL bank readers open under the staged roots (read-only)."""

    def __init__(self, roots):
        self.roots = [Path(r).resolve() for r in roots]
        self.paths = set()
        self._saved = []

    def _under(self, path):
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            return None
        for root in self.roots:
            if resolved == root or root in resolved.parents:
                return resolved
        return None

    def _record(self, path, mode=None):
        mode = str(mode or 'r')
        if set(mode)-set('rb'):              # any write/create/append mode is never a read
            return path
        found = self._under(path)
        if found is not None:
            guard_path(found)
            self.paths.add(found)
        return path

    def __enter__(self):
        import builtins
        import gzip
        import pathlib
        def bind(original, record):
            def wrapper(path, *a, **k):
                record(path, k.get('mode') or (a[0] if a else None))
                return original(path, *a, **k)
            return wrapper
        self._saved.append((pathlib.Path, 'read_text', pathlib.Path.read_text))
        self._saved.append((pathlib.Path, 'read_bytes', pathlib.Path.read_bytes))
        self._saved.append((pathlib.Path, 'open', pathlib.Path.open))
        self._saved.append((builtins, 'open', builtins.open))
        self._saved.append((gzip, 'open', gzip.open))
        pathlib.Path.read_text = bind(pathlib.Path.read_text, lambda p, m=None: self._record(p, 'r'))
        pathlib.Path.read_bytes = bind(pathlib.Path.read_bytes, lambda p, m=None: self._record(p, 'r'))
        pathlib.Path.open = bind(pathlib.Path.open, self._record)
        builtins.open = bind(builtins.open, self._record)
        gzip.open = bind(gzip.open, self._record)
        return self

    def __exit__(self, *exc):
        import builtins
        import gzip
        import pathlib
        for owner, name, original in reversed(self._saved):
            setattr(owner, name, original)
        return False


def _local_of(remote_root):
    """The local mirror of a /home/ubuntu path: the mirror/stage directory is laid out like a
    filesystem root, so /home/ubuntu/<name> reads <mirror>/home/ubuntu/<name>."""
    p = PurePosixPath(str(remote_root).replace('\\', '/'))
    if not p.is_absolute() or p.parts[1:3] != ('home', 'ubuntu') or len(p.parts) < 4:
        raise ValueError('REFUSED: staged roots must live under '+HOME+': '+str(remote_root))
    return _mirror[0].joinpath('home', 'ubuntu', *p.parts[3:])


def _rel(remote_root, resolved):
    """Manifest key: the file's path relative to /home/ubuntu (posix, no leading slash)."""
    base = PurePosixPath(str(remote_root).replace('\\', '/'))
    rel = resolved.relative_to(_local_of(remote_root))
    out = PurePosixPath(*base.parts[1:], *rel.parts)        # drop the leading '/'
    if not str(out).startswith('home/ubuntu/'):
        raise ValueError('REFUSED: staged path is not relative to '+HOME+': '+str(out))
    return str(out)


_mirror = [None]


def staging_manifest(*, remote_root, sources=(), training_seed=20261101):
    """READ-ONLY staging manifest of every bank/tape/snapshot/MAPPO-checkpoint file the new run
    reads, resolved through the REAL bank readers (rb.bank_decisions with the LS sources, the
    per_seed_v1 confirmations incl. replication/solo/all8 tapes, rb.temperature, the practice
    decision links and snapshots), plus the legacy-u3 isolated reference checkpoints and the
    composite rule file. Every entry is listed relative to /home/ubuntu with its SHA256 and
    size so an operator can rsync+sha it to cpuS. ``remote_root`` is the original run root
    under /home/ubuntu; its contents are read from the local mirror registered by
    ``manifest_source``. ``sources`` maps arm -> (remote_path, ...) LS-COLLECT label roots.

    Superset rule (disclosed in the manifest): every receipt's full branch set is staged, and
    the whole snapshot directory of each arm's own bank, because the frozen per-update practice
    plans and R's fallback rows are sampled at runtime from any admitted example.
    """
    from . import replay_bank as rb, bank_statistics, practice_starts
    if _mirror[0] is None:
        raise ValueError('REFUSED: register the local mirror first (manifest_source)')
    root_remote = str(remote_root).replace('\\', '/')
    root_local = _local_of(root_remote)
    if not root_local.is_dir():
        raise ValueError('REFUSED: local mirror of the staged root is missing: '+str(root_local))
    src_remote, src_local = {}, {}
    for arm, paths in (sources or {}).items():
        src_remote[arm] = []
        for p in paths:
            remote = str(p).replace('\\', '/')
            src_remote[arm].append(remote)
            src_local[remote] = _local_of(remote)
    watched = [root_local]+list(src_local.values())
    entries, notes = {}, []
    recorded = set()

    def under(base, resolved):
        base = Path(base)
        return resolved == base or base in resolved.parents

    def remote_of(resolved):
        if under(root_local, resolved):
            return root_remote
        for remote, local in src_local.items():
            if under(local, resolved):
                return remote
        raise ValueError('REFUSED: file outside the staged roots: '+str(resolved))

    def add(local_path, kind):
        local_path = Path(local_path)
        if local_path.is_symlink() or not local_path.is_file():
            return False
        resolved = local_path.resolve()
        guard_path(resolved)
        key = _rel(remote_of(resolved), resolved)
        if key in entries:
            return True
        entries[key] = dict(sha256=sha(resolved), bytes=resolved.stat().st_size, kind=kind)
        return True

    # 1) the REAL readers, under read recording (proves the resolution and catches every read).
    #    READ-ONLY: rb.temperature is invoked only when its sealed file already exists (the
    #    unfrozen rule is derived deterministically on cpuS from the same decisions; calling it
    #    with >=20 decisions would WRITE the file into this read-only mirror).
    with _ReadRecorder(watched) as rec:
        for arm in ('B', 'R'):
            srcs = tuple(src_local.get(p, Path(p)) for p in src_remote.get(arm, ()))
            decisions = rb.bank_decisions(root_local, arm, sources=srcs)
            if (rb.bank_root(root_local, arm)/f'TEMPERATURE_{arm}_{training_seed}.json').is_file():
                rb.temperature(root_local, arm, training_seed, decisions)
            bank_statistics.confirmations_per_seed(rb.confirmations, rb, root_local, arm, srcs)
            practice_starts._links(root_local, arm)
        recorded = set(rec.paths)

    # 2) deterministic walk of the reader-defined bank layout (the run may read any receipt's
    #    branches and any snapshot at a later update; disclosed superset)
    for arm in ('B', 'R'):
        bank = rb.bank_root(root_local, arm)
        for sub, kind in (('decisions', 'decision_link'), ('confirmations', 'confirmation'),
                          ('confirmations_safety0426', 'confirmation_overlay'),
                          ('confirmations_all8', 'confirmation_all8'),
                          ('snapshots', 'snapshot')):
            for p in sorted((bank/sub).glob('*.*')):
                add(p, kind)
        for temperature in sorted(bank.glob('TEMPERATURE_*.json')):
            add(temperature, 'temperature')
        lesson = bank/f'LESSON_LIST_{arm}.json'
        if lesson.exists():
            add(lesson, 'lesson_list')
        for name, kind in (('sumo_verification', 'bank_receipt'),
                           ('confirm_replication', 'replication_receipt'),
                           ('confirm_seed2_all8', 'seed2_receipt'),
                           ('confirm_solo', 'solo_receipt'),
                           ('confirm_solo_replication', 'solo_receipt'),
                           ('confirm_solo_all8_s1', 'solo_receipt'),
                           ('confirm_solo_all8_s2', 'solo_receipt')):
            verifications = bank/name/'verifications'
            if not verifications.is_dir():
                continue
            for p in sorted(verifications.glob('*.json')):
                add(p, kind)
                receipt = json.loads(p.read_text(encoding='utf-8'))
                for branch in receipt.get('branches') or ():
                    d = bank/name/'branches'/receipt['verification_id']/branch['branch']
                    add(d/'RESULT.json', 'tape')
                    add(d/'observable.jsonl.gz', 'trace')
        for name in ('confirm_solo_replication', 'confirm_solo_all8_s1', 'confirm_solo_all8_s2'):
            for p in sorted((bank/name/'links').glob('*.json')):
                add(p, 'solo_link')
    for arm, paths in src_remote.items():
        for remote in paths:
            src = src_local[remote]
            for p in sorted((src/'labels').glob('*.json')):
                add(p, 'label')
            for p in sorted((src/'branches').glob('*/*/RESULT.json')):
                add(p, 'tape')
                add(p.parent/'observable.jsonl.gz', 'trace')
            for p in sorted((src/'label_attempts').glob('*/*/branches/*/*/RESULT.json')):
                add(p, 'tape')
                add(p.parent/'observable.jsonl.gz', 'trace')
    # 3) MAPPO checkpoints (u0 for all three arms; legacy-u3 isolated references) + rule file.
    #    All required: the run cannot start (u0), run its isolated synergy (u3 references) or
    #    arm the composite guard (rule) without them -> fail closed here, not on cpuS.
    required = [(root_local/CHECKPOINT_000_SOURCE, 'checkpoint_u0', 'common MAPPO u0 (staged as '
                '{A,B,R}_checkpoint_000.pt in the new root)'),
                (root_local/RULE_FILE, 'safety_rule', 'composite MAIN09:57 rule (copied into the new root)')]
    required += [(root_local/name, 'checkpoint_reference_u3', 'fresh legacy-u3 isolated reference')
                 for name in ISOLATED_REFERENCE_CHECKPOINTS]
    for path, kind, why in required:
        if not add(path, kind):
            raise ValueError('REFUSED: required staging input missing: '+str(path)+' ('+why+')')
    # 4) files the readers actually opened but the walk did not cover (fail-closed completeness)
    for p in sorted(recorded):
        if p.is_file():
            add(p, 'reader_opened')
    for key, entry in entries.items():
        if entry['kind'] == 'reader_opened':
            entry['kind'] = 'bank_input'
    if not entries:
        raise ValueError('REFUSED: the staging manifest resolved no files')
    from .exact_grade import _seal
    manifest = _seal(dict(schema='coopmwm.rerun_cpus.staging_manifest.v1', home=HOME,
        root=dict(remote=root_remote, local_mirror=str(root_local)),
        sources={arm: list(paths) for arm, paths in src_remote.items()},
        training_seed=training_seed, entries=dict(sorted(entries.items())),
        counts={k: sum(1 for e in entries.values() if e['kind'] == k)
                for k in sorted({e['kind'] for e in entries.values()})},
        supersets=['every enumerated receipt\'s full branch set (RESULT.json + observable.jsonl.gz)',
                   'each arm\'s whole own-snapshot directory (practice plans sample at runtime)'],
        notes=notes, readers=['replay_bank.bank_decisions', 'replay_bank.temperature',
                              'replay_bank.confirmations', 'bank_statistics.confirmations_per_seed',
                              'practice_starts._links']), 'manifest_sha256')
    return manifest


def manifest_source(mirror):
    """Register the local mirror directory whose layout equals /home/ubuntu (read-only)."""
    _mirror[0] = Path(mirror).resolve()
    return _mirror[0]


def verify_staged(manifest, stage_dir):
    """Every manifest file must exist under ``stage_dir`` (same /home/ubuntu-relative layout)
    with the exact manifest SHA256; refuses otherwise. Read-only."""
    from .exact_grade import verify_sealed
    try:
        verify_sealed(manifest, 'manifest_sha256')
    except ValueError as e:
        raise SystemExit('REFUSED: staging manifest seal broken: '+str(e))
    if manifest.get('home') != HOME:
        raise SystemExit('REFUSED: staging manifest home differs from '+HOME)
    stage = Path(stage_dir)
    bad = []
    for key, entry in sorted(manifest['entries'].items()):
        guard_path(key)
        staged = stage.joinpath(*key.split('/'))
        if not staged.is_file() or staged.is_symlink():
            bad.append(key+': missing')
        elif sha(staged) != entry['sha256']:
            bad.append(key+': sha differs')
    if bad:
        raise SystemExit('REFUSED: %d staged file(s) do not match the manifest: %s'
                         % (len(bad), '; '.join(bad[:5])))
    return dict(files=len(manifest['entries']), bytes=sum(e['bytes'] for e in manifest['entries'].values()),
                manifest_sha256=manifest['manifest_sha256'])


def staging_files_command(manifest):
    """Operator commands: write the file list, then rsync+sha it to cpuS (relative to /home/ubuntu)."""
    files = '\n'.join(sorted(manifest['entries']))
    return ('cat > STAGING_FILES.txt <<\'EOF\'\n'+files+'\nEOF\n'
            'rsync -a --numeric-ids --files-from=STAGING_FILES.txt '
            '<original-host>:/ /home/ubuntu/   # then rsync /home/ubuntu to cpuS unchanged')


# ---------------------------------------------------------------- approval-gated dry preflight

BUDGET_REQUEST = 'RERUNCPUS_BUDGET_REQUEST.json'
APPROVAL = 'OWNER_APPROVAL.json'
OWNER_QUOTE = 'OWNER_QUOTE.txt'
VERDICT_FILE = 'EVENING_VERDICT.json'
MANIFEST_FILE = 'STAGING_MANIFEST.json'


def preflight_dry(prep_dir, *, stage_dir, budget_request=BUDGET_REQUEST, approval=APPROVAL,
                  owner_quote=OWNER_QUOTE, verdict=VERDICT_FILE, manifest=MANIFEST_FILE):
    """Zero-physics dry preflight: refuses unless OWNER_APPROVAL.json exists and binds the exact
    budget request SHA, the owner quote SHA and the verdict file SHA (verdict positive) with the
    effective-imitation commit pinned, AND every staging-manifest file SHA matches the staged
    copies. Never dispatches; new_sim_s stays 0."""
    prep, stage = Path(prep_dir), Path(stage_dir)
    req = prep/budget_request
    if not req.is_file():
        raise SystemExit('REFUSED: budget request missing: '+str(req))
    request = json.loads(req.read_text(encoding='utf-8'))
    if request.get('quote', {}).get('total') != quote()['total']:
        raise SystemExit('REFUSED: budget request total differs from the 3-arm quote')
    app = prep/approval
    if not app.is_file():
        raise SystemExit('REFUSED: no OWNER_APPROVAL.json (owner budget approval required)')
    a = json.loads(app.read_text(encoding='utf-8'))
    if a.get('budget_request_sha256') != sha(req):
        raise SystemExit('REFUSED: OWNER_APPROVAL budget sha differs from the request')
    if a.get('owner_approved') is not True:
        raise SystemExit('REFUSED: owner approval not positive')
    q = prep/owner_quote
    if not q.is_file() or a.get('owner_quote_sha256') != sha(q):
        raise SystemExit('REFUSED: owner quote missing or sha differs')
    v = prep/verdict
    if not v.is_file() or a.get('verdict_file_sha256') != sha(v):
        raise SystemExit('REFUSED: verdict file missing or sha differs')
    verdict_value = json.loads(v.read_text(encoding='utf-8'))
    if verdict_value.get('verdict') != 'positive':
        raise SystemExit('REFUSED: post-fix evening verdict not positive')
    if a.get('effective_imitation_commit') != EFFECTIVE_IMITATION_COMMIT:
        raise SystemExit('REFUSED: effective-imitation commit not pinned to '+EFFECTIVE_IMITATION_COMMIT)
    m = prep/manifest
    if not m.is_file():
        raise SystemExit('REFUSED: staging manifest missing: '+str(m))
    man = json.loads(m.read_text(encoding='utf-8'))
    staged = verify_staged(man, stage)
    return dict(ready_for_registered_launch=True, new_sim_s=0, budget_request_sha256=sha(req),
                owner_quote_sha256=a.get('owner_quote_sha256'),
                verdict_file_sha256=a.get('verdict_file_sha256'),
                manifest_sha256=man['manifest_sha256'], staged=staged)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('quote', 'budget', 'wall_clock', 'manifest', 'preflight'))
    p.add_argument('--prep', default='.')
    p.add_argument('--stage', default='.')
    p.add_argument('--mirror', default=None, help='local mirror of /home/ubuntu')
    p.add_argument('--legacy-root', default='/home/ubuntu/coopmwm20261102_warm1')
    p.add_argument('--source-b', default=None, help='LS-COLLECT label root for arm B')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    if a.cmd == 'quote':
        print(json.dumps(dict(quote=quote(), wall_clock=wall_clock(), budget=budget())))
    elif a.cmd in ('manifest', 'preflight') and a.mirror:
        manifest_source(a.mirror)
    if a.cmd == 'manifest':
        sources = {'B': [a.source_b]} if a.source_b else {}
        m = staging_manifest(remote_root=a.legacy_root, sources=sources)
        out = a.out or 'STAGING_MANIFEST.json'
        Path(out).write_text(json.dumps(m, indent=1, sort_keys=True)+'\n', encoding='utf-8')
        print(json.dumps(dict(entries=len(m['entries']), manifest_sha256=m['manifest_sha256'],
                              command=staging_files_command(m))))
    elif a.cmd == 'preflight':
        print(json.dumps(preflight_dry(a.prep, stage_dir=a.stage)))
