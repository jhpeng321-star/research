"""Pre-T2C configuration and preparation paths. No component imports or RNG draws.

Kept separate so an absent/false opt-in produces the track-base serialized config.
"""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath


SCHEDULE = dict(A=dict(lessons=[], isolated=[]),
                B=dict(lessons=[6, 12, 18, 24], isolated=[6, 12, 18, 24]),
                R=dict(lessons=[], isolated=[12, 24]))


def quote():
    a, br, lessons, isolated = 691200, 1728000, 21120, 195660
    out = dict(a_train_sim_s=a, b_train_sim_s=br//2, r_train_sim_s=br//2,
               v2v_train=a+br, v2v_lessons=lessons, v2v_isolated=isolated,
               gate_bank_verification_ext=35820, interim_u8_sim_s=345600, final_u24_sim_s=345600,
               interim_u8=dict(per_arm=115200, arms=3, updates=8, episodes_per_arm=96),
               final_u24=dict(per_arm=115200, arms=3, updates=24, episodes_per_arm=96))
    out['total_without_tests'] = a+br+lessons+isolated+35820
    out['total'] = out['total_without_tests']+691200
    return out


def budget():
    q = quote()
    phases = dict(v2v_train=q['v2v_train'], v2v_lessons=21840, v2v_isolated=q['v2v_isolated'],
                  gate_bank_verification_ext=35820, v2v_interim=345600, v2v_final=345600)
    total = sum(phases.values())
    return dict(total=total, phases=phases, allocated=total, unallocated=0, updates=24,
                episodes_per_update=24, arms=['A', 'B', 'R'], quote_total_sim_s=q['total'],
                single_ledger=True,
                rule='rerun-cpus standalone: own root, own CHARGES.jsonl; never the coop3 ledger')


def wall_clock(pace_s=1800, measured_update_wall_s=1800, diagnostic_extra_s=900,
               test_episode_wall_s=300, test_workers=2):
    per_update = max(pace_s, measured_update_wall_s)
    b = 24*per_update+len(SCHEDULE['B']['isolated'])*diagnostic_extra_s
    r = b+per_update
    tests = 2*3*96*test_episode_wall_s/test_workers
    return dict(per_update_s=per_update, A_s=24*per_update, B_s=b, R_s=r, tests_s=tests,
                total_h=round((r+tests)/3600, 1))


def runtime_config(base, *, root, reference_root, source_root, common_checkpoint_sha256):
    from . import v2v_runner
    from .rerun_cpus import EFFECTIVE_IMITATION_COMMIT
    root, reference_root, source_root = (v2v_runner.remote_name(p) for p in
                                         (root, reference_root, source_root))
    if root == reference_root:
        raise ValueError('the run root must not be the staged legacy reference root')
    if root == source_root:
        raise ValueError('the run root must not be the deployed source root')
    if len(common_checkpoint_sha256) != 64:
        raise ValueError('common original MAPPO u0 checkpoint hash required')
    base = deepcopy(base)
    base['budget'] = budget()
    cfg = v2v_runner.prepare_intent(base, root=root, legacy_root=reference_root,
        source_root=source_root, common_checkpoint_sha256=common_checkpoint_sha256,
        imitation_step_mode='kl_targeted', practice_starts=True, practice_fraction=.25,
        practice_repeat_cap=2, imitation_target_space='effective')
    cfg.update(run_root=root, legacy_root=reference_root, accounting_root=root,
               bank_source_root=root, source_root=source_root, standalone=True, rerun_cpus=True,
               rerun_arms=['A', 'B', 'R'], control_arm=dict(arm='A', root=root, update=24),
               effective_imitation_commit=EFFECTIVE_IMITATION_COMMIT,
               single_ledger=dict(root=root, charges='CHARGES.jsonl', lock='ACCOUNTING.claim',
                                  authority='AUTHORITY.md', foreign_refused=True),
               budget=budget(), a_results=dict(root=root, rule='B/R safety composites read A_UPDATE_<u> and the '
                              'same-world A references from THIS root; no remote A reads'))
    check_pins(cfg)
    return cfg


def check_pins(cfg):
    from .rerun_cpus import PINS, EFFECTIVE_IMITATION_COMMIT
    if any(cfg.get(k) != v for k, v in PINS.items() if k != 'coop_reward'):
        raise ValueError('rerun-cpus pin mismatch')
    if cfg.get('accounting_root') != cfg.get('run_root'):
        raise ValueError('REFUSED: the standalone experiment must account in its OWN root')
    if cfg.get('control_arm') != dict(arm='A', root=cfg['run_root'], update=24):
        raise ValueError('REFUSED: control arm A must run in the SAME new root (no remote A reads)')
    if cfg.get('effective_imitation_commit') != EFFECTIVE_IMITATION_COMMIT:
        raise ValueError('REFUSED: effective-imitation commit not pinned')
    if cfg.get('budget') != budget() or cfg.get('diagnostic_schedule') != SCHEDULE:
        raise ValueError('REFUSED: legacy standalone registration differs')
    return True


def preflight_dry(prep_dir, *, stage_dir, budget_request, approval, owner_quote, verdict, manifest):
    """Legacy approval/quote/receipt contract, with guarded staged destinations.

    Legacy v1 manifests have no Track 2 roles, config or registration. Never use this
    path for an opted-in launch: the Track 2 runtime explicitly selects its preflight.
    """
    from .rerun_cpus import guard_path, sha, EFFECTIVE_IMITATION_COMMIT, HOME
    from .exact_grade import verify_sealed

    def checked(path):
        guard_path(path)
        guard_path(Path(path).resolve())
        return Path(path)

    prep, stage = checked(prep_dir), checked(stage_dir)
    req = checked(prep/budget_request)
    if not req.is_file():
        raise SystemExit('REFUSED: budget request missing: '+str(req))
    request = json.loads(req.read_text(encoding='utf-8'))
    if request.get('quote', {}).get('total') != quote()['total']:
        raise SystemExit('REFUSED: budget request total differs from the 3-arm quote')
    app = checked(prep/approval)
    if not app.is_file():
        raise SystemExit('REFUSED: no OWNER_APPROVAL.json (owner budget approval required)')
    a = json.loads(app.read_text(encoding='utf-8'))
    if a.get('budget_request_sha256') != sha(req):
        raise SystemExit('REFUSED: OWNER_APPROVAL budget sha differs from the request')
    if a.get('owner_approved') is not True:
        raise SystemExit('REFUSED: owner approval not positive')
    q = checked(prep/owner_quote)
    if not q.is_file() or a.get('owner_quote_sha256') != sha(q):
        raise SystemExit('REFUSED: owner quote missing or sha differs')
    v = checked(prep/verdict)
    if not v.is_file() or a.get('verdict_file_sha256') != sha(v):
        raise SystemExit('REFUSED: verdict file missing or sha differs')
    if json.loads(v.read_text(encoding='utf-8')).get('verdict') != 'positive':
        raise SystemExit('REFUSED: post-fix evening verdict not positive')
    if a.get('effective_imitation_commit') != EFFECTIVE_IMITATION_COMMIT:
        raise SystemExit('REFUSED: effective-imitation commit not pinned to '+EFFECTIVE_IMITATION_COMMIT)
    m = checked(prep/manifest)
    if not m.is_file():
        raise SystemExit('REFUSED: staging manifest missing: '+str(m))
    man = json.loads(m.read_text(encoding='utf-8'))
    try:
        verify_sealed(man, 'manifest_sha256')
    except ValueError as e:
        raise SystemExit('REFUSED: staging manifest seal broken: '+str(e))
    if man.get('home') != HOME:
        raise SystemExit('REFUSED: staging manifest home differs from '+HOME)
    bad = []
    for key, entry in sorted(man['entries'].items()):
        guard_path(key)
        p = PurePosixPath(key)
        if (p.is_absolute() or p.parts[:2] != ('home', 'ubuntu') or len(p.parts) < 4
                or '..' in p.parts or '\\' in key or ':' in key or str(p) != key):
            raise SystemExit('REFUSED: unsafe staged destination: '+key)
        staged = checked(stage.joinpath(*p.parts))
        if stage.resolve() not in staged.resolve().parents:
            raise SystemExit('REFUSED: staged destination escapes stage: '+key)
        if not staged.is_file() or staged.is_symlink():
            bad.append(key+': missing')
        elif sha(staged) != entry['sha256']:
            bad.append(key+': sha differs')
    if bad:
        raise SystemExit('REFUSED: %d staged file(s) do not match the manifest: %s'
                         % (len(bad), '; '.join(bad[:5])))
    staged = dict(files=len(man['entries']), bytes=sum(e['bytes'] for e in man['entries'].values()),
                  manifest_sha256=man['manifest_sha256'])
    return dict(ready_for_registered_launch=True, new_sim_s=0, budget_request_sha256=sha(req),
                owner_quote_sha256=a.get('owner_quote_sha256'),
                verdict_file_sha256=a.get('verdict_file_sha256'),
                manifest_sha256=man['manifest_sha256'], staged=staged)
