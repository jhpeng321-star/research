"""Pre-T2C configuration path. No Track 2 imports, RNG draws, or worker changes.

Kept separate so an absent/false opt-in produces the track-base serialized config.
"""
from copy import deepcopy


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
