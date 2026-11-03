"""Prepare separate A/B/R-v2v roots; existing warm runner remains default-off.

MAIN13:45 authorises automatic launch after all release and funding gates.
No dispatcher is provided by this preparation module.
Component adapters are injected by the qualified V2V integration, never guessed
from unfinished OBS/MODEL/BANK modules or their development stubs.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
import time

ARMS = ('A', 'B', 'R')
REQUIRED_HOOKS = ('configure', 'accounting', 'policy_factory', 'load_checkpoint',
                  'bank_pairs', 'bank_quality', 'record_lesson', 'diagnostics',
                  'health_before_update', 'safety')


def remote_name(path):
    return str(PurePosixPath(str(path).replace('\\', '/')))


def prepare(base, *, root, legacy_root, source_root, common_checkpoint_sha256, v2v_dim,
            active_arms=ARMS):
    """Prospective binding only: no directory, checkpoint, ledger or job mutation."""
    root, legacy_root = remote_name(root), remote_name(legacy_root)
    if root == legacy_root:
        raise ValueError('V2V needs a separate run root')
    if len(common_checkpoint_sha256) != 64:
        raise ValueError('common original MAPPO u0 checkpoint hash required')
    if tuple(active_arms) not in (ARMS, ('A', 'B')):
        raise ValueError('MAIN permits three arms or budget-driven A/B-only fallback')
    if type(v2v_dim) is not int or v2v_dim <= 0:
        raise ValueError('qualified observation dimension required; no stub default')
    cfg = deepcopy(base)
    cfg.update(v2v=True, v2v_dim=v2v_dim, run_root=root, legacy_root=legacy_root,
        accounting_root=legacy_root, source_root=remote_name(source_root),
        resume_update=0, A_resume_update=0, A_stop_after_update=24, stop_after_update=24,
        active_arms=list(active_arms),final_updates={a: 24 for a in active_arms},
        start_checkpoint_sha256={a:common_checkpoint_sha256 for a in ARMS},
        common_checkpoint_sha256=common_checkpoint_sha256,
        arm_names={a: a+'-v2v' for a in ARMS}, job_namespace='v2v1',
        b_replay=True, replay_beta=1., imitation_epochs=5, pace_s=1800,
        r_concurrent_lag1_quota=True, skip_legacy_reporting=True,
        diagnostics='every update: matched lesson plus original/isolated policy synergy; composite safety',
        bank_source_root=legacy_root, bank_arms=dict(B='B', R='R'),
        bank_source_mode='read-only confirmed same-arm bank; V2V features from recorded observable traces',
        launch_authorized=False, phase_funding_authorized=False,
        launch_authority='MAIN2026-11-03 13:45 automatic on qualified integration/rehearsal/Linux/budget gates',
        additional_cap_ceiling_sim_s=3000000,total_cap_ceiling_sim_s=17500000,
        physics_reservation_sim_s=0, new_sim_s=0,
        no_outcome_conditioned_changes=True,
        legacy_dev_checks='not included in new lesson+synergy reporting recipe; scope/funding must be explicit at admission')
    # No in-episode plans or old quota/exact-grader work in B-REPLAY.
    for key in ('sumo_verify', 'quota_aware_admission', 'parallel_group_queries',
                'bounded_fanout', 'parallel_fanout_factor', 'resume_progress'):
        cfg.pop(key, None)
    return cfg


def admit(cfg, release, hooks):
    """Fail before loading models or invoking any physics/adapter on missing gates."""
    if not cfg.get('launch_authorized') or not cfg.get('phase_funding_authorized'):
        raise ValueError('MAIN V2V launch and phase funding not authorised')
    if not cfg.get('launch_amendment_sha256') or not cfg.get('funding_amendment_sha256'):
        raise ValueError('SHA-bound V2V authority required')
    if (not release.get('qualified') or not release.get('linux_cpu_pass')
            or not release.get('zero_init_parity') or not release.get('default_off_byte_identical')
            or release.get('stub', True) or release.get('v2v_dim') != cfg['v2v_dim']):
        raise ValueError('qualified non-stub V2V integration required')
    if not (release.get('rehearsal_pass') or release.get('late_rehearsal_smokes_pass')):
        raise ValueError('rehearsal or authorised late-rehearsal real smokes required')
    if not release.get('rehearsal_pass'):
        if time.time() < release.get('ready_unix',float('inf'))+3600:
            raise ValueError('late-rehearsal fallback requires 60 minutes after READY')
        if set(release.get('smoke_arms', ())) != set(cfg['active_arms']):
            raise ValueError('real one-update smoke required for every active arm')
    if not 14500000 <= cfg['budget']['total'] <= 17500000:
        raise ValueError('owner V2V budget ceiling exceeded')
    if set(cfg['start_checkpoint_sha256'].values()) != {cfg['common_checkpoint_sha256']}:
        raise ValueError('all V2V arms must start from the same MAPPO u0')
    if cfg['run_root'] == cfg['legacy_root'] or cfg['accounting_root'] != cfg['legacy_root']:
        raise ValueError('isolated outputs and shared authoritative accounting required')
    missing = [n for n in REQUIRED_HOOKS if not callable(hooks.get(n))]
    if missing:
        raise ValueError('integration hooks missing: '+','.join(missing))
    if not cfg.get('A_diagnostic_slate_authorized'):
        raise ValueError('explicit A-v2v lesson/cooperative slate required')
    if not cfg.get('reporting_scope_authorized'):
        raise ValueError('V2V diagnostic/dev reporting scope required')


@contextmanager
def patched(module, changes):
    old = {k: getattr(module, k) for k in changes}
    try:
        for k, v in changes.items():setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():setattr(module, k, v)


def run_arm(root, arm, cfg, release, hooks, *, clock=time.time, sleep=time.sleep):
    """Wire the qualified component callbacks to the inherited PPO/AWR loop.

    Each arm runs in its own durable process. No live non-V2V process is patched.
    The integration's accounting hook must bind the SAME authoritative ledger
    and lock before any wave; diagnostic hooks write only this new run root.
    """
    if arm not in cfg.get('active_arms', ()) or remote_name(root) != cfg['run_root']:
        raise ValueError('registered V2V root/arm required')
    admit(cfg, release, hooks)
    from . import warm_runner as wr, replay_bank as rb
    from .storage import read
    from .protocol import sha, checked
    root = checked(root)
    authority = (root/'AUTHORITY.md').read_text()
    if any(cfg[k] not in authority for k in ('launch_amendment_sha256', 'funding_amendment_sha256')):
        raise ValueError('V2V authority absent from new root')
    if sha(root/f'{arm}_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
        raise ValueError('common original MAPPO u0 checkpoint differs')
    hooks['accounting'](root, cfg)  # shared ledger + shared lock; never a second cap
    hooks['configure'](root, arm, cfg)
    original_wave = wr.wave
    last_start = [None]

    def wave(r, b, policies, ws, role, ident, phase, deterministic=True, extras=None):
        extra = dict(extras or {}, v2v=True, v2v_dim=cfg['v2v_dim'])
        if role == 'train':
            u = int(extra['update'])
            if arm == 'R':
                while not (root/f'B_UPDATE_{u:03}.json').exists():
                    wr.assert_running(root, arm);sleep(10)
            if last_start[0] is not None:
                while clock() < last_start[0]+cfg['pace_s']:
                    wr.assert_running(root, arm);sleep(10)
            last_start[0] = clock()
            extra['b_replay'] = arm in ('B', 'R')  # A remains plain PPO, no bank capture
        return original_wave(r, b, policies, ws, role, cfg['job_namespace']+'_'+ident,
                             phase, deterministic, extra)

    def verify(r, a):
        if Path(r) != root or a != arm:raise ValueError('V2V root/arm binding mismatch')
        return deepcopy(cfg)

    def budget():
        b = deepcopy(cfg['budget']);b['updates'] = {a: 24 for a in cfg['active_arms']};return b

    changes = dict(verify=verify, budget=budget, wave=wave,
        Policy=hooks['policy_factory'], load=lambda p: hooks['load_checkpoint'](p, config=cfg),
        before_update=hooks['health_before_update'], braking=hooks['safety'])
    bank = {n: hooks[n] for n in ('bank_pairs', 'bank_quality', 'record_lesson')}
    with patched(wr, changes), patched(rb, bank):
        wr.run_arm(root, arm, hooks=dict(diagnostics=hooks['diagnostics']))
