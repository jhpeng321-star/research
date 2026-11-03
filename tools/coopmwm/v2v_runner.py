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
            active_arms=ARMS, v2v_intent=False, v2v_intent_horizon=5, v2v_intent_loss_weight=1.,
            v2v_intent_detach_trunk=True):
    """Prospective binding only: no directory, checkpoint, ledger or job mutation.

    ``v2v_intent`` (default False = unchanged binding) adds the planned-command intent keys,
    identical for every arm; ``v2v_dim`` must then be the intent block width K*(6+H+1).
    """
    root, legacy_root = remote_name(root), remote_name(legacy_root)
    if root == legacy_root:
        raise ValueError('V2V needs a separate run root')
    if len(common_checkpoint_sha256) != 64:
        raise ValueError('common original MAPPO u0 checkpoint hash required')
    if tuple(active_arms) != ARMS:
        raise ValueError('MAIN14:06 keeps all three arms; only test/diagnostic fallbacks permitted')
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
        diagnostics='MAIN14:06 reduced schedule; composite safety every update',
        diagnostic_schedule=dict(A=dict(lessons=[],isolated=[]),
            B=dict(lessons=[6,12,18,24],isolated=[6,12,18,24]),R=dict(lessons=[],isolated=[12,24])),
        phase_map=dict(train='v2v_train'),
        diagnostic_phases=dict(lessons='v2v_lessons',isolated='v2v_isolated'),
        reporting_scope_authorized=True,final_paired_branches=False,
        bank_source_root=legacy_root, bank_arms=dict(B='B', R='R'),
        bank_source_mode='read-only confirmed same-arm bank; V2V features from recorded observable traces',
        launch_authorized=False, phase_funding_authorized=False,
        launch_authority='MAIN2026-11-03 14:06 automatic on qualified integration/Linux/budget plus rehearsal or real one-update smokes',
        additional_cap_ceiling_sim_s=3000000,total_cap_ceiling_sim_s=17500000,
        physics_reservation_sim_s=0, new_sim_s=0,
        no_outcome_conditioned_changes=True,
        legacy_dev_checks='not included in new lesson+synergy reporting recipe; scope/funding must be explicit at admission')
    # No in-episode plans or old quota/exact-grader work in B-REPLAY.
    for key in ('sumo_verify', 'quota_aware_admission', 'parallel_group_queries',
                'bounded_fanout', 'parallel_fanout_factor', 'resume_progress'):
        cfg.pop(key, None)
    if v2v_intent:
        from .v2v_intent import settings
        cfg.update(v2v_intent=True, v2v_intent_horizon=int(v2v_intent_horizon),
                   v2v_intent_loss_weight=float(v2v_intent_loss_weight),
                   v2v_intent_detach_trunk=bool(v2v_intent_detach_trunk))
        settings(cfg)  # validates v2v_dim against the intent block width
    return cfg


def prepare_intent(base, *, root, legacy_root, source_root, common_checkpoint_sha256,
                   imitation_bank_filter='cooperative_only', practice_starts=False,
                   practice_fraction=None, practice_repeat_cap=None):
    """Audit I4: the registered producer of the intent-ON A/B/R runtime config (prospective
    binding only, like prepare). Identical for every arm: v2v_dim=48 (K4 x (6+5+1)),
    v2v_intent=True, horizon 5, loss weight 1.0, detach_trunk=True; bank labels
    bank_imitation='separate_full_pass', imitation_kl_cap=.045, bank_statistics='per_seed_v1'
    (B and R; A has no bank), imitation_bank_filter='cooperative_only' (B and R) and the
    practice_starts options (owner 2026-11-04 proposal; DEFAULT OFF = the launch default; ON is a
    B/R-equal registered runtime toggle enableable at an update boundary). Exactly the keys
    v2v_launch3.bind_runtime pins (practice keys are validated, never pinned, so ON passes)."""
    from .v2v_launch3 import BANK_OPTIONS, INTENT_OPTIONS, PRACTICE_KEYS, practice_options_for
    cfg = prepare(base, root=root, legacy_root=legacy_root, source_root=source_root,
                  common_checkpoint_sha256=common_checkpoint_sha256,
                  v2v_dim=int(INTENT_OPTIONS['v2v_dim']), v2v_intent=True,
                  v2v_intent_horizon=INTENT_OPTIONS['v2v_intent_horizon'],
                  v2v_intent_loss_weight=INTENT_OPTIONS['v2v_intent_loss_weight'],
                  v2v_intent_detach_trunk=INTENT_OPTIONS['v2v_intent_detach_trunk'])
    cfg.update(BANK_OPTIONS)
    # owner 2026-11-04 00:55: identical for B and R; the V2V runtime pins 'cooperative_only'
    cfg['imitation_bank_filter'] = imitation_bank_filter
    # owner 2026-11-04 proposal: registered practice_starts options, identical for B and R.
    cfg['practice_starts'] = bool(practice_starts)
    cfg['practice_fraction'] = (BANK_OPTIONS['practice_fraction'] if practice_fraction is None
                                else float(practice_fraction))
    cfg['practice_repeat_cap'] = (BANK_OPTIONS['practice_repeat_cap'] if practice_repeat_cap is None
                                  else int(practice_repeat_cap))
    if cfg['imitation_epochs'] < 5:
        raise ValueError('full-pass separate imitation requires at least five epochs')
    for key, want in dict(INTENT_OPTIONS, **{k: v for k, v in BANK_OPTIONS.items()
                                             if k not in PRACTICE_KEYS}).items():
        if cfg.get(key) != want or isinstance(cfg.get(key), bool) != isinstance(want, bool):
            raise ValueError(f'intent runtime key {key} differs from the launcher: {cfg.get(key)!r}')
    practice_options_for(cfg, 'B'), practice_options_for(cfg, 'R')
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
    smoke = cfg.get('qualification_stage') == 'one_update_smoke'
    if smoke:
        if (not cfg.get('smoke_authority_sha256') or not cfg.get('rehearsal_late_60min')
                or cfg.get('stop_after_update') != 1 or cfg.get('A_stop_after_update') != 1
                or cfg.get('resume_update') != 0 or cfg.get('A_resume_update') != 0):
            raise ValueError('registered late-rehearsal one-update smoke only')
    elif not (release.get('rehearsal_pass') or release.get('real_update_smokes_pass')):
        raise ValueError('rehearsal or real one-update smokes required')
    if not smoke and not release.get('rehearsal_pass'):
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
    not_before=cfg.get('next_not_before_s',{}).get(arm)
    last_start = [None if not_before is None else not_before-cfg['pace_s']]

    def practice_wave(r, b, policies, ws, ident, phase, extra, u, deterministic):
        """practice_starts scheduling (owner 2026-11-04 proposal, B/R only): the registered
        fraction of the update's PPO episodes are snapshot-restart episodes built by
        practice_starts.wave_call and dispatched through the SAME original_wave path (reserve,
        admission, episode, failure replacement), so charging and receipts are the existing
        episode hooks; the remaining episodes keep the registered batch_worlds(u) order (the
        last practice-count worlds of the list are replaced)."""
        from . import practice_starts
        from .warm_protocol import SEED
        ns_ident = cfg['job_namespace']+'_'+ident
        phase_name = cfg['phase_map'].get(phase, phase)
        want = practice_starts.practice_count(len(ws), cfg['practice_fraction'])
        plan = practice_starts.plan_for_update(root, arm, u, fraction=cfg['practice_fraction'],
            repeat_cap=cfg['practice_repeat_cap'], count=want, training_seed=SEED)
        rows = plan['rows']
        results = (original_wave(r, b, policies, ws[:len(ws)-len(rows)], role, ns_ident, phase_name,
                                 deterministic, extra) if len(rows) < len(ws) else [])
        for i, row in enumerate(rows):
            call = practice_starts.wave_call(root, arm, ns_ident, phase_name, row, i, extra)
            results.extend(original_wave(r, b, policies, call['worlds'], role, call['ident'],
                                         phase_name, deterministic, call['extras']))
        return sorted((practice_starts.strip_steps(e) if e.get('practice') else e for e in results),
                      key=lambda e: (e['world']['seed'], e['arm']))

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
            from . import practice_starts
            if practice_starts.enabled(cfg, arm):  # owner 2026-11-04 proposal (default OFF)
                return practice_wave(r, b, policies, ws, ident, phase, extra, u, deterministic)
        return original_wave(r, b, policies, ws, role, cfg['job_namespace']+'_'+ident,
                             cfg['phase_map'].get(phase,phase), deterministic, extra)

    def verify(r, a):
        if Path(r) != root or a != arm:raise ValueError('V2V root/arm binding mismatch')
        return deepcopy(cfg)

    def budget():
        b = deepcopy(cfg['budget']);b['updates'] = {a: 24 for a in cfg['active_arms']};return b

    changes = dict(verify=verify, budget=budget, wave=wave,
        Policy=hooks['policy_factory'], load=lambda p: hooks['load_checkpoint'](p, config=cfg),
        before_update=hooks['health_before_update'], braking=hooks['safety'])
    def record_lesson(r,a,policy,run_root,u,*args,**kwargs):
        if u not in cfg['diagnostic_schedule'][a]['lessons']:
            return dict(arm=a,update=u,status='not_scheduled',new_sim_s=0)
        return hooks['record_lesson'](r,a,policy,run_root,u,*args,
            **dict(kwargs,phase=cfg['diagnostic_phases']['lessons']))

    def diagnostics(r,a,u,policy,b,lesson):
        schedule=cfg['diagnostic_schedule'][a]
        if u not in schedule['lessons'] and u not in schedule['isolated']:
            return lesson
        return hooks['diagnostics'](r,a,u,policy,b,lesson)

    bank = dict(bank_pairs=hooks['bank_pairs'],bank_quality=hooks['bank_quality'],record_lesson=record_lesson)
    with patched(wr, changes), patched(rb, bank):
        wr.run_arm(root, arm, hooks=dict(diagnostics=diagnostics))
