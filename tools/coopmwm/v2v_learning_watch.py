"""Receipt-only V2V learning-mechanics guard (MAIN Nov4 02:08/06:32; statistical pass-1 rule 09:00).

No physics, inference, policy selection, source mutation or process signalling.
Cooperation and intent-loss trends are reported, never health-pause criteria.
"""
import math
from pathlib import Path
import time


def complete_update(path):
    """A receipt is published with open('x'), so its first read may be partial."""
    import json
    try:
        before = path.stat()
        value = json.loads(path.read_text())
        after = path.stat()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return None, type(error).__name__
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return None, 'receipt_still_publishing'
    return value, None


def registered_kl_limits():
    """The registered imitation_kl_limit values (c4/imit-kl-option). The runtime's pinned limit
    arrives from the SHA-pinned V2V_RUNTIME.json (verified in watch) and every receipt must
    carry the SAME value; the KL-breach check uses that runtime limit, never a hardcoded bound
    and never the receipt's own claim alone."""
    from .bank_imitation import KL_LIMITS
    return KL_LIMITS


def runtime_kl_limit(runtime_value):
    """The imitation_kl_limit pinned by a (hash-verified) V2V_RUNTIME.json value. A pre-option
    runtime without the key keeps the registered default .045; anything outside the registered
    alternatives refuses (fail closed), mirroring v2v_launch3.bind_runtime."""
    from .bank_imitation import KL_LIMIT
    cfg = runtime_value.get('config') if isinstance(runtime_value, dict) else None
    limit = cfg.get('imitation_kl_limit', KL_LIMIT) if isinstance(cfg, dict) else None
    if limit not in registered_kl_limits():
        raise ValueError('unregistered runtime imitation_kl_limit ' + repr(limit))
    return limit


def inspect_update(value, runtime_kl_limit=None):
    """Evaluate only the already registered numerical learning invariants.

    runtime_kl_limit is the RUNTIME's pinned imitation_kl_limit (watch passes the value read
    from the SHA-pinned V2V_RUNTIME.json); None assumes the registered default .045. A receipt
    whose kl_limit differs from the runtime value FAILS ('imitation_kl_limit_mismatch': a
    receipt can never relax its own bound), and the KL-breach check runs against the runtime
    limit, never against the receipt's own claim."""
    bank = value.get('bank_imitation') or {}
    exact = value.get('exact_imitation') or {}
    filtered = exact.get('imitation_filter') or {}
    ppo = value.get('ppo') or []
    failures = []
    if bank.get('imitation_failed'):
        failures.append('imitation_failed')
    pairs = bank.get('pairs_total', 0)
    reported = []
    change = bank.get('pass1_loss_change') or {}
    # c4/imit-kl-option: the RUNTIME's pinned imitation_kl_limit (None = registered default .045)
    # is the only bound the checks trust; the receipt's own kl_limit must agree with it.
    from .bank_imitation import KL_LIMIT, PPO_STOP_KL
    runtime = KL_LIMIT if runtime_kl_limit is None else runtime_kl_limit
    if pairs:
        # MAIN 2026-11-04 09:00 (statistical rule, B and R identically, no hand-set threshold):
        # pass 1 pauses only when its paired per-pair loss change is significantly POSITIVE
        # (bootstrap 95% CI over imitation pairs above 0 = imitation made the loss worse).
        # A non-significant change (or a legacy receipt without the CI) is reported only.
        if change.get('significantly_worse') is True:
            failures.append('pass1_loss_significantly_worse')
        elif bank.get('pass1_effective') is not True:
            reported.append('pass1_not_effective_not_significant'
                            if change.get('ci95') is not None else 'pass1_not_effective_no_ci_reported_only')
        if bank.get('epochs_completed', 0) < 1:
            failures.append('no_complete_imitation_pass')
        kl, limit = bank.get('kl'), bank.get('kl_limit')
        if not isinstance(kl, (float, int)) or not math.isfinite(kl):
            failures.append('nonfinite_imitation_kl')
        elif (not isinstance(limit, (float, int)) or not math.isfinite(limit)
                or kl > runtime):
            failures.append('imitation_kl_out_of_limit')
        elif limit != runtime:
            # c4/imit-kl-option fix: the receipt's own kl_limit is evidence, never authority;
            # it must equal the runtime's pinned value (a mismatching receipt fails closed).
            failures.append('imitation_kl_limit_mismatch')
        elif limit not in registered_kl_limits():
            # a limit outside the registered set fails closed (unregistered values must never
            # relax this; reached only when the receipt agrees with an unregistered runtime).
            failures.append('unregistered_imitation_kl_limit')
    for i, row in enumerate(ppo):
        for key in ('loss', 'kl', 'gradient_norm'):
            number = row.get(key)
            if isinstance(number, (int, float)) and not math.isfinite(number):
                failures.append(f'nonfinite_ppo_{key}_record_{i}')
    intent = bank.get('v2v_intent') or {}
    before = (intent.get('before') or {}).get('intent_loss')
    after = (intent.get('after') or {}).get('intent_loss')
    for name, number in (('before', before), ('after', after)):
        if isinstance(number, (int, float)) and not math.isfinite(number):
            failures.append('nonfinite_bank_intent_' + name)
    own_ppo = [row for row in ppo if row.get('own_policy_steps', 0)]
    # c4/imit-kl-option gate: the report-only total-KL flag (bank_imitation sets it when the
    # runtime limit exceeds .045) is surfaced here; the monitor also derives it for a receipt
    # that carries the total but predates the flag. Report-only, never a pause criterion.
    total = bank.get('total_kl_vs_pre_update')
    total_flag = bank.get('total_kl_above_ppo_stop_plus_limit')
    if total_flag is None and runtime is not None and runtime > KL_LIMIT \
            and isinstance(total, (float, int)) and math.isfinite(total):
        total_flag = bool(total > PPO_STOP_KL + runtime)
    if total_flag is True:
        reported.append('total_kl_above_ppo_stop_plus_limit')
    return dict(arm=value['arm'], update=value['update'], timestamp_utc=value.get('timestamp_utc'),
        failures=failures, learning_mechanics_ok=not failures,
        runtime_kl_limit=runtime,
        epochs_completed=bank.get('epochs_completed'), requested_epochs=bank.get('requested_epochs'),
        pass1_effective=bank.get('pass1_effective'), pass1_loss_drop=bank.get('pass1_loss_drop'),
        pass1_loss_change_mean=change.get('mean'), pass1_loss_change_ci95=change.get('ci95'),
        pass1_loss_change_n=change.get('n'), reported_only=reported,
        imitation_failed=bank.get('imitation_failed'), kl=bank.get('kl'), kl_limit=bank.get('kl_limit'),
        total_kl_vs_pre_update=total,
        total_kl_above_ppo_stop_plus_limit=total_flag,
        action_error_before_mps2=bank.get('action_error_before_mps2'),
        action_error_after_mps2=bank.get('action_error_after_mps2'),
        imitated_cooperative_examples=filtered.get('imitated_cooperative_examples'),
        imitated_cooperative_by_seed=filtered.get('imitated_cooperative_by_seed'),
        empty_filtered_bank=filtered.get('empty_bank'),
        bank_intent_loss_before=before, bank_intent_loss_after=after,
        bank_intent_loss_increased=(after > before if before is not None and after is not None else None),
        first_minibatch_ppo_kl=own_ppo[0].get('kl') if own_ppo else None,
        ppo_kl_max=max((row.get('kl', 0.) for row in own_ppo), default=None),
        wall_s=value.get('wall_s'), eta_final_hours=value.get('eta_final_hours'),
        rule='registered mechanics only; empty fairness-capped bank is allowed; intent/cooperation trends report-only; '
             'pass 1 pauses only on a significantly worse paired loss change (bootstrap CI > 0), MAIN 2026-11-04 09:00; '
             'imitation KL is checked against the RUNTIME limit pinned in V2V_RUNTIME.json (watch passes it here); '
             'a receipt kl_limit differing from that runtime value fails closed (imitation_kl_limit_mismatch), fix round 2026-11-04 c4/imit-kl-option')


def watch(root, control):
    import fcntl
    import json
    import os
    from .protocol import sha
    from .storage import read, write
    from .replay_start import atomic, owner
    from .warm_health import pause, HealthPause
    root, control = Path(root), Path(control)
    binding = read(control/'BINDING.json')
    if str(root) != binding['run_root'] or sha(__file__) != binding['source_sha256']:
        raise ValueError('learning watcher source/root mismatch')
    if binding['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():
        raise ValueError('learning watcher registration absent')
    if sha(root/'V2V_RUNTIME.json') != binding['runtime_sha256']:
        raise ValueError('learning watcher runtime authority differs')
    # fix round (c4/imit-kl-option): the RUNTIME's pinned imitation_kl_limit (the file's SHA is
    # verified above) is the only bound the KL checks trust; every receipt must carry it too.
    runtime_limit = runtime_kl_limit(read(root/'V2V_RUNTIME.json'))
    lock = (control/'WATCH.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(control/'OWNER.json', owner())
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    while not (control/'STOP.json').exists():
        summaries, pending_reads = {}, {}
        for arm in ('A', 'B', 'R'):
            for path in sorted(root.glob(arm+'_UPDATE_*.json')):
                if not path.stem.rsplit('_', 1)[-1].isdigit():
                    continue
                target = control/(path.stem+'.json')
                if target.exists():
                    report = read(target)
                    if report['source_receipt_sha256'] != sha(path):
                        raise ValueError('immutable update receipt changed: '+path.name)
                else:
                    value, pending = complete_update(path)
                    if pending:
                        pending_reads[path.name] = pending
                        continue
                    report = dict(inspect_update(value, runtime_limit), source_receipt_sha256=sha(path),
                        unix=time.time(), new_sumo_sim_s=0)
                    safety = root/f'{arm}_SAFETY_COMPOSITE_{report["update"]:03}.json'
                    if safety.is_file():
                        s = read(safety)
                        report['safety'] = dict(release=s['release'], n=s['n'], metrics=s['metrics'])
                    write(target, report)
                summaries[arm] = report
                if arm in ('B', 'R') and report['failures']:
                    try:
                        pause(root, 'V2V_LEARNING_MECHANICS', report)
                    except HealthPause:
                        atomic(control/'PAUSED.json', report)
        atomic(control/'STATUS.json', dict(unix=time.time(), latest=summaries,
            pending_receipt_reads=pending_reads, new_sumo_sim_s=0))
        if all(summaries.get(a, {}).get('update', 0) >= 24 for a in ('A', 'B', 'R')):
            write(control/'COMPLETE.json', dict(unix=time.time(), final_update=24, new_sumo_sim_s=0))
            return
        time.sleep(30)


if __name__ == '__main__':
    import sys
    watch(*sys.argv[1:])
