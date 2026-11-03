"""Receipt-only V2V learning-mechanics guard (MAIN Nov4 02:08/06:32).

No physics, inference, policy selection, source mutation or process signalling.
Cooperation and intent-loss trends are reported, never health-pause criteria.
"""
import math
from pathlib import Path
import time


def inspect_update(value):
    """Evaluate only the already registered numerical learning invariants."""
    bank = value.get('bank_imitation') or {}
    exact = value.get('exact_imitation') or {}
    filtered = exact.get('imitation_filter') or {}
    ppo = value.get('ppo') or []
    failures = []
    if bank.get('imitation_failed'):
        failures.append('imitation_failed')
    pairs = bank.get('pairs_total', 0)
    if pairs:
        if bank.get('pass1_effective') is not True:
            failures.append('ineffective_pass1')
        if bank.get('epochs_completed', 0) < 1:
            failures.append('no_complete_imitation_pass')
        kl, limit = bank.get('kl'), bank.get('kl_limit')
        if not isinstance(kl, (float, int)) or not math.isfinite(kl):
            failures.append('nonfinite_imitation_kl')
        elif not isinstance(limit, (float, int)) or not math.isfinite(limit) or kl > limit:
            failures.append('imitation_kl_out_of_limit')
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
    return dict(arm=value['arm'], update=value['update'], timestamp_utc=value.get('timestamp_utc'),
        failures=failures, learning_mechanics_ok=not failures,
        epochs_completed=bank.get('epochs_completed'), requested_epochs=bank.get('requested_epochs'),
        pass1_effective=bank.get('pass1_effective'), pass1_loss_drop=bank.get('pass1_loss_drop'),
        imitation_failed=bank.get('imitation_failed'), kl=bank.get('kl'), kl_limit=bank.get('kl_limit'),
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
        rule='registered mechanics only; empty fairness-capped bank is allowed; intent/cooperation trends report-only')


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
    lock = (control/'WATCH.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(control/'OWNER.json', owner())
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    while not (control/'STOP.json').exists():
        summaries = {}
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
                    report = dict(inspect_update(read(path)), source_receipt_sha256=sha(path),
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
        atomic(control/'STATUS.json', dict(unix=time.time(), latest=summaries, new_sumo_sim_s=0))
        if all(summaries.get(a, {}).get('update', 0) >= 24 for a in ('A', 'B', 'R')):
            write(control/'COMPLETE.json', dict(unix=time.time(), final_update=24, new_sumo_sim_s=0))
            return
        time.sleep(30)


if __name__ == '__main__':
    import sys
    watch(*sys.argv[1:])
