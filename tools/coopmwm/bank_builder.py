"""MAIN04:06 offline adapter: delegated confirmation/synergy, existing LS labels only."""
import json
import os
from pathlib import Path
import time

PHASE = 'bank_verification_ext'


def quoted_jobs(snapshot, best, verify, solo_candidate):
    """Reserve complete replication/extended/conditional-solo pairs before physics."""
    jobs = []
    groups = [('confirm_replication', [best], 120),
              ('confirm_extended', [best], 600)]
    if len(best['members']) >= 2:
        groups += [('confirm_solo', [solo_candidate(best, m)], 120) for m in best['members']]
    for kind, candidates, tail in groups:
        for spec in verify(snapshot, candidates, tail_s=tail):
            jobs.append(dict(id=f'{kind}_{spec["verification_id"]}_{spec["branch"]}',
                             sim_s=spec['sim_s']))
    assert len({j['id'] for j in jobs}) == len(jobs)
    return jobs


def quoted_jobs_all8(snapshot, candidates, vid, verify, replicate):
    """verify_mode=all8_both_seeds: reserve the seed-2 all-K + policy branches before physics
    (same ids as the live charge hook: <kind>_<vid>_<branch>). Extended/solo verifications are
    conditional on the per-seed admission and are charged live through rb._confirm_charge."""
    from .verify_perseed import KIND_SEED2
    jobs = [dict(id=f'{KIND_SEED2}_{spec["verification_id"]}_{spec["branch"]}', sim_s=spec['sim_s'])
            for spec in verify(replicate(snapshot, vid), candidates)]
    assert len({j['id'] for j in jobs}) == len(jobs)
    return jobs


def all8_charge(original, entries):
    """Seed-2 branches use the precharged rows; every other kind keeps the live hook."""
    from .verify_perseed import KIND_SEED2
    def charge_for(run_root, phase, kind):
        if kind == KIND_SEED2:
            return precharged_charge(entries, kind)
        return original(run_root, phase, kind)
    return charge_for


def run_all8_round(root, control, config, immutable, phase=PHASE):
    """One builder pass under verify_mode=all8_both_seeds (accepted AND rejected decisions)."""
    import traceback
    from . import replay_bank as rb, sumo_verify as sv, verify_perseed as vp
    from . import accel_storage
    from .storage import ledger
    original_charge = rb._confirm_charge
    for r in vp.pending(root, 'B', sources=config['sources']):
        vid = r['verification_id']
        attempt = control/'attempts_all8'/(vid+'.json')
        claim = control/'attempts_all8'/(vid+'.claim.json')
        if attempt.exists():
            continue
        if claim.exists():
            immutable(attempt, dict(verification_id=vid, status='interrupted',
                reason='existing claim without completion; never replay', unix=time.time()))
            continue
        snapshot = rb._decision_snapshot(root, 'B', r, {})
        candidates = vp.first_seed_candidates(r, r['_branches'])
        jobs = quoted_jobs_all8(snapshot, candidates, vid, sv.build_branches, rb.replicate)
        try:
            accel_storage.reserve(root, jobs, phase)
        except ValueError as error:
            immutable(control/f'BUDGET_WAIT_{time.time_ns()}.json', dict(unix=time.time(), verification_id=vid,
                required_sim_s=sum(j['sim_s'] for j in jobs), error=repr(error), new_sumo_sim_s=0,
                verify_mode=vp.ALL8, action='wait before claim; no partial confirmation'))
            break
        immutable(claim, dict(unix=time.time(), verification_id=vid, jobs=jobs, verify_mode=vp.ALL8,
            precharged_sim_s=sum(j['sim_s'] for j in jobs),
            conditional_live_charges='extended/solo verifications via rb._confirm_charge'))
        charges = {j['id']: j for j in ledger(root)}
        rb._confirm_charge = all8_charge(original_charge, charges)
        try:
            receipt = {k: v for k, v in r.items() if not k.startswith('_')}
            def runner_for(work):
                return sv.SubprocessBranchRunner(Path(work)/'branches', workers=2,
                    timeout_s=config['branch_timeout_s'])
            out = vp.confirm_all8(root, 'B', receipt, snapshot, runner_for, str(root), phase, workers=2,
                version=r['_policy_version'], candidates=candidates, branches_dir=r['_branches'])
            result = dict(unix=time.time(), verification_id=vid, status='complete', verify_mode=vp.ALL8,
                admitted=out['admitted'], winners=out['winners'], record_sha256=out['confirmation_sha256'],
                precharged_sim_s=sum(j['sim_s'] for j in jobs), charged_sim_s=out['charged_sim_s'])
        except Exception as error:
            result = dict(unix=time.time(), verification_id=vid, status='failed', verify_mode=vp.ALL8,
                error=repr(error), traceback=traceback.format_exc(),
                precharged_sim_s=sum(j['sim_s'] for j in jobs), replay=False)
        finally:
            rb._confirm_charge = original_charge
        immutable(attempt, result)
        print(json.dumps(result), flush=True)


def precharged_charge(entries, kind):
    """Read-only delegated charge hook: all jobs were reserved in one transaction."""
    def charge(vid, specs):
        for spec in specs:
            job = entries[f'{kind}_{vid}_{spec["branch"]}']
            assert job['phase'] == PHASE and job['sim_s'] == spec['sim_s']
    return charge


def run(control):
    import fcntl
    import signal
    import traceback
    from . import replay_bank as rb, sumo_verify as sv
    from . import accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    from .warm_protocol import install_accounting
    from .warm_spool import immutable
    from .ls_label_runtime import sumo_environment
    from .storage import ledger
    from . import bank_safety as bs
    control = Path(control)
    config = json.loads((control/'BINDING.json').read_text())
    root = Path(config['run_root'])
    source = Path(config['source_root'])
    import hashlib
    for name, expected in config['source_hashes'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected, name
    bank = rb.bank_root(root, 'B');bank.mkdir(parents=True, exist_ok=True)
    with (bank/'BUILDER_0406.claim').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        immutable(control/'OWNER.json', dict(pid=os.getpid(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            start_ticks=Path('/proc/self/stat').read_text().split()[21], unix=time.time()))
        os.nice(max(0, 19-os.nice(0)))
        os.environ.update(sumo_environment(config['sumo_home']))
        install_accounting(root, config['budget'])
        accel_storage._reserve.__globals__['CAP'] = config['operational_cap']-SUPERSEDED_CHARGE
        (control/'attempts').mkdir(exist_ok=True)
        original_charge = rb._confirm_charge
        from .verify_perseed import mode, ALL8
        if mode(config.get('verify_mode')) == ALL8:
            (control/'attempts_all8').mkdir(exist_ok=True)
            while time.time() < config['deadline_unix'] and not (control/'STOP.json').exists():
                run_all8_round(root, control, config, immutable)
                time.sleep(config['poll_s'])
            return
        while time.time() < config['deadline_unix'] and not (control/'STOP.json').exists():
            originals = bs.legacy(root, 'B')
            done = bs.confirmations(root, 'B')
            decisions = rb.bank_decisions(root, 'B', sources=config['sources'])
            for r in decisions:
                vid = r['verification_id']
                if r['status'] != 'accepted' or not (r['grade_veh_s'] or 0) > 0:
                    continue
                if vid in done and done[vid].get('safety_rule') == bs.RULE:
                    continue
                attempt = control/'attempts'/(vid+'.json')
                claim = control/'attempts'/(vid+'.claim.json')
                if attempt.exists():
                    continue
                if claim.exists():
                    immutable(attempt, dict(verification_id=vid, status='interrupted',
                        reason='existing claim without completion; never replay', unix=time.time()))
                    continue
                snapshot = rb._decision_snapshot(root, 'B', r, {})
                jobs = quoted_jobs(snapshot, r['chosen_plan'], sv.build_branches, rb.solo_candidate)
                if vid not in originals:
                    try:
                        accel_storage.reserve(root, jobs, PHASE)
                    except ValueError as error:
                        alert = control/f'BUDGET_WAIT_{time.time_ns()}.json'
                        immutable(alert, dict(unix=time.time(), verification_id=vid,
                            required_sim_s=sum(j['sim_s'] for j in jobs), error=repr(error),
                            new_sumo_sim_s=0, action='wait before claim; no partial confirmation'))
                        break
                immutable(claim, dict(unix=time.time(), verification_id=vid, jobs=jobs,
                    precharged_sim_s=sum(j['sim_s'] for j in jobs),
                    conditional_unused_solo_charges_retained=True))
                charges = {j['id']: j for j in ledger(root)}
                rb._confirm_charge = lambda run_root, phase, kind: precharged_charge(charges, kind)
                try:
                    receipt = {k: v for k, v in r.items() if not k.startswith('_')}
                    def runner_for(work):
                        return sv.SubprocessBranchRunner(Path(work)/'branches', workers=2,
                            timeout_s=config['branch_timeout_s'])
                    if vid not in originals:
                        rb.confirm(root, 'B', receipt, snapshot, runner_for, str(root),
                            PHASE, workers=2, version=r['_policy_version'])
                    original = bs.legacy(root, 'B')[vid]
                    out = bs.reconcile(root, 'B', original,
                        dict(snapshot=snapshot,first_seed_receipt=receipt), runner_for, PHASE)
                    result = dict(unix=time.time(), verification_id=vid, status='complete',
                        admitted=out['admitted'], confirmation=out,
                        new_precharge_sim_s=0 if vid in originals else sum(j['sim_s'] for j in jobs),
                        precharged_sim_s=sum(j['sim_s'] for j in jobs))
                except Exception as error:
                    result = dict(unix=time.time(), verification_id=vid, status='failed',
                        error=repr(error), traceback=traceback.format_exc(),
                        precharged_sim_s=sum(j['sim_s'] for j in jobs), replay=False)
                finally:
                    rb._confirm_charge = original_charge
                immutable(attempt, result)
                quality = bs.quality(root, 'B', sources=config['sources'])
                print(json.dumps(dict(result=result, bank_quality=quality)), flush=True)
                initial = set(config['initial_verification_ids'])
                settled = {vid for vid,c in bs.confirmations(root,'B').items()
                    if c.get('safety_rule') == bs.RULE}
                if initial <= settled and not (control/'FIRST_BATCH_COMPLETE.json').exists():
                    immutable(control/'FIRST_BATCH_COMPLETE.json', dict(unix=time.time(),
                        initial_verification_ids=sorted(initial), bank_quality=quality))
            bs.quality(root, 'B', sources=config['sources'])
            time.sleep(config['poll_s'])


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
