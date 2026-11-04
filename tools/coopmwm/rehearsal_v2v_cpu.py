"""PART 5 end-to-end CPU rehearsal of the V2V intent-sharing arms (branch c4/v2v-rehearsal).

Runs the REAL orchestration (warm_runner.run_arm, accel_runner.wave, replay_bank
collect/verify/confirm, exact grading, lesson check, capacity broker, GPU RPC
spool) for THREE arms A-v2v, B-v2v and R-v2v - each resuming a u3-like
checkpoint with ``v2v=True`` in its binding, following the same recipes as the
registered arms (A plain PPO; B with B-bank imitation; R with R-bank
imitation + fairness cap) - for TWO updates (u4, u5). CPU only: the fake
SUMO plant / GPU spool / WM evaluator of rehearsal_cpu are the only leaves.

While c4/v2v-int has not published V2V_INT_READY.json, the v2v observation and
model hooks are LOCAL STUBS with the integrated interface
(v2v_obs.V2V_DIM/v2v_features, v2v_model.extend_for_v2v) plus the flag-gated
bank wiring (bank_v2v, default off; byte-identical when off). When the
integrated branch is READY this rehearsal is re-run on the real code.

Checks rehearsed (V2V_REHEARSAL_READY.json):
- all three arms complete 2 updates with no STOP and no arm failure;
- bank imitation pairs carry v2v features (offline bank_pairs re-read of the
  finished root: every kept pair sample['context']['v2v'] present, V2V_DIM wide);
- the lesson check and the isolated policy-synergy check run with v2v
  observations (receipts carry v2v.enabled and their new-policy branches run
  under bank_v2v.enabled);
- budget charges per arm per update are reported.
"""
import copy
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .rehearsal_cpu import (AMENDMENT, BOUNDS, DEADLINE, SLOTS, _child_environment, _proposer_search,
                            _seed_check_fixtures, bank_state, event, make_remote_class, progress_signature,
                            tiny_proposer, FakeGPU, EVENTS, _seal)

ARMS = ('A', 'B', 'R')
FIRST_UPDATE, LAST_UPDATE = 4, 5   # u3-like checkpoints -> exactly 2 updates each

DEFAULTS = dict(k=8, N_exec=10, slots=4, workers=6, episodes_per_update=6, last_update=LAST_UPDATE,
                latency_s=1., stall_s=900., timeout_s=9000., transport_errors=0, malformed=0, branch_crash=0,
                die_slot=None, die_after=1, decline_withdrawn=True, parallel_groups=True,
                fanout_factor=2, r_concurrent=True, verify_workers=6,
                b_replay=True, collect_episodes=1, bank_warmup_rounds=2, verify_per_round=3,
                bank_admission_rounds=8)


# ============================================================== live-like root (three arms)

def build_root(root, cfg):
    from .accel_learner import save
    from .acceleration import Policy
    from .accel_protocol import worlds
    from .storage import write
    from . import search_learning
    from .warm_protocol import SEED
    root.mkdir(parents=True)
    for d in ('requests', 'responses', 'centre_requests', 'centre_responses', 'jobs'):
        (root/d).mkdir()
    (root/'AUTHORITY.md').write_text('V2V intent-sharing CPU rehearsal authority (temporary root)\n'+AMENDMENT+'\n')
    train = worlds('train')
    n = cfg['episodes_per_update']
    plans = {u: [train[(u*7+i*5) % len(train)] for i in range(n)] for u in (FIRST_UPDATE, LAST_UPDATE)}
    for u in plans:
        seen = set();out = []
        for w in plans[u]+train:
            if w['seed'] not in seen and len(out) < n:
                seen.add(w['seed']);out.append(w)
        plans[u] = out
    torch.manual_seed(SEED)
    for arm in ARMS:   # u3-like checkpoints for all three arms
        policy = Policy();opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
        rng = torch.Generator().manual_seed(SEED)
        save(root/f'{arm}_checkpoint_003.pt', policy, opt, rng, 3, 0)
    budget = dict(total=13_600_000, phases=dict(superseded=811640, train=3_000_000, checks=500_000, dev=500_000,
                                                sumo_verification=1_000_000, sumo_verification_k8=1_500_000,
                                                bank_verification_ext=2_600_000,
                                                exact_plan_grading=500_000, discovery=138_200))
    budget['allocated'] = sum(budget['phases'].values());budget['unallocated'] = budget['total']-budget['allocated']
    budget.update(updates=dict(A=48, B=24, R=24))
    write(root/'RESOURCE_EXTENSION_1651.json', dict(amendment_sha256=AMENDMENT, gpu_deadline=DEADLINE,
                                                    budget=budget, harness=True))
    (root/'DNS_GUIDED_RELEASE_1651.json').write_text(json.dumps(dict(harness=True)))
    write(root/'WORKERS.json', dict(workers=[dict(slot=s) for s in SLOTS]))
    write(root/'WARM_REMOTE_BINDING.json', dict(endpoints=[dict(slot=s) for s in SLOTS], deadline_unix=DEADLINE,
                                                search_sha256='harness'))
    write(root/'BROKER_CONFIG.json', dict(amendment_sha256=AMENDMENT, deadline_unix=DEADLINE,
                                          source_root=str(Path(__file__).resolve().parents[2]),
                                          receipt_tag='_H3', decline_withdrawn=cfg.get('decline_withdrawn', True)))
    for d in (root/'exact_grades', root/'sumo_verification'):
        d.mkdir(parents=True, exist_ok=True)
    smoke = _seal(dict(pass_=True, all_five_branches_graded=True, harness_fixture=True), 'smoke_sha256')
    (root/'sumo_verification'/'SMOKE_PASSED.json').write_text(json.dumps(smoke))
    (root/'exact_grades'/'SMOKE_PASSED.json').write_text(json.dumps(dict(harness_fixture=True)))
    # Live A reporting fixtures (already published by the completed live A u16
    # checkpoint); mirrored here so the rehearsal resume does not re-run them.
    (root/'A_CHECK_016.json').write_text(json.dumps(dict(harness=True)))
    (root/'A_DEV_016.json').write_text(json.dumps(dict(harness=True)))
    learned = search_learning.empty('B', SEED, BOUNDS)
    progress = {arm: dict(running=[], beta=1., learned=learned,
                          exact_rng_state=np.random.default_rng(SEED).bit_generator.state, exact_count=12)
                for arm in ARMS}
    return dict(plans=plans, budget=budget, progress=progress)


def install_runtime(root, cfg, layout):
    """rehearsal_cpu.install_runtime (all leaf/platform patches), then a v2v binding verify()."""
    from . import warm_runner as wr
    from .warm_protocol import install_accounting
    rc_install = _rc_install_runtime()
    rc_install(root, cfg, layout)

    def verify(r, a):
        install_accounting(r, layout['budget'])
        k = cfg['k']
        b = dict(runtime='fake-runtime', checkpoint='fake-checkpoint', accel_bounds=BOUNDS,
                 gpu_deadline=DEADLINE, N_exec=cfg['N_exec'], workers=cfg['workers'], warm_history=False,
                 health_guards=True, safety_reference=dict(by_world={}, unmatched_zero_event_benchmark=True,
                                                           source='harness'),
                 budget=layout['budget'], awr=dict(coefficient=1.), resume_update=3,
                 A_resume_update=3, v2v=True,
                 resume_progress=copy.deepcopy(layout['progress'][a]), runtime_receipt_suffix='_H3',
                 quota_aware_admission=True, parallel_group_queries=a == 'B' and cfg.get('parallel_groups', True),
                 precharged_unattempted=[], stop_after_update=cfg['last_update'],
                 sumo_verify=dict(root=str(r/'sumo_verification'), run_root=str(r),
                                  phase='sumo_verification_k8' if k == 8 else 'sumo_verification',
                                  k=k, workers=cfg.get('verify_workers') or k+1, timeout_s=3600, guide_beta=1.))
        if cfg.get('b_replay'):
            b.update(b_replay=True, replay_beta=1., slots=cfg['slots'])
        if cfg.get('fanout_factor'):
            b['parallel_fanout_factor'] = cfg['fanout_factor']
        if cfg.get('r_concurrent') and a == 'R':
            b['r_concurrent_lag1_quota'] = True
        return b
    wr.verify = verify


def _rc_install_runtime():
    from . import rehearsal_cpu as rc
    return rc.install_runtime


# ============================================================== scenario run

def run_scenario(name, cfg, workdir):
    from . import capacity_broker
    from . import rehearsal_cpu as rc
    EVENTS.clear()
    cfg = dict(DEFAULTS, **cfg)
    base = Path(workdir)/name
    if base.exists():
        shutil.rmtree(base)
    root = base/'run'
    _child_environment(base, cfg)
    layout = build_root(root, cfg)
    install_runtime(root, cfg, layout)
    _seed_check_fixtures(root)
    # Arm A runs its own registered check at u4 and WRITES A_CHECK_RECORDS_004.json
    # (storage.write is exclusive-create); the pre-seeded rp4 fixture for B's AB panel
    # (unused here: replay B never runs check) would collide -> fault 7 fix.
    (root/'A_CHECK_RECORDS_004.json').unlink()
    proposer = tiny_proposer(base)
    from . import mppi_search
    if not hasattr(mppi_search.search, 'harness_original'):
        mppi_search.search = _proposer_search(proposer)
    if cfg.get('r_concurrent'):
        from .storage import write
        write(root/'B_RANDOM_QUOTA_003.json', dict(sampling={'merge-dense': .95, 'ordinary': .95},
              tickets=[dict(regime='ordinary', duration_s=d, group_size=n, id=i)
                       for i, (d, n) in enumerate([(60, 2), (120, 3), (180, 2), (300, 4), (90, 2), (240, 3),
                                                   (60, 5), (150, 2), (120, 6), (30, 2)])]))
    gpus = {s: FakeGPU(s, cfg) for s in SLOTS[:cfg['slots']]}
    with (root/'WARM_REMOTE_BINDING.json').open('w') as f:
        json.dump(dict(endpoints=[dict(slot=s) for s in gpus], deadline_unix=DEADLINE, search_sha256='harness'), f)
    (root/'WORKERS.json').unlink();(root/'WORKERS.json').write_text(json.dumps(dict(workers=[dict(slot=s) for s in gpus])))
    capacity_broker.Remote = make_remote_class(gpus, cfg)
    from . import warm_runner as wr
    outcome = {}
    started = time.time()

    def arm_thread(arm):
        try:
            wr.run_arm(root, arm);outcome[arm] = 'complete'
        except BaseException as error:
            outcome[arm] = 'failed: '+repr(error)
            event('arm_failed', arm=arm, error=repr(error), tb=traceback.format_exc())

    def broker_thread():
        try:
            capacity_broker.run(root, root/'BROKER_CONFIG.json');outcome['broker'] = 'exited'
        except BaseException as error:
            outcome['broker'] = 'stopped: '+repr(error)[:400]
    threads = {a: threading.Thread(target=arm_thread, args=(a,), daemon=True, name='arm-'+a) for a in ARMS}
    bt = threading.Thread(target=broker_thread, daemon=True, name='broker');bt.start()
    collectors = {}
    if cfg.get('b_replay'):
        from . import replay_bank
        from .accel_protocol import worlds
        train = [w for w in worlds('train') if w not in sum(layout['plans'].values(), [])]
        done = threading.Event()

        def collector(arm, rounds):
            try:
                b = wr.verify(root, arm)
                st = replay_bank.collect(root, arm, b, train[(0 if arm == 'B' else 7):], lambda: done.is_set(),
                                         episodes_per_round=cfg['collect_episodes'],
                                         verify_workers=cfg.get('verify_workers'), rounds=rounds,
                                         verify_per_round=cfg.get('verify_per_round'))
                outcome.setdefault('collector_'+arm, []).append(dict(rounds=rounds, statuses=st))
            except BaseException as error:
                outcome.setdefault('collector_'+arm, []).append('failed: '+repr(error))
                (Path(base)/f'collector_{arm}.tb.txt').write_text(traceback.format_exc())
                event('collector_failed', arm=arm, error=repr(error), tb=traceback.format_exc())
        # Warmup (both banks in parallel, rp4 width) before the arms start, then
        # admission top-up rounds: the arms imitate whatever their bank holds at u4,
        # so the rehearsal waits (budget-bounded rounds, no thresholds) until each
        # bank holds at least one ADMITTED entry - the live B-REPLAY start has the
        # same preflight (bank readiness before the resumed updates).
        def admitted_count(arm):
            from .bank_safety import confirmations as amended_conf
            return sum(1 for c in amended_conf(root, arm).values() if c.get('admitted'))
        warm = [threading.Thread(target=collector, args=(a, cfg['bank_warmup_rounds']), daemon=True) for a in ('B', 'R')]
        for t in warm:
            t.start()
        for t in warm:
            t.join()
        for _extra in range(int(cfg.get('bank_admission_rounds', 8))):
            lacking = [a for a in ('B', 'R') if admitted_count(a) < 1]
            if not lacking:
                break
            event('bank_admission_topup', round=_extra, lacking=lacking)
            ts = [threading.Thread(target=collector, args=(a, 1), daemon=True) for a in lacking]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        event('bank_warmup_done', bank=bank_state(root),
              admitted={a: admitted_count(a) for a in ('B', 'R')})
        collectors = {a: threading.Thread(target=collector, args=(a, None), daemon=True) for a in ('B', 'R')}
        for t in collectors.values():
            t.start()
        started = time.time()
    # Arms run SEQUENTIALLY (A, then B, then R): three concurrent arm waves plus two
    # collector pools overloaded this host (BrokenProcessPool, the same environment
    # artifact the harden-int rehearsal recorded for b_slow_search_overload). The
    # recipes are unchanged; only the schedule is serialized.
    deadlock = False
    for a in ARMS:
        last, last_change = None, time.time()
        threads[a].start()
        while threads[a].is_alive():
            time.sleep(1.)
            snap = progress_signature(root)
            if snap != last:
                last, last_change = snap, time.time()
            elif time.time()-last_change > cfg['stall_s']:
                deadlock = True
                event('harness_stall', arm=a, idle_s=time.time()-last_change, signature=snap)
                break
            if time.time()-started > cfg['timeout_s']:
                event('harness_timeout', arm=a);break
        if deadlock or outcome.get(a) != 'complete':
            event('harness_aborted', arm=a, outcome=outcome.get(a), deadlock=deadlock)
            break
    wall = time.time()-started
    if collectors:
        done.set()
        for t in collectors.values():
            t.join(600)
    state = diagnose(root, cfg, gpus)
    state['bank'] = bank_state(root)
    # PAUSE_HARNESS_END is NOT written here: it is a registered health pause
    # (warm_health.assert_running raises on any unreleased PAUSE_*.json), so the
    # closing checklist (lesson/synergy branches) must run BEFORE it is registered
    # - main() writes it after verify_checklist (fault 12). The broker thread is
    # left running as a daemon until that write / os._exit.
    result = dict(scenario=name, config={k: v for k, v in cfg.items()}, outcome=outcome, deadlock_or_idle=deadlock,
                  wall_s=round(wall, 1), root=str(root), **state,
                  events=[{k: v for k, v in e.items() if k != 'tb'} for e in EVENTS][-40:],
                  failures=[e for e in EVENTS if e['kind'] in ('arm_failed', 'collector_failed', 'gpu_worker_exception',
                                                               'harness_stall', 'harness_timeout', 'harness_aborted')][:6])
    with (Path(base)/'result_debug.json').open('w') as f:
        json.dump(dict(result, events_full=[{k: v for k, v in e.items() if k != 'tb'} for e in EVENTS]), f, indent=1, default=str)
    return result


def diagnose(root, cfg, gpus):
    from .storage import ledger
    from .sumo_verify import quote
    stops = sorted(p.name for p in root.glob('*STOP*.json'))
    pauses = sorted(p.name for p in root.glob('PAUSE_*.json'))
    verifs = []
    for p in sorted((root/'sumo_verification'/'verifications').glob('*.json')):
        v = json.loads(p.read_text())
        verifs.append(dict(id=v['verification_id'], arm=v['episode']['arm'], update=v['episode']['update'],
                           status=v['status'], k=v['k'], t0=v['t0'], end=v['window']['end_s'],
                           charged=v['charged_sim_s'], quote_k=quote(v['t0'], v['k']),
                           matches_k_plus_1=v['charged_sim_s'] in (0, (v['k']+1)*v['window']['end_s'])))
    charges = ledger(root)
    phases = {}
    for c in charges:
        phases[c['phase']] = phases.get(c['phase'], 0)+c['sim_s']
    updates = {}
    for arm in ARMS:
        started = json.loads((root/f'WARM_{arm}_STARTED_H3.json').read_text()) if \
            (root/f'WARM_{arm}_STARTED_H3.json').exists() else {}
        cumulative = float(started.get('charged_sim_s', 0.))
        for u in (FIRST_UPDATE, LAST_UPDATE):
            p = root/f'{arm}_UPDATE_{u:03}.json'
            if not p.exists():
                continue
            v = json.loads(p.read_text())
            results = 0.
            for e in (root/'jobs').glob(f'warm_u{u:03}_{arm}_*/RESULT.json'):
                results += json.loads(e.read_text()).get('charged_sim_s', 0.)
            updates[f'{arm}{u}'] = dict(executed=v['proposal_routing']['executed'],
                                        queries=v['proposal_routing']['queries'],
                                        accepted=v['proposal_routing']['accepted'],
                                        checkpoint=(root/f'{arm}_checkpoint_{u:03}.pt').exists(),
                                        graded=v['trust'].get('graded'), wall_s=round(v['wall_s'], 1),
                                        bank=v['trust'].get('bank'),
                                        v2v_summary=(v.get('exact_imitation') or {}).get('v2v'),
                                        imitation_steps=sum(m.get('exact_imitation_steps', 0) for m in v['ppo']),
                                        ppo_steps=len(v['ppo']),
                                        train_charged_sim_s=round(results, 1),
                                        cumulative_charged_sim_s=v['charged_sim_s'],
                                        delta_charged_since_start=round(v['charged_sim_s']-cumulative, 1),
                                        lesson=(json.loads((root/f'LESSON_CHECK_u{u:03}_{arm}.json').read_text()
                                                           ).get('v2v') if arm in ('B', 'R') and
                                                (root/f'LESSON_CHECK_u{u:03}_{arm}.json').exists() else None),
                                        verifications=sum(1 for x in verifs if x['arm'] == arm and x['update'] == u))
    req = sum(1 for _ in root.glob('requests/*.json'))
    resp = sum(1 for _ in root.glob('responses/*.json'))
    return dict(stops=stops, pauses=pauses, updates=updates, plan_requests=req, plan_responses=resp,
                unanswered_requests=req-resp,
                verifications=dict(n=len(verifs), by_status={s: sum(1 for v in verifs if v['status'] == s)
                                                             for s in {v['status'] for v in verifs}},
                                   all_charges_equal_k_plus_1_branches=all(v['matches_k_plus_1'] for v in verifs),
                                   all_within_quote=all(v['charged'] <= v['quote_k'] for v in verifs),
                                   charged_sim_s=sum(v['charged'] for v in verifs),
                                   ledger_verification_sim_s=sum(c['sim_s'] for c in charges
                                                                 if c['phase'].startswith('sumo_verification'))),
                charges_by_phase=phases, ledger_entries=len(charges),
                gpu=dict({s: dict(alive=g.alive, searches=g.searches, failure=g.failure)
                          for s, g in gpus.items()}))


# ============================================================== checklist + receipts

def _harness_owner(pid=None):
    import os as _os
    import time as _time
    return dict(pid=pid or _os.getpid(), start_ticks='1', boot_id='harness', unix=_time.time())


def run_synergy_checks(root, budget, arms=('B',), update=FIRST_UPDATE):
    """Non-isolated then ISOLATED policy-synergy check with v2v=True (real code path).

    The isolated variant runs from its OWN control root: run_update's one-use
    CLAIM guard is per control dir (live ops run the two variants separately)."""
    from . import policy_synergy as ps
    from . import replay_start
    replay_start.owner = _harness_owner   # Windows rehearsal: /proc identity receipts are faked
    cfg = dict(control_root=str(Path(root)/'control'), collection_root=str(Path(root)/'ls_collect_fixture'),
               v2v=True, budget=dict(budget, total=14_500_000))   # accounting() registered total assertion
    # Fault 12: PAUSE_HARNESS_END.json is the rehearsal's own end marker (no live
    # equivalent), but assert_running treats any unreleased PAUSE_*.json as a
    # health pause - every synergy branch died on it. Hold the marker while the
    # check's branches run (live ops run this check while the arm is live, before
    # any pause), then restore it.
    hold = Path(root)/'PAUSE_HARNESS_END.json'
    held = None
    if hold.exists():
        held = Path(root)/'HELD_PAUSE_HARNESS_END.json'
        hold.rename(held)
    try:
        out = {}
        for arm in arms:
            try:
                base = ps.run_update(root, arm, update, cfg)
                isolated = ps.run_update(root, arm, update,
                                         dict(cfg, isolated=True,
                                              control_root=str(Path(root)/'control_isolated')))
                out[arm] = dict(status=base['status'], entries=len(base['entries']),
                                cooperative_entries=base['admitted_cooperative_entries'],
                                v2v=base.get('v2v'), isolated_status=isolated['status'],
                                isolated_entries=len(isolated['entries']), isolated_v2v=isolated.get('v2v'),
                                gains=[e.get('policy_cooperation_gain_veh_s') for e in base['entries']])
            except BaseException as error:
                out[arm] = dict(error=repr(error), traceback=traceback.format_exc()[-800:])
    finally:
        if held is not None and held.exists():
            held.rename(hold)
    return out


def verify_checklist(root, layout_budget, arms=ARMS):
    """The rehearsal acceptance checks; returns (checks, faults, ready)."""
    from . import bank_v2v
    from . import policy_synergy as ps
    from . import replay_start
    from . import replay_bank as rb
    from .exact_grade import verify_sealed
    from .warm_protocol import SEED
    replay_start.owner = _harness_owner
    checks, faults = {}, []
    # Fault 12 fix: the scenario-end PAUSE_HARNESS_END.json (harness-stop control
    # file for the broker/collectors) is still registered when the post-run
    # policy-synergy check executes, and warm_health refuses every branch run
    # while a pause receipt exists - both variants failed with HealthPause. By
    # synergy time broker and collectors have already exited (outcome 'stopped:
    # HealthPause' / done.set()), so the control file has no remaining role and
    # is removed before the checks.
    pause = Path(root)/'PAUSE_HARNESS_END.json'
    if pause.exists():
        pause.unlink()
    stops = sorted(p.name for p in Path(root).glob('*STOP*.json'))
    failed = sorted(p.name for p in Path(root).glob('WARM_*_FAILED*.json'))
    updates_ok = all((Path(root)/f'{a}_UPDATE_{LAST_UPDATE:03}.json').exists() and
                     (Path(root)/f'{a}_checkpoint_{LAST_UPDATE:03}.pt').exists() and
                     (Path(root)/f'WARM_{a}_PAUSED.json').exists() for a in arms)
    checks['arms_complete_two_updates'] = dict(stops=stops, failed=failed, updates_ok=updates_ok,
                                               ok=not stops and not failed and updates_ok)
    pair_report = {}
    for arm in ('B', 'R'):
        pairs, weights = rb.bank_pairs(Path(root), arm, SEED, v2v=True)
        missing = 0;dims = set()
        for s, w in pairs:
            block = (s.get('context') or {}).get('v2v')
            if not block:
                missing += 1
            else:
                dims |= {len(v) for v in block.values()}
        pair_report[arm] = dict(pairs=len(pairs), pairs_missing_v2v=missing, v2v_dims=sorted(dims),
                                stub=weights.get('v2v', {}).get('stub'),
                                v2v_counts=(weights.get('v2v') or {}).get('counts'),
                                entries=weights['counts'])
        if missing or not pairs or dims-{int(bank_v2v.V2V_DIM)}:
            faults.append(f'{arm}: bank pairs v2v check failed (pairs={len(pairs)} missing={missing} dims={sorted(dims)})')
    checks['bank_pairs_carry_v2v'] = pair_report
    lessons = {}
    for arm in ('B', 'R'):
        for u in (FIRST_UPDATE, LAST_UPDATE):
            p = Path(root)/f'LESSON_CHECK_u{u:03}_{arm}.json'
            if not p.exists():
                faults.append(f'{arm} u{u}: LESSON_CHECK receipt missing')
                continue
            v = json.loads(p.read_text())
            try:
                verify_sealed(v, 'lesson_check_sha256')
            except Exception as error:
                faults.append(f'{arm} u{u}: lesson receipt seal {error!r}')
            lessons[f'{arm}{u}'] = dict(v2v=v.get('v2v'), list_size=v.get('list_size'),
                                        graded_entries=v.get('graded_entries'), error=v.get('error'),
                                        recapture_mean=v.get('recapture_mean'))
    checks['lesson_checks_v2v'] = lessons
    synergy = run_synergy_checks(root, layout_budget, arms=('B', 'R'))
    for arm, row in synergy.items():
        if row.get('status') != 'complete' or row.get('isolated_status') != 'complete':
            faults.append(f'synergy {arm}: {row.get("status")}/{row.get("isolated_status")} {row.get("error", "")}')
        elif not (row.get('v2v') or {}).get('enabled') or not (row.get('isolated_v2v') or {}).get('enabled'):
            faults.append(f'synergy {arm}: v2v receipt missing')
    checks['policy_synergy_isolated_v2v'] = synergy
    ready = not faults and checks['arms_complete_two_updates']['ok']
    return checks, faults, ready


def build_receipt(root, state, checks, faults, ready, wall_s=None):
    return dict(
        part='PART 5 (REHEARSAL) V2V intent-sharing arms A/B/R, CPU end-to-end',
        commit=_commit(),
        branch='c4/v2v-rehearsal',
        integrated_branch_ready=_integrated_ready(),
        hooks=dict(obs='tools/coopmwm/v2v_obs.py (REAL, merged from c4/v2v-int; V2V_DIM=24, v2v_features/V2VConfig/v2v_slot_block)',
                   model='tools/coopmwm/v2v_model.py (REAL, merged from c4/v2v-int; extend_for_v2v/load_checkpoint/forward_with_v2v '
                         '+ rehearsal glue bind_forward/load_for/fit_policy appended on the real module)',
                   wiring='tools/coopmwm/bank_v2v.py (REAL, merged from c4/v2v-int; _frame_table/_obs_cfg adapter to the real '
                          'v2v_obs contract, plain-float blocks; flag v2v default off)',
                   integrate='tools/coopmwm/v2v_integrate.py (REAL widen_context/policy_call/WidenedPolicy; the rehearsal '
                             'uses the in-model bind_forward path instead - contexts are never pre-widened here)',
                   stubs_removed='the three local stub files were replaced by the merged real modules at merge commit '
                                 '28a86126; no stub code remains on the rehearsal path'),
        scenario_table=_scenario_table(state),
        charges_per_update_per_arm={k: v for k, v in state['updates'].items()},
        charges_by_phase=state['charges_by_phase'],
        outcome=state.get('outcome', {}),
        harness_events=state.get('failures', []),
        faults_found_and_fixed=_FAULTS,
        faults_open=faults,
        ready=ready,
        checks=checks,
        wall_s=wall_s,
        unix=time.time())


def evaluate_root(root):
    """Checklist + receipts on a finished run root (scenario already executed)."""
    from .rehearsal_cpu import bank_state
    root = Path(root)
    cfg = dict(DEFAULTS)
    state = diagnose(root, cfg, gpus={})
    state['bank'] = bank_state(root)
    checks, faults, ready = verify_checklist(root, _layout_budget(root), arms=ARMS)
    return build_receipt(root, state, checks, faults, ready)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workdir', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--evaluate', default=None, help='finished run root: run the checklist only')
    a = p.parse_args(argv)
    if a.evaluate:
        receipt = evaluate_root(a.evaluate)
        text = json.dumps(receipt, indent=1, default=str)
        if a.out:
            Path(a.out).write_text(text)
        print(text)
        sys.stdout.flush()
        return 0 if receipt['ready'] else 2
    work = a.workdir or os.path.join(os.environ.get('TEMP', tempfile_root()), 'v2v_rehearsal')
    result = run_scenario('v2v_arms', {}, work)
    root = result['root']
    checks, faults, ready = verify_checklist(root, _layout_budget(root))
    result['checks'] = checks;result['faults'] = faults;result['ready'] = ready
    # Register the harness-end health pause only AFTER the checklist (its branch
    # runners raise HealthPause on any registered pause; fault 12). Give the
    # broker thread its poll interval to record the guided exit.
    (Path(root)/'PAUSE_HARNESS_END.json').write_text(json.dumps(dict(harness=True)))
    time.sleep(12.)
    receipt = build_receipt(root, result, checks, faults, ready, wall_s=result['wall_s'])
    text = json.dumps(receipt, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    import multiprocessing
    for child in multiprocessing.active_children():
        child.terminate()
    sys.stdout.flush();os._exit(0)


def tempfile_root():
    import tempfile
    return tempfile.gettempdir()


def _commit():
    import subprocess
    return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[2])).stdout.strip()


def _integrated_ready():
    return Path('E:/agent-work/c4-v2v-int/V2V_INT_READY.json').exists()


def _layout_budget(root):
    from .storage import read
    return read(Path(root)/'RESOURCE_EXTENSION_1651.json')['budget']


def _scenario_table(state):
    rows = []
    for arm in ARMS:
        for u in (FIRST_UPDATE, LAST_UPDATE):
            v = state['updates'].get(f'{arm}{u}')
            rows.append(dict(arm=arm+'-v2v', update=u,
                             status='complete' if v else 'missing',
                             executed=v.get('executed') if v else None,
                             queries=v.get('queries') if v else None,
                             accepted=v.get('accepted') if v else None,
                             ppo_steps=v.get('ppo_steps') if v else None,
                             imitation_steps=v.get('imitation_steps') if v else None,
                             checkpoint=v.get('checkpoint') if v else None,
                             wall_s=v.get('wall_s') if v else None))
    rows.append(dict(arm='broker', status=(state.get('outcome') or {}).get('broker', 'exited')))
    return rows


_FAULTS = [
    'fault 1: warm_runner.run_arm arm-A resume ignored stop_after_update (hard-coded last=48); '
    'fixed to min(48, stop_after_update default 48) - identical behaviour when the key is absent',
    'fault 2: accel_plant.episode loaded every snapshot into a plain Policy(); v2v-widened weights '
    'would crash - fixed with v2v_model.fit_policy (shape-driven, no-op for base widths)',
    'fault 3: replay_bank.collect reloaded the newest checkpoint with the base loader - collectors '
    'would crash on v2v-widened checkpoints - fixed with v2v_model.load_for',
    'fault 4: rehearsal fake SUMO trace rows lacked the real backend issued_acceleration_mps2/'
    'command_present slots, so bank_v2v pair blocks would have been excluded as missing partner '
    'commands - fake_tick now records the issued command per CAV row (harness parity fix)',
    'fault 5: capacity_broker.forecast_result rejected centre-only forecasts whose record_sha256 was '
    'computed over the prepared (mode-stripped) record - every R centre rollout was declined '
    '("centre unavailable", 322 skips, R bank empty); fixed with the same two-representation centre '
    'identity rule the bank reader already applies (MAIN bank part 3 centre_identity)',
    'fault 6: arm A crashed with FileExistsError writing A_CHECK_RECORDS_004.json (its registered u4 '
    'check output; storage.write is exclusive-create) because the rp4 fixture pre-seeded the same '
    'path (only read by replay-B, which never runs check) - the rehearsal no longer seeds it',
    'fault 7 (rehearsal shims, disclosed): policy_synergy accounting() registered-total assertion '
    'needs budget.total=14_500_000 and replay_start.owner needs a /proc identity - the rehearsal '
    'passes a mirrored budget and a harness owner receipt (same pattern as FakeGPU.owner); the '
    'isolated synergy variant runs from its own control root (one-use CLAIM guard is per control dir)',
    'fault 8: with three concurrent arm waves plus two collector pools the host re-created the '
    'hard-en-int overload artifact (BrokenProcessPool kills a spawn worker -> WARM_B_FAILED -> '
    'guided STOP; rp4 recorded the same for b_slow_search_overload and re-ran alone) - the '
    'rehearsal now runs the arms SEQUENTIALLY (A, then B, then R; recipes unchanged, schedule '
    'serialized) and keeps the rp4-proven collector width',
    'fault 9: replay_bank.verify_pending took its "already decided" set from the caller\'s memory '
    'only, so any FRESH collect() on the same bank (rehearsal warmup -> background handoff, or a '
    'live collector restart) re-verified already-verified states and crashed on the exclusive '
    'SNAPSHOT.pt create (FileExistsError 17) - verified pending now seeds its decided set from the '
    'durable decisions/ + skipped/ state markers first',
    'fault 10: the rehearsal stall watchdog mis-fired on healthy silent phases (arm PPO updates are '
    'minutes of pure CPU with no file writes) and its os._exit killed a mid-update arm without any '
    'failure receipt - watchdog raised to 900 s and the debug dump keeps full tracebacks',
    'fault 11: bank_v2v v2v blocks were numpy float32 scalars (v2v_features returns a float32 '
    'ndarray); injected into branch samples they reach the episode RESULT tape and canonical JSON '
    'serialization raises TypeError - EVERY lesson and policy-synergy new-policy branch failed '
    '("Object of type float32 is not JSON serializable") - the wiring now emits plain python floats '
    '(superseded by the merged real bank_v2v adapter, which keeps the plain-float output)',
    'fault 12: the scenario-end PAUSE_HARNESS_END.json control file (stops broker/collectors) is '
    'still registered when the post-run policy-synergy check executes; warm_health refuses every '
    'branch run while a pause receipt exists, so both synergy variants failed with HealthPause and '
    'status incomplete (gains null) even though their entries and v2v receipts were correct - '
    'run_synergy_checks now HOLDS the marker (rename) for the duration of the synergy branch runs '
    'and restores it afterwards (live ops run this check while the arm is live, before any pause; '
    'no live-code change, and the end marker stays on disk for the record)',
    'fault 13 (host capacity, resume run 2026-11-04): with co-tenant sessions the host runs at ~7 GB '
    'free RAM; 12-wide spawn wave pools (each worker imports torch) had workers die abruptly '
    '(BrokenProcessPool mid B-u5, native ucrtbase abort 0xc0000409 in another attempt) - the wave '
    'width is lowered to 6. Host capacity knob only: same jobs, same seeds, same recipes; '
    'episode/verification results are per-episode deterministic so receipts are unaffected',
]

if __name__ == '__main__':
    main()
