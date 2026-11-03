"""CPU tests: practice_starts (owner 2026-11-04 proposal, DEFAULT OFF).

OFF is byte-identical (launch default, receipts without a practice block, wave path unchanged);
ON gives B/R a registered fraction of each update's PPO episodes as snapshot-restart episodes
from the arm's OWN admitted within-seed cooperative bank examples (same snapshot and CRN seed
as the example, current policy from t0, normal horizon/reward/charge), uniformly sampled with a
per-example repeat cap, R falling back to its other admitted examples, A unchanged."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from tools.coopmwm import accel_runner
from tools.coopmwm import accel_storage
from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import practice_starts
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm import v2v_launch3 as launch
from tools.coopmwm import v2v_runner
from tools.coopmwm.protocol import sha
from tools.coopmwm.test_bank_boot_cpu import noisy
from tools.coopmwm.test_bank_perseed_cpu import MEMBERS, base, make, ps, win  # noqa: F401


# ---------------------------------------------------------------- fixtures

def fp(t, delay):
    return dict(t_s=t, metrics=dict(team_delay_vs=float(delay).hex(), arrived=float(t).hex()))


def base_delay(world, t):
    return 3.*world+0.5*t*t/100.


def snapshot(t0=400, world_seed=986000, gen_seed=1):
    """A collector-grade decision snapshot: recorded prefix samples/fingerprints and a CRN state."""
    from tools.coopmwm.accel_protocol import worlds
    gen = torch.Generator().manual_seed(gen_seed)
    state = gen.get_state().clone()
    samples = {}
    for t in range(300, t0+1):
        samples[str(t)] = dict(context=dict(members=list(MEMBERS)), action=[[5.], [5.]],
            behaviour=dict(route='policy', executed=False, sample_probability=0., q=0., regime='ordinary'),
            logp=0., value=0.)
    return dict(episode=dict(id=f'warm_u005_B_{world_seed}', world=dict(worlds('train')[0], seed=world_seed),
                             arm='B', update=5, role='train'),
                policy=dict(state_dict_sha256='p'),
                sumo=dict(runtime='rt', checkpoint='ck', warm_history=False,
                          accel_bounds=[-3., 2.6], noise_seed=11),
                t0=t0, end_s=1200,
                prefixes={str(t): fp(t, base_delay(world_seed, t)) for t in range(300, t0+1)},
                samples=samples, rng=dict(generator_state_sha256=hashlib.sha256(
                    state.numpy().tobytes()).hexdigest(), memory_after_t0={}),
                generator_state=state, committed_plans=[])


def bank_with_snapshots(tmp_path, arm='B', vids=('c1only', 'crep'), snapshot_seed=1):
    """A per-seed_v1 cooperative bank with own-bank decision links and decision snapshots."""
    for i, vid in enumerate(vids):
        make(tmp_path, vid, win(), win(), arm=arm, syn1=noisy(3., .5, 300+i),
             syn2=noisy(2., .5, 400+i))                                # both seeds cooperative
    bank = rb.bank_root(tmp_path, arm)
    snaps = {}
    for i, vid in enumerate(vids):
        state_id = f'state_{arm}_{vid}'
        snap = snapshot(world_seed=986000+i, gen_seed=snapshot_seed+i)
        (bank/'snapshots').mkdir(parents=True, exist_ok=True)
        torch.save(snap, bank/'snapshots'/(state_id+'.pt'))
        (bank/'decisions').mkdir(parents=True, exist_ok=True)
        (bank/'decisions'/(state_id+'.json')).write_text(json.dumps(
            dict(verification_id=vid, state_id=state_id)))
        snaps[vid] = dict(state_id=state_id, snapshot=snap)
    return snaps


# ---------------------------------------------------------------- off identical / key equality

def _cfg(tmp_path, **kw):
    cfg = v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64, **kw)
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    return cfg, live


def test_default_off_matches_launch_defaults_and_on_differs_only_in_the_flag(tmp_path):
    assert launch.BANK_OPTIONS['practice_starts'] is False
    assert launch.BANK_OPTIONS['practice_fraction'] == 0.25
    assert launch.BANK_OPTIONS['practice_repeat_cap'] == 2
    cfg, live = _cfg(tmp_path)
    assert cfg['practice_starts'] is False and cfg['practice_fraction'] == 0.25
    assert cfg['practice_repeat_cap'] == 2
    on, _ = _cfg(tmp_path, practice_starts=True)
    assert {k for k in cfg if cfg[k] != on[k]} == {'practice_starts'}
    assert on['practice_starts'] is True
    # a legacy runtime file without the keys still binds (OFF default), ON binds with B == R
    stripped = {k: v for k, v in live.items() if k not in launch.PRACTICE_KEYS}
    assert launch.bind_runtime(dict(config=stripped), root=tmp_path/'v2v', source_root=tmp_path/'src') is stripped
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    assert launch.bind_runtime(dict(config=dict(live, practice_starts=True)),
                               root=tmp_path/'v2v', source_root=tmp_path/'src')['practice_starts'] is True


def test_br_key_equality_and_registered_values(tmp_path):
    _, live = _cfg(tmp_path)
    ok = dict(live, practice_starts=True, practice_starts_by_arm=dict(
        B=dict(practice_starts=True, practice_fraction=0.25, practice_repeat_cap=2),
        R=dict(practice_starts=True, practice_fraction=0.25, practice_repeat_cap=2)))
    assert launch.bind_runtime(dict(config=ok), root=tmp_path/'v2v', source_root=tmp_path/'src') is ok
    bad = (dict(practice_fraction=.5), dict(practice_repeat_cap=3), dict(practice_starts='yes'),
           dict(practice_starts=None), dict(practice_fraction=0),
           dict(practice_starts_by_arm=dict(B='x', R='y')),
           dict(practice_starts_by_arm=dict(B=dict(practice_starts=True, practice_fraction=.25,
                                                   practice_repeat_cap=2), R='x')))
    for kw in bad:
        with pytest.raises(launch.Refused):
            launch.bind_runtime(dict(config=dict(live, **kw)), root=tmp_path/'v2v', source_root=tmp_path/'src')
    with pytest.raises(launch.Refused):                      # A-v2v is unchanged: no bank, no practice
        launch.practice_options_for(live, 'A')
    with pytest.raises(launch.Refused):                      # the producer refuses unregistered values
        _cfg(tmp_path, practice_starts=True, practice_fraction=.5)
    with pytest.raises(ValueError):
        practice_starts.practice_count(24, 0.)
    assert practice_starts.practice_count(24, 0.25) == 6
    assert practice_starts.practice_count(24, 1.) == 24


# ---------------------------------------------------------------- eligible examples / frozen plan

def test_examples_reads_own_admitted_cooperative_examples(tmp_path, ps):
    bank_with_snapshots(tmp_path, 'B', ('c1only', 'crep'))
    make(tmp_path, 'nocoop', win(), win(), arm='B', cooperation=False)  # admitted, no_solos
    (rb.bank_root(tmp_path, 'B')/'decisions'/'state_B_nocoop.json').write_text(json.dumps(
        dict(verification_id='nocoop', state_id='state_B_nocoop')))
    rows = practice_starts.examples(tmp_path, 'B')
    assert {r['example_id'] for r in rows} == {'c1only#s1', 'c1only#s2', 'crep#s1', 'crep#s2'}
    assert all(r['fallback'] is False and r['salt'] == r['verification_id'] for r in rows)
    rows = practice_starts.examples(tmp_path, 'B', cooperative_only=False)
    assert {r['example_id'] for r in rows} == {'c1only#s1', 'c1only#s2', 'crep#s1', 'crep#s2',
                                               'nocoop#s1', 'nocoop#s2'}
    assert [r['fallback'] for r in rows if r['example_id'].startswith('nocoop')] == [True, True]
    # imported/foreign labels (no own-bank decision link) are never practice starts
    (rb.bank_root(tmp_path, 'B')/'decisions'/'state_B_ghost.json').write_text(json.dumps(
        dict(verification_id='ghost', state_id='state_ghost')))
    rows = practice_starts.examples(tmp_path, 'B')
    assert 'ghost' not in {r['verification_id'] for r in rows}


def test_examples_on_unresolvable_bank_is_empty(tmp_path, ps):
    from tools.coopmwm.bank_statistics import BankDataMissing
    monkey = pytest.MonkeyPatch()
    monkey.setattr(rb, 'confirmations', lambda root, arm: (_ for _ in ()).throw(BankDataMissing('x')))
    assert practice_starts.examples(tmp_path, 'B') == []


def test_plan_fraction_cap_and_freeze(tmp_path, ps):
    bank_with_snapshots(tmp_path, 'B', tuple(f'v{i}' for i in range(5)))
    plan = practice_starts.plan_for_update(tmp_path, 'B', 3, fraction=0.25, repeat_cap=2,
                                           count=6, training_seed=12345)
    assert len(plan['rows']) == 6 and plan['requested'] == 6
    assert plan['registry']['cooperative_examples'] == 10 and not plan['registry']['empty_bank']
    assert plan['registry']['fallback_used'] is False
    used = {}
    for row in plan['rows']:
        used[row['example_id']] = used.get(row['example_id'], 0)+1
    assert max(used.values()) <= 2 and sum(used.values()) == 6          # cap on repeats per update
    again = practice_starts.plan_for_update(tmp_path, 'B', 3, fraction=0.25, repeat_cap=2,
                                            count=6, training_seed=12345)
    assert again == plan                                                # frozen, never re-sampled
    with pytest.raises(ValueError, match='practice_plan_sha256'):
        broken = dict(plan);broken.pop('practice_plan_sha256')
        from tools.coopmwm import exact_grade as xg
        xg.verify_sealed(broken, 'practice_plan_sha256')
    other = practice_starts.plan_for_update(tmp_path, 'B', 4, fraction=0.25, repeat_cap=2,
                                            count=6, training_seed=12345)
    assert other['update'] == 4 and other != plan
    with pytest.raises(ValueError):
        practice_starts.plan_for_update(tmp_path, 'B', 3, fraction=0.25, repeat_cap=0,
                                        count=6, training_seed=1)
    path = practice_starts.plan_path(tmp_path, 'B', 3)
    record = json.loads(path.read_text())
    record['rows'] = []
    (path).write_text(json.dumps(record))
    with pytest.raises(ValueError, match='mismatch'):
        practice_starts.plan_for_update(tmp_path, 'B', 3, fraction=0.25, repeat_cap=2,
                                        count=6, training_seed=12345)


def test_r_fallback_and_b_never_falls_back(tmp_path, ps):
    # B: two cooperative decisions (4 examples) -> the cap fills the count with repeats, no fallback
    bank_with_snapshots(tmp_path, 'B', ('b1', 'b2'))
    plan = practice_starts.plan_for_update(tmp_path, 'B', 1, fraction=0.25, repeat_cap=2,
                                           count=6, training_seed=7)
    assert len(plan['rows']) == 6 and not any(r['fallback'] for r in plan['rows'])
    used = {}
    for row in plan['rows']:
        used[row['example_id']] = used.get(row['example_id'], 0)+1
    assert max(used.values()) <= 2 and plan['registry']['fallback_used'] is False
    assert plan['requested'] == 6
    # B: the repeat cap exhausts before the count -> fewer practice episodes than requested
    plan = practice_starts.plan_for_update(tmp_path, 'B', 2, fraction=0.25, repeat_cap=2,
                                           count=20, training_seed=7)
    assert len(plan['rows']) == 8 and plan['requested'] == 20     # 4 examples x cap 2
    assert plan['registry']['cooperative_examples'] == 4
    # R: one cooperative decision (2 examples) + admitted non-cooperative examples -> fallback
    bank_with_snapshots(tmp_path, 'R', ('r1',), snapshot_seed=9)
    make(tmp_path, 'r2', win(), win(), arm='R', cooperation=False)
    make(tmp_path, 'r3', win(), win(), arm='R', cooperation=False)
    for vid in ('r2', 'r3'):
        (rb.bank_root(tmp_path, 'R')/'decisions').mkdir(parents=True, exist_ok=True)
        (rb.bank_root(tmp_path, 'R')/'decisions'/(f'state_R_{vid}.json')).write_text(json.dumps(
            dict(verification_id=vid, state_id=f'state_R_{vid}')))
    plan = practice_starts.plan_for_update(tmp_path, 'R', 1, fraction=0.25, repeat_cap=2,
                                           count=6, training_seed=7)
    assert len(plan['rows']) == 6                                        # same count as B's request
    assert plan['registry']['fallback_used'] is True
    coop = [r for r in plan['rows'] if not r['fallback']]
    assert {r['example_id'] for r in coop} == {'r1#s1', 'r1#s2'}
    assert all(r['fallback'] for r in plan['rows'] if r['example_id'].startswith(('r2', 'r3')))
    # R with enough cooperative examples never uses the fallback
    plan = practice_starts.plan_for_update(tmp_path, 'R', 2, fraction=0.25, repeat_cap=1,
                                           count=2, training_seed=7)
    assert len(plan['rows']) == 2 and not plan['registry']['fallback_used']


def test_empty_bank_yields_no_practice_rows(tmp_path, ps):
    plan = practice_starts.plan_for_update(tmp_path, 'B', 1, fraction=0.25, repeat_cap=2,
                                           count=6, training_seed=1)
    assert plan['rows'] == [] and plan['requested'] == 6
    assert plan['registry']['empty_bank'] is True


# ---------------------------------------------------------------- snapshot + CRN identity

def test_seed1_and_seed2_snapshots_carry_the_example_crn(tmp_path, ps):
    snaps = bank_with_snapshots(tmp_path, 'B', ('crep',))
    rows = practice_starts.examples(tmp_path, 'B')
    by_seed = {r['seed_index']: r for r in rows}
    snap1 = practice_starts.load_snapshot(tmp_path, 'B', by_seed[1])
    stored = snaps['crep']['snapshot']
    assert torch.equal(snap1['generator_state'], stored['generator_state'])
    assert practice_starts.load_snapshot(tmp_path, 'B', by_seed[2])['rng']['replication_salt'] == 'crep'
    snap2 = practice_starts.load_snapshot(tmp_path, 'B', by_seed[2])
    expected = rb.replicate(stored, 'crep')
    assert torch.equal(snap2['generator_state'], expected['generator_state'])
    assert snap2['rng']['generator_state_sha256'] == expected['rng']['generator_state_sha256']
    assert not torch.equal(snap2['generator_state'], stored['generator_state'])
    with pytest.raises(ValueError, match='seed_index'):
        practice_starts.load_snapshot(tmp_path, 'B', dict(by_seed[1], seed_index=3))


def test_wave_call_builds_the_exact_prefix_branch_and_practice_meta(tmp_path, ps):
    snaps = bank_with_snapshots(tmp_path, 'B', ('c1only',))
    row = practice_starts.examples(tmp_path, 'B')[0]
    snap = snaps['c1only']['snapshot']
    call = practice_starts.wave_call(tmp_path, 'B', 'v2v1_warm_u005', 'v2v_train', row, 0,
                                     dict(update=5, b_replay=True, v2v=True, v2v_dim=48))
    assert call['ident'] == 'v2v1_warm_u005_p01'
    assert call['worlds'] == [snap['episode']['world']]
    x = call['extras']
    t0 = int(snap['t0'])
    assert x['branch']['time_s'] == t0 and x['branch']['takeover_s'] == t0 and x['branch']['revert'] == []
    assert x['branch']['prefixes'] == {str(t): snap['prefixes'][str(t)] for t in range(300, t0+1)}
    assert x['noise_seed'] == snap['sumo']['noise_seed']
    assert (x['runtime'], x['checkpoint'], x['warm_history']) == ('rt', 'ck', False)
    assert x['accel_bounds'] == [-3., 2.6] and x['sim_s'] == 1200 and x['deterministic'] is False
    assert x['update'] == 5 and x['b_replay'] is True and x['v2v'] is True
    meta = x['practice']['meta'] if 'meta' in x['practice'] else x['practice']
    assert x['practice']['snapshot']['generator_state'].equal(snap['generator_state'])
    assert meta['practice'] is True and meta['t0'] == t0
    assert meta['verification_id'] == 'c1only' and meta['seed_index'] == 1
    assert meta['snapshot_sha256'] == sv.snapshot_sha(snap)
    assert meta['charged_sim_s'] == 1200 and 'fallback' in meta
    call2 = practice_starts.wave_call(tmp_path, 'B', 'v2v1_warm_u005', 'v2v_train', row, 1,
                                      dict(update=5))
    assert call2['ident'] == 'v2v1_warm_u005_p02'                       # unique ids per repeat


# ---------------------------------------------------------------- charging through the existing hooks

def test_practice_jobs_charge_like_every_episode(tmp_path, ps):
    import types
    from tools.coopnative import storage
    from tools.coopmwm.accel_protocol import SUPERSEDED_CHARGE
    accel_storage._reserve = types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__, CAP=10**9, CAPS={'v2v_train': 10**9}))
    snaps = bank_with_snapshots(tmp_path, 'B', ('c1only',))
    row = practice_starts.examples(tmp_path, 'B')[0]
    call = practice_starts.wave_call(tmp_path, 'B', 'v2v1_warm_u001', 'v2v_train', row, 0,
                                     dict(update=1, b_replay=True))
    jobs = [dict(id=f"{call['ident']}_B_{w['seed']}", sim_s=call['extras']['sim_s'])
            for w in call['worlds']]
    accel_storage.reserve(tmp_path, jobs, 'v2v_train')
    ledger = storage.ledger(tmp_path)
    assert ledger == [dict(id=jobs[0]['id'], phase='v2v_train', sim_s=1200)]
    with pytest.raises(ValueError):                                      # duplicate charge guard
        accel_storage.reserve(tmp_path, jobs, 'v2v_train')
    # 24 practice episodes cost exactly what 24 normal episodes cost (per-update delta 0)
    assert practice_starts.CHARGED_SIM_S == 1200
    assert 24*practice_starts.CHARGED_SIM_S == 24*1200
    assert accel_storage.charged(tmp_path) == SUPERSEDED_CHARGE+1200


# ---------------------------------------------------------------- scheduling through v2v_runner.run_arm

def fake_step(t, value=0.):
    return dict(time_s=t, reward=-1., sample=dict(value=value, logp=0.,
        behaviour=dict(route='policy', executed=False, sample_probability=0., q=0., regime='ordinary'),
        context=dict(members=list(MEMBERS)), action=[[0.], [0.]]))


def run_one_update(tmp_path, monkeypatch, *, arm='B', practice, vids=(), snap=None, t0=400):
    """One smoke update of v2v_runner.run_arm with a recorded wave; returns (calls, receipt)."""
    from tools.coopmwm.accel_learner import save
    from tools.coopmwm.acceleration import Policy
    for name, value in (('nice', lambda inc: 0), ('getpriority', lambda which, who: 0),
                        ('PRIO_PROCESS', 0)):
        monkeypatch.setattr(os, name, value, raising=False)
    policy = Policy();opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = torch.Generator().manual_seed(1)
    root = tmp_path/'v2v';root.mkdir(parents=True, exist_ok=True)
    ck = root/f'{arm}_checkpoint_000.pt'
    if not ck.exists():                                                  # u0 is write-once
        save(ck, policy, opt, rng, 0, 0)
    if vids:
        bank_with_snapshots(root, arm, vids)                             # the run root's own bank
    base = dict(budget=dict(total=17471840, phases=dict(v2v_train=2073600)),
                accel_bounds=[-3., 2.6], runtime='rt', checkpoint='ck', workers=1,
                gpu_deadline=1e12, awr=dict(coefficient=1.))
    cfg = v2v_runner.prepare_intent(base, root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256=sha(ck),
                                    practice_starts=practice)
    cfg.update(stop_after_update=1, launch_amendment_sha256='l'*64, funding_amendment_sha256='f'*64)
    (root/'AUTHORITY.md').write_text('authority '+('l'*64)+' '+('f'*64)+'\n')
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    release = dict(qualified=True, linux_cpu_pass=True, zero_init_parity=True,
                   default_off_byte_identical=True, stub=False, v2v_dim=live['v2v_dim'],
                   rehearsal_pass=True)
    hooks = {n: Mock() for n in v2v_runner.REQUIRED_HOOKS}
    hooks['load_checkpoint'].return_value = (policy, opt, rng, dict(update=0))
    hooks['bank_pairs'].return_value = ([], dict(counts=dict(positive=0, available_admitted=0,
                                                             used_admitted=0)))
    hooks['bank_quality'].return_value = {}
    calls = []

    def fake_wave(r, b, policies, ws, role, ident, phase, deterministic=True, extras=None):
        calls.append(dict(ident=ident, role=role, phase=phase, worlds=list(ws), extras=dict(extras or {})))
        out = []
        for w in ws:
            job_practice = (extras or {}).get('practice')
            steps = ([fake_step(t0), fake_step(t0+1, value=1.)] if job_practice
                     else [fake_step(t) for t in (400, 401)])
            e = dict(id=f'{ident}_{arm}_{w["seed"]}', world=w, arm=arm, steps=steps,
                     metrics={'900': dict(hard_brake=0.)})
            if job_practice:
                e['practice'] = {k: v for k, v in job_practice.items() if k != 'snapshot'}
            out.append(e)
        return out

    from tools.coopmwm import warm_runner as wr
    from tools.coopmwm import exact_grade as xg
    monkeypatch.setattr(wr, 'wave', fake_wave)
    monkeypatch.setattr(xg, 'exact_update', lambda *a, **k: [])
    from tools.coopmwm import search_learning
    monkeypatch.setattr(search_learning, 'exact_update', lambda *a, **k: None)
    monkeypatch.setattr(bs, '_CACHE', {})
    bs.install(rb, bank_rule='per_seed_v1')
    v2v_runner.run_arm(root, arm, live, release, hooks)
    receipt = json.loads((root/f'{arm}_UPDATE_001.json').read_text())
    return calls, receipt


def test_off_run_is_byte_identical_no_practice_keys_no_extra_waves(tmp_path, monkeypatch, ps):
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=False)
    assert 'practice' not in receipt
    assert len(calls) == 1                                               # one normal 24-episode wave
    assert calls[0]['ident'] == 'v2v1_warm_u001' and len(calls[0]['worlds']) == 24
    assert calls[0]['role'] == 'train' and calls[0]['phase'] == 'v2v_train'
    assert calls[0]['extras']['b_replay'] is True


def test_fraction_honoured_six_practice_starts_of_twenty_four(tmp_path, monkeypatch, ps):
    vids = tuple(f'v{i}' for i in range(4))                              # 8 cooperative examples
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=True, vids=vids)
    practice_report = receipt['practice']
    assert practice_report['practice'] == 6 and practice_report['normal'] == 18
    assert len(calls) == 7                                               # 1 normal wave + 6 practice
    normal, practices = calls[0], calls[1:]
    assert normal['ident'] == 'v2v1_warm_u001' and len(normal['worlds']) == 18
    from tools.coopmwm.warm_protocol import batch_worlds
    assert normal['worlds'] == batch_worlds(1)[:18]                      # the last 6 worlds replaced
    assert [c['ident'] for c in practices] == [f'v2v1_warm_u001_p{i:02d}' for i in range(1, 7)]
    rows = json.loads(practice_starts.plan_path(tmp_path/'v2v', 'B', 1).read_text())['rows']
    for c, row in zip(practices, rows):
        assert c['role'] == 'train' and c['phase'] == 'v2v_train'
        assert len(c['worlds']) == 1
        x = c['extras']
        assert x['sim_s'] == 1200 and x['update'] == 1 and x['b_replay'] is True
        assert x['practice']['practice'] is True
        assert x['practice']['example_id'] == row['example_id']
        assert x['practice']['snapshot_sha256']
    assert practice_report['examples'] == sorted({r['example_id'] for r in rows})
    assert practice_report['practice_return_veh_s'] == -1.               # the single post-t0 step
    assert practice_report['normal_return_veh_s'] == -2.
    # every practice episode keeps only the post-t0 seconds as PPO steps
    assert all(e['t0'] == 400 and e['steps'] == 1 for e in receipt['practice']['episodes'])


def test_practice_worlds_are_the_example_snapshots_and_ids_are_frozen(tmp_path, monkeypatch, ps):
    vids = ('v0', 'v1')
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=True, vids=vids)
    from tools.coopmwm.accel_protocol import worlds as reg_worlds
    snap_worlds = {986000, 986001}
    for c in calls[1:]:
        (w,) = c['worlds']
        assert w['seed'] in snap_worlds                                  # the snapshot's own world
        assert dict(reg_worlds('train')[0], seed=w['seed']) == w         # a registered train world
        assert c['extras']['branch']['time_s'] == 400
        assert c['extras']['practice']['t0'] == 400
    plan = json.loads(practice_starts.plan_path(tmp_path/'v2v', 'B', 1).read_text())
    assert plan['requested'] == 6 and len(plan['rows']) == 6             # cap 2 x 4 examples
    assert plan['registry']['cooperative_examples'] == 4
    # a restart reuses the frozen plan: same rows, same job idents (one-use receipts cleared)
    root = tmp_path/'v2v'
    for name in ('WARM_B_STARTED.json', 'WARM_B_PAUSED.json', 'B_UPDATE_001.json',
                 'B_VIABILITY_001.json', 'B_RANDOM_QUOTA_001.json', 'B_SEARCH_LEARNING_001.json',
                 'B_checkpoint_001.pt'):
        (root/name).unlink(missing_ok=True)
    calls2, _ = run_one_update(tmp_path, monkeypatch, arm='B', practice=True)
    assert [c['ident'] for c in calls2[1:]] == [c['ident'] for c in calls[1:]]


def test_empty_bank_runs_a_full_normal_wave(tmp_path, monkeypatch, ps):
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=True, vids=())
    assert len(calls) == 1 and len(calls[0]['worlds']) == 24
    assert receipt['practice']['practice'] == 0 and receipt['practice']['normal'] == 24
    assert receipt['practice']['examples'] == []
    plan = json.loads(practice_starts.plan_path(tmp_path/'v2v', 'B', 1).read_text())
    assert plan['rows'] == [] and plan['registry']['empty_bank'] is True


# ---------------------------------------------------------------- episode machinery (fake plant, no SUMO)

class FakeBackend:
    def __init__(self, path, world, sim_s, checkpoint):
        self.world = world['seed'];self.actual = 300
        self.totals = dict(team_delay_vs=0., collisions=0., teleports=0., hard_brake=0.,
                           ttc_below_3_vs=0.)
    def fingerprint(self):
        return fp(self.actual, base_delay(self.world, self.actual))
    def close(self):
        pass


def fake_tick(b, controller, trace):
    from tools.coopnative.test_cpu import rows
    r = rows()
    commands = {'c1': 0., 'c2': 0.}
    commands.update({row['vid']: 0. for row in r if row['is_cav']})
    controller.apply(r, commands, b.actual)
    b.actual += 1
    b.totals['team_delay_vs'] += 7.
    return []


def practice_job(tmp_path, snap, weights, name='job'):
    from tools.coopmwm.accel_protocol import worlds
    t0 = int(snap['t0'])
    meta = dict(practice=True, verification_id='vfix', example_id='vfix#s1', seed_index=1, t0=t0,
                snapshot_sha256=sv.snapshot_sha(snap), crn_seed='test', fallback=False,
                horizon_end_s=1200, charged_sim_s=1200, world=snap['episode']['world'], rule='test')
    return dict(path=str(tmp_path/name), id=name, world=snap['episode']['world'], role='train',
                arm='B', weights=weights, runtime='rt', checkpoint='ck',
                noise_seed=snap['sumo']['noise_seed'], deterministic=False, sim_s=1200,
                b_replay=True, practice=dict(meta, snapshot=snap),
                branch=dict(time_s=t0, takeover_s=t0, revert=[],
                            prefixes={str(t): snap['prefixes'][str(t)] for t in range(300, t0+1)}))


def run_episode_job(job):
    """Run one episode against the rehearsal's fake plant (tools.leverprobe is absent locally;
    the same sys.modules injection as rehearsal_cpu._fake_leverprobe, with this test's backend)."""
    import sys
    import types
    pkg = types.ModuleType('tools.leverprobe');pkg.__path__ = []
    fake = types.ModuleType('tools.leverprobe.backend')
    fake.setup = lambda runtime, checkpoint: None
    fake.Backend = FakeBackend
    sys.modules['tools.leverprobe'] = pkg
    sys.modules['tools.leverprobe.backend'] = fake
    pkg.backend = fake
    with patch('tools.coopfree.backend.tick', fake_tick), \
         patch('tools.coopmwm.accel_plant.install_seed_scope'), \
         patch('tools.coopmwm.accel_plant.guard'):
        from tools.coopmwm.accel_plant import episode
        return episode(job)


def run_practice_episode(tmp_path, snap, weights, name='job'):
    return run_episode_job(practice_job(tmp_path, snap, weights, name))


def test_practice_episode_replays_prefix_restores_crn_and_flags_the_receipt(tmp_path):
    from tools.coopmwm.acceleration import Policy
    from tools.coopmwm.practice_starts import strip_steps
    snap = snapshot(t0=400, world_seed=986000, gen_seed=5)
    weights = Policy().state_dict()
    result = run_practice_episode(tmp_path, snap, weights, name='job1')
    assert result['practice']['practice'] is True and result['practice']['t0'] == 400
    assert result['practice']['snapshot_sha256'] == sv.snapshot_sha(snap)
    assert 'snapshot' not in result['practice']
    assert result['charged_sim_s'] == 1200
    full = [s['time_s'] for s in result['steps']]
    assert full[0] == 300 and full[-1] == 1199                           # full tape on disk
    # the snapshot's own t0 command is replayed; current policy from t0+1
    t0_step = next(s for s in result['steps'] if s['time_s'] == 400)
    assert t0_step['sample']['action'] == snap['samples']['400']['action']
    after = [s for s in strip_steps(result)['steps']]
    assert [s['time_s'] for s in after] == list(range(401, 1200))
    assert all(s['sample']['action'] != [[5.], [5.]] for s in after)     # not the recorded action
    # CRN: the same weights reproduce byte-identical post-t0 actions from the snapshot state
    again = run_practice_episode(tmp_path, snap, weights, name='job2')
    a = [s['sample']['action'] for s in strip_steps(again)['steps']]
    b = [s['sample']['action'] for s in after]
    assert a == b and len({json.dumps(x, sort_keys=True) for x in a}) > 1
    assert (tmp_path/'job1'/'RESULT.json').exists()                      # receipt stored


def test_practice_episode_enforces_the_sealed_prefix_and_crn(tmp_path):
    from tools.coopmwm.acceleration import Policy
    snap = snapshot(t0=400, world_seed=986000, gen_seed=5)
    weights = Policy().state_dict()
    bad_prefix = dict(snap['prefixes']);bad_prefix['399'] = fp(399, base_delay(986000, 399)+1e-9)
    job = practice_job(tmp_path, snap, weights, name='jobp')
    job['branch']['prefixes'] = bad_prefix
    with pytest.raises(ValueError, match='prefix mismatch'):
        run_episode_job(job)
    tampered = dict(snap, rng=dict(snap['rng'], generator_state_sha256='0'*64))
    job = practice_job(tmp_path, snap, weights, name='jobc')
    job['practice']['snapshot'] = tampered
    with pytest.raises(ValueError, match='generator snapshot SHA mismatch'):
        run_episode_job(job)


# ---------------------------------------------------------------- strip + report

def test_strip_steps_and_update_report():
    eps = [dict(id='p1', world=dict(seed=1), practice=dict(verification_id='v', example_id='v#s1',
                       seed_index=1, t0=400, fallback=False),
                steps=[fake_step(399), fake_step(400), fake_step(401, value=1.)]),
           dict(id='p2', world=dict(seed=2), practice=dict(verification_id='v', example_id='v#s2',
                       seed_index=2, t0=500, fallback=True),
                steps=[fake_step(501, value=2.)]),
           dict(id='n1', world=dict(seed=3),
                steps=[fake_step(t) for t in range(300, 1200)])]
    p1 = [e for e in eps if e.get('practice')][0]
    stripped = practice_starts.strip_steps(p1)
    assert [s['time_s'] for s in stripped['steps']] == [401]
    assert [s['time_s'] for s in p1['steps']] == [399, 400, 401]         # input untouched
    report = practice_starts.update_report([stripped] + [
        practice_starts.strip_steps(eps[1]), eps[2]])
    assert report['practice'] == 2 and report['normal'] == 1
    assert report['examples_used'] == {'v#s1': 1, 'v#s2': 1}
    assert report['practice_return_veh_s'] == -1. and report['normal_return_veh_s'] == -900.
    assert report['practice_return_per_control_s'] == -1. and report['normal_return_per_control_s'] == -1.
    assert [e['fallback'] for e in report['episodes']] == [False, True]
    assert report['episodes'][1]['seed_index'] == 2 and report['episodes'][1]['t0'] == 500


def test_replacement_of_a_failed_practice_job_is_a_normal_fresh_start():
    from tools.coopmwm.accel_protocol import worlds
    job = dict(id='v2v1_warm_u001_p01_B_986000', arm='B', world=dict(worlds('train')[0]),
               practice=dict(t0=400), branch=dict(time_s=400),
               path='x/v2v1_warm_u001_p01_B_986000')
    out = accel_runner.replacement_job(job, dict(worlds('train')[1], seed=986001), 'v2v1_warm_u001_p01')
    assert out['id'] == 'v2v1_warm_u001_p01_B_986001' and out['replaces'] == job['id']
    assert 'practice' not in out and 'branch' not in out                 # no snapshot state carried
    normal = dict(id='x', arm='B', world=dict(worlds('train')[0]), path='x/x')
    out = accel_runner.replacement_job(normal, dict(worlds('train')[1], seed=986001), 'ident')
    assert 'practice' not in out                                         # normal jobs unchanged
