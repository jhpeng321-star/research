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
    # ON differs in the flag and the v2v_train allocation only (the quoted matched A-reference
    # cost); OFF stays byte-identical to the launch defaults.
    assert {k for k in cfg if cfg[k] != on[k]} == {'practice_starts', 'budget'}
    assert on['budget']['phases']['v2v_train'] == (launch.V2V_PHASES['v2v_train']
                                                   + launch.practice_reference_quote(0.25))
    assert on['practice_starts'] is True
    # a legacy runtime file without the keys still binds (OFF default), ON binds with B == R
    stripped = {k: v for k, v in live.items() if k not in launch.PRACTICE_KEYS}
    assert launch.bind_runtime(dict(config=stripped), root=tmp_path/'v2v', source_root=tmp_path/'src') is stripped
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    assert launch.bind_runtime(dict(config=dict(live, practice_starts=True)),
                               root=tmp_path/'v2v', source_root=tmp_path/'src')['practice_starts'] is True


def test_practice_on_phase_allocation_stays_within_the_registered_ceiling(tmp_path):
    """Fix round 2 (e): with practice ON the v2v_train allocation carries the matched practice
    A-reference quote and the phase sum stays within the registered 17,471,840 sim-s cap -
    bind_runtime/accounting accept it and refuse a single sim-s above the ceiling."""
    from tools.coopmwm import v2v_live
    _, live = _cfg(tmp_path, practice_starts=True)
    phases = dict(launch.V2V_PHASES)
    phases['v2v_train'] = launch.V2V_PHASES['v2v_train'] + launch.practice_reference_quote(0.25)
    live = dict(live, budget=dict(total=launch.CEILING_SIM_S, phases=phases))
    assert sum(phases.values()) <= launch.CEILING_SIM_S
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v',
                               source_root=tmp_path/'src') is live
    root = tmp_path/'run';root.mkdir()
    (root/'CHARGES.jsonl').write_text('')
    (root/'ACCOUNTING.claim').touch()
    v2v_live.accounting(root, dict(live, accounting_root=str(root)))     # practice-ON allocation binds
    others = sum(v for k, v in phases.items() if k != 'v2v_train')
    over = json.loads(json.dumps(live))
    over['budget']['phases']['v2v_train'] = launch.CEILING_SIM_S - others + 1   # one sim-s over
    with pytest.raises(ValueError, match='cap/allocation'):
        v2v_live.accounting(root, dict(over, accounting_root=str(root)))
    with pytest.raises(launch.Refused, match='exceed total cap'):
        launch.bind_runtime(dict(config=over), root=tmp_path/'v2v', source_root=tmp_path/'src')


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


def test_r_cooperative_pool_counts_examples_times_cap_before_fallback(tmp_path, ps):
    """B/R symmetry fix: the cooperative pool first counts examples x repeat-cap; only a count
    beyond that capacity falls back to R's non-cooperative examples."""
    from tools.coopmwm import replay_bank as rb
    bank_with_snapshots(tmp_path, 'R', ('r1', 'r2', 'r3'))               # 6 cooperative examples
    make(tmp_path, 'r4', win(), win(), arm='R', cooperation=False)       # admitted, non-cooperative
    (rb.bank_root(tmp_path, 'R')/'decisions').mkdir(parents=True, exist_ok=True)
    (rb.bank_root(tmp_path, 'R')/'decisions'/'state_R_r4.json').write_text(json.dumps(
        dict(verification_id='r4', state_id='state_R_r4')))
    plan = practice_starts.plan_for_update(tmp_path, 'R', 1, fraction=0.25, repeat_cap=2,
                                           count=12, training_seed=7)
    assert len(plan['rows']) == 12 and plan['requested'] == 12
    assert plan['registry']['cooperative_examples'] == 6
    assert plan['registry']['cooperative_capacity'] == 12                # 6 examples x cap 2
    assert plan['registry']['fallback_used'] is False
    assert all(not r['fallback'] for r in plan['rows'])                  # r4 never sampled
    # beyond the cooperative capacity the fallback fills the remainder (same count as B)
    plan = practice_starts.plan_for_update(tmp_path, 'R', 2, fraction=0.25, repeat_cap=2,
                                           count=14, training_seed=7)
    assert plan['registry']['fallback_used'] is True
    assert len(plan['rows']) == 14 and plan['requested'] == 14           # beyond the 12 coop rows
    fallback_rows = sum(1 for r in plan['rows'] if r['fallback'])
    assert 2 <= fallback_rows <= 4                                       # r4 has 2 examples x cap 2
    assert {r['example_id'] for r in plan['rows'] if r['fallback']} <= {'r4#s1', 'r4#s2'}


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
    # extra-seed examples (seed_index >= 3): the same deterministic per-seed CRN derivation
    s3 = practice_starts.load_snapshot(tmp_path, 'B', dict(by_seed[1], seed_index=3))
    e3 = rb.replicate(stored, 'crep#s3')
    assert torch.equal(s3['generator_state'], e3['generator_state'])
    assert s3['rng']['replication_salt'] == 'crep#s3'
    s4 = practice_starts.load_snapshot(tmp_path, 'B', dict(by_seed[1], seed_index=4))
    assert not torch.equal(s4['generator_state'], s3['generator_state'])   # independent per seed
    assert torch.equal(practice_starts.load_snapshot(
        tmp_path, 'B', dict(by_seed[1], seed_index=3))['generator_state'], e3['generator_state'])
    with pytest.raises(ValueError, match='seed_index'):
        practice_starts.load_snapshot(tmp_path, 'B', dict(by_seed[1], seed_index=0))


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
    # the recorded snapshot sumo configuration travels INSIDE the practice payload; extras never
    # duplicate the keywords accel_runner.wave passes explicitly (a duplicate is a TypeError)
    for key in ('runtime', 'checkpoint', 'noise_seed', 'warm_history', 'accel_bounds', 'sim_s',
                'deterministic'):
        assert key not in x
    sumo = x['practice']['snapshot']['sumo']
    assert sumo['noise_seed'] == snap['sumo']['noise_seed'] == 11
    assert (sumo['runtime'], sumo['checkpoint'], sumo['warm_history']) == ('rt', 'ck', False)
    assert sumo['accel_bounds'] == [-3., 2.6]
    assert x['update'] == 5 and x['b_replay'] is True and x['v2v'] is True
    meta = x['practice']
    assert x['practice']['snapshot']['generator_state'].equal(snap['generator_state'])
    assert meta['practice'] is True and meta['t0'] == t0
    assert meta['slot'] == 'v2v1_warm_u005_p01' and meta['a_reference'] is False
    assert meta['verification_id'] == 'c1only' and meta['seed_index'] == 1
    assert meta['snapshot_sha256'] == sv.snapshot_sha(snap)
    assert meta['charged_sim_s'] == 1200 and 'fallback' in meta
    call2 = practice_starts.wave_call(tmp_path, 'B', 'v2v1_warm_u005', 'v2v_train', row, 1,
                                      dict(update=5))
    assert call2['ident'] == 'v2v1_warm_u005_p02'                       # unique ids per repeat
    assert call2['extras']['practice']['slot'] == 'v2v1_warm_u005_p02'


def test_wave_call_output_builds_real_accel_runner_wave_jobs(tmp_path, ps):
    """The REAL accel_runner.wave job construction (wave_jobs, called by wave) consumes
    wave_call output: the practice job carries the snapshot's recorded runtime/checkpoint/
    noise_seed/warm_history/accel_bounds exactly once and its frozen slot id."""
    from tools.coopmwm.accel_protocol import worlds
    from tools.coopmwm.acceleration import Policy
    snaps = bank_with_snapshots(tmp_path, 'B', ('c1only',))
    row = practice_starts.examples(tmp_path, 'B')[0]
    snap = snaps['c1only']['snapshot']
    call = practice_starts.wave_call(tmp_path, 'B', 'v2v1_warm_u005', 'v2v_train', row, 0,
                                     dict(update=5, b_replay=True, v2v=True, v2v_dim=48))
    b = dict(runtime='current-rt', checkpoint='current-ck', accel_bounds=[-9., 9.],
             gpu_deadline=1e12, workers=1, warm_history=True)
    jobs, reused = accel_runner.wave_jobs(tmp_path, b, {'B': Policy()}, call['worlds'], 'train',
                                          'u005', 'v2v_train', False, call['extras'])
    assert reused == []
    job, = jobs
    sumo = snap['sumo']
    assert job['id'] == 'v2v1_warm_u005_p01_B_986000'
    assert (job['runtime'], job['checkpoint'], job['noise_seed']) == ('rt', 'ck', 11)
    assert job['warm_history'] is False and job['accel_bounds'] == [-3., 2.6]
    assert job['sim_s'] == 1200 and job['deterministic'] is False and job['guided'] is True
    assert job['branch']['time_s'] == int(snap['t0'])
    assert job['practice']['slot'] == 'v2v1_warm_u005_p01'
    # a duplicate explicit keyword in extras stays a hard error (never a silent override)
    with pytest.raises(TypeError, match='duplicate explicit job keywords'):
        accel_runner.wave_jobs(tmp_path, b, {'B': Policy()}, call['worlds'], 'train', 'u005',
                               'v2v_train', False, dict(call['extras'], noise_seed=1))
    # the non-practice path is unchanged: binding runtime and the digest noise seed
    normal, _ = accel_runner.wave_jobs(tmp_path, b, {'B': Policy()},
                                       [dict(worlds('train')[1], seed=986001)], 'train', 'u005',
                                       'v2v_train', False, dict(update=5))
    nj, = normal
    assert nj['id'] == 'u005_B_986001'
    assert (nj['runtime'], nj['checkpoint'], nj['accel_bounds']) == ('current-rt', 'current-ck',
                                                                     [-9., 9.])
    assert nj['warm_history'] is True and nj['noise_seed'] != 11


def test_practice_batch_extras_build_one_frozen_job_per_episode(tmp_path, ps):
    """One batch wave for the whole practice update (v2v_runner.practice_wave): practice_batch
    extras are per-world, repeated snapshot worlds keep distinct frozen one-use job ids."""
    from tools.coopmwm.acceleration import Policy
    bank_with_snapshots(tmp_path, 'B', ('c1only',))                      # 2 examples, cap 2
    rows = practice_starts.plan_for_update(tmp_path, 'B', 1, fraction=0.25, repeat_cap=2,
                                           count=4, training_seed=5)['rows']
    calls = [practice_starts.wave_call(tmp_path, 'B', 'v2v1_warm_u001', 'v2v_train', row, i,
                                       dict(update=1, b_replay=True, v2v=True, v2v_dim=48))
             for i, row in enumerate(rows)]
    worlds = [c['worlds'][0] for c in calls]
    batch = [dict(practice=c['extras']['practice'], branch=c['extras']['branch']) for c in calls]
    jobs, _ = accel_runner.wave_jobs(tmp_path, dict(runtime='rt', checkpoint='ck',
        accel_bounds=[-3., 2.6], gpu_deadline=1e12, workers=1), {'B': Policy()}, worlds, 'train',
        'v2v1_warm_u001_practice', 'v2v_train', False,
        extras=dict(update=1, b_replay=True, practice_batch=batch))
    assert [j['id'] for j in jobs] == [f'v2v1_warm_u001_p{i:02d}_B_{worlds[0]["seed"]}'
                                       for i in range(1, 5)]
    assert len({j['id'] for j in jobs}) == len(jobs)                     # repeats never collide
    for j, call in zip(jobs, calls):
        assert j['practice']['snapshot_sha256'] == call['extras']['practice']['snapshot_sha256']
        assert j['branch'] == call['extras']['branch']
        assert j['noise_seed'] == 11 and j['sim_s'] == 1200


def test_a_reference_call_matches_the_practice_snapshot_and_crn(tmp_path, ps):
    snaps = bank_with_snapshots(tmp_path, 'B', ('c1only',))
    row = practice_starts.examples(tmp_path, 'B')[0]
    snap = snaps['c1only']['snapshot']
    call = practice_starts.wave_call(tmp_path, 'B', 'v2v1_warm_u001', 'v2v_train', row, 0,
                                     dict(update=1))
    ref = practice_starts.a_reference_call(tmp_path, 'B', 'v2v1_warm_u001_practiceref', row, 0,
                                           dict(update=1, v2v=True, v2v_dim=48, b_replay=True))
    assert ref['ident'] == 'v2v1_warm_u001_practiceref_pa01'
    assert ref['worlds'] == call['worlds']                               # the SAME snapshot world
    rx, x = ref['extras'], call['extras']
    assert rx['practice']['snapshot']['generator_state'].equal(
        x['practice']['snapshot']['generator_state'])                    # the SAME CRN seed
    assert rx['branch'] == x['branch']                                   # the same sealed prefix
    assert rx['practice']['a_reference'] is True
    assert rx['practice']['slot'] == 'v2v1_warm_u001_practiceref_pa01'
    assert 'A policy from t0' in rx['practice']['policy']
    assert rx['practice']['charged_sim_s'] == 1200
    for key in ('runtime', 'checkpoint', 'noise_seed', 'accel_bounds', 'sim_s', 'deterministic'):
        assert key not in rx


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
    jobs = [dict(id=f"{call['ident']}_B_{w['seed']}", sim_s=practice_starts.CHARGED_SIM_S)
            for w in call['worlds']]
    accel_storage.reserve(tmp_path, jobs, 'v2v_train')
    ledger = storage.ledger(tmp_path)
    assert ledger == [dict(id=jobs[0]['id'], phase='v2v_train', sim_s=1200)]
    with pytest.raises(ValueError):                                      # duplicate charge guard
        accel_storage.reserve(tmp_path, jobs, 'v2v_train')
    # 24 practice episodes cost exactly what 24 normal episodes cost (per-update delta 0);
    # each practice episode's matched A reference costs the same 1200 sim-s (quoted separately)
    assert practice_starts.CHARGED_SIM_S == 1200
    assert 24*practice_starts.CHARGED_SIM_S == 24*1200
    from tools.coopmwm.v2v_launch3 import practice_reference_quote
    assert practice_reference_quote(0.25) == 2*24*6*1200                 # upper bound, B and R
    assert accel_storage.charged(tmp_path) == SUPERSEDED_CHARGE+1200


# ---------------------------------------------------------------- scheduling through v2v_runner.run_arm

def fake_step(t, value=0.):
    return dict(time_s=t, reward=-1., sample=dict(value=value, logp=0.,
        behaviour=dict(route='policy', executed=False, sample_probability=0., q=0., regime='ordinary'),
        context=dict(members=list(MEMBERS)), action=[[0.], [0.]]))


def run_one_update(tmp_path, monkeypatch, *, arm='B', practice, vids=(), snap=None, t0=400,
                    b_practice=None, b_receipt_late=False, weighting=None):
    """One smoke update of v2v_runner.run_arm with a recorded wave; returns (calls, receipt).
    ``b_practice`` seeds B's realised practice count in B_UPDATE_001.json (R reads it);
    ``b_receipt_late`` instead withholds that receipt until R's first wait tick - R must WAIT
    for it (fix round 2), never crash on its absence."""
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
    waited = []
    if arm == 'R' and b_practice is not None and not b_receipt_late:
        (root/'B_UPDATE_001.json').write_text(json.dumps(dict(
            update=1, arm='B', practice=dict(practice=int(b_practice), normal=24-int(b_practice)))))
    if vids:
        bank_with_snapshots(root, arm, vids)                             # the run root's own bank
    base = dict(budget=dict(total=17471840, phases=dict(v2v_train=2073600)),
                accel_bounds=[-3., 2.6], runtime='rt', checkpoint='ck', workers=1,
                gpu_deadline=1e12, awr=dict(coefficient=1.))
    cfg = v2v_runner.prepare_intent(base, root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256=sha(ck),
                                    practice_starts=practice,
                                    **(dict(practice_weighting=weighting) if weighting is not None else {}))
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

    def fake_wave(r, b, policies, ws, role, ident, phase, deterministic=True, extras=None,
                  used_seeds=None):
        calls.append(dict(ident=ident, role=role, phase=phase, worlds=list(ws),
                          extras=dict(extras or {}), used_seeds=None if used_seeds is None
                          else set(used_seeds)))
        batch = (extras or {}).get('practice_batch')
        out = []
        for wi, w in enumerate(ws):
            job_practice = (batch[wi]['practice'] if batch is not None
                            else (extras or {}).get('practice'))
            steps = ([fake_step(t0), fake_step(t0+1, value=1.)] if job_practice
                     else [fake_step(t) for t in (400, 401)])
            e = dict(id=(f"{job_practice['slot']}_{arm}_{w['seed']}" if job_practice
                         else f'{ident}_{arm}_{w["seed"]}'),
                     world=w, arm=arm, steps=steps, metrics={'900': dict(hard_brake=0.)})
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
    kwargs = {}
    if b_receipt_late:
        def late_sleep(_seconds):
            waited.append(_seconds)
            (root/'B_UPDATE_001.json').write_text(json.dumps(dict(
                update=1, arm='B', practice=dict(practice=int(b_practice),
                                                 normal=24-int(b_practice)))))
        kwargs['sleep'] = late_sleep
    v2v_runner.run_arm(root, arm, live, release, hooks, **kwargs)
    if waited:
        calls.append(dict(ident='__waited__', worlds=[], extras={}, role='wait',
                          phase='wait', used_seeds=None, ticks=len(waited)))
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
    assert len(calls) == 2                                               # 1 normal wave + ONE batch
    normal, batch = calls[0], calls[1]
    assert normal['ident'] == 'v2v1_warm_u001' and len(normal['worlds']) == 18
    from tools.coopmwm.warm_protocol import batch_worlds
    assert normal['worlds'] == batch_worlds(1)[:18]                      # the last 6 worlds replaced
    assert batch['ident'] == 'v2v1_warm_u001_practice'                   # ONE wave for the update
    assert batch['role'] == 'train' and batch['phase'] == 'v2v_train'
    # fix round 2 (c): the practice wave's failure replacements receive the update's closed
    # normal worlds as 'used' seeds (a replacement never duplicates a normal world)
    assert normal['used_seeds'] is None
    assert batch['used_seeds'] == {w['seed'] for w in normal['worlds']}
    assert len(batch['worlds']) == 6 and len(batch['extras']['practice_batch']) == 6
    rows = json.loads(practice_starts.plan_path(tmp_path/'v2v', 'B', 1).read_text())['rows']
    from tools.coopmwm.accel_protocol import worlds as reg_worlds
    for w, item, row in zip(batch['worlds'], batch['extras']['practice_batch'], rows):
        x = item['practice']
        assert w['seed'] in {986000+i for i in range(4)}                 # the snapshot's own world
        assert dict(reg_worlds('train')[0], seed=w['seed']) == w         # a registered train world
        assert x['charged_sim_s'] == 1200
        assert item['practice']['practice'] is True
        assert x['example_id'] == row['example_id']
        assert x['snapshot_sha256'] and x['slot'].startswith('v2v1_warm_u001_p')
        assert item['branch']['time_s'] == 400
    assert 'practice_batch' not in normal['extras']                      # normal wave untouched
    assert practice_report['examples'] == sorted({r['example_id'] for r in rows})
    assert practice_report['practice_return_veh_s'] == -1.               # the single post-t0 step
    assert practice_report['normal_return_veh_s'] == -2.
    # every practice episode keeps only the post-t0 seconds as PPO steps
    assert all(e['t0'] == 400 and e['steps'] == 1 for e in receipt['practice']['episodes'])


def test_r_plans_bs_realised_practice_count_not_the_requested_count(tmp_path, monkeypatch, ps):
    """B/R symmetry: R's practice plan count is B's REALISED count for the update (here B
    realised 4 of the requested 6 after one failed practice episode was replaced)."""
    vids = ('v0', 'v1')                                                  # 4 examples, cap 2
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='R', practice=True, vids=vids,
                                    b_practice=4)
    assert receipt['arm'] == 'R'
    batch = calls[1]
    assert batch['ident'] == 'v2v1_warm_u001_practice'
    assert len(batch['worlds']) == 4                                     # B's realised count, not 6
    plan = json.loads(practice_starts.plan_path(tmp_path/'v2v', 'R', 1).read_text())
    assert plan['requested'] == 4 and len(plan['rows']) == 4
    assert receipt['practice']['practice'] == 4 and receipt['practice']['normal'] == 20
    # without B's receipt the R practice count is unbound and refuses (never the fraction count)
    (tmp_path/'v2v'/'B_UPDATE_001.json').unlink()
    with pytest.raises(ValueError, match='practice count is unbound'):
        practice_starts.realised_count(tmp_path/'v2v', 'B', 1)


def test_r_practice_count_waits_for_bs_receipt_instead_of_crashing(tmp_path, monkeypatch, ps):
    """Fix round 2 (b): R's realised practice count WAITS for B's sealed update receipt (like the
    fairness wait) - a receipt that only appears after a wait tick must never crash the wave."""
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='R', practice=True,
                                    vids=('v0', 'v1'), b_practice=4, b_receipt_late=True)
    waited, = [c for c in calls if c['ident'] == '__waited__']
    assert waited['ticks'] >= 1                                          # R genuinely waited
    batch = calls[1]
    assert batch['ident'] == 'v2v1_warm_u001_practice'
    assert len(batch['worlds']) == 4                                     # B's realised count
    assert receipt['practice']['practice'] == 4                          # from the late receipt


def test_practice_worlds_are_the_example_snapshots_and_ids_are_frozen(tmp_path, monkeypatch, ps):
    vids = ('v0', 'v1')
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=True, vids=vids)
    from tools.coopmwm.accel_protocol import worlds as reg_worlds
    snap_worlds = {986000, 986001}
    (batch,) = calls[1:]
    for w, item in zip(batch['worlds'], batch['extras']['practice_batch']):
        assert w['seed'] in snap_worlds                                  # the snapshot's own world
        assert dict(reg_worlds('train')[0], seed=w['seed']) == w         # a registered train world
        assert item['branch']['time_s'] == 400
        assert item['practice']['t0'] == 400
    plan = json.loads(practice_starts.plan_path(tmp_path/'v2v', 'B', 1).read_text())
    assert plan['requested'] == 6 and len(plan['rows']) == 6             # cap 2 x 4 examples
    assert plan['registry']['cooperative_examples'] == 4
    # a restart reuses the frozen plan: same rows, same job ids (one-use receipts cleared)
    root = tmp_path/'v2v'
    for name in ('WARM_B_STARTED.json', 'WARM_B_PAUSED.json', 'B_UPDATE_001.json',
                 'B_VIABILITY_001.json', 'B_RANDOM_QUOTA_001.json', 'B_SEARCH_LEARNING_001.json',
                 'B_checkpoint_001.pt'):
        (root/name).unlink(missing_ok=True)
    calls2, _ = run_one_update(tmp_path, monkeypatch, arm='B', practice=True)
    assert [c['ident'] for c in calls2[1:]] == [c['ident'] for c in calls[1:]]
    slots1 = [i['practice']['slot'] for i in calls[1]['extras']['practice_batch']]
    slots2 = [i['practice']['slot'] for i in calls2[1]['extras']['practice_batch']]
    assert slots1 == slots2                                              # frozen per-episode ids


def test_empty_bank_runs_a_full_normal_wave(tmp_path, monkeypatch, ps):
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=True, vids=())
    assert len(calls) == 1 and len(calls[0]['worlds']) == 24
    assert receipt['practice']['practice'] == 0 and receipt['practice']['normal'] == 24
    assert receipt['practice']['examples'] == []
    plan = json.loads(practice_starts.plan_path(tmp_path/'v2v', 'B', 1).read_text())
    assert plan['rows'] == [] and plan['registry']['empty_bank'] is True


# ---------------------------------------------------------------- v2v_live.safety (matched A references)

def _safety_root(tmp_path, monkeypatch, *, vids=('v0', 'v1'), collisions=()):
    """A B-arm practice run root: frozen plan, A checkpoint/receipt, real v2v_live hooks."""
    from tools.coopmwm.accel_learner import save
    from tools.coopmwm.acceleration import Policy
    from tools.coopmwm import warm_health, v2v_live
    monkeypatch.setattr(warm_health, 'assert_running', lambda *a, **k: None)
    root = tmp_path/'v2v';root.mkdir(parents=True, exist_ok=True)
    bank_with_snapshots(root, 'B', vids)
    practice_starts.plan_for_update(root, 'B', 1, fraction=0.25, repeat_cap=2, count=6,
                                    training_seed=1)
    policy = Policy();opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = torch.Generator().manual_seed(1)
    save(root/'A_checkpoint_000.pt', policy, opt, rng, 0, 0)
    save(root/'A_checkpoint_001.pt', policy, opt, rng, 1, 0)
    (root/'A_UPDATE_001.json').write_text(json.dumps(dict(update=1, arm='A')))
    base = dict(budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)),
                accel_bounds=[-3., 2.6], runtime='rt', checkpoint='ck', workers=1,
                gpu_deadline=1e12, awr=dict(coefficient=1.))
    cfg = v2v_runner.prepare_intent(base, root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64,
                                    practice_starts=True)
    monkeypatch.setattr(bs, '_CACHE', {})
    bs.install(rb, bank_rule='per_seed_v1')
    return root, cfg, v2v_live.hooks(root, cfg)


def _trace(root, ident):
    import gzip
    p = root/'jobs'/ident;p.mkdir(parents=True, exist_ok=True)
    with gzip.open(p/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
        f.write(json.dumps(dict(time_s=500, vehicles=[]))+'\n')


def _store(root, ident, metrics, world):
    _trace(root, ident)
    rec = dict(id=ident, world=dict(seed=world), arm='B', steps=[fake_step(400)],
               metrics={'900': dict(metrics)})
    (root/'jobs'/ident/'RESULT.json').write_text(json.dumps(rec))
    return rec


def _practice_episodes(root, collisions=()):
    """The six practice episodes of the frozen plan, as the batch wave returned them."""
    plan = json.loads(practice_starts.plan_path(root, 'B', 1).read_text())
    calls = [practice_starts.wave_call(root, 'B', 'v2v1_warm_u001', 'v2v_train', row, i,
                                       dict(update=1, b_replay=True, v2v=True, v2v_dim=48))
             for i, row in enumerate(plan['rows'])]
    eps = []
    for call in calls:
        meta = call['extras']['practice']
        world = call['worlds'][0]['seed']
        hard = dict(hard_brake=0., collisions=0., teleports=0.)
        if meta['slot'] in collisions:
            hard['collisions'] = 1.
        rec = _store(root, f"{meta['slot']}_B_{world}", hard, world)
        eps.append(dict(rec, practice={k: v for k, v in meta.items() if k != 'snapshot'}))
    return eps, calls


def _capture_reference_wave(root, monkeypatch):
    """Records the real practice_references dispatch (accel_runner.wave seam) and materialises
    the reference receipts the composite guard reads from disk."""
    seen = {}

    def fake_wave(r, b, policies, ws, role, ident, phase, deterministic=True, extras=None,
                  used_seeds=None):
        seen.update(arms=sorted(policies), worlds=list(ws), role=role, ident=ident, phase=phase,
                    deterministic=deterministic, b=dict(b), batch=list(extras['practice_batch']),
                    extra_keys=sorted(extras))
        out = []
        for w, item in zip(ws, extras['practice_batch']):
            ident_ = f"{item['practice']['slot']}_A_{w['seed']}"
            _trace(Path(r), ident_)
            rec = dict(id=ident_, world=dict(seed=w['seed']), arm='A', steps=[fake_step(400)],
                       metrics={'900': dict(hard_brake=0., collisions=0., teleports=0.)})
            (Path(r)/'jobs'/ident_/'RESULT.json').write_text(json.dumps(rec))
            out.append(rec)
        return out

    monkeypatch.setattr(accel_runner, 'wave', fake_wave)
    return seen


def test_safety_runs_one_matched_a_reference_per_practice_episode(tmp_path, monkeypatch, ps):
    """v2v_live.safety (owner 2026-11-04 review fix): practice episodes get ONE matched A-policy
    reference from the SAME snapshot and CRN seed (one batch wave, arm's practice cost), the
    composite comparison runs on the exact pairs, and hard safety bites - no check is skipped."""
    from tools.coopmwm.test_composite_health_cpu import fixture as composite_fixture
    root, cfg, hooks = _safety_root(tmp_path, monkeypatch)
    composite_fixture(root)                                              # registered composite rule
    practice, calls = _practice_episodes(root, collisions={'v2v1_warm_u001_p01'})
    seen = _capture_reference_wave(root, monkeypatch)
    with pytest.raises(Exception) as raised:                             # hard safety: collision delta
        hooks['safety'](root, 'B', 1, practice, [])
    assert type(raised.value).__name__ == 'HealthPause'
    # ONE batch reference wave: arm A, same executor role, frozen pa slots (serving arm in the
    # family), the same snapshots
    assert seen['arms'] == ['A'] and seen['role'] == 'train'
    assert seen['ident'] == 'v2v1_warm_u001_B_practiceref'
    assert seen['phase'] == 'v2v_train' and seen['deterministic'] is False
    assert seen['b']['v2v'] is True
    assert {'update', 'v2v_dim', 'b_replay', 'practice_batch'} <= set(seen['extra_keys'])
    assert len(seen['worlds']) == 6 and len(seen['batch']) == 6
    for i, (w, item) in enumerate(zip(seen['worlds'], seen['batch'])):
        assert item['practice']['a_reference'] is True
        assert item['practice']['slot'] == f'v2v1_warm_u001_B_practiceref_pa{i+1:02d}'
        assert item['practice']['charged_sim_s'] == 1200                 # arm's practice cost
        assert item['branch']['time_s'] == 400
        assert 'A policy from t0' in item['practice']['policy']
    assert [w['seed'] for w in seen['worlds']] == [c['worlds'][0]['seed'] for c in calls]
    snaps = {sv.snapshot_sha(practice_starts.load_snapshot(root, 'B', row))
             for row in json.loads(practice_starts.plan_path(root, 'B', 1).read_text())['rows']}
    assert {i['practice']['snapshot_sha256'] for i in seen['batch']} <= snaps
    # the composite comparison ran on the six exact pairs; hard safety failed on the collision
    evidence = json.loads((root/'PAUSE_SAFETY_COMPOSITE_B_u001.json').read_text())['evidence']
    out = evidence['comparison']
    assert out['n'] == 6 and out['collision_teleport_hard_constraints'] is False
    assert out['release'] is False
    assert out['metrics']['collisions']['mean_A'] == 0.                  # references are clean
    assert abs(out['metrics']['collisions']['mean_right']-1./6) < 1e-12  # the practice collision


def test_safety_passes_when_practice_episodes_match_their_references(tmp_path, monkeypatch, ps):
    from tools.coopmwm.test_composite_health_cpu import fixture as composite_fixture
    root, cfg, hooks = _safety_root(tmp_path, monkeypatch)
    composite_fixture(root)
    practice, _ = _practice_episodes(root)
    _capture_reference_wave(root, monkeypatch)
    hooks['safety'](root, 'B', 1, practice, [])                          # no pause
    out = json.loads((root/'B_SAFETY_COMPOSITE_001.json').read_text())
    assert out['n'] == 6 and out['release'] is True
    assert out['collision_teleport_hard_constraints'] is True


def test_safety_still_requires_same_world_a_references_for_normal_episodes(tmp_path, monkeypatch, ps):
    root, cfg, hooks = _safety_root(tmp_path, monkeypatch)
    practice, _ = _practice_episodes(root)
    _capture_reference_wave(root, monkeypatch)
    normal = dict(id='v2v1_warm_u001_B_986100', world=dict(seed=986100), arm='B',
                  steps=[fake_step(400)], metrics={'900': dict(hard_brake=0., collisions=0.,
                                                               teleports=0.)})
    with pytest.raises(ValueError, match='missing same-world V2V A safety reference'):
        hooks['safety'](root, 'B', 1, practice+[normal], [])


def test_seed_collision_with_a_practice_snapshot_never_changes_the_normal_pairing(
        tmp_path, monkeypatch, ps):
    """Fix round 2 BLOCKING: a practice snapshot world that EQUALS a normal episode's TRAIN seed
    (TRAIN worlds repeat across updates; here the v0 snapshot world is exactly the registered
    update-1 world 986000) must never put the practice A reference into the same-world records.
    The normal episode keeps its own pairing - 1 collision vs its own same-world A with
    1 collision is a zero delta and must NOT pause (the polluted a_records would have paired it
    with the clean practice reference and tripped the hard collision constraint)."""
    from tools.coopmwm.test_composite_health_cpu import fixture as composite_fixture
    root, cfg, hooks = _safety_root(tmp_path, monkeypatch, vids=('v0',))   # worlds: {986000}
    composite_fixture(root)                                              # registered composite rule
    practice, calls = _practice_episodes(root)                           # clean practice pairs
    assert {c['worlds'][0]['seed'] for c in calls} == {986000}
    normal = _store(root, 'v2v1_warm_u001_B_986000',
                    dict(hard_brake=0., collisions=1., teleports=0.), 986000)
    _store(root, 'v2v1_warm_u001_A_986000',
           dict(hard_brake=0., collisions=1., teleports=0.), 986000)     # its OWN same-world A
    _capture_reference_wave(root, monkeypatch)
    hooks['safety'](root, 'B', 1, practice+[normal], [])                 # NO pause
    assert not (root/'PAUSE_SAFETY_COMPOSITE_B_u001.json').exists()
    out = json.loads((root/'B_SAFETY_COMPOSITE_001.json').read_text())
    assert out['n'] == 5 and out['release'] is True                       # 4 practice + 1 normal
    assert out['collision_teleport_hard_constraints'] is True
    assert out['metrics']['collisions']['mean_right'] == out['metrics']['collisions']['mean_A']
    assert out['worlds'].count(986000) == 5                               # collision is harmless


def test_matched_reference_uses_the_pre_update_a_checkpoint(tmp_path, monkeypatch, ps):
    """Fix round 2 (a): the matched practice reference must run the SAME policy version A's own
    update-u episodes ran - the PRE-update checkpoint u-1 (A_checkpoint_u is only written after
    A's update-u PPO step; A_UPDATE_u exists from that same step onward, so u-1 is always the
    checkpoint behind A's update-u reference episodes)."""
    from tools.coopmwm import v2v_runtime
    root, cfg, hooks = _safety_root(tmp_path, monkeypatch)
    practice, _ = _practice_episodes(root)
    loaded = []
    real = v2v_runtime.load_checkpoint
    def spy(path, config=None):
        loaded.append(Path(path).name)
        return real(path, config=config)
    monkeypatch.setattr(v2v_runtime, 'load_checkpoint', spy)
    _capture_reference_wave(root, monkeypatch)
    hooks['safety'](root, 'B', 1, practice, [])                          # no pause
    assert loaded == ['A_checkpoint_000.pt']                             # pre-update, not 001


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
    out = accel_runner.replacement_job(job, dict(worlds('train')[1], seed=986001),
                                       'v2v1_warm_u001_practice')
    assert out['id'] == 'v2v1_warm_u001_practice_B_986001' and out['replaces'] == job['id']
    assert 'practice' not in out and 'branch' not in out                 # no snapshot state carried
    normal = dict(id='x', arm='B', world=dict(worlds('train')[0]), path='x/x')
    out = accel_runner.replacement_job(normal, dict(worlds('train')[1], seed=986001), 'ident')
    assert 'practice' not in out                                         # normal jobs unchanged


def test_one_failed_practice_episode_in_the_batch_wave_is_replaced_not_escalated(
        tmp_path, monkeypatch):
    """Fix 3 evidence: the whole practice batch is ONE wave, so run_train_jobs' REAL accounting
    replaces a single failed practice episode with a fresh normal start and does NOT trip the
    >20% escalation that a 1-job wave would trip on its first failure."""
    from concurrent.futures import ThreadPoolExecutor
    from tools.coopmwm.accel_protocol import worlds
    ws = [dict(worlds('train')[i], seed=986000+i) for i in range(5)]
    spare = dict(worlds('train')[0], seed=986005)                        # the unused cell-0 world
    jobs = [dict(id=f'v2v1_warm_u001_p{i+1:02d}_B_{w["seed"]}', arm='B', world=w,
                 practice=dict(t0=400), path=str(tmp_path/'jobs'/f'p{i+1:02d}'))
            for i, w in enumerate(ws)]
    used = {w['seed'] for w in ws}
    reserved = []
    monkeypatch.setattr(accel_runner, 'reserve',
                        lambda root, new_jobs, phase: reserved.extend(
                            j['id'] for j in new_jobs))
    def fake_episode(job):
        if job['id'] == 'v2v1_warm_u001_p01_B_986000':
            raise ValueError('snapshot episode failed')
        return dict(id=job['id'], world=job['world'], arm=job['arm'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = accel_runner.run_train_jobs(tmp_path, jobs, 'v2v_train',
                                              'v2v1_warm_u001_practice', used, set(),
                                              pool.submit, fake_episode)
    assert len(results) == 5                                             # 4 originals + 1 replacement
    replacement, = [r for r in results if r['id'].startswith('v2v1_warm_u001_practice_B_')]
    assert replacement['world']['seed'] == spare['seed']                 # a fresh TRAIN world
    assert 'practice' not in replacement and 'branch' not in replacement  # a NORMAL fresh start
    receipt = json.loads((tmp_path/'jobs'/'v2v1_warm_u001_p01_B_986000'/
                          'EPISODE_FAILED.json').read_text())
    assert receipt['replacement'] == replacement['id']
    assert receipt['excluded_from_update'] is True
    assert reserved == [replacement['id']]
    assert not (tmp_path/'STOP.json').exists()                           # batch accounting, no STOP


def test_practice_replacement_never_duplicates_a_normal_world_of_the_update(tmp_path, monkeypatch):
    """Fix round 2 (c): the update's normal worlds are 'used' for the practice wave - a failed
    practice episode's fresh replacement skips the world the normal wave already closed
    (986001) and takes the next unused world of the cell instead."""
    from concurrent.futures import ThreadPoolExecutor
    from tools.coopmwm.accel_protocol import worlds
    ws = [dict(worlds('train')[0], seed=986000) for _ in range(5)]       # one batch practice wave
    jobs = [dict(id=f'v2v1_warm_u001_p{i+1:02d}_B_986000', arm='B', world=w,
                 practice=dict(t0=400), path=str(tmp_path/'jobs'/f'p{i+1:02d}'))
            for i, w in enumerate(ws)]
    def failing(job):
        if job['id'] == 'v2v1_warm_u001_p01_B_986000':
            raise ValueError('snapshot episode failed')
        return dict(id=job['id'], world=job['world'], arm=job['arm'])
    normal_world = 986001                                                # closed by the normal wave
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = accel_runner.run_train_jobs(tmp_path, jobs, 'v2v_train',
                                              'v2v1_warm_u001_practice',
                                              {986000, normal_world}, set(), pool.submit, failing)
    replacement, = [r for r in results if r['id'].startswith('v2v1_warm_u001_practice_B_')]
    assert replacement['world']['seed'] == 986002                        # skips the normal world
    # control: without the normal seed in 'used' the replacement WOULD have duplicated it
    (tmp_path/'jobs'/'v2v1_warm_u001_p01_B_986000'/'EPISODE_FAILED.json').unlink()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = accel_runner.run_train_jobs(tmp_path, jobs, 'v2v_train',
                                              'v2v1_warm_u001_practice', {986000}, set(),
                                              pool.submit, failing)
    duplicate, = [r for r in results if r['id'].startswith('v2v1_warm_u001_practice_B_')]
    assert duplicate['world']['seed'] == normal_world


def test_wave_unions_passed_seeds_into_the_train_replacement_used_set(tmp_path, monkeypatch):
    """Fix round 2 (c) seam: accel_runner.wave unions ``used_seeds`` into the train wave's
    replacement 'used' set (the practice wave passes its update's closed normal worlds)."""
    from tools.coopmwm.acceleration import Policy
    from tools.coopmwm.accel_protocol import worlds
    (tmp_path/'CHARGES.jsonl').write_text('');(tmp_path/'ACCOUNTING.claim').touch()
    import tools.coopmwm.runner as runner_mod
    captured = {}
    def fake_run_train_jobs(root_, jobs, phase, ident, used, precharged, submit, episode):
        captured['used'] = set(used);return []
    monkeypatch.setattr(accel_runner, 'run_train_jobs', fake_run_train_jobs)
    monkeypatch.setattr(accel_runner, 'reserve', lambda root_, jobs, phase: None)
    monkeypatch.setattr(runner_mod, 'admission', lambda cfg: None)
    b = dict(workers=1, gpu_deadline=1e12, runtime='rt', checkpoint='ck',
             accel_bounds=[-3., 2.6])
    ws = [dict(worlds('train')[0], seed=986000)]
    accel_runner.wave(tmp_path, b, {'B': Policy()}, ws, 'train', 'ident', 'v2v_train', False,
                      None, used_seeds={986001, 986002})
    assert captured['used'] == {986000, 986001, 986002}
    accel_runner.wave(tmp_path, b, {'B': Policy()}, ws, 'train', 'ident', 'v2v_train', False, None)
    assert captured['used'] == {986000}                                  # unchanged without the kwarg
    b = dict(workers=1, gpu_deadline=1e12, runtime='rt', checkpoint='ck',
             accel_bounds=[-3., 2.6])


def test_guided_replayed_plan_start_tick_with_no_proposal_is_guarded(tmp_path):
    """accel_plant fix: a replayed snapshot tick can carry recorded plan_start behaviour with
    proposal None (a practice episode never proposes a new plan) - no TypeError, and no
    backpressure record is queued for a proposal that does not exist."""
    from tools.coopmwm.acceleration import Policy
    snap = snapshot(t0=417, world_seed=986000, gen_seed=5)               # 417: no %10 overwrite
    t0sample = dict(snap['samples']['417'])
    t0sample['behaviour'] = dict(t0sample['behaviour'], executed=True, plan_start=True,
                                 plan_id='p')
    snap['samples'] = dict(snap['samples'], **{'417': t0sample})
    weights = Policy().state_dict()
    job = practice_job(tmp_path, snap, weights, name='jobg')
    job.update(guided=True, arm='B', update=1, queue_root=str(tmp_path))
    result = run_episode_job(job)
    assert result['id'] == 'jobg' and result['practice']['t0'] == 417
    assert not (tmp_path/'guided_starts').exists()                       # proposal None: nothing queued


def test_practice_on_against_the_real_full_budget_needs_a_registered_transfer(tmp_path):
    """The deployed budget allocates every sim-s of 17,471,840: practice ON must move exactly the
    quoted increase from a real donor phase (MAIN-registered), never from 'superseded'."""
    full = dict(launch.V2V_PHASES, paired_branch=1297400, interim_u8_test=900000, superseded=811640)
    full['train'] = launch.CEILING_SIM_S - sum(full.values())
    base = dict(budget=dict(total=launch.CEILING_SIM_S, phases=full))
    kw = dict(root=tmp_path/'v2v', legacy_root=tmp_path/'legacy', source_root=tmp_path/'src',
              common_checkpoint_sha256='a'*64, practice_starts=True)
    with pytest.raises(ValueError, match='registered budget_transfer'):
        v2v_runner.prepare_intent(base, **kw)
    q = launch.practice_reference_quote(0.25)
    for bad in (dict(source='superseded', amount=q, amendment_sha256='a'*64),
                dict(source='interim_u8_test', amount=q-1, amendment_sha256='a'*64),
                dict(source='interim_u8_test', amount=q),
                dict(source='nope', amount=q, amendment_sha256='a'*64)):
        with pytest.raises(ValueError):
            v2v_runner.prepare_intent(base, budget_transfer=bad, **kw)
    cfg = v2v_runner.prepare_intent(base, budget_transfer=dict(source='interim_u8_test', amount=q,
                                    amendment_sha256='b'*64), **kw)
    phases = cfg['budget']['phases']
    assert phases['v2v_train'] == launch.V2V_PHASES['v2v_train'] + q == 2419200
    assert phases['interim_u8_test'] == 900000 - q and sum(phases.values()) == launch.CEILING_SIM_S
    assert cfg['budget']['transfers'] == [dict(source='interim_u8_test', target='v2v_train', amount=q,
        amendment_sha256='b'*64, rule='practice_starts matched A references; total unchanged')]
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True)
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
