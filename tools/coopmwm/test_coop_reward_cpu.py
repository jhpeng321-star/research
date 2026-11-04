"""CPU tests: the registered coop_reward option (c4/coop-reward, DEFAULT OFF).

Off identity (same episode object, no receipt key, PPO update inputs unchanged); the
neighbourhood uses ONLY observable frame vehicles (CAV and HDV alike, 100 m V2V range); the
registered w + provenance are required when enabled; the per-update receipt logs the own vs
team components; B and R are pinned identical; A is unaffected unless explicitly included.
No SUMO, no GPU, no ssh.
"""
import copy
import json
import math
import os
from unittest.mock import Mock

import pytest
import torch

from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import coop_reward as cr
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import v2v_launch3 as launch
from tools.coopmwm import v2v_runner
from tools.coopmwm.protocol import sha
from tools.coopmwm.test_bank_perseed_cpu import ps  # noqa: F401 (autouse budget fixture)

W = .5
PROVENANCE = 'owner 2026-11-04 coop-reward registration'
COOP = dict(coop_reward=True, coop_reward_weight=W, coop_reward_provenance=PROVENANCE)


# ---------------------------------------------------------------- frames and episodes

def frame(spec):
    """Observable receipt rows (exact runtime whitelist fields)."""
    out = []
    for vid, s, v, cav in spec:
        out.append(dict(vid=vid, s_m=s, v_mps=v, lane=0, is_cav=cav, length_m=5.))
    return out


def episode(frames, rewards):
    steps = [dict(time_s=400+i, rows=rows, reward=rewards[i],
                  sample=dict(value=0., logp=0., context=dict(members=['c0']),
                              behaviour=dict(route='policy', executed=False,
                                             sample_probability=0., q=0., regime='ordinary')))
             for i, rows in enumerate(frames)]
    return dict(steps=steps, world=dict(seed=1, cell='x'))


# ---------------------------------------------------------------- off identity

def test_off_returns_the_same_episodes_object_and_reports_none():
    frames = [frame((('c0', 0., 20., True),)), frame((('c0', 5., 10., True),))]
    eps = [episode(frames, [-1., -1.])]
    for cfg in ({}, None, dict(coop_reward=False), dict(coop_reward=False, coop_reward_weight=W)):
        assert cr.settings(cfg) is None
        assert cr.enabled(cfg, 'A') is cr.enabled(cfg, 'B') is cr.enabled(cfg, 'R') is False
        got, report = cr.training_episodes(cfg, 'B', eps)
        assert got is eps and report is None
    assert copy.deepcopy(eps) == eps                      # nothing mutated above


def test_arm_not_enabled_is_identical_even_when_the_option_is_on():
    eps = [episode([frame((('c0', 0., 0., True),)), frame((('c0', 5., 0., True),))], [-1., -1.])]
    got, report = cr.training_episodes(COOP, 'A', eps)     # default arms are B and R
    assert got is eps and report is None


# ---------------------------------------------------------------- observable neighbourhood

def test_neighbourhood_range_and_hdv_inclusion_are_observable_only():
    limit = cr.SPEED_LIMIT_MPS
    rows = frame((('c0', 0., limit, True),      # ego at the limit: own deficit ignored (not a neighbour)
                  ('h0', 100., 0., False),      # HDV at exactly 100 m: IN range, deficit 1
                  ('h1', -100., limit, False),  # HDV at exactly 100 m on the other side: deficit 0
                  ('c1', 100.5, 0., True),      # 100.5 m: OUT of the observable range
                  ('c2', 500., 0., True)))      # far CAV: out of range (own empty neighbourhood)
    term, egos, empty = cr.neighbourhood_term(rows)
    assert egos == 3 and empty == 1                          # c0, c1 and c2 are egos; c2 senses nobody
    # c0's neighbour mean = (h0 deficit 1 + h1 deficit 0)/2; c1 senses only h0 (deficit 1);
    # c2's empty neighbourhood counts as 0 -> mean over the THREE observable egos
    assert term == pytest.approx((0.5 + 1. + 0.)/3)
    # no lane filter (radio range) and no is_cav filter on neighbours: HDV h0 counts
    rows2 = frame((('c0', 0., limit, True), ('h0', 30., limit/2, False)))
    assert cr.neighbourhood_term(rows2)[0] == pytest.approx(.5)
    # only the frame fields are used: a vehicle absent from the frame never contributes
    assert cr.neighbourhood_term(frame((('c0', 0., limit, True),))) == (0., 1, 1)
    assert cr.neighbourhood_term([]) == (0., 0, 0)


def test_team_term_uses_the_post_step_frame_and_counts_the_final_fallback():
    limit = cr.SPEED_LIMIT_MPS
    f0 = frame((('c0', 0., 0., True),))                      # pre frame: deficit 1 (NOT used for t0)
    f1 = frame((('c0', 5., limit/2, True), ('h0', 35., limit/2, False)))  # post frame: mean .5
    eps = [episode([f0, f1], [-1., -2.])]
    new, report = cr.training_episodes(COOP, 'B', eps)
    assert new is not eps and new[0] is not eps[0]
    # step 400 charges frame f1 (mean neighbour deficit .5 -> +w*.5); step 401 falls back to f1
    assert [round(s['reward'], 12) for s in new[0]['steps']] == [-1. + W*.5, -2. + W*.5]
    assert report['own_frame_team_steps'] == 1 and report['steps'] == 2
    assert eps[0]['steps'][0]['reward'] == -1.               # the receipts stay untouched


def test_registered_w_scales_the_team_term_and_components_sum_exactly():
    limit = cr.SPEED_LIMIT_MPS
    frames = [frame((('c0', 0., 0., True), ('h0', 10., 0., False))),
              frame((('c0', 5., limit/4, True), ('h0', 15., limit/4, False)))]
    eps = [episode(frames, [-3., -4.])]
    for w in (0.25, 1., 2.5):
        new, report = cr.training_episodes(dict(COOP, coop_reward_weight=w), 'B', eps)
        raw = sum(max(0., 1.-r['v_mps']/limit) for r in frames[1])/2
        assert raw == pytest.approx(.75)
        assert [s['reward'] for s in new[0]['steps']] == [pytest.approx(-3. + w*.75),
                                                          pytest.approx(-4. + w*.75)]
        assert report['weight'] == w and report['provenance'] == PROVENANCE
        assert report['own_reward_sum_veh_s'] == pytest.approx(-7.)
        assert report['team_addition_sum_veh_s'] == pytest.approx(2*w*.75)
        assert report['team_addition_sum_veh_s'] == pytest.approx(
            report['team_neighbourhood_mean_veh_s']*2*w)
        assert report['team_neighbourhood_mean_veh_s'] == pytest.approx(.75)
        assert report['own_reward_mean_veh_s'] == pytest.approx(-3.5)
        assert report['arms'] == ['B', 'R'] and report['background_term'] is None


# ---------------------------------------------------------------- registration: w + provenance + arms

def test_w_and_provenance_are_required_when_enabled_and_never_defaulted():
    for bad in (dict(coop_reward=True),
                dict(coop_reward=True, coop_reward_provenance=PROVENANCE),
                dict(coop_reward=True, coop_reward_weight=0., coop_reward_provenance=PROVENANCE),
                dict(coop_reward=True, coop_reward_weight=-.5, coop_reward_provenance=PROVENANCE),
                dict(coop_reward=True, coop_reward_weight=float('nan'),
                     coop_reward_provenance=PROVENANCE),
                dict(coop_reward=True, coop_reward_weight=True, coop_reward_provenance=PROVENANCE),
                dict(coop_reward=True, coop_reward_weight=W, coop_reward_provenance=''),
                dict(coop_reward=True, coop_reward_weight=W, coop_reward_provenance='   ')):
        with pytest.raises(ValueError):
            cr.settings(bad)
    # OFF never needs a w (and a stray weight alone stays OFF)
    assert cr.settings(dict(coop_reward_weight=W, coop_reward_provenance=PROVENANCE)) is None


def test_br_pinned_identical_and_a_only_by_explicit_inclusion():
    have = cr.settings(COOP)
    assert have['arms'] == ('B', 'R') and have['weight'] == W
    assert cr.enabled(COOP, 'B') and cr.enabled(COOP, 'R') and not cr.enabled(COOP, 'A')
    wide = cr.settings(dict(COOP, coop_reward_arms=['A', 'B', 'R']))
    assert wide['arms'] == ('A', 'B', 'R') and cr.enabled(dict(COOP, coop_reward_arms=['A', 'B', 'R']), 'A')
    for bad in (['B'], ['R'], ['A'], ['A', 'B'], ['B', 'R', 'A', 'X'], (), ['B', 'B', 'R'],
                'BR', dict(B=True, R=True)):
        with pytest.raises(ValueError):
            cr.settings(dict(COOP, coop_reward_arms=bad))


def test_producer_writes_keys_only_when_enabled_and_refuses_an_incomplete_registration(tmp_path):
    base = dict(budget={})
    kw = dict(root=tmp_path/'v2v', legacy_root=tmp_path/'legacy', source_root=tmp_path/'src',
              common_checkpoint_sha256='a'*64)
    off = v2v_runner.prepare_intent(base, **kw)
    assert not any(key.startswith('coop_') for key in off)          # OFF: byte-identical output
    assert v2v_runner.prepare_intent(base, **kw) == off
    with pytest.raises(ValueError, match='coop_reward_weight'):
        v2v_runner.prepare_intent(base, coop_reward=True, **kw)
    with pytest.raises(ValueError, match='provenance'):
        v2v_runner.prepare_intent(base, coop_reward=True, coop_reward_weight=W, **kw)
    with pytest.raises(ValueError, match='B and R'):
        v2v_runner.prepare_intent(base, **dict(kw, coop_reward=True, coop_reward_weight=W,
                                               coop_reward_provenance=PROVENANCE,
                                               coop_reward_arms=['B']))
    on = v2v_runner.prepare_intent(base, coop_reward=True, coop_reward_weight=W,
                                   coop_reward_provenance=PROVENANCE, **kw)
    assert on['coop_reward'] is True and on['coop_reward_weight'] == W
    assert on['coop_reward_provenance'] == PROVENANCE and 'coop_reward_arms' not in on
    assert cr.settings(on)['arms'] == ('B', 'R')
    assert v2v_runner.prepare_intent(base, coop_reward=True, coop_reward_weight=1,
                                     coop_reward_provenance=PROVENANCE,
                                     coop_reward_arms=['A', 'B', 'R'], **kw)['coop_reward_arms'] == ['A', 'B', 'R']


def test_launch_binding_refuses_a_malformed_coop_registration_and_passes_off_through(tmp_path):
    kw = dict(root=tmp_path/'v2v', legacy_root=tmp_path/'legacy', source_root=tmp_path/'src',
              common_checkpoint_sha256='a'*64)
    off = dict(v2v_runner.prepare_intent(base := dict(budget={}), **kw),
               launch_authorized=True, phase_funding_authorized=True,
               budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    assert launch.bind_runtime(dict(config=off), root=tmp_path/'v2v', source_root=tmp_path/'src') is off
    for bad in (dict(coop_reward=True),                                # missing w and provenance
                dict(coop_reward=True, coop_reward_weight=W, coop_reward_provenance=None),
                dict(coop_reward=True, coop_reward_weight=W, coop_reward_provenance=PROVENANCE,
                     coop_reward_arms=['B'])):
        with pytest.raises(launch.Refused):
            launch.bind_runtime(dict(config=dict(off, **bad)), root=tmp_path/'v2v',
                                source_root=tmp_path/'src')
    on = dict(off, **COOP)
    assert launch.bind_runtime(dict(config=on), root=tmp_path/'v2v', source_root=tmp_path/'src') is on


# ---------------------------------------------------------------- one sealed arm update

def run_one_update(tmp_path, monkeypatch, *, arm='B', coop=None, subroot='v2v'):
    """One smoke update of v2v_runner.run_arm with a recorded wave; returns
    (raw_wave, update_episodes, update_caller, receipt). The wave steps carry observable rows;
    the PPO-update entry point is spied (exact_update for B/R, update for A)."""
    from tools.coopmwm.accel_learner import save
    from tools.coopmwm.acceleration import Policy
    for name, value in (('nice', lambda inc: 0), ('getpriority', lambda which, who: 0),
                        ('PRIO_PROCESS', 0)):
        monkeypatch.setattr(os, name, value, raising=False)
    policy = Policy();opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = torch.Generator().manual_seed(1)
    root = tmp_path/subroot;root.mkdir(parents=True, exist_ok=True)
    ck = root/f'{arm}_checkpoint_000.pt'
    if not ck.exists():                                                  # u0 is write-once
        save(ck, policy, opt, rng, 0, 0)
    base = dict(budget=dict(total=17471840, phases=dict(v2v_train=2073600)),
                accel_bounds=[-3., 2.6], runtime='rt', checkpoint='ck', workers=1,
                gpu_deadline=1e12, awr=dict(coefficient=1.))
    kw = dict(root=tmp_path/subroot, legacy_root=tmp_path/'legacy', source_root=tmp_path/'src',
              common_checkpoint_sha256=sha(ck))
    cfg = (v2v_runner.prepare_intent(base, **kw) if coop is None else
           v2v_runner.prepare_intent(base, **dict(kw, coop_reward=True,
                                                  coop_reward_weight=coop['weight'],
                                                  coop_reward_provenance=coop['provenance'],
                                                  **({'coop_reward_arms': coop['arms']}
                                                     if 'arms' in coop else {}))))
    cfg.update(stop_after_update=1, A_stop_after_update=1,
               launch_amendment_sha256='l'*64, funding_amendment_sha256='f'*64)
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
    rows = [frame((('c0', 0., 0., True), ('h0', 30., 0., False))),      # step 400 (pre frame)
            frame((('c0', 5., 0., True), ('h0', 35., 0., False)))]      # step 401 = post frame of 400
    raw_wave = []

    def fake_step(i):
        return dict(time_s=400+i, rows=rows[i], reward=-1.,
                    sample=dict(value=0., logp=0., context=dict(members=['c0']),
                                behaviour=dict(route='policy', executed=False,
                                               sample_probability=0., q=0., regime='ordinary')))

    def fake_wave(r, b, policies, ws, role, ident, phase, deterministic=True, extras=None,
                  used_seeds=None):
        for w in ws:
            raw_wave.append(dict(id=f'{ident}_{arm}_{w["seed"]}', world=w, arm=arm,
                                 steps=[fake_step(0), fake_step(1)],
                                 metrics={'900': dict(hard_brake=0.)}))
        return raw_wave

    captured = dict(episodes=None, caller=None)

    def spy(caller):
        def wrapped(policy_, optimizer_, episodes_, *a, **k):
            captured['episodes'] = episodes_
            captured['caller'] = caller
            return []
        return wrapped

    from tools.coopmwm import warm_runner as wr
    from tools.coopmwm import exact_grade as xg
    monkeypatch.setattr(wr, 'wave', fake_wave)
    monkeypatch.setattr(xg, 'exact_update', spy('exact_update'))
    monkeypatch.setattr(wr, 'update', spy('update'))
    real_evidence = wr.evidence
    def spy_evidence(eps_):
        captured['raw_evidence'] = eps_
        return real_evidence(eps_)
    monkeypatch.setattr(wr, 'evidence', spy_evidence)
    from tools.coopmwm import search_learning
    monkeypatch.setattr(search_learning, 'exact_update', lambda *a, **k: None)
    monkeypatch.setattr(bs, '_CACHE', {})
    unwrapped = getattr(rb.confirmations, 'original', rb.confirmations)
    bs.install(rb, bank_rule='per_seed_v1')
    monkeypatch.setattr(rb, 'confirmations', unwrapped)   # install is process-wide: undo at teardown
    v2v_runner.run_arm(root, arm, live, release, hooks)
    receipt = json.loads((root/f'{arm}_UPDATE_001.json').read_text())
    return raw_wave, captured, receipt


def test_off_update_seals_the_raw_episodes_and_no_coop_receipt_key(tmp_path, monkeypatch):
    raw_wave, captured, receipt = run_one_update(tmp_path, monkeypatch, arm='B')
    assert captured['caller'] == 'exact_update'
    assert captured['episodes'] is raw_wave                    # SAME objects, not copies
    assert 'coop_reward' not in receipt                        # OFF: byte-identical receipt
    assert receipt['reward_source'] == 'SUMO source-inclusive team J only'
    assert all(s['reward'] == -1. for e in captured['episodes'] for s in e['steps'])
    assert captured['raw_evidence'] is raw_wave                # diagnostics keep the raw rewards


def test_on_update_transforms_only_the_ppo_rewards_and_logs_components(tmp_path, monkeypatch):
    coop = dict(weight=W, provenance=PROVENANCE)
    raw_wave, captured, receipt = run_one_update(tmp_path, monkeypatch, arm='B', coop=coop)
    assert captured['caller'] == 'exact_update'
    coop_eps = captured['episodes']
    assert coop_eps is not raw_wave and len(coop_eps) == len(raw_wave) == 24
    # the post frame of step 400 (and the fallback frame of step 401) sees BOTH vehicles stopped:
    # neighbour deficit = 1 (HDV h0 at 30 m) -> every charged step gains exactly +w*1
    assert all(s['reward'] == pytest.approx(-1. + W) for e in coop_eps for s in e['steps'])
    assert all(s['reward'] == -1. for e in raw_wave for s in e['steps'])       # receipts untouched
    report = receipt['coop_reward']
    assert report['weight'] == W and report['provenance'] == PROVENANCE
    assert report['arms'] == ['B', 'R']
    assert report['steps'] == 48 and report['own_frame_team_steps'] == 24
    assert report['own_reward_sum_veh_s'] == pytest.approx(-48.)
    assert report['team_addition_sum_veh_s'] == pytest.approx(24*2*W)
    assert report['team_neighbourhood_mean_veh_s'] == pytest.approx(1.)
    assert report['empty_neighbourhood_egos'] == 0 and report['egos'] == 48
    assert report['neighbourhood_range_m'] == 100. and report['speed_limit_mps'] == cr.SPEED_LIMIT_MPS
    # the critic diagnostics still consumed the RAW episode list (never the coop copy)
    assert captured['raw_evidence'] is raw_wave


def test_a_is_unaffected_by_the_default_registration_and_transformed_only_when_included(tmp_path, monkeypatch):
    raw_wave, captured, receipt = run_one_update(tmp_path, monkeypatch, arm='A',
                                                 coop=dict(weight=W, provenance=PROVENANCE))
    assert captured['caller'] == 'update'
    assert captured['episodes'] is raw_wave                    # A keeps the raw team-J rewards
    assert 'coop_reward' not in receipt
    included = dict(weight=W, provenance=PROVENANCE, arms=['A', 'B', 'R'])
    raw_wave, captured, receipt = run_one_update(tmp_path, monkeypatch, arm='A', coop=included,
                                                 subroot='v2v_included')
    assert captured['episodes'] is not raw_wave
    assert all(s['reward'] == pytest.approx(-1. + W) for e in captured['episodes'] for s in e['steps'])
    assert receipt['coop_reward']['arms'] == ['A', 'B', 'R']
    assert all(s['reward'] == -1. for e in raw_wave for s in e['steps'])


# ---------------------------------------------------------------- constants pin the runtime contract

def test_neighbourhood_constants_mirror_the_registered_observable_contract():
    from wms.control.contract import SENSOR_RANGE_M, SPEED_LIMIT_MPS
    from tools.coopmwm.v2v_obs import V2V_RANGE_M
    assert cr.NEIGHBOURHOOD_RANGE_M == V2V_RANGE_M == SENSOR_RANGE_M == 100.0
    assert cr.SPEED_LIMIT_MPS == SPEED_LIMIT_MPS == 31.29
    assert math.isfinite(cr.SPEED_LIMIT_MPS) and cr.SPEED_LIMIT_MPS > 0
