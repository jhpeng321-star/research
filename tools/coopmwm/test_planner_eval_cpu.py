"""CPU tests for the c4/planner-eval dry harness (Track 1 T1D; no SUMO, no GPU, no ssh/remote).

Covers the owner 2026-11-04 requirements:
  registration - the seed list comes ONLY from the registration file passed in: strict schema
                 (unknown keys refused), registered values pinned (60 s cadence, p_star 0.8,
                 full 300 s look-ahead, 1200 s episode, registered bootstrap triple), seeds
                 unique positive ints, normalized deterministically; every path-like string
                 anywhere inside is guarded (sealed/holdout/heldout/cal_split/chunk5/j4b/inc1,
                 hetdata_truth, vType/driver/vehicle_params truth files refused);
  paired jobs  - paired worlds: one job per world x arm, identical world dict + seed in BOTH
                 arms, only the PLANNER arm carries the planner config + calibration reference;
  quote        - 2 arms x worlds x episode sim-s, dry_run_sim_s = 0, WM rollouts quoted
                 separately in WM-s (CPU inference, never SUMO sim-s);
  receipts     - sealed schema (exact_grade.verify_sealed verifies them unchanged): J, complete
                 non-negative safety counters, planner gate trace on the registered cadence
                 (fired => P >= p_star, look-ahead never shortened, MAPPO arm has no trace);
  safety       - per-world deltas: collisions/teleports hard constraints, registered MAIN
                 2026-11-03 04:26 combined hard_brake + TTC<1.5 s rule (offset allowed);
  analysis     - completeness enforced, registered paired whole-world bootstrap (10,000,
                 RNG 20261101, percentile 95% CI, two-sided CI-inversion p) reused from
                 branch_eval, per-counter safety CIs, sealed analysis record;
  dry run      - writes ONLY the sealed plan + job specs into a fresh directory, refuses a
                 non-empty/forbidden output dir, charges new_sim_s = 0; the module hard-codes
                 no seed list and never touches the sealed panel machinery.
"""
import json
from pathlib import Path

import pytest

from tools.coopmwm import planner_eval as pe

SEEDS = (777001, 777002, 777003, 777004)      # test-local seeds; the module itself has none
SAFETY = dict(collisions=0., teleports=0., ttc_below_3_vs=2., ttc_below_1p5_events=3.,
              hard_brake=1.)
TRACE = [dict(t_s=60, p_better=.9, fired=True, family='wall', members=[1, 2], lookahead_s=300)]


# ---------------------------------------------------------------- fixtures/helpers

def reg(tmp_path, seeds=SEEDS, **over):
    d = dict(schema=pe.REGISTRATION_SCHEMA,
             worlds=[dict(seed=s, demand=2400, penetration=.3) for s in seeds],
             planner=dict(decision_interval_s=60, p_star=.8, lookahead_s=300, k_candidates=8),
             episode=dict(end_s=1200),
             analysis=dict(n_boot=10000, boot_seed=20261101, alpha=.05))
    d.update(over)
    path = tmp_path/'registration.json'
    path.write_text(json.dumps(d), encoding='utf-8')
    return str(path)


def plan(tmp_path, seeds=SEEDS, **over):
    return pe.build_plan(pe.load_registration(reg(tmp_path, seeds, **over)))


def job(plan_, seed, arm):
    return next(j for j in pe.jobs(plan_) if j['world']['seed'] == seed and j['arm'] == arm)


def receipt(plan_, seed, arm=pe.TREATMENT, *, J=100., safety=None, trace='auto', outcome=None):
    out = dict(J=float(J), safety=dict(SAFETY, **(safety or {})),
               planner_trace=TRACE if trace == 'auto' and arm == pe.TREATMENT
               else ([] if trace == 'auto' else list(trace or ())),
               executor='cpu-test', result_ref='fake://result')
    out.update(outcome or {})
    return pe.make_receipt(job(plan_, seed, arm), out)


def full_records(plan_, *, planner_J=lambda s: 100.-30., mappo_J=lambda s: 100.,
                 safety=None, extra=None):
    out = [receipt(plan_, s, pe.TREATMENT, J=planner_J(s), safety=safety)
           for s in (w['seed'] for w in plan_['worlds'])]
    out += [receipt(plan_, s, pe.COMPARATOR, J=mappo_J(s))
            for s in (w['seed'] for w in plan_['worlds'])]
    return out + list(extra or ())


# ---------------------------------------------------------------- budget quote

def test_quote_math_paired_full_episodes_and_dry_zero():
    q = pe.budget_quote(4)
    assert q['per_arm_sim_s'] == 4*1200 == 4800                     # full paired episode, both arms
    assert q['total_sim_s'] == 2*4*1200 == 9600
    assert q['decisions_per_episode'] == 1200//60 == 20
    assert q['lookahead_s'] == 300 and q['arms'] == ['MAPPO', 'PLANNER']
    assert q['planner_wm_rollout_s'] is None and q['dry_run_sim_s'] == 0


def test_quote_wm_rollouts_are_cpu_inference_not_sumo():
    q = pe.budget_quote(4, k_candidates=8)
    assert q['planner_wm_rollout_s'] == 4*20*(8+1)*300 == 216000    # worlds x decisions x (K+1) x H
    assert q['total_sim_s'] == 9600                                 # unchanged by the WM quote
    with pytest.raises(ValueError, match='multiple'):
        pe.budget_quote(2, episode_end_s=905)


# ---------------------------------------------------------------- registration file

def test_registration_loads_and_normalizes(tmp_path):
    r = pe.load_registration(reg(tmp_path))
    assert [w['seed'] for w in r['worlds']] == sorted(SEEDS)
    assert r['worlds'][0] == dict(seed=SEEDS[0], demand=2400., penetration=.3)
    assert r['planner'] == dict(decision_interval_s=60, p_star=.8, lookahead_s=300, k_candidates=8)
    assert r['episode'] == dict(end_s=1200)
    assert r['analysis'] == dict(n_boot=10000, boot_seed=20261101, alpha=.05)


def test_registration_order_and_spelling_do_not_change_the_digest(tmp_path):
    a = pe.build_plan(pe.load_registration(reg(tmp_path, seeds=(777002, 777001))))
    (tmp_path/'other.json').write_text(json.dumps(
        dict(schema=pe.REGISTRATION_SCHEMA,
             worlds=[dict(seed=777001, demand=2400.0, penetration=0.3),
                     dict(seed=777002, demand=2400, penetration=.3)],
             planner=dict(decision_interval_s=60, p_star=.8, lookahead_s=300, k_candidates=8),
             episode=dict(end_s=1200),
             analysis=dict(n_boot=10000, boot_seed=20261101, alpha=.05))), encoding='utf-8')
    b = pe.build_plan(pe.load_registration(str(tmp_path/'other.json')))
    assert a['registration_sha256'] == b['registration_sha256']
    assert a['plan_sha256'] == b['plan_sha256']
    other = load(str(tmp_path/'other.json'))
    other['planner'].pop('k_candidates')
    assert pe.build_plan(other)['registration_sha256'] != b['registration_sha256']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@pytest.mark.parametrize('key,value', [
    ('p_star', .9), ('decision_interval_s', 30), ('lookahead_s', 150),
])
def test_registration_refuses_unregistered_planner_values(tmp_path, key, value):
    d = load(reg(tmp_path))
    d['planner'][key] = value
    p = tmp_path/'bad_planner.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises(ValueError, match='registered'):
        pe.load_registration(str(p))


@pytest.mark.parametrize('key,value', [
    ('end_s', 900), ('n_boot', 2000), ('boot_seed', 1), ('alpha', .1),
])
def test_registration_refuses_unregistered_episode_or_analysis_values(tmp_path, key, value):
    d = load(reg(tmp_path))
    (d['episode'] if key == 'end_s' else d['analysis'])[key] = value
    p = tmp_path/'bad_analysis.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises(ValueError, match='registered'):
        pe.load_registration(str(p))


def test_registration_strict_keys_and_schema(tmp_path):
    good = load(reg(tmp_path))
    cases = [('top', good, 'surprise'), ('analysis', good['analysis'], 'n'),
             ('episode', good['episode'], 'warmup_s'), ('planner', good['planner'], 'p_min'),
             ('world', good['worlds'][0], 'cell')]
    for name, _, key in cases:
        d = load(reg(tmp_path))
        if name == 'top':
            d[key] = 1
        elif name == 'world':
            d['worlds'][0][key] = 1
        else:
            d[name][key] = 1
        p = tmp_path/f'bad_{name}_{key}.json'
        p.write_text(json.dumps(d), encoding='utf-8')
        with pytest.raises(ValueError, match='REFUSED'):
            pe.load_registration(str(p))
    d = load(reg(tmp_path))
    d['schema'] = 'coopmwm/planner_eval/registration/v0'
    p = tmp_path/'bad_schema.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    with pytest.raises(ValueError, match='schema'):
        pe.load_registration(str(p))


def test_registration_seed_rules(tmp_path):
    good = load(reg(tmp_path))

    def bad(mutate, name):
        d = json.loads(json.dumps(good))
        mutate(d)
        p = tmp_path/f'{name}.json'
        p.write_text(json.dumps(d), encoding='utf-8')
        with pytest.raises(ValueError):
            pe.load_registration(str(p))

    bad(lambda d: d.__setitem__('worlds', []), 'empty')
    bad(lambda d: d['worlds'].__setitem__(1, dict(d['worlds'][0])), 'dup')
    bad(lambda d: d['worlds'][0].__setitem__('seed', 1.5), 'float_seed')
    bad(lambda d: d['worlds'][0].__setitem__('seed', True), 'bool_seed')
    bad(lambda d: d['worlds'][0].__setitem__('seed', -3), 'negative_seed')
    bad(lambda d: d['worlds'][0].__setitem__('seed', 0), 'zero_seed')
    bad(lambda d: d['worlds'][0].__setitem__('cell', 1), 'unknown_world_key')
    bad(lambda d: d['worlds'][0].pop('demand'), 'partial_meta')
    bad(lambda d: d['worlds'][0].__setitem__('demand', float('inf')), 'nonfinite_meta')
    bad(lambda d: d.__setitem__('worlds', 'nope'), 'not_a_list')
    with pytest.raises(ValueError, match='at least two'):
        pe.validate_registration(dict(good, worlds=[good['worlds'][0]]))


def test_registration_guard_refuses_forbidden_and_truth_paths(tmp_path):
    with pytest.raises(ValueError, match='forbidden path token'):
        pe.load_registration(str(tmp_path/'sealed_seeds.json'))
    d = load(reg(tmp_path))
    for name, cal in (('holdout_cal', dict(path='runs/holdout/calibration.pt', sha256='a'*64)),
                      ('truth_cal', dict(path='runs/vtype_truth.pt', sha256='a'*64)),
                      ('heldout_cal', dict(path='x/heldout/cal.json', sha256='a'*64))):
        p = tmp_path/f'{name}.json'
        p.write_text(json.dumps(dict(d, calibration=cal)), encoding='utf-8')
        with pytest.raises(ValueError, match='REFUSED'):
            pe.load_registration(str(p))
    hidden = json.loads(json.dumps(d))
    hidden['note'] = 'mirrored from repo/sealed/panel.json'
    p = tmp_path/'hidden_sealed.json'
    p.write_text(json.dumps(hidden), encoding='utf-8')
    with pytest.raises(ValueError, match='sealed'):
        pe.load_registration(str(p))


def test_registration_calibration_sha_and_missing_file(tmp_path):
    d = load(reg(tmp_path))
    for sha in ('z'*64, 'A'*64, 'a'*63):
        p = tmp_path/f'bad_sha_{len(sha)}.json'
        p.write_text(json.dumps(dict(d, calibration=dict(path='cal/planner.json', sha256=sha))),
                     encoding='utf-8')
        with pytest.raises(ValueError, match='sha256'):
            pe.load_registration(str(p))
    p = tmp_path/'good_cal.json'
    p.write_text(json.dumps(dict(d, calibration=dict(path='cal/planner_calibration.json',
                                                     sha256='b'*64))), encoding='utf-8')
    r = pe.load_registration(str(p))
    assert r['calibration'] == dict(path='cal/planner_calibration.json', sha256='b'*64)
    with pytest.raises(ValueError, match='missing'):
        pe.load_registration(str(tmp_path/'nope.json'))


# ---------------------------------------------------------------- plan

def test_plan_signed_and_verified(tmp_path):
    p = plan(tmp_path)
    assert pe.verify_plan(p)
    assert p['arms'] == ['MAPPO', 'PLANNER'] and p['n_worlds'] == len(SEEDS)
    assert p['panel_sha256'] == pe.digest(p['worlds'])
    assert p['budget'] == pe.budget_quote(len(SEEDS), 8)
    assert p['budget']['total_sim_s'] == 2*len(SEEDS)*1200
    tampered = dict(p, worlds=[dict(p['worlds'][0], demand=1.)]+p['worlds'][1:])
    with pytest.raises(ValueError, match='plan SHA mismatch'):
        pe.verify_plan(tampered)
    with pytest.raises(ValueError, match='budget'):
        pe.verify_plan(pe.seal(dict(p, budget=dict(p['budget'], total_sim_s=1)), 'plan_sha256'))


def test_plan_pins_registered_gate_and_rules(tmp_path):
    p = plan(tmp_path)
    assert p['planner']['decision_interval_s'] == 60 and p['planner']['p_star'] == .8
    assert p['planner']['lookahead_s'] == 300 and p['planner']['k_candidates'] == 8
    assert 'NEVER shortened' in p['planner']['horizon_rule']
    assert 'CALIBRATED' in p['planner']['gate_rule'] and 'p_star' in p['planner']['gate_rule']
    assert p['episode']['end_s'] == 1200 and 'same world seed' in p['episode']['rule']
    assert p['analysis']['bootstrap']['seed'] == 20261101 and p['analysis']['n_boot'] == 10000
    assert p['safety']['keys'] == list(pe.SAFETY_KEYS)
    assert 'hard' in p['safety']['hard'] and 'combined' in p['safety']['combined']


def test_plan_registration_digest_tracks_seeds(tmp_path):
    a = plan(tmp_path, seeds=(777001, 777002))
    b = plan(tmp_path, seeds=(777001, 777009))
    assert a['registration_sha256'] != b['registration_sha256']
    assert a['plan_sha256'] != b['plan_sha256']


def test_rehashed_plan_still_enforces_registration(tmp_path):
    p = plan(tmp_path)
    p['planner']['p_star'] = .1
    with pytest.raises(ValueError, match='registered'):
        pe.verify_plan(pe.seal(p, 'plan_sha256'))


# ---------------------------------------------------------------- paired jobs

def test_jobs_paired_worlds_same_seed_both_arms(tmp_path):
    p = plan(tmp_path)
    js = pe.jobs(p)
    assert len(js) == 2*len(SEEDS) and pe.check_paired(p, js)
    for w in p['worlds']:
        t, c = job(p, w['seed'], pe.TREATMENT), job(p, w['seed'], pe.COMPARATOR)
        assert t['world'] == c['world'] == w                     # identical paired world
        assert t['planner_enabled'] is True and c['planner_enabled'] is False
        assert c['planner'] is None
        assert t['planner']['p_star'] == .8 and t['planner']['lookahead_s'] == 300
        assert t['sim_s'] == c['sim_s'] == 1200 and t['mode'] == c['mode'] == 'run'
        assert t['deterministic'] and c['deterministic']
        assert t['plan_sha256'] == c['plan_sha256'] == p['plan_sha256']
        assert t['noise_seed'] == c['noise_seed']                # paired common random numbers


def test_noise_seed_is_deterministic_and_job_binding(tmp_path):
    p = plan(tmp_path)
    s = p['worlds'][0]['seed']
    assert pe.noise_seed(p['worlds'][0], 'PLANNER') == pe.noise_seed(dict(p['worlds'][0]), 'PLANNER')
    assert pe.jobs(p)[0]['id'].startswith('planner_eval_')


def test_check_paired_refuses(tmp_path):
    p = plan(tmp_path)
    js = pe.jobs(p)
    with pytest.raises(ValueError, match='unpaired'):
        pe.check_paired(p, [j for j in js if j['arm'] == pe.TREATMENT])
    dup = js + [dict(js[0])]
    with pytest.raises(ValueError, match='duplicate'):
        pe.check_paired(p, dup)
    outside = dict(js[0], world=dict(js[0]['world'], seed=999999))
    with pytest.raises(ValueError, match='outside the registered panel'):
        pe.check_paired(p, js + [outside])
    diverged = [dict(j) for j in js]
    for j in diverged:
        if j['arm'] == pe.TREATMENT:
            j['world'] = dict(j['world'], demand=1.)
    with pytest.raises(ValueError, match='paired worlds differ'):
        pe.check_paired(p, diverged)


def test_calibration_reference_rides_only_on_the_planner_job(tmp_path):
    d = load(reg(tmp_path))
    d['calibration'] = dict(path='cal/planner_calibration.json', sha256='c'*64)
    p2 = tmp_path/'with_cal.json'
    p2.write_text(json.dumps(d), encoding='utf-8')
    p = pe.build_plan(pe.load_registration(str(p2)))
    for w in p['worlds']:
        t, c = job(p, w['seed'], pe.TREATMENT), job(p, w['seed'], pe.COMPARATOR)
        assert t['planner']['calibration']['sha256'] == 'c'*64
        assert c['planner'] is None


def test_matching_but_unregistered_job_worlds_refused(tmp_path):
    p = plan(tmp_path)
    js = pe.jobs(p)
    for j in js:
        j['world']['demand'] = 1.
    with pytest.raises(ValueError, match='registered specification'):
        pe.check_paired(p, js)


# ---------------------------------------------------------------- receipts

def test_receipt_signed_and_verifiable_by_exact_grade(tmp_path):
    p = plan(tmp_path)
    r = receipt(p, SEEDS[0])
    pe.validate_receipt(r)
    assert r['J'] == 100. and r['arm'] == pe.TREATMENT and r['seed'] == SEEDS[0]
    assert r['safety'] == {k: float(v) for k, v in SAFETY.items()}
    assert r['planner_trace'] == TRACE
    from tools.coopmwm.exact_grade import verify_sealed
    assert verify_sealed(r, 'receipt_sha256')                    # interop with the bank machinery
    pe.validate_receipt(r, job(p, SEEDS[0], pe.TREATMENT))


def test_receipt_tamper_refused(tmp_path):
    r = receipt(plan(tmp_path), SEEDS[0])
    with pytest.raises(ValueError, match='edited after sealing'):
        pe.validate_receipt(dict(r, J=r['J']+1))
    with pytest.raises(ValueError, match='not a planner_eval receipt'):
        pe.validate_receipt(dict(r, schema='other/v1'))
    with pytest.raises(ValueError, match='unknown arm'):
        pe.validate_receipt(pe.seal(dict(r, arm='R'), 'receipt_sha256'))


def test_receipt_job_binding(tmp_path):
    p = plan(tmp_path)
    r = receipt(p, SEEDS[0])
    for wrong in (job(p, SEEDS[1], pe.TREATMENT), job(p, SEEDS[0], pe.COMPARATOR)):
        with pytest.raises(ValueError, match='differs from its job'):
            pe.validate_receipt(r, wrong)


@pytest.mark.parametrize('safety', [
    dict(collisions=-1.), dict(hard_brake=float('nan')), dict(teleports='x'),
])
def test_receipt_safety_counter_rules(tmp_path, safety):
    with pytest.raises(ValueError):
        receipt(plan(tmp_path), SEEDS[0], safety=safety)


def test_receipt_incomplete_safety_refused(tmp_path):
    s = dict(SAFETY)
    s.pop('ttc_below_3_vs')
    with pytest.raises(ValueError, match='safety keys'):
        receipt(plan(tmp_path), SEEDS[0], outcome=dict(J=1., safety=s))


def test_receipt_J_rules_and_unknown_outcome_keys(tmp_path):
    with pytest.raises(ValueError, match='finite'):
        receipt(plan(tmp_path), SEEDS[0], J=float('nan'))
    with pytest.raises(ValueError, match='unregistered outcome keys'):
        receipt(plan(tmp_path), SEEDS[0], outcome=dict(surprise=1))


@pytest.mark.parametrize('trace', [
    [dict(t_s=90, p_better=.9, fired=False)],                       # off the 60 s cadence
    [dict(t_s=0, p_better=.9, fired=False)],                        # before the first interval
    [dict(t_s=1260, p_better=.9, fired=False)],                     # past the episode end
    [dict(t_s=60, p_better=.79, fired=True)],                       # fired below p_star
    [dict(t_s=60, p_better=1.5, fired=False)],                      # p outside [0, 1]
    [dict(t_s=60, p_better=.9, fired=False, family='teleport')],    # unregistered family
    [dict(t_s=60, p_better=.9, fired=False, members=[1.5])],        # bad member id
    [dict(t_s=60, p_better=.9, fired=False, lookahead_s=299)],      # look-ahead shortened
    [dict(t_s=120, p_better=.9, fired=False), dict(t_s=60, p_better=.9, fired=False)],  # unsorted
    [dict(t_s=60, p_better=.9, fired=False), dict(t_s=60, p_better=.9, fired=False)],   # duplicate
    [dict(t_s=60, p_better=.9, fired='yes')],                       # fired must be bool
    [dict(t_s=60, p_better=.9, fired=False, surprise=1)],           # unregistered trace key
])
def test_trace_rules_refused(tmp_path, trace):
    with pytest.raises(ValueError):
        receipt(plan(tmp_path), SEEDS[0], trace=trace)


def test_trace_ok_forms(tmp_path):
    p = plan(tmp_path)
    receipt(p, SEEDS[0], trace=[dict(t_s=60, p_better=.79, fired=False)])       # abstain is fine
    receipt(p, SEEDS[0], trace=[dict(t_s=60, p_better=.8, fired=True, family='wall',
                                       members=['cav_1'], lookahead_s=300)])      # exactly at the gate
    r = receipt(p, SEEDS[0], trace=[])
    assert r['planner_trace'] == []


def test_mappo_arm_must_have_no_trace(tmp_path):
    with pytest.raises(ValueError, match='no planner trace'):
        receipt(plan(tmp_path), SEEDS[0], arm=pe.COMPARATOR, trace=TRACE)


# ---------------------------------------------------------------- safety gate

def test_safety_gate_benign_and_offset_pattern(tmp_path):
    p = plan(tmp_path, seeds=SEEDS[:3])
    gate = pe.safety_gate(p, full_records(p))
    assert gate['safety_ok'] and not gate['hard_failures'] and not gate['combined_failures']
    # Increased hard braking is offset by fewer TTC<1.5 s events under the combined rule.
    recs = full_records(p, safety=dict(hard_brake=4., ttc_below_1p5_events=0.))
    gate = pe.safety_gate(p, recs)
    assert gate['safety_ok'] and gate['per_world'][str(SEEDS[0])]['combined_rule']
    assert gate['per_world'][str(SEEDS[0])]['safety_critical_events_delta'] == 0.
    # ttc_below_3_vs is not in the hard/combined families: old rule flags it, gate still passes.
    gate = pe.safety_gate(p, full_records(p, safety=dict(ttc_below_3_vs=7.)))
    assert gate['safety_ok']
    assert not gate['per_world'][str(SEEDS[0])]['old_no_metric_increase']


def test_safety_gate_hard_failures(tmp_path):
    p = plan(tmp_path, seeds=SEEDS[:3])
    gate = pe.safety_gate(p, full_records(p, safety=dict(collisions=1.)))
    assert not gate['safety_ok'] and SEEDS[0] in gate['hard_failures']
    gate = pe.safety_gate(p, full_records(p, safety=dict(teleports=1.)))
    assert not gate['safety_ok'] and SEEDS[0] in gate['hard_failures']
    gate = pe.safety_gate(p, full_records(p, safety=dict(hard_brake=4.)))
    assert not gate['safety_ok'] and SEEDS[0] in gate['combined_failures']


def test_safety_delta(tmp_path):
    p = plan(tmp_path)
    t = receipt(p, SEEDS[0], safety=dict(collisions=1., hard_brake=2.))
    c = receipt(p, SEEDS[0], arm=pe.COMPARATOR)
    d = pe.safety_delta(t, c)
    assert d['collisions'] == 1. and d['hard_brake'] == 1. and d['ttc_below_3_vs'] == 0.


# ---------------------------------------------------------------- analysis

def test_analyze_registered_paired_bootstrap(tmp_path):
    p = plan(tmp_path)
    out = pe.analyze(full_records(p, mappo_J=lambda s: 100.+s % 7), p)
    pe.verify_analysis(out)
    assert out['plan_sha256'] == p['plan_sha256'] and out['seeds'] == sorted(SEEDS)
    prim = out['primary']
    from tools.coopmwm.branch_eval import cluster_bootstrap
    import numpy as np
    treat = np.array([70. for _ in SEEDS])[:, None]
    comp = np.array([100.+s % 7 for s in SEEDS])[:, None]
    raw = cluster_bootstrap(treat, comp, 10000, 20261101, .05)   # the SAME registered estimator
    assert prim['contrast'] == 'PLANNER-MAPPO' and prim['metric'] == 'team_delay_vs'
    assert prim['mean_diff_vs'] == raw['mean_diff_vs'] == -32.5
    assert prim['ci95'] == raw['ci95'] and prim['p_two_sided'] == raw['p_two_sided']
    assert prim['worlds_improved'] == len(SEEDS) and prim['ci95'][1] < 0
    assert prim['p_two_sided'] < .05
    assert out['completeness'] == dict(worlds=len(SEEDS), receipts=2*len(SEEDS), complete=True)


def test_analyze_null_split_gives_no_significance(tmp_path):
    p = plan(tmp_path)
    out = pe.analyze(full_records(p, planner_J=lambda s: 100.+(10. if s % 2 else -10.)), p)
    assert out['primary']['mean_diff_vs'] == 0.
    assert out['primary']['p_two_sided'] == 1. and out['primary']['ci95'][0] <= 0 <= out['primary']['ci95'][1]


def test_analyze_safety_cis_and_gate_embedded(tmp_path):
    p = plan(tmp_path)
    out = pe.analyze(full_records(p), p)
    assert set(out['safety']['paired_ci']) == set(pe.SAFETY_KEYS)
    for k, ci in out['safety']['paired_ci'].items():
        assert ci['mean_delta_vs'] == 0. and ci['ci95'] == [0., 0.]
    assert out['safety']['gate']['safety_ok']


def test_analyze_completeness_refusals(tmp_path):
    p = plan(tmp_path)
    recs = full_records(p)
    with pytest.raises(ValueError, match='incomplete'):
        pe.analyze(recs[1:], p)                                   # one arm missing
    foreign = receipt(plan(tmp_path, seeds=(888001, 888002)), 888001)
    with pytest.raises(ValueError, match='outside the registered plan'):
        pe.analyze(recs + [foreign], p)
    with pytest.raises(ValueError, match='duplicate'):
        pe.analyze(recs + [dict(recs[0])], p)
    other = plan(tmp_path, seeds=(888001, 888002))
    with pytest.raises(ValueError, match='outside the registered plan'):
        pe.analyze(recs[:1] + full_records(other)[:3], other)
    with pytest.raises(ValueError, match='edited after sealing'):
        pe.analyze([dict(r, J=r['J']+1) if r['arm'] == pe.TREATMENT else r for r in recs], p)


@pytest.mark.parametrize('field,value', [('job_id', 'wrong'), ('episode_end_s', 60),
                                         ('metric', 'wrong')])
def test_analysis_binds_rehashed_receipts_to_registered_jobs(tmp_path, field, value):
    p = plan(tmp_path)
    recs = full_records(p)
    recs[0] = pe.seal(dict(recs[0], **{field: value}), 'receipt_sha256')
    with pytest.raises(ValueError, match='differs from its job'):
        pe.analyze(recs, p)


def test_analyze_tamper_refused(tmp_path):
    out = pe.analyze(full_records(plan(tmp_path)), plan(tmp_path))
    with pytest.raises(ValueError, match='edited after sealing'):
        pe.verify_analysis(dict(out, mean=1))


# ---------------------------------------------------------------- dry run

def test_preflight_writes_nothing_and_charges_zero(tmp_path):
    out_dir = tmp_path/'dry'
    summary = pe.preflight_dry(reg(tmp_path))
    assert summary['dry_run'] is True and summary['new_sim_s'] == 0
    assert summary['worlds'] == len(SEEDS) and summary['jobs'] == 2*len(SEEDS)
    assert summary['budget']['dry_run_sim_s'] == 0 and summary['receipt_schema'] == pe.RECEIPT_SCHEMA
    assert summary['safety_keys'] == list(pe.SAFETY_KEYS)
    assert not out_dir.exists()


def test_dry_run_writes_plan_and_jobs_only(tmp_path):
    out = tmp_path/'nested'/'dry'
    summary = pe.dry_run(reg(tmp_path), str(out))
    assert summary['new_sim_s'] == 0 and summary['sim_s_charged'] == 0
    assert summary['jobs_lines'] == 2*len(SEEDS)
    files = sorted(x.name for x in out.iterdir())
    assert files == [pe.JOBS_FILE, pe.PLAN_FILE]
    p = json.loads((out/pe.PLAN_FILE).read_text(encoding='utf-8'))
    assert pe.verify_plan(p) and p['plan_sha256'] == summary['plan_sha256']
    lines = (out/pe.JOBS_FILE).read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2*len(SEEDS)
    js = [json.loads(x) for x in lines]
    assert pe.check_paired(p, js)
    assert all(j['dry_run'] is True for j in js)


def test_dry_run_refuses_non_empty_and_forbidden_out(tmp_path):
    out = tmp_path/'dry'
    pe.dry_run(reg(tmp_path), str(out))
    with pytest.raises(ValueError, match='not empty'):
        pe.dry_run(reg(tmp_path), str(out))
    with pytest.raises(ValueError, match='forbidden path token'):
        pe.dry_run(reg(tmp_path), str(tmp_path/'sealed_dry'))


# ---------------------------------------------------------------- no hard-coded seeds

def test_module_hardcodes_no_seed_list_and_opens_no_panel():
    import inspect
    src = inspect.getsource(pe)
    for marker in ('accel_protocol', 'final_test', 'TRAINING_SEEDS', '986000', '989600',
                   'traffic_shepherd.seeds'):
        assert marker not in src
    assert not hasattr(pe, 'SEEDS') and not hasattr(pe, 'WORLDS')


# ---------------------------------------------------------------- resumed edge-case verification

def test_budget_multiple_wm_samples_and_groups(tmp_path):
    d = load(reg(tmp_path))
    d['planner'].update(wm_samples=5, groups_per_decision=3)
    p = pe.build_plan(d)
    assert pe.verify_plan(p)
    assert p['budget']['wm_quote_complete']
    assert p['budget']['planner_wm_total_rollout_s'] == 216000*5*3
    assert p['budget']['total_sim_s'] == 9600
    assert not pe.budget_quote(4, 8)['wm_quote_complete']
    assert pe.budget_quote(4, 8)['planner_wm_total_rollout_s'] is None
    d['planner']['wm_samples'] = 1
    with pytest.raises(ValueError, match='multiple WM samples'):
        pe.build_plan(d)


@pytest.mark.parametrize('kwargs', [dict(n_worlds=True), dict(n_worlds=2.5),
    dict(k_candidates=-1), dict(k_candidates=True), dict(episode_end_s=0),
    dict(decision_interval_s=0), dict(lookahead_s=-300), dict(wm_samples=0),
    dict(groups_per_decision=1.5)])
def test_budget_invalid_counts_refused(kwargs):
    with pytest.raises(ValueError):
        pe.budget_quote(**dict(dict(n_worlds=4), **kwargs))


def test_multiple_groups_at_same_decision_and_fired_move_completeness(tmp_path):
    p = plan(tmp_path)
    trace = [dict(TRACE[0], group_id='near_merge'), dict(TRACE[0], group_id='upstream')]
    assert len(receipt(p, SEEDS[0], trace=trace)['planner_trace']) == 2
    for bad in (trace+[trace[0]], [dict(TRACE[0], members=[])],
                [dict(TRACE[0], members=[1, 1])], [dict(TRACE[0], members=list(range(5)))],
                [dict(TRACE[0], family='none')], [dict(TRACE[0], lookahead_s=None)]):
        with pytest.raises(ValueError):
            receipt(p, SEEDS[0], trace=bad)
    d = load(reg(tmp_path))
    d['planner']['groups_per_decision'] = 1
    bounded = pe.build_plan(d)
    with pytest.raises(ValueError, match='groups_per_decision bound'):
        receipt(bounded, SEEDS[0], trace=trace)


def test_zero_delay_panel_has_valid_json_and_no_percentage(tmp_path):
    p = plan(tmp_path)
    out = pe.analyze(full_records(p, planner_J=lambda s: 0., mappo_J=lambda s: 0.), p)
    assert out['primary']['mean_diff_vs'] == 0.
    assert out['primary']['ci95'] == [0., 0.]
    assert out['primary']['pct_delay_reduction'] is None
    assert out['primary']['pct_ci95'] == [None, None]
    assert pe.verify_analysis(json.loads(pe.canonical(out)))


def test_analysis_independent_of_receipt_order_and_accepts_iterator(tmp_path):
    p = plan(tmp_path)
    recs = full_records(p)
    assert pe.analyze(iter(reversed(recs)), p) == pe.analyze(recs, p)


def test_safety_ci_matches_registered_world_resampling_without_percentages():
    import numpy as np
    d = np.array([-2., 1., 3., 0.])
    idx = np.random.default_rng(pe.BOOT_SEED).integers(0, 4, (pe.N_BOOT, 4))
    expected = np.quantile(d[idx].mean(1), [.025, .975]).tolist()
    ci = pe._paired_ci(d, pe.N_BOOT, pe.BOOT_SEED, pe.ALPHA)
    assert ci['ci95'] == expected and ci['mean_delta_vs'] == .5


def test_resolved_path_guard_blocks_alias_before_read(tmp_path, monkeypatch):
    target = tmp_path/'sealed'/'registration.json'
    monkeypatch.setattr(Path, 'resolve', lambda self: target)
    with pytest.raises(ValueError, match='forbidden path token'):
        pe.load_registration(tmp_path/'alias.json')


def test_dry_run_reads_registration_once(tmp_path, monkeypatch):
    path = reg(tmp_path)
    original = pe.load_registration
    calls = []

    def once(p):
        calls.append(p)
        return original(p)

    monkeypatch.setattr(pe, 'load_registration', once)
    result = pe.dry_run(path, tmp_path/'dry')
    assert calls == [path] and not result['dispatch_supported']
    assert not result['calibration_pinned']


def test_missing_or_unknown_receipt_fields_refused(tmp_path):
    r = receipt(plan(tmp_path), SEEDS[0])
    missing = dict(r)
    missing.pop('executor')
    for bad in (missing, dict(r, extra='unexpected')):
        with pytest.raises(ValueError, match='receipt keys'):
            pe.validate_receipt(pe.seal(bad, 'receipt_sha256'))
