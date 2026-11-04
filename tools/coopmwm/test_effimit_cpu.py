"""CPU tests for the effective (censored) imitation target space (c4/effective-imitation).

imitation_target_space: default 'raw' is byte-identical; 'effective' derives each plan
command's binding limit u_t = min(cav_typical vType cap, recorded observables that the plan
command exceeded: issued_safe, realised acceleration) from the winning plan branch
observable tape, trains the censored (Tobit) likelihood (target below u_t: the exact
Gaussian term; target at/above u_t: P(policy >= u_t), so any dead-zone command is equally
correct), reports the action error in effective space min(command, u_t) with binding-share
diagnostics, and clips intent targets to the same space. AWR weights, the .045 KL trust
region and the kl_targeted step mode are untouched. The runtime key is pinned in
BANK_OPTIONS (one key for B and R, like imitation_bank_filter/imitation_step_mode).
"""
import copy
import gzip
import json
import math

import pytest
import torch

from tools.coopmwm import bank_imitation as bi
from tools.coopmwm import effective_imitation as eff
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm import v2v_intent as vi
from tools.coopmwm import v2v_launch3 as launch
from tools.coopmwm import v2v_runner
from tools.coopfree.policy import gaussian_log_prob
from tools.coopmwm.test_bank_imitation_cpu import Policy
from tools.coopmwm.test_replay_bank_confirm_cpu import (bank_budget, fp, snapshot, plan,
                                                        FakeRunner)
from tools.coopnative.storage import canonical


# ---------------------------------------------------------------- censored likelihood

SIGMA = torch.tensor([.5])


def test_censored_log_prob_equals_the_gaussian_when_no_target_binds():
    torch.manual_seed(3)
    for n in (1, 2, 5):
        action, mean = torch.randn(n, 1), torch.randn(n, 1)
        limits = torch.full((n, 1), float('inf'))
        assert torch.equal(eff.censored_log_prob(action, mean, .5, SIGMA, limits),
                           gaussian_log_prob(action, mean, .5, SIGMA))
        # a target strictly below its limit must give the identical value
        action = mean-.5                                   # target < limit 0.0? limits below
        limits = mean+1.                                   # target is 1 below the limit
        assert torch.equal(eff.censored_log_prob(action, mean, .5, SIGMA, limits),
                           gaussian_log_prob(action, mean, .5, SIGMA))


def test_censored_survival_matches_the_closed_form_single_member():
    # one member: the correlated joint collapses to N(mu, sigma^2), so the censored
    # likelihood is exactly log Phi((mu - u)/sigma); the quadrature must reproduce it.
    for mu, u in ((-1.3, .2), (-.3, .2), (.7, .2), (2., .2)):
        got = float(eff.censored_log_prob(torch.tensor([[10.]]), torch.tensor([[mu]]), .5,
                                          SIGMA, torch.tensor([[u]])))
        want = float(torch.log(.5*torch.erfc(torch.tensor((-(mu-u)/.5)/math.sqrt(2)))))
        assert got == pytest.approx(want, abs=1e-5)


def test_every_dead_zone_plan_command_costs_the_same():
    # plan commands at/above the limit all reduce to P(policy >= limit): the target value
    # itself drops out of the loss entirely, and the uncensored neighbour keeps the exact
    # Gaussian term.
    mean = torch.tensor([[.2], [.2]])
    limits = torch.tensor([[1.2], [1.2]])
    raw = -eff.censored_log_prob(torch.tensor([[2.6], [2.6]]), mean, .5, SIGMA, limits)/2
    raw_lower = -eff.censored_log_prob(torch.tensor([[1.5], [2.6]]), mean, .5, SIGMA, limits)/2
    raw_higher = -eff.censored_log_prob(torch.tensor([[2.6], [4.]]), mean, .5, SIGMA, limits)/2
    assert torch.equal(raw, raw_lower) and torch.equal(raw, raw_higher)
    uncensored = -eff.censored_log_prob(torch.tensor([[.5], [2.6]]), mean, .5, SIGMA,
                                        torch.tensor([[1.2], [1.2]]))/2
    assert math.isfinite(float(uncensored))
    # an uncensored target still matters (the exact density term), unlike the dead-zone ones
    uncensored_other = -eff.censored_log_prob(torch.tensor([[.8], [2.6]]), mean, .5, SIGMA,
                                              torch.tensor([[1.2], [1.2]]))/2
    assert float(uncensored) != float(uncensored_other)


def test_gradient_pushes_toward_the_limit_only_where_the_plan_binds():
    torch.manual_seed(0)
    def grad(target, mean_value, limit):
        m = torch.tensor([[float(mean_value)]], requires_grad=True)
        loss = -eff.censored_log_prob(torch.tensor([[float(target)]]), m, .5, SIGMA,
                                      torch.tensor([[float(limit)]]))
        loss.backward()
        return float(m.grad)
    # binding (a* = 2.6 >= u = 1.2): strong push up while the mean is below the limit,
    # essentially none once the mean sits above it - never a pull toward a*
    below, above = grad(2.6, 1.0, 1.2), grad(2.6, 3.0, 1.2)
    assert below < -0.5 and abs(above) < 0.01
    assert abs(above) < abs(below)/100
    # the raw term keeps pulling at mean=3.0 (toward a*=2.6); the censored term does not
    m = torch.tensor([[3.]], requires_grad=True)
    (-gaussian_log_prob(torch.tensor([[2.6]]), m, .5, SIGMA)).backward()
    raw_above = float(m.grad)
    assert abs(raw_above) > abs(above)*10
    # uncensored (a* = .5 < u): identical pull to the raw Gaussian term, toward a*
    m = torch.tensor([[1.]], requires_grad=True)
    (-gaussian_log_prob(torch.tensor([[.5]]), m, .5, SIGMA)).backward()
    assert grad(.5, 1.0, 1.2) == pytest.approx(float(m.grad), rel=1e-4)


def test_pair_imitation_requires_limits_and_fails_closed():
    p = Policy()
    sample = dict(context={'members': ['a', 'b']}, action=[[.1], [.1]],
                  behaviour=dict(executed=True, proposal_members=['a', 'b']))
    with pytest.raises(ValueError, match='binding_limits'):
        eff.pair_imitation(p, sample, 1.)
    partial = dict(sample, binding_limits={'a': .2})
    with pytest.raises(ValueError, match='b'):
        eff.pair_imitation(p, partial, 1.)
    ok = dict(sample, binding_limits={'a': 1.2, 'b': 1.2})
    loss = eff.pair_imitation(p, ok, 1.).detach()
    assert math.isfinite(float(loss)) and float(loss) != 0.
    unweighted = eff.pair_imitation(p, ok, 0.).detach()
    assert float(unweighted) == 0.                       # weight<=0: zero-grad placeholder


# ---------------------------------------------------------------- effective error report

def test_effective_action_error_and_diagnostics_match_hand_computation():
    p = Policy()
    with torch.no_grad():
        p.mean.fill_(0.5)                    # policy mean 0.5 for every member
    def sample(target):
        return dict(context={'members': ['a']}, action=[[target]],
                    behaviour=dict(executed=True, proposal_members=['a']))
    pairs = [(dict(sample(2.6), binding_limits={'a': .2}), 1.),    # binding: |min(.5,.2)-.2| = 0
             (dict(sample(.1), binding_limits={'a': .5}), 1.)]     # free:    |min(.5,.5)-.1| = .4
    assert eff.effective_action_error(p, pairs) == pytest.approx(.2)
    # the raw L1 would be |0.5-2.6|, |0.5-0.1| -> 1.25: the effective report differs
    assert bi.action_error(p, pairs) == pytest.approx(1.25)
    d = eff.effective_diagnostics(p, pairs)
    assert d['binding_share'] == .5 and d['member_actions'] == 2 and d['binding_member_actions'] == 1
    assert d['effective_action_error_binding_mps2'] == 0.
    assert d['mean_policy_action_on_binding_mps2'] == pytest.approx(.5)
    assert d['mean_limit_on_binding_mps2'] == pytest.approx(.2)
    assert json.dumps(d)
    # weight 0 and unexecuted pairs never count
    pairs[0] = (pairs[0][0], 0.)
    pairs[1] = (dict(pairs[1][0], behaviour=dict(executed=False, proposal_members=['a'])), 1.)
    assert eff.effective_action_error(p, pairs) is None
    assert eff.effective_diagnostics(p, pairs)['binding_share'] is None


# ---------------------------------------------------------------- run_separate option

def limit_pairs(n=12):
    sample = dict(context={'members': ['a', 'b']}, action=[[2.6], [2.6]],
                  behaviour=dict(executed=True, proposal_members=['a', 'b']),
                  binding_limits={'a': .2, 'b': .1})
    return [(copy.deepcopy(sample), 1.) for _ in range(n)]


def test_effective_mode_trains_and_reports_effective_space():
    p = Policy()
    with torch.no_grad():
        p.mean.fill_(0.)
    out = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-4), limit_pairs(),
                          torch.Generator().manual_seed(7), copy.deepcopy(p), passes=5,
                          imitation_target_space='effective')
    assert out['imitation_target_space'] == 'effective'
    d = out['effective']
    assert d['binding_share'] == 1. and d['binding_member_actions'] == 2*out['pairs_total']
    assert d['effective_action_error_binding_mps2'] is not None
    assert json.dumps(out)
    # the error report moved to effective space min(command, u_t)
    assert out['action_error_after_mps2'] == pytest.approx(d['effective_action_error_binding_mps2'])


def test_raw_mode_stays_byte_identical_even_with_limits_present():
    a, b, c = Policy(), Policy(), Policy()
    plain = limit_pairs()
    without = [(dict(s, binding_limits=None), w) for s, w in plain]
    for s, _ in without:
        del s['binding_limits']
    ra = bi.run_separate(a, torch.optim.Adam(a.parameters(), lr=1e-3), plain,
                         torch.Generator().manual_seed(7), copy.deepcopy(a), passes=5)
    rb_ = bi.run_separate(b, torch.optim.Adam(b.parameters(), lr=1e-3), plain,
                          torch.Generator().manual_seed(7), copy.deepcopy(b), passes=5,
                          imitation_target_space='raw')
    rc = bi.run_separate(c, torch.optim.Adam(c.parameters(), lr=1e-3), plain,
                         torch.Generator().manual_seed(7), copy.deepcopy(c), passes=5,
                         imitation_target_space='raw')
    assert json.dumps(ra, sort_keys=True) == json.dumps(rb_, sort_keys=True)
    assert json.dumps(ra, sort_keys=True) == json.dumps(rc, sort_keys=True)
    d = Policy()
    assert json.dumps(bi.run_separate(d, torch.optim.Adam(d.parameters(), lr=1e-3), without,
                                      torch.Generator().manual_seed(7), copy.deepcopy(d), passes=5),
                      sort_keys=True) == json.dumps(ra, sort_keys=True)
    assert 'imitation_target_space' not in ra and 'effective' not in ra


def test_effective_pair_without_limits_fails_closed():
    p = Policy()
    broken = limit_pairs(4)
    for s, _ in broken:
        del s['binding_limits']
    with pytest.raises(ValueError, match='binding_limits'):
        bi.run_separate(p, torch.optim.Adam(p.parameters()), broken,
                        torch.Generator().manual_seed(1), copy.deepcopy(p), passes=5,
                        imitation_target_space='effective')


def test_install_binding_and_fallback_for_the_target_space():
    with pytest.raises(ValueError, match='imitation_target_space'):
        bi.install(imitation_target_space='bogus')
    with pytest.raises(ValueError, match='imitation_target_space'):
        bi.run_separate(Policy(), torch.optim.Adam(Policy().parameters()), limit_pairs(2),
                        torch.Generator(), copy.deepcopy(Policy()), passes=5,
                        imitation_target_space='bogus')
    bi.install('fixed', True, 'effective')
    assert bi.RUNTIME_TARGET_SPACE == 'effective'
    p = Policy()
    out = bi.run_separate(p, torch.optim.Adam(p.parameters(), lr=1e-4), limit_pairs(6),
                          torch.Generator().manual_seed(7), copy.deepcopy(p), passes=5)
    assert out['imitation_target_space'] == 'effective'      # installed fallback wins
    q = Policy()
    out = bi.run_separate(q, torch.optim.Adam(q.parameters(), lr=1e-4), limit_pairs(6),
                          torch.Generator().manual_seed(7), copy.deepcopy(q), passes=5,
                          imitation_target_space='raw')
    assert 'imitation_target_space' not in out               # explicit raw overrides the install


# ---------------------------------------------------------------- bank pairs attachment

class EffimitTapeRunner(FakeRunner):
    """FakeRunner whose planned branches write the RESULT.json plan tape WITH commands and
    requests (native command 1.0 for c1 / 0.3 for c2, plan command 2.6 for both) and an
    observable trace whose CAV speeds rise by their realised acceleration with issued_safe
    audit rows: c1 realises the vType cap (1.2, no safe-speed cap), c2 is leader-limited
    (issued_safe 0.4). ``drop`` removes a member row (exclusion rule), ``free`` makes the
    plan command 0.5 with realised 0.5 (nothing binds), ``mismatch`` corrupts the recorded
    action residual."""

    NATIVE = dict(c1=1., c2=.3)
    PLAN = 2.6

    def __init__(self, work, drop=None, free=False, mismatch=False, commands=True):
        super().__init__(work)
        self.drop, self.free, self.mismatch, self.commands = drop, free, mismatch, commands
        self.t0 = 420                      # frames are also built directly in unit tests

    def command(self):
        return .5 if self.free else self.PLAN

    def realised(self, vid):
        if self.free:
            return .5
        return 1.2 if vid == 'c1' else .4

    def frame(self, t):
        # c1 LEADS far ahead and accelerates at its realised 1.2; c2 follows at 0.4
        # (leader-limited): the closing speed decays to 0 within 4 s and the trace stays
        # TTC-safe at every recorded second (positions integrate the speeds).
        k = t-self.t0
        a1, a2 = self.realised('c1'), self.realised('c2')
        rows = []
        if self.drop != 'c1':
            rows.append(dict(vid='c1', lane=0, is_cav=True, length_m=5.,
                             s_m=30.+8.*k+a1/2*k*k, v_mps=8.+a1*k,
                             issued_acceleration_mps2=a1, command_present=True))
        if self.drop != 'c2':
            rows.append(dict(vid='c2', lane=0, is_cav=True, length_m=5.,
                             s_m=10.5*k+a2/2*k*k, v_mps=10.5+a2*k,
                             issued_acceleration_mps2=a2, command_present=True))
        frame = dict(time_s=t, vehicles=rows)
        if t >= self.t0 and not self.free:
            frame['actions'] = [dict(vid='c1', issued_safe=1.2, requested_raw=self.command()),
                                dict(vid='c2', issued_safe=.4, requested_raw=self.command())]
        elif t >= self.t0 and self.free:
            frame['actions'] = [dict(vid='c1', issued_safe=.5, requested_raw=.5),
                                dict(vid='c2', issued_safe=.5, requested_raw=.5)]
        return frame

    def run(self, spec, snap):
        r = super().run(spec, snap)
        p = spec['plan']
        if p is None:
            return r
        from tools.coopmwm.accel_storage import write as awrite
        from tools.coopmwm.protocol import digest
        d = self.work/'branches'/spec['verification_id']/spec['branch']
        command = self.command()
        accels = {v: command for v in p['members']}
        steps = []
        for t in range(spec['window']['start_s'], spec['window']['start_s']+p['duration_s']):
            sample = dict(context=dict(members=list(p['members'])),
                action=[[command-self.NATIVE[v]] for v in p['members']],
                absolute_accels=accels,
                plan_executions=[dict(plan_id=p['id'], proposal_members=list(p['members']))])
            if self.commands:
                steps.append(dict(time_s=t, sample=sample, commands=dict(self.NATIVE),
                    requests={v: dict(requested_raw=command, acceleration_residual=command-self.NATIVE[v])
                              for v in p['members']}))
            else:
                steps.append(dict(time_s=t, sample=sample))
        if self.mismatch:
            steps[0]['sample']['action'] = [[0.], [0.]]
        tape = dict(steps=steps)
        awrite(d/'RESULT.json', tape)
        r['result_sha256'] = digest(tape)
        self.t0 = spec['window']['start_s']
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            for t in range(self.t0-1, self.t0+p['duration_s']):
                f.write(canonical(self.frame(t))+'\n')
        return r


def _bank(root, bank, runner):
    first = sv.verify(snapshot(420), [plan('a', 1.)], runner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted', first.get('reason')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(420), lambda work: runner(work), str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] is True, out


def test_attach_limits_derives_cap_and_leader_limits_from_the_tape(tmp_path):
    trace = tmp_path/'observable.jsonl.gz'
    runner = EffimitTapeRunner(tmp_path)
    with gzip.open(trace, 'wt', encoding='utf-8') as f:
        for t in range(419, 423):
            f.write(canonical(runner.frame(t))+'\n')
    entry = dict(start_s=420, duration_s=2, plan_id='a', weight=1.)
    pairs = [(dict(context=dict(members=['c1', 'c2']), action=[[1.6], [2.3]],
                   behaviour=dict(executed=True, proposal_members=['c1', 'c2'])), 1.)
             for _ in range(2)]
    kept, counts, by_time = eff.attach_limits(pairs, dict(steps=[
        dict(time_s=420, commands=dict(EffimitTapeRunner.NATIVE),
             requests={'c1': dict(requested_raw=2.6), 'c2': dict(requested_raw=2.6)}),
        dict(time_s=421, commands=dict(EffimitTapeRunner.NATIVE),
             requests={'c1': dict(requested_raw=2.6), 'c2': dict(requested_raw=2.6)})]),
        trace, entry)
    cap = eff.cav_accel_cap()
    assert cap == pytest.approx(1.2)                     # cav_typical vType definition
    assert counts['kept'] == 2 and counts['pairs'] == 2 and counts['cap_mps2'] == cap
    # c1: cap 1.2 vs realised 1.2 -> u = 1.2; c2: leader-limited issued_safe 0.4 -> u = 0.4
    assert kept[0][0]['binding_limits'] == {'c1': pytest.approx(1.2-1.),
                                            'c2': pytest.approx(.4-.3)}
    assert by_time == {420: {'c1': pytest.approx(1.2), 'c2': pytest.approx(.4)},
                       421: {'c1': pytest.approx(1.2), 'c2': pytest.approx(.4)}}
    # free case: the plan command binds nowhere, u stays the cap (uncensored)
    free = EffimitTapeRunner(tmp_path, free=True)
    with gzip.open(trace, 'wt', encoding='utf-8') as f:
        for t in range(419, 423):
            f.write(canonical(free.frame(t))+'\n')
    free_pairs = [(dict(context=dict(members=['c1', 'c2']), action=[[-.5], [.2]],
                        behaviour=dict(executed=True, proposal_members=['c1', 'c2'])), 1.)]
    free_steps = [dict(time_s=420, commands=dict(EffimitTapeRunner.NATIVE),
                       requests={'c1': dict(requested_raw=.5), 'c2': dict(requested_raw=.5)})]
    kept, counts, by_time = eff.attach_limits(free_pairs, dict(steps=free_steps), trace, entry)
    assert counts['kept'] == 1
    assert kept[0][0]['binding_limits'] == {'c1': pytest.approx(1.2-1.), 'c2': pytest.approx(1.2-.3)}
    assert by_time == {420: {'c1': 1.2, 'c2': 1.2}}
    # a member row missing from the trace: the pair is excluded and counted (never imputed)
    dropper = EffimitTapeRunner(tmp_path, drop='c2')
    with gzip.open(trace, 'wt', encoding='utf-8') as f:
        for t in range(419, 423):
            f.write(canonical(dropper.frame(t))+'\n')
    kept, counts, by_time = eff.attach_limits(pairs, dict(steps=[
        dict(time_s=420, commands=dict(EffimitTapeRunner.NATIVE),
             requests={'c1': dict(requested_raw=2.6), 'c2': dict(requested_raw=2.6)})]),
            trace, entry)
    # both pairs are excluded: c2 has no vehicle row at their seconds (and 421 has no step)
    assert kept == [] and counts['excluded_missing_limit'] == 2 and by_time == {}
    # corrupted recorded residual: excluded as a command mismatch
    bad = EffimitTapeRunner(tmp_path)
    with gzip.open(trace, 'wt', encoding='utf-8') as f:
        for t in range(419, 423):
            f.write(canonical(bad.frame(t))+'\n')
    corrupt = [(dict(pairs[0][0], action=[[0.], [0.]]), 1.)]
    kept, counts, _ = eff.attach_limits(corrupt, dict(steps=[
        dict(time_s=420, commands=dict(EffimitTapeRunner.NATIVE),
             requests={'c1': dict(requested_raw=2.6), 'c2': dict(requested_raw=2.6)})]),
        trace, entry)
    assert counts['excluded_command_mismatch'] == 1 and kept == []
    # unreadable trace: everything excluded, disclosed
    kept, counts, _ = eff.attach_limits(pairs, dict(steps=[]), tmp_path/'absent.gz', entry)
    assert kept == [] and counts['excluded_missing_trace'] is True


def test_bank_pairs_raw_never_attaches_limits_and_stays_byte_identical(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, EffimitTapeRunner)
    p1, s1 = rb.bank_pairs(root, 'B', 12345)
    p2, s2 = rb.bank_pairs(root, 'B', 12345, imitation_target_space='raw')
    assert json.dumps([p1, s1], sort_keys=True) == json.dumps([p2, s2], sort_keys=True)
    assert 'imitation_target_space' not in s1
    assert all('binding_limits' not in p[0] for p in p1)
    with pytest.raises(ValueError, match='imitation_target_space'):
        rb.bank_pairs(root, 'B', 12345, imitation_target_space='bogus')


def test_bank_pairs_effective_attaches_binding_limits_and_counts(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, EffimitTapeRunner)
    pairs, s = rb.bank_pairs(root, 'B', 12345, v2v=True, imitation_target_space='effective')
    assert len(pairs) == 120
    block = s['imitation_target_space']
    assert block['space'] == 'effective' and block['counts']['kept'] == 120
    assert block['counts'] == dict(pairs=120, kept=120, excluded_missing_trace=False,
                                   excluded_missing_limit=0, excluded_command_mismatch=0,
                                   cap_mps2=eff.cav_accel_cap())
    for sample, w in pairs[:2]:
        assert sample['binding_limits'] == {'c1': pytest.approx(.2), 'c2': pytest.approx(.1)}
        assert w == 1. and 'v2v' in sample['context']
    assert json.dumps([pairs, s])


def test_bank_pairs_effective_excludes_pairs_with_missing_limit_data(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, lambda work: EffimitTapeRunner(work, drop='c2'))
    pairs, s = rb.bank_pairs(root, 'B', 12345, v2v=True, imitation_target_space='effective')
    assert pairs == []
    assert s['imitation_target_space']['counts']['excluded_missing_limit'] == 120
    assert s['imitation_target_space']['counts']['kept'] == 0


def test_bank_pairs_effective_excludes_command_mismatches(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, lambda work: EffimitTapeRunner(work, mismatch=True))
    pairs, s = rb.bank_pairs(root, 'B', 12345, imitation_target_space='effective')
    assert len(pairs) == 119                       # only the corrupted first second is excluded
    assert s['imitation_target_space']['counts']['excluded_command_mismatch'] == 1
    assert s['imitation_target_space']['counts']['kept'] == 119


def test_bank_pairs_effective_clips_intent_targets(tmp_path):
    intent_cfg = dict(v2v=True, v2v_intent=True, v2v_intent_horizon=5)
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, EffimitTapeRunner)
    raw_pairs, raw_s = rb.bank_pairs(root, 'B', 12345, v2v=True, v2v_intent=intent_cfg)
    eff_pairs, eff_s = rb.bank_pairs(root, 'B', 12345, v2v=True, v2v_intent=intent_cfg,
                                     imitation_target_space='effective')
    assert len(raw_pairs) == len(eff_pairs) == 120
    for i, (raw, effective) in enumerate(zip(raw_pairs, eff_pairs)):
        members = effective[0]['context']['members']
        for j, vid in enumerate(members):
            residual_limit = effective[0]['binding_limits'][vid]
            absolute_limit = residual_limit+EffimitTapeRunner.NATIVE[vid]   # intent targets are absolute
            assert absolute_limit == pytest.approx(1.2 if vid == 'c1' else .4)
            for k in range(len(raw[0]['intent_target'][j])):
                assert effective[0]['intent_target'][j][k] == pytest.approx(
                    min(raw[0]['intent_target'][j][k], absolute_limit))
    assert json.dumps([eff_pairs, eff_s])


# ---------------------------------------------------------------- launcher / runtime (B == R)

def _cfg(tmp_path, **kw):
    cfg = v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v', legacy_root=tmp_path/'legacy',
                                    source_root=tmp_path/'src', common_checkpoint_sha256='a'*64, **kw)
    live = dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))
    return cfg, live


def test_launch3_pins_effective_for_b_and_r_together(tmp_path):
    assert launch.BANK_OPTIONS['imitation_target_space'] == 'effective'
    assert launch.IMITATION_TARGET_SPACES == ('raw', 'effective')
    cfg, live = _cfg(tmp_path)
    assert cfg['imitation_target_space'] == 'effective'
    assert launch.bind_runtime(dict(config=live), root=tmp_path/'v2v', source_root=tmp_path/'src') is live
    assert (launch.imitation_target_space_for(live, 'B'),
            launch.imitation_target_space_for(live, 'R')) == ('effective', 'effective')
    for bad in (dict(imitation_target_space='raw'), dict(imitation_target_space=None),
                dict(imitation_target_space='bogus'),
                dict(imitation_target_space_by_arm=dict(B='effective', R='raw')),
                dict(imitation_target_space_by_arm=dict(B='raw', R='raw'))):
        with pytest.raises(launch.Refused):
            launch.bind_runtime(dict(config=dict(live, **bad)), root=tmp_path/'v2v',
                                source_root=tmp_path/'src')
    ok = dict(live, imitation_target_space_by_arm=dict(B='effective', R='effective'))
    assert launch.bind_runtime(dict(config=ok), root=tmp_path/'v2v', source_root=tmp_path/'src') is ok
    for arm in ('A',):
        with pytest.raises(launch.Refused):
            launch.imitation_target_space_for(live, arm)
    assert launch.imitation_target_space_for(dict(live, imitation_target_space='raw'), 'B') == 'raw'
    with pytest.raises(launch.Refused, match='imitation_target_space'):
        launch.imitation_target_space_for(dict(live, imitation_target_space=1), 'B')
    with pytest.raises(ValueError):                       # producer refuses a non-pinned value
        _cfg(tmp_path, imitation_target_space='raw')


def test_hooks_install_and_pairs_wrapper_carry_the_target_space(tmp_path, monkeypatch):
    from tools.coopmwm import bank_statistics as bs
    from tools.coopmwm import v2v_live
    installed = []
    monkeypatch.setattr(bi, 'install',
                        lambda *, step_mode, intent_bank_term, imitation_target_space:
                        installed.append((step_mode, intent_bank_term, imitation_target_space)))
    monkeypatch.setattr(bs, 'install', lambda rb_, bank_rule='boot_v2', sources=None: None)
    cfg, _ = _cfg(tmp_path)
    seen = []
    monkeypatch.setattr(rb, 'bank_pairs', lambda root, arm, seed, **k: seen.append(k))
    h = v2v_live.hooks(tmp_path, cfg)
    assert installed == [('kl_targeted', True, 'effective')]
    h['bank_pairs'](tmp_path, 'B', 1)
    assert seen[-1].get('imitation_target_space') == 'effective'
    h['bank_pairs'](tmp_path, 'R', 1)
    assert seen[-1].get('imitation_target_space') == 'effective'
    # runtime default 'raw': the install carries it and the pairs wrapper adds no kwarg
    installed.clear()
    seen.clear()
    h = v2v_live.hooks(tmp_path, dict(bank_statistics='per_seed_v1'))
    assert installed == [('fixed', True, 'raw')]
    h['bank_pairs'](tmp_path, 'B', 1)
    assert 'imitation_target_space' not in seen[-1]
    # a B/R mismatch refuses before anything is installed or wrapped
    installed.clear()
    with pytest.raises(launch.Refused):
        v2v_live.hooks(tmp_path, dict(cfg, imitation_target_space_by_arm=dict(B='effective', R='raw')))
    assert installed == []
