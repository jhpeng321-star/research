"""CPU tests for SUMO verification of WM top-k plans with a fake deterministic plant (no SUMO)."""
import json
import threading
import time

import pytest
import torch

from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.exact_grade import BudgetBlocked, verify_sealed


def fp(t, delay):
    return dict(t_s=t, metrics=dict(team_delay_vs=float(delay).hex(), arrived=float(t).hex()))


def base_delay(world, t):
    return 3.*world+0.5*t*t/100.


def snapshot(t0=400, world=7, committed=()):
    return dict(episode=dict(id=f'warm_u005_B_{world}', world=dict(seed=world), arm='B', update=5, role='train'),
                policy=dict(state_dict_sha256='p'), sumo=dict(runtime='rt', checkpoint='ck', warm_history=True,
                                                            accel_bounds=[-3, 2], noise_seed=11),
                t0=t0, end_s=1200, prefixes={str(t): fp(t, base_delay(world, t)) for t in range(300, t0+1)},
                samples={}, rng=dict(generator_state_sha256='g'), committed_plans=list(committed))


def plan(pid, members=('c1', 'c2'), duration=120, gain=0.):
    return dict(id=pid, members=list(members), duration_s=duration, profile=[[gain]*duration for _ in members],
                saving_vs_mappo=gain, saving_vs_policy=gain)


class FakePlant:
    """Deterministic: J_window(branch) = J_policy - 10*mean(profile) for candidates; identical prefixes."""

    def __init__(self, J_policy=500., mismatch=(), fail=(), delay_s=0.):
        self.J_policy, self.mismatch, self.fail, self.delay_s = J_policy, set(mismatch), set(fail), delay_s
        self.calls = [];self.lock = threading.Lock()

    def J(self, p):
        if p is None:
            return self.J_policy
        prof = [x for row in p['profile'] for x in row]
        return self.J_policy-10.*sum(prof)/len(prof)

    def run(self, spec, snap):
        with self.lock:
            self.calls.append(spec['branch'])
        if self.delay_s:
            time.sleep(self.delay_s)
        if spec['branch'] in self.fail:
            raise RuntimeError('sumo crashed')
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        w = snap['episode']['world']['seed']
        prefixes = {str(t): fp(t, base_delay(w, t)) for t in range(300, t0+1)}
        if spec['branch'] in self.mismatch:
            prefixes[str(t0-3)] = fp(t0-3, base_delay(w, t0-3)+1e-9)
        return dict(prefixes=prefixes, final=fp(t_end, base_delay(w, t0)+self.J(spec['plan'])), actual_sim_s=t_end-300)


def charge_log():
    log = []
    return log, (lambda vid, specs: log.append((vid, [s['sim_s'] for s in specs])))


def test_identical_prefix_specs_and_window():
    snap = snapshot(t0=400);cands = [plan('a', duration=120), plan('b', duration=300)]
    specs = sv.build_branches(snap, cands)
    assert [s['branch'] for s in specs] == ['policy', 'cand1', 'cand2']
    assert all(s['window'] == specs[0]['window'] for s in specs)
    assert specs[0]['window']['end_s'] == 400+300+120
    assert len({s['snapshot_sha256'] for s in specs}) == 1
    for s in specs:
        verify_sealed(s, 'branch_sha256')
    # cap at episode end
    late = sv.build_branches(snapshot(t0=900), [plan('a', duration=300)])
    assert late[0]['window']['end_s'] == 1200 and late[0]['window']['truncated']


def test_best_choice_and_grade(tmp_path):
    snap = snapshot();log, charge = charge_log()
    cands = [plan('a', gain=1.), plan('b', gain=4.), plan('c', gain=2.), plan('d', gain=-1.)]
    r = sv.verify(snap, cands, FakePlant(), charge, root=tmp_path)
    assert r['status'] == 'accepted' and r['chosen'] == 1 and r['chosen_plan']['id'] == 'b'
    assert r['J_policy_veh_s'] == 500. and r['J_candidates_veh_s'] == [490., 460., 480., 510.]
    assert r['grade_veh_s'] == r['J_policy_veh_s']-r['J_candidates_veh_s'][1] == 40.
    verify_sealed(r, 'receipt_sha256')
    saved = json.loads((tmp_path/'verifications'/f'{r["verification_id"]}.json').read_text())
    assert saved['receipt_sha256'] == r['receipt_sha256']


def test_policy_fallback_when_no_candidate_beats_policy():
    log, charge = charge_log()
    r = sv.verify(snapshot(), [plan('a', gain=0.), plan('b', gain=-2.)], FakePlant(), charge)
    assert r['status'] == 'rejected' and r['chosen'] is None and r['chosen_plan'] is None
    assert r['grade_veh_s'] is None and r['J_candidates_veh_s'] == [500., 520.]  # tie with policy is NOT accepted
    assert r['best_margin_veh_s'] == 0.


def test_prefix_mismatch_stops_and_falls_back():
    log, charge = charge_log()
    r = sv.verify(snapshot(), [plan('a', gain=5.), plan('b', gain=1.)], FakePlant(mismatch={'cand2'}), charge)
    assert r['status'] == 'stopped' and r['chosen'] is None and r['grade_veh_s'] is None
    assert r['J_candidates_veh_s'] is None                     # never imputed
    assert [b['status'] for b in r['branches']] == ['graded', 'graded', 'stopped']
    assert r['charged_sim_s'] == 3*(400+120+120)               # precharged, no refund
    r2 = sv.verify(snapshot(), [plan('a', gain=5.)], FakePlant(fail={'policy'}), charge)
    assert r2['status'] == 'failed' and r2['chosen_plan'] is None


def test_parallel_determinism():
    cands = [plan(x, gain=g, duration=d) for x, g, d in (('a', 1., 60), ('b', 3., 300), ('c', 2., 120), ('d', .5, 90))]
    _, charge = charge_log()
    serial = sv.verify(snapshot(), cands, FakePlant(), charge, workers=1)
    plant = FakePlant(delay_s=.05)
    t = time.monotonic();par = sv.verify(snapshot(), cands, plant, charge, workers=5)
    assert time.monotonic()-t < 5*.05                          # really ran concurrently
    assert sorted(plant.calls) == ['cand1', 'cand2', 'cand3', 'cand4', 'policy']
    assert serial['decision_sha256'] == par['decision_sha256']
    assert [b['branch'] for b in par['branches']] == ['policy', 'cand1', 'cand2', 'cand3', 'cand4']


def test_budget_arithmetic(tmp_path):
    cands = [plan('a', duration=60), plan('b', duration=300)]
    charge = sv.local_charge(tmp_path)
    r = sv.verify(snapshot(t0=500), cands, FakePlant(), charge)
    t_end = 500+300+120
    assert r['branch_sim_s'] == t_end and r['n_branches'] == 3 and r['charged_sim_s'] == 3*t_end
    assert sv.quote(500, k=2, max_duration_s=300) == r['charged_sim_s']
    assert sv.quote(300, k=4, max_duration_s=300) == 5*720
    lines = [json.loads(x) for x in (tmp_path/'VERIFY_CHARGES.jsonl').read_text().splitlines()]
    assert sum(x['sim_s'] for x in lines) == r['charged_sim_s'] and len(lines) == 3

    def blocked(vid, specs):
        raise BudgetBlocked('cap')
    plant = FakePlant()
    b = sv.verify(snapshot(t0=500), cands, plant, blocked)
    assert b['status'] == 'budget_blocked' and plant.calls == [] and b['charged_sim_s'] == 0


def test_random_variant_same_function(monkeypatch):
    from tools.coopmwm import mppi_search

    def fake_sample(group, centre, bounds, regime, seed, covariance=None, config=None, duration_s=None):
        g = (seed % 1000)/100.
        return plan(f'r{seed}', group['members'], duration_s or 60, gain=g)
    monkeypatch.setattr(mppi_search, 'sample_unscored', fake_sample)
    first = plan('matched', duration=90, gain=.1)
    cands = sv.random_candidates(dict(members=['c1', 'c2']), None, (-3, 2), 'ordinary', 123, k=4,
                                 duration_s=90, first=first)
    assert len(cands) == 4 and cands[0]['id'] == 'matched' and len({c['id'] for c in cands}) == 4
    assert cands == sv.random_candidates(dict(members=['c1', 'c2']), None, (-3, 2), 'ordinary', 123, k=4,
                                         duration_s=90, first=first)   # seeds from base only
    _, charge = charge_log()
    r = sv.verify(snapshot(), cands, FakePlant(), charge, proposer='random')
    best = max(range(4), key=lambda i: (sum(cands[i]['profile'][0]), -i))
    assert r['proposer'] == 'random' and r['status'] == 'accepted' and r['chosen'] == best
    with pytest.raises(ValueError):
        sv.build_branches(snapshot(), cands, proposer='oracle')


def test_rejects_candidates_overlapping_committed_plans():
    snap = snapshot(committed=[dict(plan=plan('old', members=('c2', 'c9')), start_s=380)])
    with pytest.raises(ValueError):
        sv.build_branches(snap, [plan('a', members=('c1', 'c2'))])


# ---------------------------------------------------------------- live hook

class FakeController:
    def __init__(self, t0):
        self.events = [dict(time_s=t, sample=dict(action=[[0.]])) for t in range(300, t0)]
        self.memory = dict(c1=.1, c2=-.2)
        self.active_plans = []
        self.policy = torch.nn.Linear(1, 1)


def live_guide(proposal):
    calls = []
    def g(rows, commands, now, sample):
        return json.loads(json.dumps(proposal))
    g.completed = lambda p, executed: calls.append(executed)
    return g, calls


def hook(gen, verifier, proposal, t0=400, rp=None):
    ctrl = FakeController(t0)
    prefixes = {str(t): fp(t, base_delay(7, t)) for t in range(300, t0+1)}
    job = dict(id='warm_u005_B_7', world=dict(seed=7), arm='B', update=5, runtime='rt', checkpoint='ck', stop=1200)
    g, calls = live_guide(proposal)
    return sv.verified_guide(g, verifier, ctrl, prefixes, job, gen, random_proposer=rp), calls


def test_hook_accepts_sumo_choice_and_keeps_rng_aligned():
    cands = [plan('a', gain=1.), plan('b', gain=4.)]
    proposal = dict(accepted=True, plan=cands[0], topk=cands, behaviour=dict(route='wm'))
    seen = {}
    def verifier(snap, c, proposer):
        seen.update(snap=snap, cands=c, proposer=proposer)
        return sv.verify(snap, c, FakePlant(), lambda *a: None)
    gen = torch.Generator().manual_seed(5)
    call, calls = hook(gen, verifier, proposal)
    out = call([], {}, 400, dict(action=[[0.]]))
    assert out['accepted'] and out['plan']['id'] == 'b' and out['exact_grade_veh_s'] == 40.
    assert out['behaviour']['sumo_verification']['status'] == 'accepted'
    assert seen['proposer'] == 'wm' and [c['id'] for c in seen['cands']] == ['a', 'b']
    assert torch.equal(seen['snap']['generator_state'], torch.Generator().manual_seed(5).get_state())
    assert seen['snap']['rng']['memory_after_t0'] == dict(c1=.1, c2=-.2)
    call.completed(out, True);assert calls == [True]          # quota counts executed

    # rejected: proposal declined, one pad draw consumed (same as the beta draw of an accepted plan)
    gen2 = torch.Generator().manual_seed(5)
    bad = dict(accepted=True, plan=cands[0], topk=[plan('a', gain=-1.)], behaviour=dict(route='wm'))
    call2, calls2 = hook(gen2, lambda s, c, p: sv.verify(s, c, FakePlant(), lambda *a: None), bad)
    out2 = call2([], {}, 400, dict(action=[[0.]]))
    assert not out2['accepted'] and 'rejected' in out2['reason']
    ref = torch.Generator().manual_seed(5);torch.rand((), generator=ref)
    assert torch.equal(gen2.get_state(), ref.get_state())
    call2.completed(out2, False);assert calls2 == [False]       # quota: not executed


def test_hook_passthrough_and_random_arm():
    gen = torch.Generator().manual_seed(1)
    call, _ = hook(gen, lambda *a: pytest.fail('no verification for declined proposals'),
                   dict(accepted=False, behaviour={}))
    assert call([], {}, 400, {})['accepted'] is False
    cands = [plan('m', gain=.5), plan('x', gain=2.)]
    rp = lambda proposal, k: cands
    call, _ = hook(gen, lambda s, c, p: sv.verify(s, c, FakePlant(), lambda *a: None, proposer=p),
                   dict(accepted=True, plan=cands[0], random_match=dict(path='m', key='k'), behaviour={}), rp=rp)
    out = call([], {}, 400, {})
    assert out['plan']['id'] == 'x' and out['behaviour']['sumo_verification']['grade_veh_s'] == 20.


def test_preverification_trust_skip_never_charges_or_verifies():
    gen = torch.Generator().manual_seed(5)
    ctrl = FakeController(400);ctrl.beta = 0.
    prefixes = snapshot()['prefixes'];job = dict(id='e', world={}, arm='B', runtime='rt', checkpoint='ck')
    guide, _ = live_guide(dict(accepted=True, plan=plan('a'), behaviour={}))
    call = sv.verified_guide(guide, lambda *a: pytest.fail('no verification'), ctrl, prefixes, job, gen)
    out = call([], {}, 400, {})
    assert not out['accepted'] and out['sumo_trust_drawn']
    assert out['behaviour']['trust_follow'] is False
    ref = torch.Generator().manual_seed(5);torch.rand((), generator=ref)
    assert torch.equal(gen.get_state(), ref.get_state())


def test_completed_verification_adopts_receipt_without_physics_or_charge(tmp_path):
    cands = [plan('a', gain=1.)];snap = snapshot()
    original = sv.verify(snap, cands, FakePlant(), lambda *a: None, root=tmp_path)
    plant = FakePlant()
    adopted = sv.verify(snap, cands, plant, lambda *a: pytest.fail('duplicate charge'), root=tmp_path)
    assert adopted == original and plant.calls == []
    changed = [dict(cands[0], saving_vs_policy=99.)]
    with pytest.raises(ValueError, match='identity changed'):
        sv.verify(snap, changed, plant, lambda *a: pytest.fail('charge'), root=tmp_path)


def test_interrupted_verification_claim_never_replays(tmp_path):
    cands = [plan('a', gain=1.)];snap = snapshot()
    vid = sv.verification_id(snap['episode']['id'], snap['t0'], cands)
    sv._write_new(tmp_path/'claims'/f'{vid}.json', dict(interrupted=True))
    plant = FakePlant()
    with pytest.raises(FileExistsError):
        sv.verify(snap, cands, plant, lambda *a: pytest.fail('duplicate charge'), root=tmp_path)
    assert plant.calls == []


def test_accept_and_decline_each_consume_exactly_one_trust_draw():
    for gain in (1., -1.):
        gen = torch.Generator().manual_seed(5)
        call, _ = hook(gen, lambda s, c, p: sv.verify(s, c, FakePlant(), lambda *a: None),
                       dict(accepted=True, plan=plan('a', gain=gain), behaviour={}))
        out = call([], {}, 400, {})
        assert out['accepted'] == (gain > 0.)
        assert out['sumo_trust_drawn']
        ref = torch.Generator().manual_seed(5);torch.rand((), generator=ref)
        assert torch.equal(gen.get_state(), ref.get_state())


def test_real_controller_never_draws_trust_again_for_verified_plan(monkeypatch):
    from tools.coopnative.test_cpu import rows
    from tools.coopmwm.acceleration import Controller, Policy, encode
    from tools.coopmwm import mppi_search
    data = rows();context = encode(data);members = context['members']
    monkeypatch.setattr(mppi_search, 'group_components', lambda *a, **kw: [dict(members=members)])
    p = Policy();gen = torch.Generator().manual_seed(12);ref = torch.Generator().manual_seed(12)
    p.sample(context, ref, False, {})  # Own action sample is the only expected draw here.
    profile = dict(id='verified', members=members, duration_s=30, knots=[[.2]*10 for _ in members], family='MPPI')
    def guide(*args):
        return dict(accepted=True, sumo_trust_drawn=True, sumo_verified=True,
                    plan=profile, option='MPPI', behaviour=dict(regime='ordinary', sample_probability=.95))
    c = Controller(p, gen, guide=guide, beta=.5)
    c.apply(data, {v: 0. for v in members}, 300)
    assert c.events[-1]['sample']['behaviour']['plan_start']
    assert torch.equal(gen.get_state(), ref.get_state())
