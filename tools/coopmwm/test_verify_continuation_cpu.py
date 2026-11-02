"""Verifier CRN replay and first-window equivalence; never launches SUMO."""
import copy

import pytest
import torch

from tools.coopmwm import exact_grade as xg, sumo_verify as sv
from tools.coopmwm.protocol import digest
from tools.coopmwm.test_sumo_verify_cpu import FakePlant, fp, plan, snapshot


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('accepted', [False, True])
def test_verification_trust_replay_includes_rejection(arm, accepted):
    p = dict(accepted=accepted, sumo_trust_drawn=True)
    if arm == 'R':
        p['random_match'] = dict(key='ticket')
    step = dict(sample=dict(behaviour=dict(eligible_query=True)), proposal=p)
    actual = torch.Generator().manual_seed(321)
    torch.rand((), generator=actual)  # query admission
    if arm == 'R':
        torch.randint(2**31, (), generator=actual)  # original matched sampler
    torch.rand((), generator=actual)  # pre-verification trust draw
    replay = torch.Generator().manual_seed(321)
    for call in xg.guide_draws(step, arm):
        xg.issue(replay, call)
    assert torch.equal(actual.get_state(), replay.get_state())


def test_legacy_decline_has_no_trust_or_random_sampler_draw():
    step = dict(sample=dict(behaviour=dict(eligible_query=True)), proposal=dict(accepted=False))
    assert xg.guide_draws(step, 'R') == ['rand']


def continuation():
    snap = snapshot(t0=400)
    c = plan('winner', gain=2.)
    live = dict(id=snap['episode']['id'], prefixes={str(t): fp(t, t*2.) for t in range(400, 411)},
                final=fp(1200, 2400.))

    class WithFingerprints(FakePlant):
        def run(self, spec, state):
            r = super().run(spec, state)
            r['continuation_prefix_sha256'] = {t: digest(v) for t, v in live['prefixes'].items()}
            return r

    verification = sv.verify(snap, [c], WithFingerprints(), lambda *args: None)
    live['steps'] = [dict(time_s=400, proposal=dict(plan=c), sample=dict(behaviour=dict(
        executed=True, plan_start=True, sumo_verification=dict(receipt_sha256=verification['receipt_sha256']))))]
    return verification, live


def test_chosen_branch_matches_live_to_next_query_window():
    r, live = continuation()
    gate = sv.check_live_continuation(r, live)
    assert gate['pass_'] and gate['checked_to_s'] == 410 and gate['fingerprints'] == 11
    assert gate['new_sim_s'] == 0
    # Later queries may change traffic: no overclaim about their trajectories.
    live['prefixes']['411'] = fp(411, -1.)
    assert sv.check_live_continuation(r, live) == gate


@pytest.mark.parametrize('fault', ['missing', 'different', 'wrong_plan', 'wrong_receipt', 'not_executed'])
def test_smoke_rejects_incomplete_or_nonidentical_live_continuation(fault):
    r, live = continuation()
    live = copy.deepcopy(live)
    b = live['steps'][0]['sample']['behaviour']
    if fault == 'missing':
        live['prefixes'].pop('405')
    elif fault == 'different':
        live['prefixes']['410'] = fp(410, 820.000000001)
    elif fault == 'wrong_plan':
        live['steps'][0]['proposal']['plan']['id'] = 'another'
    elif fault == 'wrong_receipt':
        b['sumo_verification']['receipt_sha256'] = 'another'
    else:
        b['executed'] = False
    with pytest.raises((sv.PrefixMismatch, ValueError)):
        sv.check_live_continuation(r, live)


def test_actual_episode_loop_and_controller_match_verified_branch(tmp_path, monkeypatch):
    """Exercise capture, branch controller, command replay and live installation together."""
    from tools.coopmwm import accel_plant, plan_client
    from tools.coopmwm.accel_storage import read
    from tools.coopmwm.test_exact_grade_cpu import WORLD, fake_sumo, policy

    class FourCandidates:
        def __init__(self, generator):
            self.generator = generator

        def __call__(self, rows, commands, now, sample):
            if now != 320:
                return None
            from tools.coopmwm.mppi_search import group_components
            members = group_components(rows)[0]['members']
            torch.rand((), generator=self.generator)  # one admitted query
            topk = [dict(id=f'P320_{i}', members=members, duration_s=30,
                         knot_s=10, family='MPPI', knots=[[a]*4 for _ in members])
                    for i, a in enumerate((-.5, .5, 1.5, 2.6))]
            return dict(accepted=True, option='MPPI', plan=topk[0], topk=topk,
                        behaviour=dict(route='wm', eligible_query=True, regime='ordinary',
                                       sample_probability=1., executed=False))

    ident = 'warm_u001_B_986001'
    root = tmp_path/'verify'
    charges = []
    monkeypatch.setattr(plan_client, 'guide', lambda job, g: FourCandidates(g))
    monkeypatch.setattr(sv, 'SubprocessBranchRunner', lambda work, **kw: sv.LocalBranchRunner(work))
    monkeypatch.setattr(sv, 'accel_charge', lambda *a, **kw: lambda vid, specs: charges.append(specs))
    job = dict(id=ident, path=str(tmp_path/'jobs'/ident), world=WORLD, role='train', arm='B',
               weights=policy().state_dict(), runtime='', checkpoint='', noise_seed=11,
               deterministic=False, sim_s=520, stop=520, queue_root=str(tmp_path),
               accel_bounds=[-3., 2.6], gpu_deadline=0, guided=False, beta=1., warm_history=False,
               sumo_verify=dict(root=str(root), run_root=str(tmp_path), workers=1))
    with fake_sumo():
        live = accel_plant.episode(job)
    receipts = [read(p) for p in (root/'verifications').glob('*.json')]
    assert len(receipts) == 1 and receipts[0]['status'] == 'accepted'
    assert len(charges) == 1 and len(charges[0]) == 5
    assert all(b['status'] == 'graded' for b in receipts[0]['branches'])
    assert sv.check_live_continuation(receipts[0], live)['checked_to_s'] == 330
