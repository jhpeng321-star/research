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
