import pytest
from .bank_builder import PHASE, precharged_charge, quoted_jobs


def test_whole_decision_quote_separates_kinds_and_retains_solos():
    def build(s, candidates, tail_s):
        return [dict(verification_id=candidates[0]['id'], branch=b, sim_s=tail_s+100)
                for b in ('policy', 'candidate_0')]
    best = dict(id='group', members=['a', 'b', 'c'])
    def solo(p, m):return dict(id=m)
    jobs = quoted_jobs({}, best, build, solo)
    assert len(jobs) == 10 and len({j['id'] for j in jobs}) == 10
    assert sum(j['sim_s'] for j in jobs) == 2*700+8*220
    assert all(j['id'].startswith('confirm_') for j in jobs)


def test_precharge_hook_never_reserves_and_requires_exact_phase_and_cost():
    entries = {'confirm_extended_v_policy': dict(phase=PHASE, sim_s=1000)}
    charge = precharged_charge(entries, 'confirm_extended')
    charge('v', [dict(branch='policy', sim_s=1000)])
    with pytest.raises(AssertionError):charge('v', [dict(branch='policy', sim_s=999)])
    with pytest.raises(KeyError):charge('v', [dict(branch='missing', sim_s=1000)])
    entries['confirm_extended_v_policy']['phase'] = 'train'
    with pytest.raises(AssertionError):charge('v', [dict(branch='policy', sim_s=1000)])
