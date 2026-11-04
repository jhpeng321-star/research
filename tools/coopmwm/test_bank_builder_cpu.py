import json
import pytest
from .bank_builder import PHASE, precharged_charge, quoted_jobs, run_all8_round


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


def test_run_all8_round_passes_slim_and_extra_seeds_from_config(tmp_path, monkeypatch):
    """run_all8_round reads slim_rejected_tapes (and extra_seeds) from its BINDING/config and
    passes both to confirm_all8 (slim None = the C4NM_SLIM_REJECTED_TAPES env var decides) and
    records the option in the claim; no physics: confirm_all8 is stubbed. The stubs are installed
    on the SAME module instances run_all8_round's call-time relative imports resolve to (pytest
    may import this test package as 'coopmwm' or 'tools.coopmwm'; never mix the two copies)."""
    import importlib
    import sys
    bb = sys.modules[run_all8_round.__module__]
    pkg = bb.__package__
    vp = importlib.import_module(pkg+'.verify_perseed')
    rb = importlib.import_module(pkg+'.replay_bank')
    accel = importlib.import_module(pkg+'.accel_storage')
    storage = importlib.import_module(pkg+'.storage')
    decision = dict(verification_id='vfix', _branches='bd', _policy_version='u9', status='accepted')
    monkeypatch.setattr(vp, 'pending', lambda root, arm, sources=(): [decision])
    monkeypatch.setattr(vp, 'first_seed_candidates', lambda r, bd, candidates=None: [])
    monkeypatch.setattr(rb, '_decision_snapshot', lambda root, arm, r, by_vid: dict(snap='s'))
    monkeypatch.setattr(bb, 'quoted_jobs_all8', lambda snap, cands, vid, verify, replicate:
                        [dict(id=f'confirm_seed2_all8_{vid}_policy', sim_s=7)])
    monkeypatch.setattr(accel, 'reserve', lambda root, jobs, phase: None)
    monkeypatch.setattr(storage, 'ledger', lambda root: [])
    seen = {}
    def fake_confirm_all8(root, arm, receipt, snapshot, runner_for, run_root, phase,
                          workers=2, **kw):
        seen.update(kw);seen['vid'] = receipt['verification_id']
        return dict(admitted=[], winners={}, confirmation_sha256='ab'*32, charged_sim_s=0)
    monkeypatch.setattr(vp, 'confirm_all8', fake_confirm_all8)

    def round(config, name):
        control = tmp_path/name/'control'
        (control/'attempts_all8').mkdir(parents=True)
        immutable = lambda p, d: p.write_text(json.dumps(d))
        run_all8_round(tmp_path/name/'root', control, config, immutable)
        return control

    ctrl = round(dict(sources=[], branch_timeout_s=1, poll_s=0, extra_seeds=3,
                      slim_rejected_tapes=True), 'on')
    assert seen['slim_rejected_tapes'] is True and seen['extra_seeds'] == 3 and seen['vid'] == 'vfix'
    claim = json.loads((ctrl/'attempts_all8'/'vfix.claim.json').read_text())
    assert claim['slim_rejected_tapes'] is True and claim['verify_mode'] == vp.ALL8
    attempt = json.loads((ctrl/'attempts_all8'/'vfix.json').read_text())
    assert attempt['status'] == 'complete' and attempt['record_sha256'] == 'ab'*32
    seen.clear()
    ctrl2 = round(dict(sources=[], branch_timeout_s=1, poll_s=0), 'absent')
    assert seen['slim_rejected_tapes'] is None and seen['extra_seeds'] == 0   # env var decides
    claim2 = json.loads((ctrl2/'attempts_all8'/'vfix.claim.json').read_text())
    assert claim2['slim_rejected_tapes'] is None
