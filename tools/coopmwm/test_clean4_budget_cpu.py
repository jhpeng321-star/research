"""CPU contract tests for the clean4 quote/coverage/approval helpers (zero physics)."""
import pytest

from tools.coopmwm import clean4_budget as b


def _phases():
    return b.quote(lesson_sim_s=35040, probe_sim_s=45480, replicated_sim_s=143760,
                   g4_base_sim_s=153600, g4_cap_sim_s=230400, diagnostic_factor=1.5)


def test_every_phase_has_15_percent_margin_and_full_schedule():
    phases = _phases()
    assert set(phases) == {'v2v_train_waves', 'v2v_practice_refs', 'v2v_lessons', 'early_probes',
                           'isolated_replicated', 'g4_synergy_credit', 'v2v_final'}
    for p in phases.values():
        assert p['allocated_sim_s'] >= p['base_sim_s']*1.15
    assert phases['v2v_train_waves']['base_sim_s'] == 24*3*16*1200
    assert phases['v2v_final']['base_sim_s'] == 5*100*1200


def test_allocation_rounds_up_and_refuses_bad_input():
    assert b.allocation(1200) == 2400
    assert b.allocation(0) == 0
    with pytest.raises(ValueError):
        b.allocation(-1)
    with pytest.raises(ValueError):
        b.phase('x', 1000, basis='', allocated=1100)
    with pytest.raises(ValueError):
        b.quote(lesson_sim_s=1, probe_sim_s=1, replicated_sim_s=1, g4_base_sim_s=0,
                g4_cap_sim_s=0, diagnostic_factor=0.5)


def test_document_seal_fit_and_runtime_mapping():
    phases = _phases()
    doc = b.document(phases, prior=dict(a=1000), cap=10_000_000, created='t')
    assert doc['fits'] and doc['shortfall_sim_s'] == 0
    rt = doc['runtime_phases']
    assert rt['v2v_train'] == phases['v2v_train_waves']['allocated_sim_s']+phases['v2v_practice_refs']['allocated_sim_s']
    assert rt['v2v_interim'] == 0
    assert rt['track2_smoke']+rt['gate_bank_verification_ext'] == b.SMOKE_PHASE_CAP
    small = b.document(phases, prior=dict(a=1000), cap=1_000_000, created='t')
    assert not small['fits'] and small['shortfall_sim_s'] > 0


def test_coverage_refuses_short_phase():
    assert b.coverage(dict(a=10, b=5), dict(a=10))['need'] == dict(a=10)
    with pytest.raises(ValueError):
        b.coverage(dict(a=10), dict(a=11))
    with pytest.raises(ValueError):
        b.coverage(dict(a=10), dict(b=1))


def test_smoke_gate_subphase_cap():
    rows = [dict(phase='track2_smoke', sim_s=14400), dict(phase='gate_bank_verification_ext', sim_s=15160),
            dict(phase='v2v_train', sim_s=99999)]
    assert b.smoke_gate_spent(rows) == dict(smoke=14400, gate=15160, total=29560)
    with pytest.raises(ValueError):
        b.smoke_gate_spent([dict(phase='track2_smoke', sim_s=20001)])
    with pytest.raises(ValueError):
        b.smoke_gate_spent([dict(phase='track2_smoke', sim_s=20000),
                            dict(phase='gate_bank_verification_ext', sim_s=20001)])


def test_seed_ranges():
    assert b.check_seeds([989200, 989599], [989600, 989699]) == dict(train=2, final=2)
    for train, final in (([989600], [989650]), ([989199], [989600]), ([989300], [989700]),
                         ([964100], [989600]), ([989300], [989300])):
        with pytest.raises(ValueError):
            b.check_seeds(train, final)


def test_owner_approval_binding():
    doc = b.document(_phases(), prior=dict(a=1000), cap=10_000_000, created='t')
    good = dict(owner_approval='yes, approve.', quote_sha256=doc['receipt_sha256'],
                approved_sim_s=doc['total_training_sim_s'], total_cap_sim_s=10_000_000)
    assert b.approved(good, doc)['approved_sim_s'] == doc['total_training_sim_s']
    for change in (dict(owner_approval=''), dict(quote_sha256='0'*64),
                   dict(approved_sim_s=doc['total_training_sim_s']-1), dict(total_cap_sim_s=1000)):
        with pytest.raises(ValueError):
            b.approved(dict(good, **change), doc)
    tampered = dict(doc, total_training_sim_s=1)
    with pytest.raises(ValueError):
        b.approved(good, tampered)
