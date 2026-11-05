"""Cumulative clean3 admission cannot conceal charges, reservations or stale quotes."""
from copy import deepcopy

import pytest

from . import clean3_budget as cb
from .exact_grade import _seal
from .protocol import digest
from .test_clean2_budget_cpu import quote


@pytest.fixture
def inputs():
    diagnostics = {f'probe_u{u:03}': quote('probe', u, 3, [7580]*3) for u in (2, 4)}
    diagnostics.update({f'replicated_u{u:03}': quote('replicated', u, 10, [4792]*10)
                        for u in (6, 12, 24)})
    return dict(clean1_rows=[dict(id='prior', phase='v2v_train', sim_s=1214880),
                            dict(id='plan1', phase='plan_fix', sim_s=33600)],
        clean2_rows=[dict(id='prior', phase='v2v_train', sim_s=900000)],
        owner_budget=dict(owner_approval='yes, approve.', total_cap_sim_s=6000000,
                          scope=cb.OWNER_SCOPE, registered='2026-11-05 23:39'),
        diagnostic_quotes=diagnostics,
        gate_manifest=[dict(id='gate', sim_s=15160, window=dict(truncated=False))])


def admission_args(inputs):
    return dict(clean1_rows=inputs['clean1_rows'], clean2_rows=inputs['clean2_rows'],
        owner_budget=inputs['owner_budget'], accepted_levers={name: 'a'*64 for name in ('G1', 'G2', 'G4')},
        clean2_stop_sha256='b'*64, sealed_updates=dict(A=14, B=14, R=14))


def test_full_quote_keeps_every_schedule_cap_and_plan_fix_once(inputs):
    first = cb.prospective(**inputs)
    assert first['allocated_sim_s'] == 3241440
    assert first['prior_charged_sim_s'] == 2149440
    assert first['plan_fix_unspent_reserved_sim_s'] == 566400
    assert first['cumulative_upper_bound_sim_s'] == 5957280
    assert first['margin_sim_s'] == 42720
    assert first['launch_authorized'] is False
    inputs['clean1_rows'].append(dict(id='plan2', phase='plan_fix', sim_s=200000))
    later = cb.prospective(**inputs)
    assert later['cumulative_upper_bound_sim_s'] == first['cumulative_upper_bound_sim_s']
    assert later['plan_fix_unspent_reserved_sim_s'] == 366400
    admitted = cb.admit(later, **admission_args(inputs))
    assert admitted['budget']['allocated'] == 3241440
    assert admitted['smoke_and_gate_required'] and not admitted['launch_authorized']


def test_real_observed_prefix_is_blocked_even_without_protected_reserve(inputs):
    inputs['clean2_rows'][0]['sim_s'] = 1614000
    value = cb.prospective(**inputs)
    assert value['prior_charged_sim_s'] == 2863440
    assert value['cumulative_upper_bound_sim_s'] == 6671280
    assert value['margin_sim_s'] == -671280 and value['status'] == 'BLOCKED_BUDGET'
    assert value['allocated_sim_s']+value['prior_charged_sim_s'] > cb.CAP
    with pytest.raises(ValueError, match='exceeds cumulative'):
        cb.admit(value, **admission_args(inputs))


def test_later_charge_and_environment_void_make_old_quote_stale(inputs):
    value = cb.prospective(**inputs)
    inputs['clean2_rows'].append(dict(id='void', phase='track2_smoke', sim_s=9600,
                                     void_environment_failure=True))
    with pytest.raises(ValueError, match='differs from current ledgers'):
        cb.admit(value, **admission_args(inputs))
    later = cb.prospective(**inputs)
    assert later['margin_sim_s'] == value['margin_sim_s']-9600


@pytest.mark.parametrize('fault', ['duplicate', 'negative', 'bool', 'plan_over',
    'wrong_plan_root', 'approval', 'scope', 'schedule', 'truncated', 'empty_gate'])
def test_invalid_evidence_never_produces_admission(inputs, fault):
    if fault == 'duplicate':
        inputs['clean1_rows'].append(deepcopy(inputs['clean1_rows'][0]))
    elif fault == 'negative':
        inputs['clean2_rows'][0]['sim_s'] = -1
    elif fault == 'bool':
        inputs['clean2_rows'][0]['sim_s'] = True
    elif fault == 'plan_over':
        inputs['clean1_rows'][1]['sim_s'] = 600001
    elif fault == 'wrong_plan_root':
        inputs['clean2_rows'][0]['phase'] = 'plan_fix'
    elif fault == 'approval':
        inputs['owner_budget']['total_cap_sim_s'] = 7000000
    elif fault == 'scope':
        inputs['owner_budget']['scope'] = 'clean3 only'
    elif fault == 'schedule':
        del inputs['diagnostic_quotes']['replicated_u024']
    elif fault == 'truncated':
        inputs['gate_manifest'][0]['window']['truncated'] = True
    else:
        inputs['gate_manifest'] = []
    with pytest.raises(ValueError):
        cb.prospective(**inputs)


@pytest.mark.parametrize('fault', ['acceptance', 'stop', 'boundary', 'forged_phase', 'forged_reserve'])
def test_stopped_boundary_and_acceptance_are_mandatory_and_summaries_recomputed(inputs, fault):
    value = cb.prospective(**inputs)
    args = admission_args(inputs)
    if fault == 'acceptance':
        del args['accepted_levers']['G4']
    elif fault == 'stop':
        args['clean2_stop_sha256'] = None
    elif fault == 'boundary':
        args['sealed_updates']['R'] = True
    else:
        if fault == 'forged_phase':
            value['phases']['g4_synergy_credit'] = 0
        else:
            value['plan_fix_unspent_reserved_sim_s'] = 0
        value = _seal({key: val for key, val in value.items() if key != 'receipt_sha256'},
                      'receipt_sha256')
    with pytest.raises(ValueError):
        cb.admit(value, **args)


def test_native_diagnostic_manifest_cannot_be_shortened_and_resealed(inputs):
    row = inputs['diagnostic_quotes']['replicated_u024']
    row['branch_manifest'][0]['window']['truncated'] = True
    row['quote_sha256'] = digest(row['branch_manifest'])
    inputs['diagnostic_quotes']['replicated_u024'] = _seal(
        {key: val for key, val in row.items() if key != 'receipt_sha256'}, 'receipt_sha256')
    with pytest.raises(ValueError, match='truncated'):
        cb.prospective(**inputs)
