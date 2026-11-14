"""Remaining global approval is shared correctly with concurrent plan-fix work."""
from copy import deepcopy

import pytest

from . import clean2_budget as cb, rerun_cpus as rc, launch_production as production
from .exact_grade import _seal
from .protocol import digest


def quote(kind, u, cells, costs):
    rows = [dict(id=f'{kind}_{u}_{i}', sim_s=n, window=dict(start_s=300, end_s=1200,
                 truncated=False), snapshot_sha256='a'*64, branch_sha256='b'*64)
            for i, n in enumerate(costs)]
    return _seal(dict(kind=kind, update=u, cells=cells, status='dry_run', new_sim_s=0,
        branches=len(rows), branch_manifest=rows, quote_sha256=digest(rows),
        quoted_sim_s=sum(costs)), 'receipt_sha256')


@pytest.fixture
def args():
    quotes = {f'probe_u{u:03}': quote('probe', u, 3, [7580]*3) for u in (2, 4)}
    quotes.update({f'replicated_u{u:03}': quote('replicated', u, 10, [4792]*10)
                   for u in (6, 12, 24)})
    return dict(clean1_rows=[dict(id='train1', phase='v2v_train', sim_s=1214880),
                            dict(id='plan1', phase='plan_fix', sim_s=2400)],
        clean1_ledger_sha256='a'*64, clean1_stop_sha256='b'*64,
        accepted_fixes={f'F{i}':str(i)*64 for i in range(1, 7)},
        diagnostic_quotes=quotes,
        gate_manifest=[dict(id='gate1', sim_s=15160, window=dict(truncated=False))],
        canonical_budget=rc.budget(track2=True))


def test_plan_fix_is_counted_once_and_its_live_remainder_is_protected(args):
    first = cb.build(**args)
    assert first['clean1_spent_sim_s'] == 1217280
    assert first['plan_fix_unspent_reserved_sim_s'] == 597600
    assert first['protected_prior_sim_s'] == 1815840
    assert first['budget']['total'] == 3184160
    # A live plan-fix append merely converts a reserved amount to a spent amount;
    # it must neither shrink clean2 a second time nor free additional capacity.
    args['clean1_rows'].append(dict(id='plan2', phase='plan_fix', sim_s=200000))
    later = cb.build(**args)
    assert first['budget'] == later['budget']
    assert first['protected_prior_sim_s'] == later['protected_prior_sim_s']
    assert later['plan_fix_unspent_reserved_sim_s'] == 397600
    assert later['cumulative_upper_bound_sim_s'] <= 5000000


def test_voids_and_new_non_plan_fix_work_stay_spent(args):
    first = cb.build(**args)
    args['clean1_rows'].append(dict(id='void', phase='track2_smoke', sim_s=9600,
                                   void_environment_failure=True))
    later = cb.build(**args)
    assert later['budget']['total'] == first['budget']['total']-9600


@pytest.mark.parametrize('fault', ['missing_fix', 'missing_schedule', 'truncated',
                                  'missing_branch', 'bad_sha', 'over_plan_fix', 'over_total'])
def test_incomplete_or_overbudget_registration_refused(args, fault):
    if fault == 'missing_fix':
        del args['accepted_fixes']['F4']
    elif fault == 'missing_schedule':
        del args['diagnostic_quotes']['replicated_u024']
    elif fault in ('truncated', 'missing_branch', 'bad_sha'):
        row = args['diagnostic_quotes']['replicated_u024']
        if fault == 'truncated':
            row['branch_manifest'][0]['window']['truncated'] = True
            row['quote_sha256'] = digest(row['branch_manifest'])
        elif fault == 'missing_branch':
            row['branch_manifest'].pop()
        else:
            row['quote_sha256'] = 'f'*64
        args['diagnostic_quotes']['replicated_u024'] = _seal(
            {k:v for k,v in row.items() if k != 'receipt_sha256'}, 'receipt_sha256')
    elif fault == 'over_plan_fix':
        args['clean1_rows'][1]['sim_s'] = 600001
    else:
        args['clean1_rows'][0]['sim_s'] = 4000000
    with pytest.raises(ValueError):
        cb.build(**args)


def test_actual_backend_enforces_disjoint_budget_without_changing_canonical_pins(args):
    cfg = rc.runtime_config({}, root='/home/ubuntu/coopmwm20261105_clean2',
        reference_root='/home/ubuntu/reference', source_root='/home/ubuntu/source',
        common_checkpoint_sha256='a'*64, track2=True)
    cfg.update(launch_optimization=deepcopy(production.OPTIMIZATION),
        launch_contingency=deepcopy(production.CONTINGENCY), clean2_budget=cb.build(**args))
    original = deepcopy(cfg)
    native = production.backend_config(cfg)
    assert rc.check_pins(cfg) and cfg == original
    assert native['budget']['total'] == 5000000
    assert native['budget']['total']-native['prior_charged_sim_s'] == 3184160
    assert native['budget']['remaining_cap_sim_s'] == 3184160
    assert native['prior_charged_sim_s'] == 1815840
    assert native['budget']['phases']['augment_verify'] == 0
    assert native['budget']['phases']['track2_smoke'] == 20000
    assert native['budget']['phases']['gate_bank_verification_ext'] == 30320
    assert native['budget']['phases']['early_probes'] == 45480
    assert native['budget']['phases']['isolated_replicated'] == 143760
    assert native['budget']['phases']['v2v_train'] == rc.budget(track2=True)['phases']['v2v_train']
    # Even a correctly resealed edited summary cannot silently increase a phase.
    edited = deepcopy(cfg['clean2_budget'])
    edited['budget']['phases']['track2_smoke'] += 1
    cfg['clean2_budget'] = _seal({k:v for k,v in edited.items() if k != 'receipt_sha256'}, 'receipt_sha256')
    with pytest.raises(ValueError, match='differs'):
        production.backend_config(cfg)


def test_real_p1_ledger_does_not_double_subtract_prior_and_refuses_phase_overrun(args, tmp_path, monkeypatch):
    import os
    import sys
    from types import SimpleNamespace
    from . import track2_backend as p1
    from .storage import ledger
    # Only the OS lock is replaced on Windows. Exercise the actual P1 reserve
    # function and real append-only accounting, not a test-only budget replica.
    if os.name == 'nt':
        monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(
            LOCK_EX=2, LOCK_SH=1, LOCK_UN=8, flock=lambda *args: None))
    value = cb.build(**args)
    native = dict(run_root=str(tmp_path), accounting_root=str(tmp_path),
        budget=dict(value['budget'], total=cb.CAP),
        prior_charged_sim_s=value['protected_prior_sim_s'])
    amount = value['budget']['phases']['v2v_train']
    assert amount > value['budget']['total']-value['protected_prior_sim_s']
    p1._reserve(native, [dict(id='whole_train', sim_s=amount)], 'v2v_train')
    assert sum(row['sim_s'] for row in ledger(tmp_path)) == amount
    with pytest.raises(ValueError, match='whole-wave budget'):
        p1._reserve(native, [dict(id='excess_smoke', sim_s=20001)], 'track2_smoke')
    assert len(ledger(tmp_path)) == 1
