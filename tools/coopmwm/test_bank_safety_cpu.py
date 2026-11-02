import copy
import pytest
from .bank_safety import verdict, reevaluate


@pytest.mark.parametrize('collision,teleport,brake,ttc,expected', [
    (0, 0, 3, -20, True), (0, 0, 3, -92, True),
    (1, 0, -10, -20, False), (0, 1, -10, -20, False),
    (0, 0, 3, -2, False), (0, 0, -3, 3, True)])
def test_shared_combined_rule_hard_constraints(collision, teleport, brake, ttc, expected):
    d = dict(collisions=collision, teleports=teleport, hard_brake=brake,
             ttc_below_1p5_events=ttc)
    result = verdict(d)
    assert result['new_combined_rule'] is expected
    assert result['old_no_metric_increase'] == all(v <= 0 for v in d.values())


def test_revalidation_retains_original_and_does_not_relax_replication():
    old = dict(confirmation_sha256='old', admitted=False,
        checks=dict(replication=True, no_later_harm=True, safety=False),
        safety_deltas={k:dict(collisions=0, teleports=0, hard_brake=3,
            ttc_below_1p5_events=-20) for k in ('replication', 'extended')})
    before = copy.deepcopy(old)
    new = reevaluate(old)
    assert old == before and new['admitted'] is True
    assert new['old_admitted'] is False and new['old_checks']['safety'] is False
    assert new['original_confirmation_sha256'] == 'old'
    old['checks']['replication'] = False
    assert reevaluate(old)['admitted'] is False


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_reconcile_runs_missing_diagnostic_solos_once_without_repeating_grades(tmp_path, monkeypatch, arm):
    pytest.importorskip('torch')
    from . import replay_bank as rb, exact_grade as xg
    from .bank_safety import reconcile, confirmations
    original = xg._seal(dict(verification_id='original', arm=arm, admitted=False,
        first_seed_advantage_veh_s=100.,
        checks=dict(replication=True, no_later_harm=True, safety=False),
        safety_deltas={k:dict(collisions=1, teleports=0, hard_brake=3,
            ttc_below_1p5_events=-20) for k in ('replication', 'extended')}), 'confirmation_sha256')
    old_path = tmp_path/'replay_bank'/arm/'confirmations/original.json'
    xg._write_new(old_path, original)
    calls = []
    def solos(*args):
        calls.append(args)
        return dict(cooperative=True, complete=True, cooperation_gain_veh_s=10.)
    monkeypatch.setattr(rb, '_cooperation', solos)
    def forbidden(*args, **kwargs):
        raise AssertionError('saved replication/extended grades must not rerun')
    monkeypatch.setattr(rb, 'confirm', forbidden)
    snapshot = dict(snapshot={}, first_seed_receipt=dict(chosen_plan={}, grade_veh_s=100.))
    first = reconcile(tmp_path, arm, original, snapshot, forbidden, 'bank_verification_ext')
    second = reconcile(tmp_path, arm, original, snapshot, forbidden, 'bank_verification_ext')
    assert first == second and len(calls) == 1
    assert first['admitted'] is False  # collision hard constraint still rejects
    assert first['synergy_diagnostic_required'] is True and first['cooperation']['cooperative'] is True
    assert confirmations(tmp_path, arm)['original'] == first
    assert original == __import__('json').loads(old_path.read_text())
