from copy import deepcopy

import pytest

from tools.coopmwm.rerun_tracks_monitor import monitor_row, write_monitor, read_monitor, METRICS


def metrics():
    return dict(group_credit_wm=1., group_credit_checked=.9, dense_ce=.6,
        dense_accuracy=.5, bank_move_ce=.5, layer_kl=0., merge_missing_share=.25, reward=2.,
        decisions=10, credit_rows=10, credit_checked_rows=1, dense_labels=10,
        dense_eligible_labels=10, dense_checked_labels=1, dense_filtered_share=0., dense_checked_accuracy=1.,
        augmented_examples=2, weighted_starts=1, families=['none', 'gap_open'],
        trigger_counts=dict(none=10, gap_open=0),
        safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0), safety_stop=False)


def test_receipts_history_and_registered_diagnose_rules(tmp_path):
    for u in range(1, 4):
        row = write_monitor(tmp_path, 'B', u, metrics())
        assert row['action'] == 'DIAGNOSE'  # no non-safety pauses
    assert set(row['diagnose']) == {'trigger_saturated', 'layer_kl_near_zero',
        'group_credit_not_rising_3_updates', 'dense_accuracy_flat_3_updates'}
    assert row['trigger_rates'] == dict(none=1., gap_open=0.)
    assert read_monitor(tmp_path)['updates'] == dict(A=0, B=3, R=0)
    assert read_monitor(tmp_path)['latest']['B'] == row
    with pytest.raises(ValueError, match='contiguous'):
        write_monitor(tmp_path, 'B', 3, metrics())
    with pytest.raises(ValueError, match='contiguous'):
        write_monitor(tmp_path, 'B', 5, metrics())
    unsafe = metrics() | dict(safety_stop=True)
    assert write_monitor(tmp_path, 'B', 4, unsafe)['action'] == 'PAUSE_SAFETY'


@pytest.mark.parametrize('field', METRICS+('decisions', 'safety', 'trigger_counts'))
def test_missing_metrics_refused(field):
    m = metrics()
    del m[field]
    with pytest.raises(ValueError, match='missing monitor'):
        monitor_row('B', 1, m)


def test_improving_history_clears_trends_and_bad_history_is_refused():
    history = []
    for u in range(1, 4):
        m = metrics() | dict(group_credit_wm=float(u), dense_accuracy=.1*u, layer_kl=.01,
                            trigger_counts=dict(none=5, gap_open=5))
        row = monitor_row('R', u, m, history)
        assert not row['diagnose'] and row['action'] == 'CONTINUE'
        history.append(row)
    altered = deepcopy(history)
    altered[1]['metrics']['reward'] = 999
    with pytest.raises(ValueError, match='mismatch'):
        monitor_row('R', 4, metrics(), altered)
    with pytest.raises(ValueError, match='invalid monitor metric'):
        monitor_row('B', 1, metrics() | dict(dense_ce=float('nan')))
