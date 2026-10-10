"""Accounting regressions for the bounded OE41 offline analysis."""
import pytest
from output.formation_harm_v1.analyze import METRICS, totals, phase_bounds


def life(ready=610, exit_s=630, terminal=633):
    return dict(command_start_s=600,committed_hold_start_s=ready,
                exit_start_s=exit_s,terminal_s=terminal)


def test_disjoint_phase_partition():
    assert phase_bounds(life())==[('assembly',600,610),('hold',610,630),
                                  ('exit',630,633),('after_handback',633,2400)]


def test_no_hold_control_is_not_discarded():
    assert phase_bounds(life(None,601,605))==[('assembly',600,601),('exit',601,605),
                                             ('after_handback',605,2400)]


def test_abrupt_abort_has_zero_exit_duration():
    assert phase_bounds(life(None,None,607))[1]==('exit',607,607)


def test_hold_aborted_without_taper_is_accounted():
    assert phase_bounds(life(610,None,624))[1:3]==[('hold',610,624),('exit',624,624)]


def test_command_interval_boundary_not_counted_twice():
    rows=[dict(start_s=t,end_s=t+1,**{k:t-599 for k in METRICS}) for t in range(600,603)]
    assert totals(rows,600,601)['acceleration_energy']==1
    assert totals(rows,601,603)['acceleration_energy']==5
    assert totals(rows,600,603)['acceleration_energy']==6


def test_missing_metric_interval_rejected():
    with pytest.raises(AssertionError):
        totals([],600,601)


def test_empty_exit_not_missing_exposure():
    assert all(v==0 for v in totals([],605,605).values())
