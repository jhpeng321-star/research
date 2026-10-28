from wms.control.auditfix_diagnostics import report_phase_selection
from wms.control.nativeinit import select_update


def test_retained_c1_is_explicit_and_selector_is_unchanged(capsys):
    curve = [{'update': 180, 'passed': True, 'mean_delta': -1.},
             {'update': 190, 'passed': True, 'mean_delta': -.5}]
    chosen = select_update(curve)
    assert chosen == 180
    row = report_phase_selection(chosen, 180, resumed=True)
    assert row['resumed_baseline_selected'] is True
    assert row['selected_new_phase_update'] is False
    assert row['selected_phase_updates'] == 0
    assert 'no newly trained phase checkpoint' in capsys.readouterr().err


def test_new_phase_selection_has_no_baseline_warning(capsys):
    row = report_phase_selection(190, 180, resumed=True)
    assert row['selected_new_phase_update'] is True
    assert row['resumed_baseline_selected'] is False
    assert row['selected_phase_updates'] == 10
    assert capsys.readouterr().err == ''
