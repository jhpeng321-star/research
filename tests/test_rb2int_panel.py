"""Guard outcome-blind registration, split separation and singleton disclosure."""
import copy

import pytest

from tools.rb2int.audit import ROOT, WORK, checked, load, select_panel


def designs():
    return (load(WORK/'c4-wmdata/tools/wmdata/design.json'),
            load(WORK/'c4-wmdata2/tools/wmdata2/design.json'))


def test_registered_records_cannot_silently_make_fit_worlds_evaluation():
    first, second = designs()
    # Outcomes are deliberately absent: selection can only use metadata.
    panel = select_panel(first, second)
    selected = set(panel['final_comparison_worlds'])
    assert selected.isdisjoint(range(969400, 969432))
    assert set(range(969432, 969440)) <= selected
    assert len(selected) == 58
    assert set(panel['replicated_primary_worlds']) | set(
        panel['descriptive_singleton_worlds']) == selected
    assert panel['descriptive_singleton_worlds'] == [969591, 969593, 969594, 969596]
    for cell in panel['cells']:
        included = set(cell['seeds']) & set(panel['replicated_primary_worlds'])
        assert len(included) == 0 or len(included) >= 2


@pytest.mark.parametrize('defect', ['arm', 'clock', 'missing_world', 'duplicate_world'])
def test_partial_or_changed_recorded_contract_is_rejected(defect):
    first, second = copy.deepcopy(designs())
    if defect == 'arm':
        second['arms'][0] = 'generic_random'
    elif defect == 'clock':
        second['control_s'] = 300
    elif defect == 'missing_world':
        second['worlds'].pop()
    else:
        first['worlds'].append(copy.deepcopy(first['worlds'][-1]))
    with pytest.raises(ValueError):
        select_panel(first, second)


def test_forbidden_path_guard_applies_before_io():
    with pytest.raises(ValueError):
        checked(ROOT/'sealed/nonexistent.json')
