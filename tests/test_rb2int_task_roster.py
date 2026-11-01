import pytest

from tools.rbeval.run_wm import task_selection


def test_development_factor_screen_can_select_explicit_single_future():
    assert task_selection(range(969420, 969440), [969420], [0]) == ((969420,), (0,))


@pytest.mark.parametrize('worlds,futures', [([969550], [0]), ([969550], None), (None, [0]), (None, [1]*8)])
def test_final_panel_cannot_shrink_roster_or_future_ensemble(worlds, futures):
    with pytest.raises(ValueError):
        task_selection([969550, 969551], worlds, futures, final=True)


def test_full_registered_new_world_roster_is_distinct_from_legacy_dev_panel():
    worlds, futures = task_selection(range(969550, 969600), final=True)
    assert len(worlds) == 50 and futures == tuple(range(8))
    with pytest.raises(ValueError):
        task_selection(range(969420, 969440), [969550])
