from tools.rblane.validate import match_events, compress_proposals


def event(t, vid=1, source=1, dest=0):
    return [vid, source, dest, t, 10 * t]


def test_window_matching_is_one_to_one_and_max_cardinality():
    # Nearest-first with input-order ties can match real0->pred1, losing real2.
    real = [event(0), event(2)]
    pred = [event(1), event(-1)]
    scores = match_events(real, pred, 1)
    assert scores['matched'] == 2
    assert scores['recall'] == scores['precision'] == 1
    assert scores['timing_abs_sum_s'] == 2
    assert scores['position_abs_sum_m'] == 20


def test_matching_requires_identity_and_direction_and_does_not_reuse():
    real = [event(1), event(1.5)]
    pred = [event(1), event(1.5, vid=2), event(1.5, source=0, dest=1)]
    assert match_events(real, pred, 2)['matched'] == 1
    assert match_events(real, [event(1), event(1.5)], 0)['matched'] == 2


def test_run_compression_retains_separate_positive_runs():
    events = [event(.5), event(1), event(2), event(2.5)]
    assert [e[3] for e in compress_proposals(events, .5)] == [.5, 2]
