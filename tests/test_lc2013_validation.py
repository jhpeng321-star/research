from tools.rblane.validate import match_events, compress_proposals, aggregate, MOTIVATIONS


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


def test_motivation_scores_keep_false_positives_and_all_event_denominator():
    real = [event(1), event(2)]
    predicted = [event(1), event(5), event(6)]
    score = match_events(real, predicted, 0)
    base = dict(exact=score, window_2s=score, run_window_2s=score,
                positive_runs=3,
                by_transition={name: score for name in ('merge', 'overtake', 'return')})
    lc = dict(base, by_motivation={name: dict(exact=score, run_window_2s=score)
                                   for name in MOTIVATIONS.values()})
    result = aggregate([dict(scores=dict(lc2013=lc, mobil=base))])
    reason = result['lc2013']['by_motivation']['keep_right']['exact']
    assert reason['precision'] == 1 / 3
    assert reason['recall_contribution'] == 1 / 2
    assert reason['predicted'] == 3
