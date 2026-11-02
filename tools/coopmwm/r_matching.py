"""Outcome-blind best-effort random-arm matching (MAIN14:47)."""
from collections import Counter


def nearest(groups, tickets, used, traffic):
    """Prefer exact regime/size, then nearest size within one; stable ticket order.

    Durations retain the ticket's registered bucket exactly. Regime fallback is
    allowed only after same-regime eligible groups have been exhausted. No WM
    score, realized grade, or random draw participates in matching.
    """
    options = [(g, t) for g in groups for t in tickets
               if t['id'] not in used and abs(g['size'] - t['group_size']) <= 1]
    if not options:
        return None, None
    return min(options, key=lambda pair: (
        pair[1]['regime'] != traffic,
        abs(pair[0]['size'] - pair[1]['group_size']),
        pair[1]['id'], tuple(pair[0]['members'])))


def report(tickets, durations):
    """Keep every stratum, including missing/extra plans; never a stop gate."""
    def counts(rows):
        return Counter((r['regime'], r['group_size'], r['duration_s']) for r in rows)
    expected, actual = counts(tickets), counts(durations)
    strata = [dict(regime=k[0], group_size=k[1], duration_s=k[2],
                   expected=expected[k], actual=actual[k],
                   shortfall=max(0, expected[k]-actual[k]),
                   excess=max(0, actual[k]-expected[k]))
              for k in sorted(expected.keys() | actual.keys())]
    return dict(rule='MAIN1447 best-effort: regime then size distance<=1; ticket duration unchanged',
                expected_count=len(tickets), executed_count=len(durations),
                count_shortfall=max(0, len(tickets)-len(durations)),
                exact_match=expected == actual, strata=strata)
