"""Prospective checkpoint rule; explicit support receipt, no outcome-driven fallback."""
from collections import defaultdict
import math
import statistics

CANDIDATES = ('87ed02e0', 'b8797242')


def select_model(rows, support):
    """Rows are diagnosis original-component longest-history inferred TF60 exports.

    Source adapter attestation is separate from historical endpoint accuracy.
    IDs must match across candidates; incomplete/mixed panels fail closed.
    """
    eligible = [c for c in CANDIDATES if support.get(c, {}).get('team_source_entries') is True]
    if not eligible:
        raise ValueError('no candidate supports team+source+entries')
    by_model = {c:{} for c in eligible}
    for row in rows:
        c = row['checkpoint']
        if c not in eligible:
            continue
        if not (row['panel'] == 'original' and row['components'] == 'original' and
                row['history'] == 'longest_available' and row['settings'] == 'inferred' and
                row['teacher_forced'] is True and row['clock_s'] == 60):
            continue
        key = (row['cohort'], row['world'], row['start'])
        if key in by_model[c] or not math.isfinite(row['hdv_abs_endpoint_error_m']):
            raise ValueError('duplicate/nonfinite diagnosis row')
        by_model[c][key] = row['hdv_abs_endpoint_error_m']
    identities = set(by_model[eligible[0]])
    if {k[0] for k in identities} != {'pi', 'wall'}:
        raise ValueError('both diagnosis cohorts required')
    scores = {}
    for c, values in by_model.items():
        if set(values) != identities:
            raise ValueError('candidates have different diagnosis starts')
        by_world = defaultdict(list)
        for (_, world, _), value in values.items():
            by_world[world].append(value)
        scores[c] = statistics.mean(statistics.mean(v) for v in by_world.values())
    winner = min(eligible, key=lambda c:(scores[c], CANDIDATES.index(c)))
    return dict(checkpoint=winner, scores_m=scores, starts=len(identities),
                rule='equal-world TF60 common-HDV endpoint, original panel/components')
