"""Prospective checkpoint rule; explicit support receipt, no outcome-driven fallback."""
from collections import defaultdict
import math
import statistics

CANDIDATES = ('87ed02e0', 'b8797242')
WM2_SHA = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
BB_SHA = 'afaaa32cb15079014d1a9e08f78b26e74fd341298d1aa0292a24009d9f783742'


def main_rule(diagnosis_sha256):
    """MAIN18:33 prospective amendment; no endpoint or switching outcomes accepted."""
    if len(diagnosis_sha256) != 64 or any(c not in '0123456789abcdef' for c in diagnosis_sha256):
        raise ValueError('diagnosis hash required')
    return dict(rule='MAIN18:33 fixed before switching outcomes',
                diagnosis_sha256=diagnosis_sha256,
                wm_checkpoint_sha256=WM2_SHA, bb_checkpoint_sha256=BB_SHA,
                history_s=60, K=8, horizon_s=60, entrant_learned_mix=True,
                prediction_objective='on_road_fleet_delay',
                evaluation_objective='full_team_delay_fleet_plus_source',
                team_source_entries=False)


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
