"""Outcome-blind arrival-phase admission; never changes the search horizon."""
import math


def contract(artifact, search):
    if artifact['schema'] != 'wm_phys.observed_arrivals.v1':
        raise ValueError('unsupported observable arrival artifact')
    end = float(artifact['observed_phase_end_s'])
    duration = max(float(d) for d in search['durations_s'] if d <= search['max_duration_s'])
    screen = min(600., max(float(search['screen_H_min']), float(search['max_duration_s'])))
    final = min(float(search['final_H_max']), max(float(search['final_H_min']), duration + float(search['release_s'])))
    horizon = max(300., screen, final)  # includes the unchanged policy-centre rollout
    if not math.isfinite(end) or end <= 0 or not math.isfinite(horizon):
        raise ValueError('invalid model arrival phase')
    return dict(observed_phase_end_s=end, required_horizon_s=horizon,
        screen_horizon_s=screen, maximum_final_horizon_s=final,
        max_start_s=end-horizon, search_horizons_unchanged=True)


def feasible(time_s, admission):
    start = float(time_s)
    if not math.isfinite(start) or start < 0:
        raise ValueError('invalid snapshot clock')
    return start + admission['required_horizon_s'] <= admission['observed_phase_end_s']


def exclusion_counts(pending, admission):
    counts = {}
    for e in pending.values():
        if not feasible(e['time_s'], admission):
            bucket = str(100 * (int(e['time_s']) // 100))
            counts[bucket] = counts.get(bucket, 0) + 1
    return counts
