import numpy as np

from tools.rb2int.compare import MODELS, features, rate_transform, switching
from tools.rb2int.uncertainty import PairedWorldBootstrap
from tools.rbeval.core import ARMS, HS


def test_group_rate_distance_is_not_mean_of_individual_outcome_distances():
    # Truth half events; a calibrated .5 forecast beats an always-event forecast.
    rates = np.zeros((2, 5, 3))
    rates[:, 0] = 1
    rates[:, 1] = 1
    rates[:, 2] = .5
    rates[:, 3] = .5
    rates[0, 4] = 0
    rates[1, 4] = 1
    boot = PairedWorldBootstrap([dict(seed=s, demand=2400, penetration=.2) for s in (1, 2)], B=500)
    row = boot.bands(rates.reshape(2, 15), transform=rate_transform)[0]
    assert row['mean'] == -.5


def test_forecast_score_keeps_all_arms_in_single_world_and_six_contrasts():
    real, predictions = [], {label: {} for label in MODELS}
    for seed in (1, 2):
        for arm in ARMS:
            target = dict(seed=seed, arm=arm, clocks={str(h): dict(team_delay_vs=0.) for h in HS},
                bins=dict(upstream=dict(events={str(h): dict(event=False) for h in HS})))
            real.append(target)
            for m, label in enumerate(MODELS):
                predictions[label][seed, arm] = [dict(clocks={str(h): dict(team_delay_vs=float(4-m)) for h in HS},
                    bins=dict(upstream=dict(events={str(h): dict(event=False) for h in HS}))) for k in range(8)]
    scored = features(predictions, real, [1, 2])
    assert scored['delay_contrasts'].shape == (2, 18)
    assert np.array_equal(scored['delay_contrasts'][:, :3], np.full((2, 3), -2.))
    assert scored['rate_futures'].shape == (2, 8, 5, 3)


def test_switch_rule_distinguishes_point_failure_and_missing_qualification():
    delay = [dict(horizon_s=h, mean=-10., high=-1.) for h in HS]
    rates = [dict(horizon_s=h, mean=-.1, high=-.01) for h in HS]
    missing = switching(delay, rates, None, complete=True)
    assert missing['point_status'] == 'NOT_ESTABLISHED'
    assert missing['switching_eligibility'] == 'NOT_ESTABLISHED'
    endpoint = dict(mean=-2., high=-.1, all_worlds_supported=True)
    assert switching(delay, rates, endpoint, complete=True)['switching_eligibility'] == 'ELIGIBLE'
    rates[1]['mean'] = .1
    assert switching(delay, rates, None, complete=True)['point_status'] == 'FAIL'
    rates[1]['mean'] = -.1
    assert switching(delay, rates, endpoint, complete=False)['switching_eligibility'] == 'NOT_ESTABLISHED'
