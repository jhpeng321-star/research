"""Cross-branch contracts exercised by the clean1 production launch."""
import pytest

from . import practice_weighting as pw, rerun_cpus as rc
from . import track2_reward as reward


def test_structured_practice_registration_is_explicit_track2_only():
    block = dict(rc.TRACK_SLOTS['practice_weighting'])
    cfg = dict(track2=True, practice_starts=True, practice_weighting=block)
    assert pw.effective_fraction(cfg) == .5
    assert pw.options(cfg) == dict(practice_weighting=True, practice_weighted_fraction=.5)
    with pytest.raises(ValueError, match='bool'):
        pw.options(dict(cfg, track2=False))
    with pytest.raises(ValueError, match='registration'):
        pw.options(dict(cfg, practice_weighting=dict(block, fraction=.75)))


def test_p1_p2_post_action_envelope_and_json_packet_roundtrip():
    import json
    from dataclasses import asdict
    from .merge_features import SensorReport, ObservedVehicle
    own = SensorReport(301., ObservedVehicle('c1', 0., 1, 10., 5.), (0, 1, 2),
        (ObservedVehicle('h1', 20., 1, 5., 5.),))
    cfg = dict(track2_reward=dict(weight=1., provenance='registered-test'))
    sample = dict(context=dict(members=['c1']))
    episode = dict(id='e1', arm='B', steps=[dict(time_s=300, reward=-999., sample=sample,
        reward_observations={'c1':dict(own=asdict(own), partner_reports={}, partner_ids=[])})])
    envelope = reward.reward_update(cfg, 'B', 1, json.loads(json.dumps([episode])))
    assert envelope['schema'] == 'coopmwm-track2-reward-v1'
    row, = envelope['rows']
    assert row['episode_id'] == 'e1' and row['time_s'] == 300
    assert row['source'] == 'sumo'
    assert row['own_reward'] == pytest.approx(-(1-10/31.29))
    assert row['neighbour_rewards'] == {'h1':pytest.approx(-(1-5/31.29))}
    assert row['reward'] == pytest.approx(row['own_reward']+row['neighbour_rewards']['h1'])
    assert envelope['receipt']['global_reward_used'] is False
    assert episode['steps'][0]['reward'] == -999.
    del episode['steps'][0]['reward_observations']
    with pytest.raises(ValueError, match='terminal post-action'):
        reward.reward_update(cfg, 'B', 1, [episode])
