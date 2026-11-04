"""T2C consumed P2 boundary. No global/team scalar or latent data enters P2.

P2 owns track2_reward; its qualified adapter implements realised_rewards with
local before/after observable envelopes and returns one realised reward per
previously observable vehicle. T2C owns neighbourhood projection and formula.
"""
from copy import deepcopy
import math


def team_reward(component, observation, next_observation, cfg):
    ego = observation['own_sensing']['ego_id']
    vehicles = observation['own_sensing']['vehicles']
    ids = {r['vid'] for r in vehicles}
    if ego not in ids:
        raise ValueError('REFUSED: realised reward requires an observable acting CAV')
    # Only the earlier causal sensing envelope defines the reward neighbourhood.
    # The next local observation cannot smuggle newly visible/global rewards in.
    before = deepcopy(observation)
    after = deepcopy(next_observation)
    after['own_sensing']['vehicles'] = [r for r in after['own_sensing']['vehicles'] if r['vid'] in ids]
    after['cav_ids'] = [v for v in after['cav_ids'] if v in ids]
    after['bounds'] = {v: b for v, b in after['bounds'].items() if v in ids}
    after['v2v_reports'] = [r for r in after['v2v_reports'] if r.get('vid') in ids
                          and r['time_s'] <= after['time_s']-1]
    rewards = component.realised_rewards(before, after, elapsed_s=after['time_s']-before['time_s'])
    if (set(rewards) != ids or any(type(v) not in (int, float) or not math.isfinite(v)
                                  for v in rewards.values())):
        raise ValueError('REFUSED: P2 realised rewards must match the local observable neighbourhood')
    slot = cfg['track2_reward']
    if not slot['provenance'] or not math.isfinite(slot['weight']):
        raise ValueError('REFUSED: registered reward weight/provenance required')
    neighbours = sorted(ids-{ego})
    mean = sum(rewards[v] for v in neighbours)/len(neighbours) if neighbours else 0.
    return dict(ego_id=ego, own_reward=rewards[ego], neighbours=neighbours,
                neighbour_mean=mean, team_reward=rewards[ego]+slot['weight']*mean,
                weight=slot['weight'], provenance=slot['provenance'])
