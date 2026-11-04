"""Registered COOPERATION (team) reward option for the V2V PPO runtime (default OFF).

Option `coop_reward` (COOPREWARD_DESIGN.md, c4/coop-reward): the augmented per-CAV reward
``r_i' = r_i + w * mean_{j in N_i} r_j`` with ``N_i`` = the vehicles (CAV and HDV alike, no
driver-type oracle) inside the SAME observable neighbourhood the CAV already senses/receives
via V2V — the 100 m observable frame range (``v2v_obs.V2V_RANGE_M`` == the sensing contract)
— plus NO separate background term (per-vehicle background outcomes are not in the training
receipts, and the own reward is already the whole-world source-inclusive team J).

The runtime trains ONE joint policy with one per-step scalar reward and one shared
observable-state critic (COOPREWARD_DESIGN.md §2–3), so the per-CAV formula collapses to the
joint step reward ``r'_t = r_t + w * mean_{i in CAVs} mean_{j in N_i} outcome_j`` — equal
weighting of every present CAV's augmented reward. The outcome of neighbour j at the charged
second is the observable speed deficit of its post-step frame row,
``max(0, 1 - v_j/SPEED_LIMIT_MPS)`` in vehicle-seconds — the exact deficit form of the
runtime's road-delay term (``traffic_shepherd.c4.env.SPEED_LIMIT_MPS``). The charged second's
final step has no post-step frame in the receipts and falls back to its own frame (counted).

Config (registered; produced by v2v_runner.prepare_intent, validated by v2v_live.hooks and
v2v_launch3.bind_runtime, fail-closed here at use time):
  coop_reward            truthy = enabled; absent/False = OFF (byte-identical run).
  coop_reward_weight     w, positive finite float, REQUIRED when enabled (no default tuning).
  coop_reward_provenance non-empty string naming the registering authority, REQUIRED.
  coop_reward_arms       optional; default ('B','R'); B and R TOGETHER, A only by explicit
                         inclusion (one key set serves every arm; B/R pinned identical).

When OFF (or the arm is not enabled) `training_episodes` returns the SAME episode list object
and a None report — the PPO update inputs, the GAE and the receipts are byte-identical.
"""
import math

from .v2v_obs import V2V_RANGE_M

ARMS = ('A', 'B', 'R')
NEIGHBOURHOOD_RANGE_M = V2V_RANGE_M        # same observable neighbourhood contract as V2V
SPEED_LIMIT_MPS = 31.29                    # runtime road-delay deficit constant (unit test pins it)


def settings(cfg):
    """The registered option of a runtime config: None when OFF, else {arms, weight, provenance}.

    Refuses a malformed registration BEFORE any training runs: missing/invalid weight (no
    default tuning), missing/empty provenance, and arms that do not keep B and R identical
    (A only by explicit inclusion)."""
    if not cfg.get('coop_reward'):
        return None
    weight = cfg.get('coop_reward_weight')
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('coop_reward requires the registered positive finite '
                         'coop_reward_weight (no default): '+repr(weight))
    provenance = cfg.get('coop_reward_provenance')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('coop_reward requires a non-empty coop_reward_provenance: '
                         +repr(provenance))
    arms = cfg.get('coop_reward_arms', ARMS[1:])
    if (not isinstance(arms, (list, tuple)) or not arms or len(arms) != len(set(arms))
            or not set(arms) <= set(ARMS) or not set(ARMS[1:]) <= set(arms)):
        raise ValueError('coop_reward_arms must keep B and R identical (A optional, explicit): '
                         +repr(arms))
    return dict(arms=tuple(arms), weight=float(weight), provenance=provenance,
                range_m=float(NEIGHBOURHOOD_RANGE_M), speed_limit_mps=float(SPEED_LIMIT_MPS))


def enabled(cfg, arm):
    """Whether the registered option is ON for one arm (OFF and unknown arms are False)."""
    have = settings(cfg)
    return have is not None and arm in have['arms']


def neighbourhood_term(rows, range_m=NEIGHBOURHOOD_RANGE_M, speed_limit_mps=SPEED_LIMIT_MPS):
    """mean_{i in CAVs} mean_{j in N_i} deficit_j over ONE observable frame of receipt rows.

    N_i = the OTHER rows within ``range_m`` longitudinal distance of CAV i — CAV and HDV
    alike (``is_cav`` selects only the egos; it is an observable frame field, never an oracle
    driver type). Neighbour outcome = the observable speed deficit in vehicle-seconds.
    Returns ``(term, egos, empty_neighbourhood_egos)``; no observable CAV -> (0., 0, 0).
    """
    if not isinstance(rows, (list, tuple)):
        raise ValueError('observable frame rows required for the coop neighbourhood')
    for row in rows:
        for key in ('vid', 's_m', 'v_mps', 'is_cav'):
            if key not in row:
                raise ValueError('observable frame row lacks '+repr(key)+': the coop term uses '
                                 'only receipt-observable vehicles')
    egos = [row for row in rows if row['is_cav']]
    if not egos:
        return 0., 0, 0
    total = 0.
    empty = 0
    for own in egos:
        s_own = float(own['s_m'])
        neighbours = [row for row in rows if row['vid'] != own['vid']
                      and abs(float(row['s_m'])-s_own) <= float(range_m)]
        if not neighbours:
            empty += 1
            continue
        total += sum(max(0., 1.-float(row['v_mps'])/speed_limit_mps) for row in neighbours)/len(neighbours)
    return total/len(egos), len(egos), empty


def training_episodes(cfg, arm, episodes):
    """The PPO-update episodes of one arm plus the per-update own/team component report.

    OFF (or arm not enabled) -> the SAME ``episodes`` object and None (byte-identical path).
    ON -> NEW episode dicts whose step rewards are ``r_t + w * team_t`` (per-step shallow
    copies; samples/contexts are shared untouched) and a detached report of the own vs team
    components. Nothing here mutates the receipts; only the update call receives the copy.
    """
    have = settings(cfg)
    if have is None or arm not in have['arms']:
        return episodes, None
    weight, range_m, limit = have['weight'], have['range_m'], have['speed_limit_mps']
    own_sum = team_sum = raw_sum = 0.
    steps = egos_total = empty_total = own_frame_steps = 0
    transformed = []
    for episode in episodes:
        episode_steps = episode['steps']
        new_steps = []
        for index, step in enumerate(episode_steps):
            if 'reward' not in step:
                raise ValueError('coop_reward transforms receipt steps with their own reward')
            if index+1 < len(episode_steps):
                frame = episode_steps[index+1]['rows']   # post-step observable frame
            else:
                frame = step['rows']                     # charged second's last step: own frame
                own_frame_steps += 1
            term, egos, empty = neighbourhood_term(frame, range_m, limit)
            addition = weight*term
            own_sum += float(step['reward'])
            raw_sum += term
            team_sum += addition
            steps += 1
            egos_total += egos
            empty_total += empty
            new_steps.append(dict(step, reward=float(step['reward'])+addition))
        transformed.append(dict(episode, steps=new_steps))
    report = dict(arms=list(have['arms']), weight=weight, provenance=have['provenance'],
                  steps=steps, egos=egos_total, empty_neighbourhood_egos=empty_total,
                  own_frame_team_steps=own_frame_steps,
                  own_reward_sum_veh_s=own_sum, own_reward_mean_veh_s=own_sum/max(steps, 1),
                  team_addition_sum_veh_s=team_sum, team_addition_mean_veh_s=team_sum/max(steps, 1),
                  team_neighbourhood_mean_veh_s=raw_sum/max(steps, 1),
                  neighbourhood_range_m=range_m, speed_limit_mps=limit,
                  background_term=None,
                  background_rule='own reward is the whole-world source-inclusive team J; no '
                                  'per-vehicle background decomposition exists in the receipts')
    return transformed, report
