"""P2, MAIN 02:30: realised, observable-only cooperation-layer rewards.

For each acting CAV, r_i = -dt * max(0, 1 - observed_speed_i / speed_limit)
and r_i' = r_i + registered_weight * mean(r_j for j in N_i). N_i excludes
self and includes sensed vehicles of either automation status, plus vehicles
directly sensed by selected, in-range V2V publishers at t-1. This is a measured
speed-deficit outcome, NOT an estimate of the simulator's substep-integrated
global delay or its unobservable source queue. Slower neighbours lower reward.

``neighbourhood_reward`` uses the SAME atomic packets, validity, range, partner
cap, latency and duplicate precedence as merge_features. It never accepts a
world state, infers a driver type, scans a report board, or calls a simulator/WM.

``training_episodes`` adapts real episode receipts into NEW cooperation-only
training episodes. It retains the original global reward as ``base_reward``;
the original episodes must still be used for evaluation and base-policy paths.
Joint layer reward is the equal-CAV mean of r_i'. Standard observable ``rows``
support direct sensing only: each ego is clipped to its own sensing range.
For V2V, the backend can instead attach per-CAV ``reward_observations`` packets
to the charged step (see function docstring). Terminal post-action sensing is
REQUIRED: never substitute a pre-action frame or a global reward for an outcome.

``layer_records`` produces the existing CooperationLayer.update record shape.
Its historical ``group_advantage`` key carries real-outcome GAE here, never a
WM counterfactual. The frozen MAPPO value is only a detached baseline.
All settings are shared by B/R. A and default-off return inputs by identity.
"""
from __future__ import annotations

import math
from numbers import Real

from . import merge_features as mf

# Public road/control constants; no simulator import (including at worker startup).
SPEED_LIMIT_MPS = 31.29
CONTROL_DT_S = mf.CONTROL_DT_S
ARMS = ("B", "R")
REWARD_SOURCE = "SUMO observed speed deficit; local sensing + timestamped V2V only"


def _finite(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{name} must be {'positive ' if positive else ''}finite numeric")
    return float(value)


def settings(cfg):
    """Validate explicit registration; no chosen/tuned default weight.

    Keys: track2_reward (bool, default False), track2_reward_weight (positive
    finite), track2_reward_provenance (nonempty authority/registration reference).
    Optional track2_reward_arms must contain exactly B and R, without duplicates.
    Legacy coop_reward and WM group-credit/teacher switches are not consumed.
    """
    flag = cfg.get("track2_reward", False)
    if not isinstance(flag, bool):
        raise ValueError("track2_reward must be boolean")
    if not flag:
        return None
    weight = _finite(cfg.get("track2_reward_weight"), "track2_reward_weight", positive=True)
    provenance = cfg.get("track2_reward_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("track2_reward_provenance must name the weight registration")
    arms = cfg.get("track2_reward_arms", ARMS)
    if (not isinstance(arms, (list, tuple)) or len(arms) != 2
            or any(not isinstance(a, str) for a in arms) or set(arms) != set(ARMS)):
        raise ValueError("track2_reward_arms must be exactly B and R; A is unaffected")
    return dict(weight=weight, provenance=provenance, arms=list(ARMS),
                reward_source=REWARD_SOURCE, speed_limit_mps=SPEED_LIMIT_MPS,
                control_dt_s=CONTROL_DT_S)


def enabled(cfg, arm):
    """A is unconditionally unaffected, including by B/R registration errors."""
    return arm in ARMS and settings(cfg) is not None


def _outcome(row):
    return -CONTROL_DT_S * max(0.0, 1.0 - row.v_mps / SPEED_LIMIT_MPS)


def neighbourhood_reward(own, cav_id, partner_reports=None, *, now_s,
                         partner_ids=(), weight):
    """Return per-CAV own/team components and observable evidence counts.

    ``own`` and partner entries are merge_features.SensorReport instances.
    ``now_s`` is the outcome observation time (end of the charged control tick).
    Current own sensing is mandatory; missing/invalid own packets raise rather
    than fabricate zero reward. Invalid remote packets are missing evidence and
    are counted. Remote outcomes retain their exact t-1 measurement time; they
    are not extrapolated to now. Own sensing wins, then nearest publisher/ID.
    No vehicle type field is needed: all directly observed neighbours count.
    """
    now_s = _finite(now_s, "now_s")
    weight = _finite(weight, "weight", positive=True)
    if not mf._valid_report(own, cav_id, now_s):
        raise ValueError("current own SUMO sensing required for cooperation reward")
    selected = sorted(set(partner_ids) - {cav_id})
    partners = []
    for vid in selected:
        # Indexed access only: never iterate the full V2V board.
        report = (partner_reports if partner_reports is not None else {}).get(vid)
        if not mf._valid_report(report, vid, now_s - CONTROL_DT_S):
            continue
        distance = abs(report.source.s_m - own.source.s_m)
        if distance <= mf.V2V_RANGE_M:
            partners.append((distance, vid, report))
    partners.sort(key=lambda item: (item[0], item[1]))
    accepted = partners[:mf.V2V_K]
    visible = {}
    for report in [own] + [item[2] for item in accepted]:
        for row in mf._local_rows(report):
            visible.setdefault(row.vehicle_id, (row, report.timestamp_s))
    neighbours = sorted(vid for vid in visible if vid != cav_id)
    own_reward = _outcome(own.source)
    team = (math.fsum(_outcome(visible[vid][0]) for vid in neighbours) / len(neighbours)
            if neighbours else 0.0)
    addition = weight * team
    reward = own_reward + addition
    if not math.isfinite(reward):
        raise ValueError("nonfinite augmented cooperation reward")
    return dict(own_reward=own_reward, neighbourhood_reward=team,
                team_addition=addition, reward=reward,
                neighbour_rewards={vid: _outcome(visible[vid][0]) for vid in neighbours},
                neighbour_ids=neighbours,
                neighbour_timestamps_s={vid: visible[vid][1] for vid in neighbours},
                neighbours=len(neighbours), empty_neighbourhood=not neighbours,
                partner_reports=len(accepted),
                unavailable_partner_reports=len(selected) - len(accepted))


def _sensor_packet(rows, cav_id, now_s):
    """Clip an observable receipt frame BEFORE reading neighbour outcomes."""
    table = {}
    for row in rows:
        vid = row["vid"]
        if not isinstance(vid, str) or not vid or vid in table:
            raise ValueError("observable rows require unique nonempty vid values")
        table[vid] = row
    if cav_id not in table:
        raise ValueError(f"missing post-action observable ego {cav_id}")
    ego_s = _finite(table[cav_id]["s_m"], "ego s_m")

    def vehicle(row):
        return mf.ObservedVehicle(row["vid"], row["s_m"], row["lane"],
                                  row["v_mps"], row["length_m"])

    source = vehicle(table[cav_id])
    local = []
    for vid, row in table.items():
        if vid != cav_id and abs(_finite(row["s_m"], "observed s_m") - ego_s) <= mf.SENSOR_RANGE_M:
            local.append(vehicle(row))
    lanes = tuple(sorted({source.lane, *(row.lane for row in local)}))
    return mf.SensorReport(now_s, source, lanes, tuple(local))


def _decode_packet(value):
    if isinstance(value, mf.SensorReport):
        return value
    if not isinstance(value, dict) or set(value) != {'timestamp_s', 'source', 'observed_lanes', 'vehicles'}:
        raise ValueError('invalid serialized SUMO sensor packet')
    return mf.SensorReport(value['timestamp_s'], mf.ObservedVehicle(**value['source']),
        tuple(value['observed_lanes']), tuple(mf.ObservedVehicle(**v) for v in value['vehicles']))


def training_episodes(cfg, arm, episodes):
    """Return (cooperation-only copies, receipt), or (original, None) when off.

    Each step supplies time_s and sample.context.members (the acting CAV IDs).
    Outcomes use either:
      * step['reward_observations'][cav_id] = {own: SensorReport,
        partner_reports: {publisher: SensorReport}, partner_ids: [...]}, at t+1;
      * step['post_rows'] + step['post_time_s'] == t+1; or
      * the immediately following step's rows/time_s == t+1.
    No terminal fallback: the backend must record terminal post_rows/time_s or
    packets, including the acting ego's outcome. Missing evidence fails closed.
    Empty-CAV steps yield zero layer reward (no actor evidence) and are counted.
    Episode lengths, samples, base rewards and original receipt objects survive.
    """
    if not enabled(cfg, arm):
        return episodes, None
    have = settings(cfg)
    transformed = []
    totals = dict(steps=0, cav_outcomes=0, empty_cav_steps=0,
                  terminal_exits=0,
                  empty_neighbourhoods=0, neighbour_outcomes=0,
                  partner_reports=0, unavailable_partner_reports=0,
                  own_reward_sum=0.0, neighbourhood_reward_sum=0.0,
                  team_addition_sum=0.0, reward_sum=0.0)
    for episode in episodes:
        new_steps = []
        steps = episode["steps"]
        for index, step in enumerate(steps):
            if "track2_reward" in step or "base_reward" in step:
                raise ValueError("track2 reward already applied")
            now = _finite(step["time_s"], "time_s") + CONTROL_DT_S
            members = step["sample"]["context"]["members"]
            if (any(not isinstance(vid, str) or not vid for vid in members)
                    or len(members) != len(set(members))):
                raise ValueError("acting CAV IDs must be unique nonempty strings")
            packets = step.get("reward_observations")
            rows = None
            if members and packets is None:
                if "post_rows" in step:
                    timestamp, rows = step.get("post_time_s"), step["post_rows"]
                elif index + 1 < len(steps):
                    timestamp, rows = steps[index + 1]["time_s"], steps[index + 1]["rows"]
                else:
                    raise ValueError("terminal post-action sensing required (post_rows/post_time_s)")
                if _finite(timestamp, "post_time_s") != now:
                    raise ValueError("post-action sensing must be exactly one control tick later")
            components = {}
            for vid in members:
                if packets is not None:
                    packet = packets[vid]
                    if 'terminal_exit' in packet:
                        expected = dict(cav_id=vid, time_s=now,
                            before_time_s=step['time_s'], present_before=True, present_after=False,
                            source='sumo_observation_transition')
                        if set(packet) != {'terminal_exit'} or packet['terminal_exit'] != expected:
                            raise ValueError('invalid measured terminal SUMO observation transition')
                        components[vid] = dict(own_reward=0., neighbourhood_reward=0.,
                            team_addition=0., reward=0., neighbour_ids=[], neighbour_rewards={},
                            neighbour_timestamps_s={}, neighbours=0, empty_neighbourhood=True,
                            partner_reports=0, unavailable_partner_reports=0, terminal_exit=True)
                        totals['terminal_exits'] += 1
                        continue
                    own = _decode_packet(packet["own"])
                    reports = {p: _decode_packet(r) for p, r in packet.get("partner_reports", {}).items()}
                    partners = packet.get("partner_ids", ())
                else:
                    own = _sensor_packet(rows, vid, now)
                    reports, partners = None, ()
                components[vid] = neighbourhood_reward(
                    own, vid, reports, now_s=now, partner_ids=partners, weight=have["weight"])
            count = len(components)
            means = {key: math.fsum(c[key] for c in components.values()) / max(count, 1)
                     for key in ("own_reward", "neighbourhood_reward", "team_addition", "reward")}
            totals["steps"] += 1
            totals["cav_outcomes"] += count
            totals["empty_cav_steps"] += not count
            for c in components.values():
                totals["empty_neighbourhoods"] += c["empty_neighbourhood"]
                totals["neighbour_outcomes"] += c["neighbours"]
                for key in ("partner_reports", "unavailable_partner_reports"):
                    totals[key] += c[key]
            for key in means:
                totals[key + "_sum"] += means[key]
            new_steps.append(dict(step, base_reward=step["reward"], reward=means["reward"],
                                  track2_reward=dict(per_cav=components, **means)))
        transformed.append(dict(episode, steps=new_steps))
    receipt = dict(have, **totals, arm=arm, scope="cooperation layer only",
                   aggregation="equal acting-CAV mean; empty neighbourhood = 0",
                   wm_used=False, global_reward_used=False, terminal_fallback_steps=0,
                   outcome_rule="negative measured speed deficit, vehicle-seconds per control tick")
    return transformed, receipt


def reward_update(cfg, arm, u, episodes):
    """P1 envelope backed by this module's real post-action sensor extraction.

    The parent keeps its SHA-bound structured registration. P1 has a narrow
    weight/provenance mapping; convert it only at this explicit integration edge.
    Every recorded control tick is retained, including non-PPO/prefix ticks.
    """
    registration = cfg['track2_reward']
    if not isinstance(registration, dict) or set(registration) != {'weight', 'provenance'}:
        raise ValueError('P1 reward registration must contain weight and provenance')
    native = dict(cfg, track2_reward=True,
                  track2_reward_weight=registration['weight'],
                  track2_reward_provenance=registration['provenance'],
                  track2_reward_arms=['B', 'R'])
    transformed, receipt = training_episodes(native, arm, episodes)
    rows = []
    if arm in ARMS:
        for episode in transformed:
            for step in episode['steps']:
                for vid, outcome in step['track2_reward']['per_cav'].items():
                    rows.append(dict(episode_id=episode['id'], time_s=step['time_s'],
                        cav_id=vid, source='sumo', own_reward=outcome['own_reward'],
                        neighbour_rewards=outcome['neighbour_rewards'], reward=outcome['reward'],
                        weight=registration['weight'], weight_provenance=registration['provenance']))
    return dict(schema='coopmwm-track2-reward-v1', arm=arm, update=u,
                rows=rows, receipt=receipt)


def layer_records(cfg, arm, episodes):
    """Produce (CooperationLayer.update records, reward receipt) from raw episodes.

    GAE uses the existing registered UNIT/GAMMA/LAMBDA and detached rollout
    values; no new discount, horizon or scale is introduced. Executed-plan,
    deterministic, empty-group and non-layer samples never become PPO records.
    Their real rewards remain in the full episode return. Call this OR
    training_episodes on raw receipts, never shape twice.
    """
    shaped, receipt = training_episodes(cfg, arm, episodes)
    if receipt is None:
        return [], None
    from .accel_learner import advantages, own_step

    records = []
    for episode in shaped:
        for step, (advantage, _) in zip(episode["steps"], advantages(episode["steps"])):
            sample = step["sample"]
            if (sample["context"]["members"] and own_step(sample)
                    and not sample.get("deterministic") and "coop_layer" in sample):
                records.append(dict(sample=sample, group_advantage=advantage))
    return records, dict(receipt, layer_records=len(records),
                         advantage_source="real observable outcome GAE; frozen rollout value baseline")
