"""Dimensionless paired physical return and globally bounded intervention cost."""

from dataclasses import asdict, dataclass
import numpy as np
from idm.damping_credit import TeamIntervalCost, team_advantages


@dataclass(frozen=True)
class RewardContract:
    braking_scale_vehicle_seconds: float = 613.0
    effort_scale_m2_per_s3: float = 41204.235963623185
    service_weight_per_second: float = 0.1
    source_weight_per_second: float = 0.1
    ttc_weight: float = 0.1
    intervention_episode_cap: float = 0.02
    violation_penalty: float = 2.0

    def manifest(self):
        return dict(
            schema="native_selectable_team_reward/2.0",
            **asdict(self),
            units="dimensionless_return",
            normalization="no_advantage_centering_or_reward_clipping",
            native_reference="training_only_identical_requests",
            terminal="veto_positive_physical_return_on_constraint_failure_then_penalize_each_violation",
        )


def reward_series(rollout, native, contract=RewardContract()):
    own, ref = rollout["summary"], native["summary"]
    for key in ("scenario_identity", "seed", "population", "global_control_steps", "team_terminal"):
        if own[key] != ref[key]:
            raise ValueError(f"paired reference mismatch: {key}")
    t, n = own["global_control_steps"], own["population"]
    if not own["team_terminal"] or n <= 0:
        raise ValueError("complete team episode required")
    if len(rollout["intervals"]) != t or len(native["intervals"]) != t:
        raise ValueError("incomplete intervals")
    for data in (rollout, native):
        for row in data["intervals"]:
            TeamIntervalCost(**row)
        for key in ("source_intervals", "ttc_intervals"):
            if (
                np.asarray(data[key]).shape != (t,)
                or not np.isfinite(data[key]).all()
                or min(data[key]) < 0
            ):
                raise ValueError("invalid physical intervals")
        for key, expected in (
            ("source_intervals", data["summary"]["source_wait_vehicle_seconds"]),
            ("ttc_intervals", data["summary"]["ttc_below_3_vehicle_seconds"]),
        ):
            if not np.isclose(sum(data[key]), expected, rtol=1e-9, atol=1e-7):
                raise ValueError("physical summary/interval mismatch")
        if not np.isclose(
            sum(r["requested_not_arrived_vehicle_seconds"] for r in data["intervals"]),
            n * data["summary"]["system_seconds_per_requested_vehicle"],
            rtol=1e-9,
            atol=1e-7,
        ):
            raise ValueError("service summary/interval mismatch")
    duty = np.asarray(rollout["intervention_fractions"], dtype=float)
    if duty.shape != (t,) or not np.isfinite(duty).all() or (duty < 0).any() or (duty > 1).any():
        raise ValueError("invalid population-normalized intervention duty")
    actual, reference = rollout["intervals"], native["intervals"]

    def difference(key):
        return np.array([b[key] - a[key] for a, b in zip(actual, reference)])

    c = contract
    parts = dict(
        braking=0.5 * difference("braking_vehicle_seconds") / c.braking_scale_vehicle_seconds,
        effort=0.5 * difference("acceleration_energy") / c.effort_scale_m2_per_s3,
        service=c.service_weight_per_second
        * difference("requested_not_arrived_vehicle_seconds")
        / n,
        source=c.source_weight_per_second
        * (np.asarray(native["source_intervals"]) - rollout["source_intervals"])
        / n,
        ttc=c.ttc_weight
        * (np.asarray(native["ttc_intervals"]) - rollout["ttc_intervals"])
        / max(5, ref["ttc_below_3_vehicle_seconds"]),
        hard_events=2
        * (difference("added_collision_events") + difference("added_teleport_events")),
    )
    physical = sum(parts.values())
    delay = (
        own["system_seconds_per_requested_vehicle"] - ref["system_seconds_per_requested_vehicle"]
    )
    source = (own["source_wait_vehicle_seconds"] - ref["source_wait_vehicle_seconds"]) / n
    ttc = own["ttc_below_3_vehicle_seconds"] - ref["ttc_below_3_vehicle_seconds"]
    violations = dict(
        service=delay > 2,
        source=source > 2,
        ttc=ttc > max(5, 0.05 * ref["ttc_below_3_vehicle_seconds"]),
        completion=own["arrived"] < ref["arrived"]
        or own["unfinished_requested"] > ref["unfinished_requested"],
        hard_events=any(
            own["totals"][k] > ref["totals"][k]
            for k in ("added_collision_events", "added_teleport_events")
        ),
    )
    count = sum(violations.values())
    penalty = (max(0, float(physical.sum())) if count else 0) + c.violation_penalty * count
    preference = -c.intervention_episode_cap * duty / t
    rewards = physical + preference
    rewards[-1] -= penalty
    if not np.isfinite(rewards).all() or preference.sum() < -c.intervention_episode_cap - 1e-12:
        raise ValueError("reward audit failed")
    return rewards, dict(
        components={k: float(v.sum()) for k, v in parts.items()},
        intervention_cost=float(preference.sum()),
        terminal_penalty=penalty,
        violations=violations,
        reward_sum=float(rewards.sum()),
    )


def prepare_batch(rollout, native, contract=RewardContract()):
    rewards, audit = reward_series(rollout, native, contract)
    ends = np.zeros(len(rewards), dtype=bool)
    ends[-1] = True
    advantages, returns = team_advantages(
        rewards, rollout["old_values"], ends, gamma=1, gae_lambda=1
    )
    times = np.asarray(rollout["actor_times"], dtype=np.int64)
    if times.ndim != 1 or (len(times) and (times.min() < 0 or times.max() >= len(rewards))):
        raise ValueError("actor time mapping mismatch")
    if any(
        len(rollout[k]) != len(times)
        for k in ("actor_observations", "actions", "old_logp", "actor_ids")
    ):
        raise ValueError("actor alignment mismatch")
    return dict(
        rollout,
        rewards=rewards,
        returns=returns.astype(np.float32),
        team_advantages=advantages.astype(np.float32),
        actor_advantages=advantages[times].astype(np.float32),
        reward_audit=audit,
    )
