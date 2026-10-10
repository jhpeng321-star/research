"""Physical reward for the PPO pilot: per-tick scaled cost deltas.

The reward is computed from the SAME per-control-tick interval counters the
outcome evaluator sums (paired_experiments.evaluate_arm), so the episode
return is exactly the episode's physical scaled cost plus penalties. There
is no reference-outcome, future or privileged term in the reward; matched
damping is used only offline to GRADE trained policies.

r_t = -(Delta service/S_service + Delta wait/S_wait + Delta braking/S_braking
        + Delta effort/S_effort + Delta ttc/S_ttc)
      - PENALTY_COLLISION * (new collision involvements)
      - PENALTY_TELEPORT * (new starting teleports)
      - INTERVENTION_COST_PER_STEP * (1 if wall-owned this tick else 0)

Episode return = sum over the fixed 2400 s horizon; gamma = 1.0 with no
truncation, so every delayed consequence reaches every earlier decision
with full weight. Raw components, normalized components, intervention cost
and the final return are logged separately (reward_log rows).
"""
from . import ppo_contract as C

COMPONENT_KEYS = {
    "service": "requested_not_arrived_vehicle_seconds",
    "braking": "braking_vehicle_seconds",
    "effort": "acceleration_energy",
    "ttc": "ttc_below_3_vehicle_seconds",
}
SCALES = {"service": C.SCALE_SERVICE, "braking": C.SCALE_BRAKING,
          "effort": C.SCALE_EFFORT, "ttc": C.SCALE_TTC}


def tick_reward(interval, *, wall_owned, source_wait_delta=0.0, previous=None):
    """One control tick's reward row from the interval counter (deltas = values).

    The interval counter already contains this tick's increments, so
    "delta" is the counter itself. ``previous`` is unused for counters but
    kept for absolute-cumulative callers (tests).
    """
    normalized, raw = {}, {}
    for name, key in COMPONENT_KEYS.items():
        raw[name] = float(interval.get(key, 0.0))
        normalized[name] = raw[name] / SCALES[name]
    raw["wait"] = float(source_wait_delta)
    normalized["wait"] = raw["wait"] / C.SCALE_WAIT
    raw["collisions"] = int(interval.get("collision_involvements", 0))
    raw["teleports"] = int(interval.get("starting_teleports", 0))
    cost = sum(normalized.values())
    penalties = C.PENALTY_COLLISION * raw["collisions"] + C.PENALTY_TELEPORT * raw["teleports"]
    intervention = C.INTERVENTION_COST_PER_STEP if wall_owned else 0.0
    return dict(reward=-(cost + penalties) - intervention, cost_scaled=cost,
                penalties_scaled=penalties, intervention=intervention,
                raw=raw, normalized=normalized, wall_owned=bool(wall_owned))


def episode_return(reward_rows):
    """Sum of tick rewards; also the total scaled physical cost breakdown."""
    totals = dict(return_=sum(r["reward"] for r in reward_rows),
                  cost=sum(r["cost_scaled"] for r in reward_rows),
                  penalties=sum(r["penalties_scaled"] for r in reward_rows),
                  intervention=sum(r["intervention"] for r in reward_rows))
    for name in ("service", "wait", "braking", "effort", "ttc"):
        totals[name] = sum(r["raw"][name] for r in reward_rows)
    return totals


# ---- controlled fixtures for reward ordering (pure functions, no SUMO) ----

def _episode(component_totals, wall_steps=0, population=1415):
    """Build a reward-row list whose sums reproduce component_totals."""
    ticks = 2400
    rows = []
    totals = {name: float(component_totals.get(name, 0.0))
              for name in ("service", "wait", "braking", "effort", "ttc")}
    collisions = int(component_totals.get("collisions", 0.0))
    teleports = int(component_totals.get("teleports", 0.0))
    per_tick = {k: v / ticks for k, v in totals.items()}
    for t in range(ticks):
        interval = {key: per_tick[name] for name, key in COMPONENT_KEYS.items()}
        interval["collision_involvements"] = collisions / ticks
        interval["starting_teleports"] = teleports / ticks
        wait_delta = per_tick["wait"]
        wall_owned = t < wall_steps
        if wall_owned and t == wall_steps - 1:
            # place integer event counts exactly on the last wall step
            interval["collision_involvements"] = collisions
            interval["starting_teleports"] = teleports
        row = tick_reward(interval, wall_owned=wall_owned, source_wait_delta=wait_delta)
        rows.append(row)
    return rows


def fixture_unnecessary_intervention_loses():
    """An episode with a harm-free but pointless wall must score below damping."""
    damping = episode_return(_episode(dict.fromkeys(
        ("service", "wait", "braking", "effort", "ttc"), 0.0)))
    wall = episode_return(_episode(dict.fromkeys(
        ("service", "wait", "braking", "effort", "ttc"), 0.0), wall_steps=80))
    assert damping["return_"] > wall["return_"]
    assert abs(wall["intervention"] - C.INTERVENTION_MAX_CONTRIBUTION * 80 / 100) < 1e-12


def fixture_helpful_intervention_beats_despite_cost():
    """A wall that removes real braking and TTC cost can beat damping."""
    damping = episode_return(_episode(dict(braking=1800.0, ttc=2200.0)))
    helped = episode_return(_episode(dict(braking=1700.0, ttc=2100.0), wall_steps=80))
    assert helped["return_"] > damping["return_"]


def fixture_harmful_intervention_scores_worst():
    """A wall that adds collisions must score strictly below both others."""
    base = episode_return(_episode(dict.fromkeys(
        ("service", "wait", "braking", "effort", "ttc"), 0.0)))
    harmful = episode_return(_episode(
        {**dict.fromkeys(("service", "wait", "braking", "effort", "ttc"), 0.0),
         "collisions": 1}, wall_steps=80))
    assert harmful["return_"] < base["return_"]
    # one involvement (25 scaled units) dominates several full-episode TTC costs
    assert C.PENALTY_COLLISION > 5.0


def fixture_delayed_penalty_is_retained():
    """A collision 200 s after release still lands in the episode return."""
    ticks = 2400
    rows = []
    for t in range(ticks):
        interval = dict(braking_vehicle_seconds=0.0, acceleration_energy=0.0,
                        ttc_below_3_vehicle_seconds=0.0,
                        requested_not_arrived_vehicle_seconds=0.0,
                        collision_involvements=int(t == 2000), starting_teleports=0)
        rows.append(tick_reward(interval, wall_owned=False))
    total = episode_return(rows)
    assert total["penalties"] == C.PENALTY_COLLISION
    assert abs(total["return_"] + C.PENALTY_COLLISION) < 1e-9


def fixture_return_does_not_change_acceptance():
    """The scalar return never feeds the registered acceptance classification."""
    from .escalation_evidence import BenefitCriteria, benefit_label
    criteria = BenefitCriteria(
        primary_minimum_improvement={"system_seconds_per_request": 1.0,
                                     "braking_seconds_per_request": 0.02},
        maximum_cost_increase={"system_seconds_per_request": 0.5,
                               "source_wait_seconds_per_request": 0.1,
                               "braking_seconds_per_request": 0.005,
                               "effort_per_request": 0.5,
                               "ttc_seconds_per_request": 0.02})
    # A better (higher) return with a guardrail violation is still 'harmful'.
    effect = dict(control="damping", treatment="wall",
                  improvements={"braking_seconds_per_request": 0.05,
                                "effort_per_request": 0.0,
                                "system_seconds_per_request": 0.0,
                                "ttc_seconds_per_request": -0.05,
                                "source_wait_seconds_per_request": 0.0},
                  excess_collisions=0, excess_teleports=0, excess_unfinished=0,
                  control_unfinished_requested=0, treatment_unfinished_requested=0,
                  control_collision_involvements=0, treatment_collision_involvements=0)
    label = benefit_label(effect, criteria)["label"]
    assert label == "harmful"  # TTC violation decides, whatever the return says
