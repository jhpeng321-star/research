"""Frozen PPO pilot contract: actions, observations, reward, seeds, budget.

Everything in this module is declared BEFORE any training simulation and is
hashed into the pilot registration. Nothing here may be tuned against
development outcomes; the reward scale constants are round numbers chosen
by inspection of the two registered damping episodes (4107/4108 m2700 p20)
to place a typical full-episode cost near 1.0 scaled unit per component.

Physical basis (registered metric definitions, unchanged acceptance
criteria live in escalation_evidence.BenefitCriteria):
  service  = requested_not_arrived_vehicle_seconds  [veh-s]
  wait     = source waiting vehicle-seconds          [veh-s]
  braking  = time with measured acceleration < -3 m/s^2 [veh-s]
  effort   = integral of measured acceleration^2     [m^2/s^3]
  ttc      = vehicle-seconds with closing TTC < 3 s  [veh-s]
Sign convention everywhere: treatment - reference; negative = improvement.
"""
from dataclasses import asdict
import hashlib
import json

POLICY_VERSION = "wall_ppo/2"
SCHEMA = "ppo_rsu_pilot_contract/1"

# ---- action contract ----------------------------------------------------
# One centralized RSU discrete actor; the analytical feasibility gate is not
# a second learned network. Action space is Discrete(2) with phase-dependent
# meaning; unavailable actions are masked before sampling and the SAME mask
# is used for the PPO log-probability.
CADENCE_S = 5.0                 # decision ticks: t % 5 == 0 (control grid 1 s)
EARLIEST_ACTIVATION_S = 120.0   # warmup; no activation before this
LATEST_ACTIVATION_S = 1800.0    # leaves >= 540 s of post-release horizon
BASE_HOLD_S = 20.0              # existing verified lifecycle_v2 HOLD
CONTINUE_INCREMENT_S = 10.0     # one registered hold_extend/2-sized increment
MAX_HOLD_S = 40.0               # maximum total HOLD (base + 2 increments)
MAX_WALL_STEPS = 100            # bound for the intervention-cost proof
LATEST_AUTO_RELEASE_S = 2350.0  # episode-end behavior: forced taper here
MAX_ACTIVATIONS = 1             # at most one wall activation per episode
A_WAIT, A_ACTIVATE = 0, 1       # inactive-phase actions
A_RELEASE, A_CONTINUE = 0, 1    # v1 hold-phase actions (superseded; kept for
                                # historical checkpoint interpretation)
A_MAINTAIN, A_HOLD_RELEASE = 0, 1   # v2 interior hold: keep / early release
A_EXTEND, A_DEADLINE_RELEASE = 0, 1  # v2 deadline: extend / release

# Duration semantics v2 (repair of the v1 contract): during an admitted
# HOLD, interior cadence decisions choose MAINTAIN (keep the admitted
# commitment) or RELEASE (early release when physically permitted). At the
# scheduled HOLD end the decision is EXTEND (+increment, admitted on the
# proposed remaining maneuver incl. exit, using current acquired reports)
# or RELEASE; a rejected extension follows the existing release schedule.
# When the maximum admitted HOLD is committed, no further extension exists:
# the commitment is maintained to its scheduled end and the scheduled
# release executes when the admitted maximum is actually reached. The
# deadline decision is evaluated AT the scheduled end (never held beyond it
# waiting for a cadence tick).
ACTION_MEANINGS = {
    "inactive": ["WAIT", "ACTIVATE"],
    "hold_interior": ["MAINTAIN", "RELEASE"],
    "hold_deadline": ["EXTEND", "RELEASE"],
}

HOLD_DECISION_SEMANTICS = dict(
    interior="MAINTAIN | RELEASE (cadence ticks strictly before the "
             "scheduled end)",
    deadline="EXTEND | RELEASE (evaluated at the scheduled end tick)",
    at_maximum="scheduled release executes; no EXTEND decision",
    never_hold_beyond_deadline=True)

# Wall-speed profile frozen from existing evidence: the lifecycle_v2 entry
# profile (target = mean reported member speed - 2 m/s, capped at 24 m/s).
# Rationale: this profile has the most verified completed maneuvers in both
# exposed seeds (lifecycle_v2, cohort_v1, duration_release arms) and is the
# base of all registered continuation evidence. No speed search is performed.

# ---- observation contract -----------------------------------------------
HISTORY_STACK = 4               # stacked decision-tick feature vectors (20 s span)
# FEATURE_NAMES: order matters; it is the network input layout. Formulas,
# units and missing handling are documented in docs/ppo_pilot_v1/CONTRACT.md
# and mirrored in ppo_features.build_features.
FEATURE_NAMES = (
    "cand_mean_speed_mps",          # mean reported speed of frozen cohort (m/s)
    "cand_min_speed_mps",
    "cand_speed_spread_mps",        # max - min cohort speed (m/s)
    "cand_mean_accel_mps2",         # mean reported acceleration (m/s^2)
    "cand_span_m",                  # max-min distance_to_merge (m)
    "cand_mean_gap_front_m",        # mean front neighbor gap (m; None -> 0 + flag)
    "cand_mean_rel_speed_front_mps",
    "all_mean_speed_mps",           # all fresh reporting CAVs (m/s)
    "all_speed_pvariance_mps2",
    "all_mean_accel_mps2",
    "all_min_speed_mps",
    "approach_mean_speed_mps",      # reporters with distance_to_merge <= 400 m
    "approach_reporter_count",
    "downstream_mean_speed_mps",    # reporters already past the merge region
    "braking_reporter_count",       # reporters with acceleration < -3 m/s^2
    "reporter_count",               # fresh reporters this tick
    "mean_report_age_s",            # now - frame.timestamp_s (s)
    "max_report_age_s",
    "cohort_fresh_fraction",        # cohort members with fresh reports / 2
    "phase_inactive",               # one-hot phase (damping/assembly/hold)
    "phase_assembly",
    "phase_hold",
    "elapsed_hold_s",               # elapsed DELIVERED hold (now - hold start)
    "admitted_total_hold_s",        # committed total HOLD at this instant
    "remaining_admitted_hold_s",    # admitted - elapsed (>= 0)
    "remaining_extension_capacity_s",  # MAX_HOLD - admitted
    "hold_deadline_decision",       # 1 when this decision is the scheduled-end EXTEND/RELEASE decision
    "time_since_start_s",           # episode clock (s)
    "activation_used",              # 0/1: at most one activation
    "cand_mean_speed_delta_5s",     # 5 s trend of cohort mean speed (m/s)
    "all_mean_speed_delta_5s",      # 5 s trend of all-reporter mean (m/s)
    "feasible_activate_now",        # cheap mask value as an explicit feature
)
OBS_DIM = len(FEATURE_NAMES) * HISTORY_STACK

# ---- reward contract ----------------------------------------------------
# Per control tick (1 s): r_t = -(sum of scaled physical cost deltas) - penalty
# - intervention. Episode return = sum of r_t over the fixed 2400 s horizon.
SCALE_SERVICE = 250_000.0       # veh-s  (registered damping 4107 ~289.8k, 4108 ~194.5k)
SCALE_WAIT = 40_000.0           # veh-s  (~58.8k / ~8.1k)
SCALE_BRAKING = 2_000.0         # veh-s  (~1792.5 / ~1398.5)
SCALE_EFFORT = 200_000.0        # m^2/s^3 (~148.8k / ~100.9k)
SCALE_TTC = 2_000.0             # veh-s  (~2282 / ~1604)
PENALTY_COLLISION = 25.0        # per involvement; dominates all scaled costs
PENALTY_TELEPORT = 5.0          # per starting teleport
INTERVENTION_COST_PER_STEP = 5e-5  # per wall-owned control step
# Intervention bound: wall ownership lasts <= MAX_WALL_STEPS control steps
# (assembly <= 50 + HOLD <= 40 + taper <= 10), so the maximum episode
# intervention contribution is 100 * 5e-5 = 0.005 scaled units - strictly
# below the smallest registered-material benefit (service >= 1 s/request is
# ~0.006 scaled at population ~1415), so accumulation over steps can never
# dominate a meaningful physical benefit.
INTERVENTION_MAX_CONTRIBUTION = INTERVENTION_COST_PER_STEP * MAX_WALL_STEPS

# ---- PPO configuration (one declared configuration; optimizer seed 20260907)
PPO_CONFIG = dict(
    hidden=(64, 64), activation="tanh",
    lr=3e-4, clip_eps=0.2, epochs=4, minibatch_size=256,
    gamma=1.0, gae_lambda=0.95,  # superseded constants, kept for provenance
    advantage_estimator="monte_carlo_return_baseline",
    entropy_coef=0.01, value_coef=0.5,
    max_grad_norm=0.5, optimizer_seed=20260907,
    episodes=32, episodes_per_update=4, updates=8,
    deterministic_eval=False,  # training samples; deployment eval is argmax
    torch_version="2.12.1+cpu",
)
# Credit reach: episodes always complete the registered 2400 s horizon, and
# the advantage estimator is the UNDISCOUNTED Monte Carlo return-to-go with
# the learned value baseline: every decision's critic target is its exact
# remaining physical return, and a delayed consequence reaches the causing
# decision with full (undecayed) weight - quantified in
# tests/test_ppo_credit.py against the superseded GAE(lambda=0.95) trace,
# which attenuated direct credit by 0.95^steps (~1% at 450 s, ~1e-8 at the
# horizon). No credit reset at wall release; the episode return is exactly
# the physical scaled cost, accounted once.

# ---- scenarios: seeds, regimes, splits ----------------------------------
TRAINING_SCENARIOS = (  # 8 training scenario seeds (regime: loading vph)
    (4107, 2700), (4108, 2700),   # exposed current-era seeds (debug/training)
    (4109, 2700), (4110, 2000), (4111, 2700),
    (4112, 2000), (4113, 2700), (4114, 2000),
)
DEVELOPMENT_SCENARIOS = (  # 4 disjoint development scenario seeds
    (4115, 2700), (4116, 2000), (4117, 2700), (4118, 2000),
)
# Provenance: all twelve seeds lie in the declared historical exposed
# partition EXPOSED_CONTROLLER_SEEDS = 4107..4132 (seeds.py). 4107/4108 are
# the current-era exposed seeds; 4109..4118 are current-era-unused. The
# partition carried archived-era (different map/actuator) assignments,
# recorded here honestly; none of the seeds is in any sealed acceptance
# block. Training and development sets are disjoint by construction.

PENETRATION = 0.2
DECISION_TIME_S = 600           # reference snapshot/cohort time (existing contract)
HORIZON_S = 2400
COMMUNICATION = dict(max_report_age_s=1.0, history_s=10.0)

# ---- budget ceilings (simulated seconds) --------------------------------
CEILINGS = dict(
    categories={
        "mechanics": {"max_allocations": 2, "max_seconds": 240.0},
        "references": {"max_allocations": 12, "max_seconds": 28800.0},
        "training": {"max_allocations": 32, "max_seconds": 76800.0},
        "dev_eval": {"max_allocations": 12, "max_seconds": 28800.0},
    },
    total_seconds=134640.0,
)

STOPPING_RULES = (
    "A failed mechanics allocation stops all further simulation in this campaign "
    "until reported; its receipt is preserved; no silent repair-and-relaunch.",
    "Stop on any non-finite loss, reward, observation or parameter update.",
    "Stop on any observation/action/reward contract violation raised by the env.",
    "Stop on any budget rejection from the launch-path guard.",
    "A policy that legitimately prefers damping is a result, not a bug.",
)
CHECKPOINT_RULE = "final completed checkpoint (update 8); best-dev selection forbidden"


def contract_dict():
    return dict(schema=SCHEMA, policy_version=POLICY_VERSION,
                actions=dict(cadence_s=CADENCE_S, earliest_activation_s=EARLIEST_ACTIVATION_S,
                    latest_activation_s=LATEST_ACTIVATION_S, base_hold_s=BASE_HOLD_S,
                    continue_increment_s=CONTINUE_INCREMENT_S, max_hold_s=MAX_HOLD_S,
                    max_wall_steps=MAX_WALL_STEPS, latest_auto_release_s=LATEST_AUTO_RELEASE_S,
                    max_activations=MAX_ACTIVATIONS,
                    inactive=dict(wait=A_WAIT, activate=A_ACTIVATE),
                    hold=dict(v2_interior=dict(maintain=A_MAINTAIN, release=A_HOLD_RELEASE),
                          v2_deadline=dict(extend=A_EXTEND,
                                           release=A_DEADLINE_RELEASE),
                          v1_superseded=dict(release=A_RELEASE,
                                             continue_=A_CONTINUE))),
                observation=dict(feature_names=list(FEATURE_NAMES), history_stack=HISTORY_STACK,
                                 obs_dim=OBS_DIM),
                reward=dict(scale_service=SCALE_SERVICE, scale_wait=SCALE_WAIT,
                    scale_braking=SCALE_BRAKING, scale_effort=SCALE_EFFORT,
                    scale_ttc=SCALE_TTC, penalty_collision=PENALTY_COLLISION,
                    penalty_teleport=PENALTY_TELEPORT,
                    intervention_cost_per_step=INTERVENTION_COST_PER_STEP,
                    intervention_max_contribution=INTERVENTION_MAX_CONTRIBUTION),
                ppo=PPO_CONFIG,
                scenarios=dict(training=[dict(seed=s, loading=load) for s, load in TRAINING_SCENARIOS],
                    development=[dict(seed=s, loading=load) for s, load in DEVELOPMENT_SCENARIOS],
                    penetration=PENETRATION, decision_time_s=DECISION_TIME_S,
                    horizon_s=HORIZON_S, communication=COMMUNICATION),
                ceilings=CEILINGS, stopping_rules=list(STOPPING_RULES),
                checkpoint_rule=CHECKPOINT_RULE)


def contract_sha256():
    return hashlib.sha256(json.dumps(contract_dict(), sort_keys=True).encode()).hexdigest()


def profile_for_hold(hold_s):
    """Frozen wall_ppo/1 profile with a registered total HOLD."""
    from .wall_ppo_policy import PpoHoldProfile
    return PpoHoldProfile(hold_s=float(hold_s))


assert asdict  # keep dataclasses import meaningful for subclasses elsewhere
