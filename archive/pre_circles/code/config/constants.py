"""Single source of truth for Traffic Shepherd constants.

Everything that two or more modules used to define independently (mode names,
speed-mode bitmasks, acceleration limits, hardcoded SUMO edge IDs, reward-term names)
lives here. Import from this module instead of re-declaring.
"""

from __future__ import annotations

from enum import IntEnum

# --- SUMO vehicle control -------------------------------------------------
# Regard safe speed (bit 0) + max accel (bit 1) + max decel (bit 2). Bit 0
# lets SUMO clamp an RL setSpeed() to its safe car-following speed — with
# mode 6 (bit 0 off) selfish/flat_marl commands ignored the leader and TTC
# events exploded (60+) on dense traffic. Right-of-way / red-light bits stay
# off so the RL policy is not otherwise overridden by SUMO's car-following
# model. Also used by the Shepherd engine's _set_speed_safe (env/actions.py),
# so the clamp applies there too.
RL_SPEED_MODE = 7

# IDM-native residual smoothing retains all SUMO longitudinal safety checks
# while setSpeed supplies the projected analytical target.
IDM_SAFE_SPEED_MODE = 31

# Fully permissive speed/lane-change mode used by the rule-based merge policy.
RULE_BASED_SPEED_MODE = 31
RULE_BASED_LANE_CHANGE_MODE = 1621


# --- Shepherd action modes ------------------------------------------------
# Final menu: docs/GOAL_FAIR_SAFE.md (fair + safe + travel-time budget).
class Mode(IntEnum):
    """Fair+safe Discrete(5) roles + opt-in wave_damp (index 5)."""

    COAST = 0  # let SUMO drive
    FAIR_BALANCE = 1  # tip-slow advantaged stream when delay gap large
    RECOVER_SPEED = 2  # push only when merge/down clear and ego crawling
    VACATE_DROP = 3  # leave ending lane L0 toward surviving L1
    PROTECT = 4  # emergency headway on low TTC / PET
    WAVE_DAMP = 5  # opt-in: match ahead-band speed EMA (wave damping)
    MERGE_MATCH = 6  # opt-in: match the merge-zone speed far upstream
    JAD_ABSORB = 7  # opt-in: hold absorbing speed at wave front (single)
    JAD_SUPPORT = 8  # opt-in: extend large time gap upstream of absorber


MODE_NAMES: list[str] = [
    "coast",
    "fair_balance",
    "recover",
    "vacate_drop",
    "guard",
    # Opt-in 6th mode (research_v2 port): registered in the global index so
    # the engine can dispatch it, but NOT part of the default mode set —
    # config/defaults.py "shepherd_modes" stays at the original 5. Runs opt
    # in via train.py --enable-wave-damp (appends it to shepherd_modes).
    "wave_damp",
    # Opt-in 7th mode (2026-08-08): match the merge-zone speed from far
    # upstream (buffered RSU reach) to absorb merge-origin waves. NOT part of
    # the default mode set; opt in via train.py --enable-merge-match.
    "merge_match",
    # Opt-in JAD modes (2026-08-09): single absorber holds the absorbing speed
    # at the wave front; upstream CAVs extend large time gaps (support driving).
    # Gated by the RSU wave signal. Opt in via train.py --enable-jad.
    "jad_absorb",
    "jad_support",
]

NUM_MODES = len(MODE_NAMES)

# Continuous longitudinal action range for Selfish and Flat-MARL (m/s²).
ACCELERATION_MIN = -3.0
ACCELERATION_MAX = 3.0

# Flat-MARL uses a narrower action range to match the effective operating
# bounds of shepherd's discrete modes (≈ ±1.0 m/s² after rate-limiting).
FLAT_MARL_ACCEL_MIN = -1.5
FLAT_MARL_ACCEL_MAX = 1.5

# --- Reward term names (paper §III: R = w_e·eff + w_s·safe + ...) ---------
REWARD_EFFICIENCY = "efficiency"
REWARD_SAFETY = "safety"
REWARD_STABILITY = "stability"
REWARD_FAIRNESS = "fairness"
REWARD_CONTROL_COST = "control_cost"

# --- Hardcoded SUMO edge / junction IDs on highway_merge_shepherd ---------
# These are referenced by zone features (observation) and travel-time split
# (evaluation). Keep them in one place so a network change only edits here.
PRE_MERGE_EDGE = "1191554636"
MERGE_EDGE = "1487606179"
DOWNSTREAM_EDGE = "1487606178"
RAMP_EDGE = "11553048#1"
MERGE_JUNCTION = "1780739017"
# Mainline origin edge (used to split tripinfo into mainline vs ramp groups).
MAINLINE_EDGE = PRE_MERGE_EDGE

# Lane lengths used as fallbacks when TraCI cannot report them.
PRE_MERGE_LANE_LENGTH = 1144.95
RAMP_LANE_LENGTH = 378.66

# Merge-edge lane layout: 3 lanes in, 2 lanes out (verified from the net file:
# junction connects _1->1487606178_0 and _2->1487606178_1; lane _0 has NO
# outgoing connection). Lane 1487606179_0 ends at the merge, so _0 vehicles
# are forced into _1 — pairs (_0, _1) on this edge are converging-lane
# (side-conflict) races, while (_1, _2) run parallel and are NOT side
# conflicts. (Corrected 2026-07-30: the dropped lane is _0, not _2.)
MERGE_EDGE_NUM_LANES = 3
MERGE_DROPPED_LANE_INDEX = 0
MERGE_SURVIVING_LANE_INDEX = 1
DOWNSTREAM_EDGE_NUM_LANES = 2
DOWNSTREAM_LANE_LENGTH = 477.89
# Net length (m) of every merge-edge lane; the dropped lane ends here, so this
# is the side-conflict point position used by merge_arrival_pet().
MERGE_DROPPED_LANE_LENGTH = 75.44

# CAVs always use this SUMO vType when selected (coast / SUMO-in-control =
# normal driver, not the route mix of aggressive/cautious). HDVs keep the
# route file's vTypeDistribution.
CAV_VTYPE = "normal_car"

# Default vehicle length for bumper-gap corrections in TTC metrics (m).
DEFAULT_VEH_LENGTH = 5.0

# --- Lane-drop bottleneck scenario (bottleneck scene) -----------------------
# upstream: 3 lanes before bottleneck, downstream: 2 lanes after
BTLN_PRE_MERGE_EDGE = "upstream"
BTLN_MERGE_EDGE = "downstream"
BTLN_RAMP_LANE = "upstream_2"  # dropped lane (lane index 2 of upstream)
BTLN_MERGE_JUNCTION = "n1"
