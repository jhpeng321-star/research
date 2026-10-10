"""Frozen layout constants for the WM-N1 featurizer (mirrored in FORMAT.md).

Everything here is a hard contract: tensor ranks, dtypes, feature indices and
mask semantics.  Do not change without a new FORMAT.md revision.
"""
from __future__ import annotations

# ---------------------------------------------------------------- physics --
DT_S = 0.5                      # physics tick
K_MAX = 128                     # vehicle slots per tick (v2 default; see below)
N_LANES = 3                     # lane_phys classes: main_left/main_right/accel

# ----------------------------------------------------------------- format --
# K_max is a FORMAT parameter (REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md
# section 4, option "K = 192"): the frozen v2 format keeps 128 slots and stays
# byte-reproducible; the new v3 format carries 192.  The pool rule itself is
# unchanged -- only the pool size follows the format.
K_MAX_V2 = 128                  # frozen v2 (FEAT1 v2/v2c packs and shards)
K_MAX_V3 = 192                  # new v3 format (A5 amendment 5)
FORMAT_V2 = "v2"
FORMAT_V3 = "v3"
FORMAT_K_MAX = {FORMAT_V2: K_MAX_V2, FORMAT_V3: K_MAX_V3}
DEFAULT_FORMAT = FORMAT_V2


def k_max_of_format(format_name: str) -> int:
    """Registered K_max of a named feature format (``"v2"`` / ``"v3"``)."""
    try:
        return FORMAT_K_MAX[format_name]
    except KeyError:
        raise ValueError(
            f"unknown format {format_name!r}; registered formats: "
            f"{sorted(FORMAT_K_MAX)}") from None


def format_of_k_max(k_max: int) -> str:
    """Format name of a registered K_max (128 -> v2, 192 -> v3)."""
    for name, k in FORMAT_K_MAX.items():
        if int(k_max) == int(k):
            return name
    raise ValueError(
        f"K_max {k_max} is not a registered format K ("
        f"{sorted(FORMAT_K_MAX.values())})")


def resolve_k_max(format=None, k_max=None) -> tuple:
    """Resolve the ``(format, k_max)`` pair; defaults are v2 / 128.

    ``format`` and ``k_max`` may each be given (they must agree); an
    explicit ``k_max`` alone selects its registered format, and an
    unregistered pair raises ``ValueError``.
    """
    if format is None and k_max is not None:
        format = format_of_k_max(k_max)
    fmt = DEFAULT_FORMAT if format is None else str(format)
    k = k_max_of_format(fmt) if k_max is None else int(k_max)
    if k_max_of_format(fmt) != k:
        raise ValueError(
            f"format {fmt!r} carries K_max {k_max_of_format(fmt)}, "
            f"not {k}")
    return fmt, k


# ---------------------------------------------------------- vehicle-step --
VEH_F = 16
# feature index -> name (see FORMAT.md section 2)
S_DOWN, LANE_LEFT, LANE_RIGHT, LANE_ACCEL = 0, 1, 2, 3
V_IDX, A_IDX, LENGTH_IDX, IS_CAV_IDX = 4, 5, 6, 7
# action one-hot {native, damping, pi, wall, unknown}
ACT_NATIVE, ACT_DAMPING, ACT_PI, ACT_WALL, ACT_UNKNOWN = 8, 9, 10, 11, 12
WALL_MEMBER_IDX = 13
JUNCTION_IDX = 14
PRESENT_IDX = 15                # present flag (redundant with veh_mask)
VEH_FEATURE_NAMES = (
    "s_down_m", "lane_main_left", "lane_main_right", "lane_accel",
    "v_mps", "a_mps2", "nominal_length_m", "is_cav",
    "action_native", "action_damping", "action_pi", "action_wall",
    "action_unknown", "wall_member", "junction", "present")

# canonical CAV action token order (must equal actions_v04.ACTIONS_V04)
ACTIONS = ("native", "damping", "pi", "wall", "unknown")
ACTION_SLOTS = {a: s for s, a in enumerate(ACTIONS)}

NOMINAL_LENGTH_M = 5.0          # params_nominal.json vehicle.length_m

# ------------------------------------------------------------- plan tensor --
N_SEGMENTS = 19                 # 19 x 30 s = 570 s horizon
SEGMENT_S = 30.0
PLAN_F = 8
PLAN_ACTION_SLOTS = 0           # plan slots 0-4: action one-hot
PLAN_WALL_ID = 5                # float: wall index in meta.wall_ids, -1 none
PLAN_WALL_SPEED = 6             # wall_speed_mps / 40
PLAN_WALL_HOLD = 7              # planned_hold_remaining_s / 600
PLAN_FEATURE_NAMES = (
    "action_native", "action_damping", "action_pi", "action_wall",
    "action_unknown", "wall_id_index", "wall_speed_mps_over_40",
    "wall_hold_s_over_600")

N_DECISION = 20                 # t0 + 30 k, k = 0..19 (covers the 570 s plan)
DECIDE_EVERY_S = 30.0

# ------------------------------------------------------------ wall tokens --
MAX_WALLS = 4
WALL_F = 11
WALL_PHASES = ("requested", "assembling", "holding", "released", "aborted")
WALL_FEATURE_NAMES = (
    "phase_requested", "phase_assembling", "phase_holding", "phase_released",
    "phase_aborted", "wall_speed_mps_over_40", "time_in_phase_s_over_600",
    "planned_hold_remaining_s_over_600", "n_members_over_4",
    "members_mean_s_down_m_over_1000", "members_lane_span_over_2")

# -------------------------------------------------------- counter history --
N_ENTRANCES = 2                 # 0 mainline, 1 ramp
COUNTER_BINS = 30               # 30 x 10 s = 300 s
COUNTER_BIN_S = 10.0
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)

# --------------------------------------------------------- step targets --
LANE_STAY, LANE_LEFT_CLS, LANE_RIGHT_CLS, LANE_NONE = 0, 1, 2, -1

INDEX_SCHEMA = "wm_n1_features_index/1"
FORMAT_SCHEMA = "wm_n1_features_format/1"
FORMAT_SCHEMA_V2 = FORMAT_SCHEMA
FORMAT_SCHEMA_V3 = "wm_n1_features_format_v3/1"
FORMAT_SCHEMAS = {FORMAT_V2: FORMAT_SCHEMA_V2, FORMAT_V3: FORMAT_SCHEMA_V3}
