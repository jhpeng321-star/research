# ruff: noqa
# VENDORED from output/wd01/wm_n1/model/constants.py (last changed in commit 02a2d8c; copied at baseline bf7d4cb).
# Copied byte-for-byte apart from this header.
# Do not edit: behaviour is frozen evidence; WP-C only mirrors it.
"""WM-N1 model constants.

Mirror of the frozen featurizer layout in
``output/wd01/wm_n1/features/FORMAT.md`` and ``features/constants.py``.  The
model NEVER imports featurizer code at runtime (read-only contract); a test
(:mod:`model.tests.test_contract`) asserts every value here equals the
featurizer's, so the frozen tensor contract cannot silently drift.

Sources: ``WM_N1_DESIGN.md`` sections 1-3 and ``REGISTRATION_WM_N1.md``
section 3.
"""
from __future__ import annotations

# ---------------------------------------------------------------- physics --
DT_S = 0.5                      # physics tick (FORMAT.md section 0)
K_MAX = 128                     # vehicle slots per tick (v2 default)
N_LANES = 3                     # lane_phys classes

# ----------------------------------------------------------------- format --
# K_max is a FORMAT parameter (amendment 5, "K = 192"): v2 keeps 128 slots
# (frozen, byte-reproducible); the new v3 format carries 192.  Mirrors
# ``features/constants.py`` (test_contract asserts every pair).
K_MAX_V2 = 128
K_MAX_V3 = 192
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

# ---------------------------------------------------------- vehicle-step --
VEH_F = 16
S_DOWN, LANE_LEFT, LANE_RIGHT, LANE_ACCEL = 0, 1, 2, 3
V_IDX, A_IDX, LENGTH_IDX, IS_CAV_IDX = 4, 5, 6, 7
ACT_NATIVE, ACT_DAMPING, ACT_PI, ACT_WALL, ACT_UNKNOWN = 8, 9, 10, 11, 12
WALL_MEMBER_IDX = 13
JUNCTION_IDX = 14
PRESENT_IDX = 15
ACTIONS = ("native", "damping", "pi", "wall", "unknown")
NOMINAL_LENGTH_M = 5.0

# ------------------------------------------------------------- plan --------
N_SEGMENTS = 19                 # 19 x 30 s = 570 s horizon
SEGMENT_S = 30.0
PLAN_F = 8
# plan slots 0-4: action one-hot; 5 wall id; 6 wall speed; 7 wall hold
N_DECISION = 20

# ------------------------------------------------------------ wall ---------
MAX_WALLS = 4
WALL_F = 11
N_WALL_PHASES = 5
WALL_PHASES = ("requested", "assembling", "holding", "released", "aborted")

# -------------------------------------------------------- counter ----------
N_ENTRANCES = 2
COUNTER_BINS = 30
COUNTER_BIN_S = 10.0

# ------------------------------------------------------- step targets ------
LANE_STAY, LANE_LEFT_CLS, LANE_RIGHT_CLS, LANE_NONE = 0, 1, 2, -1
N_LANE_CLASSES = 3
N_MIX = 3                       # GMM components for the step acceleration

# ------------------------------------------------------- outcome heads -----
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
N_TAU = len(TAUS)
# design section 2: total delay, TTC<3 exposure and hard-brake seconds
OUTCOME_TARGETS = ("delay_s", "ttc_lt3_s", "hard_brake_s")
N_OUTCOME_TARGETS = len(OUTCOME_TARGETS)
N_QUANTILES = 19                # quantile levels for the outcome heads
# 19 levels: 0.05, 0.10, ..., 0.95
QUANTILE_LEVELS = tuple(round(0.05 * i, 4) for i in range(1, N_QUANTILES + 1))

# ------------------------------------------------------- validity ----------
A_MIN = -9.0                    # design section 2.5 acceleration clip
A_MAX = 3.0
N_MEMBERS = 5                   # deep ensemble size (design section 3)
CONTRAST_WEIGHT = 1.0           # lambda_c = 1 (design section 3, line 115)

# ------------------------------------------------- correction summary window -
#: Number of rollout steps over which the P5 CAV-motion summary is built.  The
#: TRAINER rolls out ``train.idea_a.IdeaATrainConfig.rollout_horizon_steps`` and
#: summarizes exactly that window; the evaluator / scoring API roll out much
#: longer horizons (metrics need up to 120 s / 300 s), so they MUST slice the
#: trajectory back to this SAME window or the correction head sees a summary
#: from a different distribution than it was trained on.  This is the single
#: source of that window (``IdeaATrainConfig.rollout_horizon_steps`` defaults
#: to it and ``evalh`` / ``wm_api`` call :func:`correction_window_steps`).
CORRECTION_WINDOW_STEPS = 20


def correction_window_steps(n_states: int) -> int:
    """Rollout steps feeding the correction summary for ``n_states`` states.

    The window is the registered training horizon, capped by the trajectory
    actually available (``n_states - 1`` transitions); at least one step.
    """
    return max(1, min(int(n_states) - 1, CORRECTION_WINDOW_STEPS))
