"""Registration constants and bindings for WD-01-PB-V04B.

Every number here is copied from REGISTRATION_PARTB_V04B.md (registered
FINAL, 2026-10-01; LF-sha256 e00cd98e...) -- the fit refuses to run
against a registration file whose LF-sha differs from the one recorded
at code time (reg_sha() recomputes and compares).

Splits are counted BY NAME (section 2): 39 TRAIN worlds, 5 CAL worlds,
HOLDOUT 0 (the J4 plan has none), chunk-5 never.  The plan is the only
split authority; train_worlds()/cal_worlds() re-read it and refuse on
any drift from the registered names.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

from . import _paths  # noqa: F401,E402  (sys.path shims, import side effect)

WD01 = _paths.WD01
REG_PATH = WD01 / "REGISTRATION_PARTB_V04B.md"
REG_SHA_EXPECTED_PREFIX = "e00cd98e"

SCHEMA = "wd01_partB_v04b/1"

# ---- registration file binding -------------------------------------------


def lf_sha256(path: Path) -> str:
    """sha256 over the file bytes with CRLF normalised to LF (the
    repo-wide 'LF-sha256' convention; the registration quotes this form)."""
    b = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def reg_sha() -> str:
    """LF-sha256 of the registration file; refuses to serve a wrong file."""
    s = lf_sha256(REG_PATH)
    if not s.startswith(REG_SHA_EXPECTED_PREFIX):
        raise RuntimeError(
            f"registration file {REG_PATH} has LF-sha256 {s[:8]}... but "
            f"this code is bound to {REG_SHA_EXPECTED_PREFIX}... -- do not "
            "re-point v0.4b at a different registration")
    return s


# ---- taus / quantiles / tuning grid (sections 4-5) ------------------------

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
# fixed small tuning grid of 6 candidates (section 5); NO other search
GRID = (
    # (max_iter, learning_rate, max_leaf_nodes, min_samples_leaf)
    (150, 0.05, 8, 20),
    (300, 0.05, 8, 20),
    (150, 0.1, 8, 20),
    (300, 0.1, 16, 10),
    (150, 0.05, 16, 10),
    (300, 0.05, 4, 40),
)
CV_FOLDS = 6                      # world-grouped folds on TRAIN only
BRANCH_DURATION_S = 600.0

# ---- J4 wall arms (section 2) ---------------------------------------------

WALL_ARMS = ("W-lo-early", "W-lo-late", "W-mid-early", "W-mid-late")
ARM_SPEED_FACTORS = {"W-lo": 0.6, "W-mid": 0.85}
TIMING_OFFSET_S = {"early": 0.0, "late": 60.0}
HOLD_CHOICES_S = (60.0, 120.0, 240.0)
J4_PLAN_REL = "output/campaign_walls_v1/results_plan/WORLD_PLAN.json"
J4_PLAN_SCHEMA = "wm59_walls_pilot_world_plan/1"
J4_PLAN_SEED = 20261101
J4_RUNS_ROOT = "runs/campaign_walls_pilot_chunk4"
J4_BASES_ROOT = "runs/campaign_stage1_chunk4"
# sources COPIED from the v0.4a root for the aux contrasts (training
# only; never scoring, selection-gating or acceptance) + the native
# reference every contrast pairs against
AUX_SOURCES = ("base", "pi", "fixed1", "fixed2",
               "switching1", "switching2", "switching3")
NATIVE_SOURCE = "native"

# ---- contrasts / acceptance (sections 4-6) --------------------------------

HEADLINE_CONTRAST = "wall-native"
AUX_CONTRASTS = ("damping-native", "pi-native", "fixed-native",
                 "switching-native")
CONTRAST_SOURCES = {
    "wall-native": (("W-lo-early", "W-lo-late",
                     "W-mid-early", "W-mid-late"), "native"),
    "damping-native": (("base",), "native"),
    "pi-native": (("pi",), "native"),
    "switching-native": (("switching1", "switching2", "switching3"),
                         "native"),
    "fixed-native": (("fixed1", "fixed2"), "native"),
}
ARM_FAMILY = {                     # arm source -> family one-hot slot
    "W-lo-early": "wall", "W-lo-late": "wall",
    "W-mid-early": "wall", "W-mid-late": "wall",
    "base": "damping",
    "pi": "pi",
    "fixed1": "fixed",
    "fixed2": "fixed",
    "switching1": "switching",
    "switching2": "switching",
    "switching3": "switching",
    "native": "native",
}
FAMILIES = ("wall", "damping", "pi", "fixed", "switching")

TAUS_A3 = (60.0, 120.0, 300.0)     # gating taus of A3
COVERAGE_BAND = (0.75, 0.85)       # overall 10-90 % coverage band (A1)
N_BOOT = 2000                      # world-bootstrap draws (A3-i)
BOOT_SEED = 20261001
BOOT_CI = 0.90

# ---- target heads (section 4; gating on delay only) -----------------------

HEADS = ("delay", "ttc", "brake")
HEAD_COL = {"delay": "total_delay_ref_s", "ttc": "ttc_lt3_veh_s",
            "brake": "hard_brake_s"}
GATING_HEAD = "delay"

# ---- paths (section 2/10; E: roots only, WD_FORBID_D=1) -------------------

V04A_DATA_ROOT = "output/wd01_partB_v0data_v04a"
V04B_DATA_ROOT = "output/wd01_partB_v0data_v04b"   # NEW root (post-ack fit)
RESULTS_DIR = HERE / "results"
CAL_MARKER_NAME = "CAL_READ_ONCE_V04B.json"
OWNER_ACK_NAME = "OWNER_ACK_PB_V04B.json"

# ---- planned schedule (section 12) ----------------------------------------

SEG_S = 30.0                       # segment length of the plan tensor
N_SEG = 19                         # 19 x 30 s = 570 s horizon
SCHED_ACTIONS = ("native", "damping", "pi", "wall")
