"""Frozen WM-N1-INC1 constants and path resolution (registration-bound).

Every value below is transcribed from ``REGISTRATION_WM_N1_INC1.md`` (LF-sha
``d5282ecb...``).  The runner refuses when the on-disk registration's
LF-normalised sha256 differs from :data:`REGISTRATION_LF_SHA`.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

# ---------------------------------------------------------------- path roots
INC1_ROOT = Path(__file__).resolve().parent
WD01_ROOT = INC1_ROOT.parent
OUTPUT_ROOT = WD01_ROOT.parent
WORKSPACE_ROOT = OUTPUT_ROOT.parent
WM_N1_ROOT = WD01_ROOT / "wm_n1"

REGISTRATION_PATH = WM_N1_ROOT / "REGISTRATION_WM_N1_INC1.md"
#: LF-normalised sha256 recorded in the registration brief (d5282ecb...).
REGISTRATION_LF_SHA = (
    "d5282ecb5555032dcb65b695e8e69f925055dc9ec06994fc4287fbabb4e132d7")

#: the gated inputs; the owner places both before any simulation.
AUTHORIZATION_PATH = INC1_ROOT / "AUTHORIZATION_INC1.json"
MODEL_LOCK_PATH = INC1_ROOT / "INC1_MODEL_LOCK.json"
SEED_AUDIT_PATH = INC1_ROOT / "INC1_SEED_AUDIT.json"
PLAN_DIR = INC1_ROOT / "results_plan_inc1"
PLAN_PATH = PLAN_DIR / "WORLD_PLAN.json"
RUNS_ROOT = INC1_ROOT / "runs"
#: the write-once allocation/budget log (appended, never overwritten).
BUDGET_LOG = INC1_ROOT / "INC1_BUDGET_LOG.jsonl"

# ------------------------------------------------------------- schemas/names
SEED_AUDIT_SCHEMA = "wm_n1_inc1_seed_audit/1"
PLAN_SCHEMA = "wm_n1_inc1_world_plan/1"
AUTHORIZATION_SCHEMA = "wm_n1_inc1_authorization/1"
MODEL_LOCK_SCHEMA = "wm_n1_inc1_model_lock/1"
RECEIPT_SCHEMA = "wm_n1_inc1_run_receipt/1"
BUDGET_SCHEMA = "wm_n1_inc1_budget/1"

# ------------------------------------------------------------- worlds/design
#: first-choice fresh seed block; the audit must clear it before use.
SEED_BLOCK_START = 9601
SEED_BLOCK_SIZE = 20
#: collision scan also treats the literal ``w<seed>`` token as a hit.
SEED_TOKEN_TEMPLATE = "w{seed}"

LOADINGS = (2400, 2600, 2700)
WORLDS_PER_LOADING = (7, 7, 6)
#: ADDENDUM_3 A3.1: penetrations restricted to {p10, p20, p40}; assigned
#: round-robin in seed order within each loading (2400: 3/2/2, 2600: 3/2/2,
#: 2700: 2/2/2; total p10 8 / p20 6 / p40 6).
PENETRATIONS = (0.10, 0.20, 0.40)

#: incident position classes (registration section 3).
INCIDENT_POSITIONS = ("A", "B")
INCIDENT_S_DOWN = {"A": 40.0, "B": -150.0}
#: merge start on the Part-A downstream coordinate (s_down = 0 at the merge
#: edge start, 2644.2 m on the merge-relative s_road axis).
MERGE_START_S = 2644.2
#: the incident lane: mainline right lane (lane_phys 1).
INCIDENT_LANE_PHYS = 1
#: edge-aware SUMO lane of the mainline right lane per segment (FORMAT.md 1).
INCIDENT_LANE_EDGE_LANE = {
    "entry": "1191554636_0",
    "merge": "1487606179_1",
    "exit": "1487606178_0",
}
#: the lane_phys-1 downstream origin (merge-edge right-lane segment start,
#: FORMAT.md 1): s_down = s - 2644.22.
MAIN_RIGHT_LANE_OFFSET_M = 2644.22

# ------------------------------------------------------------- sim/incident
T0_S = 300.0
ARM_HORIZON_S = 300.0
CONTROL_DT_S = 1.0
PHYSICS_DT_S = 0.5
#: fleet command speed mode (WM-46 / chunk-4), distinct from the CAV
#: baseline adapter's native31.  Mode 27 keeps SUMO's safety checks on
#: (car-following, right-of-way, junctions) while granting the command
#: authority; every R1/R2 wall command runs through CommandOwnership with
#: this mode.
CAV_SPEED_MODE = 27

OBSTACLE_MIN_GAP_M = 15.0
OBSTACLE_SEARCH_WINDOW_M = 20.0
OBSTACLE_HOLD_S = 300.0
OBSTACLE_LENGTH_M = 5.0
OBSTACLE_TYPE = "incident_hdv"

#: R1 nearest-CAV-stop.
R1_BRAKE_MPS2 = -2.0
R1_HOLD_S = 30.0
R1_RANGE_M = 500.0
#: R2 early-slow.
R2_TARGET_MPS = 15.0
R2_DECEL_MAX_MPS2 = 1.5
#: R2's commanded acceleration is capped at the CAV's native IDM acceleration
#: captured at t0 (``CavState.native_accel_mps2``).  When that bound is
#: unavailable the arm never issues positive acceleration (0.0): R2 is a
#: slow-down response and must never invent a stronger-than-native push.
R2_ACCEL_MAX_DEFAULT_MPS2 = 0.0
R2_DURATION_S = 120.0
R2_UPSTREAM_LO_M = 1000.0
R2_UPSTREAM_HI_M = 200.0
#: R4 wall-v2.2 assembly.
#: The registration says "the registered speed factor"; the only registered
#: J4 single-wall factors are 0.60 (W-lo) and 0.85 (W-mid).  R4 is a single
#: mid wall, so the mid factor is used.  Recorded in the plan as an
#: explicitly disclosed interpretation before any run.
R4_SPEED_FACTOR = 0.85
R4_REQUEST_WINDOW_S = 120.0
#: R4 is driven through the patched frozen v2.2 driver; the registered
#: v2.2 law + hook provenance are recorded here (see
#: REGISTRATION_WM_N1_INC1_ADDENDUM_1.md A1.2).  The pre-patch blocker
#: document is retained as historical evidence only.
R4_BLOCKER_DOC = "output/wd01/inc1/INC1_R4_BLOCKER.md"
R4_HOOK_PATCH = "patches/wall_driver_inc1_hook.diff"
R4_FREEZE_ADDENDUM = "WALL_V22_FREEZE_ADDENDUM_INC1.json"
#: R5/R6 shared scripted multi-CAV generators (ADDENDUM_2 A2.2/A2.3).  The
#: rule values themselves live in ONE place -- the ``inc1_gen`` package
#: (``INC1_GENERATORS.json`` gated); these are the plan-summary mirror only.
R5_DURATION_S = 120.0
R5_MIN_MEMBERS = 2
R6_DURATION_S = 180.0
R6_MIN_MEMBERS = 2
R6_MIN_OPENERS = 1
#: the gated shared generator manifest (A2.6): the launcher refuses unless the
#: live package matches the LF shas recorded there.
GENERATORS_MANIFEST_PATH = WD01_ROOT / "inc1_gen" / "INC1_GENERATORS.json"

ARM_NAMES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")
ARM_DESCRIPTIONS = {
    "R0": "do-nothing: whole fleet native",
    "R1": "nearest-CAV-stop: nearest upstream CAV brakes -2.0 m/s^2, "
          "holds 30 s, then native",
    "R2": "early-slow: CAVs with s_down in [obs-1000, obs-200) track "
          "15 m/s for 120 s, then native",
    "R3": "damping-upstream: upstream CAVs run the frozen CIRCLES damping",
    "R4": "wall-v2.2-upstream: two nearest qualifying upstream CAVs, "
          "frozen wall v2.2 driver",
    "R5": "coordinated upstream speed harmonisation (multi-CAV, "
          "ADDENDUM_2 A2.2): mainline CAVs with d in [100, 1500) track the "
          "staggered target for 120 s, then native",
    "R6": "coordinated lane-clearing and gap opening (multi-CAV, "
          "ADDENDUM_2 A2.3): adjacent-lane gap openers track a 3.0 s "
          "headway and blocked-lane approachers track min(12 m/s, native) "
          "for 180 s, then native",
}

# ------------------------------------------------------------- budget (sim-s)
BUDGET_WARMUP_S = 20 * 300.0            # 6,000
#: ADDENDUM_2 A2.5: 20 worlds x 7 arms x 300 s = 42,000 sim-s.
BUDGET_ARMS_S = 20 * 7 * 300.0          # 42,000
BUDGET_PARITY_S = 2 * 60.0              # 120
#: A1.5 harness-equivalence smoke: 2 warm-ups x 300 s + 2 states x 2 paths x
#: 300 s = 1,800 sim-s.
BUDGET_HARNESS_EQUIV_S = 2 * 300.0 + 2 * 2 * 300.0   # 1,800
BUDGET_TOTAL_S = 48_120.0 + 1_800.0     # 49,920 (A2.5)
BUDGET_ITEMS = {
    "warmup": BUDGET_WARMUP_S,
    "arms": BUDGET_ARMS_S,
    "parity_smoke": BUDGET_PARITY_S,
    "harness_equivalence_smoke": BUDGET_HARNESS_EQUIV_S,
    "total": BUDGET_TOTAL_S,
}
#: a crashed run may restart exactly once, with disclosure.
MAX_CRASHED_RESTARTS = 1
#: complete runs are never retried.
RETRY_COMPLETE = False

#: disk headroom the launcher requires on the runs drive (REVISION: the
#: world plan is small; physics logs are the bulk).  8 GB is a conservative
#: floor for 20 worlds x 5 arms x 300 s JSONL.
DISK_PREFLIGHT_MIN_FREE_GB = 8.0


def lf_sha256(path: str | Path) -> str:
    """sha256 over the text with CRLF normalised to LF (registration rule)."""
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def raw_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration_lf_sha() -> str:
    """The on-disk registration's LF-normalised sha256."""
    return lf_sha256(REGISTRATION_PATH)


def seed_token_pattern(seed: int) -> "re.Pattern[str]":
    """Standalone integer or ``w<seed>`` token; decimal floats excluded.

    ``(?<![0-9.])`` / ``(?![0-9.])`` keep a seed that is part of a decimal
    literal (e.g. the ``13.9618`` service delta in the ledger) from being
    counted as a seed identity use.
    """
    return re.compile(r"(?<![0-9.])w?%d(?![0-9.])" % int(seed))


def seed_block(start: int | None = None, count: int | None = None) -> list[int]:
    start = SEED_BLOCK_START if start is None else int(start)
    count = SEED_BLOCK_SIZE if count is None else int(count)
    return list(range(start, start + count))
