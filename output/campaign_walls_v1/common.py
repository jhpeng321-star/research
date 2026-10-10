"""WM-59 WALL PILOT (campaign_walls_v1): shared constants and helpers.

Owner-approved PLAN_DRAFT.md (approved 2026-09-28): a 60-world wall pilot
(20 worlds per p10/p20/p40 drawn from the chunk-2 + chunk-3 mode-27 bases,
splits in the stage-1 49:7:14 proportions -> 42/6/12) x 4 single-wall arms
= 240 restored 600-s branches = 144,000 simulated s.  Non-members NATIVE;
the world's EXISTING native sibling is reused as the reference (never
re-run).  W-2 (multi-wall) and p05 are DEFERRED (not implemented here).

This package edits nothing frozen: ``traffic_shepherd`` (the validated
WallGroupSession), ``campaign_stage1_v3`` and ``campaign_pilot_v1`` are
imported read-only.  ZERO SUMO here - the smoke and the launch are run
later by the integrator in a no-campaign window.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    for _root in (Path(__file__).resolve().parent.parent,      # <repo>/output
                  Path(__file__).resolve().parent.parent.parent):  # <repo>
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

from campaign_stage1_v3.common import (  # noqa: F401 - re-exported helpers
    archive_rename,
    load_json,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
    write_json_archiving,
)

WM_ID = "WM-59"
SCHEMA = "wm59_walls_pilot/1"

# --- the owner-approved pilot shape (PLAN_DRAFT.md, approved 2026-09-28) ---
PLAN_SEED = 20261101                 # the one stochastic step (replay test)
PENETRATIONS = (0.1, 0.2, 0.4)
WORLDS_PER_PENETRATION = 20
WORLDS_TOTAL = 60
ARMS = ("W-lo-early", "W-lo-late", "W-mid-early", "W-mid-late")
ARM_SPEED_FACTORS = {"W-lo": 0.60, "W-mid": 0.85}
ARM_TIMINGS = ("early", "late")
#: early = the first branch tick >= t0 with a feasible pair;
#: late = the first feasible tick >= t0 + 60.
TIMING_OFFSET_S = {"early": 0.0, "late": 60.0}
#: if no feasible pair appears within this long AFTER the target time the
#: branch is recorded ``not_feasible`` and continues native.
REQUEST_WINDOW_S = 120.0
#: wall speed = factor x the members' mean reported speed at request,
#: floored here (the floor is strictly above wall_group.SAFE_SPEED_MIN_MPS
#: = 2.0, so every request passes the dispatcher's safety validation).
SPEED_FLOOR_MPS = 2.5
#: fixed hold schedule drawn per branch from the branch seed (recorded),
#: counted from assembly completion.
HOLD_CHOICES_S = (60.0, 120.0, 240.0)
BRANCH_LENGTH_S = 600.0
BRANCHES_PER_WORLD = 4
BRANCHES_TOTAL = WORLDS_TOTAL * BRANCHES_PER_WORLD          # 240
SIMULATED_SECONDS_TOTAL = float(BRANCHES_TOTAL * BRANCH_LENGTH_S)  # 144,000
#: the 2-lane mainline edges a feasible pair must share (PLAN_DRAFT 1.4
#: scan; ONLINE checks read report frames only, never truth).
MAINLINE_2LANE_EDGES = frozenset({"1191554636", "1487606178"})
#: the pilot draws its 60 worlds in the stage-1 49:7:14 proportions.
SPLIT_COUNTS = {"TRAIN": 42, "CALIBRATION": 6, "HOLDOUT": 12}
#: per penetration block of 20 (largest remainder of 42/6/12 over 3 x 20)
SPLIT_COUNTS_PER_PEN = {"TRAIN": 14, "CALIBRATION": 2, "HOLDOUT": 4}

# --- launch staging (E:/ paths only; the integrator runs the smoke and
# --- the launch in a no-campaign window; nothing here writes them) ---
LAUNCH_RUNS_ROOT = "E:/research-dense/runs/campaign_walls_pilot_v1"
LAUNCH_OUT_DIR = "E:/research-dense/output/campaign_walls_pilot_v1_launch"
LAUNCH_AUTHORIZATION = ("E:/research-dense/output/campaign_walls_pilot_v1"
                        "/AUTHORIZATION.json")
SMOKE_RUNS_ROOT = "E:/research-dense/runs/campaign_walls_pilot_v1_smoke"
SMOKE_OUT_DIR = "E:/research-dense/output/campaign_walls_pilot_v1_smoke"
#: the SAME WM-41/WM-43 gate paths the stage-1 chunks pinned
WM41_VERDICT = ("E:/research-dense/output/campaign_pilot_v1_launch_r4"
                "/EXACTNESS_VERDICT.json")
WM41_REGISTRATION = "E:/rd-wt/wm34/output/campaign_pilot_v1/REGISTRATION.md"
WM43_SUMMARY = ("E:/research-dense/runs/active/wm40_wall_mechanics_v1_r7"
                "/LAUNCH_SUMMARY.json")
#: the wall pilot's inputs: the chunk-2 + chunk-3 plans and their bases
SOURCE_CHUNK2_PLAN = ("E:/research-dense/output/campaign_stage1_v2"
                      "/results_plan_chunk2/WORLD_PLAN.json")
SOURCE_CHUNK3_PLAN = ("E:/research-dense/output/campaign_stage1_v3"
                      "/results_plan_chunk3/WORLD_PLAN.json")
SOURCE_CHUNK2_BASES_ROOT = "E:/research-dense/runs/campaign_stage1_chunk2"
SOURCE_CHUNK3_BASES_ROOT = "E:/research-dense/runs/campaign_stage1_chunk3"

INTERPRETER = "C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe"

# --- pinned launch constants (owner-approved; never tuned on outcomes) ---
WALLS_WORKERS = 10
#: the pilot's preflight: 45 GB on the drive the runs root RESOLVES to
#: (PLAN_DRAFT section 2: ~40 GB pilot on ~120 GB free; the WM-53 85 GB
#: chunk preflight does not leave room beside the running campaign).
DISK_PREFLIGHT_MIN_FREE_GB = 45.0
CAV_SPEED_MODE = 27                  # WM-46, carried from every stage-1 arm
SMOKE_MAX_SIMULATED_S = 80.0
SMOKE_HORIZON_TICKS = 75             # <= 80 sim-s; assembly has ~75 s
SMOKE_ARM = "W-mid-early"
SMOKE_PENETRATION = 0.4
#: the whole wall arm still active (no release, no abort) at the horizon
#: end is a contract anomaly and FAILS the branch loudly - see
#: wall_driver.drive_wall_branch.
HORIZON_ACTIVE_WALL_ERROR = True

PACKAGE = Path(__file__).resolve().parent
PLAN_DIR_NAME = "results_plan"
PLAN_PATH = PACKAGE / PLAN_DIR_NAME / "WORLD_PLAN.json"
REGISTRATION_PATH = PACKAGE / PLAN_DIR_NAME / "WALLS_PILOT_REGISTRATION.md"
DIGEST_PATH = PACKAGE / PLAN_DIR_NAME / "WALLS_PILOT_REGISTRATION_DIGEST.json"
RECORD_PATH = PACKAGE / PLAN_DIR_NAME / "WALLS_PILOT_REGISTRATION_RECORD.json"


class WallsPilotError(ValueError):
    """Wall-pilot contract violation; never silently relaxed."""


def package_code_digest(directory: Path | None = None) -> str:
    """sha256 over the package's .py files (sorted by relative path).

    The smoke verdict is bound to this digest: any code edit invalidates
    the verdict and forces a re-smoke (the WM-43 launch-gate shape).
    """
    directory = Path(directory) if directory is not None else PACKAGE
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        if ".ruff_cache" in path.parts or "tests" in path.parts:
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def arm_speed_factor(arm: str) -> float:
    """0.60 for the W-lo arms, 0.85 for the W-mid arms."""
    prefix = "-".join(arm.split("-")[:2])
    if prefix not in ARM_SPEED_FACTORS:
        raise WallsPilotError(f"unknown wall arm {arm!r}")
    return ARM_SPEED_FACTORS[prefix]


def arm_timing(arm: str) -> str:
    """``early`` or ``late`` (the arm's trailing token)."""
    timing = arm.rsplit("-", 1)[-1]
    if timing not in ARM_TIMINGS:
        raise WallsPilotError(f"unknown wall arm {arm!r}")
    return timing


def wall_id_for(run_name: str) -> str:
    """The one wall session id of a wall branch."""
    return f"{run_name}:wall1"
