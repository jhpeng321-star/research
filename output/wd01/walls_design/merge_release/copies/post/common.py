"""WM-59 WALL PILOT (campaign_walls_v1): shared constants and helpers.

Owner-approved PLAN_DRAFT.md (approved 2026-09-28), RE-TARGETED to chunk 4
(WM-59 job J4, 2026-09-30): the pilot uses ONLY the chunk-4 mode-27 worlds
at p10/p20/p40 that are NOT HOLDOUT (44 worlds: TRAIN 39 = 12/13/14 and
CALIBRATION 5 = 2/2/1; the chunk-4 split labels are kept; there is NO wall
HOLDOUT) x 4 single-wall arms = 176 restored 600-s branches = 105,600
simulated s.  Reduced from the approved 60 because chunk 4 has only 44
non-HOLDOUT p10/p20/p40 worlds.  Non-members NATIVE; the reference is the
chunk-4 ``<world>__native`` run (a TRUE constant fleet-native restored at
t0 from the SAME saved state; never re-run, outcomes never read).  W-2
(multi-wall) and p05 are DEFERRED (not implemented here).

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

# --- the pilot shape (PLAN_DRAFT.md, re-targeted to chunk 4, 2026-09-30) ---
PLAN_SEED = 20261101                 # seeds the per-branch seeds / hold draws
PENETRATIONS = (0.1, 0.2, 0.4)
#: the chunk-4 non-HOLDOUT worlds per penetration (VERIFIED against the
#: chunk-4 plan at build time; any difference fails loudly)
WORLDS_BY_PENETRATION = {0.1: 14, 0.2: 15, 0.4: 15}
SPLIT_COUNTS_BY_PENETRATION = {
    0.1: {"TRAIN": 12, "CALIBRATION": 2},
    0.2: {"TRAIN": 13, "CALIBRATION": 2},
    0.4: {"TRAIN": 14, "CALIBRATION": 1},
}
WORLDS_TOTAL = 44
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
BRANCHES_TOTAL = WORLDS_TOTAL * BRANCHES_PER_WORLD          # 176
SIMULATED_SECONDS_TOTAL = float(BRANCHES_TOTAL * BRANCH_LENGTH_S)  # 105,600
#: the 2-lane mainline edges of PLAN_DRAFT 1.4 (kept for other uses; the
#: member choice no longer uses the exit edge - see MEMBER_EDGES).
MAINLINE_2LANE_EDGES = frozenset({"1191554636", "1487606178"})
#: OPTION A (WM package owner, 2026-09-30): wall MEMBERS are chosen on the
#: upstream mainline ENTRY edge only (2 lanes, no incoming edges).  40/78
#: scanned requests were on the exit edge under the PLAN_DRAFT rule.
MEMBER_EDGES = frozenset({"1191554636"})
#: request feasibility span cap (member-pair longitudinal span).  A2
#: REVERT (owner/main decision 2026-09-30): back to the 75 m
#: wall_group.MAX_WALL_SPAN_M cap.  The 2026-09-28 tightening to the
#: dispatcher's 6 m assembly tolerance existed ONLY because the old
#: identical-pacing law could never close a wide gap; the v2.1-closure
#: law (rear members pace at v_wall + min(2, 0.1*gap)) assembles wide
#: pairs (WM-43b: 40-60 m in 28-35 s), so the request may span them
#: again.  A1 (entry edge only), A3 (minimal-span choice) and A4
#: (road-left rule + road-end release) are unchanged.  Pinned equal to
#: wall_group.MAX_WALL_SPAN_M by a test.
REQUEST_SPAN_CAP_M = 75.0
#: road edge lengths from the preserved net (a test re-reads them with
#: sumolib): entry 1191554636, merge 1487606179, exit 1487606178.
UPSTREAM_EDGE_ID = "1191554636"
UPSTREAM_EDGE_LENGTH_M = 2644.2
DOWNSTREAM_ROAD_M = 75.4 + 477.9            # 1487606179 + 1487606178
#: a request needs road_left(front member) >= v_wall x this many seconds
ROAD_LEFT_REQUEST_S = 80.0
#: SUPERSEDED by law v2.2 (kept only so the old road-end outcome constant
#: still imports; the driver no longer issues a road-end release).
ROAD_END_RELEASE_S = 10.0

# --- law v2.2 'merge-release' geometry (owner-confirmed 2026-10-01) --------
#: cumulative road coordinate s_road runs from the mainline ENTRY edge origin
#: (0 m) to the network end; merge START is the entry edge length.
ENTRY_EDGE_ID = "1191554636"
MERGE_EDGE_ID = "1487606179"
EXIT_EDGE_ID = "1487606178"
ENTRY_EDGE_LENGTH_M = 2644.2
MERGE_EDGE_LENGTH_M = 75.4
EXIT_EDGE_LENGTH_M = 477.9
MERGE_START_S = ENTRY_EDGE_LENGTH_M                    # 2644.2 m
MERGE_END_S = MERGE_START_S + MERGE_EDGE_LENGTH_M      # 2719.6 m
NETWORK_END_S = MERGE_END_S + EXIT_EDGE_LENGTH_M       # 3197.5 m
#: owner binding: release at X = 0 m BEFORE the merge edge start (exactly at
#: the merge start).  The design note derives the general formula; the owner
#: override is binding over any derived positive X.
MERGE_RELEASE_X_M = 0.0
MERGE_RELEASE_POINT_S = MERGE_START_S - MERGE_RELEASE_X_M
#: owner binding: D_min = X + distance-to-assemble + a minimal meaningful
#: hold.  The minimal meaningful fixed-arm hold is the smallest registered
#: draw; the assembly distance uses the J4 MEDIAN assembly time (10 s of
#: mechanics, no outcome) at the request's wall speed.
MIN_MEANINGFUL_HOLD_S = 60.0
ASSEMBLY_TIME_ASSUMED_S = 10.0
#: the smoke restores this p40 TRAIN world when it is in the plan: the
#: TRAIN-only scan (WALL_FIX_OPTIONS.md, fe6724f) found a 1.5 m pair at the
#: first tick with 3076 m of road left (needs 1269 m at W-mid).  Falls back
#: to the first sorted p40 TRAIN world if the pin is not in the plan.
SMOKE_WORLD_PIN = "w8697m2700p40t240"
#: the WIDE-PAIR smoke mode (declared 2026-09-30, J4 prep): the smoke
#: request waits for a pair whose span lands in
#: [SMOKE_WIDE_PAIR_MIN_SPAN_M, REQUEST_SPAN_CAP_M] instead of taking the
#: minimal-span pair - a declared end-to-end check that the v2.1-closure
#: law assembles WIDE pairs in the real wiring (WM-43b evidence) and that
#: the per-tick member lane logging is on disk.  Same 75-tick (<= 80
#: sim-s) horizon and TRAIN-only smoke world selection as the normal
#: smoke; smoke-only (the driver refuses the window on real branches).
SMOKE_WIDE_PAIR_MIN_SPAN_M = 40.0
#: the walls pilot has TRAIN + CALIBRATION worlds only (NO wall HOLDOUT)
SPLIT_COUNTS = {"TRAIN": 39, "CALIBRATION": 5, "HOLDOUT": 0}
#: splits a wall world may carry; a HOLDOUT world is REFUSED everywhere
ALLOWED_SPLITS = ("TRAIN", "CALIBRATION")
CHUNK = "chunk4"

# --- launch staging (E:/ paths only; the integrator runs the smoke and
# --- the launch in a no-campaign window; nothing here writes them) ---
RD_ROOT = "E:/research-dense"
LAUNCH_RUNS_ROOT = f"{RD_ROOT}/runs/campaign_walls_pilot_chunk4"
LAUNCH_OUT_DIR = f"{RD_ROOT}/output/campaign_walls_pilot_chunk4_launch"
LAUNCH_AUTHORIZATION = (f"{RD_ROOT}/output/campaign_walls_pilot_chunk4"
                        "/AUTHORIZATION.json")
SMOKE_RUNS_ROOT = f"{RD_ROOT}/runs/campaign_walls_pilot_chunk4_smoke"
SMOKE_OUT_DIR = f"{RD_ROOT}/output/campaign_walls_pilot_chunk4_smoke"
#: the shared per-run claim dir of the rented-box launcher instances
CLAIM_DIR = f"{LAUNCH_RUNS_ROOT}/CLAIMS"
#: the SAME WM-41/WM-43 gate paths the stage-1 chunks pinned
WM41_VERDICT = (f"{RD_ROOT}/output/campaign_pilot_v1_launch_r4"
                "/EXACTNESS_VERDICT.json")
WM41_REGISTRATION = "E:/rd-wt/wm34/output/campaign_pilot_v1/REGISTRATION.md"
WM43_SUMMARY = (f"{RD_ROOT}/runs/active/wm40_wall_mechanics_v1_r7"
                "/LAUNCH_SUMMARY.json")
#: the wall pilot's input: the chunk-4 plan and its bases/natives root
SOURCE_CHUNK4_PLAN = (f"{RD_ROOT}/output/campaign_stage1_v3"
                      "/results_plan_chunk4/WORLD_PLAN.json")
SOURCE_CHUNK4_BASES_ROOT = f"{RD_ROOT}/runs/campaign_stage1_chunk4"
SOURCE_PLAN_SCHEMA = "wm58_chunk4_world_plan/1"

INTERPRETER = "C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe"

# --- pinned launch constants (owner-approved; never tuned on outcomes) ---
WALLS_WORKERS = 10
#: solo launches use exactly WALLS_WORKERS; a claim-dir launch (rented-box
#: instances) may use 1..WALLS_WORKERS (the campaign launcher's per-launch
#: cap) because several instances share the claim dir.
WALLS_WORKERS_MAX = 10
#: the pilot's preflight, on the drive the runs root RESOLVES to.  Measured
#: chunk-4 branch dirs (os.stat only, 2026-09-30, `disk_estimate.py`): mean
#: 52.8/66.9/93.5 MB at p10/p20/p40 (max 72.3/96.3/139.8 MB) -> 44 worlds x
#: 4 arms = 12.6 GB mean, 18.2 GB worst case; 25 GB = worst case + ~37 %.
DISK_PREFLIGHT_MIN_FREE_GB = 25.0
#: the declared disk estimate (see disk_estimate.py)
DISK_ESTIMATE_GB = {"mean": 12.6, "worst_case": 18.2}
CAV_SPEED_MODE = 27                  # WM-46, carried from every stage-1 arm
SMOKE_MAX_SIMULATED_S = 80.0
SMOKE_HORIZON_TICKS = 75             # <= 80 sim-s; assembly has ~75 s
SMOKE_ARM = "W-mid-early"
SMOKE_PENETRATION = 0.4
#: the smoke restores a TRAIN world (never a CALIBRATION world)
SMOKE_SPLIT = "TRAIN"
#: SMOKE-ONLY hold (s): the smoke must exercise request -> assembly -> hold
#: -> planner release -> native handback inside its 75-tick horizon, so the
#: smoke branch replaces its drawn H with this.  Registered pilot arms keep
#: their H draws (the override is a CLI flag only the smoke passes).
SMOKE_HOLD_OVERRIDE_S = 15.0
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


#: the smoke-path-only hold override: a plan or argv of a REAL launch that
#: carries anything named like this is refused (launch_walls, run_box_walls)
SMOKE_ONLY_MARKER = "hold-override"


def find_smoke_only_keys(obj, path: str = "") -> list[str]:
    """Paths of every key in a JSON-like ``obj`` naming the smoke-only hold
    override (``hold_override_s``, ``smoke_hold_override_s``, ...)."""
    found: list[str] = []
    marker = SMOKE_ONLY_MARKER.replace("-", "_")
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}/{key}"
            if marker in str(key).replace("-", "_"):
                found.append(here)
            found.extend(find_smoke_only_keys(value, here))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(find_smoke_only_keys(value, f"{path}[{index}]"))
    return found


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


#: the code the branch physics/control actually run through (WM review R1)
DEPENDENCY_TREES = ("output/campaign_stage1_v3", "output/campaign_pilot_v1",
                    "traffic_shepherd")


def dependency_digest(repo_root: Path | None = None) -> str:
    """sha256 over the sorted (relative path, file sha256) of every ``*.py``
    under output/campaign_stage1_v3, output/campaign_pilot_v1 (both without
    ``tests/``) and traffic_shepherd/ (all EXCEPT ``traffic_shepherd/planner/``,
    which a separate running experiment edits), excluding ``__pycache__``.
    The smoke verdict records it and the launch refuses on a live mismatch:
    the package digest alone does not cover the dependencies that carry the
    physics and control."""
    repo_root = (Path(repo_root) if repo_root is not None
                 else PACKAGE.parent.parent)
    entries = []
    for tree in DEPENDENCY_TREES:
        base = repo_root / tree
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts or ".ruff_cache" in path.parts:
                continue
            if tree == "traffic_shepherd" and "planner" in path.relative_to(
                    base).parts[:1]:
                continue
            if tree.startswith("output/") and "tests" in path.relative_to(
                    base).parts:
                continue
            entries.append((path.relative_to(repo_root).as_posix(),
                            hashlib.sha256(path.read_bytes()).hexdigest()))
    digest = hashlib.sha256()
    for rel, file_sha in sorted(entries):
        digest.update((rel + "|" + file_sha + chr(10)).encode("utf-8"))
    return digest.hexdigest()


#: the frozen wall law this package binds (J4 prep, 2026-09-30): the
#: registration record pins module path, WALL_LAW_VERSION and the
#: LF-normalized sha256 of the file; the registration ``check`` and a
#: dedicated test re-verify it live, so any wall_group.py drift fails
#: loudly instead of silently changing the law the pilot runs.
WALL_LAW_MODULE = "traffic_shepherd/wall_group.py"


def wall_law_pin(repo_root: Path | None = None) -> dict:
    """The pinned identity of ``traffic_shepherd/wall_group.py``: the
    imported ``WALL_LAW_VERSION`` and the sha256 over the LF-normalized
    file bytes (read-only)."""
    from traffic_shepherd import wall_group  # noqa: PLC0415 - frozen lib

    repo_root = (Path(repo_root) if repo_root is not None
                 else PACKAGE.parent.parent)
    data = (repo_root / WALL_LAW_MODULE).read_bytes()
    return {
        "module": WALL_LAW_MODULE,
        "version": wall_group.WALL_LAW_VERSION,
        "sha256": hashlib.sha256(
            data.replace(b"\r\n", b"\n")).hexdigest(),
    }


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


#: an already-loaded holdout_exclusions dict; None = the guard reads its own
#: protocols/holdout_exclusions.json.  A hook for tests only.
HOLDOUT_EXCLUSIONS_DATA: dict | None = None

HOLDOUT_GUARD_CANDIDATES = (
    PACKAGE.parent.parent / "tools" / "holdout_guard.py",
    Path(RD_ROOT) / "tools" / "holdout_guard.py",
)


def load_holdout_guard(path: str | Path | None = None):
    """Import ``tools/holdout_guard.py`` (READ-ONLY, by file path) - the
    defence-in-depth exclusion list of exposed HOLDOUT roots.  The first
    existing candidate wins (this repo's tools/, then E:/research-dense)."""
    import importlib.util

    candidates = [Path(path)] if path is not None else list(
        HOLDOUT_GUARD_CANDIDATES)
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "wm59_holdout_guard", str(candidate))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise WallsPilotError(
        "tools/holdout_guard.py not found (candidates: "
        f"{[str(c) for c in candidates]}); the wall pilot refuses to "
        "build without the HOLDOUT guard")


def refuse_holdout_and_excluded(worlds, *, exclusions=None, guard=None) -> None:
    """Defence in depth: a HOLDOUT world (split label) or any root on the
    holdout_guard exclusion list is REFUSED - never silently dropped.
    ``worlds`` are dicts with ``world`` and ``split``; ``exclusions`` is an
    optional already-loaded holdout_exclusions dict (tests inject one)."""
    bad_split = sorted(str(w["world"]) for w in worlds
                       if w.get("split") not in ALLOWED_SPLITS)
    if bad_split:
        raise WallsPilotError(
            f"wall pilot refuses non-TRAIN/CALIBRATION worlds (HOLDOUT "
            f"stays sealed; there is no wall HOLDOUT): {bad_split[:5]}")
    guard = guard if guard is not None else load_holdout_guard()
    if exclusions is None:
        exclusions = HOLDOUT_EXCLUSIONS_DATA   # test hook; None = real file
    try:
        guard.assert_final_holdout_eligible(
            [str(w["world"]) for w in worlds], exclusions)
    except ValueError as exc:      # the guard's HoldoutExclusionError
        raise WallsPilotError(str(exc)) from exc
