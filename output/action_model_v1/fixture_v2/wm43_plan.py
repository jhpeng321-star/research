"""WM-43B fixture plan constants and shape validation (stdlib only).

DRAFT (not authorized, never launched): re-validation of the WM-40 wall
mechanics fixture (WM-43 r7) under the wall-v2.1 assembly CLOSURE law
(owner decision 2026-09-30, "Option B": front holds v_wall, rear members
target v_wall + min(2.0, 0.1 * gap_to_front) until span <= 6 m; new
`assembly_timeout` safety abort at 60 s).  5 arms (A1-A3 unchanged specs,
A4 wide-pair closure, A5 assembly timeout) x 5 seeds x 600 s simulated,
physical-path integrity evidence only.

Arm schedules (wall speed is strictly above the wall_group safety floor
2.0 m/s; release terms are planner `release_at_s` absolute episode times):

- A1 planner wall lifecycle: assemble -> hold -> release at t=90 s.
- A2 wall + damping non-member: same lifecycle, release at t=300 s; the
  third CAV is damping-committed from t=0 (rear-end surveillance is
  expected NOT to abort at a safe gap).
- A4 wide-pair closure (NEW): same lanes as A1 but the members start
  40-60 m apart; closure must assemble the wall (span <= 6 m) within 60 s,
  hold, then release at t=150 s.
- A5 assembly timeout (NEW): wide pair, a scripted actuator pins the front
  member at 10.5 m/s (above the rear's maximum closure target
  v_wall + 2.0 = 10.0 m/s and outside the 1 m/s speed tolerance of v_wall)
  from assembly start so closure/assembly is impossible; the
  declared `assembly_timeout` abort must fire 60 +/- 1 s after
  assembly_start with a same-tick native handback.
- A3 induced safety abort: wall holds; from t=90 s a scripted actuator
  pins the front member to 25 m/s so the member span grows ~17 m/tick and
  the declared `member_span_exceeded` abort (ABORT_SPAN_M) fires (~t=100,
  always < 200 s), handing members back to native in the same tick.  The
  actuator is a fault injector, NOT a controller write: every control
  command still flows through CommandOwnership (single writer), and every
  injection is logged as a `scripted_actuator` telemetry row.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PLAN_SCHEMA = "wm43_fixture_plan/1"
RECEIPT_SCHEMA = "wm43_run_receipt/1"
CHECKS_SCHEMA = "wm43_run_checks/1"
PREFLIGHT_SCHEMA = "wm43_preflight/1"
SUMMARY_SCHEMA = "wm43_launch_summary/1"
AUDIT_SCHEMA = "wm43_exposure_audit/1"
ALLOCATION_SCHEMA = "wm43_allocation_row/1"

FIXTURE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURE_DIR.parents[2]
PLAN_PATH = FIXTURE_DIR / "PLAN.json"
REGISTRATION_PATH = FIXTURE_DIR / "REGISTRATION_WM43B_DRAFT.md"
DIGEST_PATH = FIXTURE_DIR / "REGISTRATION_DIGEST.json"
AUTHORIZATION_PATH = FIXTURE_DIR / "AUTHORIZATION.json"
SEEDS_DIFF_PATH = FIXTURE_DIR / "seeds_py_wm43b.diff"
EXPOSURE_AUDIT_PATH = FIXTURE_DIR / "EXPOSURE_AUDIT_WM43B.json"
SEEDS_MODULE_RELATIVE = "traffic_shepherd/seeds.py"

ARMS = ("A1", "A2", "A3", "A4", "A5")
#: FRESH block proposal (next after WM43_WALL_MECHANICS_SEEDS 920244-920248);
#: 5 seeds per arm, one identity per run.  Requires the exposure audit
#: (EXPOSURE_AUDIT_INSTRUCTIONS.md) and the seeds diff before any launch.
SEEDS_BY_ARM = {
    "A1": (920249, 920250, 920251, 920252, 920253),
    "A2": (920254, 920255, 920256, 920257, 920258),
    "A3": (920259, 920260, 920261, 920262, 920263),
    "A4": (920264, 920265, 920266, 920267, 920268),
    "A5": (920269, 920270, 920271, 920272, 920273),
}
SEEDS = tuple(seed for arm in ARMS for seed in SEEDS_BY_ARM[arm])
WALL_LAW_VERSION = "wall-v2.1-closure"
ASSEMBLY_TIMEOUT_S = 60.0          # == wall_group.ASSEMBLY_TIMEOUT_S
ASSEMBLY_TIMEOUT_TOL_S = 1.0
ASSEMBLY_SPAN_TOL_M = 6.0          # == wall_group.ASSEMBLY_SPAN_TOL_M
CLOSE_MAX_DELTA_MPS = 2.0          # == wall_group.ASSEMBLY_CLOSE_MAX_DELTA_MPS
CLOSE_GAIN_PER_S = 0.1             # == wall_group.ASSEMBLY_CLOSE_GAIN_PER_S
LOADING = 2000  # registry key only: the micro-roster generates no demand streams
PENETRATION = 1.0  # the fixture roster is entirely designated CAVs
ROLE = "exposed"  # WM-41 pilot convention: fresh audited block, "exposed" role
HORIZON_TICKS = 600
CONTROL_DT_S = 1.0
PHYSICS_DT_S = 0.5
WALL_SPEED_MPS = 8.0
RUNS_TOTAL = 25
SIMULATED_SECONDS_TOTAL = 15000
WORKERS_MAX = 6
RETRIES = 0
WALL_CLOCK_BUDGET_S = 3600  # requested authorization bound; micro-roster expected faster
RUNS_ROOT_RELATIVE = "runs/active/wm43b_wall_closure_v1"
#: Standing PRE-LAUNCH SMOKE (owner decision 2026-09-27): ONE run per
#: package, capped just past its critical point (<= 80 simulated s), into a
#: separate smoke runs root, never used as data, zero retries.  For this
#: fixture (v2): the A4 (wide-pair closure) first-seed run capped at t=70 s - scenario/vType load,
#: dispatcher wall assembly start and telemetry rows all occur inside the
#: cap.  The full launch refuses unless a write-once SMOKE_VERDICT.json
#: (passed=true, bound to the registration sha256 AND the package code
#: digest) exists and is current.
SMOKE_ARM = "A4"
SMOKE_SEED = SEEDS_BY_ARM["A4"][0]
SMOKE_HORIZON_TICKS = 70
SMOKE_MAX_HORIZON_TICKS = 80
SMOKE_RUNS_ROOT_SUFFIX = "_smoke"
SMOKE_CHECKS_SCHEMA = "wm43_smoke_checks/1"
SMOKE_VERDICT_SCHEMA = "wm43_smoke_verdict/1"
SMOKE_VERDICT_NAME = "SMOKE_VERDICT.json"

#: the smoke check set (wm43_checks.evaluate_smoke) - the capped run must
#: complete, stay clean, keep the single-writer path and reach the assembly
#: critical point inside the cap
SMOKE_CHECK_IDS = (
    "S01_smoke_horizon_complete",
    "S02_smoke_zero_collisions",
    "S03_smoke_zero_teleports",
    "S04_smoke_single_writer_path",
    "S05_smoke_assembly_started",
    "S06_smoke_assembled_by_cap",
)
INTERPRETER = "C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe"
SEEDS_DERIVATION_PREFIX = "wm43b_wall_closure_v1"

# Per-arm roster and wall schedule (see module docstring for the A3 actuator).
ARM_SPECS = {
    "A1": {
        "wall_id": "wall_a1",
        "members": ("wall.0", "wall.1"),       # lane 0, lane 1 (ordered by lane index)
        "extra_cavs": (),
        "assignment_extra": {},                 # non-member assignments (e.g. damping)
        "release_at_s": 90.0,
        "injection": None,
    },
    "A2": {
        "wall_id": "wall_a2",
        "members": ("wall.0", "wall.1"),
        "extra_cavs": ("follow.0",),
        "assignment_extra": {"follow.0": "damping"},
        "release_at_s": 300.0,
        "injection": None,
    },
    "A3": {
        "wall_id": "wall_a3",
        "members": ("wall.0", "wall.1"),
        "extra_cavs": (),
        "assignment_extra": {},
        "release_at_s": 600.0,                  # outlives the expected abort (never reached)
        "injection": {"start_s": 90.0, "target_mps": 25.0, "member": "wall.1"},
    },
    # NEW in v2: the wide-pair arms.  wall.1 is always the FRONT member.
    "A4": {
        "wall_id": "wall_a4",
        "members": ("wall.0", "wall.1"),
        "extra_cavs": (),
        "assignment_extra": {},
        "release_at_s": 150.0,
        "injection": None,
        "wide_offset_m": (40.0, 60.0),
    },
    "A5": {
        "wall_id": "wall_a5",
        "members": ("wall.0", "wall.1"),
        "extra_cavs": (),
        "assignment_extra": {},
        "release_at_s": 600.0,                  # never reached (timeout abort)
        # phase "assembly": the pin starts once assembly has started (v1's
        # A3 pin waits for the assembled event)
        "injection": {"start_s": 0.0, "target_mps": 10.5, "member": "wall.1",
                      "phase": "assembly"},
        "wide_offset_m": (40.0, 60.0),
    },
}

#: Machine-evaluated per-run check ids (implemented in wm43_checks.py,
#: specified in REGISTRATION.md).  A1_* checks are shared verbatim by A2/A3.
COMMON_CHECKS = (
    "C01_run_complete",
    "C02_zero_collisions",
    "C03_zero_teleports",
    "C04_single_writer_path",
    "C05_speed_mode_31_contract",
    "C06_roster_and_wall_identity",
    "C07_law_version_identity",
)
CHECK_LISTS = {
    "A1": COMMON_CHECKS + (
        "A1_1_assembly_start",
        "A1_2_assembled_event",
        "A1_3_hold_speed_tolerance",
        "A1_4_released_event",
        "A1_5_native_handback_receipts",
        "A1_6_no_wall_after_release_no_abort",
        "A1_7_wall_pacing_identity",
    ),
    "A2": COMMON_CHECKS + (
        "A1_1_assembly_start",
        "A1_2_assembled_event",
        "A1_3_hold_speed_tolerance",
        "A1_7_wall_pacing_identity",
        "A2_1_no_abort_release_at_300",
        "A2_2_follower_damping_path",
        "A2_3_surveillance_no_region_entry",
        "A2_4_cross_path_single_writer",
        "A2_5_handback_receipts_at_release",
    ),
    "A3": COMMON_CHECKS + (
        "A1_1_assembly_start",
        "A1_2_assembled_event",
        "A1_3_hold_speed_tolerance",
        "A1_7_wall_pacing_identity",
        "A3_1_injections_logged",
        "A3_2_abort_reason_span",
        "A3_3_same_tick_handback",
        "A3_4_injections_cease",
        "A3_5_hold_until_injection",
    ),
    "A4": COMMON_CHECKS + (
        "A1_1_assembly_start",
        "A4_1_assembled_within_60s",
        "A4_2_span_at_assembled",
        "A4_3_front_paced_every_assembly_tick",
        "A4_4_rear_closure_identity",
        "A1_3_hold_speed_tolerance",
        "A1_4_released_event",
        "A1_5_native_handback_receipts",
        "A1_6_no_wall_after_release_no_abort",
        "A1_7_wall_pacing_identity",
        "A4_5_no_rear_end_region_entry",
    ),
    "A5": COMMON_CHECKS + (
        "A1_1_assembly_start",
        "A1_7_wall_pacing_identity",
        "A5_1_injections_logged",
        "A5_2_abort_reason_assembly_timeout",
        "A5_3_same_tick_handback",
        "A5_4_injections_cease",
        "A5_5_never_assembled",
    ),
}
#: Declared bounds used by the checks (assembly must complete early; the A3
#: abort must land well before the release term; hold speeds within the
#: wall_group assembly tolerance).
ASSEMBLY_DEADLINE_S = 30.0
A3_ABORT_LATEST_S = 200.0
HOLD_SPEED_TOL_MPS = 1.0  # == wall_group.ASSEMBLY_SPEED_TOL_MPS
MIN_HOLD_SAMPLES = 20

_SAFE_NAME = re.compile(r"^[a-z0-9_]+$")


def run_name(arm: str, seed: int) -> str:
    return f"{arm.lower()}_seed{seed}"


def check_safe_name(name: str) -> bool:
    return bool(_SAFE_NAME.match(name))


def expected_runs() -> list[dict]:
    """The canonical 25-run plan body (5 arms x 5 seeds)."""
    runs = []
    for arm in ARMS:
        spec = ARM_SPECS[arm]
        for seed in SEEDS_BY_ARM[arm]:
            runs.append({
                "run": run_name(arm, seed),
                "arm": arm,
                "seed": seed,
                "loading": LOADING,
                "penetration": PENETRATION,
                "role": ROLE,
                "duration_s": HORIZON_TICKS,
                "wall_id": spec["wall_id"],
                "members": list(spec["members"]),
                "extra_cavs": list(spec["extra_cavs"]),
                "wall_speed_mps": WALL_SPEED_MPS,
                "release_at_s": spec["release_at_s"],
                "injection": dict(spec["injection"]) if spec["injection"] else None,
                "checks": list(CHECK_LISTS[arm]),
            })
    return runs


def check_plan_shape(plan: dict) -> list[str]:
    """Structural + budget validation; returns a list of problems."""
    problems = []
    runs = plan.get("runs", [])
    constants = plan.get("constants", {})
    if len(runs) != RUNS_TOTAL:
        problems.append(f"plan carries {len(runs)} runs, expected {RUNS_TOTAL}")
    names = [r.get("run") for r in runs]
    if len(set(names)) != len(names) or any(not check_safe_name(str(n)) for n in names):
        problems.append("run names must be unique and match [a-z0-9_]+")
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS_BY_ARM[arm]}
    seen = {(r.get("arm"), r.get("seed")) for r in runs}
    if seen != expected or len(runs) != len(expected):
        problems.append("runs must be exactly the 5 arms x 5 registered seeds")
    total_s = 0
    for run in runs:
        if run.get("duration_s") != HORIZON_TICKS:
            problems.append(f"{run.get('run')}: duration_s must be {HORIZON_TICKS}")
        if run.get("loading") != LOADING or run.get("role") != ROLE \
                or abs(float(run.get("penetration", -1)) - PENETRATION) > 1e-12:
            problems.append(f"{run.get('run')}: loading/role/penetration differ "
                            "from the registered fixture constants")
        if run.get("wall_speed_mps") != WALL_SPEED_MPS:
            problems.append(f"{run.get('run')}: wall_speed_mps must be {WALL_SPEED_MPS}")
        if sorted(run.get("checks", [])) != sorted(CHECK_LISTS.get(run.get("arm"), ())):
            problems.append(f"{run.get('run')}: check list differs from the "
                            f"registered {run.get('arm')} check list")
        spec = ARM_SPECS.get(run.get("arm"), {})
        expected_injection = dict(spec["injection"]) if spec.get("injection") else None
        if run.get("injection") != expected_injection:
            problems.append(f"{run.get('run')}: injection schedule differs from "
                            "the registered arm spec")
        total_s += int(run.get("duration_s", 0))
    if total_s != SIMULATED_SECONDS_TOTAL:
        problems.append(f"simulated seconds {total_s} != {SIMULATED_SECONDS_TOTAL}")
    if constants.get("runs_root_relative") != RUNS_ROOT_RELATIVE:
        problems.append("constants.runs_root_relative differs from the registered root")
    if constants.get("workers_max") != WORKERS_MAX or constants.get("retries") != RETRIES:
        problems.append("constants.workers_max/retries differ from the registered budget")
    if constants.get("interpreter") != INTERPRETER:
        problems.append("constants.interpreter must pin the rl_driving CPython")
    if constants.get("wall_clock_budget_s") != WALL_CLOCK_BUDGET_S:
        problems.append("constants.wall_clock_budget_s differs from the requested bound")
    smoke = plan.get("smoke")
    if smoke is None:
        problems.append("plan carries no registered pre-launch smoke block")
    else:
        if smoke.get("arm") != SMOKE_ARM or smoke.get("seed") != SMOKE_SEED:
            problems.append("smoke.arm/seed differ from the registered smoke run")
        horizon = smoke.get("horizon_ticks")
        if (isinstance(horizon, bool) or not isinstance(horizon, int)
                or not 1 <= horizon <= SMOKE_MAX_HORIZON_TICKS):
            problems.append(f"smoke.horizon_ticks must be an int in "
                            f"[1, {SMOKE_MAX_HORIZON_TICKS}]")
        elif horizon != SMOKE_HORIZON_TICKS:
            problems.append(f"smoke.horizon_ticks must be {SMOKE_HORIZON_TICKS} "
                            "(just past the assembly critical point)")
        if smoke.get("runs_root_relative") != RUNS_ROOT_RELATIVE + SMOKE_RUNS_ROOT_SUFFIX:
            problems.append("smoke.runs_root_relative must be the registered runs "
                            "root plus the _smoke suffix")
        if smoke.get("simulated_seconds") != horizon:
            problems.append("smoke.simulated_seconds must equal smoke.horizon_ticks")
        if smoke.get("used_as_data") is not False:
            problems.append("smoke.used_as_data must be false")
    return problems


def load_plan(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else PLAN_PATH
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema must be {PLAN_SCHEMA}")
    return plan
