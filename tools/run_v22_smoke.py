"""WALL22-R: the declared wall-law v2.2 mechanics smoke driver.

Runs EXACTLY the three runs declared in ``SMOKE_V22.md`` /
``SMOKE_V22.json`` (repo root) - the owner-approved 3 x 300 sim-s
mechanics smoke of wall law v2.2 'merge-release' - SEQUENTIALLY, one
fresh SUMO process per run inside the child, and evaluates MC1-MC5
field-by-field from each run's wall events / branch record / receipt,
writing the write-once ``SMOKE_V22_VERDICT.json``.

J4BFIX (2026-10-01): the driver also accepts the R2 RE-smoke fixture
``output/campaign_walls_v1/SMOKE_V22_R2.json`` (schema
``wm59_walls_v22_smoke_r2_fixture/1``; frozen, LF ``c5df40b0...``) -
the owner's post-hoc, disclosed amendment after the R1 failure on a
registered safety abort ('fix the test, approve J4b').  The law is
UNCHANGED (e7fd3536); only the smoke pass rule changes:

- RC1/RC2/RC3/RC4/RC4b/RC6 per run, RC5 across the runs (the fixture's
  ``checks`` are implemented field-by-field in ``evaluate_rc_checks`` /
  ``evaluate_rc5``);
- a REGISTERED v2.2 safety abort (any frozen reason except
  ``member_left_network``, stripped of any ``:<vehicle>`` suffix) is
  RECORDED in the verdict's diagnosis table and fails NOTHING;
  ``member_left_network`` and any reason OUTSIDE the frozen list fail
  the run (RC2);
- NO stop-on-fail: ALL THREE declared runs always execute, failures are
  reported per run;
- the write-once runs root defaults to ``runs/campaign_walls_v22_smoke_r2``
  (the R1 root ``runs/campaign_walls_v22_smoke`` stays frozen);
- the verdict is ``SMOKE_V22_R2_VERDICT.json``; ONLY when the fixture
  PASSES the driver also writes ``V22_SMOKE_PASS.json`` (schema
  ``wm59_walls_v22_smoke_pass/1``) binding the fixture LF sha256, the
  law registration LF sha256 (e7fd3536...), the verdict sha256 and the
  frozen J4B registration LF sha256 (2463f5be...) - the file the J4B
  launch gate (``launch_walls.check_v22_smoke_pass``) accepts INSTEAD
  of the J4 ``SMOKE_VERDICT.json``.

Zero SUMO here: this driver only dispatches the REAL child
(``campaign_walls_v1.run_wall_branch.py _exec_wall_branch``) with
``--law-version v2.2-merge-release``, ``--horizon-ticks 300`` and NO
hold override (the registered 240 s draws stay as drawn - the
merge-start cap is what must end the hold).  Tests inject a fake
launcher and fake run artifacts; the integrator runs this in a
no-campaign window.

Binding (deliberate deviation from SMOKE_V22.md section 2, step 2): the
declared ``results_plan/WALLS_PILOT_REGISTRATION_V22.md`` was never
finalized and is NOT invented here.  The smoke binds
``--registration REGISTRATION_WALL_V22.md`` (repo root; LF sha256
e7fd3536..., the owner-approved v2.2 registration, listed in
``output/campaign_walls_v1/WALL_V22_REGISTRATION_ALLOWLIST.json``)
together with the FROZEN walls-pilot plan
(``output/campaign_walls_v1/results_plan/WORLD_PLAN.json``).  The child
enforces the same gate (``check_v22_registration``) and refuses a runs
root holding runs under the other wall law.

Smoke runs are mechanics fixtures, NEVER data: they are never read as
outcomes and never tune anything.  Any FAIL stops the R1 fixture; the
R2 fixture never stops early.  A re-smoke uses a NEW runs root and is
registered as such.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0,
                    str(Path(__file__).resolve().parent.parent / "output"))

from campaign_stage1_v3.common import (  # noqa: E402 - bootstrap above
    load_json,
    sha256_file,
    utc_now,
    write_json,
)
from campaign_walls_v1.common import (  # noqa: E402
    DISK_PREFLIGHT_MIN_FREE_GB,
    ENTRY_EDGE_ID,
    EXIT_EDGE_ID,
    J4B_REGISTRATION_LF_SHA256,
    MERGE_EDGE_ID,
    PACKAGE,
    SMOKE_MIN_FREE_GB,
    WALL_LAW_VERSION_V22,
    WALL_V22_REGISTRATION_LF_SHA256,
    WallsPilotError,
    check_v22_registration,
    lf_sha256_file,
    normalize_wall_law_version,
)
from campaign_walls_v1.launch_walls import check_disk_preflight  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

SMOKE_VERDICT_SCHEMA = "wm59_walls_v22_smoke_verdict/1"
SMOKE_VERDICT_NAME = "SMOKE_V22_VERDICT.json"
DEFAULT_FIXTURE_PATH = REPO_ROOT / "output" / "campaign_walls_v1" / "SMOKE_V22.json"
#: the owner-approved v2.2 registration this smoke binds (workspace copy)
DEFAULT_REGISTRATION_PATH = REPO_ROOT / "output" / "campaign_walls_v1" / "REGISTRATION_WALL_V22.md"
#: the FROZEN walls-pilot plan (worlds/arms/seeds/hold draws carried)
DEFAULT_PLAN_PATH = (PACKAGE / "results_plan" / "WORLD_PLAN.json")
#: the write-once smoke runs root (must NOT exist)
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "campaign_walls_v22_smoke"
#: the bases root of the declared commands (chunk-4 mode-27 saved states)
DEFAULT_BASES_ROOT = "E:/research-dense/runs/campaign_stage1_chunk4"

# --- the R2 RE-smoke (J4BFIX, 2026-10-01; SMOKE_V22_R2.json is authoritative)
R2_FIXTURE_SCHEMA = "wm59_walls_v22_smoke_r2_fixture/1"
R2_FIXTURE_PATH = PACKAGE / "SMOKE_V22_R2.json"
R2_VERDICT_NAME = "SMOKE_V22_R2_VERDICT.json"
R2_RUNS_ROOT = REPO_ROOT / "runs" / "campaign_walls_v22_smoke_r2"
#: ONLY on an R2 PASS: the J4B launch-gate binding file
V22_SMOKE_PASS_SCHEMA = "wm59_walls_v22_smoke_pass/1"
V22_SMOKE_PASS_NAME = "V22_SMOKE_PASS.json"
DEFAULT_V22_SMOKE_PASS_PATH = REPO_ROOT / V22_SMOKE_PASS_NAME

#: the merge edges a member must NOT occupy before the release tick
POST_MERGE_EDGES = frozenset({MERGE_EDGE_ID, EXIT_EDGE_ID})
#: float tolerance for the release-window / cap-point comparisons
EPS = 1e-6
#: terminal outcomes RC4 must bound at/below the merge start (the
#: ``merge_released`` window is checked separately)
RC4_NONRELEASE_OUTCOMES = ("assembled_released", "aborted_in_hold",
                           "aborted_in_assembly")


class SmokeDriverError(WallsPilotError):
    """A v2.2 smoke-driver contract violation."""


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def fixture_kind(fixture: dict) -> str:
    """``"r2"`` for the RE-smoke fixture (abort-tolerant, no stop-on-fail),
    ``"r1"`` for the original declared smoke (MC1-MC5, stop-on-fail)."""
    return "r2" if fixture.get("schema") == R2_FIXTURE_SCHEMA else "r1"


def select_fixture_runs(fixture: dict, plan: dict) -> list[dict]:
    """The declared runs resolved IN the plan, in fixture order.  Refuses
    any drift: a run missing from the plan, a non-wall arm, an arm name
    or a registered hold draw that differs from the fixture (the draw
    stays AS DRAWN - no hold override)."""
    by_run = {b["run"]: (w, b) for w in plan["worlds"]
              for b in w["branches"]}
    resolved = []
    for row in fixture["selection"]["worlds"]:
        entry = by_run.get(row["run"])
        if entry is None:
            raise SmokeDriverError(
                f"declared smoke run {row['run']!r} is not in the plan")
        world, branch = entry
        if branch.get("mode") != "restore_wall":
            raise SmokeDriverError(
                f"declared smoke run {row['run']!r} is not a wall branch "
                f"(mode {branch.get('mode')!r})")
        if branch.get("arm") != row["arm"]:
            raise SmokeDriverError(
                f"declared smoke run {row['run']!r} has plan arm "
                f"{branch.get('arm')!r} != fixture arm {row['arm']!r}")
        wall = branch.get("wall") or {}
        if float(wall.get("hold_s") or -1.0) != \
                float(row["registered_hold_s"]):
            raise SmokeDriverError(
                f"declared smoke run {row['run']!r} carries plan hold_s "
                f"{wall.get('hold_s')!r} != the registered draw "
                f"{row['registered_hold_s']!r} (the draw stays as drawn; "
                "the merge-start cap ends the hold)")
        resolved.append({"fixture_row": row, "world": world,
                         "branch": branch})
    return resolved


def build_child_argv(*, branch, plan_path, registration_path, runs_root,
                     bases_root, horizon_ticks: int, law_version: str,
                     smoke_min_free_gb: float) -> list[str]:
    """The exact child command for one declared run (SMOKE_V22.md section
    4 shape, with the WALL22-R law flag, the SMOKE-ONLY disk-floor
    record, and WITHOUT any hold override)."""
    argv = [
        sys.executable,
        str((PACKAGE / "run_wall_branch.py").resolve()),
        "_exec_wall_branch",
        "--run", branch["run"],
        "--plan", str(Path(plan_path).resolve()),
        "--runs-root", str(Path(runs_root)),
        "--registration", str(Path(registration_path).resolve()),
        "--bases-root", str(bases_root),
        "--horizon-ticks", str(int(horizon_ticks)),
        "--law-version", law_version,
        "--smoke-min-free-gb", str(float(smoke_min_free_gb)),
    ]
    if any("hold-override" in str(a) for a in argv):
        raise SmokeDriverError(
            "the declared smoke carries NO hold override (the registered "
            "240 s draws stay as drawn; the merge-start cap ends the hold)")
    return argv


def evaluate_mc_checks(run_dir, fixture: dict, fixture_row: dict) -> dict:
    """MC1-MC5 for one run, field-by-field per ``SMOKE_V22.json``
    ``checks``, parsed from the run's wall events, branch record and
    receipt.  Every check fails loudly with the numbers it saw."""
    run_dir = Path(run_dir)
    release_point = float(fixture["law"]["merge_release_point_s"])
    registered_hold_s = float(fixture_row["registered_hold_s"])
    horizon_ticks = float(fixture["budget"]["simulated_seconds_per_run"])
    events = _read_jsonl(run_dir / "run" / "wall_events.jsonl")
    record = load_json(run_dir / "run" / "branch_record.json")
    receipt = load_json(run_dir / "RUN_RECEIPT.json")
    checks: list[dict] = []

    def _add(check_id: str, ok: bool, detail: str) -> None:
        checks.append({"id": check_id, "ok": bool(ok), "detail": detail})

    wall_events = [e for e in events if e.get("kind") == "wall_event"]
    assembled = [e for e in wall_events if e.get("event") == "assembled"]
    releases = [e for e in wall_events if e.get("event") == "merge_release"]
    requests = [e for e in events if e.get("kind") == "wall_request"]
    members = ((requests[0].get("members")
                if len(requests) == 1 else None)
               or (assembled[0].get("members") if len(assembled) == 1
                   else None)) or []
    member_set = {str(m) for m in members}
    release_ts = (float(releases[0]["timestamp_s"])
                  if len(releases) == 1 else None)
    assembled_ts = (float(assembled[0]["timestamp_s"])
                    if len(assembled) == 1 else None)
    lane_rows = [e for e in events if e.get("kind") == "wall_member_lanes"]
    pre_release = [r for r in lane_rows
                   if release_ts is None
                   or float(r["timestamp_s"]) < release_ts - 1e-9]

    def _edges(lane_row: dict) -> dict:
        lanes = lane_row.get("lanes") or {}
        return {m: ((lanes.get(m) or {}).get("edge")) for m in member_set}

    # MC1: the wall ASSEMBLED, and every member-lanes row before the
    # release tick shows BOTH members on the entry edge (upstream)
    mc1_rows_ok = bool(pre_release) and all(
        set(lane_row.get("lanes") or {}) == member_set
        and all(edges == ENTRY_EDGE_ID for edges in _edges(lane_row).values())
        for lane_row in pre_release)
    _add("MC1_assembly_upstream_of_merge",
         len(assembled) == 1 and mc1_rows_ok,
         f"assembled={len(assembled)}; {len(pre_release)} pre-release "
         f"wall_member_lanes rows, all members on edge {ENTRY_EDGE_ID}: "
         f"{mc1_rows_ok}")

    # MC2: outcome merge_released; exactly ONE merge_release at the
    # release point; the lead is at/just past the point (<= one control
    # tick of travel); NO member on a post-merge edge before the release
    lead = record.get("lead_s_road_at_release_m")
    wall_speed = float(record.get("wall_speed_mps") or 0.0)
    release_ok = (record.get("outcome") == "merge_released"
                  and len(releases) == 1
                  and releases[0].get("release_point_s") is not None
                  and abs(float(releases[0]["release_point_s"])
                          - release_point) <= 1e-6)
    lead_ok = (lead is not None and wall_speed > 0
               and release_point - 1e-6 <= float(lead)
               < release_point + wall_speed - 1e-6)
    no_early_merge_edge = all(
        edge not in POST_MERGE_EDGES
        for lane_row in pre_release
        for edge in _edges(lane_row).values())
    _add("MC2_merge_release_at_point",
         release_ok and lead_ok and no_early_merge_edge,
         f"outcome={record.get('outcome')!r}; merge_release rows="
         f"{len(releases)} at release_point_s="
         f"{[r.get('release_point_s') for r in releases]} (want "
         f"{release_point}); lead_s_road_at_release_m={lead} in "
         f"[{release_point}, {release_point + wall_speed}); "
         f"pre-release rows on {sorted(POST_MERGE_EDGES)}: "
         f"{not no_early_merge_edge}")

    # MC3: ZERO safety_abort of ANY kind (a v2.2 smoke run with any
    # safety_abort fails; member_left_network is structurally gone)
    aborts = [e for e in wall_events if e.get("event") == "safety_abort"]
    _add("MC3_zero_member_left_network", not aborts,
         f"{len(aborts)} safety_abort rows "
         f"({[a.get('reason') for a in aborts]}) - any abort FAILS")

    # MC4: exactly one hold_cap row - the registered draw, capped at the
    # merge start, effective = merge_release - assembled < the draw,
    # at the cap point; the branch record carries the cap fields
    caps = [e for e in events if e.get("kind") == "hold_cap"]
    cap = caps[0] if len(caps) == 1 else None
    effective = (float(cap["effective_hold_s"])
                 if cap is not None and cap.get("effective_hold_s") is not None
                 else None)
    cap_ok = bool(
        cap
        and abs(float(cap.get("requested_hold_s") or -1.0)
                - registered_hold_s) <= 1e-6
        and cap.get("capped_at_merge") is True
        and effective is not None and effective < registered_hold_s - 1e-9
        and assembled_ts is not None and release_ts is not None
        and abs(effective - (release_ts - assembled_ts)) <= 1e-6
        and cap.get("cap_point_s") is not None
        and abs(float(cap["cap_point_s"]) - release_point) <= 1e-6)
    record_ok = (record.get("hold_capped_at_merge") is True
                 and record.get("merge_release_s") is not None)
    cap_detail = (
        f"hold_cap rows={len(caps)}; "
        f"requested={(cap or {}).get('requested_hold_s')} "
        f"(want {registered_hold_s}), "
        f"capped_at_merge={(cap or {}).get('capped_at_merge')}, "
        f"effective={effective} = release {release_ts} - assembled "
        f"{assembled_ts} (want < {registered_hold_s}), "
        f"cap_point_s={(cap or {}).get('cap_point_s')}; branch record "
        f"hold_capped_at_merge={record.get('hold_capped_at_merge')}, "
        f"merge_release_s={record.get('merge_release_s')}")
    _add("MC4_hold_cap_logged", cap_ok and record_ok, cap_detail)

    # MC5: the receipt contract (complete, ONE loadState, the smoke cap
    # horizon, 600 physics rows = 300 sim-s x 2) + the run's wall law
    rows_written = receipt.get("physics_rows_written")
    mc5_ok = (receipt.get("exit_status") == "complete"
              and receipt.get("load_state_calls") == 1
              and receipt.get("horizon_mode") == "smoke_cap"
              and rows_written == int(2 * horizon_ticks)
              and rows_written == receipt.get("physics_rows_expected")
              and receipt.get("executed_time_s") is not None
              and abs(float(receipt["executed_time_s"]) - horizon_ticks)
              <= 1e-6
              and receipt.get("law_version") == WALL_LAW_VERSION_V22)
    _add("MC5_receipt", mc5_ok,
         f"exit_status={receipt.get('exit_status')!r}, "
         f"load_state_calls={receipt.get('load_state_calls')}, "
         f"horizon_mode={receipt.get('horizon_mode')!r}, "
         f"physics_rows={rows_written}/"
         f"{receipt.get('physics_rows_expected')} (want "
         f"{int(2 * horizon_ticks)}), executed_time_s="
         f"{receipt.get('executed_time_s')} (want {horizon_ticks:g}), "
         f"law_version={receipt.get('law_version')!r}")

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# --- the R2 RE-smoke checks (SMOKE_V22_R2.json ``checks``, field-by-field)

def strip_abort_reason(reason) -> str:
    """A frozen v2.2 abort reason stripped of any ``:<vehicle>`` suffix
    (``non_member_rear_end_risk:ramp.29`` -> ``non_member_rear_end_risk``)."""
    return str(reason or "").split(":", 1)[0]


def _load_run_artifacts(run_dir) -> tuple[list | None, dict | None,
                                          dict | None]:
    """(wall_events.jsonl rows | None, branch_record | None, receipt |
    None) - a missing artifact stays None (its RC fails loudly)."""
    run_dir = Path(run_dir)
    events = (_read_jsonl(run_dir / "run" / "wall_events.jsonl")
              if (run_dir / "run" / "wall_events.jsonl").is_file()
              else None)
    record_path = run_dir / "run" / "branch_record.json"
    record = load_json(record_path) if record_path.is_file() else None
    receipt_path = run_dir / "RUN_RECEIPT.json"
    receipt = load_json(receipt_path) if receipt_path.is_file() else None
    return events, record, receipt


def evaluate_rc_checks(run_dir, fixture: dict, fixture_row: dict) -> dict:
    """RC1, RC2, RC3, RC4, RC4b and RC6 for one run, field-by-field per
    ``SMOKE_V22_R2.json`` ``checks``.  A REGISTERED safety abort (a
    frozen v2.2 reason except ``member_left_network``, stripped of any
    ``:<vehicle>`` suffix) is RECORDED (``registered_aborts``) and fails
    NOTHING; ``member_left_network`` or a reason outside the frozen list
    fails RC2 (unregistered law behaviour)."""
    run_dir = Path(run_dir)
    release_point = float(fixture["law"]["merge_release_point_s"])
    registered_hold_s = float(fixture_row["registered_hold_s"])
    horizon_ticks = int(fixture["budget"]["simulated_seconds_per_run"])
    frozen_reasons = set(fixture.get("frozen_v22_abort_reasons") or ())
    events, record, receipt = _load_run_artifacts(run_dir)
    checks: list[dict] = []
    registered_aborts: list[dict] = []

    def _add(check_id: str, ok: bool, detail: str) -> None:
        checks.append({"id": check_id, "ok": bool(ok), "detail": detail})

    wall_events = ([e for e in events if e.get("kind") == "wall_event"]
                   if events is not None else [])
    aborts = [e for e in wall_events if e.get("event") == "safety_abort"]
    for abort in aborts:
        stripped = strip_abort_reason(abort.get("reason"))
        if stripped in frozen_reasons and stripped != "member_left_network":
            # abort_rule: RECORDED, fails nothing (diagnosis table)
            registered_aborts.append({
                "reason": abort.get("reason"),
                "registered_reason": stripped,
                "timestamp_s": abort.get("timestamp_s"),
                "wall_id": abort.get("wall_id"),
            })

    assembled = [e for e in wall_events if e.get("event") == "assembled"]
    assembled_ts = (float(assembled[0]["timestamp_s"])
                    if len(assembled) == 1 else None)
    finish_events = [e for e in wall_events
                     if e.get("event") in ("merge_release", "safety_abort")]
    finish_ts = (float(finish_events[0]["timestamp_s"])
                 if len(finish_events) == 1 else None)
    lane_rows = ([e for e in events if e.get("kind") == "wall_member_lanes"]
                 if events is not None else [])

    def _member_maps(lane_row: dict) -> list[dict]:
        return [dict(lane or {})
                for _member, lane in sorted(
                    (lane_row.get("lanes") or {}).items())]

    # RC1: law v2.2 everywhere it is logged (events rows, branch record,
    # receipt)
    drifted = [row for row in (events or [])
               if row.get("law_version") is not None
               and row.get("law_version") != WALL_LAW_VERSION_V22]
    record_law = (record or {}).get("law_version")
    receipt_law = (receipt or {}).get("law_version")
    rc1_ok = (events is not None and not drifted
              and record_law == WALL_LAW_VERSION_V22
              and receipt_law == WALL_LAW_VERSION_V22)
    _add("RC1_law_version_v22", rc1_ok,
         f"wall_events.jsonl {'MISSING' if events is None else 'read'}; "
         f"{len(drifted)} rows with a non-v2.2 law_version "
         f"({sorted({str(r.get('law_version')) for r in drifted})[:3]}); "
         f"branch_record.law_version={record_law!r}, "
         f"receipt.law_version={receipt_law!r} "
         f"(want {WALL_LAW_VERSION_V22!r})")

    # RC2: ZERO member_left_network; any abort reason OUTSIDE the frozen
    # list also fails (unregistered law behaviour).  Registered aborts
    # are recorded above and never fail.
    bad_aborts = [a for a in aborts
                  if strip_abort_reason(a.get("reason"))
                  not in frozen_reasons - {"member_left_network"}
                  or strip_abort_reason(a.get("reason"))
                  == "member_left_network"]
    rc2_ok = not bad_aborts
    _add("RC2_no_member_left_network", rc2_ok,
         f"{len(aborts)} safety_abort row(s) "
         f"({[a.get('reason') for a in aborts]}); "
         f"{len(registered_aborts)} registered abort(s) recorded and "
         f"tolerated; failing aborts: "
         f"{[a.get('reason') for a in bad_aborts]}")

    # RC3: exactly ONE hold_cap row (the registered draw at the cap
    # point; the wall assembled and finished) + the full receipt contract
    caps = ([e for e in events if e.get("kind") == "hold_cap"]
            if events is not None else [])
    cap = caps[0] if len(caps) == 1 else None
    receipt_ok = bool(receipt) and all([
        (receipt or {}).get("wall_summary") is not None,
        (receipt or {}).get("exit_status") == "complete",
        (receipt or {}).get("load_state_calls") == 1,
        (receipt or {}).get("horizon_mode") == "smoke_cap",
        (receipt or {}).get("physics_rows_written") == 2 * horizon_ticks,
    ])
    rc3_ok = bool(
        cap
        and abs(float(cap.get("requested_hold_s") or -1.0)
                - registered_hold_s) <= EPS
        and cap.get("cap_point_s") is not None
        and abs(float(cap["cap_point_s"]) - release_point) <= EPS
        and receipt_ok)
    _add("RC3_hold_cap_row_and_receipt", rc3_ok,
         f"hold_cap rows={len(caps)} "
         f"(requested={(cap or {}).get('requested_hold_s')} want "
         f"{registered_hold_s:g}, cap_point_s="
         f"{(cap or {}).get('cap_point_s')} want {release_point:g}); "
         f"receipt {'present' if receipt else 'MISSING'}: "
         f"wall_summary={(receipt or {}).get('wall_summary') is not None}, "
         f"exit_status={(receipt or {}).get('exit_status')!r}, "
         f"load_state_calls={(receipt or {}).get('load_state_calls')}, "
         f"horizon_mode={(receipt or {}).get('horizon_mode')!r}, "
         f"physics_rows_written="
         f"{(receipt or {}).get('physics_rows_written')} "
         f"(want {2 * horizon_ticks})")

    # RC4: no terminal event PAST the merge start.  A merge_release sits
    # in [release_point, release_point + v_wall) (at most one control
    # tick of travel); every other terminal outcome keeps the lead at or
    # before the merge start (the cap fired late or not at all = FAIL).
    outcome = (record or {}).get("outcome")
    lead = (record or {}).get("lead_s_road_at_release_m")
    wall_speed = float((record or {}).get("wall_speed_mps") or 0.0)
    releases = [e for e in wall_events if e.get("event") == "merge_release"]
    if outcome == "merge_released":
        rc4_ok = bool(
            len(releases) == 1
            and releases[0].get("release_point_s") is not None
            and abs(float(releases[0]["release_point_s"]) - release_point)
            <= EPS
            and lead is not None and wall_speed > 0
            and release_point - EPS <= float(lead)
            < release_point + wall_speed + EPS)
    elif outcome in RC4_NONRELEASE_OUTCOMES:
        rc4_ok = lead is None or float(lead) <= release_point + EPS
    else:
        # not_feasible (no terminal wall event) or an unknown outcome
        rc4_ok = outcome == "not_feasible" and lead is None
    _add("RC4_no_terminal_event_past_merge_start", rc4_ok,
         f"outcome={outcome!r}; merge_release rows={len(releases)} at "
         f"release_point_s="
         f"{[r.get('release_point_s') for r in releases]} "
         f"(want {release_point:g}); "
         f"lead_s_road_at_release_m={lead} vs release window "
         f"[{release_point:g}, {release_point + wall_speed:g}) / "
         f"non-release bound <= {release_point + EPS:g}; v_wall="
         f"{wall_speed:g}")

    # RC4b: while the wall is active (from assembly to the release/abort
    # tick, EXCLUSIVE) no member is on the merge/exit edge or past the
    # merge start in s_road
    past = []
    if assembled_ts is not None:
        window = [r for r in lane_rows
                  if assembled_ts - EPS <= float(r["timestamp_s"])
                  and (finish_ts is None
                       or float(r["timestamp_s"]) < finish_ts - EPS)]
        for lane_row in window:
            for lane in _member_maps(lane_row):
                edge = lane.get("edge")
                s_road = lane.get("s_road")
                if edge in POST_MERGE_EDGES or (
                        s_road is not None
                        and float(s_road) > release_point + EPS):
                    past.append({"timestamp_s": lane_row.get("timestamp_s"),
                                 "edge": edge, "s_road": s_road})
    rc4b_ok = assembled_ts is not None and not past
    _add("RC4b_no_member_past_merge_start_in_wall", rc4b_ok,
         f"assembled_ts={assembled_ts}, finish_ts={finish_ts}; "
         f"{len(lane_rows)} wall_member_lanes row(s), "
         f"{len(past)} member observation(s) on {sorted(POST_MERGE_EDGES)} "
         f"or past s_road {release_point:g} while in the wall: {past[:3]}")

    # RC6: the receipt's integrity counts (collisions == 0, teleports == 0)
    collisions = (receipt or {}).get("collisions")
    teleports = (receipt or {}).get("teleports")
    rc6_ok = collisions == 0 and teleports == 0
    _add("RC6_no_collision_no_teleport", rc6_ok,
         f"receipt collisions={collisions!r}, teleports={teleports!r} "
         "(want 0/0; integrity counts only)")

    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "registered_aborts": registered_aborts}


def evaluate_rc5(runs_root, fixture: dict, declared: list[dict],
                 run_entries: list[dict]) -> dict:
    """RC5 ACROSS runs: at least ONE run ended ``merge_released`` with its
    ``hold_cap`` row ``capped_at_merge == true`` and the branch record's
    ``merge_release_s`` set, and in THAT run no ``wall_member_lanes`` row
    BEFORE the release tick shows a member on the merge or exit edge."""
    qualifying: list[str] = []
    candidates: list[str] = []
    details: list[str] = []
    for entry in run_entries:
        run_dir = Path(runs_root) / entry["run"]
        _events, record, _receipt = _load_run_artifacts(run_dir)
        outcome = (record or {}).get("outcome")
        if outcome != "merge_released":
            continue
        candidates.append(entry["run"])
        caps = ([e for e in _read_jsonl(
                    run_dir / "run" / "wall_events.jsonl")
                 if e.get("kind") == "hold_cap"])
        capped = len(caps) == 1 and caps[0].get("capped_at_merge") is True
        released_s = (record or {}).get("merge_release_s")
        if not (capped and released_s is not None):
            details.append(f"{entry['run']}: capped_at_merge={capped}, "
                           f"merge_release_s={released_s!r}")
            continue
        release_ts = float(released_s)
        bad = []
        for lane_row in _read_jsonl(run_dir / "run" / "wall_events.jsonl"):
            if lane_row.get("kind") != "wall_member_lanes":
                continue
            if float(lane_row["timestamp_s"]) >= release_ts - EPS:
                continue  # BEFORE the release tick only
            for _member, lane in sorted(
                    (lane_row.get("lanes") or {}).items()):
                if (lane or {}).get("edge") in POST_MERGE_EDGES:
                    bad.append({"timestamp_s": lane_row.get("timestamp_s"),
                                "edge": (lane or {}).get("edge")})
        if bad:
            details.append(f"{entry['run']}: {len(bad)} pre-release member "
                           f"observation(s) on the merge/exit edge")
            continue
        qualifying.append(entry["run"])
    ok = bool(qualifying)
    return {"id": "RC5_merge_release_observed", "ok": ok,
            "detail": (f"{len(qualifying)} qualifying merge_released run(s) "
                       f"({qualifying}) of {len(candidates)} candidate(s) "
                       f"({candidates}); {'; '.join(details) if details else 'no other candidates'}")}


def run_v22_smoke(*, fixture_path=None, registration_path=None,
                  plan_path=None, runs_root=None,
                  bases_root: str = DEFAULT_BASES_ROOT, out_path=None,
                  launcher=None, smoke_min_free_gb: float | None = None,
                  disk_free_fn=None, pass_out_path=None) -> dict:
    """The declared smoke, end to end: gate the binding, preflight the
    runs-root drive at the SMOKE-ONLY floor, run the three declared
    children SEQUENTIALLY, evaluate the checks per run, write the
    write-once verdict, return it.

    R1 fixture (``SMOKE_V22.json``): MC1-MC5 per run; any child failure
    or failed check STOPS the fixture (later runs are not launched).

    R2 fixture (``SMOKE_V22_R2.json``, J4BFIX): RC1/RC2/RC3/RC4/RC4b/RC6
    per run + RC5 across runs; a registered safety abort is recorded and
    fails nothing; ALL THREE runs always execute (no stop-on-fail); the
    runs root defaults to ``runs/campaign_walls_v22_smoke_r2``; the
    verdict is ``SMOKE_V22_R2_VERDICT.json`` and, ONLY on a PASS, the
    write-once ``V22_SMOKE_PASS.json`` (J4B launch binding) is written
    next to it (``pass_out_path``).

    ``smoke_min_free_gb`` (default ``common.SMOKE_MIN_FREE_GB`` = 5) is
    the WALL22-R-addendum SMOKE-ONLY disk-floor override: the declared
    smoke writes < 1 GB, so the driver preflights at this floor instead
    of the campaign floor (which stays unchanged and launcher-only).
    The override is recorded in the verdict and passed to every child
    for its receipt.  ``disk_free_fn`` is injectable (tests)."""
    fixture_path = (Path(fixture_path) if fixture_path is not None
                    else DEFAULT_FIXTURE_PATH)
    registration_path = (Path(registration_path) if registration_path
                         is not None else DEFAULT_REGISTRATION_PATH)
    plan_path = (Path(plan_path) if plan_path is not None
                 else DEFAULT_PLAN_PATH)
    fixture = load_json(fixture_path)
    is_r2 = fixture_kind(fixture) == "r2"
    out_path = (Path(out_path) if out_path is not None
                else REPO_ROOT / (R2_VERDICT_NAME if is_r2
                                  else SMOKE_VERDICT_NAME))
    runs_root = (Path(runs_root) if runs_root is not None
                 else (R2_RUNS_ROOT if is_r2 else DEFAULT_RUNS_ROOT))
    law_version = normalize_wall_law_version(fixture["law"]["version"])
    if law_version != WALL_LAW_VERSION_V22:
        raise SmokeDriverError(
            f"fixture declares law {law_version!r}, not "
            f"{WALL_LAW_VERSION_V22!r}")
    horizon_ticks = int(fixture["budget"]["simulated_seconds_per_run"])
    if out_path.exists():
        raise SmokeDriverError(
            f"{out_path} already exists (write-once; a re-smoke is a NEW "
            "fixture run, registered as such)")
    if (is_r2 and pass_out_path is not None
            and Path(pass_out_path).exists()):
        raise SmokeDriverError(
            f"{pass_out_path} already exists (the V22_SMOKE_PASS.json is "
            "write-once; a re-smoke registers a NEW runs root and archives "
            "the old pass file first)")
    if not registration_path.is_file():
        raise SmokeDriverError(
            f"registration not found: {registration_path}")
    gate = check_v22_registration(registration_path)
    if gate["registration_lf_sha256"] != WALL_V22_REGISTRATION_LF_SHA256:
        raise SmokeDriverError(
            "the declared smoke binds REGISTRATION_WALL_V22 "
            f"({WALL_V22_REGISTRATION_LF_SHA256[:16]}...), got "
            f"{gate['registration_lf_sha256'][:16]}...")
    if not plan_path.is_file():
        raise SmokeDriverError(f"plan not found: {plan_path}")
    plan = load_json(plan_path)
    declared = select_fixture_runs(fixture, plan)
    if len(declared) != int(fixture["budget"]["runs"]):
        raise SmokeDriverError(
            f"fixture declares {fixture['budget']['runs']} runs, resolved "
            f"{len(declared)}")
    if runs_root.exists():
        raise SmokeDriverError(
            f"smoke runs root already exists (write-once; a re-smoke uses "
            f"a NEW runs root): {runs_root}")
    if not runs_root.parent.is_dir():
        raise SmokeDriverError(
            f"smoke runs parent does not exist: {runs_root.parent}")
    # WALL22-R addendum: the SMOKE-ONLY disk floor (default 5 GB - the
    # declared smoke writes < 1 GB), NOT the campaign floor (which stays
    # launcher-only and unchanged); refused before anything is created
    if smoke_min_free_gb is None:
        smoke_min_free_gb = SMOKE_MIN_FREE_GB
    smoke_min_free_gb = float(smoke_min_free_gb)
    disk_preflight = check_disk_preflight(
        runs_root, min_free_gb=smoke_min_free_gb, disk_free_fn=disk_free_fn)
    runs_root.mkdir()

    passed = True
    runs: list[dict] = []
    registered_aborts: list[dict] = []
    rc5_check: dict | None = None
    for declared_run in declared:
        row = declared_run["fixture_row"]
        branch = declared_run["branch"]
        entry = {"run": branch["run"], "world": declared_run["world"]["world"],
                 "arm": branch["arm"]}
        runs.append(entry)
        argv = build_child_argv(
            branch=branch, plan_path=plan_path,
            registration_path=registration_path, runs_root=runs_root,
            bases_root=bases_root, horizon_ticks=horizon_ticks,
            law_version=law_version, smoke_min_free_gb=smoke_min_free_gb)
        code = (launcher(argv) if launcher is not None else
                subprocess.run([str(a) for a in argv]).returncode)
        entry["exit_code"] = int(code)
        run_dir = runs_root / branch["run"]
        if is_r2:
            # NO stop-on-fail: every declared run is launched and
            # evaluated; a registered safety abort never fails
            evaluation = evaluate_rc_checks(run_dir, fixture, row)
            registered_aborts.extend(
                {"run": entry["run"], **abort}
                for abort in evaluation["registered_aborts"])
            entry["checks_ok"] = evaluation["ok"]
            entry["checks"] = evaluation["checks"]
        else:
            receipt_path = run_dir / "RUN_RECEIPT.json"
            complete = (receipt_path.is_file()
                        and load_json(receipt_path).get("exit_status")
                        == "complete")
            if code == 0 and complete:
                evaluation = evaluate_mc_checks(run_dir, fixture, row)
            else:
                evaluation = {"ok": False, "checks": [{
                    "id": "child_complete", "ok": False,
                    "detail": f"child exit_code={code}, complete receipt="
                              f"{complete} (no MC evaluation without it)"}]}
            entry["checks_ok"] = evaluation["ok"]
            entry["checks"] = evaluation["checks"]
            if not (code == 0 and evaluation["ok"]):
                # any FAIL stops the fixture (no retry in this runs root)
                passed = False
                break

    if is_r2:
        # RC5 across the runs + the R2 pass rule: per-run RC1/RC2/RC3/
        # RC4/RC4b/RC6 for EACH run AND >= 1 qualifying merge_release;
        # child exit codes are surfaced through the per-run checks (a
        # dead child cannot produce the RC3 receipt)
        rc5_check = evaluate_rc5(runs_root, fixture, declared, runs)
        checks_ok = all(entry.get("checks_ok") for entry in runs)
        passed = bool(runs) and checks_ok and rc5_check["ok"]

    verdict = {
        "schema": SMOKE_VERDICT_SCHEMA,
        "created_utc": utc_now(),
        "law_version": law_version,
        "fixture": str(fixture_path),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_lf_sha256": lf_sha256_file(fixture_path),
        "fixture_schema": fixture.get("schema"),
        "registration": str(registration_path),
        "registration_lf_sha256": gate["registration_lf_sha256"],
        "v22_registration_gate": gate,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "bases_root": str(bases_root),
        "runs_root": str(runs_root),
        "horizon_ticks": horizon_ticks,
        "simulated_seconds_total": horizon_ticks * len(runs),
        "hold_override_s": None,
        # WALL22-R addendum: the SMOKE-ONLY disk floor actually applied
        # (the campaign floor stays launcher-only and unchanged)
        "smoke_min_free_gb": smoke_min_free_gb,
        "disk_preflight": disk_preflight,
        "disk_preflight_min_free_gb_declared": DISK_PREFLIGHT_MIN_FREE_GB,
        "disk_floor_note": "SMOKE-ONLY floor override (default 5 GB): the "
                           "declared smoke writes < 1 GB (3 x 300 sim-s "
                           "mechanics fixtures); the campaign floor "
                           f"({DISK_PREFLIGHT_MIN_FREE_GB:g} GB) is NOT "
                           "applied to smoke runs and is unchanged",
        "passed": passed,
        "runs": runs,
        "note": "declared mechanics smoke (SMOKE_V22.md"
                + (" / SMOKE_V22_R2.md" if is_r2 else "")
                + "); smoke runs are "
                "NEVER data and never read as outcomes",
    }
    if is_r2:
        verdict["checks_rule"] = (
            "SMOKE_V22_R2.json: RC1/RC2/RC3/RC4/RC4b/RC6 per run + RC5 "
            "across runs; a registered v2.2 safety abort is recorded and "
            "fails nothing; NO stop-on-fail (all three runs execute)")
        verdict["registered_aborts"] = registered_aborts
        verdict["rc5"] = rc5_check
    write_json(out_path, verdict)
    if is_r2 and passed:
        # ONLY on a PASS: the J4B launch-gate binding file (write-once)
        pass_out_path = (Path(pass_out_path) if pass_out_path is not None
                         else DEFAULT_V22_SMOKE_PASS_PATH)
        if pass_out_path.exists():
            raise SmokeDriverError(
                f"{pass_out_path} already exists (write-once); archive it "
                "before re-smoking")
        smoke_pass = {
            "schema": V22_SMOKE_PASS_SCHEMA,
            "created_utc": utc_now(),
            "fixture": str(Path(fixture_path).resolve()),
            "fixture_lf_sha256": lf_sha256_file(fixture_path),
            "registration": str(Path(registration_path).resolve()),
            "registration_lf_sha256": gate["registration_lf_sha256"],
            "verdict": str(Path(out_path).resolve()),
            "verdict_sha256": sha256_file(out_path),
            "j4b_registration_lf_sha256": J4B_REGISTRATION_LF_SHA256,
            "runs_root": str(runs_root),
            "passed": True,
            "note": "written ONLY on an R2 PASS; the J4B launch gate "
                    "(launch_walls.check_v22_smoke_pass) verifies these "
                    "bindings and the R2 verdict sha256",
        }
        write_json(pass_out_path, smoke_pass)
        # returned dict only - the verdict FILE is write-once and stays
        # exactly the bytes the pass file just bound
        verdict["smoke_pass"] = str(pass_out_path)
    return verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=None,
                        help="the machine-readable fixture (default: "
                             "SMOKE_V22.json at the repo root; the R2 "
                             "re-smoke passes output/campaign_walls_v1/"
                             "SMOKE_V22_R2.json)")
    parser.add_argument("--registration", type=Path,
                        default=DEFAULT_REGISTRATION_PATH,
                        help="the v2.2 registration to bind (default: "
                             "REGISTRATION_WALL_V22.md at the repo root; "
                             "LF sha e7fd3536...)")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH,
                        help="the frozen walls-pilot plan (default: "
                             "output/campaign_walls_v1/results_plan/"
                             "WORLD_PLAN.json)")
    parser.add_argument("--runs-root", type=Path, default=None,
                        help="the write-once smoke runs root (must NOT "
                             "exist; default: runs/campaign_walls_v22_smoke"
                             " for the R1 fixture, "
                             "runs/campaign_walls_v22_smoke_r2 for R2)")
    parser.add_argument("--bases-root", default=DEFAULT_BASES_ROOT,
                        help="the chunk-4 bases root the branches restore "
                             "from (default: the declared E:/ path)")
    parser.add_argument(
        "--smoke-min-free-gb", type=float, default=SMOKE_MIN_FREE_GB,
        help="SMOKE-ONLY disk-floor override (WALL22-R addendum; default "
             f"{SMOKE_MIN_FREE_GB:g} GB): the declared smoke writes < 1 GB, "
             "so the driver preflights the runs-root drive at this floor "
             "instead of the campaign floor; recorded in the verdict and "
             "each receipt. The real launcher refuses this flag")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"the write-once verdict path (default: "
                             f"{SMOKE_VERDICT_NAME} at the repo root for "
                             f"R1, {R2_VERDICT_NAME} for R2)")
    parser.add_argument("--pass-out", type=Path, default=None,
                        help=f"the write-once J4B launch binding written "
                             f"ONLY on an R2 PASS (default: "
                             f"{V22_SMOKE_PASS_NAME} at the repo root)")
    args = parser.parse_args(argv)
    fixture_path = args.fixture or DEFAULT_FIXTURE_PATH
    verdict = run_v22_smoke(
        fixture_path=fixture_path, registration_path=args.registration,
        plan_path=args.plan, runs_root=args.runs_root,
        bases_root=args.bases_root, out_path=args.out,
        smoke_min_free_gb=args.smoke_min_free_gb,
        pass_out_path=args.pass_out)
    print(json.dumps({
        "passed": verdict["passed"],
        "runs": [{"run": r["run"], "checks_ok": r["checks_ok"],
                  "exit_code": r.get("exit_code")} for r in verdict["runs"]],
        "verdict": str(args.out or (
            REPO_ROOT / (R2_VERDICT_NAME
                         if fixture_kind(load_json(fixture_path)) == "r2"
                         else SMOKE_VERDICT_NAME))),
        **({"smoke_pass": verdict["smoke_pass"]}
           if verdict.get("smoke_pass") else {})}))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
