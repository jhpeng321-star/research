"""WM-59 WALL-2: the wall-pilot registration (draft / finalize / check).

Modelled on ``campaign_stage1_v3.chunk3_registration`` (read-only import
surface, same ordering contract): the registration is written BEFORE
anything launches; its SHA-256 and UTC time are recorded beside it
(digest + record).  ``draft`` renders and rewrites the three files while
only a DRAFT exists; ``finalize`` is the write-once formal registration
(delete by hand to re-register).  ``check`` verifies the registration
against the CURRENT live state (base-state digests re-hashed from the
chunk roots, native-sibling receipts required complete) and round-trips
for both kinds.  This package writes no AUTHORIZATION file and launches
nothing.  Pinned here (owner-approved PLAN_DRAFT, 2026-09-28):

- the plan (results_plan/WORLD_PLAN.json, sha pinned): 60 worlds
  (20 per p10/p20/p40), 240 branches, 144,000 simulated s, plan seed
  20261101 (replay bit-exact);
- the base-state digests of the 60 worlds (state.xml + STATE_MANIFEST
  sha256 - re-checked at check time against the chunk-2/3 roots);
- the native sibling run ids (the EXISTING stage-1 native runs reused as
  the reference; their receipts must be complete at check time; their
  outcomes are never read);
- workers = 10; the 45 GB disk preflight on the drive the runs root
  RESOLVES to; speedMode 27 on every CAV path;
- the smoke requirement: a passed, write-once SMOKE_VERDICT.json (<= 80
  simulated s, request from reports, assembly complete OR logged abort,
  members locked, non-members never commanded, one loadState) bound to
  THIS registration's sha256 and the package code digest - the launch
  refuses without it;
- the WM-41/WM-43 gates (the SAME paths the stage-1 chunks pinned);
- E:/ paths only (runs root, out dir, authorization, smoke runs/out) and
  the rendered smoke/launch/resume commands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from campaign_stage1_v3.common import (  # noqa: E402 - bootstrap above
    archive_rename,
    load_json,
    sha256_file,
    utc_now,
    write_json_archiving,
)
from campaign_walls_v1.common import (  # noqa: E402
    BRANCHES_TOTAL,
    CAV_SPEED_MODE,
    DIGEST_PATH,
    DISK_PREFLIGHT_MIN_FREE_GB,
    INTERPRETER,
    LAUNCH_AUTHORIZATION,
    LAUNCH_OUT_DIR,
    LAUNCH_RUNS_ROOT,
    PACKAGE,
    PLAN_PATH,
    RECORD_PATH,
    REGISTRATION_PATH,
    SIMULATED_SECONDS_TOTAL,
    SMOKE_ARM,
    SMOKE_HORIZON_TICKS,
    SMOKE_MAX_SIMULATED_S,
    SMOKE_OUT_DIR,
    SMOKE_RUNS_ROOT,
    SOURCE_CHUNK2_BASES_ROOT,
    SOURCE_CHUNK3_BASES_ROOT,
    WALLS_WORKERS,
    WM41_REGISTRATION,
    WM41_VERDICT,
    WM43_SUMMARY,
    WORLDS_PER_PENETRATION,
    WORLDS_TOTAL,
    WallsPilotError,
    package_code_digest,
    PENETRATIONS,
)
from campaign_walls_v1.plan_walls_pilot import PLAN_SCHEMA  # noqa: E402

WM_ID = "WM-59"
REGISTRATION_SCHEMA = "wm59_walls_pilot_registration/1"
SMOKE_VERDICT_KIND = "wm59_walls_pilot_smoke/1"
AUTHORIZATION_SCHEMA = "wm59_authorization/1"
WORLD_ENTRY_SCHEMA = "wm59_walls_pilot_world_entry/1"

#: the recorded per-world fields the check re-verifies live
WORLD_PIN_FIELDS = ("world", "penetration", "split", "t0_s", "source_chunk",
                    "bases_root", "base_run", "native_sibling_run")


class WallsRegistrationError(WallsPilotError):
    """A wall-pilot registration contract violation."""


def _runs_root_drive(runs_root: str = LAUNCH_RUNS_ROOT) -> str:
    """The drive the runs root RESOLVES to (junctions followed) - the
    drive the 45 GB preflight watches."""
    return Path(runs_root).resolve().drive or "E:"


def render_commands(package: Path | None = None) -> dict:
    """The exact integrator commands (smoke, launch, resume)."""
    package = Path(package) if package is not None else PACKAGE
    smoke = (
        f"{INTERPRETER} {(package / 'smoke_walls.py').as_posix()} "
        f"--plan {PLAN_PATH.as_posix()} "
        f"--registration {REGISTRATION_PATH.as_posix()} "
        f"--runs-root {SMOKE_RUNS_ROOT} "
        f"--out {SMOKE_OUT_DIR}"
    )
    launch = (
        f"{INTERPRETER} {(package / 'launch_walls.py').as_posix()} launch "
        f"--plan {PLAN_PATH.as_posix()} "
        f"--authorization {LAUNCH_AUTHORIZATION} "
        f"--registration {REGISTRATION_PATH.as_posix()} "
        f"--out {LAUNCH_OUT_DIR} --runs-root {LAUNCH_RUNS_ROOT} "
        f"--smoke-verdict {SMOKE_OUT_DIR}/SMOKE_VERDICT.json "
        f"--workers {WALLS_WORKERS}"
    )
    return {
        "smoke": smoke,
        "launch": launch,
        "launch_resume": launch + " --resume",
    }


def worlds_record(plan: dict) -> list[dict]:
    """The per-world pinned identity (digests + sibling ids)."""
    rows = []
    for world in plan["worlds"]:
        rows.append({
            "world": world["world"],
            "penetration": world["penetration"],
            "split": world["split"],
            "t0_s": world["t0_s"],
            "source_chunk": world["source_chunk"],
            "bases_root": world["bases_root"],
            "base_run": world["base_run"],
            "base_state": {
                "state_xml_sha256": world["base_state"]["state_xml_sha256"],
                "state_manifest_sha256": world["base_state"][
                    "state_manifest_sha256"],
                "saved_state_id": world["base_state"]["saved_state_id"],
            },
            "native_sibling_run": world["native_sibling"]["run"],
            "native_sibling_runs_root": world["native_sibling"]["runs_root"],
        })
    return rows


def render_registration(plan: dict, kind: str) -> str:
    """The markdown registration (the pinned contract, human-readable)."""
    commands = render_commands()
    per_pen = {str(pen): sum(1 for w in plan["worlds"]
                             if w["penetration"] == pen)
               for pen in PENETRATIONS}
    splits = {k: sum(1 for w in plan["worlds"] if w["split"] == k)
              for k in ("TRAIN", "CALIBRATION", "HOLDOUT")}
    return f"""# {{WM_ID}} WALL PILOT REGISTRATION - {{title_kind}}

- registered_utc: {{registered_utc}}
- registration_kind: {{registration_kind}} (draft = re-renderable while no
  final exists; final = the write-once registration)
- wm_id: WM-59 (WALL-1 + WALL-2, owner-approved PLAN_DRAFT 2026-09-28)
- plan: results_plan/WORLD_PLAN.json (sha256 {{plan_sha256}}, plan seed
  {plan["plan_seed"]}, replay bit-exact from the seed)

## Question

Do planner-chosen single walls (4 arms: 0.60/0.85 x the members' mean
speed at request, early/late request timing) assemble, hold and hand
back safely on restored stage-1 states - with members locked through the
dispatcher, non-members NATIVE, and the world's existing native sibling
as the reference - producing the wall_events/branch_record data Part A/B
need (p_assembly, assembly time, p_abort by phase, hold-length effects)?

## Design (pinned)

- {plan["budget"]["worlds"]} worlds: {per_pen} (20 per p10/p20/p40 from
  the chunk-2 + chunk-3 mode-27 bases, drawn deterministically from plan
  seed {plan["plan_seed"]}); splits {splits} (the stage-1 49:7:14
  proportions).  HOLDOUT worlds ARE included; their outcomes are never
  read before a registered final pass.
- {plan["budget"]["branches_total"]} branches (4 arms x 60 worlds) x
  {plan["budget"]["branch_length_s"]:g} s = {{budget_sim_s}} simulated s
  (one restored branch per fresh SUMO process, EXACTLY ONE loadState).
- ARMS: wall speed = 0.60/0.85 x the members' mean REPORTED speed at
  request (floor 2.5 m/s, always above wall_group.SAFE_SPEED_MIN_MPS =
  2.0); request timing early (first tick >= t0 with a feasible pair) /
  late (>= t0 + 60); feasible pair = >= 2 live CAVs on the same 2-lane
  mainline edge (1191554636 / 1487606178), different lanes, span <= 75 m
  (MAX_WALL_SPAN_M), evaluated ONLINE from CAV report frames only - never
  truth.  Members = the feasible pair nearest upstream of the merge, one
  per lane.  No feasible pair within 120 s after the target: the branch
  is recorded not_feasible and continues native.
- Hold H in {{60, 120, 240}} s drawn from the branch seed (recorded in
  the plan), counted from assembly completion; then release - members
  handed back to NATIVE.  Safety aborts (wall_group rules, ANY phase)
  also hand members back to native; the abort reason is logged.
- Non-members NATIVE for the whole branch.  Each wall branch restores
  <bases_root>/<base_run> saved_state with the SAME arrival seed as its
  world's EXISTING native sibling (chunk-2/3 roots below); the sibling
  is REUSED as the reference (recorded run + path; never re-run).
- Every CAV path carries speedMode {CAV_SPEED_MODE} (WM-46, carried).
- A wall still active at the horizon end fails the branch loudly (never
  recorded as a release).

## Pinned per world

The record (WALLS_PILOT_REGISTRATION_RECORD.json) pins, for each of the
{WORLDS_TOTAL} worlds: penetration, split, t0_s, source chunk, bases
root, base run, the base-state digests (state.xml + STATE_MANIFEST.json
sha256 + saved_state_id - RE-VERIFIED at check time against the live
chunk roots), and the native sibling run id + runs root.  Chunk roots:
{SOURCE_CHUNK2_BASES_ROOT}
{SOURCE_CHUNK3_BASES_ROOT}

## Workers: {WALLS_WORKERS}

--workers {WALLS_WORKERS} is the declared worker count (above it the
launch refuses).

## Disk preflight: {DISK_PREFLIGHT_MIN_FREE_GB:g} GB

The launch refuses unless the drive the runs root RESOLVES to
({_runs_root_drive()} for {LAUNCH_RUNS_ROOT}) has at least
{DISK_PREFLIGHT_MIN_FREE_GB:g} GB free (the ~40 GB pilot fits beside the
running campaign; the WM-53 85 GB chunk preflight does not).

## Smoke (<= {SMOKE_MAX_SIMULATED_S:g} simulated s; REQUIRED)

The launch refuses without a passed, write-once SMOKE_VERDICT.json
(kind {SMOKE_VERDICT_KIND}) at
{SMOKE_OUT_DIR}/SMOKE_VERDICT.json bound to THIS registration's sha256
AND the package code digest: restore one p40 pilot world's base, request
a {SMOKE_ARM} wall, run <= {SMOKE_HORIZON_TICKS} ticks.  PASS requires:
the request was made from reports; the event log shows assembly complete
OR a logged abort with a reason; members were locked in dispatcher_rows
during assembly/hold; non-members were never commanded by the wall; the
one-loadState invariant held.  Smoke command (no AUTHORIZATION required;
run right before the launch so the code digest is fresh):

```
{commands["smoke"]}
```

## Launch staging and gates

WM-41 exactness gate ({WM41_VERDICT}; registration
{WM41_REGISTRATION}), WM-43 wall-mechanics fixture gate
({WM43_SUMMARY}), authorization gate (owner-written AUTHORIZATION.json
carrying the exact SHA-256 of THIS registration at
{LAUNCH_AUTHORIZATION}), the disk preflight above, write-once receipts,
zero retries.  E:/ paths only.

Launch (after the owner writes the authorization):

```
{commands["launch"]}
```

Resume (only after an interrupted attempt):

```
{commands["launch_resume"]}
```

## Freeze

Constants above are declared before any run exists and are never tuned
on outcomes.  HOLDOUT wall-branch outcomes are sealed until the
registered final pass.  Scoring/analytics packages register separately.
"""


def _require_plan(plan_path: Path | None = None) -> dict:
    path = Path(plan_path) if plan_path is not None else PLAN_PATH
    if not path.is_file():
        raise WallsRegistrationError(
            f"registration requires the plan file first: {path} "
            "(run plan_walls_pilot first)")
    plan = load_json(path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise WallsRegistrationError(
            f"{path} schema {plan.get('schema')!r} != {PLAN_SCHEMA!r}")
    return plan


def _write_registration(plan: dict, kind: str, *, package: Path | None = None,
                        paths: dict | None = None) -> dict:
    package = Path(package) if package is not None else PACKAGE
    pathset = paths or {
        "registration": REGISTRATION_PATH,
        "digest": DIGEST_PATH,
        "record": RECORD_PATH,
        "plan": PLAN_PATH,
    }
    registration_path = Path(pathset["registration"])
    text = render_registration(plan, kind)
    text = (
        text.replace("{WM_ID}", WM_ID)
        .replace("{title_kind}", "stage-1 wall pilot (prepare only; nothing launched)")
        .replace("{registered_utc}", utc_now())
        .replace("{registration_kind}", kind)
        .replace("{plan_sha256}", sha256_file(Path(pathset["plan"])))
        .replace("{budget_sim_s}", f"{SIMULATED_SECONDS_TOTAL:,.0f}")
    )
    archive_rename(registration_path, tag="superseded")
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(text, encoding="utf-8")
    registration_sha = sha256_file(registration_path)
    digest = write_json_archiving(Path(pathset["digest"]), {
        "schema": "wm59_walls_pilot_registration_digest/1",
        "registration_kind": kind,
        "registration_sha256": registration_sha,
        "plan_sha256": sha256_file(Path(pathset["plan"])),
        "written_utc": utc_now(),
    })[0]
    record = {
        "schema": REGISTRATION_SCHEMA,
        "wm_id": WM_ID,
        "registration_kind": kind,
        "registered_utc": utc_now(),
        "plan_schema": plan["schema"],
        "plan_seed": plan["plan_seed"],
        "plan_sha256": sha256_file(Path(pathset["plan"])),
        "worlds": WORLDS_TOTAL,
        "worlds_per_penetration": {str(pen): sum(
            1 for w in plan["worlds"] if w["penetration"] == pen)
            for pen in PENETRATIONS},
        "worlds_per_penetration_expected": {
            str(pen): WORLDS_PER_PENETRATION for pen in PENETRATIONS},
        "splits": {k: sum(1 for w in plan["worlds"] if w["split"] == k)
                   for k in ("TRAIN", "CALIBRATION", "HOLDOUT")},
        "splits_expected": {"TRAIN": 42, "CALIBRATION": 6, "HOLDOUT": 12},
        "branches_total": BRANCHES_TOTAL,
        "simulated_seconds_total": SIMULATED_SECONDS_TOTAL,
        "worlds_table": worlds_record(plan),
        "declared_workers": WALLS_WORKERS,
        "disk_preflight_min_free_gb": DISK_PREFLIGHT_MIN_FREE_GB,
        "disk_preflight_drive": _runs_root_drive(),
        "cav_speed_mode": CAV_SPEED_MODE,
        "smoke": {
            "kind_required": SMOKE_VERDICT_KIND,
            "arm": SMOKE_ARM,
            "max_simulated_s": SMOKE_MAX_SIMULATED_S,
            "horizon_ticks": SMOKE_HORIZON_TICKS,
            "verdict_path": f"{SMOKE_OUT_DIR}/SMOKE_VERDICT.json",
            "bound_to": ["registration_sha256", "package_code_sha256"],
            "runs_root": SMOKE_RUNS_ROOT,
            "out_dir": SMOKE_OUT_DIR,
        },
        "package_code_sha256_at_registration": package_code_digest(package),
        "launch_paths": {
            "runs_root": LAUNCH_RUNS_ROOT,
            "out_dir": LAUNCH_OUT_DIR,
            "authorization": LAUNCH_AUTHORIZATION,
        },
        "gate_paths": {
            "wm41_verdict": WM41_VERDICT,
            "wm41_registration": WM41_REGISTRATION,
            "wm43_summary": WM43_SUMMARY,
        },
        "commands": render_commands(package),
        "registration_sha256": registration_sha,
    }
    record_path = write_json_archiving(Path(pathset["record"]), record)[0]
    check(record_path, package=package, verify_gate_files=False)
    return {
        "registration": str(registration_path),
        "digest": str(digest),
        "record": str(record_path),
        "registration_kind": kind,
        "registration_sha256": registration_sha,
    }


def draft(*, plan_path: Path | None = None, package: Path | None = None,
          paths: dict | None = None) -> dict:
    """Render and write the DRAFT registration (refused once a FINAL
    exists - delete by hand to re-register)."""
    record_path = Path(paths["record"]) if paths else RECORD_PATH
    if record_path.is_file():
        existing = load_json(record_path)
        if existing.get("registration_kind") == "final":
            raise WallsRegistrationError(
                f"{record_path} is a FINAL registration; draft refuses to "
                "overwrite - delete by hand to re-register")
    plan = _require_plan(plan_path)
    return _write_registration(plan, "draft", package=package, paths=paths)


def finalize(*, plan_path: Path | None = None, package: Path | None = None,
             paths: dict | None = None) -> dict:
    """The write-once FINAL registration (delete by hand to re-register)."""
    record_path = Path(paths["record"]) if paths else RECORD_PATH
    if record_path.is_file():
        existing = load_json(record_path)
        if existing.get("registration_kind") == "final":
            raise WallsRegistrationError(
                f"{record_path} exists as FINAL; finalize refuses to "
                "overwrite - delete by hand to re-register")
    plan = _require_plan(plan_path)
    return _write_registration(plan, "final", package=package, paths=paths)


def check(record_path: Path | None = None, *, package: Path | None = None,
          verify_gate_files: bool = True) -> dict:
    """Verify the registration against the CURRENT live state: digests,
    plan sha, pinned constants, the 60 base-state digests re-hashed from
    the chunk roots, and every native-sibling receipt complete."""
    package = Path(package) if package is not None else PACKAGE
    record_path = Path(record_path) if record_path is not None else RECORD_PATH
    record = load_json(record_path)
    plan_dir = record_path.resolve().parent
    registration_path = plan_dir / "WALLS_PILOT_REGISTRATION.md"
    if sha256_file(registration_path) != record["registration_sha256"]:
        raise WallsRegistrationError(
            "record digest does not match the current "
            "WALLS_PILOT_REGISTRATION.md")
    digest = load_json(plan_dir / "WALLS_PILOT_REGISTRATION_DIGEST.json")
    if digest["registration_sha256"] != record["registration_sha256"]:
        raise WallsRegistrationError(
            "digest and record disagree on the registration sha")
    plan_path = plan_dir / "WORLD_PLAN.json"
    if sha256_file(plan_path) != record["plan_sha256"]:
        raise WallsRegistrationError(
            "plan sha256 mismatch (the plan changed after registration)")
    if record.get("registration_kind") not in ("draft", "final"):
        raise WallsRegistrationError(
            f"unknown registration_kind {record.get('registration_kind')!r}")
    # pinned constants
    if record.get("branches_total") != BRANCHES_TOTAL or \
            record.get("simulated_seconds_total") != SIMULATED_SECONDS_TOTAL \
            or record.get("worlds") != WORLDS_TOTAL:
        raise WallsRegistrationError(
            "record budget is not the registered 60 worlds / 240 branches "
            "/ 144,000 sim-s")
    if record.get("worlds_per_penetration") != \
            record.get("worlds_per_penetration_expected") or \
            record.get("splits") != record.get("splits_expected"):
        raise WallsRegistrationError(
            "record world/split counts differ from the approved shape")
    if record.get("declared_workers") != WALLS_WORKERS:
        raise WallsRegistrationError(
            f"record worker count is not {WALLS_WORKERS}")
    if record.get("disk_preflight_min_free_gb") != DISK_PREFLIGHT_MIN_FREE_GB:
        raise WallsRegistrationError(
            f"record disk preflight is not {DISK_PREFLIGHT_MIN_FREE_GB:g} GB")
    if record.get("disk_preflight_drive") != _runs_root_drive():
        raise WallsRegistrationError("record preflight drive mismatch")
    if record.get("cav_speed_mode") != CAV_SPEED_MODE:
        raise WallsRegistrationError("record speed mode mismatch")
    smoke = record.get("smoke") or {}
    if smoke.get("kind_required") != SMOKE_VERDICT_KIND or \
            smoke.get("max_simulated_s") != SMOKE_MAX_SIMULATED_S or \
            smoke.get("arm") != SMOKE_ARM or \
            smoke.get("horizon_ticks") != SMOKE_HORIZON_TICKS or \
            "package_code_sha256" not in (smoke.get("bound_to") or []):
        raise WallsRegistrationError(
            "record smoke requirement is not the registered <= 80 sim-s "
            "report-request smoke bound to the registration + code digests")
    if record.get("launch_paths", {}).get("runs_root") != LAUNCH_RUNS_ROOT or \
            record.get("launch_paths", {}).get("authorization") != \
            LAUNCH_AUTHORIZATION:
        raise WallsRegistrationError("record launch paths mismatch")
    if record.get("gate_paths", {}).get("wm41_verdict") != WM41_VERDICT or \
            record.get("gate_paths", {}).get("wm43_summary") != WM43_SUMMARY:
        raise WallsRegistrationError("record gate paths mismatch")
    if verify_gate_files:
        for name, path in record.get("gate_paths", {}).items():
            if not Path(path).is_file():
                raise WallsRegistrationError(
                    f"pinned gate file missing at check time ({name}): {path}")
    # the 60 worlds: pinned identity, LIVE base-state digests, COMPLETE
    # native-sibling receipts (receipt status only - outcomes sealed)
    rows = record.get("worlds_table") or []
    if len(rows) != WORLDS_TOTAL:
        raise WallsRegistrationError(
            f"record pins {len(rows)} worlds != {WORLDS_TOTAL}")
    for row in rows:
        for field in WORLD_PIN_FIELDS:
            if field not in row:
                raise WallsRegistrationError(
                    f"world row missing {field!r} (schema "
                    f"{WORLD_ENTRY_SCHEMA})")
        state_dir = Path(row["bases_root"]) / row["base_run"] / "saved_state"
        state_xml = state_dir / "state.xml"
        manifest_path = state_dir / "STATE_MANIFEST.json"
        if not state_xml.is_file() or not manifest_path.is_file():
            raise WallsRegistrationError(
                f"{row['world']}: base saved state missing under "
                f"{state_dir}")
        if sha256_file(state_xml) != row["base_state"]["state_xml_sha256"]:
            raise WallsRegistrationError(
                f"{row['world']}: base state.xml digest changed since "
                "registration")
        if sha256_file(manifest_path) != \
                row["base_state"]["state_manifest_sha256"]:
            raise WallsRegistrationError(
                f"{row['world']}: STATE_MANIFEST.json digest changed since "
                "registration")
        receipt_path = (Path(row["native_sibling_runs_root"])
                        / row["native_sibling_run"] / "RUN_RECEIPT.json")
        if not receipt_path.is_file():
            raise WallsRegistrationError(
                f"{row['world']}: native sibling receipt missing: "
                f"{receipt_path} (the reference must exist before the "
                "pilot launches)")
        receipt = load_json(receipt_path)
        if receipt.get("exit_status") != "complete":
            raise WallsRegistrationError(
                f"{row['world']}: native sibling "
                f"{row['native_sibling_run']} is not complete "
                f"(exit_status {receipt.get('exit_status')!r})")
    return {
        "verified": True,
        "registration_kind": record.get("registration_kind"),
        "worlds": WORLDS_TOTAL,
        "branches": BRANCHES_TOTAL,
        "simulated_seconds_total": SIMULATED_SECONDS_TOTAL,
        "registration_sha256": record["registration_sha256"],
        "plan_sha256": record["plan_sha256"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("draft")
    sub.add_parser("finalize")
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--record", type=Path, default=None)
    check_parser.add_argument("--skip-gate-files", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "draft":
        print(json.dumps(draft()))
        return 0
    if args.command == "finalize":
        print(json.dumps(finalize()))
        return 0
    print(json.dumps(check(args.record,
                           verify_gate_files=not args.skip_gate_files)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
