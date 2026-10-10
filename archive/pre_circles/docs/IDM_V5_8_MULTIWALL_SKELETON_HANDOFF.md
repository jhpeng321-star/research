# v5.8 multiwall skeleton handoff

Date: 2026-09-02.  Status: **preparatory engineering only — no SUMO branch and
no scientific outcome were opened.** The skeleton is isolated and nonexecuting
by design; nothing here integrates with the active v5.7.1 Gate A experiment.

## 1. Files created (all new; no existing file modified)

| File | Role |
| --- | --- |
| `policies/multiwall_coordinator.py` | Pure-Python two-cohort state machine and damping-preserving command-composition layer. Stdlib imports only (`copy`, `dataclasses`, `enum`, `hashlib`, `json`, `math`, `typing`). No TraCI, SUMO, Ray, RLlib, or active policy import. |
| `experiments/idm/wall_v5_8_multiwall_contract.py` | Isolated v5.8 draft contract: split mechanical vs scientifically unresolved fields; refuses to build an executable profile while unresolved fields remain; fingerprints only fully specified profiles. |
| `tests/test_multiwall_coordinator.py` | 28 focused synthetic-only tests (#01–#28, mapped below). |
| `docs/IDM_V5_8_MULTIWALL_DRAFT_PROTOCOL.md` | Draft scientific protocol, non-claims, risks, and the future M0–M4 gated sequence. |
| `docs/IDM_V5_8_MULTIWALL_SKELETON_HANDOFF.md` | This document. |

No additional helper module was needed; the coordinator, contract and tests
are each self-contained stdlib + the two sibling files. The existing
`policies/__init__.py` was not edited; the new module is imported directly as
`policies.multiwall_coordinator` and adds nothing to the package surface.

## 2. Architecture summary

`MultiwallCoordinator` is a pure state machine:

- **Cohort lifecycle** (`CohortLifecycle`): INACTIVE → ASSEMBLING → HOLDING →
  TAPERING → RELEASED, with ABORTED from any active state. Each cohort has
  its own id, members/reserves by lane, activation step/time, assembly/HOLD/
  taper elapsed timers, requested duration, target speed, requested/realized
  authority, lane coverage, center/span, speed spread, handoff count,
  abort/release reason, and event/change logs (all mandatory per-cohort
  fields are in `CohortState`).
- **Ownership rules** are global and disjoint; `register_cohort` rejects any
  cross-wall member/reserve overlap, and `promote_reserve`/`handoff` move
  reserve→member atomically.
- **Damping-preserving composition** (`compose(damping_actions,
  wall_commands, ...)`): the output starts as an exact deep copy of the
  supplied damping map; only explicitly active members get replaced by their
  opaque wall payload. Nonmembers, released/aborted members and removed
  members retain the identical supplied damping payload. Payloads are
  arbitrary (dict/list/tuple/numeric), never interpreted, never mutated; the
  input mappings are not mutated (verified by test #28).
- **Lane coverage** is explicit: required vs covered lanes, fraction, missing
  lanes, duplicate assignments, full-coverage flag; activation fails closed if
  a required lane is uncovered.
- **Spatial interaction** uses per-cohort `SpatialGeometry(center_m, span_m)`
  with distance-to-merge ordering (larger center = upstream). Overlap,
  minimum-spacing violation, upstream-catching-downstream, order reversal and
  insufficient storage are detected with exact reasons; activation rejects
  conflicts, `update_spatial` applies the fact and records/returns them, and
  missing spatial data fail closed.
- **Authority accounting**: per-wall decel/accel limits, per-cohort requested
  vs realized authority, and a configurable combined budget with rejection of
  budget-exceeding combinations. Two mild walls sharing a budget can be
  expressed, but no equivalence or safety claim is made.
- **Release coordination**: independent taper/release per cohort; configurable
  `ReleaseOrder` (downstream-first / upstream-first / external),
  simultaneous-release prohibition, and minimum release separation — all
  fail closed with exact conflict reasons.
- **Hard safety priority**: `compose` consumes externally computed
  `SafetyStatus` (front/rear/TTC/lane-coverage/compactness/member-identity/
  storage/release safe) and aborts an unsafe cohort before any timer advance
  or taper; absent safety status also aborts (fail closed). The other cohort
  is untouched unless `SafeAbortPolicy.COORDINATED_ALL` is configured.
- **Diagnostics** are deterministic and JSON-serializable (`json.dumps(
  sort_keys=True, allow_nan=False)`), include a `diagnostic_sha256`, and
  report every item the handoff requires (active ids, lifecycle per cohort,
  ownership, override ids, damping-preservation, dose, coverage, center/span,
  speed spread, inter-wall distance/ordering/storage, individual and combined
  authority, member changes, handoffs, transitions, reasons, release
  separation, conflicts, returned ids).

`wall_v5_8_multiwall_contract.py` hard-codes the mechanical spec, the 12+1
unresolved scientific fields with validation requirements, a serializable
`draft_manifest()` (`executable: False`, `profile_sha256: None`) and an
`executable_profile()` that raises `UnresolvedFieldsError` until every
scientific field is bound and valid — fingerprinting only complete profiles.

## 3. Tests run and result

Command (equivalent to the mandated PowerShell invocation; no SUMO launched,
no TraCI connection opened):

```powershell
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
& "C:\Users\jpeng\Miniconda3\envs\rl_driving\python.exe" -m pytest -q tests\test_multiwall_coordinator.py
```

Result: **28 passed in 0.51 s** (0 failed, 0 skipped). Coverage of the
required cases: #01 disjoint registration, #02 third-active rejection, #03
duplicate membership, #04 primary/reserve overlap, #05 atomic reserve
transfer, #06 nonmember damping unchanged, #07 override only active members,
#08 released members return exactly, #09 aborted members return exactly, #10
one-wall abort leaves the other unchanged, #11 missing lane coverage fails
closed, #12 overlap detected, #13 spacing violation detected, #14 catch-up and
order reversal detected, #15 missing spatial info fails closed, #16 storage
insufficiency detected, #17 combined authority budget enforced, #18 per-wall
limits enforced, #19 simultaneous release prohibited, #20 minimum release
separation enforced, #21 configurable release ordering, #22 hard safety
overrides HOLD/taper, #23 diagnostics deterministic and JSON-serializable, #24
state transitions deterministic, #25 unresolved draft cannot execute, #26
stable fingerprint on complete synthetic profile, #27 reset clears everything,
#28 input maps not mutated.

Note: the test session loads the repository `tests/conftest.py`, which (as
pre-existing shared infrastructure) imports `env.sumo_utils`/`env.snapshot`
and therefore binds the `traci` client library at module scope. Module-level
binding opens no connection and starts no process; no TraCI call is made by
the new coordinator or by the tests, and the running Gate A SUMO/Python
processes are untouched and not connectable from this session.

## 4. Mechanically complete

- Full lifecycle state machine with per-cohort independent state, timers and
  events; at most two active cohorts.
- Global disjoint ownership with atomic reserve promotion and same-lane
  handoff.
- Damping-preserving composition with opaque payloads, exact restoration, no
  input mutation, and no uncontrolled release path.
- Lane-coverage computation and fail-closed activation gate.
- Spatial ordering/overlap/spacing/catch-up/order-reversal/storage checks with
  exact conflict reasons and fail-closed missing-data behavior.
- Per-wall and combined authority accounting with budget enforcement.
- Release-order (downstream-first/upstream-first/external), simultaneous-
  release prohibition and minimum-separation enforcement, fail closed.
- Hard-safety consumption and abort policy with default abort-only-cohort.
- Deterministic JSON diagnostics with sha256.
- Draft contract split (mechanical vs unresolved) with executable gating and
  fingerprinting only of full profiles.

## 5. Remaining unresolved (deliberately unset; do not guess)

Minimum inter-wall spacing; assembly zones; HOLD zones; activation offset;
per-wall authority; combined authority budget; storage threshold; release
order; minimum release separation; candidate-selection rule; CAVs per lane;
treatment duration — plus wall target speed. Each is `None`/absent in the
skeleton defaults and in the draft manifest, reported as "unresolved" in
diagnostics, and rejected by `executable_profile()` until bound. The
feasibility audit (Gate M0/M1) is what may eventually bind them — from
treatment-blind telemetry, never from wall or outcome evidence.

## 6. Integration points that must wait until Gate A finishes

1. Do not edit the active v5.7.1 contract, runner, its output directory, or
   `policies/counterfactual_escalation.py`; this skeleton does not.
2. Any SUMO-side policy adapter (wiring `MultiwallCoordinator.compose` into a
   replay branch as the two-cohort overlay) is future work — only after Gate A
   finishes, Gates M0–M3 pass, and a profile becomes executable.
3. Wall-command payload construction (what a two-wall member command is) is
   not defined by this skeleton; the adapter must supply per-member payloads,
   and missing payloads are recorded (`wall_payload_missing`) with damping
   retained for that member.
4. Candidate/position telemetry ingestion (per-CAV lane, distance, speed,
   gaps, rear/front identity per step) must come from the existing
   instrumented pipelines once they are free; the coordinator only stores and
   checks what it is given.
5. The combined-authority budget is a configured number; only a validated
   profile may set it. Nothing in Gate A's baseline code may be changed to
   accommodate it.

## 7. Recommended next treatment-blind telemetry fields

For Gate M0/M1 the stored decision-time telemetry must be able to answer
"can two disjoint full-lane cohorts exist at t=880?" Recommended additions to
the treatment-blind row (recorded per CAV at the decision absolute time, plus
a short 20 s kinematic window before it):

- per-CAV id, lane id, distance-to-merge (m), speed (m/s), acceleration
  (m/s²), road edge id;
- front leader id and gap; rear follower id and gap; front/rear closing speed
  and TTC;
- lifecycle runway: whether the CAV remains in the eligible region for the
  full requested treatment window under the low-authority speed profile;
- per-lane CAV ordering/coordinate list to evaluate two separated cluster
  centers and spans without any wall definition;
- distance between the two best disjoint cluster centers and their edge-to-
  edge gap (storage capacity is deduced from the gap, never from outcomes);
- flags for front-safe/rear-safe/TTC-safe/low-authority-safe per candidate;
- explicit statement whether any stored row already contains these or whether
  a separate capture is required (the audit file says positions are absent).

## 8. Confirmation

- No SUMO run was launched, no TraCI connection was opened, and no existing
  process or file was touched by this work.
- The active v5.7.1 Gate A experiment (contract, runner, output directory,
  configuration, and `policies/counterfactual_escalation.py`) is unchanged
  and was not read from or written to beyond the required reading.
- No scientific outcome, feasibility conclusion, or result is claimed for a
  two-wall treatment anywhere in this handoff.
