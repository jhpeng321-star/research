# REGISTRATION WM-43B (DRAFT - NOT AUTHORIZED - NEVER LAUNCHED)

**Status: DRAFT.** Prepared 2026-09-30 by the wall-closure worker (branch
`zcode/wd01wallB`, zero simulation). Nothing here authorizes a run. A launch
requires (1) the integrator's exposure re-audit of the seed block on the main
tree, (2) the seeds diff applied by the integrator, (3) the standing smoke
verdict, and (4) an owner-signed `AUTHORIZATION.json` bound to the sha256 of
the FINAL (non-draft) registration. The output directory
`runs/active/wm43b_wall_closure_v1/` must NOT exist before launch and was not
created by this package.

## 1. Question

Does the WM-40 per-CAV dispatcher wall group, on the REAL TraCI / ownership
path, (a) still pass its registered mechanics under the new wall assembly law
`wall-v2.1-closure`, (b) actually CLOSE a wide member pair (40-60 m, up to the
75 m feasibility cap) to span <= 6 m within 60 s, and (c) end an assembly that
cannot close with the declared `assembly_timeout` safety abort and a same-tick
native handback?

This is physical-path integrity evidence only. It is NOT efficacy evidence and
makes no controller comparison (short mechanics runs are not efficacy evidence).

## 2. Changed variable

The wall command law (owner decision 2026-09-30, "Option B"), identified in
every lifecycle event as `law_version = "wall-v2.1-closure"`:

- ASSEMBLY: the FRONT member (largest downstream position `s_m =
  -distance_to_merge_m`) targets `v_wall`; every REAR member targets
  `v_wall + min(2.0 m/s, 0.1 1/s * gap_to_front)` until span <= 6 m; the target
  is turned into an absolute acceleration by the unchanged bounded clip
  `[-3, +1.5] m/s^2` (`_pace`). Completion test unchanged: all speeds within
  1 m/s of `v_wall` AND span <= 6 m.
- HOLD: unchanged (all members paced to `v_wall`).
- NEW safety abort `assembly_timeout` when assembly has not completed 60 s
  after `assembly_start` (members handed back to native immediately). All old
  aborts (`member_frame_missing`, `member_span_exceeded` at 150 m,
  `lane_coverage_violation`, rear-end surveillance) are unchanged and evaluated
  first.

Code identity: `traffic_shepherd/wall_group.py` (commit of branch
`zcode/wd01wallB`; see `CHANGES_WALL_CLOSURE.md`). Part A emulator mirror:
`output/wd01/partA_ctrl/emulators.py::wall_pace` (parity-tested to 1e-9).

## 3. Comparators and reuse

- A1-A3: specifications UNCHANGED from WM-43 r7 (`output/action_model_v1/fixture`,
  `REGISTRATION.md` kept as `REGISTRATION_WM43_r7_REFERENCE.md`). The r7 outcome
  of A1-A3 is the comparator; the integrator must cite the exact r7 verdict from
  the ledger (this worker did NOT verify the r7 outcomes; the only outcome text
  read is the r6 note in FIXTURE_REPORT.md: 11/15 passed, zero collisions, 4/5
  A2 aborts with `member_position_unavailable`, fixed in r7). Expected: identical
  pass/fail per arm with the closure law (members <= 4 m apart, so the rear Delta
  <= 0.4 m/s); a difference is itself a finding.
- A4/A5: NEW; no prior comparable experiment. Closest prior: WM-43 A1 (same
  lanes and wall speed; members 1-4 m apart) and A3 (scripted fault injector).
  The exact unanswered difference: wide pairs never assembled under the v2
  identical-pacing law (a real smoke: 55 m span stayed 55 m).
- The repeat of A1-A3 is a REPLICATION FOR A CHANGED IMPLEMENTATION (the law
  changed), on fresh seeds, not a repeat for its own sake.

## 4. Arms (wall speed 8 m/s, 600 s each, 5 seeds per arm)

| Arm | Spec | Status |
| --- | --- | --- |
| A1 | assemble -> hold -> `release_at_s=90`; members 1-4 m apart, lanes 0/1 | unchanged |
| A2 | A1 lifecycle + damping non-member `follow.0`, `release_at_s=300` | unchanged |
| A3 | scripted actuator pins the front member to 25 m/s from t=90 s -> `member_span_exceeded` abort | unchanged |
| A4 | **wide-pair closure**: same lanes as A1, `wall.1` (front) 40-60 m ahead of `wall.0`, both 19-21 m/s; `release_at_s=150` | NEW |
| A5 | **assembly timeout**: A4 geometry; scripted actuator pins the front member `wall.1` at 10.5 m/s from assembly start (above the rear's max closure target 10.0 m/s and outside the 1 m/s speed tolerance), so closure is impossible; `release_at_s=600` (never reached) | NEW |

A5 pin arithmetic (why the abort is `assembly_timeout` and not
`member_span_exceeded`): rear max speed 10.0 m/s, front pinned 10.5 m/s, so the
span grows at most 0.5 m/s * ~57 s ~= 29 m on top of <= 60 m initial (minus the
initial deceleration overlap) -> < 150 m at t = 60 s. The injection is a logged
fault injector (`scripted_actuator` rows), not a control write: every command
still flows through `CommandOwnership`.

## 5. Machine-evaluated checks (`wm43_checks.py`, parsed from run-dir evidence only)

Common (all arms): C01 run complete, C02 zero collisions, C03 zero teleports,
C04 single-writer path, C05 speed-mode/wall-speed contract, C06 roster and wall
identity, **C07 law_version identity** (`assembly_start`/`assembled` events
carry `wall-v2.1-closure`).

New telemetry row `wall_frames` (per wall-active tick: member speed and
downstream coordinate exactly as the session saw them) lets the checks
re-derive the law offline; `A1_7_wall_pacing_identity` now verifies EVERY wall
row (front v_wall / rear closure formula / hold v_wall) with no skipped ticks.

- A1: A1_1..A1_7 as in r7 (A1_7 = the closure-aware identity).
- A2: A1_1, A1_2, A1_3, A1_7, A2_1..A2_5 as in r7.
- A3: A1_1, A1_2, A1_3, A1_7, A3_1..A3_5 as in r7.
- A4: A1_1 (assembly_start at the plan tick), A4_1 assembled within 60 s of
  `assembly_start`, A4_2 span <= 6 m and speeds within 1 m/s at the assembled
  tick, A4_3 front paced to `v_wall` on every assembly tick, A4_4 rear commands
  equal the closure formula on every assembly tick (and closure is actually
  exercised: at least one tick with gap > 6 m), A1_3 hold speed, A1_4 released
  event (150 s), A1_5 same-tick native handback receipts, A1_6 no wall rows
  after release and no abort, A1_7, A4_5 no rear-end region entry (no abort of
  any kind; the A4 roster has no non-member CAV, so surveillance is vacuously
  silent), plus C02 zero collisions.
- A5: A1_1, A1_7, A5_1 injections logged (first at/just after assembly start),
  A5_2 exactly one abort with reason `assembly_timeout` at 60 +/- 1 s after
  `assembly_start`, A5_3 same-tick native handback, A5_4 injections cease,
  A5_5 the wall never assembled or released, plus C02 zero collisions.

Unit-tested with fakes (no SUMO): every arm's checks pass on a fake-TraCI run
and fail on tampered evidence (`tests/test_wm43_package.py`).

## 6. Seeds (FRESH block proposal; not yet registered)

25 identities, one per run, contiguous after `WM43_WALL_MECHANICS_SEEDS`
(920244-920248): A1 920249-920253, A2 920254-920258, A3 920259-920263, A4
920264-920268, A5 920269-920273. Fixture loading 2000 (registry key only: the
micro-roster generates no stochastic demand), role `exposed`, penetration 1.0.
Roster derivation: `random.Random("wm43b_wall_closure_v1:{arm}:{seed}")`.
Smoke (standing, data-free, <= 80 s): the A4 first-seed run 920264 capped at 70 s.

Exposure audit: `exposure_audit_wm43b.py` (read-only `git grep -P` over tracked,
untracked and git-ignored contents; identity-like = digits not embedded in a
longer number/decimal). A preliminary run on the worktree found zero
identity-like hits except the declared non-use token 920249 as the exclusive
end bound of `range(920244, 920249)`. **The integrator must re-run it on the
main tree (`--repo E:/research-dense`) before applying `seeds_py_wm43b.diff`**;
registry absence is not proof of freshness. The diff adds
`WM43B_WALL_CLOSURE_SEEDS` / `WM43B_SEED_LOADING_MAP` and one accept clause in
`require_development_case`; `traffic_shepherd/seeds.py` is NOT edited by this
package.

## 7. Budget

25 runs x 600 simulated s = 15,000 simulated s (+ one 70 s standing smoke, not
data); single replay each; zero retries; zero training; <= 6 workers (one child
OS process per run); wall-clock request <= 1 h (authorization bound; the
micro-roster runs are expected to finish materially faster); no
acceptance-sealed data touched; no wall-cap or seed expansion authorized by
more workers. Output root: `runs/active/wm43b_wall_closure_v1/` (never
pre-created; the runner creates the leaf), smoke root
`runs/active/wm43b_wall_closure_v1_smoke/`.

## 8. Launch procedure (all steps require the stated authorization)

1. Integrator: re-run `exposure_audit_wm43b.py --repo <main tree> --write`;
   verify FRESH. Apply `seeds_py_wm43b.diff` to `traffic_shepherd/seeds.py`.
2. Integrator: `python output/action_model_v1/fixture_v2/make_digest.py` (writes
   `REGISTRATION_DIGEST.json` for the FINAL registration text), remove the DRAFT
   banner in a new revision of this file, ledger-register the experiment
   (novelty check, budget, sources) per AGENTS.md before execution.
3. `python output/action_model_v1/fixture_v2/run_fixture.py preflight` must pass.
4. Standing smoke: `run_fixture.py smoke` (A4 first seed, <= 70 s, writes the
   write-once `SMOKE_VERDICT.json` bound to the registration sha256 and package
   code digest).
5. Owner writes `AUTHORIZATION.json` (template in the v1 package) bound to the
   registration sha256. Only then `run_fixture.py launch --workers 6`.
6. Close-out: evaluate checks, update ledger/open questions/ROUTE per AGENTS.md.

## 9. Stop rules and constraints preserved

Frozen network bytes untouched; SUMO 1.27.1; `CommandOwnership` sole
longitudinal writer; speedMode 31; no bespoke controller learning; no
successor experiment follows automatically from any outcome.

## 10. Open questions for the owner / integrator

- One seed per run (25 seeds) is proposed; the v1 fixture reused the same 5
  seeds across arms. Reusing 5 seeds would need only a 5-seed block but roster
  derivation is per (arm, seed) so rosters are distinct either way.
- Confirm the A5 pin speed (10.5 m/s) and the 60 +/- 1 s abort tolerance.
- The stray dependency modules `output/controller_verification_v{2,4,7}` are
  byte-copied into this branch (needed by `traffic_shepherd/pi_session.py`);
  in the main tree they are git-ignored, untracked files (merge caution).
