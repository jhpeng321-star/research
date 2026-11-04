# T2C fix round for the current independent r1 review

Resume coordination: the original harness was observed active again after this
receipt was started (it committed lifecycle step `624b0815` and launched
`.t2c-r1-final-prep`). This resumed harness is leaving its remaining tests and
readiness edits to that active harness, and will inspect their final receipts
before reporting completion. Additional completed lifecycle evidence below may be
included in the final ready receipt; no overlapping suite was started here.

Input: `REVIEW_T2C_r1.md`, verdict BLOCK at `7b148d87`, and the complete shared
specification through MAIN 02:30. This supersedes the older, differently scoped
`T2C_FIX_R1.md`/`T2C_FIX_R2.md` receipts.

## Recovery

The resumed session preserved all prior commits, edits, test directories and the
replacement review. Recovered implementation commits:

- `c2a753ba`: blocker 5, opt-in public quote/budget/preparation compatibility.
- `21148f70`: blocker 1, observable realised team reward; WM teaching/credit OFF.
- `0360812b`: blocker 2, successful native all8 augmented-execution admission.
- `b500a3c9`: blocker 3, independent per-CAV contexts and isolated member masks.

The prior lifecycle pytest process (PID 23512) was allowed to finish. Its final
output was unavailable to the resumed harness, so it is not claimed as passing
evidence. No matching process remained before subsequent CPU checks. No dispatch
or live job was launched.

## Blocker 4 — smoke-to-full-run lifecycle

One approval/config/registration/root and one ledger cover both phases. Smoke uses
`smoke/` artifacts, `T2C_SMOKE.claim`, and `t2c_smoke_` episode/reference IDs. Full
training uses the root artifacts, `T2C_RUNTIME.claim`, and `t2c_` IDs, loading fresh
common-u0 models. An exclusive start claim serialises first admission. Full-run
admission requires a completed self-hashed matching smoke receipt if smoke was
attempted; interrupted/duplicate attempts remain refused.

Augmentation is generated/verified once, charged to its own approved allocation,
and reused through immutable validated root-level caches. Full training reloads
the existing ledger, preserving smoke charges and job identities. Both per-phase
and total budget ceilings apply. Smoke's 9,600 episode/reference sim-s are separate
from the common 143,280 augmentation allowance.

CPU check: requested Python, CUDA hidden, bytecode/cache disabled, OMP/MKL=1:

`python -m pytest -q --tb=short -p no:cacheprovider tools/coopmwm/test_t2c_lifecycle_cpu.py --basetemp=E:/agent-work/c4-rerun-tracks/.t2c-resume-lifecycle`

**3 passed**, 29.72 s, exit 0. Covers completed smoke then all three full update-1
arms, fresh u0 weights, immutable smoke checkpoints/caches, disjoint unique job
IDs, cumulative charges, duplicate refusal, interrupted smoke refusal and a
self-hashed receipt with a different approval binding. The inherited unknown
`cache_dir` warning results from disabling pytest's cache provider.

## Completed implementation and verification

`624b0815` completed the lifecycle fix. The final validation pass also binds every
augmented target to the native all8 receipt's plan, snapshot identity, TRAIN seed,
actual execution-tape digest, effective sample and own-execution AWR weight. It
reserves each complete native verification wave before dispatch and validates the
original bank before proposing any variants. Both success and negative native
both-seed confirmations ran against a CPU tape plant; native materialisation was
checked independently of inherited source evidence.

Final focused coverage: **161 passed, 1 skipped**, across 11 explicit T2C files.
The Windows file-symlink privilege skip is inherited; hard-link/junction checks
ran. Final regression coverage: **55 passed, 5 deselected**. Four inherited
Linux `/proc` cases and the fixture that creates its own Git commit were excluded.
There are **216 distinct passing tests**, with no failures in the final checks.
Earlier failed intermediate runs and the timed-out reward invocation are not
counted as passing evidence. No T2C process remained before resuming the checks.

All checks used `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q
--tb=short -p no:cacheprovider`, CUDA hidden, bytecode disabled, OMP/MKL=1. They
were split into short sequential calls with unique worktree-local basetemp roots.
The exact selections and counts are recorded in `T2C_READY.json`.

Current CPU smoke: `T2C_CPU_SMOKE.json`, six episodes/one update, 30 isolated dry
cells, 9,600 virtual smoke sim-s plus 4,800 shared virtual augmentation sim-s,
**zero new real sim-s**. The full-run lifecycle test then starts all three update-1
arms from fresh common u0 while retaining the smoke checkpoint and caches.

The approval-bound quote is **4,310,220 sim-s INCLUDING smoke**, host cpuV,
689,780 below the 5M cap. WM dense teaching/credit have zero phase allocations;
augmentation has one common 143,280 ceiling, below 200,000. The operator contract
and exact monitor/smoke/run commands are in `RERUN_TRACKS_READY.md`. Qualified
sibling exports and fresh deployment/owner bindings remain live-launch prerequisites.

## Fresh-session final receipt verification

Recovered HEAD `31b9a11e` already contains the completed five-blocker fix round.
The tracked worktree was clean. A scoped Git comparison confirms that the
implementation, tests, operator documentation and smoke record are unchanged from
tested implementation commit `9f06f049`; the saved smoke receipt matches both
`T2C_CPU_SMOKE.json` and the readiness record. `git diff --check c4/track-base`
passed.

CPU-only collection of the same 11 focused test files completed with **162 tests
collected**, exit 0, in 14.53 s. This was collection only: the existing **216
distinct passing tests** remain evidence from the completed runs above, with no
new passing-test claims. An existing staging/integration pytest process was
observed and left untouched; no completed suite, runtime or dispatch was repeated.
All prior commits and untracked artifacts were preserved.

`T2C_READY.json` was regenerated with these resume checks. The existing owner
shortcut in `T2C_ACCEPTED.json` delegates deployment integration to S-LAUNCH while
retaining the mandatory end-to-end smoke and safety gate. CPU fix readiness is
unchanged; no live execution occurred in this resume.
