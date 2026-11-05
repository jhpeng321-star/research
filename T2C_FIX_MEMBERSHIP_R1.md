# T2C r1 dynamic-membership fix round

## Recovery and scope

Recovered HEAD: `70911f7aad20e5b4674d7dc7efb162064260737e` on
`c4/rerun-tracks`, 2026-11-05. Git status/log, the full shared specification
through MAIN 02:30, the replacement `REVIEW_T2C_r1.md`, and the existing
ready/fix/acceptance receipts were read before implementation work.

The independent review already accepts the preceding five fixes and reports one
new blocker: retained multi-CAV commitments can reference partners that leave
sensing or exit between decision ticks. The production hook currently treats
that ordinary turnover as an invalid command and aborts.

Existing uncommitted review text and all untracked test/receipt artifacts are
preserved. Process inventory showed no active pytest invocation. Existing
processes were left untouched; no runtime, dispatch or verification was launched.
Completed prior suites will not be duplicated as recovery work.

## Required fix and verification

Retire an entire retained move when a member is no longer locally observable
and commandable. Do not resample that actor until the registered decision tick;
continue the native base and independently valid residual control. Actor exit
discards its context; reappearance gets a fresh context, as with any new actor.
Keep invalid fresh proposals and invalid residuals fail-closed.

Add CPU production-hook coverage for out-of-range partners still on-road,
departures, actor exit/reappearance, overlapping contexts, isolated all/solo
masks, decision cadence and reward/learning receipt attribution. Exercise the
native V2V controller over the complete controlled episode with changing frames.

## Implementation and production-hook verification

The hook cancels the complete retained commitment as soon as any member is
outside the actor's current observable/commandable set. Its sampled move/onset
remain available for learning attribution; cancellation does not select a new
move or consume trigger RNG. Visibility return does not resurrect the move. The
next registered decision tick can select normally. Departed actors are still
discarded, and returning actors start a fresh context.

Cancelled actors retain deterministic ownership of their own zero-correction
base fallback. This prevents a returning partner's overlapping group from
indirectly reapplying a move to an actor whose own commitment was cancelled.
Independently valid residuals still run. Fresh proposals and residual outputs
remain checked against the observable/commandable set, including zero-valued
invalid residual entries. Training rows record `move_cancelled_at_s` and
`move_unavailable_members`; reward accumulation remains attached to the original
decision row, with no invented decision or off-cadence label.

Before implementation, the new range-turnover regression failed at the exact
reviewed `HDV/nonobservable production command` exception (1 expected failing
test, stopped at first failure). Intermediate runs exposed a zero-fallback
ownership case and two test-fixture expectations; those were corrected and are
not counted as passing evidence.

Completed CPU check (exit 0, 46.29 s):

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q --tb=short -p no:cacheprovider tools/coopmwm/test_t2c_multicav_cpu.py tools/coopmwm/test_t2c_backend_cpu.py --basetemp=E:/agent-work/c4-rerun-tracks/.t2c-membership-hooks-final
```

**43 passed**, including **27 added regressions**. Eighteen cases drive every
controlled second (t=300..1199) with three overlapping actors, range loss,
departure/reappearance or missing actuator, and training/all/group/solo/empty
masks. They check exact cadence, single-owner commands, base fallback,
non-resurrection, row identity, cancellation records and conserved reward steps.
Negative cases retain refusal for invalid fresh proposals/residuals. Two native
V2V worker cases (B/R) run the entire controlled window with dynamic sensor
frames, including an empty-CAV frame, then update the layer from the actual
reward rows while verifying that the base model remains frozen.

Environment: CUDA hidden, bytecode disabled, OMP/MKL=1. The only warning is the
inherited unknown `cache_dir` option with the pytest cache provider disabled.
No SUMO/GPU/SSH/dispatch or real sim-seconds were used. Implementation and these
tests were committed in `94b58dfc` after the recovery checkpoint `7b66877e`.

## Affected integration checks and final readiness

Two additional short sequential invocations completed at `94b58dfc`, with the
same interpreter, arguments and environment above:

```text
tools/coopmwm/test_t2c_runtime_cpu.py tools/coopmwm/test_t2c_redesign_cpu.py --basetemp=E:/agent-work/c4-rerun-tracks/.t2c-membership-runtime-final
```

**31 passed**, exit 0, 54.13 s. Includes runtime smoke, shared-final orchestration,
realised-reward learning and disabled WM-path boundaries.

```text
tools/coopmwm/test_t2c_compat_cpu.py tools/coopmwm/test_v2v_runner_cpu.py tools/coopmwm/test_v2v_runtime_cpu.py --basetemp=E:/agent-work/c4-rerun-tracks/.t2c-membership-compat-final
```

**26 passed**, exit 0, 15.97 s. Checks default-off compatibility and native
V2V runner/controller boundaries. Thus this round has **100 distinct passing
tests**, including 27 added regressions, with no failures in the final checks.
The preceding round's 216 passes/1 skip/5 exclusions remain historical evidence;
they are preserved separately in readiness rather than claimed as fresh runs.

`T2C_READY.json` v7 binds implementation `94b58dfc`, the replacement review and
these exact invocations. `T2C_CPU_SMOKE.json` now matches the completed runtime
test's smoke receipt: 6 episodes, 1 update, 30 isolated dry cells, 14,400 virtual
sim-s (9,600 smoke + 4,800 once-only augmentation), **zero real sim-s**. This is
a CPU test backend result; the full-window native-hook B/R tests above provide
the dynamic-membership evidence. Deployment integration, owner bindings, the
real end-to-end smoke and safety gate remain required as already documented.

The production lifecycle is documented in `RERUN_TRACKS_READY.md`. The independent
review is preserved unchanged; its blocker is addressed by this implementation,
not by rewriting its verdict. CPU fix status: **READY FOR REVIEW**.

Final scoped `git diff --check c4/track-base` passed for the implementation,
tests, review, operator documentation and readiness/smoke/fix receipts.
