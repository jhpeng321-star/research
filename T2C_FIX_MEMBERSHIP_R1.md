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

Status: inspection complete; implementation and verification pending.
