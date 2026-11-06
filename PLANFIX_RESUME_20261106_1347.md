# PLANFIX preserved-session recovery — 2026-11-06 13:47 +08

## Step 1 — recover before execution

Recovered task worktree `E:/agent-work/c4-variety-collect` at `737a08f3`.
`E:/agent-work/_sol` is the harness directory, not a Git repository.
Tracked diff is empty. Pre-existing untracked `VARCOLLECT_READY.blocked3.json`
and `VARCOLLECT_READY.blocked4.json` are preserved.

Read history/status, tracks_spec, central REGISTRATION last 200 lines
(7657–7856), diagnosis, central bank results/readiness and the prior 12:49
recovery plus its successful 12:50 CPU status receipt before taking action.
The previous recovery already established terminal bank dispositions and
exited WM/coordinator identities; no replacement status job is needed to
decide whether the historical dispatch can resume.

Owner decisions recorded at 02:44 supersede S-VARIETY with G1/mining;
03:00 releases the 566,400 unspent plan_fix reservation. Later registrations
through 13:41 do not restore it. Clean1 is owner-stopped at A9/B9/R9,
clean2 stopped at u18, and clean3 is owner-stopped after its u1 fault.
The historical clean1/clean2 u12 feed-in cannot be applied prospectively.

Disposition: BLOCKED_AUTHORITY / NOT READY. No new SSH/GPU transfer, WM
dispatch, SUMO, signal, ledger amendment or bank switch in this recovery.
Preserve finished tests, failed inputs, source packages and result evidence.
Continuation requires explicit restored plan_fix funding within the current
aggregate cap, intended run/update boundary, and MAIN/owner task ownership.

## Follow-up found during recovery

Central `PLAN_FIX_READY.md` correctly records supersession and the 12:50
worker audit, but `PLAN_FIX_READY.json` still describes a 600,000 active
allocation, target u12 and obsolete missing-input/clean2-not-launched blockers.
Reconcile its metadata while retaining `ready=false`, `apply=false`, and
`outputs=[]`; do not publish any executable bank output.

## Step 2 — existing receipt consistency PASS

Local Python assertions verify eight source dispositions, twelve measured
zero-gain/zero-lower-bound outcomes, two null-gain startup exclusions, zero
survivors, exactly 33,600 retained charges, and agreement of all six native
confirmation hashes with `PLANFIX_BANK_TERMINAL.json`. The preserved 12:50
CPU status has both exact identities exited and an empty gate set.
No fresh liveness or native scientific re-verification is claimed.
`PLANFIX_RECEIPT_CHECK_20261106_1347.json` pins the two input file hashes.
Existing full suites remain adopted; this is bookkeeping-only verification.

## Step 3 — readiness metadata and progress reconciled

Updated central `PLAN_FIX_READY.json` to retain the historical allocation and
bank-completion spend separately from zero active allocation/remaining authority.
Run and actual update target are null; the original u12 is historical only.
Obsolete missing-input/clean2-not-launched blockers are replaced with supersession,
released reservation and the observed terminal CPU gate/search state. The
machine consumer remains disabled (`ready=false`, `apply=false`, `outputs=[]`).
Added a dated section to `PLAN_FIX_READY.md` and four PLANFIX lines to
`CLEAN1_STATUS.md`. No AUTHORITY or ledger is amended: this recovery grants no
budget. No remote command or process action was needed. Existing unrelated
untracked files and every earlier receipt remain intact.

Final local assertions PASS: switch disabled, null run/update, zero active
allocation, historical33,600 + released566,400 =600,000 and unchanged bank
results hash. Readiness metadata SHA is pinned in the receipt-check JSON.
Central run artifacts are not tracked changes in `E:/research-dense`; recovery
documentation and their content-addressed check are committed in this worktree.
