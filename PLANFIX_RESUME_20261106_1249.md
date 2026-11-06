# PLANFIX preserved-session recovery — 2026-11-06 12:49 +08

## Step 1 — receipts and authority recovered

Recovered `E:/agent-work/c4-variety-collect` at `d8626035`. The harness
directory `E:/agent-work/_sol` is not a Git worktree. Initial task-worktree
status contains only the pre-existing untracked `VARCOLLECT_READY.blocked3.json`
and `VARCOLLECT_READY.blocked4.json`; both are preserved.

Read the task history/status, full `tracks_spec.md`, central `REGISTRATION.md`
lines 7603–7802 (last 200 lines), full `CLEAN1_DIAGNOSIS_20261105.md`,
`PLANFIX_CPU_HANDOFF.md`, `PLANFIX_RECOVERY_20261106.md`, and the complete
462-line central `PLAN_FIX_BANK_RESULTS.json` before remote action.

### Completed bank pass — adopt, do not replay

Eight source seed examples have terminal dispositions: six measured examples,
each with two CRN results, plus two excluded startup failures. All twelve
measured translated gains and one-sided lower bounds are zero; none is admitted.
The two startup failures have null translated gains and are not scientific
negative results. Original gain, translated gain, family, source example,
native receipt hash and charges remain in `PLAN_FIX_BANK_RESULTS.json`.
Total charged: **33,600 sim-s** (2 × 2,400 + 6 × 4,800).

Prior 301-test/9-subtest collector qualification and eleven executor checks
are retained. This receipt-only recovery does not rerun completed suites or
claim a newly qualified scientific implementation.

### Conflicting authority — additional physics blocked

The resumed prompt's original clean1/u12/5M allocation predates central owner
decisions recorded at 02:44 (S-VARIETY superseded by G1/mining) and 03:00
(remaining **566,400 sim-s released**). Subsequent clean3/mining/probe
allocations protect other phases; they do not restore the plan_fix reservation.
The last registration now records clean3's u1 fault and new probe tasks at 12:46.
Reopening this task cannot safely treat released funds as still reserved.

Do not run new SUMO or apply the historical switch without reconciliation by
MAIN/owner: explicitly restore a plan_fix allocation inside the current total
cap, identify the intended run/update boundary and confirm task ownership.
Existing prepared switch remains NOT READY: all four requested family counts
zero, admitted seed coverage empty, actual switch index UNSET.

## Step 2 — bounded worker status check succeeded

Historical own identities: CPU WM381108/start8612088 and resource coordinator
381105/start8612061, boot2c19986e-dd5e-4de9-95ca-3212b1ea0568. Last successful
status is 00:37; previous recovery's 08:44 F6 admission attempt timed out before
remote execution. Present state is UNKNOWN until a fresh read succeeds.

One read-only `planfix_deploy_cpu_worker.py status --generation 7` through the
preserved F6 shared admission transport succeeded at **12:50:05 +08**,
attempt 1/1, return code 0. Durable fresh receipt:
`PLANFIX_CPU_STATUS_v7_1793940606467549300.json`.

- WM381108/start8612088: `alive_same_identity=false`.
- Resource coordinator381105/start8612061: `alive_same_identity=false`.
- Bootstrap PASS6.314276s and query/export ABI preflight remain preserved.
- **0/8 startup gate receipts; no completed speed search.**
- Actual v2 coordinator terminal reason is
  `controller_error_no_retry:AssertionError()`. Its traceback is the unchanged
  clean2 STOP/FAULT assertion in `probe_health`, not a numerical startup-gate
  failure. The terminal receipt records that it stopped its own unfinished model
  at controller exit; the model is no longer alive at the current check. This
  audit does not establish who subsequently retired it or why.
- No `CPU_FAILED.json` exists in the returned inventory. The automatically
  appended status line's `failure=None` refers to that missing model receipt;
  it does **not** erase the resource coordinator's recorded terminal error.

No dispatcher, transfer, process signal, new search, simulator work, ledger
mutation or bank switch occurred. Existing source/input/chunk/result evidence
and the two pre-existing untracked files are preserved.

## Step 3 — disposition and handoff

**BLOCKED_AUTHORITY / NOT READY.** Zero translated bank survivors, zero newly
verified variety plans, empty admitted seed coverage, switch index UNSET.
New physical execution requires an explicit restored plan_fix allocation within
the current aggregate cap and a prospective run/boundary registration. The old
u12 switch cannot be applied retrospectively. Current WM qualification/search
is incomplete and no model is automatically restarted by this recovery.

Central `PLAN_FIX_READY.md` and `CLEAN1_STATUS.md`, plus the task's CPU handoff,
are updated with the successful fresh read and the authority conflict. Completed
qualification is reused; documentation/receipt changes are checked with Git.
