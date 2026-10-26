# S-RESTART — WAIT_OWNER (2026-10-26)

**Restart not launched. The continuation-wide second operational failure stop rule was reached during local worktree setup.** No new SUMO, seed allocation, simulation charge, trainer, collector or detached driver was started. No task-owned remote PIDs exist.

## Completed before the stop

- Read NATIVEINIT.md, RESUMEPREP.md and its exact RUNBOOK/control/staging implementation, LONGCRITIC.md, current CYCLE0_RESULT.md/C0DIAG.md, FOLLOWUPS items7–10, R17ENTRY.md and the latest shared registration entries.
- Appended the owner-authorized S-RESTART amendment to `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md` **before integration or launch**. Exact source/config/candidate smoke bindings remain unbound; this is not dispatch authorization without the prescribed gates.
- Created branch `c4/restart` at `c4/mainstart` tip `1aee086ecb667c69895c157b9ce9e01bb308801c`. No source merges were attempted.
- Read-only SSH probes: gpuA RTX4090D49140MiB,952MiB used/47559MiB free; `/root`5.3GB free. gpuR `/root`12GB free; old `/root/main14_loop/LOOP14_STATE.json` and staging helper directory present. No remote files/processes were changed.
- Host decision: gpuA under the latest owner plan. S-RESUMEPREP matched microprobes show policy7.836s R/8.616s A and WM60.774s R/57.771s A, no decisive overall winner;48GB headroom favors A. Planned main policy/WM caps0.20each, combined numerical threads<=10; peer round17 settings untouched.

## Operational failures and recovery for documentation only

1. Fresh `git worktree add --no-checkout -b c4/restart E:/agent-work/c4-restart c4/mainstart` succeeded. Sparse setup followed by `git restore --worktree --source=HEAD -- .` attempted a wider checkout and failed with **No space left on device**. No user work existed in this new worktree.
2. `git read-tree -mu HEAD` recovery failed again with **No space left on device**, plus already-present-path diagnostics. This exhausted the authorized single operational retry; the stop applies across setup as well as runtime phases. No launch retry is permitted in this continuation.

The task-created incomplete checkout was removed using `git worktree remove --force` (only this fresh worktree), retaining branch `c4/restart`. For this receipt only, a new no-checkout worktree on that same branch was initialized with **`git read-tree HEAD` without `-u` before sparse selection**, then limited to `/RESTART.md` and `/tools/restart/`. This restored clean index semantics without another source checkout or launch attempt. Post-cleanup E: free space was only **3,772,416 bytes**. No other worktree, unrelated file or peer process was modified; no reset, clean, stash or push was used for recovery.

## Gates and accounting at handoff

- Integrated tests, box staging, prearm, mini-policy, native-init real mechanics, rejected-checkpoint veto demonstration, smoke/conversion/two-step WM mini, MINI_PASS and detached main launch: **NOT RUN**.
- S-C0DIAG-SMOKE-1 remains unconsumed by this task: **0/1,922 sim-s dispatched**. Its prescribed seeds968142/968143 still require the live-ledger recheck immediately before reservation; no claim of current availability is made.
- Historical accounting remains the supplied87,399charged/28,353actual, cap210,000. No state/ledger mutation was made. The rejected `fa2427772a49e9462c0be3172e402c515616a15e2930a263e1eafeacb23a0d5d` remains historical, never adopted here. Accepted base remains round16 SHA`5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2`.
- Native-dispatch audit reached `src/wms/control/dreamer.py:860`: nativeinit production code passes `self.counterfactual.native_cav` explicitly; `confident_gate.py` also passes explicit mode/head settings. These observed callers do not exhibit config-object-to-HOLD fallback. Collection replay and remaining callers still need the complete integrated audit; no R17ENTRY patch/flag was activated.

## Required next authorization / remaining exact sequence

Resolve local disk capacity within authorized scope and explicitly authorize a new continuation after this second failure. Install the recorded amendment into the **actual successor loop REGISTRATION.json** (not yet created), retaining87,399charges, exclusions, seed history and the210,000total. Merge `c4/nativeinit`93426d22, `c4/resumeprep`ab876da4 and `c4/longcritic`7a210420 on this branch; resolve native barrier/historical-generation interactions and run the integrated offline suite before deployment.

Audit all native-reference/residual/replay callers; use R17ENTRY exact explicit dispatch wherever config objects are passed, without changing learned drivers. Keep LONGWM loss/uncertainty OFF. Fresh zero residual including native lanes, policy_ramp_merge alignment,30measured updates, SHA-bound native-relative veto and first-four-world300s scientific guard remain required. Imagination initially10s, ceiling60s with CURRENT-WM10/30/60s registered gates;90s sign evidence alone cannot authorize expansion.

Enable LONGCRITIC/native paired B300s targets and registered per-current-policy share selection (measured1.0),30–60s persistent exploration,300s SINGLE-CAV+GROUP branches,7,800primary long sim-s/8,000combined-attempt cap,36,933primary/cycle. Enable approved safe speed-ups/immutable-WM overlap,4.5h WM timeout **and** trainer max-hours/registered allowance,40steps unchanged. Stage from boxes per RUNBOOK into new task-owned directories; verify CPU executors before prearm. Recheck smoke seeds, append exact hash bindings, charge1,922before dispatch, run once, verify native initialization trajectory/commands/speeds/lanes/merges and fa242777 veto, conversion/two WM steps and matching MINI_PASS. Only then launch detached new cycle0 and verify exact PID/start tick and first scientific gates. No automatic extra SUMO replay; scientific failure or second operational failure WAIT_OWNER.

Machine-readable stop: `tools/restart/WAIT_OWNER.json`. Shared registration amendment and stop entries are append-only and remain outside this worktree commit.
