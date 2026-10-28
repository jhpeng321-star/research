# S-C4PREP-1 — build verified; empirical preparation BLOCKED (2026-10-28)

## Registered scope (retained from initial receipt)

Prepare parallel gpuA cycle-4 candidates A (cycle1 c01.a3, SHA64f70af4)
and B (cycle3 c03.a6 best, SHA0978aa72), identical 150-update settings,
LONGMEM microbatch1/segment8/host offloads and allocator cap0.45 each.
Entrant ON/OFF variants require separately compatible native-anchor artifacts.
Only bounded two-update dry runs are authorized here. No SUMO or policy-phase
launch. The initial ensemble-binding requirement is superseded below.

## Authority and recovered work

Registration was appended before preparation in the authoritative
`E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md` and committed
as `3951042b`. Prerequisite merges already exist: anti-exploitation `de3b6a38`
via `1deed3d6`, entrant `875cc443` via `719f9485`. Existing inspection helpers
are preserved. No prior C4 preparation runtime/job directory exists on gpuA
at recovery; no fitting/training job was dispatched by this recovery step.

MAIN13:40 amendment supersedes the original ensemble instruction: pessimism
is current-WM3 K=8 paired spread plus the native anchor. No older-checkpoint
copies or cross-recipe ensemble disagreement.

## Live prerequisites at recovery

- WM3 `41e18f09b6060a1f4e4039f02655fcbcd497e32a360cffb71be716fffeaadc84`
  has measured `checks/c03.a2/ACCEPT.json` decision ACCEPT. The registered
  horizon gate is still running; the authoritative status/loop boundary has
  not yet advanced from WM2. No guessed horizon is usable.
- S-ENTRANTFIX current branch tip remains `875cc443`; final factorial/adoption
  verdict is pending. Its final tip must be merged and bound before launch.
- gpuA has 187,488,133,120 free bytes and GPU use 1 MiB at recovery.
- Historical real returns expose start-state `entry_backlog`; converted paired
  episode histories lack complete queue/request/CAV clocks. A genuine replay
  must use observable inputs and correct historical actor bindings, not scalar
  saved predictions duplicated into eight futures.
- Historical actor identities: C1 `64f70af4`, C2 `44f3eb66`, C3 `0978aa72`.
  Fit worlds C1/C2 and validation worlds C3 remain fixed by registration.

## Execution boundary

Preparation, genuine offline inference/fitting and two-update checks only.
No SUMO or full policy phase is authorized here. Shared chain stop/budget
markers and peer runtimes/processes remain under their respective owners.
Real finite/memory dry-run receipts remain blocked as detailed below.

## Implemented preparation

- `tools/c4prep/launch.py`: two parallel, separately resumed candidates A/B;
  isolated outputs and owned PID/start-tick receipts; five-second owned-process
  GPU-memory sampling; allocator/process-memory verification against cap0.45.
  Final WM/gate, source, actor, anchor and entrant bindings are checked before
  dispatch. No collection, WM-FT or shared chain-state mutation path.
- `configs/c4_policy14_c4prep_{on,off}.toml`: same150-update budget,
  microbatch1/segment8/both host offloads, full residual actor training,
  q10/K8/native anchor, uncertainty's old std term OFF; probes OFF.
  Looplong supplies direct real credit and disables critic bootstrap. The
  frozen native-relative panel/selector and signal gate remain unchanged.
- `policy14_cli --prep-dry-run-updates 2`: two actual scratch update calls,
  finite checks and memory telemetry, then STOP with `completed_dry_run` and
  no adoption receipt. Full-phase invocation still uses150 updates.
- `tools/c4prep/calibrate.py`: exhaustive observable-input preflight and
  guarded genuine current-WM paired inference/fitting path. The inference path
  has NOT run: missing clocks refuse before any GPU inference. No fitted scale
  or empirical unbiasedness verdict exists.
- Immutable exact-byte source snapshots are SHA verified. Staging reuses
  verified source bytes from gpuA's existing runtime without modifying it;
  only source deltas cross SSH. No WM checkpoint is copied.

## Measured receipts and blockers

gpuA root: `/root/autodl-tmp/c4/restart20261026/arm/loop/c4prep`.
`CURRENT_RUNTIME.json` identifies the immutable runtime; `STAGING_*.json`
preserves each staging receipt. Existing peer processes are not discovered or
signalled. Neither failed large-upload attempt created a runtime or compute job.

`INPUT_PREFLIGHT_1.json` SHA
`e320e30e2e6f515cda15d3ada50826f63ace5df9edd41149b8b1bb5da0df1870`:
all78paired starts/156episodes inspected; all three historical actor SHAs
verified; all156 live episodes lack the complete recorded queue/request/CAV
clock set. Start backlogs and full300s real-return totals exist, but do not
identify pre-t0 requested demand or partial-horizon source-inclusive costs.
C1/C2 backlog re-exports exist separately; they cannot supply C3 validation.
The prior backlog reconstruction requires `scenario/requests.rou.xml`, a
route/type-assignment input prohibited by this task; it was NOT opened.
Read-only cpuP/cpuQ observable-tape checks both failed at SSH connection close.

The first corrected isolated CPU suite passed **13/13** in1.39s, including
normal-path preservation, the exact two-update stop, configuration parity,
restricted-path refusal and unresolved-input refusal before dispatch. This is
control-path verification with test doubles, NOT the requested real GPU dry run.
Local fatal-error Ruff passed. The laptop has no torch; authoritative tests run
on gpuA with CUDA hidden. Earlier failing test/setup receipts remain preserved.

Final source `36a0baec4267` was restaged with1180 verified files; the updated
13-check suite passed in1.33s (log SHA
`7686df86725b79215adde82b9b6c7728e1e42010cb4b115f0447d00ea308763f`).
Compact local receipt: `tools/c4prep/evidence/FINAL_VERIFIED.json`.

WM3's registered gate remains incomplete, and the final entrant tip/verdict
has not landed (current committed tip875cc443; peer uncommitted work preserved).
Therefore **anchor fitting, both real two-update GPU/memory checks, final entrant
merge and launch readiness remain BLOCKED**. GPU use was1MiB at the receipt;
charged264544/280000 and STOP_DRIVER remain unchanged. No preparation compute
coordinator, SUMO or full policy job is running from this task.

## Exact one-command MAIN launch per mode (not executed; prerequisites required)

Run from `E:/agent-work/c4-c4prep` after final integration and verified dry runs:

```text
python -B tools/c4prep/main_launch.py --entrant off
python -B tools/c4prep/main_launch.py --entrant on
```

Each command resolves the current immutable runtime, validates the corresponding
`resolved_OFF/CONFIG.json` or `resolved_ON/CONFIG.json`, and detaches ONE owned
coordinator for the TWO parallel A/B candidates. MAIN selects ONE mode from the
final entrant verdict. Missing resolved configs, calibrated active horizons,
final entrant integration or matching dry-run receipts refuse launch.

## Remaining prerequisite sequence (no automatic retry/dispatch armed)

1. Obtain allowed observable queue/request/CAV clocks for C1–C3 without reading
   restricted route assignments; export a fresh immutable paired dataset and
   preserve original observable arrays, pair hashes and real-J reconciliation.
2. Re-merge final `c4/entrant`, bind its final verdict/main choice, stage a NEW
   immutable source snapshot, and resolve its WM3 gate compatibility separately
   for ON/OFF. Never assign the OFF gate to ON by assumption.
3. Generate genuine paired K8 futures at every active horizon using each
   historical collection actor, current WM3 and matched causal inputs/CRNs.
   Fit C1/C2; validate disjoint C3 with5000 world-bootstrap draws,
   seed20261028. No enablement when its interval excludes zero. No claim that
   an interval including zero proves unbiasedness.
4. `launch.py prepare --config CONFIG_<MODE>.json --anchor <artifact>
   --entrant-final-receipt <integration-receipt>` creates exclusive resolved
   config/boundary files. The integration receipt must bind `task=S-ENTRANTFIX`,
   `final=true`, the final merged/staged `merged_source_commit`, and MAIN's
   `selected_entrant_control`; it does not replace the underlying final verdict.
5. Run `launch.py dry-run --config <resolved config>` separately for each mode.
   Only two scratch updates per candidate are permitted; retain finite metrics,
   effective flags and measured memory in `DRY_RUN_VERIFIED.json`. Both candidates
   must pass before the corresponding MAIN launch command is usable.
