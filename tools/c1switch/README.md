# S-C1SWITCH operational receipt

## MAIN20:52 authorised immutable continuation (v7)

`c1908877` registers the corrected WM-only canary and total490860sim-s.
New roots: cpuQ `/home/ubuntu/c1switch20261031/main_v7`, gpuR
`/root/c1switch20261031/main_v7`. The v6 scientific adapter hashes are unchanged.
Original WM canary12450sim-s is superseded evidence and remains in accounting;
BB original episode+15branches are imported, conditional on all10 corrected picks.
Old STOPs and complete trees are retained. Do not use legacy `mainops` dispatch.

Use `python -B -m tools.c1switch.continuation progress` first on every resume.
Write-once operations: `wm_gpu`, `wm_parity`, `relay`, `canary` (WM only),
`bb_gpu`, `bb_parity`, `bb_reuse`, then `gate` and `main` only after all gates.
Inspect existing dispatch/PID receipts rather than relaunching any operation.
The GPU original-draw parity workers are diagnostic-only; live stochastic keys
are frozen anonymous observables. New services carry WM2557/BB1664s prior debit,
leaving8243/1936s; all loading/validation/idle counts. GPU hard caps remain3h/1h.
`V7_STAGE.json` binds the sources and imported71910sim-s. CPU28cpuQ/29gpuR PASS.
Validation now PASS: both full60s GPU parity gates; all10 BB reuse picks;
corrected WM1episode+15branches; exact paired warmup; all30live-prefix gates,
zero collisions/teleports, cumulative84360sim-s charged=actual. See
`V7_CANARY_GATE.json`, `V7_GPU_PARITY.json`, `V7_PAIRED_WARMUP_GATE.json`.
Main remains write-once and bounded by the original GPU hard caps; observed
WM mean81.59s/forecast creates throughput pressure. No partial-panel efficacy.

## Historical SCIENTIFIC STOP (v6 observable-key retention gate FAIL)

The following stop was superseded only by MAIN's explicit20:52 continuation.
`evidence/V6_CANARY_REVALIDATION.json` / `V6_FINAL_STOP.json` bind the stop:
WM j0 retains CTRL0, but corrected anonymous-observable sampling at j1/t360
picks PI instead of the original canary's MAPPO. The registered all20-identical
retention gate failed; no main dispatcher was created. This is a validation
failure, not an efficacy verdict. v6 original-draw GPU parity was not reached.

Actual=charged71910sim-s (Step047010+WM12450+BB12450); GPU allocation including
all loading/validation/idle phases is WM2548.557475s+BB1658.436705s=4206.994180s.
Own revalidator3800051 and GPU598515 exited; v6 BB was never allocated. All local
relays cooperatively closed; other tasks/processes untouched. v4/v5/v6 trees,
original model requests, failed gate and all charged ledgers are preserved.

No automatic retry/seed replacement/refund. MAIN must explicitly authorize a
new continuation namespace/budget before any corrected-policy physics. A WM-only
first-world repeat would add12450sim-s to planned478410 (=490860), leaving9140
reserve, provided BB reuse is separately validated; this proposal is BLOCKED.
GPU throughput and remaining per-model caps also need an executable gate.

## Completed integration evidence before the stop

Builds `db3d8683/4a4fe8cc` executed one paired world969112: real cross-host
WM2/BB K8, full600s control, all30full-prefix branches and clocks PASS,
charged=actual71910 cumulative, zero collisions/teleports. No repeated physics.
`evidence/MAIN_CANARY.json` is the mechanical gate, not a model-benefit result.
Original GPU services gracefully COMPLETE: WM1899.304206s, BB1339.300185s.
v5 full60s original-draw GPU sample parity PASS; no new SUMO. Both services
gracefully COMPLETE after319.14s each; v6 cumulative conservative debits2224/1664s
leave8576/1936s against original10800/3600 caps.

CPU-proven exact callback speedups are in `adapter.py/merge_fast.py`; v6 also
skips unused WM pivot quadrature diagnostics and memoizes deterministic BB K8.
Both still require original-draw full60s GPU sample parity. Native
PI/CIRCLES envelope is model-supplied when available, inferred-head otherwise.
The end-to-end audit found v4 CRN hashing included discarded detector IDs.
v6 removes them before both RPC and hashing. MAIN remains blocked until full60s
GPU original-draw parity AND all20canary picks under corrected anonymous keys
pass. If any corrected pick differs, STOP; do not repeat/replace the canary.
The two immutable v4 episodes/branches stay on disk regardless of this gate.

Prospective v6 main: cpuQ14episode+<=14branch workers nice19 (owner<=28), BB
first for its tighter GPU wall allocation. GPU services2.5GiB each,5GiB combined.
`mainops stage/launch_gpu/relay/parity/revalidate_canary/revalidation_status`
are gated, write-once operations; `mainops main` also requires both GPU gates.
Complete-only `analyze.py` checks all pairing/accounting before seed0/2000-draw
world bootstrap with best-fixed identity recomputed inside each draw.

## Historical cpuQ Step0 continuation (MAIN 18:33)

Design commit `ec322b36`, SHA256
`428f1850cc33350f4bc63c08e89b6270d859cd17a6662c660ddaa1977e478470`
is registered: total478,410 sim-s, including Step0 47,010; GPU4h hard cap.
Main16worlds, eight episode arms, no oracle episode; WM/BB172,500 EACH,
random/five fixed14,400 EACH. WM2b8797242+mix/full60s observable history and
BBafaaa32c minimize predicted on-road delay; evaluation is full fleet+source.
This MAIN rule supersedes the historical model/source and budget blocks below.

cpuQ immutable `/home/ubuntu/c1switch20261031/fullrerun_v3` is staged;
`evidence/CPUQ_STAGE.json` binds sources/imported930 debit. cpuQ13testsPASS,
28seedguardsPASS, newSUMO0. Existing frozen runtime/controller hashes verify.
No gpuR v2 dispatch/lock and no additional legacy debit were found on resume.
Dispatched ONCE after build `2608bb92`, cpuQ ownedPID3525930; first five-way
state969101/t300 PASS (native full-rerun+90repeat bitwise, all candidate prefixes).
Now COMPLETE16/16states,96/96new runs closed; ownPID exited; all bitwise gates
PASS and headroom gate PASS. Charged=actual47,010sim-s incl retained930; GPU0.
Unique60s wins native2/CTRL0 9/MAPPO3 +2ties; unique90s native2/CTRL0 7/
MAPPO1/PI3/CIRCLES1 +2ties. Repeat difference0.0; median margin1.8324/6.8538veh-s.
Report and CSVs: `docs/paper_results/C1SWITCH_WINNERS.md`, `c1switch/step0b_*`.
Use `python -m tools.c1switch.cpuops progress` for receipts; never re-stage,
launch again or restart. Initial Step0 one
nice19 CPU worker, CUDA hidden; MAIN permits <=28workers on cpuQ.
Main still requires actual Step0 bitwise/headroom gates and real observable
feedback inference adapters; source entries are no longer required in prediction.
`evidence/MODEL_SELECTION.json` records MAIN's fixed choice, not adapter validation.
`episode.py` commits each model pick before exact truth is computed, measures only
the registered three clocks, and checks all main branches against the LIVE prefix.
CPU core33testsPASS; host-specific model/transport adapters are still unvalidated.
Main capture now projects both native half-second ticks (CAV self, nearest
same-lane neighbour sensing, RSU zones) and requires all121ticks over60s.
Inference-side `observable.py` uses the existing anonymous BeliefTracker: true HDV
IDs are discarded before association, no true length/type or penetration enters.
Remote CPU19testsPASS, including HDV-ID noninterference and raw unseen-row
noninterference; local33PASS/1module skipped (no local torch). Existing cpuQ
Python used, no environment created, SUMO calls intercepted, GPU0.
These new sources live beside the immutable running Step0, not inside it.
`cpuops collect` exports exact completed-state receipts/observable conditions;
`python -m tools.c1switch.headroom` independently verifies complete16-state
prefix/repeat/accounting gates and emits CSVs, refusing a partial panel.

## Historical full-rerun staging (MAIN 17:53; never dispatched)

Design commit `74082ebd` supersedes saved-state execution below. Step0 v2 is
47,010 sim-s cumulative: 930 retained + 38,400 five-way reruns + 7,680 native
repeats. Eight unused reserved worlds, decision times 300/480, exact60/90.
`rerun.replay` rebuilds controller memory by executing the full prior schedule;
checks full warmup/trunk tape + belief + metric/counter fingerprint BEFORE any
alternative tick. Native90 repeated continuation passes before nonnative runs.
The legacy runner now refuses execution; old deployment/STOP/ledger preserved.

CPU: 13 task tests PASS locally and on gpuR, all36 seed guards PASS remotely,
ruff PASS. Immutable version `/root/c1switch20261031/fullrerun_v2` is staged;
`evidence/FULL_STAGE.json` binds its sources/design and imported930 debit.
No v2 SUMO dispatch: gpuR load41.61 exceeded20 admission ceiling. No queued
process or second job was created. Respect SUMOCF priority; resume by inspecting
`python -m tools.c1switch.fullops inspect`, then `... progress`.
Only when admitted: `python -m tools.c1switch.fullops launch` ONCE. Existing
dispatch/lock/START/STOP blocks retries; do not rerun `stage` or legacy `ops` jobs.

Main is hard-blocked by budget arithmetic: even an optimally reused live oracle
candidate needs626,400 sim-s for24worlds x10decisions, exceeding500,000 alone.
Prospective three-clock WM/BB measurement plan totals1,447,410 including Step0;
not approved. MAIN must resolve oracle count or cap before main implementation/
dispatch. Existing source-entry/model eligibility gates also remain unresolved.
New C1PIDIAG report is present, but explicitly entries-OFF/conditional presence;
it does not attest a valid fleet+source+entrant adapter for either candidate.

## Historical saved-state attempt (receipts only; commands retired)

Design `docs/paper_results/C1SWITCH_DESIGN.md`, commit `53e38608`, was registered
before analysis/SUMO; exact plan 498,360 sim-s, hard cap 500,000; Step0 59,160.
CPU verification: `python -m pytest -q tests/test_c1switch.py tests/test_command_ownership.py`
(25 passed); `python -m ruff check --select E9,F tools/c1switch tests/test_c1switch.py` passed.

Modules: `core` (selection, receipt boundary, handover, selected-trunk branching),
`sumo` (frozen-plant backend), `runner` (canary/Step0/nine arms), `rpc` (GPU file
service), `model` (prospective choice), `storage` (precharge/receipts), `ops`
(exposure/deploy/preflight/recorded exports).

Exposure audit `evidence/EXPOSURE.json`: exact registry identities/ranges clear;
C1GAP corpus IDs clear; gpuR 338,348 and gpuA 858,502 safe names clear. Forbidden
directories pruned before traversal, file content never read by remote scanner.
Frozen runtime verification hashes only allowed source files, not parameter or
assignment files. No seed is allocated before both frozen guards pass on gpuR.

Deployment: `python -m tools.c1switch.ops deploy`, then `... preflight`.
Canary on gpuR existing Python: `python -m tools.c1switch.runner canary` from the
staged `pytools`, nice19, CUDA hidden, existing conda bin on PATH. Exactly 480
precharged sim-s (300+90+90), native live vs restored full tape AND team metrics.
Any mismatch creates durable STOP. No Step0 panel/main after a failing gate.
No dispatch script auto-retries or refunds reservations. Old START/lock/STOP is
never removed; an interrupted job requires explicit reconciliation.

The real-model adapter is deliberately unbound: C1PIDIAG has not published its
requested final diagnosis; neither model's observable-source/entries adapter has
passed an integration test for this task. RPC accepts a source-hashed adapter
factory and MODEL_SELECTION binding only after those gates. Do not represent the
CPU framework as a validated WM/BB inference experiment or run switching early.
No GPU has been allocated. A source/entry adapter must sample unobserved arrivals
and source queues from observations/priors, without consulting the SUMO ledger.

The shared `REGISTRATION.md` is outside this Git worktree and updated additively;
never force-add it. No push, new environment, deletion or peer PID operation.

## Current stopped state

Canary passed (`00d94895`); Step0 dispatcher524236 then failed at first-world
CTRL0 snapshot,450sim-s after warmup/partial branches. Total930charged=930actual,
zero complete decisions,zero GPU-hours. Both owned PIDs gone. STOP is retained;
do not repeat canary or use the stale STEP0_STATUS=RUNNING as a live-process sign.
Snapshot repair is CPU-verified separately under `repairs/cc96a3c44efc`; original
deployed files remain unchanged. Full panel needs a prospective continuation
binding from MAIN. Details and paper figure: `docs/paper_results/C1SWITCH_WINNERS.md`.
