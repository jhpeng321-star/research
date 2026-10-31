# S-C1SWITCH operational receipt

## Current cpuQ continuation (MAIN 18:33)

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
Use `python -m tools.c1switch.cpuops progress` for receipts; never re-stage,
launch again or restart. Initial Step0 one
nice19 CPU worker, CUDA hidden; MAIN permits <=28workers on cpuQ.
Main still requires actual Step0 bitwise/headroom gates and real observable
feedback inference adapters; source entries are no longer required in prediction.
`evidence/MODEL_SELECTION.json` records MAIN's fixed choice, not adapter validation.
`episode.py` commits each model pick before exact truth is computed, measures only
the registered three clocks, and checks all main branches against the LIVE prefix.
CPU core33testsPASS; host-specific model/transport adapters are still unvalidated.

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
