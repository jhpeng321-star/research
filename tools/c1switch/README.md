# S-C1SWITCH operational receipt

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
