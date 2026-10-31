# S-ENSPROBE execution and provenance

Worktree `E:/agent-work/c4-ensprobe`, branch `c4/ensprobe`.
Approval and prospective amendments: task `REGISTRATION.md`, master C4-V7FT-20261017 register, and `docs/paper_results/ENSPROBE_DESIGN.md`.

## Current owned deployment

**COMPLETE:** 80/80 branches, all80 physical gates PASS,49,869 charged=actual. Owned worker93751 exited. All K8 physical-tape hashes match within each state/arm; zero seed spread. Final paper report34lines,7CSVs and onePNG are under `docs/paper_results/`.

- cpuQ `/home/ubuntu/ensprobe20261031/recovery_v4`, one nice19/single-thread CPU worker, CUDA hidden.
- Immutable deployed `BINDING.json` lists exact source hashes; local `evidence/v4_{stage,cpu,launch}.json` binds source, CPU gate and owned PID.
- Prior original/v2/v3 namespaces and STOPs are retained. No redeploy/retry is implicit in the progress/collection commands.
- Priority admission uses spare capacity: pause on task-local YIELD or load>20 (32core host); explicit STOP cooperatively exits our worker. No peer PID changes.
- Exact cap usage on completion: 48,000 scientific branch sim-s + 1,800 native prefix sim-s + 69 scheduling-aborted partial sim-s = 49,869; 131 unallocated.
- Original first600 reservation was resumed after0 ticks; second600 reservation combines its remaining531 with69 newly precharged reserve after the scheduling stop. Existing completed native600 result is SHA-bound and reused. No refunds.

## Commands (run from this worktree)

1. `python -B -m tools.ensprobe.ops v4_progress` — read-only current status, no dispatch.
2. After terminal COMPLETE: `python -B -m tools.ensprobe.ops v4_collect` — permitted JSON receipts only; complete-panel gate.
3. `python -B -m tools.ensprobe.analyze` — unchanged AST-extracted LRDIAG BIC, CSVs/onePNG/≤50line paper report; no SUMO.
4. `python -B -m pytest -q tests/test_ensprobe.py tests/test_ensprobe_analysis.py` — eight focused CPU checks.
5. `python -B -m tools.ensprobe.analyze --verify-outputs` — independent CSV arithmetic, dimensions and canonical-LF artifact hashes; `--verify-staged` checks those hashes against Git's staged blobs before commit.

`collection.json` exports physical kinematics/routes/queue checks, bins, objective clocks, accounting, immutable bindings and SHA256s. It never reads scenario assignment/truth or saved-state XML contents. Native SUMO necessarily consumes its own generated inputs internally. File content inspection is confined to permitted source/configuration and public output receipts.

## Mechanical history

1. Initial pending-queue gate stopped at0 continuation ticks: `getPendingVehicles()` cache36→0 after load, while36/36 actual due waiting objects remained loaded/off-road/not departed with correct routes. Direct waiting-object gate added; cache mismatch preserved.
2. One extra300 native prefix exactly reproduced original physical trunk. One native600 continuation completed.
3. A second0tick stop caught deepcopy memo reuse; fresh-memo fix and zero-tick first-state reconstruction added. Frozen `env.state()` is current-frame-only; no native history enters MAPPO. Retained prefix bins remain exact. Later four worlds use full native histories.
4. Overstrict any-running-peer yield on32cores caused starvation despite load≈1; owned cooperative stop retained69partial sim-s, then bounded spare-CPU admission resumed collection. All mechanics were registered before scientific outcome inspection; seeds/arms/detector/objective unchanged.

The ensemble conditions on the same SUMO-restored physical state, realized demand and driver population. It does not claim latent-memory identity with an un-restored live trunk, or identify real-world uncertainty conditional on limited observations.
