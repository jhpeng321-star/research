# S-C1SWITCH2 per-CAV import failure — Nov 2 fresh-session recovery

- Reconciliation commit `f4a4f416`; original design/panel/budgets unchanged.
- Original CPU-only observer20260 remains LIVE; no replacement or duplicate jobs.
- Observer admitted original cpuQ scope once at Unix1793538230, own2380230,
  `/home/ubuntu/c1switch2_run_10950e88a99994d0`, one nice19 worker/GPU0.
- Durable STOP on `cav_mappo__s969138`; two completed episodes,75 decision receipts,
  3000charged/2980closedactualsim-s. Original gpuR scope retains300charged/noCLOSE.
- Exact failure: `ModuleNotFoundError: traffic_shepherd.c4.switchable_wall` from
  the frozen per-CAV executor's direct `_new_wall` import on its first WALL group.
- Frozen v7 precedes the additive staged modules on the import path. Fleet WALL
  installs that extension lazily; mixed execution bypassed the fleet facade.
  The former CPU fixture imported WALL itself, masking this pristine-process gap.
- Correction: the common WALL facade exposes `prepare()`, installing the existing
  hash-bound additive paths and resolving the real controller class. Per-CAV
  backend invokes it BEFORE base backend/SUMO construction; selector CPU admission
  also exercises the actual start/stop lifecycle before any one-use dispatch claim.
- Regression removes the additive path and cached controller module, then executes
  the actual peer atomic-WALL group and exact fleet handback through BOTH optional
  ownership-hook ABIs. Before correction:2PASS/2FAIL with the exact production error.
  After correction: all156task CPU tests PASS; targeted Ruff PASS. A separate test
  verifies unresolved WALL dependencies refuse before physics backend construction.
- Immutable remote stages, failed worlds, original locks, charges and receipts
  remain unchanged. No terminal scope restart or replay was attempted.
- MAIN continuation request: exclude failed969138(cpuQ) and969139(gpuR); permit
  only the nine/three wholly unstarted episode worlds respectively plus all
  still-unstarted registered branches of the two successful cpuQ episodes.
  This is76840additionalsim-s/GPU0 (cpuQ62460/gpuR14380), including8120sim-s for
  the two completed worlds'12six-way+2held-joint full-prefix branches. Prospective
  immutable plan, no-prior-charge proofs and host CPU gates are required before
  dispatch; no authorization is inferred here and no corrected stage is admitted.
- Fresh aggregate250980charged/250660closedactualsim-s; finalizedGPU27787.12987509556s.
  All112fixed/random+16fleet episodes/96fleet oracles+36WM oracles remain COMPLETE.
  WM has6episodes only; full paired primary efficacy panel remains incomplete.
- OWNER20:14GPUhold remains ACTIVE; all task GPU services EXITED. No new GPU work.
  Existing CPU-only observer continues20min receipts/commits through Nov2 12:14+08.
