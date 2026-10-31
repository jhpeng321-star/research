# S-SUMOCF artifact check — HOLD

Owner authority: MAIN 2026-10-31 17:17 registration. This supersedes the
earlier launch permission until the artifact check passes.

## Recovery and hold

- Preserved clean inherited HEAD `7a5e2da8` and all prior results.
- Exact owned supervisor `517331/start4206463446` was running update-1
  simulations; the 48-job DEV MAPPO/CTRL0 baseline had finished.
- No `UPDATES.jsonl` existed: no optimizer update had completed.
- Wrote `HOLD_OPTIMIZER.json` and used the existing `STOP` mechanism to
  drain active workers, discard pending dispatch, and fail before the optimizer.
  No process was signalled. In-flight/precharged work remains charged.

## Existing receipts first (zero SUMO)

`tools/sumocf/evidence/ARTIFACT_EXISTING.json` preserves every eligible original
probe sample's D_i, D_G, C_G, executed-command deviation RMS/absolute sum,
and unchanged-command share. Across those rows, corr(D_G, deviation RMS)=0.534
and corr(|D_G|, deviation RMS)=0.572; these pooled associations do not establish
causality or validate the credit. The old traces contain no pre-safety command,
so exact safety-clamp share is **not reconstructible** from those receipts.
Unchanged-command share is reported separately and is not called clamp share.

Audit refinement (also zero SUMO): the exact original raw command **can** be
reconstructed from the saved sampled latent, hash-bound [-3,2.6] decoder bounds,
NumPy tanh and float32 assignment in frozen frame_to_state. This supersedes the
initial missing-raw limitation above. ARTIFACT_EXISTING_RECONSTRUCTED.json
contains per-sample exact safety-clamp shares; the first three pair10s rows all
have clamp share1.0. The original audit is preserved, not silently overwritten.

Code inspection found that the original sampled latent was an **absolute
branch-start latent held for the entire window**, whereas B recomputed its
mean every second. Thus setting the original noise to zero would still hold
the initial mean instead of following the current policy mean. The intervention
also decoded via NumPy float64 while the MAPPO decoder uses Torch float32;
bitwise zero-noise equivalence requires the identical decode/command path.
These are candidate defects to reproduce with CPU tests and two TRAIN states.

## Saved-state zero-noise gate FAILED — root cause confirmed

Both corrected zero-noise saved-state samples fail. At TRAIN969203,t465,
all loaded subsets have J600106131.25161385666 versus live B107364.59136128555:
false DG+1233.339747428894. At TRAIN969204,t596, loaded J60040899.11289396204
versus live B39343.32949724115: false DG-1555.7833967208862. Every loaded subset
has the same trajectory hash, independently of membership. The first physical
step differs even though issued actions are identical. These are the dominant
original probe values; **the original SNR/CG findings are invalid artifacts**.
The t300 canary passed but did not validate these later sampled states.
Evidence: ARTIFACT_RESTORE_FAILURE.json (full observable first-step comparison).
No further saved-state SUMO, no optimizer update. Use the originally registered
full same-seed rerun fallback, including all prefix costs.

## Pending fallback gate

Two-state zero-noise G==B bitwise; same-state small/large-noise comparison;
corrected-engine clamp/deviation receipts; CPU regressions; a corrected probe
of at most 100,000 sim-s inside the original 8M TRAIN cap; raw-D_G SNR
reselection and registration. No resume is authorised by this document yet.

## CPU correction

Replaced absolute-latent/manual NumPy decode with persistent **offsets around
the current mean** through the original Torch head and safety/memory execution.
Zero offsets skip arithmetic; current mean/raw/executed/clamp diagnostics are
logged. Added HOLD checks before dispatch and immediately before optimizer.step.
14 CPU tests PASS/2.64s on gpuR, no new environment or SUMO, including changing-
state bitwise action/memory equivalence, exact nonzero-offset decoder, hold and
the nested100k artifact precharge cap. Receipt: evidence/CPU_ARTIFACT.json.
Prospective gate and corrected-probe bounds are registered in SUMOCF_DESIGN.md:
<=86400sim-s combined, with100000hard cap, all inside original TRAIN8M.

Fallback prospective bound supersedes that saved-state plan:7061failed sim-s
retained+<=36000full-rerun gate+<=36000reduced probe=<=79061<100000. Reduced
probe fixes rho.5 and all three windows on the same two previously bound states;
all-zone exact identity aliases reused, other rho/size4 cells explicitly NA.
