# S-SUMOCF artifact check — PASS for full reruns; corrected learner launch pending

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

## Historical pending fallback gate

Two-state zero-noise G==B bitwise; same-state small/large-noise comparison;
corrected-engine clamp/deviation receipts; CPU regressions; a corrected probe
of at most 100,000 sim-s inside the original 8M TRAIN cap; raw-D_G SNR
reselection and registration were required before resumption.

### Full-rerun gate completed

Both TRAIN states pass: all five zero-offset subsets reproduce B **bitwise in
J600 and full trajectory SHA**, every D_i/D_G/D_all/C_G/M_i exactly0.
On969203, noise scale.25 ->4 increases issued-safe deviation RMS.0532 ->1.4788
m/s², realized acceleration RMS versus B.0510 ->.3680, and realized speed RMS
.0444 ->.7036m/s. DG changes+1233.63 ->-2352.41veh-s; its magnitude increases
but sign reverses, consistent with nonlinear long-horizon effects rather than
a monotone reward. CG is-.00590 ->-.10147veh-s, not beneficial cooperation.

On969204, issued-safe RMS.00650 ->.08104 grows12.46x, but **realized physical
trajectories remain exactly B**, and DG/CG stay0. SUMO's final execution caps
erase these command changes; this is a genuine no-effect intervention, not a
branch artifact. Log issued and realized effects separately. Safety-layer clamp
share remains96.6%, versus66.3–80.8% on969203; erased-exploration fractions are
reported separately in ARTIFACT_REVIEW.json. Thus nonzero issued deviation alone
does not imply a physically effective intervention.

All full-rerun gate jobs cost33915sim-s; including the failed saved gate,
artifact charged=actual40976sim-s. Corrected six-sample probe is now running;
optimizer remained held throughout. Noise scale1 (original sigma.3) remains fixed for probe
and training; measured small/large scales are diagnostics, not reward shaping.

### Corrected probe and decision

Six complete samples cost31905sim-s; total artifact72881charged=actual<100000,
including the failed saved-state gate. No completed simulator job was repeated.
An analysis-only alias check initially failed when reconstructing group geometry
from normalized actor inputs; it now verifies the exact observable baseline trace
at t0. All six receipts reused unchanged, dispatch disabled during recomputation.

Full27-label machine table: evidence/ARTIFACT_PROBE.json. Distinct pair/rho.5 rows
(two worlds each; means/SD in veh-s):
-10s: DG0/0, pooledDi0/0, CG0/0, rawSNR0.
-30s: DG.000719547/.001017594, pooledDi.000359774/.000719547, CG0/0,
 rawSNR.7071067811865475.
-60s: DG.001293659/.001829510, pooledDi.000646830/.001293659, CG0/0,
 rawSNR.7071067811865476; meanMi.000646830.
All-zone rows are exact pair aliases; size4 missing, other rho not sampled under
the prospectively reduced cost plan. Every row costs5317.5sim-s/sample on average.

The exact registered maximum selects **pair/60s**, nominalrho.5; training still
samples rho0/.5/1. The30/60 SNR distinction is floating-point roundoff, not a
scientific difference. The SNR arises from one tiny nonzero effect plus one zero;
CG is0 on both worlds. No useful cooperation/efficacy signal is established.
There was no registered SNR cutoff. Mechanics PASS authorises the pre-approved
learner after new launch binding; it is not a positive scientific finding.
ARTIFACT_PASS.json records this distinction. CVbeta0, originalsigma.3 retained.

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
