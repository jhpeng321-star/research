# S-SELMAPPO2 — per-CAV MAPPO controller picker

## Status (2026-11-01)

Implementation and CPU integration complete; production training has not yet
been dispatched. Learning curves and final evaluation are pending, not zero-valued
results. Frozen final169 is the only eligible evaluation checkpoint.

## Registered experiment

`SELMAPPO2_DESIGN.md` SHA256
`09499f2af60311b3690a940dca8d94605d710465afb9c9eb05d5b0a79b23900c`.
Shared REGISTRATION.md contains the exact advance-approved per-CAV cap of
**7,012,800 sim-s**:6,996,600 production +1,800 mechanics +14,400 evaluation
reservation. The evaluation reservation remains unstarted because S-C1SWITCH2
owns the panel physics and additional J900 tail. Training is169x46 fresh900s
worlds; optimization seed1; one training replicate; no retries or top-ups.

Thirty decisions at300..880 every20s, six-controller categorical library including
WALL; full-team road+source delay reward, gamma=lambda1. Shared environment,
features, PPO and runtime are imported from S-SELMAPPO; local code only owns
per-CAV dispatch/accounting and the atomic consenting WALL roster.
One frozen WallLaw group selects its eligible cross-lane pair; consenting
nonmembers use CIRCLES, other CAVs keep their selected controller. Active-member
withdrawal/departure/re-entry interrupts the group atomically. No new behavioural
thresholds. WALL receives current observable receipts; missing junction flags
and traffic outside sensor coverage are not filled from truth.

## Verification and source binding

- Local56/56 CPU tests pass, Ruff passes.
- cpuQ56/56 CPU tests pass, zero skips; real CTRL0/MAPPO checkpoint constructors,
  frozen WALL formation/preservation/stop, per-CAV frozen seed aliases,
  30x20s reward windows and actual shared PPO learning pass with SUMO intercepted.
- Registry allocation audit passes. Pruned metadata-only exposure census:
  cpuQ185,978 safe filenames; gpuR303,475; no conflicts or traversal errors.
- Immutable staged binding:
  `b42c257b09813c175fcb7261e341c7599d0bb5c4c971529e2856890d2fc8038c`.
  Parent:`/home/ubuntu/selmappo2_20261101/`; build name is `build_` plus this hash.
- Earlier CPU-only stage35514f91 failed because its private test import omitted
  the frozen baseline search path. Original stage/evidence retained; no physics,
  budget reservation or training occurred there. Corrected56-test stage passed.

## Learning curves and evaluation

Pending training. Supervisor records per-update J600 mean/std/range, episode
return, PPO loss/KL-stop, sim-s/hour, actual/reserved charge and checkpoint SHA256
in `training/training.jsonl`. Initial000 and every001..169 checkpoint are immutable;
`FROZEN_FINAL.json` binds the final policy and sources. One worker completes both
smoke worlds before expansion to at most12spare nice19/single-thread CPU workers.

S-C1SWITCH2 panel969136..969151 is disjoint from production18100000..18107773 and
smoke18107998/18107999. The peer's complete hash-bound J600 receipts will supply
evaluation and paired WM/BB/six-fixed contrasts (2000world bootstraps, seed0,
95%CI, best-fixed reselected per draw). No partial-checkpoint efficacy claim;
single-training-seed uncertainty and richer per-CAV action space will be disclosed.
Frozen speed-MAPPO library pretraining and WM/BB data/compute are separate costs.

Evidence is under `docs/paper_results/selmappo2/`.
