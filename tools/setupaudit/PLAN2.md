# S-SETUPAUDIT2 plan and immutable scope

Start: c4/distill f84b34a9 (documentation successor of requested b56444ab).
No training, SUMO, push, destructive git operation or forbidden data access.
GPU: gpuR only, initially idle (2026-10-30 14:46); memory fraction <= .90.

1. Restore update-190 TRAIN sampler; draw 24 permitted starts spanning episode time,
   demand and penetration. Bind checkpoint/config/start hashes and sampler selection.
2. Score mix-ON paired CTRL0/MAPPO using the prior cap/Euler/native adapter,
   K8 common futures; take 90/300/600 s prefixes if 600 is executable.
   Preserve per-start/future values and paired world-cluster bootstrap intervals.
3. Cross longitudinal and lane controllers on the same evolving geometry;
   freeze CTRL0 admission/source tapes as a separate diagnostic if supported.
   Decompose road/source, spatial zones, CAV/HDV, initial/entrant and time bins.
4. Compare real definitions from named historical documents:
   CTRLARM.md @266d5e37; E3FINAL_A.md @69b4524d. Both include source waiting
   and all requested vehicles. Distinguish physical J from weighted suffix/q10
   training credit and from the older t0-HDV-only qualification target.
5. Diagnose only measured mechanisms; write <=90-line SETUPAUDIT2.md, bind final
   audited code/evidence SHA, and commit each completed step. Never push.

The real CTRLARM MAPPO-CTRL0 estimate is -623.95 [-2044.25,664.51] veh-s:
its interval includes zero. E3FINAL_A is C1, not CTRL0, on different worlds.
