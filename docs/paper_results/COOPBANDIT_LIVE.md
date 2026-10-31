# Registered live test implementation receipt

Scientific plan remains COOPBANDIT_DESIGN.md SHA a7178a97, lines59–67.
Fresh dev981080–981091; full300state TRAIN fit and nestedCV calibration only.
The registered all-state GBT/radius policy is frozen before any live episode;
the companion available-only GBT/linear diagnostic cannot replace it.

`tools/coopbandit/live.py` implements the fixed reservation:
- 12 MAPPO +12 LEARNED episodes, warmup300/scored900, 28,800sim-s;
- baseline-t300 restricted four-option exact oracle, 43,200sim-s;
- first qualifying non-O0 event at<=540 in first8worlds, reference+2solos,
  <=27,360sim-s. Factual later decision schedule/targets are fixed in twins.
No event means NA and no credit physics. At an early event the full1140s
worst-case reservation remains charged while actual ends at event_time+600.
Total ceiling99,360; no implicit retry/refund and no additional pilot physics.
The oracle is a single-decision upper bound, not an oracle closed-loop episode.

All attempts reuse original backend/pre-safety/native lanes/noise and require
bitwise paired trunks; all-O0 episodes must match MAPPO clocks bitwise.
`live_ops.prepare` binds source/design/approval/final-policy hashes and allocation.
Eight nice19 single-thread CPU workers, load<=24/disk>=2GiB/memory>=8GiB.
`live_report` reconciles actual/reserved charges, paired J600/J900 and whole-world
2000-draw95% CIs, safety, exact event C_G, and S-COOPBEHAV descriptive detectors.
Detector signatures never assert unmeasured native-release attribution.

CPU QA: immutable cpuQ `/home/ubuntu/coopbandit20261031_liveqa2`;
20/20 tests PASS, zero skips,1.465s, TraCI start/step patched to fail (newSUMO0).
Includes actual orchestration with fake plant, all15decision clocks, paired
bitwise refusal, event solo isolation, exact report/bootstrap/credit arithmetic,
worst-case reservation vs shorter actual and incomplete-charge no-replay gates.
Initial local full-suite attempt exposed Linux-only fcntl/missing staged probe
paths; final verification ran on Linux with the genuine frozen runtime imports.
Existing S-COOPBEHAV cpuQ reporter imports and reads the public existing baseline
tape successfully (840 recorded frames/2 descriptive events); zero new physics.
Dispatch still waits for full labels/fit; retain first partial135 publication.
