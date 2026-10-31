# LC2013 source contract — SUMO 1.27.1

Public immutable tag: https://github.com/eclipse-sumo/sumo/tree/v1_27_1
Read-only local source cache: `E:/Temp/opencode/sumo-1.27.1-rb-lane`.
No source code from that cache is executed. New Python implementation follows
the highway mathematical rules; it is not a complete SUMO simulator port.

- `src/microsim/lcmodels/MSLCM_LC2013.cpp`: 48–78 constants;
  99–150 defaults and derived thresholds; 1233–1245 lookahead speed memory;
  1294–1333 strategic urgency; 1502–1516 remaining time and negotiation;
  1547–1569 anticipated lane speeds and inconvenience;
  1643–1669 cooperative wish; 1694–1881 speed/keep-right accumulation;
  1902–1951 anticipated follow speed; 2056–2057 safety factor `1/lcAssertive`.
- `src/microsim/cfmodels/MSCFModel_IDM.cpp`: 218–220 secure gap;
  225–268 IDM internal integration used for accelerating leaders.
- `src/microsim/MSLaneChanger.cpp`: secure front/back checks also subtract
  follower minGap from bumper clearance; selected vehicle's safety factor applies
  to both secure gaps. No generic kinematic-braking gap substitution for IDM.
- LC2013 uses fixed-point wish counters with precision 10,000,000, threshold
  left `0.2/lcSpeedGain`, right `0.2/(lcSpeedGainRight*lcSpeedGain)`;
  defaults `lcSpeedGainRight=0.1`, `lcLookaheadLeft=2`,
  `lcSpeedGainRemainTime=20`, `lcKeepRightAcceptanceTime=-1` (legacy),
  `lcStrategic=lcCooperative=lcSpeedGain=lcKeepRight=lcAssertive=1`.
- `lcImpatience` is not used by LC2013's secure-gap factor: unlike SL2015,
  LC2013 returns `1/myAssertive` directly. Do not add a new waiting-time
  gap-relaxation heuristic. Its timing is the persistent wish and lookahead.

Validation must distinguish exact frame matches from time-window event matches,
report false positives/precision as well as recall, and use recorded motion only.
The historical MOBIL 1.4% came from a different panel; rerun MOBIL on the same
FIT observations for the paired comparison. Teacher forcing is a lane diagnostic,
not a long-horizon distributional qualification or counterfactual evaluation.

## Implementation scope (first step)
`physics/lc2013.py` ports fixed-point left/right/keep-right wishes, 0.9/s
lookahead memory, strategic route-end lookahead, early strategic return,
cooperative lane clearing, asymmetric speed gain and five-second keep-right.
IDM secure-gap safety includes each follower's minGap and ego's lcAssertive.
Anticipation uses the discrete safe-stop inversion (CFModel.cpp:860-882),
brakeGapEuler, and IDM internal substeps for accelerating leaders.
Blocking-follower Euler cut-in-help advice follows LC2013.cpp:712-791.
Only matching named inferred parameters map; all other LC parameters default.
`lcImpatience` does not change this LC2013 safety formula.

Explicit parity gaps: synchronous instead of downstream-first serial messages;
leader overtake/fall-behind/full follower negotiation;
arbitrary route best-lane metadata unless supplied by the caller; nonlegacy
keep-right follower adjustment; speed-gain lookahead extension (default zero);
nonhighway special cases. This is a testable highway subset, not full parity.

## Source completeness amendment (after preserving first full panel)
The SUMO default right-overtaking prohibition is now ported: positive left wish
and next-step own speed advice from LC2013.cpp:1335-1356 and
AbstractLCM.cpp:1119-1133; keep-right's cannot-pass-current-leader check from
LC2013.cpp:1748-1752. Congestion semantics are **SUMO's** existing physics rule,
`v < 60/3.6 OR laneSpeedLimit < 60.1/3.6` (MSVehicle.cpp:1494-1495), not an
invented behavioural cutoff. Default global overtaking-right=false; explicit
pure-module override available. Emergency classes and stochastic
lcOvertakeRight>0 are outside this default-population subset.

Extra public sources read only, downloaded into NEW EMPTY directories:
- `E:/Temp/opencode/sumo-1.27.1-lane-extra/MSVehicle.h`, SHA256
  `9e9f7b35cfb959dd9c12f01f0d651ab54dcb214f000bb9c54feb4b75ca33fee9`
- same directory `StdDefs.h`, SHA256
  `a44924b7205c63675e0eca79a3ca4591146941dd897b939ffa2be7100ee58243`
- `E:/Temp/opencode/sumo-1.27.1-lane-vehicle/MSVehicle.cpp`, SHA256
  `142eef6ef8dd16d6ece698f73ae2d34837fbb16f1ca6677629dc7e7fee5d966c`.
No downloaded code executed. No parameter fitting/selection from tape outcomes.
Revised-candidate panel reuses the completed MOBIL scores after exact input,
physics, cohort and horizon checks; the original candidate results are retained.

## Precision continuation amendments

- LC2013.cpp:1785 keep-right request must remain inside1716's non-slower-target
  branch. Corrected in568d0314; 255FIT false-positive steps removed, no TP lost.
- `lc2013_geometry.py` reconstructs edge-local position and finite LaneQ.length
  from the fixed corridor net and completed obsstate helper. LENGTH, not remaining
  distance, enters1547/1740/1809/1855; strategic1294/1303 subtracts position.
  MSVehicle.cpp:6163,6295 sum edge lengths; MSEdge.cpp:126 chooses first lane.
  Both-zero route offsets resolve toward the target at1144–1155, while1330/1874
  retain equality with the original offset for route-required strategic requests.
- Public `src/config.h.cmake:236/239` defines NUMERICAL_EPS=.001 and
  POSITION_EPS=.1. Earlier1e-6 in this lane subset was wrong. Candidate6 uses
  .001 for secure-stop numerical margin, IDM near-zero gap and speed-gain's
  positive-gain veto. This matches INT's independently discovered source values.
  Downloaded READ ONLY to NEW EMPTY `E:/Temp/opencode/sumo-1.27.1-lane-precision`,
  config.h.cmake SHA2565e48e2b4acfdde7860e6eed0ba3d0ffca529046693151cc65f82b2ad7bc214c3.

## Coordinate / sequencing amendment (candidate9)

MSNet.cpp:872/878/884 confirms plan→execute movement→lane change. The original
post-motion ordering stays. MSLaneChanger.cpp:682/760 uses raw edge-local lane
positions for same-edge neighboring gaps; upstream normalization must restore
the left/right1.45m origin difference. MSLane.cpp:3036/3136–3161 includes internal
link lengths during consecutive-leader lookup and gates search at the preceding
normal lane end, not the next leader's position. MSVehicle.cpp:5617–5712 preserves
raw position on lane change, requiring the corresponding normalized-s shift.
Both runtime adapter and scorer now respect that transform. The scorer's
explicitv2 contract prevents mixing cached MOBIL decisions from old coordinates.
Source download SHA/path and default-OFF bit-exact proof are in the iteration diary.

## Urgency / message semantics completion

LC2013.cpp:1661/1861 requests explicit follower help for mature left wishes;
lcSpeedGainUrgency is50 by default, not a fitted threshold. MSCFModel_IDM.cpp:82
uses max(comfortableDecel,min(emergencyDecel,1.5)) for minNextSpeed. LC2013.cpp:
283–300 checks raw advice against feasible bounds, then blends foreign advice
with wanted speed; own advice is unscaled. Explicit presence distinguishes a
zero-acceleration request from no message. Tests and decision-equivalence scope
are in the morning report and immutable receipts; complete negotiation is still
outside this subset's parity claim.
