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
leader overtake/fall-behind/full follower negotiation; right-overtake prohibition;
arbitrary route best-lane metadata unless supplied by the caller; nonlegacy
keep-right follower adjustment; speed-gain lookahead extension (default zero);
nonhighway special cases. This is a testable highway subset, not full parity.
