# HANDOFF — rbh-lcspec (LC2013 spec + test vectors)

Status: DONE (2026-11-01 ~01:00).

## What exists
- `docs/rbh_lcspec_LC2013_spec.md` — implementable spec of SUMO 1.27.1 MSLCM_LC2013:
  pipeline, LCA_* bit values, state, parameter defaults (incl. all hard-coded constants),
  strategic/speedGain/keepRight/cooperative formulas, secure-gap math (base + IDM override),
  blocker negotiation (informLeader/informFollower/slowDownForBlocked/blocker reservation),
  patchSpeed rules, commit gate, resets — all with file:line citations to tag v1_27_1.
- Deliverable copy for the rebuild agents: `E:/agent-work/rb-shared/lcspec/spec.md`
  (+ `DONE.md` summary there).

## Consumable facts (quick reference)
- Probabilities are fixed-point ints × 1e7 (`HYST_PRECISION`).
- Thresholds: speedGain right = (0.2/lcSpeedGainRight)/lcSpeedGain*1e7 = 2e7 (default),
  left = 2e6; keepRight change when `prob*lcKeepRight < -2e7`.
- Strategic lookahead `laDist = laSpeed*10*lcStrategic*(right?1:lcLookaheadLeft) + 2*lenWithGap`;
  urgent iff `usableDist/|bestLaneOffset| < laDist` (and bestLaneOffset == curr offset).
- URGENCY divisor 2.0 (5.0 opposite) enters `remainingSeconds`, which gates all messaging.
- IDM secure gap `max(0, v*T + v*(v-vLead)/(2*sqrt(a*b)))`; changer blocked test multiplies
  by `getSafetyFactor() = 1/lcAssertive`.
- 10 hand-computed test vectors (inputs -> expected flags/commit) at the end of the spec,
  IDM a=1.5, b=4.5, T=1, ballistic, TS=1.

## Tests
- None executable here (doc-only deliverable); the 10 vectors are hand-derived from the
  cited formulas and include the intermediate arithmetic.

## Known gaps
- Opposite-direction driving (`lcOpposite`, changeOpposite in MSLaneChanger) only summarized;
  not part of the 10 vectors.
- MSLCM_LC2013_CC (connector-control variant) not covered.
- Continuous lane changing (INSUFFICIENT_SPACE/SPEED gates) described but not vectorized
  (default build uses instant LC).
- SUMO sources fetched read-only into `E:/agent-work/_sol/sumo-src-1271/` (never executed).
