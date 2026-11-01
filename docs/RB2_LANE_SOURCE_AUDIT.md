# SUMO1.27.1 LC2013 round-2 source comparison

Read-only reference tag `v1_27_1`, local cache
`E:/Temp/opencode/sumo-1.27.1-rb-lane`; downloaded source is never executed.
Base `22cc32f6` has the round-1 source corrections already applied. This audit
does not pretend that its existing fixed-point rules were missing.

## Accumulation and thresholds, line by line

- Constructor99–123 / initDerivedParameters143–150: zero initial three counters;
  `lcStrategic`, `lcCooperative`, `lcSpeedGain`, `lcKeepRight`, `lcAssertive` default1;
  right gain multiplier0.1. `long long` thresholds right
  `(0.2/lcSpeedGainRight)/lcSpeedGain * 10000000`, left
  `0.2/lcSpeedGain * 10000000`, truncated toward zero. Nonpositive speedGain
  uses LLONG_MAX for both; consequently it also disables keep-right accumulation.
  Our `_settings`/thresholds match. The FIT reader preserves legal zero motivations
  and cooperative[0,1], assertive>0; it never clips zero to a positive epsilon.
- 1694–1695: `(vTarget-vCurrent)/max(vTarget,10)` matches `relative_gain`.
- 1710–1718: right current-lane advantage>5/3.6 decays **right speed-gain only**
  by `0.5^actionStep`; otherwise add truncated actionStep×relativeGain×precision.
  Keep-right does not decay in this branch (source1715 is commented out).
- 1721–1728: legacy acceptance `7*(vMax/laneSpeedLimit)*max(1,currentSpeed)`.
  We retain this default, not a hand-set dwell time. Nonlegacy follower-dependent
  acceptance1729–1737 is still outside the subset and is not used by this fit.
- 1740–1753: full-speed gap uses LaneQ **length**, minus discrete CF brakeGap;
  slower target leader caps it by gap-minus-IDM-secure-gap. Slow original leader
  further caps driving time when overtaking-right is prohibited. Signed closing
  speed remains signed. Round-1 already repaired this; do not add a positivity clamp.
- 1755–1757: subtract truncated actionStep×thresholdRight×
  (fullSpeedSeconds/acceptance)/5. Right1758+ wishes persist through temporary
  blocking. 1785 compares truncated keepCounter×lcKeepRight to **negative** right
  threshold. Both multiply/truncation and branch placement match our kernel.
- 1808–1816: right gain request only above threshold and target continuation
  length/currentSpeed>20s. 1820–1826: left decay0.5^actionStep if worse,
  0.8^actionStep if equal; otherwise accumulate. 1853–1855 additionally needs
  relativeGain>NUMERICAL_EPS0.001 (or right-overtake avoidance). Matching rules.
- 1874–1881: nonurgent route-strategic request comes after speed/keep-right;
  equality refers to ORIGINAL bestLaneOffset, not the equally-good-lane fallback.
  Our reason precedence matches the early returns.
- No identifier `changeWish` and **no generic wish decay** occurs in this tag's
  LC2013. Applying decay to mature keep-right to improve precision would be a
  new behavioral rule. `changed()`1064–1078 resets all wishes/lookahead; our
  memory resets on actual observed/free-running lane changes, not hypotheticals.

## Lookahead, safety and cadence

- 1233–1245: lookahead grows instantly, otherwise linear memory factor
  `1-0.1*actionStepLength`, floored at0, then10×strategic×leftLookahead2
  plus twice lengthWithGap. Strategic urgency subtracts edge-local position;
  anticipation/full-speed/remain-time use full LaneQ.length. Source agreement.
- MSLaneChanger329–443: sublane-off, downstream-first, **right before left**,
  only on `vehicle.isActive()`. Default actionStepLength is the simulation step.
  On the two through lanes each ego has one legal adjacent through target;
  the ramp's legal target is lane1. No random per-step event is called for.
- NEW: the old kernel aged lookahead even when `decision_mask` excluded that
  ego, and always used simulation dt for wish updates. Round2 freezes lookahead
  on nondecision ticks and accepts documented `actionStepLength` for all wish
  accumulation/decay. Simulation dt still controls CF/braking. The caller must
  supply the actual isActive decision mask for nondefault action steps; no new
  timing threshold or inferred action-step schedule is invented.
- The same legal-decision guard matters under DEFAULT settings on the one-lane
  upstream ramp: MSLaneChanger380 never calls wantsChange there. Old code carried
  an upstream speed peak into the merge-edge lookahead despite no LC decision.
  The fixed kernel starts lookahead from the first legal merge-edge speed.
- MSLaneChanger851–925: ego's assertiveness divides both secure gaps; each
  follower's minGap is removed before comparison. IDM.cpp218–220 returns
  max(0,v*tau+v*(v-vLeader)/(2sqrt(ab))). Body overlap still vetoes independently.
  NEW: for actionStepLength!=TS, secure-gap speeds use source tauRemainder
  `max(egoTau-TS,0)`, with positive follower/negative leader acceleration only.
- Blocked wishes must keep accumulating: source1676–1680's blanket blocked
  early return is commented out. Current implementation correctly accumulates
  wishes but never executes blocked/overlapping proposals. Do not suppress all
  blocked accumulation merely to reduce positives.

## Runtime / inference defects addressed

- FULL-v4 recipe explicitly rejects fitted LC. New explicit opt-in
  `variants.FULL.fitted_lc=true` plus `lane_parameter_directory` enables the exact
  frozen raw-fitted population. Default `wm_phys.toml` is unchanged.
- IDM `with_motion_posteriors` deliberately drops LC (longitudinal evidence is
  not lane evidence). Keep that contract: attach separately reconditioned LC
  **after** causal IDM conditioning. Native tape saved-ID weights are not
  transplanted to another export's numeric ID space.
- Each sampled driver profile is persistent/coupled across arms. Entrants use
  the learned population. Likelihood emission is only for causal inference,
  never a runtime lane/no-lane coin flip. No population refit or black-box training.
- Replay the causal60s under each sampled driver's parameters and observed
  lane changes to initialize wish memory; map into packed slots by observable
  identity. Cold wishes and packed-slot-as-ID mistakes are not permitted.

## Remaining limits (must be reported with evaluation)

Full serial negotiation/leader overtake/fall-behind, junction critical leaders,
zipper unsafe-link veto, and nondefault nonlegacy acceptance are not complete.
Our synchronous snapshot cannot claim bitwise SUMO equivalence. The frozen fit
was estimated with an older source kernel and pooled IDM; later source hashes
are recorded when reconditioning. IDM/LC blocks remain independently sampled.
FIT proposal precision alone cannot certify free-running counts or congestion.
