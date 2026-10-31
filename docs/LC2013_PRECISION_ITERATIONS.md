# LC2013 precision iterations — resumed 2026-11-01 00:10

## Preserved starting point (127a106f physics, 6fc46b16 handoff)

Clean worktree on resume. PANEL2_COMPLETE.json records exited owned worker
214459; live identity-bound status confirmed exited/complete at 00:14.
160 FIT episodes, 3255 recorded events, 44333 predicted positive steps,
1136 exact matches: precision 2.5624%, recall 34.9002%. Return-right alone:
30116 proposals / 207 matches (0.6873% precision); overtake 10845/587;
merge 3372/342. Original receipts are immutable. No completed MOBIL scoring
will be repeated.

## Source audit hypotheses before the new measurements

1. `keep_request` is evaluated outside the source's `thisLaneVSafe - 5/3.6
   <= neighLaneVSafe` branch. A previously accumulated keep-right wish can
   therefore trigger even while the destination is much slower. Source
   MSLCM_LC2013.cpp:1710–1791 gates BOTH accumulation and the request.
2. Default through-lane route distances are 1e6 m. Actual frozen WM geometry
   ends at s=553.33 (batched.py:38–46). SUMO caps full-speed keep-right driving
   time by route-end distance minus brakeGap (LC2013.cpp:1740–1741), so the
   infinite default can accumulate far too quickly. Finite distances also
   constrain strategic stay and 20s speed-gain remaining-time tests.
3. Source thresholds are already correctly scaled: 0.2/lcSpeedGain left and
   0.2/(lcSpeedGainRight*lcSpeedGain) right, fixed-point truncation each update.
   No outcome-based threshold adjustment is authorized or planned.

## Measurement addition

The preserved scores do not contain predicted motivation. A new diagnostic
pass is necessary to attribute the old candidate's false positives. Added
exact precision and recall contribution for strategic/cooperative/speed-gain/
keep-right and compressed ±2s runs. Recorded motion has no motivation labels:
recall per predicted reason is a contribution to all-event recall, NOT true
class-conditional recall. Baseline MOBIL is reused with original SHA gates.
One nice19/single-thread CPU only; load admission <20. At 00:14 cpuQ load21.4
blocked dispatch; no GPU present on cpuQ, no new job launched at that check.
