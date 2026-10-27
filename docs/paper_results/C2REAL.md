# S-C2REAL-1: cycle-2 real-SUMO evidence

**Analysis only, 2026-10-27.** All unique collected main-loop policy/native CRN pairs: cycles 0/1/2 have **4/13/13 worlds**, two starts/world (60 pairs; 120 tapes). Cycle 0 registered thirteen but collected four; nine are missing. Its four conversions duplicate the same samples. Thirteen failed cycle-1 attempts contain no paired outcomes.

## Real outcomes

J includes road delay plus entrance waiting over 300 s. Negative policy-minus-native means improvement. Mean veh-s/start [95% CI]: cycle 0 **−623 [−1,203,+63]**; cycle 1 **+53 [−589,+676]**; cycle 2 **−263 [−1,061,+512]**; pooled **−174 [−620,+272]**. Equal-world means; 5,000 whole-world percentile bootstrap draws, seed 20261027. CSV contains every start/world, two-start sums, and CAV-share/demand strata; singleton-stratum intervals are descriptive, not replication evidence. The four-world guard uses two-start sums, explaining its different scale and population.

**“Cycle 2 better than cycle 1 in real SUMO”: inconclusive.** Difference **−316 [−1,321,+718] veh-s/start**. Worlds differ: the cross-cycle contrast is **unpaired**, independently resampling thirteen worlds/cycle; within-start policy/native pairing is retained.

## Imagination versus reality

Bias = realised minus predicted policy/native dJ; positive means over-promise. Existing same-start, eligibility-selected ALL/B predictions:

- 30 s: cycles 1/2 use 19/21 starts; bias **+0.70 [−4.24,+6.27] / +4.64 [+0.74,+8.51] veh-s**; Pearson/Spearman **0.45/0.45 / 0.35/0.51**.
- 300 s: 9/8 starts; bias **+1,086 [+58,+2,094] / −743 [−1,933,+461]**; Pearson/Spearman **−0.45/−0.53 / 0.16/−0.07**.

**“WM over-promises”: supported for cycle 2 at 30 s; unidentified at 60 s (bias CI unavailable).** No collected horizon-matched 60 s predictions exist. The selection-panel change −0.384→−0.693 cannot establish 60 s fidelity. Increased optimism at 30 s is inconclusive: **+3.94 [−2.93,+10.38]**; exploitation is not established.

## Behaviour

Frozen COOPMEAS **9119cd77**, events/CAV-hour, policy/native for cycles 0;1;2: metering **0.81/1.19;0.44/1.32;0.75/1.44**; multi-CAV slowing **0.20/0.50;0.06/0.56;0.09/0.46**; side-by-side **0.10/0.10;0/0.06;0/0.12**; lane-vacate **2.33/1.99;1.60/2.06;2.22/2.13**. These are behavioural proxies, not causal cooperation.

**“Metering increased”: inconclusive.** Cycle-2-minus-cycle-1 policy/native rate-effect difference **+0.187 [−0.321,+0.683] events/CAV-hour**; unpaired world bootstrap. CSV supplies all rate CIs and counts/exposures.

Evidence: `tools/c2real/evidence/{VERIFIED,COVERAGE,INPUTS,RESULTS}.json`. Computation: gpuA CPU, nice 19, one numerical thread, CUDA disabled; zero new SUMO.
