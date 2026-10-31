# S-C1CLOSED — frozen current-model closed-loop C1
**COMPLETE recorded TRAIN/dev panel; scoped common-plant evidence, provisional algorithm novelty.**
WM2 `b8797242` + learned entrant mix, K8; frozen BB-traj `afaaa32c`, BB-id `e12c81c0`, historical structured `87ed02e0`, K1.
40 worlds 969400–969439, t300, genuine observed 60 s histories, 480 world/future tasks; no SUMO or fitting.
Primary comparison uses shared external CAV longitudinal/gap-merge plant (no ordinary CAV lane changes); production-native WM2 is separate.
Real PI/CIRCLES/MAPPO feedback observes predictions; autonomous HDV/entry/exit/source evolution, no original command tape or future motion inputs.
S-C1GAP supports PI/MAPPO absence from audited WM2 corpora; pre-R16 initialization exclusion remains incomplete. Reused worlds are not unseen-world evidence.
BBs have their original different/no-PI training corpus; BB-id learns CAV motion by ID and cannot execute feedback MAPPO. No baseline retraining.
Wall feedback pending: 120 genuine-history starts survive, but zero executor snapshots/controller maps and only 120 s cached futures; causal restoration not established. CIRCLES/native/CTRL0 are diagnostics.
Endpoint errors: median initial-HDV AE within world/draw, then median K8, then world median; paired models use the identical cohort surviving truth and every WM draw.
All 10/30/60/90/120/180/300 s clocks are in CSV; surviving counts exported, no zero imputation; synthetic entrants have no matched identity.
95% percentile CIs: 2,000 whole-world resamples, seed 0; differences formed within world before bootstrap. Negative WM−BB error differences favour WM.
## Headline 60 s matched endpoints (m)
- pi: WM 57.6 [54.6, 63.7] (n=40); BB-traj 61.9 [51.9, 78.3] (n=40); paired WM−BB -5.17 [-18, 7.09] (n=40).
- mappo: WM 56.7 [53.5, 62.1] (n=40); BB-traj 68.9 [53.4, 77.1] (n=40); paired WM−BB -3.45 [-13.4, 4.58] (n=40).
## All-horizon paired position error WM−BB-traj (m)
- 10 s: PI 1.8 [1.13, 2.56] (n=40); MAPPO 1.61 [0.737, 2.33] (n=40).
- 30 s: PI 2.99 [0.287, 4.68] (n=40); MAPPO 1.27 [-1.79, 4.59] (n=40).
- 60 s: PI -5.17 [-18, 7.09] (n=40); MAPPO -3.45 [-13.4, 4.58] (n=40).
- 90 s: PI -15.4 [-67.3, 0.27] (n=40); MAPPO -14.1 [-46.9, 16.9] (n=40).
- 120 s: PI -72.6 [-135, 23.2] (n=33); MAPPO -97.5 [-134, 34.9] (n=33).
- 180 s: PI 100 [-303, 316] (n=6); MAPPO -143 [-453, 115] (n=5).
- 300 s: PI unavailable; MAPPO unavailable.
## Long-horizon aggregate fidelity (secondary)
Merge queue: -200≤s≤75.44 m, v<5 m/s; endpoint and time-mean counts. Fixed field cells: 100 m×10 s, -1000..500 m, all lanes pooled, density veh/km road.
Field speed is occupancy-weighted (empty=0); occupancy mismatch exported. Queue/fields/crossings use pointwise draw medians. These are traffic-level errors, not causal diagnoses.
- 60 s J absolute-error difference (veh-s): PI -102 [-310, 31.2] (n=40); MAPPO -116 [-375, -11.1] (n=40).
- 120 s J absolute-error difference (veh-s): PI -2.49e+03 [-2.9e+03, -1.96e+03] (n=40); MAPPO -2.63e+03 [-2.87e+03, -1.88e+03] (n=40).
- 180 s J absolute-error difference (veh-s): PI -7.98e+03 [-8.42e+03, -7.05e+03] (n=40); MAPPO -7.78e+03 [-8.37e+03, -7.1e+03] (n=40).
- 300 s J absolute-error difference (veh-s): PI -2.37e+04 [-2.5e+04, -2.28e+04] (n=40); MAPPO -2.36e+04 [-2.57e+04, -2.21e+04] (n=40).
## Delay accounting and controller effects
J = source waiting + upstream(s<0) + merge(0..75.44) + discharge(s>75.44) speed-deficit integrals; left-boundary .5 s rectangles, vref31.29 m/s.
Common total-J median-order draw weights preserve component additivity within world. Signed component biases use equal-world means (also additive); absolute errors use world medians.
- WM 300 s pi component biases source/upstream/merge/discharge (veh-s): 1.86e+03 [689, 3.11e+03] (n=40); 3.32e+03 [2.62e+03, 4.07e+03] (n=40); 1.09e+03 [832, 1.37e+03] (n=40); 1.34e+03 [1.18e+03, 1.49e+03] (n=40).
- pi WM E2 direction accuracy 60/300 s: 0.351 [0.211, 0.5] (n=37) / 0.55 [0.4, 0.7] (n=40).
- WM 300 s mappo component biases source/upstream/merge/discharge (veh-s): 1.23e+03 [-44.3, 2.6e+03] (n=40); 4.23e+03 [3.14e+03, 5.42e+03] (n=40); 1.25e+03 [924, 1.59e+03] (n=40); 1.47e+03 [1.15e+03, 1.82e+03] (n=40).
- mappo WM E2 direction accuracy 60/300 s: 0.615 [0.462, 0.769] (n=39) / 0.575 [0.425, 0.725] (n=40).
- Five-arm oracle regret 300 s: WM 809 [509, 1.13e+03] (n=40); BB-traj 710 [462, 978] (n=40); includes exposed CTRL0 diagnostic.
E2 ΔJ=treatment−native, negative improves; exactly-zero true effects excluded only from direction denominators. Pairwise rankings exclude true ties; native-first deterministic choice ties.
WMFIXMERGE v2-DATA / WMFIX-ES secondary slot: no independently passing, hash-bound verdict was supplied; primary WM2 is unchanged.
SUMO RNG divergence-floor slot: unavailable pending MAIN-supplied approved recordings; figure leaves it explicitly unmeasured, not zero or a proven lower bound.
Files: `C1CLOSED_per_world.csv`, `C1CLOSED_paired.csv`, `C1CLOSED_bootstrap.csv`, `C1CLOSED_error_horizon.png`, `C1CLOSED_ANALYSIS.json`.
Reproduce: `python -B tools/c1closed/sync.py`; `python -B tools/c1closed/analyze.py`; CPU/source/PID receipts in `tools/c1closed/evidence/`.
No qualification threshold is borrowed from E2; relative BB advantages, absolute fidelity and controller selection are distinct claims.
## Interpretation and numerical limitation
At 60 s neither PI nor MAPPO paired HDV-position contrast excludes zero; current closed-loop endpoint superiority is inconclusive.
Lower 120–300 s team-delay error versus BB-traj coexists with large absolute errors and weak selection; it does not establish accurate controller evaluation.
Mixed-hardware exact cross-host parity failed; the prospectively registered numerical/ensemble gate passed 158/158. See `C1CLOSED_RUNTIME.md`; this is not a SUMO divergence floor.
