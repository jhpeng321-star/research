# R13Y — learned cooperative merge yield

Base: `c4/v10r13c` tip `eb75010c`; implementation branch `c4/v10r13y`.

## Mechanism and initialization

`model.mainline_merge_yield_learnable=false` by default. Disabled models create no
yield tensors and consume no extra RNG. Enabled models require learned driver
parameters and the `lane_learnable` structured core. The existing V6MERGE
`merge_end_blocked` mask identifies slow lane-end ramp cars without secure gaps.
The nearest lane-1 follower strictly behind each merger's rear sees its speed and
position as a virtual IDM leader. Pairing/mask selection is discrete; selected
positions, speeds, driver parameters and both yield scalars retain their adjoints.
CAV followers are excluded without reassigning their pair to a farther HDV.

Two shared `nn.Parameter` scalars, `merge_yield.log_decel` and `log_distance`, map
through `exp` to positive deceleration and influence distance. **2 m/s² and 60 m
are initializations**, from BLOWUP.md arm Y, not fixed behavioural thresholds.
The follower's inferred minimum gap supplies both the virtual clearance reserve
and the width of `sigmoid((distance-front_separation)/min_gap)`. This replaces
V6MERGE's non-learnable distance cutoff; its legacy operator remains unchanged.
Own learned IDM braking also limits yield. No driver settings are overwritten.

Courtesy repair A/residuals first produce their usual acceleration; yield supplies
a separately computed physics cap. The final value is their **minimum**, never
the sum of two yielding offsets. Existing stronger braking wins; positive
corrections retain their uninfluenced share, so distant pairs cannot erase them. The forward
`physics_accel_mps2` decomposition remains the uncapped baseline; direct
`physics_accel()` returns the constrained core. Merge decisions, overlap vetoes,
stop-time integration and the pathwise adjoint are preserved.

## Offline evidence

CPU-only S-R12C inference: 48 fixed TRAIN-corpus states × 2 samples × 120 s;
12 predetermined matching states × 2 samples × 300 s. Native CAV replay, entries
off, recorded exits; CRN across off/yield/yield+emergency9. Export source-internal
partition is validation within the explicitly approved outer TRAIN corpus.
All 360 arm runs completed. `docs/r13y_offline_arms.csv` contains all requested
metrics and absolute/paired 95% world-cluster CIs (4,000 draws). Manifest/receipt
pin data, code and checkpoint hashes. Zone windows:

```text
seconds arm occupancy/SUMO[CI95] standing_vehicles[CI95]
60-120 off .536[.446,.662] .419[.231,.644]
60-120 yield .755[.613,.949] .736[.434,1.075]
60-120 yield+E9 .753[.603,.976] .681[.408,.990]
120-300 off .549[.275,.793] .196[.012,.459]
120-300 yield .635[.290,.868] .254[.014,.634]
120-300 yield+E9 .659[.276,.875] .263[.005,.658]
```

120 s zone speed: 14.61 → 12.45 m/s (SUMO 12.16); stuck ramps 64 → 30.
**Acceptance PARTIAL: overlap gate FAILS.** Yield adds HDV–HDV onsets/run
+5.35[2.23,9.13]; 52/96 short and 13/24 long paired runs worsen. E9 reduces
HDV–HDV onsets to .115/run versus 13.0 off, but the full 13E combination needs
verification. Keep the flag off pending that gate.

Verification: 81 CPU tests pass (2 legacy fixture/config cases deselected);
2,238 tensors are byte-identical against the tip, including eight short rollouts.
Post-study change only repositions a default-omission entry for clean 13E merging;
config and byte-identity checks pass again.

## Merge into round 13E

Merge `c4/v10r13y` into `c4/v10r13e`. Preserve 13E's emergency
baseline and secure-gap veto when resolving shared model/config hunks. Set
`model.mainline_merge_yield_learnable=true` before construction/optimizer setup;
include `merge_yield.*` in core recalibration's trainable parameters. Legacy
weights initialize only absent yield tensors; native checkpoints preserve them.
Freeze the fitted yield parameters with the core during residual-only training.
Do not duplicate the final cap in the baseline acceleration. Run the focused
tests and the combined emergency/secure-gap/yield gate before launch.

## Risks

A bounded cooperative deceleration is not a collision-proof emergency controller.
Discrete pair switches remain non-differentiable. The 300 s fixed-roster check is
censored by recorded exits and excludes new arrivals. The emergency9 arm is a
proxy, not 13E's learned emergency-plus-secure-gap mechanism. Initialization-only
inference cannot establish post-recalibration fidelity or planner qualification.
No training, GPU operation, SUMO run or push is part of this change.
