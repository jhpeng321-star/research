# C4 diagnosis after the 2026-10-16 Dreamer-loop run (zero-simulation analysis)

Sources (local copies in `runs/active/C4-V5-20261014/final_20261016/`):
- final SUMO evaluation (seeds 964000–964041, 600 s episodes);
- the 300 s SUMO branching measurement of the after-loop policy (92 decision points, demand 2400, 20/40 % CAVs);
- policy training logs (updates 1–1273: before the loop, then on refreshes 1–4).

Sign conventions: SUMO deltas are policy − native (positive = worse) unless marked; D_all/C_G are delay saved vs
native (positive = better).

## Findings

1. **The policy is still worse than native, and the loop helped a lot.**
   Final SUMO evaluation, system time per vehicle vs native: before the loop +11.53 [+10.99, +12.06] s, after the
   loop +4.15 [+3.81, +4.50] s. For comparison, MAPPO −1.51 and PI −1.16.

2. **Most of the remaining harm is blocked entries, not slower driving on the road.**
   300 s branching, all CAVs on the policy vs all native, same state:

   | Measure | Policy − native |
   |---|---|
   | Team delay (D_all) | −2,643 [−3,152, −2,126] veh-s (harm) |
   | Source wait | +1,879 veh-s (≈ 71 % of the harm) |
   | Road delay | +764 veh-s |
   | Arrivals | −8.4 of 188 |

   The CAVs reduce how many vehicles can enter the network.

3. **The world model is wrong mainly about entries.**
   - In imagination the policy's source-wait term fell 5× during training (0.57 → 0.11 per step).
   - In SUMO the after-loop policy increases source waiting by 1,879 veh-s over 300 s.
   - The world model's entry/insertion process does not capture how CAV behaviour near the entrance blocks
     insertions. This is the specific world-model gap to fix; overall position accuracy is not the bottleneck.

4. **The safety constraint made the policy cautious early, and that habit persisted.**
   - The TTC cost is summed over every same-lane closing pair, human–human pairs included, against an absolute
     budget (3.6). It is not relative to native traffic.
   - The Lagrange multiplier rose to about 1.8 (updates 150–450), then fell to about 0 after update about 750,
     once the cost dropped below budget.
   - In SUMO the after-loop policy has 18 % lower TTC cost than native (551 → 454 over 300 s) while costing delay.
     It buys safety beyond native at the expense of throughput.

5. **The delay-learning signal is weak once the safety term switches off.**
   - The actor gradient norm fell from about 10–15 (while the multiplier was active) to about 0.6 (updates 750+).
   - Entropy collapsed from 1.28 to 0.28, below its 0.5 target.
   - The delay gradient through 30 s of imagination is about 20× weaker than the safety gradient was, so the
     cautious habits learned early are undone only slowly.
   - Most later progress came when a refreshed world model changed the landscape (each refresh saturated after
     about 150–200 updates).

6. **No cooperation.** At 300 s the cooperation gain C_G is +154 [−782, +1,048] (not significant). The harm is
   collective: D_all −2,643 against D_G −217 and sum of D_i −371 (both not significant). It appears when all CAVs
   act, consistent with the entry-blocking finding.

## Conclusion
The limit is **not mainly overall world-model accuracy**. It is three specific things:
- **(a) Objective:** an absolute, all-traffic TTC budget that rewards being safer than native at the cost of delay.
- **(b) One world-model component:** entry/insertion, the source of 71 % of the real harm, which imagination
  thinks is solved.
- **(c) Optimisation:** from-scratch start, a weak 30 s delay gradient, and collapsed exploration.

## Recommended order
0. **Cheap checks** (GPU, no SUMO):
   - score the MAPPO actor and native inside the refresh-4 world model: does the model rank MAPPO above native?
   - compare imagined vs SUMO source wait on the loop's policy batches.
1. **Fix the safety objective:** make the TTC cost relative to native, or count only pairs involving a CAV, and
   scale the Lagrangian term like the return. Re-check that the policy's SUMO TTC cost ends near native, not 18 %
   below it.
2. **Fix entries in the world model:** add source wait / insertions on held-out policy episodes to the refresh
   acceptance test, and up-weight entry losses on policy data.
3. **Fix the optimisation:**
   - start from the MAPPO actor (learned, −1.51) rather than from scratch;
   - keep entropy at its target (check the alpha update);
   - consider 60 s imagination once the 60 s error allows.
4. **Run the loop overnight** (10+ refreshes) with the model-gap metrics (delay and source wait, imagined vs SUMO)
   as the headline.
5. **Then cooperation:**
   - run near-capacity scenarios where opening a gap for a ramp car pays off;
   - use more what-if scenes, started at the merge;
   - measure with 300 s branching plus behaviour counts;
   - in parallel, try runtime planning with the world model (score joint plans per situation).

## Addendum (laptop checks, 2026-10-16 ~23:20; runs/active/C4-DIAG-20261016)

7. **Root behaviour: the CAVs fail to accelerate out of the merge bottleneck.**
   - Just past the merge, policy CAVs accelerate +0.27 m/s² against native +0.51.
   - They stay 4–5 m/s slower all the way downstream (22.9 vs 27.4 m/s) and slow the HDVs behind them.
   - A lower discharge rate means a bigger upstream queue, which means blocked entrances (finding 2).
   - Likely causes:
     - the zero-acceleration initialisation ("hold speed");
     - the TTC penalty that punishes closing in on a leader, so it fights accelerating;
     - the weak delay gradient that never undid this (finding 5).
8. **The lane head never learned.** Its output weights are still about 0.003 (initialised at 0) after 1,273
   updates, and imagined lane entropy stays at exactly ln 2. REINFORCE with this signal makes no progress. In SUMO
   the near-uniform logits issue about 2–3 lane commands per CAV per episode. A stay-only variant is NOT better
   (+8.9 vs +7.5 s/veh at 40 %), so the lane changes are not what causes the harm, but lane control is effectively
   untrained.
9. **Entropy temperature collapsed.** alpha fell to about 2e-5 by update 600 and recovers only slowly in log space,
   so entropy sat below its target (0.28 vs 0.5) for the last 500 updates.
10. **Not problems:** the 25 % native share in the deployment logs is the native warmup; the native fallback works
    as designed.

## Addendum 2: world-model side (2026-10-17, CPU check)
11. **The world model's merge is too congested, not too easy.**
    - HDVs in the merge zone are 1–3.7 m/s slower than SUMO, and the model discharges 5–8 % fewer vehicles in
      120 s (native and wall arms).
    - The bottleneck in imagination is mostly the model's own merge friction, so accelerating out of the merge earns
      little credit. This matches the policy's slow post-merge behaviour.
    - Fix: calibrate the merge zone and its discharge against SUMO, using the merge-zone speed and discharge flow as
      validation targets. The merge rules fixed blow-ups but did not tune the flow.
12. **The harm lands outside imagination.** The entrances are about 2.6 km upstream of the merge and minutes away,
    while imagination covers 30 s, so the policy cannot see the entrance blocking it causes. Fix: a critic or
    backlog term trained on real SUMO outcomes from loop data, or a local discharge-flow reward term.
13. **The refresh acceptance test** is position error on the original data, so none of 11–12 is checked. Add
    merge-zone speed, discharge and entrance wait on held-back policy episodes.

## Addendum 3: corrections from the controller-ranking baseline (2026-10-17 ~03:40)
Source: runs/active/C4-DIAG-20261016/ranking/RESULTS_v5m.md.
- Setup: 6 controllers, from the identical t = 300 s state, 48 states, 120 s horizon, entries off.

**Correction to finding 11.**
- Part of the "WM discharges 5–8 % fewer vehicles" result was an artifact: the predictor removed WM vehicles when
  their SUMO twin left the road.
- With `--hdv-exits none` the discharge shortfall shrinks to −1.1 (native arm) and −2.4 (wall arm) per 120 s.
- "The merge is too slow in the WM" still holds.

**New: v5m ranks the controllers like SUMO on HDV delay.**
- Spearman +1.00.
- It predicts the after-loop policy's delay harm almost exactly: +88 vs SUMO +88 veh-s; +98 with
  `--hdv-exits none`.
- It underestimates the downstream slowdown: −0.47 vs −1.56 m/s.

**Consequence.** Over 120 s and with entries off, the WM is broadly right about which controllers hurt. The bigger
limits are on the policy side and the horizon:
- the 30 s imagination;
- the entrance effects beyond the horizon;
- the weak learning signal.

The WM merge calibration is still worthwhile, but it is not the main lever.
