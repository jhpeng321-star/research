# Fresh IDM collapse-delay sensitivity (2026-08-25)

## Purpose and status

This is a model-form sensitivity experiment for the IEEE IV framing. It is not
verification of the counterfactual escalation predictor and it must not be
presented as such. The tested wall arm is the established `sup_v43` supervisor,
not the still-to-be-verified v2 counterfactual treatment-effect gate.

The experiment asks two narrower questions:

1. Does an IDM car-following population exhibit a stronger delay consequence
   as the corridor moves from the stochastic-breakdown band into persistent
   collapse?
2. Can coordinated moving-wall control offset that delay at sufficiently high
   demand, relative to both no control and the frozen decentralized damping
   checkpoint?

## Frozen experiment slice

- Fresh development seeds: 151--162 (12 paired seeds per cell).
- Long corridor map: `D:\research\highway_merge_shepherd_long`.
- IDM demand tags: `light_m2200r400idm`, `light_m2400r400idm`, and
  `light_m2700r400idm`.
- Mainline demand: 2200, 2400, or 2700 veh/h; ramp demand: 400 veh/h.
- CAV penetration: 0.20; scatter placement.
- Episode cap: 1800 controller steps; demand ends at 900 s and the evaluated
  episodes cleared after about 1000 steps.
- HDV population: the same aggressive/normal/cautious mixture as the Krauss
  comparator, with only `carFollowModel="IDM"` added to each vType.
- Arms:
  - no control (`no_op`);
  - frozen decentralized damping (`rl_pruned3`) using
    `runs/rllib/shepherd_pruned_v1_dense_mpen/checkpoints/best`;
  - established coordinated moving-wall supervisor (`sup_v43`).
- Sustained collapse: health below 0.15 for at least 60 consecutive steps after
  step 100, using the existing stage-10 definition.
- Delay: SUMO trip `timeLoss`, averaged over completed vehicles (s/vehicle).

## Fresh 12-seed results

At 2200 veh/h, no control collapsed in 8/12 seeds with mean delay 14.308
s/vehicle. Damping collapsed in 9/12 with mean delay 14.621. The wall collapsed
in 2/12 with mean delay 15.620. Thus, in the transition band the wall bought a
large protection gain at a mean delay cost of 1.312 s/vehicle versus no control
(bootstrap 95% CI 0.582 to 2.019; paired Wilcoxon p=0.0093) and 0.999 s/vehicle
versus damping (95% CI 0.194 to 1.791; p=0.0640).

At 2400 veh/h, no control and damping both collapsed in 12/12 seeds. Their mean
delays were 21.234 and 19.010 s/vehicle, respectively. The wall collapsed in
4/12 and had mean delay 21.612. Relative to no control, the mean wall difference
was +0.378 s/vehicle (95% CI -2.763 to 3.255; p=0.622), which is delay parity in
this small sample. Relative to damping, the wall cost +2.602 s/vehicle (95% CI
0.140 to 4.614; p=0.0640). Individual cases were strongly heterogeneous: seed
159 improved from 27.61 (no control) and 26.12 (damping) to 18.25 s/vehicle,
while several low-cost collapse realizations paid a wall premium.

At 2700 veh/h, every arm met the sustained-collapse definition in 12/12 seeds,
but the wall managed the persistent breakdown more efficiently. Mean delay was
44.190 s/vehicle under no control, 44.927 under damping, and 39.787 under the
wall. The wall saved 4.403 s/vehicle versus no control (95% CI 0.336 to 8.528 s
saved; paired Wilcoxon p=0.0523; 10/12 seeds improved) and 5.140 s/vehicle versus
damping (95% CI 1.150 to 9.160 s saved; p=0.0523; 9/12 seeds improved). This
fresh effect reproduces the earlier independent 50-seed result of approximately
4.4 s/vehicle saved versus no control (reported p=0.005).

Across the no-control density ladder, mean delay rose from 14.308 to 21.234 to
44.190 s/vehicle while sustained-collapse frequency rose from 0.667 to 1.000.
At 2400, the seed-level association between health and delay was strong
(Spearman rho=-0.909, p=0.00004); at 2700 it remained significant (rho=-0.629,
p=0.0283). Because demand changes across cells and collapse is not randomized,
these are descriptive model-regime results, not a causal estimate of the
economic cost of real-world breakdown.

Within fixed density and arm, collapsed episodes also had higher mean delay
whenever both outcomes were observed. At 2200, the conditional collapsed-minus-
healthy difference was +0.883 s/vehicle for no control, +0.697 for damping, and
+2.138 for the wall. At 2400 it was +2.119 for the wall. These conditional
differences are small-sample descriptive premiums, not treatment effects.

## Safety and service checks

No collisions were recorded in any of the 108 fresh runs. Total TTC events for
no-control / damping / wall were 11 / 15 / 6 at 2200, 19 / 12 / 5 at 2400, and
66 / 68 / 25 at 2700. Mean discharge remained essentially matched within each
density (about 2297, 2473, and 2692--2696 veh/h), so the high-density delay gain
does not come from simply abandoning throughput.

The saved `hard_brake_steps` field was zero in all arms, but that field records
the policy wrapper's hard-brake counter and should not be substituted for a
full all-vehicle deceleration exposure audit. TTC and collision results are the
clean safety evidence from this particular probe.

## Damping checkpoint limitation at 2700

At 2700, some timesteps had more than 32 active CAVs. The frozen checkpoint's
packed centralized-critic observation truncates overflow peers after the first
32 sorted IDs. The per-CAV actor still executes, but 2700 must be described as a
stress/transfer case for this checkpoint. This limitation strengthens the need
to keep the primary counterfactual evaluation in the frozen Krauss contract and
use IDM as transfer/model-form sensitivity evidence.

## Defensible paper wording

Use wording close to this:

> Under an IDM model-form stress test, the delay consequence of sustained
> congestion is substantially amplified as demand increases. Coordinated
> moving-wall control trades about 1 s/vehicle for collapse prevention near the
> stochastic transition, is delay-neutral relative to no control at 2400 veh/h,
> and saves 4.4 s/vehicle relative to no control (5.1 s/vehicle relative to the
> frozen damping controller) during persistent overload at 2700 veh/h. These
> results motivate a selective escalation gate: the wall is valuable in some
> states but is not uniformly delay-beneficial.

Do not write that IDM "matches reality better" based on this simulation alone.
The defensible statement is that IDM supplies a stronger collapse-delay regime
and tests model-form sensitivity. Claiming superior realism requires empirical
trajectory/capacity-drop calibration or authoritative literature tied to the
chosen IDM parameters.

## Reproduction commands

Use the SUMO/RL environment documented in `AGENTS.md`, then run each tag with:

```powershell
python -u _probe_metastable_feasibility.py --stage 10 `
  --tag light_m2200r400idm --variants rl_pruned3,sup_v43 `
  --seeds 151,152,153,154,155,156,157,158,159,160,161,162 `
  --workers 4 --steps 1800 --pen 0.2 --long
```

Repeat with `light_m2400r400idm` and `light_m2700r400idm`.

Raw rows are stored under:

- `runs/eval/noop_cache_light_m{2200,2400,2700}r400idm_long/`
- `runs/probe/metastable_feasibility/stage10_light_m{2200,2400,2700}r400idm_long/`
