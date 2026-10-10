# Stage 1b-prep-2 results — tail-braking scan and delay decomposition

Date: 2026-09-02. Status: **development-only; no arm selected, no gate
applied** (see "claims not made").

Reference Arm C per seed on every grid is the 1b-prep replayed C with the
rule-blind t_d = 200 window, digest-identical to the stored Stage 1a Arm C
(verified 30/30 for every scan arm).

## Part A — artifact-only

### A1. Collapse premium at 2200/p20 (Arm C, n = 30)

| | collapsing (24) | non-collapsing (6) | premium |
| --- | ---: | ---: | ---: |
| mean delay s/veh | 14.06 | 11.74 | **+2.31** |
| median delay s/veh | 12.85 | 11.50 | **+1.35** |

Regression of C delay (s/veh) on C total collapse duration (s), n = 30, OLS:
slope **+2.87 s/veh per 100 s of collapse**; seed-bootstrap (2000 resamples,
seed 20260902) CI90 [+1.71, +4.03], CI95 [+1.31, +4.43].

### A2. WHOLD_soft value decomposition (per-seed classes)

| class | n | mean Delta_delay (s/veh) | median | mean Delta_duration (s) |
| --- | ---: | ---: | ---: | ---: |
| prevented | 13 | **+2.57** | +2.58 | −226.9 |
| co-collapsing | 11 | +1.69 | +2.28 | −175.5 |
| induced | 1 (6009) | +4.75 | +4.75 | −28.0 |
| never-collapsing | 5 | **+3.52** | +3.62 | −105.6 |

Engagement cost (mean Delta_delay over never-collapsing seeds) =
**+3.52 s/veh**. **The prevented-seed mean Delta_delay is positive (+2.57),
not negative** — the wall paid delay on every class; the value of
prevention is duration reduction (mean −227 s on prevented seeds), not a
delay saving.

### A3. Prevented vs co-collapsing seed features

| | h(200) | h(150) | C onset (s) | C delay (s/veh) |
| --- | ---: | ---: | ---: | ---: |
| prevented (13) | 0.000 | 0.393 | **356** | 12.76 |
| co-collapsing (11) | 0.000 | 0.466 | **190** | 13.17 |

The full per-seed listing is in `prep2_summary.json → a3` (medians above;
prevented seeds have notably later C onset — 356 vs 190 s). h(200) is 0 for
both classes; h(150) does not separate them.

### A4. Hard-brake severity (stored)

Stored: yes — the event-start `acceleration_mps2` field. WHOLD_soft
approach-edge event starts (n = 4357): p10/p50/p90 = **−4.5 / −4.5 /
−4.5 m/s²**; min −5.67; 94.9% of onsets sit *exactly* at the SUMO IDM
deceleration cap (4.5 m/s² — the hard-brake criterion is −4.5, and the
model's decel is bounded at −4.5); 5.1% reach −4.5 to −5.67; **fraction
≤ −6.0 m/s² = 0.0**. Same result for W-only (no matching C event) onsets
(n = 3011).

### A5. Same-cell throttle law (merge-edge flow, 300-900 s)

Arm C median 2602.7 veh/h; WHOLD_soft median 2593.7 veh/h; median delta
**+3.0 veh/h** — the throttle is preserved: the 10 m/s sustained wall does
not reduce mean merge-edge discharge in the 300-900 s window.

## Part B — wall-speed scan (t_d = 200 rule-blind, hold to 900 s)

Arms W15/W18/W21 = WHOLD_soft except wall speed 15/18/21 m/s; band 200-400 m
mainline, tau 3.0 s, 0.3 m/s² clip. 120 branches (30 seeds × C+3); integrity:
prefix identity through t_d = 200 and first divergence at step 201 on
90/90 treatment branches, zero collisions/teleports in 120/120, tau restored
90/90, C replay digest matches stored Stage 1a C 30/30, all holds ran the
full 700 s and released at the 900 s cap (time_cap 30/30 per arm).

| metric | W10 (WHOLD_soft) | W15 | W18 | W21 |
| --- | ---: | ---: | ---: | ---: |
| health incidence C → arm | 24 → **12** | 24 → 20 | 24 → 20 | 24 → 19 |
| paired reduction mean (CI90 / CI95) | 0.40 [0.233, 0.567] / [0.200, 0.600] | 0.133 [−0.033, 0.300] / [−0.067, 0.333] | 0.133 [−0.033, 0.300] / [−0.067, 0.333] | 0.167 [0.067, 0.267] / [0.033, 0.300] |
| prevented / induced | 13 / 1 | 7 / 3 (6005, 6009, 6025) | 7 / 3 (6005, 6025, 6029) | 5 / 0 |
| total duration delta median (CI90) | −180 s [−203, −158] | −127 s [−160, −96] | −138.5 s [−156, −81] | −84 s [−109, −62.5] |
| longest run delta median | −42 s | −15 s | −13.5 s | −11 s |
| delay median (p10 / p90; CI90) | +3.09 (−0.41 / +4.63; [2.43, 3.58]) | +1.58 (−0.38 / +5.80; [0.57, 2.70]) | +1.21 (−0.72 / +4.99; [0.25, 2.50]) | +1.05 (−0.51 / +4.13; [0.29, 2.04]) |
| fraction delay-beneficial | 0.133 | 0.167 | 0.200 | 0.233 |
| hard-brake excess total median | +166.5 | +119.0 | +105.5 | +89.5 |
| approach-edge HB excess median | +190.5 | +133.0 | +107.5 | +102.0 |
| HB phase (during / release-60 / after) | +164.5 / +1.5 / 0.0 | +105.5 / 0.0 / 0.0 | +93.5 / 0.0 / 0.0 | +89.5 / 0.0 / 0.0 |
| HB location (approach / merge / post) | +190.5 / −21.5 / 0.0 | +133.0 / −20.0 / 0.0 | +107.5 / −20.0 / 0.0 | +102.0 / −11.5 / +1.0 |
| approach-edge event-start speed p10/50/p90 (m/s) | 14.5 / 17.2 / 29.7 | 14.6 / 20.4 / 30.4 | 14.7 / 21.5 / 30.7 | 14.8 / 21.8 / 30.8 |
| low-TTC excess median (merge edge) | −1.0 | 0.0 | −0.5 | −1.0 |
| merge-edge flow 300-900 s, C / arm / delta | 2602.7 / 2593.7 / +3.0 | 2602.7 / 2605.7 / 0.0 | 2602.7 / 2602.7 / +3.0 | 2602.7 / 2590.7 / −6.0 |
| member-seconds median | 1205.5 | 994.5 | 951.0 | 913.5 |
| per-lane coverage median / full-coverage fraction | 0.589 / 0.359 | 0.530 / 0.288 | 0.507 / 0.265 | 0.493 / 0.254 |

### Descriptive paragraph (no selection)

Protection retained relative to W10 (CI90 lower bound of the incidence
reduction ≥ 0.15): **only W10 itself** (0.233). W15 and W18 sit at
[−0.033, 0.300] and W21 at [0.067, 0.267] — none of the scan arms retains
protection at the W10 level under the stated rule; the point reduction falls
from 0.40 (W10) to 0.13-0.17 (W15/W18/W21), with W15 and W18 each inducing
three collapses that C did not have (including 6025, which the W10 wall
prevented) while W21 induces none but prevents only five. Approach-edge
hard-brake excess relative to W10: W15 0.70, W18 0.56, W21 0.54 — the
excess falls monotonically as wall speed rises. Delay relative to W10:
W15 0.51, W18 0.39, W21 0.34 — monotonic decrease with speed. Merge-edge
flow stays within ±6 veh/h of C on all arms (the throttle is preserved as
speed rises — flow is essentially unchanged). In short: the axis trades
protection (W10 best) against delay and hard-brake cost (W21 best); the
middle speeds keep less of both and W15/W18 additionally create collapses
that C did not have.

## Artifacts

- `runs/probe/sustained_wall_v1_stage1b_prep2/prep2_part_a.json` — Part A
  (premium, regression, decomposition, a3 listings, severity, throttle).
  SHA-256 `6df8d7f03c8e74e3ece3c04d97fc4d0436b155fb39d0cee05a55d3de2ba4119f`.
- `runs/probe/sustained_wall_v1_stage1b_prep2/prep2_summary.json` — Part A
  + Part B (per-arm blocks, descriptive paragraph). SHA-256
  `8e395c201ea384953d159cdc6ed29808e406441c9b3a7eb338be1878ba474b11`.
- `runs/probe/sustained_wall_v1_stage1b_prep2_wallspeed_scan_m2200_p20_s6001_6030/`
  — scan manifest + per-seed artifacts (C/W15/W18/W21 branches, integrity,
  dose).
- Inputs reused (unchanged): `runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/`,
  `runs/probe/sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030/`.
- Code: `experiments/idm/run_sustained_wall_v1_stage1b_prep2_wall_speed_scan.py`,
  `experiments/idm/analyze_sustained_wall_v1_stage1b_prep2.py`.

## Claims not made

- No arm is selected, no gate is applied, no cell is endorsed; no model was
  fitted or screened (A1's regression is a descriptive slope with a
  bootstrap interval, not a predictor).
- No validation or prospective claim: seeds 6001-6030 remain development
  seeds excluded from prospective evaluation.
- The hard-brake speed/severity distributions are event-start granularity
  (per-step per-vehicle data were not persisted); the phase and road
  marginals are exact step counts.
- No generalization beyond 2200/400, p = 0.20, IDM, this corridor.
