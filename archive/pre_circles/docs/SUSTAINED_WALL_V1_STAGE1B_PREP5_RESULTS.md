# Stage 1b-prep-5 results — locator cells 2400-2700, damping arm D, damping+wall full hold, pooled observability

Date: 2026-09-03. Status: **development-only; no gate, no cell/arm selection**
(see "claims not made"). Amended scope: locator extended to 2700 (B), a
damping arm at every cell (B2), the full hold run on the damping
counterfactual at all six cells (C), and the pooled observability AUC (C2).

## B — locator rows (Arm C, native IDM; 2400/2600/2700 new seeds
6121-6150 / 6151-6180 / 6181-6210, registered as
`sustained_wall_v1_stage1b_prep5_locator` + exclusion set; grep-audited clean
before registration)

| cell | health incidence | onset median (s) | delay median (s/veh) |
| --- | ---: | ---: | ---: |
| 2000 | 9/30 | 383 | 11.37 |
| 2100 | 17/30 | 303 | 12.82 |
| 2200 | 24/30 | 258.5 | 12.44 |
| **2400** | **29/30** | **181** | **19.97** |
| **2600** | **30/30** | **171.5** | **37.42** |
| **2700** | **30/30** | **182.5** | **45.99** |

The native-IDM cell ladder saturates at 2400 (29/30) and fully at 2600/2700,
with the delay doubling from 2200 to 2600 and tripling at 2700 — the
metastable band of the physical label being empty is unchanged (0/30
everywhere). All 90 new C branches ended by verified natural clearance.

## B2 — damping arm D (frozen analytical damping, zero residual, from t = 0)

Repeat identity: exact digest identity on the first two seeds of every cell
(12/12 pairs). Command activity at p = 0.20 (per cell means): 2000 367
CAV-seconds / 27 distinct CAVs / 31.3% of steps with a command; 2100 382 /
26 / 32.8%; 2200 463 / 32.8 / 37.8%; 2400 553 / 38.9 / 42.9%; 2600 782 /
57.9 / 57.9%; 2700 797 / 60.7 / 59.1% — the damping controller is clearly
active at p = 0.20 and its activity rises with demand.

D − C per cell (paired, n = 30):

| cell | incidence C → D | delay Δ median (mean; CI90) | fraction beneficial | comfort (decel-cap) approach / merge |
| --- | ---: | ---: | ---: | ---: |
| 2000 | 9 → 12 | +0.041 (+0.13; [0.010, 0.068]) | 0.333 | 0.0 / −1.0 |
| 2100 | 17 → 16 | −0.055 (−0.32; [−0.121, −0.030]) | 0.733 | 0.0 / −1.5 |
| 2200 | 24 → **29** | +1.891 (+2.08; [1.81, 2.06]) | 0.067 | +10.5 / +17.5 |
| 2400 | 29 → 29 | +0.092 (−0.44; [−0.91, 0.44]) | 0.433 | −1.0 / +4.5 |
| 2600 | 30 → 30 | −1.193 (−1.70; [−2.99, 0.44]) | 0.633 | −1.0 / −1.5 |
| 2700 | 30 → 30 | +0.607 (−0.04; [−1.50, 2.08]) | 0.433 | −1.0 / −1.5 |

**Damping gives limited (and at 2200 negative) help**: it increases
incidence at 2000 (9→12) and 2200 (24→29) and costs +1.89 s/veh at 2200;
at 2100 and 2600-2700 it is roughly neutral on incidence and mildly
delay-reducing at 2600 (mean −1.70 s/veh, CI90 of median crosses zero).
Collapse premium within D (collapsing minus non-collapsing delay): 2000
+0.45, 2100 +1.16, 2200 +2.15, 2400 +5.80 s/veh; undefined at 2600/2700
where all D seeds collapse.

## C — damping + wall full hold (branching from D, t_d = 250, hold to 900,
10 m/s, tau 3.0, 200-400 m mainline, 5 s taper; 180 branches)

Integrity: prefix identity through 250 and first divergence at 251 on
180/180 branches; zero collisions/teleports; verified natural clearance.
W − D per cell:

| cell | incidence D → W | delay Δ median (mean; CI90) | fraction beneficial | duration Δ median (s) | comfort appr / merge | merge flow med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 12 → 6 | +2.19 (+2.25; [1.79, 2.47]) | 0.000 | −92.5 | +83 / −23 | 2396 |
| 2100 | 16 → 9 | +2.64 (+2.77; [2.43, 3.00]) | 0.000 | −108.5 | +113 / −30 | 2483 |
| 2200 | 29 → 19 | +2.56 (+2.57; [2.20, 3.46]) | 0.100 | −189.5 | +140.5 / −22 | 2600 |
| 2400 | 29 → 28 | +2.29 (+1.93; [1.21, 4.50]) | 0.267 | −201.0 | +193.5 / −2.5 | 2782 |
| 2600 | 30 → 30 | −2.43 (−1.50; [−4.82, 1.04]) | 0.600 | −116.5 | +214 / +22.5 | 2902 |
| 2700 | 30 → 30 | **−8.56** (−5.99; [−9.52, −6.43]) | 0.800 | −45.5 | +192 / +14 | 2971 |

Low-TTC excess (merge edge) −1.0/−0.5/−1.0/−1.0/0.0/−1.5; approach edge
+2 at 2100, +1 at 2400/2700, 0 elsewhere. The W treatment halves collapse
incidence at 2000-2200 (12→6, 16→9, 29→19) at a cost of ~+2.2-2.6 s/veh of
delay, barely moves incidence at 2400-2700 (all cells already at 29-30), and
**reduces delay at 2600 (−1.5 mean) and 2700 (−6.0 mean, CI90 [−9.5,−6.4])**
— the management channel (wall orders the discharge of a persisting
breakdown). Note the incidence never decreases at 2400-2700 despite the
large delay wins — "delay benefit without incidence change" is the 2600-2700
regime.

**Sign-change curve (one axis across all six cells, mean Δdelay W − D):**
2000 +2.25, 2100 +2.77, 2200 +2.57, 2400 +1.93, **2600 −1.50**, **2700
−5.99** (medians +2.19 / +2.64 / +2.56 / +2.29 / −2.43 / −8.56). The sign
flips between 2400 and 2600 — exactly the monotonic-vs-controlled transition
the escalation gate would have to learn; at 2700 the wall is strongly
delay-beneficial in 80% of seeds.

## C2 — pooled observability on the damping prefix (Arm D vector, t = 250)

The full RSU observation vector was recorded in every D branch at
t = 150/200/250/300 (same schema as prep-4; per-seed step_features stored).
Pooled across all six cells (n = 180, 80 positive for the target
"W − D Δdelay < 0"), univariate AUC at t = 250:

1. **near_merge_count 0.883**
2. **upstream_count 0.859**
3. **merge_edge_count 0.851**
4. post_merge_count 0.707
5. approach_count 0.695
6. near_merge_speed_std 0.582
7. ramp_speed_std 0.581
8. post_merge_speed_std 0.561
9. merge_edge_speed_std 0.540
10. ramp_count 0.520

The first direct observability test: the pooled signal is dominated by the
**counting** variables (near-merge, upstream and merge-edge vehicle counts
0.85-0.88), i.e. by the throughput/density levels, while speed-dispersion
variables are near chance. Caveat recorded: the pooled target is strongly
cell-correlated (the sign flips with demand), so these AUCs partly measure
cell separability; a within-cell test is the next check.

## Artifacts and hashes

- `runs/probe/sustained_wall_v1_stage1b_prep5/prep5_summary.json` — all
  sections above. SHA-256 `ce0bf560d5b20b536e5c256ccc037b9f1284efc8dd28c47741c9bc8123812c1e`.
- Per-seed artifacts: `prep5_C_m{2400,2600,2700}_p20_s*/`,
  `prep5_D_m{2000,2100,2200,2400,2600,2700}_p20_s*/` (with command activity,
  obs features and repeat checks), `prep5_W_m{...}_p20_s*/` (D and W branches,
  integrity, dose).
- Code: `policies/damping_plus_wall.py`,
  `experiments/idm/run_sustained_wall_v1_stage1b_prep5.py`,
  `experiments/idm/analyze_sustained_wall_v1_stage1b_prep5.py`,
  `tests/test_damping_plus_wall.py` (2 tests; full suite **1014 passed**
  before compute and after the final code).
- Seed registration: `idm/contract.py` blocks for 2400/2600/2700 + exclusion.

## Production notes

- All stages ran strictly sequentially (8 workers) after the prep-4
  concurrency lesson; every C/W prefix and D repeat is digest-verified.
- The runner's audit whitelist now includes every prep-family root and all
  prep-5 siblings (glob).

## Claims not made

- No gate, no cell or arm selection; the sign-change curve is descriptive.
- C2 AUCs are univariate rank statistics on the pooled damping prefix; no
  model, no predictor claim, and the pool conflates cell membership with
  the target — a within-cell test is required before any observability claim.
- The damping arm is the frozen analytical controller (zero residual);
  "limited help" describes the measured D − C row, not a general damping
  verdict.
- No validation or prospective claim: all seeds (6001-6210) are development
  seeds in the prospective-exclusion set.
- No generalization outside 1900-2700 veh/h, 400 veh/h ramp, p = 0.20, IDM.
