# Stage 1b-prep-4 results — legacy actuator fidelity, minimum-hold pilot, observation map

Date: 2026-09-02/03. Status: **development-only; no gate, no arm selection**
(see "claims not made").

## Part A — legacy actuator fidelity at 2200/p20

### A1. Recovery (provenance recorded)

`tools/legacy_probe/_probe_metastable_feasibility.py` was recovered verbatim
from git history (commit 8d993e9; SHA-256
`0d9c4635b98517f0644f2f4ab04535e2242ed786e925bbca4aaa2f3a514930e4`; identical
at its last pre-deletion commit 6f8aad7; deleted by the 2026-08-27 cleanup).
Dependencies survive in the current tree (`evaluate_policy.run_policy_evaluation`,
`policies.baselines.NoOpPolicy`, `policies.metastable_rule.HealthRecordingWrapper`).
**Recorded discrepancy**: the recovered driver's variant tables stop at stage
7; the §7 stage-10 table (with `w10_t30_h025`) was never committed (full-
history grep for `w10_t30` finds only naming comments in `env/wall_supervisor.py`
and `policies/collapse_gate.py`). The stage-10 rows used here are a documented
reconstruction in `tools/legacy_probe/run_stage10_wall_probe.py`:
w10_t30_h025 = WallHoldPolicy with the RUNGS[1] operating point named
"w10_t30" (band 200-400 m, 10 m/s, tau 3.0) + legacy health trigger
arm 0.25 / release 0.45 (+0.2 default), 0.3 m/s² clip, instant tau. The
alternative band reading (200-1500 m default) is recorded but not used.

### A2. Legacy driver run

`run_stage10_wall_probe.py --seeds 6001-6030` on
`_light/osm.dense.light_m2200r400idm.sumocfg` (long map, pen 0.2, steps 1800)
produced 30 `w10_t30_h025` rows (schema 2: sustained_collapse, loss, tt,
first_collapse_step, ttc_events, dv_mean, h_mean, upper_frac) and 30 shared
no_op cache rows (same scene). Row files:
`runs/probe/sustained_wall_v1_stage1b_prep4_legacy_probe/{rows,noop_cache}/`.

### A3. Legacy vs ours (same seeds)

- **no_op vs our Arm C: 100% per-seed agreement (30/30); both 24/30
  collapse incidence; zero disagreeing seeds.** The legacy no-op path
  (NoOpPolicy mode-0 + HealthRecordingWrapper) is physics-identical to our
  no-command Arm C on the same seeds — this is the actuator-fidelity control.
- **w10_t30_h025 vs our HRULE (band-identical, both trigger 0.25/0.45):**
  legacy wall incidence 17/30; our HRULE 19/30 (C 24/30). Per-seed agreement
  (relative to C): **both prevent 7** (6002, 6006, 6010, 6022, 6024, 6027,
  6030), **only legacy prevents 3** (6007, 6017, 6028), **only ours prevents
  1** (6015), **13 both collapse**, **6 C-no-collapse**, neither-prevent 0.
  Mean per-vehicle time loss (legacy loss field): w10 ≥ no_op on 7 of the
  10 seeds where either prevented (6015 +1.0, 6028 −3.8 are the two
  exceptions) — the legacy wall also costs delay where it prevents.
- Differences remain (1 vs 2 health-EMA advances per step: the legacy
  wrapper reads h once per step, our corridor adds a second read; different
  band behavior on ramp CAVs — see A4) — enough to move 4 of 30 seeds
  (3 legacy-only + 1 ours-only).

### A4. TraCI command diff (seed 6001, steps 100-400; ours = the
`sustained_wall.py` sustained actuator in a 180 s pulse; legacy = the
trigger wall driven from t=0)

Command-kind facts (full logs stored in `part_a4.json`):

1. **Engagement trigger (KIND difference)**: ours = rule-blind window
   (first command at step 201); legacy = health hysteresis from step 1
   (first command at step 100 on seed 6001).
2. **Membership population (KIND difference)**: ours commands only mainline
   (PRE_MERGE_EDGE) CAVs (`_mainline_obs` filter); legacy's WallHoldPolicy
   receives the full active-CAV dict, so **ramp CAVs with
   distance-to-merge in [200, 378.66] m are also walled** (legacy log shows
   `ramp.20` configured + commanded at step 173).
3. **Breadth of the release path (KIND difference)**: legacy's inactive
   path issues `setTau` + `setSpeed(-1)` for every mainline CAV passed each
   OFF step (4834 setTau / 4517 setSpeed in-window); ours restores only
   wall-touched CAVs (2347 setTau / 2095 setSpeed) — a deliberate deviation
   (same-value setTau clones the vTypeDistribution child type, which is a
   digest-visible but physics-neutral artifact).
4. **Taper (KIND difference)**: ours ends with the existing 5 s linear
   taper (`act_with_targets` blended to native speed + `set_tau_blend` of
   the wall tau offset), then `restore_tau`; legacy releases instantly
   (tau original + speed −1 in one shot, no blend).
5. **setSpeed vs slowDown**: neither uses `slowDown`; both use `setSpeed`
   (with −1 as release) — same kind, value-only differences.
6. **speedMode / laneChangeMode**: both use `configure_cav` → 31 / 1621 on
   commanded CAVs (identical bits); ours 31 configures in-window vs legacy
   10 (different on-times).
7. **t = 100 gate**: ours engages only via the rule wrapper (t_d-based);
   the legacy trigger has no gate.
8. **tau restore timing**: legacy at the release step; ours at taper end
   (blended during taper).

## Part B — minimum-hold pilot at 2100/p20 (t_d = 250 rule-blind)

Integrity: prefix identity through 250, first divergence at 251, zero
collisions/teleports, tau restored, verified natural clearance, C replay
digest matches the stored band-locator C 30/30 on 60/60 branches.

| metric | SE250 (instant, prep-3) | SE250_H120 | SE250_H300 |
| --- | ---: | ---: | ---: |
| C → arm incidence | 17 → 15 | 17 → **9** | 17 → **8** |
| reduction (CI90 / CI95) | 0.067 [0.000, 0.133] / [0, 0.167] | 0.267 [0.100, 0.433] / [0.100, 0.467] | 0.300 [0.133, 0.467] / [0.100, 0.500] |
| prevented / induced | 2 / 0 | 9 (6092, 6098, 6102, 6104, 6108, 6114, 6115, 6116, 6119) / 1 (6093) | 10 (+6120) / 1 (6093) |
| duration Δ median (longest run) | — | −128.0 s (−11.5) | −131.0 s (−12.5) |
| delay s/veh median (mean; p10 / p90; CI90 of median) | 0.000 (0.00; −0.28 / +0.06; [−0.007, 0.0004]) | +2.28 (+2.07; +0.67 / +4.07; [2.12, 2.66]) | +2.41 (+2.13; +0.67 / +4.07; [2.12, 2.67]) |
| fraction delay-beneficial | 0.40 | 0.067 | 0.067 |
| realized hold median (p10 / p90) | 9.5 (1.0 / 52.2) | 650 (650 / 650) | 650 (650 / 650) |
| release reasons | health 30/30 | demand_end 29, health 1 | demand_end 30, health 0 |
| lead time prevented (median; min-max) | 35.5 (−32…+103) | +120 (−32…+571) | +111.5 (−32…+571) |
| lead time co-collapsing median | 53 | +5.5 | −21 |
| comfort (decel-cap) excess: approach / merge | — | +128 / −29 | +133 / −30 |
| low-TTC excess: approach / merge | — | +1 / −1 | +1 / −1 |
| merge-edge flow 300-900 s (C / arm / Δ) | — | 2491.8 / 2479.9 / −6.0 | 2491.8 / 2479.9 / −6.0 |

Reading: with the 30 s health-persistence release rule, almost no seed
releases before the demand end at 2100 (the post-250 health stays below
0.45), so the minimum-hold distinction is nearly moot — both H120 and H300
behave like a full 251-900 hold, cutting incidence 17→~9-8 at a +2.3 s/veh
median delay (7x the SE250-instant delay for 4-5x the protection; about a
quarter of the WHOLD_soft-type delay at 2200 for the same duration).

## Part C — raw-observation map and descriptive AUC

The Arm C replay was repeated per cell with exact digests (30/30 per cell:
0 mismatches) and the full RSU aggregate vector was recorded at every step
40-310 and evaluated at t = 100/150/200/250/300 (variables: per-region
count / density / mean / harmonic / std for upstream>400 m, approach
200-400 m, near-merge <200 m, merge edge, post-merge, ramp; mainline lane
imbalance; ramp queue length and 30 s ramp arrivals; 30 s upstream inflow
minus merge discharge; 10/30/60 s merge-density and approach-std backward
differences; CAV band/upstream counts; h). Artifacts:
`runs/probe/sustained_wall_v1_stage1b_prep4_obsmap_m{2000,2100,2200}_p20_s*/`.
The per-seed per-t table plus the full AUC array is in `prep4_summary.json →
part_c` (target positives: seeds with onset strictly inside (t, t+horizon]).

Top five variables by AUC (descriptive, no model), at the two times of
interest (t = 150 and t = 200; h>0.45 fraction shown):

| cell / target | t | h>0.45 | top-five (AUC) |
| --- | ---: | ---: | --- |
| 2000 / 100 s | 150 | 0.40 | upstream_speed_std 0.78, post_merge_mean 0.75, post_merge_harmonic 0.74, merge_edge_count 0.69, merge_edge_density 0.69 |
| 2000 / 100 s | 200 | 0.27 | post_merge_speed_std 1.00, ramp_harmonic 1.00, ramp_mean 1.00, lane_imbalance 0.97, post_merge_count 0.97 (npos = 1 — degenerate) |
| 2000 / 200 s | 150 | 0.40 | upstream_speed_std 0.78, post_merge_mean 0.75, post_merge_harmonic 0.74, merge_edge_count 0.69, merge_edge_density 0.69 |
| 2000 / 200 s | 200 | 0.27 | ramp_speed_std 0.91, lane_imbalance 0.78, ramp_count 0.76, upstream_harmonic 0.75, upstream_mean 0.75 |
| 2100 / 100 s | 150 | 0.57 | approach_count 0.70, approach_density 0.70, post_merge_mean 0.69, post_merge_harmonic 0.66, near_merge_speed_std 0.65 |
| 2100 / 100 s | 200 | 0.30 | **h 0.80**, near_merge_speed_std 0.73, near_merge_count 0.68, near_merge_density 0.68, merge_edge_harmonic 0.63 |
| 2100 / 200 s | 150 | 0.57 | post_merge_harmonic 0.78, post_merge_mean 0.78, ramp_mean 0.75, ramp_harmonic 0.75, approach_count 0.67 |
| 2100 / 200 s | 200 | 0.30 | h 0.73, merge_edge_harmonic 0.70, upstream_speed_std 0.65, merge_edge_mean 0.62, post_merge_mean 0.61 |
| 2200 / 100 s | 150 | 0.47 | upstream_count 0.70, upstream_density 0.70, approach_harmonic 0.68, lane_imbalance 0.67, approach_mean 0.67 |
| 2200 / 100 s | 200 | 0.33 | upstream_speed_std 0.73, merge_edge_mean 0.72, near_merge_count 0.72, near_merge_density 0.72, h 0.71 |
| 2200 / 200 s | 150 | 0.47 | lane_imbalance 0.70, upstream_count 0.67, upstream_density 0.67, approach_harmonic 0.57, ramp_speed_std 0.57 |
| 2200 / 200 s | 200 | 0.33 | **h 0.76**, near_merge_count 0.70, near_merge_density 0.70, merge_edge_mean 0.64, upstream_speed_std 0.62 |

**Does anything beat 0.75 at t = 150/200 where h has saturated?** At t = 150
(h>0.45 = 0.40-0.57, not yet saturated): 2000's upstream_speed_std 0.78 (+
post_merge_mean 0.75) exceed 0.75 (npos = 3-9); 2100/2200 do not (top 0.70).
At t = 200 (h>0.45 = 0.27-0.33, saturated): **h itself is the (only)
reliable variable where it exceeds 0.75 — 0.80 at 2100/100 s (npos = 5) and
0.76 at 2200/200 s (npos = 10)**; 2000's 1.00 values come from exactly one
positive event (degenerate, not interpretable). The post-merge speed
signals at 2000/150 s and the near-merge/ramp/lane-imbalance cluster at
2000/200 s also reach 0.75-0.91 with npos = 3 — weak but non-degenerate in
the cell where h is least informative. No variable exceeds 0.75 at t = 150
where h is still mostly alive.

## Artifacts and hashes

- `runs/probe/sustained_wall_v1_stage1b_prep4/prep4_summary.json` — Parts B
  and C (SHA-256 `f322663a5d95957ae9dce7ada9f6a708956b7864b49d297c2d0b74d8a71a4c47`).
- `runs/probe/sustained_wall_v1_stage1b_prep4_legacy_probe/part_a3.json`,
  `.../part_a4.json` — A3/A4 machine outputs.
- `runs/probe/sustained_wall_v1_stage1b_prep4_legacy_probe/{rows,noop_cache}/`
  — 60 legacy rows.
- `runs/probe/sustained_wall_v1_stage1b_prep4_minhold_m2100_p20_s6091_6120/`,
  `runs/probe/sustained_wall_v1_stage1b_prep4_obsmap_m{2000,2100,2200}_p20_s*/`.
- Code: `tools/legacy_probe/` (recovered driver + README + stage-10 runner),
  `policies/sustained_wall_minhold.py`,
  `experiments/idm/run_sustained_wall_v1_stage1b_prep4_minhold.py`,
  `..._observation_map.py`, `analyze_sustained_wall_v1_stage1b_prep4.py`,
  `..._prep4_part_a.py`, `tests/test_sustained_wall_minhold.py`.
- pytest: 1012 passed before compute and after the final code.

Production note (recorded): the minhold/obsmap batches failed under
concurrent execution (two chains of 8 workers); sequential single-chain
reruns reproduced every stored digest exactly (30/30 per run).

## Claims not made

- No gate, no arm/cell selection; AUCs are univariate descriptive rank
  statistics, not a fitted model and not a predictor claim.
- The legacy stage-10 table is a documented reconstruction (the committed
  driver ends at stage 7) — the legacy w10 numbers here are produced by the
  recovered harness with the RUNGS[1] operating point, not by a
  byte-recovered pre-cleanup configuration.
- No validation or prospective claim: all seeds are development seeds in
  the prospective-exclusion set.
- Per-region densities use published geometry constants (upstream span,
  band lengths, merge-edge 300 m / 3 lanes); post-merge and ramp densities
  are not defined and are not reported.
- AUCs with a single positive event (2000 t=200 onset-100 s) are reported
  as degenerate and are not interpretable.
