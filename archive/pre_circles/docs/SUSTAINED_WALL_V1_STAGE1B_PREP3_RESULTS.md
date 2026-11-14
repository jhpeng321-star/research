# Stage 1b-prep-3 results — HRULE closed-loop baseline and single-engagement pilot

Date: 2026-09-02. Status: **development-only; no gate applied, no arm
selected** (see "claims not made").

## HRULE semantics implemented (recorded contract)

Reproduced from the legacy mechanisms (see `policies/inflow_control.py →
WallHoldPolicy._wall_active`; `env/wall_supervisor.py` corridor/health
conventions; `policies/metastable_rule.py` was inspected and does NOT define
the wall rule — it is the per-CAV h-table metering controller and its 5-step
release persistence applies to that module's release, not to the wall
trigger):

- Signal: merge-health h = `metastable_health_detail()[0]` (EMA-based),
  read once per controller step (policy pre-step; corridor post-step —
  identical EMA advance pattern to every other arm).
- Arm: h < 0.25, instantaneous (no persistence, no minimum-on time).
- Release: h > 0.45, instantaneous (no minimum-off time, no taper on
  release; released CAVs return to native IDM with their original tau).
- Re-engagement: allowed immediately after release (latch resets).
- Gate: the first arm condition that can fire is the h observed at t = 100;
  its first wall command is at step 101, so replay prefix identity with
  Arm C is exact through t = 100.

Ambiguities in the legacy code and how they were resolved (all recorded):
(a) the legacy `WallHoldPolicy` trigger has no t = 100 start gate — this
task adds it explicitly; (b) the legacy release decision is made on the
pre-step health read (a one-second lag in the release decision); (c) the
legacy trigger default release threshold is trigger + 0.2 = 0.45, matching
the task's 0.45; (d) release is instantaneous with no taper (the 5 s taper
exists only in the counterfactual wall machinery, not the legacy trigger
wall); (e) `policies/metastable_rule.py`'s 5-step release persistence is
part of that module's own metering rule and is NOT part of the legacy wall
trigger semantics.

SE(t_d) uses the same actuator and release rule: engage at the fixed
rule-blind step t_d regardless of h; release (instant) when the pre-step
health read exceeds 0.45, first evaluated one controller step after
engagement (so an engagement always applies at least one wall second), or
at the demand end (t = 900 s); no re-engagement.

Actuator (both arms): 200-400 m mainline band, wall speed 10 m/s, tau
+3.0 s via setTau, 0.3 m/s² clip, instant tau change; out-of-band CAVs are
released each ON step with their original tau (legacy passage semantics).
Rule state is owned by the wrapper; the WallHoldPolicy actuator is run with
its trigger removed, and the wrapper only ever issues tau-restore commands
to CAVs the wall actually touched (same-value setTau on hdv-mix vehicles
makes SUMO clone the type, which is why the touched-set bookkeeping exists —
a digest-visible, physics-neutral artifact).

## Results (filled after compute; see prep3_summary.json)

### Integrity

All 270 branches: replay identity with the stored cell C digests (30/30 per
run); prefix identity exact through t = 100 (HRULE) / t = t_d (SE) with first
divergence at the first actual engagement (≥ 101 / t_d + 1); zero
collisions and teleports in every branch; tau restored after every release
(probe at last-on + 3/6/10 s); verified natural clearance terminal in all
branches. The rule arms engage at very different times per seed (the health
condition decides), so divergence steps are heterogeneous by design.

### HRULE per cell (paired vs Arm C, n = 30)

| cell | C → HRULE incidence | reduction (CI90 / CI95) | prevented / induced | delay median (p10 / p90; CI90) | fraction beneficial | wall-on s median (IQR) | engagements / episode median | first engage s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2200 | 24 → 19 | 0.167 [0.000, 0.333] / [−0.033, 0.367] | 8 / 3 | +0.65 (−2.79 / +1.45; [0.38, 0.90]) | 0.267 | 535.5 (40.8) | 30.5 | 110.5 |
| 2100 | 17 → 11 | 0.200 [0.033, 0.367] / [0.000, 0.400] | 8 / 2 | +0.26 (−0.51 / +0.90; [0.05, 0.60]) | 0.200 | 446.0 (80.8) | 31.0 | 109 |
| 2000 | 9 → 7 | 0.067 [−0.067, 0.200] / [−0.100, 0.233] | 4 / 2 | +0.41 (−0.19 / +0.91; [0.26, 0.50]) | 0.167 | 405.5 (102.5) | 33.0 | 116 |

Wall-on split around the C onset (collapsing seeds): 2200 98 s before /
427.5 s after (n = 24); 2100 92 / 393 (n = 17); 2000 144 / 286 (n = 9) —
the rule spends most of its wall time AFTER the (native) onset. The
hysteresis is chattery: ~30 engagements per episode with first engagement
just after the 100 s gate on most seeds. Comfort cost (decel-cap exposure
excess ≥ −4.5 m/s², i.e., hard-brake steps): approach-edge median +87 (2200)
/ +56.5 (2100) / +41.5 (2000); merge-edge median 0.0 / −10 / −12;
post-merge ≈ 0. Low-TTC excess (merge edge): 2200 −1, 2100 −1, 2000 +1.
Merge-edge flow 300-900 s (C / HRULE): 2602.7 / 2593.7 (Δ −6.0); 2491.8 /
2491.8 (Δ 0.0); 2399.0 / 2402.0 (Δ 0.0) — the throttle is preserved.
Zero collisions/teleports in HRULE at all cells.

### 2200 one-axis (never / 180 s pulse / WHOLD_soft / HRULE)

| arm | health-collapse incidence | delay Δ median (s/veh) | wall-on time (s) |
| --- | ---: | ---: | ---: |
| never (C) | 24 | (baseline 13.6 s/veh mean) | 0 |
| 180 s pulse (Stage 1a W) | 24 | +0.655 | 180 |
| WHOLD_soft (700 s hold, 10 m/s) | 12 | +3.09 | 700 |
| HRULE (closed-loop h-rule) | 19 | +0.65 | 535.5 |

HRULE buys intermediate protection (5 prevented, 3 induced) at the same
delay cost as the short pulse (+0.65 s/veh) with 3x its wall-on time; the
blanket 700 s hold buys the most protection (12 prevented) but 4.7x the
delay cost.

### Part B — SE single-engagement pilot at 2100 (n = 30)

| arm | C → SE incidence | reduction (CI90 / CI95) | prevented / induced | delay Δ median (mean; fraction beneficial) | hold duration s median (p10 / p90) | release reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SE150 | 17 → 17 | 0.000 [0, 0] / [0, 0] | 0 / 0 | 0.000 (+0.020; 23.3%) | 1.5 (1.0 / 45.5) | health_release 30/30 |
| SE250 | 17 → 15 | 0.067 [0.000, 0.133] / [0.000, 0.167] | 2 (6098, 6108) / 0 | 0.000 (−0.344; 40.0%) | 9.5 (1.0 / 52.2) | health_release 30/30 |

At t = 150 the 2100 health is still above 0.45 on most seeds (see Part C2),
so the legacy release rule ends the engagement after a median of 1.5 s —
SE150 is a near-no-op (no incidence change, ~zero delay). SE250 holds a
median of 9.5 s, prevents 2 collapses (one seed with lead +103 s; the other
with lead −32 s — onset before the engagement), and shows a mean delay
reduction (−0.34 s/veh; 40% of seeds beneficial). Lead times
(C onset − t_d): SE250 prevented 35.5 median (range −32…+103); co-collapsing
+53 median (−114…+571). Delay-beneficial 2x2: **both** 2 seeds, **neither**
13, SE150-only 5, SE250-only 10.

### Part C — artifact-only

C1 wall-on vs Δdelay slope (s/veh per 100 s of wall-on, per cell): **2000
+0.50** (bootstrap CI90 [0.17, 0.73]) — consistent with the blanket-hold
engagement cost of +0.5 s/veh per 100 s; **2100 −0.23** (CI90 [−0.81, 0.27])
and **2200 −0.18** (CI90 [−2.08, 1.19]) — indistinguishable from zero and
not consistent with the blanket-hold reference. Interpretation: within the
chattery rule, total wall-on time varies little between seeds (IQR 41-102 s
against means of 405-535 s), so per-second cost is dominated by noise; only
at 2000, where the rule sees a scan of engage durations, does the
blanket-hold slope reappear. (The 2100 with the rule shows no cost at all.)

C2 h(t) > 0.45 fraction (Arm C): per-seed h at 100/150/200/250/300 is in
`prep3_summary.json → part_c.c2_health_signal_map.per_seed` (compact
listing). Fractions:

| t (s) | 2000 | 2100 | 2200 |
| --- | ---: | ---: | ---: |
| 100 | 0.700 | 0.733 | 0.700 |
| 150 | 0.400 | 0.567 | 0.467 |
| 200 | 0.267 | 0.300 | 0.333 |
| 250 | 0.200 | 0.267 | 0.100 |
| 300 | 0.500 | 0.333 | 0.167 |

The health signal is alive (> 0.45) on ~70% of seeds at t = 100 in every
cell, decays through 200-250, and partially recovers by 300 at 2000/2100
(0.50 / 0.33) — the map that motivates the SE150/S E250 choice and explains
SE150's immediate release.

## Artifacts

- `runs/probe/sustained_wall_v1_stage1b_prep3/prep3_summary.json` — all
  sections above (SHA-256 `d58b0aad6f77b6468bff1673dbed823465dd9c406dad38ba9881524081ac6978`).
- `runs/probe/sustained_wall_v1_stage1b_prep3_hrule_m{2200,2100,2000}_p20_s*/` —
  per-cell manifests and per-seed artifacts (C + HRULE branches, integrity,
  wall telemetry, tau probes).
- `runs/probe/sustained_wall_v1_stage1b_prep3_se_m2100_p20_s6091_6120/` —
  SE150/SE250 artifacts (C + 2 arms per seed).
- Code: `policies/sustained_wall_hrule.py`,
  `experiments/idm/run_sustained_wall_v1_stage1b_prep3.py`,
  `experiments/idm/analyze_sustained_wall_v1_stage1b_prep3.py`,
  `tests/test_sustained_wall_hrule.py` (5 tests; full suite 1010 passed
  before compute and after the final code).

## Claims not made

- No gate, no arm selection, no cell selection, no model fitting; comparisons
  are paired effect sizes with intervals.
- No validation or prospective claim: all seeds are development seeds in the
  prospective-evaluation exclusion set.
- SE hold durations below the demand-end maximum are classified as health
  release by construction (release reason is determined by the realized
  hold; the wall always releases by health before the end in this dataset).
- No generalization outside 2000-2200 veh/h, 400 veh/h ramp, p = 0.20, IDM.
