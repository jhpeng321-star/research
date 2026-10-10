# Sustained-wall v1 Stage 1a — effect-existence pilot results (2200/400, p=0.20)

Date: 2026-09-02. Status: **development-only effect-existence pilot; no gate
is applied and no claim of efficacy or safety passes** (see the "claims not
made" section). Follows the estimand reset
(`docs/ESTIMAND_RESET_2026-09-02.md`).

## 1. Frozen definitions used

Cell and seeds (per `experiments/idm/sustained_wall_v1_contract.py` and
`idm/contract.py`):

- Scenario `scenarios/idm_transition/osm.dense.idm_m2200r400.sumocfg`
  (network `D:/research/highway_merge_shepherd_long/osm.net.xml.gz`, demand
  0-900 s, all vTypes `carFollowModel="IDM"`); mainline 2200 veh/h, ramp
  400 veh/h; CAV penetration 0.20 (seeded Bernoulli scatter, `cav_placement
  scatter`, CAVs set to `normal_car`); SUMO tick 0.5 s, controller step 1.0 s.
- Seeds 6001-6030 registered as `sustained_wall_v1_stage1a_development` and
  added to `PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS` (with all other development
  and validation partitions). Grep audit of docs/, runs/ manifests and code
  found no prior assignment of 6001-6030 (no hits).
- Horizons: common 1080 s or verified natural clearance (v5.7.1 terminal rule).
  In the event all 90 branches terminated by **natural clearance** (SUMO
  `terminated`, final live count 0, expected 0, no collision/teleport) at
  final sim time 1010-1040 s (median 1024.5), i.e. before the 1080 s cap.

Treatment arms (all native IDM; all branches replay from t = 0 with the
frozen SUMO/CAV seed = 7xx... seed id):

- **Arm C** (control): native IDM, no commands at all (`policies/noop_base.py`
  returns `{}` and never touches a vehicle; the env receives an empty action
  dict).
- **Arm W** (treatment): sustained health-gated moving wall.
  Decision landmark t_d: first controller step t ≥ 200 with h(t) ≤ 0.45 and
  h(t) − h(t−30) ≤ 0, h = `env.core…metastable_health_detail()[0]` on the
  Arm C prefix (the arms share the prefix). First wall command at step
  t_d + 1 (twin-replay convention: decision at t_d, commands from t_d + 1;
  this is also the measured first-divergence step). Actuator = the legacy
  `WallHoldPolicy` band-passage semantics: every CAV whose distance-to-merge
  is in [200, 400] m on a mainline lane is a member while in the band;
  members hold 10 m/s (0.3 m/s² rate clip) with tau raised to
  original + 3.0 s; leaving the band releases the CAV to native IDM
  immediately (tau restored). No latched cohort, no Gate F, no compactness.
  Release on (i) h > 0.45 for 20 consecutive pre-step readings, (ii) t_d +
  180 s of wall, (iii) t = 900 s; then the existing 5 s linear taper (speed
  blend to the native IDM speed plus tau-offset blend) on the current
  members and final tau restore.
- **Arm P** (perturbation noise floor): at t_d + 1 the CAV nearest the 300 m
  distance-to-merge point on a mainline lane receives a single one-step
  slowDown of 0.5 m/s (setSpeed v−0.5 for exactly one step), then returns to
  native IDM (setSpeed −1).

Outcome labels (frozen): health collapse = h < 0.15 for ≥ 60 consecutive
seconds after t = 100; physical collapse = ≥ 5 premerge vehicles < 15 m/s
and premerge mean ≤ 15 m/s for 60 consecutive seconds; hard brake = accel
≤ −4.5 m/s²; low TTC = TTC < 1.5 s; delay = incremental time-loss over the
post-decision window divided by distinct vehicles seen.

## 2. Implementation choice (recorded)

The CounterfactualEscalationPolicy path was examined first (its wall is a
latched member-slot cohort with formation gates, activation-safety vetos,
abort machinery and step-boundary-only taper) and rejected: it cannot
express membership-by-passage or the health-persistence release, and wiring
it in would require modifying frozen policy code. **Choice: wrap
`WallHoldPolicy` directly in the new direct-control policy
`policies/sustained_wall.py`** (plus `policies/noop_base.py`) and reuse the
twin-replay identity machinery (`capture_step`, `compare_traces`,
`ReplayTrace`) and `counterfactual/outcomes.py` inside the new runner
`experiments/idm/run_sustained_wall_v1_stage1a.py`. Secondary choices
recorded: membership restricted to the mainline approach edge (ramp CAVs
excluded); tau change instant (legacy default `tau_ramp_step = 0`); the wall
runs with zero members so ex-members are released every step; no members at
release ⇒ release is instant (no taper needed); the t = 900 s cap can never
bind because t_d ≤ 221 → t_d + 180 ≤ 401 (it is kept for completeness);
the [t_d, t_release+60] analysis window for Arm P uses the same seed's
Arm W release step (P has no own release); health is read
exactly once per step by the policy (pre-step) and once by the corridor
observation (post-step) in every arm so the health EMA advance pattern is
arm-invariant.

## 3. Integrity results

All 30 seeds, per-seed records in
`runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/seeds/seed*.json`:

- Prefix identity: exact replay digest identity through t_d for W vs C and
  P vs C on **30/30** seeds.
- First divergence: **exactly step t_d + 1** for both W and P on all 30
  seeds (no divergence was ever recorded earlier); divergence was present
  for P on every seed (the nudge was recorded, on 1 of 30 seeds it changed
  outcomes materially — seed 6002).
- Arm C repeat (seeds 6001 and 6002): **exact digest identity** between two
  independent runs on both seeds.
- Collisions and teleports: **zero** unique vehicles and zero vehicle-steps
  in all 90 branches.
- Terminal classification: natural clearance in 90/90 branches (final live
  count 0, expected 0, end reason `terminated`); paired branches of the
  same seed share the same terminal class, reason and final SUMO time.
- Wall engagement and dose integrity: wall active ≥ 1 step on 30/30 seeds;
  tau restored after release (max CAV tau at release+3/6/10 s = 1.1-1.5 s
  on all 30 seeds; 4.1 s never observed post-release).
- Per-branch wall-clock (dry run, seeds 6001-6002): 78-97 s, well under the
  500 s budget (seed 6001: C 80.3 s, C-repeat 93.3 s, P 92.1 s, W 96.5 s;
  seed 6002: C 78.0 s, C-repeat 91.2 s, P 89.8 s, W 94.9 s).
- Re-run note: seeds 6005, 6012, 6018, 6019, 6020, 6022, 6027, 6028 and
  6001 were re-run after a fixed one-line bug in the tau-integrity probe
  (dead code referencing a removed variable; it had failed those branches
  after the SUMO compute, before any JSON was written). Exact-replay checks
  confirm determinism of the final artifacts.

## 4. Analysis sections

### 4.1 Landmark statistics

**30/30 seeds had a landmark**; zero no-landmark seeds. t_d distribution
(min/median/max 200/200/221; values: 200 ×20, 201 ×2, 202, 204, 205, 208 ×2,
214, 220, 221). Health at landmark h(t_d) ≤ 0.45 by construction; typical
values were 0.03-0.44. Note: on several seeds the health collapse label
(the h < 0.15 run) had already started before t = 200 (earliest onset 119 s),
so the landmark is often the earliest allowed step, 200 s, during an
in-progress collapse.

### 4.2 Collapse incidence and duration

Health label (h < 0.15 for ≥ 60 s after t = 100):

| arm | incidence |
| --- | ---: |
| C | 24/30 (0.80) |
| W | 24/30 (0.80) |
| P | 24/30 (0.80) |

Paired incidence reduction W − C: mean 0.000 (no seed toggled *in both*
directions — one prevented [6012 C=1/W=0] and one caused [6025 C=0/W=1],
net 0). Seed-level bootstrap (2000 resamples) of the mean reduction:
95% CI [−0.100, +0.100], 90% CI [−0.067, +0.067]. P − C: identical (0.000;
CI [−0.067, +0.067] 90%, [−0.100, +0.100] 95%).

Physical label (v5.5 queue label): **0/30 in all three arms** — the label
has zero events in this cell, exactly the 2200-2400 removal the estimand
reset diagnosed. All physical-label duration deltas are 0.

Duration changes (W − C, health label): total duration median −30 s,
IQR 88.5 s (q25 −90.0 s, q75 −1.5 s), p10 −120.5 s / p90 +30.7 s,
min −343 s, max +183 s; bootstrap CI of the median: 95%
[−62.5, −15], 90% [−54, −18]. Longest run median 0 s (IQR 21 s; p10 −195.8 s,
p90 +37.3 s; min −560 s, max +476 s); CI of median 95% [−10, 0], 90%
[−5.5, 0]. 22/30 seeds shortened total duration, 7 lengthened, 1 unchanged.
P − C: median 0 s; p10 −0.3 s (interpolated), p90 0 s; longest run delta
0 s everywhere; onset moved by ±2 s on a handful of seeds (non-zero total
duration deltas: +57, −80, −6, −3, +2 s on 5 seeds).

### 4.3 Delay

Mean per-vehicle delay delta (s/veh, post-decision window):

| arm | median | IQR | p10 | p90 | min | max | CI90/95 of median | fraction beneficial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| W | +0.655 | 1.14 | −0.329 | +1.952 | −8.26 | +13.48 | [+0.203, +0.797] / [+0.146, +0.880] | 7/30 = 23.3% |
| P | +0.00003 | 0.0001 | −0.0056 | +0.0004 | −0.203 | +0.594 | [~+0.00001, +0.00008] | 7/30 = 23.3% |

W's delay effect is a **positive** median (+0.66 s/veh, CI excluding 0 —
the wall costs delay) with a heavy right tail (max +13.5 s/veh on seed 6006);
on 23% of seeds delay improved. P's delay effect is effectively zero except
for seed 6002 (+0.59 s/veh) and small negatives (−0.20).

### 4.4 Noise floor (perturbation P 5-50-95 percentiles; fraction of Delta_W outside)

Network-wide (whole horizon; delay = per-step incremental delay in s):

| metric | Delta_P p5 | p50 | p95 | fraction Delta_W outside |
| --- | ---: | ---: | ---: | ---: |
| hard-brake vehicle-steps | −3.0 | 0.0 | +0.55 | 30/30 = 1.00 |
| low-TTC exposure steps | 0.0 | 0.0 | 0.0 | 15/30 = 0.50 |
| delay (s) | −45.7 | +0.02 | +1.75 | 27/30 = 0.90 |
| merge discharge count | 0.0 | 0.0 | 0.0 | 0/30 = 0.00 |

Window [t_d, t_release + 60 s]:

| metric | Delta_P p5 | p50 | p95 | fraction Delta_W outside |
| --- | ---: | ---: | ---: | ---: |
| hard-brake vehicle-steps | −4.65 | 0.0 | +0.55 | 30/30 = 1.00 |
| low-TTC exposure steps | 0.0 | 0.0 | 0.0 | 10/30 = 0.33 |
| delay (s) | −0.49 | +0.03 | +41.1 | 29/30 = 0.97 |
| merge discharge count | 0.0 | 0.0 | 0.0 | 16/30 = 0.53 |

Delta_W hard-brake was **** positive on all 30 seeds (median +35, p10 +8.8,
p90 +62.6, min +2, max +154 vehicle-steps) — the wall's hard-brake effect
sits above the perturbation band on every seed (upward direction). Delta_W
low-TTC (median 0, min −4, max +6) and discharge (network change 0 on all
seeds) are within or near the P band. Delta_P itself is mostly zero with a
2-seed upper tail (+1, +20 hard-brake in-window; +18 network; delay +41 s
in-window on seed 6006) and small dips on a few seeds.

### 4.5 Dose

Realized wall duration: 180 s on 27 seeds; 136 s (6006), 106 s (6008),
44 s (6030) on the 3 health-recovery seeds; median 180 s (p10 175.6; the
p10 reflects the health-recovery runs). Release reasons: `max_duration` 27,
`health_recovery` 3, `time_cap` 0 (the 900 s cap cannot bind: t_d ≤ 221 ⇒
t_d + 180 ≤ 401). Member-seconds: median 250.0, p10 180.8, p90 398.7.
Mean member count per wall step: median 1.46 (min 0.85, max 3.13);
max member count per seed: median 4 (p10 3, p90 6.1). Per-lane coverage
fraction (occupied mainline lanes with ≥ 1 member): per-seed mean median
0.527 (min 0.356, max 0.728); per-step distribution over engaged steps
p10 0.0 / median 0.5 / p90 1.0.

### 4.6 Per-seed table (seed, t_d, C/W collapse (health), delay, hard-brake, wall)

Plain-text listing (`summary.json` → `per_seed_table` is the machine form):

| seed | t_d | C | W | P | delay ΔW | delay ΔP | hb ΔW | wall s | reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6001 | 221 | 1 | 1 | 1 | +0.10 | −0.000 | +36 | 180 | max_duration |
| 6002 | 205 | 1 | 1 | 1 | +0.87 | +0.59 | +57 | 180 | max_duration |
| 6003 | 200 | 0 | 0 | 0 | +0.89 | 0.00 | +27 | 180 | max_duration |
| 6004 | 200 | 1 | 1 | 1 | −0.02 | 0.00 | +9 | 180 | max_duration |
| 6005 | 220 | 0 | 0 | 0 | −0.16 | 0.00 | +7 | 180 | max_duration |
| 6006 | 200 | 1 | 1 | 1 | **+13.48** | −0.04 | +154 | 136 | health_recovery |
| 6007 | 200 | 1 | 1 | 1 | +2.06 | −0.20 | +51 | 180 | max_duration |
| 6008 | 208 | 1 | 1 | 1 | +0.67 | 0.00 | +30 | 106 | health_recovery |
| 6009 | 208 | 0 | 0 | 0 | +0.64 | 0.00 | +11 | 180 | max_duration |
| 6010 | 201 | 1 | 1 | 1 | +0.72 | 0.00 | +52 | 180 | max_duration |
| 6011 | 200 | 1 | 1 | 1 | +1.39 | 0.00 | +35 | 180 | max_duration |
| 6012 | 200 | 1 | **0** | 1 | +0.19 | 0.00 | +11 | 180 | max_duration |
| 6013 | 200 | 1 | 1 | 1 | **−8.26** | 0.00 | +77 | 180 | max_duration |
| 6014 | 200 | 1 | 1 | 1 | +1.94 | 0.00 | +61 | 180 | max_duration |
| 6015 | 200 | 1 | 1 | 1 | +1.30 | 0.00 | +39 | 180 | max_duration |
| 6016 | 204 | 0 | 0 | 0 | +0.68 | 0.00 | +23 | 180 | max_duration |
| 6017 | 200 | 1 | 1 | 1 | −0.00 | 0.00 | +5 | 180 | max_duration |
| 6018 | 200 | 1 | 1 | 1 | −1.88 | 0.00 | +145 | 180 | max_duration |
| 6019 | 200 | 1 | 1 | 1 | +0.09 | 0.00 | +15 | 180 | max_duration |
| 6020 | 201 | 1 | 1 | 1 | +1.03 | 0.00 | +55 | 180 | max_duration |
| 6021 | 200 | 1 | 1 | 1 | −0.07 | 0.00 | +13 | 180 | max_duration |
| 6022 | 200 | 1 | 1 | 1 | +0.30 | −0.09 | +10 | 180 | max_duration |
| 6023 | 200 | 1 | 1 | 1 | +0.42 | 0.00 | +35 | 180 | max_duration |
| 6024 | 202 | 1 | 1 | 1 | +1.31 | 0.00 | +52 | 180 | max_duration |
| 6025 | 200 | **0** | **1** | 0 | +2.26 | 0.00 | +45 | 180 | max_duration |
| 6026 | 214 | 1 | 1 | 1 | +0.65 | 0.00 | +34 | 180 | max_duration |
| 6027 | 200 | 1 | 1 | 1 | +0.66 | 0.00 | +19 | 180 | max_duration |
| 6028 | 200 | 1 | 1 | 1 | −3.29 | 0.00 | +40 | 180 | max_duration |
| 6029 | 200 | 0 | 0 | 0 | +1.36 | 0.00 | +50 | 180 | max_duration |
| 6030 | 200 | 1 | 1 | 1 | +0.10 | 0.00 | +2 | 44 | health_recovery |

(C/W columns: 1 = health collapse present, 0 = absent. P column = same as C
except 6002/6007/6022 onset/duration shifts; see 4.2. Δs are s/veh and
hard-brake vehicle-steps.)

## 5. Artifacts

- `runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/manifest.json`
  (frozen cell/arm definitions, outcome definition, exclusion note, terminal
  rule, prior-seed artifact audit = no conflicts).
- `runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/summary.json`
  — **SHA-256 `bef7a1a61e225d8f61fb38a727c3fb2ef0edde83e35b8a6ea3fd7e749302f4cf`**
  (all sections 4.1-4.6 machine-readable; 2000-resample bootstraps seeded
  20260902 for reproducibility).
- `runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/seeds/seed6001.json`
  … `seed6030.json` (per-seed: branch digests, outcomes, health/physical
  collapse statistics, per-step series, dose telemetry, integrity and repeat
  checks).
- Code: `policies/noop_base.py`, `policies/sustained_wall.py`,
  `experiments/idm/sustained_wall_v1_contract.py`,
  `experiments/idm/run_sustained_wall_v1_stage1a.py`,
  `experiments/idm/analyze_sustained_wall_v1_stage1a.py`,
  `tests/test_sustained_wall_v1.py` (15 focused tests; full suite 1000
  passed before any SUMO compute and 1000 after the final code).

## 6. Claims not made

- No predictor, screening model, or outcome fit was trained or evaluated on
  these data (no sklearn/torch on outcomes anywhere in this run).
- No prospective or validation claim: 6001-6030 are development seeds,
  registered in the exclusion set for prospective evaluation; nothing here
  is a frozen or validated result.
- No generalization beyond the single 2200/400, p = 0.20 cell.
- No pass/fail gate was applied and no treatment is endorsed: the wall
  showed **no incidence reduction** (24/30 → 24/30), a **positive median
  delay cost** (+0.66 s/veh, CI excluding 0), a modest duration reduction
  (−30 s median total-duration delta, CI [−54, −18] 90%), and **hard-brake
  excess on every seed** above the perturbation-noise floor (+2…+154
  vehicle-steps vs P p95 +0.55). Whether the duration reduction is science
  or delay-queue accounting, and whether a different gate/dose changes the
  picture, is for the separately versioned stages.
