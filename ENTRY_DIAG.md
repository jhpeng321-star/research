# ENTRY_DIAG: why the WM-S imagination under-predicts the v5 policy's entry-queue effect (2026-10-15)

Branch `c4/v6entry`. Code + tests + this diagnosis only (no SUMO run, no GPU, no training).
Evidence: the six recorded eval episodes under `_ref/eval` (native `c00`, v5 policy `c01`,
MAPPO `c02`; worlds s4107_m2400_p0400 / s4108_m2400_p0100), the v5 WM `_ref/wm_s_best.pt`
(physics.json + TRAIN_RECEIPT.json: sumo_fidelity IDM a=2.49, `entry_insert_rule=sumo_like`,
entries_v2) and the v5 policy `_ref/policy_best.pt`. Scripts: `_out/v6entry/*.py`
(measure_b, v0_check, rule_check, wm_replay, entry_dynamics); outputs as JSON next to them.

## 1. Where the +597 veh-s lives: the mainline source, not the ramp

Per-entrance window ledger (t in [300, 900] s, ledger semantics of
`traffic_shepherd.metrics.RequestLedger`; veh-s, insertions in window, delay p50 of inserted):

| episode (arm) | entrance | due | ins | never | wait (veh-s) | delay p50 |
|---|---|---|---|---|---|---|
| s4107 p0.4 native | mainline | 599 | 489 | 110 | 45,820 | 79.3 |
| s4107 p0.4 **policy** | mainline | 599 | 486 | 113 | **46,691 (+871)** | 79.8 |
| s4107 p0.4 MAPPO | mainline | 599 | 489 | 110 | 45,811 (−9) | 79.3 |
| s4108 p0.1 native | mainline | 539 | 485 | 54 | 28,087 | 52.1 |
| s4108 p0.1 **policy** | mainline | 539 | 480 | 59 | **29,584 (+1,496)** | 54.3 |
| all four arms | ramp | 83–101 | all | 0 | 48–59 (**bit-identical across arms**) | 0.76 |

The whole effect is at the **mainline** entrance; the on-ramp (demand 101/600 s) is free in
every arm (entry gaps p50 128–162 m, entry speed = lane cap). The mainline source is
**saturated in every arm**: demand 2,400 veh/h = 0.665 veh/s vs a recorded service capacity
of 0.54 veh/s (two lanes x 0.27 = one insertion per 3.5 s and lane), leaving a standing queue
of 47 (p0.1) to 77 (p0.4) vehicles whose wait is 99 %+ of the window total.

**The service process** (per entrance lane, window, native arm): SUMO refuses while the
bumper gap between the departure position (front at lane start + length + 0.1,
`departPos="base"`) and the rearmost vehicle's rear is below the vType minGap — 98.9–99.9 %
of queued refusal ticks have end-of-tick bumper gap < 2.0 m (p50 −2.9) — and inserts the
moment it clears (gap at insertion p05/p50 = 2.0–2.1 / 2.9–3.2 m end-of-tick). Insertion
speed = **stop speed**, not follow speed: median recorded `v0` = 0.15–0.17 m/s while the
rearmost vehicle already moves at ~3.4–4.0 m/s; `v0` is reproduced by
`-tau*b + sqrt((tau*b)^2 + 2 b (g' - s0))` with the leader's END-OF-STEP rear position `g' =
g + v_lead * 0.5` (stop-speed at g': MAE 0.28–0.29 m/s, corr 0.6; crediting the leader speed
 overshoots by ~1 m/s, p50 1.7–2.5). Each entrant then needs ~4–4.5 s to clear the next
insertion's 12.1 m zone (acc0-2 s p50 0.93–1.02 m/s² → the 3.5 s median headway).

**The policy's mechanism**: v5 CAVs accelerate at a near-constant ~1.0 m/s² (recorded
requested accel median 0.99; native IDM uses up to 1.2), so CAVs leaving the entrance hold
the 12.1 m zone longer: acc0-2 s of inserted CAVs 0.89–0.94 (policy) vs 0.95–1.02 (native),
entry-edge CAV accel p50 0.99–1.04 vs 1.05–1.08, clear-15 m slower, per-lane service
0.537 vs 0.542 veh/s. On a 47–77 vehicle standing queue that marginal service loss
integrates to **+871 (p0.4) / +1,496 (p0.1) veh-s**; MAPPO (native-like accel, 0.04 % lane
commands) shows none (−9). Lane commands play at most a secondary role here: the rearmost
queue vehicle is a CAV only 11 % (p0.1) to 43 % (p0.4) of the time, the effect is present in
both, and lane-change opportunities within the first 100 m are rare on a 2.6 km entrance
edge.

## 2. The `sumo_like` rule does not reproduce this (rule check on SUMO's own states)

For every mainline-lane tick, the rule decision on the exact state SUMO saw (end of the
previous tick) vs SUMO's actual decision (inserted at the next tick / refused while queued).
True accept = share of SUMO's insertions admitted; false accept = share of queued refusal
ticks the rule would insert (`_out/v6entry/rule_check.json`):

| episode (lane) | SUMO ins / ref ticks | sumo_like true / false | sumo_like_v2 true / false | v0 p50 SUMO / like / v2 |
|---|---|---|---|---|
| native s4107 L0 | 163 / 1036 | **0.000** / 0.000 | **0.810** / 0.001 | 0.16 / – / 0.61 |
| native s4107 L1 | 162 / 1037 | **0.000** / 0.000 | **0.858** / 0.002 | 0.15 / – / 0.59 |
| native s4108 L0 | 162 / 1037 | **0.000** / 0.000 | **0.858** / 0.002 | 0.17 / – / 0.79 |
| policy s4107 L0 | 167 / 1032 | **0.000** / 0.000 | **0.832** / 0.002 | 0.17 / – / 0.63 |
| policy s4108 L1 | 158 / 1041 | **0.013** / 0.000 | **0.842** / 0.007 | 0.16 / 1.44 / 0.62 |

On a **saturated** entrance `sumo_like` refuses essentially every insertion SUMO actually
makes: it tests the PRE-step bumper gap (median 1.0 m at SUMO's insertions) against minGap
2.5 without the rearmost vehicle's within-step movement, and would insert at the FOLLOW-safe
speed (1.4–2.5+ m/s) where SUMO inserts at the stop speed (0.15). It "works" only on its own
optimistic roads (below). `sumo_like_v2` (this branch) tests the end-of-step gap
`g' = g + v_lead * dt` against the textbook minGap (2.0) and departs at the stop speed:
**true accept 0.81–0.86 at false accept 0.001–0.011**, predicted v0 p50 0.56–0.79 (right
scale; SUMO's own events sit exactly at the admission boundary where the stop speed is
0.15–0.2). The empirical admission boundary (end-of-step bumper gap at SUMO's insertions:
p05 1.64–1.74; at refusals: p99 1.57–1.74, max 2.2) is the textbook minGap 2.0; the legacy
vType table's 2.5 admits only 0.58.

## 3. Teacher-forced WM replay: predicted vs recorded (the fix on/off)

`_out/v6entry/wm_replay.py`: the v5 WM rolled 30 s (60 x 0.5 s, the Dreamer horizon) from
t0 in {300, 360, ..., 840} (10 windows per episode), t0 CAVs teacher-forced to their
recorded trajectories (the policy's own SUMO actions), HDVs are the WM's predictions,
entries ON (feat arrivals from the t0 entry features, counter rate p50 0.527–0.549 veh/s),
insertion rule switched. Predicted source wait = dt x backlog summed (per entrance):

| episode (arm) | rule | mainline ins / 30 s | mainline wait (veh-s, p90) | entrant v0 p50 | entry-zone occupancy pred (rec) |
|---|---|---|---|---|---|
| native s4107 | recorded | 16.4 | 2,239 | 0.15 | 1.00 |
| | sumo_like | 12.8 | 42 (95) | 3.0 | 0.91 |
| | **sumo_like_v2** | **13.8** | 39 (67) | **0.0** | **0.96** |
| native s4108 | recorded | 16.2 | 1,388 | 0.16 | 1.00 |
| | sumo_like | 15.7 | 5 (14) | 17.6 | 0.44 |
| | **sumo_like_v2** | 15.6 | 4 (9) | 7.9 | **0.60** |
| policy s4107 | recorded | 16.0 | 2,280 | 0.15 | 1.00 |
| | sumo_like | 13.8 | 22 (43) | 3.7 | 0.89 |
| | **sumo_like_v2** | 13.5 | 22 (46) | 2.5 | 0.92 |
| policy s4108 | recorded | 16.2 | 1,460 | 0.14 | 1.00 |
| | sumo_like | 15.7 | 7 (27) | 11.8 | 0.58 |
| | **sumo_like_v2** | 15.4 | **13 (43)** | 5.4 | **0.77** |

(ramp predicted wait 0.0–0.15 veh-s vs recorded 2.4–3.4 in all cells — the 0.5 s step
accounting; insertions match.)

- The trainer's "0.07 veh-s/step" imagination source wait is reproduced: the sumo_like
  p0.1 cells give 5–7 veh-s / 60 steps = 0.08–0.12/step.
- **Why the level is ~50x too small (both rules):** the recorded window wait IS the
  pre-t0 queue (mean queue x 30 s: 76.2 x 30 = 2,286 vs 2,239; 46.7 x 30 = 1,400 vs 1,388);
  the entry process starts `pending = 0` and the RSU cannot see SUMO's source backlog.
  Additionally the arrival rate is the INSERTION counter (0.53–0.55 veh/s = capacity), not
  the demand (0.665), so the imagined queue can only grow from `counter - service`.
- **Why the service margin is too high:** (i) the rule (section 2 — fixed here) and (ii)
  the WM's own road dynamics: on the entry edge the WM predicts t0-HDV speed gains of
  +2.16–2.45 m/s over 2 s where SUMO recorded +1.57–1.84 (+33 %; `_out/v6entry/entry_dynamics.json`),
  so the predicted 12.1 m insertion zone is occupied only 44–96 % of steps (recorded 100 %)
  and the imagined entrance serves 0.46–0.52 veh/s against arrivals of 0.53–0.55.
- With the fix the insertion EVENTS become SUMO-like (v0 near 0, occupancy +0.03–0.19,
  mainline insertions within 6 % of recorded in 3 of 4 cells), and the p0.1 policy cell
  now shows the effect in the right direction (native 4.4 vs policy 13.4 veh-s; recorded
  1,388 vs 1,460). The p0.4 cells remain dominated by window-to-window renewal noise
  (+-20 veh-s at n = 10 windows).

## 4. The fix: `insert_rule = "sumo_like_v2"` (default OFF, bit-identical)

`wms.model.entries`: `EntryConfig.insert_rule = "sumo_like_v2"` (`INSERT_RULES` extended)
reproduces SUMO 1.27's queued-entrance semantics measured above: the admission test runs at
the END of the insertion step (`g' = bumper gap + relu(v_lead) * dt`), the entrant departs
at the STOP speed behind that end-of-step rear position (leader speed not credited), with
the v5 textbook vType settings (`SUMO_V2_TAU_S/_DECEL_MPS2/_MIN_GAP_M`: cars tau 1.2,
decel 2.0, minGap 2.0; truck unchanged). Admission stays "end-of-step bumper gap >= minGap";
free lanes still enter at the lane-limit cap (recorded 31.29 / 22.22 m/s). Random draws are
shared with v2/sumo_like (common random numbers). Keys, all default `"v2"`:
- WM training: `data.entry_insert_rule = "sumo_like_v2"` (as in v4/v5 for sumo_like).
- Dreamer policy imagination: `trainer.entry_insert_rule = "sumo_like_v2"`
  (`configs/c4_policy_v6_entry.toml`, extends `c4_policy_v5.toml`).
- `wms.control.srcgrad` (the `source_wait_grad` surrogate, now usable together with the new
  rule): the soft test moves to the end-of-step gap against the textbook minGap, leader
  speed only credited as within-step movement.

## 5. What the fix does not repair (recorded limits, next actions)

- **Queue level** (the 47–77 vehicle pre-t0 backlog and demand > capacity) is invisible to
  the RSU-observable entry process (`pending` starts at 0, arrivals from the insertion
  counter). Closing that needs a queue-state input (e.g. the `critic_entry_backlog` channel
  fed by an RSU detector estimate) — policy-side config alone cannot do it.
- **WM road dynamics**: the +33 % entry-edge HDV acceleration bias keeps the imagined
  service at ~0.5 veh/s >= the counter rate, so the backlog still grows only slowly. The v6
  WM should train with `data.entry_insert_rule = "sumo_like_v2"` (SUMO-accurate entrances
  in the imagined roads) and the entry-edge dynamics revisited if the next SUMO head-to-head
  still shows the imagination insensitive at demand 2400.

## 6. Verification

- **Bit-identity of the defaults (golden)**: `_out/v6entry/golden_parity.py` runs this
  branch's entries module against the pre-change module (`git show 55be83a:...`) for rules
  "v2" and "sumo_like" (8 trials x 60 steps, shared arrivals, busy roads, seeded pending):
  masks, states, pending and summaries identical.
- `tests/unit/test_v6entry.py` (13 tests): defaults bit-identical (default == explicit v2,
  30-step rollouts); the saturated scene (sumo_like refuses both lanes, v2 admits at the
  analytic stop speed on the larger-gap lane, front at lane start + length + eps); stop
  speed maximal-safe and stationary-leader semantics; refuses inside minGap; free lane at
  the cap; randomised never-unsafe/never-overlapping (both lane modes); one-step-earlier
  queue service; dreamer/data/legacy-flat plumbing; the config file; the srcgrad soft
  surrogate follows the end-of-step gap, agrees with the hard process decision (>100
  sweeps), negative d p_block / d s; recorded-episode check (skipped when `_ref` absent):
  sumo_like < 5 % vs sumo_like_v2 > 70 % true accept on the v5 policy episode.
- Suites (OMP_NUM_THREADS=1): test_v6entry + test_entryfix + test_c4_policyv5 +
  test_c4_srcgrad + test_c4_srcwait_imagination + test_c4_dreamer: **78 passed**;
  test_c4_core + test_import_graph + test_wmloss + test_c4_srcwait_mappo +
  test_c4_policy2: **269 passed**; test_datahyg + test_r5_switches + test_c4_branching +
  test_c4_loop: **54 passed, 4 errors** (pre-existing on this host: the gitignored
  `data_cache/` pack roots of `configs/data_registry.toml` are absent; unrelated to this
  change).
- `ruff check` clean on src/wms, tests and configs (the repo does not enforce
  `ruff format`; unchanged files fail it too).
