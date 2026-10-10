# DIAGNOSIS — WD-01-PA2G-LC1: the residual lane-change mismatch of Part A
# v2 general on chunk-4 __native_base TRAIN truth

- **Registration:** `output/wd01/REGISTRATION_PA2G_LC1.md` (LF-sha256
  prefix c09131957836def4; authority: owner decision (b), 2026-09-30).
- **Status:** COMPLETED 2026-09-30. Zero SUMO/TraCI. TRAIN worlds only;
  opened run files recorded per script and asserted (`assert_train_only`
  passed in every pass; lists saved under `results/lc1/*opened*.json`).
- **Panel:** the same 32 PA2G-A2 TRAIN worlds (`results/lc1_a2_worlds32.json`),
  18 windows each (w = t0+30k, k=1..18), H ∈ {10,30,60,120}.
- **Delay-scope note (owner decision (c)):** Part A is tracked-scope;
  Part B carries untracked arrivals. All counts below are tracked-cohort
  counts; nothing here speaks to system delay.

## Headline (post-LC1-fixes eval, Table 1')

v2g lane-change counts, predicted/true summed over windows
(`results/eval_a2_post_lc1/summary.json`; the class-(ii) counting fix of
§1 is applied to both sides):

| model|H | n | merge p/t | disc p/t | disc ratio |
|---|---|---|---|---|
| v2g\|H10 | 576 | 398/607 | 2034/442 | 4.60x |
| v2g\|H30 | 576 | 647/1749 | 6226/1132 | 5.50x |
| v2g\|H60 | 576 | 1737/2320 | 8364/1397 | 5.99x |
| v2g\|H120 | 512 | 1951/2120 | 7247/1389 | 5.22x |

The residual mismatch stands after the counting fix: discretionary
over-production ~4.6–6.0x and merge under-production at short H (0.66x at
H10, 0.37x at H30) flipping to slight over-production at H120 (0.92x).

## 1. Counting audit (done FIRST)

**Method.** An independent counter
(`diag_lc1_audit.py`) re-counts every truth lane change directly from the
raw `physics_steps.jsonl` rows. It does NOT import `partA_data.pdl` or the
eval's `ModelMapper`: the lane_phys table is hand-derived from the frozen
network (`diag_lc1_netcheck.py`; accel lane `1487606179_0` is a DEAD END
at junction `:13638027693` — it has no onward connection, so every truth
merge is a genuine SUMO lane change on the 75.44 m merge edge). Events =
lane_phys changes between consecutive 0.5 s rows per vehicle; runs of
junction-internal rows resolve to the next physical lane (the eval's
`retrospective_junction` convention); every event is classified by
(from lane_phys, to lane_phys, s, junction flag).

**Whole-run truth event census (32 worlds, all vehicles):**
`1->0` 2409, `0->1` 1222, `2->1` 2120 — **no other class occurs**
(vehicles never change onto/off the accel lane except by merging).
5770 events; **100% carry SUMO's own `lane_changed` flag** on the
physical row. The raw flag is set on 50,352 ticks: the excess ~44.6k are
junction-internal-lane transitions and intra-junction lateral moves (e.g.
`ramp.34` crossed junction `:1780739017` onto main_right at t=313.0 and
re-changed to main_left at t=313.5) — the flag alone is NOT a valid
physical-LC definition; the phys-transition definition is.

**Eval vs independent counter, per (world, w, H) row (2240 rows):**

| tracked-cohort rule | equal | mismatch | eval extra events |
|---|---|---|---|
| raw physical rows only | 2174 | 66 | +47 merge, +8 disc_left, +11 disc_right |
| mid-junction placed at destination (eval retro-fix convention) | **2240** | **0** | — |

**Verdict:** the eval's lane-change counting itself is correct and
semantically identical to the independent counter. The only truth-side
difference is the tracked-cohort rule: the eval (via `retrospective_
junction`) TRACKS vehicles that are junction-INTERNAL at a window start,
placing them at their destination segment start (s0). Prediction
snapshots never contain junction states, so the convention cannot be
mirrored by the prediction side — class **(ii)**.

**Class-(ii) fix (applied, tested):** `NativeLog` now records the raw
internal (tick, vid) membership (`inj`), and `tracked_state(..., inj_at)`
excludes mid-junction vehicles from the tracked cohort
(`eval_a.py`, used by `eval_a2.py` and `eval_fixcheck.py`).
Regression tests: `partA_v2g/tests/test_lc1_counting_fix.py`
(5 tests, green). After the fix, my counter matches the saved post-fix
eval rows **exactly on all 2240 truth rows and all 2240 prediction rows**
(`diag_lc1_keycheck.py`).

**Post-LC1-fixes A2 re-run** (same 32 worlds, same flags, labelled
"post-LC1 fixes", `results/eval_a2_post_lc1/`): the fix moves the
headline negligibly (truth disc 443→442 @H10, 1393→1389 @H120; truth
merge 611→607 @H10; v2g disc pred 2044→2034 @H10; s_RMSE/delay shifts
are second-order; full before/after table in
`results/lc1/../lc1_beforeafter.out` and the ledger draft). **The
4.6–6.0x disc mismatch and the short-H merge under-production are NOT a
counting artifact.**

Audit note (transparency): the first version of my OWN independent
counter had a bug — it skipped the direct event check on the row that
resolves a junction run, missing 19 events (eval-higher mismatches
included my misses). Fixed and re-run; all numbers above are from the
fixed counter.

## 2. Decomposition (tracked cohort, events in (w, w+120]; 576 windows;
machine tables in `results/lc1/tables.md` / `tables.json`)

Lane pair: **merge 2213 pred / 2406 truth (0.92x); disc_left 6791/1012
(6.71x); disc_right 1878/510 (3.68x)** — the surplus is 2.3x larger in
the leftward (overtaking) direction.

s-position: disc events spread over the whole zone in both, but pred
over-fires everywhere (every 100 m bin ≥ 4x truth except the last two);
merge position: truth mean s=432.3 m with **22.4% of merges below s=415**
(early edge), pred mean 440.3 m with only **7.5% below 415** — the model
merges late, concentrated in the last ~25 m.

Regime at the event (same-lane neighbours within ±50 m | their mean
speed): truth events sit mostly in FREE/LOW-density regimes (n0: 764,
n1-2: 527 of 3928); pred events cluster in congested regimes
(n3-5|v<10: 1824, n6+|v<10: 2327 of 8669). The model manufactures its
lane changes inside its own congestion.

CAV vs HDV: per 1000 vehicle-seconds the over-production is essentially
equal (CAV 0.56→1.66, HDV 0.59→1.63) — no CAV-specific effect.

Time since window start: **pred 2436 vs truth 1049 events in the first
10 s (2.32x) while the initial states are IDENTICAL by construction** —
the rule over-triggers from step one; the ratio then grows (5.7x
aggregate) as the predicted state diverges.

Repeats: pred 9626 vehicles change (vs 3809 truth); 1203 pred vehicles
change 2+ times (42 change 4–9 times; truth: 0); pred change-back within
10 s rate 5.7% vs truth 1.8%.

Merge outcomes (merge-edge vehicles at w): truth n=1163, merged within
120 s 88%, never-merged 139, **stopped dead at the edge end 48** (SUMO
had to rescue them later; junction transits then force the merge);
pred n=1121, merged 98%, never-merged 24, stopped 0 (the urgency-relaxed
rule always eventually accepts; the engine never fully blocks).
**Dwell: truth median 12 s, pred median 34.5 s** — this dwell gap is the
short-H under-production: within H=10/H=30 most pred merges simply
haven't happened yet.

## 3. Per-event MOBIL traces (worst windows)

`results/lc1/traces.md` / `traces.json`: for the 12 worst windows
(disc surplus 35–47 per window), every dumped decision is an ACCEPTED
predicted change with NO truth lane change of that vehicle within ±5 s
(30–48 pred-only decisions per window). Representative trace
(`w8694m2600p40t240`, w=690, `mainline.298`, t=724, 1→0):

- terms: gain 0.618 ≥ eff 0.400 (thr 0.10, bias 0.30, p 0.50);
  da_ego = 0.849 − 0.448; d_new_foll = 0.861 − 0.996; d_old_foll =
  0.941 − 0.372; gap_lead 5.2 m (v_lead 28.1), gap_foll 24.9 m.
- truth beside it: the vehicle stays on lane 1 (truth gap ahead 19.4 m)
  and does NOT change within ±5 s (nor at all).

Reading: under 0.5 s ballistic integration the instantaneous IDM
accelerations make ANY accelerating queue look like a MOBIL win — the
ego "gains" 0.4 m/s² by jumping to a lane whose leader is 5 m away but
fast. The textbook criterion (instantaneous accelerations, no maneuver
time, no hysteresis, no strategic lane preference) has no mechanism that
reproduces the SUMO fleet's reluctance; the keep-right bias (0.3) only
shifts the direction mix.

Decision-side diagnostics (all 576 windows): rejected-candidate tallies
are dominated by disc|b_safe 1.43M and disc|threshold 0.99M; every
accepted decision's re-derived term vector was asserted equal to the
engine's own decision (trace engine; zero assertion failures) — **no
class-(i) implementation bug found in the engine's lane-change layer.**

## 4. Descriptive ablations — LABELLED: NOT ADOPTED, NOT SELECTED

32 worlds × 18 windows × full 120 s rollouts; `params_nominal.json`
UNTOUCHED (deep-copied overrides in memory). Event totals across all
windows:

| variant | disc_left | disc_right | merge | disc/window | change-back rate |
|---|---|---|---|---|---|
| **truth (reference, window slices)** | **1012** | **510** | **2406** | **2.64** | **0.018** |
| nominal (p 0.5, bias 0.3, thr 0.1) | 6791 | 1878 | 2213 | 14.15 | 0.057 |
| politeness 0.2 | 15863 | 10720 | 2266 | 40.54 | 0.417 |
| politeness 1.0 | 6728 | 1818 | 2193 | 13.94 | 0.038 |
| bias 0 (SYMMETRIC MOBIL) | 10583 | 4583 | 2141 | 23.50 | 0.288 |
| threshold 0.3 | 5668 | 1275 | 2246 | 11.25 | 0.029 |
| urgency onset mid-edge (s=416.88) | 6768 | 1858 | 2215 | 14.08 | 0.056 |

(The nominal-vs-symmetric pair IS the registered asymmetric-vs-symmetric
comparison: the nominal keep-right bias 0.3 is the textbook European
asymmetric form.)

Findings: (a) politeness is the main stabilizer — 0.2 triples the churn
and quadruples ping-pong; 1.0 changes nothing (−1.4%); (b) removing the
keep-right bias makes churn WORSE (+75%, all leftward); (c) threshold
0.3 removes only 20% — no tested parameter combination approaches the
truth rate; (d) merge-urgency onset is immaterial (13/499 window merge
counts differ, total 2215 vs 2213): merging is gated by the safe-gap
kinematics (rejected merge candidates: foll_bsafe 53k, lead_headway
28.8k, foll_s0 14.3k, threshold 76.5k), not by the urgency schedule.
**No ablation was adopted; params_nominal.json is unchanged (sha256
verified pre/post).**

## 5. Attribution to classes (i)–(iv), with rough shares

- **(i) Implementation bug: NONE.** Trace-engine assertions (decision
  terms == engine decisions on every candidate), counting identity on
  all 4480 eval rows, and the regression tests found nothing; the PA2G
  churn fix held.
- **(ii) Measurement/classification: found and FIXED (~1.5–2% of truth
  events).** The mid-junction tracked-start convention gave the eval 66
  extra events (+47 merge, +8 disc_left, +11 disc_right; 104 events on
  the pre-fix convention) — class (ii) fix + tests applied; before/after
  A2 metrics labelled "post-LC1 fixes" (§1). The residual mismatch is
  real.
- **(iii) Model structure: DOMINANT for disc over-production.** At t≤10 s
  (identical initial states) the model already fires 2.32x the truth
  rate — that floor is pure decision-rule semantics (instantaneous MOBIL
  gains under 0.5 s ballistic integration, no maneuver time, no
  lane-hold hysteresis, no strategic preference). Ablations confirm no
  threshold/bias/politeness value in the registered grids removes it.
  For merges, the late-merging (dwell median 34.5 vs 12 s; 7.5% vs
  22.4% early-edge merges) is structural: strict kinematic safe-gaps +
  MOBIL ego-gain on a target lane that is slower near the merge, versus
  SUMO's strategic LC2013 merging early on the edge and its forced
  dead-end rescue (48 truth stops, 0 pred).
- **(iv) Input-state effects: amplification, not the root.** Pred events
  concentrate in the model's own congestion (n3+/n6+ regimes 48% of pred
  events vs 5% of truth events), and the surplus ratio grows from 2.32x
  (t≤10 s) to 5.7x (120 s) as predicted queues compress and discharge
  differently (free-road truncation, no arrivals). Rough shares of the
  disc surplus: (iii) ≈ the entire 2.32x floor at matched states (≥40%
  of the final 5.7x), (iv) divergence amplification ≈ the remaining
  ~60%, (ii) ≈ 1.5–2% (fixed), (i) 0.

## 6. Recommended owner options

1. **Keep the textbook model and make C4L1 (b) reported-only** — accept
   the disc over-production (4.6–6.0x) and short-H merge lag as
   characterized model error of textbook MOBIL/merge under dt = 0.5 s.
   The counting is now provably identical and the residual is honest.
2. **Approve a specific structural addition** (separately registered):
   candidates with ablation-supported mechanisms: (a) a lane-hold /
   maneuver-time hysteresis (targets the 5.7% change-back and repeat
   changers), (b) a strategic early-merge incentive on the accel lane
   (targets dwell 34.5→12 s and the early-edge share), (c) an
   anticipation/lateral-dynamics term. Politeness/threshold/bias
   re-tuning alone is NOT sufficient (§4).
3. **Approve a parameter change on TRAIN evidence:** not recommended on
   this evidence — the best single change (threshold 0.3) removes only
   20% of the surplus and politeness ≥ 0.5 is already at the flat end.

## 7. Provenance and artifacts

- Registration: `output/wd01/REGISTRATION_PA2G_LC1.md`.
- Scripts (`output/wd01/partA_v2g/`): `diag_lc1_audit.py` (counting
  audit + truth events), `diag_lc1_pred.py` (instrumented v2g rollouts),
  `diag_lc1_truth_merge.py`, `diag_lc1_ablation.py`,
  `diag_lc1_tables.py` (aggregation only), `diag_lc1_traces.py`,
  `diag_lc1_beforeafter.py`, `diag_lc1_netcheck.py` (+ small diff/inspect
  helpers `diag_lc1_mminspect/mmdiff/mmdiff2/rampr95/r95t810/wcheck/
  keycheck/kcheck/uncov/slicecheck/validate/ablcheck/ablsplit.py`).
- Raw rows BEFORE aggregation: `results/lc1/truth_events.jsonl` (5770),
  `truth_windows.jsonl` (2240), `pred_events.jsonl` (11076),
  `pred_decisions_accepted.jsonl` (11272), `pred_rejections.jsonl` (576),
  `pred_window_rows.jsonl` (576), `pred_states.jsonl` (576),
  `pred_merge_outcomes.jsonl` (2410), `merge_outcomes_truth.jsonl`
  (8187), `ablation_rows.jsonl` (3456), `ablation_events.jsonl` (87807).
- Aggregated: `results/lc1/tables.json`, `tables.md`, `traces.json`,
  `traces.md`, `audit_report.json`, `worst_windows.json`.
- Post-fix A2 re-run: `results/eval_a2_post_lc1/` (raw rows
  8432 lines, summary.json, opened_run_files_full.json; run log
  `lc1_postfix_run.out`; wall 1722 s at 4 workers).
- Budget: registration allowed ≤ 3 h CPU at ≤ 4 workers. Charged passes:
  audit 3×~60 s, post-fix A2 1722 s, pred runs (3 launches: 2 invalidated
  by writer/shard bugs, preserved in logs; final 170 s), truth-merge
  63 s, ablation 559 s — total ≈ 0.83 h wall at ≤ 4 workers, zero SUMO.
- params_nominal.json unchanged: LF-sha256
  `079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7`
  (raw-file sha256 = param_set_id
  `BA1FD93F0A21B1FFD2CDF20693E74FB8E3FBB8E4A1C57E7F88D9A3929E3455CE`),
  verified before and after this task.
- TRAIN only: plan split asserted before every open; CALIBRATION/
  VALIDATION/HOLDOUT never opened; opened-file lists:
  `results/lc1/{pred,traces,ablation_}opened_run_files.json`,
  `truthmerge_opened_run_files.json`, `audit_report.json.opened_run_files`,
  `results/eval_a2_post_lc1/opened_run_files_full.json`.

Delay-scope note (owner decision (c)): Part A is tracked-scope; Part B
carries untracked arrivals.
