# WD-01 VALIDATION LOOK #8 — REPORT (frozen p3o candidate, gate v2)

**VAL look #8; holdout not accessed.** ZERO simulation (no SUMO/TraCI
anywhere; the only dynamics is the pure-torch part-3 rollout engine).
This is the single, final validation look for this candidate
(WD-01-DEV-2 ledger: 7 prior validation looks; ONE look in val8). The
evaluation pass ran exactly once with the frozen settings; everything
below is aggregated from the raw rows saved by that one pass
(`results/raw/`, written BEFORE aggregation; `aggregate_val8.py`
re-rolls nothing).

## Verdict (single overall verdict at the PRIMARY horizon)

**GATE v2 at H = 30 (PRIMARY, AMENDMENT_1): FAIL — criterion (c) only**
(delay-phys widened coverage 0.7692 < 0.80; TTC coverage and the CRPS
leg pass). (a), (b), (d) pass. At H = 60 (secondary) gate v2 PASSES
for the frozen candidate (L6c). The old gate passes at H = 30 and
fails at H = 60 (speed RMSE 5.0152 ≥ 4.97). Per the one-look policy
there is no re-run, no refit, no tuning; diagnosis continues on TRAIN
only.

## Hash verification (eval + aggregation both re-verify; LF-normalised sha256)

Registered freeze 2026-09-27T05:09:19Z (sidecars
`candidate_p3o_iter2.json.FROZEN.sha256` /
`calibration_p3o.json.FROZEN.sha256`, both
`frozen_utc=2026-09-27T05:09:19.400476+00:00`). The brief registered
the gate-file hashes truncated (84b8a23d…a47f / 27aeddab…79d4); the
full values below match those prefixes/suffixes.

```
HASH OK: candidate_p3o_iter2.json        458890d9c4b0ee6ee03de145be50498d9bd0873505a4e86f2bf8a1e224ebe45d
HASH OK: calibration_p3o.json            c7ffa7540768fa67e75e99881eec6f801201990dea4ace90af93c202f789c22a
HASH OK: VALIDATION_GATE_V2.md           84b8a23df1ff654f8d602e1fd75119517c4bd7cb458dfb2191d2d088452da47f
HASH OK: VALIDATION_GATE_V2_AMENDMENT_1.md 27aeddab51a899dd1e7a7e24f073fbfe4d29efbe6c2cd3db7e992663d39d79d4
SIDECAR OK: candidate_p3o_iter2.json.FROZEN.sha256 / calibration_p3o.json.FROZEN.sha256
HASH GATE PASSED (candidate, calibration, gate v2, amendment 1).
```

## What was evaluated (exactly as registered)

- Frozen candidate `candidate_p3o_iter2.json`: L6 kind='idm_merge',
  5 members, part-3o engine flags `lc_scale_disc` +
  `merge_accept_prob` (winners: lc_scale_disc 0.0016,
  merge_accept_prob 0.55, tau_accept 2.4, b_accept 1.0,
  end_force_s 60.0). No parameter changed anywhere in this look.
- Frozen spread factors `calibration_p3o.json` (mean-centred
  widening): delay_phys {30: 2.7638, 60: 2.5198}, ttc3
  {30: 3.0314, 60: 6.3496}. L6c = L6 + these factors.
- 4 native Part A validation roots cv20_7005m2000, cv20_7006m2000,
  cv30_7305m2700, seed_7535m2400; t0 every 30 s in [120, 480]
  (13 × 4 = 52 windows); M = 5 members; H ∈ {30, 60}.
- L0 (deterministic IDM-normal) rolled in the SAME run on the same 52
  windows — the (c) MAE and the (d) per-horizon reference.
- Split asserts: validation ∩ (fit ∪ calibration ∪ holdout) = ∅; the
  5 holdout roots were never passed to `root_arrays`
  (`holdout_accessed = false`).

Raw rows saved before aggregation: **L6 520 = 5 members × 52 windows
× 2 H; L0 104.**

## Gate v2 — per criterion, H = 30 (PRIMARY) and H = 60

Pasted from `results/stdout/stdout_aggregate_val8.txt`
(L6c = frozen candidate + frozen spread; L6 = raw ensembles):

```
== GATE v2 at H=30 (PRIMARY (AMENDMENT_1)) -- (d) BINDING per-horizon form ==
  L6c: (a) coll/1000=0.022 PASS | (b) lc 13.904/12.077 (15.1%) PASS | (c) ttc3 0.8077 delay-phys 0.7692 crps 37.49 vs mae 293.39 -> FAIL | (d) speed 4.2393 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  L6:  (a) coll/1000=0.022 PASS | (b) lc 13.904/12.077 (15.1%) PASS | (c) ttc3 0.4231 delay-phys 0.2115 crps 39.59 vs mae 293.39 -> FAIL | (d) speed 4.2393 vs 1.05xL0@H=4.3975 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
  ==> OVERALL GATE v2 H=30 (PRIMARY, frozen candidate L6c): FAIL: c_coverage_crps

== GATE v2 at H=60 (secondary) -- (d) BINDING per-horizon form ==
  L6c: (a) coll/1000=0.018 PASS | (b) lc 25.712/23.942 (7.4%) PASS | (c) ttc3 0.9038 delay-phys 0.8462 crps 94.31 vs mae 738.77 -> PASS | (d) speed 5.0152 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 PASS
  L6:  (a) coll/1000=0.018 PASS | (b) lc 25.712/23.942 (7.4%) PASS | (c) ttc3 0.4423 delay-phys 0.3654 crps 89.84 vs mae 738.77 -> FAIL | (d) speed 5.0152 vs 1.05xL0@H=5.2263 (PASS, binding) / fixed 5.2185 (PASS) ==> v2 FAIL: c_coverage_crps
```

Reading of the primary failure: criterion (c) is a conjunction; at
H = 30 the TTC coverage leg passes (0.8077 ≥ 0.80) and the CRPS leg
passes (37.49 < L0 MAE 293.39), but the physics-delay 10–90 % band
covers 0.7692 of windows (40/52) — 2 windows short of the 0.80
threshold. On the TRAIN-internal calibration split the same factors
gave 0.8571 (H30) / 0.8214 (H60), so this is a genuine transfer
shortfall, not a bookkeeping difference. Widening is exactly what
carries the candidate: raw L6 delay coverage is 0.2115 / 0.3654.

## OLD gate (always reported alongside)

```
== OLD gate at H=30 ==
  L6c: speed 4.2393 < 4.97 True | coll 0.022 True | lc 15.1% True | delay minmax 0.8462 True ==> PASS
  L6:  speed 4.2393 < 4.97 True | coll 0.022 True | lc 15.1% True | delay minmax 0.3462 False ==> FAIL

== OLD gate at H=60 ==
  L6c: speed 5.0152 < 4.97 False | coll 0.018 True | lc 7.4% True | delay minmax 0.8846 True ==> FAIL
  L6:  speed 5.0152 < 4.97 False | coll 0.018 True | lc 7.4% True | delay minmax 0.5385 True ==> FAIL
```

## Caveat table 1 — LC by type (accel merges vs discretionary), pred vs true per horizon

Per window, validation roots, members pooled (from the same stdout;
per-root rows are in the stdout and `results/gate_v2_val8.json`):

```
   H=30 pooled   predM  5.64 trueM  3.54 ( +59.4%) | predD  8.18 trueD  8.54 (  -4.2%) | total 13.81/12.08 ( +14.4%)
   H=60 pooled   predM  9.15 trueM  7.08 ( +29.2%) | predD 16.40 trueD 16.86 (  -2.8%) | total 25.55/23.94 (  +6.7%)
```

The merge-type over-count evidenced on TRAIN (p3o iter2: +40.4 % /
+23.8 % pooled) is LARGER on validation (+59.4 % / +29.2 %), while the
discretionary type stays close (−4.2 % / −2.8 %). The total stays
inside the ±20 % gate band at both horizons (+14.4 % / +6.7 %), and
the by-type deviations are exactly the residual declared at freeze
(`H30:b_type_pooled`, `H60:b_type_pooled`). Per-root totals are
strongly heterogeneous (−41 % to +89 %): the pooled pass hides
cancelling root-level errors.

## Caveat table 2 — TTC<3 and physics-delay CRPS vs CLIMATOLOGY (TRAIN only)

Climatology built from the 8 TRAIN roots' truth sides ONLY (4 fit
roots from `part3o/results/raw/raw_fit_truth_m0.jsonl` + 4 calibration
roots from `raw_train_internal_iter2_m0.jsonl`; 56 samples per
(output, H); the truth sides are member-independent). "clima(mean)" is
the per-window TRAIN-mean point forecast; "clima(emp)" the TRAIN
empirical distribution.

```
   rung  output       H   cov    med_band  truth_iqr | crps     clima(emp) clima(mean) n_train | verdict
   L6    delay_phys   30  0.2115     62.40     906.51 |    39.59   223.46   258.95      56 | BEATS; informative=False
   L6    delay_phys   60  0.3654    163.20    1784.14 |    89.84   464.29   541.39      56 | BEATS; informative=False
   L6    ttc3         30  0.4231      5.45      17.45 |     3.66     4.27     5.48      56 | BEATS; informative=False
   L6    ttc3         60  0.4423      8.75      27.25 |     6.10     7.44     9.60      56 | BEATS; informative=False
   L6c   delay_phys   30  0.7692    172.46     906.51 |    37.48   223.46   258.95      56 | BEATS; informative=False
   L6c   delay_phys   60  0.8462    411.22    1784.14 |    94.31   464.29   541.39      56 | BEATS; informative=True
   L6c   ttc3         30  0.8077     16.52      17.45 |     4.27     4.27     5.48      56 | does NOT beat; informative=False
   L6c   ttc3         60  0.9038     55.56      27.25 |    11.89     7.44     9.60      56 | does NOT beat; informative=False
```

The widened delay bands beat both climatology baselines at both
horizons (informative at H = 60). The widened TTC<3 bands pass
coverage but do NOT beat climatology (4.27 vs 4.27 at H = 30 — a tie
at printed precision, scored as "does NOT beat"; 11.89 vs 7.44 at
H = 60): passing coverage without beating climatology counts as
UNINFORMATIVE, unchanged from the TRAIN-internal finding at freeze
(owner decision had been pending; it now also holds on validation).

## Caveat table 3 — TTC coverage breakdown (below / in / above; truth = 0 ties)

From `results/gate_v2_val8.json` (`coverage.*`, aggregate_p3m
convention: truth == 0 ties are counted inside below/above):

```
  L6@H30:  ttc3 in 0.4231 (below 0.3462 [3 windows with truth=0], above 0.2308 [0])
  L6c@H30: ttc3 in 0.8077 (below 0.0577 [0], above 0.1346 [0])
  L6@H60:  ttc3 in 0.4423 (below 0.1923 [1 window with truth=0],  above 0.3654 [0])
  L6c@H60: ttc3 in 0.9038 (below 0.0000 [0], above 0.0962 [0])
```

## Provenance note (required)

The evaluated candidate came from **3 TRAIN-internal selection
iterations on the calibration roots (p3o iter0–2)** — fit roots
cv17, cv19_4107, cv19_4108, cv20_7001m2000; calibration/fold roots
cv20_7002m2000, cv20_7003m2000, cv20_7004m2000, cv20_7007m2700 —
after **7 prior validation looks (WD-01-DEV-2 ledger)**. This was
look #8, the single final look for this candidate. The holdout was
never accessed (no holdout root loaded, no registration hash created
or passed); `holdout_block` is authoritative. Data root
`WD01_DATA_ROOT=D:/rd-wt/wd01/output/wd01` used read-only.

## Artifacts

- `eval_val8.py` — the one evaluation pass (hash gate → resumable
  per-member raw rows → L0 in-run).
- `aggregate_val8.py` — aggregation only (re-reads saved raw rows).
- `results/raw/raw_val8_m{0..4}.jsonl`, `results/raw/raw_val8_L0.jsonl`
- `results/stdout/stdout_eval_val8.txt`, `results/stdout/stdout_aggregate_val8.txt`
- `results/gate_v2_val8.json` — machine-readable verdicts + tables
- `LEDGER_DRAFT_VAL8.json`
