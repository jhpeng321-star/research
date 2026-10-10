# WM-N1-NOISEFLOOR-20261004 — result (COMPLETE, analysis only)

Registration: REGISTRATION.md sha256 `4bf47db59f1214c2c1272077a57902055a3246d0e4dfaba58a233c04ae4f9136`
(frozen, unedited). Executor: opencode / DeepSeek (took over stalled GLM task
085edbf7). Analysis completed 2026-10-05; no simulation was run.

## Status
- [x] Registration read and hash-verified
- [x] 90/90 registered runs present and complete (`RUNS.jsonl`, receipts)
- [x] Determinism: R0 = original true-native arm (bit-exact, all 30 worlds)
- [x] v0.4a t==t0 labels rebuilt for R0/N1/S1 (same `build_v04a_cov` path)
- [x] Existing frozen native/pi labels read for the same worlds
- [x] Noise floor + PI-native spread ratios + world-cluster bootstrap
- [x] Verdict written

## 1. Inventory
All 90 runs (30 worlds x R0/N1/S1) are present: 90 `start`, 90 `done`, all
`exit_code=0`, all `RUN_RECEIPT.json` `exit_status=complete`, all
`physics_steps.jsonl` written 1200/1200 rows (600 s post-t0 at 0.5 s). No
missing, failed or partial runs. Artifact: `analysis/inventory.json`.

## 2. Determinism (R0 vs original native arm)
For every one of the 30 worlds:
- R0 window labels at all taus are **exactly** equal to the original true-native
  labels (max abs diff 0.0);
- the normalized per-tick physics digest (sorted `(id, lane, pos@6dp,
  speed@6dp)` over all 1200 rows) of R0 equals the original native run;
- `arm_verify` (native_fleet) passes.

S1 (SUMO seed+1) is **bit-identical to R0** (raw sha256 equal for all 30
worlds): the saved-state branch restores the SUMO RNG, so a seed change is not a
physical perturbation. **The seed-change noise floor is exactly zero.**
Artifact: `analysis/determinism.json`.

## 3. Labels
v0.4a window labels (`total_delay_ref_s` = delay_s, `ttc_lt3_veh_s`,
`hard_brake_s`) computed at the `t == t0` row, taus 30/60/120/300/570 s, using
the exact `build_v04a_cov` path (`v0lib.build_run_trace` for the branch with
`t_ins_pre` from the world's `__base` run + `v0lib.window_label`,
`skip_first_tick=False`). Existing native/pi t0 labels read from the frozen
v0.4a root. Artifact: `analysis/labels.csv`.

## 4. Noise-floor table — delay_s (primary)
dY values; sd = sample sd (ddof=1) over the 30 TRAIN worlds. `sd ratio PI/N1`
is bootstrap point estimate with 90% world-cluster CI (5000 resamples, seed 0).

| tau | N1-native mean / sd / median\|dY\| / frac>0 | S1-native | PI-native mean / sd / median\|dY\| / frac>0 | sd PI/N1 [90% CI] | clears? |
|----:|-------------------------------------------|-----------|---------------------------------------------|-------------------|---------|
| 30  | 0.424 / 1.021 / 0.235 / 0.800 | 0 (exact) | 12.786 / 27.730 / 6.197 / 0.700 | 27.17 [16.32, 44.71] | yes |
| 60  | -1.185 / 9.533 / 0.238 / 0.800 | 0 (exact) | 32.073 / 89.787 / 34.082 / 0.600 | 9.42 [5.13, 104.21] | yes |
| 120 | 9.586 / 41.219 / 0.238 / 0.700 | 0 (exact) | 101.423 / 263.745 / 156.751 / 0.700 | 6.40 [4.01, 158.31] | yes |
| 300 | 9.146 / 43.905 / 0.238 / 0.700 | 0 (exact) | 307.512 / 914.019 / 530.595 / 0.567 | 20.82 [12.24, 508.38] | yes |
| 570 | 50.820 / 656.351 / 0.238 / 0.667 | 0 (exact) | 212.703 / 3252.354 / 1998.522 / 0.467 | 4.96 [3.00, 1724.41] | yes |

R0-native and S1-native are identically 0 at every tau (bit-exact determinism).

### Secondary — hard_brake_s (point sd ratios PI/N1)
| tau | N1-native sd | PI-native sd | sd PI/N1 |
|----:|-------------:|-------------:|---------:|
| 30  | 0.42 | 1.90 | 4.5 |
| 60  | 0.49 | 5.48 | 11.3 |
| 120 | 4.96 | 15.10 | 3.0 |
| 300 | 3.34 | 38.02 | 11.4 |
| 570 | 2.91 | 86.01 | 29.5 |

## 5. Verdict
At **every** registered tau (30, 60, 120, 300 and 570 s) the PI-native spread
clearly exceeds the null-perturbation noise floor: the 90% world-cluster CI of
`sd(PI-native)/sd(N1-native)` lies entirely above 1 (lower bounds 16.3, 5.1,
4.0, 12.2, 3.0). The seed-change floor (`PI/S1`) is undefined because
`sd(S1-native)=0` exactly — a seed change is not a perturbation. The secondary
`hard_brake_s` contrast agrees (point ratios 3.0–29.5).

Therefore the PI-native delay difference is a systematic action effect, not
generic amplified divergence. Two honest caveats: (a) the null perturbation is
itself heavy-tailed at long horizons (N1 sd reaches 656 s at tau 570, from
individual worlds diverging under a one-CAV nudge), so the "noise floor" is
itself wide at 570 s even though PI is still clearly wider; (b) frac>0 for
PI-native falls to 0.467 at tau 570 — the policy effect is not uniformly
positive, it is a materially larger spread.

## Failures / deviations
- S1 (SUMO seed+1) is bit-identical to R0 on all 30 worlds: the saved-state RNG
  restore makes a seed change physically neutral. Recorded as a finding; the
  seed-based noise floor is exactly zero, so `sd(PI/S1)` is undefined.
- No run failed; no retries; no simulation performed during analysis.

## Artifacts
- `analysis/noisefloor_analysis.py` — analysis driver (no simulation)
- `analysis/inventory.json`, `analysis/determinism.json`,
  `analysis/labels.csv`, `analysis/analysis.json`
