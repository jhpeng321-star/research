# WD-01 part 3o REPORT — LC over-count fix (gate v2 (b)) of the native Part A emulator

**Status: completed (TRAIN-internal only).** The lane-change over-count
(gate v2 (b)) of the `idm_merge` emulator (p3n refit) is FIXED at both
horizons on the TRAIN-internal 2-fold split: **(b) +6.5 % @H30 and
−1.7 % @H60** (both inside the ±10 % ideal band), with gate v2 **fully
PASSING at BOTH horizons for the first time** (including criterion (c)),
(a) collisions ≤ 1/1000 and (d) speed-per-horizon kept PASS, and the
delay-proxy CRPS kept informative (beats climatology ~4.5x at both H;
better than p3n iter1 at H30).

**No validation look was taken; the 4 validation roots were never
loaded or scored; holdout was not accessed (blocked in
`p3lib.root_arrays`); ZERO simulation (no SUMO/TraCI anywhere — the
only dynamics is the pure-torch part-3 rollout engine).**

Date: 2026-09-27. Workspace `D:\rd-wt\wd01o`, branch `zcode/wd01-p3o`.

---

## 1. Scope and protocol

Inherited from p3n, unchanged:

| role | roots |
|---|---|
| fit (refit) | cv17, cv19_4107, cv19_4108, cv20_7001m2000 (× t0 120..480 step 60 = 28 windows) |
| TRAIN-internal gate | fold A = cv20_7002m2000 + cv20_7003m2000, fold B = cv20_7004m2000 + cv20_7007m2700 (× 7 t0 = 28 windows, M=5, H ∈ {30, 60}) |
| validation | **NOT used anywhere in part 3o** (never loaded, never scored) |
| holdout | blocked (physical block authoritative; never passed) |

2-fold gate protocol: when scoring fold X, the spread factors per
(output, horizon) are fitted on fold Y's rows only.

## 2. Engine change (minimal flags, defaults are exact no-ops)

`output/wd01/part3/engine.py`, `Rollout.__init__` gains two kwargs:

- `lc_scale_disc=None` — scales ONLY the idm_merge discretionary 0↔1
  hazard (M4); `None` falls back to `lc_scale` (shipped behaviour).
  `idm_sampled` and all other kinds are untouched.
- `merge_accept_prob=1.0` — a gap-accepted M2 merge candidate merges
  with this probability; draws come from a **dedicated** RNG stream
  (seed `(member_seed, t0, 9107)`) that is created only on the
  `mprob < 1.0` path, so at the default `1.0` no stream is touched and
  the merge decision is byte-identical to the shipped deterministic
  acceptance. Prob-thinned candidates are counted in a new
  `merge_stats['rej_prob']` key that exists only when the flag is
  active (the canonical key set at defaults is unchanged — part 3l's
  key-spelling test still passes), and they fall into no
  rejection-cause bucket.

Defaults are no-ops by construction and by test (same-config re-rolls
are deterministic; the part3/part3l `__new__`-stub tests read the new
attributes via `getattr` defaults). Regression against the committed
p3n iter1 gate: pooled LC totals differ by 0.07 (H30) / 0.16 (H60)
lc/window (0.5–0.7 %) — cross-platform float jitter vs a p3n session
whose raw rolls were not kept; far below any effect a non-no-op default
could produce (10–40 %). `results/stdout_base.txt` documents the check.

## 3. Diagnosis (typed LC, the basis the p3n iteration lacked)

iter0 = the p3n iter1 members re-rolled with the typed row builder
(`raw_train_internal_iter0_m*.jsonl`; engine flags at no-op defaults).
LC per window by type, calibration roots, members pooled:

| H | pred merge | true merge | dev | pred disc | true disc | dev | total dev |
|---|---|---|---|---|---|---|---|
| 30 | 4.68 | 3.25 | **+44.0 %** | 8.91 | 7.96 | +11.9 % | +21.2 % |
| 60 | 8.18 | 6.54 | **+25.1 %** | 17.80 | 16.64 | +7.0 % | +12.1 % |

The over-count is dominated by the **mandatory merge type
(accel→main_right)**; discretionary (0↔1) is much closer. Per fold,
fold A's total is on target (−2.8 %) while fold B blows out (+52.7 %):
fold B's truth carries systematically fewer LCs, which **global** hazard
levers cannot reconcile per-fold — the per-fold tables are therefore
reported as diagnostics and the package gates on the pooled by-type
band (same pooling as the formal gate-v2 (b)).

## 4. Iterations (all fits on the 28 TRAIN fit windows only)

Loss: p3m stage-1 + p3n stage-2 (LC by type, H30 primary + H60 half
weight, TTC<3 exposure, physics delay, mr/occupancy guards) — the p3n
loss, unchanged. Every iteration's stdout and JSON are committed
(`results/stdout_fit_iter*.txt`, `fit_lines_iter*.json`,
`fit_p3o_iter*.json`, `stdout_gate_iter*.txt`,
`gate_train_internal_iter*.json`).

### Iteration 1 — separate the two hazards (new flags)

Lines over base = p3n member 0: `merge_accept_prob` (1.0, 0.85, 0.7,
0.55), `lc_scale_disc` (0.002, 0.0016, 0.0012), `tau_accept` (1.9, 2.4).
Winners: mprob 0.55, disc 0.0016, tau 2.4 → fit joint loss
2.774 → 2.266. Members 1–4: p3n member bases unchanged + softmax draws
over the two new fields. Gate (train-internal):

| H | (a) | (b) total | (c) | (d) |
|---|---|---|---|---|
| 30 | 0.012 PASS | **11.914/11.214 (+6.2 %) PASS** | FAIL (delay-phys cov 0.786) | 3.602 ≤ 3.781 PASS |
| 60 | 0.009 PASS | **22.771/23.179 (−1.8 %) PASS** | PASS | 4.266 ≤ 4.353 PASS |

CRPS 33.74 ≤ 34.95 (@H30, better than p3n iter1) but 88.27 > 82.26
(@H60, worse). Pooled by type: merge +39.8 %/−, disc −8.2 %/−12.5 %.

### Iteration 2 — count-effective merge levers

The merge over-count survived the thinning lever: **merges are
arrival-limited with per-tick retries, so `merge_accept_prob` delays a
merge by ~1/p ticks but does not prevent it within the horizon** (fit
lines: occupancy improves 0.485→0.333, lcM30 loss flat 0.118→0.110).
Most queue merges accept inside the relaxed end-force zone
(`MERGE_FORCE_TAU=0.5`, `MERGE_FORCE_B=6.0`; `end_force_s=60` ⇒ zone
[394, 454] m), where tau/b_accept are overridden. Iteration 2 lines
(base = iter1 member 0): `end_force_s` (60, 45, 30, 15), `tau_accept`
(2.4, 3.0, 3.6), `b_accept` (2.0, 1.5, 1.0), `merge_accept_prob`
(0.55, 0.7, 0.85), `lc_scale_disc` (0.0016, 0.0018, 0.002).

**Negative result (lever evidence):** shrinking the zone 60→15 m cuts
lcM30 only 0.105→0.087 while the joint loss explodes 2.27→4.35
(delay 0.300→0.382, occupancy guard included); tau 3.0/3.6 and b 1.5
are count-flat at worse loss. The merge-type over-count is **structural
for the declared levers** — cutting it costs delay informativeness,
exactly the trade the brief forbids. Winner: b_accept 1.0 (marginal),
everything else at the iter1 optimum; joint 2.2639. Members 1–4 drew
`end_force_s=60` + mprob 0.7–0.85.

Gate (train-internal, 2-fold):

| H | (a) | (b) total | (c) | (d) | v2 |
|---|---|---|---|---|---|
| 30 | 0.013 PASS | **11.943/11.214 (+6.5 %) PASS** | ttc3 0.821, dphys 0.857, crps 33.61 PASS | 3.609 ≤ 3.781 PASS | **PASS** |
| 60 | 0.009 PASS | **22.793/23.179 (−1.7 %) PASS** | ttc3 0.893, dphys 0.821, crps 85.93 PASS | 4.279 ≤ 4.353 PASS | **PASS** |

Pooled by type @H30: merge +40.4 %, disc −8.0 %; @H60: merge +23.8 %,
disc −12.2 %. CRPS vs p3n iter1 reference: 33.61 ≤ 34.95 @H30 ✓,
85.93 > 82.26 @H60 (residual; better than iter1's 88.27; climatology
CRPS is 189.3/386.6 — the ensembles beat it ~5.6x/4.5x, and the p3n
iter1 reference itself only reached 82.26 with delay coverage 0.714/0.786,
below the (c) target that iter2 passes at 0.857/0.821).

## 5. Frozen candidate (for a possible later single VAL look — NOT run)

`results/candidate_p3o_iter2.json` — sha256
`458890d9c4b0ee6ee03de145be50498d9bd0873505a4e86f2bf8a1e224ebe45d`,
frozen_utc `2026-09-27T05:09:19.400476+00:00`.

Member 0 (main fit): `lc_scale=0.002, lc_scale_disc=0.0016,
merge_accept_prob=0.55, merge_params={tau_accept 2.4, b_accept 1.0,
p_coop_scale 0.0, end_force_s 60.0, v_merge_blend 0.0}, dawdle 1.0,
seed 1`. Members 1–4: p3n bases (seeds 18/35/52/69) with the drawn
p3o fields (see the frozen JSON); M=5 ensemble.

`results/calibration_p3o.json` — sha256
`c7ffa7540768fa67e75e99881eec6f801201990dea4ace90af93c202f789c22a`,
frozen_utc `2026-09-27T05:09:19.400476+00:00`. Spread factors (mean-
centred widening), fitted on the FULL calibration split (part 3m/3n
production protocol; the 2-fold fitting was the gate's anti-leakage
device): `delay_phys` s_out 2.7638 @H30 / 2.5198 @H60, `ttc3` 3.0314 /
6.3496; widened coverage 0.821 / 0.857 / 0.821 / 0.821.

Known residuals recorded in the frozen candidate
(`--accept-residuals`, formal (b)/(a)/(d) all PASS):
`H30:b_type_pooled, H60:b_type_pooled` (merge-type band, evidenced
structural above), `H60:crps_not_worse` (+3.7 points vs p3n iter1,
which itself fails (c) coverage at both H).

## 6. TTC and climatology diagnostics (owner decision pending; not optimised)

ttc3 coverage (2-fold, widened): 0.821 @H30, 0.893 @H60. ttc3 ensemble
CRPS does NOT beat the TRAIN climatology (4.05 vs 2.63 @H30; 10.55 vs
5.14 @H60) — same finding as p3n, unchanged by part 3o (never
optimised for, per brief). Physics delay CRPS beats climatology at both
horizons (33.6 vs 189.3; 85.9 vs 386.6).

## 7. Verification

- `pytest` — part 3o: 9 passed (flag no-op neutrality + determinism,
  by-type table algebra, goal-package semantics incl. fold reporting,
  resumability, typed-row consistency, holdout/val exclusion);
  part3/part3l/part3m/part3n suites: all pass (101 total incl. the
  p3n holdout-block and merge_stats key-spelling guards; 1 skip =
  p3n's TASK-A raw-row pin, skipped because the p3n raw rows were
  never committed).
- Evidence files: `results/stdout_base.txt`, `stdout_fit_iter1.txt`,
  `stdout_fit_iter2.txt`, `stdout_gate_iter0/1/2.txt`,
  `gate_train_internal_iter0/1/2.json`, `fit_p3o_iter1/2.json`,
  `fit_lines_iter1/2.json`, frozen sidecars `*.FROZEN.sha256`.

## 8. Resumability

Every expensive step writes per-unit files and skips completed ones on
restart: base rolls per member (`results/raw/raw_train_internal_iter0_m*.jsonl`,
`raw_base_L0.jsonl`, `raw_fit_truth_m0.jsonl`), fit line entries
appended per value (`fit_lines_iter*.json`), gate rolls per member
(`raw_train_internal_iter{k}_m{mi}.jsonl`; `--fresh` forces re-rolls).
The whole pipeline was restarted mid-way twice during this session and
resumed exactly (once from a bug fixed after line entries were cached,
once for the base summary re-print).

## 9. Not done / owner items

- Validation look (deliberately not part of part 3o; the frozen
  candidate + spread factors are ready for ONE look if the owner
  accepts the recorded residuals).
- Holdout evaluation (blocked until the owner opens it).
- Merge-type pooled band (+40 %/+24 %): structural within the declared
  levers (evidence in §4); closing it needs a mechanism-level change
  (e.g. a retry-cooldown or zone-entry lever), which was out of scope.
- Delay-phys CRPS @H60 vs the p3n iter1 number: +3.7 points; note the
  p3n reference bought its sharpness with sub-target coverage that
  iter2 does not.
