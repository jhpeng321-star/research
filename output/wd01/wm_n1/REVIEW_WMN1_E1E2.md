# First-pass review: WM-N1 e1 (model) + e2 (evalh)

Reviewed 2026-09-30 against `WM_N1_DESIGN.md` sections 2-3/6-7 and the frozen
`REGISTRATION_WM_N1.md`. CPU only, zero SUMO, no training beyond the existing
CPU smoke. Executor under review: e1 = `zcode/wmn1model` @ `43a3e06`;
e2 = `zcode/wmn1eval` @ `8c6ebac` (branch tip also carries `1ce7f7d`).

Integration: `zcode/wmn1eval` merged into `zcode/revwm2`
(merge commit `d3db398`, clean, no conflicts). Review fixes:
`9616803` (model), `1f1c99a` (evalh).

## Verdict

**PARTIAL — one larger defect reported, small defects fixed.** The evalh
harness matches the registration. The model's encoder/losses/validity
primitives match the design, but the autoregressive rollout **never advances
vehicle positions**, so e1's "rollout" is not a usable trajectory generator.

## PASS/FAIL per acceptance item

### Model (e1)

| # | item | verdict | evidence |
| --- | --- | --- | --- |
| M1 | masking-invariance test is real | PASS | `model/tests/test_net.py:32` appends junk padded slots; independent re-check pads 1/4/16 slots with `randn*5` junk and all outputs match (`slot_hidden`, `global_token`, `outcome_quantiles`, `collision_logits`, `step.*`, `arrival.*`) at atol 1e-5 |
| M2 | slot-permutation test is real | PASS | `test_net.py:56`; independent re-check with `randperm(8)` confirms equivariance of slots and step head |
| M3 | validity: non-negative speed | PASS | `project_step` clamps `v_new >= 0` and counts (`rollout.py:79-82`); `test_validity_clips_accel_and_speed` |
| M4 | validity: no same-lane overlap | PASS (primitive) | same-lane gap projection enforces rear-bumper gap >= 0 (`rollout.py:116-132`); `test_validity_gap_projection_keeps_nonnegative_gap`; see defect D1 for the rollout-glue caveat |
| M5 | validity: lane bounds | PASS | `_lane_targets` only allows 0<->1 and 1<->2; `test_validity_lane_bounds`, `test_validity_rejects_lane_change_into_occupied_gap` |
| M6 | contrast loss matches design sec 3 | PASS | sibling-arm comonotone quantile difference `q_a - q_b` vs `y_a - y_b`, sign = treatment−reference (`losses.py:116-136`); `test_contrast_loss_prefers_correct_sign`, `test_contrast_loss_zero_for_identical_predictions_and_outcome`. Sibling identity (same world+t0) is a data-pipeline responsibility; the loss consumes `a`/`b` row indices |
| M7 | loss weights cite the design | PASS | only `lambda_c` is design-specified; `CONTRAST_WEIGHT = 1.0` cites `WM_N1_DESIGN.md` line 115 (`constants.py:67`, `losses.py:9-10`). Other terms default 1.0; the design gives no other weights |
| M8 | parameter count in 2-5 M/member | PASS | measured 3,901,286 (3.90 M) per member; ensemble 19,506,430 / 5 = 3.90 M |
| M9 | only FORMAT.md tensors (no world/penetration/vType) | PASS | `WmN1Net.forward` signature is exactly the six FORMAT tensors; `test_forbidden_inputs.py` rejects `t0`, `seed`, `type_id`; `test_contract.py` pins constants to `features/constants.py`. `is_cav` allowed per design sec 1 |
| M10 | rollout deterministic under a seed | PASS | `test_rollout_deterministic_under_fixed_seed`; separate `torch.Generator` + numpy RNG both seeded |
| M11 | rollout advances the world (autoregressive) | **FAIL** | D1 below |

### Evalh (e2)

| # | item | verdict | evidence |
| --- | --- | --- | --- |
| E1 | paired dV CRPS, world-bootstrap 90% CI | PASS | `build_delta_pairs` / `delta_metrics` (`metrics.py`); `test_build_delta_pairs_comonotone_comonotone_pairing`, `test_delta_metrics_reports_ci_and_threshold`; CI excludes/contains 0 as expected |
| E2 | sign accuracy on `|dY| > TRAIN-median` pairs | PASS | `sign_accuracy` + threshold wired from the source/CLI (`run_eval.py:168-171`). Now also carries a world-bootstrap CI (review fix, D4) |
| E3 | 10-90 coverage and PIT | PASS | `coverage`, `pit_values`, `level_metrics`; `test_coverage_bounds` |
| E4 | decision-rule text byte-identical to registration | PASS | programmatic comparison of `REGISTRATION_WM_N1.md` lines 79-88 with `metrics.DECISION_RULE_QUOTE` → `quote_equal=True` (556 == 556 bytes); verified after merge |
| E5 | truth table tested | PASS | `test_decision.py` (8 tests): each of the four items flips the verdict; item 1 needs >=2 contrasts and >=2 taus; CI must exclude 0 |
| E6 | CAL read-once | PASS | `seals.read_once` / `CAL_MARKER`; `test_cal_is_read_once`, `test_run_eval_cal_consumes_marker` |
| E7 | TEST read-once + `--final-test` + both frozen LF-sha | PASS | chunk-5 `f8c7d760...f304c38214b31e`, WM-N1 `666233aa...9f3c9b56dd93246`; recomputed LF-sha equals both constants; `test_missing_final_test_refused`, `test_wrong_*_sha_refused`, `test_test_is_read_once` |
| E8 | CRLF-safe hashing | PASS | `lf_normalize` maps `\r\n`/`\r`→`\n`; both files are CRLF in this checkout yet hash to the LF constants; `test_crlf_copy_accepted_via_lf_normalisation` |
| E9 | HOLDOUT refused | PASS | canonical `E:/research-dense/tools/holdout_guard.py` present; literal `HOLDOUT` always refused; `test_literal_holdout_refused`, `test_excluded_root_refused` |
| E10 | timing-gate thresholds cite the design | PASS | `BUDGET_HOURS_MAX=50` (design:226), `GATE_LIMIT_HOURS=1.0` (design:230), `ROLLOUT_FRACTION=0.30`/`EVAL_HOURS=2.0` (design:227); citation string asserted in `test_timing_gate.py` |

## Defects

### D1 (larger, NOT fixed) — the rollout never integrates position

`project_step` computes `v_new = v + a*dt` but sets `s_new = s.clone()` and
only ever *reduces* `s` via the gap projection; it never adds `v*dt`
(`model/rollout.py:84`). `_rollout_once` calls it as the only state transition
and does not integrate either (`model/rollout.py:234-265`). Consequence: over a
rollout the positions stay frozen at their t0 values.

Reproduction (real stdout):

```
positions per step, vehicle 0 (slot 0): [-150.0, -150.0, -150.0, -150.0, -150.0, -150.0]
speeds   per step, vehicle 0 (slot 0): [8.0, 8.077, 8.6, 9.502, 9.276, 9.293]
position range over all steps/vehicles: -150.0 0.0
```

Impact: the short-horizon metrics (`rollout_metrics` s_RMSE/v_RMSE, design
sec 6) would compare against constant positions; the validity gap projection is
never actually exercised (vehicles cannot close gaps); and the model README
overstates the rollout. Not fixed here because the correct update scheme
(explicit `s += v*dt + 0.5*a*dt**2`, semi-implicit, or trapezoidal with
`a_used`) is a design decision for the executor. Suggested minimal fix:
integrate positions in `project_step` using the clipped acceleration and
`dt=0.5`, then keep the existing gap projection as the constraint operator.
**Executor action required; e1 should not be treated as rollout-complete.**

### D2 (small, FIXED `9616803`) — inserted arrivals kept zero length in the gap projection

`_rollout_once` captured `length` once at t0 and never refreshed it after
`_insert_arrivals`, so an inserted vehicle's length was 0 in every later
`project_step` call; `min_gap` was then too small and the rear-bumper-gap
guarantee could be violated by up to half a vehicle length once D1 is fixed.
Fixed by refreshing `length = veh[..., C.LENGTH_IDX]` after insertion
(`rollout.py`). Latent today only because of D1.

### D3 (small, FIXED `9616803`) — README parameter count

`model/README.md` claimed "~3.7 M parameters per member"; measured 3.90 M.
Corrected to "~3.9 M".

### D4 (small, FIXED `1f1c99a`) — sign accuracy lacked its required CI

Design sec 6 and registration line 92 require "sign accuracy ... with a
world-bootstrap CI". `delta_metrics` returned only a point estimate. Added
`sign_accuracy_ci` and `variant_sign_ci`/`hybrid_sign_ci`/`sign_n_used` fields
plus two regression tests.

### D5 (small, FIXED `1f1c99a`) — masked s_RMSE produced NaN

`rollout_metrics` sets masked predicted positions to NaN but `rmse_percent`
averaged with `np.mean`, so any masked slot made `s_rmse_pct` NaN while the
parallel `s_rmse_m` used `nanmean`. `rmse_percent` now uses `np.nanmean`.

### D6 (minor, reported, NOT fixed) — sign-accuracy threshold provenance

The design says the contrast sign threshold is the **TRAIN median**.
`SyntheticSource.sign_threshold` (both `sources.py` and its use in
`run_eval.run`) computes the median on the *current* split's rows. This is a
test double, and the production `EvalSource` contract exposes `sign_threshold`
for the real TRAIN median, but the synthetic default is not the registered
quantity. Flagging so the real source supplies the TRAIN median.

## Torch 2.5.1 compatibility

Static grep over `model/` and `evalh/` for APIs newer than torch 2.5 found
**none**: `flex_attention`, `torch.nn.attention`, `torch.compiler`,
`torch.compile`, `scaled_dot_product_attention`, `torch.func`, `torch.vmap`,
`torch.export`, `torch.amp`/`torch.ao`/`torch.distributed`,
`set_default_device`, `weights_only`, `float8`, `uint16/uint32/uint64` — all
absent. APIs used are long-stable (`nn.TransformerEncoder*`,
`softmax/logsumexp/lgamma/logit/nan_to_num`, `torch.Generator`,
`torch.set_num_threads`). Note: this environment runs torch 2.12.1+cpu, so the
2.5 claim is a static review, not a runtime proof; the target GPU box pins to
the recon's env and should be smoke-checked there.

## Executor-summary overclaims

1. `model/README.md:33-45` and commit `43a3e06` describe an "autoregressive
   rollout" / "validity rollout"; the rollout never advances positions (D1).
2. `model/README.md:33` "~3.7 M parameters" vs measured 3.90 M (fixed, D3).
3. `evalh/README.md:28-29` and `metrics.py` present sign accuracy as
   registration-complete; the required world-bootstrap CI was missing (fixed,
   D4).
4. `model/tests/test_rollout.py` advertises coverage of "rollout ...
   safety" but only checks shapes, non-negative speed and determinism; it never
   asserts that the world moves (D1 slipped through).
5. `WM_N1_OWNER_SUMMARY.md:56` "about a day of training time" is optimistic
   versus the design's own 25-50 GPU-h +30 % estimate (up to ~2.7 days on the
   upper bound); pre-timing-gate, so tolerable.

## Tests run (real stdout)

Model (post-fix):

```
$env:WD_FORBID_D="1"
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest output/wd01/wm_n1/model/tests -q
27 passed, 2 warnings in 114.13s (0:01:54)
```

Evalh (post-fix):

```
$env:WD_FORBID_D="1"
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest output/wd01/wm_n1/evalh/tests -q
49 passed in 32.30s
```

Also: `ruff check output/wd01/wm_n1/model output/wd01/wm_n1/evalh` →
`All checks passed!`

## git log -5

```
1f1c99a review fix (evalh): add world-bootstrap CI to sign accuracy (design sec 6 / reg:92); nan-safe masked s_RMSE
9616803 review fix (model): refresh inserted-vehicle lengths before gap projection; correct README param count to ~3.9 M
d3db398 Merge zcode/wmn1eval (WM-N1 evalh e2) into zcode/revwm2 for first-pass review
1ce7f7d wd01 WM-N1 evalh: restore the blank line in the verbatim decision-rule quote
8c6ebac wd01 WM-N1 evalh: make the decision-rule quote byte-verbatim with the registration
```

## Artifacts

- `output/wd01/wm_n1/REVIEW_WMN1_E1E2.md` (this file)
- review fix commits `9616803`, `1f1c99a`
- merge commit `d3db398`
