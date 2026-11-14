# FIRST-PASS REVIEW — WM-N1 Idea A pieces P2+P4 and P5+P6

Reviewer: execution agent (first pass). Date: 2026-10-01.
Baseline: `zcode/revA1` @ `99ad833` (`zcode/wmn1base2`).
Merged (all `--no-ff`, disjoint ownership, no conflicts):

| merge | source branch | tip | subject |
| --- | --- | --- | --- |
| `ee03b8c` | `zcode/wmn1p24` | `05f195c` | Idea A P2+P4: HDV kinematics residual net + counter-driven arrival head |
| `4457be8` | `zcode/wmn1p56` | `cdb4c9c` | Idea A P5/P6: corr head + CAV-motion OOD and events tokens/masks |

Contracts reviewed: `INTERFACES_V2.md`, `REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md`
(`git show zcode/wd01:...`), `PREFLIGHT_GATES_IDEA_A.md`.

Commands (from `output/wd01/wm_n1`, env `WD_FORBID_D=1`, Python
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, torch 2.12.1+cpu):

```
python -m ruff check hdv arrival corr events
python -m pytest -q hdv arrival corr events
python -m pytest -q model features
python -m pytest -q hdv arrival corr events model features
```

---

## PASS / FAIL by acceptance item

### P2 `hdv/` — HDV residual net

| # | acceptance item | verdict | evidence |
| --- | --- | --- | --- |
| P2.1 | attention inputs are kinematics only; independently perturbing every non-kinematic channel (is_cav/action/strategy) in a full v1+v2 feature dict leaves output identical | **PASS** | `HdvResidualNet.forward(kin, kin_mask, *, cav_flag=None)` accepts nothing else; `input_schema` forbids `action/strategy/is_cav/hdv_base/cav_plan_ch/labels`. `features.idea_a.kinematics_view` selects only indices 0-6. Strengthened `test_full_v1v2_dict_every_nonkinematic_channel_perturbed` perturbs each of v1 columns 7-15 independently plus `hdv_base/cav_plan_ch/event_tok/event_mask/is_cav/action_code/strategy` -> outputs bit-identical. Adversarial manual run (all 9 non-kinematic columns set to random 1000-scale values) confirmed identical. |
| P2.2 | ablation adds exactly one CAV flag | **PASS** | `HdvConfig.in_features == KIN_F+1`; schema moves only `cav_flag` to `optional_inputs`; net concatenates exactly one column (`net.embed[0].in_features == KIN_F+1`). `test_ablation_adds_exactly_one_cav_flag_channel`. |
| P2.3 | zero_residual gives exact zeros | **PASS** | `test_zero_residual_switch_is_exact_noop` now also asserts `accel_log_sigma` is exactly zero. |
| P2.4 | masking / permutation invariance | **PASS** | `test_masking_isolates_absent_slots` (absent slots -> exactly 0, present slots invariant), `test_slot_permutation_equivariance`, neighbourhood/attention-row tests. |
| P2.5 | torch 2.5 compatible | **PASS** (static) | Ops used are 2.5-safe (`nn.Linear/LayerNorm/GELU/ModuleList`, `softmax`, `masked_fill`, `nan_to_num`, `index_select`, `eye`, `isfinite`, `full_like`, `tanh`, `clamp`, `where`). No `torch.compile`, `autocast`, `scaled_dot_product_attention`, `flex_attention`, `torch.export`, or `torch.load(weights_only=)`. No 2.5.1 interpreter is installed on this box, so compatibility is a static scan, not a 2.5.1 run. |

### P4 `arrival/` — arrival head + sampler

| # | acceptance item | verdict | evidence |
| --- | --- | --- | --- |
| P4.1 | counter causality (no count after t used) | **PASS** | Structural: the frozen `features.counters.DepartureLog.counter_at` restricts to `0 <= age` (`<= t`); `IdeaAArrivalHead.forward` accepts only already-past counters and zeroes `valid_bins` that are not observed. `test_counter_causality_only_observed_bins_matter` proves masked bins cannot affect the output. |
| P4.2 | NB/speed params sane | **PASS** | `count_mu = softplus >= 0`; `speed_mu = softplus+1 > 1`; `speed_log_sigma` clamped `[-6, 2]`. Sampler NB uses `n = max(r, 1e-3) > 0`, `p` clipped to `[1e-6, 1-1e-6]`; mean ≈ `|mu|`. `test_arrival_head_shapes_and_ranges`, `test_sample_counts_deterministic_clipped_int64`. |
| P4.3 | sampler deterministic | **PASS** | Both `sample_counts`/`sample_speeds` take a caller `rng` and return equal tensors for equal seeds; `insert` is deterministic. `test_sample_*_deterministic*`, `test_insert_*`. |
| P4.4 | inserted vehicles carry no CAV flag | **PASS** | `insert` sets `is_cav=False`, `wall_group=-1`, `length_m=NOMINAL_LENGTH_M`; asserted in `test_insert_places_generic_hdv_behind_lane_occupancy`. Input world is not mutated. |

### P5 `corr/` — correction head + CAV-motion summary + OOD

| # | acceptance item | verdict | evidence |
| --- | --- | --- | --- |
| P5.1 | correction head has no label input (schema + perturbation) | **PASS after fix** | Signature is exactly `(summary, global_token)` and `cav_motion_summary(positions, speeds, accel, cav_mask)`; both checked against a forbidden-substring list. Coverage gap: the original "controller name" test never passed a label. Review fix adds `test_correction_head_ignores_label_channels_in_full_batch` (strategy/action/outcome channels present in the batch but perturbed -> output bit-identical). |
| P5.2 | quantiles monotone | **PASS** | Constructive: `softplus` increments over a `softplus` base `torch.cat([base, base+cumsum(inc)])`; `test_head_output_shapes_and_monotone_quantiles` pins `q[...,1:] - q[...,:-1] >= -1e-6`. |
| P5.3 | motion summary depends only on trajectories | **PASS** | Review fix adds `test_summary_depends_only_on_cav_trajectory`: perturbing every non-CAV slot's position/speed/accel leaves the summary bit-identical. |
| P5.4 | OOD primitives sound | **PASS** | `motion_density_fit/score` (diagonal Gaussian, standardized MSE, higher = unfamiliar) and `ensemble_disagreement` (member spread) tested for monotone behaviour and zero on identical members. |

### P6 `events/` — tokens + masks

| # | acceptance item | verdict | evidence |
| --- | --- | --- | --- |
| P6.1 | zero-mask no-op | **PASS** | `test_zero_mask_is_noop`: `event_token_mask == 0`, `blocked_lane_mask` all-False, `speed_limit_mask` all-NaN, `apply_speed_limit` returns its input unchanged. `test_empty_events_is_noop` covers the empty list/`EventTokens`. |
| P6.2 | fixtures cover stopped obstacle / blocked lane / speed limit | **PASS** | `events.synthetic.synthetic_event_batch` emits exactly one of each kind (`test_synthetic_fixture_shapes`, `test_masks_from_synthetic_fixture`); `encode_events` covers both flat and nested-by-world inputs, including the `speed_limit_change` magnitude branch. |

## Review fixes applied

1. `ce2c2ce` — P2 test `hdv/tests/test_hdv_leak.py`: added
   `test_full_v1v2_dict_every_nonkinematic_channel_perturbed` (independent
   perturbation of every v1 column 7-15 and the v2 channels/strategy); P2 test
   `hdv/tests/test_hdv_interface.py`: zero-residual now also asserts
   `accel_log_sigma` is zero; P5 test `corr/tests/test_corr_interface.py`:
   added `test_summary_depends_only_on_cav_trajectory` and
   `test_correction_head_ignores_label_channels_in_full_batch`.
2. `a209d94` — P4 `arrival/head.py`: corrected the counter-order wording (the
   frozen featurizer puts the most recent departure in bin 0; see
   `features/counters.py:60`, `features/tests/test_featurizer.py:122`). No
   behavioural change; `past_bin_mask` semantics unchanged.

## Overclaims and reported (not fixed) issues

- **O1 (larger, frozen v1).** `FORMAT.md` §5 line 117 and the `counters.py`
  module docstring state bin `b` covers `[t-300+10b, ...)` with the current
  window in bin 29, but the frozen code (`b = floor(age/10)`) and the frozen
  test (`test_featurizer.py:122`, "age 1.0 -> bin 0") put the most recent
  departure in **bin 0**. `INTERFACES_V2.md` line 17 ("prefix of past bins")
  is consistent with bin 0 = most recent. The contradiction is in the frozen
  v1 prose, which this review may not edit; P4 consumes whatever the featurizer
  produces, so its function is unaffected. Integrator should reconcile the doc.
- **O2.** The pre-fix P2 leak test was titled as a "full feature dict"
  perturbation but only flipped `is_cav` and the action one-hots and never
  passed the dict to the net. Addressed by fix 1.
- **O3.** P5 `test_summary_invariance_to_controller_name` defined
  `controller_a`/`controller_b` but never used them, so it was not a real
  perturbation. Addressed by fix 1 (real label-batch perturbation added); the
  weak test is retained.
- **O4 (low risk, reported).** `IdeaAArrivalHead.forward`'s `valid_bins`
  handling left-pads dims; a `[B]`-shaped mask would broadcast to `[1,1,B]`
  rather than `[B,1,1]`. No caller uses that shape.
- **O5 (cosmetic).** `HdvResidualNet.input_schema().identity_outputs` lists
  only `accel_delta_mps2`, `lane_logit_delta`, omitting the also-zeroed
  `accel_log_sigma`.
- **O6 (low risk).** `events.tokens._flatten_events` decides flat vs nested
  from `events[0]`'s type only; a flat list whose first element is not a
  `Mapping` is silently treated as nested.

No label leak, future-count use, or CAV-flag leakage was found in the P2/P4/P5
paths reviewed. All items pass after the two small "review fix" commits.

---

## Evidence — pytest

`python -m pytest -q hdv arrival corr events` (post-merge, pre-fix):

```
44 passed, 2 warnings in 6.16s
```

`python -m pytest -q model features`:

```
69 passed in 22.03s
```

`python -m pytest -q hdv arrival corr events model features` (final, post-fix):

```
..........
[hdv] parameter_count full=749151 one-layer=351191
...........
[arrival] parameter_count head=250120 small=100552
................................................... [ 62%]
............................................                             [100%]
116 passed in 23.47s
```

`python -m ruff check hdv arrival corr events`: `All checks passed!`

## Evidence — git log -5

```
a209d94 review fix (P4 docs): state the frozen counter-bin ordering (bin 0 most recent) and drop the incorrect bin-29 wording
ce2c2ce review fix (P2/P5 tests): perturb every non-kinematic v1+v2 channel and the label-bearing batch channels; zero-residual checks log-sigma
4457be8 Merge zcode/wmn1p56 (P5 corr head + P6 events) into zcode/revA1
ee03b8c Merge zcode/wmn1p24 (P2 HDV residual + P4 arrival head) into zcode/revA1
cdb4c9c wd01 WM-N1 P5/P6: implement corr head + CAV-motion OOD and events tokens/masks
```
