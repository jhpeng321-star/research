# V6ENC: opt-in physical driver parameters

`model.driver_param_mode = "legacy"` and `model.driver_ema_update = true`
preserve the old arithmetic, RNG stream and state-dict keys. Both flat config
keys are omitted at their defaults. Checkpoints persist non-default switches;
physical checkpoints also save the learned centre and range buffers.

## Physical map

For the five longitudinal parameters, let `r = n / (n + 8)` and use
`lo + (hi - lo) * sigmoid(centre_logits + r * raw)`:

- `v0`: 0.6–1.4 times the road speed limit.
- `T`: 0.4–3.0 s; `s0`: 0.5–5.0 m.
- `a`: 0.3–4.0 m/s²; `b`: 0.8–6.0 m/s².

`centre_logits` is a shared five-element `nn.Parameter`, initialized from the
model's legacy physics defaults, clipped just inside the range for finite
logits. No observations means exactly the learned centre. Reliability shrinkage
acts in logit space so any positive observation count retains access to the
full open interval; shrinking already-bounded physical deviations would recreate
observation-dependent floors. Lane parameters retain their legacy mapping.

`driver_ema_update = false` suppresses the algebraic v0/T updates while retaining
the prediction-trained innovation embedding. `configs/c4_wm_v6.toml` inherits v5,
enables the physical map, disables EMA parameter updates and the driver-prior
loss, and uses `runs/active/c4_wm_v6`. Driver inference uses observed motion and
prediction error; the CAV plant table is unchanged.

## Verification (Windows, PyTorch 2.8.0+cpu, at most two threads)

- `tests/unit/test_v6enc.py`: **24 passed**, including textbook reachability and
  Jacobians at 1/10/61 observations, learnable centre, lane identity, EMA-off,
  checkpoint round trips, off-path values/gradients and a two-step synthetic
  forward/backward using the v6 recipe with the prior disabled.
- V6 plus `test_infrav5.py`, `test_lanev5.py`, `test_v5data.py`, `test_execcap.py`:
  **101 passed, 1 deselected**. The deselected optional V5DATA smoke reads a
  dataset outside this worktree.
- WMLANE, LANEFIX, R5 switches, config train/load/golden-preset and model/checkpoint
  equivalence: **152 passed, 5 deselected**. Two deselections require an external
  C1 checkpoint; three are the frozen numeric assertions described below.
- Ruff on all changed Python files and `git diff --check`: **passed**.

Three existing frozen assertions do not match this CPU environment:
`test_wmlane::test_switch_off_is_bit_identical_to_pre_wmlane_code`,
`test_lanefix::test_switch_off_hazard_training_step_is_bit_identical`, and
`test_r5_switches::test_default_unroll_record_is_bit_identical`. They fail with
**identical values on the unmodified baseline `c7aca52`** (e.g. lane-event loss
0.5079182386398315 versus frozen 0.5079182982444763). Frozen expectations were
not changed. Loading the eight changed source modules directly from that Git
commit and comparing against the working tree gives **3/3 bit-exact fixed-batch
comparisons**, including complete records, model weights and available gradients.
Local verification script/artifacts are under `runs/active/v6enc_*` (ignored).

No data-backed training smoke was run; verification used synthetic CPU batches.
Full frozen-golden sign-off still needs the environment used to capture them.
