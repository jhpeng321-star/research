# MERGE_CORE: the learned LC2013-like secure-gap merge core (SECUREGAP)

Branch `c4/v10mergecore`, from `c4/v10int` at `de1c6b43`. Owner-approved 2026-10-21.
Inference/fit only: no SUMO, no GPU, no training run (a CPU 2-step wiring proof and the
recorded-data parameter fit are the only optimisation runs). TRAIN-val only; the tools refuse
evaluation seeds 964000-964199 and restricted split paths; no driver truth table is read.

## 0. Summary

* `model.merge_decision = "secure_gap"` is the learned two-sided secure-gap core behind a
  default-off switch; the default `"hazard"` path creates no module and is bit-identical.
* The fitted core parameters (from recorded TRAIN-val ramp rows) are
  `s0 = 2.161 m`, `T = 0.862 s`, `b = 1.968 m/s^2`, `tau_a = 1.079 s`,
  `temperature = 1.278 m`, `alpha_log_sigma = 0.128`; the two-sided identification diagnostic
  fits `s0 = 1.988`, `T = 0.823`, `b = 1.280`, `tau_a = 0.732`, `temperature = 0.472`,
  `alpha_log_sigma = 0.467`.
* The two-sided identification fit reproduces the diagnosis: held-out **median |timing error|
  0.5 s**, event recall **0.684**, within 2 s **63.7 %** (row P/R/F1 0.119 / 0.628 / 0.201).
* The deployed free-side fit (missing gap side = physically free space) has held-out
  median |error| 2.75 s, event recall 0.905, within 2 s 49.2 %, i.e. it accepts earlier on the
  recorded rows; it is the variant that does not deadlock in rollouts.
* No-training 24-state check (the 72-state tool, entries off, 3 workers) on round 2
  `E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt`: stuck in lane 2 **13 -> 3** (SUMO 1),
  HDV ds60 **35.9 -> 34.0 m**, ds120 **32.9 -> 27.5 m**, blow-ups **2/24 -> 0/24**;
  median absolute departure error **16.8 -> 16.0 s**. The paired ds60/ds120 medians are
  0.00. The hybrid two-sided fallback variant had stuck 11 and ds120 50.1 m, which is why the
  deployed core treats a missing side as free space.

## 1. Mechanism

The core is the two-sided car-following secure gap of `LC2013_MECHANISM.md` section 4.2,
with every behavioural parameter fitted:

```text
v_ant   = v + a tau_a,   v_l_ant = v_l + a_l tau_a,   v_f_ant = v_f + a_f tau_a
gap_l_a = gap_l + (v_l - v) tau_a,   gap_f_a = gap_f + (v - v_f) tau_a
req_l   = s0 + T v_ant   + max(0, v_ant^2   - v_l_ant^2) / (2 b)
req_f   = s0 + T v_f_ant + max(0, v_f_ant^2 - v_ant^2)   / (2 b)
margin  = min(gap_l_a - req_l / alpha, gap_f_a - req_f / alpha) + residual
p_merge = sigmoid(margin / temperature)
```

* `s0`, `T`, `b`, `tau_a`, `temperature` are `softplus(raw)` parameters, initialised from
  `src/wms/model/merge_core_init.json` (the fit output).
* `alpha = exp(loc + sigma * eps)` is the persistent per-driver assertiveness (`lcAssertive`
  analogue). `loc` is a zero-initialised bounded function of the driver encoding and `sigma`
  is a learned positive scale. `eps` is drawn once per car at its first ramp appearance and
  carried in `merge_core_alpha` / `merge_core_has_alpha` / `merge_core_present`; an absent slot
  that becomes present again is a new car and redraws. The training likelihood integrates
  `eps` out with Gauss-Hermite quadrature.
* The residual is a small zero-initialised bounded head on
  `[leader margin, follower margin, speeds, gaps, merge intent]`; it is exactly zero at load.
* A missing lane-1 leader/follower is **physically free space** (no constraint). This is the
  deployed semantics; it avoids a lane-end deadlock. A two-sided diagnostic (both sides must
  be identified, except in the last 5 m) is available in the fit tool and is the variant that
  reproduces the diagnosis's ~0.5 s timing.
* The decision is the smooth `sigmoid(margin / temperature)`: the rollout samples it on
  envelope-feasible in-zone steps (`lane_v5_acts`), so every recorded merge has positive
  likelihood; the hard physical envelope is still a veto and is never relaxed.
* A zero-initialised follower-yield head acts on the lane-1 follower of a ramp car whose
  follower margin is near zero, on `[follower margin, ramp intent, speeds, gap, relative
  speed]`; its output is non-positive (`-softplus(scale) * tanh(raw / 2)`) and is scattered to
  the follower slot. It is supervised by the R5YIELD Huber term on recorded pre-merge follower
  accelerations with the core acceleration as prediction.

## 2. Losses (`loss.merge_core`)

* `weight` (default 0): censored discrete-time survival NLL of the SAME recorded ramp car's
  merge on the free-rollout states, with `alpha` integrated by Gauss-Hermite quadrature. It
  reuses the R5FR compatibility censoring (same car / absolute clock; censor at the last
  comparable state; early merges are censored and counted).
* `tf_weight` (default 0): teacher-forced acceptance likelihood of the same secure-gap margin
  on the RECORDED states (absolute next-step merge label), direct supervision of
  `s0, T, b, tau_a` and the temperature.
* `yield_weight` (default 0): the R5YIELD Huber follower-yield supervision on
  `merge_core_yield_accel`.
* `metrics` builds the diagnostics without a loss; `n_quad` (default 8), the four compatibility
  tolerances and `location_reg_weight` are the R5FR-compatible keys.

The term replaces `loss.merge_first_passage` / `merge_timing` / `merge_survival` for the merge
head when `weight > 0`. All weights 0 = no term, bit-identical.

## 3. Fitted parameters and held-out timing

Data: the 384 recorded TRAIN-val starts (`recs_native.pkl` + `recs_genrand.pkl`, 192 worlds)
under `merge_calib/data`; 143,527 in-zone ramp rows, 1,934 recorded merges. Every fifth world
is held out (38 worlds, 421 ramp cars, 411 merges). The fit maximises event-recall x
within-2 s on the fit worlds (timing target), selecting among deterministic random candidates.

| fit | s0 m | T s | b m/s2 | tau_a s | temp m | alpha sigma | held P | held R | held F1 | event recall | med err s | abs med s | within 2 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed (free side) | 2.161 | 0.862 | 1.968 | 1.079 | 1.278 | 0.128 | 0.101 | 0.776 | 0.179 | 0.905 | -2.75 | 2.75 | 49.2 % |
| two-sided diagnostic | 1.988 | 0.823 | 1.280 | 0.732 | 0.472 | 0.467 | 0.119 | 0.628 | 0.201 | 0.684 | **-0.5** | **0.5** | 63.7 % |

The two-sided diagnostic reproduces the evidence table's secure-gap row (~0.5 s median,
66 % within 2 s); the deployed free-side variant is earlier on recorded rows but is the one
that does not leave ramp cars stuck at the lane end.

Commands:

```text
python tools/merge_core_fit.py --search 800 --seed 0
python tools/merge_core_fit.py --search 800 --seed 0 --require-both \
    --out _out/merge_core/two_sided_init.json \
    --report runs/active/C4-V7FT-20261017/merge_sumo_diag/merge_core_fit_two_sided.json
python tools/merge_core_smoke.py        # CPU 2-step wiring proof (loss 1.867 -> 1.918, finite)
```

## 4. No-training 72-state A/B on round 2

Checkpoint `E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt` (hazard, round 2). The 72-state
tool with entries off and `--workers 3`; the acceptance allows >= 24 states, so the run used
the first 24 every-2 states (`--limit 24`). The core is the same checkpoint with
`model_config={"merge_decision": "secure_gap"}`, no training, fitted init.

| variant | states | ramp HDVs | stuck in lane 2 (SUMO) | median dep. err s | abs med s | early>2 / late>2 | HDV ds60 | HDV ds120 | ramp ds60 (state/pooled) | ramp ds120 (state/pooled) | blow-ups |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round-2 hazard | 24 | 120 | 13 (1) | -1.5 | 16.8 | 51 / 38 | 35.9 | 32.9 | 107.7 / 72.9 | 207.0 / 176.8 | 2/24 |
| SECUREGAP free side | 24 | 120 | **3** (1) | -8.2 | 16.0 | 63 / 38 | 34.0 | **27.5** | 122.9 / 56.2 | 163.7 / 138.8 | **0/24** |
| SECUREGAP hybrid two-sided | 24 | 120 | 11 (1) | +4.2 | 21.0 | 39 / 55 | 28.5 | 50.1 | 41.6 / 25.4 | 156.6 / 145.6 | 1/24 |

Paired ds60/ds120 medians against round 2 are 0.00 for the free-side core (+0.04 m mean ds60,
0.00 ds120). The core cuts the lane-2 stuck count by 10/13 and removes the two blow-ups; the
hybrid variant's stricter mid-zone identification does not deadlock less than the free-side
core, so it is not deployed.

Evidence: `_out/merge_core/check_24_free_fit.pkl` (deployed) and
`_out/merge_core/check_24_hybrid.pkl` (diagnostic). Reproduce:

```text
PYTHONPATH="src;.;tools" python tools/v8wm3_check.py \
  --var "base=E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt" \
  --var "core=E:/Temp/dc_laptop/ck_round2/wm_s_stage240.pt|mc=1" \
  --workers 3 --limit 24 --entries off --out _out/merge_core/check_24_free_fit.pkl
```

## 5. Enable keys

```toml
[model]
lane_model = "hazard_v5"          # required (the core consumes the lane-v5 envelope)
lane_decision = "gated"
merge_decision = "secure_gap"     # default "hazard" is bit-identical

[loss.merge_core]
weight = 1.0                      # censored survival on free-rollout states
tf_weight = 1.0                   # teacher-forced acceptance on recorded states
yield_weight = 0.1                # follower-yield Huber
n_quad = 8
metrics = true
```

Loading an older checkpoint with the switch on is supported: `merge_core.*` is an optional
checkpoint prefix, and the new parameters start at the fitted init
(`src/wms/model/merge_core_init.json`). No checkpoint is modified.

## 6. Tests

`tests/test_merge_core.py`: default switch off / no module; alpha persists across steps and is
redrawn after a slot is reused; hand-built closing-follower rejection and matched-speed small
gap acceptance; absent side is free space; core gradients finite; the survival + teacher-forced
loss is finite and differentiable; round-2 loads with the switch on at the fitted init
(skipped when the checkpoint is absent). `tools/merge_core_smoke.py` runs two Adam steps on a
three-step synthetic rollout.

## 7. Limits / next

* The deployed free-side fit is earlier than the recorded merges; the two-sided diagnostic is
  the ~0.5 s timing match. The next step is to condition the free-side relaxation on the
  lane-end approach (a small learned gate) instead of a fixed structural choice, then rerun
  the fit and the 72-state A/B.
* The 24-state check is the acceptance-allowed subset; a full 72-state run is the next
  confirmation before any training launch.
* Round 2 was scored as-is; the core is not trained in this brief. Any fine-tune should use
  `loss.merge_core.weight` on the free-rollout states plus `tf_weight` on the recorded states,
  with `yield_weight` for the follower response.
