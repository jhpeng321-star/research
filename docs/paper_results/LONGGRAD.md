# S-LONGGRAD — 2026-10-30, CPU diagnosis and default-OFF gradient controls

Final tested implementation/probe SHA: `a1898d8b85bb062c5d2dd4225c8c7d8959a49bcd`.
Controls SHA: `6a87106f`; diagnosis: `4b129cf7`. Final delivery SHA is resolved by
`git log -1 --format=%H -- docs/paper_results/LONGGRAD.md` (avoids a self-hash).
Worktree `E:/agent-work/c4-longgrad`, branch `c4/longgrad`, base `66dcc61c`.
## Findings and strength of evidence
1. Supplied `c4-longmem300/tools/longmem300/evidence/` receipts
   `s8_update191_long300_updates.json` / `s9_update191_dense90_updates.json`:
   pre-clip norm **230,145,616 vs 0.291780204** (7.8876e8x); long/dense scalar
   credit **-6107.969 / -73.832 = 82.73x**; return scales 2418.276 / 2417.776.
   Module norms ON: residual_head 229,677,709; own_encoder 6,784,934;
   report_encoder 3,197,711; attention 9,126,615; group_encoder 8,694,330.
   The head contributes 99.59% of squared norm. OFF head norm = 0.291121.
2. Gate ON share .20 means **one of five eligible rows**, not 20% of updates.
   Gain-only accepts positive improvements; a negative weighted long credit with
   one ON row identifies a **negative-coefficient flip-one row (-10.3)**. Its
   precise index is unavailable. Signed joint q10 gives at most **two futures**
   per start, weights .3/.7; which dominates and which step explodes are unlogged.
   Current candidate A retired flips (`n_flip=0`, `w_team=0`): this stopped-actor
   measurement is not a gradient measurement of the current production row set.
3. No missing reduction found: row coefficients average over starts; tail_credit's
   K multiplier cancels mean(K); replay sums disjoint future VJPs. Long deliberately
   sums t0..300 s; dense averages suffix returns. Dividing by rows/horizon changes
   the registered objective. Baselines, confidence decisions and ranks are detached.
4. No mode-2 `_ImmediateVJP` or pair-bias mis-scaling found. Existing LONGMEM300
   records real CUDA 90 s bitwise equality with both flags ON/OFF, not a 300 s proof.
   CPU tests also cover actual reward/TTC at tiny gaps and unused TTC cotangents.
   follow_speed already has the 184542ab inf/tiny-gap fix. TTC divides by floored
   closing speed, not gap; it is report-only, absent from candidate A's long reward.
5. An independent **pool bug** was found/fixed: worker snapshots omitted gated300,
   WM IDs and horizon curriculum, losing the long output/prefix semantics. CPU
   snapshot + actual worker forward/VJP parity tests pass. This cannot explain s8,
   which was serial; it does not establish CUDA 300 s pool memory qualification.
6. Most plausible remaining cause is large pathwise rollout sensitivity, but the
   supplied logs cannot prove a particular dynamics branch/state or growth rate.
   The reduced hybrid-WM probe (untrained learned encoder, four synthetic slots,
   two HDVs, no entries/damping) gives synthetic summed-reward actor norms at
   10/30/60/90/120 s: **12.945 / 128.594 / 401.297 / 674.002 / 946.707**.
   Modes 0/2 match exactly. Growth becomes roughly linear; production's catastrophic
   amplification is **not reproduced**. State-cotangent clipping is deferred until
   the new traces localize a harmful path; no arbitrary dynamics/row cutoff added.
7. Clip=100 multiplies s8's combined gradient by ~4.35e-7, erasing dense credit.
   Adam sees the clipped norm <=100, not 2.3e8; its squared-moment impulse can still
   be ~1.2e5x dense's, with beta2=.999's ~1000-update decay time constant.

## Recommended controls and telemetry
`long_grad_balance=false` by default. ON computes **g_dense + a*g_long**, where
`a=min(1, ||g_dense||/||g_long||)` (a=1 if long is zero; zero dense suppresses long).
One global actor norm, ratio exactly 1, no tuned constant. Scalar objective, K=8,
q10, gain-only gate, learned drivers/WM and H_G=300 are unchanged. This is optimizer
preconditioning, **not an assertion that the modified gradient is the exact derivative**.
Nonfinite term norms fail before Adam; no silent row dropping. OFF retains the
original combined backward bitwise. Dense includes entropy/safety/anchor/direct
and lane terms where applicable. No long-credit critic/value function is added.

Always in `updates.jsonl`: `grad_norm_dense`, `grad_norm_long`, `grad_cosine`,
`grad_long_scale`, `grad_norm_long_scaled`, `grad_group_norms_{dense,long,combined}`
(optimizer groups and named modules), plus `grad_long_futures` in memory replay.
Each future has pre-scale norm/module norms and `observed_state_cotangent` peak,
top eight physics steps, per-row maxima and finite flag. Hooks on s/v/a are passive
accumulated adjoints at controller inputs, not independent per-step parameter
attributions or a full history/latent-state Jacobian. Gate telemetry adds row kind,
start, coefficient, ON mask and K-by-1-by-row long-output cotangent (selected futures).
No long term: long norm/cosine=0. ON uses two replay passes (dense + long); OFF
diagnostics use three including original combined backward. Expect higher wall time;
no new GPU timing/memory claim. Diagnostics do not obey the older audit-every knob.

## Apply to the running gpuA run (instructions only; no SSH/job performed here)
Use the existing STOP-between-updates mechanism, wait for its owned process to exit,
then take the highest complete `policy_*.pt` by saved update, including a newer
`policy_latest.pt` from STOP. Stage this branch/runtime and regenerate the source
manifest with the new entry config; `longgrad.py`/`policy14.py` are hash-bound.
The opt-in `configs/c4_policy14_candA_longgrad.toml` preserves the serial co-located
0.50 card fraction and sets exactly these continuation changes:
```toml
[trainer]
long_grad_balance = true
long_grad_reset_actor_adam = true
[policy14]
zero_residual_output = false
```
The second key defaults OFF and clears **actor Adam only on explicit new-phase
load**, retaining weights, update count and other optimizers. Ordinary resume
retains newly learned Adam history even when the saved flag is true. Omit this
key/its allow key if intentionally retaining old moments. Disabling zero-start is
essential: the launch preset otherwise zeros the loaded trained head again.

Keep the running launcher's WM/config/returns/boundary/panel/native-init/seed arguments
and execution/speed/entry environment, including `longmem300_serial_coloc.env`,
`WMS_LM300_RECOMPUTE_REWARD=2`, `WMS_LM300_PAIRBIAS_CKPT=1`, segment=2, K8.
Replace its config, resume path, output and the entire historical C1 allow-list by:
```sh
--config configs/c4_policy14_candA_longgrad.toml \
--resume "$LATEST_POLICY" --new-phase \
--resume-allow trainer.long_grad_balance \
--resume-allow trainer.long_grad_reset_actor_adam \
--output-dir "$NEW_OUTPUT_DIR" --max-updates 10
```
Native-init requires a multiple of ten; max-updates is additional phase updates.
Only those two trainer keys should differ from a matching current-run checkpoint.
If entropy annealed (or another saved runtime trainer/reward value differs), carry
its checkpoint value into the requested TOML; inspect the exact resume diff rather
than reusing C1 approvals. No resume-allow key for policy14.zero_residual_output,
the long-credit definition, or unchanged memory geometry. Plain resume alone would
restore saved trainer flags and lose the requested migration. Use a fresh output dir.

First update: verify H_G=300, unchanged credits/ON masks, finite separate/module
norms, `grad_norm_long_scaled <= grad_norm_dense`, scale<1 when long dominates,
and pre-clip actor norm <=2*dense (roundoff). Check cosine for cancellation, selected
rows/futures and early-step adjoint peaks; verify the Adam-reset receipt and finite
checkpoint. Gate-OFF long norm must be zero. A small/zero dense gradient is a useful
signal, not a reason to invent a floor. Rescore with the existing registered panel.

## CPU verification
92 tests passed, 2 CUDA-only skipped across `test_longgrad.py`, `test_gated300.py`,
`test_longmem300_budget.py`, `test_longmem.py`, `test_auditfix_resume.py`; metadata
follow-up: 6 passed. Ruff passed on the new module/tests/probe. Probe command:
`$env:PYTHONPATH='src;.'; .\.venv\Scripts\python.exe tools/longgrad/diagnose.py --horizons 10 30 60 90 120`.

## S-LONGSKIP — code diagnosis (2026-10-30)

Starting point: `508ac026`. The supplied production update-181 measurement is
3708 s/update: 861 s full no-grad K8/300 s forward plus eight approximately
357 s replay/VJPs (rounding accounts for the small difference). Gate share was 0.
This is a supplied measurement, not a new CPU/GPU benchmark.

* `imagination_memory.memory_imagine.Replay.backward` unconditionally loops over
  every cached future and calls `imagine` and `autograd.grad`; it only tests for
  `None` cotangents, never exact zero. `parallel_futures.backward_futures` likewise
  schedules every start-microbatch/future. Cached inactive futures are already
  reused as constants, but the active future is always recomputed in full.
* `longgrad.actor_backward` runs dense and long replay separately when balance is
  ON; OFF diagnostics run both plus the original combined backward. An all-OFF
  gain-only gate still has a differentiable long loss with zero output cotangent,
  so it does not bypass the long pass. Gate-ON rows do not identify all live
  futures: signed joint q10 and all other dense cotangents must also be checked.
* `dreamer.imagine` runs one long trajectory per future, then `gated300.prefix`
  slices rewards/actions/costs/states/etc. to the dense horizon. Crucially it
  **rebuilds dense returns from prefix rewards**, using reverse cumsum, rather
  than slicing the 300 s suffix returns. Values are zero in gated imagination;
  there is no 90 s critic bootstrap or 90..300 s dense actor dependency. Long
  credit is still the sum from t0..300 s, paired against detached native totals.
* Exact prefix replay must keep original-horizon stochastic/exploration plans,
  row geometry, checkpoint segment boundaries and full-K reductions. Cached
  detached tails can restore the full collection shapes before the unchanged
  prefix/long reductions. Shortening `cfg.horizon` globally would change random
  draw shapes and is not a valid implementation.
* The initial no-grad 300 s forward is needed even with gate share zero: the
  gate and q10 decisions require all K full trajectories. It cannot be removed
  prospectively on the basis of a previous update's gate.
