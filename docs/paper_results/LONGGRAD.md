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

### Implemented exact skips and boundary qualification

Implementation/test commit: `3539e238`. Two independent **default-OFF** boolean
trainer keys can be added to `configs/c4_policy14_candA_longgrad.toml`:

```toml
[trainer]
imagination_skip_zero_vjp = true
imagination_replay_dense_prefix = true
```

`configs/c4_policy14_candA_longgrad_skip.toml` is the combined preset: both flags
plus `long_grad_balance=true`, `long_grad_reset_actor_adam=true`, the serial
co-located 0.50 card fraction and `policy14.zero_residual_output=false`. It has
the same inheritance depth as the existing longgrad preset (extending that
preset again would exceed the loader's depth limit). Longskip source is included
in `tools/candA/authority.py`'s runtime byte manifest.

The planner checks **every output cotangent**, sliced to the actual start
microbatch and future. None and exactly zero are inactive; there is no tolerance,
row-share heuristic or threshold. A nonzero long-output cotangent keeps 300 s.
Unknown nonzero output paths, shared disagreement cotangents and nonfinite
cotangents conservatively keep full replay. The switches operate on the existing
`imagination_offload_futures=true` cached-future path; other memory paths retain
full replay. Serial and pooled VJPs share the planner and execution code.

For zero total cotangent, the whole future replay/VJP is skipped. Parameter
connectivity is remembered within this one forward's repeated backwards, so
skipping does not replace unused `None` gradients with zeros (which would affect
Adam). If connectivity is unknown and the current accumulator still has unused
parameters, a conservative full replay establishes it. Thus a standalone first
all-zero backward can still need support-discovery replay; normal dense replay
establishes support before the balanced long pass. An all-OFF long pass then
launches no VJP workers and replays no futures. ON rows alone are insufficient:
only futures selected by the actual signed q10 cotangent run the long pass.

For zero long but nonzero dense cotangent, the active WM runs only the necessary
prefix; full-horizon random/exploration plans, CRN tape draws, row topology and
output reductions stay unchanged. The original detached cached tail is reused.
There is an important **next-state boundary**: with entries, reward transition
t reads `recorder.states[t+1]`, the next callback's **post-insertion** WM input.
Stopping exactly at 90 s would substitute a terminal state before those next
insertions. Replay therefore executes **one extra physics step**, rounded up to
a boundary shared by control steps and the original checkpoint segments:

`physics_steps = ceil((dense_steps*2 + 1) / lcm(2, segment_or_1)) * lcm(2, segment_or_1)`.

With production segment=2 and physics dt=.5 s, the logged replay horizon is
**91 s**, not 90 s. Only the original first 90 s outputs/states have live credit;
the rest is restored from the cached full forward. This also avoids replacing a
nonterminal dense state by WM-specific terminal fields. Larger segments round
up accordingly; if the boundary reaches the full horizon, no truncation occurs.
There is no second no-grad trajectory to eliminate: the original full forward is
already reused for inactive futures and now for the shortened active future's tail.

### Per-update telemetry and update-181 projection

`updates.jsonl` now includes `futures_replayed` (actual start-microbatch/future
VJP executions **across all backward phases**, not unique future IDs),
`replay_horizon_s` (records with `phase`, `start`, `future`, actual horizon, full
horizon and host wall time), and `replay_seconds_saved_estimate`. Zero horizons
identify fully skipped tasks. `grad_long_futures` contains only executed long
VJPs; skipped zero futures have no state-adjoint trace. Existing dense/long norm,
gate, coefficient and output-cotangent telemetry remains available.

The saved-seconds field multiplies saved trajectory seconds by the same update's
observed replay wall time per executed trajectory second. It is a **linear work
estimate**, not a GPU timing measurement or parallel critical-path prediction;
it is null if no replay was timed. Host timing does not force a CUDA synchronize.
Unchanged full-forward/reduction/critic/launch overhead is not claimed as saved.

Using the supplied update-181 **single-pass** profile (861 s forward, 357 s per
full future), conservatively retaining all eight dense futures, no periodic
extra gradient audit, and gate-OFF long cotangents:

* Combined preset / balance ON: **8 x 91 s dense VJPs, zero long VJPs**.
  Estimate `861 + 8*357*(91/300) = 1727 s/update` (**28.8 min**), approximately
  **1981 s saved / 2.15x faster** than the measured 3708 s update. 150 updates
  project to **72.0 h**, rather than 154.5 h. This is still above 24 h.
* Without skips, the earlier balance-ON implementation's two full replay passes
  would project to `861 + 16*357 = 6573 s/update`. Against that matched two-pass
  baseline the estimated saving is **4846 s/update** (3.81x).
* If `m` futures have nonzero long cotangent, balance ON projects to
  **`1727 + 357*m` s/update**. For example `m=2` gives **2441 s / 40.7 min**,
  or **101.7 h** for 150 updates. An ON-row share of .2 does **not** imply .2 of
  futures: m is the union of selected futures across starts/rows and must be
  read from the actual cotangents/telemetry.
* Balance OFF retains its original third combined backward for bitwise legacy
  arithmetic. With both skips and all long cotangents zero, dense and combined
  each need eight 91 s VJPs: about **2594 s/update**, plus any extra audit work.

These are extrapolations from the supplied production profile, not new timing or
memory qualifications. The untouched 861 s no-grad forward is the limiting floor.

### Latest-policy restart (instructions only)

Use the owned STOP-between-updates mechanism and wait for its process to exit.
Choose the **highest complete saved update**, considering both numbered policy
checkpoints and a newer STOP-written `policy_latest.pt`; set `LATEST_POLICY` to
that path, and `NEW_OUTPUT_DIR` to a fresh phase directory. Stage this runtime,
including `longskip.py`, and regenerate the source/config byte manifest with
`configs/c4_policy14_candA_longgrad_skip.toml` as entry. Keep the currently bound
WM, reward/returns, gate/boundary, learned-driver/mix, execution and entry inputs.
The historical C1 actor binding in the launch authority is provenance, **not**
the policy checkpoint to resume.

In the existing launcher's execution/speed/entry environment, use the same serial
co-located recipe (`tools/candA/longmem300_serial_coloc.env`, mode-2 reward,
pair-bias checkpointing, segment=2, K8). The following uses the existing
`stepZ.sh` input paths; retain the current launcher's paths if it was relocated:

```sh
set -a
. tools/candA/longmem300_serial_coloc.env
set +a
L=/root/autodl-tmp/c4/restart20261026/arm/loop
RUNTIME=/root/autodl-tmp/c4/candA_184542ab_zs

/root/miniconda3/bin/python -B -m wms.control.policy14_cli \
  --config configs/c4_policy14_candA_longgrad_skip.toml \
  --wm-ckpt "$L/wm/c02/wm_s.pt" \
  --wm-config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml \
  --real-returns "$L/returns/returns_c03.a1.npz" \
  --longcritic-native-returns "$L/returns/native_returns_c03.a1.npz" \
  --paircrit-receipt "$L/returns/pairing_c03.a1.json" \
  --paircrit-policy-sha256 0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e \
  --gate-panel /root/autodl-tmp/c4/restart20261026/arm/gate_panel.json \
  --horizon-gate "$L/longint/gate_wm2/HORIZON_GATE_WM2_REG.json" \
  --looplong-boundary "$RUNTIME/authority_gen/BOUNDARY_WM2.json" \
  --critic-warmup-updates 0 --native-init --phase-relative-signal-gate \
  --resume "$LATEST_POLICY" --new-phase \
  --resume-allow trainer.long_grad_balance \
  --resume-allow trainer.long_grad_reset_actor_adam \
  --resume-allow trainer.imagination_skip_zero_vjp \
  --resume-allow trainer.imagination_replay_dense_prefix \
  --device cuda --seed 965140 --max-updates 10 \
  --output-dir "$NEW_OUTPUT_DIR" --compact
```

Those **four** allow keys are the complete expected diff from the running
pre-longgrad candidate-A checkpoint: missing historical keys default to false.
If continuing a checkpoint that already has longgrad controls ON, only the two
skip keys differ; omit the two unchanged longgrad allow keys (unused approvals
are rejected). Actor Adam reset is inherited from the recovery preset: it runs
only on explicit new-phase load, not ordinary resume. To retain Adam history,
set `long_grad_reset_actor_adam=false` and omit its allow key when it then matches
the checkpoint. If entropy or another saved trainer/reward setting differs,
carry its exact checkpoint value into the requested config before migration.
Do not reuse the original historical C1 allow-list. Plain resume restores saved
skip flags and would lose this requested change. `zero_residual_output=false`
protects the trained head and requires no trainer allow key. Ten additional
updates meet native-init's multiple-of-ten requirement; extend after observing
the first complete update.

First update-181-like all-OFF receipt: expect eight dense records at 91 s and
eight long records at zero; `futures_replayed=8`, finite identical scalar credit
definitions, zero long norm, and unchanged dense scale/normalization. For ON
updates, full 300 s long records must correspond to actual nonzero long output
cotangents; dense records remain 91 s. Confirm Adam reset receipt, finite policy
checkpoint and measured wall time before replacing the projection.

### S-LONGSKIP CPU verification

126 tests passed across `test_longskip.py` (34), `test_longgrad.py`,
`test_gated300.py`, `test_longmem300_budget.py`, `test_longmem.py`, and
`test_auditfix_resume.py`; two CUDA cases were explicitly deselected. Two existing
resume tests initially rejected pytest's external temp root, then passed with
an allowed root under `E:/Temp/opencode`. Six targeted worker/planner/zero/resume
cases also passed after the fallback worker-budget refinement. Ruff passed.
Preset/source-manifest resolution and the exact four-key pre-longgrad / two-key
already-longgrad new-phase migrations were checked separately after packaging.
Tests prove bitwise parameter gradients, actor parameters and Adam state vs
flags OFF for both balance modes, gate ON/OFF, independent/combined flags and
reward modes 0/2; unchanged scalar credit/gate/q10 and RNG; actual worker
forward/VJP transport; checkpoint resume; and the real learned hybrid WM with
native-residual trunk/group training, sampled futures/lanes, source-wait-gradient
entries and the next-state boundary. Unknown output/nonfinite cotangent and
unused-parameter cases are conservative. Only CPU/synthetic inputs were used.
