# BRAKEFIX-01 — implementation ready; real TRAIN verification pending

Branch: `wms/brakefix`. Base: `6ebcf95` (hybrid rollout). Date: 2026-10-09.
The cancelled session's partial implementation was reviewed and continued.

**Local result:** CPU verification passes: **290 passed, 2 skipped**, no warnings.
The synthetic `wms.cli train` argv smoke completed one update at each of
10/30/60/120/300 s and passed both gates. Peak OS working set of the final test
process was **865,292,288 bytes**, below the enforced 6,000,000,000-byte limit.

**Real-data status:** not run in this session. The requested pack junction and
identity/overlay sources are outside the permitted workspace. Exact coordinator
commands are below. No recorded-data coverage percentages or braking-improvement
claim is made. The inherited `_out/brakefix_audit.json` and
`_out/brakefix_dry.py` are preserved, but are not independently verified evidence
for this implementation. This note supersedes their readiness implications.

## Design and compatibility

The supplied `_in/DIAG120.md` motivates braking supervision; its diagnosis is
reused, not re-estimated. No self-recovery/pushforward training was added.

* `--asym-speed-loss ALPHA`, default **1/off**: add `(ALPHA-1)` times the
  positive-error position/speed Huber mass. Speed and position signs are tested
  independently, including ahead-but-slower cases.
* `--decel-weight W`, default **1/off**, and `--decel-threshold A`, default **3**:
  multiply the HDV speed/position loss by W only when the recorded destination
  acceleration is strictly below `-A`. It composes multiplicatively with
  asymmetry; the existing step weights and global valid-target denominator stay
  unchanged. CAVs/entrants do not receive these additions.
* `--propagation-loss W`, default **0/off**, and `--prop-window S`, default **3**:
  recorded same-lane nearest-ahead leader search, then follower speed and realized
  acceleration Huber supervision during transitions `t+1 .. t+S/dt` following
  **each** hard-braking step. Sustained braking extends the rolling window; this
  is not an onset-only detector. Recorded targets specify the response timing,
  including the diagnosis's 1–1.5 s lag, rather than imposing instantaneous
  follower braking. No distance cutoff applies unless `--prop-max-gap-m` is set.
  Pairwise leader search is time-chunked; cumulative sums expand the window;
  there are no vehicle loops or device-to-host transfers in those helpers.
  Event-count/loss logging transfers the accumulated new metrics once per unroll.
* Realized acceleration uses consecutive **predicted** speeds and the model's
  timestep, including the zero-speed floor, projection and optional state noise.
  Raw model acceleration and legacy rollout output keys are preserved.
* `--residual-cap-neg-mps2`, default **None/off**: negative residual tanh bound
  independent of the existing positive bound. Equal bounds reproduce the
  symmetric function. No physics clamp, lane cap or projection default changed.
  The checkpoint evaluation adapter restores this new bound only when recorded.
* Enabled terms `asym_speed`, `decel_weight`, `propagation` are logged and included
  in gradient-share diagnostics. Both multi-stage gates automatically derive
  required terms from `BRAKEFIX_CONFIG.json` (written before updates) or the
  adjacent `TRAIN_RECEIPT.json`. Missing/non-numeric terms on any row fail;
  finite zero terms pass. `--train-config` supports an explicit metadata path.
* Defaults omit new config keys and new artifacts/log keys. Tests compare the
  actual pre-fix trainer source at `6ebcf95` against this trainer: **exact loss,
  deterministic record and gradient parity** in residual, driver_params and
  hybrid modes, including traffic loss and loss-share logging. The historical
  hybrid/traffic diagnostic term-reuse behavior is retained on the off path;
  enabled braking losses use the correct traffic tensor.

Recorded acceleration alignment (`state[t+1, channel 5]` versus the preceding
speed change) is an assumption awaiting the real-data audit. The audit emits
streaming correlation and absolute-difference diagnostics; synthetic alignment
is tested, and no unverified real correlation is asserted in source comments.

## Braking-reach audit

These are **theoretical outer lower envelopes**, not fitted-model ability or
state-dependent coverage. Tanh bounds are limiting endpoints; `gate(0)=0` allows
no residual on the first transition. Driver deceleration bounds are default-
centered encoder envelopes; history shrinkage can narrow them.

| Setting | Raw-acceleration lower envelope, m/s² | Share of recorded TRAIN braking reachable |
|---|---:|---|
| residual, uncapped | -9.0 | Pending coordinator audit |
| residual, symmetric cap 1.5 | -9.0 | Pending; global bound can hide state-dependent cap deficits |
| driver_params, default decel 4.2 | -4.2 | Pending; residual cap flags do not affect this mode |
| hybrid, uncapped | -9.0 | Pending coordinator audit |
| hybrid, symmetric cap 1.5, decel 4.2 | -5.7 | Pending coordinator audit |
| hybrid, symmetric cap 1.5, decel 2.7 | -4.2 | Pending coordinator audit |
| hybrid, symmetric cap 1.5, decel 5.7 | -7.2 | Pending coordinator audit |
| hybrid, positive 1.5 / negative 3, decel 4.2 | -7.2 | Pending coordinator audit |
| hybrid, positive 1.5 / negative 3, decel 2.7 | -5.7 | Pending coordinator audit |
| hybrid, positive 1.5 / negative 4.5, decel 4.2 | -8.7 | Pending coordinator audit |

`wms.brakefix_audit` samples TRAIN windows one at a time using read-only `.npy`
memory maps and bounded physics/leader-search chunks. It reports the recorded
deceleration histogram, threshold counts, global-envelope inclusion shares,
**recorded-state raw-acceleration interval inclusion shares**, and required
negative corrections for each fixed-default driver scenario. The interval
diagnostic accounts for both residual sides, the final raw acceleration clamp,
and zero-step gating; it remains an optimistic unprojected teacher-state bound,
not learned-model skill. Boundary equality and correction deficits are reported
separately. No checkpoint/driver inference or recorded-outcome selection occurs.

* **Lane-logit cap:** directly restricts lane-logit corrections, not acceleration.
  It can affect leader selection through lane choice; numerical braking coverage
  after those choices needs a separately interpreted rollout, not an acceleration
  cap percentage.
* **Projection:** constrains post-step speed/lane/position; it does not rewrite
  raw acceleration. It can alter effective speed deceleration. The zero-speed
  floor truncates braking at a stop; raw/effective acceleration are distinguished
  in supervision and audit limitations. Synthetic projection behavior is tested.
* **Physics:** IDM/raw final clip is `[-9,3]`; per-driver physics additionally
  clamps at `-decel_i`. Residual/hybrid output retains the final clip. Recorded
  braking below these limits can be unreachable. Merge calibration and injected
  physics are outside the fixed-default audit and explicitly identified as such.
* `old_hist_chain.sh` uses **driver_params**, not hybrid: broadening the residual
  cap cannot repair that acceleration path. The helper retains its TRAIN keep/
  exclude identities but uses hybrid intentionally to exercise the new cap.

## Verification and preserved receipts

Run from `E:/rd-wt/ws-brakefix`; the interpreter is used directly throughout:

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/brakefix_verify.py --tag final_08 -- -m pytest -q output/wd01/wm_n1/wms/tests --basetemp="E:/rd-wt/ws-brakefix/_out/pytest_final_08"
```

Result: `290 passed, 2 skipped in 192.51s`. The skips are unavailable TRAIN
chunk-4 fixtures `w8676m2700p20t360` and `w8677m2700p20t360`.
For reproduction choose a fresh tag and basetemp, rather than overwriting receipts.

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m ruff check tools/brakefix_verify.py output/wd01/wm_n1/wms/brakefix.py output/wd01/wm_n1/wms/brakefix_audit.py output/wd01/wm_n1/wms/brakefix_smoke.py output/wd01/wm_n1/wms/tests/test_wms_brakefix.py output/wd01/wm_n1/wms/tests/test_wms_brakefix_audit.py output/wd01/wm_n1/wms/cli.py output/wd01/wm_n1/wms/train.py output/wd01/wm_n1/wms/model.py output/wd01/wm_n1/wms/gate_s2.py output/wd01/wm_n1/wms/gate_stages.py output/wd01/wm_n1/wms_eval/wms_plugin.py
git diff --check
```

Both pass. New coverage includes each term's direction/multipliers, combined
weight identity, masking, finite gradients, vectorized/slow-reference equality,
window/chunk boundaries, nondefault timestep, actual cap limits, replay plumbing,
gate fail-closed behavior, bounded mmap TRAIN slicing and streaming audit counts.

Final command log and memory receipt:
`_out/brakefix_verification/final_08/{command.log,verification.json}`.
Synthetic smoke artifacts:
`_out/pytest_final_08/test_cli_train_cpu_all_stages_0/smoke/`
(`TRAIN_RECEIPT.json`, `train_log.jsonl`, `loss_share.jsonl`, synthetic checkpoints).
The smoke substitutes only the synthetic dataset and small model; production CLI
parsing, config, trainer, optimizer, logging and both gates are exercised.

Earlier receipts are retained: initial 32-test pass; `focused_02` was interrupted
by the shell timeout; `parity_03` exposed three Windows GBK decoding failures in
the baseline-source test, fixed by explicit UTF-8; `focused_04` passed 64 tests;
`regression_05` passed 219/2 skipped; `audit_06` passed 24 tests; `final_07` passed
289/2 skipped. `final_08` supersedes them after removing the default propagation
distance cutoff and adding its regression. No failed attempt was silently retried.

## Exact coordinator real-data commands — NOT executed here

The external paths below come from the inherited local dry-run script and the
three-pack identities/LPO contract in `old_hist_chain.sh`. Their availability and
canonical mapping were not inspected here. Use these read-only paths or provide
their identity-equivalent staged TRAIN replacements if unavailable. No VAL list
is supplied; CAL/HOLDOUT/sealed/chunk-5/J4B/INC1 are not evaluation inputs.

**1. Kinematics-only audit: 60 TRAIN windows, 120 s maximum/window.** Output
must be new. The explicit external-read option is for the coordinator; outputs
remain workspace-local. It never opens identity/outcome labels.

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/brakefix_verify.py --tag real_audit_01 -- -m output.wd01.wm_n1.wms.brakefix_audit --readonly-external-packs --pack "E:/rd-wt/ws-cl/_ext/pack_v3" --pack "E:/rd-wt/dspeed-data/_out/pack_dspeed" --pack "E:/rd-wt/damping-snap/_out/pack_damping" --window-list "E:/rd-wt/ws-brakefix/_out/lpo/windows_train_noPI.json" --exclude-window-list "E:/rd-wt/ws-brakefix/_out/lpo/pi_excluded_runs.json" --n-windows 60 --horizon-steps 240 --chunk-steps 60 --threads 1 --out "E:/rd-wt/ws-brakefix/_out/BRAKEFIX_REAL_AUDIT_01.json"
```

Check the acceleration-channel diagnostics and populate the pending shares from
the report's **state-dependent** interval rows, keeping the global-envelope rows
separate. Do not infer actual skill from either bound.

**2. Actual `wms.cli train` argv CPU smoke: at most two eligible windows per
pack, five total optimizer updates, stages 10/30/60/120/300 s.** Selection uses
only keep/exclude metadata, preserving structured window identities; no braking-
rich outcome search or dataset-order assumption is used. The helper records exact
argv, logs/checkpoints, and automatically checks both gates.

```powershell
$env:WM_N1_TRAIN_IDENTITIES = "E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities"
$env:WM_N1_CANONICAL_PACK = "D:/rd-staging/wmn1_feat/pack_v3"
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/brakefix_verify.py --tag real_smoke_01 -- -m output.wd01.wm_n1.wms.brakefix_smoke --pack "E:/rd-wt/ws-cl/_ext/pack_v3" --identities "E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities" --pack "E:/rd-wt/dspeed-data/_out/pack_dspeed" --identities "E:/rd-wt/a9/_out/identities_dspeed" --overlay "E:/rd-wt/dspeed-data/_out/labels_overlay_dspeed" --pack "E:/rd-wt/damping-snap/_out/pack_damping" --identities "E:/rd-wt/damping-snap/_out/identities_damping" --overlay "E:/rd-wt/damping-snap/_out/labels_overlay_damping" --outcomes "D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z" --window-list "E:/rd-wt/ws-brakefix/_out/lpo/windows_train_noPI.json" --exclude-window-list "E:/rd-wt/ws-brakefix/_out/lpo/pi_excluded_runs.json" --out "E:/rd-wt/ws-brakefix/_out/BRAKEFIX_REAL_SMOKE_01"
```

The monitor enforces a 6 GB limit for the single training process and preserves
full stdout/stderr. `num-workers=0`, CPU-only, truncated BPTT and memory-mapped
packs keep allocations bounded. This is a smoke/integrity check, not a training
campaign or efficacy evaluation. Preserve failure receipts and choose new tags
for any corrected follow-up.

## Suggested first experimental weights (implementation defaults remain OFF)

`ALPHA=2`, `decel W=5`, `A=3 m/s²`, `propagation W=1`, `S=3 s`, and hybrid
positive cap `1.5` / negative cap `3 m/s²`. A modest twofold sign penalty targets
under-braking without erasing slow-side error; fivefold rare-event weighting
raises braking influence without blanket reweighting; three seconds covers the
diagnosed follower lag and its immediate response. The asymmetric cap broadens
negative authority while retaining positive authority and the existing final
physics clip. These are reasoned starting values, not tuned/validated weights.
The real audit can identify an unreachable-target issue; it cannot establish
that a larger cap improves a trained model. A larger negative cap or physics-clamp
change requires explicit subsequent interpretation rather than a silent change.

## Recordkeeping

Stable engineering record: `_out/BRAKEFIX_TASK.json` (`BRAKEFIX-01`). Zero SUMO,
zero real-data fitting/evaluation, no new scientific conclusion. Synthetic
optimizer updates are routine verification and recorded in this task, not a
scientific experiment. No master-ledger entry or shared ROUTE/README/INDEX
scientific-status update is claimed; this focused branch's close-out is the READY
note and verification record. Real-data completion remains a coordinator action.
