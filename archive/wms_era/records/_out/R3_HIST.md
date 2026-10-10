# R3-HIST: longer causal per-driver history

**WM-S-R3-HIST-20261009 — implemented and CPU-verified on `wms/r3-hist`.**
Code, synthetic tests and engineering documentation only. No training run,
GPU, SUMO, real-state scoring or acceptance-data access was performed. The
scientific ledger and conclusions are unchanged; verification is recorded here
rather than allocating a scientific experiment for unit tests.

## Flags and backward compatibility

- `--history-s H`: opt-in driver-history duration, normally **30 or 60 s**.
  Omitted (`None`) preserves the existing **10 native ticks**, including t0,
  whose endpoints span 4.5 s. The rolling motion buffer remains `[B,K,10,6]`.
- `--history-stride-s S`: sample the longer driver history every S seconds,
  default **1 s**; must be a positive multiple of the 0.5 s physics step. It is
  inert when `--history-s` is omitted.
- H must be finite and positive. Samples are anchored at t0 and satisfy
  `t0 - H <= t <= t0`: `floor(H / S) + 1` samples. For 30/60 s at 1 s stride
  this is 31/61 samples. Fractional H is rounded down to native physics ticks;
  the earliest sampled tick never precedes the requested interval.
- Supported structured modes: `residual`, `driver_params`, `hybrid`, including
  the existing optional `--lane-head history` in the latter two modes.
  The two black-box variants refuse the new flag rather than silently ignoring it.
- Default checkpoints, parameter shapes, RNG progression, configuration
  serialization, model outputs, losses and gradients stay bit-identical.
  Regression tests load the actual pre-change `bf298cf` model/trainer source
  through Git and compare all three modes exactly. Explicit OFF is also tested.

## Design and causal input contract

`wms/history.py` is the shared TRAIN/evaluation featuriser. Lazy indexed windows
read only native `veh_kin`/`veh_mask` rows through t0, and `target_mask` rows
**strictly before t0**, within the same run. Every native continuity transition
is ANDed backward before downsampling, including transitions between selected
samples. Slot reuse, late entry, early t0 and missing observations are masked;
short windows are retained with zero front-padding, never dropped.

The fixed driver input is `[B,K,L,9]` with a `[B,K,L]` boolean mask:
speed/30, acceleration/3, lane one-hot (3), observed leader gap/100,
displacement from t0/250, leader-relative speed/30, and leader-observed status.
An absent leader has finite free-road gap, zero relative speed and an explicit
missing-leader indicator. Unobserved values are zeroed before arithmetic, so
even NaN padding cannot affect the embedding. Historical leader cues use
tick-local observed neighbours, including those which left before t0.

The opt-in `DriverEncoder` uses a small **48-wide masked GRUCell**. Missing
samples retain the hidden state; masked attention pools its temporal outputs.
The existing bounded parameter head and 102-wide fused driver embedding are
retained. In residual/hybrid mode the embedding conditions the acceleration
and lane residual heads through the existing conditioning adapter. In
driver-parameter modes it also conditions the bounded IDM/MOBIL parameters;
with long history, driver_params additionally permits the conditioned lane
correction. The explicit history lane head receives the same embedding.

The embedding and inferred parameters are fixed through the rollout. They are
cached across steps and detached across existing truncated-BPTT chunks; predicted
future states only update the separate rolling buffer. No future label, hidden
driver parameter or controller outcome enters the history encoder. The tests
mutate every post-t0 source value and check unchanged history, embedding and
parameter inference. GRU order sensitivity and masked-value invariance are tested.

`TrajectoryDataset.from_real_packs` accepts both flags, records the padding
policy in its receipt and passes them through original, switching and monitoring
datasets. Array/NPZ windows may provide `history_kin [N,T,K,7]`,
`history_mask [N,T,K]`, `history_target_mask [N,T-1,K]` explicitly ending at the
window's t0 snapshot. Without prior arrays, only the initial state is observed:
the remaining requested history is masked, never filled with target states.
`collate` fails on inconsistent mixed-history records. Model/trainer flag
configuration must match, and both flags are serialized only when enabled.

## Memory estimate (no GPU measurement)

Let B be batch size, K slots, L sampled ticks and D=48. History work/storage
scales **O(B K L D)**, not O(rollout horizon); slot-neighbour featurisation is
CPU-side, one sampled tick at a time, with O(K²) temporary pair tensors.
Native source arrays remain memmapped; only the bounded history window is read.

For **B=8, K=192, FP32**, long input+mask storage is about **1.68 MiB at 30 s**
or **3.31 MiB at 60 s**, versus about 0.37 MiB for the old 10x6 buffer (which
is still retained separately). A planning allowance of 8–12 FP32 hidden-width
values per GRU sample, plus inputs/masks, is **72–107 MiB at 30 s** or
**141–210 MiB at 60 s**, extra per batch for history activations before allocator
overhead. These are estimates, not measured GPU peaks. Increasing stride to
2 s nearly halves activation storage. First-chunk history graphs are freed by
the existing backward/chunk boundary; later chunks carry detached embeddings.

The GRU adds 8,496 parameters: about 0.13 MiB for FP32 weights+gradients+two Adam
moments. Modes previously lacking residual conditioning also add the existing
102→128→128 adapter (29,696 parameters, about 0.45 MiB with Adam). Hybrid already
has that adapter. The zero-initialized adapter initially blocks upstream residual
history gradients; the unit gradient fixture sets a tiny nonzero adapter without
an optimizer step to prove the complete differentiable path.

CPU verification peak process working set was **896,675,840 bytes (<0.90 GB)**;
final targeted verification used **373,972,992 bytes**. No GPU allocator was used.

## Eval state requirement and materialisation

Today's wallc1 states with approximately 10 ticks cannot supply 30–60 s of past.
Re-materialise from a permitted source into a **new directory** using the same
duration/stride flags. `wms_eval.materialize` and the driver-tools
`wms_eval.driver materialise` now support these flags. No existing state was
opened or rewritten for this task. The legacy raw `prepare_states` path does
not produce driver histories and is unsuitable for a long-history checkpoint.

The minimal state change is to extend `history.kin`, `mask` and `target_mask`
to **61/121 native 0.5 s ticks** for 30/60 s. Keep native transitions even when
the encoder samples at 1 or 2 s, so reuse between sampled ticks stays detectable.
Add `history_s`, `history_stride_s`, `dt_s=0.5`, `available_span_s` and boolean
`run_start_limited`. For early t0, explicitly report run-start clipping; the
loader supplies the missing front-padding. Long histories retain all pack
columns as neighbour context, with anonymous names for empty t0 columns; actual
rollout IDs select their driver histories only after featurisation.

Checkpoint loading restores the flags, validates metadata/native continuity,
rejects undeclared legacy short histories, and uses the same featuriser in
forecasting and per-driver recovery. Flags-off records retain the original keys
and byte-identical serialization. A checkpoint can consume a longer declared
state, but uses only its requested interval and requires matching stride metadata.

At K=192, packed FP32 kinematics plus two one-byte masks are approximately
**0.335 MiB/state at 30 s** or **0.665 MiB/state at 60 s**, versus 0.055 MiB
for 10 ticks. JSON list encoding is larger: budget roughly **2–4 MB/state**
for a 60 s history (precision/occupancy dependent), before compression. Existing
recorded futures are unchanged. Materialisation stays streamed by state.

## CLI examples (documented, not executed)

From workspace root; substitute separately permitted TRAIN sources and fresh
output paths. These examples do not authorize training or scoring:

```powershell
$env:PYTHONPATH = "$PWD/output/wd01/wm_n1"
python -m wms.cli train `
  --pack TRAIN/pack_v3 --identities TRAIN/identities `
  --history-s 60 --history-stride-s 1 `
  --rollout-mode hybrid --lane-head history --out NEW_RUN

python -m wms_eval.materialize `
  --pack pack_v3=TRAIN/pack_v3 --identities pack_v3=TRAIN/identities `
  --from-set TRAIN=TRAIN/wallc1_states.json --history-s 60 --history-stride-s 1 `
  --out-dir _out/NEW_STATES

python -m wms_eval materialise `
  --pack TRAIN/pack_v3 --history-s 60 --history-stride-s 1 `
  --out _out/NEW_DRIVER_STATES.json
```

Scoring uses checkpoint flags automatically; no evaluation-time duration override
can silently change the trained input contract.

## Verification and close-out

All pytest commands set TMP/TEMP to `_out/r3_hist_tmp`, disable CUDA visibility,
use OMP/MKL=1, suppress bytecode writes, and set basetemp inside this worktree.
The direct `rl_driving` interpreter has torch **2.12.1+cpu**. Passing-fixture
retention is disabled to respect the nearly full E: volume.

- **Broad suite:** 573 passed, 4 skipped; `_out/R3_HIST_FULL02.xml`.
  Full `output/wd01/wm_n1/wms/tests` plus materialize/plugin/main-compat/main-switch
  and new eval tests, excluding `test_rollout2_parity.py` because it opens two
  real TRAIN trajectory fixtures outside this synthetic-only task. Four skipped
  tests require absent staged wall reference source, not driver-history code.
- **Final targeted suite after eval context/recovery integration:** 118 passed;
  `_out/R3_HIST_ACCEPT03.xml`. Includes new history tests, materialize,
  main-compat, plugin, main-switch and driver-tool tests.
- **Driver streaming close-out:** 48 passed; `_out/R3_HIST_DRIVER04.xml`.
  Long driver-state exports now stream one record at a time; legacy exports
  retain the same payload/serialization. Checked driver tools, new eval history
  contracts and existing driver-history preparation tests after this change.
- Full WM-S/eval scoped ruff and `git diff --check`: passed.
- Initial failure/install/disk receipts are retained in
  [R3_HIST_FAILURES.md](R3_HIST_FAILURES.md). Incomplete receipts are not counted
  as passing evidence. No prior run or source was overwritten.

Reproducible verification (use a fresh basetemp for any new attempt):

```powershell
$env:TMP = "$PWD/_out/r3_hist_tmp"; $env:TEMP = $env:TMP
$env:CUDA_VISIBLE_DEVICES = ""; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$P = "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe"
& $P -m pytest -q output/wd01/wm_n1/wms/tests `
  tests/test_wms_eval_long_history.py tests/test_wms_eval_materialize.py `
  tests/test_wms_eval_main_compat.py tests/test_wms_eval_wms_plugin.py `
  tests/test_wms_eval_main_switch.py `
  --ignore=output/wd01/wm_n1/wms/tests/test_rollout2_parity.py `
  -o tmp_path_retention_policy=failed --basetemp=_out/pytest_r3_hist_NEW
& $P -m ruff check output/wd01/wm_n1/wms output/wd01/wm_n1/wms_eval `
  tests/test_wms_eval_long_history.py
git diff --check
```

Review status: independent coordinator acceptance checked pre-change bitwise
equivalence, causal slicing/masking, bounded shape restoration and complete
synthetic forward/backward paths. Task verification record updated; no scientific
experiment allocation, efficacy claim or successor training is implied.
