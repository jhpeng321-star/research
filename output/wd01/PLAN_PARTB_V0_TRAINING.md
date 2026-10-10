# Part B round 0: V0 training plan (ready to dispatch after the GPU restart)

Plan only. 2026-09-27, world model session. It follows DESIGN_PARTB_V2.md §3–§4 and the step-0 amendment (policy-conditioned V0).

## Data

- **Source:** V0 dataset (output/wd01/partB_v0data, MANIFEST in the worktree wd01n / merged c18cd2b + 5ea9417).
- **Policies:** `source=base` is the damping fleet; `source=native` is native.
- **Flags:** chunk_id, cav_speed_mode, pre_fix and p40 on every file.
- **Splits (by world, from WORLD_PLAN):**
  - TRAIN (49 worlds): fitting.
  - CALIBRATION (7): spread calibration, early stopping and model selection.
  - HOLDOUT (14): not built. Evaluated later only under a registration.
- **Chunk-2 (speedMode 27):** worlds are appended per world when the builder is run on chunk 2. Training v0.1 uses chunk 1 only. v0.2 retrains with chunk 2 included and a `pre_fix` input flag, and must report the pre-/post-fix split.
- **Thin long-τ native coverage:** τ = 570 native labels are ≈ 3 per world. Report per-policy counts by τ. If fewer than about 150 native τ = 570 labels remain after filtering, flag it and use the costed option (native base runs / longer native branches) for the owner.

## Model

- **Encoder, shared across policies:**
  - a lane-bin field (3 lanes × 10 m bins: count, mean speed, speed spread) through 1-D conv per lane;
  - CAV tokens (position, lane, speed, action; wall fields empty in chunk 1);
  - entrance-queue scalars;
  - the future-demand summary (per-minute scheduled arrivals per entrance over the remaining τ);
  - demand regime and penetration.
  A transformer over tokens feeds a pooled embedding. About 1–2 M parameters.
- **Conditioning:** τ is sinusoidal-embedded. policy_id is a learned embedding over {native, damping_fleet}, extensible.
- **Heads, all conditioned on τ and policy:**
  - V_delay: symlog-space twohot distribution (255 bins);
  - V_exp (TTC<3 exposure) and V_brake: symlog twohot;
  - P_coll: Bernoulli;
  - P_closing_pair (for future_has_closing_pair): Bernoulli;
  - future_min_ttc_q05: quantile head. Trained, but not trusted until G-1 is validated.
- **Normalisation:** per demand regime (2000/2400/2600/2700) with running percentiles of the return (DreamerV3-style).
- **Ensemble:** M = 5, bootstrap over TRAIN worlds, different seeds.

## Targets and losses

- **Monte-Carlo anchors:** labels(t, τ) for τ ∈ {30, 60, 120, 300, 570}. Cross-entropy on the twohot target (symlog).
- **TD (finite-horizon, τ-conditioned):**
  `target = D[t, t+30] + V̄(s_{t+30}, τ − 30, same policy)`, using a slow EMA target net (τ_ema 0.995).
  - Use λ-returns where consecutive tuples exist.
  - Weight TD against MC with λ_TD = 0.5 at start, annealed towards 0.8.
- **Auxiliary heads:** MC targets only; TD-bootstrap them too if their MC loss plateaus.
- **Weights:** equalise native and damping rows per τ, so the thin native long-τ rows count.

## Training

- **Device:** CUDA (RTX 3060 12 GB), device-agnostic code (WD_DEVICE); CPU only for smoke tests.
- **Optimiser:** AdamW, lr 3e-4 cosine, batch 512, bf16/fp16 autocast on CUDA. Early stopping on CALIBRATION native-row CRPS at τ = 300.
- **Checkpoints:** state_dict only (weights_only loadable), sha256 pinned in a manifest, copied to D:/research-dense-backups/wd01/partB_v0/ with verified SHA256SUMS on every save (MIG-04 rule).
- **Budget:** about 30–60 min on the 3060 for M = 5.

## Evaluation (CALIBRATION split; the native reference is scored on source=native rows only)

1. **Calibration and sharpness per τ:**
   - CRPS vs climatology (TRAIN per-τ and per-demand mean/empirical);
   - PIT histogram;
   - 10–90% coverage;
   - reported separately for native and damping rows.
2. **TD consistency of the model:** `V(s_t, τ) − [D + V(s_{t+30}, τ − 30)]` residual distribution.
3. **Policy contrast (the planner's quantity, CRN pairs):**
   - At each world's t0, the base (damping) and native branches share the restored state.
   - Compare the predicted ΔV = V(s_t0, 570, damping) − V(s_t0, 570, native) with the realised delay difference over (t0, t0 + 570]: sign agreement, Spearman, MAE vs the "ΔV = 0" baseline.
   - This is single-realisation truth, so the noise caveat from A2 step 2 applies. The replicate campaign, if approved, supplies expected-value truth.
4. **Safety heads (diagnostic):** P_coll calibration (Brier; collisions are rare) and exposure CRPS vs climatology. These remain G-1/G-2 gated.
5. **Speed:** predictions per second on GPU for a planner call: 50 plans × M5 × K16.

## Deliverables

- Code: output/wd01/partB_v0/ (model, train, eval, tests).
- Weights: pinned and backed up.
- REPORT_V0.md and LEDGER_DRAFT_V0.json.
- One registration-free CALIBRATION evaluation. No holdout.

## Addendum 2026-09-28: cloud target is Google Colab (there is no local GPU)

Windows does not detect the RTX 3060, so training is done on Google Colab
(Pay-As-You-Go, T4 16 GB, possibly an L4). There is no CPU training. CPU is
used for smoke tests only.

- **Bundle.**
  - ONE zip: the dataset v0.2 (chunk 1 + chunk 2), the partB_v0 code, and
    MANIFEST.json with the sha256 of every file.
  - ONE notebook, `train_partB_v0_colab.ipynb`, with these cells:
    1. check the GPU (`nvidia-smi`, `torch.cuda`);
    2. upload the zip (`files.upload`) or mount Google Drive;
    3. unzip and verify the MANIFEST hashes, stopping on any mismatch;
    4. `pip install` only what Colab lacks beyond its preinstalled torch,
       with pinned versions, and record the torch/CUDA/driver versions
       actually used;
    5. train, saving a checkpoint to Drive every N minutes, with
       `--resume` so a disconnect loses at most N minutes;
    6. evaluate on the CALIBRATION native rows (the plan's evaluation
       items 1–3 and 5);
    7. zip the outputs (weights, metrics, logs, the env report, and an
       output MANIFEST with sha256) and download them.
  - A README that gives the owner's steps in plain language.
- **Fit and time.** It must fit a T4 (16 GB) with fp16 autocast (T4 has no
  bf16) and finish within about 1–1.5 h for M = 5. Scale the epochs and the
  batch from a timing probe inside the notebook; the probe reports the
  estimated total before training starts.
- **Return path.** When the owner brings the output zip back, verify its
  MANIFEST, copy the weights to E:/research-dense-backups/wd01/partB_v0/
  with SHA256SUMS, and record the hashes in the V0 report.

## Addendum 2 (2026-09-28, evening): Docker replaces Colab (owner change)

The owner no longer needs the Colab notebook. The deliverable is retargeted
to a Docker environment so training can run on other computers. The
Dockerfile spec is pending owner confirmation, so nothing Docker-specific
is built yet.

- **Unchanged:** the training code, B2. It is platform-independent:
  - one CLI (`train` / `eval`) with `--data`, `--out`, `--resume`,
    `--ckpt-every-min`, and a device option (cuda, falling back to cpu);
  - a timing probe;
  - an output MANIFEST with sha256;
  - an env report (torch, CUDA, driver, GPU).
- **Unchanged:** the data path, B1 (`make_v0_bundle.py`): incremental
  chunk3 / chunk1_fix27 builds, verification, and a data archive with its
  MANIFEST.
- **Dropped:** the Colab notebook, `files.upload`, and Drive mounting.
- **Draft Docker target (awaiting confirmation):**
  - a PyTorch CUDA runtime base image;
  - code plus a pinned env baked in;
  - data mounted as a volume, or baked in since it is only ~60 MB;
  - `docker run --gpus all ... train`;
  - outputs go to a mounted folder with a SHA manifest;
  - a CPU fallback for smoke tests.

## Addendum 3 (2026-09-28): Docker spec decided (owner, relayed by main discussion)

- **Target:** a rented Linux amd64 cloud GPU (RunPod, Vast.ai, or
  similar). The provider supplies the NVIDIA driver and the container
  runtime.
- **Base image:** an official `pytorch/pytorch` CUDA 12.x *runtime*
  image.
  - The minimum host driver is stated in OWNER_GUIDE.md.
  - Dependencies are pinned.
  - `/out` is writable by non-root users.
  - No secrets or credentials go into the image.
- **Data:** baked into the image. The data version tag is the V0 MANIFEST
  sha256. A `-v <dir>:/data` mount overrides it.
- **ENTRYPOINT `train` (the default):**
  - train, then evaluate on CAL;
  - write weights, metrics, logs, the env report and the SHA manifest to
    `/out`;
  - save a checkpoint to `/out` every N minutes;
  - re-running the same command resumes from the last checkpoint;
  - fall back to CPU when there is no GPU.
- **Scope:** training only. Data generation stays on the laptop.
- **Deliverables:**
  1. the Dockerfile and a build script;
  2. a local CPU smoke test with Docker Desktop, using a tiny epoch;
  3. `docker save` to
     `E:/research-dense/output/partB_docker/<tag>.tar.gz`, with its sha;
  4. OWNER_GUIDE.md in plain language, covering:
     - two routes: RunPod via a private Docker Hub image, or upload the
       tar, then `docker load` and `docker run`;
     - downloading `/out`;
     - what to send back.
- **Order:** built via ZCode after B2 and B1 are green.
- **Local state on 2026-09-28:** the Docker CLI 29.2.0 is installed, but
  the Docker Desktop engine is not running. The owner must start it before
  the smoke test. Free space: C: 108 GB, E: 261 GB.

### Addendum 3a (2026-09-28): RunPod/Vast behaviour

On RunPod and Vast, the owner works inside the container. There is no docker
CLI there, and the pod stops when the entrypoint exits.

- **Output directory:** set by the env var `OUT_DIR`, default `/out`. On
  RunPod, set `OUT_DIR=/workspace/out` so outputs land on the persistent
  volume. Checkpoints and resume use `OUT_DIR`.
- **After train and eval:**
  - write `OUT_DIR/results.zip`, a single file holding the weights,
    metrics, logs, the env report and the SHA manifest, and record its own
    sha256 in `OUT_DIR/results.zip.sha256`;
  - print `DONE — download <OUT_DIR>/results.zip`;
  - then **keep the container alive** by sleeping until it is stopped, so
    the owner can download through the provider's file browser, Jupyter
    or SSH.
  - `KEEP_ALIVE=0` disables the sleep, for the local smoke test and CI.
- **OWNER_GUIDE options:**
  - Option A: RunPod or Vast, using the image from a private Docker Hub
    repo.
  - Option B: `docker load` then `docker run`, only for full VMs or lab
    servers that have Docker (Lambda, cloud VMs). Not for RunPod or Vast.

### Addendum 3b (2026-09-28): plain bundle alongside the Docker image

The owner is likely to rent on AutoDL (China), possibly also on Colab.
Neither service runs Docker images, so D also produces a PLAIN BUNDLE:
`partB_v0_bundle_<tag>.zip`.

- **Contents:**
  - code, data, and a MANIFEST with the sha256 of every file;
  - `requirements_plain.txt`, which pins everything except torch (torch
    comes from the host's PyTorch image);
  - `README_PLAIN.md`.
- **README_PLAIN.md steps, AutoDL:**
  1. Rent an RTX 4090 or 3090 with the PyTorch 2.x / CUDA 12 image.
  2. Upload the zip to `/root/autodl-tmp` and unzip it.
  3. `pip install -r requirements_plain.txt`.
  4. Run the one train command with `--out` on the data disk. Re-running
     the same command resumes.
  5. Download `results.zip`.
  6. SHUT DOWN the instance.
- **README_PLAIN.md steps, Colab:**
  1. Mount Drive.
  2. Unzip to `/content`.
  3. `pip install`.
  4. Train with `--out` on Drive. Re-running resumes.
- **Build:** both the image and the bundle use the same training entry point
  (`python -m partB_v0 train`), and `make_v0_bundle.py` builds both.
- **Future briefs:** maxSubagents = 0 explicitly.

## Addendum 4 (2026-09-28, late): v0.3 trainer requirements, from the v0.2 AutoDL run

Observed on the RTX 4090 (v0.2, owner's live run):

- The probe measured 34 ms/step. The real rate was about 76 ms/step, and the
  GPU was mostly idle (0.95 of 24 GiB), so the bottleneck is CPU overhead.
- `--time-budget-min` was ONE global deadline across all members. Members 3
  and 4 would have received about 0 steps and been saved as if final.
- The owner restarted with `--max-steps 22700 --time-budget-min 0`.

v0.3 must include:

1. **Per-member budget.** Each member gets total budget / M. The plan is
   recomputed after the first ~500 steps from the MEASURED ms/step, including
   the periodic CAL eval, checkpoint and logging overhead. The cosine LR
   follows the re-planned total.
2. **No partial members.** A member with 0 steps or an unfinished schedule is
   never saved as final. metrics.json marks it incomplete, and results.zip is
   refused unless `--allow-partial` is given.
3. **Speed.** Target: a full M = 5 run in 45 min or less on a 4090.
   - Train the M members concurrently on one GPU: a vectorised ensemble
     (torch.func / stacked weights) or parallel streams/processes. Keep a
     per-member best checkpoint, and keep resume bitwise per member.
   - Make the CAL eval less frequent and cheaper.
   - Speed up the batch pipeline: pre-tensorise the whole dataset on the
     device (it is small), use pinned memory, and avoid per-step host work.
   - Use bf16 autocast where supported (Ampere+). Use fp16 + GradScaler on
     T4.
4. **ea1bae89 fixes:**
   - an OUTPUT_MANIFEST that is consistent with the logs;
   - ckpt/ excluded from the zip;
   - a measured CUDA peak and an automatic OOM batch-halving;
   - rerun-on-completed covered by a test;
   - the numpy warning fixed.
5. **README_PLAIN:** use `nohup … > train.log 2>&1 &` plus `tail -f`, since
   tmux is not installed on the AutoDL image; `apt-get install -y tmux` is
   optional.

### Addendum 4b (2026-09-28): model selection and regularisation, from the v0.2 member-0 curve

Observed: the best CAL native CRPS@300 was 3317 at step 2837, with no
improvement through step 14,755, while the train loss kept falling. mc_d
plateaued near 0.9 nat. Much of that plateau is expected: the twohot target
entropy puts a floor of about 0.4–0.6 nat under the loss, and the delay labels
are single realisations. The CAL selection signal is noisy, because it rests
on 14 worlds whose decision times are autocorrelated.

**Design flaw:** v0.2 uses CAL for both early stopping and the final
evaluation, so its reported CAL numbers are optimistic.

v0.3 must change the following.

1. **Select on out-of-bag worlds, not CAL.**
   - Early stopping and model selection use each member's out-of-bag TRAIN
     worlds (those not in its bootstrap).
   - Mean CRPS is taken over τ ∈ {60, 120, 300} and both policies.
   - Patience is 5 evals. On stopping, restore the best state.
   - CAL is used for the final report only.
2. **Regularisation options:** weight decay, dropout, and a small width
   (d_model 128). Run a quick ablation (base / higher weight decay /
   dropout 0.1 / small) scored on OOB CRPS. Report CAL only for the chosen
   setting.
3. **Log the gap at every eval:**
   - TRAIN-in-bag CRPS;
   - OOB CRPS;
   - CAL CRPS (report only);
   - TRAIN climatology CRPS.
4. **Expected step counts:** with early stopping, a member should need a few
   thousand steps, which supports the ≤ 45 min target.

### Addendum 4c (2026-09-28, ~23:50): v0.2 end-of-run failure, a v0.3 requirement

What happened on the owner's AutoDL run:
- The run was killed partway through member 0, then resumed with
  `--max-steps 12000`.
- On the resume, member 0 took the "already at step >= planned; skipping"
  path. That path never writes `member_0.pt`, because `member_k.pt` is only
  written when that member's `train()` returns in the current process.
- `load_members()` then stops at the first missing index, so it reported "no
  member_*.pt weights" even though members 1–4 had been saved.
- v0.2 also never persists a member's best-state weights to disk, so a member
  that is interrupted and resumed loses its best state.

Recovery on the instance: export `ckpt/m0_state.pt` (the last state) as
`member_0.pt` with a sidecar, then run `python -m partB_v0 eval`. This was
verified locally.

v0.3 must:
1. Persist each member's best-state weights in `ckpt/` (`m{k}_best.pt` plus
   meta) at every new best.
2. Write `member_k.pt` from the persisted best state on EVERY path: natural
   completion, the skip path, and the deadline or budget stop.
3. Make `load_members` require all M members and name any missing ones. It
   must never stop silently at the first gap.
4. Add a test: interrupt member 0, resume with a smaller `--max-steps`, and
   check that all `member_k.pt` files exist and eval runs.

### Addendum 4d (2026-09-29, 00:00): early-stopping data point from the v0.2 run

In the v0.2 run, every member's best CAL native CRPS@300 came at about step
3000 (eval every 1500 steps), with no improvement through step 12000:

| member | best CRPS | at step |
| --- | --- | --- |
| m0 | 3317 | 2837 |
| m1 | 3694 | 3000 |
| m2 | 4302 | 3000 |
| m3 | 4373 | 3000 |
| m4 | 4602 | 3000 |

v0.3 defaults:
- **Early stopping:** evaluate on the OOB worlds every 500 steps, with a
  patience of 3 evals and a maximum of 8000 steps. A member should then
  finish at about 3–5k steps.
- **Report:** show the spread across members, both OOB and CAL, next to
  climatology.

### Addendum 4e (2026-09-29): v0.2 run-1 results → v0.3 diagnostics

Results from `output/partB_results/v0.2_run1` (backed up in
`research-dense-backups/wd01/partB_v0/v0.2_run1`):

- **Skill vs climatology.** The CAL CRPS is 1703 against 1852 for
  climatology, about 8% better. The model LOSES to climatology at
  τ = 30 on both chunks and at τ = 60/570 on chunk 2. It wins by 26–44% at
  τ = 60–300 on chunk 1.
- **Calibration.**
  - 10–90% coverage is about 0.70, below the 0.80 target.
  - The PIT histogram is skewed low, which means the model over-predicts
    delay.
- **TD consistency.** There is a negative gap at τ = 300/570, so the value
  function is not self-consistent at long horizons.
- **Policy contrast.** Sign agreement is 0.57, but the MAE is 7177 against
  3506 for the dV = 0 baseline.

v0.3 must add:

1. **A short-horizon diagnostic**, covering:
   - an audit of the time alignment between the feature snapshot t and the
     label window (t, t + τ];
   - a check that the pre_fix flag and chunk handling are correct;
   - per-τ training curves;
   - a τ = 30 comparison against a simple state-based baseline (for example,
     linear regression on queue length and mean speed).
2. **Report precision.** Print coverage and PIT means with 3 decimals, and
   print the PIT histogram per τ.
3. **TD vs MC at long τ.** Report TD-vs-MC consistency per τ, and consider
   chained TD targets for τ = 300/570.

## Addendum 5 (2026-09-29): v0.3 WORK ORDER (owner, relayed by main discussion)

This supersedes the ordering in addenda 4–4e. Their items are still included.

1. **τ = 30 is worse than climatology. Diagnose first, on CPU, before
   finalising B3.**
   - (a) Label/feature time alignment: recompute D[t, t+τ] from raw physics
     for about 100 random rows per τ and chunk, and check that the feature
     timestamp equals the decision time.
   - (b) τ input: check its encoding and scale.
   - (c) Demand: confirm the demand level is present in the features
     (climatology is demand-conditioned).
   - (d) Baseline: fit a quick GBM or linear model on the same features at
     τ = 30/60/300. This separates a trainer problem from a feature/label
     problem.
   - Fix whatever the diagnosis finds.
2. **Bias high and over-confident.**
   - Normalise the target per τ (divide by a τ-typical delay before
     symlog/twohot).
   - Use the median as the point forecast.
   - Add post-hoc per-τ calibration (shift/scale or isotonic on quantiles),
     fitted on OOB TRAIN and reported on CAL.
3. **Long-τ TD is inconsistent.**
   - Give MC more weight as τ grows.
   - Use n-step targets where the data allow.
   - Add an explicit consistency penalty.
   - Avoid off-grid TD interpolation, or make it exact.
   - Let the shared encoder learn from the damping base rows, but score
     native only on native rows.
4. **Overfitting and noise.** Select on OOB, early-stop at about 3k steps,
   run a weight-decay / dropout / small-model ablation, and train the
   members concurrently.
5. **Policy contrast.** Add a paired Δ head trained on CRN pairs (arm vs
   native, from the same saved state and the same branch seed) that predicts
   Δdelay directly. Evaluate it by sign agreement and by error against the
   zero-difference baseline. Swap in REP-P replicate-mean targets when they
   exist.
6. **Minor.**
   - 3-decimal report.
   - results.zip holds only whitelisted outputs (no stray dirs such as
     ckpt_m0_old).
   - The budget, skip-path and best-persist fixes (addenda 4, 4c).

Item 1's diagnosis is reported to main discussion first.

### Addendum 5a (2026-09-29): item-1 diagnosis result

The problem is in the trainer or model, not in the features or labels. See
`partB_diag/diag_tau30.py` and `diag_tau30.json` (CPU, read-only).

On CAL, MAE of the point forecast (median):

| model | τ = 30 | τ = 300 |
| --- | --- | --- |
| ridge on summary features | 84 | 1361 |
| GBM | 79 | 1739 |
| climatology | 696 | 5634 |
| v0.2 | 744 (bias +610) | 4205 |

The v0.2 bias pattern compresses the forecast toward the middle horizon: it
over-predicts short τ and under-predicts long τ. That points to weak τ
conditioning.

v0.3 fixes, added to item 1:

- normalise the target per τ, and use a twohot grid over the realistic
  range only;
- condition every head on τ with FiLM or concat, plus a per-τ output scale;
- standardise the input features, and add a summary-feature token;
- select on OOB averaged over all τ;
- keep a ridge/GBM baseline permanently in the eval report. v0.3 must beat
  the ridge at every τ, or the report must say so.

### Addendum 6 (2026-09-29): chunk-4 native_base sources (reconstructor, owner-approved)

Chunk 4 (runs root `E:/research-dense/runs/campaign_stage1_chunk4`, 90
worlds) adds one `<world>__native_base` run per world. Each run:

- uses the same world, seed and demand as that world's damping base;
- has ALL CAVs native from t = 0, with speedMode 27;
- covers [0, t0 + 600] and saves no state;
- has the same files as a base run.

In the plan it is arm `native_base` with policy `native`. HOLDOUT stays
sealed.

What the builder must do (chunk-4 follow-up, after B1):
- Ingest `source = native_base` with `policy = native`, on the base-style
  decision grid [120, t0 + 570]. This gives long-τ native labels from t = 120
  and fixes the thin native τ = 570 coverage.
- Align native_base and base at the same (world, t) as CRN-like pairs from
  t = 0. They share demand and seed but NOT the state after t = 0, so they
  are paired by scenario, not by restored state.
- Use them as extra pairs for the Δ head, flagged `pair_kind = scenario`
  (versus `restore` for the t0 branch pairs).

### Addendum 7 (2026-09-29, ~06:05): v0.3 = hybrid value model on CPU; GPU reserved for the new net (agreed with main discussion)

- **v0.3 value model.** v0.3 is the HYBRID model: a ridge point model plus
  quantile-GBM on the ridge residuals, conformally calibrated on OOB/TRAIN
  folds. The fallback is pure quantile-GBM.
  - It is fitted and evaluated locally on CPU on data v0.3. No GPU run is
    needed.
  - CAL is never used for selection or calibration, so the CAL numbers are
    clean.
- **Deliverable.**
  - The v0.3 eval report:
    - per-τ HARD acceptance vs ridge, on CAL and on world-grouped OOB;
    - calibration;
    - policy contrast;
    - planner speed of at least 1000 pred/s.
  - The model files, sha-pinned and backed up.
- **Next GPU run.** It is reserved for B3b's new net once that passes its CPU
  smoke. Its acceptance bar is to beat the HYBRID at every τ, on both OOB and
  CAL.
- **Old net path.** The old architecture (v0.2) is not retrained, because it
  underperforms ridge about 9× at τ = 30.
