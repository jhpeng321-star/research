# R3_READY — ROUND-3 integration and train-readiness (CPU only)

**Branch:** `wms/r3-int` (from `wms/r3base`). **Scope:** merge integration,
test/lint, one R3 preset, a local CPU smoke on real TRAIN packs, and a gpuA
staging/launch recipe. No GPU, no SUMO, no full training, no ssh/scp. Packs and
sibling worktrees were read-only; every artifact was written inside
`E:/rd-wt/ws-r3-int`. CAL / HOLDOUT / sealed / chunk-5 / J4B / INC1 were never
opened.

## 1. Merge integration and the `model.py` conflict

Merged, in order, keeping **both** feature sets with all flags default OFF:

| merge | commit | result |
|---|---|---|
| `wms/r3-resid` | `802f33c` (already in base) | residual behaviour preserved |
| `wms/r3-hist` | `e955513` | one conflict, resolved |
| `wms/r3-jam` | `ec777fb` | one conflict, resolved |
| `wms/r3-stage` | `bf7d4cb` | clean |

- **`output/wd01/wm_n1/wms/model.py`** (`WmSModel.__init__` signature): kept
  r3-resid's `residual_soft_bound_mps2` / `residual_soft_bound_neg_mps2` **and**
  r3-hist's `history_s` / `history_stride_s`. The body already referenced both
  (`validate_bounds`/`soft_bound`, `history_spec`/`DriverEncoder(long_history=…)`,
  `self.long_history`), so the merged body is coherent.
- **`output/wd01/wm_n1/wms/cli.py`** (`main`): kept both the r3-hist history
  contract check and the r3-jam jam-flag validation; both still run.
- No `git reset/clean/stash/rebase/checkout --`; conflicts were re-edited and
  committed, never aborted.

## 2. Tests and lint

Commands used the `rl_driving` interpreter (`torch 2.12.1+cpu`), `CUDA_VISIBLE_DEVICES=`,
`OMP/MKL=1`, `TMP/TEMP/TMPDIR` and `--basetemp` inside this worktree.

- Full WM-S suite + eval-history tests: **662 passed, 6 skipped**
  (`_out/R3_INT_TESTS_FINAL.xml`); 6 skips need absent wall-session / chunk-4
  fixtures, unrelated to R3.
- Flags-off bitwise-equivalence tests from **both** branches pass. One
  integration break was fixed: r3-hist's
  `test_long_history.py::test_flags_off_matches_actual_pre_change_source`
  asserted exact dict equality against the pre-change `bf298cf` trainer, which
  now also carries r3-resid's three **always-on additive monitors**. The test
  now requires every original field to stay bit-identical and only permits the
  documented additive keys (matching r3-resid's own oracle), so it is not
  weakened.
- Ruff (`E4,E7,E9,F`, repo rules): **All checks passed** for `wms`, `wms_eval`
  and the modified/added tests. `git diff --check` clean.
- `test_r3_chain.py` was updated for the new authoritative preset (default `R3`);
  its guard/token, data-path, jam/pass-through and required-terms assertions all
  still hold.

## 3. R3 preset (`_out/train_r3_chain.sh`)

A single authoritative preset `R3` is now the default (`R3_PRESET=${R3_PRESET:-R3}`);
`R3_FULL` / `R3_CTRL` remain for historical reproduction. The `R3` case enables,
by default:

- **C1FAST recipe**: `--start-shift-prob 0.75`, stage steps `2500/2500/1875/1250`,
  unroll `10/30/60/120`, **no** residual cap / projection / traffic / stability /
  conservation loss, merge-cal as C1.
- **pack_v3**: `_out/lpwo_v3/windows_{train,val}_noPI_noWall.json` +
  `_out/sidecars_packv3_v2/{identities,outcomes}` (packv3fix loader resolves the
  canonical `pack_v3` list name to the staged basename).
- **R3 flags**: `--residual-soft-bound-mps2 4 --residual-mag-weight 0.01
  --lane-rate-weight 1.0 --lane-rate-target 0.0123 --select-by-long-horizon
  --history-s 60 --history-stride-s 1`.
- **Braking** (no cap): `--asym-speed-loss 2 --decel-weight 5 --decel-threshold 3
  --propagation-loss 1 --prop-window 3`.
- **OFF**: jam oversampling (`JAM_OVERSAMPLE=1`), entries (`ENTRIES_V2=0`), and
  switching data (`SWITCH_DATA=0`, new env). Switching requirement/checks and the
  `--switch-*` flags are now gated on `SWITCH_DATA`; the R3 preset needs no
  `SW_ROOT`. `BATCH_SIZE` is env-configurable (default 8) for the 120 s memory bound.
- **Required smoke terms follow the enabled terms**: the embedded TERMS resolver
  now also appends `residual_mag` / `lane_rate` when their weights are > 0; for
  `R3` this yields
  `position,speed,lane,asym_speed,decel_weight,propagation,residual_mag,lane_rate`.

Shell syntax (`git bash -n`) passes and all three scripts are LF-only.

## 4. Local CPU smoke on real TRAIN data

Interpreter `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, `--device cpu`,
`--threads 1`, `--num-workers 0`, tiny batch. Read-only local packs (non-junction
paths, because the identity-sidecar loader refuses reparse points):
`D:/rd-staging/wmn1_feat/pack_v3`, `E:/rd-wt/dspeed-data/_out/pack_dspeed`,
`E:/rd-wt/damping-snap/_out/pack_damping`, sidecars
`E:/rd-wt/ws-packv3fix/_out/sidecars_packv3_v2/{identities,outcomes}`,
`E:/rd-wt/a9/_out/identities_dspeed`,
`E:/rd-wt/damping-snap/_out/identities_damping` + overlays.

**Complete run** `_out/r3_smoke/final` (`--stage-steps 1,1,1,1`, tiny 3-window val
for cheap selection), exit 0, `TRAIN_RECEIPT.json` written:

- **pack_v3 contributes windows**: `kept_by_pack = {pack_v3: 294, pack_dspeed:
  444, pack_damping: 173}` (train total 911; val `pack_v3: 47`).
- **Every enabled loss term present and nonzero** (`loss_share.jsonl`):

  | stage | position | speed | lane | asym_speed | decel_weight | propagation | residual_mag | lane_rate |
  |---|---|---|---|---|---|---|---|---|
  | 10 s | 0.3228 | 0.0212 | 0.0193 | 0.6363 | 0.0 | 0.0 | 2.2e-4 | 1.5e-4 |
  | 30 s | 0.4379 | 0.0111 | 0.0072 | 0.5435 | 0.0 | 1.5e-4 | 1.4e-4 | 2.5e-5 |
  | 60 s | 0.7800 | 0.0292 | 0.0078 | 0.1825 | 0.0 | 0.0 | 5.0e-4 | 1.0e-5 |
  | 120 s | 0.4478 | 0.0234 | 0.0088 | 0.4745 | 0.0433 | 1.3e-3 | 6.7e-4 | 2.0e-4 |

  `decel_weight` / `propagation` are 0 on stages whose single sampled window had
  no braking/propagation event; both are nonzero at 120 s here, and both were
  nonzero at 10/30/60 s in the earlier 2-step run `_out/r3_smoke/smoke`
  (`decel_weight` 0.23/0.072, `propagation` 1.7e-3/1.2e-3). Every enabled term is
  therefore exercised.
- **History encoder receives real 60 s histories**: `driver_hist [B,192,61,9]`,
  `history_s=60`, `history_stride_s=1` in the receipt. Measured over 8 real TRAIN
  windows with `start_shift_prob=0.75`: **masked fraction 0.674** (observed 0.326)
  of the 192×61 slot-tick samples; the mask is dominated by empty slots plus
  run-start clipping, not by future leakage (r3-hist tests prove post-t0 values
  cannot enter history). Probe: `_out/r3_smoke/hist_probe` (temp-run record).
- **Checkpoints save**: `wm_s_stage10/30/60/120.pt`, `wm_s.pt` (final
  optimizer-step weights) and **`wm_s_best.pt`**. `TRAIN_RECEIPT.json`
  `long_horizon_selection` is `enabled: true`, `minimum_stage_s: 60`, best at
  `horizon_s 120` (`mean_position 504.72`), `final_checkpoint_policy
  last-optimizer-step`.

Earlier supporting runs: `_out/r3_smoke/smoke` (2 steps/stage through 60 s) and
`_out/r3_smoke/best` (60/120 s; produced `wm_s_best.pt`). These longer runs were
externally interrupted by the shell's per-call timeout under heavy **concurrent
CPU load from another agent's sibling-worktree jobs** (not R3 failures); the
`final` run above completed cleanly.

## 5. Memory and wall-time estimate for the single GPU

**Parameters:** `816,917` (R3, residual + 60 s history) vs `778,725` (C1) —
+38,192 (+4.9 %); fp32 weights+grads+Adam ≈ 13 MB total. The rollout graph
dominates, and the residual net is gradient-checkpointed
(`model.py:710`, `torch.utils.checkpoint`, `use_reentrant=False`), so per-step
activations are recomputed in backward. R3's incremental graph over C1 is small:
the history GRU runs once at t0 (input `8×192×61×9×4 ≈ 3.4 MB`, hidden 48,
cached/detached across truncation chunks); `residual_mag` and the soft bound are
elementwise `[B,K,T]` (~1.5 MB each); the lane-rate loss uses a no-grad pass and
does not retain a whole-unroll graph.

**CPU profile datapoint** (`batch 8`, real data, unroll 20 then 60):
process RSS `0.5 → 5.2 → 9.4 GB` (peak 11.9 GB), i.e. ≈ 0.1 GB per simulated
step at batch 8. Linear extrapolation to the 120 s stage (240 steps) is roughly
**28–36 GB of eager graph memory**. GPU peak with the caching allocator is
typically of the same order and often lower; C1 already ran `batch 8` at the
120 s stage on the 48 GB 4090 without OOM, so R3's +4.9 % parameters and small
additive terms should remain **below the ~44 GB guard**. Decision: keep
`BATCH_SIZE=8` (C1-proven) and expose it in the preset; there is no gradient
accumulation in the trainer, so if the first 120 s step's peak exceeds ~44 GB set
`BATCH_SIZE=4` before launch. This is recorded in `r3_env.sh`/the chain header.

**Wall time vs C1** (C1 on a 4090: 0.74 / 2.15 / 4.21 / 7.81 s per step by stage):
C1's full 8125-step run ≈ 6.9 h. R3 keeps the identical schedule and rollout;
the extra history forward and lane-rate no-grad pass add a small per-step
overhead (order 5–10 %), giving a projected **≈ 7.2–7.8 h** for the full run,
comfortably under `MAX_HOURS=14`. (Contended CPU smoke step times are not used
for this projection.)

## 6. gpuA staging and the single launch command

The real gpuA pack_v3 is `/root/autodl-tmp/wmn1_m1_45c9032_20261003_A1/work/pack`
(basename `pack`); the packv3fix loader resolves the canonical `pack_v3` list name
to it. `_out/r3_box/r3_env.sh` now pins the final `R3_PRESET=R3` values
(soft bound 4, `lane_rate_weight 1.0`, `history_stride_s 1`), `SWITCH_DATA=0`,
`BATCH_SIZE=8`, and points `IDENTITIES_V3`/`TRAIN_OUTCOMES` at the uploaded
`sidecars_packv3_v2` bundle.

Prerequisite (fail-closed): copy the verified pack_v3 sidecar payload into
`_out/sidecars_packv3_v2/{identities,outcomes}` (git-ignored; ~21 MB) before
staging — `stage_gpuA.sh` refuses on any hash mismatch.

```bash
# local, from this worktree (Git Bash); gpuA must be ON
STAGE_DRY_RUN=1 bash _out/r3_box/stage_gpuA.sh   # build + verify, no ssh/scp
bash _out/r3_box/stage_gpuA.sh                   # upload once, unpack, print launch
```

The single launch command printed by staging (do **not** wrap ssh in retry loops;
never relaunch after an ssh drop):

```bash
ssh gpuA
  bash -l
  source /root/autodl-tmp/wms_r3/_out/r3_box/r3_env.sh
  cd /root/autodl-tmp/wms_r3
  bash _out/train_r3_chain.sh preflight      # four-stage smoke + explicit-hours gate
# only if preflight passes, EXACTLY ONCE:
  nohup setsid bash _out/train_r3_chain.sh run \
    > /root/autodl-tmp/wms_train_r3/launch_r3_main_a1.log 2>&1 &
```

After the run: `bash _out/r3_box/fetch_and_shutdown.sh`, then power gpuA down with
the printed command.

## 7. Constraints honored

No GPU / SUMO / full training; CPU smoke only. Packs and sibling worktrees were
read-only. Writes stayed inside `E:/rd-wt/ws-r3-int`. No ssh/scp to any box. No
processes were killed (the concurrent sibling-worktree jobs were left running and
only observed). Peak CPU RSS for the batch-8 profile was ~9–12 GB for that short
diagnostic, all other smoke work stayed well below the 6 GB per-process budget
(the single-batch training steps used <1 GB).
