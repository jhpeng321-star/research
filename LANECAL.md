# LANERATECAL: closed-loop lane-change RATE calibration for WM-S under `physics.lane_learnable`

Branch `c4/v10lanecal` (worktree `E:/agent-work/c4-v10lanecal`, from `c4/v10int` `e80d323b`).
CPU only, no GPU, no SUMO, TRAIN-val starts only (never evaluation seeds 964000-964199, never
`sealed/`, `holdout`, `heldout`, `cal_split`, `chunk5`, `j4b`, `inc1`).

## 1. Problem

With `physics.lane_learnable` (c4/v10unblock) every lane head samples its own probability, so the
WM's **free rollout** changes lanes far more often than SUMO on the same TRAIN-val starts
(`v10unblock/eval_same.json`, `eval_different.json`, 16 starts / 120 s, arm `ii = v5m | ll`):

| event | WM (`ii`) | SUMO | ratio |
|---|---|---|---|
| overtake 1->0 | 282 | 187 | 1.51 |
| return 0->1 | 214 | 106 | 2.02 |
| merge 2->1 | 66 | 64 | 1.03 |
| `ds120` median [m] | 183.9 | - | ~2x the as-is 94.8 |

`ds120` roughly doubles when the false mainline changes are added. WMLANE `LANE_REDESIGN.md`
showed the offsets do **not** transfer across checkpoints, so the calibration is per checkpoint.

## 2. Method

`tools/lane_rate_calib.py` (inference only; no training, no SUMO):

1. select `N` (`--n-native` + `--n-genrand`, default 2x24 = 48) TRAIN-val episodes from each
   source's `val_list.txt` (`split_role == "val"` in `C4_EPISODES_MANIFEST.json`),
2. build the t0=420 s start records (`merge_calib.episode_dict` + `c4_episode_states`), teacher-force
   the CAVs from the recorded arm and **free-roll** the HDVs with `WmsPredictor` under
   `physics.lane_learnable` (`hdv_exits="sumo"`, 120 s, per-start lane seed for common random
   numbers),
3. count the t0-roster HDV transitions **overtake 1->0, return 0->1, merge 2->1** in the WM and in
   the recorded SUMO frames of the same starts,
4. iterate the damped clipped fixed point (default 6 rounds, stop at +-10 %)

   ```
   offset_h += damping * clip(log(target_count_h / wm_count_h), +-max_step)   # 0.7 / 1.0
   offset_h  = clip(offset_h, +-max_offset)                                  # 6.0
   ```

   with the offsets applied as `model.lane_logit_offsets = config_offsets + calib` on the three
   heads (`--heads overtake,return`; `merge` optional),
5. write the **best iterate** to `<checkpoint dir>/lane_calib.json`
   (`format: lane_rate_calib/1`, `offsets` by head, `offset_vector`, target/WM counts, per-round
   log, `checkpoint_sha256`).  `wms.model.checkpoint.load_checkpoint` reads that file when the
   policy stage loads the **same** checkpoint path: the additive offsets are folded into
   `lane_logit_offsets` and the document is kept in `model.lane_rate_calib`.  No file = no-op
   (bit-identical).  A calibration whose `checkpoint_sha256` does not match the file is refused,
   so it can never transfer to another checkpoint.

CLI (many-core Linux box; paths are arguments):

```
python tools/lane_rate_calib.py fit CKPT --native-dir DIR --genrand-dir DIR \
  --n-native 24 --n-genrand 24 --horizon-s 120 --workers 48 --device cpu --lane-learnable \
  --rounds 6 --tol 0.10 --heads overtake,return --out-dir CKPT_DIR
```

## 3. Local smoke (4 starts, CPU, 4 workers, no GPU, no SUMO)

`v5m_ref/wm_s_best_ap.pt` copied into the worktree, `--lane-learnable`, 2 native + 2 genrand
TRAIN-val starts, 120 s, 5 rounds.

| round | offsets (overtake, return) | WM overtake / SUMO | WM return / SUMO | rel err (o/ret) | ds10 | ds30 | ds60 | ds120 | blowups |
|---|---|---|---|---|---|---|---|---|---|
| 0 (before) | 0.000, 0.000 | 58 / 47 | 41 / 21 | 0.234 / 0.952 | 3.97 | 14.74 | 41.58 | 168.99 | 0 |
| 1 | -0.147, -0.468 | 71 / 47 | 30 / 21 | 0.511 / 0.429 | 3.78 | 14.22 | 38.39 | 243.15 | 1 |
| 2 (best) | **-0.436, -0.718** | **44 / 47** | **20 / 21** | **0.064 / 0.048** | 3.79 | 15.00 | 32.33 | 185.95 | 0 |

* Rate matching works: overtake 58 -> 44 (target 47, -6.4 %), return 41 -> 20 (target 21,
  -4.8 %); merge is not fitted (20 vs 19).  The damped fixed point is noisy round-to-round on
  4 starts (round 1 overshoots), hence the damping/clip and the best-iterate choice.
* `ds60` follows the false-positive drop (41.6 -> 32.3 m).  `ds120` did **not** drop on these 4
  starts (169 -> 186 m), i.e. the expected "120 s error drops when false changes drop" is only
  visible at the 60 s window here; 4 starts / 2 sources are too few to claim a general 120 s gain.
  Blow-ups stay 0 in the uncalibrated and best rounds.
* End-to-end load hook: `load_checkpoint(smoke_ckpt/wm_s_best_ap.pt)` returns
  `lane_logit_offsets == (-0.436, -0.718, 0.0)` and `lane_rate_calib["offset_vector"]` equal to
  that vector.

## 4. Exact gpuR command (coordinator)

`python = /root/miniconda3/bin/python`, repo = `/root/autodl-tmp/c4/v5/repo_<sha>`,
data = `/root/autodl-tmp/c4/v5/data/v5_native` + `/root/autodl-tmp/c4/v5/data/genrand_plain`.
Run on the gpuR CPUs (many-core), after the WM fine-tune, before the policy update:

```
cd /root/autodl-tmp/c4/v5/repo_<sha>
PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/bin/python tools/lane_rate_calib.py fit \
  /root/autodl-tmp/c4/v5/<wm_out>/wm_s_best.pt \
  --native-dir /root/autodl-tmp/c4/v5/data/v5_native \
  --genrand-dir /root/autodl-tmp/c4/v5/data/genrand_plain \
  --n-native 24 --n-genrand 24 --horizon-s 120 --workers 48 --device cpu --lane-learnable \
  --rounds 6 --tol 0.10 --heads overtake,return \
  --out-dir /root/autodl-tmp/c4/v5/<wm_out>
```

The policy update then loads the same `wm_s_best.pt`; `load_checkpoint` picks up the adjacent
`lane_calib.json` automatically, so no extra flag is needed.

## 5. Loop-stage integration

`scripts/c4/loop_v10.py` + `configs/c4_dreamer_loop_v10loop.json`:

* new stage **`lane_calib`**, planned after `wm_check` and before `policy_update` (also iter-0,
  both `iter0_mode='existing'` and `'train'`); host `gpuR` (many-core CPU);
* config key `wm.lane_calib` (`enabled`, `n_starts`, `n_native`/`n_genrand`, `workers`, `rounds`,
  `horizon_s`, `tol`, `damping`, `max_step`, `max_offset`, `heads`, `lane_learnable`, `out`);
  validated in `validate_config`;
* the stage writes `lane_calib.json` next to `wm_s_best.pt`; `policy_update` loads the same
  calibrated checkpoint (its note records this).

## 6. Tests

```
python -m pytest tests/unit/test_lane_rate_calib.py tests/unit/test_c4_loop_v10.py
  tests/unit/test_lanev5.py tests/unit/test_mergecal.py tests/unit/test_lanefix.py
  tests/unit/test_lane_learnable.py tests/unit/test_v8lane.py -q
# 106 passed
python -m ruff check tools/lane_rate_calib.py src/wms/model/checkpoint.py src/wms/model/wms.py \
  scripts/c4/loop_v10.py tests/unit/test_lane_rate_calib.py tests/unit/test_c4_loop_v10.py
# All checks passed!
```

New coverage: event counting, the damped/clipped update + convergence, the `lane_calib.json`
parser/hook (no-op, additive, sha mismatch refused, malformed refused) and the loop stage order,
command and policy-loading note.

## 7. Notes / limits

* 4-start local smoke is small; the miscalibration ratio (2.0x) is much clearer at 48 starts, which
  is the default for the gpuR run.
* `merge` is left uncalibrated by default (the ramp merge is handled by the physical envelope /
  the forced mechanism; the smoke shows 20 vs 19).
* The best iterate minimises the **rate** error, not `ds120`; at 4 starts `ds60` improved while
  `ds120` did not.  Report the 48-start `ds10/30/60/120` and blow-ups from the gpuR run before
  treating the calibration as a fidelity win.
