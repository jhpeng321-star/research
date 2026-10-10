# Golden reference outputs (captured from wms/r3-int, bf7d4cb)

CPU only, no GPU, no SUMO. Deterministic: 2 torch threads, `torch.use_deterministic_algorithms(True)`,
fixed seeds (model/trainer 101, batch permutation 2026, per-item 7000+n, loss 900+j, train 1100+j).
`python tests/golden/capture_golden.py --out tests/golden/data [--parts selection,batches,forward,losses,train,eval]`
regenerates everything (~8 min). `pytest tests/golden -q` re-captures into a temp dir and compares with
`data/` (`GOLDEN_PARTS=...` restricts parts). Interpreter: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`.
Comparison helpers and the tolerance table: `golden_compare.py` (`load_npz`, `load_json`, `compare_npz`, `compare_json`).

Inputs are read-only external paths listed at the top of `capture_golden.py` (packs, lpwo_v3 lists, sidecars,
wall_same.json). The R3 configuration is built by running the real `wms.cli train` argument path (a trainer subclass
captures dataset/model/config instead of fitting): R3 preset, soft residual bound 4, residual-mag 0.01, lane-rate 1.0
(target 0.0123), history 60 s stride 1, asym 2 / decel 5 @3 / propagation 1 over 3 s, merge-cal, start-shift 0.75,
truncation 20, teacher CAVs, batch 2, device cpu.

## Artifacts (data/)
| file | content | tolerance |
|---|---|---|
| meta.json | git HEAD at capture, seeds, versions, timings | not compared |
| selection.json | dataset receipts and kept window keys `[pack, batch, world, arm, t0_s]` in dataset order for train (911: v3 294, dspeed 444, damping 173) and val (151: v3 47, dspeed 76, damping 28) | exact |
| batches_u{10,60}_hist{on,off}.npz/.json | 3 decoded batches of 2 windows (tensors `b{j}_{name}`: states, present, slot_index, loss_valid, vehicle_ids, cav_teacher, is_cav, hist, hist_mask, driver_hist, driver_hist_mask, controller_id); json holds dataset indices, start-shift ticks, shapes | exact |
| c1/wm_s.pt, c1/TRAIN_RECEIPT.json | read-only copy of the C1 checkpoint (needed inside the workspace for the eval loader) and its receipt | input |
| forward_c1.npz/.json | C1 per-step outputs (accel_mps2, raw_residual_mps2, gate, lane_logits, positions, speeds, lanes, masks, lane_change_probabilities), keys `u10_b{0,1,2}_*` (histoff batches) and `u60_b0_*`; json has checkpoint sha and state checksum | atol 1e-6 |
| forward_r3.npz/.json | same for a fresh seed-101 R3 model on histon batches | atol 1e-6 |
| r3_init_state.npz | fresh R3 initial weights (names with `/` for `.`) | exact (same seed) |
| losses_r3.json | per batch (10 s, fresh model): every loss term, total, gate curve, per-parameter and per-group gradient norms | atol 1e-5 |
| train3_r3.json, train3_final_params.npz | 3 Adam steps (batches 0-2): per-step records, param sha256 checksums, final parameters | atol 1e-5 (sha256 keys compared only with `strict_hashes=True`) |
| eval_rollouts.json, eval_pred_frames.npz | first 3 wall_same states sorted by (world, t0_s, key); 60 s C1 rollout, CAVs teacher-forced, entries off, logic copied from main2_score.py; per-arm car metrics (hdv/cav/all |ds|, |dv|), section delays, predicted frames | atol 1e-5 |

Integers, bools, strings and selection data are always exact.

## How the rewrite must compare
1. Load the same inputs; build the rewritten data path and compare kept keys and counts with `selection.json`; decode
   the batches with the stated seeds/indices and compare with `batches_*.npz` (exact).
2. Load `c1/wm_s.pt` (physics from `c1/TRAIN_RECEIPT.json`) and the batches, run teacher-forced rollouts, compare
   with `forward_c1.npz`; load `r3_init_state.npz` into the rewritten R3 model and compare with `forward_r3.npz`.
3. With that init, same seeds before each pass, compare loss terms/gradient norms (`losses_r3.json`) and the
   parameters after 3 steps (`train3_final_params.npz`). Hash keys need bit-identical op order; use the npz.
4. Run the eval path on the three states and compare with `eval_*`.
Use `compare_npz(arrays, "forward_c1.npz")` / `compare_json(obj, "losses_r3.json")`; empty list means match.

Observations: driver-encoder gradient is exactly 0 at init for the residual-mode R3 model (recorded as is).
Tests use no sealed/CAL/HOLDOUT/chunk-5 data.
