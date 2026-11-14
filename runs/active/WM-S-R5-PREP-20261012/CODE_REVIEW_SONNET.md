# Read-only code review of wms/r5-prep (Claude Sonnet 5.5 subagent, 2026-10-12 ~00:30)

Nothing was edited or run in the worktree. (b) was verified with a small pipe experiment; (a) is from code reading, magnitude not measured.

## (b) Exit hang = smoke-wrapper pipe deadlock, not a wms bug (verified)
- `E:/agent-work/tmp/run_smoke.py:51` starts the child with stdout=PIPE, stderr=STDOUT and only poll()s until communicate() at :73.
- `src/wms/cli/train.py:97` prints the full result as indented JSON at the end of training (receipt + history + val_by_stage, >= 40 KB) -> the child blocks writing to the full pipe (0% CPU) after TRAIN_RECEIPT.json is written. Repro: child printing 9/40/200 KB never exits; 3 KB exits. Matches the 252-byte console (stderr lines only).
- Fix: wrapper writes stdout to a file (no PIPE) or uses a reader thread; recommended also: cli/train.py writes the summary to out/RUN_SUMMARY.json and prints one line.
- Verify: same smoke with stage_steps [1,1,1,1] -> exits seconds after TRAIN_RECEIPT.json.

## (a) ~20 GB RSS at batch 2 (R4 was ~2-3 GB)
1. HIGH confidence mechanism: in hybrid mode `_ShareAccumulator.add` (`src/wms/model/wms.py:54-61`) builds non-detached physics/residual magnitudes -> `roll["residual_magnitude"]`, `roll["physics_accel_magnitude"]` (wms.py:837-840) -> accumulated into `acc.hybrid_residual` / `acc.hybrid_physics` in `src/wms/train/unroll.py:195-197` WITHOUT detach (only detached at :277). In the stage-end loss-share pass (trainer.py ~366, forward_backward(backward=False, share=True), outside no_grad, enabled by loss_share_logging=true) no backward runs, so the graph of ALL chunks (up to 240 steps) stays alive. R4 (rollout_mode=residual) never created these accumulators.
   - Rough scale (hypothesis): ~40 MB per sample-step -> batch 8 x 240 steps ~80 GB if not freed.
   - Fix: accumulate `res.detach()` / `phys.detach()` at unroll.py:195-197 (keep `roll[...]` graph-carrying for ResidualPenalty, `src/wms/train/losses/residual.py:45`); optionally run non-probe chunks of the share pass under torch.no_grad. Fallback: `loss.loss_share_logging=false`.
2. MEDIUM: `driver_grad_across_chunks` re-encodes drivers every chunk (unroll.py ~108-134, wms.py ~701-704): 12 extra 61-step GRU passes at 240 steps, and `_encode_drivers` (wms.py ~345-352) uses the chunk-start state in `_driver_interaction`, so settings drift per chunk and differ from inference (encoded once at t0). Better: encode once at t0, use detached leaf tensors in every chunk, then `torch.autograd.backward([emb, params], [emb_leaf.grad, params_leaf.grad])` after the last chunk -> exact gradient, one encoder pass, train/inference consistent.
3. LOWER: per-step `checkpoint(use_reentrant=False)` (wms.py:679-714) is sound, but `lead_neighbour` + `push_history` (wms.py ~745-750, K x K per step), `_record_step` and `section_ids` are outside the checkpoint -> maybe 1-5 MB per sample-step retained; several GB at batch 8 x 240. Move into the checkpointed function if measured memory is high.

## Owner rules / defaults (by reading)
- No driver-type list, labels or true parameters in model/training/cli (encoder consumes only observed pre-t0 history; driver_sidecar pre-existing and null).
- Switches default OFF with the original call path (lane_gate -> shared gate; checkpoint_step False -> step_fn; driver_grad_across_chunks False -> original detach; stage_plan fills global truncation; new keys in OMIT_WHEN_DEFAULT). tests/unit/test_r5_switches.py exists (not run by the reviewer).

## Cheap verification
1. After the detach + wrapper fixes: r5_base CPU, batch 1, unroll_s [10,30], stage_steps [1,1], loss_share_logging true -> peak RSS should stop growing with steps.
2. Probe UnrollRunner.run(share=True, backward=False) at 40/80/120 steps with psutil RSS (linear now, flat after fix).
3. checkpoint_step=true, truncation 40, backward=True -> real per-chunk memory -> decide whether batch 8 x 240 fits 24 GB.
4. Bit-identity unit test + default-config golden step before/after.
