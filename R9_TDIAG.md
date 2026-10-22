# R9_TDIAG - round-9 training-path launch blockers (branch c4/v10r9tdiag), 2026-10-22

Evidence: gpuR RTX 4090, real TRAIN batches (v5_native / genrand_plain / genrand_branch, round-8 stage300 init, round-9 config,
batch 1, 60 s and 120 s stages), `scripts/r9_tdiag_probe.py` (PROBE ONLY: wraps `score_chunk` to re-score every event slot alone and
describe it, wraps `geometry` for rollout scene validity, nan_to_nums the loss so a non-finite batch still reaches backward; nothing
of it is in `src/`). Raw rows: `runs/active/tdiag_gpu/out_a*.jsonl` (gitignored), viewers `scripts/r9_tdiag_{show,tab,on,grad,tf}.py`.

## Blocker 2 (NO GRAD_FN) - solved, a different cause than assumed

Not `merge_pivot_state_st`, not the freeze list: **the merge_pivot tensors do receive gradients.** Per-batch evidence (state_st=false,
clean batches, 8 trainable tensors): batches with no non-finite chunk give nonzero grad on 8/8 tensors, e.g. a1 step 5
(s_f 2.6e-3, T_f 3.3e-3, b_f 2.6e-3, s_l 3.9e-3, T_l 7.5e-4, b_l 8.3e-4, tau 3.3e-2, w 1.2e-2), step 9/10 8/8 as well (max 0.23).
The "element 0 of tensors does not require grad and does not have a grad_fn" is raised AFTER the chunk loop, in
`src/wms/train/unroll.py` `_finish` (`value.value.backward()`, traceback captured): the differentiable **unroll** terms
(`LaneEventLoss`, `StabilityLoss`) are back-propagated unconditionally. Round 9 freezes everything except merge_pivot (lane_v5, residual_net ...)
but the config chain (c4_wm_v5 -> ... -> round9) still has `loss.lane_event.weight = 10.0`, whose value therefore has no graph.
So EVERY real round-9 batch raised at the end of `run()` (after finite chunk gradients were accumulated), including batches with a finite
loss; batches without scored merge cars simply showed the same error first.
- Fix: `unroll.py` (`_finish`): skip the backward of a differentiable unroll term only if its value has no autograd graph
  (`torch.is_tensor(v) and v.requires_grad`); the value is still logged, a term WITH a graph is always back-propagated. No loss value changes.
- Tests: `tests/test_r9_tdiag.py` (2): frozen-input stability term no longer aborts and merge_pivot `tau_raw.grad > 0` while no frozen tensor gets a grad
  (fails with the old code: RuntimeError reproduced); trainable-input stability term is still back-propagated (residual_net grads nonzero).
  Optional (owner): set `[loss.lane_event] weight = 0` in the round-9 config - it is dead weight in the logged loss.

## Blocker 1 (NON-FINITE BATCHES) - root cause: ON-POLICY rollout divergence, not the teacher-forced path

24 distinct real batches (a1: 12 at 60 s, a4: 8 at 60 s seed 11, a5: 4 at 120 s seed 12; a3 repeats a1's first 6 with state_st=true):
non-finite chunk in 7/12, 1/8, 3/4 = 11/24 (46 %; memory probe 12/21, 6/18, 4/5). By term:

| term | zero-mass events (cells) | notes |
|---|---|---|
| teacher-forced recorded path (tf) | **1** (a5 step 2, window row 0) | the known ~0.3 % class: ego standing at the lane end (x = 75.44, v = 0), follower overlaps the rear by 3.9 m (`gap_l = -3.88`), recorded merge executed by SUMO; recorded geometry vetoes it (`overlap`). Not window-boundary bookkeeping (row 0 but x is the real lane end). Same class as the offline 12/3701 unsupported cars |
| on-policy (rollout) event/survival term | **33** (in the 18 batches tabulated; 84 vetoed event cells of which only 20 pass the support mask) | every one has `rec_eligible = True` (recorded geometry legal) and WM decision geometry `eligible = False`, support mask TRUE |
| F2 pre-window prefix | 0 | prefix paths only supply history; no event is scored there |

Why on-policy events are zero-mass: the recorded event is scored on the WM's OWN rollout state at the same time. The WM scene diverges and its
decision geometry (frozen one-step motion) physically forbids the merge: `overlap` of the ego with a WM target-lane body (also CAV incoming
`fixed_overlap`). Examples (rec vs WM position of ego / blocking partner): ego 13.3 vs 43.2 m, blocking CAV recorded 110.6 vs WM 44.9 m;
ego 14.7 vs 26.7 m, partner CAV 21.9 vs 26.4; ego 9.3 vs 21.5, HDV partner 2.1 vs 25.7 m. |dpos| at vetoed on-policy event cells: median 38 m
(p25 8 m, p75 138 m) vs median 4.9 m for the 9 eligible ones; the support mask lets them through because `loss.merge_core` tolerances
are `pos_tol 50 m / speed_tol 8 m/s / gap_tol 40 m / gap_speed_tol 8 m/s` (round-9 toml; code defaults 25/5/20/5), 3 orders looser than the 2-5 m
bumper gaps of a queue merge. So: offline fitter/audit only cover the teacher-forced path (0.3 %); the on-policy path has never been audited, and
nothing in `score_chunk` removes a vetoed event: `support` only checks position/speed/gap *similarity*, not that the WM scene still affords the recorded
legal opportunity.
Secondary, unresolved: in the rollout geometry 1 % of ramp-car steps (62 / 8074; 13 of 3014 rows) have an invalid *scene* (a present slot with `length <= 0`,
171 slot-steps; `geometry()` voids every ramp car in that row). `GEO len0_examples` in run a6 isolates which slots (see addendum).

Genuine contract bug found on this path: none that changes the zero-mass count; the ZM events are physically consistent WM-vs-record divergences.
What is NOT allowed to fix it (so nothing was changed): dropping vetoed on-policy events from the NLL, a floor, relaxing the veto, tightening the tolerances.
Options for the owner (each a decision, none implemented):
1. keep the on-policy term and accept the exact semantics: "a recorded legal merge that the rollout scene cannot afford is unsupported" as an explicit,
   counted exclusion class, with the same physical definition published in the audit (needs a rollout-side audit first; today no audit exists);
2. run round 9 on the teacher-forced term only (`loss.merge_core.weight = 0`, on-policy still logged via `metrics = true`, so zero-mass counts stay visible);
3. tighten the support tolerances to physical scale (hand-set thresholds - owner call).

## Real-data check of the actual entry (see addendum below for numbers)

Real training entry `python -m wms train --config configs/c4_wm_v10round9.toml --mode run` on gpuR (repo = this branch, round-8 init, real TRAIN data,
60 s stage, batch 1, truncation 60, `model.merge_pivot_state_st=true` as the owner requires, 4 steps requested; run dirs
`/root/autodl-tmp/c4/v5/tdiag_r9/run_b1`, `run_b2`):
- **b1, exactly the round-9 recipe**: aborts at step 0, `RuntimeError: non-finite first_passage loss` - tf: 2 events, 2 positive mass, 0 zero mass;
  on-policy: 1 event, 0 positive mass, **1 zero mass (physical veto)**, `on_nll = inf`. Blocker 1 is NOT closed by any bookkeeping fix.
- **b2, same + `loss.merge_core.weight=0.0` (on-policy term logged only, a diagnostic, NOT a proposal I applied)**: **4/4 optimizer steps complete**,
  finite loss (150.0, 111.3, 207.3, 186.7; includes lane_event w=10 value), finite nonzero grad_norm (9.3e-4, 2.6e-3, 4.4e-3, 1.0e-3, unclipped),
  tf zero mass 0 on all 4 steps (2, 2, 5, 5 events, all positive mass), on-policy zero-mass counted and visible (1, 2, 2, 4 of 1, 2, 2, 4 events),
  tau 0.07274 -> 0.07278, w 1.0 -> 0.9982 (both move). Blocker-2 fix verified end to end under state_st=true (before the fix this raised at the end of every batch).
- Per-tensor merge_pivot gradients (state_st=false probe, clean batches): 8/8 nonzero, see Blocker 2; state_st=true batch-1 run a3 reproduced the same
  batch-0..5 values (identical forward) with the full-graph backward completing (no OOM at 60 s).

Entrant slots (open, not fixed): in rollouts the WM geometry sees present slots k >= 192 (generated entrants/queue) with `length_m = 0` and positions such as
-1733 m or -2625 m (run a6: 19 slot-steps in one batch, 7 in another); `geometry()` treats a present slot with length <= 0 as an unobserved scene and vetoes
EVERY ramp car in that row (`ramp_scene_invalid` 62 of 8074 ramp-car steps overall, up to 16 % of the steps of single batches). Whether these slots should be
present, carry a length, or be excluded from `scene_valid` is a model-contract question for the entry/geometry owner; I did not change it because it needs
its own equivalence check against the recorded (fitter) path and its share of the 33 on-policy zero-mass events was not measured in the time available.

## Tests / commits
- `PYTHONPATH="src;.;tests" python -m pytest -q tests/test_merge_pivot_training.py tests/test_r9_tdiag.py` -> 48 passed, 0 failed (local CPU).
- Commits on c4/v10r9tdiag: probes; `unroll.py` fix + `tests/test_r9_tdiag.py`; this note. Not pushed. gpuR work dir `/root/autodl-tmp/c4/v5/tdiag_r9` (kept; 33 MB).
