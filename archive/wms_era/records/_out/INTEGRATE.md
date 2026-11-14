# WM-S integrator log (branch wms/integrate)

Plan: `E:/research-dense/docs/WM_STATE_PLAN.md`. This file is the merge/test log.
Policy: merge only (never reset/rebase/clean/stash); minimal fixes preferring
flag-gated default-off; record branch -> merged sha -> tests -> notes.

## Pass 1 (2026-10-05 22:40 relaunch, executor restart)

Base before pass: `1c29f22` (wms/integrate).

| branch | tip | merge result | tests | notes |
|---|---|---|---|---|
| wms/p2 | `430360f` | fast-forward `1c29f22..430360f` | wms 4 failed/7 passed; driver 14 passed; rollout2 58 passed; closedloop 1 failed -> fixed (see below) | p2 already contains wms/torchphys (`7879170`) + wms/driver (`9d7215e`). Adds trainer `wms/train.py`, `wms/model.py`, `wms/physics_torch.py`, `wms/cli.py`, `driver/*`. |
| wms/torchphys | `7879170` | contained in p2 (already ancestor) | covered by p2 | no extra merge needed |
| wms/driver | `9d7215e` | contained in p2 (already ancestor) | covered by p2 | changed `rollout2/step.py` -> `accelerations(v0_override=...)` |
| wms/merge | `1c29f22` | no-op (already the base) | - | nothing ahead |
| wms/eval | `1c29f22` | no-op (already the base) | - | nothing ahead |
| wms/p4 | `5d729c0` -> merge `47dd4fb` | merged | tests/test_wms_decide.py 3 passed | adds `wms_decide.py` at repo root |
| wms/p5 | `5bc66f1` -> merge `dbbb1b9` | merged | p5splits/tests 9 passed | adds `_out/p5_splits.json` (30k lines), `p5splits/*` |
| wms/cl | `44e5f33` -> merge `8cf615a` | merged | closedloop/tests re-run 27 passed | adds `closedloop/wm_s_predictor.py`, policies/runner edits |
| wms/p2data | `182fa11` -> merge `1d6cbb3` | merged | - | `_out/T5.md` record only (BLOCKED) |

No merge conflicts occurred; every merge applied with the `ort` strategy.

### Fixes applied
- `output/wd01/wm_n1/closedloop/skip.py`: the `install_trace_hook` wrapper
  `def accelerations(self, w)` did not accept the new `v0_override` keyword that
  `wms/driver` added to `V2GCore.accelerations`. Changed to
  `def accelerations(self, w, *args, **kwargs)` forwarding through. This was the
  only merge-induced regression (closedloop adapter test); commit `55f4a2e`.

### Test status after pass (rl_driving env, OMP/MKL threads 4)
- `output/wd01/wm_n1/wms/tests`: **4 failed, 7 passed** (branch-internal to wms/p2).
- `output/wd01/wm_n1/wms_eval/tests`: path does not exist (nothing to run).
- `output/wd01/wm_n1/closedloop/tests`: 27 passed (after fix).
- `output/wd01/wm_n1/rollout2/tests`: 58 passed.
- `output/wd01/wm_n1/model/tests`: 56 passed, 1 skipped (CUDA unavailable).
- `output/wd01/wm_n1/train/tests`: 43 failed, 416 passed, 4 skipped.
- `output/wd01/wm_n1/driver/tests`: 14 passed.
- `output/wd01/wm_n1/p5splits/tests`: 9 passed.
- `tests/test_wms_decide.py`: 3 passed.

### Failure attribution
- **wms/tests (4)**: `test_physics_torch.py`. `physics_torch.py` and the test are
  byte-identical to `wms/p2` (`e270528...`), so these are branch-internal defects
  from unverified executor code (T6/T4 admit torch was unavailable), not caused by
  integration. Concrete causes: (a) `test_eight_world_numpy_parity_*` oracle updates
  leader-then-follower sequentially while the batched step is simultaneous and the
  tolerance (2e-4) is below float32/float64 drift; (b) `test_projection_exit_entries`
  relies on a version-specific `torch.poisson(seed=7)` draw; (c)
  `test_mandatory_merge_*` selects a follower gap of 5 m < `merge_follower_gap_m=8`
  and the mandatory side is hard-coded to `lane+1`; (d) `test_eight_world_lane_decision`
  starts a merge-lane vehicle at s=150, below `merge_start_m=378.66`. Culprit: wms/p2.
- **train/tests (43)**: all `CorpusRefusal` / `IdeaAConfigError` / `LabelIsolationError`
  from `train/dataset.py`, `train/idea_a.py`, `train/label_isolation.py` — no merged
  branch touched `train/`, so these pre-date the merges (identity/INDEX gate vs
  fixtures). Not merge-induced.

### Dry-run gate
`wms/p2:_out/T4.md` does **not** contain `READY FOR GPU: yes` (it reports the local
TRAIN-pack dry run as the remaining gap). Per brief, the real-data dry run (defaults /
sumo_fidelity + open road / calibrated merge) was **not** triggered. No code archive
produced yet.

Certified integration (code) commit for this pass:
`55f4a2ef47ccf74bf8612118ee32b27330411b9f`. Only log-only commits may sit on
top of it (this file is the sole delta).

READY FOR GPU: no + 55f4a2ef47ccf74bf8612118ee32b27330411b9f

## Pass 2 (2026-10-06 00:00-01:30, WM-S integrator relaunch after executor crash)

Base before pass: `76260bb` (wms/integrate; pass 2 had merged p2/p4/cl/p5/eval/torchphys/merge
but crashed before updating this file). Reused: `_out/T4.md`, `_out/T4_FIX.md` (dose-run recipe),
`E:/rd-wt/ws-review/_out/REVIEW.md`.

| branch | tip | merge result | tests | notes |
|---|---|---|---|---|
| wms/torchphys | `4a8b9de` | merged `2e18f29` | wms 15 passed (after reconcile) | completes blockers 1,2: `MODEL_FRAME_LANE_LENGTHS` (mainline 553.33 / accel 75.44), lane-specific exits, mandatory-merge uses the target lane directly. |
| wms/p2 | `cf78c6f` | merged `e0b5d4e` | wms 15 passed | report commits only (`_out/T4.md`, `_out/T4_FIX.md`); blocker-3 fix `1c46c3f` already in tree. |
| wms/p4 | `107ae10` -> `c6d7d98` | merged `e2cc36c`, `b13b39c` | decide + eval plugin tests pass | blockers 4,5 (`b560d49`); `c6d7d98` adds `wms_eval/wms_plugin.py` (resolves the missing plugin referenced by `b2dd684`). |
| wms/merge | `6d0edee` -> `f131347` | merged `e311f3f`, `eabd727` | rollout2 64 passed (after fix) | calibrated `merge_params.json`; `f131347` added `MergeCalCore.accelerations` which missed the driver `v0_override` kwarg (regression, fixed below). |
| wms/cl | `d0034fd` | already merged `21c2812` | closedloop passed | blocker 6 (`124e43b`). |
| wms/p5 | `83dedcb` | already merged `ca4e67f` | p5splits 10 passed | blocker 7 (`bf3f330`). |
| wms/eval | `5299395` | already merged `76260bb` | eval tests pass | harness. |
| wms/driver | `9d7215e` | already merged `ac94e46` | rollout2 passed | `v0_override`. |

### Review blockers (FIXED/OPEN per merged sha)

1. **FIXED** `wms/torchphys 4a8b9de` (merge `2e18f29`): model-frame geometry is the frame
   default (`merge_start_m=0.0`, `merge_end_m=75.44`, lane ends `(553.33,553.33,75.44)`).
2. **FIXED** `4a8b9de`: mandatory merge checks the actual `merge_target_lane` occupancy
   (`_neighbors(query_lane=merge_target_lane)`), not an index into the `[lane-1,lane+1]` pair.
3. **FIXED** `wms/p2 1c46c3f` (merge `ac94e46`): per-tick CAV identity masks; regression
   `test_cav_mask_follows_identity_when_a_slot_is_reused` passes.
4. **FIXED** `wms/p4 b560d49` (merge `dd574d8`): per-row TRAIN + 64-hex provenance validation.
5. **FIXED** `b560d49`: `total_delay` is registered normalized seconds (`test_..._normalized...`).
6. **FIXED** `wms/cl 124e43b` (merge `21c2812`): closed-loop predictor fails closed on short/
   misaligned predictions.
7. **FIXED** `wms/p5 bf3f330` (merge `ca4e67f`): J4 requires wall policy AND frozen v2.1 law.
13. **FIXED** in integrate `7bc523e` (additional review blocker): `wms_eval.cli._inside_workspace`
    resolves links via `realpath`, so a symlink/junction cannot escape the workspace;
    `tests/test_wms_eval_paths.py` reproduces the junction escape.

Major 8/9/10/11/12 are outside the 7-tracked set; 8's bumper gap is available behind the
flag-gated `PhysicsConfig.bumper_gap` (default off), and 11 was already fixed in `1c46c3f`.

### Integrator fixes (minimal, flag/contract preserving)

- `7bc523e`: blocker-13 path confinement + `physics_torch._project_gaps` out-of-place column
  updates (the crashed rewrite's in-place `safe_s[:, i] =` broke backward once the entry path ran)
  + reconciled the 4 stale `test_physics_torch` cases to the corrected contract.
- `e0d87fb`: parity reference aligned with post-`4a8b9de` contract (`bumper_gap=True`,
  unbounded road so the new model-frame exit does not end the comparison; 10-step horizon because
  IDM platoons are chaotic and float32-invalid beyond ~5 s).
- `cdfc8e7`: `MergeCalCore.accelerations(..., v0_override=None)` forwards to `V2GCore` — fixes the
  3 merge-induced `test_merge_cal` failures (culprit: `wms/merge f131347`).
- `97c662d`: restored `tests/test_wms_eval_wms_plugin.py` direct import now that
  `wms_eval/wms_plugin.py` lands (`wms/p4 c6d7d98`).

### Test status at frozen HEAD `97c662d` (rl_driving py3.10, OMP/MKL threads 4)

- Combined: `wms tests, closedloop tests, rollout2 tests, model tests, p5splits tests,
  tests/test_wms_eval_metrics.py, tests/test_wms_eval_wms_plugin.py,
  tests/test_wms_eval_paths.py, tests/test_wms_decide.py`
  -> **196 passed, 1 skipped** (skip = CUDA unavailable).
- `train/tests`: **43 failed, 416 passed, 4 skipped** — unchanged pre-existing identity/split-gate
  refusals (`CorpusRefusal` / `IdeaAConfigError` / `LabelIsolationError`); no merged branch touches
  `train/`. Not merge-induced.
- `wms_eval/tests` path does not exist; the harness tests live in `tests/test_wms_eval_*.py`.

### Real-data CPU dry run (READ-ONLY TRAIN; no SUMO/GPU/SSH; threads 4)

Recipe from `_out/T4_FIX.md` / `E:/rd-wt/ws-p2/_out/dry.sh`; run inside this workspace:
`WM_N1_TRAIN_OUTCOMES=D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z`,
`WM_N1_TRAIN_IDENTITIES=E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities`,
`wms.cli dry-run --pack D:/rd-staging/wmn1_feat/pack_v3 --pack dspeed --pack damping
 --identities (3) --overlay (2) --out E:/rd-wt/ws-integrate/_out/wms_dry_final3
 --threads 4 --sumo-fidelity`. Open road is the model default; calibrated merge params exist at
`output/wd01/wm_n1/rollout2/merge_params.json` (the trainer CLI takes no merge-param input).
Result: **status passed**, 2 optimizer steps, 3060 TRAIN records, loss 8.6548 -> 8.4686,
valid_targets 5981, g(0)=0.0, gate monotone, finite. Receipt `_out/wms_dry_final3/TRAIN_RECEIPT.json`.

### Pass 3 (2026-10-06 ~01:50, continued polling + final freeze)

Further branch movement was merged and tested (all `ort`, no conflicts after the two
resolutions below):

| branch | tip | merge | tests | notes |
|---|---|---|---|---|
| wms/eval | `3a11839` | merged `3460b62` | eval plugin tests pass | adds tracked `wms_plugin.py` + `report.py`. add/add conflict with the p4 plugin; resolved by taking the wms/eval version (canonical), 196 passed after. |
| wms/torchphys | `a0ab9bc` | merged `a0c2c8c` | wms 17 passed, 2 skipped | author's own fix of blockers 1-2: functional `_project_gaps` (same autograd fix), uses `target_valid`, adds `test_rollout2_parity.py` and the reviewer-named tests. Conflict resolved by taking the torchphys version wholesale (supersedes the integrator stopgap test edits). |
| wms/torchphys | `e09f51a` | merged | - | `_out/T6.md` evidence trim only. |
| wms/cl | `402d238` | merged | closedloop pass | blocker-6 residual: fail-closed t0 slot coverage + uniform `dt_s` time grid. |
| wms/p5 | `b8d7033` | merged | lpo 18 tests pass | metadata-only leave-PI-out window lists (`output/wd01/wm_n1/lpo`). |

Final merged-tests at frozen HEAD `c05650d` (threads 4):
`wms, closedloop, rollout2, model, p5splits, lpo, tests/test_wms_eval_*,
tests/test_wms_decide.py` -> **210 passed, 3 skipped** (2 = `test_rollout2_parity`
chunk-4 TRAIN states absent from this workspace; 1 = CUDA unavailable).
`train/tests` remains **43 failed / 416 passed / 4 skipped** (pre-existing identity/split gate;
`train/` untouched by every merged branch).

Final real-data CPU dry run at HEAD `c05650d` (same recipe/env as above, out
`_out/wms_dry_c05650d`): **status passed**, 2 optimizer steps, loss 8.6788 -> 8.4678,
valid_targets 5981, g(0)=0.0, monotone, finite. Receipt `_out/wms_dry_c05650d/TRAIN_RECEIPT.json`.

### Archive (final)

`git archive --format=tar --output=_out/code_wms_integrated_c05650d.tar c05650d output/wd01/wm_n1`
SHA-256 `5d0fe8c299159fe596863c603a9b9451eb926526732a387e85060939ee4ef9c2`
size 9,400,320 bytes (`_out/code_wms_integrated_c05650d.tar.sha256`).

READY FOR RUN 2: yes + c05650d208043db2f09ea8252ba84ff83b18a7b6
