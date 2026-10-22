# R10_PREP: round 10 = round 9 + fix B (cut-in reaction), prepared 2026-10-22

Branch `c4/v10r10` (worktree E:/agent-work/c4-v10r10). Not pushed. Nothing run on gpuR, no SUMO, no GPU, TRAIN data only
(local copy E:/agent-work/c4-v10r9fix/data/r9_train, read-only). Review packet for astra:
E:/agent-work/astra-insight/src_r10/ROUND10_PROPOSAL.md (the full argument; this file is the operational summary).

## 1. Merge
`git merge c4/v10r9k2` into c4/v10r10 (base 0aba97cc): clean, no conflicts (merge commit 7030ad63). Round-9 merge-pivot code
and the cut-in code coexist; none of the files reserved for the round-9 agents was edited (tools/r9_*.py,
src/wms/train/losses/merge_pivot.py, configs/c4_wm_v10round9.toml, scripts/round9_gate.sh).
The two "inherited round-8 config test failures" of R9_K.md (stale normalisers 1.0 vs 395.2/2.15) were already fixed at the
root by d679c0e6 (R9A) in the base: tests/test_merge_core_v2_config.py passes unchanged (no weakening/skip/xfail).

## 2. Fix B finished
- **History hook installed** (`UnrollRunner.run`, src/wms/train/unroll.py): when `cutin_reaction` is in the composer,
  `targets.prepared['cutin_reaction_inputs']` = {history, history_mask, driver_history, driver_history_mask} = the
  rollout's own detached pre-t0 observations (C4 windows keep stable slots, so no `reset` mask is needed).
- **Telemetry** in the train_log record: `cutin_reaction_{n_events,n_event_rows,n_control_rows,censored_windows,event_loss,control_loss}`
  (plus the existing `cutin_reaction`); forward-only/no_grad passes skip the replay and clear stale stats.
- **Replay cost/memory**: per-step activation checkpointing (same as `WmSModel.step`) and autograd only on steps that
  carry a supervised row (others advance the carried state under no_grad = truncated carry). Values identical to the plain replay (tests).
- **Leader-change features, hooks 1-6**: deliberately NOT installed (a second mechanism, new zero-init parameters and a
  rollout carry; violates "one change"; not needed because the cut-in loss acts on total acceleration). To remove the silent
  no-op trap, `ModelSpec.leader_change_features=true` now raises (fail closed); default False, tracker still used for replay.
- Config hooks (settings/compose/legacy flat) were already in the merged branch.

## 3. Tests (CPU, Python = rl_driving env, PYTHONPATH=src:.)
Env: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest ... -q -p no:cacheprovider` from the worktree, PYTHONPATH=src:. All runs on 2026-10-22.
1. `tests/test_cutin_reaction.py tests/test_r10_config.py tests/test_merge_context.py tests/test_merge_core.py tests/test_merge_core_v2_onpolicy.py tests/test_merge_core_v2_seqtf.py tests/test_merge_core_v2_config.py` (deselect `test_r10_config.py::test_cutin_weight_comes_from_the_documented_rule_file`, run separately below): **106 passed**.
   New in this task: 6 cut-in tests (grad-step truncation, checkpoint equivalence, runner history hook + telemetry, no_grad skip, fail-closed leader_change_features) and 5 config-pin tests (tests/test_r10_config.py).
2. `tests/test_merge_pivot.py tests/test_merge_pivot_hybrid.py tests/test_merge_pivot_pre_window.py tests/test_merge_pivot_runtime.py tests/test_merge_pivot_training.py tests/test_r9_support_audit.py`: **225 passed**.
   `tests/test_r9_exact.py tests/test_r9_fit_exact.py tests/test_r9_runtime.py`: **83 passed** once `data/r9_train/wm_s_stage300.pt` and the one episode `v5_native/episodes/w963004m1900p40t300__native` are present inside the worktree (they are gitignored data, copied read-only from c4-v10r9fix; without them 69 tests ERROR with FileNotFoundError, which is an environment gap of the round-9 tests, not a regression).
3. `tests/unit/config tests/unit/train tests/unit/test_c4_slowres.py tests/unit/test_wmloss.py tests/unit/test_wmlane.py`: **274 passed**.
4. `tests/equivalence`: **60 passed, 16 skipped** (skips are pre-existing golden/checkpoint-availability skips in that suite, untouched).
5. `tests/test_r10_config.py::test_cutin_weight_comes_from_the_documented_rule_file`: passes with the provisional weight file (5 passed in tests/test_r10_config.py in total).
`ruff check` on every file I touched (unroll.py, cutin_reaction.py, tools/r10_*.py): clean.

## 4. Component attribution (tools/r10_attribution.py; json: tmp_r10/attr_r9.json, copy in the astra packet)
740 insertions / 164 windows (16 local episodes, all manifest-role `val` of the round-8 TRAIN split, i.e. not fit by round 8),
teacher-forced replay of the round-8 weights in the round-9/10 model. Follower post-insertion min acceleration, model - recorded:
full -0.66 [-0.79,-0.52]; IDM core only -0.21 [-0.33,-0.09]; IDM+attention head -0.55; IDM+temporal -0.35.
**Verdict: both over-react, by regime.** Tight gaps (gap < 8.6 m, n=245): the IDM core alone brakes to p50 -1.4/-1.3 vs
recorded +0.2/+0.1 and the residual does not help. Larger gaps: IDM at/above recorded, the learned residual pulls it below.
The learned residual is about 2/3 of the mean deficit. **The residual attention head is dead**: pre-tanh activation in
[-26.2,-23.4] on all 45,132 cached rows, tanh' <= 1.9e-20, constant -1.5 m/s^2 before the gate (-0.58 at the saturated gate 0.386),
exactly zero gradient. Also: recorded a[t+1] ~ a[t] (Huber 0.009 vs model 0.32, IDM 0.28), a[t] is a residual input
(persistence shortcut risk).

## 5. What round 10 trains
Trainable: `temporal.*` (MLP, 6.5k params, lr x5; pilot rule in the astra packet) and `gate_g_max_logit`, `gate_tau_raw` (lr x100). Frozen bitwise:
merge_pivot, driver_encoder/updater, lane_gate_*, lane_v5, history_proj, k_embedding, merge_core, the whole residual_net
(incl. the dead attention head). Justification and the "one change" decision: round 9 trains only merge_pivot; round 10
freezes it so every round-10-vs-round-9 difference is attributable to fix B and the merge result of round 9 stays valid;
override for a joint run: `--set optim.freeze_prefixes=[...]` without "merge_pivot". The IDM core's only handle (encoder) is
not trained (would alter inferred driver settings for every vehicle); the tight-gap IDM over-brake therefore only
partially fixable (stated limit).

## 6. Config / weight rule
configs/c4_wm_v10round10.toml: extends round 2 directly with round 9's blocks copied inline (round 9 already uses the full 16-file
extends depth, so extending round 9 is refused by the loader; tests/test_r10_config.py pins that the effective config differs
from round 9 only in name, model.init_from, loss.cutin_reaction.*, optim.freeze_prefixes, optim.lr_scale, runtime.out_dir).
Init: /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round9/wm_s_stage300.pt. `[loss.cutin_reaction]` weight 1.0 (provisional), control_weight 1,
pre_s 1, max_window_s 6, huber 0.5.
Weight rule (tools/r10_weight_rule.py, no hand-set number): weight = G_brake / G_cutin(weight=1), the geometric-mean accumulated
gradient norm of the explicit braking terms (brake_persist + brake_persist_quantile as composed in round 9/10) over the
round-10 trainable parameters divided by the gradient norm of the unweighted cut-in term, measured with the real runner on real
local TRAIN windows at the round-8 stage-300 weights. NOT COMPLETED (same CPU limit): the config uses weight 1.0 = the unweighted cut-in term (event mean + control mean) and docs/r10_weight_rule.json is marked PROVISIONAL. Run `python tools/r10_weight_rule.py --ckpt <round8 or round9 ckpt> --out ...` on gpuR/CPU and set loss.cutin_reaction.weight to its `weight_geomean_ratio` (the config test reads docs/r10_weight_rule.json).

## 7. Micro-smoke (tools/r10_micro_smoke.py: real WmSTrainer.train_batch, local TRAIN, 2 steps, batch 2, 20 s windows each with a recorded insertion)
NOT COMPLETED in the time available: the shared CPU (100% load, 3-5x slow) did not finish the first optimizer step (rollout + round-9 losses + replay, batch 2, 20 s) in 30+ minutes of wall time; window building (1 s warm) and trainer construction succeed on the round-10 config (INIT_FROM.json: 149 tensors loaded, 8 new = merge_pivot from the round-8 checkpoint, 0 mismatch). Command: `python tools/r10_micro_smoke.py --ckpt <round8 stage300> --roots tmp_r10/local_roots/v5_native --out <dir> --steps 2 --batch 2 --horizon-steps 40 --pre-steps 24`. What IS verified: cut-in loss nonzero/finite/telemetry/gradient reach on the real runner at small scale (tests/test_cutin_reaction.py), the trainable/frozen set on the real model (tests/test_r10_config.py), the telemetry keys (runner test) and the gate's checks (bash -n).

## 8. Gate
scripts/round10_gate.sh (LF, `bash -n` clean): same structure as round 9 (smoke then `exec` the full run; out dir
runs/c4_wm_v10_round10, repo dir repo_r10, exit 2/3/4). Checks: >= 4 steps, finite loss, no traceback, the round-9 merge_pivot
checks (events > 0, zero-mass == 0, scored seconds > 0, tau/w finite positive), cut-in events/rows > 0 and finite positive loss,
INIT_FROM.json complete, frozen tensors and merge_pivot bitwise unchanged, a trainable tensor moved. Not run on gpuR.

## 9. Blockers / caveats
- Only 20 pilot windows and a short micro-smoke fit in the time and CPU available (shared machine at 100% load).
- Free-rollout over-reaction not re-measured (needs SUMO-free rollouts on GPU); attribution is teacher-forced at matched state.
- The dead attention head means round 10 can only change the residual through the temporal MLP and the gate (global scale).
- Round-9 outcome unknown: the gate requires its checkpoint; if round 9 failed the merge, round 10 inherits it.

## STATUS (owner pause, 2026-10-22 ~17:10)
Done: merge (clean), fix B finished (history hook, telemetry, checkpointed replay, fail-closed leader_change_features), tests green (sections 3), attribution verdict (section 4), round-10 config + config-pin test, scripts/round10_gate.sh, astra packet src_r10/ROUND10_PROPOSAL.md, offline pilot.
Not done: the weight measurement and the 2-step local micro-smoke did not finish on the loaded CPU (weight is PROVISIONAL 1.0, docs/r10_weight_rule.json; rerun tools/r10_weight_rule.py and tools/r10_micro_smoke.py, commands in sections 6-7). Gate not run on gpuR.
Notes: data/ in the worktree (r9_train ckpt + one episode) is untracked test data; round-9 postponed so round 10 is not launchable until round 9 exists.
