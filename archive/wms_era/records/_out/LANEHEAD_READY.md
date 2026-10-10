# Learned history-conditioned lane head readiness

Status: **READY — implementation, focused verification, a real-pack CPU CLI
dry-run, and a bounded mechanics check are complete. No GPU, SUMO, or remote job
was launched.**

Implementation commits: `4ca4de8` (history-conditioned lane-change head) and
`8c4ae8d` (finite-guard fix in the balanced lane loss), branch `wms/lanehead`.

## Design

The default `--lane-head off` leaves the previous decision path in place. In the
legacy path, `mobil_merge_logits()` supplies absolute-lane scores (stay = 0;
adjacent moves use MOBIL plus the mandatory-merge incentive and safe-gap masks).
Residual mode adds the learned temporal/residual lane logits through
`bounded_value(gate * lane_delta, lane_logit_cap)`. In `driver_params` mode the
physics IDM/MOBIL baseline is used without a lane residual. The new head is
selected only by `--lane-head history` and requires `--rollout-mode driver_params`.

The new three-class head predicts stay/left/right from the frozen pre-t0
DriverEncoder embedding and a 23-feature local context: current normalized
kinematics/lane, leader and follower gaps and relative speeds in all three lanes,
distance to the merge-lane end, time since the last lane change, and the
physics/MOBIL lane scores. Initial directional biases are -3.5 (a low hazard
prior); MOBIL is an input and does not choose the action. Hard masks reject
nonadjacent moves, unsafe gaps, and physics-infeasible targets. On the final 25 m
of the merge lane, a safe mandatory target can override dwell and stay is masked;
if no safe gap exists, the change remains refused. Decisions use argmax, move at
most one adjacent lane per vehicle per control step, and enforce a 3-second
minimum dwell except for that safe mandatory merge. The rollout carries the
encoder embedding and dwell counter across truncated chunks. Evaluator
checkpoints require and consume slot-aligned pre-t0 history rather than silently
substituting an empty history.

Training classifies recorded per-step transitions as stay/left/right with
inverse-frequency class weights capped at 20. Existing rollout position and
speed losses are retained. New files/tests do not enable the feature by default.

### Fix (`8c4ae8d`)

`balanced_lane_loss()` overflowed to `inf` on real data: a recorded transition can
name a directional action that the current hard feasibility mask forbids, so its
logit is `finfo.min`, and `weight * cross_entropy` then exceeds the float32
maximum. The loss now drops steps whose target action is hard-masked and returns a
finite graph-connected zero when no step remains (`logits.sum() * 0.0` was itself
`-inf * 0 = nan`). A regression test covers both paths.

## Verification and dry-run

Environment used directly: `C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe`
(torch 2.12.1+cpu).

- `python -m pytest output/wd01/wm_n1/wms/tests tests/test_wms_eval_wms_plugin.py -q`:
  **222 passed, 2 skipped**. The two skips are unavailable TRAIN chunk-4 states
  `w8676m2700p20t360` and `w8677m2700p20t360`.
- Focused lane/evaluator test selection
  (`wms/tests/test_wms_lane_head.py tests/test_wms_eval_wms_plugin.py`):
  **19 passed** (lane-head file 8 of 8).

### Real-data CPU dry-run (real `python -m wms.cli train` argv path)

Ran the real trainer on a tiny TRAIN slice of the original packs (6 windows: two
each from pack_v3/pack_dspeed/pack_damping, taken from the registered
`windows_train_noPI.json`; `--exclude-window-list pi_excluded_runs.json`), with
`--rollout-mode driver_params --lane-head history`, `--batch-size 2`,
`--sumo-fidelity --start-shift-prob 0.75 --truncation-steps 20`,
`--stage-steps 1,1,1,1` over all four curriculum stages (10/30/60/120 s).

Read-only inputs (real paths behind the frozen junctions):

- `pack_v3 = E:/rd-wt/ws-cl/_ext/pack_v3` (canonical `D:/rd-staging/wmn1_feat/pack_v3`,
  passed through `WM_N1_CANONICAL_PACK`),
- `pack_dspeed = E:/rd-wt/dspeed-data/_out/pack_dspeed`,
- `pack_damping = E:/rd-wt/damping-snap/_out/pack_damping`,
- identities `E:/rd-wt/ws-p2/_ext/identities`, `E:/rd-wt/a9/_out/identities_dspeed`,
  `E:/rd-wt/damping-snap/_out/identities_damping`,
- overlays `labels_overlay_dspeed`, `labels_overlay_damping`,
- outcomes `D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z` via `WM_N1_TRAIN_OUTCOMES`,
- LPO window lists in `_out/lpo/`.

Result: **rc=0, 4 optimizer steps** (one per stage), every logged record
`finite: true`; measured **peak RSS 803.9 MB** (sampled, parent + children),
far under the 6 GB ceiling; wall time ~366 s. Lane loss per stage (10/30/60/120 s):
**0.7133, 0.3837, 0.7064, 0.9799**; position 12.54/82.65/133.63/349.65; speed
0.569/1.579/1.574/2.509; recovered driver parameters stayed finite and inside
their hard bounds. This is a mechanics/argv check on a real pack slice, not an
efficacy result.

### Bounded mechanics check (21 states, mechanics only)

On `E:/rd-wt/ws-evalh/_out/c1_states_v2c/unseen_u_new_world.json` (read-only, one
record at a time; `_out/LANEHEAD_MECHANICS.json`), a 30 s rollout with all CAVs
forced to HDV and untrained (no-checkpoint) model weights:

| predictor | lane changes | accuracy @10 s | accuracy @30 s |
| --- | --- | --- | --- |
| frozen (t0 lane held) | 0 | 0.9731 | 0.9314 |
| MOBIL (lane_head off) | 29 | 0.9736 | 0.9319 |
| untrained head | 13 | 0.9751 | 0.9331 |
| recorded | 188 | — | — |

These are mechanics sanity numbers (the untrained head barely leaves "stay"),
not accuracy or efficacy claims.

## Chain and GPU preparation

`wms/lanehead_chain.sh` carries the old-history/C1 pack, window-list, stage-gate,
and step-budget configuration forward, adding only `--lane-head history` through
the `LANE_HEAD` switch. The script itself is LF-only and it LF-normalises its copy
of `old_hist_chain.sh` in the writable run directory before invoking Bash, so a
CRLF Windows checkout cannot break the Linux run. It keeps the 10/30/60/120 s
stages, the fixed 2,500/2,500/1,875/1,250 step budget, the multi-stage gate, the
400-step timing gate, and the `MAX_HOURS=14` refusal. Estimated duration is
approximately **9–11 hours**, extrapolated from the C1 chain's estimate; the new
head's GPU overhead is unmeasured.

Prepared GPU launch command (after TRAIN-only inputs and code are staged on gpuA;
not executed here):

```bash
bash "$W/code/output/wd01/wm_n1/wms/lanehead_chain.sh"
```

Code archive (LF), built from the code commit `8c4ae8d` (the tar pax header
embeds the commit id, so build from that ref to reproduce the hash):

```bash
git -c core.autocrlf=false archive --format=tar \
  --mtime="2026-10-08 00:00:00 +0000" -o code_lanehead_8c4ae8d.tar \
  8c4ae8d output/wd01/wm_n1
```

`_out/code_lanehead_8c4ae8d.tar` (9,687,040 bytes), SHA-256:

```text
f1d845b203b5abae7b164e04e181773801b132c33eefba0078cb9afbad20b5fa
```

`lanehead_chain.sh` and `old_hist_chain.sh` inside the archive are LF-only.

## Remaining action

Stage the TRAIN-only C1 inputs and this archive on gpuA, then run the launch
command. The chain self-checks with `wms.gate_s2` / `wms.check400` and refuses the
full run above 14 hours; refresh the estimate from its own smoke timing.
