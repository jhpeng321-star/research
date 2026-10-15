# V6MERGE: lane-end waiting and cooperative mainline yielding (2026-10-16)

Branch `c4/v6merge2` (from c4/int 916559d). Two opt-in physics switches, both default OFF:
`physics.merge_end_wait` and `physics.mainline_merge_yield`. The OFF path is bit-identical to HEAD.
They are meant for the v6 fine-tune `configs/c4_wm_v6_merge_ft.toml`.

## Root cause (V5_DIAGNOSIS.md Q1, verified in code)

Stopped ramp HDVs at the accel-lane end merged after a median 2.5 to 3.2 s in the model. SUMO
takes 39.8 s on the same states. The lane-1 merge queue never formed.

1. **`gap_stopped_relax=True` (v5/v6 physics.json).** A ramp merger below 1 m/s is admitted
   with only 0.5 m bumper clearance to both lane-1 neighbours, plus the follower's kinematic
   term `v_f^2 / 12`. Example: a follower crawling at 3 m/s needs only 1.25 m, so the merger
   squeezes into almost any gap of a crawling queue (`test_stopped_relaxation_admits_tight_gap_only_when_off`).
   The un-relaxed speed rule is also permissive: its constants are TRAIN-fitted (follower s0
   0.5 m, T 0.6 s, b 6 m/s^2).
2. **`merge_hazard_zone='forced_late'` (lane_v5).** Any feasible merge in the last 5 m gets a
   logit of at least 8, so with the relaxed mask a stopped car at the end merges at once.
3. **Lane application (`wms.py` step).** Nothing stops a lane-2 car from passing the lane end
   when the merge is masked; only the IDM lane-end obstacle holds it back. On the v6 stage-10
   checkpoint, 10 HDVs drove past 75.44 m on lane 2 (517 HDV-steps) on 72 wall states.
4. **No mainline reaction.** Nothing in the IDM core slows lane-1 traffic for a blocked merger.
   `merge_cal`'s yield exists, but it is off in v5/v6 and not wired to the speed gap rule.

## The rules (`src/wms/physics/mobil.py`, `merge_v6.py`, `accel.py`, `model/lane_v5.py`, `model/wms.py`)

- **`merge_end_wait`**
  - Applies to lane-end mergers: a ramp car within 25 m of the end and at or below 2 m/s
    (`merge_end_wait_distance_m`, `merge_end_wait_speed_mps`).
  - Such a car may enter lane 1 only if BOTH lane-1 gaps are SUMO-secure with the road's IDM
    constants:
    - leader gap `>= s0 + T v + max(0, v^2 - v_l^2) / (2b)`;
    - follower gap `>= s0 + T v_f + max(0, v_f^2 - v^2) / (2b)`;
    - no footprint overlap;
    - all checked on the current AND the one-step-anticipated configuration.
  - There is no stopped-vehicle relaxation. The rule is a final veto in `mobil_merge_logits`
    (a masked prior), so no lane head can resurrect the merge: not the forced-late window, the
    hazard models, or the residual lane delta.
  - The lane-v5 teacher-forced fast mask applies the same veto (equality tested).
  - In the step, a car staying on lane 2 never passes the lane end: it is clamped there with v = 0.
- **`mainline_merge_yield`**
  - For each blocked lane-end merger, only the NEAREST lane-1 vehicle reacts. That vehicle must
    have its front behind the merger's rear and be within 60 m front to front.
  - It treats the merger as a virtual leader with its OWN IDM (per-driver learned parameters
    when present). The virtual leader stands the road's s0 behind the merger's rear, so a
    yielding follower always leaves the merger's standstill secure gap.
  - Without that s0 offset, yielding deadlocked merges whenever the driver's learned s0 was
    below the road's 2.6 m. That first version gave longer waits than wait-only.
  - The reaction never accelerates the follower and brakes by at most 2 m/s^2 (and the
    driver's own decel).
  - No phantom braking: nothing happens for a merger behind the follower, overlapped with it,
    farther than 60 m, or not blocked, and nothing for the second vehicle back.
- Both rules are equal physics for every vehicle (HDV or CAV) and add no parameters. They are
  fully vectorised:
  - `[B, K, K]` masks with no loops and no host syncs;
  - one blocked mask per step, shared by the accel and lane logits;
  - the discrete pair selection runs without an autograd graph;
  - the gradient flows only through the selected `[B, K]` gap and speed.

## Plumbing

The switches are `physics.merge_end_wait` and `physics.mainline_merge_yield`. They are wired
through these files, like `stop_time_integration`:
- TrainConfig `PhysicsOptions`;
- `legacy_flat` (flat keys, omitted when default);
- `TrainSettings`;
- `build_physics`;
- `PhysicsConfig`, which stores the switches plus 4 constants.

The trainer persists all of them in `physics.json` (`asdict`). Old `physics.json` files load with
both switches OFF, and `from_params_file` never reads the switches.

## Evidence

**(a) Tests.** `tests/unit/test_v6merge.py` has 41 tests:
- veto vs relaxation, the both-gaps table, the speed/zone scope and fast-mask equality;
- closed loop: the merger waits more than 10 steps longer than OFF, merges only when both gaps
  are secure, and never passes the end;
- yield: the follower decelerates, there is no phantom braking (5 cases), per-driver IDM with
  gradients, and no deadlock;
- OFF bit-identical: physics logits/accel, the fast mask and a full v5-config model step all
  match a fixture captured on unmodified HEAD 916559d (`tests/fixtures/v6merge_off_golden.pt`,
  single-threaded, because NN matmuls differ bitwise across thread counts);
- config/checkpoint round trip, constant validation, ON backward finite, and fused gap check
  equal to the reference.

Relevant suites, on HEAD vs this branch:

| Run | Passed | Skipped |
|---|---|---|
| unmodified HEAD, 11 files | 190 | 10 |
| this branch, same files + test_v6merge | 229 | 10 |

The 11 files are physics (tests/unit/physics, equivalence, gaprule), lane (lane_rule, lanefix),
v6integ, v6enc, infrav5, lanev5, wmlane and golden_new.

The skips are the same on HEAD: 10 golden tests need local packs that are not registered on this
laptop. No frozen golden test fails on HEAD here.

**(b) Speed.** `tools/v6merge_bench.py`: v6 fine-tune architecture, B=16, K=96, 10 steps × 8
alternating repeats. Run on gpuW CPU (nice, 2 threads) because laptop timings varied ±25%. All
times in ms per step, OFF vs ON:

| What | OFF ms/step | ON ms/step | Ratio |
|---|---|---|---|
| physics core only | 24.5 | 31.3 | +28% (+6.8 ms) |
| no-grad rollout step | 212.4 | 219.2 | **+3.2%** |
| train fwd+bwd step | 438.6 | 432.6 to 440.6 | **+0.4 to 0.6%** |

The physics-core increase is small in absolute terms. The fused check (commit 0b52f26) gives the
same rollout numbers: +2.6% no-grad, +0.4% train.

As a GPU launch-count proxy, a torch.profiler count of aten compute ops per step went from
11475 to 11940 no-grad (+4.1%) and from 14976 to 15663 fwd+bwd (+4.6%). The lane_conflict_resolve
loop (2.7× slowdown) is not used.

**(c) Probe without retraining** (`tools/v6merge_probe.py`)
- Setup:
  - wall_same, 60 s-history set `states_h60full` on gpuW, every 2nd state (72 states, 69 with a
    blow-up verdict);
  - variant E: estimated drivers, residual on, CAVs teacher-forced, entries off;
  - the switches are toggled at inference only.
- Definitions:
  - blow-up = per-state median |ds| > 50 m at 60 s AND > 100 m at 120 s (an_walls.py rule);
  - |ds| = median over states of the per-state median HDV |ds|;
  - "stopped wait" = HDVs on lane 2 at s > 60 m and v < 1 at t0: time to their first lane
    change (SUMO 39.8 s, n=56 instances);
  - "all lane-2 wait" = every lane-2 HDV at s >= 0 at t0 (SUMO 32.5 s, n=174).
- WAIT = merge_end_wait only.

| Checkpoint | Mode | Blow-ups | \|ds\| 30/60/120 s | Mean of state medians @60 | Stopped wait | All lane-2 wait |
|---|---|---|---|---|---|---|
| v5 best | OFF | 11/69 | 12.1 / 24.0 / 68.7 | 31.6 | 3.2 s | 17.0 s |
| v5 best | **ON** | **3/69** | 11.5 / 27.6 / **54.7** | **29.4** | 11.0 s | 29.5 s |
| v5 best | WAIT | 9/69 | 11.2 / 24.6 / 68.7 | 30.0 | 16.8 s | 34.5 s |
| v6 cont (30 s stage, step 436) | OFF | 19/69 | 13.7 / 35.5 / 84.9 | 42.7 | 3.2 s | 20.5 s |
| v6 cont | **ON** | **14/69** | **12.8 / 33.3 / 60.7** | **39.8** | 14.2 s | 28.0 s |
| v6 cont | WAIT | 18/69 | 13.5 / 33.8 / 84.9 | 41.4 | 17.8 s | 30.5 s |
| v6 full stage10 | OFF | 26/69 | 51.2 / 89.8 / 83.8 | 111.1 | 3.2 s | 23.0 s |
| v6 full stage10 | ON | 24/69 | 48.2 / 84.0 / 79.8 | 111.8 | 16.5 s | 32.0 s |

Paired per state, ON − OFF |ds|:

| Checkpoint | Better / worse @60 s (share of states) | Mean diff @60 s | Mean diff @120 s |
|---|---|---|---|
| v5 best | 0.39 / 0.19 | −2.2 m | −88 m |
| v6 cont | 0.33 / 0.17 | −2.9 m | −61 m |

On v6 stage10, lane-2 overshoots past the end drop from 517 HDV-steps (10 vehicles) to 0.

Other runs:
- A local run on the older 5 s-history set (`_ref/states_h60`, v5) with the first yield version
  gave the same direction: blow-ups 10 → 8, stopped wait 2.5 → 31.2 s.
- The v6 cont checkpoint was copied mid-run (`wm_s_latest.pt`, step 436 of the 30 s stage) to
  `/root/autodl-tmp/c4/v5/diag_merge2/ckpt_v6cont/`.

Interpretation:
- Waiting moves toward SUMO without retraining. The stopped wait is still short (11 to 17 s vs
  40 s), because the learned residual and lane heads were trained with the relaxed rule.
- ON is the best mode for tails and blow-ups.
- The v5 median at 60 s rises by 3.6 m while the paired mean falls. That is a distribution
  shift of the median-of-medians, not a typical-state loss.
- Fine-tuning with the rules ON should close more of the gap.
- The constants were NOT tuned on these states. wall_same is the unseen-wall evaluation set; all
  values are the a-priori defaults.

Probe and bench outputs (summary.json and logs) are in `docs/v6merge/`:
- `*_fix_*` and `v6cont_*` use the final rule;
- `*firstyield*` use the first yield version (without the s0 offset).

The raw `results.pkl` files are on gpuW under `/root/autodl-tmp/c4/v5/diag_merge2/`.

**(d) Fine-tune config** `configs/c4_wm_v6_merge_ft.toml`
- Contents:
  - extends `c4_wm_v6_cont.toml`;
  - both switches on;
  - `model.init_from` placeholder `ckpt_v6_final`;
  - `unroll_s [60, 120]`, `stage_steps [200, 120]`, `truncation [120, 120]`;
  - `lr_per_stage [1e-4, 5e-5]`, `max_hours 3.5`;
  - `out_dir runs/active/c4_wm_v6_merge_ft`.
- `load_train_config` and `build_physics` give ON physics plus the v6 integration switches
  (tested).
- A 2-update CPU smoke (tools/c4_smoke.py on textbook data) passed: finite losses 12.79 → 8.82,
  with init_from the v5 best (117 tensors loaded, 3 new).

## For the integrator

- Set `model.init_from` at launch to the final c4_wm_v6_cont checkpoint (absolute path).
  init_from copies weights only; the physics (rules ON) comes from the config and is written to
  the new run's `physics.json`.
- Every consumer of the fine-tuned checkpoint then gets the rules from `physics.json`
  automatically: eval, probe, and the C4 policy training inside the WM.
- To evaluate an older checkpoint with the rules, edit or override its physics.
- `wms.control.belief` has its own simplified physics and is unchanged.
- No new learned parameters, so all checkpoint tensors stay compatible.
- `lane_conflict_resolve` stays OFF, as in v6_cont.
