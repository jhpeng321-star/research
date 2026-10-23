# R10B_LAUNCH - round 10b launch record (S-R10BLAUNCH), 2026-10-23

## Registration (C1 of R10B_REVIEW.md; registered BEFORE exec)
Run: round 10b = round 10 (fix B) + learned pre-merge follower courtesy head. Commit 6a6f2afd (branch c4/v10r10b).
Init: /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round9/wm_s_stage300.pt. Config unchanged: weight 4.719, lr x1000, rollout_grad=false, hidden 32, Huber 1.0.
Run dir: /root/autodl-tmp/c4/v5/runs/c4_wm_v10_round10b (smoke: runs/smoke_r10b). Host: gpuR, sharing the GPU with round 10 (PID 354035, untouched).
Morning scoring (C2): unseen walls/PI per protocol, never sealed data; three arms: round 9 (re-scored with current code), 10b, 10b with the head zeroed (follower_courtesy.net.4.* = 0, exact ablation).
Reject criteria (from R10B_REVIEW.md, fixed before launch):

| id | check | reference | reject the head (zero it) if |
|---|---|---|---|
| M1 | rollout event profile of target-follower excess over [-6, +2] s | SUMO: onset -0.14 at -6 s, peak -0.45 at -1.5 s, ~0 at the merge | mean over [-3, 0) s < -0.9 (2x the SUMO peak) |
| M2 | R9_YIELD 5.3 removal test | near pairs 3-25 m: dv(10 s) of the SUMO order (about -1 m/s); far placebo >= 60 m: about 0 | far placebo dv(10 s) < -0.5, or its CI excludes 0 in the braking direction |
| M3 | negative controls, one-step and rollout: dA>30 m, d_end>40 m, alongside (bumper gap < 1 m), ramp car behind (must be exactly 0), lane 0 | recorded rows in the same groups | \|model - recorded\| > 0.3 m/s^2 |
| M4 | head output on rollout states: p0.1 / p99.9, share < -3 m/s^2; head_min/max telemetry per step | - | head is regularly < -3 m/s^2 on any group |
| M5 | safety and stuck cars vs round 9 and SUMO: collisions, TTC<1.5 s count, min gap at insertion, hard-brake (< -3) count, lane-1 jerk at partner switches, braking waves; stuck ramp cars, merge count, wait distribution | round 9 and SUMO | collisions or TTC worse than round 9: reject the round, even if the means improve |
| M6 | fix-B reads (R10_REVIEW R3): post-insertion min-accel p10, onset-row bias split by outcome, tight-gap non-insertion braking p10 | as in R10_REVIEW | as in R10_REVIEW |
| M7 | 90 steps completed; courtesy loss and head telemetry trends | - | - |

M6 is read on both the 10b and the head-zeroed arms.

## Ship
Repo dir: /root/autodl-tmp/c4/v5/repo_r10b_6a6f2afd = cp -a of repo_r10_d19c2066 (commit d19c2066, an ancestor of 6a6f2afd) plus the 25 files changed d19c2066..6a6f2afd, taken as LF git blobs (no CR; gate .sh md5 bef04899188b40503097ffeb8e22408f matches local).
Verification: all 7752 tracked files at HEAD checked on gpuR by git blob hash: 520 exact (includes the 25 shipped files), 7227 identical after CR removal (the round-10 base copy has CRLF text files, harmless for python), 5 absent (R8_A/B/C.md, tests/test_merge_core_v2_onpolicy.py, tests/test_merge_core_v2_seqtf.py; not used by the gate). No file differs in content.
Disk after copy: 907 MB free (run dirs ~15-25 MB each).

## Smoke (runs/smoke_r10b, log smoke_r10b.log, verdict smoke_r10b.verdict = "smoke: ok", report smoke_r10b.gate.json)
Launched 18:17 (CST) with nohup/setsid bash scripts/round10b_gate.sh (PID 400794, same PID continues as the full run via exec). Total smoke wall time about 63 min while sharing the GPU with round 10 (stage120 ended 18:44, stage300 and checks 19:20).
Gate metrics: 4 optimizer steps, cuda_peak_mb 3308 (gate metric; per-step log shows 1704 MB at step 1), courtesy rows 4298, cut-in events 88 (454 event rows), TF zero-mass events 0, scored seconds 3237 (120 s) / 1565 (300 s), shared replay 4, courtesy loss last 0.887, cut-in loss last 1.076. All 3 checkpoints reload strictly, head read-out moved, nothing changed outside the trainable set.
"changed_tensors: 6" in the gate report counts only keys that exist in the round-9 init (temporal.0/2 weight+bias = 4, gate_g_max_logit, gate_tau_raw); the 6 courtesy head tensors are new keys, all nonzero and all moved between stage120 and stage300. Total trainable = 12 tensors / 7911 parameters.

## Full run (runs/c4_wm_v10_round10b, log gate_r10b.out)
Started about 19:21 (CST), PID 400794, GPU shared with round 10 (PID 354035, untouched). Memory on the GPU about 12.0 GB total used with both jobs (round 10 alone about 7.4 GB).
Step 1: loss finite (231.7), courtesy loss 1.019 on 844 rows (889 partner rows), head output exactly 0 (zero-init no-op, base residual mean -0.173), cut-in events 26 (124 event rows, 16337 control rows), TF zero mass 0, merge_pivot_on_objective_disabled 1, merge_pivot_grad_norm 0, grad_norm 6.17 (clipped), lr 0.05.
Step 2: courtesy 0.910 on 1236 rows, head_min -0.377. Step time at stage 120 (batch 4, shared GPU): 124.5 / 267.7 / 108.9 s for steps 1-3 (round 10 stage-300 median 209 s for reference).
