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
