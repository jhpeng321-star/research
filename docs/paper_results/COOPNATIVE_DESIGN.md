# S-COOPNATIVE — prospective native-to-cooperation learner, 2026-11-01
Status: BUILD/CPU/QUOTE ONLY; MAIN approval required before any SUMO, including timing.
Goal: learn road-wide multi-CAV control from native; MAPPO169 d9042cc5 is comparator only.
Prior: c4-nativeinit/NATIVEINIT.md (10-26 WM residual pilot; native zero must abstain).
Prior: c4-coopfree COOPFREE.md/COOPFREE_DESIGN.md (tracking, broadcast, correlated search).
Prior: c4-coopplan COOPPLAN.md (exact benefit/walls at H300, not a move library).
Prior: c4-coopbandit COOPBANDIT.md (helpful options only45/300 states; avoid prescribed moves).
Algorithm: fresh shared observable actor + centralised observable critic; on-policy clipped PPO.
No MAPPO weights/teacher/fallback; no exact-rerun training credit; exact twins evaluation only.
24 independent full episodes/update, 144updates; 4PPO epochs, minibatch128 joint decision frames.
Adam3e-4, clip.2, value coefficient.5/value clip.2, gradient cap.5, joint KL stop.045.
Actions: ALL live CAVs including ramp/upstream/downstream, held W=10s; entrants native until next W.
Two Gaussian residual coordinates: delta speed and delta log-time-gap, initial mean exactly0.
Anchor is decision-time sensed speed and sensed headway (1s when no observed leader).
Exact pair(0,0) abstains for WHOLE W: native driving, not a zero-acceleration/speed-hold command.
Nonzero pair selects fixed absolute speed=max(0,v+delta), tau=exp(logtau+delta); no move thresholds.
This abstention mapping is discontinuous at exact0; stochastic exploration is not native parity.
Reuse COOPFREE requests/tracking5s; optional zone=None extends roster only, default unchanged.
Original safe_accel (cap1.5/floor-3), env.step(lanes=None,fallback=native), CommandOwnership unchanged.
Native zero bypasses external actuation AFTER safety; no direct TraCI actuator writes in learner.
Actors: own kinematics/lane + nearest front/back per lane within100m + shared4D RSU latent.
RSU zone[-100,75.44], per-lane observed count/mean speed/position/CAV count; zero-init broadcast.
Critic: pooled road-wide CAV-local features + RSU zones; no privileged state/queue/driver labels.
Strict whitelist; no world/seed/demand/true-share/driver-type channels; IDs only identity/tie ordering.
Joint exploration sqrt(.5)*shared+sqrt(.5)*individual Gaussian; sigma4.5m/s/.5log-s, learned scale.
PPO scores exact correlated JOINT density, no independent-marginal pseudo likelihood.
Team reward=-J increment: road speed-deficit PLUS source waiting, identical comparator reducer.
No behavioural, cooperation, throughput, lane or safety reward bonuses; common reward unit10000veh-s.
900s controlled task +300s native warmup; critic terminal0 at task end; plant horizon901 matches cache.
gamma=exp(-10/600), lambda=exp(-10/1200): GAE e-folding400s, credits>=300s walls.
TRAIN fresh986000–986071: six worlds/all12share×demand cells (.05/.1/.2/.4×1900/2400/2700).
Rotate two repeats/cell/update; fresh on-policy noise/physics under current checkpoint each update.
Independent TRAIN-cell checks986072–986083, one/cell, never optimizer/selection input.
Disjoint registry-range + safe-name exposure audits on admitted hosts and BOTH seed guards required.
Never964000–964199/962xxx/977xxx; prohibited paths/truth/assignments never opened.
Every8updates: paired check J600/J900 vs native AND comparator-only MAPPO, safety + behaviour stats.
Before training collect12native+12MAPPO check controls once; all physics charged including warmup.
Behaviours: speed/loggap variance, group slow-down signs, unchanged COOPBEHAV lane-hold detectors.
Dev checkpointsu8/final fixed in advance; no efficacy-based checkpoint selection or hyperparameter tuning.
Dev EXACT S-DISTILL/S-WMPLAN984000–984023 panelSHA72776ff1, two worlds/all12cells, high-share first.
Reuse all24 bitwise MAPPO baseline episode bytes/hashes; add native and actual frozen CTRL0 arms.
Report paired J600/J900,10000 world-bootstrap95%CIs/RNG20261101, collisions/TTC<3/hard braking.
End: detector-first chronological group behaviours, complete native-release/joint/every-member-solo twins.
Replay identical full prefix; hold identical group targets; score C_G at120/300/600s, report coverage.
No positive-only discovery selection; reserve whole groups before physics; no twins used for gradients.
Fairness: production2 RUN_SUMMARY SHA8b25ee1e records169×46×600=4664400controlled env steps/sim-s.
MAPPO warmup7774×300 gives6996600total sim-s; horizon600, GAE.99/.95, fourPPO epochs.
MAPPO reward=team_reward (delay/throughput/smooth/brake/fairness weights1/.5/.05/1/1), cost dual0 at169.
MAPPO169 inherited actor aecd7d70 from earlier iteration170; earlier-training cost not yet included.
Our fresh-native primary uses STRICTER4664400 TOTAL sim-s ceiling; publish curve vs total+train sim-s.
Total proposed4664400: TRAIN4147200 +checks288000 +DEV115200 +discovery96000 +canary1200 +unallocated16800.
Checks=18×12×1200 learned+24×1200controls; DEV=2×24×1200learned+48×1200native/CTRL0; MAPPO reuse0.
Every attempt/warmup/failure/full prefix precharged; no refund/retry/deletion; incomplete charges STOP resume.
Immutable full policy/critic/Adam/RNG/update/counters checkpoints; exact optimizer continuation CPU-tested.
CPU shared priority: WMPLAN first, C1SWITCH2 then COOPFREE; initial cpuQ<=2 nice19 single-thread workers.
Optional gpuA/gpuR<=1CPU worker each only after priority/headroom admission and frozen parity tests.
Yield between waves at load>12(cpuQ)/16(other), memory<8GiB or disk<2GiB; no peer PID operations.
Quote uses existing one-episode CPU timing + new CPU learner overhead; no unapproved physics timing.
Register source/design/CPU/audits/budget before SUMO; MAIN must append S-COOPNATIVE BUDGET APPROVED.
After approval bind immutable runtime/actual CTRL0/MAPPO-cache hashes, verify zero physical canary before learning.
Publish every learning check to COOPNATIVE.md; own-PID observer only; never auto-clear durable STOP.
