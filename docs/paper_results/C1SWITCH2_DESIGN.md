# S-C1SWITCH2 protocol v2 — prospective design
Status: DESIGN/BUILD/CPU; owner23:48 approval observed; MAIN00:29 integration amendment applied.
Worktree E:/agent-work/c4-c1switch2, branch c4/c1switch2; inherited frozen ancestor1121117f.
Reuse v7 full-prefix reruns, anonymous history/seeding, WM/BB RPC, ownership/PI/MAPPO hand-over.
No saved-state continuation; merge S-WALLSW1b0581d0; frozen runtime/manifests remain byte-unchanged.
Question: does WM-guided picking beat fleet PPO and the best fixed controller on paired team delay?
Never select models, worlds, checkpoints or reporting scope using switching outcomes.
Library order: native, CTRL0 zero.py24095a10, speed-only MAPPO169d9042cc5, PI external-U, CIRCLES, WALL.
WALL is the S-WALLSW frozen wall-v2.1/v2.2 group law; assembly/release belongs inside the controller.
Decisions t=300,320,...,880 (30); imagine each held candidate60s, execute20s, re-observe/re-plan.
WM/BB scoring: mean of K8 predicted ON-ROAD delays; strict argmin (MAIN00:29 permitted amendment).
Exact ties retain incumbent, otherwise library order; no dwell/hysteresis/manual switching penalties.
Switch/reset/assembly costs must arise in imagined controller feedback and source/traffic dynamics.
Genuine121 half-second observable frames over60s; CAV self IDs only, anonymous HDVs before RPC AND RNG.
WM2b8797242 vs frozen BBafaaa32c; mix configured but INACTIVE with entries OFF; no retraining.
Optional WM-PHYS iff complete S-RB-EVAL beats WM2: lower CRPS at120/300/600, closer breakdown rate,
and no worse HDV position error60s; freeze qualification+checkpoint hashes before switching outcomes.
Registered common road-only objective for BOTH models: initial observable road fleet, entries OFF;
no source prediction. Primary evaluation and branch oracle remain FULL TEAM; disclose this limitation.
Fleet PPO primary and per-CAV MAPPO secondary use final frozen S-SELMAPPO/S-SELMAPPO2 checkpoints.
Same cadence/library/observable inputs/execution/memory semantics for every picker; training accounted separately.
Fresh TRAIN main969136..969151: all12 (.05,.10,.20,.40)x(1900,2400,2700) cells, then four high cells
(.20,2400),(.20,2700),(.40,2400),(.40,2700); canary969152(.40,2700), excluded from analysis.
Both frozen guards17/17 required; exact registry intervals + pruned cpuQ/gpuR/gpuA filename audit before allocation.
Disjoint from all registered experiment blocks; forbidden seed/path/truth rules apply to every audit/runtime.
All16 worlds paired across WM,BB,fleet PPO,per-CAV MAPPO,six fixed,random; optional WM-PHYS same worlds.
One shared measured index/world = random.Random(f'S-C1SWITCH2:{seed}').sample(range(30),1), fixed now.
Indices by ascending seed:10,12,22,1,6,24,10,13,28,15,5,8,26,19,16,2.
After committed episode picks, each measured state gets six exact held-controller60s fresh reruns.
Every replay starts t0, native300, prior actions held20s; compare full live prefix BEFORE intervention.
Branch cost=sum_world 6*(300+20*j+60)=60,600sim-s per picker; truth never reaches any picker.
Per-CAV MAPPO additionally branches its held joint action:10,100sim-s; signed gap to fleet-library oracle
may be negative and is not labelled nonnegative regret. Fleet arms use chosen-cost minus library minimum.
J600=full-team delay t300..900; J900=t300..1200. Hold the last action t900..1200, no extra decisions.
Every main episode costs1,200sim-s (300native+600decisions+300tail), sixteen episodes=19,200/arm.
EXACT SIM-S: WM79,800; BB79,800; fleet PPO79,800; per-CAV MAPPO89,900.
EXACT SIM-S fixed: native19,200; CTRL019,200; MAPPO19,200; PI19,200; CIRCLES19,200; WALL19,200.
EXACT SIM-S random19,200; canaries4,680; baseline TOTAL468,380.
Optional WM-PHYS79,800 => TOTAL548,180 <=600,000;51,820 unallocated, no automatic retries/refunds.
Canary4,680=2native(300+60)+6alternatives(300+60)+2scripted six-way900s repeats; native repeat gates alternatives.
EXACT GPU HARD allocations incl loading/parity/validation/idle: WM16h, BB3h, optional WM-PHYS16h.
Fleet PPO0h; per-CAV MAPPO0h; EACH fixed0h; random0h; SUMO canaries0h; total19h or35h with WM-PHYS.
Both gpuA/gpuR serve each inference arm: half its aggregate allocation/card (WM8h,BB1.5h,PHYS8h each).
Reuse frozen optimizations; require real-checkpoint/controller-feedback and cross-device sample parity first.
No claim of measured v2 throughput yet; log per-pick latency and stop at hard service caps.
cpuQ<=14 and gpuR<=4 spare nice19 single-thread SUMO workers; CUDA hidden, yield loads28/20 respectively.
Disjoint host assignment: world index mod4==3 ->gpuR, others cpuQ; per-host caps sum exactly to global caps.
GPU memory admission>=7GiB free, allocator<=2.5GiB/service; release owned services immediately after picks finish.
All evaluation physics is owned HERE; peer selector tasks supply final checkpoints and reuse our receipts,
leaving their prospective evaluation allocations unused; never evaluate/charge the same episode twice.
Primary: WM-minus-fleet-PPO and WM-minus-best-fixed paired J600,2,000 world bootstrap draws seed0,95%CIs;
reselect best fixed within each draw; negative favours WM. Complete panels only, no survivor-only claims.
Secondary: per-decision regret/signed joint gap,BB contrasts,J900,safety,switches,WALL share/observable conditions.
Report wall conditions descriptively without behavioural cutoffs; include failures, training sim-s and WM data costs.
Runner: tools/c1switch2/{episode,rerun,backend,core,rpc,runner,supervisor,canary,gate,analyze}.py.
CPU tests cover30x20s/60s lookahead,36 hand-over pairs,WALL start/stop,anonymous RNG,prefix rejection and budgets.
Main admission: bound WALL/six-way models/RL/parity/CPU/exposure/canary; mechanics canary needs CPU/WALL only.
Poll REGISTRATION with tools.c1switch2.gate; approval alone never bypasses missing scientific dependencies.
