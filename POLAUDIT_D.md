# S-POLAUDIT-D — deployment / evaluation audit before candidate A (2026-10-28)

Scope: how a trained policy runs in real SUMO and is scored vs native / MAPPO. Read-only code
review of the frozen E3-final source (`E:/agent-work/c4-e3final`, manifest b3c031ad), COOPMEAS2,
and `c4/c4prep` (candidate A), plus zero-SUMO probes on ONE recorded E3-final world (977000,
5 %/1900, all three arms; observable `episode.npz` only, copied read-only from cpuP; no
scenario/route/vType file opened). Scripts: `tools/polaudit_d/*.py`; outputs in
`_audit_data/*.json` (untracked; checkpoints/episodes are NOT committed). Zero SUMO, zero GPU.

## BLOCKER-1 — the C1 residual has (almost) no authority in SUMO; ours-vs-native is an executor artifact

Evidence (`residual_probe.py`, `residual_probe2.py`, `parity_check.py`; C1 64f70af4 replayed through
the deployment `PolicyExecutor` on the recorded E3-final states):

* C1 residual is POSITIVE on 100 % of 1,658 CAV control ticks: min +0.012, median +0.041,
  max +0.073 m/s² (radius 1.2). Coop-off twin e1fa4881 differs by <= 0.0038 m/s².
* SUMO composes `getFollowSpeed`-native + r and executes `setAcceleration` under speedMode 27
  (bit 0 = safe speed). Realised first-substep accel is <= the composed command's native part in
  >95 % of ticks (quantiles of realised − native_est: −0.67/−0.026/−0.015/−0.007/−0.000002);
  realised ≈ requested in only 1.1 % of ticks. A positive residual is clipped away by the
  safe-speed cap (`src/wms/control/nativeinit.py:95-120`, `traffic_shepherd/c4/env.py:473-474,557-562`).
* Therefore every C1-controlled CAV in SUMO is an "IDM safe-speed follower" (TraCI-influenced,
  no SUMO lane-change speed patches such as cooperative yielding), whatever the trained weights.
  Zero residual would release to true native, but C1 never outputs zero.
* Imagination does NOT model this: `bridge.py:264-275, 291` composes the learned native **head** + r
  (head ≈ true native incl. yielding) and only then applies the IDM exec cap. Imagination credits
  r on top of the native head; SUMO jumps to the IDM cap for any r > 0 and ignores r.

Consequences (inferred from 1 probed world + the 228-world E5 identity; confirm with B1.3):
(a) E3FINAL ours−native −1,093.9 veh-s (and E3LONG/E5 −970) predominantly measures the
executor switch native → capped-follow, not learned control; (b) the cycle-3 "imagination −0.834,
SUMO +8 worse" pattern is the expected symptom; (c) candidate A restarts from C1 inside the same
imagination and will again learn residuals SUMO cannot execute.

Fix before Thursday (pick one, register it, add a CPU regression):
1. Imagination parity: for residual-acting CAVs, baseline = the SUMO executor baseline
   (`exec_accel_cap`, i.e. IDM safe-speed command) and effective accel = cap + min(r, 0);
   keep the native **head** only for the native reference arm. Or
2. SUMO parity: release r >= 0 to true native (no TraCI influence), compose only r < 0.
3. Either way add a registered SUMO control arm "executor-only" (composer with r ≡ +ε) on the
   same 228 worlds; report ours vs executor-only as the learned-control contrast.
4. Pre-SUMO gate: replay the candidate on recorded states (this probe) and refuse deployment if
   the residual is non-negative on > 95 % of CAV ticks (inert policy).

## BLOCKER-2 — coop-off / ON-OFF comparisons are degenerate (coordinator lead S-NOISEFLOOR)

Not a wiring bug: E5 audit (`tools/autonomy_e5/identity_audit.py`) and my hash check confirm two
distinct checkpoints (64f70af4 vs e1fa4881, residual-head max diff 0.0017) were loaded. The
byte-identical 228 E5 trajectories and the identical coopoff_c1 / ours_c1 real dJ in all 36
macroloss groups follow mechanically from BLOCKER-1 (both residuals > 0 → both capped).
Implications: E5 "ON−OFF = 0" says nothing about cooperation; any gate/selection panel pooling
coopoff_c1 and ours_c1 counts duplicate outcomes (72 distinct of 108 in NOISEFLOOR's panel) —
deduplicate by trajectory hash before sign-agreement statistics (0.868421 gate). COOPMEAS2
ONLY/GROUP toggles switch the right vehicles (e3final `branching.py` `accel.subset(mask)`), but for
C1 they measure executor-switch interactions, not learned cooperation. Planned backup-1 (coop
residual on top of MAPPO): MAPPO commands saturate (90 % >= +0.5, mean +1.95 m/s², realised
mean +0.15; `slowdown_probe2.py`), so a residual only acts if it pulls the command below the
safe-speed cap (≈ r < −1.8 m/s²); its coop-off ablation will also tie unless designed for this.

## MAJOR findings

* M1 MAPPO arm also mostly executor-capped: 1,500/1,666 commands >= +0.5 realised +0.15 mean;
  its 59 negative commands are realised to within 0.1 m/s² (93 %/87 %). MAPPO−native mixes the
  same executor switch with genuine learned braking. Report vs the executor-only arm (B1.3).
* M2 Cycle-4 arm source mismatch: `c4/c4prep` (and `c4/int`, `main`) lack e3final's residual-aware
  `BranchController`/`split_output` (TypeError via `NativeResidualCommand.__array__`), deploy's
  `observer` and `--population` (c4prep `deploy.py` builds `CavSumoEnv` with default
  **legacy3**). `evalprep.run_e3` would fail closed; a manual `python -m wms.control.deploy` run
  would silently use the wrong HDV population. Fix: run cycle-4 on the frozen b3c031ad runtime
  (+ policy-loader diff only), or port those hunks and re-run a native+MAPPO canary that must be
  bit-identical to the 977000 E3-final metrics before adding the arm to the same worlds.
* M3 Seed hygiene gaps: every training/gate/accept guard excludes only 964000–964199
  (`policy14_cli.py:118,141`, `accept.py:check_roots`, `real_returns14.py:299`, `loop14.py:50`…);
  977000–977999 (E3-final eval) and 973000–973227 are not refused anywhere offline. Add both to all
  guards. E3FINAL outcomes are now known; freeze candidate A's config/launch SHA and show it
  predates 2026-10-28 15:18 (commit 69b4524d) or evaluate on a fresh registered block too.
* M4 Horizon authority: candidate A trusts 90 s via the inherited WM2 gate
  (`c4launch_authority.py`, WM2_GATE be0297ab), not a WM3 + entrant-ON measurement; the WM2
  gate's entrant setting is UNVERIFIED. NOISEFLOOR shows 0.868 is unreachable at <= 120 s under
  its noise proxy. Measure WM3/entrant-ON on a deduplicated pairs300 panel before relying on 90 s.

## MINOR findings

* m1 Two longitudinal writers: residual CAVs get `CommandOwnership.slowDown` (as "wall"/direct)
  at the control tick, then `execute_sumo_residual` overwrites with `setSpeed(-1)` +
  `setAcceleration` every 0.5 s (env.py:557-562). CAVs released by an ownership fallback are
  still accelerated by the residual executor while logged native. Recorded "requested/issued"
  commands are never what SUMO executed. Route through CommandOwnership or log executed values.
* m2 Envelope mismatch: SUMO upper = min(policy 1.2, vType accel); imagination compose bound
  native_max 2.6 (bridge.py:268-269). Moot while the cap binds; align with the B1 fix.
* m3 `safe_accel` applied in imagination's composition (bridge.py:275) but not on the SUMO
  residual path (policy.py:515-525 returns before safe_accel at :529). Matters for negative residuals.
* m4 `check_case` admits 964000–964199 (V5_EVAL in ALLOWED_SEEDS); population default legacy3 in
  env/collect/deploy. Assert textbook + forbidden-seed refusal inside the cycle-4 runner.
* m5 `C4_POLICY_STOCHASTIC`/`C4_POLICY_ACCEL_NOISE` are read at load time; E3-final forces 0
  (`e3final_operate.py:34`, verified). Keep and record in each result.
* m6 HARMFUL-only smoke guard = 4-world percentile bootstrap (nativeinit.py:141); with inert
  residuals its real deltas are executor effects. Fine as a harm stop, not as evidence.

## OK-verified

* Decision parity: bridge (imagination) residual == deploy `PolicyExecutor` residual,
  float32-bitwise, 120 recorded ticks / 472 CAV rows (`parity_check.py`); obs features, 100 m
  sensing, 400 m reports, RSU-zone field and command/intent zeros identical by construction.
* Arms: native = true native (all CAVs released; warmup native in every arm); MAPPO aecd7d70
  (hash-asserted, c4_dreamer/3 export, own lane rules, deterministic, safe_accel as in its SUMO
  training); ours 64f70af4 hash-bound with a SHA-matching native gate receipt.
* CRN/pairing: per-world identical initial-state SHA asserted; ours−MAPPO per world; equal-weight
  cells, shared 5,000-draw world bootstrap; treatment − reference sign; t0 = 300 s, cumulative
  120/300/600 s; road + source and mainline + ramp conservation asserted (e3long.py analyze).
* Observability: no driver types/global metrics in C1 inputs (breakdown head disabled; C1 has
  no `breakdown` payload). Lane-head policies with consequence masks use full env history
  (`policy_pairs.WmSettings`) — N/A for C1/A.
* accept.py receives `cfg["wm"]["config"]` (TOML preset path) in loop14/loop14_fast.

UNVERIFIED: gpuA/gpuR live configs (not inspected), WM2 gate entrant flag, generality of the
probe beyond world 977000 (one world; the E5 228-world identity corroborates).
