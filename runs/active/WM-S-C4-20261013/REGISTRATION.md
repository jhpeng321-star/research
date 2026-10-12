# WM-S-C4-20261013 — learned multi-CAV control inside the world model (REGISTRATION DRAFT)

**STATUS: DRAFT — owner approval pending. Nothing in this file authorises training, SUMO, GPU use or any machine
start.** Prepared 2026-10-12 by the C4 ops worker (branch `c4/ops`) from `docs/WM_C4_DESIGN.md` and
`src/wms/control/contract.py`. Format follows `docs/EXPERIMENT_RECORD_TEMPLATE.md`; fields marked **[OWNER]** need a
decision, **[PENDING]** need an artifact another worker is still producing. Scope update (coordinator, 2026-10-12):
MAPPO training is out of scope for now (stub `scripts/c4/run_mappo.sh`, deferred); comparators are native, CIRCLES
damping, PI external-U, wall law (optional); the policy is evaluated alone and with an RSU world-model check in two
variants (oracle state, belief state).

### REGISTRY ENTRY — WM-S-C4-20261013
- Date registered / family / name: 2026-10-12 (draft) / WM-S, contribution C4 / "Dreamer-style learned multi-CAV
  controller, trained in WM-S, evaluated across CAV penetration".
- Scientific question / hypothesis: A single shared per-CAV policy (direct acceleration, CAV-to-CAV attention, RSU
  group signal, physics safety layer) trained by analytic-gradient imagination inside a physics-informed world model
  that has never seen a designed controller (a) lowers delay without hurting braking, safety or the worst-off HDV
  group at the penetrations it was trained on (5/10/20/40 %), (b) keeps working at penetrations it never saw
  (2/15/30/60 %) because penetration changes only the number of CAVs, and (c) improves further when the RSU checks the
  policy's plan in the world model before acting (oracle-state check first; belief-state check when the reconstructor
  exists). Null: it does not beat native/damping/PI, or beats them only at the trained penetrations.
- Status: proposed (draft, not registered).
- Record kind: campaign (three training stages + one evaluation campaign).
- Controller/actuator + source/version identity: policy = `wms.control` (this branch, commit recorded in
  `BUNDLE_MANIFEST.json` at bundle time); WM init = R5b `wm_s_best.pt` sha256
  `6f000331498a285ff7671b8fe055a211c09a566e1ffa8e51e102108933874806`; C1FAST checkpoint
  `87ed02e0707e5e4eefdd2dbb0fd282f834c1288776f7f395b41ab4d13826aaef` ships as a comparator world model only.
  Actuation through `CommandOwnership` (`wall_commands={vid: accel}`), 1 s control / 0.5 s physics / 100 m sensing /
  accel [-3, 1.5] m/s^2 / speedMode 31 (traffic_shepherd/config.py), unchanged.
- Damping baseline and car-following model: frozen CIRCLES damping (`baselines/circles`, model sha256
  `762fde07...3ccee`), HDVs IDM/MOBIL (SUMO), world model IDM/MOBIL core + gated residual (R5b recipe).
- Network/scenario; seed metadata: preserved long merge network `assets/research/highway_merge_shepherd_long`
  (2400 s episodes, demand 1800 s, ramp 400 veh/h); mainline demands 2400/2600/2700 veh/h. Seeds: section 7.
- Runtime observation contract; privileged information: actor = CAV-observable only (own state, same-lane gaps within
  100 m, other-CAV reports relayed by the RSU with age/missingness). No true penetration, density or driver type as an
  online input. Critic (training only) may use full WM state. **The oracle RSU check uses the true SUMO state and is
  labelled `oracle`; it is an upper bound, not a deployable result.** The belief check uses a reconstructor posterior
  **[PENDING: reconstructor checkpoint]**.
- Decision frame: continuation (a standing controller per CAV, decision every 1 s); the RSU check re-decides every
  few seconds (interval **[OWNER]**, default 5 s, 10-30 s imagination).
- Changed variables: controller family (learned policy, with/without RSU check), CAV penetration. Fixed: network,
  demand cells, seeds, horizon, sensing, actuation limits, HDV model, metrics.
- Comparators: native, CIRCLES damping, PI external-U (`desired_speed 31.29`), wall law v2.x (optional, `--include-wall`),
  c4_policy alone, c4_policy_rsu_check (oracle), c4_policy_rsu_check (belief, when available). All on identical
  (seed, demand, penetration) cells (paired). Reuse: none — the classic arms are rerun on the new seeds so the pairing
  is exact. Prior arms at other seeds (CV-31/CV-32, WL-04) are context only.
- Closest prior experiments and the exact unanswered difference: CV-31/CV-32 (retain/native/PI and wall arms over
  fixed controllers; wall never beat the cell's best controller), WL-04 (envelope-gated wall-v2 vs no wall, -30.2 s/veh at
  920002/m2700), OE-08 (predictors could not select beneficial walls), WM-S-R5B (world model fidelity, 1.7/4.5 m median
  error at 10/30 s). None trains a controller from scratch inside a world model, evaluates a shared per-CAV policy
  across penetrations, or checks actions in a world model before execution. Unanswered: whether such a controller
  generalises across penetration and whether the check adds value. `tools/experiment_history.py closest` was run
  2026-10-12; lexical retrieval returns CV-32, CU-01, CV-31, WM-44, WM-72, none with the same variable.
  **[Rerun `closest`/`pending` immediately before approval.]**
- Immutable registration references: registration file SHA-256 **[PENDING at approval]**; commit SHA of the ledger
  update **[PENDING]**; `eval_seeds.json` sha256 recorded in the file (`sha256_of_seeds_by_demand`).
- Paths: this file; code `scripts/c4/`, `src/wms/control/`, `traffic_shepherd/c4/` (workers' branches); configs
  `configs/c4_wm.toml`, `c4_policy.toml` **[PENDING]**; planned outputs `/root/autodl-tmp/c4/runs/<RUN_ID>/` on the
  boxes, small files fetched to `runs/active/WM-S-C4-20261013/fetched/`.

## 1. Training stages

| Stage | What | Where | Script | Output |
|---|---|---|---|---|
| a | C4 world model: continue from R5b best on native + random direct-output CAV data (`configs/c4_wm.toml`) | gpuA | `run_wm.sh preflight / smoke / run` | `wm_s_best.pt` |
| b | Policy + critic by imagination (analytic gradient through the frozen WM, 30 s rollouts, from scratch, no demonstrations) | gpuA | `run_policy.sh run` | `policy_best.pt` |
| c | Dreamer loop, 2-3 iterations: collect SUMO episodes with the current policy -> convert to WM data -> WM fine-tune -> policy fine-tune | CPU box (collect, convert), gpuA (wm, policy) | `run_loop.sh run` | per-iteration checkpoints |
| e | Final evaluation campaign | CPU box | `run_eval.sh run` | per-job outcome directories |

Stage d (MAPPO, model-free comparator) is **deferred by owner decision**; `run_mappo.sh` is a stub that refuses to run.
A model-free comparison, if later wanted, needs a fresh registration with an equal-SUMO-interaction budget.

## 2. Data for the world model (TRAIN only, no designed controllers)

Source: the leave-PI/wall-out TRAIN/VAL lists (`lpwo_v3`, 911/151 windows, `all split == TRAIN`) and the
switching-pack direct-mix list, filtered by `scripts/c4/pack_bundle.py --policies native,trajectory`
(`data_cache/c4_lists/windows_{train,val}_c4.json`, `windows_train_direct_mix_c4.json`). Dry-run counts
(2026-10-12, `pack_bundle.py --dry-run`):

| pack | windows kept / listed | runs | ticks selected | role |
|---|---|---|---|---|
| pack_v3 | 162 / 341 | 162 (true native: `chunk4`, `truearms_chunk2/3`) | 217,722 of 3,190,623 (6.8 %) | native driving |
| pack_dspeed | 520 / 520 | 520 | 624,520 of 773,444 | random sampled target-speed (direct-output) CAV trajectories |
| pack_damping | 32 / 201 | 32 (native arm only) | 38,432 of 315,863 | native driving |
| pack_switching_train | 90 / 180 | 90 (arm `direct-mix`) | 216,090 of 432,180 | random direct-output CAV actions |

Excluded by default and in the bundle: all CIRCLES-damping windows (`policy == damping`), the `switching` arm of the
switching pack (it mixes damping/PI segments; the design's open check defaults to excluded), PI and wall exposure
(already removed by LPWO v2), the evaluation worlds of the three scored state sets, every CALIBRATION-split run, and
every path containing a restricted token. The bundle ships only the array rows of the selected runs (row-range
archives, full-size sparse arrays on the box); metadata JSON still names non-TRAIN runs because indices must stay valid,
but no array bytes of those runs leave the laptop. Chunk-1..3 `__native`/`__pi` arms are per-CAV random switching
(memory note); none of them are in the selected set (native windows come from `chunk4` and `truearms_chunk2/3`).
**[OWNER confirm]** that `pack_dspeed` target-speed trajectories count as "random direct-output actions".

Stage c adds SUMO episodes driven by the learned policy. These are TRAIN data for the WM refresh and use a
**separate collection seed block that is exposure-audited and registered before collection [PENDING]**; the collect
stage refuses to start without `COLLECT_SEEDS_FILE`. Collection seeds are disjoint from `eval_seeds.json`.

## 3. Compute budget (registered ceilings — **[OWNER]** to confirm)

| Resource | Allocation | Ceiling |
|---|---|---|
| gpuA stage a | one detached run, 4 stages 10/30/60/120 s (R5b measured ~8 s/step at 60 s, ~16 s/step at 120 s, batch 32, GPU 4.5 GB at 60 s) | 8 h wall, one launch, no retry; a failed run is preserved and any restart is a new registered allocation |
| gpuA stage b | policy + critic imagination training | **[PENDING policy worker's smoke]**; proposal 6 h |
| gpuA stage c | 2-3 iterations x (WM fine-tune + policy fine-tune) | proposal 2 h + 2 h per iteration |
| CPU box stage c | collection: per iteration 4 penetrations x 3 demands x S seeds (S **[OWNER]**, proposal 10) = 120 episodes, 2400 s each | 3 iterations = 360 episodes; wall = episodes x min/episode / workers |
| CPU box stage e | 8 pen x 3 demands x 8 seeds x 6 controllers = 1152 episodes (+192 wall, optional; belief variant 192 of the 1152 wait for the reconstructor) | `eval_campaign.py plan` prints the estimate; SUMO min/episode **[PENDING sumo worker]** |

Allocations are recorded when they start (RUN_ID, start time, PID, `RUN_SUMMARY_C4.json`); each run uses a new output
directory; a failed or killed run is preserved. Stopping rules: stage a stops on non-finite loss, GPU peak above 44 GB
(R5 rule) or an explicit owner stop; evaluation jobs that time out (`--timeout`, default 7200 s) or exit non-zero are
recorded `FAILED` and not rerun inside the campaign. The campaign does not shrink the seed set after seeing outcomes.

## 4. Evaluation design

- Grid: controllers {native, damping, pi, c4_policy, c4_policy_rsu_check_oracle, c4_policy_rsu_check_belief, (wall)}
  x penetration {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60} x demand {2400, 2600, 2700} x 8 seeds per demand.
- Trained penetrations 0.05/0.10/0.20/0.40; **unseen 0.02/0.15/0.30/0.60** (never in WM or policy data).
- Paired: every controller and penetration uses the same (seed, demand); analysis is on within-cell differences.
- Metrics (per episode): vehicle delay including ramp/source waiting (all requested service retained), throughput,
  hard-braking count and acceleration effort (sum a^2), TTC < 3 s vehicle-seconds, collisions, teleports, and fairness:
  p90 and p95 per-vehicle delay, worst-group mean delay (ramp vs mainline; vehicles behind CAV formations), plus for the
  RSU check the fraction of decisions where the check overrode the policy. Coordination evidence: CAV formation metrics
  (side-by-side across lanes, synchronised pre-merge slowdowns) vs the hand-designed wall.
- WM honesty: WM-predicted vs SUMO-measured gain for every evaluated controller and horizon.
- Ablations (separately registered if run): no CAV-to-CAV attention, no group signal.

## 5. Original criteria (draft, to be frozen at approval; never weakened later)

1. Primary: paired difference in mean vehicle delay (treatment - reference, **negative = improvement**), c4_policy
   minus native, over the 24 (seed, demand) cells of each penetration, with a 95 % paired bootstrap interval; reported
   separately for trained and unseen penetrations. Supported only if the interval lies below 0 at every trained
   penetration and at the majority of unseen penetrations.
2. Against the best classic arm (min of damping, PI per cell): reported, no threshold; the penetration-agnostic claim
   requires c4_policy not worse than the best classic arm at the unseen penetrations.
3. Safety (constraints, not trade-offs): collisions and teleports not above native; TTC<3 s exposure, hard-brake count
   and p95 per-vehicle delay intervals not significantly worse than native at any penetration. A violation is reported
   as a failure of that penetration regardless of delay.
4. RSU check: c4_policy_rsu_check minus c4_policy on the same cells; oracle result labelled an upper bound.
**[OWNER]** confirm or edit these before freezing; no effect-size threshold is invented here.

## 6. Safety constraints and refusal rules
- Physics safety layer in every execution (imagination and SUMO); acceleration bounds [-3, 1.5]; speedMode 31.
- Training gradients never use a designed controller as a target or bonus; no wall action in the policy.
- A job that collides is not retried or removed; it counts.

## 7. Seeds and exposure
- Evaluation seeds: `eval_seeds.json` from `scripts/c4/make_eval_seeds.py` (rule printed in the file): the
  exposed/development partitions of `traffic_shepherd/seeds.py`, accepted only through
  `require_development_case` (the gate `traffic_shepherd.workflow.evaluate` applies), never SEALED_SEEDS, the sealed
  chunk-5 or TEST_REPL blocks or FINAL_TEST, minus every seed that appears as a world in a WM TRAIN pack (330 seeds).
  Result 2026-10-12: demand 2400 -> WM28 seeds 920050,059,065,070,080,094,103,109; demand 2600 -> OE65 920008-920015;
  demand 2700 -> OE65 920016-920023. These are **development** seeds that earlier studies already exposed; they carry no
  freshness claim. Registry absence is not proof of freshness (AGENTS.md); I did not read sealed material.
- A final, claim-bearing evaluation needs a new audited block registered with the owner's approval **[PENDING]**
  (same procedure as WM-34/WM-41); this campaign is development evidence.
- Collection seeds: section 2, **[PENDING]**.

## 8. Integrity limitations known at drafting
- The pack arrays are shipped as row ranges; correctness relies on `unpack_bundle.py` (sha256 of every row block,
  shape/dtype/header checks) and was tested on synthetic packs only. The first real use is `box_setup.sh` followed by
  `run_wm.sh preflight` and `smoke` on the real pack (box smoke must pass before the full run).
- Interfaces between scripts and the not-yet-written modules (`wms.control.train_cli`, `traffic_shepherd.c4.collect`,
  `traffic_shepherd.c4.to_pack`, `traffic_shepherd.c4.evaluate`, `configs/c4_*.toml`) are assumptions listed in
  OPS_NOTES.md; they must be reconciled before the first launch.
- The oracle RSU check cannot be reproduced online (it needs the true full state).

## 9. What must NOT be inferred
Nothing here is a result. Development seeds are not acceptance seeds. A good unseen-penetration result does not show the
policy is deployable without the reconstructor, and the oracle-check result is an upper bound only.
