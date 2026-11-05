Plain summary — S-LANEMAPPO, 2026-11-06:
1. Recommendation: NO to swapping the planned clean3 base now; YES, a separately versioned lane-on successor is technically feasible.
2. Found and independently SHA-verified aecd7d70 locally and in cpuV's cpuq_final copy; it is a trained CavPolicy with acceleration and three-way lane heads.
3. Its SUMO E3FINAL_A delay benefit versus native is 1,777.4 veh-s at 600 s; versus CTRL0 on 23 reused worlds the 623.95 veh-s benefit is inconclusive.
4. There is no verified matched lane-on versus retrained speed-only comparison here; PANEL120 and CTRLARM do not establish lane-on superiority over speed-only.
5. Clean3 has TWO models: an external frozen CavPolicy MAPPO and a separate flat-feature residual/V2V model; do not replace the latter's u0 with aecd7d70.
6. Keep residual V2V width 48 and its intent/cooperation heads; replace the external executor through a new lane-aware adapter, safety path and replay contract.
7. Existing native-lane guards, twice-called safety interception, old bank/prefix bindings and G1/G4/G5 semantics require work; estimate 18–30 engineering hours plus separately budgeted qualification/certification.
8. A/B/R must share the lane-on base and executor; the paper's registered primary comparator remains speed-only MAPPO unless the owner explicitly changes the scientific question.
9. Best immediate use: preserve aecd7d70 as an additional frozen evaluation baseline; this check ran no SUMO, training, optimizer steps or charges.

# Lane-on MAPPO feasibility: clean2/clean3

## 1. Scope, evidence and decision

**Analysis-only record S-LANEMAPPO.** The requested report is the only research artifact written. Local source inspection used `E:/research-dense` and `E:/agent-work`; remote research reads were confined to permitted files under **cpuV `/home/ubuntu/cpuq_final`**. No cpuQ connection was needed. Remote commands used the existing WSL `tools.coopmwm.cpuv_ssh.Transport` shared multiplex master, bounded 90–100 s operations, no retries, and reads spaced more than 60 s apart. Python ran with bytecode disabled; remote checkpoint inspection was CPU-only/one numerical thread. No simulator, training, accounting dispatcher, process-control operation or hidden parameter/truth reader was run. No prohibited-path or hidden truth contents were returned or used. An initial registration grep unexpectedly expanded to neighboring files; subsequent registration searches were filename-filtered. Directory-name inventories were not content reads of excluded files.

Local compatibility source was `E:/agent-work/c4-rerun-launch`, observed HEAD **`e8bd3c8fcd396742fde8b170305aa9d25235c913`**. Existing dirty/untracked preparation documents were preserved. G-module receipts were read from their separate worktrees; readiness is not deployment or physical qualification.

**Decision: do not substitute lane-on MAPPO into the currently prepared clean3.** This would change the background controller, lateral-action regime, certified lessons and counterfactual reference at once, while clean3 is already integrating G levers. It is not an optimizer-resume/checkpoint-path change. A new lane-on lineage is feasible and potentially useful, but should begin from a separately bound common base with refreshed teaching/certification and a matched comparator. Nothing in this report claims that lane-on initialization will produce positive group-over-best-solo synergy.

## 2. Checkpoint identity, architecture and training

### 2.1 Independently verified files

- Local: `E:/research-dense/runs/active/C4-HEADROOM-20261017/mappo_v5_final.pt`.
- cpuV archive copy: `/home/ubuntu/cpuq_final/e3final/inputs/mappo_v5_final.pt`.
- Both are **454,236 bytes**, SHA256 **`aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564`**. Both byte hashes were recomputed in this check.
- Payload: `format="c4_dreamer/3"`, `trainer="mappo"`, `lane_head=true`, `iteration=170`; contains actor, group encoder, lane rules, bounds, reward weights, resolved config and budget. This is a deployment export, **not** a full trainer/critic/optimizer/RNG checkpoint. Exact original PPO resume state was not recovered; frozen-base use does not need it.
- Registration corroboration: `REGISTRATION.md:1592,2375–2377,2815,3494,4936`; the HEADROOM README identifies the historical gpuW relay source. Historical source locations are provenance, not a claim those hosts remain available.

### 2.2 Architecture verified against the tensors and loader

This is **not one flat observation vector of dimension 29 or 77**:

- Own observation: **13** features, `[B,K,13]`; own MLP `13 → 64 → 64`, SiLU.
- Relayed CAV report: **13** features/report, up to **16** masked reports/CAV; report MLP `13 → 64`, then **64-wide, four-head attention**. Report features already include the preceding commanded acceleration and lane intent; these are not clean3's five-second learned intent messages.
- Learned RSU group signal: **8** features/CAV, produced by `RsuGroupEncoder`; CAV input weight `[64,14]`, sensor-cell input `[64,10]` (six sensor features plus geometry), set/field attention and group output `[8,64]`. Actor/group-encoder constructors use four attention heads; these counts are source architecture, not separately encoded in the export.
- Acceleration head: concatenated own64 + report-context64 + group8 = **136**; weights `head.0.weight=[64,136]`, `head.2.weight=[2,64]`. The two outputs are latent mean and log-standard-deviation, **not acceleration and lane**. Tanh-Gaussian decoding has stored bounds **[-3,2.6] m/s²**.
- Lane head: same136 + feasible-choice mask3 = **139**; `lane_head.0.weight=[64,139]`, `lane_head.0.bias=[64]`, `lane_head.2.weight=[3,64]`, `lane_head.2.bias=[3]`. Masked categorical **stay/left/right**; deterministic deployment uses masked argmax.
- `PolicyExecutor` maintains command memory and lane dwell, builds observations/group signals, chooses feasible lanes and applies acceleration safety for **both current and requested target lane** before recording the command/intent.

Sources: `src/wms/control/{contract,policy}.py`, especially `policy.py:80–100,159–213,239–273,420–468`. The local C4 component loader successfully loaded the actual checkpoint and reported lane head ON and bounds [-3,2.6].

### 2.3 Actual resolved training config, not documentation defaults

From the export itself:

- 170 completed iterations ×46 episodes = **7,820 episodes**; seed1, exposed world seeds **4107–4148**; native warmup300 s + controlled600 s.
- Discrete penetrations {.05,.1,.2,.4}; demands {2400,2600,2700} vph. `charge_warmup=false`.
- Recorded budget: **4,692,000 controlled seconds / env steps**, **7,038,000 simulated seconds including warmup**, zero wasted simulation seconds. Config ceiling100,000,000 is a ceiling, not expenditure.
- PPO gamma .99, GAE lambda .95, actor/value clipping .2, four epochs, minibatch1,024 frames, actor/encoder LR3e-4, critic LR1e-3, acceleration/lane entropy .01/.01, value coefficient .5, gradient cap .5, target KL .03, return normalization ON; update threads44, gradient-noise measurement every5 iterations.
- Actual `cost_limit_per_step=3.6`, Lagrange LR .01/max50. Do not substitute the older .01 limit described in generic MAPPO notes.
- Reward weights: delay1, throughput.5, smooth.05, hard-brake1, fairness1; TTC limit3 s. Lane rules: front gap base1 m/time1.2 s/closing4.5 s; rear base3 m/time1.25 s/closing6 s; dwell3 s; edge guard10 m.
- MAPPO used joint acceleration+lane likelihood, a centralized critic during training and the same C4 observation/execution machinery (`MAPPO_NOTES.md`). No privileged critic is needed for frozen deployment.

The evaluated lane regime matters independently of tensor shapes. `traffic_shepherd/c4/env.py:37–52,385–429,513–518` applies **laneChangeMode513** to every finite-command CAV whenever a lane tensor is supplied, even if that tensor contains only -1/no-change targets. Strategic/route merges remain enabled; discretionary cooperative/speed-gain/keep-right changes are disabled. TraCI requests retain SUMO gap/braking safety, and junction/edge/ramp-entry guards still apply. Thus “no lane request this tick” is not generally equivalent to `lanes=None`/native lanes.

## 3. Existing real-SUMO outcomes and comparison limits

All delay contrasts below are **treatment minus reference**, in veh-s; negative is better. Bracketed intervals are the reports' 95% bootstrap intervals. These are recorded physical outcomes, not WM predictions. Different panels must not be subtracted to invent a lane-on/speed-only treatment effect.

### 3.1 E3FINAL_A: lane-on versus native and historical ours(C1)

`E:/research-dense/docs/paper_results/E3FINAL_A.md:3–35,76–125`, matching CSV alongside it:

- 228 paired TRAIN/dev worlds, seeds977000–977227; 12 share×demand cells,19 worlds/cell; native300 s warmup +600 s continuous intervention. Fixed MAPPO **aecd7d70** and ours(C1) **64f70af4**. Equal-cell means;5,000 paired world-cluster draws within cells, seed20261027.
- MAPPO−native total delay: **120 s -54.6 [-88.1,-21.0]**; **300 s -437.8 [-632.0,-241.4]**; **600 s -1,777.4 [-2,336.8,-1,231.3]**.
- At600 s, native/MAPPO mean delay **56,437.7 /54,660.3**; throughput difference **+37.3 [+25.7,+49.2] vph**.
- Historical ours(C1)−MAPPO primary600 s contrast **+683.5 [+256.9,+1,122.4]**: lane-on MAPPO was better on that comparison. Ours(C1) is not clean2/clean3 and this is not a speed-only MAPPO result.
- MAPPO−native hard-braking exposure **-2.7 [-3.7,-1.6]**, TTC<3 exposure **-37.2 [-53.4,-21.0]**; zero observed collisions/teleports/dropped steps in all arms. Observed zeros do not prove safety equivalence.

### 3.2 PANEL120: separate short-horizon descriptive evidence

`E:/research-dense/docs/paper_results/PANEL120.md:9–18,54–84`:

- 60 paired fresh worlds977228–977287, five/cell; deterministic native and **lane-on aecd7d70**, same frozen E3 runtime;300+120 s only.
- MAPPO−native at120 s **-21.6 [-98.1,+52.5]**, SD300.7, median+14.8; native mean about7,468. Report labels this descriptive provenance, not a registered efficacy verdict.
- These records were reserved for WM horizon verdict scoring, not fitting/selection. There is no300/600 s PANEL120 result to quote.
- Original records are documented at cpuQ `/home/ubuntu/panel120/run1/jobs`; observable package at `/home/ubuntu/panel120/package` and gpuA. Only the permitted local report was read for this check.

### 3.3 CTRLARM: lane-on versus the execution-switch control

`E:/research-dense/docs/paper_results/CTRLARM.md:5–23`, corresponding CSV and world CSV:

- 23 reused E3 worlds977000–977022,300+600 s; equal-world means,2,000 paired bootstrap draws/seed0; exploratory.
- CTRL0 is **forced-on exact-zero residual/base-law longitudinal execution, speedMode27, native lanes**. It is neither native release nor lane-on MAPPO with a zero cooperation layer.
- **MAPPO−CTRL0 -623.95 [-2,044.25,+664.51]**; **CTRL0−native -1,956.53 [-4,086.66,+144.50]**; **ours(C1)−CTRL0 -45.71 [-464.12,+342.63]**. Every delay interval includes zero.
- MAPPO−CTRL0 hard-braking exposure **-3.07 [-5.61,-.41]**, TTC<3 exposure **-21.39 [-70.82,+23.74]**. No observed collisions/teleports/dropped steps in any arm.

### 3.4 The retrained speed-only comparator is different

Registration `:4936–4941` records the **Oct30 owner decision that the primary MAPPO comparator must be speed-only**, correcting the earlier lane-controlling setup. `:4970` records initialization from aecd7d70 with its lane head stripped, followed by additional SUMO training. This is not merely the old actor with a deployment lane toggle.

- Independently hashed cpuV copies `/home/ubuntu/cpuq_final/mappospeed/production2/{policy_00169.pt,policy_latest.pt}` are both416,744 bytes, SHA256 **`d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca`**.
- Payload confirmed iteration169, `lane_head=false`, `lane_control="native"`, same own/report13-feature attention architecture and acceleration head, no lane-head tensors. Its additional training records **4,664,400 controlled /6,996,600 simulated seconds**;169×46 episodes, same training-world block and demand/share choices. Export counters exclude inherited training and operational failure charges.
- Existing SPEEDHEAD real-SUMO summary, `REGISTRATION.md:5298,5434`: eight paired TRAIN worlds963880–963887,300+600 s; speed-only MAPPO−CTRL0 **300 s approximately -543 [-2,033,+887]**, **600 s +33 [-3,920,+3,869]**; no collisions. Both signs occur at large magnitude. The advertised SPEEDHEAD report was not present in the inspected local report directories, so these rounded numbers are attributed to the registration summary, not a newly recomputed analysis.
- **No verified same-world, same-executor lane-on versus retrained speed-only contrast was recovered.** E3's significant benefit versus native and SPEEDHEAD's uncertain benefit versus CTRL0 cannot establish lane-on superiority over speed-only. A native-lane proxy of aecd7d70 also is not the d9042cc5 trained comparator. No new outcome comparison was run.

## 4. What clean2/clean3 actually loads: two distinct checkpoint roles

This distinction materially reduces the required migration relative to replacing every network:

1. **External physical MAPPO base:** production config `checkpoint` points to speed-only `policy_00169.pt`/d9042cc5; frozen C4 `PolicyExecutor` supplies the underlying absolute acceleration. `tools/leverprobe/backend.py` is runtime-supplied, absent from the inspected local checkout. Its archived source was read at cpuV `/home/ubuntu/cpuq_final/leverprobe20261031_d36d557c/pytools/tools/leverprobe/backend.py`: `MAPPO_SHA=d9042cc5`, `setup:29–30` enforces it, constructor `:108–110` requires `native_lanes` and disallows residual-on-native; `advance:154–179` rejects lane output/counters.
2. **Trainable correction model, also called common MAPPO u0 in rerun documents:** `initial_checkpoint`/`A|B|R_checkpoint_000.pt` is a separate `tools.coopmwm.acceleration.Policy` checkpoint. Archived coop3 A/B u0 files are identical107,114-byte files, SHA256 **`5ea2296470c2af2855f9e292b03eeaa62bafdfec08146712d7347d7ce117d734`**; update0 payload has `policy,optimizer,generator,torch_rng,numpy_rng,python_rng,charged,extra`. Its flat sensor29 + broadcast4 feed `actor.0.weight=[64,33]`; action output `[1,64]`; critic first input `[128,41]`; scalar learned `log_std`. It is not the CavPolicy export itself.

`accel_plant.py:34–43,87` loads the external physical checkpoint and correction weights separately. `acceleration.requests` adds the correction to the physical base's current absolute command. `track2_backend._load_base:492–497` loads the correction checkpoint; A trains that correction with PPO, while B/R freeze copies and train their cooperation layer/trigger. “A plain MAPPO PPO” in the clean-run docs therefore does not mean the external CavPolicy is being fine-tuned.

**Actual CPU-only load probes in this check:** the C4 loader accepts aecd7d70; current `v2v_model.load_checkpoint(aecd7d70,v2v_dim=48)` raises **`KeyError: 'actor.0.weight'`**; `extend_for_v2v(CavPolicy,48)` raises **`AttributeError: 'CavPolicy' object has no attribute 'actor'`**. Neither probe sampled a simulator or performed an optimizer step. These are confirmed direct-swap failures, not hypothetical architectural concerns.

## 5. Compatibility and required changes

### 5.1 V2V widening and predictive intent: retain the existing correction model

- `v2v_model.py:34–38,97–125,206–218,275–299` inserts48 zero columns **after sensor column29**, before the actor's4-dimensional broadcast tail and critic's12-dimensional RSU tail. Actor first weight becomes **[64,81]**, critic **[128,89]**. `V2VInputLinear` scales only the48 partner channels; optimizer moments/fresh-column steps are remapped.
- `v2v_intent.py:89–147,241–274` adds a zero-initialized **[5,64] intent weight** (bias5), observing four partners ×(six telemetry fields +five future commands +validity) = **48**. Publication is t-1 causal; detached-trunk training is the registered default. B/R base/intent freezing must remain enforced.
- **Minimal route:** retain these tensors, observation builders, likelihoods and optimizer schema exactly; substitute a lane-aware external CavPolicy executor underneath. No widening of CavPolicy is necessary merely to keep clean3's existing V2V layer. Recompute intent labels from the new composed absolute commands and rebuild its history on each replay.
- A full migration of all V2V/intent learning into CavPolicy's attention architecture would need new observation/optimizer/head adapters and would be a different, larger method. The old lane head cannot be copied into the flat residual trunk just because both hidden widths are64: its first input is139, and its features/mask semantics differ.

### 5.2 Frozen-base cooperation and likelihoods

Keep the external actor **and group encoder** immutable and deterministic. Preserve the existing correction model as the interface to `ProductionLayer`/`TriggerPolicy`, rather than pretending CavPolicy implements `forward(context) -> mean,value`, `.sample`, `.score`, scalar `.log_std` and the shared AR(1) Gaussian.

`coop_layer.py:65–111,161–213` copies/freezes its base, excludes all `base.*` parameters from the layer optimizer and enforces eval/no-gradient state; `track2_backend.ProductionLayer:152–179` adds full-strength library moves plus zero-initialized learned correction. Retain those protections for the correction model **and** independently hash/check the external CavPolicy/group encoder. Define identity as **no move +zero learned correction reproduces the original lane-on requested/issued acceleration, lane targets, memory and mode state**. F1 intentionally executes selected moves from update0; a zero neural residual alone does not turn those moves off.

If only the external base has frozen lane actions, the correction's PPO likelihood can remain acceleration-only: its trainable random action remains the Gaussian correction. This route inherits the old lane decisions; it does not fine-tune the old lane head. G5 can add separately owned learned/selected lane macros. If A or B/R starts learning an additional stochastic lane policy, joint categorical likelihood/KL/entropy and recorded feasible masks are required; that is an additional design, not implied by choosing a lane-on base.

### 5.3 Safety injection and single writer: substantive production work

Three concrete failures occur before meaningful lane-on collection:

- The runtime backend rejects the different checkpoint SHA and the absence of native lanes. Replace this with an explicit versioned base-family/mode binding and exact SHA check, retaining the old speed-only mode unchanged.
- `tools/coopfree/backend.py:38–40,59,69–71` rejects non-None lane output, sends `lanes=None`, and treats any lane counter as an anomaly. It must pass the lane-on executor's targets through `env.step` and distinguish authorized requests/refusals from genuine anomalies.
- **Do not simply remove the one-call assertion:** `tools/coopfree/execution.py:47–83` intercepts `safety.safe_accel` and calls `controller.apply` once per safety invocation. A changing-lane `PolicyExecutor` calls safety once for the current lane and again for the target lane (`policy.py:455–460`). The present interceptor would sample/update the controller twice and fail `expected one native-lane safety boundary`. Refactor composition to happen **once per decision, before both safety projections**, with one recorded correction/action, then use the same raw composed command for current/target-lane safety. Command/intent memory must record the resulting composed safe command.

Retain CommandOwnership, speedMode27, existing physical caps and SUMO lane-request safety. The checkpoint's decoder ceiling2.6 does not raise the clean move interface's registered **1.2 m/s²** feasibility ceiling (`moves.py:67,201`); lane-on initialization does not itself eliminate longitudinal saturation/dead zones. Preserve mode513 on all lane-on policy ticks, including stay-only ticks; restore previous modes on handback. Include entrants, junctions, rejected requests, lane ends and slot reuse in parity checks. A longitudinal correction that appears helpful before target-lane projection may have no physical effect; retain requested/raw/safe/issued/measured distinctions.

### 5.4 Effective imitation and old teaching evidence

The correlated-Gaussian/Tobit **residual-layer likelihood can be retained**, because the correction action distribution need not change. But residual labels/limits are expressed relative to the contemporaneous external base command (`effective_imitation.py:19–25,131–151,261–268`). A new base changes that command and may change the binding leader/target lane, trajectories and member availability.

Do not reinterpret historical speed-only tape residuals as lane-on labels, reuse their causal advantages as if measured over the new background, or assume the old safe-speed limit remains binding after a lane change. Build/revalidate absolute teacher-command labels and limits against the new composed executor and record command differences; reverify executable synergy under the new background. Physical ceiling provenance can remain the registered interface constant—the production bank view already explicitly supplies it (`track2_backend.py:480–485`); no population-truth read is needed.

### 5.5 Practice starts, isolated branches and G levers

- **Practice/replay:** `practice_starts.py:3–24,203–230` and `accel_plant.py:51–65` require exact prefix/sample/CRN replay and a preserved t0 action. A speed-only prefix does not become a lane-on prefix by replacing weights. Either preserve and explicitly label the old speed-only prefix with a lane-on switch strictly after t0, or collect/rebind new lane-on prefixes; these answer different questions. Restore full external command-memory/dwell/lane-mode/last-target state as well as correction AR noise, V2V boards and active commitments. All group/base/solo continuations must share the same declared prefix and background.
- **G1 executable certificates:** existing `base_policy_sha256`, runtime, snapshot/prefix, t-1 geometry, exact macro/decision/roles and issued-effect bindings change. A native-plan or speed-only executable certificate is not a lane-on certificate. Reissue group/base/**all** solos on the original full window and CRNs; do not merely update its SHA string. G1 currently certifies F3 trigger lessons/practice, not every historical continuous AWR label.
- **G2 opportunities:** graph/slate tensors need not change when only the external base changes. But traffic geometry, availability and event ordering will change; preserve causal t-1 views, pair-first ownership, exact recorded PPO slates and B/R symmetry. G2 alone does not add complementary lane actions.
- **G4 sparse measured credit:** retain signed `min_i J_solo_i - J_group`, full windows and paired sampling, but bind the lane-on external executor/state to every capture/worker and restore its history. In a solo or member-reversion branch, the reverted member returns to **lane-on base**, not native lanes or speed-only MAPPO. Background actors continue the same lane-on base; no hidden change to the counterfactual reference.
- **G5 lane macros:** arbitrate base lane choice versus committed macro through one lane-command owner, with declared priority/release/dwell semantics and target-lane safety. Group/solo masking must revert overridden lane commands as well as acceleration. Background base lanes continue. Record request/issued/refused/completed outcomes; do not count a requested change as execution.
- The earlier `c4-lane-policy/LANEPOLICY_READY.json` and `c4-lane-plans/LANEPLANS_READY.json` provide reusable default-off execution/precedence/CE tests, **not a ready CavPolicy converter**. Their learned lane+intent path changes V2V width to**72**, uses a new flat `Linear(64,3)` head, and treats all-keep ticks as native lanes. That differs from this checkpoint's masked139-input lane head and persistent513 regime. G5 executable macros do not by themselves require changing the correction's48-wide predictive-acceleration-intent block; only adopt72 if the new lane-intent design is deliberately selected and rebound.

G evidence: `E:/agent-work/_sol/tracks_spec.md:70–77`; `c4-grp-certify/{G1_READY.json,G1_CERTIFICATION.md}`; `c4-grp-opportunity/G2_READY.json`; `c4-grp-credit/{G4_READY.json,tools/coopmwm/g4_runtime.py}`. At the inspected local cutoff, the launch blocker document records G1/G2 accepted, G4 independent acceptance pending; G5 had no readiness receipt in its inspected worktree. This is a document cutoff, not a current remote-process status assertion.

## 6. Effort and verification envelope

Engineering estimates for one engineer familiar with this stack, **not elapsed training time or approved expenditure**:

- Versioned external-base loader/hash/config binding while retaining the correction checkpoint ABI: **2–3 h**.
- Once-per-decision composition, dual-lane safety, lane pass-through, modes and zero-correction parity: **3–5 h**.
- Replay/practice/diagnostic identity plus absolute-label/effective-limit rebinding: **4–6 h**.
- G1/G4 bindings and G5 arbitration/reversion integration: **4–7 h**.
- Focused CPU/fake-boundary regression qualification, source/config manifests and reproducible handoff: **3–5 h**.
- Total **16–26 h of identified work; plan 18–30 h with integration contingency**, roughly2–4 focused working days. Some G5 implementation can be reused once independently accepted. This is a production lane-on successor, not a wholesale CavPolicy/V2V rewrite.

Before any later physical qualification, CPU checks should cover exact no-move identity, frozen actor/group/correction tensors, one sample per decision despite two lane-safety projections, replay/reload/slot-reuse identity, A/B/R base equality, correct macro/solo lane reversion and no duplicate command writer. Future SUMO zero-layer parity, teaching recertification and the full smoke/gate must be separately quoted/charged; no successful physics qualification is claimed here. Existing G1/G4 conservative ceilings are230,400 sim-s each, but they do not automatically fund an additional base migration or refreshed banks.

The inspected `CLEAN3_PREPARATION_BLOCKERS.md:3–17` already reports a **788,880 sim-s shortfall** in its00:35 quote when protecting prior work and the newly separate500,000 mining allocation. That is a historical running-prefix quote requiring recomputation, not a new budget calculation or permission to transfer reserves. Lane-on migration cannot be assumed to fit the existing clean3 quote.

## 7. Fairness and paper interpretation

1. **Common base means all A/B/R, not only B.** Bind aecd7d70, its group encoder, preprocessing, bounds, lane rules, deterministic settings, mode513 and entrant treatment identically. Start A/B/R from the same correction-u0 initialization and data-independent seeds. A remains the registered plain-correction PPO arm; B/R preserve their frozen-base layer contract and differ only in guidance source under matched code, family support, practice/sampling and verification effort.
2. Report the **untouched frozen lane-on MAPPO** in addition to its A continuation. Comparing only B against a continued A cannot show improvement over the original base. Keep total fleet/source delay, safety and true group-over-best-solo synergy distinct: a multi-CAV-capable lane head is not proof of beneficial cooperation.
3. Preserve the Oct30 **speed-only d9042cc5 primary comparator** for the current paper/run. A future lane-on method should report a same-action-regime lane-on comparator as a required additional matched control and label speed-only comparisons as cross-action-regime. A claim that WM guidance caused the gain needs B−R/B−A/common-base contrasts under the same lane regime, not just lane-on B versus speed-only MAPPO.
4. The two MAPPOs have different histories. Speed-only was warm-started from lane-on and trained further; its export's6.9966M seconds do not erase the inherited7.038M. Disclose both stages, retained failed-attempt charges and later continuation/teacher/verification costs when discussing efficiency. Existing evaluations are not a controlled ablation of lane control alone.
5. Changing A/B/R to lane-on changes the registered reference/estimand. Preserve clean1/clean2 evidence, diagnostics and u24 synergy definition; identify the successor lineage prospectively rather than pool old and new histories into an unchanged clean3 result. Existing PANEL120/verdict outcomes must not become teacher-training data or an outcome-selected initialization sweep.

## 8. Recommended immediate alternative: extra frozen evaluation baseline

**Use aecd7d70 as an additional lane-on evaluation baseline, keeping the planned speed-only lineage.** This answers whether the present system competes with a controller that already has learned lateral authority, without conflating new G-lever learning with inherited lane behavior. It also restores clear labels to older E3/CTRLARM/PANEL120 results.

Load the exact C4 export with its original group encoder/lane rules; execute its full lane-aware path and mode513, not `with_lanes=False`, a native-lane proxy, a neutral new lane head or the current native-only LEVERPROBE guard. For a later authorized comparison, run on the same prospectively registered permitted worlds, full warmup/control windows, request population and reporting conventions as the clean3 candidates, include native/CTRL0 and d9042cc5 where budgeted, and report paired CIs and safety. Existing E3 numbers remain historical evidence, not substitutes for matched clean3 outcomes. The baseline adapter/report integration is approximately **2–4 engineering hours** using the existing E3 deployment path, with a separately quoted physical evaluation cost.

**Bottom line:** the checkpoint is real, usable and demonstrably lane-controlling. The most feasible training route is to retain clean3's residual/V2V architecture and replace its external frozen executor in a new, qualified lane-on lineage. For the current clean3, the evidence and implementation state favor **extra evaluation baseline now; no base switch within the existing recipe**.
