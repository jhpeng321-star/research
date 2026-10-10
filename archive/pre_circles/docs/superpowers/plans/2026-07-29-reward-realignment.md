# Reward Realignment Plan: Beat-HDV Objective with Safety/Fairness Gates

> **For agentic workers:** implement task-by-task in order. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip Task 0 (context) — it contains constraints derived from a post-mortem of v16–v20.

**Goal:** Make "beat the all-HDV baseline on travel time" something the shepherd reward can actually express, then prove or disprove the result with a paired, seed-matched evaluation protocol. Objective (user-stated): **beat HDV on travel time as much as possible, subject to safety (TTC/collisions) and fairness (time-loss) being no worse than HDV.**

**Architecture:** Reward changes confined to `env/rewards.py` (keep `reward_fn=` injection intact); one guard relaxation in `env/actions.py`; checkpoint selection moves from `policy_reward_mean` to gated eval travel time; final evaluation is paired seed-matched policy-vs-HDV on identical demand.

**Tech Stack:** SUMO + TraCI, Ray RLlib PPO, pytest. Env: conda `rl_driving`, `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`, `PYTHONPATH=D:\research\traffic_shepherd_rl`.

---

## Task 0: Post-mortem context (READ FIRST — constraints, not optional)

Findings from the v16–v20 investigation (2026-07-29). These explain *why* the plan is shaped this way; do not re-litigate without new data.

1. **The v18–v20 reward's optimum IS the HDV baseline.** `MergeOutflowTracker.sys_norm` (env/rewards.py:199-211) is clipped at 1.2 and calibrated so uncontrolled HDV ≈ 1.0. In healthy flow the max intervention gain is 0.5×(1.2−1.0) = **+0.1/step**, while the min intervention cost is −0.1 (control cost) − stability − ego-speed loss. Cost ≥ benefit in every improvable state → PPO correctly learned "behave like an HDV". The term only punishes (jam → −0.5/step); it never rewards improvement.
2. **The v16 medium win (64.8 s vs HDV 66.2 s) was mostly the ego-speed lever.** Old `_efficiency` = full-weight ego speed. ~80 CAVs × ~13 s self-improvement ≈ 1.3 s off the all-vehicle mean ≈ the observed 1.4 s. The 0.5·ego + 0.5·sys redesign halved this lever.
3. **Fairness pays for free in light traffic, is demand-noise in heavy traffic.** r_ref: light 0.074 / medium 0.22 / dense 0.38 (config/defaults.py:148-152). Typical medium local mean rate ≈ 0.05–0.1 → R_fair ≈ +0.55..+0.77 × 0.5 ≈ +0.3/step for doing nothing. Negative excursions are seed-level demand luck, correlated across all ~80 agents → gradient noise. (Same pathology as the removed neighbor-jerk term.)
4. **Congestion guard vetoes ALL modes < 8 m/s** (env/actions.py:192-194) → policy can never learn queue recovery; dense parity is its structural ceiling.
5. **Checkpoint selection on `policy_reward_mean` selects "most HDV-like"**, not fastest traffic. Training reward climbed 40.6 → 48.4 across v17–v20m while eval travel time stagnated — the two are decoupled under the current reward.
6. **Invalid/contaminated results — discard:**
   - v16 dense evals (only 577/876 vehicles arrived, 10,000-step runs) — never comparable.
   - ALL of v17 (trained 05:19–06:32 while reward/env code was being edited mid-run).
   - Any cross-version comparison of `policy_reward_mean` or eval `total_reward` (reward function changed).
7. **Seed noise floors (5 seeds, all-HDV):** medium mean 66.2 s ± 1.6 s (range 63.9–67.9); dense mean 81.7 s ± 6.0 s (range 77.0–91.5). Mean differences smaller than these spreads are not evidence; use paired per-seed win/loss.
8. **HDV has ZERO sub-1.5 s TTC events in all 5 dense and 5 medium seeds**; every RL version produces 2–14 per eval. This is the most consistent un-solved gap.
9. `diag_scenario.py` has stale hardcoded reward weights (stability 0.5, control_cost 0.2) and calls `_efficiency` without the flow tracker — fix before trusting its decomposition (Task 3).
10. Current v20d run (old reward, dense) is the **control arm** — let it finish (~3 h from 16:52), do not kill it. Its final paired eval is the baseline the new reward must beat.
11. **Tau pinning confound (found 2026-07-29, paired no_op probes).** `non_intervention` (mode 0) executes `setTau(base_tau=1.0)` + `setSpeed(-1)` EVERY step on every CAV (env/actions.py:183-185; config/defaults.py:40). But HDV driver taus from the route vTypeDistribution are aggressive 0.95 / normal 1.1 / **cautious 1.5** (tools/build_demand.py). So every CAV — including under `no_op` — has its driver-type tau overwritten to 1.0 (cautious CAVs forced to follow at 1.0 s). Measured effect of `no_op` at p=0.10 vs all-HDV, paired: medium ≈ neutral (mean −0.9 s); **dense: seed-dependent swings of −17.8 s (seed 43) to +5.5 s (seed 44)**. Consequences: (a) all-HDV alone is NOT a sufficient control on dense — the correct "do no harm" baseline is `no_op` at the same penetration; (b) vs no_op paired on dense, rule_shepherd is worse on all 5 seeds (+0.1..+14.4, mean +6.3), v18 (+3.6) and v19 (+8.3) are worse than doing nothing too; (c) on medium, rule_shepherd ≈ no_op (+0.2) and no_op ≈ HDV, so medium conclusions stand; (d) rule_shepherd has TTC=0 and fewer collisions than HDV on all 10 paired runs — the action space is safe; its gap rule NEVER fires on dense (disabling it changed nothing; its flow-stab rule helps: removing it made dense worse: 92.6→95.2 seed 44).

---

## Task 1: Generate all-HDV references for selection seeds 47–49

Checkpoint selection (Task 4) needs HDV references on reserved seeds {47, 48, 49} for medium and dense. Seeds {47,48,49} are used for selection ONLY; final reporting stays on {42–46} (references already exist in `runs/eval/all_hdv_{medium,dense}.seed4[2-6].json`).

- [ ] Run all-HDV evals for seeds 47, 48, 49 on medium and dense (same procedure/seeds style as existing `all_hdv_medium.seed4X.json` outputs; see `evaluate_sumo_baseline.py` / the tool that produced the 7/27–7/28 seeded HDV files).
- [ ] Verify: each output has all vehicles arrived (800 medium / 876 dense), `collisions` present (not silent-zero), TTC = 0 (expected for HDV).

## Task 1.5: Fix tau pinning (env/actions.py)

Prerequisite for interpreting ANY dense result (see Task 0 finding 11).

- [ ] Cache each CAV's original tau (its vTypeDistribution-sampled value) at activation; `non_intervention` and the congestion-guard veto must restore THAT tau (and `setSpeed(-1)`), not global `base_tau=1.0`. Mode-specific tau deltas still apply on top of base_tau for ACTIVE modes (unchanged behavior while intervening).
- [ ] Test: under `no_op` at p=0.10, a CAV's effective tau equals its vType tau every step.
- [ ] Re-run the no_op paired probe (5 seeds × medium+dense) after the fix: expected ≈ all-HDV per seed on BOTH densities (any residual = genuine CAV-mechanics effect, report it).

## Task 2: Reward v21 (env/rewards.py) + guard relaxation (env/actions.py)

**Files:**
- Modify: `D:\research\traffic_shepherd_rl\env\rewards.py` (`_efficiency`, docstring)
- Modify: `D:\research\traffic_shepherd_rl\config\defaults.py` (weights)
- Modify: `D:\research\traffic_shepherd_rl\env\actions.py` (guard)
- Test: `D:\research\traffic_shepherd_rl\tests\test_reward.py`, `tests\test_reward_structure.py`, `tests\test_shepherd_actions.py`

**Reward v21 — the core change is a POSITIVE slope above HDV parity:**

```
R_eff  = ego_speed/max_speed + w_flow * F_flow          (w_flow = 0.4)
F_flow = clip(sys_norm, 0, 2.0) - 1.0                   # ∈ [-1, +1]
```

- `_efficiency` returns `ego + w_flow * F_flow` (restore ego to FULL weight — the v16 lever). `sys_norm` = 0.5·q_norm + 0.5·s_norm unchanged in form; tracker, refs, EMA unchanged. Without a tracker, `F_flow = 0` (neutral).
- **REQUIRED — raise the inner clips:** `MergeOutflowTracker.q_norm`/`s_norm` currently clip at **1.2** (env/rewards.py:199-207). Left untouched, sys_norm ≤ 1.2 → F_flow ≤ +0.2 → max flow bonus +0.08/step, still below the −0.1 control cost (the same trap as v18–v20). Raise both clips to **2.0** (or drop the inner clips and rely on the outer clip in `F_flow`) so the full [-1, +1] range of F_flow is reachable.
- Semantics: matching HDV flow = 0; jam → down to −0.4 (at w_flow 0.4); beating HDV flow → up to **+0.4/step**, profitable against the −0.1 control cost + stability + ego-speed cost of an intervention.
- `R_safe`, `R_stab` (0.25), `C_ctrl` (0.1) unchanged. Fairness: same mean-rate form, **weight 0.5 → 0.2** (it pays for free in light traffic — do not let it dominate the gradient; the fairness GATE is enforced at selection/eval time, not in the reward).
- Weights in `config/defaults.py`: efficiency 1.0, safety 1.0, stability 0.25, fairness 0.2, control_cost 0.1; add `flow_weight: 0.4` (or wire via reward_weights) — keep single-source-of-truth, no magic numbers in env code.
- Update the module docstring term table.

**Guard relaxation (env/actions.py:192-194):** below `guard_speed`, veto only DECELERATING modes (speed_factor < 1.0: gap_creation, conservative_following); allow flow_stabilization (1.00) and release (1.05). Braking in a queue stays forbidden; the policy can now act inside queues.

**Verification:**
- [ ] Update affected reward tests to v21 semantics; add: (a) sys_norm = 1.3 → F_flow = +0.3 (positive slope above parity), (b) sys_norm = 1.5 → F_flow = +0.5 (mid-range, requires the raised inner clips — regression test for the 1.2-clip trap), (c) sys_norm = 2.5 → F_flow = +1.0 (clip), (d) sys_norm = 0.4 → F_flow = −0.6, (e) no tracker → neutral.
- [ ] Guard test: below guard speed, gap_creation is overridden but release executes.
- [ ] `python -m pytest -q` full suite green; regenerate `tests/golden/golden_trace_dense.json` ONLY from the green baseline (`python -u tools/golden_trace.py dump --out tests/golden/golden_trace_dense.json`) if obs/reward semantics changed the trace.

## Task 3: Fix diag_scenario.py before using it

- [ ] Update hardcoded weights to match v21 (stability 0.25, control_cost 0.1, fairness 0.2, flow 0.4); pass the flow tracker into `_efficiency`; fix the mislabeled `ego_speed` stat (it currently logs `episode_time`).
- [ ] Smoke train 30k steps on medium; run `diag_scenario.py` on the smoke checkpoint and sanity-check the decomposition: F_flow ≈ 0 at HDV-like flow, positive only when discharge/upstream speed exceed refs.

## Task 4: Checkpoint selection by gated eval travel time

Replace `policy_reward_mean` selection (AGENTS.md: "best/ selected by policy_reward_mean" — update that note when done).

- [ ] Every 10 training iterations, eval the current checkpoint on selection seeds {47, 48, 49} at the run's density, p=0.10.
- [ ] Score = mean travel time over the 3 seeds. **Gates (must all pass, else score = +inf):** TTC events ≤ 1 per eval; collisions ≤ HDV mean on same seeds; mean_time_loss ≤ 1.05 × HDV mean on same seeds.
- [ ] `best/` = lowest gated score. Implementation may be a standalone script run between/after training phases if an in-training callback is too invasive — do not block on automation.

## Task 5: Training runs (after v20d finishes)

- [ ] Train A: v21 reward, medium, 600k steps, hyperparameters as v20m (`train.py --baseline shepherd --density medium --total-timesteps 600000 --logdir runs/rllib/shepherd_v21m`).
- [ ] Train B: v21 reward, dense, 600k steps (`.../shepherd_v21d`).
- [ ] Optional ablation C: v21 dense WITHOUT the Task 2 guard relaxation (isolates guard effect).

## Task 6: Final paired evaluation + win criterion

- [ ] Seeds {42–46}, policy vs all-HDV AND vs `no_op` @ p=0.10 on IDENTICAL demand (three-way paired). Densities: medium and dense. (Already on disk from 2026-07-29 probes, reusable after Task 1.5 re-baseline: `rule_shepherd_{medium,dense}_p010.seed4X.json`, `noop_{medium,dense}_p010.seed4X.json`, `rule_nogap_dense_p010.seed4X.json`, `rule_safetyonly_dense_p010.seed4{4,6}.json`.)
- [ ] Report per density: median TT, per-seed win/loss vs HDV **and vs no_op**, worst-case TT, TTC events, collisions, mean_time_loss, ramp/mainline TT split.
- [ ] **Win criterion:** ≥4/5 seeds won on TT vs BOTH controls AND TTC ≤ 1/eval AND collisions ≤ HDV AND time_loss ≤ 1.05 × HDV. Beating all-HDV while losing to no_op is reported as "worse than doing nothing", not a win.
- [ ] Include the finished v20d (old reward) in the same paired protocol as the control arm.

## Task 7: Documentation

- [ ] Update `docs/PROJECT_GUIDE.md` and `AGENTS.md`: new reward formula, new checkpoint-selection rule, paired-eval protocol, and the "invalid results" list (v16 dense, v17) so future sessions don't cite them.

---

## Pitfalls (from the post-mortem)

- Do NOT compare reward values across reward versions — only paired travel time / safety / fairness metrics.
- Do NOT leave the 1.2 inner clips in `q_norm`/`s_norm` — that silently re-caps F_flow at +0.2 and reproduces the v18–v20 "improvement is never profitable" trap.
- Do NOT raise fairness weight above 0.2 or reintroduce terms that pay regardless of action.
- Do NOT select checkpoints on training reward.
- Do NOT use seeds {47,48,49} in final reporting (selection-contaminated) or {42–46} for selection (report-contaminated).
- If dense stays gridlock-prone after v21, check `FLOW_V_REF dense = 19.7` / `FLOW_Q_REF dense = 0.95` calibration (config/defaults.py:127-141, measured seed 42 only) before touching reward weights.
- Expect a SMALL margin (a few seconds) — 10% penetration caps the ceiling. A verified parity-with-zero-TTC on dense is also a publishable result; do not chase a big number by loosening gates.

---

## ADDENDUM (2026-07-29 23:30): v21m results + residual reward-hacking diagnosis

**Tau-fix verification: PASS.** Post-fix `noopfix_medium` == all-HDV exactly on all 5 seeds (identical travel times). no_op is now a true null control.

**v21m paired verdict (medium, seeds 42–46, checkpoint=iter_40 via gated selection): PARITY, not WIN.**
Per-seed v21 vs HDV: +2.4, −0.6, −0.8, +1.7, −1.3 (3W/2L; mean 66.5 vs 66.2). Gates: TTC=0 all seeds PASS; collisions total 9 vs HDV 15 PASS (but per-seed fails on 44: 1>0 and 46: 5>4); time_loss ≤1.05×HDV passes 3/5 (fails 42, 45). Controls: v20m (old reward) mean 69.1 → v21 is 2.6 s better; rule_shepherd post-fix mean 67.5 → v21 wins 4/5. Selection seeds: iter_40 mean 65.65 (best of iter_30..100; late checkpoints 90/100 degrade to 73–76 s).

**Residual reward hacking (why training reward kept rising while eval TT worsened after ~iter_40), evidenced by diag_scenario.py on iter_40 vs iter_100:**
1. ~~**Fairness term is saturated/dead.**~~ **CORRECTION (2026-07-30, R1.2):** the diag's `fair = 0.9333` was a DIAG TOOL ARTIFACT (its `_fairness` call sees near-zero local rates, e.g. stale/missing tracker). Direct FCD measurement of the actual v21m runs shows the true local delay rate ≈ 0.21 vs r_ref 0.22 → the live term reads ≈ +0.05 ± seed noise: small but NOT saturated. Real issue is weaker than claimed: fairness punishes local congestion (clip −1 → −0.2/step at w 0.2) but its everyday signal (±0.01/step) is tiny next to safety/efficiency terms. Recalibrated references (validated, R1.2): **r_ref medium 0.22 → 0.24, dense 0.38 → 0.45** (local same-edge-200 m rates measured from all-HDV FCDs, seeds 42–46; network-formula validated against existing 0.22/0.38 within 1%).
2. **Timidity exploit.** iter_100 vs iter_40: braking modes 21.7% vs 5.1% of actions, non-intervention 73.9% vs 91.6%, diag duration 69.3 s vs 64.3 s. Steady conservative_following buys safety-penalty reduction + smoothness at ~−0.04/step ego cost, while the delay imposed on followers is under-priced on medium (F_flow only responds when the merge drops BELOW HDV reference; medium slack absorbs CAV braking waves). Reward↑, TT↑. iter_40 precedes the exploit dominating.
3. **Leader-only safety.** `_safety`/TTC only watch the same-lane leader; side-merge conflicts are invisible → TTC=0 with 2–5 collisions on some seeds. FIX: min-TTC over neighbors / lane-change-conflict term.
4. **OFFICIAL EVAL TTC METRIC IS BROKEN (found 2026-07-30, R1.1).** `evaluation.py:_compute_ttc` uses `distance = leader.pos − follower.pos` WITHOUT subtracting vehicle length (~5 m) → TTC inflated by ~5 m/closing-speed → almost never < 1.5 s. All "TTC = 0" results (HDV AND v21m) are artifacts. Corrected analysis (gap = pos_diff − 5 m, same-lane adjacent, edge-triggered): leader-TTC events v21m vs HDV per seed = 85/76, 98/92, 69/67, 96/77, 92/99 (total 440 vs 411, +7%) — v21m leader safety is HUMAN-LEVEL, and the old "RL has 2–14 TTC, HDV has 0" claim was vacuous (old policies were extreme enough to trip even the broken metric). Also: v21m's 9 collisions (seeds 42–46) are ALL on the merge edge (collisions XML, collider/victim) — side-swipes at the lane drop, consistent with missing side-conflict safety term.

**Process validation:** gated eval-TT selection worked exactly as designed — reward-based best/ would have picked iter_100 (36.7) over iter_40 (33.1); the gates picked iter_40.

**Next (Phase B decision):** dense v21d was killed before launch (2026-07-29 22:35); per the decision gate, medium outcome = PARITY-with-zero-TTC → candidate next step is imitation warm-start from rule_shepherd OR fairness-recalibration + retrain, before any dense run.

**UPDATE (2026-07-30, R2.1 DONE — v22 reward implemented, 191 tests green):** (a) `side_ttc` added to env/traci_query.py — converging-lane TTC restricted to genuine lane-drop races on the merge edge (pairs on lanes {2,1}; parallel lanes ignored); wired into `_safety` as −1 below 1.5 s (R_safe bound now [-10, 0]). (b) `evaluation.py::_compute_ttc` vehicle-length fix (was the "TTC always 0" bug) + new `evaluate_side_ttc` metric (key `ttc_side` in evaluate_run output). (c) `FAIRNESS_R_REF_BY_DENSITY` recalibrated from R1.2 local-rate measurement: medium 0.22→0.24, dense 0.38→0.45 (light unchanged, pending measurement). (d) constants: MERGE_DROPPED_LANE_INDEX=2, DEFAULT_VEH_LENGTH=5.0. Real-data sanity (seed42): fixed leader-TTC v21m 131 vs HDV 120; side-TTC v21m 1256 vs HDV 988 (+27% — the v22 reduction target). New tests: tests/test_side_ttc.py (10). Golden trace regenerated from green suite. NEXT: train v22m (600k, medium) → gated selection → paired eval (protocol unchanged; gates now use the FIXED TTC metrics).

**UPDATE 2 (2026-07-30, v22m VERDICT + v23m in flight):**
- **v22m = FIRST WIN vs HDV on medium.** Checkpoint iter_70 (gated selection): travel time 65.24 vs HDV 66.20 mean, **4/5 paired seeds won** (42 −0.9, 43 −3.4, 44 −0.2, 45 −1.2; lost 46 +0.9); vs no_op identical (no_op ≡ HDV post-tau-fix). Corrected leader-TTC 26–33 vs HDV 95–158 per seed (**−75%**); collisions 8 vs HDV 15 total (but worse per-seed on 44: 1>0, 46: 5>4); time_loss better 4/5 (seed46 1% over gate). Beats v21m by 1.3 s mean, beats rule_shepherd 4/5.
- **GEOMETRY CORRECTION (PET analysis, 2026-07-30):** dropped lane is **`_0`** (no outgoing junction connection), NOT `_2` as coded. All 14 recorded collisions are crawl-speed rear-enders of aggressive_car (tau 0.95 < 1 s step — SUMO tau artifact) inside the `_0` queue; ZERO CAVs involved in any of them. v22's side_ttc watched the wrong pair ({2,1} = parallel lanes). **Fixed to {0,1} for v23m.**
- **PET (severe <1.0 s/pair, 5 seeds):** v22m better than HDV on 42/46, worse on 43/44/45 — merge-conflict tightness ≈ HDV (the wrong-lane side-TTC "dominant win" was metric noise). Corrected side-TTC (lane _0): v22m below HDV 3/5.
- **Seed-46 failure autopsy: CHAOS, not defect.** Identical demand/types; v22m's congestion wave shifted later (500–900 s); throughput unharmed; CAVs ≥ HDV speeds; warmup breaks at first CAV (t=1s) so "pre-warmup collision exclusion" premise was FALSE (dropped). v22m even removed HDV's t=563 collision.
- **Timidity hypothesis REFUTED:** v22m braking modes 0.2% vs v21m 5.1%; shifted to flow_stab 6.5% / release 2.9%. No w_flow defense needed; bumped 0.4→0.5 purely as the speed lever.
- **v23m = v22m + dropped-lane fix {0,1} + w_flow 0.5** — training started 04:47 (600k medium), gated selection (corrected paired TTC gates: ttc/side_ttc ≤ HDV per-seed) + paired eval + PET armed as post-driver.
- Eval artifact for the report: collisions in this scenario are dominated by the aggressive-car tau<step artifact — a SUMO/modeling caveat, not a policy signal; both policies suffer it.

**UPDATE 3 (2026-07-30 ~07:15, v23m VERDICT + dense transfer — overnight work complete):**
- **v23m (lane fix + w_flow 0.5, winner iter_50):** mean TT 65.06 (−1.7% vs HDV — best mean yet) but only 3/5 per-seed wins (won 43/45/46 incl. seed 46 64.3 vs 67.6 = v22's lost seed; lost 42/44). Safety: leader-TTC 23–41 (−75%), **side-TTC (corrected lane _0) below HDV on ALL 5 seeds**, collisions 7 vs 15. PET severe rate: worse than HDV on 4/5 (better instantaneous gaps, slightly worse merge arrival coordination).
- **v22m vs v23m are COMPLEMENTARY:** v22 wins 42/43/44/45, v23 wins 43/45/46. Per-seed best-of-both = 64.76 mean, 5/5 won. v22m remains the report model (4/5 TT wins); v23m = robustness variant (best mean, best worst-seed 66.2 vs 68.5).
- **Dense transfer (medium-trained, no dense training):** v22m 83.66 mean ≈ no_op 83.76 (not harmful — first medium model to transfer at neutral; vs HDV 81.64 → +2.5%, 2W/3L). v23m 87.14 → worse than no_op (medium-tuning hurt it). Collisions dense: v22 12 = HDV 12.
- **PET metric now canonical for merge safety** (runs/eval/analysis/pet_at_merge.py + pet_results JSONs); side-TTC corrected to lane {0,1}; the pre-v22 wrong-lane side-TTC numbers are noise (discard).
- Data-integrity lesson: eval FCD filenames are per-policy, not per-run — always copy FCDs immediately post-eval (paired drivers now do this).

**UPDATE 4 (2026-07-30 18:30, v26m FAILS mean-first bar; tail-term post-mortem):**
- **Decomposition finding (drove v26m):** v22m's win is ~88% externality; residual loss = mainline-HDV loser TAIL in peak window (~1,900 s/seed positive mass, p90 +12 s, median delta 0). Ramp Q3 small net loser; CAV self-speed saturated (not a lever).
- **v26m = v22m + one-sided p90 tail penalty** (−clip((p90_local−0.34)/0.34,0,1), w=0.25, refs measured medium 0.34/dense 0.53), warm-start iter_70, 600k new steps, winner iter_140. **Verdict: FAIL — mean 66.44 (>65.24 bar, >HDV 66.20); 3W/2L; seed44 +3.4 with 5 collisions.**
- **The tail GREW instead of shrinking:** mainline positive mass +41–64% on seeds 43/44/45 (1754/3682/2205 vs v22's 1075/2238/1563); smaller only on 42/46. Mechanism: paying CAVs for nearby-HDV delay incentivizes braking near queues → braking exports MORE delay downstream (timidity exploit in new costume). Lesson: "protect others" penalties that act through braking are self-defeating in this reward family; tail protection must come from non-braking mechanisms (earlier flow-stab timing, release timing), not penalties.
- **Safety side-effects of v26m:** side-TTC below HDV all seeds (best yet), PET better 3/5, leader-TTC −75%, collisions 14 ≈ HDV 15. The policy became more merge-cautious — and paid for it in TT and tail.
- **Wide-seed robustness champion (free):** v22m_iter60 — 5/9 wins (seeds 47–55), mean 65.32, worst-seed 67.5; beats iter_70 on the full 14-seed union (8 vs 7 wins, 65.55 vs 65.76). iter_70 remains the report-seed model (4/5 wins, 65.24); iter_60 is the robustness alternate.
- **Gap_creation ablation (forced_mode):** always-on is catastrophic in every configuration (medium 178.6 s, radius-200 125.7 s, dense 201.1 s vs baselines 67.9/67.9/82.7 s). ~0% usage by learned policies is a finding, not a deficiency — continuous-yield strategies are negative-sum at this merge; only flow-stab/release are positive-sum.
- **Multi-penetration (v25m) verdict:** off-dose harm flattened (5% +0.39→−0.02, 20% +0.04→−0.26) at cost of trained-dose win (10% −0.97→−0.14); no consistent win anywhere → generalization attempt failed; specialist + documented dose-response limitation is the report position.

**UPDATE 5 (2026-07-30 19:00, v27d launched — "v22m recipe on dense"):**
- **Task B0 done:** FLOW refs re-measured across seeds 42–46. FLOW_Q_REF dense 0.948 ≈ 0.95 (unchanged); **FLOW_V_REF dense 19.7 → 17.5** (seed 42 was a lucky draw; per-seed 13.5–22.0) — dense flow-term zero-point now honest (was ~11% too strict). Medium refs left at 0.90/27.0 (v22m banked under them).
- **v27d recipe (faithful to v22m + bug fixes):** from scratch, 600k dense, w_flow **0.4** (v22m's value, restored from 0.5), fairness 0.2 (r_ref dense 0.45), safety = brake + leader-TTC + side-TTC(corrected lanes {0,1}) + merge-arrival-PET, **tail penalty DISABLED (0.0)** after v26m's proven failure, congestion guard relaxed (v22-era). 208 tests green, golden regenerated.
- **v22m fairness analysis (for the report):** mean delay better 4/5; TT std LOWER than HDV 4/5 (more equal outcomes); worst-decile TT better than HDV 4/5 (Rawlsian fair); systematic unfairness = ramp funds mainline on 2–3 seeds (+1.9..+2.3 s); seed 46 a loss on all measures.
- Post-driver armed: 8 checkpoints × seeds 47–49 gated selection (corrected dense TTC refs) → paired eval 42–46 + PET. ETA ~21:30.
