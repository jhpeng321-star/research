# WMFIX independent adversarial review — 2026-10-31 (read-only; Claude reviewer, not Astra)
Scope: c4/wmfixmerge HEAD 03ea0a28 plus the untracked verdict files in the worktree; GLM branches c4/wmfix-g1/g2/g3.
The G branches were NOT merged (integrator re-implemented data/prepare/objective/feasibility/verdict), so G*_RESULT test counts do not apply.
CPU tests (8 request files): **68 passed**, 113 s (request says 65). One throw-away admission/mix probe (temp script, deleted); no GPU/SUMO/SSH/sealed.

## 1. Claims
1. **PARTLY VERIFIED, overstated.**
   - The output goes before the envelope only if `physics.emergency_braking` is set (wms.py:1045, 1295-1300); `enable_merge_response` (wms.py:308-316) does not assert it.
   - CAV ownership holds: `~is_cav` (merge_response.py:60) plus the later CAV override (wms.py:2775).
   - The "full-upstream field" is 4 RBFs with sigma ~907 m over -2646..75 m (merge_response.py:31-34). The 75 m merge zone and the -379 m ramp entrance share one basis; the bottleneck is unresolved.
   - It is another learned HDV accel residual on regional aggregates (cf. PRIOR #31, #41, RAMPFIX). The novelty is the target, not the mechanism.
2. **VERIFIED (arithmetic).**
   - Ledger: due-inserted (source_ledger.py:28-34; entries.py:377-389), no clip/drop when ON; Q0 comes from the label queue0 (runtime.py:70-72).
   - Slot exhaustion: no slot, then break, so the backlog is kept (entries.py:355-357, 994-997).
   - OFF bitwise parity versus fc69fcfd is tested (test_wmfix_admission.py:80-97).
3. **WRONG under multi-birth.** Admission keys candidates by attempt order, `_candidate_index+attempt` (entries.py:987-988).
   - LearnedMix assigns ordinals in `newborn.nonzero()` slot order (tools/errsrc/inference.py:30-38).
   - A CAV takes the lowest *virgin* slot; an HDV takes the lowest free (recycled) slot (entries.py:351-354).
   - **Reproduced:** CAV attempt0, then HDV attempt1 into recycled slot 0. The HDV is admitted with ordinal-1 v0 3.33 but moves with the ordinal-0 draw, v0 20.0.
   - Only the single-birth fixture is tested (test_wmfix_admission.py:27-57).
   - The learned path drops the incumbent `sumo_min_speed` rejection (entries.py:1179) and uses gap<min_gap (learned_admission.py:49-51). That is not "the incumbent formula". CAV entrants get HDV-pool settings.
4. **NOT VERIFIED; offset likely.** Labels sum q_0..q_{n-1} (labels.py:141; prepare.py:43), while the runtime samples after insertion, q_1..q_n (runtime.py:110, 135).
   One-tick shift: 0.5(q_n-q_0) per row, 0.5·Δq_n per contrast. No HARNESS_CONTRACT exists.
5. **VERIFIED as stated, but the gate is weak.**
   - No STE; source is numpy `pending`, with no gradient (runtime.py:110).
   - V2 FAIL is confirmed. The V3/V4 "pass" moved J MAE 0.008005 -> 0.008003 (0.03%).
   - The real probe is one group, one arm, 60 s, and "any decrease" (feasibility.py:97, 130, 148). The real receipt is BLOCKED (file not found).
6. **VERIFIED.**
   - Allowlist: set equality, FIT role, mappo skipped before open (data.py:8-9, 53-71); loader filters registered worlds/arms (runtime.py:35-42).
   - Q0/due/paired-history checks: prepare.py:37-69. Path-substring guard plus audit hook: data.py:10, 27-32, 105-112.
   - Caveat: IMPLEMENTATION_FILES (data.py:17-24) omits tools/errsrc/inference.py (the ordinal source), labels.py, inputs.py and verdict*.py.
7. **VERIFIED.**
   - 44·32+32+33 = 1473 params; zero last layer (merge_response.py:20-23); allowlist at :64-73.
   - Bitwise frozen check plus reload (train.py:31-55); flags default OFF; 21600 s cap and stages (train.py:74, 112-115).
   - CUDA memory is unmeasured. The head adds [B,K,K] and [B,K,K,3] pairwise tensors per tick (merge_response.py:43-47), with K<=1024 (runtime.py:27).
8. **PARTLY VERIFIED.** Gates exist (verdict_metrics.py:44-168). But:
   - (a) The A base is WM2 with legacy entries, while the candidate is head+conserve+learned admission (verdict_panels.py:41-61). That is a bundle with no attribution (PRIOR #4).
   - (b) accept.py/C1 never install conserve/admission (verdict_panels.py:112-113).
   - (c) The B/C1 producers are null and `accept_roots` is [] (verdict_plan.json:12).
   - (d) The verdict files are untracked and unhashed.

## 2. Launch blockers (before any GPU)
1. Fix ordinal binding: record (b,slot)->ordinal at `_place` and have LearnedMix consume it. Add tests for CAV+HDV same-step births, recycled slots, both entrances, exhaustion and retries.
2. Prepare/hash the real TRAIN corpus (feasibility BLOCKED). 3. Sign the source-clock contract, quantifying the one-tick shift vs WMREQUAL3.
4. Log per-term loss shares at step0 on real TRAIN.
   Absolute anchors are normalized by *paired-contrast* RMS (prepare.py:102; train.py:120), so they likely dominate.
   The paired J term with non-differentiable source (src/wms/train/merge_response.py:27-31) rewards road contrasts that cancel source error. Drop J from the gradient loss or justify it with measured shares.
5. Strengthen the feasibility gate: all 12 groups, both arms, 120/300 s.
   Require teacher ceiling < initial, and an optimizer step that lowers paired J MAE beyond the noise of a frozen re-run.
6. Score the step0 zero head with conserve+admission as the A/B attribution arm and as the retarget-rule residual re-check.
7. Bind the B/C1 producers and accept roots. Make C cover the deployed admission path, or record it as uncovered.
8. Commit the verdict files and extend IMPLEMENTATION_FILES. Run a gpuR single 300 s update microbenchmark (memory + seconds) and confirm the head is inside checkpointed steps.

## 3. Prior-pattern risk
- **High:** #40/#47/#48 "best step0, final sign/r regress". This is the same small FT recipe: weights move at most ~0.036 (lr 3e-4, <=120 updates), and the first layer has zero gradient at init. Effectively it is a random-feature readout.
- **High:** bundle confound (#4, #11), from 1.8a. **Medium:** absolute fit hides response/cancellation (#50), via blocker 4.
- **Medium:** short-horizon gate does not transfer (#17, #36). Feasibility is 60 s only, and the 300 s stage runs last, so it is first cut by the cap.
- **Medium:** tiny-effect regional aggregates (#31, #41). Four broad bases give no resolution where discharge forms.

## 4. Plausibility
Training sees ~24 paired contrasts (12 worlds x 2 matched arms, if one t300 start each) at batch 1, with 30+30+<=60 updates; each 300 s contrast is seen ~2-3 times.
The target error is directional (J300 r -0.235, sign .556), not a level shift.
- **Magnitude** is reachable (a 0.1-0.3 m/s² HDV bias can move upstream delay ~10^3 veh-s); **learning** an arm-conditional response from 24 noisy contrasts via 4 broad bases is unlikely.
- **Most likely failure:** an anchor-driven common-mode level/count correction. A then shows lower bias but an MAE CI that includes 0, with sign/r/confident-gain flat. Or it overfits the 12 worlds and regresses at 120-180 s. Either way the outcome is "best = step0".
- **Secondary:** the admission change alone moves J and is hidden by the bundle.

## 5. Verdict: **FIX-FIRST** (blockers 1-8; 1, 4, 6 are scientific, not paperwork)
Single most valuable change:
- After fixing the ordinal bug, run the zero-head candidate path (conserve + learned admission, K8, native lanes) on the corrected WMREQUAL3 A panel on CPU.
- Use it as the retarget check (does the upstream residual survive the new entry path?) and verdict baseline; if the residual vanishes or changes, do not train.
