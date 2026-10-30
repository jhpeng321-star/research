# WMFIX prior-attempt audit — 2026-10-31
Sources: docs/WM_CHANGE_LOG.md; named C4-V5/C4-V7FT REGISTRATION.md; W1/W2/W4/W5/W6CAP/W7/W8_VERDICT.md and ERRSRC.md in E:/research-dense/docs/paper_results; task-named WMREQUAL/2 and SPEEDHEAD. No simulation/training.
W3_VERDICT.md is absent: registration documents zero-update/OOM and structural gradient gates, not a scientific verdict.
Each entry below gives change → judged target → result/status; grouped entries retain distinct outcomes.
1. C1 IDM/MOBIL + gated learned residual → unseen PI/wall position → KEPT reference (wall60 8.71 m).
2. Random start shifts .75 → same target → kept C1 value; switching/event oversampling separately unisolated/deferred.
3. Hard residual caps (including asymmetric braking run) → position/blowups → REJECT: saturation killed gradients; MAIN2 ~5x worse.
4. MAIN2 bundle (entries/projection/caps/losses/merge-cal/start-shift) → 60/120s position → REJECT bundle, attribution confounded.
5. Traffic/stability/conservation penalties → rollout state/loss shares → traffic .01–2%, stability/conservation 0%; not structural conservation.
6. Gap projection/merge calibration/lane-logit cap → C1 position → modest optional gains/neutral, not intervention validation.
7. pack_v3 loader + eval-world-disjoint lists → usable windows/leak prevention → KEPT loader fail-closed, +294/+47 windows.
8. Entries at inference on entry-untrained C1 → position → worse120 (112→170 m); rejected as inference-only rescue.
9. driver_params-only scoring, pushforward/self-recovery, black-box retrains → position → respectively drops residual/rejected, owner-vetoed, stopped/crashed.
10. Direct120 predictor, jam oversampling, role-neighbour networks → proposals/deferred; no measured success.
11. R3 soft bound/size penalty/lane-rate/best-checkpoint/history/braking bundle → unseen walls/history → no new best; residual/lane drift reduced, over-slow.
12. R3 diagnostics → position/speed/brake recall → history/calibration not culprit; braking overweight suspected; lane-rate target counted junction toggles.
13. LANEDWELL hysteresis/dwell; LANECAL change offsets → lane flips/position → fewer flips without long-horizon gain; reject inference rules.
14. LANECAL gate cap; R4 learned saturating gate/corrected lane target/no braking → position → fingerprints better,120 error unchanged; C1 retained.
15. R4 history/prior clamps → position → physical settings inert in residual mode; jam discharge failure persists; oracle future lanes/types diagnostic only.
16. Type identification and correction fade → position → promising diagnostic, selection caveats; no truth/type-based solution authorized here.
17. R5 learned settings + full BPTT + bounded separate lane gate/flip loss → position → exploding gradients; R5b damping/lower rates rescued short horizons,120 n.s.
18. Lane-event/LC2013 audit and R5b switch-offs → identity/position → settings/cap useful; correct lanes large lever; counts alone insufficient.
19. v5 lane prefit/textbook corpus; v6 reachable settings/invariants/long pairs → wall position → v5m selected over v6m (19 blowups); unfinished v6merge not included.
20. v5m merge-end wait + mainline yield → walls → historical base, later diagnosed artificial merge deadlocks; not universal behavioural law.
21. v7 scratch/v7ft merge speed+discharge losses/policy data → discharge/ranking/walls → local flow gains but120 regression/blowups; reject recipe.
22. v8 survival timing/slim losses/forced hazard/yield/gap variants PC1–7 → stuck/departure/walls → overcorrection (223 stuck), impatience trades tails; no sweep passes all gates.
23. Full v8/heavy-data variants → walls/ramp/ranking → modest state gains, bad stuck/discharge or criterion unmeasured; v5m retained.
24. Budget lane counts/lane_flow and K16 hedging → counts/position → realistic counts, wrong identities,120 worsens; not adopted as fix.
25. v9entry queue initialization/demand vs service/CAV mix/single insert/shared draws → source/total delay → KEPT bookkeeping; source sign restored, per-seed rank weak/road2.7x.
26. Merge-head AP gate/offset recalibration/secure-gap relaxation → timing/count/stuck → offsets saturate, mask remains; rate calibrated≠correct timing.
27. Noyield/CMR/reference cap+hold symmetry → closed-loop road levels/choice → bias improved, discrimination near chance; CMR adds nothing.
28. paired_diff heavy branches + noyield → controller choice/walls → rank .13→.23, wall120 worse and blowups doubled; not adopted.
29. v10 learnable acceptance/all-risk labels/free lanes → support/merges/position → supports merges but overshoots mainline; iter0 position nearly doubles.
30. Censor/runtime-envelope/callback/CAV-teacher/exit repairs → contracts/position → retained correctness repairs; several alleged physics defects were harness artifacts.
31. iter0b regional aggregates/courtesy lanes → delay/stuck → aggregate tiny; paired term NEVER active on60s branches; total matches by regional cancellation.
32. iter0c courtesy features/brake following; R1 persistence/CAV weighting/240s paired data → stuck/waves/delay → courtesy alone fails; R1 partial gain, no300s unroll.
33. R2 exits-in-training; R3 negotiation+survival/300s teacher curriculum+learned insertion; R4 brake-wave → regional/paired/walls → retention/stuck remains, R3 paired regresses; admission v0 improves but service drifts.
34. R5 threshold/first-passage/yield/extra features; R6 frailty; R7 secure-gap core; R8 sequence/on-policy normalization → merge likelihood/support/timing → led to pivot redesign, not training-J qualification.
35. R9 hybrid first passage/F2 left truncation/risk-retain/lane-end support/exclusions → event likelihood/stuck → fitted law supported, rollout motion/teacher bugs and closed-loop gap mismatch remain.
36. R9 motion one-step; R10 cut-in; R10b/R11 courtesy contrast → one-step/yield/walls → local targets do not establish long-run intervention fidelity; courtesy state gradients/intent later repaired.
37. R12/R12C physical learned settings/reinit/prior removal; R13C/E frozen-core residual/courtesy/temporal-off; R14 braking; R16 hybrid timing → state/safety gates → useful lineage, not proof of training-J accuracy.
38. R15 paired_long full-road regional contrasts/shared draws → road paired response → old terms t0-only, tiny (<1%), short bins; new gradient weights highly variable; prepared gate blocks unready launch.
39. PAIRFT12 corrected updates → road ΔJ/metering → adapter-invalid trial retained; corrected A300 MAE6633→6208 but B60 worse, feedback MAE54.1→55.4; KEEP OFF.
40. DEEP/LONGLOSS/COMBINED/LOSSONLY/mainline residual+lane unfreeze → old paired road gates → no qualified replacement; LOSSONLY selects exact step0, just2 long-loss updates.
41. MACROLOSS/MACROLONG regional30s speed/density/flow and lane variant → old long-road gates → step-starvation prompted extension; no training-J success established.
42. W1 longwm all-pairs high-rate FT → old road bias/sign+CCT → does_not_help, accept REJECT (62 vs59 overlaps).
43. W2 lr/10+EMA+batch stability → same → both candidates does_not_help, ACCEPT; gain-only H_G300 is not all-effect fidelity.
44. W3 lane-ST/unfreeze → gradient gate → initial zero path; later audit OOM, zero updates; no accuracy verdict.
45. W4 ramp/merge unfreeze; W5 lanes+ramp → same → both does_not_help; W4 validation MAE/r regression, W5 ACCEPT but no qualified gain.
46. CAPFIX/W6 leader connector/phantom repair → fit crossings+old gates → fit has no discharge drop; runtime repair retained candidate, neither trained checkpoint qualifies.
47. W7 expected lanes+jam ST offsets+group recipe/accumulation → same → best step0, final correlation regression; offsets/gradients≠successful response.
48. W8 standstill hold-time supervision (entrant NLL graph zero; onset unbuilt) → same → best step0, final sign/r regress; no qualified hold-duration rescue.
49. ERRSRC learned entrant mix/noise widths → old t0-HDV ΔJ → mix small kept option, noise no; real+68/WM−297 veh/h counters incompatible, not proven capacity mechanism.
50. WMREQUAL training-J+source → upstream dominates paired error, source benefits fictitious; WMREQUAL2 finds shared lane/native ownership confound; corrected report pending.
## Common failure pattern and binding exclusions
State/one-step/count or t0-HDV road proxies ≠ paired all-vehicle training-J incl. source; absolute fits hide response/cancellation.
Tiny/inactive terms, zero/saturated/indirect gradients, few updates, bundled recipes and global timing/scalar tweaks recur.
Bookkeeping/ownership/exits/admission contracts must be fixed before interpreting model error; association is not capacity causation.
Do not repeat hard training clamps, uncontrolled bundles, driver_params scoring, vetoed pushforward, or assumed-live agents.
