# S-LRDIAG-1 live handoff (not the final report)

Latest checkpoint: **93b6f409** contains LRDIAG.md (520 words at that point) and
the complete primary E3 CSV/figure package. Both CPU input audits passed all2151
staged-file SHA checks. All288 full-class output upgrades passed with exactly0
J/speed/count difference; fullfield timeline is complete (36 worlds/288 draws),
class-delay accounting max7.25e-5veh-s. Pulse repairs are now COMPLETE96/96 with
zero retry failures; pulse aggregation passed (29952 amplitudes/1152 chain gains),
world gain tables and supplemental figure inspected. Only pairs300 remains
running (recent P647+ / Q743+ pool completions out of1296 each, includes144 upgrades).
Direct SSH banner handshakes began timing out; authorized `-J gpuR` / SCP
`-o ProxyJump=gpuR` works without GPU use or project writes. `gather.py --jump`
supports this route. See latest progress receipts rather than old counts below.

## Rules and location
- E:/agent-work/c4-lrdiag, c4/lrdiag, base b74837d0. Commits so far: 95b829ad, 535946d8, 0b31d9e7; later tool additions are being checkpointed.
- Analysis only: no SUMO, training, GPU inference, push, reset/clean/stash. All model inference CPU on cpuP/cpuQ under ~/lrdiag, <=24 active workers, collection yield. Own explicit PID/start-token/parent checks only.
- Never open prohibited paths or driver-type truth/assignment files. Only observable CAV/HDV classes and kinematics/actions are used. Verdict numbers exclusively E3 validation (36 worlds).
- Shared registration already appended first, with subsequent method/compute receipts, at E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md.

## Implemented / running
- CPU copies contain E3 validation ONLY: native/mappo/ours_c1, 108 episodes. WM2 hash b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436. Model load 171 tensors, zero new/unused.
- E3 core complete: 36 worlds x 6 variants x K8 = 1728 batch-three rollouts, 300 s / 600 ticks. Source in ~/lrdiag/results. Primary objective is the registered gate's t0-HDV speed-deficit J, arm-native, 31.29 m/s reference.
- Dense attempt archived in ~/lrdiag/dense_reference (24 complete units/box). Supported compact-attention path used consistently in final K8; short parity passed. Numerical controls retained, no mixed dense samples in final statistics.
- Production loader real masks are t0-cohort masks. Full-traffic breakdown is recomputed from raw converted arrays including arrivals, valid mapped lanes, using preregistered BIC dominant downward step+trend vs constant/linear, 60 s baseline, 10 s bins, no speed threshold. Majority confusion separate from expected event probability.
- Full-road CAV/HDV timeline needed dynamic entrant classes. Existing pairs300 coordinator first upgrades baseline E3 metadata with SAME WM/seeds, preserving original J and all-class speed/count bins; original trace NPZs unchanged. Old/new SHA receipts under fullclass_upgrades; old JSONs archived. Version 2 adds full-road class speed/count/delay bins. fullfield.py outputs the arrival-inclusive timeline, separate from t0-cohort timelines.
- Core owners: cpuP 1763026 / cpuQ 1633451 (ended). Continuations: P 1763027 / Q 1633452. Current P pairs-pool owner 1987213; always read current OWNER/PAIRS_OWNER + start ticks before any control. No other job/process has been altered.
- Continuations run pulses (P only), then baseline class-output upgrades + pairs300 in the SAME <=24-worker pool. Fixed pairs panel = 48 groups, sample seed 20261026, from WMCOMBINED score_pairs. Expected raw p300 results 2304; pool jobs include 144 metadata upgrades per host plus 1152 pairs jobs.
- gather.py merges Q parity into P via SSH, incremental SHA checks, atomic writes, hash-bound class-metadata upgrades. Two initial long transfers reset; owned partials preserved/repaired. Current NPZs successfully read by reducer.
- analyze_owned.py pauses one verified owned worker/subtree, runs statistics/plots in its active slot, then resumes; yields to collection. Use modes partial/complete/pairs/pulses/figures/verify. Reducer caches NPZ members (old inefficient stats PID 1832965 was explicitly stopped by owned token; no model process affected).

## Established E3 validation results (K8, 36 worlds)
- Baseline pooled 300 s slope .0575464 [−.35935,.68435], sign .666667, bias +165.711, MAE 217.954 veh-s. MAPPO slope .246775; ours −.211918. At 120 s pooled slope .557031, bias +67.499.
- Crucial decomposition: slope = Pearson .0822458 x SD ratio .6996885. Mean |predicted dJ| / mean |real dJ| .69437. Predicted mean +50.568 vs real −115.143. Low regression slope is mostly wrong worldwise alignment, not a 17x physical gain loss.
- Native upstream real event .777778; WM expected probability .961806, gap CI [.06597,.31944]. Majority TP/FN/FP/TN = 28/0/7/1. Onset MAE 65.54 s, bias +38.75 s. Merge real .722222 vs WM .805556, onset MAE 61.6 s. WM selected steps are larger, not absent.
- Best OFF point estimate: ramp_adapt_off pooled slope .373186 [.20713,.74669], bias +132.459, MAE 166.984, sign unchanged .666667. Pearson .677120; SD ratio FALLS to .551137. MAPPO slope .489435, ours .206525.
- Paired slope-toward-one improvement CI [−.17366,.58913]; bias reduction 33.252 CI [−29.327,112.371]; MAE reduction 50.970 CI [−10.884,129.786]. Promising target, NOT a confirmed improvement/adoption.
- Other pooled 300 s slopes: physics-only residual OFF −.630624; population defaults −.094007; courtesy OFF −.241400; yield OFF −.452527. All worsen calibration; keep learned stack rather than turn it off wholesale.
- Ramp module source inspected: src/wms/model/ramp_speed_adapt.py. Three global learned scalars, deceleration-only virtual-leader constraint, selects nearest mainline front at/ahead (including negative bumper gap alongside). Cannot express signed gap-targeting / acceleration into a gap ahead of a yielding CAV.
- Structural protocol confound: _device_timeline pads entrant CAV commands with native. Recorded post-t0 arrivals carry 99.60% of MAPPO upstream direct-CAV vehicle-time in 120–300 s (ours 99.85%). This is missing exposure, not a measured causal fraction of bias. Correct identity/controller continuation BEFORE fitting more long paired losses.
- t0-HDVs: 300 s upstream paired errors +113.10 MAPPO / +123.33 ours veh-s. CAV errors appear early; full-road timeline will separate these from cohort exits. Paired RMS is not uniformly attenuated around 90 s; late fixed-cohort fade partly reflects exits. RMS of ensemble-mean response and mean path RMS both retained, with support and survival.

## Pulse qualification repair
- Recorded pulses exist on P at ~/c4/runs/pulse/{jobs,src/all}, seeds 969824–969835, 30 s, 3 targets, pulse_m/pulse_p/nopulse. Outside E3 validation: SUPPLEMENTAL ONLY, no numeric verdict from them.
- Initial 32/96 units succeeded; 64 failed only an overly broad initial-position alignment assertion including unmapped internal-lane coordinates. For failed seeds 969825/969834, mapped initial s/v differences are exactly zero; unmapped counts 1/2.
- pulses.py now checks the valid mapped initial domain, matching the actual following platoon domain; dynamics unchanged. pulse_retry.py repairs one failed unit at a time AFTER a P pairs rollout in that existing slot (parent waits, one serial child only). New replay.py hook installed on P; process cleanup handles own child only. Monitor PULSE_RETRY_QUEUE / CHILD / DONE; need 96 complete results, no retry failures. Do not treat original PULSE_DONE failures as final.

## Next steps
1. Inspect fullclass_upgrades and FULLFIELD_COVERAGE after latest gather/reducer; expect all 288 baseline units version 2, scalar/count parity passing. Full-field output required, not just cohort timeline.
2. Monitor PULSE_RETRY_*; investigate remaining failures if any, without SUMO/truth labels. Run pulse_summary once 96 complete.
3. Let both pairs pools finish; inspect PAIRS_DONE.failed (ALL_DONE alone does not prove success). Re-run only failed owned analysis units if needed. No background process-name killing.
4. python tools/lrdiag/gather.py locally; then use analyze_owned.py complete / pairs / pulses / figures / verify on P. Full field needs fullfield.py deployed (done). Read latest stats logs/returncodes. CI bootstrap worlds, never K draws.
5. Fetch ~/lrdiag/evidence CSV/JSON/SVG/PNG into new evidence_lrdiag/ in worktree, inspect PNGs via Read, verify source/data hashes and counts; LRDIAG.md <=600 words.
6. Plain verdict: over-damped/no-breakdown explanation is unsupported as the main mechanism; strict linear string stability not identified by E3 isolated pulses (none). Rank ramp/mainline mis-coupling (best direct ablation), missing CAV action continuity (certain confound), and timing/cohort/exposure errors. Do not claim ramp OFF statistically confirmed.
7. ONE promising learned candidate: flag-gated signed, gap-/CAV-intent/history-aware ramp response v2 replacing the rigid decel-only virtual leader; preserve other learned components. Input contract must include correct entrant controller/actions. Proposed quick offline check: recorded-ramp longitudinal-acceleration oracle with CAV/entry context held correct; assess 300 s slope/correlation/bias and upstream timeline. No training or new SUMO in this task.
8. Append completion receipt at actual shared-registration tail, commit intended artifacts after each step, no push. Final chat <=8 lines, self-contained, links to report/evidence.
