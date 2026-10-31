# S-WMREQUAL3 — corrected speed-only/native-lane qualification
Final audited evidence SHA: `79b75b27ed7159727fa9267ff3af8983747cbe44`; report delivery SHA: `git log -1 --format=%H -- docs/paper_results/WMREQUAL3.md`.
**Speed-only/native-lane accuracy is now MEASURED and remains NOT QUALIFIED on this recorded validation contract; the earlier failure was not solely a lane-ownership artefact.**
Matched arms: **ours_c1** (`64f70af4`) and **coopoff_c1** (`e1fa4881`), each 36 worlds; 72 matched arm−native contrasts. All recorded checkpoint hashes match; both policies declare residual-on-native, native lanes, no lane head.
**MAPPO is separate, unmatched diagnostic only:** its checkpoint (`aecd7d70`) has a lane head, but the packed original timeline lacks lane requests. Its original speeds now run with native lanes; entrant policy crossings remain explicit. Its 36 contrasts cannot qualify either the native-lane contract or genuine MAPPO lane control.
Repair `fc69fcfd`: default-OFF scorer switch; ON delegates original/native-fallback ramp stays and native discretionary lanes, with cohort masks composed correctly. CPU: frozen/default/explicit-OFF eight-row 2 s equality, 44/44 original ramp CAVs delegated in every arm (21 worlds), HDV-twin first-passage equality, native entrant targets and explicit-crossing preservation pass.
WM2 `b8797242`, 36 worlds, **K8, mix ON only**, 600 half-second ticks; 1,152 raw rows. Original longitudinal commands are replayed; synthetic entrants use same-arm feedback. Both original and entrant native references now have native lane ownership.
J=all-present original/entrant road speed-deficit delay + source waiting, veh-s; paired K8 median, not the complete weighted suffix/group training credit. Errors=predicted−recorded arm−native ΔJ; zero recorded effects excluded from sign denominators only.
## Accuracy by horizon
Each cell=sign agreement / MAE veh-s / Pearson r / bias veh-s. n=36 per arm, 72 matched; MAPPO is never included in the matched column.
| h s | ours_c1 | coopoff_c1 | Matched native lanes | MAPPO diagnostic |
|---:|---|---|---|---|
| 60 | 0.571 / 52.2 / −0.001 / +11.4 | 0.571 / 52.5 / −0.000 / +11.9 | 0.571 / 52.4 / −0.001 / +11.7 | 0.571 / 50.5 / −0.346 / +45.7 |
| 90 | 0.444 / 148.6 / −0.263 / +41.4 | 0.444 / 148.9 / −0.261 / +42.4 | 0.444 / 148.8 / −0.262 / +41.9 | 0.472 / 143.1 / −0.442 / +128.1 |
| 120 | 0.556 / 279.1 / −0.357 / +77.7 | 0.556 / 281.0 / −0.357 / +80.4 | 0.556 / 280.0 / −0.357 / +79.0 | 0.472 / 282.7 / −0.273 / +243.6 |
| 180 | 0.500 / 588.0 / −0.215 / +157.8 | 0.500 / 604.3 / −0.203 / +171.2 | 0.500 / 596.2 / −0.209 / +164.5 | 0.500 / 713.6 / +0.077 / +588.7 |
| 300 | 0.583 / 1388.3 / −0.241 / +359.9 | 0.528 / 1364.9 / −0.228 / +275.4 | 0.556 / 1376.6 / −0.235 / +317.7 | 0.556 / 1593.4 / +0.179 / +1125.5 |
Matched J300 MAE falls **24.3%**, 1818.0→1376.6; paired world-bootstrap MAE-change 95% CI **[−970.9,+10.9] veh-s**, so improvement is not statistically decisive. J120/J180 MAE changes +0.9%/−1.1%; long-horizon correlations remain negative.
## Confident gain/harm calls
Conditional σ=0 diagnostic: median gain/harm>0 AND paired K8 q10(gain/harm)>0. Cells=G count:correct/nonzero-real calls; H likewise. Zero-real calls count toward coverage only; “—” means no calls.
| h s | ours_c1 G; H | coopoff_c1 G; H | Matched G; H | MAPPO diagnostic G; H |
|---:|---|---|---|---|
| 60 | 17:15/16; 11:2/11 | 17:15/16; 11:2/11 | 34:30/32; 22:4/22 | 0:—; 25:14/25 |
| 90 | 13:10/13; 7:2/7 | 13:10/13; 7:2/7 | 26:20/26; 14:4/14 | 0:—; 22:10/22 |
| 120 | 14:9/14; 7:2/7 | 14:9/14; 8:2/8 | 28:18/28; 15:4/15 | 0:—; 19:8/19 |
| 180 | 4:2/4; 5:1/5 | 4:2/4; 4:0/4 | 8:4/8; 9:1/9 | 1:1/1; 9:3/9 |
| 300 | 3:2/3; 2:0/2 | 4:2/4; 2:0/2 | 7:4/7; 4:0/4 | 1:1/1; 8:2/8 |
Matched gain accuracy [95% world-bootstrap CI] at 120/180/300 s: .643 [.385,.889] / .500 [.000,1.000] / .571 [.000,1.000]. Bootstrap=2,000 whole-world draws, seed 0; per-arm gain/harm CIs and wrong-call magnitudes are retained.
Matched conditional CCT-1-style diagnostics fail all horizons; conditional CCT-2-style gain diagnostics pass only 60 s and fail 90/120/180/300 s. At 180/300 s there are only 8/7 gain calls (<20), with 4/8 and 4/7 correct; all four J300 harm calls are wrong.
Training perturbation σ remains unmeasured; these σ=0 calls do not supply it or the registered pooled training gate. No training horizon authority is created; operational H_G=0.
## Additive source/upstream/downstream errors
S=source; U=upstream/merge road s≤75.44 m; D=downstream road. Tuples=bias / MAE veh-s / covariance share %. Shared total-J median order-statistic weights preserve additivity; shares=100 E[e_component e_J]/E[e_J²], not causal fractions; negative shares mean cancellation.
| h s | ours_c1: S; U; D | coopoff_c1: S; U; D | Matched: S; U; D | MAPPO diagnostic: S; U; D |
|---:|---|---|---|---|
| 60 | S −38.4/38.4/−7.1%; U +62.8/76.8/138.5%; D −13.0/28.9/−31.5% | S −38.6/38.6/−8.2%; U +63.8/79.8/140.9%; D −13.3/28.6/−32.7% | S −38.5/38.5/−7.7%; U +63.3/78.3/139.7%; D −13.1/28.7/−32.1% | S +0.5/3.0/1.1%; U +60.9/75.0/144.2%; D −15.6/38.7/−45.4% |
| 90 | S −90.8/90.8/−8.6%; U +141.2/191.0/120.1%; D −9.1/47.0/−11.5% | S −88.2/88.2/−7.9%; U +141.9/192.0/119.3%; D −11.3/44.8/−11.4% | S −89.5/89.5/−8.3%; U +141.6/191.5/119.7%; D −10.2/45.9/−11.4% | S +2.5/7.1/1.6%; U +143.9/161.3/113.3%; D −18.4/50.9/−14.9% |
| 120 | S −145.6/148.2/−7.8%; U +236.8/320.6/110.6%; D −13.5/63.8/−2.8% | S −145.4/148.0/−8.6%; U +239.4/323.9/112.2%; D −13.6/63.4/−3.6% | S −145.5/148.1/−8.2%; U +238.1/322.2/111.4%; D −13.6/63.6/−3.2% | S −1.2/12.2/−0.5%; U +249.2/288.4/93.9%; D −4.4/88.7/6.6% |
| 180 | S −318.4/322.4/−8.9%; U +392.6/641.2/103.1%; D +83.7/147.8/5.8% | S −309.6/313.7/−10.9%; U +396.2/635.4/104.7%; D +84.7/149.7/6.2% | S −314.0/318.0/−9.9%; U +394.4/638.3/103.9%; D +84.2/148.8/6.0% | S +3.1/30.3/−0.4%; U +496.4/632.3/80.8%; D +89.1/192.1/19.5% |
| 300 | S −866.2/866.6/−6.9%; U +1062.2/1529.7/96.9%; D +163.9/324.5/10.0% | S −875.5/875.9/−7.3%; U +1000.0/1465.5/95.0%; D +151.0/309.5/12.2% | S −870.9/871.3/−7.1%; U +1031.1/1497.6/96.0%; D +157.4/317.0/11.1% | S −2.3/107.4/0.2%; U +837.0/1397.9/87.8%; D +290.9/410.2/12.0% |
The corrected matched upstream/merge component still contributes **96.0%** of J300 squared-error covariance; source/downstream contribute −7.1%/+11.1%. Ownership repair changes the error pattern but does not remove the upstream failure.
## Corrected verdict and next model fix
Speed-only native-lane control is **no longer unmeasured** for these recorded longitudinal interventions: corrected J120/J180/J300 sign agreement is .556/.500/.556, and confident gain/harm decisions remain unreliable. This is not a qualification of every possible closed-loop native-lane policy.
**Further WM accuracy repair is still needed. Merge-discharge/queue-propagation remains the supported next model-side target, but its specific necessity is not causally established by this scoring-only repair.** Residual localization justifies testing it; no unconditional capacity scalar or source-bias offset is justified.
GPU: gpuA only after room checks; retained batch wall-time 2.63 h, sampled total-owned peak **11.40 GB ≤12 GB**. Two allocator-OOM attempts are preserved; exact-verified pair-bias row chunking completes the panel without changing K8/CRNs. All eight owned process incarnations are finished; zero owned GPU allocations; no pilot process actions.
Verification: CPU ownership/44-CAV/frozen-OFF checks; corrected serial/grouped CPU parity; six bitwise GPU pair-bias checks plus exact 10 s corrected rollout; five summary tests; `python -B tools/wmrequal3/analyze.py --verify`; `python -B tools/wmrequal3/verify_report.py` checks numeric tables, hashes, ≤60 lines and released GPU ownership.
Evidence: `tools/wmrequal3/evidence/{CPU,PANEL_AUDIT,BIAS_PARITY,ANALYSIS,FINISH}.json`, all 36 raw validation batches and memory/PID receipts; execution details in `tools/wmrequal3/AUDIT.md`. No SUMO, training, excluded-input access, assignment/truth reads, deletions or push.
