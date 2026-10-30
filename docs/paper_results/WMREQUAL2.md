# S-WMREQUAL2 — MAIN 2026-10-30 22:55
Final audited analysis/evidence SHA: `9ce0b125971da601f74574573b31b94da588ee99`; report delivery SHA: `git log -1 --format=%H -- docs/paper_results/WMREQUAL2.md`.
**NOT QUALIFIED remains operationally correct; the specific SETUPAUDIT2 True-switch bug is absent, but a separate shared timeline/native-reference ownership mismatch prevents a clean WM or speed-only verdict.**
Scope: 36 validation worlds, 108 arm−native contrasts, 2,304 retained rows; WM2 b8797242, K8, mix ON below. Original rows and both-mix pooled statistics reproduce exactly on CPU.
Accrued J=all-present/entrant road speed-deficit delay + source waiting, veh-s; paired K8 median, not the complete weighted suffix/group training credit. Errors=predicted−recorded ΔJ.
## Execution-path audit
All three arms (`mappo`, `ours_c1`, `coopoff_c1`) use `TimelineControllers` for original CAVs; only synthetic entrants invoke `policy_feedback → make_cav_controller` (the policy/MAPPO-compatible bridge).
All three entrant bridges use **policy_ramp_merge=False**, native_ramp_merge=wm_law: ramp stays delegate to native first passage and explicit policy crossings keep ownership. No validation arm uses the audited True path; excluding that specific bug removes **zero** arms.
**Separate confirmed mismatch:** timeline callbacks hold original CAV lanes and never delegate merges; native-reference entrants also use this undelegated timeline fallback. Thus all three contrasts have a mismatched native reference, and original ramp CAVs are blocked in every arm.
CPU census: 44 original ramp CAVs in 21/36 worlds, **44/44 merge in each recorded arm** but none receives WM delegation. Original mainline vehicles with any recorded lane change: native 87, MAPPO 107, ours/coopoff 98 each; their lanes are also held in the scorer.
Eight CPU cohort/arm callback checks with hash-verified actual policy checkpoints confirm original delegation native/MAPPO/ours/coopoff=F/F/F/F, entrant delegation=F/T/T/T. Six frozen source hashes match the source manifest; no switch overrides in controller specs.
## Training-J accuracy by arm and exclusion sensitivity
Each cell=sign agreement / MAE veh-s / Pearson r / bias veh-s; n=36 per arm, 108 all, 72 without MAPPO. Zero recorded effects excluded from signs only. “No MAPPO” is a sensitivity, not a clean execution subset.
| h s | MAPPO | ours_c1 | coopoff_c1 | All / excluding specific True bug | No MAPPO |
|---:|---|---|---|---|---|
| 60 | .514 / 49.3 / −.509 / +41.6 | .600 / 52.4 / −.013 / +8.0 | .600 / 52.2 / −.014 / +8.2 | .571 / 51.3 / −.133 / +19.3 | .600 / 52.3 / −.013 / +8.1 |
| 90 | .472 / 136.2 / −.484 / +114.0 | .444 / 148.3 / −.181 / +33.6 | .444 / 147.3 / −.171 / +33.2 | .454 / 143.9 / −.239 / +60.2 | .444 / 147.8 / −.176 / +33.4 |
| 120 | .417 / 250.0 / −.230 / +200.5 | .528 / 278.4 / −.289 / +57.3 | .556 / 276.5 / −.282 / +56.7 | .500 / 268.3 / −.267 / +104.9 | .542 / 277.5 / −.286 / +57.0 |
| 180 | .528 / 544.3 / +.225 / +310.1 | .583 / 605.5 / −.255 / +57.0 | .583 / 600.4 / −.243 / +52.8 | .565 / 583.4 / −.115 / +140.0 | .583 / 603.0 / −.249 / +54.9 |
| 300 | .583 / 1572.7 / +.284 / −341.4 | .583 / 1806.4 / −.249 / −519.8 | .583 / 1829.6 / −.248 / −491.1 | .583 / 1736.2 / −.047 / −450.8 | .583 / 1818.0 / −.248 / −505.5 |
## Per-arm additive error decomposition
S=source; U=upstream/merge road s≤75.44 m; D=downstream road. Within each component: bias / MAE veh-s / covariance share %. Shared total-J median order-statistic weights make errors additive; shares=100 E[e_component e_J]/E[e_J²], not causal fractions; negative shares indicate cancellation.
| h s | MAPPO: S; U; D | ours_c1: S; U; D | coopoff_c1: S; U; D |
|---:|---|---|---|
| 60 | S +0.6/2.3/−0.3%; U +55.9/73.3/149.1%; D −14.9/34.0/−48.8% | S −40.2/40.2/−6.3%; U +60.5/78.3/147.5%; D −12.3/24.2/−41.2% | S −40.4/40.4/−6.1%; U +61.5/78.4/147.3%; D −12.9/24.4/−41.2% |
| 90 | S +1.1/5.8/1.2%; U +125.1/153.2/113.3%; D −12.3/47.3/−14.5% | S −85.9/86.0/−7.1%; U +129.8/174.0/121.1%; D −10.4/40.2/−13.9% | S −85.5/85.7/−6.3%; U +128.3/172.4/120.0%; D −9.5/39.3/−13.6% |
| 120 | S −2.2/16.0/0.3%; U +209.1/267.7/101.2%; D −6.4/70.2/−1.5% | S −146.8/149.2/−5.9%; U +194.9/299.5/107.9%; D +9.3/64.8/−2.0% | S −149.0/151.3/−6.2%; U +194.7/301.0/108.2%; D +11.0/64.1/−2.0% |
| 180 | S +14.2/35.1/2.5%; U +210.8/614.4/101.4%; D +85.0/183.2/−3.9% | S −333.3/333.8/−3.8%; U +276.9/673.1/98.9%; D +113.4/171.6/4.9% | S −328.6/329.1/−4.8%; U +269.2/688.6/100.4%; D +112.3/180.7/4.4% |
| 300 | S +15.6/118.1/1.6%; U −610.0/1796.1/112.4%; D +253.1/413.6/−14.0% | S −853.5/857.7/6.0%; U −67.9/2101.1/104.6%; D +401.5/527.8/−10.6% | S −865.5/869.8/5.3%; U −11.7/2058.2/104.1%; D +386.1/493.9/−9.5% |
Without MAPPO at 300 s: S −859.5/863.8/5.7%; U −39.8/2079.7/104.4%; D +393.8/510.9/−10.0%. MAPPO contributes only 25.0% of all-arm squared J300 error; removing it worsens MAE 4.7%.
## Confident-gain/harm calls and qualification
Conditional σ=0 diagnostic: median gain/harm>0 AND paired K8 q10(gain/harm)>0. Cells=G count:correct/nonzero-real calls; H count:correct/nonzero-real calls. Zero-real calls count toward coverage only; “—” means no calls. This does not supply training perturbation σ.
| h s | MAPPO G; H | ours_c1 G; H | coopoff_c1 G; H | All / excluding specific True bug G; H | No MAPPO G; H |
|---:|---|---|---|---|---|
| 60 | 0:—; 26:14/26 | 20:18/19; 7:1/7 | 20:18/19; 7:1/7 | 40:36/38; 40:16/40 | 40:36/38; 14:2/14 |
| 90 | 0:—; 23:10/23 | 16:12/16; 5:1/5 | 16:12/16; 5:1/5 | 32:24/32; 33:12/33 | 32:24/32; 10:2/10 |
| 120 | 0:—; 18:9/18 | 15:10/15; 6:1/6 | 15:10/15; 6:1/6 | 30:20/30; 30:11/30 | 30:20/30; 12:2/12 |
| 180 | 4:3/4; 8:3/8 | 13:9/13; 5:1/5 | 12:8/12; 5:1/5 | 29:20/29; 18:5/18 | 25:17/25; 10:2/10 |
| 300 | 15:13/15; 3:0/3 | 13:7/13; 3:0/3 | 13:7/13; 3:0/3 | 41:27/41; 9:0/9 | 26:14/26; 6:0/6 |
No-MAPPO gain accuracy [95% world-bootstrap CI] at 120/180/300 s: .667 [.429,.900] / .680 [.389,.920] / .538 [.286,.818]; 2,000 world draws, seed 0. Per-arm gain/harm CIs, coverage and wrong-call magnitudes are retained in ANALYSIS.json.
Both all-arm and no-MAPPO conditional CCT-1 fail every horizon; conditional CCT-2 fails 120/180/300. MAPPO J300 gain accuracy .867 has only 15 calls (<20). No registered pooled training H_G is measurable: perturbation σ remains unknown and pairs300 K8 remains blocked; operational H_G=0.
Excluding **all arms exposed to the confirmed blocked-original/native-reference ownership path leaves n=0**; its qualification statistics are unavailable, not a pass. The 15 worlds without original ramp CAVs still use the mismatched native-entrant/reference path.
## Corrected interpretation and rescore decision
The verdict is **not driven solely by MAPPO or SETUPAUDIT2's specific True-switch bug**: that path is absent and no-MAPPO scores still fail. Whether the newly confirmed shared execution bug drives the upstream error is unmeasured without corrected rollouts.
**Speed-only with genuinely native lanes: UNMEASURED / NOT AUTHORIZED**, rather than proven inaccurate. Lane-held original playback is not native-lane control, and the recorded policy-arm outcomes cannot label a different speed-only intervention.
No specific-True-path arms exist to rescore with policy_strategic_merge repair ON; that bridge-only switch cannot repair timeline/native-reference ownership. No corrected K8 effect is claimed. Historical dual-mix panel cost 4.01 GPU h; no new GPU allocation or rollout was launched.
**Merge-discharge WM fix: still a candidate, not established as necessary.** First fix original/native-entrant ramp delegation and native-reference discretionary lanes, preserve explicit crossings, then requalify a matched native-lane contract; only residual error can justify the learned discharge/queue-propagation fix. Prior upstream-error localization remains descriptive, not causal proof.
Evidence: `tools/wmrequal2/evidence/{RUNTIME,RUNTIME_TEXT,ANALYSIS,CPU_PROBE}.json`; details in `tools/wmrequal2/AUDIT.md`. CPU probe 9.87 s/two threads; no training/SUMO, excluded-input access, GPU use, pilot process operations, deletions or push.
Verification: `python -B tools/wmrequal2/analyze.py --verify` reproduces raw contrasts, original aggregates and new analysis exactly; five summary tests and eight CPU ownership cases pass. `python -B tools/wmrequal2/verify_report.py` checks all numeric tables, ≤60 lines, four scripts and frozen-source/fast-adapter receipt hashes.
