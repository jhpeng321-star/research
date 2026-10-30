# S-WMREQUAL — 2026-10-30, owner 22:30 cutoff
Final audited scoring/analysis SHA: `8347672e93cea663d1b2580513c097b2e23985af`; documentation delivery SHA: `git log -1 --format=%H -- docs/paper_results/WMREQUAL.md`.
**WM2+mix is NOT qualified for the training objective. Measured pooled H_G is unavailable; operational H_G=0. Upstream/merge road error dominates validation paired-J error.**
## Scope, completion and contract
Validation: complete 36 starts/36 worlds, 108 arm-minus-native contrasts, 2,304 raw rows (native+3 arms, K8, mix ON/OFF); completed work reused.
pairs300: 48 starts/20 worlds, 96 contrasts, exact real J reconstructed; **new K8 scoring blocked**: all 144 tapes lack per-entrance Q0 (total Q0=30..220 is known), recorded identity-name maps absent, metadata-derived originals unavailable. No guessed split or substitute panel.
Calibration/PANEL120 were skipped in this final analysis. Eight previously completed calibration scoring batches and all earlier real labels retained; excluded chain stopped, not resumed.
WM2 `b8797242`, config `f95f7c4c`, coupled seed 140014, K8 driver scale=1/embedding SD=.05, physical exits, learned entrant mix ON/OFF; frozen WM, no optimizer/SUMO. Native reference only; no recorded CTRL0 contrasts.
J=all-present/entrant road speed-deficit delay + source waiting, veh-s, no vehicle divisor; paired median over 8 futures, q10 of paired gain for gate. Scored target is accrued J, not the whole dense grouped-credit functional.
Source labels: independent `damping_requests_v1:<seed>:<stream>` departure-time RNG + recorded append-only identity registry; total Q=due−inserted, t=300 per-source counters independently checked. No assignment/driver-truth inputs. Real source integral uses S-GAPCHECK's left-boundary 0.5 s rectangles, not continuous-departure ledger interpolation. Late-start total Q remains exact; missing per-source Q0 prevents initializing the WM entrance dynamics.
## Side-by-side objective statistics (mix ON)
Each tuple = sign agreement / MAE / Pearson r / mean signed error; errors=predicted−recorded ΔJ, veh-s. Zero real effects excluded from signs only.
| h s | Validation old t0-HDV road, archived K1 | Validation training J, new K8 median |
|---:|---|---|
| 60 | .438 / 27.8 / .325 / +19.2 | .571 / 51.3 / −.133 / +19.3 |
| 90 | .500 / 71.3 / .370 / +47.3 | .454 / 143.9 / −.239 / +60.2 |
| 120 | .481 / 108.3 / .390 / +66.3 | .500 / 268.3 / −.267 / +104.9 |
| 180 | .639 / 148.3 / .400 / +90.8 | .565 / 583.4 / −.115 / +140.0 |
| 300 | .657 / 192.5 / .014 / +123.5 | .583 / 1736.2 / −.047 / −450.8 |
pairs300 retains the old contract only; its recorded training-objective effects are still measurable:
| h s | pairs300 old K1: sign / MAE / r / bias | Recorded mean training ΔJ veh-s (new WM unavailable) |
|---:|---|---:|
| 60 | .938 / 64.5 / .799 / +57.4 | +168.8 |
| 90 | .896 / 150.6 / .689 / +145.5 | +254.1 |
| 120 | .677 / 244.6 / .643 / +236.6 | +303.5 |
| 180 | .656 / 331.0 / .609 / +312.7 | +393.2 |
| 300 | .688 / 390.4 / .522 / +369.8 | +421.8 |
The first table changes K/runtime/exit handling as well as objective: it is not a cost-only ablation. Same new K8 rollout's t0-HDV 300 s sign/MAE/r=.407/543.7/−.237. Training J300 OFF MAE=1832.7; ON improves it 5.3%.
## Noise floor and confident calls
Recomputed validation equivalent-control contrast (ours_c1−coopoff_c1): n=36, training-J SD=0 and 2σ=0 at **all five horizons**. These controls are outcome-equivalent, not independent null perturbations.
Training-objective perturbation σ and its 2σ threshold are **unmeasured**: neither registered panel contains null-perturbation arms. Old σ below is an inherited sensitivity scale, not training σ; WM residual SD is not noise.
Old CCT-2 uses ΔJ<−2σ; training-style diagnostic below uses median gain>0 AND joint-K8 q10(gain)>0 (conditional σ=0). Bootstrap: 2,000 world draws, seed 0; counts include zero-real calls, agreement excludes them.
| h s | Old 2σ veh-s | Old pooled K1 gain count / correct fraction | Validation K8 σ=0 gains / correct [95% CI] | K8 harms / correct |
|---:|---:|---|---|---|
| 60 | 19.00 | 4 / .250 | 40 / .947 [.842,1.000] | 40 / .400 |
| 90 | 50.76 | 9 / .667 | 32 / .750 [.533,.941] | 33 / .364 |
| 120 | 82.44 | 7 / .857 | 30 / .667 [.429,.900] | 30 / .367 |
| 180 | 84.22 | 26 / .962 | 29 / .690 [.417,.913] | 18 / .278 |
| 300 | 87.82 | 31 / .968 | 41 / .659 [.458,.857] | 9 / .000 |
Old CCT-1 fails both panels at 90/120/180/300; old pooled CCT-2 H_G=300. New validation median-only CCT-1 at σ=0 fails every horizon (108 confident; sign as first table). Exact coverage/wrong-call sizes/CIs are in SUMMARY.json.
New σ=0 gain-only checks fail at 120/180/300; these are **not a registered pooled verdict**. Borrowing old σ gives validation 120 s 20 gains/.800 but 180 s 29/.690 and 300 s 41/.659; this sensitivity cannot confer authority.
## Source waiting and total entrance queue prediction
Tuples=bias / MAE / r. Absolute predictions=K8 means on intervention arms; paired changes use the total-J median's same two order-statistic draws. Source waiting W is veh-s; queue Q is vehicles; NA=zero variance.
| h s | Absolute W ON | Absolute W OFF | ΔW ON | ΔW OFF |
|---:|---|---|---|---|
| 60 | +143.3/283.7/.976 | +141.4/282.9/.976 | −26.7/27.6/.065 | −26.5/26.9/.068 |
| 90 | +272.3/506.9/.966 | +266.2/504.9/.966 | −56.8/59.2/.068 | −59.1/60.1/.081 |
| 120 | +461.6/800.6/.957 | +449.7/799.6/.956 | −99.3/105.5/.078 | −98.9/102.5/.078 |
| 180 | +1008.4/1674.5/.936 | +982.8/1673.5/.936 | −215.9/232.7/.072 | −218.9/229.8/.083 |
| 300 | +2843.0/4352.8/.896 | +2788.4/4348.8/.895 | −567.8/615.2/.097 | −579.5/598.7/.114 |

| h s | Absolute Q ON | Absolute Q OFF | ΔQ ON | ΔQ OFF |
|---:|---|---|---|---|
| 60 | +4.17/7.40/.944 | +4.05/7.39/.943 | −.95/.99/NA | −.88/.88/NA |
| 90 | +5.15/9.32/.924 | +4.99/9.23/.924 | −1.12/1.34/NA | −1.26/1.37/NA |
| 120 | +7.61/11.79/.912 | +7.42/11.70/.912 | −1.51/1.77/NA | −1.52/1.69/NA |
| 180 | +11.02/17.81/.861 | +10.76/17.69/.860 | −2.33/2.53/NA | −2.28/2.43/NA |
| 300 | +19.38/27.05/.843 | +19.23/27.06/.842 | −3.72/4.00/.070 | −3.68/3.87/.073 |
Real ΔW is nonzero in only 1/108 contrasts through 180 s and 10/108 at 300 s; real ΔQ is zero through 180 s and nonzero in only 1/108 at 300 s. High absolute correlation does not imply intervention fidelity.
Per demand/penetration at 300 s, tuples again bias/MAE/r; each cell has 3 worlds/9 contrasts:
| demand/share | ΔW ON | ΔW OFF | ΔQ ON | ΔQ OFF |
|---|---|---|---|---|
| 1900/.05 | −59.5/97.5/NA | −93.1/94.0/NA | −.72/.83/NA | −.67/.67/NA |
| 1900/.10 | −130.4/143.7/NA | −199.1/201.0/NA | −.78/.78/NA | −1.28/1.28/NA |
| 1900/.20 | −742.5/742.5/.34 | −567.6/578.1/.28 | −4.78/4.89/NA | −3.89/3.89/NA |
| 1900/.40 | −589.7/600.6/.24 | −453.8/493.2/.21 | −3.61/3.61/.19 | −2.78/3.00/.18 |
| 2400/.05 | −147.0/259.3/NA | −182.1/212.3/NA | −1.33/2.11/NA | −1.39/1.61/NA |
| 2400/.10 | −333.8/342.7/NA | −472.4/511.3/NA | −2.06/2.06/NA | −2.39/3.06/NA |
| 2400/.20 | −738.6/780.3/−.24 | −649.2/649.2/−.49 | −4.11/4.33/NA | −4.28/4.39/NA |
| 2400/.40 | −1523.9/1642.4/.05 | −1589.6/1589.6/.22 | −10.11/10.67/NA | −10.06/10.06/NA |
| 2700/.05 | −172.2/179.4/−.65 | −278.0/278.0/−.39 | −.78/1.00/NA | −1.33/1.44/NA |
| 2700/.10 | −285.6/351.7/.36 | −325.9/395.3/.35 | −2.17/2.83/NA | −1.83/2.61/NA |
| 2700/.20 | −747.9/788.1/.41 | −720.4/746.6/.48 | −5.17/5.17/NA | −4.56/4.67/NA |
| 2700/.40 | −1342.4/1454.3/−.40 | −1422.6/1436.3/−.48 | −9.00/9.78/NA | −9.67/9.78/NA |
All horizons' per-cell absolute W/Q and paired W/Q bias/MAE/r for both mix settings are retained in `tools/wmrequal/evidence/SUMMARY.json` → panels.validation.cells; tiny/constant real effects make cell correlations descriptive only.
## Merge discharge and additive error decomposition (validation, mix ON)
Disjoint source + road at s≤75.44 m (upstream/merge) + downstream sum exactly using common total-J median weights. Each component tuple=bias/MAE/100×E[e_component e_J]/E[e_J²]; negative shares mean cancellation, not negative causal error.
| h s | Source veh-s/% | Upstream/merge veh-s/% | Downstream veh-s/% | J MAE if upstream exact | J MAE if source exact |
|---:|---|---|---|---:|---:|
| 120 | −99/105/−4.4% | +200/289/106.2% | +5/66/−1.8% | 135.5 | 274.9 |
| 180 | −216/233/−2.7% | +252/659/100.1% | +104/179/2.6% | 266.4 | 615.8 |
| 300 | −568/615/4.6% | −230/1985/106.4% | +347/478/−11.0% | 689.6 | 1786.5 |
Discharge-error-only OLS explains 48.3/47.3/68.4% of centered paired-J error variance at 120/180/300 s; incremental discharge R² given source=51.1/48.0/68.4 percentage points. Source alone explains .08/.17/~0%; association is not causal attribution to capacity drop.
Full-flow continuous-identity crossings give mean real/WM arm−native discharge changes +1.08/−1.05, +2.59/+4.06, +1.00/+13.01 vehicles at 120/180/300 s. These full-horizon intervention contrasts do not re-estimate ERRSRC's onset-aligned +68 vs −297 veh/h drop.
Source waiting dominates absolute cost in some starts, but **upstream/merge road error dominates paired training-J error here**. Perfect source correction alone worsens aggregate J MAE; mix ON hardly fixes source sensitivity.
## Next concrete WM fix and delivery
Build an observable merge-discharge/queue-propagation correction with identity-bracketed full-flow crossing targets, upstream cumulative-delay supervision and per-entrance conservation (due−inserted=queue); constrain intervention responses jointly, rather than applying an unconditional capacity-drop scalar or source-bias offset.
Expected direction: reduce upstream paired-delay/discharge-response errors and fictitious source benefits. Perfect upstream-only correction gives a diagnostic ceiling of 49/54/60% J-MAE reduction at 120/180/300 s, **not a forecast**. Recover recorded per-source Q0 and matched perturbation σ before rerunning pooled CCT-2; no authority issued.
Evidence: `tools/wmrequal/evidence/{SUMMARY,PAIRS300_BLOCKER,OWNER_2230_STOP,GPU_RELEASE}.json`, `validation/`, `SOURCE.json`; `python -B tools/wmrequal/summarize.py` reproduces analysis. Checks: 3 label + 5 summary tests pass; 6 changed Python files parse; earlier 3 CPU mix/parity checks retained.
gpuA: exact owned parent 938486/child 975471 stopped; all 8 receipt-owned process incarnations absent on final audit, cap=.85. No training, SUMO, peer process operation, deletions or push; prior work and excluded completed rows preserved.
