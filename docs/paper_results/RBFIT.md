# S-RB-FIT — observable-motion physical parameter learning
**Delivery rechecked2026-11-04: IDM exported; current-source LC fitted/verified, but exact-event fidelity remains inadequate.**

## Contract
- Owner WM-PHYS registration5683, FIT5703/addendum5736; branch `c4/rb-fit`, base `f0db3c82`; all prior work retained.
- Only native FIT worlds969400–969431; six allowlisted observable files, hashes in `artifacts/rbfit/*/provenance.json`.
- Fit [240,300]s:229,839 mainline acceleration observations,2,956 fitted HDVs;3,130 driver records incl174 population fallbacks;1,154 have full60s motion histories.
- Validation [300,360]s:232,648 acceleration observations;9,758/8,303 stable-leader5s/10s windows. Leaders/lanes teacher-forced; endpoint positions actually recorded.
- Exclude identity/lane/leader changes, connectors, negative bumper gaps; stopped traffic retained. Ramp-only CF histories use population, not an invented stop leader.

## IDM: fits and validation
- SUMO1.27.1 IDM substeps0.25s, observation step0.5s, delta4 fixed, emergencyDecel9; no black-box training or SUMO runs.
- All-row pooled LS candidate `pooled_full_v5/idm_pooled.json`:v0=29.839994m/s,tau=.770429s,a=.795080,b=1.772102m/s²,minGap=1.933667m.
- Later-FIT acceleration RMSE **.2991 vs .9025** initial settings; MAE .1980 vs .7071m/s²; bias +.000002m/s².
- Pooled5s/10s position MAE **1.95/5.58m**, RMSE2.80/8.16m; constant-speed MAE3.96/13.81m, RMSE6.75/23.96m.
- Gaussian empirical-Bayes population + full-covariance driver Laplace posteriors are preserved in `gaussian_v1/`; driver10s MAE4.16m improves, RMSE8.35m worsens.
- `finite_mixture_v4/`:1,026 motion-derived numerical support profiles before exact-zero removal,949 retained; population masses/residual scale by discrete-sequence marginal-likelihood EM, exact finite-support driver Bayes.
- Finite population q10–q90:v0 **27.29–34.17m/s**,tau **.780–1.558s**,a **.399–1.497**,b **.776–3.599m/s²**,minGap **1.084–2.685m**. Heterogeneity inferred without labels; finite support is an approximation, not globally identified true settings.
- Physical-profile acceleration CRPS population/posterior **.1656/.1635**, vs pooled initial-fit point MAE .1975; expected-acceleration RMSE .3567/.3778m/s².
- Nominal95% physical-profile coverage **95.0% population /68.8% posterior**. Driver posterior is underdispersed; no calibrated conditional uncertainty claim or automatic adoption.
- NEW `shrinkage_v1/`: learn prior_count **19.35568** by physical-acceleration CRPS, prefix240..270→270..300; posterior fraction n/(n+prior_count), zero evidence=population. No action labels; shared pre-t0 empirical-Bayes population makes this internal calibration.
- Frozen later-FIT shrunk/raw/population CRPS **.14657/.16351/.16560**, expected-acceleration RMSE **.3478/.3778/.3567m/s²**; shrunk nominal95% coverage **90.48%**, improved but still under nominal. All3,130 posterior records materialized for UNC,174 zero-evidence fallbacks.
- Unrestricted and finite-variance Student-t attempts scale-collapsed; verified own PIDs stopped, STOP receipts/partial evidence preserved, neither selected.

## LC2013
- NEW `lane_v2/`: source-backed highway subset `c4/rb-lane` commit811cd7e8, source SHA1bdded8e…;65 numerical profiles of lcStrategic/Cooperative/SpeedGain/KeepRight/Assertive, conditional on pooled_full_v5 IDM; earlier lane_v1 retained.
- Fit3,130 drivers/658 exact-time lane events in253,468 at-risk intervals; observed positions/gaps define decision geometry; persistent wishes start at zero at240s and reset on observed changes.
- Population parameters are a fitted discrete joint mixture; per-driver posterior weights retain aliases where histories cannot identify parameters. Observation-emission nuisance probabilities are learned, not runtime driving rules.
- Later-FIT626 events/258,691 intervals:posterior/population Brier **.002828/.003071**, event-rate baseline **.002414**; log loss .01715/.01895 vs .01700. Improved from lane_v1 posterior.003059, but **LC fit not adopted**.
- NEW `lane_directional_v3/`: source-conditioned stay/target0/target1 likelihood fixes Boolean wrong-destination matches; causal informative-evidence shrinkage. All3,130 posterior sequences independently verified;1,630 uninformative histories=population. Later-FIT event Brier posterior/population **.002359/.002463**; raw physical proposals still7811vs626observed, versus source-default12095; NOT ADOPTED. Original support permits cooperative>1; v4 corrects domain.
- Latest `lane_raw_5a5b15ef118c/`: LANE80541278/SHAe61f897c5e7a,snapshotv2,209profiles,3,130posteriors/1,599uninformative fallbacks. Raw LS Brier.0023075,435.655proposals/60.983exact directed matches vs626events:precision14.0%,recall9.74%; default2.12%/42.01%. NOT ADOPTED.

## Delivery / limits
- All paths above are under `artifacts/rbfit/`; formats/API in `HANDOFF.md`, runtime `src/wms/physics/motion_fit.py`, reproduction in `tools/rbfit/README.md`. NEW `condition_motion` conditions fixed profiles on causal60s at arbitrary starts without guessing tape VIDs; real FIT weight parity5.6e−17.
- `FittedHiddenState.from_directory('artifacts/rbfit/shrinkage_v1')` in UNC commitf8b6faea reads the exact schema directly for incumbents+entrants; policy already applied. Raw finite_mixture_v4 remains; draw joint profiles ONCE per future/identity, never tick noise.
- Prior22CPU tests+11current-source numerical tests PASS. Latest LC export:3,130actual UNC driver draws+16paired/replayed planned futures PASS; independent posterior error≤2.06e-15. Raw population puts94.55% mass atlcKeepRight0; temperature.265641 learned from causal prefix, already applied; no runtime emission noise.
- v0 maps to speedFactor by division by actual lane limit; shared CF compatibility note records NUMERICAL_EPS=.001/emergencyDecel9 source defaults.
- These are conditional held-in FIT engineering results; no unseen-world, intervention, free-running breakdown, queue-clearing or action-value improvement claim. Pooled point candidate is ready for INT/EVAL testing.
- SUMOCORE saved-six-FIT-state diagnosis: inference changes600s MAPPO/PI effects by **+4171/+3909veh-s**, almost entirely road; native road improves8523 vs controlled4351/4617veh-s. Incumbents+entrant donors changed together; culpable parameter unidentified. See `RBFIT_SUMOCORE_DIAG.md`; no demonstrated sign repair.
- Fresh2026-11-04 resume3c3588bc:29local hashes PASS (`receipts/DELIVERY_MIRROR_RESUME_3c3588bc.json`); committed LANE80541278 unchanged. Peer worktree paths unavailable; committed handoffs read via Git. Numerical/UNC and remote completion checks remain historical; no repeated fit/inference/SUMO/GPU/training.
