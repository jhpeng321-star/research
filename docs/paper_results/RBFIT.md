# S-RB-FIT — observable-motion physical parameter learning
**Complete; useful pooled-IDM candidate, uncertainty/LC qualification not established.**

## Contract
- Owner WM-PHYS amendment REGISTRATION.md:5694; branch `c4/rb-fit`, base `f0db3c82`.
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
- Unrestricted and finite-variance Student-t attempts scale-collapsed; verified own PIDs stopped, STOP receipts/partial evidence preserved, neither selected.

## LC2013
- Source-backed highway subset `c4/rb-lane` commitc6b1edab, source SHA7fd41b35…;65 numerical profiles of lcStrategic/Cooperative/SpeedGain/KeepRight/Assertive, conditional on motion-fitted pooled IDM.
- Fit3,130 drivers/658 exact-time lane events in253,468 at-risk intervals; observed positions/gaps define decision geometry; persistent wishes start at zero at240s and reset on observed changes.
- Population parameters are a fitted discrete joint mixture; per-driver posterior weights retain aliases where histories cannot identify parameters. Observation-emission nuisance probabilities are learned, not runtime driving rules.
- Later-FIT626 events/258,691 intervals:posterior/population Brier **.003059/.003300**, event-rate baseline **.002414**; log loss .01846/.02023 vs .01700. **LC fit not adopted.**

## Delivery / limits
- All paths above are under `artifacts/rbfit/`; formats, causal-time guards and API in `HANDOFF.md`, runtime `src/wms/physics/motion_fit.py`, reproduction in `tools/rbfit/README.md`.
- Finite driver posterior is lossless UTF-8JSON gzip (32.8MB); original104MB JSON preserved locally/remote, not committed. Sample joint profiles ONCE per future/identity, never noise each tick.
- **15 CPU tests pass**, including independent SUMO numeric vectors, synthetic parameter recovery, causal identity guards, nonidentification and gzip runtime draws. Artifact SHA/schema/PSD/likelihood checks PASS.
- v0 maps to speedFactor by division by actual lane limit; shared CF compatibility note records NUMERICAL_EPS=.001/emergencyDecel9 source defaults.
- These are conditional held-in FIT engineering results; no unseen-world, intervention, free-running breakdown, queue-clearing or action-value improvement claim. Pooled point candidate is ready for INT/EVAL testing.
