# S-RB-FIT — why better traffic fields can reverse action effects

## Measured localization (saved forecasts; no new simulation)
- Input: S-SUMOCORE `scores/forecasts.json`, SHA bound in
  `artifacts/rbfit/sumocore_diagnosis_v1/diagnosis.json`.
- ONLY six FIT worlds969421/422/425/426/427/428; native/MAPPO/PI,
  both settings variants, paired RNG101/202/303:108 saved600s futures.
  No non-FIT world contributes to the numbers below or any fit.
- 600s MAPPO-minus-native delay: default **−1164**, inferred **+3007veh-s**.
  Inference changes the predicted effect by **+4171veh-s**: road **+4172**,
  source **−1.64**. PI: **−3294→+615**, change **+3909**: road **+3907**,
  source **+2.44veh-s**. Source waiting is NOT the immediate effect-sign cause.
- Inference reduces native road delay by **8523veh-s**, but reduces MAPPO/PI
  road delay by only **4351/4617**. The native baseline improves more than the
  controlled branches: better absolute road forecasts need not preserve gains.
- Inference increases absolute source delay by about **11703veh-s** in EVERY
  arm; this common error largely cancels in paired effects. Do not fit a global
  delay offset or select settings solely on absolute forecast CRPS.
- All six leave-one-world-out mean changes remain positive: MAPPO1653..6342,
  PI2046..4806veh-s. These are descriptive influence checks, NOT independent
  samples or an exclusion of the collision-flagged969425 world.

## Source audit / what remains unidentified
- `tools/sumocore/inference.py:12–19` maps v0/31.29 to speedFactor once and
  preserves tau/minGap/accel/decel; MOBIL settings are not mapped to LC2013.
- `prepare_bundle.py:86–103` changes BOTH incumbent HDV settings and future
  HDV donor settings. Thus the ablation does not isolate encoder error from
  entrant-population error or identify a culpable IDM coordinate.
- LC stays fixed; nonlinear leader/lane/junction interactions amplify changed
  car-following. Initialization collisions remain a validity defect. Neither
  these saved forecasts nor one-step fitting establishes the causal mechanism.

## FIT response, chosen without action labels
- Existing exact finite posterior undercovers later motion (68.8% nominal95%).
  Correlated residuals and unmodeled constraints are plausible reasons for
  overconfidence; their shares are unidentified, not hidden-type recovery.
- New shrinkage uses pre300 motion only: learn a population-contamination
  strength with prefix240..270→270..300 physical-acceleration CRPS; zero usable
  evidence means the learned population exactly. No behavioral threshold/noise.
- Exports preserve joint physical support; missing/weak evidence cannot force
  an off-support encoder point. Population donors also cover future entrants.
- This mitigates one identified inference failure mode. **Avoiding the600s
  sign reversal is NOT established** until INT/EVAL tests the fitted futures.
