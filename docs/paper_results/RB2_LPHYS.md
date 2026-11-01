# S-RB2-LPHYS — completed development ablation, 2026-11-01
**Engineering ready; no useful congestion/credibility repair established. Keep default OFF.**

## Fixed-form learned parameters
- Base: frozen FULL-v4 physics22cc32f6; no SUMO execution, black-box training or acceleration residuals.
- `physics_params='learned_v1'` + `physics_params_file='artifacts/rb2lphys/learned_v1.json'`.
- `configs/wm_phys_lp.toml` differs from baseline by exactly those two fields; INT has FULL-v5-LP integration.
- Three population probabilities select documented SUMO departLane first/free and departSpeed last/max.
- Learned first-lane=0, mainline-last=0, ramp-last=0.011075188190315551; persistent ordinal-keyed CRN draws.
- FIT/LANE retain IDM/five-LC parameter ownership; CAL handles physical spread. No duplicate refit/network.
- GP Bayesian optimisation:57trials,16common support draws,1068native FIT32 births240..300s.
- Objective: equal-world/entrance normalized birth-speed CRPS +lane/birth Brier; long-run metrics are checks.
- Later native development12 birth objective0.08274163→0.08279937 (worse),2059births300..600s.
- Converted views stop600s; the original900s requested bound is corrected by the support audit, not a refit.

## Complete autonomous DEVELOPMENT before→after
- All12worlds969420..969431, fivearms, K2subset of identicalK8 plans, H600;48tasks/240arm-futures COMPLETE.
- Delay CRPS120/300/600:841.09→841.43 /3705.55→3696.28 /14034.18→14024.29veh-s.
- 600s mean delay bias12820.87→12816.69veh-s; gains are tiny relative to the unresolved bias.
- Upstream breakdown300/600:94.17→94.17% /100→100%; recorded61.67% /90.00% on this development subset.
- 300s mean cell×arm rate distance40.83→40.83pp; no breakdown repair.
- 600s upstream speed CRPS2.68130→2.69049m/s; onset IBS0.386389→0.386944 (both worse).
- 600s LC/world329.642→329.617 vs264.333real; LC count CRPS57.7208→57.7250.
- LC10s count CRPS2.40632→2.40264; ramp-lane road loss2031.34→2032.56veh-s.
- Ramp-source wait20.9003→21.0836veh-s; full ramp-origin delay remains unscored, not conflated with lane loss.
- Mean600s MAPPO/PI/CIRCLES effects−388/−920/+2826→−396/−1176/+2886; real−1296/−3107/+793veh-s.
- MAPPO120/300 mean point signs remain wrong. K2delay range coverage600s21.67→20.00%; no credibility fix.
- Narrow delay/rate point screen passes; broader scores are mixed. Singleton-cell95%CIs are NA; NOT QUALIFIED.

## Delivery / exposure / checks
- Before/after table:`docs/rb2lphys/comparison/before_after.csv`; full scores/effects/audit in that directory.
- Artifact/code audit:`docs/rb2lphys/DELIVERY_AUDIT_FINAL.json`; hashes distinguish LF vs evaluated CRLF.
- New fit/diagnosis: native969400..969431 only; selection: development12/fivearms only; no other new TRAIN tapes.
- Reserved969432..969439/WMDATA2 never fit/selection inputs; inherited reference-bundle rows excluded from scores.
- 31CPU PASS/1CUDA skip;14remote checks PASS; all48tasks finite/conserved/CRN verified; learned corrections OFF.
- gpuR coordinator893753 and all own children EXITED;13:14GPU0%/1MiB. No peer PID/file action.
- Implementation throughf6cf52ab; numerical stages06eaf9b3/1c5722e9. Integrated FULL-v5 qualification remains INT-owned.
