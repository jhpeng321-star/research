# COOPAUDIT — measured cooperation in real SUMO

S-COOPMEASURE v2 COMPLETE; exploratory TRAIN audit, owner2026-10-31 15:45/17:20.
D(S)=J(native)-J(S); C_G=D_GROUP-sum_i D_ONLY:i; positive C_G is beneficial interaction.
Native means no acceleration/lane request; disabled members use IDM. All vehicles released after60s.
Each branch starts a fresh SUMO process and reruns the same native300s prefix; NO save/loadState.
Frozen observable pair at t0=300s, zone[-150,75.44]m; corrected upstream grid7/pair60s adopted.
MAPPO iter169 d9042cc5; wall-v2.1-closure; fixed final SUMOCF u15 e4b51f64 (deployed mean, no noise).
TRAIN969650-969677:28 worlds,24 high-share(.2/.4 x2400/2700)+4 coverage;21 eligible pairs,20 high-share.
Seven missing groups remain missing; no replacement/truncation. Only3/21 pairs have feasible wall geometry.
All28 allocation-weighted CIs are UNDEFINED (missing strata); never treat missing cooperation as zero.
Below: conditional eligible21-world means and95% stratified world-bootstrap CIs; eligible-cell weights.
5000 paired world draws, seed20261031; units veh-s. Raw strict share C_G>0 is reported separately.
J_H integrates fleet+source delay over H seconds AFTER t0 (J300/J600 end at absolute600/900s).

## Team J600
- mappo: C_G +67.679 [-1179.061,+1154.090]; share>0=0.381; D_G +893.181; sumD_i +825.502; n21.
- wall: C_G -0.000 [-0.001,+0.000]; share>0=0.000; D_G -0.475; sumD_i -0.475; n21.
- sumocf: C_G +67.757 [-1178.903,+1154.094]; share>0=0.429; D_G +893.160; sumD_i +825.402; n21.
## Team J300
- mappo: C_G +116.945 [-197.765,+441.431]; share>0=0.333; D_G +382.083; sumD_i +265.138; n21.
- wall: C_G -0.000 [-0.001,+0.000]; share>0=0.000; D_G -0.422; sumD_i -0.422; n21.
- sumocf: C_G +117.023 [-197.463,+441.509]; share>0=0.381; D_G +382.062; sumD_i +265.039; n21.
## Merge-local J at window+120s
- mappo: C_G -15.645 [-97.251,+82.718]; share>0=0.238; D_G -31.953; sumD_i -16.308; n21.
- wall: C_G -0.000 [-0.000,+0.000]; share>0=0.000; D_G -0.423; sumD_i -0.422; n21.
- sumocf: C_G -15.567 [-97.135,+82.821]; share>0=0.286; D_G -31.974; sumD_i -16.408; n21.

## Prespecified high-share coverage and pairing
- High-share eligible20/24, mappo: C_G600 +71.063 [-1238.014,+1211.794] (conditional eligible worlds).
- High-share eligible20/24, wall: C_G600 -0.000 [-0.001,+0.000] (conditional eligible worlds).
- High-share eligible20/24, sumocf: C_G600 +71.146 [-1237.847,+1211.799] (conditional eligible worlds).
- Paired SUMOCF-MAPPO delta C_G, 300: +0.078 [-0.001,+0.231]; n21.
- Paired SUMOCF-MAPPO delta C_G, 600: +0.078 [-0.001,+0.231]; n21.
- Paired SUMOCF-MAPPO delta C_G, local: +0.078 [-0.001,+0.231]; n21.

## Safety, mechanics and audit trail
- native GROUP/B across21 worlds: TTC<3 8289.5; hard-braking 286.5veh-s; collision/teleport/drop=0/0/0.
- mappo GROUP/B across21 worlds: TTC<3 7808.0; hard-braking 282.5veh-s; collision/teleport/drop=0/0/0.
- wall GROUP/B across21 worlds: TTC<3 8289.5; hard-braking 286.5veh-s; collision/teleport/drop=0/0/0.
- sumocf GROUP/B across21 worlds: TTC<3 7807.0; hard-braking 282.5veh-s; collision/teleport/drop=0/0/0.
- Corrected canary16 full900s reruns: native/no-op/three-controller repeats bitwise; wall106commands, assembly307s/handback353s.
- Canary wall C_G600+1658.239758 (D_G+1616.622358, sumD_i-41.617400); MAPPO/SUMOCF C_G0. One mechanics world, NOT panel inference.
- Main231 full-prefix gates PASS;247 including canary. Historical saved-state outcomes excluded; all prior data retained.
- Budget: 243480actual/243780charged of500000sim-s;300 unused setup reservation retained; no simulator retries/refunds.
- Local cohort: vehicles observed in s[0,75.44] during window+120s, their road+source delay at that endpoint.
- C_G, D_G, sumD_i, raw branches/safety, all cells/coverage, paired CIs and numerical-null counts: tools/coopaudit/evidence/v2_analysis/.
- Traces: gpuR /root/autodl-tmp/coopaudit20261101_v2/v2_sSEED/traces/*.jsonl.gz; members+nearby ramp vehicles,(270,390]s/0.5s.
- Portable SHA-verified trace copies: evidence/v2_complete/jobs/; TRACE_MANIFEST.csv and tools/coopaudit/TRACE_HANDOFF.md.
- Design AUDIT_DESIGN.md; upstream ARTIFACT_PASS372d3326/probe420a4904; CPU26/26PASS; canary commitd58db191.
- Exposure disclosure: initial legacy parameter scan opened forbidden truth JSON; no values used; evidence/EXPOSURE_NOTE.md. No such reads in v2.
- Interpret the conditional C_G CIs directly: fleet gain alone is not synergy; a fixed-pair null does not exclude cooperation in other groups/windows.
