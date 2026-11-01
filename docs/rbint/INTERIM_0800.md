# ~08:00 interim — complete reserved8 FULL-v4, switching rule FAIL

Delay and60s position predictions improve,but breakdown calibration does not.
This fails the owner's three-part switching rule on point comparisons,
independently of the uncertainty limitation. Do not adopt from CRPS alone.

- Immutable08publication:`noon/20261101T075846+0800/PUBLICATION.json`;
  collection started07:58:46 and finished after08:00. FULL-v4/source22cc32f6
  reserved969432–969439,fivearms,K8,all64tasks complete. Baseline originalWM2.
- Pooled five-arm absolute delay CRPS120/300/600,WM2→FULL-v4:
  1499→1105,6817→3777,24752→13105veh-s. Paired changes−393/−3040/−11647.
- Pooled upstream300s rate:real85.0%,WM285.625%,FULL-v491.25%.
  Absolute distance worsens+5.625pp.120/600s rates remain in the full report.
- HDV60 mean-world-median absolute position error change−15.26m on1237common
  t0-HDV/arm endpoints. Original-ID censoring and cohort attrition retained.
- Full-world95%CIs unavailable:singleton scored cells have no within-cell
  replication. Detailed mixed-cell conditional intervals are retained as
  diagnostics,not promoted to full-panel confidence claims. Points unchanged.
- FULL-v4600s mean-delay bias still+14648veh-s. Native upstream speed CRPS2.83
  vs cell2.11m/s,worse; native onset IBS.215 vs.286,better. Clearing unqualified.
- Selected>=.8calls observed win52.9/55.6/70.0% versus forecast92.6/92.4/93.1%
  at120/300/600s. Absolute range coverage45.0/65.0/57.5%. Not calibrated action
  confidence,and these saved-state calls are not executed policy outcomes.
- A2/B2/FULL-v3 each have all8reserved worlds; C2 currently1complete world.
  Earlier4015b641 A2/B2/C2 and latest FULL-v4 differ in engineering as well as
  settings; they are not a controlled settings-only ablation. No outcome tuning.
- SUMOCORE remains a separate8world/3arm/K3 panel with18startup collision flags,
  not a clean information ceiling. Integrated ramp-delay target is unavailable;
  the native FIT diagnostic is not substituted for this panel.
- Updated57-line WMPHYS_MORNING.md explicitly retains08interim and all three
  switching comparisons.10CPU report tests PASS1.18s. No inference repeated.
- Own FULL-v1/v3 stopped only after preserved reserved8 completion; A/B/C and
  latest FULL-v4 continue. Keep09:30engineering freeze/11:30own bounds/11:45report.
- Collector11932 acknowledged owner pause after08publication. Its frozen-source
  guard will stop before another analysis after this renderer update; successor
  resumes this committed publication/cache. No inference process is affected.
