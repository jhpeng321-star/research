# S-LRDIAG evidence

Primary numerical verdict evidence is S-E3DATA **validation only**, 36 worlds,
three arms, WM2 b8797242, K=8. Supplemental filenames beginning `pairs300_` or
`pulse_` are explicitly non-verdict and must not be pooled with E3 statistics.

- `ablation_summary.csv`: 120/300 s pooled and arm-specific bias, MAE, sign,
  slope, world-bootstrap CIs, correlation/SD decomposition, paired improvements.
- `ablation_worlds.csv`, `within_world_slopes.csv`: per-world contrasts and
  explicitly defined ratio/through-origin summaries; `ablation_draws.csv` retains K.
- `breakdown_worlds.csv`, `breakdown_draws.csv`, `breakdown_confusion.csv`:
  dominant BIC speed-loss events, probabilities, majority confusion, onset errors.
- `timeline_*`: the gate's fixed-t0-cohort, dynamic-region CAV/HDV timeline.
- `fullfield_timeline_*`: all mapped on-road traffic, arrivals included; source
  waiting excluded. Empty-class speed means retain support counts; class delay
  sums are defined even when empty.
- `paired_signal_*`: mean path RMS versus RMS of the ensemble-mean paired
  response, conditional support and initial-cohort survival. Exits are imputed at
  free speed for the fixed-cohort signal; conditional-survivor RMS is separate.
- `recorded_controller_coverage.csv`, `controller_coverage_summary.csv`:
  observed post-t0 direct-CAV exposure missing from synthetic-entrant command plans.
- `arm_delays.csv`, `ablation_regions_*`, `regional_summary.csv`,
  `horizon_curve.csv`: absolute t0-HDV values and regional attribution.
- `COVERAGE.json`, `FULLFIELD_COVERAGE.json`, `NUMERICS.json`,
  `INPUTS_VERIFIED.json`, `VERIFIED.json`: completion, invariants and SHA receipts.
- `pulse_amplitude_*`, `pulse_signal_draws.csv`, `pulse_gain_*`: 12 recorded
  qualification worlds, 30 s, K=8; amplitude along the initial following platoon,
  adjacent-vehicle / per-100 m gains and positive-amplitude log fits. World gains
  use median over K, then targets within world; support and exact-zero counts are
  retained. These worlds are outside E3 validation and are not verdict evidence.

Methods, execution guards and reproduction commands: `tools/lrdiag/README.md`
and `CONTRACT.md`. Primary J uses the registered speed-deficit proxy, not full-fleet
source-inclusive travel time. BIC is a dominant downward-step detector, not an
absolute jam-speed classifier or an asymptotic string-stability eigenvalue test.

This directory is refreshed as the remaining supplemental jobs complete. The
final verification receipt is authoritative for complete delivery.
Generated data/figure bytes are preserved by the directory's `.gitattributes`
so recorded SHA256 values remain portable across checkout line-ending settings.
