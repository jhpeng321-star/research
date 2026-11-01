# Zero-variance CI numerical correction — 2026-11-01 ~16:52 +08

First INTERIM published/committed5512f402 at16:46:50: two matched K8 worlds
969436/969590 in demand1900/penetration.05. Delay v5−v4 points120/300/600 are
+205.6/+2333.2/+11427.9veh-s; HDV60−8.7m. All point estimates stay unchanged.

Review found an analysis-only floating-point defect: `std(draws,axis=0)` on10000
identical nonbinary floats in a multicolumn matrix can accumulate a spurious
mean-reduction error >1e-14. Constant direction-accuracy columns consequently
printed equal-endpoint confidence intervals instead of zero-estimable-variance NA.
Those constant-column16:46intervals are superseded; retained original receipt.

Repair centers the draws at their point before estimating variance in uncertainty,
credibility ratios and gate adapter; guards numerical zero relative to the metric
scale (64machine-epsilon), not a behavioral/calibration threshold. No point,
forecast, model, source-data, bootstrap weights, estimand or selection changed.
36CPU tests PASS9.01s; Ruff PASS, including exact nonbinary constant regression,
large constant-cost support and actual Windows PID/creation-time/exit binding.

## Explicit local-continuation transition

No remote supervisor or GPU process is touched. Original collector9660 detects
changed analysis hashes at its next ordinary poll and safely stops before new
collection; original cache-only publisher24320 already exited on its source guard.
Their receipts/cache/forecasts/scores are preserved. `resume_ci.py` binds their
exact saved creation times, waits for BOTH exact expected STOPs and actual exits
without signals, then starts read-only collectorV3 and cache-only publisherV2
ONCE. It reuses the same validated cache, transferring only newly complete worlds.
No overlapping collectors, forecast retries or remote dispatch. Any unexpected
STOP, existing successor root or source change prevents a successor launch.

The corrected publisher refreshes the first available replicated view, immediately
publishes complete reserved8 when ready, then updates each~3h. Its final progress-
report guard handover now binds collectorV3. All original final registered gates,
four models, six contrasts and source/recipe/input hashes remain unchanged.
