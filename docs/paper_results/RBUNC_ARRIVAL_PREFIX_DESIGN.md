# Frozen causal arrival-prefix check (2026-11-01 continuation)

Purpose: check the learned entrance-count uncertainty using existing observable
FIT counters, independently of traffic rollout error. No new recordings, driver
labels, SUMO, inference workers or training. Preserve all earlier diagnostics.

Before running: use only `rbunc/observed_arrivals.json`, exactly native FIT worlds
969400–969431, 30 ten-second bins/world. Sum bins 0–150s as causal evidence and
predict the disjoint 150–300s aggregate separately for mainline/ramp. The midpoint
is an analysis split, not a behavioral threshold. No alternate cut is selected.

Frozen forecast: the implemented Jeffreys Gamma(count+0.5, rate=150s) posterior
with a conditional Poisson clock gives an exact negative-binomial count law.
Compare with a Poisson at the SAME posterior mean to isolate rate uncertainty;
report plug-in MLE Poisson separately. Neither comparator changes the runtime.

Metrics: predictive mean/variance, mean count negative log likelihood and discrete
CRPS, central90% interval coverage plus its exact discrete nominal mass, midpoint
PIT diagnostic, and mean standardized squared residual. Paired whole-world
bootstrap CIs use10000resamples/seed20261101 and keep both entrances together.
All32per-world forecasts/targets and source/code hashes are retained. Numerical
tests compare the Gamma–Poisson PMF with independent quadrature and discrete CRPS
with the pairwise-expectation identity. Truncation error is numerical only.

Interpretation: this is a FIT-only diagnostic of request counters. Family choice
was already examined on these worlds, so it is not independent qualification.
Small32-world samples cannot justify extra dispersion by themselves. Do not tune
hidden-driver spread to across-world traffic SD or change physical driver support
from this count check. Any arrival-law change needs explicit source-family and
causal counter evidence plus a separately frozen comparison.
