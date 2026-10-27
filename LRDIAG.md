# S-LRDIAG — long-run response diagnosis

**Status:** E3 validation analysis complete; supplemental pulse/pairs300 jobs are finishing.

[CSV/figures](evidence_lrdiag/) · [definitions and reproduction](tools/lrdiag/README.md).

**Verdict: the stated over-damped / “no imagined breakdown” explanation is not supported as the main mechanism.** The WM produces excessive, mistimed speed-loss events and poorly aligned controller effects. Strict string stability is not identified by isolated E3-validation pulses: none were recorded there.

## Evidence and method

WM2 `b8797242`, **36 E3 validation worlds**, native/MAPPO/ours, matched warmup-end observations, K=8 coupled 300 s CPU rollouts. No SUMO, training, GPU inference, or driver-type truth. Primary dJ is the registered **t0-HDV speed-deficit proxy**, arm minus native; pooled slopes use OLS with intercept, CIs resample worlds. Per-world single-contrast “slopes” are explicitly paired ratios, not independently identified regressions.

**Breakdown:** COOPMEAS supplied no collapse detector. We registered a BIC-selected dominant downward speed step plus trend against constant/linear alternatives, using full recorded traffic, 10 s bins and 60 s history; no speed cutoff. The final 30 s are right-censored by the post-change support requirement. Native upstream occurrence is **77.8% real versus 96.2% WM expected**; majority TP/FN/FP/TN = **28/0/7/1**. True-positive onset MAE is **65.5 s**, signed error **+38.8 s**. WM drops are stronger, not absent. Merge occurrence/onsets and every world/arm/draw are in the CSVs.

**Response:** baseline pooled 300 s slope **0.058 [−0.359, 0.684]** decomposes as correlation **0.082 × SD ratio 0.700**. It is not a 17-fold physical amplitude loss. Predicted mean dJ is **+50.6**, real **−115.1 veh-s**. Fixed-cohort paired signals grow through approximately 90 s; late fade must be interpreted with the supplied survival/support curves. Qualification-pulse gains per vehicle/per 100 m are supplemental, outside validation, and do not determine this verdict.

**Switch-offs:** only ramp-speed adaptation OFF moves the pooled 300 s point slope toward 1: **0.373 [0.207, 0.747]**; MAPPO **0.489**, ours **0.207**. Bias falls **165.7→132.5**, MAE **218.0→167.0 veh-s**, sign stays **66.7%**. Importantly, SD ratio falls further, **0.700→0.551**. Paired improvement CIs include zero: this is not a confirmed fix. Other pooled slopes: physics-only **−0.631**, population defaults **−0.094**, courtesy OFF **−0.241**, yield OFF **−0.453**. Complete 120/300 s, per-arm and per-world results are provided.

**Timeline:** errors are evident before 120 s and increase strongly afterward. At 300 s, fixed-cohort upstream HDV errors are **+113/+123 veh-s** (MAPPO/ours); full-road HDV errors, arrivals included, are **+534/+585**. Separate 10 s region/class tables prevent confusing cohort exits with damping.

## Ranked mechanisms and one fix

1. **Wrong ramp↔mainline coupling:** the strongest model-side suspect. The current adapter has three shared scalars and only brakes toward the nearest adjacent-lane leader; removing it improves alignment while reducing amplitude.
2. **Missing controller continuity:** synthetic entrant CAVs receive native commands. Post-t0 arrivals supply **99.6%** of MAPPO’s recorded upstream direct-control vehicle-time during 120–300 s. This is missing-input exposure, not an identified causal share of bias.
3. **Phase/survival errors**, rather than uniformly decaying perturbations.

**One learned candidate:** flag-gated `ramp_response_v2`, a signed gap-/CAV-action/history-conditioned ramp-acceleration residual replacing the rigid brake-only adapter. Restore entrant identity/action continuity before fitting; use E3 fit worlds, calibration selection, validation verdict. **Quick offline check:** a recorded-ramp longitudinal-acceleration oracle, with CAV/entry exposure matched and learned lanes/merge retained; test 300 s correlation, paired bias and upstream error recovery. No global wave-gain increase is justified.
