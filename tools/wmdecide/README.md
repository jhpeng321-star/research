# WM-PHYS probabilistic decisions and credibility

CPU-only, NumPy-only tooling. No simulator, model inference, fitting, driver-label
access or training. Team delay is full-population road delay + source waiting in
veh-s on the evaluator's registered horizon and clock convention.

## Picker API (also for S-C1SWITCH2)

```python
from tools.wmdecide import Future, decide

# Producer supplies identical hidden-driver and future-arrival draws for all actions.
actions = {
    "incumbent": [Future("draw0", 100., False), Future("draw1", 200., True)],
    "candidate": [Future("draw0", 90., False), Future("draw1", 180., True)],
}
choice = decide(actions, incumbent="incumbent")
```

`summarize(actions)` reports mean/SD/quantiles/range/Monte Carlo SE of delay,
breakdown probability, and ordered paired action comparisons. It aligns by
`draw_id`, rejects duplicate/missing/nonfinite futures, and never drops failures.
Unknown breakdown labels yield an unknown probability, not a reduced-K estimate.
Actual common-random-number coupling is the producer's responsibility: identical
ID strings alone cannot prove identical draws. IDs must identify the full shared
hidden-driver/arrival realization and action-independent random streams.

**Registered switch rule:** switch only when strict paired
`P(candidate delay < incumbent delay) >= 0.8` **and** the candidate has lower
expected delay. Select the lowest expected-delay eligible candidate; lexical
action name breaks equal candidate means. Otherwise retain the incumbent.
Ties count as neither win; no epsilon, dwell time or behavioural threshold.
The **0.8 is the owner's registered confidence level**, not a fitted parameter.
For K8 this requires at least 7/8 strict wins. It does not mean an 80% confidence
interval on an expected-value difference, nor establish real-world calibration.
MC SE measures finite-future sampling only, not shared dynamics-model error.

Verify with `python -B -m unittest tools.wmdecide.test_decision -v`.
