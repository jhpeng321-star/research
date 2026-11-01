# S-RB2-CAL integration contract

## Physical latent sampler v1

Implementation/evidence source `0711a02646bb014a785e190f24e0838460ed8e23`;
base FULL-v4 `22cc32f65a80af482a1a277c5f22377fa727d0a3`.

- New runtime `src/wms/control/spread_calibration.py`.
- Three-line opt-in `BirthMix` hook in `src/wms/control/hidden_state.py`.
- Recipe loader and evaluation identity hooks in `tools/rbint/{recipe,evaluate}.py`.
- New `configs/wm_phys_rb2_cal_v1.toml` and frozen learned artifact
  `docs/paper_results/rb2cal/v1/fit/calibration.json`.
- Switch `variants.FULL.spread_calibration_json`. Omit it for the original
  adapter. A retention=1/rate variance=1/fixed-population-infinity artifact is
  draw-for-draw equivalent to the original tested path.
- Population, shrinkage and arrival-family byte hashes must match. Causal motion
  conditioning is mandatory. No saved-driver identities can carry into a start.
- Present-driver settings are drawn once by observable VID. Entrant mixture
  weights are drawn once per future/world; entrance ordinals couple admission
  settings to the admitted driver's actual motion settings.
- All nine original parameter coordinates and all original CAV settings are
  preserved except the five explicitly fitted joint IDM coordinates for HDVs.
  LC fit remains OFF, learned trajectory/admission corrections remain OFF.

The fitted concentration is **zero prior pseudocount**, not zero entrant
variance: the Dirichlet has the causal drivers' soft posterior counts as its
shape. No evidence with zero prior count falls back explicitly to the fixed
population. Infinity is serialized as null. No independent-coordinate noise.

The arrival variance multiplier applies to the **Gamma intensity variance**, not
the variance of total request counts: Poisson counting variance remains. It
preserves `(N+0.5)/exposure` and uses shape `(N+0.5)/multiplier`, scale
`multiplier/exposure`. Multiplier=0 means a point intensity, not zero arrivals.

## Decision probability reporting v1 (separate opt-in)

`tools/rbcal/probability.py` provides `calibrated_decide` and an offline CLI.
It preserves strict ties, expected costs, future draw IDs and the original 0.8
gate. It does not change or manufacture physical futures.

FULL-v4's development-fitted map is in `probability_fullv4_v1/calibration.json`.
It is bound to **FULL-v4's exact model contract**, not to the new physical v1
candidate. Its full-fit lambdas0.55325/0.35578/0.53994 imply maximum no-tie
probabilities0.77662/0.67789/0.76997, so no action can pass0.8. This is a measured
lack of reliable switch support, not a useful-policy qualification.

After integrating other workers, rerun on development forecasts and fit a NEW
map. Supply the exact source fingerprint to the API; changed sources are refused:

```python
from tools.rbcal.probability import calibrated_decide, fingerprint

choice = calibrated_decide(actions, incumbent, horizon_s=600,
    calibration=artifact, source_fingerprint=fingerprint(producing_model_contract))
```

CLI after producing a development-only S-RB-DECIDE observation report:

```text
python -B -m tools.rbcal.probability --report REPORT/physical --out NEW_MAP
```

## Evaluation and exposure

- Development snapshots admit EXACT `w969420..969431_k0..7.json` names before
  opening tasks. Do not point the generic scorer directly at a mixed source
  directory during fitting.
- `tools/rbcal/outcomes.py` uses the registered evaluator validator and DECIDE
  metrics. Its factual reader stops at the first reserved seed header, before
  decoding that episode's outcome. Hashing the full factual file binds bytes;
  it does not make reserved outcomes available to parameter fitting.
- Scalar crossfits exclude the whole target world. FIT's population already
  used all32 worlds, so this is internal empirical Bayes, not unseen-world data.
- All five arms have now been exposed on development12, including MAPPO, PI and
  CIRCLES outcome order. These must not be called unseen-controller results.
- Owner-mandated morning reports exposed historical reserved aggregate results;
  no reserved raw future task or reserved episode outcome was decoded/scored by
  this worker. The final comparison must be source-fixed and separately audited.
- No full-world CI is published from singleton cells; SVG deciles have support
  counts and no CI bars. Replicate every scored demand×penetration×split cell
  using recorded worlds not used in this fit before claiming valid final CIs.

## Evidence pointers

- `DEVELOPMENT_BASELINE.md` + JSON: original physical model and whole-world OOF
  mean-preserving output-scale limitation experiment. Output-scaled numbers
  are NOT simulated futures and must never be used as a fixed-model candidate.
- `dev_fullv4_v1/{physical,diagnostic_oof_scale}/`: CRPS, coverage, selected-call
  Brier, reliability CSV/SVG and exact input manifest.
- `probability_fullv4_v1/`: source-bound full-fit map and OOF reliability evidence.
- `v1/fit/`: all learned physical parameters, objective coefficients, input
  hashes, prefix evidence and scalar crossfits; no reserved fit data.
- `COMPLETION.json` and `DEVELOPMENT_PHYSICAL_V1_CPU.md` are produced only after
  the finite collector sees all96 complete fresh physical tasks. Their absence
  means fresh physical outcome calibration has not completed.
