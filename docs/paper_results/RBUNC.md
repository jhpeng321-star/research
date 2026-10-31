# S-RB-UNC — hidden drivers and future demand, deterministic conditional physics

**Owner00:40 extension is complete:** [fitted posteriors + learned arrivals +
paired-action CRN](RBUNC_ADDENDUM.md). That report preserves a distinct raw-fitted
32-state diagnostic and identifies FIT's newer preferred shrinkage artifact.
The original encoder-input results below remain unchanged and historical.

## Result

**Implementation ready for WM-PHYS variant A. Source-attribution checks pass.**
The default-off adapter samples observer-hidden settings and future entrants;
it introduces no acceleration, embedding, lane-decision or merge-event noise.
This is not a distributional qualification or a matched WM2-versus-WM-PHYS win.

Code: `src/wms/control/hidden_state.py`. API/integration example:
`tools/rbunc/README.md`. Frozen diagnostic design: `RBUNC_DESIGN.md`.
Merge `c4/rb-unc`; integrator should retain its own root HANDOFF on conflict.
Useful increments: `1f918969` initial adapter, `ebea8483` physical support + FIT
probe, `9739d304` exact SUMO admission/newborn bridge, `73cb13be` bounded likelihood.

## What the switch does

- `plan_hidden_state_futures(..., future_sampling='off')` returns None without
  encoding or RNG work. `'hidden_state'` returns K `HiddenFuture` objects with
  `.kwargs` for the existing rollout and `.receipt` for provenance.
- Motion-inferred current HDV settings sampled once and persisted. Supplied
  calibrated `posterior_samples` are supported. The incumbent untrained
  `uncertainty` head is ignored rather than mislabeled a posterior.
- Fallback is an empirical joint population spread around the encoder estimate,
  weighted by a causal acceleration likelihood. Its residual scale comes from
  observed motion; plant bounds are the existing model bounds. The encoder's
  existing reliability smoothly mixes with the population for short histories;
  zero history gives a population draw exactly. No new behavioral cutoff.
- Future arrivals use independent entrance Poisson clocks at observed request
  demand rates, including the observable t0 queue. No future realization is read.
  Entrant settings are joint learned-population draws keyed by entrance/ordinal.
- Paired actions share hidden-state/demand streams. A per-future bridge ensures
  SUMO insertion tests precisely the settings subsequently used by newborn motion,
  even after MERGE installs its insertion adapter inside rollout.
- Existing checkpoint weights and shared wms/spec/factory/trainer files untouched.

## Recorded FIT verification

All **32 native FIT states969400–969431**, t0=300, authentic60s histories, K8,
H120. Frozen available encoder SHA `5b256941...`, the same encoder used by
S-SUMOCORE. It is not the main EVAL panel's `b8797242...` checkpoint binding.
Reference is deterministic legacy IDM/MOBIL with all learned corrections absent
and an explicit IDM CAV callback retaining current lanes. It is **not** the
integrated LC2013/SUMO-IDM efficacy model or its native execution-parity contract.

Final v2 **mean within-K8 SD**, versus **real native across-world SD**, veh-s:
- Team delay: **574.4 [506.6,644.9]** versus **4,200.5**.
- Road delay: **229.3 [202.1,257.6]** versus **1,140.4**.
- Source waiting: **490.4 [424.6,558.4]** versus **3,475.7**.

Brackets are 95% whole-world bootstrap CIs for the mean within-K SD,
10,000 resamples/seed20261031. Team RMS within-K SD608.8, range138.9–1060.6.
Across-world variability includes observable-state/demand heterogeneity;
it is **not** the desired conditional ensemble SD and was never a tuning target.

First-state969400 source ablations, team-delay sample SD (veh-s):
- Both hidden-driver + arrival/newcomer sources: **696.7**, eight distinct tapes.
- Driver source only, future request/newcomer realization fixed: **134.1**.
- Arrival/newcomer source only, current driver settings fixed: **755.6**.
- Both fixed: **0.0**, **all eight complete physical tapes hash-identical**.

Source variances are not additive: interactions and shared nonlinear traffic
dynamics mean these SDs must not be interpreted as variance fractions.
Driver-only request streams match exactly; arrivals-only current-driver spread
is exactly zero. No global physics RNG advancement in any probe rollout.
Exact due-minus-inserted queues, no dropped requests, no padding refusal, finite
positions/speeds verified for all **280 v2 physical futures**.

Mean per-HDV K8 SD: desired speed0.807m/s, headway0.0258s, gap0.138m,
acceleration0.0506m/s², comfortable deceleration0.165m/s². These scales come
from inferred vectors/motion likelihood, not generic runtime perturbations.

Request-count mean within-K8 SD at120/300/600s: **8.82/14.88/20.48 vehicles**;
mean Poisson theoretical SD: **9.50/15.02/21.24**. K8 is a small Monte Carlo sample.
Recorder `arrivals` means destination exits, so those counters are explicitly
not compared with entrance requests. Traffic-delay spread was measured at120s;
300/600s here are request-clock diagnostics only.

## Calibration limits and corrected mechanics

The likelihood is empirical-Bayes/approximate, not coverage-certified. The
reference120s mean team bias is +332.8veh-s, empirical CRPS611.4veh-s and K8
range coverage59.4%. Those are scoped diagnostics, not a WM-PHYS gate PASS.
One factual continuation per observation cannot identify true conditional SD.

The v1 diagnostics exposed comparison of observed executed acceleration with
raw unbounded IDM at compressed/censored gaps. Applying the existing plant
bounds removes that mismatch: maximum residual MSE3.18e10 -> **42.51**;
median0.0764 and median particle ESS38.85 across2153 driver histories in v2.
Both complete versions are retained; no factual delay selected a width/threshold.

LC2013 settings not identified by the old encoder remain documented population
defaults. MOBIL/merge parameters are not reinterpreted as LC2013 parameters.
An absent current-driver donor pool requires an explicit learned population;
an unobserved CAV newcomer pool fails closed rather than invent settings.
Full WM-PHYS120/300/600s action/distribution qualification belongs to INT/EVAL.

## Checks, artifacts and compute

- New CPU suite: **12 passed** (latest18.64s); existing futures regression:
  **13 passed** in the combined run before the new-module-only follow-ups.
- Actual INT-runtime LC2013+SUMO-insertion synthetic smoke: exact tested/assigned
  settings for both entrances; two five-tick replays bit-identical; global RNG
  unchanged. `RBUNC_INTEGRATION_SMOKE.json` records exact source hashes.
- Independent verification PASS:96 metric/world cells, CRPS re-evaluated as
  empirical-CDF integral, SD as pairwise variance, exact recorded target equality
  and all source invariants. `rbunc/v2_verification.json`.
- Final data `rbunc/fit_h120_fit_probe_v2.json`; summary `rbunc/v2_summary.json`;
  own dispatch/PID/terminal receipts `rbunc/v2_dispatch_terminal.json`.
- V2 raw SHA `db82294e9e5c47c362192055ca5111ceb8aac17c305313980066e58bf21eb703`;
  sampler SHA `c758e1d9bd3702b2b26e75ed3a3080c6c833d5fb98362ad09bcfa4e4d021e752`.
- V1 outputs/analysis retained with explicit historical count-label correction.
  560 reference futures total over both versions =67,200 model-seconds;
  **SUMO0, GPU0, training0**. Existing CPU conda, single-thread nice19 workers,
  maximum8 plus the canary; initial and queued-work free-resource checks.
  All32 worlds complete; all31 supervised children return0. Peer processes intact.
