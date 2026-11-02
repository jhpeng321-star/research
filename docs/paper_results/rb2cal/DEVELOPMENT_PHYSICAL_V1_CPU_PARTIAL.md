# S-RB2-CAL physical v1 — partial development delivery

**Over-confidence remains unresolved.** The fitted physical sampler does not
establish nominal traffic-outcome coverage or reliable 0.8 calls on the retained
development cases. This is an incomplete, availability-selected diagnostic,
not a qualified replacement model or an independent final comparison.

## Frozen versions and execution

- Physical sampler `rb2-cal-v1`: code/evidence `0711a026`, fit
  `v1/fit/calibration.json`, recipe `configs/wm_phys_rb2_cal_v1.toml`.
  Posterior retention1; Gamma intensity variance multiplier1.667235195661008;
  entrant prior pseudocount0. Parameters remain the original dev-fitted values.
- Strict-win probability map `probability_fullv4_v1/calibration.json` remains
  bound to FULL-v4 only. It cannot be applied to physical v1 or integrated v5.
  No new map was fitted on this partial sample.
- Remote CPU run/source/archive identities are preserved in the fetched
  CANARY/PANEL receipts. Runtime calibration SHA256
  `bb195c12d1af607156220cc7264b81d3efbc281eae73195bce1c5aab87e84a99`;
  verifier matches frozen content with LF/CRLF normalization only.
- MAIN stopped CAL workers448399/448400 Nov2 02:28 for S-COOPMWM resource
  priority. Both absent on direct Nov3 audit. Original collector exited BLOCKED
  on SSH failure; its Nov2 09:00 deadline has elapsed.
- Retained57/96 tasks:48 tasks from complete K8 worlds969420–969425 scored;
  nine tasks from incomplete969426/969427 retained but excluded.39 tasks missing.
  Five arms, horizons120/300/600; no reserved raw outcomes decoded or scored.

## Physical v1 results (horizon order120 /300 /600 seconds)

- Absolute-delay CRPS: **829.05 /3307.60 /10817.04 veh-s**.
- Absolute central50% coverage: **6.7 /23.3 /20.0%**.
- Absolute central80% coverage: **66.7 /50.0 /33.3%**.
- Absolute central90% coverage: **70.0 /60.0 /63.3%**.
- Absolute inclusive K8 range coverage: **70.0 /70.0 /70.0%**.
- Paired-delay CRPS: **175.53 /1097.94 /2730.81 veh-s**.
- Paired central50% coverage: **18.3 /18.3 /23.3%**.
- Paired central80% coverage: **43.3 /43.3 /45.0%**.
- Paired central90% coverage: **51.7 /51.7 /50.0%**.
- Paired inclusive K8 range coverage: **58.3 /56.7 /51.7%**.
- All unordered-pair Brier: **0.23125 /0.26094 /0.19245**;60 pairs per horizon.
- Selected0.8-call Brier: **0.28613 /0.27969 /0.13058**; calls16 /10 /14.
- Selected forecast win: **96.1 /96.3 /95.5%**; observed **68.8 /70.0 /85.7%**.
  Those are11/16,7/10,12/14 strict wins. Calls across incumbents/arms are
  dependent within world; they are not independent Bernoulli trials.

## Same-world frozen FULL-v4 comparison

Baseline is the saved FULL-v4 **GPU** forecast, matched to the same six worlds;
candidate is **CPU**. This is not a device-controlled physical-parameter ablation.
Baseline absolute CRPS845.04 /3207.77 /10111.75 veh-s; paired
CRPS176.20 /1032.76 /2784.23. Candidate absolute CRPS is slightly better at120s
and worse at300/600s. Pairwise Brier improves at120/600s but worsens at300s.

Mean absolute predictive SD increases832→902 /3178→3699 /10622→12718 veh-s,
while mean paired predictive SD is152→150 /908→900 /2272→2263 veh-s.
Thus wider marginal futures have not produced wider action-contrast futures.
These diagnostics do not identify which latent family causes the remaining
action-effect uncertainty; retain common-random-number coupling across actions.
Absolute600s mean delay bias remains+8594 veh-s on this subset, versus+10065
for the matched baseline; neither is the owner's original reserved8 estimate.

## Receipts, validity and remaining blocker

- `dev_physical_v1_cpu_partial/RESULTS.json`: exact scores, admission hashes,
  source identity and caveats; `physical/` and `baseline_matched/` contain
  observations, summary CSV/JSON, reliability deciles and both reliability SVGs.
- `dev_physical_v1_cpu_partial/VERIFICATION.json`:57 retained hashes,48 complete
  conservation checks, complete-world/arm/clock support, frozen artifact match
  and all confidence intervals suppressed. Recovery implementation `813ee355`.
- Calibration/recovery suite **20PASS** using existing Miniconda, one BLAS/OMP
  thread. Initial default-Python collection lacked SciPy; an unbounded-thread
  run timed out. The final single-thread run completed38.84s. Ruff passes.
- K8 interpolated central intervals are descriptive, not exact nominal
  guarantees. Continuous iid K8 range reference is7/9, not90%; neither empirical
  coverage nor in-development probability fitting establishes final validity.
- Full dev12 and independent replicated-cell comparison remain incomplete.
  Completing the39 missing futures requires renewed MAIN resource/admission
  authority and an explicit source/claim/deadline reconciliation. Do not
  restart the expired collector or refit an exposed frozen candidate.
