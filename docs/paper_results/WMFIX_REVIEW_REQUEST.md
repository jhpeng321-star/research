# WMFIX independent review request — Astra

**Requested decision: review the mechanism/CPU implementation, identify launch blockers, then approve or reject only after real gates.**
Branch `c4/wmfixmerge`, E:/agent-work/c4-wmfixmerge. No GPU training, SUMO or worker dispatch was launched here.
Prior/design commits 1a3a4631/388a0cd4 are preserved; implementation d8d7df49/48ae1c31/17d0467e;
corrected-evidence retarget 1e28f311. This file is a request, not an Astra approval.

## Evidence to check first
- `WMFIX_PRIOR.md` is59 lines; `WMFIX_DESIGN.md` is88 lines. Fifty grouped prior attempts, binding exclusions and acceptance tests are retained.
- Delivered task-named WMREQUAL3 report SHA256 `6a692d83b4cc949c5cad1ff37ff972cde15ece91b9142189e8f57e67c497af25`:
  matched J300 MAE1376.6/r−.235/sign.556; upstream covariance96.0%, source bias−870.9veh-s.
- That supports the conditional discharge/admission hypothesis after ownership correction, not a causal capacity scalar.
- Old lane-head MAPPO FIT tapes are now excluded before tape open; matched native/ours_c1/coopoff_c1 only supervise the correction.
- SPEEDHEAD is evaluation-only for this correction:963880–963887, final MAPPO/PI/CIRCLES/CTRL0, true t300 histories/queues.

## Key claims requiring adversarial checks
1. **Different mechanism/target:** full-upstream spatial occupancy/speed/CAV-flux field, not a local RAMPFIX residual or capacity multiplier.
   New output is added before the existing emergency envelope/plant bounds; CAV command ownership is retained.
   Targets are paired full-fleet J/source/upstream/crossing response, with absolute crossing/road anchors.
2. **Conservation by construction:** per-entrance due−inserted is exact, including initial Q0, deferred service and finite pool exhaustion.
   No dropped demand, guessed Q0 split or clipping with conserve_queue ON; original OFF clipping/RNG remains bitwise.
3. **Learned-only admission:** candidate physical stop-speed uses the SAME inferred entrance/ordinal draw as subsequent newborn motion.
   No textbook class-setting lookup in this path; geometric nominal length/lane limits only. Check retries/multiple births/entrances.
   Candidate-index/mix-count alignment must remain exact under pool exhaustion and backpressure, not only the easy fixture.
4. **Source clocks:** recorded labels use left-boundary queue rectangles; inherited WM scoring samples after insertion.
   Measure their paired contract explicitly against WMREQUAL3 before signing HARNESS_CONTRACT; do not infer parity from report existence.
5. **Gradient honesty:** hard service has zero source gradient away from events. No admission straight-through estimator is used.
   Probe separately reports paired-J/crossing/source gradient norms and central finite differences, and must improve actual paired J.
   Retained V2 synthetic receipt FAILS: total loss fell while paired-J MAE worsened. V3 conditional teacher passes, still synthetic.
   Real TRAIN teacher ceiling and a free-running optimizer step are NOT yet available locally; no real feasibility PASS is claimed.
6. **Data boundaries:** exactly12 allowlisted S-E3DATA FIT seeds in data.py; explicit arms, t300 starts, hash-bound production arrays.
   Reject wrong roles/paths/seeds/identities/Q0/due schedules before fitting; recorded exogenous demand only is teacher-forced.
   No validation/pairs300/SPEEDHEAD,964000–964199,962xxx or977xxx enters fitting; no truth/assignment/type-file inputs.
7. **Frozen fit:** optimizer exactly merge_response.*,1,473 new parameters; old tensors checked bitwise every immutable update/reload.
   Flags default OFF. Six-hour cap, native+one-arm microbatches,60/120/300s stages, final completed checkpoint primary.
   gpuR24GB estimate8–20GB/120–300s per300s update is UNMEASURED. MAIN must verify a single-update CUDA microbenchmark.
8. **Verdict integrity:** A K8/mix ON uses corrected native ownership; MAPPO lane-policy panel remains diagnostic.
   Same-world bootstrap2000/seed0; pooled sign/r/MAE/confident-gain improvements, horizon regression gates, zero-call blocking.
   B checks all four absolute arms and arm−CTRL0 directions/magnitudes300/600s; source+road must sum to J.
   C checks unchanged C1 PI/unseen-wall cohorts plus canonical accept.py thresholds; missing/failed safety never adopts.
   Step0 safety reference has exact WM2 legacy tensors/zero head. Check that C1/full deployment covers inferred admission too;
   canonical accept's historic cohort/entry convention is retained and cannot by itself certify all new entrance behaviour.

## Current blockers (review must keep these explicit)
- Remote registered TRAIN corpus has not been prepared/hashed by this session; local real feasibility receipt is BLOCKED.
- Corrected report is now read and residual persists; source-clock reconciliation and TRAIN free-running gradient/optimizer evidence remain pending.
- `verdict_plan.json` has null B/C1 producers and no canonical accept TRAIN roots bound.
  The one-command driver, concrete A/safety adapters and metric gates exist; exact SPEEDHEAD/C1 runtime producers are still needed.
  Bind final d9042cc5 speed-only MAPPO and frozen CTRL0/PI/CIRCLES, not update190 or original-slot command replay.
- Port/verify the corrected scorer's exact memory/runtime closure (WMREQUAL3 reports pair-bias chunking) before claiming source-hash equivalence.
- No independent approval, GPU memory/time measurement, complete A+B+C verdict or qualified WM checkpoint exists yet.

## Reproduce CPU verification
`C:/Users/jpeng/Miniconda3/python.exe -m pytest tests/unit/test_wmfixmerge.py tests/unit/test_wmfix_data.py tests/unit/test_wmfix_training.py tests/unit/test_wmfix_admission.py tests/unit/test_wmfix_verdict.py tests/unit/test_entryfix.py tests/unit/test_v9entry.py tools/wmrequal/test_labels.py -q -p no:cacheprovider`
Complete run: **65 passed**,144.29s (one-thread new model fixtures); two inherited pytest/NumPy warnings only.
After the matched-only FIT retarget: data/training tests **16 passed**,25.39s, including2 new production-filter/OFF checks.
Exact CPU preparation/feasibility, eventual six-hour training and one-command verdict are in `tools/wmfixmerge/README.md`.
Please return findings with file/line references and a launch decision. If eventually approving, bind REVIEW.json to passing real FEASIBILITY.json.
