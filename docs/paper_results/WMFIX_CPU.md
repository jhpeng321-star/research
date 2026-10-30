# WMFIX CPU implementation receipt

## Resume and core — 2026-10-31
- Preserved commits 1a3a4631 (59-line audit), 388a0cd4 (69-line design + GLM packages).
- Recovered interrupted, empty wms.py from HEAD before adding the opt-in integration.
- Observed local Python/SSH command lines before work: no WMFIX job or dispatch identified; none started.
- WMREQUAL3.md at the task-named location remains absent at resume. Retarget gate stays closed.
- New zero-initialized observable spatial field runs before the incumbent emergency envelope/plant bounds.
- `ModelSpec.merge_response=False` and `EntryConfig.conserve_queue=False` are defaults.
- Conservation preserves pending demand across actual insertion failure/pool exhaustion; legacy clipping stays exact OFF.
- Paired full-fleet component objective retains exact nondifferentiable source costs; no admission straight-through estimator.
- CPU command: `C:/Users/jpeng/Miniconda3/python.exe -m pytest tests/unit/test_wmfixmerge.py -q -p no:cacheprovider`.
- Result: **10 passed** (36.26 s); warning is pytest cache_dir with cacheprovider explicitly disabled.
- Includes committed-legacy forward + 3-step rollout/RNG/key parity, finite-difference state gradient,
  new-head-only optimizer/frozen tensors, source error retention and identity/slot-reuse exclusions.
- Unit/synthetic evidence is not the registered real TRAIN free-running feasibility gate.

## TRAIN pipeline verification
- Added full production-array hash binding, fit-role/seed/arm/path checks, identical-prefix checks,
  exact recorded request/backlog reconstruction and common due-schedule checks.
- Added two-row paired runtime, frozen-head trainer, same-config export/reload and immutable checkpoints.
- Config/OFF legacy-flat omission and checkpoint canonical reload tested. No CUDA or SUMO invoked.
- Stronger feasibility requires paired-J improvement separately from aggregate normalized loss.
- Original synthetic common-mode teacher: total loss improved but paired-J MAE 0.00026059→0.00026083;
  retained `WMFIX_SYNTHETIC_V2.json` is FAIL. This is a real acceptance-gate discovery, not a model result.
- Conditional observable-state teacher: loss 9.14834e-5→9.11202e-5, paired-J MAE .00800538→.00800323;
  derivative −.00104930 vs central difference −.00104756; frozen tensors exact (`WMFIX_SYNTHETIC_V3.json`).
- Production absolute anchors now crossing/road only, matching design; paired J/source/upstream/crossing remain explicit.
- Combined CPU check: **52 passed** in101.71s: wmfixmerge/data/training, entryfix, v9entry and wmrequal labels.
- An earlier combined invocation timed out at120s; process inspection confirmed it ended before retry.
- Retried with one CPU math thread for the new model fixtures; no unrelated processes were signalled.
- Real probe receipt is BLOCKED: registered prepared corpus is remote and absent locally.
- WMREQUAL3.md checked again before this step: absent. Source-clock/retarget contract and Astra approval remain blocking.
- Exact eventual launch, input roots, six-hour estimate and gate schemas: `tools/wmfixmerge/README.md`.

## Learned-only admission audit amendment
- Inherited `sumo_like_v2` selected stop-speed driver settings from textbook class tables before birth sampling.
- New default-OFF `EntryConfig.learned_admission` instead uses the SAME entrance/ordinal-keyed
  pre-t0 inferred-pool draw as `LearnedMix` assigns to that newborn's subsequent motion.
- v0/headway/decel/min-gap are inferred; no truth/type assignment input or new behavioural threshold.
- Inferred admission requires conserved queues and the physical sumo_like_v2 formula; hazard bundles reject.
- Four new CPU tests cover exact candidate→newborn setting equality, actual insertion speed/conservation,
  invalid settings/bundles, and actual OFF insertion/RNG parity against committed WM2 entries.
- Targeted combined admission/verdict/entryfix/v9entry command: **38 passed** (66.70s).
- Real feasibility now records ledger-only vs inferred-admission/zero-head outputs and separate J/crossing/source gradient norms.
- Launch checks bind the implementation bytes to CPU evidence; all three opt-in flags are required.

## Corrected-harness delivery / retarget
- WMREQUAL3.md appeared before final handoff and was read immediately; SHA256
  `6a692d83b4cc949c5cad1ff37ff972cde15ece91b9142189e8f57e67c497af25`, evidence commit79b75b27.
- Corrected matched J300 MAE1376.6/r−.235/sign.556; upstream covariance share96.0%; source bias−870.9veh-s.
- The merge/queue target remains justified as a testable hypothesis, not a proven capacity mechanism.
- Report availability resolves the pending-report blocker; remote TRAIN feasibility, clock comparison and independent review still block launch.
- Retargeted FIT supervision to native/ours_c1/coopoff_c1 only: old MAPPO lane-head requests are unrepresented.
- Added default-OFF production-loader arm filtering before any episode open; excluded MAPPO tapes never open for fitting.
- After this change, data/training targeted tests: **16 passed** (25.39s), including2 new source-filter/OFF checks.

## Review items 7/8 (04:00 addendum): loss rebalance and differentiable-only gradients
- `paired_objective` gradients now flow only through the differentiable paired columns
  (upstream, discharge surrogate, road); paired J and source are exact detached reports
  (`objective_terms`: per-term values, paired/anchor shares, paired MAE for J and source).
- Absolute anchors keep discharge+road levels but are normalized by their own TRAIN
  absolute-value RMS (`anchor_scales` in `TRAIN_PREPARED.json`), not paired-contrast RMS;
  trainer writes `STEP0_TERMS.json` and per-step term shares. Old prepared corpora lack
  the key and fail closed; MAIN must re-run prepare.
- Fixed latent objective_terms bug: raw-list references made the active mask a scalar
  bool, unsqueezing the MAE mean over inactive rows (3.0 vs 6.0 in the fixture).
- Synthetic probe declares its single differentiable column explicitly; real probe passes
  production anchor_scales. Tests updated to the 5-component contract.
- Combined rerun: wmfix 7-file set **51 passed** (30.25s); entryfix/v9entry/v9entry_eval
  **31 passed** (8.85s).

## Review blockers 5/9 and data-need statement (resume session)
- Panel A is now three-way: base (WM2 legacy entries) vs entry (zero-head WM2 with
  conserve+learned admission, same seeds) vs candidate (head); gates stay base>candidate,
  base>entry and entry>candidate attribution CIs reported (schema wmfix_validation/2).
- Real feasibility gate strengthened per blocker 5: `panel_gate` covers all 12 groups x
  both arms x {120,300}s, reports per-horizon paired-J MAE and upstream teacher ceiling,
  measures frozen-rerun noise, and launch additionally requires the post-step panel to
  beat the pre-step panel beyond that noise at both horizons (`gate_panel_ok`).
  Feasibility accepts `--device cuda` for MAIN (48-rollout panel is slow on CPU).
- WMFIX_DATA_NEED.md states the 24-contrast verdict and the owner options (reuse-first
  allowlist, ~100k sim-s 36-world collection, or focused smaller head).
- Verdict/zero-head tests updated; full wmfix set re-run below.
- Producer receipt corrected: speedhead/c1 panel files do NOT exist here; packages OPEN,
  GPU-host side, with the exact wm_gpu generalization spec in tools/wmfixmerge/README.md.
- Final complete verification (resume session): **86 passed**,17.95s over
  wmfixmerge/data/training/admission/contract/verdict/zero_head + entryfix + v9entry
  + v9entry_eval + wmrequal labels. Review-request addendum records items 1-10 status.
