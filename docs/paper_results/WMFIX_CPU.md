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
