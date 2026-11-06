# W4 finding: extra frozen final baseline is quoted but not executed

2026-11-06, read-only review of exact r12 source `3e5bc9f9`.

Owner 01:16 requires the lane-on MAPPO `aecd7d70` as an additional **descriptive** frozen comparator at final paired evaluation. The speed-only primary comparator and u24 synergy criterion stay unchanged.

## Evidence

- `tools/coopmwm/clean3_restart.py:57-59` defines the extra-baseline label; `:501` writes `cfg['extra_final_baselines']`.
- `prepare(..., laneon=None)` is called by `ops/clean3_chain.py:231` without `laneon`, so that slot has metadata but no executable/checkpoint binding in this path.
- `tools/coopmwm/launch_evaluation.py:124-146` builds checkpoints and jobs only for `cfg['final_evaluation']['arms']`; it never reads `extra_final_baselines`.
- `launch_evaluation.episode` always loads `<arm>_checkpoint_<evaluation_update>.pt` and the standard policy loader. There is no extra-baseline dispatch/analysis path there.
- Targeted content search across `tools/coopmwm/*.py` finds `extra_final_baselines` only in `clean3_restart.py` (configuration and AUTHORITY emission), no consumer.
- Budget reserves 133,200 sim-s for 96 x 1200 full-window lane-on jobs plus >=15% margin, merged into the 531,600 `v2v_final` allocation. This finding is about execution, not missing budget.

## Required follow-up for the existing launch operator / MAIN

Before final evaluation, qualify an additive executable path for the SHA-bound frozen lane-on MAPPO using the **same registered paired worlds**, full windows, descriptive safety and paired estimates; keep it separate from the primary criterion. Include an actual consumer/dispatch test and immutable runtime/authority bindings. Do not treat the quoted slot as a completed evaluation.

W4 has not edited or signalled the running r12 chain, changed a release check, dispatched any baseline simulation or duplicated another operator's implementation. This finding does not establish a training safety event. Primary operator retains deployment/fix ownership.
