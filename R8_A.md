# Round 8 agent A — on-policy merge-core v2 (c4/v10r8a)

## Files changed

- `src/wms/train/losses/merge_core.py`
- `tools/merge_core_norm.py` (new)
- `tests/test_merge_core_v2_onpolicy.py` (new)
- `R8_A.md` (this file)

## Implemented

- `MergeCoreConfig` gained `version`, `z_tf`, `z_on`, `sigma_reg_weight`, `sigma_fit`
  with the contract validation (version 1/2; positive finite normalisers).
- Version 1 remains the original code path, unchanged; `version=1` is the default.
- Version 2 on-policy supported-prefix survival:
  - absolute-clock censoring (`c0 > 0` repaired) and terminal per-car support that
    persists across chunks and resets only on a genuinely new identity (slot reuse);
  - first-failure support reasons (position / speed / gap / topology / identity);
  - supported event only when feasible and supported, survival strictly before the
    recorded event, no post-event survival, late model merges ignored;
  - early model merge (ramp -> target lane only) included as a non-event only when its
    pre-decision state is supported, then terminated; non-target departures are support
    failures, not early merges;
  - administrative censoring at the last observed ramp interval and zero-exposure
    identities;
  - persistent normalized Gauss-Hermite posterior across chunks
    (`A_q = log pi_q + g_q`, `-logsumexp(A)`, detached update) and a birth-fixed
    alpha location/scale taken from the live core head at first ramp appearance;
  - partitioned terminal categories (`event`, `censored_admin`, five support-failure
    reasons, `early_nonevent`, `zero_exposure`) and exposure/scored-seconds metrics.
- Version 2 loss:
  `tf_weight * (sum_tf_nll / n_eligible) / z_tf
   + weight * (sum_on_nll / n_eligible) / z_on
   + location_reg_weight * mean_birth(mu^2)
   + sigma_reg_weight * mean(log(sigma / sigma_fit))^2`,
  with `sigma_fit=0` falling back to `alpha_log_sigma` in
  `src/wms/model/merge_core_init.json`.
- Recorded-state term calls agent B's `build_recorded_scene` once in `prepare` and
  `seq_tf_chunk` per chunk. The import is guarded: `version=2, tf_weight=0` works without
  B's module; `tf_weight>0` without B raises a clear error.
- `tools/merge_core_norm.py` loads a checkpoint + preset, forces `secure_gap`, rolls
  fixed TRAIN windows through the normal chunked unroll without optimizer steps, and
  prints per-window/mean `Z_ON`, `Z_TF` and terminal/exposure statistics. It refuses
  restricted split paths and eval seeds `964000-964199`.

## Not implemented / limits

- Agent B's `merge_core_seqtf.py` is not on this branch. The version-2 TF path is coded
  against `R8_INTERFACE.md` and tested with a stub; it was not exercised against B's
  implementation here.
- `tools/merge_core_norm.py` was syntax/CLI checked only; no checkpoint/TRAIN data were
  read from outside the worktree.

## Verification

- `python -m pytest tests/test_merge_core.py tests/test_merge_core_v2_onpolicy.py -q`
  -> **29 passed** (8 pre-existing merge-core tests + 21 new version-2 tests).
- `python -m ruff check src/wms/train/losses/merge_core.py tools/merge_core_norm.py tests/test_merge_core_v2_onpolicy.py`
  -> **All checks passed**.
- No tolerances, sealed/holdout paths, driver truth tables or eval seeds were used.
- Commit: see the branch head after this file.
