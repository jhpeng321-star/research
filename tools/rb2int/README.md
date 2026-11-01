# Round-2 integration/evaluation receipts

Status and immutable prospective design: `docs/rb2int/`; teacher signal:
`HANDOFF.md`. `smoke_ops` can launch only the explicit development smoke/canary;
it cannot dispatch a final panel. Development release is distinct from final.

- `audit`: outcome-blind design metadata, exact panel roster, prior fit artifact,
  and worker handoff snapshots. This does not automatically clear exposure.
- `resources`: read-only resource/path/INT receipt audit. Recheck GPU utilization,
  CPU quota/load, effective memory **and free disk** at actual admission.
- `freeze_design`: exclusive registration receipt. Never overwrite it.
- `inputs`: explicit registry of observable t0/history/result hashes; keeps the
  default round-1 loader unchanged. No missing-source-counter substitutions.
- `evaluate`: selected orchestration code with separately frozen model/runtime
  import precedence, exact source/config/checkpoint checks, and hash-bound panel
  release. WM2 retains its own original runtime; v4 cannot import merged v5 code.
  Registered extended tasks also save60s endpoints, without another forecast.
- `dev_canary`/`smoke_ops`: exclusive, incarnation-recorded development execution
  on the immutable INT stage. Closed60s smoke or dev969420/969421 K1/all5arm/H600;
  no replay of a completed smoke root. Resource/disk admission precedes claiming.
- `prepare_inputs`/`input_ops`: one-use causal source exporter, sanitized producer
  receipts, pre-t0-only speed bins, pairing/hash checks and verified archive
  transfer. It never reads future trajectories/cumulative targets. Ancillary
  result containers supply only start hashes/initial IDs; no outcome field is
  copied into the forecaster's input root. Production bridge/release still needed.
- `uncertainty`: joint18-contrast score/rate families; fixed-cell whole-world
  bootstrap, paired future resampling for forecast means. Singleton factual cells
  remain NA even when the K8 forecast varies.
- `gate_bootstrap`: original distributional gate adapter, with per-feature
  complete-stratum support. Missing metric worlds cannot borrow other worlds'
  bootstrap weights. Constant/unidentifiable bands are NA.
- `predictions`: explicitly approved run identities and frozen model manifests;
  checks all five arms, causal prefixes, BIC events, K8 completeness and delay
  ledger. Old reserved and new extended orchestration identities require a
  committed development parity bridge. Duplicate task copies are errors.
- `score_gate`: original full12-cell gate with the new bootstrap adapter and
  declared `primary54`, `all58` or `reserved8` scope. Baselines remain original
  development969400–969431. Missing12-cell/precision conditions prevent PASS.
- `compare`: six predeclared four-model contrasts, simultaneous95% intervals,
  empirical/fair CRPS, rate-distance, common-cohort HDV60 and switching rule.
  Reuses prefix-verified existing WM2 reserved endpoint tasks if supplied; this
  attaches diagnostic records in memory without changing saved prediction bytes.

## Required artifacts before final execution/scoring

1. Four delivered worker source/artifact exposure audits and candidate freezes;
   development-only CPU/smoke/bridge receipts, all with immutable hashes.
2. A `READY` release bound to the exact registration. `execution_purposes` must
   explicitly allow `final_comparison` (or only `development_bridge` for an early
   mechanics check). `models` binds every runtime/source/config/checkpoint.
3. A `rb2int-panel-inputs/1` manifest binds the registration/release, selected
   worlds, observable registry and factual receipt. Input preparation is implemented;
   production verification/bridging and panel dispatch remain. Do not manufacture these files
   as a substitute for missing exposure/smoke evidence.
4. A `rb2int-approved-runs/1` registry lists exact approved shard identity hashes,
   the frozen semantic model source subset and each real-receipt hash. It binds
   the same final release plus successful development bridge receipts.

All output receipts are exclusive creations. Current CLI analysis tools do not
retry, replace worlds, select candidates, open assignment/truth data, signal peer
processes, start SUMO, or train black boxes. These missing execution stages are
explicitly pending, not evidence that a full evaluation has happened.
