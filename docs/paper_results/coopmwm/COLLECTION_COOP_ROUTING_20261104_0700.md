# Cooperative-search config routing repair — Linux qualified, not activated

## Diagnosis from public metadata and call paths
`SEARCH_CURRENT_AUDIT_1793746420172116000.json` reports zero pending B/R
snapshots. The latest saved B search answered at unix 1793733032 (03:10 +08),
and all 80 B / 146 R snapshots have responses. The four GPU processes retain
their loaded models; a point sample on gpuA was 0% utilisation. A plain restart
of the existing supervisors cannot create fresh training snapshots.

Two confirmed routing defects are repaired:
1. `replay_start.collection_options` now reads only the additive, registered
   `coop_search` and `coop_alternation` keys from `LS_COOP_SEARCH_CONFIG.json`.
   Absent-file output is unchanged; invalid keys/types or alternation without
   qualified niche admission fail loudly.
2. `replay_bank.collector_guide` forwards those keys when obtaining the next
   niche admission config. The outcome-blind 2x2 sequence is preserved:
   niches 1/4 with cooperative scoring on, then niches 1/4 with it off.
   R still publishes centre-only queries without WM-scored candidate config.

## Qualification
Immutable cpuQ control `coopmwm20261104_collectrouting2`, based on the live
qualified `launchcoop_src3`: **41 unittest checks + 9 pytest checks PASS**,
zero skips/SUMO, 13.30 s. Tests exercise the actual loader -> collector guide ->
signed request path, four admissions, R isolation, and invalid overrides.
The cooperative-objective-off golden uses frozen Git `b20bec2f` under the same
Linux interpreter, not the Windows golden. First attempt's missing test setup
directory and cross-platform-golden failure are retained in the V1 receipt.

## Separate remaining live integration defects
- `v2v_runner.wave` sets `b_replay`, but supplies neither `collect` nor
  `collector_policy_update`; `accel_plant` installs the snapshot hook only when
  `job['collect']` is present. The live V2V PPO waves therefore create no fresh
  bank queries. Their shared bank link directories still show historical
  snapshots, not evidence of current-policy capture.
- `policy_wm.load_actor` always constructs the original residual `Policy`,
  not the 48-wide intent policy. Merely attaching V2V policy weights to new
  queries is not a qualified current-policy WM rollout. This must be wired and
  parity-qualified before admitting such requests, rather than using old-policy
  centres or silently dropping intent inputs.
- The active GPU workers are still the NICHENULL3 numerical release. The
  cooperative-search numerical sources must be qualified per slot before
  claiming cooperative search active; no source or request was overwritten.

## Resource and accounting observations
Backfill coordinator 1132856 is alive, 38 decisions complete (19 each), and
properly disk-paused; its threshold remains >9 GiB. cpuQ free 7,212,777,472
bytes (~6.72 GiB). Regular mover 1237127 is alive; latest scan has zero eligible
files, source protection is retained, and a transport timeout alert is retained.
Do not lower its protection or the backfill threshold to force progress.

The old `search_restart_ops audit` phase-room calculation reads historical
`V2V_CAP_1406.json` and does not apply later additive transfers. Its negative
bank-phase remainder is stale, not a spending-authority result. Reconcile against
the live runtime/binding before new dispatch. This repair made no new charge,
changed no trained checkpoint, restarted no process, and opened no sealed outcome.
