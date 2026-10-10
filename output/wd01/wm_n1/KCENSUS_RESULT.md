# KCENSUS_RESULT (A5 §1-§3, read-only drop-location census)

- sample: `KCENSUS_SAMPLE.json` sha `affce2e2177a1a44…` (2026-10-01T08:17:46Z, seed 20261001, 300 runs); census elapsed 293 s (2026-10-01T09:35:19Z → 2026-10-01T09:40:12Z)
- totals: rostered 40,855,639 vehicle-ticks, dropped 2,677,481 (in-zone 725,448 / out 1,952,033); still dropped at K=192: 57,157

## §3 rule quantities (numbers only; no decision applied)

- `out_share` = 0.729056 (1,952,033 / 2,677,481 pooled dropped vehicle-ticks outside the zone)
- `in_zone_drop_share(L)` = dropped in-zone / rostered in-zone:

| loading | share | dropped_in | rostered_in | bootstrap 95% CI |
|---|---|---|---|---|
| m2000 | 0.000000 | 0 | 1,148,685 | [0.000000, 0.000000] |
| m2400 | 0.014933 | 40,238 | 2,694,534 | [0.007196, 0.023436] |
| m2600 | 0.034559 | 222,483 | 6,437,755 | [0.029135, 0.040502] |
| m2700 | 0.042920 | 462,727 | 10,781,187 | [0.038680, 0.047318] |

- `out_share` bootstrap 95% CI: [0.711158, 0.745202]

## Descriptive extras (not rule inputs)

| loading | roster p50 | roster p99 | roster max | residual drop share @K=192 |
|---|---|---|---|---|
| m2000 | 72 | 97 | 104 | (pooled) |
| m2400 | 106 | 128 | 128 | (pooled) |
| m2600 | 122 | 128 | 128 | (pooled) |
| m2700 | 128 | 128 | 128 | (pooled) |

- pooled residual drop share at K=192: 0.001399 (57,157 / 40,855,639)

## Integrity / reconcile

- raw source fidelity: sha256(physics_steps.jsonl) == pack INDEX `input_sha256` verified before parsing for all 290 censused runs
- per run: census `dropped_total` == `SLOT_POOL_CENSUS.json` `dropped_vehicle_ticks` (pack_v2c meta `n_dropped_slot_steps`): 290/290 ok

## Missing raw sources (10 of 300 sampled runs; numbers cover the 290 censused runs)

- `chunk1/w8475m2000p40t360__pi`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 7a9de161d069028e…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8475m2000p40t360__switching2`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 c63896422d7654b7…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8469m2400p40t420__switching3`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 6c3628f54754bf11…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8476m2600p40t480__switching2`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 6b70674c1ae06ac5…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8468m2600p40t240__fixed1`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 0546c9711e2aa64f…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8476m2600p40t480__pi`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 73c31abb15cd1148…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8472m2700p40t300__native`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 57804c327c14fb77…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8473m2700p40t540__fixed2`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 f295414e9db3eda5…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8465m2700p40t240__switching3`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 e39519edfa74dd70…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- `chunk1/w8465m2700p40t240__switching2`: raw physics_steps.jsonl absent at the shard meta input_physics path; pack INDEX input_sha256 eaa452d8a67491d9…; relocated candidates (chunk1_fix27 re-runs) sha-mismatched; source not recoverable from this workspace
- spot check chunk1/w8414m2000p02t240__switching3: shard meta 0 == slot-pool census 0 -> ok
- spot check chunk1/w8416m2000p02t300__fixed2: shard meta 0 == slot-pool census 0 -> ok
- spot check chunk1/w8416m2000p02t300__native: shard meta 0 == slot-pool census 0 -> ok
