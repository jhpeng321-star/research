# S-C1CLOSED — runtime expansion and numerical sensitivity

Operational supplement, not an efficacy result. Frozen primary model, K8,
480-task/40-world panel, controllers, scoring and original finalizer are unchanged.

## Cross-host verification
- 420 runtime/start files match across hosts; concrete model, policy, candidate,
  execution config and all16 inherited WM configs are byte-identical.
- EXPORT_BOUND metadata differ by host/source diagnostics; executable fields and
  actual MAPPO169 SHA256 match. gpuR's missing config chain was copied only into
  absent destinations; no peer source/config/checkpoint was overwritten.
- Both environments: Python3.12.3, torch2.8.0+cu128, numpy2.3.2; GPUs4090D/4090.
- Full300s w0/k0/common replay: exact cross-host parity **FAIL**, preserved.
  Independent same-host repeats match bitwise on both hosts through300s.
- Cross-host60s maximum original-ID position/speed differences:0.0003663m /
  0.00006557m/s; total delay0.0002442veh-s, counts/queues identical.
- At300s, total-delay difference<=23.92veh-s (<=0.101%); one arm differs by
  13 endpoint queue vehicles and9 crossings. This is hardware sensitivity,
  not an irreducible SUMO RNG floor; local field differences can be substantial.
- Separate numerical gate was registered in commit51ba73ca BEFORE its K8-spread
  evaluation: early physical-unit tolerances plus all late scalar/field/common-ID
  differences bounded by fixed same-world K8 spread. **PASS158/158**, no waiver.
  This operational acceptance is not a claim that exact parity passed.
- All three original BB/historical10s CUDA paths pass against their original
  CPU checks: maximum position0.0004883m, speed0.00006104m/s. No retraining.

## Exclusive execution and collection
- Original own scheduler PID211623/start881579797 paused; all CUDA children drained.
  100 completed tasks reused, two CPU tasks reserved outside BOTH new assignments.
- Remaining tasks: gpuA31 (two GPU lanes); gpuR347 (up to four GPU lanes).
  Assignment uses roster/runtime only, never recorded errors or controller outcomes.
- gpuR budget8,000,000,000 bytes; gpuA12,000,000,000; >=1GiB reserve, 3GiB
  allocator/process, own PID+start checks. Sampled owned MiB:2298/6078 (A/R).
- New dispatchers: gpuA252290/start882699960; gpuR564599/start4207337319.
- Publication bridge Windows36960 is a scheduler/output transport only; it atomically
  publishes verified successful gpuR outputs into the original gpuA panel.
- Sole scientific collector/finalizer remains Windows39680; every original source
  hash is unchanged. It runs analysis/verification/commit only after480/480 completion.
- Exact failure, prospective numerical acceptance, assignments and live resource
  receipts: `tools/c1closed/evidence/EXPAND_*.json`; code `tools/c1closed/expand*.py`.
- `prefetch.py` can bulk-copy immutable own output/receipt files into the frozen
  sync cache with per-file/archive hashes and safe member validation. One128-file
  receipt batch and separate one-off200-file transfer both verified. It
  performs no inference/analysis/finalization and cannot change the complete gate.
- Latest live receipt:157/480 unique tasks complete, both expansion runners and
  sole collector/bridge live, no failed expansion tasks. Sampled completed-task
  owned peaks3750MiB (gpuA) /7316MiB (gpuR), below the respective byte budgets.
- Cached partial panel validates129 tasks against frozen model/source/initial-state
  bindings and all200 recorded label sets. Remaining351 cache tasks block scoring;
  this identity check is not a partial C1 estimate. `CACHE_VALIDATED.json` records it.

## Separate reference slot
S-C1FLOOR now supplies an approved native-only tiny-speed-pulse sensitivity on
worlds969400–969405 (`c4-c1floor/docs/paper_results/C1FLOOR.md`). It explicitly is
NOT an RNG-only irreducible floor or matched PI/MAPPO reference. Keep that distinction
when integrating its supplied curve; the registered RNG-only slot stays unavailable.
