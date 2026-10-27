# S-LOOPLONG — preparation complete

All switches default OFF; `looplong.enabled` is the master, `enable_cycle=3`.
Work preserved from 29d7490b. CPU engineering only; no SUMO, production training,
GPU allocation, live deployment or push.

## Flags

- `wm_loss`: frozen-core, current-cycle real POLICY/native 300s tapes;
  S-LONGWM physical/paired losses, measured scales and gradient-parity weights.
  Preserve ordinary replay and optimizer-step budget.
- `auto_horizon`: unchanged acceptance plus registered 10/30/60s nonregression;
  same 96-group pairs300 panel/thresholds at 60/90/120/180/300s. Largest PASS
  sets trusted reach; curriculum inherits previous reach. Hash gate/thresholds.
- `long_credit`: trusted WM accrued returns; **critic bootstrap/baseline OFF**.
  Explicit `imagination_horizon_s` extends beyond the gate using genuine K=8
  paired futures, conditional bias-correcting quantiles and world-max 80% CQR.
  Disjoint validation requires coverage and within-world paired-benefit ranking
  lower95>chance; singleton worlds cannot qualify. Unqualified, exploratory,
  changed-actor and unvalidated cooperation contrasts are pessimistic-only;
  insufficient conformal rank gives zero long credit. Source-inclusive clocks
  reconcile to real J. Direct real paired 300s credit has share **1.0**, via an
  explicitly off-policy executed-action secant surrogate, not a value model.
  Calibration/validation worlds cannot supply actor fitting starts/real gradients.
- `speed`: bounded window caching, shared acceptance decoding, redundant
  full-graph probe removal, cached deterministic courtesy targets, two-lane
  gradient-parity VJPs. Retain learned-forward checkpointing; retain outer
  physics operations instead of nested recomputation (additional activation memory).
- `hot_swap`: owned post-conversion overlap; first update boundary after ACCEPT.
  Preserve actor/optimizer/count, reset score competition, refit CQR, log swap.
- `exploration`: correlated 30–60s holds and metering denominators.
  `coop_credit=false` zeros individual/group/cooperation/team schedules only.

## Verified evidence

**83 tests pass (16.41s); Ruff and handoff checks pass.** Includes R16
first-passage flag-OFF/ON trajectory/RNG/gradient parity and reversed-within-world
ranking refusal. Two tiny WM updates and three 90s policy updates are finite;
critic loss zero; synthetic metering holds **0/24** starts. Production frequency
and CQR qualification remain unmeasured.

`tools/looplong/{TESTS.log,DRY_RUN.json,CHECKPOINT_SPEED.json,CYCLE1_TIMING.json}`:
checkpointed synthetic CPU WM-FT **17.50→11.52 s/step (34% faster)**, two updates
per arm; checkpoint tensors/RNG identical. Gradient-parity recomputation dominates.
Production ~264s/step is not remeasured. Historical 40-step FT/acceptance:
7,859.67/1,688.25s. Acceptance toy decodes 4→2, metrics identical; timing flat.

Using the corrected 36.512GiB/60s/K2 reference, rough fixed-batch linear
120/300s projections are **73.0/182.6GiB (K2)** or **292.1/730.2GiB (K8)**.
Static allocations/entrant growth are unresolved; 48GB gpuA (0.65 cap≈30.8GiB)
requires the planned long-memory work before production use.

## Exact prospective enable commands

In each deployed code root; main binds these new configs at the boundary.
E5 twins use identical flags except cooperation credit.

```sh
R=/root/autodl-tmp/c4/restart20261026
python -B -m tools.looplong.prepare --base "$R/arm/CONFIG.json" --out "$R/arm/CONFIG_LOOPLONG.json" --pairs300 /root/autodl-tmp/c4/v5/data/pairs300 --threshold configs/coopw_horizon_gate_13E_b2.json --enable --imagination-horizon-s 300 --coop-credit on
R=/root/autodl-tmp/c4/coopoff220261027
python -B -m tools.looplong.prepare --base "$R/arm/CONFIG.json" --out "$R/arm/CONFIG_LOOPLONG.json" --pairs300 /root/autodl-tmp/c4/v5/data/pairs300 --threshold configs/coopw_horizon_gate_13E_b2.json --enable --imagination-horizon-s 300 --coop-credit off
```

Reproduce via `tools/looplong/cpu_tests.py`: `stage`, `tests`, `dry-run`;
checkpoint timing: `start-speed-checkpoint --run-id NEW`, then poll/collect.
Verify: `python -B tools/looplong/verify_handoff.py`.
