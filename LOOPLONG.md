# S-LOOPLONG — preparation only

All switches default OFF. No SUMO, production training, GPU allocation, runtime
deployment or push occurred. Main enables at the cycle-2→3 boundary.

## Flags

`looplong.enabled` is the master; `enable_cycle=3` preserves earlier policy phases.

- `wm_loss`: frozen-core fine-tuning on current-cycle real POLICY/B300s pairs;
  physical/paired-effect losses use measured S-LONGWM scales/gradient-parity
  weights. Retain ordinary replay and optimizer-step budget.
- `auto_horizon`: unchanged acceptance AND registered 10/30/60s nonregression;
  evaluate 60/90/120/180/300s on the same 96-group pairs300 panel/threshold file.
  Select largest PASS; curriculum starts from the preceding achieved horizon.
  Gate/threshold hashes enter cycle events. Existing low-share cap remains.
- `long_credit`: beyond60s, K=8 paired-benefit bounds. World-weighted residual
  quantile/width fitting plus disjoint-world, world-max80% CQR; insufficient rank
  gives zero long credit. Partial clocks must reconcile to source-inclusive J.
  Bind calibration to collection actor/new WM; updated-actor use is off-policy.
- `speed`: bounded observable-window caching, acceptance decoding once per
  window across arms, and removal of redundant full-graph paired-return probes.
- `hot_swap`: owned overlap starts after paired conversion, alongside WM/check;
  swap at the first update boundary after ACCEPT. Preserve actor/optimizer/count,
  reset cross-WM score competition, refresh reliability/CQR, log swap update.
- `exploration`: retain correlated noise/30–60s persistence; log upstream
  multi-CAV sub-native-speed holds/denominators. `coop_credit=false` zeros
  individual/group/coop/team schedules; other new flags remain identical.

## Verification and resources

CPU source-only suite and synthetic integration/dry-run evidence:
`tools/looplong/{TESTS.log,DRY_RUN.json,CYCLE1_TIMING.json}`.
**71tests PASS19.03s; Ruff PASS.**
Two tiny WM updates/three90s toy-policy updates are finite; synthetic metering
holds0/24 sampled starts. Production hold frequency/CQR qualification: unmeasured.

Historical cycle1:40 optimizer steps7,859.67s; acceptance1,688.25s.
Dry WM kernel1.685→1.182s; decoder27.21→2.84ms; acceptance-rollout1.591→1.679s
(decodes4→2, no measured speedup). Checkpoint/objective/RNG and acceptance metrics
match. Gradient-parity reverse passes dominate the new loss's profile.

From12.086GiB/60s/K2, rough fixed-batch linear projections for120/300s are
24.17/60.43GiB(K2),96.69/241.72GiB(K8), excluding entrant growth/static-memory
effects. gpuA's0.65 cap≈30.8GiB: longer K8 runs are **not resource-qualified**.

## Exact enabled-config commands

After deploying this commit, run in each configured code root. Commands create
new configs; main binds them prospectively at the boundary. Existing gates apply.

```sh
R=/root/autodl-tmp/c4/restart20261026
python -B -m tools.looplong.prepare --base "$R/arm/CONFIG.json" --out "$R/arm/CONFIG_LOOPLONG.json" --pairs300 /root/autodl-tmp/c4/v5/data/pairs300 --threshold configs/coopw_horizon_gate_13E_b2.json --enable --coop-credit on
R=/root/autodl-tmp/c4/coopoff220261027
python -B -m tools.looplong.prepare --base "$R/arm/CONFIG.json" --out "$R/arm/CONFIG_LOOPLONG.json" --pairs300 /root/autodl-tmp/c4/v5/data/pairs300 --threshold configs/coopw_horizon_gate_13E_b2.json --enable --coop-credit off
```

Reproduce: `python -B tools/looplong/cpu_tests.py stage`, then `tests`, then
`dry-run`. Optional historical GPU-only paired fixtures are outside this suite.
