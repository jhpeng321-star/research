# S-LONGLOSSTEST-1 — offline long-run paired-loss test

Preparation on `c4/longloss`, from `c4/looplong@6a0f5dff`. No launch yet.
Target: gpuR `/root/autodl-tmp/c4/longloss20261027`, mirrored autonomously to
the same path on gpuA; terminal `DONE.json`. Zero new simulation.

## Prospective design

Start each variant independently from WM1 SHA256
`5257a64c7cbb6a8cb828cb26230e97c686d7715cc33825bd3e1ca95f7bf51d8e`.
Reuse S-WMDEEPTEST-1's exact world split, 48-group scoring panel, loop-world
acceptance panel and original thresholds. Globally exclude its 25 scoring seeds
from fitting. WM1's historical exposure remains acknowledged.

LONGLOSS-frozen first, then LONGLOSS-deep sequentially, each <=3 h including
assembly/validation, <=400 updates; same fixed16-window position validation,
every5steps, patience3, strict minimum, step0 eligible. Use the loop's real
cycle0/1/2 POLICY/B 300s tapes and paired-return provenance. New cycle2 worlds
are assigned prospectively by a fixed hash, with validation never used to fit.
Inherited scoring worlds remain unchanged.

Enable exactly the looplong WM hook: longwm loss ON, complete300s graph,
600tick truncation, batch2, coupled paired randomness, stage paired horizon.
These required changes from DEEP's120s/batch4 are explicit; this tests the wired
LONGLOSS package, not an objective-only matched ablation. Keep original replay
shares and paired-batch fraction. Deep additionally unfreezes lane_v5,
merge_core, merge_pivot, ramp_speed_adapt, driver_encoder, driver_updater,
residual_net.lane at1.5e-6; residual1.5e-5, existing courtesy/yield rates retained.
Use the speed-fixed trainer from6a0f5dff, learned driver settings only.

Original horizon_gate at10/30/60/90/120/180/300s and accept.py follow training.
Report signed paired-dJ mean error and centered SD at120/300s. Primary improvement
is abs(WM1 mean300s error)-abs(candidate mean300s error), paired world-cluster
bootstrap95% percentile CI,2000 draws seed0, retaining all starts/arms per world.
Verdict **helps** only if improvement>0, its CI lower bound>0, and sign agreement
at each10/30/60s meets the unchanged threshold. Acceptance reported separately;
missing endpoints give unavailable. No automatic adoption.

CUDA allocator cap0.85; disk floor3GiB (stop trigger3.25GiB). Detached own-PID
chain, no laptop dependency; per-variant failures retained and terminal receipt
mirrored. Training ETA <=6 h total plus fixed scoring; actual launch ETA follows.

Results pending; main reads them after this launch-only handoff.
