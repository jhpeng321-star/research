# S-INTEG — candidate B integration (owner 2026-10-30 16:30)
Worktree `E:/agent-work/c4-candB`, branch `c4/candB-int`, base `9613f956`.
Final verified implementation SHA: `dfbbb17607eedc5d38a28d4a1cfaaf61e6eee860`.
Documentation delivery SHA: `git log -1 --format=%H -- docs/paper_results/CANDB_INTEG.md`.

## Merges and integration seams
- caprel `28ecaa65` -> merge `7402e58b`; eslong `8f29a489` -> `9f36e6f5`.
- gatemeas `0f217127` -> `54130be5`; distill `b4dddcd4` -> `7829a7b1`.
- distill2 `d8361808` -> `ae74989d`; newer CAPREL zero/autograd and mix replay retained.
- Later distill refresh after parity: already up to date at `b4dddcd4`.
  **SETUPAUDIT2 final head has not landed**; re-merge it when published, then test its changes.
- Defaults remain OFF. Opt-in CTRL0 reference hook is importable, worker-transported,
  rebound on resume; CAPREL residual decode slope is explicitly exposed to ES calibration.
- Fixed historical receipt normalization and a distillation-fixture environment leak.

## CPU verification
- Torch2.14.1+cpu: complete requested union **502 passed, 2 CUDA-only skipped**.
  Includes CAPREL/ESLONG/GATEMEAS/DISTILL/SETUPAUDIT(2), longgrad/longskip/longmem,
  gated300/CCT1/CCT2/gate-panel/coopw/antiexploit, candA/integration/zerostart,
  keyerr/nanfix/entrant and all seven execution-parity suites; no CPU deselections.
- Final targeted integration/ranking/authority/budget/Policy14/recovery/setup suite:
  **55 passed**. Real tiny-WM K8 comparator objectives, zero=CTRL0, config/source/mix
  drift refusals, whole-world bootstrap and pre-update rank-gate refusal are covered.
- Ruff on all touched Python: PASS; `bash -n` launch/stage: PASS; diff check: PASS.

## Preset and required ranking gate
`configs/c4_policy14_candB.toml` extends `c4_policy14_candA_longgrad_skip.toml`.
Every override cites its source inline: CAPREL ctrl0 law; ESLONG antithetic M2/auto
sigma; GATEMEAS CTRL0 reference/training-window scoring/headroom starts; SETUPAUDIT
connected mix/objective-match refusal; balance/skips ON; fresh zero output; K8 retained.
Two-forward-worker prospective memory recipe: coordinator .40, workers .24 each,
budget .90. GPU memory/concurrency are not qualified by these CPU checks.
`tools/candB/rankcheck.py`: native/CTRL0/MAPPO-aecd7d70-native-lanes/CIRCLES/PI/current/
fresh-zero B, WM2 b8797242+mix, >=16 actual TRAIN-sampler starts, paired K8 at90/300s.
Actual dense-suffix/joint-q10 + gated-long actor objective is evaluated, scale1;
physical J and exact low-share dense caps are separately labelled. Published PI/damping
bridge and aligned absolute-MAPPO cap/Euler/native-background path are reused.
Raw future/start JSON + text table; 5,000-draw WORLD-cluster paired CIs, seed20261030.
Owner R2 gate and full runtime/config/source/WM-config/sampler/init bindings are enforced.
**No numerical rankcheck or launch authority is claimed:** matching all-vehicle/source/
CTRL0/K8/mix CCT-2 qualification is still required; old HDV-only/K1 authority refuses.

## Reproduction/staging commands — prospective, no training launched
Full exact argument lists and named input paths: `tools/candB/README.md`.
```sh
PYTHONPATH=src:. python -m tools.candB.rankcheck --help
PYTHONPATH=src:. bash tools/candB/stage.sh /existing/dir/candB_SOURCE.tar.gz
# Optional explicit staging target only: append gpuA /root/autodl-tmp/c4/candB_SOURCE
PYTHONPATH=src:. python -m tools.candB.authority build --help
PYTHONPATH=src:. python -m tools.candB.authority verify authority_gen/CANDB_AUTHORITY.json
RUNTIME=/root/autodl-tmp/c4/candB_SOURCE bash tools/candB/launch.sh /root/autodl-tmp/c4/candB_dry dry
RUNTIME=/root/autodl-tmp/c4/candB_SOURCE bash tools/candB/launch.sh /root/autodl-tmp/c4/candB_pilot pilot
```
Staging binds committed LF blobs + full byte manifest; launch independently verifies
all inputs and refuses an occupied GPU. `dry` is two optimizer updates, not a no-op.

## Registered SUMO execution parity — PASS
S-CAPREL-PARITY appended BEFORE SUMO to the named shared REGISTRATION.md.
gpuR, eclipse-sumo1.27.1, nice19, one CPU worker; training seeds963110/963111 only.
Paired zero scientific arrays bitwise identical; fixed -0.01 probe: **18,702** stable
rows, pooled RMSE **0.12230764831008893 <= 0.1603 m/s²**; native-head0.12769870693529828.
Deployment `verify_parity`: PASS on the immutable LF runtime;104 nonstable rows excluded.
First staging-import failure at t0 retained900s charge; registered fresh amendment ran
2700s. **Total charged3600, actual2700; allocation exhausted**. All attempts retained.
Receipt `tools/candB/evidence/CAPREL_PARITY.json`; REPORT/identity/budget/ownership beside it.
Remote `gpuR:/root/autodl-tmp/c4/candB_caprel_20261030_r2/parity_run/`; own437069 exited.
No production training, push, GPU job changes or additional SUMO; one-world parity is not efficacy.
