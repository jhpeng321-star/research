# S-DISTILL2 — MAPPO-assisted WM warm-start building block (2026-10-30)

## Design and status
- Branch `c4/distill2`, base `9613f956`; final implementation SHA recorded below after verification.
- Frozen teacher `mappo_v5_final.pt` SHA `aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564`.
- WM2 SHA `b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436`; explicit WM config required.
- Execution: always-commanded mode 2, SUMO leader rule ON, existing learned newcomer mix ON.
- Imported S-CAPREL shared `residual.executed_from_residual` from `37bc2a57`; both reachability laws use it.
- `tools.distill.represent`: per-present-CAV physics-tick executed MAPPO−CTRL0 ledger, entrants included;
  exact reachable-interval fractions (all ticks/nonzero deviations), L1 deviation fraction reproducible;
  lane changes, ramp merge choices and command/native switching reported separately from acceleration fit.
- MAPPO has private command memory and a lane head; absolute commands meet the shared physics cap.
  Between-decision births abstain to WM-native MAPPO behaviour; student holds zero residual until decision.
- `tools.distill.train`: teacher- and student-driven 60 s trajectories each round; aggregate bounded FIFO replay.
  Student = candidate-A CavPolicy + RSU encoder, teacher trunk initialization, deterministic bounded residual.
  Inputs allowlist: deployable kinematics/CAV reports/RSU field, zero command/intent convention; no teacher memory,
  inferred driver labels, driver truth, future controls/presence, or requested penetration enter student features.
- Data-fitted inverse-execution least-squares head initialization crosses the current-law cap dead zone.
  Thereafter only Huber loss on executed acceleration, differentiable through the shared law; live CAV mask.
- Logs: merge/ramp/mainline/entrant MAE/RMSE; reproduced braking magnitude, absolute and relative to CTRL0.
- Exact `c4_dreamer/3` writer, residual/native-lane metadata, critic/optimizers/RNG; imitation clock separate
  from RL update_count=0. DAgger replay/Adam/sampler/RNG persist in additive `distillation` metadata.
  RL Adam starts fresh. `policy14_cli --native-init --resume ... --new-phase` loads the exported checkpoint.
  `configs/c4_distill_mappo.toml` disables inherited zero_residual_output so resume retains the learned head.
- Training/export currently uses the incumbent law. Cap-relative reachability is implemented; cap-relative
  training/export refuses until the full S-CAPREL policy14 runtime is integrated (prevents wrong-law resumes).
- `tools.distill.check`: distilled-only/MAPPO/CTRL0 feedback, K8, 60/120/300 s prefixes, paired dJ before
  reductions, raw draws/median/q10/q90 and world-cluster CI. Existing e3_score group loaders, delay_proxy
  and PairedRandom streams are reused. Both registered t0-HDV delay and fleet+source J are retained.
  Requires existing W-verdict JSON rows: exactly 48 pairs300 keys and 36 validation keys; no new selection.
  Entries/endogenous exits are simulated; recorded future controls/presence are never imagination inputs.
- No production representability, trained actor, or real-panel dJ is claimed: registered weights/corpora/panel
  receipts are not staged here. SETUPAUDIT.md was absent at the last check; adoption remains audit-dependent.

## Training provenance and resources
- Optimization seed: `965160`; smoke network seed `701` (synthetic only). No production traffic training run yet.
- Only existing TRAIN corpora: `native_v5` (`v5_native_q`) and `genrand_plain` (`genrand_plain_q`), from
  `configs/c4_policy14_starts.toml`; disjoint development windows via split seed `965140`, val_fraction=0.1.
- Every source manifest is preflighted before arrays load; seeds 964000–964199 and 977000–977999 refused.
  Actual selected training worlds and source-manifest hashes are listed in every checkpoint/summary signature.
- CPU checks: synthetic tiny WM, entrants, shared-law gradients/parity, loss decrease, observable isolation,
  bitwise interrupted/uninterrupted DAgger resume, real policy14_cli full new-phase load, K8 zero=CTRL0 pairing.
  Verification: 22 passed (14 distillation + 8 KEYERR regression), CPU, 66.26 s; pytest cache_dir warning only.
- GPU planning ONLY, unmeasured: 600 updates = 144,000 WM physics ticks + 19,200 small actor minibatches.
  Rough budget gpuA (47 GB): 2–8 h, 4–12 GiB; gpuR (24 GB): 3–12 h, 4–12 GiB, sequential no WM backward.
  Full 84-start K8 three-arm panel = 1,209,600 WM ticks: rough 8–40 GPU h. Qualify actual cost before launch.
  CPU-only work in this build; no SSH, GPU, SUMO, or external traffic generation was used.

## Exact prospective commands (repository root; run CPU here)
```sh
export PYTHONPATH=src:. CUDA_VISIBLE_DEVICES='' C4_ALLOWED_ROOTS=/root/autodl-tmp/c4:/root/e3long/inputs
export WMS_DATA_V5_NATIVE_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q
export WMS_DATA_GENRAND_PLAIN_Q=/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q
WM=/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt
CFG=/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml
MAPPO=/root/e3long/inputs/mappo_v5_final.pt
python -m tools.distill.represent --wm "$WM" --wm-config "$CFG" --mappo "$MAPPO" --out runs/distill_repr --rollouts 8 --horizon 60 --seed 965160 --device cpu
python -m tools.distill.train --wm "$WM" --wm-config "$CFG" --mappo "$MAPPO" --out runs/distill_first --updates 100 --seed 965160 --device cpu
python -m tools.distill.train --wm "$WM" --wm-config "$CFG" --mappo "$MAPPO" --out runs/distill_final --updates 600 --seed 965160 --device cpu --resume runs/distill_first/policy_latest.pt
# Supply the existing frozen W-verdict JSON receipts at these explicit paths before panel scoring:
python -m tools.distill.check --wm "$WM" --wm-config "$CFG" --mappo "$MAPPO" --student runs/distill_final/policy_latest.pt --pairs300-root /root/autodl-tmp/c4/pairs300 --validation-root /root/autodl-tmp/c4/e3data --pairs300-panel runs/w_verdict/pairs300.json --validation-panel runs/w_verdict/validation.json --out runs/distill_check --seed 140014 --device cpu
python -m pytest -q -p no:cacheprovider tests/unit/test_distill.py tests/unit/test_keyerr.py
```
- Output directories must be fresh; the resume update limit is TOTAL, not additional. Gates are unchanged.
