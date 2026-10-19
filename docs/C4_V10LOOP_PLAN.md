# C4 v10loop — closed Dreamer loop plan (PREPARED, NOT LAUNCHED)

Branch `c4/v10loop` (from `c4/v10entzone` `b75475ad`). Status: **prepared only** — no training
launched, no GPU used, no evaluation seed touched. Execution requires owner approval, the merge of
`c4/v10unblock` and `c4/v10horizon`, and `execute_guard.allow = true` on gpuR.

Machines in scope (only these): **gpuR** (1 GPU, 224 CPUs, base `/root/autodl-tmp/c4/v5`) and
**cpuN** (48 CPUs, `~/c4`, reachable from gpuR as ssh host `cpuN`).

Artifacts:
`scripts/c4/loop_v10.py` (orchestrator, stdlib), `configs/c4_dreamer_loop_v10loop.json`,
`configs/c4_wm_v10loop_iter0.toml`, `tests/unit/test_c4_loop_v10.py`.

## Closed iteration

* **Iteration 0 (bootstrap).** Fine-tune the WM `v5m_ref` on **native/random TRAIN data only**
  (`configs/c4_wm_v10loop_iter0.toml`) with the `lane_learnable` key passed through; then start the
  **policy from scratch** inside that WM (`configs/c4_policy_v10entzone_v5m.toml`,
  `trainer.entrance_zone_native = true`).
* **Iteration k ≥ 1 (closed).**
  1. `collect` — current policy drives SUMO on gpuR CPUs + cpuN from the next fresh training seeds;
  2. `convert` — raw episodes → WMS data dir (`wms.data.c4_episodes`);
  3. `wm_finetune` — continue the previous WM with `lane_learnable` + horizon curriculum + mixed
     prioritised replay passed through verbatim;
  4. `wm_check` — finite losses and validation sanity vs the previous WM;
  5. `policy_update` — continue the previous policy inside the new WM (`WMS_LOOP_REFRESH=1 --resume`);
  6. `sumo_check` — seeds **969824–969835** vs `runs/sumo_check_v8pol_u600/native` with
     `pair_check.py` + `coop_metrics.py` + WM fidelity;
  7. `rollback` — revert + halve on regression (below), append `iterations.jsonl`.

## Seed block

* Proposed fresh training block: **`969000–969199`** (inside the owner-approved Dreamer-loop training
  block `C4_LOOP_TRAIN = 966000–969999`, `traffic_shepherd/seeds.py:323`).
* Excluded: live-test seeds `965990–965993` and SUMO-check seeds `969824–969835`.
* Evaluation block `964000–964199` is refused by `check_seed`; the SUMO-check block is never
  allocated for training.
* Runtime guard: `allocate_seeds` skips `seed.exclude`, the eval range and the check range, and the
  plan greps remote ledgers (`used_ledger_globs`, e.g. `/root/autodl-tmp/c4/v5/loop*/seed_ledger.jsonl`)
  before a real launch.

### Verification evidence (grepped 2026-10-19, this worktree)

```text
$ rg -n --no-heading "969000|969001|...|969199" protocols runs scripts configs traffic_shepherd docs
configs/c4_dreamer_loop_v10loop.json:15:      969000,
configs/c4_dreamer_loop_v10loop.json:16:      969199
protocols/experiment_history.json:2442: "cost_treatment": 86.619691004   # float substring, not a seed
```
No `protocols/` or `runs/` manifest references the block as seeds.

```text
$ rg -n --no-heading "96982[0-9]|96983[0-9]" protocols/*.json runs/active/*/*.json runs/active/*/*/*.json
(no output)
```
No protocol/run manifest references the SUMO-check block `969824–969835`.

```text
$ rg -n "C4_LOOP_TRAIN_SEEDS|C4_V6_LONGPAIR_SEEDS" traffic_shepherd/seeds.py
traffic_shepherd/seeds.py:320:C4_V6_LONGPAIR_SEEDS = frozenset(range(965000, 966000))
traffic_shepherd/seeds.py:323:C4_LOOP_TRAIN_SEEDS = frozenset(range(966000, 970000))
```
The block is inside the approved training parent; SUMO-check and eval ranges are disjoint.

> Residual risk: a prior gpuR loop may already have consumed part of `969000–969199`. The runtime
> ledger grep (`seed.used_ledger_globs`) is mandatory before the first real launch; if a collision is
> found, shift the block to the next free 200-seed window inside `966000–969999`.

## Stages and commands

Config: `configs/c4_dreamer_loop_v10loop.json`. Commands below are the exact strings produced by
`plan`/`dry-run`; `{...}` are the values the orchestrator fills in. `SHA` = deployed `repo_<sha>`.

### gpuR (WM + policy, local)

```bash
# iteration 0 - WM bootstrap on native/random TRAIN data, lane_learnable passed through
cd /root/autodl-tmp/c4/v5/repo_SHA && \
  PYTHONPATH=src:. OMP_NUM_THREADS=12 WMS_JAM_WORKERS=14 WMS_LATEST_EVERY=100 \
  WMS_DATA_V5_NATIVE=/root/autodl-tmp/c4/v5/data/v5_native \
  WMS_DATA_GENRAND_PLAIN=/root/autodl-tmp/c4/v5/data/genrand_plain \
  WMS_DATA_GENRAND_BRANCH=/root/autodl-tmp/c4/v5/data/genrand_branch \
  /root/miniconda3/bin/python -m wms train --config configs/c4_wm_v10loop_iter0.toml --mode run \
    --out /root/autodl-tmp/c4/v5/runs/c4_wm_v10_iter000 \
    --set model.init_from=/root/autodl-tmp/c4/v5/runs/active/C4-V7FT-20261017/v5m_ref/wm_s_best_ap.pt \
    --set curriculum.stage_steps=[60,40] --set model.lane_learnable=true

# iteration k>=1 - WM fine-tune with all three parallel key groups (verbatim pass-through)
  ... -m wms train --config configs/c4_wm_v5m_refresh.toml --mode run \
    --out /root/autodl-tmp/c4/v5/runs/c4_wm_v10_iterNNN \
    --set model.init_from=<previous WM wm_s_best.pt> --set curriculum.stage_steps=[60,40] \
    --set model.lane_learnable=true --set curriculum.horizon_curriculum=true \
    --set 'curriculum.unroll_s=[60.0,120.0,180.0]' \
    --set data.mixed_replay.prioritised=true --set data.mixed_replay.priority_alpha=0.6
  # WMS_DATA_V6_POLICY=<converted pool of this iteration>

# policy - from scratch (iter 0) / continued (iter k>=1), entrance_zone_native=true
cd /root/autodl-tmp/c4/v5/repo_SHA && \
  PYTHONPATH=src:. WMS_LOOP_REFRESH=1 WMS_DATA_V6_POLICY=<converted> \
  /root/miniconda3/bin/python -m wms.control.train_cli \
    --config configs/c4_policy_v10entzone_v5m.toml --wm-ckpt <new WM>/wm_s_best.pt \
    --device cuda --max-updates <N> --output-dir /root/autodl-tmp/c4/v5/runs/c4_policy_v10_gNNN
  # iteration k>=1 adds: --resume <previous policy_best.pt>
```

### gpuR + cpuN (collection)

```bash
# gpuR, 25 episodes/iteration, 160 workers
cd /root/autodl-tmp/c4/v5/repo_SHA && PYTHONPATH=src:. OMP_NUM_THREADS=1 \
  /root/miniconda3/bin/python -m traffic_shepherd.c4.collect --policy <policy_ckpt> \
    --population textbook --seeds <gpuR part> --case-plan weighted-cycle \
    --penetrations 0.05,0.1,0.2,0.4 --pen-weights 1,1,2,2 --demands 1900,2100,2400,2700 \
    --warmup 300 --horizon 900 --workers 160 --no-physics-log \
    --out /root/autodl-tmp/c4/v5/loop_v10/iter_NNN/collect/gpuR

# cpuN, 25 episodes/iteration, 40 workers (from gpuR via ssh host 'cpuN')
ssh cpuN "cd ~/c4/repo_SHA && PYTHONPATH=src:. ~/c4env/bin/python -m traffic_shepherd.c4.collect \
  --policy <policy_ckpt> --population textbook --seeds <cpuN part> --case-plan weighted-cycle \
  --penetrations 0.05,0.1,0.2,0.4 --pen-weights 1,1,2,2 --demands 1900,2100,2400,2700 \
  --warmup 300 --horizon 900 --workers 40 --no-physics-log --out ~/c4/loop_v10/iter_NNN/collect/cpuN"
```

The SUMO-side entrance masking (the `entrance_mask_loader.py` NaN fallback) is a property of the
SUMO check/collection environment, not of this orchestrator; the policy side matches it with
`trainer.entrance_zone_native = true` in `configs/c4_policy_v10entzone_v5m.toml`.

### SUMO check (gpuR)

```bash
python -m traffic_shepherd.c4.collect --policy <policy_ckpt> --seeds 969824-969835 ... --out <check>/policy
python <v7ft>/pair_check.py   --policy <check>/policy --native runs/sumo_check_v8pol_u600/native --out <check>/pairs.json
python <v7ft>/coop_metrics.py --pairs <check>/pairs.json --out <check>/coop.json
python <v7ft>/wm_fidelity.py  --wm <new WM>/wm_s_best.pt --states <fixed TRAIN-val states> \
  --horizons 10,30,60,120 --out <check>/wm_fidelity.json
```
The v7ft tool flags are data in `check.commands` (`<v7ft>` = `runs/active/C4-V7FT-20261017`); adjust
there without touching the orchestrator.

## Pass-through parameters (parallel branches)

`configs/c4_dreamer_loop_v10loop.json → wm.extra_set` carries, verbatim and unparsed:

| branch | key(s) currently wired |
|---|---|
| `c4/v10unblock` | `model.lane_learnable=true` |
| `c4/v10horizon` | `curriculum.horizon_curriculum=true`, `curriculum.unroll_s=[60.0,120.0,180.0]`, `data.mixed_replay.prioritised=true`, `data.mixed_replay.priority_alpha=0.6` |

If the branches finalise a different dotted spelling, edit only that JSON list. `wm.extra_set_iter0`
holds the iteration-0 subset (`model.lane_learnable=true` only).

## Rollback rule

Per iteration, from `<check>/coop.json` (`total_D.policy_minus_native`) and `<check>/wm_fidelity.json`
(`hdv.abs_ds.60`):

```
revert iff  SUMO total D(policy - native) > previous + rollback.sumo_total_D_max_worsen (= 0.0)
        or  WM_60 / previous_WM_60                    > 1 + rollback.wm60_rel_tolerance (= 0.10)
on revert: restore the previous WM + policy, and next_policy_updates = max(min_policy_updates,
            floor(policy_updates * halve_factor))  = max(50, floor(N * 0.5))
on keep:   next_policy_updates = policy_updates
```

`Revert`/`keep` and the reasons are appended to `<loop_root>/iterations.jsonl`; the self-test in the
dry-run exercises both terms.

## Budget (per iteration; from `loop_v10.py budget`)

| iteration | collect episodes | check episodes | collect sim-s | check sim-s | WM steps | policy updates | GPU h (WM) | GPU h (policy) | GPU h (total) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (bootstrap) | 0 | 0 | 0 | 0 | 100 | 200 | 1.2 | 1.6 | **2.8** |
| k ≥ 1 | 50 (25 gpuR + 25 cpuN) | 12 | 60 000 | 14 400 | 100 | 200 | 1.2 | 1.6 | **2.81** |

4-iteration run (iter 0 + 4 closed iterations): ~`14.0` GPU h, ~`240 000` collect sim-s. Rates are
config-driven (`budget.gpu_hours_per_wm_step=0.012`, `..._per_policy_update=0.008`,
`..._per_check_episode=0.0005`) and should be re-measured on gpuR before launch.

## MAPPO fairness counter

`loop_v10.py sample_counter` reports, cumulatively over `iterations.jsonl`:

* `train_episodes` — SUMO policy episodes collected;
* `policy_sim_s` — policy-controlled simulated seconds;
* `control_samples = policy_sim_s / mappo_control_dt_s` (dt = 0.5 s) — the number of policy control
  steps. **Train MAPPO on the same cumulative `control_samples`** so the comparison is sample-fair.
* At iteration 1: 50 episodes → 45 000 policy sim-s → **90 000 control samples**.

## Dry-run

Everything at tiny sizes; no external command runs (`executed_external_commands` = 0).

```bash
python scripts/c4/loop_v10.py dry-run --config configs/c4_dreamer_loop_v10loop.json \
  --out-dir <tmp>/v10loop_dryrun
```

Observed summary (excerpt; full trace in `<out-dir>/dry_run.jsonl` and `dry_run_summary.json`):

```json
{
  "stage_count": 14,
  "stages": ["wm_finetune_iter0","wm_check_iter0","policy_update_iter0",
             "collect","collect","convert","wm_finetune","wm_check","policy_update",
             "sumo_check","sumo_check","sumo_check","sumo_check","rollback"],
  "seed_block": [969000, 969199],
  "seeds_iteration1": "969000-969001",
  "budget_iteration1": {"collect_episodes": 2, "check_episodes": 12, "sim_s": 30.0, "gpu_hours_total": 0.046},
  "rollback_self_test": {"action": "revert", "next_policy_updates": 50,
                         "reasons": ["SUMO total D (policy - native) worsened by +0.0500",
                                     "WM 60 s HDV |ds| worse by 1.150x (> 1.10)"]},
  "executed_external_commands": 0
}
```

Representative rendered commands (iteration 1):

```text
[iter 1] wm_finetune: [wm] cd /root/autodl-tmp/c4/v5/repo_REPO_SHA && ... python -m wms train \
  --config configs/c4_wm_v5m_refresh.toml --mode run --out <...>/iter_001/wm/out \
  --set model.init_from=<...>/iter_000/wm/out/wm_s_best.pt --set curriculum.stage_steps=[1,1] \
  --set model.lane_learnable=true --set curriculum.horizon_curriculum=true \
  --set 'curriculum.unroll_s=[60.0,120.0,180.0]' --set data.mixed_replay.prioritised=true \
  --set data.mixed_replay.priority_alpha=0.6
[iter 1] policy_update: [gpuR] ... train_cli --config configs/c4_policy_v10entzone_v5m.toml \
  --wm-ckpt <...>/iter_001/wm/out/wm_s_best.pt --device cuda --max-updates 2 --resume <...>/iter_000/policy/out/policy_best.pt
[iter 1] sumo_check: [gpuR] ... pair_check.py / coop_metrics.py / wm_fidelity.py ...
```

## Verification evidence

* `python -m pytest tests/unit/test_c4_loop_v10.py` — 14 passed (seed guards, stage coverage,
  pass-through verbatim, rollback keep/halve/floor, sample counter, budget, WM sanity, guarded
  execution refused, injected-runner iteration logging, plan-doc sections).
* `python -m ruff check scripts/c4/loop_v10.py tests/unit/test_c4_loop_v10.py` — clean.
* `python scripts/c4/loop_v10.py plan` and `dry-run` — artifacts above; 0 external commands.

## Safety / stop rules

* No `git reset`/`clean`/`stash drop`; commits on `c4/v10loop` only; **do not push**.
* Never evaluation seeds `964000–964199`; never the SUMO-check block for training.
* Never open `sealed/`, `holdout`/`heldout`/`cal_split`/`chunk5`/`j4b`/`inc1`.
* `run` refuses unless `execute_guard.allow` is true (default false) — dry-run only in this worktree.
* Rollback reverts to the previous WM/policy and halves the policy-update count on regression.

## Next actions

1. Owner approval + merge `c4/v10unblock` and `c4/v10horizon`; confirm the pass-through key spelling.
2. On gpuR, `sed "s/REPO_SHA/<sha>/g"` the config, grep the remote seed ledgers, then set
   `execute_guard.allow = true` and run `loop_v10.py run --execute`.
3. Copy this plan to `runs/active/C4-V7FT-20261017/v10loop/LOOP_PLAN.md` (done in this prep).
