# S-COOPEVAL — evaluation + figure pipeline for the cooperation paper

Authority: REGISTRATION.md 2026-10-31 15:45 (workstream launch, 15:50) and the
16:00 S-SUMOCF contract (docs/paper_results/SUMOCF_DESIGN.md). S-COOPEVAL is a
NO-SUMO workstream: code + tests only; runs are dispatched later by
MAIN/S-SUMOCF inside the registered caps (train 8,000,000 sim-s, dev 300,000
sim-s). Nothing here launches SUMO, training, retries or any process; the
tools are read-only over a mirrored run tree and write only under `--out`.

## What it produces

1. **Dev-panel paired evaluation** (dev worlds 969600-969623, two full 12-cell
   repeats): learned policy vs speed-only MAPPO vs CTRL0. J at 120/300/600 s =
   E3 `team_delay_vs` (fleet + source delay, veh-s). Paired world bootstrap,
   2000 draws, seed 0, 95 % percentile CIs. **Negative learned-minus-comparator
   difference = learned policy better** (S-SPEEDHEAD convention). Safety:
   collisions, teleports, TTC<3 s exposure (veh-s), hard braking (veh-s),
   worlds-with-collision counts, plus the registered collision-increase
   pairing check.
2. **Cooperation tables**: per checkpoint (update u -> `policy_u{u:02d}.pt`)
   and pooled: D_G, sum_i D_i, C_G = D_G - sum_i D_i, mean_i M_i, D_all —
   improvements in veh-s, positive = better — with seed-cluster bootstrap CIs
   (rows averaged within a seed first) and the share of rows with C_G > 0
   (rows pooled within resampled seeds). Singleton/empty-group rows are logged
   but excluded from interaction credit (SUMOCF_DESIGN.md). Cross-checked
   against the trainer's own receipts (CG_positive/CG_rows/mean_CG).
3. **Ablation scaffolding configs** (`configs/coopeval/`, no runs):
   - `ablation_full.json` — the registered S-SUMOCF curriculum (reference).
   - `ablation_ind_only.json` — individual-only credit: w_group = w_coop =
     w_team = 0 (w_all = q/4, w_ind = 1 kept).
   - `ablation_mappo_continue.json` — equal-budget MAPPO-continuation with its
     own reward: identical init/worlds/branching structure/episode count/sim-s;
     only the learning signal differs.
   All three share init (MAPPO iter169 d9042cc5...), worlds
   (train_worlds 969200-969391 + dev 969600-969623 + canary 969624) and budget
   accounting (198 episodes x 900 s per update; 5,702,400 train + 280,800 dev
   planned). `tools/coopeval/ablations.py` validates anti-gaming
   (w_coop <= w_ind), nonnegative expanded coefficients and cross-arm
   comparability. **These are scaffolds; running them needs a separate
   registration + ledger + owner approval.**
4. **Paper figures** (PNG + the numbers behind them as CSV):
   - `fig_training_curves.png` / `training_curves.csv` — mean C_G (95 % CI)
     and paired dev J600 diffs vs cumulative training sim-s.
   - `fig_dev_bars.png` / `dev_bars.csv` — paired dev J differences at
     120/300/600 s vs both comparators, CI error bars.
   - `fig_safety_table.png` / `safety_table.csv` — safety means per arm.
   - `fig_coop_rates.png` / `coop_rates.csv` — cooperation-behaviour rates,
     consumed from S-COOPBEHAV outputs **when present** (see below).

## Input contract (mirrored S-SUMOCF run tree, `--root`)

Only these registered paths are ever opened:

    receipts/DEV_BASELINE.json    24 dev worlds x {mappo, ctrl0} worker results
    receipts/dev_u{NN}.json       dev check per registered update
    receipts/credit_u{NN}.json    per-(seed, sample) credit records
    UPDATES.jsonl                 one receipt per update (or receipts/update_u*.json)
    accounting.jsonl              precharged episode ledger
    receipts/CANARY.json, EVENTS.jsonl, FINISHED.json

J: `J600` (cumulative_600) is authoritative for the 600 s horizon; 120/300 s
come from the E3 `metrics.windows` blocks (reported as missing, never
fabricated, when absent). Paths containing restricted tokens (sealed, holdout,
heldout, cal_split, chunk5, j4b, inc1, hetdata_truth, vtype) are refused.

## Exact commands

From the worktree root (`E:/agent-work/c4-coopeval`, branch `c4/coopeval`):

```bash
# everything (dev tables + coop tables + figures + status + summary),
# once a mirrored run tree exists:
python -m tools.coopeval.cli all --root /path/to/sumocf20261031_mirror \
    --out _out/coopeval \
    --coopbehav /path/to/COOPBEHAV/rates.json \
    --mappo-train-sim-s <MAPPO production receipts sim-s or omit>

# a specific dev-check update instead of the latest:
python -m tools.coopeval.cli all --root <mirror> --out _out/coopeval --update 12

# validate the ablation scaffolding configs (safe any time):
python -m tools.coopeval.cli ablations            # exit 0 = valid, equal budget

# refresh only the figures from already-written reports:
python -m tools.coopeval.cli figures --root <mirror> --out _out/coopeval

# unit tests (synthetic data only, no SUMO):
python -m pytest tools/coopeval -q
```

S-COOPBEHAV consumption (`--coopbehav`, optional, may be repeated): JSON files
with schema `coopbehav_rates/1`, `{"rates": [{"controller", "behaviour",
"rate", "lo", "hi", "unit"}]}`, or a CSV with those columns. Any other file is
skipped, never guessed at; when absent the figure step is skipped and the
summary says so.

## Output map (`--out`, default `_out/coopeval/`)

| file | content |
|---|---|
| `DEV_EVAL.md` / `dev_eval.json` | paired J tables (120/300/600), safety, splits, pairing integrity |
| `dev_j_worlds.csv` | per-world J and safety per arm (the numbers behind the tables) |
| `dev_safety.csv` | per-arm safety means @600 s |
| `COOP_TABLES.md` / `coop_tables.json` | per-checkpoint + pooled credit tables, shares, consistency |
| `coop_credit_rows.csv` | every pair row of credit |
| `fig_training_curves.png` / `training_curves.csv` | C_G + dev J600 vs sim-s |
| `fig_dev_bars.png` / `dev_bars.csv` | paired dev bars with CIs |
| `fig_safety_table.png` / `safety_table.csv` | safety table |
| `fig_coop_rates.png` / `coop_rates.csv` | S-COOPBEHAV rates (when present) |
| `run_status.json`, `COOPEVAL_SUMMARY.md` | canary/finished/accounting, fairness record, conventions |

## Conventions and guarantees

- J = E3 team delay (fleet + source, veh-s), **lower better**;
  learned-minus-comparator **negative = better**; credit D **positive =
  better**; C_G = D_G - sum D_i; M_i = D_G - D_(G\i).
- Bootstrap: world/seed-cluster, 2000 draws, seed 0, 95 % percentile, in
  `tools/coopeval/stats.py` and replicated by the trainer-side receipts.
- Missing inputs are reported (exit 2 for an empty tree; explicit notes in
  outputs); malformed inputs fail loudly; nothing is imputed or fabricated.
- Fairness: MAPPO's original training cost is carried as
  `--mappo-train-sim-s` / `run_status.json`; **no equal-budget claim is made
  unless the equal-budget comparator arm is actually registered and run.**
- Rules: own PIDs only; no processes launched; no deletions; no push; no new
  venvs; no SUMO anywhere in this pipeline; forbidden-path guard on `--root`.
