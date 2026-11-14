# RUN2_READY — WM-S Run 2 launch preparation

Status: **prep complete including long-horizon stability; full GPU run NOT
started.** Branch `wms/run2b` (= `wms/run2` e61d653 with `wms/stab` merged, merge
commit **6ee1d8a096f6**). The stability merge kept **every feature flag-gated**;
run 2 turns the recommended ones ON in `_out/run2_chain.sh` /
`_out/run2_train_config.json`. TRAIN-only, CPU dry runs, no SUMO/GPU/SSH. No
reset/rebase/clean/checkout/stash.

## Merge contract

`git merge wms/stab` into `wms/run2b`: only two conflicts (`model.py`,
`train.py`), both resolved by keeping **both** feature sets. In `model.rollout`
the run-2 `entries`/state-noise path and the stab `projection` path are combined;
the return dict carries `entrant_seq`/`sections`/`state`/`history`/`history_mask`
and adds `projection_stats` only when a projection is supplied. In `train.py`
the stability imports, trainer knobs and the conservation/projection reporting
sit alongside the trafloss/entries hooks. Off-path is unchanged (all run-2 flags
default OFF; `bounded_value(x, None) == x`).

## Feature-branch status

| branch | tip | merged into run2b? | content |
|---|---|---|---|
| `wms/entry2`  | 33ad317 | yes (via run2) | entries v2 (observed-history arrivals) + trainer `--entries-v2` |
| `wms/trafloss`| e09b365 | yes (via run2) | traffic-level losses (B1), state noise (B4, off), 300 s plan_steps |
| `wms/stab`    | 6f9baaf | yes (6ee1d8a) | feasibility projection, capped/contractive residual, conservation, error growth |

## Merged code contract

`wms.cli train --help` now exposes the run-2 flags `--merge-cal`,
`--merge-cal-params`, `--entries-v2`, `--entry-pad`, `--traffic-loss`,
`--state-noise-std-m`, `--state-noise-std-mps` **and** the stability flags
`--projection`, `--projection-min-gap-m`, `--residual-cap-mps2`,
`--lane-logit-cap`, `--stability-weight`, `--stability-margin`,
`--stability-probes`, `--conservation-metric`, `--conservation-weight`. All
default OFF. New config knobs: `conservation_enabled` (metric) distinct from
`conservation_weight` (loss). `wms.gate_run2` gains
`--require-projection`/`--require-conservation`.

## Run-2 training config (`_out/run2_train_config.json`)

Run-1 defaults + junction fix (active) + `merge_cal` ON
(`_out/merge_params.json`, sha256 `c23385f9…`) + `--entries-v2 --entry-pad 8` +
`--traffic-loss` + curriculum `10/30/60/120/300` s, plus the stability block:

| feature | flag | value |
|---|---|---|
| feasibility projection | `--projection --projection-min-gap-m` | 2.0 m |
| capped residual | `--residual-cap-mps2 --lane-logit-cap` | 1.5 / 3.0 (stab default) |
| contractive regulariser | `--stability-weight --stability-margin --stability-probes` | 0.01 / 1.0 / 1 |
| conservation metric | `--conservation-metric` | ON (loss term OFF, weight 0.0) |
| state noise | `--state-noise-std-m/-mps` | OFF (0.0) |

Frozen: batch 8, truncation 20, start-shift 0.75, sumo-fidelity, packs
`pack_v3+pack_dspeed+pack_damping`, `windows_train_noPI.json` (1307), seed 101.

## Real-data CPU dry runs (batch 4, threads 4, one process)

Command: `python _out/run2_dry.py --tag run2b_10 --unroll-s 10 --steps 2
--merge-cal --entries-v2 --traffic-loss --extra --projection
--projection-min-gap-m 2.0 --residual-cap-mps2 1.5 --lane-logit-cap 3.0
--stability-weight 0.01 --stability-margin 1.0 --stability-probes 1
--conservation-metric` (and `--unroll-s 300 --steps 1`). Data opened READ-ONLY;
`split=TRAIN`; receipts `_out/run2_dry_run2b_{10,300}/TRAIN_RECEIPT.json` confirm
`merge_cal=true`, `entries_v2=true`, `traffic_loss=true`, all stability knobs on,
`state_noise_std_*=0.0`.

| run | valid targets | entrants | position | speed | lane | traffic | stab | cons | proj frac | proj gap/clamp | step s | peak RSS |
|-----|---------------|----------|----------|-------|------|---------|------|------|-----------|----------------|--------|----------|
| 10 s (2 steps) | 7809 → 6928 | 22 → 17 (sum 39) | 13.16 → 11.48 | 0.739 → 0.630 | 0.139 → 0.183 | 0.0 (ramp) | 0.0 | 0.0 | 0.0035 → 0.0046 | 13→26 / 13→26 | 96.0, 58.6 (mean 77.3) | 7623.6 MB |
| 300 s (1 step) | 69519 | 32 | 354.52 | 2.558 | 0.563 | 6.367e8 (finite) | 0.0 | 0.0 | 0.130 | 57 / 57 | 1554.9 | 8442.3 MB |

All logged terms are finite. Entrants > 0 in both; projection-active fractions,
gap/s-clamp counts, conservation metric (max violation 0) and loss terms are all
present. The conservation metric is 0 (no unexpected creations/removals). The
300 s step is a single 600-sim-step unroll with batch 4 on CPU: the loss
magnitude and the >6 GB peaks are the differentiable IDM/MOBIL autograd graph
(compute, not data) — the same effect documented for C1/run-2 prep. The data
memory rule (mmap `mode='r'`, one batch materialised, one run at a time) is
honored; the CPU peak is a compute-graph peak. Artifacts:
`_out/run2_dry_run2b_10.json`, `_out/run2_dry_run2b_300.json` (+ per-run
`TRAIN_RECEIPT.json`, `train_log.jsonl`, `stdout_tail.txt`).

## Tests (OMP/MKL threads 4)

`pytest output/wd01/wm_n1/wms/tests output/wd01/wm_n1/closedloop/tests
output/wd01/wm_n1/rollout2/tests output/wd01/wm_n1/p5splits/tests
tests/test_wms_eval_c1.py tests/test_wms_eval_confinement.py
tests/test_wms_eval_seen.py tests/test_wms_decide.py` → **270 passed, 2 skipped**
(2 pre-existing TRAIN chunk-4 parity skips) in 445.1 s; plus
`tests/test_wms_eval_wms_plugin.py tests/test_wms_eval_run1_report.py` → **13
passed** in 54.0 s. Total **283 passed, 2 skipped**
(`_out/run2b_final_pytest.txt`). `ruff check output/wd01/wm_n1/wms
output/wd01/wm_n1/wms_eval tests/test_wms_eval_wms_plugin.py` clean.

## Code tar

`_out/code_run2_6ee1d8a096f6.tar` (git archive of the merge commit 6ee1d8a,
excluding `archive/`, `runs/`, `output/wd01/part*`, prior `_out/code_run2_*.tar*`;
includes `_out/merge_params.json` and all `wms` code), 35.54 MB, sha256
`af65104e15503894aed8c33ba61a5072191b7d091b1d7d57fb447057de7c5a2a`
(`_out/code_run2_6ee1d8a096f6.tar.sha256`).

## GPU chain

`_out/run2_chain.sh` (gpuA): flag-presence guard now also checks `--projection
--residual-cap-mps2 --stability-weight --conservation-metric`; smoke400 gate
(400 records, all numeric terms finite, traffic-loss term present, entrant total
> 0, **projection and conservation metrics present**) via
`wms.gate_run2 … --require-projection --require-conservation` → `plan_steps` over
the 5 stages with `BUDGET_S=18000` (~5 h) and `HARD_CAP_S=28800` (8 h; refuse
above) → one full run. Stability flags are passed on every training invocation
(smoke and full).

## Registration

Draft: `_out/REGISTRATION_WM-S-RUN2_DRAFT.md` (sha256
`d207c06a82ce3b7ea67d2c443e16c83712865676b10684e208f76b541965bca6`), comparator
**WM-S-TRAIN-20261005**, budget 18000 s / cap 28800 s, pending owner freeze and
ledger append.

READY FOR RUN 2 GPU: yes — commit 6ee1d8a096f645483de418833f720897a2a8f7f2
