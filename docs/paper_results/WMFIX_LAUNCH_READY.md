# WMFIX launch readiness — 2026-10-31 (resume session; MAIN reviews and launches)

Status: **DRAFT — gates (b) feasibility and (c) retarget pending the zero-head panel**
(Estimated panel completion ~11:45 after the batch-11 OOM restart; this file is finalized
when the gates close. MAIN/Claude replaces the unavailable Astra REVIEW.json.)

## What passed already
1. CPU implementation + tests (WMFIX_CPU.md): 81-86 passed locally per stage; box CPU
   suite 50 passed (2 git-history parity tests are repo-only by design).
2. `TRAIN_PREPARED.json` regenerated on gpuA from the 12 FIT seeds (ledger-free S-E3DATA
   tapes; exact counter/schedule reconstruction with fail-closed checks), `verify()` PASS,
   sha `7bbd41f380b481858f5a2f07d08326a6cfafe899ad7425a5c55403e90f7ca572`,
   `ledger_checked:false`, anchor_scales discharge 104.553 / road 5590.36.
3. Corrected report `WMREQUAL3.md` byte-identical on gpuA (`6a692d83...`).
4. Candidate-config loading fixed end-to-end (`TrainSettings` carries `merge_response`);
   quartet worker canary PASS on gpuR (2.5 s / 3.5 GiB, zero-head pseudo-candidate
   `c3e3faa4...` — never a verdict candidate).
5. Verdict producers bound (verdict_plan.json sha `f47a8f4c...`): B = 69-file hash
   closure (producer_b.py + worker + reals + base quartet), C1 = fail-closed driver with
   explicit missing cohort/runner blockers, SAFETY accept root = /root/autodl-tmp/c4/e3data.
   Real SPEEDHEAD absolutes cross-checked against the frozen report table.

## Pending gates (blocking, in order)
- (c) zero-head entry-path panel (PID 100400, gpuA, groups 0-35, 14+/36 batches at
  09:20) -> `zero_head_compare.py` -> **WMFIX_RETARGET.md** decision.
- HARNESS_CONTRACT.json (`harness_contract.py`, residual-remains flag per that decision).
- (b) real TRAIN feasibility, `--device cuda`, strengthened panel gate (12 groups x
  both arms x {120,300} s, teacher ceiling, frozen-rerun noise, post-step improvement
  beyond noise). Must produce `FEASIBILITY.json` with `launch_ready:true`.
- REVIEW.json (MAIN/Claude in Astra's place) binding the passing FEASIBILITY sha.

## Exact launch command (MAIN, after gates)
Run on **gpuA** (all receipts live in /root/autodl-tmp/c4/wmfixmerge20261031; gpuR use
requires rsyncing the staged tree + TRAIN_PREPARED and re-verifying hashes):

```bash
cd /root/autodl-tmp/c4/wmfixmerge20261031 && mkdir -p train && timeout --signal=INT --kill-after=30s 21570s \
/root/miniconda3/bin/python -B -m tools.wmfixmerge.train --enable-merge-response --conserve-queue \
--learned-admission --device cuda \
--prepared /root/autodl-tmp/c4/wmfixmerge20261031/TRAIN_PREPARED.json \
--config /root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml \
--checkpoint /root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt \
--controllers /root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json \
--corrected-report /root/autodl-tmp/c4/wmfixmerge20261031/WMREQUAL3.md \
--feasibility /root/autodl-tmp/c4/wmfixmerge20261031/FEASIBILITY.json \
--review /root/autodl-tmp/c4/wmfixmerge20261031/REVIEW.json \
--out /root/autodl-tmp/c4/wmfixmerge20261031/train
```

GPU budget: <= 6 h (21,570 s cap), one card, ~8-20 GB estimated per update, every-update
immutable checkpoints. gpuA (48 GB) recommended; co-resident jobs must stay under ~20 GB
total. A single-update CUDA microbenchmark is still a MAIN pre-launch requirement
(CPU engineering cannot certify the estimate).

## Post-training verdict (one command)
1. Copy gpuR `/root/autodl-tmp/c4/speedhead20261030/wmfix_candidate_quartet` (worker run
   with the FINAL checkpoint + candidate_config.json) to gpuA
   `/root/autodl-tmp/c4/wmfixmerge20261031/quartet_candidate` after its COMPLETE.json exists.
2. C1 remains BLOCKED until the unseen-PI/unseen-wall cohort + lineage runner are staged
   (producer exits 2 with explicit blockers; missing cohort/runtime blocks adoption by design).
```bash
cd /root/autodl-tmp/c4/wmfixmerge20261031 && mkdir -p verdict && /root/miniconda3/bin/python -B \
-m tools.wmfixmerge.verdict --plan tools/wmfixmerge/verdict_plan.json \
--train-dir /root/autodl-tmp/c4/wmfixmerge20261031/train \
--out /root/autodl-tmp/c4/wmfixmerge20261031/verdict --execute --device cuda
```

## Known schedule slip
Batch-11 CUDA OOM (transient 10 GB peer + own 11.8 GiB cap) restarted the zero-head panel
at 06:55 with a 0.35 fraction; training start moved from 06:00-07:00 to ~14:00 at the
earliest, verdict ~20:30-22:00 same day.
