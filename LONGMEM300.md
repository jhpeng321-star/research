# S-LONGMEM300 - memory recipe for the gain-only 300 s long credit

Owner option B (2026-10-30 00:46) + MAIN priority message (fit first, speed second). Branch `c4/longmem300`
(worktree `E:/agent-work/c4-longmem300`, from `c4/candA-int` d04129f7). **Memory / infrastructure only**:
no change to the registered objective, credits, K=8, q10, WM (WM2 b8797242 + learned mix), CCT-2
authority (H_G 300, w_long 1.0) or the gain-only gate. All new switches default OFF in the library and are
exact (bitwise equal gradients in the CPU tests below).

Rig: gpuR (RTX 4090, 23.5 GiB). Staged copy of the zero-start runtime `/root/autodl-tmp/c4/longmem300/rt`
(= gpuA `candA_184542ab_zs` f4ee6cc0 + the patches below). Actor = candidate-A phase-1 checkpoint
`policy_000190.pt` (a realistic non-zero actor from the stopped run), `--new-phase` fork, seed 965140, nine rows,
K8, entrants ON, leader rule, always-commanded law, monitors ON, 1 start/update, `[long_credit] enabled = true`
(`configs/c4_policy14_longmem300.toml` extends the candidate-A config; only `long_credit.enabled` and
`policy14.zero_residual_output = false` differ).

## Verdict (STATUS: see "Measured result" at the end for the full-update line)

A single 24 GiB card CAN run the gain-only 300 s long credit, serially (one future at a time, no worker pool):
**peak 15.8 GiB allocated / 18.2 GB device (of 23.5) per future VJP, 417 s per future (replay 156 s + backward
261 s), 915 s for the no-grad forward of 8 futures** -> about 75-80 min per update. Memory is NOT the
binding resource any more; host RAM (5.6 GB RSS) and disk are fine.

Recipe (all exact; `probe.sh` sets the env defaults):

```
WMS_LM300_RECOMPUTE_REWARD=2   # per-step reward/TTC recomputed one call at a time (see Cause 2)
WMS_LM300_PAIRBIAS_CKPT=1      # WM pair-bias MLP recomputed in backward (see Cause 3)
[trainer] imagination_checkpoint_segment = 2   (the POLSPEED2 value; microbatch 1, both offloads unchanged)
single process: WMS_SPEED_FUTURE_WORKERS unset/1 (serial futures), [run] gpu_memory_fraction = 0.96
```

## What actually blocked 300 s (root causes, in the order they were hit)

1. **Forward worker size.** The no-grad forward of one 300 s future peaks at 10.1 GiB allocated (11.7 GiB
   reserved): the per-step state clones (5.4 GiB) plus a 1.28 GiB `[B,K,K,64]` pair-bias temporary with
   K ~ 760 slots (the slot count grows with the entry pad at 300 s). The 90 s fractions (.04 of 49 GB = 1.9 GiB)
   OOM at 3.6 GiB (attempt 1) and at 7.1 GiB (attempt 2); a forward worker needs >= ~11 GiB, so two forward
   workers cannot share a 24 GiB card.
2. **Host RAM kill (the real wall).** In the replay forward the per-step `reward_fn` (`_ttc_cost` is
   `O(K^2)` pairwise, ~10 saved `[1,K,K]` tensors per row per step) is *outside* the segment checkpoints, so
   LONGMEM's saved-tensor spill copied hundreds of GB of pairwise tensors into pinned host RAM; the cgroup
   (96.6 GB on both boxes) killed the process (`Killed`, rc 137, attempt 4). At 90 s this was ~36x smaller.
   Fix 2a (`WMS_LM300_RECOMPUTE_REWARD=1`): checkpoint the reward call -> spill 1.4 GB, RSS 5 GB. But the
   engine unpacks (recomputes) hundreds of reward frames before the TTC branches run, so 16 GiB of
   recomputed `[K,K]` intermediates were alive at once and backward OOMed at 21.8 GiB (attempts 5, 6).
   Fix 2b (`WMS_LM300_RECOMPUTE_REWARD=2`, `dreamer._ImmediateVJP`): a custom autograd node that stores only
   the inputs and recomputes AND differentiates one call inside its own backward, so at most one call's
   intermediates exist. Same ops on the same inputs -> bitwise equal values and gradients (CPU test).
3. **Segment window.** With (2b) the OOM moved to the WM segment recompute: the pair-bias MLP saves ~3 x 1.28 GiB
   per step, so a segment-2 window held 7.7 GiB (segment 8: 10+ GiB at the 9th allocation); the resident floor
   (state clones 5.4 + merge-pivot boundary states 2.3 + history/other ~3) is 12 GiB. Fix 3
   (`WMS_LM300_PAIRBIAS_CKPT=1`, `residual_net._pair_bias`): checkpoint the pure pair-bias MLP so the window
   shrinks to one step's transient. Segment 2 then peaks at 15.8 GiB allocated.
4. Segment size: the *larger* segment is NOT cheaper at 300 s (segment 8 OOMed earlier than segment 2): the
   window grows ~3.8 GiB/step while boundaries shrink only 0.5 GiB/segment doubling. Keep segment 2 (measured);
   segment 1 was not needed (run cancelled after 2 min when pair-bias checkpointing was ready).

## Attempt table (gpuR 23.5 GiB, one 300 s update, long credit ON, a190 actor)

| # | config | result |
|---|--------|--------|
| 1 | pool W=2 slots 1, fwd .15 (3.6 GiB) | forward OOM (3.39 GiB alloc) |
| 2 | pool W=2, fwd .30 (7.1 GiB) | forward OOM (needs a 1.28 GiB temp on 5.2 GiB) |
| 3 | serial, inherited `gpu_memory_fraction` .45 (10.6 GiB) | forward OOM |
| 4 | serial, fraction .96, seg 2, no LM300 flags | forward OK (917 s, 10.1 GiB), replay -> **host RAM kill** (cgroup 96 GB) |
| 5 | + reward ckpt mode 1, seg 2 | replay OK (10.2 GiB resident), backward OOM 21.8 GiB (16 GiB TTC frames held) |
| 6 | mode 1, seg 8 | same OOM (TTC frames), resident 7.8 GiB |
| 7 | mode 2, seg 8 | OOM in segment recompute (pair-bias 1.28 GiB x 8 steps) |
| 8 | mode 2, seg 2 | OOM, window 7.7 GiB + resident 12 GiB |
| 9 | **mode 2 + pair-bias ckpt, seg 2, serial** | **PASS: future-0 VJP peak 15.76 GiB alloc, 18.2 GB device; 417 s/future** |

## Files changed (all default OFF, exact)

* `src/wms/control/dreamer.py`: `_ImmediateVJP/_ImmediateRecompute`; `WMS_LM300_RECOMPUTE_REWARD` (1 = standard
  reward checkpoint, 2 = one-call-at-a-time recompute), memory mode only.
* `src/wms/model/residual_net.py`: `WMS_LM300_PAIRBIAS_CKPT=1` checkpoints `_KinAttention._pair_bias`.
* `src/wms/control/parallel_futures.py`: `WMS_SPEED_FUTURE_BUDGET_CAP` (default .60 unchanged; <=.95) so a
  dedicated card may exceed the shared-card 60% guard; `WMS_SPEED_FORWARD_WORKERS` sizes the no-grad forward
  pool separately from the VJP pool (default = `WMS_SPEED_FUTURE_WORKERS`).
* `src/wms/control/imagination_memory.py`: `WMS_LM300_TRACE=1` per-event CUDA peaks / host RSS / spill counters;
  `WMS_LM300_SNAPSHOT=<dir>` allocator snapshots (forward-done + on OOM). Diagnostics only.
* `configs/c4_policy14_longmem300.toml`, `tools/longmem300/{probe.sh,sampler.sh}`.
* `tests/unit/test_longmem300_budget.py` (13 tests, CPU): default-OFF, bitwise-equal gradients for reward
  recompute modes 1 and 2 and for the pair-bias checkpoint, budget-cap and forward-pool validation. Pre-existing
  environmental failures in `test_longmem.py` (2) are identical on the c4prep baseline.

## Exact fork command (single 24 GB-class GPU; works on gpuA too, alone on the card)

Fork from a candidate-A phase checkpoint `policy_0000NN.pt` (written with `imagination_checkpoint_segment = 2`,
so NO `--resume-allow` key is needed or accepted; a segment other than 2 needs exactly
`--resume-allow trainer.imagination_checkpoint_segment`). `--max-updates` must be a multiple of 10 (<= 150)
for native-init.

```sh
# on the box, staged tree = c4/longmem300 (or candA zs tree + these patches) at $RT; $R holds out/
cd /root/autodl-tmp/c4/longmem300        # R; probe.sh also takes RT, INP, DATA_NATIVE, DATA_GENRAND
TAG=fork_NN FORK_CKPT=/root/autodl-tmp/c4/<candA_run>/policy_0000NN.pt MAXU=30 \
  nohup setsid bash tools/longmem300/probe.sh > out/fork_NN.log 2>&1 < /dev/null &
```

`probe.sh` writes `configs/_lm300_<TAG>.toml` (extends `c4_policy14_longmem300.toml`; segment, coordinator
fraction), exports the full S-ZEROSTART env (leader rule, mode-2 law, FGA flags) plus the two LM300 switches, runs
`wms.control.policy14_cli ... --new-phase --device cuda --seed 965140 --max-updates $MAXU`, and samples GPU /
host memory every 2 s into `out/<TAG>.gpu.csv`. Stop with `touch out/<TAG>/STOP` (checked between updates).

## Projected time (serial recipe)

Per update: startup/rescore overhead ~6 min (first update) + forward 915 s + 8 x (156 s replay + 261 s VJP)
= ~75 min -> **20 updates ~25 h, 30 updates ~37 h** on gpuR-class hardware (4090 24 GB). Not within a night.

## gpuA (47 GB) expected configs - UNMEASURED (extrapolated from the numbers above)

* Serial (same recipe) alone on the card: identical memory (~18 GB device), ~75 min/update.
* **Two-VJP-worker pool** (recommended speed-up, needs a 1-update qualification on gpuA before adoption):
  `WORKERS=2 SLOTS=2 FWD_WORKERS=3`, VJP fractions `ACTIVE=.40` (19 GiB each; measured need 15.8 GiB alloc),
  forward `FWD=.26` (12.3 GiB each; measured need 10.1 GiB alloc), `LOW=.02`, `CAP=.90`, coordinator `.10`.
  Expected: forward 915/3 = ~5 min, VJP 8 x 417/2 = ~28 min -> ~35-40 min/update (~12-14 h for 20 updates).
  Three VJP workers do not fit (3 x ~17 GiB reserved = 51 GB).

## Measured result of the complete update

(appended below when the first full update finished)
