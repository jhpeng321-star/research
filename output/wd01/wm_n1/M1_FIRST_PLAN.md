# M1-FIRST PLAN — one Idea A model first (PREP ONLY)

**Status:** preparation only (2026-10-01). No dispatch to any GPU box, no
training, no real-data reads. Branch `zcode/m1bind` at `zcode/m1first`
`38c68fe`; cwd `output/wd01/wm_n1`. This document records the commands, gates
and estimates for the owner's "ONE model first" decision, now BOUND to the
frozen Amendment 4 CAL-tune/CAL-cal split
(`REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md`, `CAL_SPLIT_A4.json`,
`A4_FREEZE.json`).

**Scope implemented in this change**

1. GPU-A M1-FIRST mode: the 1-h timing gate and the owner `APPROVAL.json` before
   member 1 are UNCHANGED; the queue can stop after Idea A member 1. Members 2-5
   are ADDITIONALLY gated by a separate owner file `APPROVAL_A_M2_5.json`.
2. GPU-B mode: run ONLY held-out (i) `J4 -> J4B` (one member), then stop.
   Held-out (ii), (iii) and the variant-N comparator are NOT queued.
3. `train.idea_a` QUICK-TUNE: a CAL-only improvement loop that trains member 1
   on a fixed, world-stratified TRAIN subset and a fraction of the epochs, with
   the same seed.
4. `bundle.envcheck` accepts Ada `sm_89` (only Blackwell needs `sm_120`).

Nothing here changes the frozen baselines, the four frozen registration
records, the sealed chunk-5 TEST, or the member-1-first rule.

---

## 1. New box-config variant: `bundle/box_configs/2box/c_m1only/`

`bundle.multibox.build_m1_only_configs()` / `write_m1_only_configs()` generate
the files (the existing `2box/{a,b,c}`, `3box`, `4box` configs are unchanged):

| box | `assigned_configs` | `stop_after` | role |
| --- | --- | --- | --- |
| `box-A` | `["member-1"]` | `member-1` | Idea A member 1 ALONE, then STOP |
| `box-B` | `["heldout-i-j4b"]` | `heldout-i-j4b` | held-out (i) J4→J4B only, then STOP |

Amendment 5 (`REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md`, LF `7788c270…`):
the M1-only configs run on the NEW **pack_v3** (`corpus_source`
`D:/rd-staging/wmn1_feat/pack_v3`, `format: "v3"`, `k_max: 192`), not the frozen
pack_v2.  The 1-h timing gate and `bundle.envcheck` are v3 there (`--format v3`);
every other box plan keeps pack_v2.

`box_configs/2box/c_m1only/index.json` records `"m1_only": true`. The loader
rejects any `stop_after` that is not one of the box's assigned configs.

Box A command (from `output/wd01/wm_n1` on the GPU-A box):

```
python -m bundle.envcheck --device cuda --format v3
python -m bundle.multibox run-box-config \
    --config bundle/box_configs/2box/c_m1only/boxA.json --work-dir $WORK
```

or, through the sequential queue (same stages, explicit stop):

```
python -m bundle --manifest $WORK/upload/manifest.json --work-dir $WORK \
    --approval $WORK/approvals/APPROVAL.json \
    --stop-after member-1 --device cuda
```

This runs `verify-upload -> convert+verify -> timing-gate -> approval ->
member-1` and STOPS. The timing gate runs on pack_v3 (K=192) and writes
`$WORK/TIMING_GATE.json`; the box STOPS there until the owner `APPROVAL.json`
binds that file's sha. The default `--stop-after timing-gate` and the
`stop_after="member-1"` M1-FIRST mode both refuse to train members 2-5.

Box B command:

```
python -m bundle.envcheck --device cuda --format v3
python -m bundle.multibox run-box-config \
    --config bundle/box_configs/2box/c_m1only/boxB.json --work-dir $WORK
```

Equivalent single-member argv (the exact training entry point):

```
python -m train.idea_a --member 1 --seed 101 --corpus $WORK/pack \
    --out $WORK/members/heldout-i-j4b --device cuda --ensemble-k 1
python -m evalh.idea_a heldout-config --test i \
    --out $WORK/eval/heldout-i-j4b \
    --predictor adapter.predictor:IdeaABundlePredictor \
    --reference evalh.climatology:ClimatologyPredictor
```

---

## 2. Gate chain (unchanged before member 1)

| # | gate | file | enforced by |
| --- | --- | --- | --- |
| 1 | sealed-shard refusal | upload manifest | `bundle.queue.SealedShardRefusal` |
| 2 | pack verify | `<work>/pack` | `corpus.verify` |
| 3 | 1-h real-member timing gate | `<work>/TIMING_GATE.json` | `bundle.timing.run_timing_gate` (v3, K_max=192 pack_v3) |
| 4 | owner approval before member 1 | `APPROVAL.json` | `bundle.approval.require_approval` |
| 5 | member-1-first | `members/member_1.pt` | `train.idea_a.IdeaATrainer` |

### 2.1 Members 2-5: separate `APPROVAL_A_M2_5.json`

Members 2-5 are gated in addition to gate 4. The separate owner file
`schema: "wm_n1_approval_a_m2_5/1"` must bind:

* `registration_lf_sha256` = the same frozen `REGISTRATION_WM_N1.md` LF-sha
  (`666233aa…93246`), re-hashed from the record (CRLF-safe);
* `amendment_lf_sha256` = the same frozen amendment-1 LF-sha
  (`decb9656…16c57`) (`REGISTRATION_WM_N1_AMENDMENT_1.md`);
* `amendment_4_lf_sha256` = the frozen Amendment 4 LF-sha
  (`d392b300…0822f`) (`REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md`);
* `cal_split_lf_sha256` = the frozen CAL split LF-sha
  (`bc4e32f2…20d46`) (`CAL_SPLIT_A4.json`);
* `m1_checkpoint_sha256` = raw sha256 of the completed
  `members/member_1.pt`;
* `winning_config_sha256` = raw sha256 of the selected `WINNING_CONFIG.json`
  (`bundle.winning_config`; schema `wm_n1_winning_config/1`, A4.2 knobs, round
  log, CAL-tune scores — validated by the gate, not just hashed).

The member-1 `APPROVAL.json` (schema `wm_n1_approval/1`) binds the same
Amendment 4 + CAL-split LF-shas in addition to registration/amendment-1 and the
TIMING_GATE sha; every per-config `APPROVAL.json`
(`wm_n1_multibox_approval/1`, including the `2box/c_m1only` boxA/boxB configs)
does too (`bundle.multibox.require_config_approval`).  The per-config binding is
unchanged in shape and additionally requires the frozen Amendment 5 LF-sha
`amendment_5_lf_sha256 = 7788c270…d13ae7`
(`REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md`, CRLF-safe); a missing, wrong or
tampered amendment-5 record is refused
(`bundle/tests/test_multibox.py::test_approval_wrong_sha_refused[...amendment_5...]`).

Until this file matches the actual checkpoint/config bytes, `members-2-5`
refuses (`bundle.approval.ApprovalMissing`) and no member 2-5 step runs.

Continue to members 2-5 only after the owner places both files:

```
python -m bundle --manifest $WORK/upload/manifest.json --work-dir $WORK \
    --approval $WORK/approvals/APPROVAL.json \
    --a-m2-5-approval $WORK/approvals/APPROVAL_A_M2_5.json \
    --winning-config $WORK/WINNING_CONFIG.json \
    --stop-after members-2-5 --device cuda
```

`Approval`/`Timing-Gate`/`Registration`/`Amendment` identities are unchanged
for gates 1-4; the A_M2_5 file is purely additive.

---

## 3. QUICK-TUNE (CAL-only improvement loop)

CLI (member 1, same seed as the full member):

```
python -m train.idea_a --member 1 --seed 101 --corpus $WORK/pack \
    --out $WORK/quick/m1 --device cuda --ensemble-k 1 \
    --quick-tune --subset-frac 0.25 --epoch-frac 0.35
```

* `--subset-frac 0.25`: keep `round(0.25 * n_world)` items of each world (at
  least one), drawn by `world_stratified_subset(items, frac, seed)` with the
  member seed. Deterministic: the same `(items, frac, seed)` selects the same
  positions; every world survives, so the subset stays world-stratified.
* `--epoch-frac 0.35`: run `round(0.35 * epochs)` epochs (never < 1).
* Same seed (101) as the full member, so quick and full share RNG and world
  selection; the quick run writes its subset receipt into `TRAIN_RECEIPT.json`
  (`quick_tune`, `subset_frac`, `epoch_frac`, `subset.per_world`).
* `--dry-run` prints the quick-tune setting without touching data.
* CAL is used for the improvement loop diagnostics only; per amendment A2 a
  quick/early model supports NO selection or superiority claim, and never runs
  on chunk 5.

### 3.2 Amendment 4 CAL split (selection vs calibration) — FROZEN

`REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md` (LF `d392b300…0822f`) freezes a
MANDATORY world-level CAL split in `CAL_SPLIT_A4.json` (LF `bc4e32f2…20d46`),
materialised before any training:

* **CAL-tune** — the improvement-loop selection metric and any quick-tune
  scoring read ONLY these worlds;
* **CAL-cal** — early stopping, calibration, blend weights and variant choice
  read ONLY these worlds.

`evalh.cal_split` is the single loader/filter. `train.selection_worlds` /
`calibration_worlds`, `evalh.run_eval.selection_worlds` / `calibration_worlds`
(`--cal-half`) and `combine.guards.selection_worlds` / `calibration_worlds`
(`guard_cal_only(..., half=…)`) all resolve through it and raise
`CalSplitError` if handed a world from the other half. The improvement loop
selects the winner into `WINNING_CONFIG.json` (`bundle.winning_config`), which
`APPROVAL_A_M2_5.json` binds.

### 3.1 Rank-fidelity check design

Quick-tune only reduces wall-clock; it must not be trusted to choose a config
unless its ranking matches the full run. The check:

1. Fix round 0 with the same seed (101) and the same world-stratified subset.
2. Take the 3 candidate configs with the largest expected effect (e.g. the 3
   top variants from the design's CAL improvement loop).
3. Run each candidate BOTH quick (`--quick-tune --subset-frac 0.25
   --epoch-frac 0.35`) and full (`epochs=15, steps_per_epoch=2000`), all
   diagnostic-only on CAL.
4. Compare the CAL diagnostic loss rank order. Accept quick-tune for pruning
   only if the top-1 candidate agrees AND the Spearman rank correlation over
   the 3 configs is 1.0. Otherwise widen to 5 configs or fall back to full.
5. Record the quick/full ranks in the experiment ledger; a quick-tune ranking
   is never reported as a result.

---

## 4. Estimates for a 48 GB RTX 4090 D

**Assumptions**

* Design envelope (amendment 3 A3.7): one Idea A member costs **9-17 GPU-h**
  (midpoint 13.0) on an **A10G-class** reference card.
* The RTX 4090 D (Ada, sm_89, 48 GB) is **~2-3x** A10G throughput; divide the
  reference by 2 (optimistic) to 3 (conservative). This is a throughput
  scaling assumption, to be replaced by the actual 1-h timing gate.
* torch `>=2.7,<2.13` cu128; `python -m bundle.envcheck --format v3` must PASS
  on Ada (sm_89 accepted; no sm_120 token required for non-Blackwell).
* Host: 208 vCPU / 1 TB RAM. The packed corpus loader is numpy memmap; with
  1 TB RAM the working pack can be fully page-cached, so dataloading is not
  the bottleneck. Use several CPU worker processes, bounded by the GPU feed
  rate (start ~8-16, tune with `nvidia-smi`/loader timing); CPU affinity and a
  single GPU stream keep the estimate GPU-bound.

| job | reference GPU-h (A10G) | 4090 D @2x | 4090 D @3x | planning midpoint (@2.5x) |
| --- | --- | --- | --- | --- |
| Idea A member 1 alone (GPU-A) | 9-17 | 4.5-8.5 | 3.0-5.7 | **~5.2 h** |
| held-out (i) J4→J4B alone (GPU-B) | 9-17 (1 member) | 4.5-8.5 | 3.0-5.7 | **~5.2 h** |
| one quick-tune round (`epoch_frac=0.35`) | 0.35 x member | 1.6-3.0 | 1.05-2.0 | **~1.8 h** |

Notes:

* The quick-tune round is `epoch_frac x` the full member wall-clock at a fixed
  `steps_per_epoch`; `subset_frac` changes data coverage (and therefore rank
  fidelity), not per-step cost. If a round additionally scales
  `steps_per_epoch` by `subset_frac`, the round falls to
  `0.25 x 0.35 = 0.0875 x` the full member (~0.26-0.74 h on a 4090 D).
* Each single-member job (M1 and held-out (i)) is one 1-member model; the
  envelope is reused from the per-member design cost.
* GPU-A M1-FIRST total before any human decision: ~5.2 h (midpoint); GPU-B
  held-out (i) total: ~5.2 h. The two boxes run in parallel, so the one-model
  milestone lands in ~5-9 h wall-clock (timing-gate dependent).

### 4.1 CPU micro-benchmark (quick vs full ratio)

Run on the local CPU box with synthetic windows only (no real corpus, no
SUMO): real Idea A fake-module bundle, member seed 101, `batch_size=2`,
`steps_per_epoch=5`, after a warm-up fit.

| run | epochs | steps | median wall (s) |
| --- | --- | --- | --- |
| full | 6 | 30 | 1.113 |
| quick (`subset_frac=0.25`, `epoch_frac=0.35`) | 2 | 10 | 0.368 |

Measured quick/full wall ratio = **0.331** vs the expected step ratio
**0.333** (`0.35` epoch fraction rounded to 2/6 epochs), confirming the
wall-clock model is `epoch_frac x` full and that the world subset does not
change per-step cost. Harness (synthetic only): `sys.path.insert(0,
"train/tests")`, a tiny `_WindowCorpus(windows, worlds)`, and
`test_idea_a_fakes.seeded_bundle(101)`; time `IdeaATrainer.fit()` for the full
config (`epochs=6, steps_per_epoch=5`) and the quick config (`epochs=6,
steps_per_epoch=5, quick_tune=True, subset_frac=0.25, epoch_frac=0.35`), median
of 3 after a warm-up fit. Real stdout:

```
full median_s=1.113 (steps=30)
quick median_s=0.368 (steps=10, epochs=2, n_kept=4/8)
measured quick/full wall = 0.331 ; epoch_frac = 0.350 ; expected ratio = 0.333
```

---

## 5. Environment (Ada acceptance)

`bundle.envcheck` refuses only: torch `<2.7`, a Blackwell-class GPU (compute
capability major 12) whose arch list lacks `sm_120`/`compute_120`, or
`--device cuda` with CUDA unavailable. An Ada `(8, 9)` / sm_89 card passes:
only Blackwell needs sm_120. Test:
`bundle/tests/test_envcheck.py::test_evaluate_accepts_ada_sm89`.

```
python -m bundle.envcheck --device cuda --format v3
```

---

## 6. Verification (this prep change)

```
python -m pytest bundle/tests train/tests -q
```

New coverage: `stop_after member-1` M1-first mode; `APPROVAL_A_M2_5` refusal
(missing/wrong m1 checkpoint sha, wrong winning-config sha, missing files);
Amendment 4 + CAL-split binding on every `APPROVAL.json` (incl. the
`2box/c_m1only` boxA/boxB configs) and on `APPROVAL_A_M2_5.json` (wrong sha,
tampered record, invalid winner); `c_m1only` box configs and `run-box-config`
stopping after member 1 (GPU-A) and after held-out (i) (GPU-B); quick-tune
subset determinism + world stratification + epoch fraction; Ada sm_89
acceptance; `bundle.winning_config` schema/writer; CAL-tune/CAL-cal refusal on
each of the train, evalh and combine paths (`evalh.cal_split` synthetic
world lists).

**Stop / dispatch decision.** This is prep only. GPU dispatch requires the
owner's explicit approval of the timing-gate sha, and members 2-5 require the
additional `APPROVAL_A_M2_5.json` binding.
