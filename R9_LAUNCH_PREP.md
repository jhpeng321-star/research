# R9 launch preparation (S-PREP), 2026-10-22

Branch `c4/v10r9launch` (base `0aba97cc`). Scope: configs/scripts/docs only; `src/wms/train/losses/merge_pivot.py`
and `tools/r9_*.py` untouched. **Nothing was launched on gpuR; nothing on gpuR was deleted.**
Evidence labels: [Measured] = observed in this session, [Read] = from code/docs, [Proposed] = my recommendation.

## 0. Verdict

* Config `configs/c4_wm_v10round9.toml` is correct against the code's option names and the astra advice; **no edit needed**.
* The real entrypoint runs end-to-end on local TRAIN data up to the loss, but **no optimizer step completed**:
  the first-passage loss aborts (`non-finite first_passage loss`) because of zero-mass events (S-SUP's area).
* **NEW BLOCKER for the gpuR gate (not S-SUP's zero-mass issue): hybrid quadrature memory.** On the real TRAIN
  slice the F2 pre-window prefix is up to 2120 steps long and `hybrid_log_masses` keeps an O(cars x T^2) autograd
  graph. One 30 s step with batch 1 reached >12 GB host RAM (killed by my guard) at quadrature order 8 and ~8.5 GB
  at order 32 even after an exact compaction. gpuR has a 24 GB GPU; a batch-4 60/120 s smoke is very likely to
  OOM. See section 3.
* Gate script, ship script and gpuR inventory are ready (sections 4-6).

## 1. Config check (task 1)

`model.merge_decision` is the real key (the brief's "merge_mode" does not exist); `merge_passage_law`, `merge_w_init`,
`merge_quadrature_order`, `merge_negotiation` exist in `src/wms/config/train.py` (validation: first_passage requires
B off, `loss.merge_core` requires first_passage/secure_gap). The extends-depth cap is 16; round 9 already sits at
exactly 16, so a child config of it fails ("chain is too deep") -- local smoke therefore uses `--set` overrides.

Effective diff round 8 -> round 9 ([Measured], resolved configs):

```
model.merge_decision: secure_gap -> first_passage        model.merge_passage_law: strict -> hybrid
loss.merge_core.yield_weight 0.4 -> 0.0                  loss.merge_core.location_reg_weight 1e-3 -> 0.0
loss.merge_core.sigma_reg_weight 1e-3 -> 0.0             physics.merge_end_wait True -> False
optim.freeze_prefixes: [merge_core.temperature_raw, merge_core.residual_head, merge_core.residual_scale_raw]
   -> [driver_encoder, driver_updater, gate_, lane_gate_, history_proj, k_embedding, lane_v5, merge_core, residual_net, temporal]
optim.lr_scale: {lane_v5 .05, driver_* .3, ..., merge_core* 20-100}  ->  {merge_pivot: 20.0}
runtime.out_dir -> runs/active/c4_wm_v10_round9
unchanged and relevant: merge_w_init=1.0, merge_quadrature_order=32, merge_negotiation=false, z_tf=395.2, z_on=2.15,
weights tf 0.6 / on 0.4, tau_reg_weight=w_reg_weight=0 (penalties only via grouped inner validation, per advice F1.8),
init round 8 stage300, P4 context off, B off.
```

Checks [Measured] from the optimizer log line of the real run: `trainable_parameters: 8, trainable_tensors: 8`, one
group `{prefix: merge_pivot, lr_scale: 20}`, `frozen_parameters: 846445`, `unmatched_freeze_prefixes: []`,
`unmatched_lr_scale_prefixes: []`; `init_from wm_s_stage300.pt: loaded 149 tensors, expanded 0, new 8, shape-mismatch 0,
unused 0` (the 8 new tensors are the six gap parameters, tau, w). F2 (survivor conditioning from recovered pre-window
history) has no config key: it is switched on by `first_passage` through `merge_pivot_history=True` in
`src/wms/cli/assemble.py`; the run logged `pre_window_recovered 4 / unknown_age_after 0`. Initial `w=1.0`, `tau=0.0727`.
The stage learning rate 5e-5/3e-5 is multiplied by 20 for the merge_pivot group as drafted; the advice does not
specify an lr scale, so it is left as the owner's draft value.
`wms train --mode preflight` reports two pre-existing, config-chain issues that are NOT run-mode failures:
"select_by_long_horizon requires a validation window list" (the C4 sources provide the split; run mode accepts it)
and the missing local `genrand_branch` / `pairs300` roots (present on gpuR).

## 2. Local CPU micro-smoke (task 2)

Data: trimmed copies of 4 episodes per pack from `E:/agent-work/c4-v10r9fix/data/r9_train/{v5_native,genrand_plain}`
into `runs/active/smoke_data` (gitignored). The original 8 episodes per pack are all `split_role: val`, so the copied
manifests relabel 3 as train + 1 val (derived smoke manifest only; the source files and manifests are unchanged). These
are TRAIN-pack worlds; no sealed/holdout path was opened. Entry: `scripts/r9_local_smoke.sh` (real `python -m wms train`
with the round-9 config, `--set` overrides only for sources/paired terms/validation/device/curriculum).

| run | settings | result |
|---|---|---|
| local (1st) | 60/120 s, batch 2 | CPU OOM in `hybrid_log_masses` (on-policy scoring), ~17 min in |
| local2 | 30/60 s, batch 1, order 32 | reached 16 GB RSS, OOM in `conditional_log_hazards` |
| local3 / local4 | same, order 8 | killed by my 12 GB guard; shape probe: `hybrid_log_masses` called with path shape (4, 2120) and (1, 238, 2180) (T = 2120-2180 steps, finite columns ~230-280) |
| local5 | same + exact column compaction (probe), order 32 | peak 8.5 GB; reached the loss, then `RuntimeError: non-finite first_passage loss` |
| local6 | same + compaction, `runtime.seed=7` | same abort |

Metrics at the abort (identical key set on every run), [Measured]:
```
local5: tf_events 4, tf_positive_mass_events 3, tf_zero_mass_events 1, tf_physical_fail_events 1, tf_nll inf,
        on_events 0, on_scored_intervals 7, on_zero_mass_events 0, tf_scored_seconds 70.5, tau 0.0727, w 1.0
local6: tf_events 4, tf_positive_mass_events 3, tf_zero_mass_events 1, tf_physical_fail_events 1,
        on_events 3, on_zero_mass_events 3, on_physical_fail_events 3, on_nll inf, on_scored_intervals 180
```
So zero-mass events are nonzero on this snapshot (every zero-mass event coincides with a physical-veto event): that is
the support problem S-SUP is fixing. The trainer's behaviour is a hard abort, so the gate cannot pass until it is fixed.

**train_log.jsonl keys.** Not observed in a file (no step finished). They are written by
`record.update(merge_pivot_metrics(...))` in `src/wms/train/unroll.py` (same function whose dict appears in the abort
message) plus `loss`, `stage_s`, `step` from the trainer. Exact names [Measured in the dict / Read for the log path]:
`merge_pivot_on_events`, `merge_pivot_tf_positive_mass_events`, `merge_pivot_tf_zero_mass_events`,
`merge_pivot_on_zero_mass_events`, `merge_pivot_on_scored_seconds` (= scored_intervals x dt), `merge_pivot_tau`,
`merge_pivot_w` (learned hybrid width; last chunk value), `loss`, `stage_s`. All gate keys exist under those names.

## 3. Hybrid-quadrature memory finding (new blocker)

* `hybrid_log_masses` builds, per step, tensors over [cars, knots x order] and keeps them for backward: O(cars x T^2).
  Synthetic probe [Measured] (`order=32`, CPU, random finite c, with grad): peak RSS per car ~0.02 GiB at T=60,
  ~0.1 GiB at T=120, ~0.4 GiB at T=240, ~2.6 GiB at T=600 (quadratic).
* Real windows are far longer than the chunk: the F2 pre-window replay covers the car's whole risk history
  (`pre_window_replayed_intervals 2120`), and `previous_logS = hybrid_log_masses(history, ...)` / the chunk path both
  integrate the full padded prefix. On gpuR the batch is 4 and stages are 120/300 s, so the same pattern at
  120-300 s windows x batch 4 x tens of risk cars is a GPU-OOM risk [Proposed estimate, not measured on GPU].
* Exact mitigation I verified (probe `scripts/r9_probe_compact_hybrid.py`, not applied to src): vetoed (`-inf`) steps have
  zero hazard, so columns that are `-inf` in every row can be dropped before quadrature and re-expanded after (events
  `-inf`; survival carried from the previous kept column, 0 before the first). Self-test with atoms and per-row
  vetoes: max |diff| 8.9e-16 in event and survival masses, identical tau/w gradients. It shrinks T from ~2150 to ~230-280
  (union of finite columns) and let the real run get past scoring (8.5 GB RSS at order 32, batch 1, 30 s). It is a
  numerical-exactness-preserving change in `src/wms/model/merge_pivot.py` (S-SUP's/main's file), so I did not apply it.
* Further levers [Proposed]: per-row compaction instead of the union of columns, segment gradient checkpointing of the
  time loop, smaller `optim.batch_size`/`truncation_steps` for the gate smoke. Lowering `merge_quadrature_order` is
  an accuracy trade (F1 doc: order 32 is not a blanket guarantee; needs the refinement certificate) -- do not use it
  just to fit memory without that certificate.
* Before the gpuR gate: run it with a GPU memory log (`nvidia-smi --query-gpu=memory.used -l 5`), and consider adding
  the compaction to `hybrid_log_masses` (needs its own regression test) first.

## 4. Gate script (task 3): `scripts/round9_gate.sh`

Copied from `E:/Temp/round9_gate.sh`; LF endings (pinned in `.gitattributes`); takes an optional repo dir argument
(`bash scripts/round9_gate.sh /root/autodl-tmp/c4/v5/repo_r9_<sha8>`), and refuses to run if `runs/smoke_r9`,
`runs/c4_wm_v10_round9` or `smoke_r9.log` already exist under `/root/autodl-tmp/c4/v5` (never overwrites a prior run;
exit 4). Checks (all kept): >= 4 steps; finite `loss` on every row; sum `merge_pivot_on_events` > 0;
sum `merge_pivot_tf_positive_mass_events` > 0; sum of tf+on zero-mass events == 0; sum of
`merge_pivot_on_scored_seconds` over rows with `stage_s >= 120` > 0; no `Traceback` in the log.
Changed/added: tau must be a finite number > 0 on every row (the old test accepted `null`); **new: `merge_pivot_w`
must be a finite number > 0 on every logged step** (missing key or null fails). The summary line prints `w_ok`.
Note the abort path: a non-finite loss raises, the log may then have < 4 rows and the traceback check fires, so the gate
exits 3 and does not start the full run. Exits: 2 repo missing, 3 smoke failed, 4 prior artifacts exist.
The smoke settings (60/120 s, stage steps 2+3, batch 4, max_hours 0.5) are the original ones; given section 3
they may need the memory mitigation first.

## 5. Ship script (task 4): `scripts/ship_r9_gpuR.sh`

`scripts/ship_r9_gpuR.sh <local-commit> [--local-only | --stage-only] [--skip-base-check] [--base <commit>]`
1. Base check: LF-normalised md5 of every text file under src/configs/tools/scripts/tests of base commit `21ca456f`
   versus gpuR `repo_r9_base` (aborts on any src/ or configs/ mismatch). gpuR files are CRLF (earlier Windows tars), so
   the comparison strips CR.
2. Delta = `git diff --name-status --no-renames 21ca456f <commit>`; A/M files are read with `git cat-file blob`
   (byte exact LF, no autocrlf/archive eol conversion) into a staging dir under `runs/active/ship_r9_<sha8>/`,
   tar'd (`--force-local`, needed on Windows paths), md5 manifest written, tar self-verified locally. Deletions are only
   listed (`deleted_not_applied.txt`), never applied. CRLF in a shipped `.sh` aborts.
3. Uploads tar + manifest to `gpuR:/root/autodl-tmp/c4/v5/ship_r9_<sha8>/` (refuses if it exists), compares tar md5.
4. Full mode: refuses if `repo_r9_<sha8>` exists, checks `need = du(repo_r9_base) + du(ship dir) + 512 MB < avail`,
   `cp -a repo_r9_base repo_r9_<sha8>` (hard links are deliberately not used: tar would write through them into the base),
   extracts, `md5sum -c` of every shipped file, `compileall src/wms`. `--stage-only` extracts the delta into
   `ship_r9_<sha8>/check/` (a few MB) and verifies without creating a repo dir.
No training, no gate, no deletion is performed by the script.

[Measured] dry runs: `--local-only` on `0aba97cc`: 72 files, tar 2,539,520 bytes, local self-check OK.
`--stage-only` on gpuR: remote verify OK for 72 files (0aba97cc) and 77 files (31145f43); staged under
`/root/autodl-tmp/c4/v5/ship_r9_0aba97cc/` (2.5 MB tar + check/ copy, left in place; the full `cp -a` dry run was not
done: 296 MB base copy exceeds the 200 MB allowance).
Base check after hashing fix [Measured, `--check-base-only`, 10 s]: **0 mismatched, 3 missing** -- the missing files are
`tests/test_merge_core_v2_{config,onpolicy,seqtf}.py` (never shipped for round 8; they arrive with the delta). (The first
dry run showed 3 spurious test mismatches caused by CRLF inside committed blobs and intermittent fork failures in the
per-file `git cat-file` loop; the script now hashes through one `git cat-file --batch` process with CRLF normalisation
on both sides.) Staging dirs locally are time-stamped (`runs/active/ship_r9_<sha8>_<time>`); the remote dir is keyed by
sha and never reused. `--stage-only` leaves `ship_r9_<sha8>/` (tar + check copy, ~5 MB) on gpuR; two such dirs exist now
(`ship_r9_0aba97cc`, `ship_r9_31145f43`, both from dry runs; remove them yourself if wanted, I did not delete anything).
Real run: `bash scripts/ship_r9_gpuR.sh <final-merged-commit>` (add `--skip-base-check` to skip the 10 s check).

## 6. gpuR state (read-only inspection, 2026-10-22 ~13:50 local)

* GPU: RTX 4090, 24564 MiB, 1 MiB used, 0 % util, no compute processes. Host RAM 754 GB total / ~327 GB free, `nproc` 224.
* Disk `/root/autodl-tmp` (md0): 50 G total, 48 G used, **2.9 G free (95 %)**, unchanged by the 2.5 MB stage-only dry run. A fresh
  `repo_r9_<sha8>` costs ~296 MB; a round-9 run's logs/checkpoints are ~tens of MB. Fine, but do not stage anything large.
* `repo_r9_base` (296 MB, mtime 2026-10-20 19:15, files CRLF): equals **commit `21ca456f` (round-8 config commit)** for
  every src/configs/tools/scripts file by git-blob sha1 after CR normalisation (best match of 150 candidate commits; the
  only differences are 3 test files absent and fixtures/binary-normalisation noise). No `.git`. Older dirs `repo_r2..r8`,
  `repo_r9` (stale Oct 20 copy), `repo_s3` exist; do not reuse `repo_r9` -- the gate refuses stale artifacts and the ship
  script writes `repo_r9_<sha8>`.
* Round-8 checkpoint `/root/autodl-tmp/c4/v5/runs/c4_wm_v10_round8/wm_s_stage300.pt`: md5
  `9f3a835943709577ff8dbf745ec7b502` = identical to the local copy `E:/agent-work/c4-v10r9fix/data/r9_train/wm_s_stage300.pt`.
* Data present: `data/{v5_native,genrand_plain,genrand_branch,pairs300,pairs300b}` (pairs300b held out). No `runs/smoke_r9` or
  `runs/c4_wm_v10_round9` exists yet. `runs/r9_measure` (36 K) exists.
* Delta size from base to `0aba97cc`: 72 files (31 added, 32 modified in code dirs + root docs), 2.5 MB tar.

## 7. Blockers / hand-offs

1. Support gate: zero-mass events (S-SUP). Required before the gate can pass.
2. Hybrid quadrature O(T^2) memory with long F2 prefixes: needs an exact compaction (verified) or checkpointing in
   `src/wms/model/merge_pivot.py` and a GPU memory trial before launch.
3. After the S-SUP branch merges: re-run `scripts/ship_r9_gpuR.sh <merged-commit>` (base unchanged), then the gate with
   the new repo dir, with GPU memory logging; confirm a real `train_log.jsonl` row carries the keys of section 2.
4. Committing note: the working tree here is `core.autocrlf=true`; `.gitattributes` now pins the two launch scripts to LF
   (blobs are LF either way and the ship script reads blobs).
