# WM-N1 TRAIN stage launcher

`run_train_stage.py` defines three separate, preregistered stages. It is source
only at this handoff: no pack, shard, sidecar, outcomes, or training data were
opened, and no stage was run. The integrating reviewer controls the actual
allocations, test guard, and experiment records.

The invocation is `python run_train_stage.py --policy POLICY.json
--policy-sha256 REVIEWED_RAW_SHA256`. The policy is an immutable JSON file in
the current worktree. Its raw SHA is recorded by an external reviewer before
launch. The launcher never calculates a missing policy, source, guard, runtime,
or data pin for inclusion in the policy. One invocation runs one stage and
creates one new, exact E: sandbox; it never resumes or retries. The policy has
`version: 1`, `stage`, `sandbox`, `stage_inputs`, `files`, `interpreter`, and
`pythonpath`. `files` maps absolute literal paths to `{role, sha256}` where
role is `code`, `config`, or `data`. The exact guard files
`sitecustomize.py`, `train_stage_guard.py`, and `run_train_stage.py` must all be
externally pinned as `code`; so must the project sources imported in that
stage. `interpreter` contains externally pinned `path` and raw `sha256` for
the reviewed `rl_driving` interpreter. `pythonpath` includes this tools folder
and `output/wd01/wm_n1`. There are no external helpers.

The production `stage_inputs.pack` is exactly
`D:/rd-staging/wmn1_feat/pack_v3`. The sidecar stage alone gets
`shards_root` exactly `D:/rd-staging/wmn1_feat/shards_v3`. A synthetic test
fixture must explicitly set `synthetic_test_mode: true` and name its fresh E:
`synthetic_root`; it has no production authority. `sandbox` is a fresh direct
child of `E:/tmp/wmn1_train_runs_20261002`, outside the worktree and runtime.
All outputs are fresh direct children
of that sandbox. Every prior output used by a later stage is separately
reviewed and registered by exact path and raw SHA. An output cannot
automatically advance to the next stage.

1. `metadata-plan`: `stage_inputs` has `pack`, `plan_out` only. Its data
   registrations are precisely the pack `manifest.json` and `INDEX.json`.
   Under the installed child guard, the accepted
   `train.identity_sidecar._plan` reconciles every INDEX run and manifest item,
   refuses protected content, and derives only TRAIN shard identities. The
   fresh `PLAN_RECEIPT.json` records the complete plan, exact pack path,
   metadata raw SHAs, and canonical plan SHA. No shard or outcome is opened.
2. `train-sidecars`: inputs add the exact prior `plan_receipt` file,
   `shards_root`, the ordered `selected_shards` list, and an ordered
   `positive_shards` subset copied from the reviewed plan's runs with
   `n_outcome_rows > 0`, plus fresh `identity_out` and `sidecar_out`. Its only
   shard registrations are `DONE.json`, `slot_vid.json`, and `slot_index.npy`
   for every selected TRAIN run, with `outcomes.parquet` only for positive-row
   runs. The parent validates both lists against the pinned plan before any
   shard hash. The guarded child rederives the plan and checks each DONE
   identity, count, and source hash; zero-row DONE records must omit the
   outcome output key and have no outcome file. Only then does
   `train.label_isolation.build_sidecar` read selected TRAIN outcomes. The
   mixed pack outcome table, root shard INDEX, and CALIBRATION shards are not
   registered. The label builder's sole receipt rename is the guard's narrow
   fresh `.tmp` to `TRAIN_ISOLATION_RECEIPT.json` exception.
3. `cpu40`: inputs add the prior `plan_receipt`, `identity_dir`,
   `sidecar_dir`, a fresh `cpu_out`, and `num_workers` fixed to 0 or 2. Its
   data registrations are pack metadata, the plan receipt, exactly 15
   `TRAIN_ARRAYS` pack feature arrays, four identity sidecar files, and two
   label sidecar files. No shard path is registered. Both sidecar receipts
   validate against the exact pack and selected TRAIN coverage before the
   first array or parquet read. The frozen A6 document is externally pinned
   and checked by `evalh.seals.require_frozen_amendment_6`. The launcher calls
   the current `train.idea_a.main` with member 1, seed 101, CPU, K=192,
   ensemble size 5, one epoch, 40 steps, every-step logging, and the explicit
   worker count. The canonical config keeps batch size 8, all A3.2 losses,
   A6 state token, and full TRAIN sampling. It does not set quick tune,
   incomplete-loss, or A6-off flags.

The parent validates the policy structure and exact stage file set before
hashing registered data. It imports no project or native module; only the
guarded child calls `train.identity_sidecar._plan`. The parent hashes each
selected data file once and passes the exact path-to-SHA `verified_data`
attestation and, for later stages, `selection_sha256` in an exclusively created
`sandbox/child_spec.json`. The child's environment contains only that spec's
absolute path and raw SHA, avoiding Windows environment size limits.
For CPU, it also sets the exact approved canonical pack and TRAIN sidecar
environment paths; other stages clear those overrides. `sitecustomize` installs the
guard before any child project or native import; Windows DataLoader workers
inherit the same startup policy. The guard rechecks source/runtime pins on
startup, and the child rederives the TRAIN plan before stage work.

For a separate synthetic metadata-stage startup proof, the reviewed policy
sets `synthetic_test_mode: true`, includes the exact source pin for
`tests/synthetic_startup_probe.py`, and adds that tests directory to
`pythonpath`. After the guarded metadata plan, the probe starts one inherited
Python child and records distinct PIDs, the identical raw spec SHA and policy
fingerprint, and active native wrappers. It also refuses two public Arrow
filesystem URI factories on a lexical forbidden D: probe path before any
native open, then checks Arrow/pandas Parquet and NumPy read-only mmap on tiny
new files under its guarded sandbox. Those synthetic files remain for review.
Production stages do not invoke it.

Every stage keeps `receipts/STARTED.json`, `child.log`, and a new
`receipts/RESULT.json` plus `SUCCEEDED.json` or `FAILED.json` where possible.
The logs and any partial outputs remain in place on failure. The guard is a
CPython in-process policy with the native interception limits documented in
`train_stage_guard.py`; it is not an OS sandbox. A registered stage is an
execution allocation, so the integrating reviewer must register and review
each stage before running it and update the ledger afterward.
