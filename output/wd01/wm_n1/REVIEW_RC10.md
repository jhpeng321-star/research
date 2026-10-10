# REVIEW — RC-10 chunk-5 TEST package (read-only)

**Reviewer:** independent acceptance review (WD-01, zero SUMO).
**Date:** 2026-09-30.
**Package under review:** `E:/rd-wt/rc10` uncommitted working tree (branch `checkpoint-2026-09-28`, HEAD `7350ee1`; a worktree of `E:/research-dense`).
**Frozen reference:** `REGISTRATION_CHUNK5_TEST.md` (this repo), LF-sha256
`f8c7d760baedeb359c501cd90a3e2a599fb0ddfef390ea555cf304c38214b31e`;
freeze record `output/wd01/wm_n1/C5TEST_FREEZE.json`.
**Scope:** read-only. No file in `E:/rd-wt/rc10` or `E:/research-dense` was edited;
no `runs/` outcome file, HOLDOUT or CAL file was opened; no SUMO/TraCI was started;
`git status` of `E:/rd-wt/rc10` was byte-identical before and after this review.
No run directory named `w8707..w8736` exists in `E:/rd-wt/rc10/runs` or
`E:/research-dense/runs`, and no `runs/campaign_stage1_chunk5` exists (checked by
name only).

## Verdict table

| # | criterion | verdict |
| --- | --- | --- |
| 1 | seeds diff LF-sha `efe1286d...`, adds only 8707-8736 + map, `git apply --check` on a copy, post-apply `71b0663a...` | **PASS** (see O1) |
| 2 | `WORLD_PLAN.json`: 30 worlds / 15 pairs x2 / seeds 8707-8736 once / TEST / t0 pool + plan seed / 12 runs / 234,000 sim-s | **PASS** |
| 3 | `chunk5_launch.py` refuses without AUTHORIZATION bound to `f8c7d760`; wall law v2.1 `6e0a0805`; chunk-4 code reused unchanged | **PASS** (see O4) |
| 4 | `grade_chunk5.py` reads only plan + receipts; never outcome fields | **PASS** (caveat O3) |
| 5 | `chunk5_test_seal.json` + `tools/holdout_guard.py` refuse chunk-5 TEST except `WM-N1_FINAL_TEST`; tests pass | **PASS** (see O5, O6) |
| 6 | `RC10_REPORT.md` claims match the files | **PASS** (minor O2) |

Overall: the package reproduces every frozen requirement; findings O1-O7 are
non-blocking for the rc10 staging checks. **O1 is the one item to fix before the
real box launch.**

## Criterion 1 — seeds diff

- LF bytes, 0 CR, sha256 = `efe1286d49583ef9b65037ce0f5118a127a45a43338f3b17bc9fbe46dc220ba7` (matches the recorded value; rc10 `output/campaign_stage1_v3/seeds_py_wm_c5.diff`).
- The diff is 25 lines with **no `-` lines**: it adds `WM58_CHUNK5_SEEDS = frozenset(range(8707, 8737))`, `WM58_CHUNK5_LOADING_MAP` (`seeds_py_wm_c5.diff:10-11`), and one `role == "exposed"` clause inside `require_development_case` (`:20-22`). No existing seed set is touched (only context lines around the WM-58 chunk-4 block).
- `git apply --check --verbose` against a **copy** of `E:/research-dense/traffic_shepherd/seeds.py` in a tmp dir: exit `0` (`Checking patch traffic_shepherd/seeds.py...`).
- Post-apply copy sha256 = `71b0663a89a71816189393f5d8632c863de79f1e9fda29e581484ebcbad1045b` — reproduced with `git -c core.autocrlf=false apply` (and by LF-normalizing the default apply). Live `seeds.py` was **not** edited (`git status` unchanged).
- A second run of the patched module accepts exactly the 30 planned `(seed, loading)` pairs and refuses the rest (`test_chunk5_seeds.py` passed).

## Criterion 2 — WORLD_PLAN.json

`output/campaign_stage1_v3/results_plan_chunk5/WORLD_PLAN.json` (sha256
`eac2998f...`): 30 worlds, 15 `(loading, penetration)` pairs x 2 replicates;
loadings `{2400,2600,2700}` x penetrations `{0.02,0.05,0.10,0.20,0.40}`; seeds
exactly `8707..8736` each once; every world `split="TEST", role="exposed"`; t0
balanced 5 each over `{240,300,360,420,480,540}`, `plan_seed=20261012`.
Per world 12 runs and 7,800 sim-s: `base`/`native_base` 900 s; `native`, `pi`,
`fixed1/2`, `switching1/2`, `W-lo-early/late`, `W-mid-early/late` 600 s. Total
234,000 sim-s; 360 distinct run names. Independent replay `build_plan(CHUNK5_SEEDS)`
equals the on-disk plan (modulo `created_utc`) and `build_seal` equals the on-disk
seal; seal roots equal the world names. Wall-law pin version `wall-v2.1-closure`,
sha256 `6e0a0805a63da905f7d7a1b38dad70526da3540d09bd0721411f0b3449c04bfa` = the
actual `traffic_shepherd/wall_group.py` bytes.

## Criterion 3 — launcher

`chunk5_launch.py` gates (each loud): claim-config + worker bounds (`:217-218`);
seed audit present/fresh (`:221-222`, `:112-125`); owner authorization schema +
`registration_sha256 == f8c7d760` (`:223-225`, `:77-109`); plan shape
30/360/234,000/all-TEST (`:226-227`, `:128-161`); disk preflight; live-seeds drift;
new runs root unless `--resume`/claim dir; write-once `LAUNCH_SUMMARY.json`, zero
retries. The refusal paths run headless with a fake launcher (`test_chunk5_launch.py`,
10 tests). Wall law v2.1 is used unchanged via `campaign_walls_v1`; the four wall
arms dispatch through `run_wall_branch.py _exec_wall_branch`, and the stage-1 arms
through `run_chunk.py _exec_base/_exec_branch`; the child argv keys match those
parsers exactly. No chunk-4 module is modified (only imported:
`_claim_world`, `_machine_identity`, `_launch_child`, `StartStagger`,
`check_claim_config`, `check_disk_preflight`).

## Criterion 4 — grading

`grade_chunk5.py` reads the plan and each `<run>/RUN_RECEIPT.json` only
(`:115-127`): run/world/arm/seed, `exit_status`, `registration_sha256`,
`cav_speed_mode==27`, `physics_rows_written == expected` (row COUNTS), and
teleport/collision COUNTS from the receipt. `OUTCOME_FILE_MARKERS` (`:41-45`) and
a `builtins.open`/`io.open` spy prove no `physics_steps`/`outcome_record`/... is
opened (`test_chunk5_grade.py:40-69`). No delay/TTC/speed sample is consumed.

## Criterion 5 — seal + guard

`protocols/chunk5_test_seal.json` (`allowed_reader=WM-N1_FINAL_TEST`,
`registration_sha256=f8c7d760...`) lists exactly the 30 chunk-5 roots.
`tools/holdout_guard.assert_chunk5_not_sealed` refuses every other reader
(including run-name style `world__arm`) and passes the WM-N1 final test
(`:58-84`). Tests: `test_chunk5_grade.py` seal cases pass; `tests/test_holdout_exclusions.py`
3 passed and `tests/planner/test_holdout_guard.py` 18 passed.

## Test commands and results (WD_FORBID_D=1 / RD_FORBID_D=1, basetemp outside rc10)

```
python -m pytest -p no:cacheprovider --basetemp=<tmp> \
  output/campaign_stage1_v3/tests/test_chunk5_{plan,seeds,launch,grade}.py
  -> 28 passed, 1 warning in 4.86s

python -m pytest -p no:cacheprovider --basetemp=<tmp> tests/test_holdout_exclusions.py
  (RESEARCH_DENSE_ROOT=E:/rd-wt/rc10)
  -> 3 passed, 1 warning in 0.49s

python -m pytest -p no:cacheprovider --basetemp=<tmp> tests/planner/test_holdout_guard.py
  -> 18 passed, 1 warning in 4.01s

python -m pytest -p no:cacheprovider --basetemp=<tmp> output/campaign_walls_v1/tests
  -> 169 passed, 1 warning in 424.27s

python -m ruff check --no-cache --config output/campaign_stage1_v3/ruff.toml <new files>
  -> All checks passed!
```

## Objections (with file:line and concrete fix)

**O1 (medium-high, fix before launch) — the sha gates are EOL-sensitive.**
`chunk5_registration.sha256_file` hashes the *raw* bytes
(`chunk5_registration.py:53-54`) and is compared to the frozen LF-sha in
`check_authorization` (`chunk5_launch.py:88-92`). On this host system
`core.autocrlf=true` (`file:C:/Program Files/Git/etc/gitconfig`), so the committed
`REGISTRATION_CHUNK5_TEST.md` checks out as CRLF: the worktree copy at
`E:/rd-wt/revrc10/output/wd01/wm_n1/REGISTRATION_CHUNK5_TEST.md` has raw sha256
`bc19c411...` (92 CRLF), while the committed git blob is LF and hashes to
`f8c7d760...`. Result: a normal checkout on the launch box makes the launcher
refuse ("registration file does not match the frozen chunk-5 sha"). The same
hazard applies to the recorded seeds shas (`seeds_chunk5.py:45-46`, used in
`live_seeds_state` `:90-127`): a CRLF checkout of `seeds.py` is neither the
pre-apply `90187c6a...` nor the post-apply `71b0663a...` and is reported as
*drift*. This is also visible in criterion 1: a plain `git apply` (default
`autocrlf=true`) produces a CRLF file with sha `5d31944e...`, not `71b0663a...`.
*Fix:* hash LF-normalized bytes (e.g. a `registration_lf_sha256()` that replaces
`\r\n`→`\n` before `sha256`), or add `.gitattributes`
(`REGISTRATION_CHUNK5_TEST.md text eol=lf`, `traffic_shepherd/seeds.py text eol=lf`),
and record the same EOL-exact apply command in `RC10_REPORT.md`.

**O2 (low) — report arithmetic.** `RC10_REPORT.md:106-112` reports
`59 failed / 294 passed / 44 errors` without the new files and
`59 / 320 / 44` with them, "where the +26 are the new tests", but the same report
(`:103`) and this review both count **28** new tests. *Fix:* correct the delta (or
state which two new cases are not collected in the full-suite run).

**O3 (low) — forbidden-receipt guard is incomplete for the real receipt shape.**
Real `RUN_RECEIPT.json` carries `suffix_summary` (`run_chunk.py:2429`) and
`evaluate_summary` (`run_chunk.py:2773-2774`), both outcome payloads, but
`FORBIDDEN_RECEIPT_FIELDS` (`grade_chunk5.py:47-50`) omits them, and the spy test
uses a minimal synthetic receipt (`test_chunk5_grade.py:22-29`). Grading does not
*consume* those fields (the core requirement holds), but the defensive guard never
fires against a real receipt. *Fix:* add `evaluate_summary`/`suffix_summary` to the
forbidden set (and write outcome-free receipts), plus a test using the real
receipt key set.

**O4 (low) — child argv is never executed.** `test_chunk5_launch.py` injects
`launcher=lambda argv: 0` (`:59`), so the 360 dispatches are counted, not run; the
`_exec_base`/`_exec_branch`/`_exec_wall_branch` invocations are only statically
matched (verified here by reading the parsers). *Fix:* add an argv/`--dry-run`
assertion test per arm.

**O5 (low) — default guard root vs seal location.** `tools/holdout_guard.py:19-20`
defaults `ROOT=E:/research-dense`, so `load_chunk5_seal()` with no path would look
for `E:/research-dense/protocols/chunk5_test_seal.json`, which does **not** exist
(check `Test-Path = False`); the seal lives at `E:/rd-wt/rc10/protocols/`. *Fix:*
ensure the seal is committed to the integrated root, or make the seal root
explicit, and add a default-path test.

**O6 (info) — divergent exclusion lists.** `E:/rd-wt/rc10/protocols/holdout_exclusions.json`
(`kept_disclosed=["w8477m2600p02t480"]`) differs from
`E:/research-dense/protocols/holdout_exclusions.json` (adds `w8617m2700p02t300`).
`tests/test_holdout_exclusions.py` expects the rc10 copy and passes only with
`RESEARCH_DENSE_ROOT=E:/rd-wt/rc10`; with the guard's default root it fails. Chunk-5
roots are in neither list, so the risk is low. *Fix:* reconcile the two files before
integration.

**O7 (info) — freeze-record path.** `C5TEST_FREEZE.json:11` records the seed audit
at `output/campaign_stage1_v3/results_audit_chunk5/C5_SEED_AUDIT.json`, which does
not exist; the audit is at the workspace root `C5_SEED_AUDIT.json` (sha256
`a2812857...`, matching the recorded digest). *Fix:* correct the record path (or
place the file where recorded).

## Criterion 6 — report vs files

The deliverable table and every checked numeric claim match the files (LF-sha
`efe1286d`, pre-apply `90187c6a`, post-apply `71b0663a`, wall-law `6e0a0805`,
30/360/234,000, plan seed 20261012, 28 package tests, 169 walls tests, ruff clean).
The only mismatch is the O2 arithmetic. The box commands use `E:/rd-wt/rc10/...`
staging paths, which is expected for this worktree and changes with `WORKSPACE_ROOT`
after integration.
