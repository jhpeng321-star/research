# partB_v04b — WD-01-PB-V04B direct-difference pipeline (code + tests)

Implementation of `output/wd01/REGISTRATION_PARTB_V04B.md` (LF-sha256
`e00cd98e...`, registered FINAL 2026-10-01): one quantile-GBM per
(tau x sensing variant) predicting the ARM DIFFERENCE dY = Y(arm) −
Y(native sibling) DIRECTLY on J4 wall pairs (same world + t0, same
saved state), with the PLANNED-SCHEDULE action descriptor (19 x 30 s
segments + wall hold/speed/cap; "damp 60 s then native" != "damp 570
s"), the auxiliary chunk-4 contrasts as training rows only, the
registered A1/A3 acceptance with duration/onset x tau stratified
reporting, the ONE-CAL-read guard, and the J-PULSE retrain hook.

**State: CODE_READY. Nothing has been fitted; the ONE CAL read has not
happened.** The TRAIN-only dry-run (<= 4 worlds, no CAL row) was run as
authorized by the owner brief; its log is
`results/dryrun/dryrun_log.txt` (fresh complete re-run, ends with
"dry-run complete: NO CAL row was read"). The first saved attempt's
truncated log is preserved unchanged under
`results/dryrun_attempt1_stale/` for provenance; its report JSON was
complete, only the log tail was lost when the previous task hit its
runtime limit. The ledger allocation entry
`output/wd01/LEDGER_DRAFT_PB_V04B_ALLOCATION.json` is committed with
`status: planned` (registration s.10: committed BEFORE the build) and
stays planned until the owner's ack.

Everything below runs from `output/wd01` with
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, `WD_FORBID_D=1`,
CPU, 1 worker, zero SUMO.

## Layout

| module | role |
|---|---|
| `reg.py` | registration binding (LF-sha), tau/quantiles, the fixed 6-candidate grid, wall arms, contrasts, acceptance constants |
| `j4plan.py` | J4 plan access; split FIRST by name (39 TRAIN / 5 CAL); discloses the one recorded name drift (`w8662m2700p10t300` in the registration text vs `w8662m2600p10t300` in the plan — the plan is the split authority) |
| `schedule_desc.py` | the s.12 PLANNED-SCHEDULE descriptor (segment shares + wall block), duration/onset strata, first-60-only counts |
| `pairs_v04b.py` | pair builder: wall arms + aux contrasts vs the native sibling at the same world + t0; saved-state assertion refuses a mismatched world |
| `build_v04b.py` | t0-row build: aux/native rows COPIED from the v0.4a root; wall rows from the raw J4 runs (labels via `v0lib.window_label`, request-time wall descriptors via `wall_tokens_v04`, action tokens via `actions_v04`); NEW-root COPY + MANIFEST + SHA256SUMS |
| `descriptors_v04b.py` | design matrix: deployable strict/detector state + v0.4a action tokens + policy/pi + arm-family one-hot + the planned-schedule descriptor |
| `model_v04b.py` | DirectDiffGBM: world-grouped CV over the registered grid (pinball, delay head, headline rows), conformal shifts on headline OOB residuals, monotone quantiles, ridge-on-dY reference, wall-only ablation |
| `acceptance_v04b.py` | A1 + A3(i)/(ii) + verdict + s.12.1 stratified tables; identical CRPS code path as v0.4a (imports `partB_v0.acceptance_v04a` read-only) |
| `cal_guard.py` | OWNER_ACK gate + `CAL_READ_ONCE_V04B.json` read-once marker + the TRAIN-only hard guard |
| `cal_read_v04b.py` | THE ONE CAL read (guarded): frozen models predict, acceptance + verdict report, no refit |
| `pulse_hook.py` | J-PULSE disclosed-retrain config + the registered OOF diagnostic (diagnostic-only) |
| `dryrun.py` / `__main__.py` | entry points |

## Sequence after the owner's ack (exact commands)

```powershell
# from output/wd01, rl_driving env, WD_FORBID_D=1
$env:WD_FORBID_D = "1"

# 0. read-only integrity check
python -m partB_v04b selfcheck

# 1. the allocation entry is ALREADY committed with status "planned"
#    (registration s.10: before the build).  On the owner's go, flip it
#    to in_progress and commit that BEFORE starting the build:
$p = "output/wd01/LEDGER_DRAFT_PB_V04B_ALLOCATION.json"
(Get-Content -Raw $p).Replace('"status": "planned"', '"status": "in_progress"') | Set-Content -NoNewline -Encoding ascii $p
git add -f output/wd01/LEDGER_DRAFT_PB_V04B_ALLOCATION.json
git commit -m "wd01: PB-V04B registered run starts (owner ack received)"

# 2. TRAIN build of the NEW data root (COPY of v0.4a sources + wall t0 rows;
#    MANIFEST + SHA256SUMS; never opens CAL/HOLDOUT/chunk-5)
python -m partB_v04b build
#   -> E:/research-dense/output/wd01_partB_v0data_v04b

# 3. full TRAIN fit (both variants + wall-only ablation; still no CAL row)
python -m partB_v04b fit

# 4. drop the owner's ack file, then THE ONE CAL read
#    output/wd01/partB_v04b/results/OWNER_ACK_PB_V04B.json:
#    {"ack": "<owner's go>", "registration_lf_sha256": "e00cd98edadbdfdafc9267eefdd9f004fb72cc7de8c531e00ef27e8be8b5ad9d"}
python -m partB_v04b cal-read
#   -> results/cal_read/acceptance_report.json (+ finished read-once marker)
#    NO refit after this point (registration s.7/s.8)

# 5. report + ledger close-out (REPORT_V04B.md, LEDGER_DRAFT_PB_V04B.json,
#    CHANGES_V04B.md, RESULTS_SUMMARY regen, validate/verify-sources)
```

## Never (hard guards)

CAL/HOLDOUT rows outside the guarded cal-read; chunk-5;
`dispatcher_rows.jsonl`; `insertions.jsonl`; any outcome block of
`branch_record.json` (only `ALLOWED_RECORD_FIELDS` are bound); git
reset/clean/checkout--/stash drop; D: paths; deleting files.

## Tests

```powershell
$env:WD_FORBID_D = "1"
python -m pytest output/wd01/partB_v04b/tests -q
```

73 tests, all synthetic (no run file, no v0.4a file, no CAL world is
opened by the unit tests): descriptor distinguishes durations; pairs
same world + t0 only with saved-state refusal; CV folds world-grouped;
metrics on truth tables; CAL guard refuses without OWNER_ACK, on a
second read, and on a foreign-sha marker; J-PULSE hook contract;
synthetic end-to-end pipeline. The real-data dry-run is run separately
(`python -m partB_v04b dry-run`) and bounded to <= 4 TRAIN worlds.
