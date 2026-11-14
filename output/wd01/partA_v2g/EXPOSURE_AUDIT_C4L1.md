# EXPOSURE AUDIT — WD-01-C4L1 (run time, 2026-09-30)

Frozen registration `REGISTRATION_C4L1.md` (LF-sha256
`5e4a772137b8b427823b4860a20440eaa65abcbea735551eff681aa67b80b124`)
requires, BEFORE freezing and BEFORE the pass: "an exposure audit, re-run
at run time". This is that audit. Executed from worktree
`E:\rd-wt\wd01c4l1` (branch `zcode/wd01c4l1` at 9f716b4), before any
C4L1 eval code ran and before any C4L1 output existed.

## Command

Full-text search of every Part A output tree for the 9 chunk-4
CALIBRATION world names, then for the 15 HOLDOUT world names (13
excluded + kept_disclosed `w8477m2600p02t480` + clean `w8451m2700p20t420`
from `protocols/holdout_exclusions.json`). ripgrep 15.0.0, fixed-string,
binary included, `.git` excluded, and the two registration files that
legitimately name the worlds excluded (`REGISTRATION_C2L1_V2G_DRAFT.md`,
`REGISTRATION_C4L1.md`). This audit record itself names the worlds and
is written after the search; exclude it when re-running.

```powershell
# roots (49, all that exist):
#   E:\rd-wt\wd01*\output\wd01            (44 worktrees with output\wd01)
#   E:\research-dense\output\wd01*        (5 dirs: wd01_partB_v0data,
#       wd01_partB_v0data_v03, wd01_partB_v0data_v031,
#       wd01_partB_v0data_v032, wd01_rep)
rg -l -F -uuu --no-require-git -g '!**/.git/**' `
   -g '!REGISTRATION_C2L1_V2G_DRAFT.md' -g '!REGISTRATION_C4L1.md' `
   -f cal_patterns.txt @roots
# cal_patterns.txt: the 9 CAL world names, one per line
# holdout_patterns.txt: the 15 HOLDOUT world names, one per line
```

## Output — AUDIT 1 (the 9 CAL worlds)

**ZERO files matched.** `rg -c ...` (per-file counts) wrote an empty
file; `rg -l` returned nothing. No Part A code, report, ledger draft,
opened-file list, raw-row file, table, diagnosis, plan, or
manifest under any of the 49 roots contains any of:

w8621m2700p02t240, w8622m2700p02t360, w8639m2700p05t360,
w8640m2000p05t540, w8657m2700p10t300, w8658m2600p10t360,
w8674m2600p20t300, w8675m2700p20t300, w8692m2600p40t300

(The pre-audit of 2026-09-30 08:30 found the same: none.)

## Output — AUDIT 2 (the 15 HOLDOUT worlds)

148 files matched, all of exactly two kinds, none a Part A eval output
or opened-file list:

1. `partB_v0data/` MANIFEST.json / DATA_CARD_V0.md / build_v0data.py
   (and their `__pycache__` copies, and the research-dense
   `wd01_partB_v0data*` mirrors): the HOLDOUT worlds appear as members
   of Part B v0 training-data records. Part B data construction is not
   Part A and never opened chunk-4 `__native_base` Part A truth.
2. `partA_act/tests/test_act.py` line 243 (`test_holdout_refused`):
   `ActWorld("w8407m2400p02t360", 360.0)` inside
   `pytest.raises(ValueError)` — the world name appears only in a test
   asserting that the HOLDOUT world is REFUSED.

A targeted re-grep restricted to the Part A analysis trees
(`partA_v2g`, `partA_ctrl`, `partA_data` in wd01 and wd01c4l1):
**zero matches** — no HOLDOUT world appears in any opened-file list,
summary, raw-row file or report of a Part A evaluation.

## Verdict

- None of the 9 CAL worlds appears in any prior Part A output. ✔
- No HOLDOUT run file/world was opened by any Part A code; the only
  Part A mention is the refusal test. ✔

**AUDIT PASSED — the look may proceed.** If this file is re-read by a
later audit, exclude it (it quotes the world names); the audit above
ran before this file existed.
