# WD-01 path roots, D:-removal, and the forbid-D hook

The research data used to live behind D: junctions (`D:/research-dense`,
`D:/rd-wt`, `D:/research-dense-backups` -> E:).  The owner will delete
those junctions, so all live code under `output/wd01/**` resolves its
roots via `wdpaths.py` and every place that OPENS a path read back from
a manifest/record/registration file routes it through `wdpaths.remap()`
(those frozen files keep their historical D: strings; they are never
edited).  ZERO simulation anywhere in this tree.

## Roots (`output/wd01/wdpaths.py`)

| constant | env var | default |
| --- | --- | --- |
| `RESEARCH_DENSE_ROOT` | `RESEARCH_DENSE_ROOT` | `E:/research-dense` |
| `RD_WT_ROOT` | `RD_WT_ROOT` | `E:/rd-wt` |
| `BACKUPS_ROOT` | `RESEARCH_DENSE_BACKUPS_ROOT` | `E:/research-dense-backups` |

Env vars are read once at import (like the rest of the wd01 libs);
`wdpaths` also mirrors the resolved values back into the environment via
`os.environ.setdefault`, so read-only REFERENCE modules that read the
same variables directly (the frozen `hybrid_state_v1/repo.py`, whose
baked-in default is the retired D: path) resolve to the live roots too.
Explicit user exports always win.

`remap(path)` rewrites a leading `D:/research-dense`,
`D:/research-dense-backups` or `D:/rd-wt` prefix (any slash style, drive
letter case-insensitive, `\\?\D:` extended-length forms) to the
corresponding root and leaves every other path unchanged.  str in ->
str out, `Path` in -> `Path` out.  Unit tests:
`output/wd01/tests/test_wdpaths.py`.

Import pattern used by the parts (the usual sys.path shim, one level up):

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # output/wd01
import wdpaths
```

Cross-worktree defaults are RD_WT_ROOT-relative, e.g. the V0 chunk-1
copy source `RD_WT_ROOT/wd01n/output/wd01/partB_v0data`; untracked data
comes from the populated worktrees via `WD01_DATA_ROOT`
(= `E:/rd-wt/wd01/output/wd01`) and `WD01_PARTA_DATA`.

## Forbid-D hook (`output/wd01/forbid_d_hook.py`)

`install()` registers a `sys.addaudithook` that raises
`ForbiddenDriveDError` on the file-system audit events (`open`,
`os.listdir`, `os.scandir`, `os.mkdir`, `os.rename`, `os.replace`,
`os.remove`, `os.rmdir`, `os.chmod`, `os.link`, `os.symlink`,
`shutil.copyfile`, ...) whenever an argument resolves to drive D:
(after `os.fspath`; `\\?\D:` forms included).  Readers that bypass
Python-level audit events are wrapped with the same check:
`numpy.load`, `pandas.read_parquet`, `DataFrame.to_parquet`,
`pyarrow.parquet.read_table/write_table`.

Activation: `WD_FORBID_D=1`.  `output/wd01/conftest.py` (loaded for
every suite because pytest's rootdir is the repo root) installs the hook
before any test module is imported.  Self-test:

```powershell
$env:WD_FORBID_D = "1"
python output/wd01/forbid_d_hook.py     # prints PASS lines + SELFTEST OK
```

## Running the suites

From the main checkout (bash), with the populated-data overrides:

```bash
cd E:/research-dense
export WD01_DATA_ROOT=E:/rd-wt/wd01/output/wd01
export WD01_PARTA_DATA=E:/rd-wt/wd01/output/wd01/partA_data
export WD01_SRC=E:/research-dense
export WD_DEVICE=cpu
export WD_FORBID_D=1                    # forbid-D audit hook
for s in part1 part2 part3 part3j part3k part3l part3m part3n part3o \
         part3q part4 partA_act partA_ctrl partA_data partB_v0data predict
do python -m pytest "E:/rd-wt/wd01s/output/wd01/$s/tests" -q; done
python -m pytest E:/rd-wt/wd01s/output/wd01/tests -q   # wdpaths + hook units
```

Without `WD01_DATA_ROOT`/`WD01_PARTA_DATA` the libs keep the in-repo
layout, so real-data tests skip/fail in a fresh worktree (untracked data
absent) exactly as before this change; path handling is unaffected.
