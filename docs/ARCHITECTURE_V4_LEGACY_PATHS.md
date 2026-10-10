# ARCHITECTURE_V4 - legacy import paths after WP-F (guide for the integrator)

WP-F (branch `wms/clean-f`) moved the legacy WM-S trees unchanged with `git mv`:

| old | new |
|---|---|
| `output/wd01/wm_n1/wms/` (41 py, tests, shell chains) | `archive/wms_era/wm_n1/wms/` |
| `output/wd01/wm_n1/wms_eval/` (30 py) | `archive/wms_era/wm_n1/wms_eval/` |

The extra `wm_n1/` level is deliberate: moved code uses `Path(__file__).parents[k]` locators
(for example `wms_eval/data.py` `WORKSPACE = parents[4]`, `wms/tests/*` `parents[5]`) that
resolve to the repository root only at this depth. Every file's old->new path and sha256 is in
`archive/wms_era/PATH_MAP.json`.

## What still works unchanged (thin shims)

`output/wd01/wm_n1/wms/__init__.py` and `.../wms_eval/__init__.py` are shims: they set
`__path__` to the archived directory and run the archived `__init__`. These therefore resolve
exactly as before:

* `from output.wd01.wm_n1.wms.model import WmSModel` (dotted path from the repo root; no sys.path needed)
* `sys.path.insert(0, "output/wd01/wm_n1"); import wms, wms_eval`, and `python -m wms_eval ...`
* `lpwo`, root `wms_decide.py` and the root `tests/test_wms_eval_*.py` (verified: same results as before the move)

## Repointing the equivalence/golden tests on the other branches

Preferred: keep the dotted path `output.wd01.wm_n1.wms...` / `output.wd01.wm_n1.wms_eval...`.
It cannot collide with the new package, which is imported as plain `wms` from `src/`.

Collision hazard: the archived code itself does `from wms.model import ...` and
`from wms_eval... import ...` (top-level names), and `model.py`/`train.py` insert their parent
directory into `sys.path`. Rules for the integrator:

1. Never put `output/wd01/wm_n1` ahead of `src` on `sys.path` in a process that also needs the new
   `wms`; the shim `wms` would shadow it. Either run legacy-dependent equivalence tests in a
   separate process (pytest subprocess / marker), or import the legacy model through the dotted
   alias `output.wd01.wm_n1.wms` (repo root on `sys.path`; sibling legacy packages `hdv`, `features`,
   `closedloop`, `rollout2` still live in `output/wd01/wm_n1`).
2. If a test did `sys.path.insert(0, ".../output/wd01/wm_n1")` then `import wms.model` for the
   **legacy** model, switch it to the dotted import above. If it reads legacy sources as files,
   e.g. `ROOT / "wms" / "cli.py"`, point it at `REPO / "archive/wms_era/wm_n1/wms/cli.py"`
   (the same fix was applied to `output/wd01/wm_n1/lpwo/tests/test_wall_chains.py`).
3. Running the archived unit tests in place needs `PYTHONPATH=output/wd01/wm_n1` (for `hdv`,
   `features`, ...): `PYTHONPATH=output/wd01/wm_n1 python -m pytest archive/wms_era/wm_n1/wms/tests -q`
   (spot check of six files: 55 passed, 4 skipped).
4. Root `wms_decide.py` still reaches the legacy `wms_eval` through the shim; once WP-E's
   `src/wms/eval/decide.py` lands, reduce the root file to the 3-line shim.

## Other relocations touching active paths

* Root `tests/test_wms_eval_ctm.py` and `tests/test_wms_eval_run1_report.py` -> `archive/wms_era/tests/`
  (their modules are archived).
* 11 dead root tests -> `archive/tests_dead/` (see its `PATH_MAP.json`).
* `_out/` keeps only what the R3 chain needs: `train_r3_chain.sh`, `train_main_chain.sh` (sourced by
  it), `trainsetup_cpu.py`, `merge_params.json`, `train_main_data.json`, `lpo/`, `lpwo_v3/`,
  `sidecars_packv3_v2/`, `r3_packv3_subset/`, `r3_box/`, `wms_ckpt/`, `test_train_main_chain.py`.
  Everything else tracked there is in `archive/wms_era/_out/` (scripts) or
  `archive/wms_era/records/_out/` (records).
* Small edits outside the archive: `output/wd01/wm_n1/lpwo/tests/test_wall_chains.py`,
  `wms_decide.py` (harness-present check now uses `wms_eval/__init__.py`), and links to moved `_out`
  records in `README.md` / `docs/INDEX.md`.
