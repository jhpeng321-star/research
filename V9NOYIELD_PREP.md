# V9NOYIELD PREP

Branch `c4/v9noyield` in worktree `E:/agent-work/c4-v9noyield`, created from `c4/v9entry`
HEAD `317904d1`. No push. Local, CPU-only work; no SUMO, GPU, launch or training run.

## Why

A diagnosis (ONROAD_DIAG.md) found the WM-S `mainline_merge_yield` rule deadlocks the merge
zone and makes the WM penalise any CAV that deviates from its native CAV. Turning it off at
inference fixes the level (native-road delay WM/SUMO 1.25 -> 1.00) and the mean deviation
penalty (PI D -114 -> -16 vs SUMO +23; MAPPO -119 -> +63 vs +43). This change lets the same
override be applied **inside Dreamer imagination** at policy-training time.

## Files changed (commit c0bf6907)

```
 configs/c4_policy_v9noyield_v5m.toml       |  18 +++++   (new)
 src/wms/control/dreamer.py                 |  79 +++, 3 ---
 tests/unit/test_c4_wm_physics_overrides.py | 103 +++++   (new)
 3 files changed, 200 insertions(+), 3 deletions(-)
```

* `src/wms/control/dreamer.py`
  * `DreamerConfig.wm_physics_overrides: dict` (default `{}` = bit-identical) plus a mapping
    type check; field name is exempted from the numeric `asdict` validation loop.
  * `_physics_holders(model)` / `apply_wm_physics_overrides(model, overrides)`: the same
    attribute/`_modules` holder-walk as `tools/wm_eval_common.py` and the diagnosis's
    `coop_cl_noyield.py`. It replaces **every** `PhysicsConfig` holder with
    `dataclasses.replace(base, **overrides)`. An empty mapping returns immediately (no swap),
    so the default path is unchanged. The loaded fields are cached in a plain **dict** (not a
    `PhysicsConfig`) so the walk never re-finds it and repeated applications start from the
    checkpoint's physics. Unknown field names raise `ValueError`.
  * `DreamerTrainer.__init__`: applies the override to each member right after it is loaded,
    before `_model_identity` is fixed, and prints the effective physics
    (`[wm-physics] member N: {...} applied to H PhysicsConfig holder(s); effective
    mainline_merge_yield=...`).
* `configs/c4_policy_v9noyield_v5m.toml`: equals `configs/c4_policy_v9entry_v5m.toml` with
  exactly one added line (the switch). `entry_queue_state = true` is retained.
* `tests/unit/test_c4_wm_physics_overrides.py`: the three acceptance tests below.

No launch/loop files, no other worktree or branch was touched. Nothing under `sealed/` or any
restricted-token path was read.

## Exact config diff vs `configs/c4_policy_v9entry_v5m.toml`

```diff
--- a/configs/c4_policy_v9entry_v5m.toml
+++ b/configs/c4_policy_v9noyield_v5m.toml
@@ -15,3 +15,4 @@ wm_config = "configs/c4_wm_v5m_refresh_q.toml"
 
 [trainer]
 entry_queue_state = true
+wm_physics_overrides = {mainline_merge_yield = false}
```

`run.output_dir` is deliberately left as `runs/active/C4-POLICY-V9ENTRY-V5M` so the file
differs by exactly the one switch. **Before any launch, retarget `output_dir`** to a fresh
directory (e.g. `runs/active/C4-POLICY-V9NOYIELD-V5M`) so a run cannot write into the v9entry
run's output.

## Test results

Command (worktree root): `python -m pytest <args> -q` with
`PYTHONPATH=src;.` and `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`.

| suite | result |
| --- | --- |
| `tests/unit/test_c4_wm_physics_overrides.py` (new) | **6 passed** |
| `tests/unit/test_c4_dreamer.py` | **23 passed** |
| `tests/unit/config` | **157 passed** |
| `tests/unit/test_import_graph.py` | **203 passed** |
| `test_c4_cfmem`, `test_c4_futures`, `test_c4_srcgrad`, `test_c4_srcwait_imagination`, `test_c4_policy2`, `test_c4_policyv5`, `test_c4_v6explore` | **77 passed** |
| `ruff check` on the 3 changed files | clean |

Acceptance mapping:
1. **empty option -> physics unchanged**: `test_empty_override_is_bit_identical` (no holder is
   replaced, `model.physics is` the loaded object, value stays `True`).
2. **option set -> every holder False**: `test_override_reaches_every_holder` and
   `test_trainer_applies_override_at_init` (a real `WmSModel` and a two-holder stub, each
   verified through the holder-walk after `DreamerTrainer.__init__`).
3. **config loads and resolves**: `test_policy_config_loads_and_resolves` reads the TOML
   through `train_cli._read_config`, checks the switch, `entry_queue_state`, and constructs
   `DreamerConfig(**raw["trainer"])`.

No tiny CPU training smoke was run: the frozen WM checkpoint and the `WMS_DATA_*` start-window
data are not present in this worktree (no `*.pt` under `runs/active`, `world_model.checkpoints`
is empty and requires `--wm-ckpt`). Per the brief, unit tests only in that case.

## Commit

`c0bf6907` — "V9NOYIELD policy option: trainer.wm_physics_overrides turns WM
mainline_merge_yield off in imagination (default empty = bit-identical)".
