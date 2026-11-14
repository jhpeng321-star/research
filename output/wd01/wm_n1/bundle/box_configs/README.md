# WM-N1 per-box config files (`bundle/box_configs/`)

Machine-readable packing of the `bundle/GPU_PLAN.md` 3-box and 4-box
schedules.  A reconstructor drops one `boxN.json` into the GPU box's
`gpu_bundle` for the physical box it is provisioning; each file lists the
configs that box runs **sequentially** (boxes run in parallel, `member-1` is
box 1's first job).

```
bundle/box_configs/
  README.md
  3box/box1.json box2.json box3.json index.json
  4box/box1.json box2.json box3.json box4.json index.json
```

The `N` in `boxN.json` is the physical box number of that plan (3-box:
`box1..box3`; 4-box: `box1..box4`).  The plan is the directory, so the two
plans never collide on the same file name.

## Generation

The files are exactly `bundle.multibox.plan_fleet(n)` (the GPU_PLAN tables),
enriched with per-config argv/seed/output metadata:

```
python -m bundle.multibox plan-fleet --boxes 3
python -m bundle.multibox plan-fleet --boxes 4
```

The generator lives outside the repo (one-off); `index.json` records the
planner output (per-box configs + finish hours) so a regenerated file can be
diffed.

## Schema (`wm_n1_box_config/1`)

| field | meaning |
| --- | --- |
| `plan` | `"3-box"` / `"4-box"` |
| `box_id` | planner box id (`box-1` …) |
| `wall_clock_hours`, `box_finish_hours` | design midpoints (amendment 3 A3.7) |
| `assigned_configs` | ordered list (member-1 first per family) |
| `gpu_env` | Python 3.10/3.11 + `torch>=2.7,<2.13` (cu128, Blackwell sm_120) install line |
| `corpus_source` | featurised pack placeholder `D:/rd-staging/wmn1_feat/pack_v2` (the M1-only `2box/c_m1only` configs use `pack_v3`, `format: "v3"`, `k_max: 192`, amendment 5) |
| `configs[]` | one entry per config (below) |

Each `configs[]` entry:

| field | meaning |
| --- | --- |
| `config`, `kind`, `family`, `member`, `seed` | identity; seeds `101/202/303/404/505` |
| `trainer` | `train.idea_a.IdeaATrainer` (Idea A / held-out) or the v1 `train.trainer.Trainer` (variant N) |
| `train_argv` | exact training command (`python -m train.idea_a …` / `python -m train …`) |
| `eval_argv`, `score_argv` | exact `evalh.run_eval` commands (split + predictor + reference) |
| `expected_outputs` | written files (`member_<k>.pt`, `.json`, `TRAIN_RECEIPT.json`, eval `REPORT.json`, …) |
| `start_hours`, `gpu_hours` | planner start / midpoint estimate |
| `approval`, `timing_gate` | per-config owner `APPROVAL.json` + `TIMING_GATE.json` paths |
| `box_cli` | `python -m bundle.multibox run-config …` for one-config-per-box runs |

`$WORK` is the box-local run directory.  Each config keeps its own owner
approval; a member-k (k>1) refuses until its family's member-1 completes
(amendment 1 A1).  The CPU box never reads `D:/rd-staging`; the placeholder is
only where the GPU box mounts the FEAT1 pack.
