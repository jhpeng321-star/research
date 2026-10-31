# S-RB-MERGE — in progress (2026-10-31)

Base: `f0db3c82`, branch `c4/rb-merge`. Owner registration read (line 5682).

## Interface reserved for S-RB-INT / S-RB-LANE
- Switch: `PhysicsConfig.merge_model = "legacy" | "sumo"` (default legacy).
- New mechanics: `src/wms/physics/junction_merge.py`; new insertion adapter:
  `src/wms/model/sumo_insertion.py`. No new model tensors or training.
- Merge owns lane 2 -> 1 inside model-frame `[0, 75.44]`, lane-end stopping,
  target-gap acceptance and speed adaptation. Lane stream owns mainline changes.
- Planned lane hook: callable `lc2013_lane_logits(kin, present, physics, driver_params)`;
  merge implementation can consume mainline proposals and resolves merge conflicts.
- `merge_model="sumo"` bypasses learned merge/courtesy/ramp adaptation and learned
  admission; exact `conserve_queue` remains mandatory. Driver settings come only
  from the existing motion encoder or population defaults.
- SUMO source downloaded as text into newly created empty directory
  `E:/Temp/opencode/rb-merge-sumo-1271-20261031`; no downloaded code executed.

## Status / known gaps
- Fresh-session recovery: HEAD f0db3c82; only this untracked handoff survived.
  No implementation, evaluation receipt, PID or dispatch exists in this worktree.
- Authorization verified at REGISTRATION.md:5683; all four WHY documents read.
- Source review + implementation underway. No completed tests/evaluation yet.
- S-RB-LANE and S-RB-INT HANDOFF.md did not exist on initial read (~23:15).
- Runtime recovered: `py -3.13`, torch 2.15.0+cpu. Use one torch thread.
- NVIDIA driver unavailable; CPU load 0% at resume resource check.
- Shared mergespec folder not present on resume; no helper work duplicated.
- Validation will restrict recorded FIT seeds to 969400–969431 and observable fields.
